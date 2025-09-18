import json
import os
import argparse
import requests
from typing import List, Dict, Any, Tuple
import openai
from datasets import load_dataset
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, BooleanType, StructType, StructField
import sys
from datetime import datetime
import logging
import prompt
import term_dict

class EdgarRAGPipeline:
    
    def __init__(self, openai_api_key: str, year = '2018', key_options = ["REVENUE"], log_file_path: str = None):

        # logging to file 
        if log_file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file_path = f"logs/edgar_pipeline_{timestamp}.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file_path, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)

        self.openai_api_key = openai_api_key
        self.dataset = None
        self.year = year
        self.key_options = key_options

        # init Sentence Transformer
        self.logger.info("loading Sentence Transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # init PySpark
        self.spark = None
        self.spark_available = False
        self.max_concurrent = 4
        self._init_pyspark(self.max_concurrent)


    def _init_pyspark(self, max_concurrent):
        try:
            self.logger.info("Init PySpark...")
            
            self.spark = SparkSession.builder \
                .appName("EdgarRAGPipeline") \
                .master("local[*]") \
                .config("spark.driver.memory", "1g") \
                .config("spark.executor.memory", "1g") \
                .config("spark.executor.cores", "1") \
                .config("spark.executor.instances", str(self.max_concurrent)) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.driver.maxResultSize", "512m") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.python.worker.reuse", "true") \
                .getOrCreate()
            
            self.spark.sparkContext.setLogLevel("WARN")
            test_df = self.spark.createDataFrame([("test", 1)], ["text", "number"])
            test_count = test_df.count()
            self.spark_available = True
            self.logger.info(f"PySpark init！")
            self.logger.info(f"  - Spark version: {self.spark.version}")
            self.logger.info(f"  - core available: {self.spark.sparkContext.defaultParallelism}")
            self.logger.info(f"  - df size: {test_count}")
            
        except Exception as e:
            self.logger.info(f"PySpark init failed: {e}")
    
    def load_edgar_data(self):
        self.logger.info("loading EDGAR dataset...")
        try:
            self.dataset = load_dataset(
                "json",
                data_files={
                    "test": f"/Users/jingyawang/Downloads/edgar/{self.year}/test/*.jsonl"
                }
            )

            # ds = load_dataset(
            #     "json",
            #     data_files={
            #         "test": "/Users/jingyawang/Downloads/edgar/2018/test/test.jsonl"
            #     }
            # )
            # code = '10795'
            # self.dataset = ds.filter(lambda x: x['cik'] == code)

            self.logger.info(self.dataset["test"]["filename"])
            return True
        except Exception as e:
            self.logger.info(f"Failed loading dataset : {e}")
            return False
   
    # confirm year 
    def get_test_data_year(self) -> List[Dict]:
        if not self.dataset:
            self.logger.info("Dataset not found")
            return []
        
        test_data = self.dataset['test']
        
        data_year = []
        for item in test_data:
            if self.year in str(item.get('filename', '')) or self.year in str(item.get('year', '')):
                data_year.append(item)
        
        self.logger.info(f"Found {len(data_year)} items in year {self.year}")
        return data_year
    
    def chunk_by_sections(self, document: Dict) -> Dict[str, str]:
        chunks = {}
        
        for section_key in document.keys():
            if section_key.startswith('section_') and document[section_key]:
                section_content = document[section_key]
                lines = section_content.split('\n')
                for line_idx, line in enumerate(lines):
                    if line.strip():
                        chunk_key = f"{section_key}_line_{line_idx}"
                        chunks[chunk_key] = line.strip()
        
        self.logger.info(f"Generated {len(chunks)} chunks")
        return chunks

    def filter_chunks_by_tokens_and_tfidf(self, chunks: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        # Step 1: token filtering
        token_filtered = {}
        for key, text in chunks.items():
            tokens = text.split()
            if len(tokens) > 5:
                token_filtered[key] = text
        self.logger.info(f"Chunks filtered by token>3 : {len(token_filtered)}")
        if not token_filtered:
            return {}

        # Step 2: TFIDF filtering
        chunk_keys = list(token_filtered.keys())
        chunk_texts = list(token_filtered.values())

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)  # 1-gram and 2-gram
        )

        tfidf_matrix = vectorizer.fit_transform(chunk_texts)
        feature_names = vectorizer.get_feature_names_out()

        results = {}
    
        # Process each feature in key_options
        for feature in self.key_options:
            self.logger.info(f"Processing feature: {feature}")

            # retrieving feature related terms
            feature_keywords = term_dict.TERM_DICT[feature] # ['revenue', 'revenues', 'total revenue', 'net revenue', 'sales', 'income']
            feature_indices = []
    
            for keyword in feature_keywords:
                if keyword in feature_names:
                    idx = np.where(feature_names == keyword)[0][0]
                    feature_indices.append(idx)
                    # print("PICKED KEYWORDS: "+keyword)
    
            if not feature_indices:
                self.logger.info("Feature related terms not found in TFIDF feature")
                results[feature] = token_filtered  # return result from step-1
                continue
    
            # tfidf score for each chunks 
            feature_scores = []
            for i in range(tfidf_matrix.shape[0]):
                score = 0
                for idx in feature_indices:
                    score += tfidf_matrix[i, idx]
                feature_scores.append(score)
    
            # filter by score>0 
            selected_chunks = {}
            for i, (key, text) in enumerate(zip(chunk_keys, chunk_texts)):
                chunk_preview = text[:100] + "..." if len(text) > 100 else text
                # print("TO PICK: "+chunk_preview)
                if feature_scores[i] > 0:
                    # print("SELECTED")
                    selected_chunks[key] = text
            self.logger.info(f"Chunks filtered by TFIDF: {len(selected_chunks)}")
            results[feature] = selected_chunks
    
        return results

    def semantic_filter_with_sentence_transformer(self, chunks_by_feature: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        results = {}

        # Process each feature
        for feature in self.key_options:
            chunks = chunks_by_feature.get(feature,{})
            self.logger.info(f"To apply sementic filter for feature {feature}. Starting with {len(chunks)} chunks")
        
            if not chunks:
                self.logger.info(f"No chunks available for {feature}")
                results[feature] = {}
                continue

            feature_keywords = term_dict.TERM_DICT[feature]
            # queries = [keyword + " of " + self.year for keyword in feature_keywords]
            queries = [keyword for keyword in feature_keywords]
            self.logger.info(f"Using {len(queries)} queries:")
            for i, query in enumerate(queries):
                self.logger.info(f"  Query {i}: '{query}'")

            chunk_keys = list(chunks.keys())
            chunk_texts = list(chunks.values())
        
            query_embeddings = self.sentence_model.encode(queries)
            chunk_embeddings = self.sentence_model.encode(chunk_texts, show_progress_bar=True)
            all_similarities = cosine_similarity(query_embeddings, chunk_embeddings)
        
            selected_chunks = {}
       
            self.logger.info("Similarity ranking result and samples:") 
            for chunk_idx in range(len(chunk_texts)):
                # for every chunk, check the score for every query
                chunk_similarities = all_similarities[:, chunk_idx]
                max_score = np.max(chunk_similarities)
                # chunk_preview = chunk_texts[chunk_idx][:100] + "..." if len(chunk_texts[chunk_idx]) > 100 else chunk_texts[chunk_idx]
                # self.logger.info(f"     Similarity score: {max_score:.4f}")
                # self.logger.info(f"     Preview: {chunk_preview}")
                if max_score > term_dict.similarity_threshold[feature]:
                    chunk_key = chunk_keys[chunk_idx]
                    selected_chunks[chunk_key] = chunk_texts[chunk_idx]
                    chunk_preview = chunk_texts[chunk_idx][:100] + "..." if len(chunk_texts[chunk_idx]) > 100 else chunk_texts[chunk_idx]
                    self.logger.info(f"  Selected {chunk_key}")
                    self.logger.info(f"     Similarity score: {max_score:.4f}")
                    self.logger.info(f"     Preview: {chunk_preview}")
       
            results[feature] = selected_chunks 
            self.logger.info(f"Chunks filtered by semantic for feature {feature}: {len(selected_chunks)} (thresholding at: {term_dict.similarity_threshold[feature]})")
        return results 

    @staticmethod
    def process_partition(api_key_broadcast, partition_iterator, year_broadcast = '2018', key_options_broadcast = ["REVENUE"]):
        import openai
        client = openai.OpenAI(api_key=api_key_broadcast.value)
        year = year_broadcast.value
        key_options = key_options_broadcast.value
        
        for row in partition_iterator:
            chunk_key = row.chunk_key
            content = row.content
            filename = row.filename
            feature = row.feature # only one feature is processed in a row

            feature_upper = feature.upper()
            if feature_upper == "REVENUE":
                sys_prompt = prompt.SYS_PROMPT_REVENUE.format(year=year)
                user_prompt = f"Extract total revenue information from this SEC filing text:\n\n{content[:4000]}"
            if feature_upper == "LOSS":
                sys_prompt = prompt.SYS_PROMPT_LOSS.format(year=year)
                user_prompt = f"Extract total loss information from this SEC filing text:\n\n{content[:4000]}"
            if feature_upper == "INDUSTRY":
                sys_prompt = prompt.SYS_PROMPT_INDUSTRY.format(year=year)
                user_prompt = f"Extract industry information from this SEC filing text:\n\n{content[:4000]}"
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": sys_prompt 
                        },
                        {
                            "role": "user", 
                            "content": user_prompt
                        }
                    ],
                    max_completion_tokens=3000,
                    temperature = 0.1
                    # reasoning_effort="medium"
                )
                
                full_response = response.choices[0].message.content.strip()
                
                # Parse JSON response
                try:
                    response_json = json.loads(full_response)
                    feature_lower = feature.lower()
                    feature_info = response_json.get(f"{feature_lower} analysis", "No analysis found")
                    feature_value = response_json.get(f"{feature_lower} value", "Not found")
                except json.JSONDecodeError:
                    feature_info = full_response
                    feature_value = "JSON parse error" + full_response

            except Exception as e:
                feature_info = f"API error: {str(e)}"
                feature_value = "API error"

            yield (chunk_key, filename, content, feature, feature_info, feature_value)

    def pyspark_openai_extraction(self, chunks_by_feature: Dict[str, Dict[str, str]], filename: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
        try:
            chunks_list = []
            for feature, chunks_dict in chunks_by_feature.items():
                for chunk_key, content in chunks_dict.items():
                    chunks_list.append((chunk_key, content, filename, feature))
            
            self.logger.info(f"parallelized processing {len(chunks_list)} chunks")

            schema = StructType([
                StructField("chunk_key", StringType(), True),
                StructField("content", StringType(), True),
                StructField("filename", StringType(), True),
                StructField("feature", StringType(), True)
            ])
            df = self.spark.createDataFrame(chunks_list, schema).repartition(self.max_concurrent) 

            output_schema = StructType([
                StructField("chunk_key", StringType(), True),
                StructField("filename", StringType(), True),
                StructField("content", StringType(), True),
                StructField("feature", StringType(), True),
                StructField("feature_info", StringType(), True),
                StructField("feature_value", StringType(), True)
            ])          
 
            api_key_broadcast = self.spark.sparkContext.broadcast(self.openai_api_key)
            year_broadcast = self.spark.sparkContext.broadcast(self.year)
            key_options_broadcast = self.spark.sparkContext.broadcast(self.key_options)
            def partition_processor(partition_iterator):
                return EdgarRAGPipeline.process_partition(api_key_broadcast, partition_iterator, year_broadcast, key_options_broadcast)

            result_df = df.rdd.mapPartitions(partition_processor).toDF(output_schema)
            
            self.logger.info("Reduce to collect results...")
            results_rows = result_df.select("chunk_key", "filename", "feature", "feature_info", "feature_value").collect()
            
            final_results = {}
            feature_values = {}
            for feature in self.key_options:
                final_results[feature] = {}
                feature_values[feature] = ""

            for row in results_rows:
                feature = row.feature
                full_key = f"{row.filename}_{row.chunk_key}"
                feature_info = row.feature_info
                final_results[feature][full_key] = feature_info
                self.logger.info(f"Final results - {feature} - {full_key}: {feature_info}")
                if row.feature_value != "Not found" and row.feature_value != "API error":
                    feature_values[feature] = row.feature_value
          
            for feature in self.key_options: 
                self.logger.info(f"Final value - {feature} - {feature_values[feature]}") 
            return final_results, feature_values
            
        except Exception as e:
            self.logger.info(f"PySpark OpenAI API failed: {e}")

 
    def process_document(self, document: Dict) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:

        filename = document.get('filename')

        self.logger.info("="*80)
        self.logger.info(f"\nProcessing file: {filename}")
        self.logger.info("Step 1: chunk by section...")
        init_chunks = self.chunk_by_sections(document)

        if not init_chunks:
            self.logger.info(f"  No section found in {filename}")
            return {filename: "No section data found"}

        self.logger.info(f"Starting with {len(init_chunks)} chunks")

        self.logger.info("\nStep 2: token, TFIDF filtering...")
        tfidf_filtered = self.filter_chunks_by_tokens_and_tfidf(init_chunks)

        has_any_chunks = any(chunks for chunks in tfidf_filtered.values())
        if not has_any_chunks:
            self.logger.info(f"  No chunks found in {filename} after step 2")
            return {feature: {filename: "No chunks found"} for feature in self.key_options}
        for feature, chunks in tfidf_filtered.items():
            self.logger.info(f"Found {len(chunks)} chunks for {feature} after step 2")

        self.logger.info("\nStep 3: Sentence Transformer filtering...")
        semantically_filtered = self.semantic_filter_with_sentence_transformer(tfidf_filtered)
        
        has_any_chunks = any(chunks for chunks in semantically_filtered.values())
        if not has_any_chunks:
            self.logger.info(f"  No chunks found in {filename} after step 3")
            return {feature: {filename: "No chunks found"} for feature in self.key_options}, {}
        for feature, chunks in semantically_filtered.items():
            self.logger.info(f"Found {len(chunks)} chunks for {feature} after step 3")

        results, values = self.pyspark_openai_extraction(semantically_filtered, filename)

        return results, values
    
    def run_pipeline(self, n_files = 5):
        self.logger.info("Starting EDGAR RAG Pipeline...")
        
        if not self.load_edgar_data():
            return {}
        test_data_year = self.get_test_data_year()
        if not test_data_year:
            self.logger.info("No data found in {self.year}")
            return {}
        
        all_results = {}
        all_values = {}
        
        for i, document in enumerate(test_data_year[:n_files]):
            self.logger.info(f"\n=== Reading files {i+1}/{min(n_files, len(test_data_year))} ===")
            
            results, values = self.process_document(document)
            filename = document.get('filename', f'doc_{i}')
            all_results[filename] = results
            all_values[filename] = values
   
        return all_results, all_values
