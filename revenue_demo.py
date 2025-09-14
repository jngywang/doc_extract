import json
import os
import requests
from typing import List, Dict, Any
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

class EdgarRAGPipeline:
    
    def __init__(self, openai_api_key: str, log_file_path: str = None):

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
        self.logger = logging.getLogger(__name__)

        # init openAI 
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.dataset = None

        # init Sentence Transformer
        self.logger.info("loading Sentence Transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def load_edgar_data(self):
        self.logger.info("loading EDGAR dataset...")
        try:
            # dataset = datasets.load_dataset("eloukas/edgar-corpus", "year_2018", split="test")

            self.dataset = load_dataset(
                "json",
                data_files={
                    "test": "/Users/jingyawang/Downloads/edgar/2018/test.jsonl"
                }
            )

            # ds = load_dataset(
            #     "json",
            #     data_files={
            #         "test": "/Users/jingyawang/Downloads/edgar/2018/test.jsonl"
            #     }
            # )
            # code = '1597892'
            # self.dataset = ds.filter(lambda x: x['cik'] == code)

            self.logger.info(self.dataset["test"]["filename"])
            return True
        except Exception as e:
            self.logger.info(f"Failed loading dataset : {e}")
            return False
   
    # confirm year 
    def get_test_data_2018(self) -> List[Dict]:
        if not self.dataset:
            self.logger.info("Dataset not found")
            return []
        
        test_data = self.dataset['test']
        
        data_2018 = []
        for item in test_data:
            if '2018' in str(item.get('filename', '')) or '2018' in str(item.get('year', '')):
                data_2018.append(item)
        
        self.logger.info(f"Found {len(data_2018)} items in year 2018")
        return data_2018
    
    def chunk_by_sections(self, document: Dict) -> Dict[str, str]:
        chunks = {}
        
        for i in range(1, 16):
            section_key = f'section_{i}'
            # section_key = 'section_1'
            if section_key in document and document[section_key]:
                section_content = document[section_key]
                lines = section_content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        chunk_key = f"{section_key}_line_{i}"
                        chunks[chunk_key] = line.strip()
        self.logger.info(f"Generated {len(chunks)} chunks")
        
        return chunks

    def filter_chunks_by_tokens_and_tfidf(self, chunks: Dict[str, str]) -> Dict[str, str]:
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

        try:
            tfidf_matrix = vectorizer.fit_transform(chunk_texts)
            feature_names = vectorizer.get_feature_names_out()

            # retrieving revenue related terms
            revenue_keywords = ['revenue', 'revenues', 'total revenue', 'net revenue', 'sales', 'income']
            revenue_indices = []

            for keyword in revenue_keywords:
                if keyword in feature_names:
                    revenue_indices.append(np.where(feature_names == keyword)[0][0])

            if not revenue_indices:
                self.logger.info("Revenue related terms not found in TFIDF feature")
                return token_filtered  # return result from step-1

            # tfidf score for each chunks 
            revenue_scores = []
            for i in range(tfidf_matrix.shape[0]):
                score = 0
                for idx in revenue_indices:
                    score += tfidf_matrix[i, idx]
                revenue_scores.append(score)

            # filter by score>0 
            selected_chunks = {}
            for i, (key, text) in enumerate(zip(chunk_keys, chunk_texts)):
                if revenue_scores[i] > 0:
                    selected_chunks[key] = text
            self.logger.info(f"Chunks filtered by TFIDF: {len(selected_chunks)}")

            return selected_chunks

        except Exception as e:
            self.logger.info(f"Error in TFIDF filtering: {e}")

            return token_filtered  # return result from step-1 
    
    def semantic_filter_with_sentence_transformer(self, chunks: Dict[str, str], 
                                                 query: str = "total revenue of 2018") -> Dict[str, str]:
        self.logger.info(f"To filter by Sentence Transformer. Starting with {len(chunks)} chunks")
        self.logger.info(f"Query: '{query}'")
        
        if not chunks:
            return {}
        
        chunk_keys = list(chunks.keys())
        chunk_texts = list(chunks.values())
        
        query_embedding = self.sentence_model.encode([query])
        chunk_embeddings = self.sentence_model.encode(chunk_texts, show_progress_bar=True)
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1]
        
        selected_chunks = {}
        similarity_threshold = 0.5
       
        self.logger.info("Similarity ranking result and samples:") 
        for i, idx in enumerate(top_indices):
            similarity_score = similarities[idx]
            chunk_key = chunk_keys[idx]
            chunk_preview = chunk_texts[idx][:100] + "..." if len(chunk_texts[idx]) > 100 else chunk_texts[idx]
            
            if similarity_score > similarity_threshold:
                selected_chunks[chunk_key] = chunk_texts[idx]
                self.logger.info(f"  {i+1}. {chunk_key}")
                self.logger.info(f"     Similarity score: {similarity_score:.4f}")
                self.logger.info(f"     Preview: {chunk_preview}")
        
        self.logger.info(f"Chunks filtered by semantic: {len(selected_chunks)} (thresholding at: {similarity_threshold})")
        return selected_chunks

    def extract_revenue_with_openai(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                  {
                        "role": "system",
                        "content": """You are a financial analyst specializing in SEC EDGAR filings from 1993-2020. 

                        Look for these revenue-related terms and their associated numbers:
                        - "total revenue" or "total revenues"
                        - "net revenue" or "net revenues" 
                        - "total net sales"
                        - "revenues"
                        - "revenue"
                        - "net sales"
                        - "total sales"
                        - "Operation revenue"

                        Extract the numerical value ( just number or with units like millions, thousands, etc.) associated with these terms.
                        If multiple values are found, report all with brief introduction.
                        If no related terms are found, respond with "No revenue information found".
                        If only related information found, but no exact number found, respond with "No exact revenue number found".
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"Extract revenue information from this SEC filing text:\n\n{text[:2000]}"
                    }
                ],
                max_completion_tokens=3000,
                reasoning_effort="medium"
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"API error: {str(e)}"
    
    def process_document(self, document: Dict) -> Dict[str, str]:

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

        if not tfidf_filtered:
            self.logger.info(f"  No chunks found in {filename} after step 2")
            return {filename: "No chunks found"}

        self.logger.info(f"Found {len(tfidf_filtered)} chunks after step 2")

        self.logger.info("\nStep 3: Sentence Transformer filtering...")
        semantically_filtered = self.semantic_filter_with_sentence_transformer(
            tfidf_filtered, 
            "total revenue of 2018"
        )
        
        if not semantically_filtered:
            self.logger.info(f"  No chunks found in {filename} after step 3")
            return {filename: "No chunks found"}
        self.logger.info(f"Found {len(semantically_filtered)} chunks after step 3")

        ### openai query and result collection
        results = {}
        # processing by sections 
        for section_name, content in semantically_filtered.items():
            if content:
                self.logger.info(f"processing {section_name}... ") #with content ==> {content}")

                revenue_info = self.extract_revenue_with_openai(content)
                results[f"{filename}_{section_name}"] = revenue_info
                self.logger.info(f"  {revenue_info}")

                # concurrency control with API rate limit 
                time.sleep(1)
            else:
                results[f"{filename}_{section_name}"] = "section not valid"
        ###
        
        return results
    
    def run_pipeline(self, n_files):
        self.logger.info("Starting EDGAR RAG Pipeline...")
        
        if not self.load_edgar_data():
            return {}
        test_data_2018 = self.get_test_data_2018()
        if not test_data_2018:
            self.logger.info("No data found in 2018")
            return {}
        

        all_results = {}
        
        for i, document in enumerate(test_data_2018[:n_files]):
            self.logger.info(f"\n=== Reading files {i+1}/{min(n_files, len(test_data_2018))} ===")
            
            results = self.process_document(document)
            filename = document.get('filename', f'doc_{i}')
            all_results[filename] = results
            
            self.logger.info(f"\nRetrieval results for file {filename} :")
            for key, value in results.items():
                self.logger.info(f"  {key}: {value}")
        
        return all_results

def main():
    API_KEY = os.getenv("OPEN_API_KEY")
 
    if API_KEY == "your-openai-api-key-here":
        print("OpenAI API Key needed!")
        return
    
    pipeline = EdgarRAGPipeline(API_KEY)
    results = pipeline.run_pipeline(n_files = 10)
    
    pipeline.logger.info("\n" + "="*60)
    pipeline.logger.info("FINAL RESULTS:")
    pipeline.logger.info("="*60)
    
    for filename, file_results in results.items():
        pipeline.logger.info(f"\nFile: {filename}")
        for section_key, revenue_info in file_results.items():
            pipeline.logger.info(f"  {section_key}: {revenue_info}")

if __name__ == "__main__":
    main()
