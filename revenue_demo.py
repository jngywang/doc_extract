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

class EdgarRAGPipeline:
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.dataset = None

        # init Sentence Transformer
        print("loading Sentence Transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def load_edgar_data(self):
        print("loading EDGAR dataset...")
        try:
            # dataset = datasets.load_dataset("eloukas/edgar-corpus", "year_2018", split="test")
            # self.dataset = load_dataset(
            #     "json",
            #     data_files={
            #         "test": "/Users/jingyawang/Downloads/edgar/2018/test.jsonl"
            #     }
            # )
            ds = load_dataset(
                "json",
                data_files={
                    "test": "/Users/jingyawang/Downloads/edgar/2018/test.jsonl"
                }
            )
            code = '817720'
            self.dataset = ds.filter(lambda x: x['cik'] == code)
            print(self.dataset["test"]["filename"])
            return True
        except Exception as e:
            print(f"Failed loading dataset : {e}")
            return False
   
    # confirm year 
    def get_test_data_2018(self) -> List[Dict]:
        if not self.dataset:
            print("Dataset not found")
            return []
        
        test_data = self.dataset['test']
        
        data_2018 = []
        for item in test_data:
            if '2018' in str(item.get('filename', '')) or '2018' in str(item.get('year', '')):
                data_2018.append(item)
        
        print(f"Found {len(data_2018)} items in year 2018")
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
        print(f"Generated {len(chunks)} chunks")
        
        return chunks

    def filter_chunks_by_tokens_and_tfidf(self, chunks: Dict[str, str]) -> Dict[str, str]:
        # Step 1: token filtering
        token_filtered = {}
        for key, text in chunks.items():
            tokens = text.split()
            if len(tokens) > 5:
                token_filtered[key] = text
        print(f"Chunks filtered by token>3 : {len(token_filtered)}")
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
                print("Revenue related terms not found in TFIDF feature")
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
            print(f"Chunks filtered by TFIDF: {len(selected_chunks)}")

            return selected_chunks

        except Exception as e:
            print(f"Error in TFIDF filtering: {e}")

            return token_filtered  # return result from step-1 
    
    def semantic_filter_with_sentence_transformer(self, chunks: Dict[str, str], 
                                                 query: str = "total revenue of 2018", 
                                                 top_k: int = 3) -> Dict[str, str]:
        print(f"To filter by Sentence Transformer. Starting with {len(chunks)} chunks")
        print(f"Query: '{query}'")
        
        if not chunks:
            return {}
        
        chunk_keys = list(chunks.keys())
        chunk_texts = list(chunks.values())
        
        query_embedding = self.sentence_model.encode([query])
        chunk_embeddings = self.sentence_model.encode(chunk_texts, show_progress_bar=True)
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1] #[:top_k]
        
        selected_chunks = {}
        similarity_threshold = 0.1
       
        print("Similarity ranking result and samples:") 
        for i, idx in enumerate(top_indices):
            similarity_score = similarities[idx]
            chunk_key = chunk_keys[idx]
            chunk_preview = chunk_texts[idx][:100] + "..." if len(chunk_texts[idx]) > 100 else chunk_texts[idx]
            
            print(f"  {i+1}. {chunk_key}")
            print(f"     Similarity score: {similarity_score:.4f}")
            print(f"     Preview: {chunk_preview}")
            print()
            
            if similarity_score > similarity_threshold:
                selected_chunks[chunk_key] = chunk_texts[idx]
        
        print(f"Chunks filtered by semantic: {len(selected_chunks)} (thresholding at: {similarity_threshold})")
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
                        - "net sales"
                        - "total sales"

                        Extract the numerical value (with units like millions, thousands, etc.) associated with these terms.
                        If multiple values are found, report all.
                        If no revenue terms are found, respond with "No revenue information found".
                        If only revenue related information found, but no exact number found, respond with "No exact revenue number found".
                        # Otherwise, don't reply with empty, reply a short summary.
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"Extract revenue information from this SEC filing text:\n\n{text[:4000]}"
                    }
                ],
                max_completion_tokens=2000,
                reasoning_effort="medium"
            )
            
            return response.choices[0].message.content  #.strip()
        
        except Exception as e:
            return f"API error: {str(e)}"
    
    def process_document(self, document: Dict) -> Dict[str, str]:
        filename = document.get('filename')
        print(f"\n Processing file: {filename}")

        print("Step 1: chunk by section...")
        init_chunks = self.chunk_by_sections(document)

        if not init_chunks:
            print(f"  No section found in {filename}")
            return {filename: "No section data found"}

        print(f"Starting with {len(init_chunks)} chunks")

        print("\nStep 2: token, TFIDF filtering...")
        tfidf_filtered = self.filter_chunks_by_tokens_and_tfidf(init_chunks)

        if not tfidf_filtered:
            print(f"  No chunks found in {filename} after step 2")
            return {filename: "No chunks found"}

        print(f"Found {len(tfidf_filtered)} chunks after step 2")

        print("\nStep 3: Sentence Transformer filtering...")
        semantically_filtered = self.semantic_filter_with_sentence_transformer(
            tfidf_filtered, 
            "total revenue of 2018", 
            top_k=3
        )
        
        if not semantically_filtered:
            print(f"  No chunks found in {filename} after step 3")
            return {filename: "No chunks found"}
        print(f"Found {len(semantically_filtered)} chunks after step 3")

        results = {}

        # processing by sections 
        for section_name, content in semantically_filtered.items():
            if content:
                print(f"  processing {section_name}... ") #with content ==> {content}")

                revenue_info = self.extract_revenue_with_openai(content)
                results[f"{filename}_{section_name}"] = revenue_info
                for key, value in results.items():
                    print(f"  {key}: {value}")

                # concurrency control with API rate limit 
                time.sleep(1)
            else:
                results[f"{filename}_{section_name}"] = "section not valid"        
        
        return results
    
    def run_pipeline(self, n_files):
        print("Starting EDGAR RAG Pipeline...")
        
        if not self.load_edgar_data():
            return {}
        test_data_2018 = self.get_test_data_2018()
        if not test_data_2018:
            print("No data found in 2018")
            return {}
        
        all_results = {}
        
        for i, document in enumerate(test_data_2018[:n_files]):
            print(f"\n=== Reading files {i+1}/{min(n_files, len(test_data_2018))} ===")
            
            results = self.process_document(document)
            filename = document.get('filename', f'doc_{i}')
            all_results[filename] = results
            
            print(f"\n Retrieval results for file {filename} :")
            for key, value in results.items():
                print(f"  {key}: {value}")
        
        return all_results

def main():
    API_KEY = os.getenv("OPEN_API_KEY")
 
    if API_KEY == "your-openai-api-key-here":
        print("OpenAI API Key needed!")
        return
    
    pipeline = EdgarRAGPipeline(API_KEY)
    results = pipeline.run_pipeline(n_files = 10)
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    
    for filename, file_results in results.items():
        print(f"\nFile: {filename}")
        for section_key, revenue_info in file_results.items():
            print(f"  {section_key}: {revenue_info}")

if __name__ == "__main__":
    main()
