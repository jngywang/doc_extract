import os
import argparse
from edgar_rag_pipeline import EdgarRAGPipeline

def main():
    API_KEY = os.getenv("OPEN_API_KEY")
    if API_KEY == "your-openai-api-key-here":
        print("OpenAI API Key needed!")
        return

    parser = argparse.ArgumentParser(description='year of file')
    parser.add_argument('year', type=int, help='years in 1900-2030')
    parser.add_argument('--format', default='YYYY')
    args = parser.parse_args()
    if args.year < 1993 or args.year > 2020:
        print("Files are in years 1993-2020")
        return
    year = str(args.year)

    # key_options: ["REVENUE", "LOSS", "INDUSTRY"]
    key_options = ["REVENUE", "LOSS"]
    key_options = ["REVENUE", "LOSS", "INDUSTRY"]
    pipeline = EdgarRAGPipeline(API_KEY, year, key_options)
    results, values = pipeline.run_pipeline(n_files = 10)

    with open("../feature_extraction_results", "w") as f:
        f.write("\n" + "="*60)
        f.write("FINAL ANALYSIS:")
        f.write("="*60)

        for feature in key_options:
            for filename, file_results in results.items():
                feature_results = file_results.get(feature, {})

                f.write(f"\n=== {feature} Results for file {filename}===\n")
                f.write(f"Number of processed chunks: {len(feature_results)}\n")
            
                for chunk_key, chunk_result in feature_results.items():
                        f.write(f"--{chunk_key}: {chunk_result}\n")

        f.write("\n" + "="*60)
        f.write("FINAL VALUES:")
        f.write("="*60)

        for feature in key_options:
            f.write(f"\n=== {feature} Results ===\n")
            for filename, file_results in values.items():
                feature_value = file_results.get(feature, "")
                f.write(f"\n--File: {filename} has {feature} in year {year}: {feature_value}")

if __name__ == "__main__":
    main()
