import os
import argparse
from edgar_rag_pipeline import EdgarRAGPipeline

REVENUE_KEYWORDS = [
                    "total revenue",
                    "total revenues",
                    "net revenue",
                    "net revenues",
                    "total net sales",
                    "revenues",
                    "revenue",
                    "net sales",
                    "total sales",
                    "operation revenue"
                ]
TERM_DICT = {
                "revenue": REVENUE_KEYWORDS,
            }


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
    pipeline = EdgarRAGPipeline(API_KEY, year, key_options = ["REVENUE"])
    results, values = pipeline.run_pipeline(n_files = 5)

    pipeline.logger.info("\n" + "="*60)
    pipeline.logger.info("FINAL ANALYSIS:")
    pipeline.logger.info("="*60)
    for filename, file_results in results.items():
        pipeline.logger.info(f"\nFile: {filename}")
        for section_key, revenue_info in file_results.items():
            pipeline.logger.info(f"  {section_key}: {revenue_info}")

    pipeline.logger.info("\n" + "="*60)
    pipeline.logger.info("FINAL VALUES:")
    pipeline.logger.info("="*60)

    for filename, file_results in values.items():
        pipeline.logger.info(f"\nFile: {filename} gets total revenue in year {year}: {file_results}")

if __name__ == "__main__":
    main()
