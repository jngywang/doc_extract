import os
import re
import argparse
import csv
from datetime import datetime
from edgar_rag_pipeline import EdgarRAGPipeline

# features whose extracted value should be reduced to a plain number
NUMERIC_FEATURES = {"REVENUE", "LOSS"}

_EMPTY = {"", "not found", "n/a", "na", "none", "null", "api error", "unknown"}


def to_number(value):
    """Parse an extracted string like '$34.0 million' or '(130,999)' into a
    plain int/float. Returns '' when there is no usable number."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in _EMPTY:
        return ""

    negative = (s.startswith("(") and s.rstrip().endswith(")")) or \
               bool(re.search(r"-\s*\$?\s*\d", s.replace("—", "-")))

    low = s.lower()
    if re.search(r"\b(billion|bn|bln)\b", low) or re.search(r"\d\s*b\b", low):
        scale = 1_000_000_000
    elif re.search(r"\b(million|mm|mn|mln)\b", low) or re.search(r"\d\s*m\b", low):
        scale = 1_000_000
    elif re.search(r"\b(thousand|k)\b", low) or re.search(r"\d\s*k\b", low):
        scale = 1_000
    else:
        scale = 1

    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    if not m:
        return ""
    result = float(m.group(0).replace(",", "")) * scale
    if negative:
        result = -abs(result)
    return int(result) if result == int(result) else result


def norm(feature, value):
    """Numeric features -> plain number; everything else unchanged."""
    return to_number(value) if feature in NUMERIC_FEATURES else value


# ground-truth file and which column backs each feature
GT_PATH = "../ground_truth_2018.csv"
GT_COLUMNS = {"REVENUE": "total_revenue", "LOSS": "net_income_loss"}


def load_gt_lookup(features):
    """filename -> set of features that have a usable ground-truth value.
    Files / features absent here are skipped by the pipeline (treated as
    'not found'), since there is nothing to score them against."""
    lookup = {}
    if not os.path.exists(GT_PATH):
        print(f"WARNING: {GT_PATH} not found - running all features for all files")
        return None
    with open(GT_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            have = set()
            for feat in features:
                col = GT_COLUMNS.get(feat)
                if col and str(row.get(col, "")).strip().lower() not in ("", "none", "nan"):
                    have.add(feat)
            lookup[row["filename"]] = have
    return lookup


def main():
    API_KEY = os.getenv("OPEN_API_KEY")
    if API_KEY == "your-openai-api-key-here":
        print("OpenAI API Key needed!")
        return

    parser = argparse.ArgumentParser(description='year of file')
    parser.add_argument('year', type=int, help='years in 1900-2030')
    parser.add_argument('--format', default='YYYY')
    parser.add_argument('--n_files', type=int, default=2, help='number of files to process')
    args = parser.parse_args()
    if args.year < 1993 or args.year > 2020:
        print("Files are in years 1993-2020")
        return
    year = str(args.year)

    # available: ["REVENUE", "LOSS", "INDUSTRY"]
    key_options = ["REVENUE", "LOSS"]
    gt_lookup = load_gt_lookup(key_options)
    pipeline = EdgarRAGPipeline(API_KEY, year, key_options)
    results, values = pipeline.run_pipeline(n_files = args.n_files, gt_lookup = gt_lookup)

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
                feature_value = norm(feature, file_results.get(feature, ""))
                f.write(f"\n--File: {filename} has {feature} in year {year}: {feature_value}")

        f.write("\n" + "="*60)
        f.write("PROCESSING TIME PER FILE:")
        f.write("="*60 + "\n")
        for filename, elapsed in pipeline.file_timings.items():
            f.write(f"\n--File: {filename} took {elapsed:.2f} seconds")
        f.write(f"\n--Total: {sum(pipeline.file_timings.values()):.2f} seconds\n")

    with open("../feature_extraction_results.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Filename', 'Year', 'Feature_Value'])

        for feature in key_options:
            for filename, file_results in values.items():
                feature_value = norm(feature, file_results.get(feature, ""))
                writer.writerow([feature, filename, year, feature_value])

    # compact result file: filename, processing time, final value(s) only
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"../result_{timestamp}.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Processing_Time_Sec'] + key_options)
        for filename, file_results in values.items():
            row = [filename, f"{pipeline.file_timings.get(filename, 0):.2f}"]
            row += [norm(feature, file_results.get(feature, "")) for feature in key_options]
            writer.writerow(row)


if __name__ == "__main__":
    main()
