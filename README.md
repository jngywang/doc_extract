# doc_extract

`doc_extract` is an EDGAR document-extraction component of a broader research project: **Developing an AI-agent-based system to construct large-scale multimodal training datasets for public-safety applications**. Development of the broader project began in **September 2025**.

The long-term project aims to build scalable AI agents that reconstruct natural and environmental disasters, transportation and infrastructure incidents, and human-caused or technological emergencies from decades of publicly available records. It will connect authoritative event records with relevant text, imagery, video, structured data, and time-series observations using transformer encoders, geospatial-temporal matching, cross-modal similarity analysis, and source-quality checks.

Within that larger effort, this repository provides an early, text-focused extraction workflow. It retrieves relevant passages from SEC EDGAR filing data and uses an LLM to convert selected passages into structured feature values. The current implementation focuses on the financial-document domain and serves as a reusable pattern for evidence retrieval, filtering, provenance-aware extraction, and structured dataset construction.

> **Current scope:** This repository extracts information from SEC EDGAR filings; it does not yet perform disaster reconstruction or multimodal public-safety event matching.

## What it does

The pipeline processes sectioned EDGAR documents for a selected year and extracts the following features:

- **REVENUE** — annual total revenue / net sales information;
- **LOSS** — annual total or net loss information;
- **INDUSTRY** — an industry, sector, SIC/NAICS, or business-activity description.

For each document, the workflow:

1. Loads JSON EDGAR records listed in the `data` manifest.
2. Selects records associated with the requested year (currently **1993–2020**).
3. Splits populated `section_*` fields into line-level chunks.
4. Removes very short chunks and applies TF-IDF keyword filtering.
5. Uses `all-MiniLM-L6-v2` sentence embeddings and cosine-similarity thresholds to retain semantically relevant chunks.
6. Uses the OpenAI API in parallel PySpark partitions to extract a JSON analysis and a feature value from the retained evidence.
7. Writes detailed results to `feature_extraction_results` and a tabular summary to `feature_extraction_results.csv`.

## Repository layout

```text
.
├── data                           # Manifest of JSON data paths (one path per line)
├── feature_extraction_results     # Example detailed extraction output
├── feature_extraction_results.csv # Example structured extraction output
└── src/
    ├── edgar_rag_pipeline.py      # Retrieval, filtering, embedding, and LLM pipeline
    ├── feature_extraction.py      # Command-line entry point and result writers
    ├── prompt.py                  # Feature-specific LLM prompts and JSON schemas
    └── term_dict.py               # Keywords and semantic-similarity thresholds
```

## Requirements

- Python 3.9+ (recommended)
- Java runtime compatible with PySpark
- An OpenAI API key
- Access to the EDGAR JSON files referenced by `data`

Install the Python dependencies:

```bash
pip install openai datasets numpy sentence-transformers scikit-learn pyspark findspark requests
```

The first run may download the `all-MiniLM-L6-v2` sentence-transformer model.

## Data manifest

`data` is a plain-text manifest, not a directory. Add one JSON path per line; blank lines and lines starting with `#` are ignored. Paths can include a `{year}` placeholder:

```text
/path/to/edgar/{year}/test.json
/path/to/edgar/{year}/train.json
```

The pipeline assigns each file's Hugging Face dataset split name from the parent directory in its path. The current implementation reads the `test` split, so ensure that the manifest produces a `test` entry (for example, a path ending in `test/test.json`). Each document should include a `filename` field and one or more non-empty fields named `section_*`.

## Configuration

Set the API key in the environment variable used by the current code:

```bash
export OPEN_API_KEY="your-openai-api-key"
```

> The variable name is intentionally shown as `OPEN_API_KEY` because that is the name currently read by `src/feature_extraction.py`.

Feature keywords and semantic thresholds are defined in `src/term_dict.py`. The extraction prompts and expected JSON response formats are defined in `src/prompt.py`.

## Run the pipeline

Run the command from the `src` directory so the repository's relative paths resolve as expected:

```bash
cd src
python feature_extraction.py 2018
```

Replace `2018` with any year from **1993** through **2020**. The entry point currently processes up to 10 documents and enables all three supported features:

```python
key_options = ["REVENUE", "LOSS", "INDUSTRY"]
```

Results are written to the repository root:

- `feature_extraction_results` contains chunk-level analyses;
- `feature_extraction_results.csv` contains `Feature`, `Filename`, `Year`, and `Feature_Value` columns;
- timestamped logs are created under `src/logs/`.

## Method and broader research relevance

The broader public-safety dataset initiative is intended to create event-centered, multimodal records that capture:

- conditions surrounding an incident;
- response measures and deployed resources; and
- measurable outcomes, including casualties, property and infrastructure damage, response expenditures, service disruptions, environmental impacts, and recovery time.

This repository contributes a narrow but important capability: turning large collections of public documents into screened, structured, model-ready evidence. Its retrieval-and-extraction design can be adapted to authoritative reports, maintenance and inspection records, transportation and environmental observations, news archives, and other properly accessible public sources used by future public-safety data agents.

## Responsible use

Use only lawfully accessible public data and comply with source terms, licenses, rate limits, privacy obligations, and applicable regulations. LLM-generated values should be treated as candidate extractions and validated against the original filing text before use in analytical, operational, or safety-critical settings.

## Status

This is an active research prototype. The code and documentation describe the current EDGAR-focused implementation and are expected to evolve as the broader multimodal public-safety dataset system develops.
