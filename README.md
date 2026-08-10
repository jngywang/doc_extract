# doc_extract

`doc_extract` is part of the project **Developing an AI-agent-based system to construct large-scale multimodal training datasets for public-safety applications**, initiated in September 2025.

The long-term project aims to build scalable AI agents that reconstruct public-safety events in multimodality. The agents will identify and reconstruct events from decades of publicly available records. The resulting event-centered datasets are intended to support training and decision-support research on public-safety events and responding strategies.

This repository focuses on one of the central technical challenges required by that larger system:

> **How can an AI agent extract the correct structured fact from a large text collection with high precision, while preserving the evidence needed to verify the result?**

The approach is designed for precise extraction from **any large-scale text corpus**. The current implementation provides a prototype architecture for locating relevant evidence, eliminating plausible but incorrect alternatives, producing structured outputs, and retaining document- and passage-level traceability.

## Technical objective

A target value may appear alongside many misleading alternatives. Long documents and document collections often contain:

- values associated with different dates, entities, locations, versions, or reporting periods;
- partial values that should not be treated as totals or event-level facts;
- repeated statements, summaries, tables, footnotes, and narrative references;
- conflicting terminology, formats, units, or levels of aggregation;
- semantically related passages that are not valid evidence for the requested field; and
- statements that mention a target concept but do not support a precise extractable value.

For public-safety event reconstruction, similar problems arise when extracting casualties, damage estimates, infrastructure impacts, response resources, costs, service disruptions, environmental measures, or recovery indicators from large and heterogeneous document collections.

The goal is therefore not merely to retrieve text that appears relevant. The goal is to identify the **right evidence**, determine whether it satisfies the extraction criteria, return a normalized structured value only when warranted, and make the result auditable against its supporting passages.

## Method

The pipeline uses staged retrieval and constrained extraction rather than relying on a model to interpret an entire document collection at once.

1. **Document segmentation**  
   Large documents are divided into traceable evidence candidates.

2. **Lexical candidate filtering**  
   Short or clearly irrelevant passages are removed. Target-specific terms identify an initial set of potentially relevant candidates.

3. **Semantic relevance filtering**  
   A sentence-transformer encoder generates embeddings for target concepts and candidate passages. Cosine similarity is used to retain passages that are semantically related to the requested field, including relevant language that may not exactly match a keyword.

4. **Target-specific thresholds**  
   Similarity thresholds can be configured separately for different extraction targets, allowing the retrieval stage to balance recall and precision according to the characteristics of each target.

5. **Constrained LLM extraction**  
   Only the reduced evidence set is submitted to an LLM. Target-specific prompts instruct the model to:
   - extract the requested information;
   - distinguish complete values from partial, ambiguous, or mismatched values;
   - return `Not found` when the available evidence is insufficient; and
   - produce structured JSON with both an analysis and a candidate value.

6. **Parallel processing**  
   PySpark distributes extraction work across partitions so that large candidate sets can be processed concurrently.

7. **Structured outputs and logs**  
   The pipeline writes detailed extraction analyses, document-level structured values, and runtime logs to support review, error analysis, and future evaluation.

## Current capabilities

The current prototype implements:

- configurable target-specific keyword dictionaries;
- TF-IDF-based lexical candidate selection;
- semantic filtering with the `all-MiniLM-L6-v2` sentence-transformer model;
- cosine-similarity thresholding;
- target-specific prompts with required JSON response schemas;
- OpenAI API-based extraction;
- PySpark-based parallel execution;
- detailed extraction output and CSV summaries; and
- timestamped runtime logging.

The included feature configurations are examples of how a target can be represented through keywords, semantic thresholds, and extraction instructions. The same architecture can be extended to other text-based facts by defining new targets and evaluation criteria.
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

Configure target-specific keywords and semantic thresholds in `src/term_dict.py`. Configure extraction rules and JSON response schemas in `src/prompt.py`.

To adapt the pipeline to a new extraction target:

1. define a target name;
2. add a domain-appropriate keyword dictionary;
3. set and evaluate a semantic similarity threshold;
4. create a constrained prompt that specifies what counts as valid evidence and what should be rejected; and
5. validate extraction quality against source passages and task-specific ground truth.

## Run

Run from the `src` directory so the existing relative paths resolve correctly:

```bash
cd src
python feature_extraction.py 2018
```

The current command-line interface accepts a year argument and processes up to 10 documents per run.

## Outputs

The pipeline produces:

- `feature_extraction_results`  
  Detailed chunk-level extraction analyses.

- `feature_extraction_results.csv`  
  A structured summary containing:

  ```text
  Feature, Filename, Year, Feature_Value
  ```

- `src/logs/edgar_pipeline_<timestamp>.log`  
  Runtime logs, including retrieval and filtering information.

## Research direction

The main research question is how to make automated extraction **precise, scalable, and auditable** when the correct evidence is sparse and the surrounding text contains many plausible but incorrect alternatives.

Planned directions include:

- extraction targets beyond the current examples;
- improved normalization of units, dates, entities, locations, and provenance;
- confidence scoring and cross-passage consistency checks;
- systematic evaluation of false positives and false negatives;
- agentic retrieval across multiple authoritative text sources;
- contradiction detection and evidence-ranking methods; and
- integration with image, video, geospatial, and time-series evidence for multimodal public-safety event reconstruction.

## Responsible use

Use only lawfully accessible data and comply with applicable licenses, source terms, rate limits, privacy obligations, and regulations. Extracted values are candidate outputs and should be validated against underlying source material before use in high-stakes, operational, or safety-critical contexts.

## Status

Active research prototype. This repository concentrates on high-precision text extraction as a foundational capability for the broader multimodal public-safety dataset construction system.
