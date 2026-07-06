# OfficeQA Financial RAG Challenge

This project builds a Baseline and Engineered RAG system for the Databricks OfficeQA dataset.

## Goal

Use U.S. Treasury Bulletin records to answer financial questions and compare:

- Baseline RAG
- Engineered RAG with metadata filtering and better chunking

## Dataset

The answer key comes from:

```python
from datasets import load_dataset

ds = load_dataset(
    "databricks/officeqa",
    data_files="officeqa_full.csv",
    split="train"
)
```

The RAG corpus should come from the Treasury Bulletin `.txt` / Markdown files from the Databricks OfficeQA source.

## Setup

```bash
git clone <your-repo-url>
cd officeqa-rag-challenge
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

pip install -r requirements.txt
cp .env.example .env
```

## Recommended Run Order

### 1. Load and filter benchmark questions

```bash
python src/load_answer_key.py
```

This creates:

```text
data/processed/officeqa_questions_2022_2025.csv
```

### 2. Add Treasury Bulletin text files

Put `.txt` files into:

```text
data/raw/treasury_txt/
```

Expected filename style:

```text
treasury_bulletin_2024_01.txt
treasury_bulletin_2023_06.txt
```

### 3. Build baseline ChromaDB index

```bash
python src/build_index.py --mode baseline
```

### 4. Build engineered ChromaDB index

```bash
python src/build_index.py --mode engineered
```

### 5. Run evaluation

```bash
python src/run_eval.py --mode baseline
python src/run_eval.py --mode engineered
```

Outputs go to:

```text
results/
```

## Architecture

### Baseline

- ChromaDB vector database
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk size: 1000 characters
- Chunk overlap: 100 characters
- Retrieval: top K=5 similarity search
- No metadata filtering

### Engineered

- ChromaDB vector database
- Embedding model: `BAAI/bge-small-en-v1.5`
- Chunk size: 512 tokens approx.
- Chunk overlap: 100 tokens approx.
- Metadata required:
  - `year`
  - `month`
  - `source_file`
- Retrieval:
  - First filters by year/month when available
  - Then retrieves top K=5

## Metrics

### Search / Retriever Metrics

- Hit Rate@5
- MRR
- Recall proxy

### Generator Metrics

- Groundedness
- Factual Accuracy
- Hallucination Rate

The generator metrics are implemented with lightweight helper functions. You may improve them using an LLM judge.

## Discussion Board Template

Name: Chikezie Okoli | Recent Years Used: 2022, 2023, 2024, 2025

GitHub Link: [Insert GitHub link]

| Metric | Baseline | Engineered |
|---|---:|---:|
| Hit Rate@5 | results | results |
| MRR | results | results |
| Groundedness | results | results |
| Factual Accuracy | results | results |
| Hallucination Rate | results | results |

### Reflection

The bottleneck in the baseline was likely retrieval. A low Hit Rate@5 or MRR means the correct Treasury Bulletin file was not appearing near the top of the results. After adding year/month metadata, retrieval improved because the search space became smaller and more targeted.

If scaling to the full 1939–2025 archive, the first component likely to become slow would be retrieval/indexing because the chunk count would grow significantly. A production solution would need stronger metadata partitioning, hybrid search, reranking, and possibly a more scalable vector database.
