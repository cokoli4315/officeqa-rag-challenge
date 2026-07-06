from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TREASURY_TXT_DIR = RAW_DIR / "treasury_txt"

RESULTS_DIR = ROOT_DIR / "results"

CHROMA_DIR = ROOT_DIR / "chroma_db"

YEARS = [2022, 2023, 2024, 2025]

K = 5

BASELINE_COLLECTION = "officeqa_baseline"
ENGINEERED_COLLECTION = "officeqa_engineered"

BASELINE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENGINEERED_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
