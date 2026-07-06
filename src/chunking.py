import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


def parse_year_month_from_filename(path: Path):
    match = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def simple_char_chunks(text: str, chunk_size: int = 1000, overlap: int = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def tokenish_chunks(text: str, chunk_size: int = 512, overlap: int = 100):
    # Lightweight approximation: split by whitespace.
    # This avoids requiring a tokenizer and works well enough for class projects.
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


def load_treasury_chunks(txt_dir: Path, mode: str) -> List[Chunk]:
    if not txt_dir.exists():
        raise FileNotFoundError(
            f"Missing directory: {txt_dir}. Put Treasury .txt files there."
        )

    files = sorted(txt_dir.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {txt_dir}. Add OfficeQA Treasury Bulletin text files."
        )

    all_chunks: List[Chunk] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        year, month = parse_year_month_from_filename(file_path)

        if mode == "baseline":
            split_texts = simple_char_chunks(text)
        elif mode == "engineered":
            split_texts = tokenish_chunks(text)
        else:
            raise ValueError("mode must be baseline or engineered")

        for i, chunk_text in enumerate(split_texts):
            metadata = {
                "source_file": file_path.name,
                "chunk_id": i,
                "year": year,
                "month": month,
                "mode": mode,
            }

            all_chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=metadata,
                )
            )

    return all_chunks
