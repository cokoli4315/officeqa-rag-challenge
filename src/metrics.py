import re
from typing import List, Dict, Any


def normalize_source(value: Any) -> str:
    return str(value).strip()


def hit_rate_at_k(retrieved_docs: List[Dict], expected_source: str, k: int = 5) -> int:
    expected_source = normalize_source(expected_source)

    retrieved_sources = [
        normalize_source(doc["metadata"].get("source_file"))
        for doc in retrieved_docs[:k]
    ]

    return int(expected_source in retrieved_sources)


def reciprocal_rank(retrieved_docs: List[Dict], expected_source: str) -> float:
    expected_source = normalize_source(expected_source)

    for rank, doc in enumerate(retrieved_docs, start=1):
        source_file = normalize_source(doc["metadata"].get("source_file"))

        if source_file == expected_source:
            return 1.0 / rank

    return 0.0


def recall_proxy(retrieved_docs: List[Dict], expected_source: str, k: int = 5) -> float:
    # OfficeQA gives source file, not every relevant snippet.
    # This proxy checks whether at least one chunk from the expected file appears.
    return float(hit_rate_at_k(retrieved_docs, expected_source, k))


def extract_number(text: str):
    if text is None:
        return None

    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(text))

    if not matches:
        return None

    value = matches[0].replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def factual_accuracy(generated_answer: str, expected_answer: str, tolerance: float = 0.01) -> int:
    expected_num = extract_number(expected_answer)
    generated_num = extract_number(generated_answer)

    if expected_num is None or generated_num is None:
        return int(str(expected_answer).strip().lower() in str(generated_answer).strip().lower())

    if expected_num == 0:
        return int(abs(generated_num - expected_num) <= tolerance)

    relative_error = abs(generated_num - expected_num) / abs(expected_num)
    return int(relative_error <= tolerance)


def simple_groundedness(generated_answer: str, retrieved_docs: List[Dict]) -> float:
    # Lightweight heuristic:
    # If numbers in the answer appear in the retrieved context, treat as grounded.
    answer_numbers = set(re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(generated_answer)))

    if not answer_numbers:
        return 0.0

    context = " ".join(doc["text"] for doc in retrieved_docs)
    supported = sum(1 for num in answer_numbers if num in context)

    return supported / len(answer_numbers)


def hallucination_rate(generated_answer: str, retrieved_docs: List[Dict]) -> float:
    return 1.0 - simple_groundedness(generated_answer, retrieved_docs)
