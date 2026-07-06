import argparse
import pandas as pd
from tqdm import tqdm

from config import PROCESSED_DIR, RESULTS_DIR, K
from retriever import OfficeQARetriever
from generator import generate_answer
from metrics import (
    hit_rate_at_k,
    reciprocal_rank,
    recall_proxy,
    factual_accuracy,
    simple_groundedness,
    hallucination_rate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "engineered"], required=True)
    parser.add_argument(
        "--questions",
        default=str(PROCESSED_DIR / "officeqa_questions_2022_2025.csv"),
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.questions)

    if len(df) == 0:
        raise ValueError("No questions found. Check year filtering.")

    retriever = OfficeQARetriever(args.mode)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        expected_answer = row["answer"]
        expected_source = row["source_files"]
        year = row.get("year")
        month = row.get("month")

        retrieved_docs = retriever.retrieve(
            question=question,
            year=year,
            month=month,
            k=K,
        )

        generated_answer = generate_answer(question, retrieved_docs)

        rows.append(
            {
                "uid": row.get("uid"),
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "expected_source": expected_source,
                "year": year,
                "month": month,
                "hit_rate_at_5": hit_rate_at_k(retrieved_docs, expected_source, K),
                "mrr": reciprocal_rank(retrieved_docs, expected_source),
                "recall_proxy": recall_proxy(retrieved_docs, expected_source, K),
                "groundedness": simple_groundedness(generated_answer, retrieved_docs),
                "factual_accuracy": factual_accuracy(generated_answer, expected_answer),
                "hallucination_rate": hallucination_rate(generated_answer, retrieved_docs),
                "retrieved_sources": [
                    doc["metadata"].get("source_file") for doc in retrieved_docs
                ],
            }
        )

    result_df = pd.DataFrame(rows)

    detail_path = RESULTS_DIR / f"{args.mode}_details.csv"
    summary_path = RESULTS_DIR / f"{args.mode}_summary.csv"

    result_df.to_csv(detail_path, index=False)

    summary = {
        "mode": args.mode,
        "questions": len(result_df),
        "hit_rate_at_5": result_df["hit_rate_at_5"].mean(),
        "mrr": result_df["mrr"].mean(),
        "recall_proxy": result_df["recall_proxy"].mean(),
        "groundedness": result_df["groundedness"].mean(),
        "factual_accuracy": result_df["factual_accuracy"].mean(),
        "hallucination_rate": result_df["hallucination_rate"].mean(),
    }

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\nSummary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nSaved details to {detail_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
