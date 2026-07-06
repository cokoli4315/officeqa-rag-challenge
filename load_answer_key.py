import re
import pandas as pd
from datasets import load_dataset
from config import PROCESSED_DIR, YEARS


def extract_year_month(source_files: str):
    text = str(source_files)
    match = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "databricks/officeqa",
        data_files="officeqa_full.csv",
        split="train",
    )

    df = ds.to_pandas()

    if "source_files" not in df.columns:
        raise ValueError(f"Expected source_files column. Found: {list(df.columns)}")

    df[["year", "month"]] = df["source_files"].apply(
        lambda x: pd.Series(extract_year_month(x))
    )

    filtered = df[df["year"].isin(YEARS)].copy()

    output_path = PROCESSED_DIR / "officeqa_questions_2022_2025.csv"
    filtered.to_csv(output_path, index=False)

    print(f"Saved {len(filtered)} filtered questions to {output_path}")
    print(filtered["year"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
