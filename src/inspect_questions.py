import pandas as pd
from config import PROCESSED_DIR

path = PROCESSED_DIR / "officeqa_questions_2022_2025.csv"
df = pd.read_csv(path)

print(df[["question", "answer", "source_files", "year", "month"]].head(20))
print("\nCounts by year:")
print(df["year"].value_counts().sort_index())
