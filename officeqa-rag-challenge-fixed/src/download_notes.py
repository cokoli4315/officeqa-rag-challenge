"""
This project intentionally does not auto-download the full Treasury Bulletin corpus.

Why?
- The OfficeQA benchmark answer key is easy to load with Hugging Face datasets.
- The Treasury Bulletin corpus may require access through the Databricks OfficeQA repository/Hugging Face files.
- For class projects, manually downloading only the needed .txt files is often faster and safer.

Put your .txt files here:

data/raw/treasury_txt/

Expected examples:

treasury_bulletin_2022_01.txt
treasury_bulletin_2023_06.txt
treasury_bulletin_2024_12.txt
treasury_bulletin_2025_01.txt
"""
