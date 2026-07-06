"""
Download Treasury Bulletin text files from the OfficeQA dataset source URLs.
"""
import re
import requests
from pathlib import Path
from datasets import load_dataset
from config import TREASURY_TXT_DIR, YEARS
from bs4 import BeautifulSoup


def extract_filename(source_files: str):
    """Extract treasury bulletin filename from source_files."""
    text = str(source_files)
    match = re.search(r"(treasury_bulletin_\d{4}_\d{2}\.txt)", text)
    if match:
        return match.group(1)
    return None


def extract_year(filename: str):
    """Extract year from filename."""
    match = re.search(r"treasury_bulletin_(\d{4})", filename)
    if match:
        return int(match.group(1))
    return None


def download_from_fraser(url: str):
    """Download Treasury Bulletin document from Fraser URL."""
    try:
        print(f"  Fetching from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML to extract text content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"  Error downloading: {e}")
        return None


def main():
    print("Loading OfficeQA dataset from HuggingFace...")
    ds = load_dataset("databricks/officeqa", "officeqa_full")
    
    # Handle DatasetDict - get the train split
    if isinstance(ds, dict):
        df = ds["train"].to_pandas()
    else:
        df = ds.to_pandas()
    
    # Create treasury_txt directory
    TREASURY_TXT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract unique Treasury files for our years of interest
    unique_files = {}
    for idx, row in df.iterrows():
        source_files = row["source_files"]
        source_docs = row["source_docs"]
        
        filename = extract_filename(source_files)
        if filename:
            year = extract_year(filename)
            if year in YEARS and filename not in unique_files:
                unique_files[filename] = source_docs
    
    print(f"Found {len(unique_files)} unique Treasury Bulletin files for years {YEARS}")
    print(f"Files: {list(unique_files.keys())}\n")
    
    # Download and save each file
    saved_count = 0
    for filename, url in sorted(unique_files.items()):
        output_path = TREASURY_TXT_DIR / filename
        
        if output_path.exists():
            print(f"Already exists: {filename}")
            saved_count += 1
            continue
        
        print(f"Downloading: {filename}")
        content = download_from_fraser(url)
        
        if content:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Saved to: {output_path}")
            saved_count += 1
        else:
            print(f"  Failed to download")
    
    print(f"\nSuccessfully saved {saved_count}/{len(unique_files)} Treasury Bulletin files")


if __name__ == "__main__":
    main()
