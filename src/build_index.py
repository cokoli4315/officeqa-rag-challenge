import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    TREASURY_TXT_DIR,
    CHROMA_DIR,
    BASELINE_COLLECTION,
    ENGINEERED_COLLECTION,
    BASELINE_EMBEDDING_MODEL,
    ENGINEERED_EMBEDDING_MODEL,
)
from chunking import load_treasury_chunks


def get_settings(mode: str):
    if mode == "baseline":
        return BASELINE_COLLECTION, BASELINE_EMBEDDING_MODEL

    if mode == "engineered":
        return ENGINEERED_COLLECTION, ENGINEERED_EMBEDDING_MODEL

    raise ValueError("mode must be baseline or engineered")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "engineered"], required=True)
    args = parser.parse_args()

    collection_name, embedding_model_name = get_settings(args.mode)

    print(f"Loading chunks for {args.mode}...")
    chunks = load_treasury_chunks(TREASURY_TXT_DIR, args.mode)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)

    batch_size = 64

    for batch_start in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[batch_start: batch_start + batch_size]

        texts = [c.text for c in batch]
        metadatas = [c.metadata for c in batch]
        ids = [
            f"{c.metadata['source_file']}::chunk_{c.metadata['chunk_id']}"
            for c in batch
        ]

        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"Built Chroma collection: {collection_name}")


if __name__ == "__main__":
    main()
