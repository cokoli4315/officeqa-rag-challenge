import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    BASELINE_COLLECTION,
    ENGINEERED_COLLECTION,
    BASELINE_EMBEDDING_MODEL,
    ENGINEERED_EMBEDDING_MODEL,
    K,
)


class OfficeQARetriever:
    def __init__(self, mode: str):
        if mode == "baseline":
            self.collection_name = BASELINE_COLLECTION
            self.embedding_model_name = BASELINE_EMBEDDING_MODEL
        elif mode == "engineered":
            self.collection_name = ENGINEERED_COLLECTION
            self.embedding_model_name = ENGINEERED_EMBEDDING_MODEL
        else:
            raise ValueError("mode must be baseline or engineered")

        self.mode = mode
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(self.collection_name)
        self.model = SentenceTransformer(self.embedding_model_name)

    def retrieve(self, question: str, year=None, month=None, k: int = K):
        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
        ).tolist()[0]

        where = None

        if self.mode == "engineered":
            if year is not None and month is not None:
                where = {
                    "$and": [
                        {"year": int(year)},
                        {"month": int(month)},
                    ]
                }
            elif year is not None:
                where = {"year": int(year)}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = []

        for i in range(len(results["documents"][0])):
            docs.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "rank": i + 1,
                }
            )

        return docs
