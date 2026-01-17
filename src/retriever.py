import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        persist_dir="vector_db",
        collection_name="stack_overflow",
        embedding_model_name=None,
    ):
        load_dotenv()
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-m3"
        )
        self.model = SentenceTransformer(self.embedding_model_name)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def search(self, query, top_k=5):
        embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not results.get("ids") or not results["ids"][0]:
            return []
        hits = []
        for idx in range(len(results["ids"][0])):
            hits.append(
                {
                    "id": results["ids"][0][idx],
                    "document": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": results["distances"][0][idx],
                }
            )
        return hits
