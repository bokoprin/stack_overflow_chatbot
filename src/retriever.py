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
        self.is_e5 = "e5" in self.embedding_model_name.lower()
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def search(self, query, top_k=5, where=None):
        query_text = _apply_query_prefix(query, self.is_e5)
        embedding = self.model.encode(query_text, normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=_normalize_where(where),
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

    def get_by_question_id(self, question_id, chunk_type=None):
        where = {"question_id": question_id}
        if chunk_type:
            where["chunk_type"] = chunk_type
        results = self.collection.get(
            where=_normalize_where(where),
            include=["documents", "metadatas"],
        )
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        hits = []
        for idx in range(len(ids)):
            hits.append(
                {
                    "id": ids[idx],
                    "document": docs[idx],
                    "metadata": metas[idx],
                    "distance": None,
                }
            )
        return hits


def _normalize_where(where):
    if not where:
        return None
    if len(where) == 1 or any(k.startswith("$") for k in where.keys()):
        return where
    return {"$and": [{k: v} for k, v in where.items()]}


def _apply_query_prefix(query, is_e5):
    if not is_e5:
        return query
    if not query:
        return "query:"
    return f"query: {query}"
