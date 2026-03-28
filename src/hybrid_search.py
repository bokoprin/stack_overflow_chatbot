import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, ids, documents, metadatas):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.tokens = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query, top_k=5):
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            results.append(
                {
                    "id": self.ids[idx],
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "bm25_score": float(score),
                }
            )
        return results

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


class HybridSearcher:
    def __init__(self, retriever, cache_path=None):
        self.retriever = retriever
        if cache_path:
            self.cache_path = Path(cache_path)
        else:
            collection_name = getattr(retriever.collection, "name", "default")
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
            default_path = f"data/processed/bm25_cache_{safe_name}.pkl"
            self.cache_path = Path(os.getenv("BM25_CACHE_PATH", default_path))
        self._bm25 = None

    def _load_bm25(self):
        if self._bm25 is not None:
            return self._bm25

        collection = self.retriever.collection
        count = collection.count()
        refresh = os.getenv("BM25_CACHE_REFRESH", "false").lower() in {"1", "true", "yes", "on"}
        if self.cache_path.exists():
            with self.cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                not refresh
                and cached.get("count") == count
                and cached.get("collection_name") == getattr(collection, "name", None)
                and cached.get("persist_dir") == getattr(self.retriever, "persist_dir", None)
            ):
                self._bm25 = BM25Index(
                    cached["ids"], cached["documents"], cached["metadatas"]
                )
                return self._bm25

        data = collection.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        self._bm25 = BM25Index(ids, documents, metadatas)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "collection_name": getattr(collection, "name", None),
                    "persist_dir": getattr(self.retriever, "persist_dir", None),
                    "count": count,
                    "ids": ids,
                    "documents": documents,
                    "metadatas": metadatas,
                },
                handle,
            )
        return self._bm25

    def search(self, query, top_k=5, alpha=0.5, candidate_multiplier=3, filter_fn=None, where=None):
        bm25 = self._load_bm25()
        candidate_k = max(top_k * candidate_multiplier, top_k)
        vector_results = self.retriever.search(query, top_k=candidate_k, where=where)
        bm25_results = bm25.search(query, top_k=candidate_k)

        if filter_fn:
            vector_results = [r for r in vector_results if filter_fn(r)]
            bm25_results = [r for r in bm25_results if filter_fn(r)]

        vector_scores = {item["id"]: 1.0 - float(item.get("distance", 0.0)) for item in vector_results}
        bm25_scores = {item["id"]: float(item.get("bm25_score", 0.0)) for item in bm25_results}

        vector_norm = _normalize(vector_scores)
        bm25_norm = _normalize(bm25_scores)

        merged = {}
        for item in vector_results:
            merged[item["id"]] = {
                **item,
                "bm25_score": bm25_scores.get(item["id"], 0.0),
            }
        for item in bm25_results:
            merged.setdefault(
                item["id"],
                {
                    "id": item["id"],
                    "document": item["document"],
                    "metadata": item["metadata"],
                    "distance": None,
                    "bm25_score": item.get("bm25_score", 0.0),
                },
            )

        for item_id, item in merged.items():
            v_score = vector_norm.get(item_id, 0.0)
            b_score = bm25_norm.get(item_id, 0.0)
            item["hybrid_score"] = alpha * b_score + (1 - alpha) * v_score

        ranked = sorted(merged.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return ranked[:top_k]


def _normalize(scores):
    if not scores:
        return {}
    values = list(scores.values())
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
