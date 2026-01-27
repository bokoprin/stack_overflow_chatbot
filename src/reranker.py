import os
import re

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        load_dotenv()
        self.model_name = os.getenv("CROSS_ENCODER_MODEL", "").strip()
        self.device = os.getenv("CROSS_ENCODER_DEVICE")
        self.ce_weight = float(os.getenv("RERANK_CE_WEIGHT", "0.6"))
        self.base_weight = float(os.getenv("RERANK_BASE_WEIGHT", "0.2"))
        self.accepted_weight = float(os.getenv("RERANK_ACCEPTED_WEIGHT", "0.15"))
        self.score_weight = float(os.getenv("RERANK_SCORE_WEIGHT", "0.1"))
        self.tag_weight = float(os.getenv("RERANK_TAG_WEIGHT", "0.1"))
        self._model = None

        if self.model_name:
            if self.device:
                self._model = CrossEncoder(self.model_name, device=self.device)
            else:
                self._model = CrossEncoder(self.model_name)

    def rerank(self, query, results, tag_hints=None):
        if not results:
            return results

        base_scores = _normalize_scores([_base_score(r) for r in results])
        meta_scores = [_meta_score(r, query, tag_hints, self.accepted_weight, self.score_weight, self.tag_weight) for r in results]

        ce_scores = None
        if self._model:
            pairs = [(query, r.get("document") or "") for r in results]
            raw = self._model.predict(pairs)
            ce_scores = _normalize_scores(list(map(float, raw)))

        ranked = []
        for idx, item in enumerate(results):
            score = (
                self.base_weight * base_scores[idx]
                + meta_scores[idx]
            )
            if ce_scores is not None:
                score += self.ce_weight * ce_scores[idx]
            ranked.append((score, item))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked]


def _base_score(item):
    if item.get("hybrid_score") is not None:
        return float(item.get("hybrid_score") or 0.0)
    if item.get("distance") is not None:
        return 1.0 - float(item.get("distance") or 0.0)
    if item.get("bm25_score") is not None:
        return float(item.get("bm25_score") or 0.0)
    return 0.0


def _normalize_scores(scores):
    if not scores:
        return []
    min_v = min(scores)
    max_v = max(scores)
    if max_v == min_v:
        return [1.0 for _ in scores]
    return [(s - min_v) / (max_v - min_v) for s in scores]


def _meta_score(item, query, tag_hints, accepted_weight, score_weight, tag_weight):
    meta = item.get("metadata") or {}
    score = 0.0
    if meta.get("is_accepted"):
        score += accepted_weight
    raw_score = meta.get("score")
    if raw_score is not None:
        score += score_weight * min(float(raw_score), 20.0) / 20.0

    tags_text = (meta.get("tags") or "").lower()
    tags = {t.strip() for t in tags_text.split(",") if t.strip()}
    hints = set(tag_hints or [])
    if not hints:
        hints = set(re.findall(r"[a-zA-Z0-9_]+", (query or "").lower()))
    if tags and hints and tags.intersection(hints):
        score += tag_weight
    return score
