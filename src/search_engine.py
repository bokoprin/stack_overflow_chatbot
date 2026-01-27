import os

from hybrid_search import HybridSearcher
from mmr import mmr_select


class SearchEngine:
    def __init__(
        self,
        retriever,
        query_processor,
        hybrid_alpha=0.5,
        candidate_multiplier=3,
        parent_child=None,
    ):
        self.retriever = retriever
        self.query_processor = query_processor
        self.hybrid_alpha = hybrid_alpha
        self.candidate_multiplier = candidate_multiplier
        if parent_child is None:
            self.parent_child = os.getenv("PARENT_CHILD_MODE", "true").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self.parent_child = parent_child
        self.hybrid_searcher = HybridSearcher(retriever)

    def search(self, query, top_k, fallback_en=None, filters=None):
        filters = filters or {}
        queries = self.query_processor.build_search_queries(query, fallback_en=fallback_en)
        search_query = queries[-1] if queries else query
        effective_top_k = self.query_processor.adjust_top_k(
            query,
            top_k,
            min_k=int(os.getenv("TOP_K_MIN", "3")),
            max_k=int(os.getenv("TOP_K_MAX", "10")),
        )

        where = {}
        if self.parent_child and not self.query_processor.uses_dual():
            where["chunk_type"] = "qa_parent"
        if filters.get("language"):
            where["language"] = filters["language"]

        require_parent = self.parent_child and not self.query_processor.uses_dual()
        filter_fn = _build_filter_fn(filters, require_parent=require_parent)

        results = []
        if self.query_processor.uses_dual():
            results = _dual_search(
                self.retriever,
                queries,
                effective_top_k,
                filter_fn,
                where or None,
            )
        elif self.query_processor.uses_tag_split() and filters.get("tags"):
            results = _tag_split_search(
                self.retriever,
                queries,
                effective_top_k,
                filter_fn,
                where or None,
                filters,
            )
        elif self.query_processor.uses_multistage():
            results = _multistage_search(
                queries,
                self.retriever,
                self.hybrid_searcher if self.query_processor.uses_hybrid() else None,
                effective_top_k,
                self.hybrid_alpha,
                self.candidate_multiplier,
                filter_fn,
                where or None,
            )
        else:
            if self.query_processor.uses_hybrid():
                results = self.hybrid_searcher.search(
                    search_query,
                    top_k=effective_top_k,
                    alpha=self.hybrid_alpha,
                    candidate_multiplier=self.candidate_multiplier,
                    filter_fn=filter_fn,
                    where=where or None,
                )
            else:
                results = self.retriever.search(
                    search_query, top_k=effective_top_k, where=where or None
                )
                results = _apply_filter_fn(results, filter_fn)

        if self.query_processor.uses_dual():
            pass
        elif self.parent_child:
            if self.query_processor.uses_mmr():
                results = _apply_mmr(results, effective_top_k)
            results = _expand_children(self.retriever, results, filters)
        else:
            if self.query_processor.uses_mmr():
                results = _apply_mmr(results, effective_top_k)

        results = self.query_processor.rerank_results(
            results, search_query, tag_hints=filters.get("tags")
        )
        return search_query, results


def _build_filter_fn(filters, require_parent=False):
    tags = [t.strip().lower() for t in (filters.get("tags") or []) if t.strip()]
    language = filters.get("language")
    min_score = filters.get("min_score")
    accepted_only = filters.get("accepted_only", False)

    def _match(item):
        meta = item.get("metadata") or {}
        if require_parent and meta.get("chunk_type") != "qa_parent":
            return False
        if language and meta.get("language") != language:
            return False
        if min_score is not None:
            raw = meta.get("score")
            if raw is None:
                raw = meta.get("answer_score")
            if raw is None or float(raw) < float(min_score):
                return False
        if accepted_only and not meta.get("is_accepted") and not meta.get("has_accepted"):
            return False
        if tags:
            meta_tags = {
                t.strip().lower() for t in (meta.get("tags") or "").split(",") if t.strip()
            }
            if not meta_tags.intersection(tags):
                return False
        return True

    return _match


def _apply_filter_fn(results, filter_fn):
    if not results:
        return results
    return [item for item in results if filter_fn(item)]


def _expand_children(retriever, parent_results, filters):
    if not parent_results:
        return []
    question_ids = []
    for item in parent_results:
        meta = item.get("metadata") or {}
        qid = meta.get("question_id")
        if qid is not None and qid not in question_ids:
            question_ids.append(qid)

    children = []
    for qid in question_ids:
        children.extend(retriever.get_by_question_id(qid, chunk_type="answer_child"))

    filter_fn = _build_filter_fn(filters, require_parent=False)
    children = _apply_filter_fn(children, filter_fn)
    if filters.get("accepted_only"):
        accepted = [c for c in children if (c.get("metadata") or {}).get("is_accepted")]
        if accepted:
            children = accepted
    return children


def _multistage_search(
    queries,
    retriever,
    hybrid_searcher,
    top_k,
    hybrid_alpha,
    candidate_multiplier,
    filter_fn,
    where,
):
    merged = {}
    for query in queries:
        if hybrid_searcher:
            results = hybrid_searcher.search(
                query,
                top_k=top_k,
                alpha=hybrid_alpha,
                candidate_multiplier=candidate_multiplier,
                filter_fn=filter_fn,
                where=where,
            )
        else:
            results = retriever.search(query, top_k=top_k, where=where)
            results = _apply_filter_fn(results, filter_fn)
        for item in results:
            item_id = item.get("id")
            if item_id in merged:
                merged[item_id] = _prefer_item(merged[item_id], item)
            else:
                merged[item_id] = item
    return list(merged.values())


def _prefer_item(a, b):
    score_a = _base_score(a)
    score_b = _base_score(b)
    return a if score_a >= score_b else b


def _base_score(item):
    if item.get("hybrid_score") is not None:
        return float(item.get("hybrid_score") or 0.0)
    if item.get("distance") is not None:
        return 1.0 - float(item.get("distance") or 0.0)
    if item.get("bm25_score") is not None:
        return float(item.get("bm25_score") or 0.0)
    return 0.0


def _apply_mmr(results, top_k):
    lambda_param = float(os.getenv("MMR_LAMBDA", "0.7"))
    max_candidates = int(os.getenv("MMR_MAX_CANDIDATES", "30"))
    return mmr_select(results, top_k=top_k, lambda_param=lambda_param, max_candidates=max_candidates)


def _dual_search(retriever, queries, top_k, filter_fn, where):
    q_where = _merge_where(where, {"chunk_type": "question_only"})
    a_where = _merge_where(where, {"chunk_type": "answer_only"})
    q_results = []
    a_results = []
    for query in queries:
        q_results.extend(retriever.search(query, top_k=top_k, where=q_where))
        a_results.extend(retriever.search(query, top_k=top_k, where=a_where))

    q_results = _apply_filter_fn(q_results, filter_fn)
    a_results = _apply_filter_fn(a_results, filter_fn)

    q_weight = float(os.getenv("DUAL_Q_WEIGHT", "0.55"))
    a_weight = float(os.getenv("DUAL_A_WEIGHT", "0.45"))

    merged = {}
    for item in q_results:
        merged[item["id"]] = _weighted_item(item, q_weight)
    for item in a_results:
        if item["id"] in merged:
            merged[item["id"]] = _merge_weighted(merged[item["id"]], _weighted_item(item, a_weight))
        else:
            merged[item["id"]] = _weighted_item(item, a_weight)
    return sorted(merged.values(), key=lambda x: x.get("weighted_score", 0.0), reverse=True)[:top_k]


def _tag_split_search(retriever, queries, top_k, filter_fn, where, filters):
    tag_filters = dict(filters)
    tag_filters["tags"] = filters.get("tags")
    strict_filter_fn = _build_filter_fn(tag_filters, require_parent=False)

    tag_results = []
    for query in queries:
        results = retriever.search(query, top_k=top_k, where=where)
        tag_results.extend(_apply_filter_fn(results, strict_filter_fn))

    if len(tag_results) >= top_k:
        return tag_results[:top_k]

    global_results = []
    for query in queries:
        global_results.extend(retriever.search(query, top_k=top_k, where=where))
    global_results = _apply_filter_fn(global_results, filter_fn)

    merged = {}
    for item in global_results:
        merged[item["id"]] = _weighted_item(item, 1.0)
    for item in tag_results:
        merged[item["id"]] = _weighted_item(item, 1.2)

    ranked = sorted(merged.values(), key=lambda x: x.get("weighted_score", 0.0), reverse=True)
    return ranked[:top_k]


def _weighted_item(item, weight):
    score = _base_score(item) * weight
    payload = dict(item)
    payload["weighted_score"] = score
    return payload


def _merge_weighted(a, b):
    return a if a.get("weighted_score", 0.0) >= b.get("weighted_score", 0.0) else b


def _merge_where(where, extra):
    if not where:
        return extra
    if len(where) == 1 or any(k.startswith("$") for k in where.keys()):
        return {"$and": [where, extra]}
    return {"$and": [{k: v} for k, v in where.items()] + [{k: v} for k, v in extra.items()]}
