import os

from hybrid_search import HybridSearcher


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
        search_query = self.query_processor.build_search_query(query, fallback_en=fallback_en)
        effective_top_k = self.query_processor.adjust_top_k(
            query,
            top_k,
            min_k=int(os.getenv("TOP_K_MIN", "3")),
            max_k=int(os.getenv("TOP_K_MAX", "10")),
        )

        where = {}
        if self.parent_child:
            where["chunk_type"] = "qa_parent"
        if filters.get("language"):
            where["language"] = filters["language"]

        filter_fn = _build_filter_fn(filters, require_parent=self.parent_child)

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

        if self.parent_child:
            results = _expand_children(self.retriever, results, filters)

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
