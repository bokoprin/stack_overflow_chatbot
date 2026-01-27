import re


def mmr_select(results, top_k, lambda_param=0.7, max_candidates=None):
    if not results:
        return []
    if top_k <= 0:
        return []

    candidates = results[: max_candidates or len(results)]
    base_scores = [_base_score(item) for item in candidates]
    selected = []
    selected_idx = []

    # pick the best base score first
    first_idx = max(range(len(candidates)), key=lambda i: base_scores[i])
    selected.append(candidates[first_idx])
    selected_idx.append(first_idx)

    while len(selected) < min(top_k, len(candidates)):
        best_idx = None
        best_score = None
        for i, item in enumerate(candidates):
            if i in selected_idx:
                continue
            similarity = 0.0
            for s in selected:
                similarity = max(similarity, _jaccard(item.get("document"), s.get("document")))
            score = lambda_param * base_scores[i] - (1.0 - lambda_param) * similarity
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break
        selected.append(candidates[best_idx])
        selected_idx.append(best_idx)

    return selected


def _base_score(item):
    if item.get("hybrid_score") is not None:
        return float(item.get("hybrid_score") or 0.0)
    if item.get("distance") is not None:
        return 1.0 - float(item.get("distance") or 0.0)
    if item.get("bm25_score") is not None:
        return float(item.get("bm25_score") or 0.0)
    return 0.0


def _jaccard(text_a, text_b):
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(inter) / len(union) if union else 0.0


def _tokenize(text):
    if not text:
        return set()
    tokens = re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龯]+", text)
    return {t.lower() for t in tokens if t.strip()}
