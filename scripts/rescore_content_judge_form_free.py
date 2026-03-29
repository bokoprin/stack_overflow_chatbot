import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龯]+", text.lower())


def _norm_tokens(text: str, min_len: int = 2) -> set[str]:
    return {t for t in _tokenize(text) if len(t) >= min_len}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _score_grounding(answer: str, question: str, evidence_qids: str, context_qids: str, id_match_score: float) -> tuple[float, str]:
    ans = (answer or "").strip()
    qset = _norm_tokens(question, min_len=2)
    aset = _norm_tokens(ans, min_len=2)

    if not ans:
        return 0.0, "empty answer"

    # Low-confidence wording is not penalized by itself in form-free mode.
    target_q = {x for x in (evidence_qids or "").split("|") if x}
    ctx_q = {x for x in (context_qids or "").split("|") if x}
    hit_ratio = (len(target_q & ctx_q) / len(target_q)) if target_q else 0.0

    # Lexical relevance to question only (avoid length/bullet dependence).
    rel = (len(aset & qset) / len(qset)) if qset else 0.0

    # retrieval consistency gets larger weight than phrasing.
    norm_id = _clamp(id_match_score / 5.0)
    grounding = 5.0 * _clamp(0.55 * norm_id + 0.30 * hit_ratio + 0.15 * rel)
    reason = f"id_norm={norm_id:.2f}, qid_hit={hit_ratio:.2f}, rel={rel:.2f}"
    return round(grounding, 3), reason


def _score_task_success(answer: str, question: str, constraints: str) -> tuple[float, str]:
    ans = (answer or "").strip()
    if not ans:
        return 0.0, "empty answer"

    qset = _norm_tokens(question, min_len=2)
    cset = _norm_tokens(constraints, min_len=2)
    aset = _norm_tokens(ans, min_len=2)

    # Ask-focused and constraint-focused coverage; no length/format terms.
    q_cov = (len(aset & qset) / len(qset)) if qset else 0.0
    c_cov = (len(aset & cset) / len(cset)) if cset else 0.0

    # Mild penalty only when answer is effectively refusal-only.
    refusal_words = {"わかりません", "不明", "判断できません", "情報不足"}
    has_refusal = any(w in ans for w in refusal_words)
    nontrivial = len(aset) >= 12
    refusal_penalty = 0.25 if (has_refusal and not nontrivial) else 0.0

    task = 5.0 * _clamp(0.65 * q_cov + 0.35 * c_cov - refusal_penalty)
    reason = f"q_cov={q_cov:.2f}, c_cov={c_cov:.2f}, refusal_penalty={refusal_penalty:.2f}"
    return round(task, 3), reason


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _summary(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        try:
            score = float(r.get("final_score", ""))
        except (TypeError, ValueError):
            continue
        bucket[(r.get("answer_model", ""), r.get("method", ""))].append(score)

    out: dict[str, dict[str, float]] = {}
    for k, vals in bucket.items():
        out[f"{k[0]}::{k[1]}"] = {
            "count": len(vals),
            "avg_final_score": round(sum(vals) / len(vals), 4) if vals else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore existing content_judge outputs without format bias.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    with in_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if r.get("error"):
            continue
        answer = r.get("answer", "")
        question = r.get("question", "")
        constraints = r.get("constraints", "")
        evidence_qids = r.get("evidence_question_ids", "")
        context_qids = r.get("context_qids", "")
        try:
            id_match = float(r.get("id_match_score", "0") or 0)
        except ValueError:
            id_match = 0.0

        grounding, g_reason = _score_grounding(
            answer=answer,
            question=question,
            evidence_qids=evidence_qids,
            context_qids=context_qids,
            id_match_score=id_match,
        )
        task_success, t_reason = _score_task_success(
            answer=answer,
            question=question,
            constraints=constraints,
        )
        # Keep the same final weighting for comparability.
        final = 0.4 * id_match + 0.3 * grounding + 0.3 * task_success

        r["grounding_score"] = str(round(grounding, 3))
        r["task_success_score"] = str(round(task_success, 3))
        r["final_score"] = str(round(final, 3))
        r["judge_reason"] = f"form_free | {g_reason} | {t_reason}"

    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(Path(args.output_csv), rows, fieldnames)
    Path(args.summary_json).write_text(
        json.dumps(_summary(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
