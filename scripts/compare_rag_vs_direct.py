import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Keep embedding retrieval on CPU so it does not compete with llama.cpp GPU memory.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from llm_client import LLMClient  # noqa: E402
from preprocessor import strip_html  # noqa: E402
from query_processor import QueryProcessor  # noqa: E402
from retriever import Retriever  # noqa: E402
from search_engine import SearchEngine  # noqa: E402


DEFAULT_STRATEGY = "translate_hybrid_mmr_rerank_llm_expand_compress_fusion_tag_split_dual"
FALLBACK_STRATEGY = "hybrid_mmr_rerank_tag_split_dual"
DEFAULT_TOP_K = 5


class LocalLlamaCppClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 180,
        max_retries: int = 4,
        retry_wait: float = 2.0,
        max_answer_chars: int = 1200,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.max_answer_chars = max_answer_chars

    def generate(self, user_prompt: str, system_prompt: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 1024,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        text = self._post_with_retry(url, payload)
        text = _strip_think_tags(text).strip()
        if self.max_answer_chars > 0 and len(text) > self.max_answer_chars:
            text = text[: self.max_answer_chars].rstrip() + "…"
        return text

    def _post_with_retry(self, url: str, payload: dict[str, Any]) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("local llm response has no choices")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    chunks = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            chunks.append(part.get("text", ""))
                    content = "".join(chunks)
                if not isinstance(content, str):
                    raise RuntimeError("local llm response content is invalid")
                return content
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait * (2 ** attempt))
        raise RuntimeError(f"local llm request failed: {last_exc}")


class OpenAIJudge:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4-mini",
        timeout: int = 120,
        max_retries: int = 4,
        retry_wait: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    def judge(self, question: str, answer_rag: str, answer_direct: str) -> dict[str, Any]:
        prompt = (
            "あなたは厳密な技術評価者です。質問に対する2つの回答を採点してください。\n"
            "観点は accuracy, grounding, specificity, safety の4つ（各0〜5点）。\n"
            "overall_score は4観点の平均（小数可）にしてください。\n"
            "winner は rag/direct/tie のいずれか。\n"
            "必ずJSONのみを返してください。\n\n"
            "出力JSONスキーマ:\n"
            "{\n"
            '  "rag": {"overall_score": 0.0, "reason": "...", "aspects": {"accuracy": 0, "grounding": 0, "specificity": 0, "safety": 0}},\n'
            '  "direct": {"overall_score": 0.0, "reason": "...", "aspects": {"accuracy": 0, "grounding": 0, "specificity": 0, "safety": 0}},\n'
            '  "winner": "rag"\n'
            "}\n\n"
            f"質問:\n{question}\n\n"
            f"回答A (rag):\n{answer_rag}\n\n"
            f"回答B (direct):\n{answer_direct}\n"
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "top_p": 1,
            "messages": [
                {"role": "system", "content": "JSONのみを返す採点者として振る舞ってください。"},
                {"role": "user", "content": prompt},
            ],
        }
        result = self._chat_completion(payload)
        parsed = _parse_json_loose(result)
        if not isinstance(parsed, dict):
            raise RuntimeError("judge output parse failed")
        return _normalize_judge_payload(parsed)

    def _chat_completion(self, payload: dict[str, Any]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("openai response has no choices")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if not isinstance(content, str):
                    raise RuntimeError("openai response content is invalid")
                return content
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait * (2 ** attempt))
        raise RuntimeError(f"judge request failed: {last_exc}")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龯]+", text.lower())


def _score_raw_item(query_tokens: list[str], item: dict[str, Any]) -> float:
    title = (item.get("title") or "").lower()
    body = strip_html(item.get("body") or "").lower()
    tags = item.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tag_text = " ".join(str(t).lower() for t in tags)

    title_tokens = set(_tokenize(title))
    body_tokens = set(_tokenize(body[:3000]))
    tag_tokens = set(_tokenize(tag_text))
    q_tokens = set(query_tokens)
    if not q_tokens:
        return 0.0

    title_overlap = len(q_tokens & title_tokens) / len(q_tokens)
    body_overlap = len(q_tokens & body_tokens) / len(q_tokens)
    tag_overlap = len(q_tokens & tag_tokens) / len(q_tokens)
    contains_bonus = 1.0 if any(t in title or t in body for t in q_tokens if len(t) >= 3) else 0.0

    return 3.0 * title_overlap + 1.0 * body_overlap + 2.0 * tag_overlap + contains_bonus


def _select_best_answer(answers: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not answers:
        return None
    for a in answers:
        if a.get("is_accepted"):
            return a
    return max(answers, key=lambda x: x.get("score") or -10**9)


def _build_direct_contexts(question: str, raw_items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_tokens = _tokenize(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in raw_items:
        score = _score_raw_item(query_tokens, item)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [item for _, item in scored[:top_k]]

    contexts = []
    for idx, item in enumerate(selected, start=1):
        qid = item.get("question_id")
        tags = item.get("tags") or []
        body = strip_html(item.get("body") or "")
        body = re.sub(r"\s+", " ", body).strip()
        body = body[:1200]
        best_answer = _select_best_answer(item.get("answers") or [])
        answer_id = best_answer.get("answer_id") if best_answer else None
        answer_text = strip_html((best_answer or {}).get("body") or "")
        answer_text = re.sub(r"\s+", " ", answer_text).strip()[:1200]
        title = item.get("title") or ""
        doc = (
            f"Q: {title}\n"
            f"Tags: {', '.join(tags)}\n"
            f"Body: {body}\n"
            f"A: {answer_text}"
        )
        contexts.append(
            {
                "id": f"raw_{idx}_{qid}",
                "document": doc,
                "metadata": {
                    "chunk_type": "raw_question",
                    "question_id": qid,
                    "answer_id": answer_id,
                    "title": title,
                    "link": item.get("link"),
                    "score": item.get("score"),
                },
            }
        )
    return contexts


def _extract_context_ids(results: list[dict[str, Any]]) -> str:
    ids: list[str] = []
    for item in results:
        meta = item.get("metadata") or {}
        qid = meta.get("question_id")
        if qid is None:
            continue
        qid_s = str(qid)
        if qid_s not in ids:
            ids.append(qid_s)
    return "|".join(ids)


def _extract_context_answer_ids(results: list[dict[str, Any]]) -> str:
    ids: list[str] = []
    for item in results:
        meta = item.get("metadata") or {}
        aid = meta.get("answer_id")
        if aid is None:
            continue
        aid_s = str(aid)
        if aid_s not in ids:
            ids.append(aid_s)
    return "|".join(ids)


def _parse_pipe_ids(value: str) -> set[str]:
    return {x.strip() for x in (value or "").split("|") if x.strip()}


def _judge_by_target_ids(
    *,
    target_qid: str,
    target_aid: str,
    rag_qids: str,
    rag_aids: str,
    direct_qids: str,
    direct_aids: str,
) -> dict[str, Any]:
    tq = str(target_qid or "").strip()
    ta = str(target_aid or "").strip()
    rag_qset = _parse_pipe_ids(rag_qids)
    rag_aset = _parse_pipe_ids(rag_aids)
    direct_qset = _parse_pipe_ids(direct_qids)
    direct_aset = _parse_pipe_ids(direct_aids)

    def _score_side(qset: set[str], aset: set[str]) -> tuple[float, str, bool, bool]:
        hit_q = bool(tq and tq in qset)
        hit_a = bool(ta and ta in aset)
        if hit_q and hit_a:
            return 5.0, "target_question_id と target_answer_id の両方に一致", hit_q, hit_a
        if hit_a:
            return 4.0, "target_answer_id に一致", hit_q, hit_a
        if hit_q:
            return 3.0, "target_question_id に一致", hit_q, hit_a
        return 0.0, "ターゲットID不一致", hit_q, hit_a

    rag_score, rag_reason, rag_hit_q, rag_hit_a = _score_side(rag_qset, rag_aset)
    direct_score, direct_reason, direct_hit_q, direct_hit_a = _score_side(direct_qset, direct_aset)
    if rag_score > direct_score:
        winner = "rag"
    elif direct_score > rag_score:
        winner = "direct"
    else:
        winner = "tie"

    return {
        "rag": {
            "overall_score": rag_score,
            "reason": rag_reason,
            "hit_question_id": rag_hit_q,
            "hit_answer_id": rag_hit_a,
        },
        "direct": {
            "overall_score": direct_score,
            "reason": direct_reason,
            "hit_question_id": direct_hit_q,
            "hit_answer_id": direct_hit_a,
        },
        "winner": winner,
    }


def _parse_json_loose(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _strip_think_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned


def _clamp_0_5(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, num))


def _normalize_judge_payload(data: dict[str, Any]) -> dict[str, Any]:
    def _norm_side(obj: Any) -> dict[str, Any]:
        obj = obj if isinstance(obj, dict) else {}
        aspects = obj.get("aspects") if isinstance(obj.get("aspects"), dict) else {}
        out_aspects = {
            "accuracy": _clamp_0_5(aspects.get("accuracy")),
            "grounding": _clamp_0_5(aspects.get("grounding")),
            "specificity": _clamp_0_5(aspects.get("specificity")),
            "safety": _clamp_0_5(aspects.get("safety")),
        }
        overall = obj.get("overall_score")
        if overall is None:
            overall = sum(out_aspects.values()) / 4.0
        out_overall = round(_clamp_0_5(overall), 3)
        reason = str(obj.get("reason") or "").strip()
        return {"overall_score": out_overall, "reason": reason, "aspects": out_aspects}

    rag = _norm_side(data.get("rag"))
    direct = _norm_side(data.get("direct"))
    winner = str(data.get("winner") or "").strip().lower()
    if winner not in {"rag", "direct", "tie"}:
        if rag["overall_score"] > direct["overall_score"]:
            winner = "rag"
        elif direct["overall_score"] > rag["overall_score"]:
            winner = "direct"
        else:
            winner = "tie"
    return {"rag": rag, "direct": direct, "winner": winner}


def _load_questions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _load_raw_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported raw JSON shape: {path}")


def _load_existing_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        out = {}
        for row in reader:
            row_id = str(row.get("id") or "").strip()
            if row_id:
                out[row_id] = row
        return out


def _write_results_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    rag_scores = []
    direct_scores = []
    winners = {"rag": 0, "direct": 0, "tie": 0}
    aspect = {
        "rag": {"accuracy": [], "grounding": [], "specificity": [], "safety": []},
        "direct": {"accuracy": [], "grounding": [], "specificity": [], "safety": []},
    }

    for row in rows:
        try:
            rag_score = float(row.get("judge_score_rag", ""))
            direct_score = float(row.get("judge_score_direct", ""))
        except (TypeError, ValueError):
            continue
        rag_scores.append(rag_score)
        direct_scores.append(direct_score)
        winner = (row.get("winner") or "").strip().lower()
        if winner in winners:
            winners[winner] += 1
        for side in ("rag", "direct"):
            for key in ("accuracy", "grounding", "specificity", "safety"):
                col = f"judge_{key}_{side}"
                try:
                    aspect[side][key].append(float(row.get(col, "")))
                except (TypeError, ValueError):
                    pass

    def _avg(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    return {
        "count_total_rows": len(rows),
        "count_scored_rows": len(rag_scores),
        "avg_score_rag": _avg(rag_scores),
        "avg_score_direct": _avg(direct_scores),
        "winners": winners,
        "aspect_averages": {
            "rag": {k: _avg(v) for k, v in aspect["rag"].items()},
            "direct": {k: _avg(v) for k, v in aspect["direct"].items()},
        },
    }


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# RAG vs Direct Summary",
        "",
        f"- total_rows: {summary.get('count_total_rows')}",
        f"- scored_rows: {summary.get('count_scored_rows')}",
        f"- avg_score_rag: {summary.get('avg_score_rag')}",
        f"- avg_score_direct: {summary.get('avg_score_direct')}",
        f"- winners: {summary.get('winners')}",
        "",
        "## Aspect Averages",
        "",
        "| side | accuracy | grounding | specificity | safety |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for side in ("rag", "direct"):
        a = summary.get("aspect_averages", {}).get(side, {})
        lines.append(
            f"| {side} | {a.get('accuracy')} | {a.get('grounding')} | {a.get('specificity')} | {a.get('safety')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Compare RAG vs direct raw-data answering.")
    parser.add_argument("--input-csv", default=os.getenv("EVAL_QUESTIONS_CSV", "data/processed/eval_questions_c_cpp_rtos.csv"))
    parser.add_argument("--raw-json", default=os.getenv("RAW_SOURCE_JSON", "data/raw/stackoverflow_qa_merged_20260122_202231.json"))
    parser.add_argument("--output-dir", default=None, help="If omitted, creates data/processed/eval_compare_<timestamp>.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--judge-mode", choices=["target_id", "openai", "none"], default="target_id")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    raw_json = Path(args.raw_json)
    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")
    if not raw_json.exists():
        raise FileNotFoundError(f"raw json not found: {raw_json}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("data/processed") / f"eval_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "results.csv"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"

    questions = _load_questions(input_csv)
    if args.limit is not None:
        questions = questions[: args.limit]

    raw_items = _load_raw_items(raw_json)

    retriever = Retriever(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "vector_db"),
        collection_name=os.getenv("CHROMA_COLLECTION", "stack_overflow"),
    )
    query_processor = QueryProcessor(
        strategy=args.strategy,
        enable_expansion=True,
        dynamic_top_k=True,
    )
    search_engine = SearchEngine(retriever, query_processor, parent_child=True)
    fallback_processor = QueryProcessor(
        strategy=FALLBACK_STRATEGY,
        enable_expansion=True,
        dynamic_top_k=True,
    )
    fallback_search_engine = SearchEngine(retriever, fallback_processor, parent_child=True)
    prompt_builder = LLMClient(host="http://localhost:0", model="dummy", enforce_japanese=False)

    local_client = LocalLlamaCppClient(
        base_url=os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080"),
        model=os.getenv("LLAMA_CPP_MODEL", "qwen3.5-27b-iq3m-96k"),
        max_answer_chars=int(os.getenv("ANSWER_MAX_CHARS", "1200")),
    )

    judge: OpenAIJudge | None = None
    if not args.skip_judge and args.judge_mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            judge = OpenAIJudge(
                api_key=api_key,
                model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.4-mini"),
            )
        else:
            print("OPENAI_API_KEY is not set; judge is disabled.")

    existing = _load_existing_rows(results_csv)
    results: list[dict[str, str]] = []

    base_fields = [
        "id",
        "domain",
        "question",
        "target_question_id",
        "target_answer_id",
        "target_title",
        "target_link",
        "target_tags",
        "answer_rag",
        "answer_direct",
    ]
    append_fields = [
        "rag_context_ids",
        "rag_context_answer_ids",
        "direct_context_ids",
        "direct_context_answer_ids",
        "hit_target_qid_rag",
        "hit_target_aid_rag",
        "hit_target_qid_direct",
        "hit_target_aid_direct",
        "judge_score_rag",
        "judge_score_direct",
        "judge_reason_rag",
        "judge_reason_direct",
        "winner",
        "judge_accuracy_rag",
        "judge_grounding_rag",
        "judge_specificity_rag",
        "judge_safety_rag",
        "judge_accuracy_direct",
        "judge_grounding_direct",
        "judge_specificity_direct",
        "judge_safety_direct",
        "error",
    ]
    fieldnames = base_fields + append_fields

    system_answer = (
        "あなたは技術アシスタントです。与えられた抜粋のみを根拠に日本語で回答してください。"
        "根拠が不足する場合は、わかる範囲と不明点を明確に述べてください。"
    )

    for idx, q in enumerate(questions, start=1):
        qid = str(q.get("id") or "").strip()
        question = (q.get("question") or "").strip()
        if not qid or not question:
            continue

        prev = existing.get(qid)
        prev_has_answers = bool((prev or {}).get("answer_rag")) and bool((prev or {}).get("answer_direct"))
        prev_has_result = bool((prev or {}).get("judge_score_rag")) or bool((prev or {}).get("error"))
        if prev and prev_has_answers and prev_has_result:
            results.append(prev)
            continue

        row = {k: q.get(k, "") for k in base_fields}
        row.setdefault("answer_rag", "")
        row.setdefault("answer_direct", "")
        for col in append_fields:
            row[col] = ""

        try:
            try:
                _, rag_hits = search_engine.search(question, top_k=args.top_k, filters={})
            except Exception as rag_exc:
                print(f"rag strategy fallback for id={qid}: {rag_exc}")
                _, rag_hits = fallback_search_engine.search(question, top_k=args.top_k, filters={})
            rag_prompt = prompt_builder.build_prompt(question, rag_hits)
            answer_rag = local_client.generate(rag_prompt, system_answer)
            row["answer_rag"] = answer_rag
            row["rag_context_ids"] = _extract_context_ids(rag_hits)
            row["rag_context_answer_ids"] = _extract_context_answer_ids(rag_hits)

            direct_hits = _build_direct_contexts(question, raw_items, top_k=args.top_k)
            direct_prompt = prompt_builder.build_prompt(question, direct_hits)
            answer_direct = local_client.generate(direct_prompt, system_answer)
            row["answer_direct"] = answer_direct
            row["direct_context_ids"] = _extract_context_ids(direct_hits)
            row["direct_context_answer_ids"] = _extract_context_answer_ids(direct_hits)

            if not args.skip_judge and args.judge_mode == "target_id":
                judgement = _judge_by_target_ids(
                    target_qid=row.get("target_question_id", ""),
                    target_aid=row.get("target_answer_id", ""),
                    rag_qids=row.get("rag_context_ids", ""),
                    rag_aids=row.get("rag_context_answer_ids", ""),
                    direct_qids=row.get("direct_context_ids", ""),
                    direct_aids=row.get("direct_context_answer_ids", ""),
                )
                row["judge_score_rag"] = str(judgement["rag"]["overall_score"])
                row["judge_score_direct"] = str(judgement["direct"]["overall_score"])
                row["judge_reason_rag"] = judgement["rag"]["reason"]
                row["judge_reason_direct"] = judgement["direct"]["reason"]
                row["winner"] = judgement["winner"]
                row["hit_target_qid_rag"] = str(judgement["rag"]["hit_question_id"])
                row["hit_target_aid_rag"] = str(judgement["rag"]["hit_answer_id"])
                row["hit_target_qid_direct"] = str(judgement["direct"]["hit_question_id"])
                row["hit_target_aid_direct"] = str(judgement["direct"]["hit_answer_id"])
            elif judge is not None:
                judgement = judge.judge(question, answer_rag, answer_direct)
                row["judge_score_rag"] = str(judgement["rag"]["overall_score"])
                row["judge_score_direct"] = str(judgement["direct"]["overall_score"])
                row["judge_reason_rag"] = judgement["rag"]["reason"]
                row["judge_reason_direct"] = judgement["direct"]["reason"]
                row["winner"] = judgement["winner"]
                row["judge_accuracy_rag"] = str(judgement["rag"]["aspects"]["accuracy"])
                row["judge_grounding_rag"] = str(judgement["rag"]["aspects"]["grounding"])
                row["judge_specificity_rag"] = str(judgement["rag"]["aspects"]["specificity"])
                row["judge_safety_rag"] = str(judgement["rag"]["aspects"]["safety"])
                row["judge_accuracy_direct"] = str(judgement["direct"]["aspects"]["accuracy"])
                row["judge_grounding_direct"] = str(judgement["direct"]["aspects"]["grounding"])
                row["judge_specificity_direct"] = str(judgement["direct"]["aspects"]["specificity"])
                row["judge_safety_direct"] = str(judgement["direct"]["aspects"]["safety"])

            print(f"[{idx}/{len(questions)}] done id={qid}")
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[{idx}/{len(questions)}] failed id={qid}: {exc}")

        results.append(row)
        _write_results_csv(results_csv, results, fieldnames)

    summary = _summarize(results)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_md(summary_md, summary)

    print(f"saved: {results_csv}")
    print(f"saved: {summary_json}")
    print(f"saved: {summary_md}")


if __name__ == "__main__":
    main()
