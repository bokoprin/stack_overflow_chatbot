import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Keep embeddings on CPU to avoid competing with llama.cpp GPU memory.
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


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龯]+", text.lower())


def _parse_pipe_ids(value: str) -> set[str]:
    return {x.strip() for x in (value or "").split("|") if x.strip()}


def _strip_think_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def _select_best_answer(answers: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not answers:
        return None
    for a in answers:
        if a.get("is_accepted"):
            return a
    return max(answers, key=lambda x: x.get("score") or -10**9)


def _score_raw_item(query_tokens: list[str], item: dict[str, Any]) -> float:
    title = (item.get("title") or "").lower()
    body = strip_html(item.get("body") or "").lower()
    tags = item.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tag_text = " ".join(str(t).lower() for t in tags)

    q_tokens = set(query_tokens)
    if not q_tokens:
        return 0.0
    title_tokens = set(_tokenize(title))
    body_tokens = set(_tokenize(body[:3000]))
    tag_tokens = set(_tokenize(tag_text))

    title_overlap = len(q_tokens & title_tokens) / len(q_tokens)
    body_overlap = len(q_tokens & body_tokens) / len(q_tokens)
    tag_overlap = len(q_tokens & tag_tokens) / len(q_tokens)
    contains_bonus = 1.0 if any(t in title or t in body for t in q_tokens if len(t) >= 3) else 0.0
    return 3.0 * title_overlap + 1.0 * body_overlap + 2.0 * tag_overlap + contains_bonus


def _build_direct_contexts(question: str, raw_items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    query_tokens = _tokenize(question)
    for item in raw_items:
        score = _score_raw_item(query_tokens, item)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [item for _, item in scored[:top_k]]

    out = []
    for idx, item in enumerate(selected, start=1):
        best_answer = _select_best_answer(item.get("answers") or [])
        answer_id = best_answer.get("answer_id") if best_answer else None
        answer_text = strip_html((best_answer or {}).get("body") or "")
        answer_text = re.sub(r"\s+", " ", answer_text).strip()[:1200]
        body = strip_html(item.get("body") or "")
        body = re.sub(r"\s+", " ", body).strip()[:1200]
        title = item.get("title") or ""
        tags = item.get("tags") or []
        doc = f"Q: {title}\nTags: {', '.join(tags)}\nBody: {body}\nA: {answer_text}"
        out.append(
            {
                "id": f"raw_{idx}_{item.get('question_id')}",
                "document": doc,
                "metadata": {
                    "chunk_type": "raw_question",
                    "question_id": item.get("question_id"),
                    "answer_id": answer_id,
                    "title": title,
                    "link": item.get("link"),
                },
            }
        )
    return out


def _extract_context_qids(results: list[dict[str, Any]]) -> str:
    ids = []
    for r in results:
        qid = (r.get("metadata") or {}).get("question_id")
        if qid is None:
            continue
        s = str(qid)
        if s not in ids:
            ids.append(s)
    return "|".join(ids)


def _extract_context_aids(results: list[dict[str, Any]], qid_to_aid: dict[str, str]) -> str:
    ids = []
    for r in results:
        meta = r.get("metadata") or {}
        aid = meta.get("answer_id")
        if aid is None:
            qid = meta.get("question_id")
            aid = qid_to_aid.get(str(qid)) if qid is not None else None
        if aid is None:
            continue
        s = str(aid)
        if s not in ids:
            ids.append(s)
    return "|".join(ids)


def _id_match_score(target_qids: set[str], target_aids: set[str], context_qids: set[str], context_aids: set[str]) -> tuple[float, bool, bool]:
    hit_q = bool(target_qids & context_qids)
    hit_a = bool(target_aids & context_aids)
    if hit_q and hit_a:
        return 5.0, hit_q, hit_a
    if hit_a:
        return 4.0, hit_q, hit_a
    if hit_q:
        return 3.0, hit_q, hit_a
    return 0.0, hit_q, hit_a


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


class LocalLlamaCppClient:
    def __init__(self, base_url: str, model: str, timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def answer(self, system_prompt: str, user_prompt: str) -> str:
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
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if isinstance(content, list):
            content = "".join((p.get("text") or "") for p in content if isinstance(p, dict))
        return _strip_think_tags(str(content))


class OpenAIChatClient:
    def __init__(self, api_key: str, model: str, timeout: int = 180) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0,
            "top_p": 1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        return _strip_think_tags(str(content))


class CodexCliClient:
    def __init__(self, timeout: int = 240) -> None:
        self.timeout = timeout

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        prompt = (
            f"システム指示:\n{system_prompt}\n\n"
            f"ユーザー指示:\n{user_prompt}\n"
        )
        out_file = Path("/tmp") / f"codex_eval_{int(time.time() * 1000)}.txt"
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--output-last-message",
            str(out_file),
            prompt,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"codex exec failed: {proc.stderr.strip() or proc.stdout.strip()}")
        if not out_file.exists():
            # fallback: parse final stdout chunk
            text = proc.stdout.strip()
            return _strip_think_tags(text)
        text = out_file.read_text(encoding="utf-8")
        return _strip_think_tags(text)


def _load_raw_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported raw json: {path}")


def _build_evidence_text(raw_by_qid: dict[str, dict[str, Any]], qids: set[str], max_items: int = 2) -> str:
    parts = []
    for qid in list(qids)[:max_items]:
        item = raw_by_qid.get(qid)
        if not item:
            continue
        title = item.get("title") or ""
        body = re.sub(r"\s+", " ", strip_html(item.get("body") or "")).strip()[:500]
        best = _select_best_answer(item.get("answers") or [])
        ans = re.sub(r"\s+", " ", strip_html((best or {}).get("body") or "")).strip()[:500]
        parts.append(f"qid={qid}\nQ:{title}\nBody:{body}\nA:{ans}")
    return "\n\n".join(parts)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _content_judge(question: str, constraints: str, evidence: str, answer: str) -> tuple[float, float, str]:
    ans = (answer or "").strip()
    if not ans:
        return 0.0, 0.0, "empty answer"
    low_conf = "わかりません" in ans or "不明" in ans

    ans_tokens = [t for t in _tokenize(ans) if len(t) >= 2]
    ev_tokens = [t for t in _tokenize(evidence) if len(t) >= 2]
    q_tokens = [t for t in _tokenize(question) if len(t) >= 2]
    c_tokens = [t for t in _tokenize(constraints) if len(t) >= 2]
    aset = set(ans_tokens)
    eset = set(ev_tokens)
    qset = set(q_tokens)
    cset = set(c_tokens)

    overlap_ev = (len(aset & eset) / max(1, len(aset))) if aset else 0.0
    overlap_q = (len(aset & qset) / max(1, len(qset))) if qset else 0.0
    overlap_c = (len(aset & cset) / max(1, len(cset))) if cset else 0.0

    grounding = 5.0 * _clamp01(0.75 * overlap_ev + 0.25 * overlap_q)
    if low_conf:
        grounding = min(grounding, 2.0)

    bullet_count = ans.count("\n-") + ans.count("\n*") + ans.count("1.")
    length_score = 1.0 if len(ans) >= 220 else (0.6 if len(ans) >= 120 else 0.2)
    structure_score = 1.0 if bullet_count >= 2 else (0.5 if bullet_count >= 1 else 0.2)
    constraint_score = _clamp01(0.7 * overlap_c + 0.3 * overlap_q)
    task_success = 5.0 * _clamp01((length_score + structure_score + constraint_score) / 3.0)
    if low_conf:
        task_success = min(task_success, 1.5)

    reason = (
        f"ev_overlap={overlap_ev:.2f}, q_overlap={overlap_q:.2f}, "
        f"c_overlap={overlap_c:.2f}, low_conf={low_conf}"
    )
    return round(grounding, 3), round(task_success, 3), reason


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        try:
            score = float(r.get("final_score", ""))
        except (TypeError, ValueError):
            continue
        bucket[(r.get("answer_model", ""), r.get("method", ""))].append(score)
    out = {}
    for k, vals in bucket.items():
        out[f"{k[0]}::{k[1]}"] = {"count": len(vals), "avg_final_score": round(sum(vals) / len(vals), 4)}
    return out


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fair eval: RAG vs Direct across answer models.")
    parser.add_argument("--questions-csv", default="data/processed/eval_questions_rag_fair_20.csv")
    parser.add_argument("--raw-json", default=os.getenv("RAW_SOURCE_JSON", "data/raw/stackoverflow_qa_merged_20260122_202231.json"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--answer-models", default="qwen,codex", help="comma separated: qwen,codex")
    parser.add_argument("--judge-mode", choices=["id_only", "content_judge"], default="id_only")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions_csv = Path(args.questions_csv)
    raw_json = Path(args.raw_json)
    if not questions_csv.exists():
        raise FileNotFoundError(questions_csv)
    if not raw_json.exists():
        raise FileNotFoundError(raw_json)

    out_dir = Path(args.output_dir) if args.output_dir else Path("data/processed") / f"eval_fair_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results_long.csv"
    summary_json = out_dir / "summary.json"

    with questions_csv.open(encoding="utf-8", newline="") as f:
        questions = list(csv.DictReader(f))
    if args.limit is not None:
        questions = questions[: args.limit]

    raw_items = _load_raw_items(raw_json)
    raw_by_qid = {str(x.get("question_id")): x for x in raw_items if x.get("question_id") is not None}
    qid_to_aid: dict[str, str] = {}
    for qid, item in raw_by_qid.items():
        best = _select_best_answer(item.get("answers") or [])
        if best and best.get("answer_id") is not None:
            qid_to_aid[qid] = str(best.get("answer_id"))

    retriever = Retriever(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "vector_db"),
        collection_name=os.getenv("CHROMA_COLLECTION", "stack_overflow"),
    )
    qp = QueryProcessor(strategy=args.strategy, enable_expansion=True, dynamic_top_k=True)
    rag_engine = SearchEngine(retriever, qp, parent_child=True)
    qp_fb = QueryProcessor(strategy=FALLBACK_STRATEGY, enable_expansion=True, dynamic_top_k=True)
    rag_engine_fb = SearchEngine(retriever, qp_fb, parent_child=True)
    prompt_builder = LLMClient(host="http://localhost:0", model="dummy", enforce_japanese=False)

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    answer_clients = {}
    for name in [x.strip() for x in args.answer_models.split(",") if x.strip()]:
        if name == "qwen":
            answer_clients[name] = LocalLlamaCppClient(
                base_url=os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080"),
                model=os.getenv("LLAMA_CPP_MODEL", "qwen3.5-27b-iq3m-96k"),
            )
        elif name == "codex":
            answer_clients[name] = CodexCliClient()

    system_prompt = (
        "あなたは技術アシスタントです。与えられた抜粋のみを根拠に日本語で回答してください。"
        "根拠が不足する場合は、不明であることを明示してください。"
    )

    rows: list[dict[str, Any]] = []
    fieldnames = [
        "id",
        "answer_model",
        "method",
        "question",
        "constraints",
        "evidence_question_ids",
        "evidence_answer_ids",
        "context_qids",
        "context_aids",
        "hit_target_qid",
        "hit_target_aid",
        "id_match_score",
        "grounding_score",
        "task_success_score",
        "final_score",
        "judge_reason",
        "answer",
        "error",
    ]

    for idx, q in enumerate(questions, start=1):
        qtext = (q.get("question") or "").strip()
        constraints = (q.get("constraints") or "").strip()
        target_qids = _parse_pipe_ids(q.get("evidence_question_ids") or "")
        target_aids = _parse_pipe_ids(q.get("evidence_answer_ids") or "")
        if not qtext:
            continue

        try:
            try:
                _, rag_hits = rag_engine.search(qtext, top_k=args.top_k, filters={})
            except Exception as exc:
                print(f"rag fallback id={q.get('id')}: {exc}")
                _, rag_hits = rag_engine_fb.search(qtext, top_k=args.top_k, filters={})
            direct_hits = _build_direct_contexts(qtext, raw_items, top_k=args.top_k)
        except Exception as exc:
            for model_name in answer_clients.keys():
                for method in ("rag", "direct"):
                    rows.append(
                        {
                            "id": q.get("id"),
                            "answer_model": model_name,
                            "method": method,
                            "question": qtext,
                            "constraints": constraints,
                            "evidence_question_ids": q.get("evidence_question_ids"),
                            "evidence_answer_ids": q.get("evidence_answer_ids"),
                            "error": str(exc),
                        }
                    )
            continue

        contexts_by_method = {"rag": rag_hits, "direct": direct_hits}
        for model_name, client in answer_clients.items():
            for method in ("rag", "direct"):
                contexts = contexts_by_method[method]
                prompt = prompt_builder.build_prompt(qtext, contexts)
                context_qids = _extract_context_qids(contexts)
                context_aids = _extract_context_aids(contexts, qid_to_aid)
                id_score, hit_q, hit_a = _id_match_score(
                    target_qids,
                    target_aids,
                    _parse_pipe_ids(context_qids),
                    _parse_pipe_ids(context_aids),
                )

                row = {
                    "id": q.get("id"),
                    "answer_model": model_name,
                    "method": method,
                    "question": qtext,
                    "constraints": constraints,
                    "evidence_question_ids": q.get("evidence_question_ids"),
                    "evidence_answer_ids": q.get("evidence_answer_ids"),
                    "context_qids": context_qids,
                    "context_aids": context_aids,
                    "hit_target_qid": str(hit_q),
                    "hit_target_aid": str(hit_a),
                    "id_match_score": str(id_score),
                    "grounding_score": "",
                    "task_success_score": "",
                    "final_score": "",
                    "judge_reason": "",
                    "answer": "",
                    "error": "",
                }
                try:
                    answer = client.answer(system_prompt, prompt)
                    row["answer"] = answer
                    grounding = 0.0
                    task_success = 0.0
                    reason = ""
                    if args.judge_mode == "content_judge":
                        evidence_text = _build_evidence_text(raw_by_qid, target_qids)
                        grounding, task_success, reason = _content_judge(qtext, constraints, evidence_text, answer)
                    row["grounding_score"] = str(round(grounding, 3))
                    row["task_success_score"] = str(round(task_success, 3))
                    row["judge_reason"] = reason
                    if args.judge_mode == "content_judge":
                        final = 0.4 * id_score + 0.3 * grounding + 0.3 * task_success
                    else:
                        final = id_score
                    row["final_score"] = str(round(final, 3))
                except Exception as exc:
                    row["error"] = str(exc)
                rows.append(row)

        _write_csv(results_csv, rows, fieldnames)
        print(f"[{idx}/{len(questions)}] done")

    summary = _summary(rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {results_csv}")
    print(f"saved: {summary_json}")


if __name__ == "__main__":
    main()
