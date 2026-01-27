import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append("src")

from indexer import StackOverflowIndexer  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from query_processor import QueryProcessor  # noqa: E402
from search_engine import SearchEngine  # noqa: E402
from retriever import Retriever  # noqa: E402


class ProgressLogger:
    def __init__(self, path, interval_seconds=1200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        self.next_log_time = time.time()

    def log(self, message, **kwargs):
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": message,
        }
        payload.update(kwargs)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_if_due(self, message, **kwargs):
        now = time.time()
        if now >= self.next_log_time:
            self.log(message, **kwargs)
            self.next_log_time = now + self.interval_seconds


class EvalScorer:
    def __init__(self, host, model, timeout=120):
        self.host = host
        self.model = model
        self.timeout = timeout

    def score(self, question, gold, answer):
        prompt = (
            "あなたはRAGの評価者です。質問と正解、回答を比較して0〜3点で採点してください。\n"
            "採点基準:\n"
            "3: 正確で十分\n"
            "2: 要点は合っているが不足あり\n"
            "1: 一部正しいが誤りあり\n"
            "0: 間違い/無関係\n"
            "JSONで次の形式だけ返してください: {\"score\": 2, \"reason\": \"...\"}\n\n"
            f"質問:\n{question}\n\n"
            f"正解:\n{gold}\n\n"
            f"回答:\n{answer}\n"
        )
        raw = self._generate(prompt)
        data = self._parse_json(raw)
        if not data:
            return 0, "parse_failed"
        score = data.get("score")
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        reason = data.get("reason", "")
        return score, reason

    def _generate(self, prompt):
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        response = __import__("requests").post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return (data.get("response") or "").strip()

    def _parse_json(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None


def _latest_merged_json():
    candidates = sorted(Path("data/raw").glob("stackoverflow_qa_merged_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _select_gold_answer(answers):
    accepted = [a for a in answers if a.get("is_accepted")]
    if accepted:
        return accepted[0]
    if not answers:
        return None
    return max(answers, key=lambda a: a.get("score") or 0)


def _strip_html(text):
    return StackOverflowIndexer.strip_html(text or "")


def _truncate(text, limit=2000):
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "..."


def build_dataset(records, size, processor, progress_logger, dataset_path):
    candidates = []
    for record in records:
        answers = record.get("answers") or []
        gold = _select_gold_answer(answers)
        if not gold:
            continue
        candidates.append((record, gold))

    random.seed(42)
    random.shuffle(candidates)

    dataset = []
    done_qids = set()
    if dataset_path.exists():
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        done_qids = {item.get("question_id") for item in dataset}

    processed = 0
    for record, gold in candidates:
        if len(dataset) >= size:
            break
        qid = record.get("question_id")
        if qid in done_qids:
            continue
        processed += 1
        title = record.get("title") or ""
        body = _strip_html(record.get("body") or "")
        query_en = title.strip() or _truncate(body, 200)
        try:
            query_ja = processor.translate_en_to_ja(query_en)
            query_en_translated = processor.translate_ja_to_en(query_ja)
            gold_en = _strip_html(gold.get("body") or "")
            gold_ja = processor.translate_en_to_ja(_truncate(gold_en, 1200))
        except Exception as exc:
            progress_logger.log(
                "dataset_translate_failed",
                question_id=qid,
                error=str(exc),
            )
            continue

        dataset.append(
            {
                "question_id": qid,
                "tags": record.get("tags", []),
                "link": record.get("link"),
                "query_en": query_en,
                "query_ja": query_ja,
                "query_en_translated": query_en_translated,
                "gold_answer_en": gold_en,
                "gold_answer_ja": gold_ja,
                "gold_is_accepted": bool(gold.get("is_accepted")),
                "gold_score": gold.get("score"),
            }
        )
        done_qids.add(qid)
        if processed % 5 == 0:
            dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

        progress_logger.log_if_due(
            "dataset_building",
            processed=len(dataset),
            target=size,
        )

    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset


def evaluate_strategy(
    strategy,
    dataset,
    retriever,
    llm_client,
    scorer,
    processor,
    output_dir,
    top_k,
    progress_logger,
):
    os.environ["ENABLE_CONTEXT_COMPRESS"] = "true" if "compress" in strategy else "false"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"answers_{strategy}.jsonl"

    completed_qids = set()
    if results_path.exists():
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    completed_qids.add(data.get("question_id"))
                except json.JSONDecodeError:
                    continue

    total = len(dataset)
    hits = 0
    mrr_total = 0.0
    score_total = 0
    scored = 0

    search_engine = SearchEngine(retriever, processor)

    for idx, item in enumerate(dataset, start=1):
        qid = item.get("question_id")
        if qid in completed_qids:
            continue

        query_ja = item["query_ja"]
        query_en = item.get("query_en_translated") or item.get("query_en")
        search_query, results = search_engine.search(query_ja, top_k=top_k, fallback_en=query_en)

        rank = None
        for r_idx, res in enumerate(results, start=1):
            meta = res.get("metadata") or {}
            if meta.get("question_id") == qid:
                rank = r_idx
                break

        hit = rank is not None
        if hit:
            hits += 1
            mrr_total += 1.0 / rank

        prompt = llm_client.build_prompt(query_ja, results)
        answer = llm_client.generate_answer(prompt)
        score, reason = scorer.score(query_ja, item["gold_answer_ja"], answer)
        score_total += score
        scored += 1

        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "strategy": strategy,
                        "rank": rank,
                        "hit": hit,
                        "score": score,
                        "reason": reason,
                        "answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        progress_logger.log_if_due(
            "evaluating",
            strategy=strategy,
            processed=idx,
            total=total,
            hits=hits,
            scored=scored,
        )

    hit_rate = hits / total if total else 0.0
    mrr = mrr_total / total if total else 0.0
    avg_score = score_total / scored if scored else 0.0
    accuracy = score_total / (scored * 3) if scored else 0.0

    metrics = {
        "strategy": strategy,
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "avg_score": round(avg_score, 4),
        "accuracy": round(accuracy, 4),
        "total": total,
        "scored": scored,
    }

    with (output_dir / f"metrics_{strategy}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    return metrics


def select_best(metrics_list):
    def key(item):
        return (item["avg_score"], item["hit_rate"], item["mrr"])

    return sorted(metrics_list, key=key, reverse=True)[0] if metrics_list else None


def update_env_best_strategy(strategy):
    env_path = Path(".env")
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith("QUERY_STRATEGY="):
            new_lines.append(f"QUERY_STRATEGY={strategy}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"QUERY_STRATEGY={strategy}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run improvement loop for RAG evaluation.")
    parser.add_argument("--input", default=None, help="Merged JSON path.")
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--time-limit-hours", type=int, default=7)
    parser.add_argument("--log-interval-minutes", type=int, default=20)
    parser.add_argument(
        "--strategies",
        default=(
            "baseline,translate,hybrid,translate_hybrid,translate_rerank,translate_hybrid_rerank,"
            "translate_hybrid_mmr,translate_hybrid_mmr_rerank,"
            "translate_hybrid_mmr_rerank_multi,translate_hybrid_mmr_rerank_llm_expand,"
            "translate_hybrid_mmr_rerank_llm_expand_compress"
        ),
    )
    parser.add_argument("--eval-model", default="qwen3:8b")
    parser.add_argument("--translate-model", default="qwen3:8b")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    start_time = datetime.now()
    deadline = start_time + timedelta(hours=args.time_limit_hours)
    progress_logger = ProgressLogger("logs/progress.log", args.log_interval_minutes * 60)

    input_path = Path(args.input) if args.input else _latest_merged_json()
    if not input_path or not input_path.exists():
        raise FileNotFoundError("merged JSON not found")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data/processed") / f"eval_{start_time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_logger.log("start", input=str(input_path), output=str(output_dir))

    records = json.loads(input_path.read_text(encoding="utf-8"))
    processor = QueryProcessor(strategy="translate", translate_model=args.translate_model)
    dataset_path = output_dir / "dataset.json"
    dataset = build_dataset(records, args.size, processor, progress_logger, dataset_path)

    retriever = Retriever()
    llm_client = LLMClient()
    scorer = EvalScorer(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"), model=args.eval_model)

    metrics_all = []
    for strategy in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        progress_logger.log("strategy_start", strategy=strategy)
        strategy_processor = QueryProcessor(strategy=strategy, translate_model=args.translate_model)
        metrics = evaluate_strategy(
            strategy=strategy,
            dataset=dataset,
            retriever=retriever,
            llm_client=llm_client,
            scorer=scorer,
            processor=strategy_processor,
            output_dir=output_dir,
            top_k=args.top_k,
            progress_logger=progress_logger,
        )
        metrics_all.append(metrics)
        progress_logger.log("strategy_end", strategy=strategy, metrics=metrics)

        if datetime.now() >= deadline:
            progress_logger.log("deadline_reached", strategy=strategy)
            break

    best = select_best(metrics_all)
    if best:
        (output_dir / "best_strategy.json").write_text(
            json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        update_env_best_strategy(best["strategy"])
        progress_logger.log("best_selected", strategy=best["strategy"], metrics=best)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(metrics_all, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_logger.log("end", metrics=metrics_all)


if __name__ == "__main__":
    main()
