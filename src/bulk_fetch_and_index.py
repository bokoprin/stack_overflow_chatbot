import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from data_loader import StackOverflowDataLoader
from indexer import StackOverflowIndexer


DEFAULT_TAGS = [
    "machine-learning",
    "deep-learning",
    "pytorch",
    "c",
    "c++",
    "linux",
    "linux-kernel",
    "embedded",
    "rtos",
    "arduino",
    "raspberry-pi",
    "stm32",
]


def _now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _latest_merged_json(data_dir="data/raw"):
    data_path = Path(data_dir)
    candidates = sorted(
        data_path.glob("stackoverflow_qa_merged_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_existing_question_ids(paths):
    seen = set()
    for path in paths:
        if not path or not Path(path).exists():
            continue
        with Path(path).open(encoding="utf-8") as handle:
            try:
                records = json.load(handle)
            except json.JSONDecodeError:
                continue
        for record in records:
            qid = record.get("question_id")
            if qid is not None:
                seen.add(int(qid))
    return seen


def _parse_throttle_wait_seconds(message):
    match = re.search(r"more requests available in\\s+(\\d+)\\s+seconds", message)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _load_state(state_path):
    path = Path(state_path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(state, state_path):
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    records = []
    if not Path(path).exists():
        return records
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_jsonl_slice(path, start_line):
    records = []
    if not Path(path).exists():
        return records, 0
    total_lines = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            if total_lines <= start_line:
                continue
            records.append(json.loads(line))
    return records, total_lines


def bulk_fetch(
    *,
    tags,
    base_json,
    output_dir,
    state_path,
    pagesize,
    max_wait_seconds,
    quota_stop_threshold,
    request_delay,
    retry_wait,
    max_retries,
):
    existing_sources = []
    if base_json:
        existing_sources.append(base_json)
    seen_question_ids = _load_existing_question_ids(existing_sources)

    state = _load_state(state_path) if state_path else None
    if state and state.get("output_dir"):
        output_dir = state["output_dir"]
    output_dir = str(output_dir)

    progress = {}
    if state and state.get("progress"):
        progress = state["progress"]
    else:
        progress = {tag: {"page": 1, "done": False, "fetched": 0} for tag in tags}

    new_jsonl = str(Path(output_dir) / "new_records.jsonl")
    if Path(new_jsonl).exists():
        with Path(new_jsonl).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = record.get("question_id")
                if qid is None:
                    continue
                try:
                    seen_question_ids.add(int(qid))
                except ValueError:
                    continue

    loader = StackOverflowDataLoader(
        api_key=os.getenv("STACK_OVERFLOW_API_KEY"),
        sort="votes",
        order="desc",
        pagesize=pagesize,
        max_questions=pagesize,  # per page
        request_delay=request_delay,
        retry_wait=retry_wait,
        max_retries=max_retries,
    )

    latest_quota = None
    latest_quota_max = None
    total_new = int(state.get("total_new", 0)) if state else 0
    total_skipped = int(state.get("total_skipped", 0)) if state else 0
    indexed_records = int(state.get("indexed_records", 0)) if state else 0

    active_tags = [tag for tag in tags if not progress.get(tag, {}).get("done")]
    round_count = 0

    while active_tags:
        round_count += 1
        for tag in list(active_tags):
            if progress[tag]["done"]:
                active_tags.remove(tag)
                continue

            loader.tag = tag
            page = progress[tag]["page"]
            try:
                data = loader._request(
                    "questions",
                    {
                        "page": page,
                        "pagesize": pagesize,
                        "order": "desc",
                        "sort": "votes",
                        "tagged": tag,
                        "filter": "withbody",
                    },
                )
            except RuntimeError as exc:
                wait_seconds = _parse_throttle_wait_seconds(str(exc))
                if wait_seconds is not None and wait_seconds <= max_wait_seconds:
                    time.sleep(wait_seconds)
                    continue
                state_payload = {
                    "output_dir": output_dir,
                    "base_json": str(base_json) if base_json else None,
                    "tags": tags,
                    "progress": progress,
                    "seen_question_ids": len(seen_question_ids),
                    "total_new": total_new,
                    "total_skipped": total_skipped,
                    "indexed_records": indexed_records,
                    "last_error": str(exc),
                    "updated_at": _now_stamp(),
                }
                if state_path:
                    _save_state(state_payload, state_path)
                raise

            latest_quota = data.get("quota_remaining", latest_quota)
            latest_quota_max = data.get("quota_max", latest_quota_max)
            if latest_quota is not None and latest_quota <= quota_stop_threshold:
                state_payload = {
                    "output_dir": output_dir,
                    "base_json": str(base_json) if base_json else None,
                    "tags": tags,
                    "progress": progress,
                    "seen_question_ids": len(seen_question_ids),
                    "total_new": total_new,
                    "total_skipped": total_skipped,
                    "indexed_records": indexed_records,
                    "quota_remaining": latest_quota,
                    "quota_max": latest_quota_max,
                    "updated_at": _now_stamp(),
                }
                if state_path:
                    _save_state(state_payload, state_path)
                return {
                    "output_dir": output_dir,
                    "new_jsonl": new_jsonl,
                    "total_new": total_new,
                    "total_skipped": total_skipped,
                    "quota_remaining": latest_quota,
                    "quota_max": latest_quota_max,
                }

            items = data.get("items", [])
            if not items or not data.get("has_more"):
                progress[tag]["done"] = True
                active_tags = [t for t in active_tags if t != tag]
                continue

            for question in items:
                qid = question.get("question_id")
                if qid is None:
                    continue
                qid_int = int(qid)
                if qid_int in seen_question_ids:
                    total_skipped += 1
                    continue

                # Reduce answer calls: fetch only top voted answers (up to 30); fetch accepted by id if missing.
                question_id = qid_int
                try:
                    answers = loader._request(
                        f"questions/{question_id}/answers",
                        {
                            "order": "desc",
                            "sort": "votes",
                            "pagesize": 30,
                            "filter": "withbody",
                        },
                    ).get("items", [])
                except RuntimeError as exc:
                    print(f"api error answers qid={question_id}: {exc}")
                    continue

                accepted_answer = None
                top_non_accepted = None
                for answer in answers:
                    if answer.get("is_accepted"):
                        accepted_answer = answer
                    elif top_non_accepted is None:
                        top_non_accepted = answer

                accepted_answer_id = question.get("accepted_answer_id")
                if not accepted_answer and accepted_answer_id:
                    try:
                        accepted_answer = loader.fetch_answer_by_id(accepted_answer_id)
                    except RuntimeError as exc:
                        print(f"api error accepted answer id={accepted_answer_id}: {exc}")
                        accepted_answer = None

                selected_answers = []
                if accepted_answer:
                    selected_answers.append(loader._select_answer_fields(accepted_answer))
                if top_non_accepted and (
                    not accepted_answer
                    or top_non_accepted.get("answer_id") != accepted_answer.get("answer_id")
                ):
                    selected_answers.append(loader._select_answer_fields(top_non_accepted))

                record = loader._select_question_fields(question)
                record["answers"] = selected_answers

                _append_jsonl(new_jsonl, record)
                seen_question_ids.add(qid_int)
                progress[tag]["fetched"] += 1
                total_new += 1

            if latest_quota is not None:
                print(
                    f"tag={tag} page={page} fetched_total={progress[tag]['fetched']} "
                    f"new_total={total_new} skipped_total={total_skipped} "
                    f"quota_remaining={latest_quota}/{latest_quota_max}"
                )
            else:
                print(
                    f"tag={tag} page={page} fetched_total={progress[tag]['fetched']} "
                    f"new_total={total_new} skipped_total={total_skipped}"
                )

            progress[tag]["page"] += 1
            if state_path and round_count % 2 == 0:
                _save_state(
                    {
                        "output_dir": output_dir,
                        "base_json": str(base_json) if base_json else None,
                        "tags": tags,
                        "progress": progress,
                        "seen_question_ids": len(seen_question_ids),
                        "total_new": total_new,
                        "total_skipped": total_skipped,
                        "indexed_records": indexed_records,
                        "quota_remaining": latest_quota,
                        "quota_max": latest_quota_max,
                        "updated_at": _now_stamp(),
                    },
                    state_path,
                )

    return {
        "output_dir": output_dir,
        "new_jsonl": new_jsonl,
        "total_new": total_new,
        "total_skipped": total_skipped,
        "quota_remaining": latest_quota,
        "quota_max": latest_quota_max,
    }


def merge_records(base_json, new_records, output_path):
    base = []
    if base_json and Path(base_json).exists():
        with Path(base_json).open(encoding="utf-8") as handle:
            base = json.load(handle)
    by_qid = {int(item["question_id"]): item for item in base if item.get("question_id") is not None}
    for record in new_records:
        qid = record.get("question_id")
        if qid is None:
            continue
        by_qid[int(qid)] = record
    merged = list(by_qid.values())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)
    return merged


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Bulk fetch Stack Overflow data and append index.")
    parser.add_argument("--tags", default=",".join(DEFAULT_TAGS), help="Comma-separated tags.")
    parser.add_argument("--base-json", default=None, help="Existing merged JSON to avoid duplicates.")
    parser.add_argument("--output-dir", default=None, help="Output directory under data/raw.")
    parser.add_argument("--state", default="data/raw/bulk_state.json", help="State file for resume.")
    parser.add_argument("--pagesize", type=int, default=100)
    parser.add_argument("--max-wait-seconds", type=int, default=600)
    parser.add_argument("--quota-stop-threshold", type=int, default=10)
    parser.add_argument("--request-delay", type=float, default=0.15)
    parser.add_argument("--retry-wait", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--index-batch-size", type=int, default=2)
    parser.add_argument("--embedding-device", default=os.getenv("EMBEDDING_DEVICE", "cpu"))
    args = parser.parse_args()

    state = _load_state(args.state) if args.state else None
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    base_json = Path(args.base_json) if args.base_json else _latest_merged_json()
    if base_json is None:
        print("base_json not found; continuing without de-dup baseline.")
    output_dir = args.output_dir or str(Path("data/raw") / f"bulk_{_now_stamp()}")

    result = bulk_fetch(
        tags=tags,
        base_json=base_json,
        output_dir=output_dir,
        state_path=args.state,
        pagesize=args.pagesize,
        max_wait_seconds=args.max_wait_seconds,
        quota_stop_threshold=args.quota_stop_threshold,
        request_delay=args.request_delay,
        retry_wait=args.retry_wait,
        max_retries=args.max_retries,
    )

    new_records = _read_jsonl(result["new_jsonl"])
    base_count = 0
    if base_json and Path(base_json).exists():
        with Path(base_json).open(encoding="utf-8") as handle:
            base_count = len(json.load(handle))

    stamp = _now_stamp()
    merged_out = Path("data/raw") / f"stackoverflow_qa_merged_{(base_count + len(new_records))}_{stamp}.json"
    new_out = Path("data/raw") / f"stackoverflow_qa_new_{len(new_records)}_{stamp}.json"
    with new_out.open("w", encoding="utf-8") as handle:
        json.dump(new_records, handle, ensure_ascii=False, indent=2)
    merged = merge_records(base_json, new_records, merged_out)

    if new_records:
        already_indexed = int((state or {}).get("indexed_records", 0))
        to_index, total_lines = _read_jsonl_slice(result["new_jsonl"], already_indexed)
        os.environ["EMBEDDING_DEVICE"] = args.embedding_device
        indexer = StackOverflowIndexer(batch_size=args.index_batch_size)
        chunks = indexer.create_chunks(to_index)
        count = indexer.build_index(chunks, reset=False)
        print(f"indexed chunks: {count}")
        if args.state:
            state_payload = state or {}
            state_payload.update(
                {
                    "output_dir": str(Path(result["new_jsonl"]).parent),
                    "indexed_records": total_lines,
                    "updated_at": _now_stamp(),
                }
            )
            _save_state(state_payload, args.state)

    print(f"saved new: {new_out}")
    print(f"saved merged: {merged_out} (total questions: {len(merged)})")
    if result.get("quota_remaining") is not None:
        print(f"quota_remaining: {result['quota_remaining']}/{result.get('quota_max')}")


if __name__ == "__main__":
    main()
