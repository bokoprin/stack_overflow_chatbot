import argparse
import os

from dotenv import load_dotenv

from indexer import StackOverflowIndexer, find_latest_json
from llm_client import LLMClient
from retriever import Retriever


def run_index(args):
    input_path = args.input or str(find_latest_json())
    records = StackOverflowIndexer.load_json(input_path)
    indexer = StackOverflowIndexer(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )
    chunks = indexer.create_chunks(records)
    count = indexer.build_index(chunks, reset=args.reset_index)
    print(f"indexed chunks: {count}")


def _print_sources(results):
    if not results:
        return
    print("\nSources:")
    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        line = f"[{idx}] type={metadata.get('chunk_type')} qid={metadata.get('question_id')}"
        if metadata.get("score") is not None:
            line += f" score={metadata.get('score')}"
        if metadata.get("is_accepted") is not None:
            line += f" accepted={metadata.get('is_accepted')}"
        if metadata.get("link"):
            line += f" {metadata.get('link')}"
        print(line)


def answer_query(query, retriever, llm_client, top_k):
    results = retriever.search(query, top_k=top_k)
    prompt = llm_client.build_prompt(query, results)
    answer = llm_client.generate_answer(prompt)
    print(answer)
    _print_sources(results)


def run_query(args):
    retriever = Retriever(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )
    llm_client = LLMClient()
    answer_query(args.query, retriever, llm_client, args.top_k)


def run_interactive(args):
    retriever = Retriever(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )
    llm_client = LLMClient()
    print("対話モード開始。終了する場合は exit / quit / q を入力してください。")
    while True:
        try:
            query = input("質問> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n対話モード終了。")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("対話モード終了。")
            break
        answer_query(query, retriever, llm_client, args.top_k)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Stack Overflow RAG CLI")
    parser.add_argument("--index", action="store_true", help="Build the vector index.")
    parser.add_argument("--reset-index", action="store_true", help="Reset the index before building.")
    parser.add_argument("--input", help="Path to QA JSON file.")
    parser.add_argument("--query", help="User query text.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "vector_db"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "stack_overflow"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.index and not args.query and not args.interactive:
        parser.print_help()
        return

    if args.index:
        run_index(args)
    if args.query:
        run_query(args)
    if args.interactive:
        run_interactive(args)


if __name__ == "__main__":
    main()
