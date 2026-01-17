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


def run_query(args):
    retriever = Retriever(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )
    results = retriever.search(args.query, top_k=args.top_k)
    llm_client = LLMClient()
    prompt = llm_client.build_prompt(args.query, results)
    answer = llm_client.generate_answer(prompt)
    print(answer)
    if results:
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


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Stack Overflow RAG CLI")
    parser.add_argument("--index", action="store_true", help="Build the vector index.")
    parser.add_argument("--reset-index", action="store_true", help="Reset the index before building.")
    parser.add_argument("--input", help="Path to QA JSON file.")
    parser.add_argument("--query", help="User query text.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "vector_db"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "stack_overflow"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.index and not args.query:
        parser.print_help()
        return

    if args.index:
        run_index(args)
    if args.query:
        run_query(args)


if __name__ == "__main__":
    main()
