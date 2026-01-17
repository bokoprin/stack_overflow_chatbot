import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from src.retriever import Retriever


def main():
    load_dotenv()
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "vector_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "stack_overflow")
    data_path = Path(persist_dir)
    if not data_path.exists():
        print(f"vector db not found: {persist_dir}")
        return

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        print(f"collection not found: {collection_name}")
        return

    count = collection.count()
    print(f"collection: {collection_name}")
    print(f"chunks: {count}")

    if count == 0:
        return

    retriever = Retriever(persist_dir=persist_dir, collection_name=collection_name)
    results = retriever.search("How to sort a list in python", top_k=3)
    if not results:
        print("sample query returned no results")
        return
    print("sample results:")
    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        doc = (item.get("document") or "").replace("\n", " ")
        snippet = doc[:160] + ("..." if len(doc) > 160 else "")
        print(
            f"[{idx}] type={metadata.get('chunk_type')} "
            f"qid={metadata.get('question_id')} "
            f"score={metadata.get('score')} "
            f"accepted={metadata.get('is_accepted')} "
            f"{snippet}"
        )


if __name__ == "__main__":
    main()
