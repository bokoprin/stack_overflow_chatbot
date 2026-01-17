import argparse
import json
import os
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


class StackOverflowIndexer:
    def __init__(
        self,
        embedding_model_name=None,
        persist_dir="vector_db",
        collection_name="stack_overflow",
        batch_size=16,
    ):
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-m3"
        )
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.embedding_model_name)

    @staticmethod
    def load_json(path):
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def strip_html(text):
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        return " ".join(soup.stripped_strings)

    def create_chunks(self, records):
        chunks = []
        for record in records:
            question_id = record.get("question_id")
            title = record.get("title", "")
            body = record.get("body", "")
            question_text = self.strip_html(f"{title}\n\n{body}".strip())
            question_chunk = {
                "id": f"q_{question_id}",
                "text": question_text,
                "metadata": {
                    "question_id": question_id,
                    "chunk_type": "question",
                    "chunk_index": 0,
                    "tags": record.get("tags", []),
                    "title": title,
                    "link": record.get("link"),
                },
            }
            chunks.append(question_chunk)

            answers = record.get("answers", [])
            for idx, answer in enumerate(answers, start=1):
                answer_text = self.strip_html(answer.get("body", ""))
                answer_chunk = {
                    "id": f"a_{question_id}_{idx}",
                    "text": answer_text,
                    "metadata": {
                        "question_id": question_id,
                        "chunk_type": "answer",
                        "chunk_index": idx,
                        "score": answer.get("score"),
                        "is_accepted": answer.get("is_accepted"),
                    },
                }
                chunks.append(answer_chunk)
        return chunks

    def build_index(self, chunks, reset=False):
        client = chromadb.PersistentClient(path=self.persist_dir)
        if reset:
            try:
                client.delete_collection(name=self.collection_name)
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        total = len(chunks)
        for start in range(0, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [item["text"] for item in batch]
            ids = [item["id"] for item in batch]
            metadatas = [item["metadata"] for item in batch]
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
            )
            embeddings = [vector.tolist() for vector in embeddings]
            collection.upsert(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
            print(f"indexed {min(start + self.batch_size, total)}/{total}")
        return collection.count()


def find_latest_json(data_dir="data/raw"):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    candidates = sorted(
        data_path.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    return candidates[0]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build ChromaDB index.")
    parser.add_argument("--input", default=None, help="Path to QA JSON file.")
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "vector_db"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "stack_overflow"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reset", action="store_true", help="Reset the collection.")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_latest_json()
    records = StackOverflowIndexer.load_json(input_path)
    indexer = StackOverflowIndexer(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )
    chunks = indexer.create_chunks(records)
    count = indexer.build_index(chunks, reset=args.reset)
    print(f"indexed chunks: {count}")


if __name__ == "__main__":
    main()
