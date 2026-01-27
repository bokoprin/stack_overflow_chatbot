import argparse
import json
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from domain_dict import normalize_terms
from preprocessor import QAPreprocessor, detect_language, strip_html

class StackOverflowIndexer:
    def __init__(
        self,
        embedding_model_name=None,
        persist_dir="vector_db",
        collection_name="stack_overflow",
        batch_size=16,
        device=None,
    ):
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-m3"
        )
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.device = device or os.getenv("EMBEDDING_DEVICE")
        if self.device:
            self.model = SentenceTransformer(self.embedding_model_name, device=self.device)
        else:
            self.model = SentenceTransformer(self.embedding_model_name)
        self.is_e5 = "e5" in self.embedding_model_name.lower()

    @staticmethod
    def load_json(path):
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def strip_html(text):
        return strip_html(text)

    def create_chunks(self, records):
        chunks = []
        for record in records:
            question_id = record.get("question_id")
            title = record.get("title", "")
            question_text = record.get("question_text")
            if not question_text:
                body = record.get("body", "")
                question_text = strip_html(f"{title}\n\n{body}".strip())
            question_text = normalize_terms(question_text)
            tags = record.get("tags", [])
            tags_text = ",".join(tags)
            language = record.get("language") or detect_language(question_text)
            question_score = record.get("score")

            answers = record.get("answers", [])
            accepted_answer = None
            for answer in answers:
                if answer.get("is_accepted"):
                    accepted_answer = answer
                    break
            if not accepted_answer and answers:
                accepted_answer = answers[0]

            accepted_text = ""
            accepted_score = None
            accepted_id = None
            if accepted_answer:
                accepted_text = accepted_answer.get("body_text") or strip_html(
                    accepted_answer.get("body", "")
                )
                accepted_text = normalize_terms(accepted_text)
                accepted_score = accepted_answer.get("score")
                accepted_id = accepted_answer.get("answer_id")

            parent_text = f"Q: {question_text}\nA: {accepted_text}".strip()
            parent_max_chars = int(os.getenv("PARENT_CHUNK_MAX_CHARS", "800"))
            if len(parent_text) > parent_max_chars:
                parent_text = parent_text[:parent_max_chars]

            parent_id = f"p_{question_id}"
            chunks.append(
                {
                    "id": parent_id,
                    "text": parent_text,
                    "metadata": {
                        "question_id": question_id,
                        "chunk_type": "qa_parent",
                        "chunk_index": 0,
                        "tags": tags_text,
                        "title": title,
                        "link": record.get("link"),
                        "language": language,
                        "question_score": question_score,
                        "answer_score": accepted_score,
                        "answer_id": accepted_id,
                        "has_accepted": bool(accepted_answer and accepted_answer.get("is_accepted")),
                    },
                }
            )

            # question-only chunk (dual index)
            chunks.append(
                {
                    "id": f"qonly_{question_id}",
                    "text": question_text,
                    "metadata": {
                        "question_id": question_id,
                        "chunk_type": "question_only",
                        "chunk_index": 0,
                        "tags": tags_text,
                        "title": title,
                        "link": record.get("link"),
                        "language": language,
                        "question_score": question_score,
                    },
                }
            )

            # answer-only chunk (accepted/top answer)
            if accepted_text:
                answer_only_max = int(os.getenv("ANSWER_ONLY_MAX_CHARS", "1200"))
                answer_only_text = accepted_text[:answer_only_max]
                chunks.append(
                    {
                        "id": f"aonly_{question_id}",
                        "text": answer_only_text,
                        "metadata": {
                            "question_id": question_id,
                            "chunk_type": "answer_only",
                            "chunk_index": 0,
                            "tags": tags_text,
                            "title": title,
                            "link": record.get("link"),
                            "language": language,
                            "score": accepted_score,
                            "is_accepted": bool(
                                accepted_answer and accepted_answer.get("is_accepted")
                            ),
                            "answer_id": accepted_id,
                        },
                    }
                )

            answer_max_chars = int(os.getenv("ANSWER_CHUNK_MAX_CHARS", "1200"))
            for ans_idx, answer in enumerate(answers, start=1):
                answer_text = answer.get("body_text") or strip_html(answer.get("body", ""))
                answer_text = normalize_terms(answer_text)
                segments = _semantic_split(answer_text, max_chars=answer_max_chars)
                for seg_idx, segment in enumerate(segments, start=1):
                    chunks.append(
                        {
                            "id": f"c_{question_id}_{ans_idx}_{seg_idx}",
                            "text": segment,
                            "metadata": {
                                "question_id": question_id,
                                "chunk_type": "answer_child",
                                "chunk_index": seg_idx,
                                "parent_id": parent_id,
                                "tags": tags_text,
                                "title": title,
                                "link": record.get("link"),
                                "language": answer.get("language") or language,
                                "score": answer.get("score"),
                                "is_accepted": answer.get("is_accepted"),
                                "answer_id": answer.get("answer_id"),
                            },
                        }
                    )
        return chunks

    def build_index(self, chunks, reset=False, start=None, end=None):
        if start is not None or end is not None:
            start = start or 0
            end = end if end is not None else len(chunks)
            chunks = chunks[start:end]
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
                _apply_embedding_prefix(texts, self.is_e5, prefix="passage:"),
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
        [p for p in data_path.glob("*.json") if p.name.startswith("stackoverflow_qa")],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    return candidates[0]


def _semantic_split(text, max_chars=1200, min_chars=120):
    text = (text or "").strip()
    if not text:
        return []
    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
    segments = []
    for block in blocks:
        if len(block) <= max_chars:
            segments.append(block)
            continue
        sentences = re.split(r"(?<=[.!?。！？])\s+", block)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 <= max_chars:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    segments.append(current)
                current = sentence
        if current:
            segments.append(current)

    merged = []
    buffer = ""
    for segment in segments:
        if len(buffer) + len(segment) + 1 <= max_chars:
            buffer = f"{buffer} {segment}".strip()
        else:
            if buffer:
                merged.append(buffer)
            buffer = segment
    if buffer:
        merged.append(buffer)

    final = []
    for segment in merged:
        if len(segment) >= min_chars or not final:
            final.append(segment)
        else:
            final[-1] = f"{final[-1]} {segment}".strip()
    return final


def _apply_embedding_prefix(texts, is_e5, prefix="passage:"):
    if not is_e5:
        return texts
    prefixed = []
    for text in texts:
        if text:
            prefixed.append(f"{prefix} {text}")
        else:
            prefixed.append(prefix)
    return prefixed


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build ChromaDB index.")
    parser.add_argument("--input", default=None, help="Path to QA JSON file.")
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "vector_db"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "stack_overflow"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reset", action="store_true", help="Reset the collection.")
    parser.add_argument("--start", type=int, default=None, help="Start index for chunk slicing.")
    parser.add_argument("--end", type=int, default=None, help="End index for chunk slicing.")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_latest_json()
    records = StackOverflowIndexer.load_json(input_path)
    preprocessor = QAPreprocessor()
    records = preprocessor.preprocess(records)
    indexer = StackOverflowIndexer(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )
    chunks = indexer.create_chunks(records)
    count = indexer.build_index(chunks, reset=args.reset, start=args.start, end=args.end)
    print(f"indexed chunks: {count}")


if __name__ == "__main__":
    main()
