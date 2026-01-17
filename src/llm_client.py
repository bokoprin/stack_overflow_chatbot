import os

import requests
from dotenv import load_dotenv


class LLMClient:
    def __init__(self, host=None, model=None, timeout=120):
        load_dotenv()
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:14b")
        self.timeout = timeout

    def build_prompt(self, query, contexts):
        context_blocks = []
        for idx, item in enumerate(contexts, start=1):
            metadata = item.get("metadata") or {}
            header = f"[{idx}] type={metadata.get('chunk_type')} qid={metadata.get('question_id')}"
            if metadata.get("score") is not None:
                header += f" score={metadata.get('score')}"
            if metadata.get("is_accepted") is not None:
                header += f" accepted={metadata.get('is_accepted')}"
            if metadata.get("title"):
                header += f" title={metadata.get('title')}"
            if metadata.get("link"):
                header += f" link={metadata.get('link')}"
            block = f"{header}\n{item.get('document', '')}"
            context_blocks.append(block)
        context_text = "\n\n".join(context_blocks) if context_blocks else "No context available."
        prompt = (
            "You are a helpful assistant. Use the following Stack Overflow excerpts to answer the question.\n"
            "If the answer is not contained in the excerpts, say you don't know.\n"
            "Answer in Japanese.\n\n"
            f"Question:\n{query}\n\n"
            f"Excerpts:\n{context_text}\n\n"
            "Answer:"
        )
        return prompt

    def generate_answer(self, prompt):
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
