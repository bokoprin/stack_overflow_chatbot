import os
import re

import requests
from dotenv import load_dotenv


class LLMClient:
    def __init__(self, host=None, model=None, timeout=120, enforce_japanese=True):
        load_dotenv()
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:14b")
        self.timeout = timeout
        self.enforce_japanese = enforce_japanese

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
        context_text = "\n\n".join(context_blocks) if context_blocks else "参照情報なし。"
        prompt = (
            "あなたは有能なアシスタントです。以下のStack Overflow抜粋だけを根拠に回答してください。\n"
            "抜粋に答えが含まれない場合は「わかりません」と答えてください。\n"
            "必ず日本語で回答してください。\n\n"
            f"質問:\n{query}\n\n"
            f"抜粋:\n{context_text}\n\n"
            "回答:"
        )
        return prompt

    def generate_answer(self, prompt):
        answer = self._generate(prompt, system="必ず日本語で回答してください。")
        if self.enforce_japanese and answer and not self._contains_japanese(answer):
            translated = self._translate_to_japanese(answer)
            if translated:
                return translated.strip()
        return answer.strip()

    def _generate(self, prompt, system=None):
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def _contains_japanese(self, text):
        return re.search(r"[\\u3040-\\u30ff\\u4e00-\\u9fff]", text) is not None

    def _translate_to_japanese(self, text):
        prompt = (
            "以下の回答を日本語に翻訳してください。\n"
            "- ``` で囲まれたコードブロックの中身は変更しない\n"
            "- URL、ライブラリ名、関数名はそのまま\n"
            "翻訳対象:\n"
            f"{text}\n\n"
            "翻訳結果:"
        )
        return self._generate(prompt, system="日本語の翻訳結果のみを返してください。")
