import os
import re
import time

import requests
from dotenv import load_dotenv

from reranker import Reranker


_SYNONYMS = {
    "ai": ["artificial intelligence", "machine learning", "ml", "deep learning", "dl"],
    "機械学習": ["machine learning", "ml", "ディープラーニング"],
    "深層学習": ["deep learning", "dl"],
    "rtos": ["real-time operating system", "リアルタイムos"],
    "リアルタイムos": ["rtos", "real-time operating system"],
    "arduino": ["microcontroller", "マイコン"],
    "ラズパイ": ["raspberry pi", "raspi"],
    "raspberry pi": ["raspi", "ラズパイ"],
    "c++": ["cpp", "cplusplus"],
    "c言語": ["c language", "c"],
    "linux": ["gnu/linux", "linux kernel"],
}

class QueryProcessor:
    def __init__(
        self,
        strategy=None,
        host=None,
        translate_model=None,
        enable_expansion=None,
        dynamic_top_k=None,
        timeout=120,
        max_retries=2,
        retry_wait=5,
    ):
        load_dotenv()
        self.strategy = strategy or os.getenv("QUERY_STRATEGY", "baseline")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.translate_model = translate_model or os.getenv("QUERY_TRANSLATE_MODEL", "qwen3:8b")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.enable_expansion = (
            enable_expansion
            if enable_expansion is not None
            else os.getenv("ENABLE_QUERY_EXPANSION", "true").lower()
            in {"1", "true", "yes", "on"}
        )
        self.dynamic_top_k = (
            dynamic_top_k
            if dynamic_top_k is not None
            else os.getenv("DYNAMIC_TOP_K", "true").lower()
            in {"1", "true", "yes", "on"}
        )
        self.reranker = Reranker()

    def uses_translation(self):
        return "translate" in self.strategy

    def uses_rerank(self):
        return "rerank" in self.strategy

    def uses_hybrid(self):
        return "hybrid" in self.strategy

    def uses_mmr(self):
        return "mmr" in self.strategy

    def uses_multistage(self):
        return "multi" in self.strategy

    def uses_llm_expansion(self):
        return "llm_expand" in self.strategy

    def build_search_query(self, query, fallback_en=None):
        if self.uses_translation():
            if fallback_en:
                base = fallback_en
            else:
                base = self.translate_ja_to_en(query)
        else:
            base = query
        expanded = base
        if self.enable_expansion:
            expanded = self.expand_query(base)
        if self.uses_llm_expansion():
            expanded = self.expand_query_llm(expanded, base_query=base)
        return expanded

    def build_search_queries(self, query, fallback_en=None):
        base = query
        if self.uses_translation():
            base = fallback_en or self.translate_ja_to_en(query)

        queries = [base]
        if self.enable_expansion:
            expanded = self.expand_query(base)
            if expanded not in queries:
                queries.append(expanded)
        if self.uses_llm_expansion():
            llm_expanded = self.expand_query_llm(queries[-1], base_query=base)
            if llm_expanded not in queries:
                queries.append(llm_expanded)
        return queries

    def rerank_results(self, results, query_for_search, tag_hints=None):
        if not self.uses_rerank():
            return results
        return self.reranker.rerank(query_for_search, results, tag_hints=tag_hints)

    def adjust_top_k(self, query, base_top_k, min_k=3, max_k=10):
        if not self.dynamic_top_k:
            return base_top_k
        tokens = self._tokenize_ascii(query)
        length = len(tokens)
        if length <= 2:
            return min(max_k, base_top_k + 4)
        if length <= 4:
            return min(max_k, base_top_k + 2)
        if length >= 12:
            return max(min_k, base_top_k - 2)
        return base_top_k

    def expand_query(self, query):
        if not query:
            return query
        q_lower = query.lower()
        additions = []
        for key, synonyms in _SYNONYMS.items():
            if key.lower() in q_lower:
                for syn in synonyms:
                    if syn.lower() not in q_lower:
                        additions.append(syn)
        if not additions:
            return query
        return f"{query} {' '.join(additions)}"

    def expand_query_llm(self, query, base_query=None):
        if not query:
            return query
        if not self._should_use_llm_expansion(query):
            return query
        prompt = (
            "You are expanding a search query for technical Q&A retrieval. "
            "Return 5 short keywords or phrases to improve recall. "
            "Return keywords in the same language as the input, separated by spaces.\n\n"
            f"Query:\n{query}\n\n"
            "Keywords:"
        )
        try:
            response = self._generate(
                prompt,
                system="Return keywords only.",
                options={"temperature": 0.2, "num_predict": 64},
            )
        except Exception:
            return query
        keywords = self._clean_keywords(response)
        if not keywords:
            return query
        base = base_query or query
        return f"{base} {' '.join(keywords)}"

    def _should_use_llm_expansion(self, query):
        max_len = int(os.getenv("LLM_EXPANSION_MAX_CHARS", "120"))
        return len(query) <= max_len

    def _clean_keywords(self, text):
        if not text:
            return []
        tokens = re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龯]+", text)
        seen = set()
        cleaned = []
        for token in tokens:
            token_l = token.lower()
            if token_l in seen:
                continue
            seen.add(token_l)
            cleaned.append(token)
        return cleaned

    def translate_ja_to_en(self, text):
        if not text:
            return ""
        prompt = (
            "Translate the following Japanese text into English. "
            "Do not translate code blocks, URLs, or library/function names. "
            "Return English only.\n\n"
            f"Text:\n{text}\n\n"
            "English:"
        )
        return self._generate(
            prompt,
            system="Return English only.",
            options={"temperature": 0, "num_predict": 256},
        )

    def translate_en_to_ja(self, text):
        if not text:
            return ""
        prompt = (
            "Translate the following English text into Japanese. "
            "Do not translate code blocks, URLs, or library/function names. "
            "Return Japanese only.\n\n"
            f"Text:\n{text}\n\n"
            "Japanese:"
        )
        return self._generate(
            prompt,
            system="日本語だけを返してください。",
            options={"temperature": 0, "num_predict": 512},
        )

    def _generate(self, prompt, system=None, options=None):
        url = f"{self.host}/api/generate"
        payload = {"model": self.translate_model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        last_exc = None
        for _ in range(self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return (data.get("response") or "").strip()
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(self.retry_wait)
        raise last_exc

    def _tokenize_ascii(self, text):
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())
