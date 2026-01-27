import os
import random
import re
from typing import List

from bs4 import BeautifulSoup
from dotenv import load_dotenv


def strip_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return " ".join(soup.stripped_strings)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def detect_language(text: str) -> str:
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text or ""):
        return "ja"
    return "en"


class QAPreprocessor:
    def __init__(
        self,
        min_question_score: int | None = None,
        min_answer_score: int | None = None,
        min_question_chars: int = 40,
        min_answer_chars: int = 80,
        max_answers_per_question: int = 2,
        accepted_only: bool = False,
        target_ja_ratio: float | None = None,
        max_per_primary_tag: int | None = None,
        random_seed: int = 42,
    ):
        load_dotenv()
        self.min_question_score = (
            min_question_score
            if min_question_score is not None
            else _env_int("MIN_QUESTION_SCORE")
        )
        self.min_answer_score = (
            min_answer_score if min_answer_score is not None else _env_int("MIN_ANSWER_SCORE")
        )
        self.min_question_chars = int(
            os.getenv("MIN_QUESTION_CHARS", str(min_question_chars))
        )
        self.min_answer_chars = int(os.getenv("MIN_ANSWER_CHARS", str(min_answer_chars)))
        self.max_answers_per_question = int(
            os.getenv("MAX_ANSWERS_PER_QUESTION", str(max_answers_per_question))
        )
        self.accepted_only = _env_bool("ACCEPTED_ONLY", accepted_only)
        self.target_ja_ratio = _env_float("TARGET_JA_RATIO", target_ja_ratio)
        self.max_per_primary_tag = _env_int("MAX_PER_PRIMARY_TAG", max_per_primary_tag)
        self.random = random.Random(int(os.getenv("BALANCE_RANDOM_SEED", str(random_seed))))

    def preprocess(self, records: List[dict]) -> List[dict]:
        cleaned = []
        seen = set()
        for record in records:
            title = record.get("title", "")
            body = record.get("body", "")
            question_text = strip_html(f"{title}\n\n{body}".strip())
            if len(question_text) < self.min_question_chars:
                continue

            question_score = record.get("score")
            if self.min_question_score is not None and (question_score or 0) < self.min_question_score:
                continue

            dedupe_key = normalize_text(question_text)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            answers = record.get("answers", []) or []
            answers_sorted = sorted(
                answers,
                key=lambda a: (
                    0 if a.get("is_accepted") else 1,
                    -(a.get("score") or 0),
                ),
            )

            selected = []
            for answer in answers_sorted:
                if self.accepted_only and not answer.get("is_accepted"):
                    continue
                answer_text = strip_html(answer.get("body", ""))
                if len(answer_text) < self.min_answer_chars:
                    continue
                if self.min_answer_score is not None and (answer.get("score") or 0) < self.min_answer_score:
                    continue
                answer["body_text"] = answer_text
                answer["language"] = detect_language(answer_text)
                selected.append(answer)
                if len(selected) >= self.max_answers_per_question:
                    break

            if not selected:
                continue

            record["question_text"] = question_text
            record["language"] = detect_language(question_text)
            record["answers"] = selected
            cleaned.append(record)

        cleaned = self._balance_language(cleaned)
        cleaned = self._balance_tags(cleaned)
        return cleaned

    def _balance_language(self, records: List[dict]) -> List[dict]:
        if self.target_ja_ratio is None:
            return records
        ja = [r for r in records if r.get("language") == "ja"]
        other = [r for r in records if r.get("language") != "ja"]
        total = len(records)
        target_ja = int(round(total * float(self.target_ja_ratio)))
        target_ja = max(0, min(total, target_ja))

        if len(ja) > target_ja:
            ja = self.random.sample(ja, target_ja)
        elif len(ja) < target_ja and other:
            target_other = max(0, total - target_ja)
            if len(other) > target_other:
                other = self.random.sample(other, target_other)
        return ja + other

    def _balance_tags(self, records: List[dict]) -> List[dict]:
        if self.max_per_primary_tag is None:
            return records
        groups: dict[str, list[dict]] = {}
        for record in records:
            tags = record.get("tags") or []
            primary = tags[0] if tags else "unknown"
            groups.setdefault(primary, []).append(record)

        balanced = []
        for primary, items in groups.items():
            if len(items) > self.max_per_primary_tag:
                balanced.extend(self.random.sample(items, self.max_per_primary_tag))
            else:
                balanced.extend(items)
        return balanced


def _env_int(name: str, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default=False):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}
