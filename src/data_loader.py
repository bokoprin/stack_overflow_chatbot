import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


class StackOverflowDataLoader:
    def __init__(
        self,
        api_key=None,
        site="stackoverflow",
        tag="python",
        pagesize=100,
        max_questions=100,
        order="desc",
        sort="votes",
        timeout=30,
        request_delay=0.1,
        max_retries=5,
        retry_wait=10,
    ):
        self.api_key = api_key
        self.site = site
        self.tag = tag
        self.pagesize = min(pagesize, 100)
        self.max_questions = max_questions
        self.order = order
        self.sort = sort
        self.timeout = timeout
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.base_url = "https://api.stackexchange.com/2.3"
        self.session = requests.Session()

    def _request(self, path, params):
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = dict(params)
        params["site"] = self.site
        if self.api_key:
            params["key"] = self.api_key
        for attempt in range(self.max_retries + 1):
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = self.retry_wait
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                time.sleep(wait_time)
                continue

            try:
                data = response.json()
            except ValueError:
                data = {}

            backoff = data.get("backoff")
            if backoff:
                time.sleep(backoff)

            if response.status_code >= 400 or "error_id" in data:
                message = data.get("error_message", response.text)
                raise RuntimeError(f"Stack Exchange API error: {message}")

            time.sleep(self.request_delay)
            return data

        raise RuntimeError("Stack Exchange API error: exceeded retry limit")

    def fetch_questions(self):
        questions = []
        page = 1
        while len(questions) < self.max_questions:
            page_size = min(self.pagesize, self.max_questions - len(questions))
            params = {
                "page": page,
                "pagesize": page_size,
                "order": self.order,
                "sort": self.sort,
                "tagged": self.tag,
                "filter": "withbody",
            }
            data = self._request("questions", params)
            items = data.get("items", [])
            if not items:
                break
            questions.extend(items)
            if not data.get("has_more"):
                break
            page += 1
        return questions

    def fetch_answers(self, question_id):
        params = {
            "order": "desc",
            "sort": "votes",
            "pagesize": 100,
            "filter": "withbody",
        }
        data = self._request(f"questions/{question_id}/answers", params)
        return data.get("items", [])

    def fetch_answer_by_id(self, answer_id):
        params = {"filter": "withbody"}
        data = self._request(f"answers/{answer_id}", params)
        items = data.get("items", [])
        return items[0] if items else None

    def fetch_question_with_answers(self, question):
        question_id = question.get("question_id")
        accepted_answer_id = question.get("accepted_answer_id")
        answers = self.fetch_answers(question_id)

        accepted_answer = None
        top_non_accepted = None
        for answer in answers:
            if answer.get("is_accepted"):
                accepted_answer = answer
            elif top_non_accepted is None:
                top_non_accepted = answer

        if not accepted_answer and accepted_answer_id:
            accepted_answer = self.fetch_answer_by_id(accepted_answer_id)

        selected_answers = []
        if accepted_answer:
            selected_answers.append(self._select_answer_fields(accepted_answer))
        if top_non_accepted and (
            not accepted_answer
            or top_non_accepted.get("answer_id") != accepted_answer.get("answer_id")
        ):
            selected_answers.append(self._select_answer_fields(top_non_accepted))

        record = self._select_question_fields(question)
        record["answers"] = selected_answers
        return record

    def process_questions_with_answers(self):
        questions = self.fetch_questions()
        results = []
        for index, question in enumerate(questions, start=1):
            question_id = question.get("question_id")
            try:
                record = self.fetch_question_with_answers(question)
                results.append(record)
                print(f"[{index}/{len(questions)}] fetched question {question_id}")
            except requests.RequestException as exc:
                print(f"[{index}/{len(questions)}] request failed for {question_id}: {exc}")
            except RuntimeError as exc:
                print(f"[{index}/{len(questions)}] api error for {question_id}: {exc}")
        return results

    def save_to_json(self, data, path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return output_path

    def build_summary(self, data):
        question_count = len(data)
        answer_count = sum(len(item.get("answers", [])) for item in data)
        accepted_count = sum(
            1
            for item in data
            for answer in item.get("answers", [])
            if answer.get("is_accepted")
        )
        return {
            "questions": question_count,
            "answers": answer_count,
            "accepted_answers": accepted_count,
        }

    def _select_question_fields(self, question):
        return {
            "question_id": question.get("question_id"),
            "title": question.get("title"),
            "body": question.get("body"),
            "score": question.get("score"),
            "tags": question.get("tags", []),
            "link": question.get("link"),
            "creation_date": question.get("creation_date"),
        }

    def _select_answer_fields(self, answer):
        return {
            "answer_id": answer.get("answer_id"),
            "body": answer.get("body"),
            "score": answer.get("score"),
            "is_accepted": answer.get("is_accepted"),
            "creation_date": answer.get("creation_date"),
        }


def _default_output_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/raw") / f"stackoverflow_qa_{timestamp}.json"


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch Stack Overflow QA data.")
    parser.add_argument("--tag", default=os.getenv("STACK_OVERFLOW_TAG", "python"))
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--pagesize", type=int, default=100)
    parser.add_argument("--site", default="stackoverflow")
    parser.add_argument("--sort", default="votes")
    parser.add_argument("--order", default="desc")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    loader = StackOverflowDataLoader(
        api_key=os.getenv("STACK_OVERFLOW_API_KEY"),
        site=args.site,
        pagesize=args.pagesize,
        max_questions=args.max_questions,
        order=args.order,
        sort=args.sort,
    )
    data = loader.process_questions_with_answers()
    output_path = args.output or _default_output_path()
    saved_path = loader.save_to_json(data, output_path)
    summary = loader.build_summary(data)
    print(f"saved: {saved_path}")
    print(
        "summary: "
        f"{summary['questions']} questions, "
        f"{summary['answers']} answers, "
        f"{summary['accepted_answers']} accepted"
    )


if __name__ == "__main__":
    main()
