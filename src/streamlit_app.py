import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from llm_client import LLMClient
from retriever import Retriever


@st.cache_resource
def _get_retriever(persist_dir: str, collection_name: str) -> Retriever:
    return Retriever(persist_dir=persist_dir, collection_name=collection_name)


@st.cache_resource
def _get_llm_client(host: str, model: str) -> LLMClient:
    return LLMClient(host=host, model=model)


def _format_source_line(index: int, item: dict) -> str:
    metadata = item.get("metadata") or {}
    line = f"[{index}] type={metadata.get('chunk_type')} qid={metadata.get('question_id')}"
    if metadata.get("score") is not None:
        line += f" score={metadata.get('score')}"
    if metadata.get("is_accepted") is not None:
        line += f" accepted={metadata.get('is_accepted')}"
    if metadata.get("link"):
        line += f" link={metadata.get('link')}"
    return line


def _log_perf(payload: dict) -> None:
    try:
        log_path = Path("logs") / "perf.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
        log_path.write_text("", encoding="utf-8") if not log_path.exists() else None
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _contains_japanese(text: str) -> bool:
    return re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text or "") is not None


def main():
    if get_script_run_ctx() is None:
        raise SystemExit(
            "このファイルは `python src/streamlit_app.py` では起動できません。\n"
            "`venv/bin/python -m streamlit run src/streamlit_app.py` で起動してください。"
        )

    st.set_page_config(page_title="Stack Overflow RAG Chatbot", page_icon="💬", layout="wide")

    default_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "vector_db")
    default_collection = os.getenv("CHROMA_COLLECTION", "stack_overflow")
    default_model = os.getenv("OLLAMA_MODEL", "qwen3:14b")
    default_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    st.title("Stack Overflow RAG Chatbot")

    with st.sidebar:
        st.header("設定")
        persist_dir = st.text_input("ChromaDB永続化ディレクトリ", value=default_persist_dir)
        collection = st.text_input("ChromaDBコレクション", value=default_collection)
        top_k = st.slider("検索件数 (top_k)", min_value=1, max_value=10, value=5)
        host = st.text_input("Ollama Host", value=default_host)
        model = st.text_input("Ollama Model", value=default_model)
        show_sources = st.checkbox("参照（Sources）を表示", value=True)
        show_timing = st.checkbox("処理時間を表示", value=True)
        if st.button("履歴をクリア"):
            st.session_state.pop("messages", None)
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                meta = message.get("meta") or {}
                if show_timing and meta.get("timing"):
                    st.caption(meta["timing"])
                if show_sources and meta.get("sources"):
                    with st.expander("参照", expanded=False):
                        st.markdown("\n".join(meta["sources"]))

    user_text = st.chat_input("質問を入力してください")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    retriever = _get_retriever(persist_dir=persist_dir, collection_name=collection)
    llm = _get_llm_client(host=host, model=model)

    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            t0 = time.perf_counter()
            try:
                t_retrieval_start = time.perf_counter()
                results = retriever.search(user_text, top_k=top_k)
                t_retrieval = time.perf_counter() - t_retrieval_start

                prompt = llm.build_prompt(user_text, results)
                t_llm_start = time.perf_counter()
                answer = llm.generate_answer(prompt)
                t_llm = time.perf_counter() - t_llm_start

                sources = [_format_source_line(i, item) for i, item in enumerate(results, start=1)]
                timing = f"retrieval={t_retrieval:.2f}s, llm={t_llm:.2f}s, total={time.perf_counter() - t0:.2f}s"
                _log_perf(
                    {
                        "query": user_text,
                        "top_k": top_k,
                        "retrieval_sec": round(t_retrieval, 3),
                        "llm_sec": round(t_llm, 3),
                        "total_sec": round(time.perf_counter() - t0, 3),
                        "collection": collection,
                        "persist_dir": persist_dir,
                        "ollama_host": host,
                        "ollama_model": model,
                        "answer_japanese": _contains_japanese(answer),
                    }
                )
            except Exception as exc:
                answer = f"エラー: {exc}"
                sources = []
                timing = None
                _log_perf(
                    {
                        "query": user_text,
                        "top_k": top_k,
                        "collection": collection,
                        "persist_dir": persist_dir,
                        "ollama_host": host,
                        "ollama_model": model,
                        "error": str(exc),
                    }
                )

            st.markdown(answer)
            if show_timing and timing:
                st.caption(timing)
            if show_sources and sources:
                with st.expander("参照", expanded=False):
                    st.markdown("\n".join(sources))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "meta": {"sources": sources, "timing": timing},
        }
    )


if __name__ == "__main__":
    main()
