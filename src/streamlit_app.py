import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from llm_client import LLMClient
from query_processor import QueryProcessor
from search_engine import SearchEngine
from retriever import Retriever


@st.cache_resource
def _get_retriever(persist_dir: str, collection_name: str) -> Retriever:
    return Retriever(persist_dir=persist_dir, collection_name=collection_name)


@st.cache_resource
def _get_llm_client(host: str, model: str) -> LLMClient:
    return LLMClient(host=host, model=model)


@st.cache_resource
def _get_query_processor(
    strategy: str,
    host: str,
    translate_model: str,
    enable_expansion: bool,
    dynamic_top_k: bool,
) -> QueryProcessor:
    return QueryProcessor(
        strategy=strategy,
        host=host,
        translate_model=translate_model,
        enable_expansion=enable_expansion,
        dynamic_top_k=dynamic_top_k,
    )


@st.cache_resource
def _get_search_engine(
    retriever: Retriever,
    query_processor: QueryProcessor,
    hybrid_alpha: float,
    candidate_multiplier: int,
) -> SearchEngine:
    return SearchEngine(
        retriever=retriever,
        query_processor=query_processor,
        hybrid_alpha=hybrid_alpha,
        candidate_multiplier=candidate_multiplier,
    )


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
    return re.search(r"[\u3040-\u30ff]", text or "") is not None


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
    default_strategy = os.getenv("QUERY_STRATEGY", "baseline")
    default_translate_model = os.getenv("QUERY_TRANSLATE_MODEL", "qwen3:8b")
    default_hybrid_alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
    default_dynamic_top_k = os.getenv("DYNAMIC_TOP_K", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    default_expansion = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    st.title("Stack Overflow RAG Chatbot")

    with st.sidebar:
        st.header("設定")
        persist_dir = st.text_input("ChromaDB永続化ディレクトリ", value=default_persist_dir)
        collection = st.text_input("ChromaDBコレクション", value=default_collection)
        top_k = st.slider("検索件数 (top_k)", min_value=1, max_value=10, value=5)
        host = st.text_input("Ollama Host", value=default_host)
        model = st.text_input("Ollama Model", value=default_model)
        base_strategy = st.selectbox(
            "検索戦略",
            options=[
                "baseline",
                "translate",
                "hybrid",
                "translate_hybrid",
                "translate_rerank",
                "translate_hybrid_rerank",
            ],
            index=[
                "baseline",
                "translate",
                "hybrid",
                "translate_hybrid",
                "translate_rerank",
                "translate_hybrid_rerank",
            ].index(default_strategy)
            if default_strategy
            in {
                "baseline",
                "translate",
                "hybrid",
                "translate_hybrid",
                "translate_rerank",
                "translate_hybrid_rerank",
            }
            else 0,
        )
        use_mmr = st.checkbox("MMR（多様性確保）", value=False)
        use_multistage = st.checkbox("マルチステージ検索", value=False)
        use_llm_expand = st.checkbox("LLMクエリ拡張", value=False)
        use_context_compress = st.checkbox("重要文抽出（圧縮）", value=False)
        translate_model = st.text_input("翻訳モデル", value=default_translate_model)
        hybrid_alpha = st.slider("ハイブリッド比率（BM25寄り）", min_value=0.0, max_value=1.0, value=default_hybrid_alpha, step=0.1)
        candidate_multiplier = st.slider("候補拡張係数", min_value=1, max_value=5, value=3)
        enable_expansion = st.checkbox("クエリ拡張を有効化", value=default_expansion)
        dynamic_top_k = st.checkbox("動的top_kを有効化", value=default_dynamic_top_k)
        parent_child = st.checkbox("親子チャンク検索", value=True)
        tag_filter = st.text_input("タグフィルタ(カンマ区切り)", value="")
        language_filter = st.selectbox("言語フィルタ", options=["auto", "ja", "en"], index=0)
        min_score = st.number_input("最小スコア", min_value=0, max_value=50, value=0)
        accepted_only = st.checkbox("承認回答のみ", value=False)
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
    strategy = base_strategy
    if use_mmr and "mmr" not in strategy:
        strategy = f"{strategy}_mmr"
    if use_multistage and "multi" not in strategy:
        strategy = f"{strategy}_multi"
    if use_llm_expand and "llm_expand" not in strategy:
        strategy = f"{strategy}_llm_expand"
    os.environ["ENABLE_CONTEXT_COMPRESS"] = "true" if use_context_compress else "false"

    query_processor = _get_query_processor(
        strategy=strategy,
        host=host,
        translate_model=translate_model,
        enable_expansion=enable_expansion,
        dynamic_top_k=dynamic_top_k,
    )
    search_engine = _get_search_engine(
        retriever=retriever,
        query_processor=query_processor,
        hybrid_alpha=hybrid_alpha,
        candidate_multiplier=candidate_multiplier,
    )

    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            t0 = time.perf_counter()
            try:
                t_retrieval_start = time.perf_counter()
                filters = {
                    "tags": [t.strip() for t in tag_filter.split(",") if t.strip()],
                    "language": None if language_filter == "auto" else language_filter,
                    "min_score": min_score if min_score > 0 else None,
                    "accepted_only": accepted_only,
                }
                search_engine.parent_child = parent_child
                search_query, results = search_engine.search(
                    user_text, top_k=top_k, filters=filters
                )
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
                        "strategy": strategy,
                        "search_query": search_query,
                        "hybrid_alpha": hybrid_alpha,
                        "candidate_multiplier": candidate_multiplier,
                        "dynamic_top_k": dynamic_top_k,
                        "enable_expansion": enable_expansion,
                        "use_mmr": use_mmr,
                        "use_multistage": use_multistage,
                        "use_llm_expand": use_llm_expand,
                        "use_context_compress": use_context_compress,
                        "parent_child": parent_child,
                        "filters": filters,
                        "collection": collection,
                        "persist_dir": persist_dir,
                        "ollama_host": host,
                        "ollama_model": model,
                        "translate_model": translate_model,
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
                        "strategy": strategy,
                        "collection": collection,
                        "persist_dir": persist_dir,
                        "ollama_host": host,
                        "ollama_model": model,
                        "translate_model": translate_model,
                        "hybrid_alpha": hybrid_alpha,
                        "candidate_multiplier": candidate_multiplier,
                        "dynamic_top_k": dynamic_top_k,
                        "enable_expansion": enable_expansion,
                        "use_mmr": use_mmr,
                        "use_multistage": use_multistage,
                        "use_llm_expand": use_llm_expand,
                        "use_context_compress": use_context_compress,
                        "parent_child": parent_child,
                        "filters": filters,
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
