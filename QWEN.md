# QWEN.md — Stack Overflow RAG Chatbot

## Directory Overview

This directory is a **RAG (Retrieval-Augmented Generation) chatbot prototype** that answers Japanese questions by retrieving relevant Q&A pairs from Stack Overflow. The long-term goal is to abstract the data source so it can be swapped for internal systems like Redmine.

- **Language**: Python 3.x
- **Key libraries**: ChromaDB (vector DB), sentence-transformers/bge-m3 (embeddings), Ollama/qwen3:14b (LLM), Streamlit (UI), rank-bm25 (keyword search)
- **Data sources**: Stack Overflow API (Option C strategy — accepted answer + highest-scoring non-accepted answer per question)

## Project Structure

```
stack_overflow_chatbot/
├── src/                          # Core application code (16 Python files)
│   ├── data_loader.py            # Fetches Q&A from Stack Overflow API
│   ├── preprocessor.py           # HTML cleaning, language detection, QA filtering
│   ├── indexer.py                # Chunking + embedding + ChromaDB persistence
│   ├── retriever.py              # ChromaDB wrapper (vector similarity search)
│   ├── hybrid_search.py          # BM25 keyword search on top of vector index
│   ├── mmr.py                    # Maximal Marginal Relevance for diversity
│   ├── reranker.py               # Cross-encoder weighted fusion reranking
│   ├── query_processor.py        # Query transform: translate, expand, domain normalize
│   ├── domain_dict.py            # Domain term normalization (e.g. "raspberry pi" <-> "ラズパイ")
│   ├── search_engine.py          # Orchestrator: combines all retrieval steps
│   ├── llm_client.py             # Ollama client: prompt building, generation, Japanese enforcement
│   ├── streamlit_app.py          # Browser UI with sidebar config for all pipeline knobs
│   ├── main.py                   # CLI entry point (index build / single query / interactive mode)
│   └── bulk_fetch_and_index.py   # Bulk multi-tag data acquisition script
├── scripts/                      # Operational & evaluation scripts
│   ├── compare_rag_vs_direct.py           # A/B: RAG vs direct LLM answering
│   ├── eval_fair_rag_direct_models.py     # Fair comparison with id_only / content_judge modes
│   ├── improvement_loop.py                # Automated scoring loop for iterative tuning
│   ├── rescore_content_judge_form_free.py# Grounding/task-success scoring (lexical overlap fallback)
│   ├── nightly_run.sh                     # Nightly orchestration: fetch → index → evaluate
│   └── nightly_supervisor.sh              # Supervisor wrapper for nightly runs
├── data/
│   ├── raw/                       # Raw SO API JSON outputs
│   └── processed/                 # Eval CSVs, evaluation result artifacts
├── vector_db/                     # ChromaDB persistent files (cosine space)
├── documents/                     # Design docs, setup guide, TODO management
├── tests/                         # Currently empty (no automated test suite)
├── logs/                          # Runtime logs + perf.log
├── .env.example                   # 60+ configuration variables documented
├── requirements.txt               # Core dependencies
└── check_indexer_status.py        # Quick index sanity-check CLI tool
```

## Current State (as of 2026-05-09)

| Phase | Status | Notes |
|-------|--------|-------|
| 0: Environment setup | ✅ Done | Dev env, model selection (ChromaDB, bge-m3, qwen3:14b) |
| 1: MVP prototype | ✅ Done | Data loading → indexing → search → answer generation end-to-end |
| 2: RAG evaluation | ⚠️ Partial | Evaluation scripts exist; 500-question eval set only partially built (20 questions) |
| 3: Improvement loop | ⚠️ Partial | Automated scoring scripts exist but baseline not fully established |
| 4: Feature expansion | 📋 Planned | Parent-child chunks, hybrid ratio experiments, reranker model comparison |
| 5: Productionization | 📋 Planned | Data source abstraction (Redmine), scheduled updates |

**Data scale**: ~11,414 questions fetched → ~60,677 chunks in ChromaDB.

## Building and Running

### Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit with your API keys
```

### Data Pipeline
```bash
# Fetch data (Option C: accepted + top-scoring answer per question)
venv/bin/python src/data_loader.py --tag python --max-questions 100

# Build vector index
venv/bin/python src/indexer.py --reset
```

### Querying
```bash
# Single query
venv/bin/python src/main.py --query "リストを逆順にするには？"

# Interactive conversation mode
venv/bin/python src/main.py --interactive

# Streamlit browser UI (full config sidebar)
venv/bin/python -m streamlit run src/streamlit_app.py

# Index sanity check
venv/bin/python check_indexer_status.py
```

### Evaluation
```bash
# RAG vs direct comparison
venv/bin/python scripts/compare_rag_vs_direct.py

# Fair eval (id_only judge mode — default policy)
venv/bin/python scripts/eval_fair_rag_direct_models.py --judge-mode id_only

# Automated improvement loop
venv/bin/python scripts/improvement_loop.py
```

## Configuration (.env)

All configuration is via environment variables (see `.env.example`). Key groups:

| Group | Variables | Purpose |
|-------|-----------|---------|
| **APIs** | `STACK_OVERFLOW_API_KEY`, `OPENAI_API_KEY` | Data fetching & optional OpenAI judge |
| **LLM** | `OLLAMA_HOST`, `OLLAMA_MODEL` (default qwen3:14b) | Answer generation backend |
| **Embedding** | `EMBEDDING_MODEL` (BAAI/bge-m3), `EMBEDDING_DEVICE` | Vector encoding |
| **Vector DB** | `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION` | ChromaDB location & name |
| **Query Strategy** | `QUERY_STRATEGY` — controls full pipeline chain | Pipeline orchestration |
| **Hybrid Search** | `HYBRID_ALPHA` (0.5 = 50/50 BM25:vector) | Keyword vs vector weight |
| **MMR** | `MMR_LAMBDA` (0.7), `MMR_MAX_CANDIDATES` (30) | Diversity parameter |
| **Reranking** | `CROSS_ENCODER_MODEL`, `RERANK_*_WEIGHT` | Multi-signal fusion weights |
| **Preprocessing** | `MIN_QUESTION_SCORE`, `MIN_ANSWER_CHARS`, etc. | Data quality filters |

## Coding Conventions

- **Style**: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes
- **Module separation**: Each pipeline stage has its own module (loader/indexer/retriever/reranker)
- **No formal linter/formatter** is enforced; keep naming and style consistent with neighboring files
- **Configuration**: CLI flags + environment variable fallbacks (see `main.py` for pattern)

## Testing

- **No automated test suite exists yet** (`tests/` is empty of Python files).
- Development testing relies on:
  - CLI smoke tests (`venv/bin/python src/main.py --query "..."`)
  - Index sanity check (`venv/bin/python check_indexer_status.py`)
  - Manual evaluation scripts in `scripts/`
- When adding new logic, consider writing tests under `tests/test_<feature>.py`.

## Evaluation Policy

- **Default judge mode**: `judge-mode id_only` — checks if the answer references target question/answer IDs (no cloud API judge scoring).
- Gold answers are derived on-the-fly from raw SO data (accepted answer or highest-scoring answer per question).
- There is no persistent standalone gold answer dataset file.

## Key Design Decisions

1. **Option C answer selection**: Each question gets "accepted answer + top-scoring non-accepted answer". Chosen for accuracy over implementation simplicity.
2. **Offline search design**: Data is pre-indexed; queries do not hit live APIs during inference.
3. **Minimal LlamaIndex usage**: Critical pipeline parts are hand-implemented for extensibility and debugging.
4. **Parent-child chunking**: Three chunk types per question — `qa_parent` (question+answer combined), `question_only`, `answer_only`, plus semantic `answer_child` segments. This enables flexible search strategies (dual index, tag-split, multistage).
5. **Query strategy pipeline**: The `QUERY_STRATEGY` env var chains operations like `translate_hybrid_mmr_rerank_llm_expand_compress_fusion_tag_split_dual`. Each stage can be toggled via CLI flags or the Streamlit sidebar.

## What's NOT in Scope (for now)

- Parent-child chunk redesign (Phase 4)
- Hybrid search ratio experiments (Phase 4)
- Reranker model comparison (Phase 4)
- Scheduled data updates (Phase 5)
- Web API / authentication / multi-user features (Phase 5+)
