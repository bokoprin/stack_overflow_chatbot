# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core Python implementation for ingestion, indexing, retrieval, reranking, and UI (`main.py`, `indexer.py`, `streamlit_app.py`).
- `scripts/`: operational scripts (nightly runs, supervisor, restart helpers).
- `data/raw/` and `data/processed/`: fetched Stack Overflow data and evaluation artifacts.
- `vector_db/`: persistent ChromaDB files (large generated artifacts; do not hand-edit).
- `logs/`: runtime logs and PID files.
- `documents/`: setup and architecture notes.
- `tests/`: currently minimal; add automated tests here as coverage grows.

Keep application logic in `src/` and keep generated outputs under `data/`, `vector_db/`, and `logs/`.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate`: create and activate local environment.
- `pip install -r requirements.txt`: install dependencies.
- `cp .env.example .env`: create local config, then fill required values.
- `venv/bin/python src/data_loader.py --tag python --max-questions 100`: fetch sample data.
- `venv/bin/python src/indexer.py --reset`: rebuild vector index.
- `venv/bin/python src/main.py --query "..."`: run one-shot CLI query.
- `venv/bin/python src/main.py --interactive`: start interactive CLI mode.
- `venv/bin/python -m streamlit run src/streamlit_app.py`: launch browser UI.
- `venv/bin/python check_indexer_status.py`: quick index sanity check.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, `snake_case` functions/variables, `PascalCase` classes.
- Keep modules focused by pipeline stage (loader/indexer/retriever/reranker).
- Prefer explicit CLI flags and environment-variable fallbacks (existing pattern in `src/main.py`).
- No formatter/linter is enforced yet; keep imports, naming, and argument style consistent with neighboring files.

## Testing Guidelines
- No full test suite is currently enforced.
- For each change, run targeted checks:
  - `venv/bin/python check_indexer_status.py`
  - relevant CLI path (`src/main.py --query ...`) and/or Streamlit smoke run.
- Add new automated tests under `tests/` using `test_<feature>.py` naming when introducing non-trivial logic.

## Commit & Pull Request Guidelines
- Local history shows short Japanese summaries (for example, `更新`, `Streamlit UI追加...`).
- Prefer clear, scoped commit messages over generic `更新`:
  - Example: `indexer: add parent-child chunk metadata`
- PRs should include:
  - purpose and user-facing impact,
  - verification commands/results,
  - related issue/context,
  - screenshots only for UI changes.

## Security & Configuration Tips
- Never commit `.env` or API keys.
- Treat `data/raw/`, `vector_db/`, and `logs/` as generated/runtime state; avoid unnecessary diffs.
- When adding new config, update both `.env.example` and `documents/setup_guide.md`.
