# Repository Guidelines

## Project Structure & Module Organization
This repository currently includes `documents/CLAUDE.md`, `documents/todo.md`, and a hidden `.env`. The working layout described in `documents/CLAUDE.md` expects these directories when the code is present:

- `src/`: core Python modules (e.g., data loader, indexer, retriever, LLM client).
- `data/raw` and `data/processed`: Stack Overflow JSON input and cleaned artifacts.
- `vector_db/`: ChromaDB persistence for embeddings.
- `documents/`: design notes and setup references.

Keep source logic under `src/`, keep generated artifacts under `data/` or `vector_db/`, and avoid mixing data with code.

## Build, Test, and Development Commands
Typical local workflow (Python 3.12):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python src/data_loader.py
python src/indexer.py
python check_indexer_status.py
```

Use `src/data_loader.py` to fetch Stack Overflow data, `src/indexer.py` to build the vector index, and `check_indexer_status.py` for a quick sanity check.

## Coding Style & Naming Conventions
- Indentation: 4 spaces, Python 3.12.
- Naming: English `snake_case` for functions/variables.
- Comments: Japanese is acceptable; keep them short and purposeful.
- Error handling: prefer explicit `try/except` around I/O and external calls.
- Formatting/linting: none configured yet; keep code consistent with existing modules.

## Testing Guidelines
No test framework is set up in this snapshot. If you add tests, place them under `tests/` and document the command (e.g., `pytest`) in this guide and in any PR description.

## Commit & Pull Request Guidelines
There is no Git history in this snapshot, so no established commit convention is visible. Use clear, imperative messages such as `Add indexer status check`. For PRs, include:
- What changed and why
- How to run or verify (commands, sample inputs)
- Linked issues or context
- Screenshots only if you introduce UI assets

## Security & Configuration Tips
`.env` likely contains secrets or API keys. Do not commit it. If you add new configuration values, also add an `.env.example` template with placeholder values.
