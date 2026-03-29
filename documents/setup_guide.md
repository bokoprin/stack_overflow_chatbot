# セットアップガイド

このガイドはローカル環境でRAGチャットボットを動かすための手順をまとめたものです。

## 1. Python仮想環境の作成

```bash
python -m venv venv
```

有効化:

```bash
source venv/bin/activate
```

## 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## 3. 環境変数の準備

`.env.example` をコピーして `.env` を作成し、値を設定します。

```bash
cp .env.example .env
```

設定項目:

- `STACK_OVERFLOW_API_KEY`: Stack Overflow APIキー
- `OLLAMA_HOST`: OllamaのURL（既定は `http://localhost:11434`）
- `OLLAMA_MODEL`: 使用するモデル名（例: `qwen3:14b`）
- `CHROMA_PERSIST_DIR`: ChromaDBの永続化ディレクトリ
- `OPENAI_API_KEY`: 採点用OpenAI APIキー
- `OPENAI_JUDGE_MODEL`: 採点モデル（既定: `gpt-5.4-mini`）
- `LLAMA_CPP_BASE_URL`: llama.cppサーバURL（既定: `http://127.0.0.1:8080`）
- `LLAMA_CPP_MODEL`: llama.cppで使うモデル名（既定: `qwen3.5-27b-iq3m-96k`）
- `EVAL_QUESTIONS_CSV`: 比較対象の質問CSV
- `RAW_SOURCE_JSON`: 生データ直接参照で使うJSON

## 4. Ollamaの準備

Ollamaをインストールした上で、モデルを取得します。

```bash
ollama pull qwen3:14b
```

## 5. データ取得とインデックス構築

```bash
python src/data_loader.py
python src/indexer.py
```

## 6. 動作確認

```bash
python check_indexer_status.py
```

## 7. RAG vs 生データ直接の比較評価

```bash
venv/bin/python scripts/compare_rag_vs_direct.py --limit 2
```

フル実行時は `--limit` を外してください。結果は `data/processed/eval_compare_YYYYMMDD_HHMMSS/` 配下に保存されます。
