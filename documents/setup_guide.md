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
