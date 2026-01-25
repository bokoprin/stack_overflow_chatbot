# Stack Overflow RAG Chatbot

Stack Overflowの質問・回答データを使ったRAGチャットボットのプロトタイプです。ローカルLLMとベクトル検索を組み合わせ、日本語の質問に対して関連する回答を返します。将来的には社内Redmineへ展開可能な汎用RAG基盤として拡張します。

## 目的

- Stack OverflowデータによるRAG構成の検証
- 検索精度・回答品質のベースライン確認
- データソース差し替え可能な構造の準備

## 構成

```
stackoverflow-chatbot/
├── documents/         # 設計・仕様・タスク管理
├── data/
│   ├── raw/           # Stack Overflow API取得データ
│   └── processed/     # 前処理済みデータ
├── vector_db/         # ChromaDB永続化
├── src/               # 実装コード
├── tests/             # テスト（未整備）
├── .env.example       # 環境変数テンプレート
└── README.md
```

## セットアップ

セットアップ手順は `documents/setup_guide.md` を参照してください。

## 使い方

データ取得 → インデックス構築 → 質問の順で進めます。

```bash
# データ取得
venv/bin/python src/data_loader.py --tag python --max-questions 100

# インデックス構築
venv/bin/python src/indexer.py --reset

# 単発の質問
venv/bin/python src/main.py --query "リストを逆順にするには？"

# 対話モード
venv/bin/python src/main.py --interactive

# ブラウザUI（Streamlit）
venv/bin/python -m streamlit run src/streamlit_app.py

# インデックス状態の確認
venv/bin/python check_indexer_status.py
```

## 現状

- Phase 1-3（検索・回答生成・対話モード）まで実装済み

## ライセンス

未定
