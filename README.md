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

## 現状

- Phase 0: 環境準備中
- Phase 1-3（検索・回答生成）に進む前の準備段階

## ライセンス

未定
