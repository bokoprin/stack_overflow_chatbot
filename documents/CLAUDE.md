# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Stack Overflowの質問・回答データを使ったRAGチャットボットシステム。最終的には会社のRedmineチャットボットへ展開し、各部署が独自データソースで使える汎用RAGシステム基盤を構築する。

**重要**: このプロジェクトはAIとの協働を前提としており、コードベースで構築する理由は「AIがシステム構造を把握・理解・修正できる」ため。Difyなどノーコードツールは避ける。

**現在の進捗**: Phase 1-2完了（データ取得・インデックス構築）、Phase 1-3（検索・回答生成）に進行中

## 技術スタック

- **言語**: Python 3.12
- **RAGフレームワーク**: LlamaIndex（薄く使用、重要部分は自前実装）
- **LLM**: qwen3:14b（ローカル実行 via Ollama）
- **ベクトルDB**: ChromaDB（MVP用。将来的にQdrantへ移行予定）
- **埋め込みモデル**: bge-m3（多言語対応、日英両対応）
- **データソース**: Stack Overflow英語版（stackoverflow.com）
- **運用言語**: 日本語（bge-m3による多言語検索で実現）

## 開発環境セットアップ

### 初回セットアップ
```bash
# 仮想環境作成
python -m venv venv

# 仮想環境の有効化
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 依存パッケージインストール
pip install -r requirements.txt

# 環境変数設定
cp .env.example .env
# .envファイルを編集して設定値を入力

# Ollamaモデルのダウンロード（別途Ollamaのインストールが必要）
ollama pull qwen3:14b
```

### よく使うコマンド

```bash
# 仮想環境の有効化（毎回必要）
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Stack Overflowからデータ取得（100件のPython質問）
venv\Scripts\python.exe src/data_loader.py  # Windows
python src/data_loader.py  # Linux/macOS

# ベクトルDBインデックス構築（初回は10-30分かかる：bge-m3モデルダウンロード）
venv\Scripts\python.exe src/indexer.py  # Windows
python src/indexer.py  # Linux/macOS

# インデックス構築状態の確認
venv\Scripts\python.exe check_indexer_status.py  # Windows
python check_indexer_status.py  # Linux/macOS
```

## アーキテクチャの重要な設計原則

### 1. データ構造：軽量親子チャンク方式（現在の実装）

**実装済み（Phase 1-2）**:
```
1質問につき 2-3チャンク:
├─ チャンク1: 質問（タイトル + 本文）
├─ チャンク2: 回答1（承認回答 or 高スコア回答）
└─ チャンク3: 回答2（存在する場合の高スコア回答）
```

**データ取得戦略（Option C）**:
- Stack Overflow APIから承認回答 + 高スコア回答（非承認の中で最高スコア）を取得
- 質問100件 → 約287チャンク（平均2.87チャンク/質問）

**将来の拡張（Phase 2以降）**:
```
親チャンク: 1質問スレッド全体
 ├─ 子チャンク: 質問タイトル
 ├─ 子チャンク: 質問本文
 ├─ 子チャンク: 回答1（accepted answer）
 ├─ 子チャンク: 回答2
 └─ 子チャンク: コメント
```

検索フロー（将来）:
1. 子チャンクに対してハイブリッド検索（キーワード + ベクトル）
2. リランクモデルで絞り込み
3. `parent_question_id`から親チャンクを取得
4. 親チャンク全体をコンテキストとしてLLMに渡す

### 2. ハイブリッド検索の比率実験（Phase 2で実装予定）

**現在（Phase 1）**: ベクトル検索のみ

**将来実装する3パターン**:
- **10:0** - キーワード100%（専門用語が明確な場合に強い）
- **5:5** - ハイブリッド50:50（バランス型）
- **0:10** - ベクトル100%（曖昧な質問や言い換えに強い）

同じ質問で3パターンを実行し、最適な比率を検証する。

### 3. オフライン検索設計

データは事前にベクトルDB化し、検索時はStack Overflow APIを叩かない。これにより高速・低負荷を実現。

### 4. LlamaIndexの使い方

- **コア部分**: LlamaIndexに任せる
- **データ取得・前処理**: 自前実装（カスタマイズ可能にする）
- **理由**: 拡張性を確保し、データソース差し替え（Redmine、社内Wiki等）を容易にする

## 段階的実装戦略

### Phase 1: MVP（現在のフェーズ）

**完了済み**:
- ✅ Phase 1-1: Stack Overflow APIから100件のPythonデータ取得（Option C戦略）
- ✅ Phase 1-2: 軽量親子チャンク分割 + ChromaDBインデックス構築
  - 質問（タイトル+本文）で1チャンク
  - 各回答で1チャンク
  - 合計約287チャンク

**作業中**:
- 🔄 Phase 1-3: ベクトル検索 + LLM回答生成機能
- ⏳ Phase 1-4: 統合・動作確認

**MVP成功条件**: 日本語質問に対して関連するStack Overflow回答を返すこと

### Phase 2: 機能拡張（将来）
- ハイブリッド検索追加（5:5固定）
- 比率実験機能（10:0, 5:5, 0:10）
- リランク機能追加
- 詳細な親子チャンク分割（質問本文、タイトル、コメント別）

### Phase 3: 本番化（将来）
- データソース抽象化（Redmine対応）
- 定期更新機能
- 本番環境対応

## 現在のプロジェクト構成

```
stackoverflow-chatbot/
├── documents/                   # 設計ドキュメント
│   ├── AGENTS.md               # リポジトリ運用ガイド
│   ├── CLAUDE.md               # このファイル
│   └── todo.md                 # タスク管理
├── data/
│   ├── raw/                    # Stack Overflow API取得データ（JSON）
│   └── processed/              # 処理済みデータ
├── vector_db/                  # ChromaDB保存先
├── src/                        # 実装コード
├── tests/                      # テスト（未整備）
└── .env                        # 環境変数（gitignore対象）
```

**実装済みのファイル（Phase 1-3）**:
- `src/retriever.py` - ベクトル検索機能
- `src/llm_client.py` - LLM接続・回答生成
- `src/main.py` - CLI統合
- `check_indexer_status.py` - インデックス状態確認

## コーディング規約

- **コメント**: 日本語OK（開発者が日本語ネイティブ）
- **関数・変数名**: 英語（snake_case）
- **docstring**: 必要に応じて記述
- **エラーハンドリング**: try-catchで適切に処理

## 開発者の背景と注意点

- 組込みエンジニア出身（C言語、マイコン）
- AI・機械学習、LlamaIndex、RAGシステムは初めて
- 完璧主義になりがち → **まず動くものを作ることを優先**
- 細部にこだわりすぎない → **プロトタイプ段階では荒くてOK**

## 実装済みの重要機能

### src/data_loader.py
- `StackOverflowDataLoader` クラス
- `fetch_questions()`: Stack Overflow APIから質問リストを取得
- `fetch_answers()`: 指定した質問の回答を取得
- `fetch_question_with_answers()`: Option C戦略（承認回答 + 高スコア回答）
- `process_questions_with_answers()`: 質問+回答をまとめて取得
- `save_to_json()`: データをJSON形式で保存（data/raw/）

**Option C戦略の詳細**:
- 承認回答（accepted_answer）があれば必ず取得
- 承認回答以外で最高スコアの回答を1件取得
- 1質問あたり最大2件の回答を取得

### src/indexer.py
- `StackOverflowIndexer` クラス
- `strip_html()`: Stack OverflowのHTMLタグを除去（BeautifulSoup4使用）
- `create_chunks()`: 軽量親子チャンク分割
  - 質問（タイトル+本文）→ 1チャンク
  - 各回答 → 1チャンク
- `build_index()`: ChromaDBへのインデックス構築
- 埋め込みモデル: bge-m3（HuggingFace）
- 初回実行時にbge-m3モデル（約2GB）を自動ダウンロード

### check_indexer_status.py
- ベクトルDBディレクトリの存在確認
- ChromaDBコレクションの確認
- 簡易検索テスト（動作確認用）

### src/retriever.py
- `search()`: 質問のベクトル検索
- 検索結果（本文・メタデータ）を返却

### src/llm_client.py
- `build_prompt()`: コンテキスト付きプロンプト生成
- `generate_answer()`: Ollama問い合わせ

### src/main.py
- `--index` でインデックス構築
- `--query` で検索 + 回答生成
- `--interactive` で対話モード

## メタデータ構造（実装済み）

各チャンクに付与するメタデータ:
- `question_id`: 質問ID（全チャンクに付与）
- `chunk_type`: タイプ（question / answer）
- `chunk_index`: チャンク番号（0から開始）
- `tags`: Stack Overflowのタグ（質問チャンクのみ）
- `score`: スコア（回答チャンクのみ）
- `is_accepted`: 承認された回答かどうか（回答チャンクのみ）
- `title`: 質問タイトル（質問チャンクのみ）
- `link`: Stack Overflowのリンク（質問チャンクのみ）

## 選定基準

### ベクトルDB
- ローカル実行可能
- Pythonから簡単に使える
- LlamaIndexと連携可能
- 軽量

### 埋め込みモデル
- ローカル実行可能
- メモリ使用量が少ない
- 日本語・英語両対応
- LlamaIndexと連携可能

## 次のステップ（Phase 1-3）

Phase 1-3の実装は完了。次は動作確認を行う:

- `python src/indexer.py` でインデックス構築
- `python src/main.py --query "質問文"` で回答生成

## 参照ドキュメント

- [documents/design_decisions.md](documents/design_decisions.md) - 技術選定・設計判断の記録
- [documents/todo.md](documents/todo.md) - タスク管理・進捗記録
- [documents/setup_guide.md](documents/setup_guide.md) - 詳細なセットアップ手順
- [documents/仕様書.md](documents/仕様書.md) - システム仕様の詳細
- [README.md](README.md) - プロジェクト概要説明
- [START_HERE.md](START_HERE.md) - 作業開始時の確認事項
- [OVERNIGHT_REPORT.md](OVERNIGHT_REPORT.md) - Phase 1-2完了レポート
