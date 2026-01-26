## Streamlit運用手順

### 前提条件

- Python仮想環境が有効（`source venv/bin/activate`）
- 依存関係をインストール済み（`pip install -r requirements.txt`）
- Ollamaが起動している（`OLLAMA_HOST`が有効）

### 起動

```bash
venv/bin/python -m streamlit run src/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

- 同一PCからのアクセス: `http://127.0.0.1:8501`
- 自宅LAN内の別PCからのアクセス: `http://<このPCのIP>:8501`

### 停止

```bash
kill $(cat logs/streamlit.pid)
```

### 再起動（起動済みなら再起動、未起動なら起動）

```bash
bash scripts/restart_streamlit.sh
```

### PIDファイル更新（必要時のみ）

```bash
ss -ltnp | rg ":8501"
```

上記の出力にある `pid=xxxx` を `logs/streamlit.pid` に書き換えてください。

### 起動確認

```bash
ss -ltnp | rg ":8501"
curl -sSf http://127.0.0.1:8501 >/dev/null && echo "http ok"
```

### トラブルシュート

- 8501使用中のエラー: 既存プロセスを停止してから起動
- 日本語で回答しない: `src/llm_client.py` 更新後に再起動
- 別PCから見えない: `--server.address 0.0.0.0` で起動し、必要ならFWで `8501/tcp` を許可
