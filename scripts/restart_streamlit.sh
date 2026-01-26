#!/usr/bin/env bash
set -euo pipefail

APP_PATH="src/streamlit_app.py"
HOST="0.0.0.0"
PORT="8501"
PYTHON_BIN="venv/bin/python"
PID_FILE="logs/streamlit.pid"
LOG_FILE="logs/streamlit.log"

mkdir -p logs

get_pid_from_pidfile() {
  if [ -f "$PID_FILE" ]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return
    fi
  fi
}

get_pid_from_port() {
  ss -ltnp 2>/dev/null | rg ":${PORT}\\b" | sed -n 's/.*pid=\\([0-9]*\\).*/\\1/p' | head -n 1
}

stop_pid() {
  local pid="$1"
  echo "stopping pid=${pid}"
  kill "$pid" 2>/dev/null || true
}

wait_port_free() {
  local limit=15
  local count=0
  while ss -ltnp 2>/dev/null | rg -q ":${PORT}\\b"; do
    sleep 1
    count=$((count + 1))
    if [ "$count" -ge "$limit" ]; then
      echo "port ${PORT} is still in use"
      return 1
    fi
  done
}

running_pid="$(get_pid_from_pidfile || true)"
if [ -z "$running_pid" ]; then
  running_pid="$(get_pid_from_port || true)"
fi

if [ -n "$running_pid" ]; then
  stop_pid "$running_pid"
  wait_port_free
else
  echo "no running process detected, starting new instance"
fi

nohup "$PYTHON_BIN" -m streamlit run "$APP_PATH" \
  --server.address "$HOST" \
  --server.port "$PORT" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
sleep 2

if ss -ltnp 2>/dev/null | rg -q ":${PORT}\\b"; then
  echo "started pid=$(cat "$PID_FILE")"
  exit 0
fi

echo "failed to start streamlit (see ${LOG_FILE})"
exit 1
