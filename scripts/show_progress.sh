#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"

echo "== indexer =="
if [[ -f "$LOG_DIR/indexer_run.log" ]]; then
  grep -E "indexed [0-9]+/[0-9]+" "$LOG_DIR/indexer_run.log" | tail -n 1 || true
else
  echo "indexer_run.log not found"
fi

echo ""
echo "== evaluation =="
if [[ -f "$LOG_DIR/progress.log" ]]; then
  tail -n 1 "$LOG_DIR/progress.log"
else
  echo "progress.log not found"
fi
