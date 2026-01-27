#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/nightly_run.log"
PID_FILE="$LOG_DIR/nightly_run.pid"
INDEX_PID_FILE="$LOG_DIR/indexer_run.pid"

mkdir -p "$LOG_DIR"
echo $$ > "$PID_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

fail() {
  log "ERROR: $*"
}

cleanup() {
  log "nightly_run finished"
}
trap cleanup EXIT

check_cmd() {
  command -v "$1" >/dev/null 2>&1
}

check_disk() {
  local min_kb="${MIN_DISK_KB:-1048576}" # 1GB
  local avail_kb
  avail_kb=$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')
  if [[ -z "$avail_kb" ]]; then
    fail "disk check failed"
    return 1
  fi
  if (( avail_kb < min_kb )); then
    fail "low disk space: ${avail_kb}KB available"
    return 1
  fi
  log "disk ok: ${avail_kb}KB available"
  return 0
}

ollama_host() {
  echo "${OLLAMA_HOST:-http://localhost:11434}"
}

ollama_ready() {
  local host
  host="$(ollama_host)"
  curl -sS "$host/api/tags" >/dev/null 2>&1
}

start_ollama() {
  if ! check_cmd ollama; then
    fail "ollama not found"
    return 1
  fi
  log "starting ollama serve"
  nohup ollama serve > "$LOG_DIR/ollama_serve.log" 2>&1 &
  sleep 3
  return 0
}

ensure_model() {
  local model="$1"
  if [[ -z "$model" ]]; then
    return 0
  fi
  if ! check_cmd ollama; then
    fail "ollama missing; cannot pull $model"
    return 1
  fi
  if ollama list 2>/dev/null | awk '{print $1}' | grep -Fxq "$model"; then
    log "ollama model present: $model"
    return 0
  fi
  log "ollama pull $model"
  if ! ollama pull "$model"; then
    fail "ollama pull failed: $model"
    return 1
  fi
  return 0
}

wait_for_pid() {
  local pid="$1"
  local timeout="${2:-21600}" # 6h
  local start
  start=$(date +%s)
  while kill -0 "$pid" >/dev/null 2>&1; do
    local now
    now=$(date +%s)
    if (( now - start > timeout )); then
      fail "timeout waiting for pid $pid"
      return 1
    fi
    sleep 30
  done
  return 0
}

start_or_wait_indexer() {
  if [[ -f "$INDEX_PID_FILE" ]]; then
    local pid
    pid="$(cat "$INDEX_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      log "indexer already running (pid=$pid), waiting"
      wait_for_pid "$pid" "${INDEX_TIMEOUT_SEC:-21600}" || return 1
      return 0
    fi
  fi

  log "starting full reindex"
  PYTHONUNBUFFERED=1 nohup "$ROOT_DIR/venv/bin/python" "$ROOT_DIR/src/indexer.py" --reset \
    > "$LOG_DIR/indexer_run.log" 2>&1 &
  echo $! > "$INDEX_PID_FILE"
  wait_for_pid "$(cat "$INDEX_PID_FILE")" "${INDEX_TIMEOUT_SEC:-21600}" || return 1
  return 0
}

run_eval_loop() {
  local eval_model="${EVAL_MODEL:-qwen3:8b}"
  local translate_model="${TRANSLATE_MODEL:-qwen3:8b}"
  local strategies="${EVAL_STRATEGIES:-baseline,translate,hybrid,translate_hybrid,translate_rerank,translate_hybrid_rerank,translate_hybrid_mmr,translate_hybrid_mmr_rerank,translate_hybrid_mmr_rerank_multi,translate_hybrid_mmr_rerank_llm_expand,translate_hybrid_mmr_rerank_llm_expand_compress,translate_hybrid_mmr_rerank_llm_expand_compress_fusion,translate_hybrid_mmr_rerank_llm_expand_compress_fusion_tag_split,translate_hybrid_mmr_rerank_llm_expand_compress_fusion_tag_split_dual}"
  local size="${EVAL_SIZE:-500}"
  local top_k="${EVAL_TOP_K:-5}"
  local limit_hours="${EVAL_TIME_LIMIT_HOURS:-7}"

  local embedding_models="${EMBEDDING_MODELS:-}"
  if [[ -z "$embedding_models" ]]; then
    embedding_models="${EMBEDDING_MODEL:-BAAI/bge-m3},intfloat/multilingual-e5-base"
  fi

  local reranker_models="${CROSS_ENCODER_MODELS:-none,BAAI/bge-reranker-base}"

  IFS=',' read -r -a model_list <<< "$embedding_models"
  IFS=',' read -r -a reranker_list <<< "$reranker_models"
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)"

  for model in "${model_list[@]}"; do
    local clean_model
    clean_model="$(echo "$model" | tr '/:.' '___')"
    local collection="stack_overflow_${clean_model}"
    log "reindex for embedding model: $model (collection=$collection)"
    EMBEDDING_MODEL="$model" "$ROOT_DIR/venv/bin/python" "$ROOT_DIR/src/indexer.py" \
      --reset \
      --collection "$collection" \
      >> "$LOG_DIR/indexer_${clean_model}.log" 2>&1 || fail "indexer failed: $model"

    for reranker in "${reranker_list[@]}"; do
      local reranker_tag
      if [[ "$reranker" == "none" || -z "$reranker" ]]; then
        reranker_tag="none"
        export CROSS_ENCODER_MODEL=""
      else
        reranker_tag="$(echo "$reranker" | tr '/:.' '___')"
        export CROSS_ENCODER_MODEL="$reranker"
        log "preloading reranker: $reranker"
        "$ROOT_DIR/venv/bin/python" - <<PY >> "$LOG_DIR/reranker_${reranker_tag}.log" 2>&1
from sentence_transformers import CrossEncoder
CrossEncoder("${reranker}")
PY
      fi
      local output_dir="$ROOT_DIR/data/processed/eval_${run_id}_${clean_model}_${reranker_tag}"
      log "starting improvement loop (collection=$collection, reranker=$reranker_tag)"
      CHROMA_COLLECTION="$collection" EMBEDDING_MODEL="$model" "$ROOT_DIR/venv/bin/python" \
        "$ROOT_DIR/scripts/improvement_loop.py" \
        --size "$size" \
        --top-k "$top_k" \
        --time-limit-hours "$limit_hours" \
        --strategies "$strategies" \
        --eval-model "$eval_model" \
        --translate-model "$translate_model" \
        --output-dir "$output_dir" \
        >> "$LOG_DIR/improvement_loop_${clean_model}_${reranker_tag}.log" 2>&1
    done
  done
}

log "nightly_run started"
check_disk || true

if ! ollama_ready; then
  start_ollama || true
fi

if ollama_ready; then
  ensure_model "${OLLAMA_MODEL:-qwen3:14b}" || true
  ensure_model "${QUERY_TRANSLATE_MODEL:-qwen3:8b}" || true
else
  log "ollama not available; evaluation steps may be skipped"
fi

if start_or_wait_indexer; then
  log "indexer completed"
else
  fail "indexer failed or timed out"
fi

if ollama_ready; then
  run_eval_loop || fail "improvement loop failed"
else
  fail "skipping evaluation: ollama not available"
fi

log "nightly_run done"
