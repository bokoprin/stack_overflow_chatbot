#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/nightly_supervisor.log"
STATE_FILE="$LOG_DIR/nightly_supervisor.state"
RUN_LOG="$LOG_DIR/nightly_run.log"
PID_FILE="$LOG_DIR/nightly_run.pid"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

get_timeout_count() {
  if [[ -f "$RUN_LOG" ]]; then
    grep -c "timeout waiting for pid" "$RUN_LOG" || true
  else
    echo 0
  fi
}

read_state() {
  if [[ -f "$STATE_FILE" ]]; then
    cat "$STATE_FILE"
  else
    echo 0
  fi
}

write_state() {
  echo "$1" > "$STATE_FILE"
}

start_nightly() {
  log "starting nightly_run"
  nohup bash "$ROOT_DIR/scripts/nightly_run.sh" >> "$LOG_DIR/nightly_run.out" 2>&1 &
  echo $! > "$PID_FILE"
}

stop_nightly() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      log "stopping nightly_run pid=$pid"
      kill "$pid" 2>/dev/null || true
    fi
  fi
}

main_loop() {
  local delay="${SUPERVISOR_CHECK_INTERVAL:-300}"
  local restart_delay="${SUPERVISOR_RESTART_DELAY:-60}"
  local timeout_count
  local last_timeout_count

  last_timeout_count="$(read_state)"

  while true; do
    timeout_count="$(get_timeout_count)"
    if (( timeout_count > last_timeout_count )); then
      log "detected indexer timeout in log (count ${timeout_count})"
      stop_nightly
      sleep "$restart_delay"
      start_nightly
      last_timeout_count="$timeout_count"
      write_state "$last_timeout_count"
      sleep "$delay"
      continue
    fi

    if [[ -f "$PID_FILE" ]]; then
      local pid
      pid="$(cat "$PID_FILE" 2>/dev/null || true)"
      if [[ -z "$pid" || ! -d "/proc/$pid" ]]; then
        log "nightly_run not running; restarting"
        start_nightly
      fi
    else
      log "nightly_run pid file missing; starting"
      start_nightly
    fi

    write_state "$last_timeout_count"
    sleep "$delay"
  done
}

log "nightly_supervisor started"
main_loop
