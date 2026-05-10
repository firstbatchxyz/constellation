#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/resume_labeling_node.sh start
  scripts/resume_labeling_node.sh bootstrap-deps
  scripts/resume_labeling_node.sh seed-agenttrove-labels
  scripts/resume_labeling_node.sh start-sglang
  scripts/resume_labeling_node.sh start-hermes-streams
  scripts/resume_labeling_node.sh start-label-loop
  scripts/resume_labeling_node.sh status
  scripts/resume_labeling_node.sh stop

Gold-path setup for resuming Constellation labeling on a fresh GPU node:
  1. install runtime dependencies,
  2. download already-uploaded labeled AgentTrove shards from Hugging Face,
  3. start a local SGLang Qwen/Qwen3.5-0.8B server,
  4. stream Hermes Kimi/GLM into canonical shards,
  5. continuously label closed canonical shards.

Environment overrides:
  REPO_DIR                         Repo checkout path. Default: script parent.
  CONSTELLATION_RUNS_DIR           Artifact root. Default: ~/constellation-runs
  HF_REPO                          HF dataset repo for uploaded AgentTrove labels.
                                   Default: driaforall/constellation-agenttrove-labeled
  UV_BIN                           uv binary. Default: uv
  SGLANG_MODEL                     Label model. Default: Qwen/Qwen3.5-0.8B
  SGLANG_PORT                      SGLang port. Default: 30000
  SGLANG_MEM_FRACTION_STATIC       Default: 0.70
  SGLANG_LAUNCHER                  docker-rtx6000, host, or auto. Default: auto
  SGLANG_ENABLE_JIT_DEEPGEMM       Default: 0 for RTX6000 portability.
  SGLANG_JIT_DEEPGEMM_PRECOMPILE   Default: 0 for RTX6000 portability.
  ENSURE_CUDNN_VERSION             Default: 9.16.0.29. Set empty to skip.
  LABEL_CONCURRENCY                llm-label HTTP concurrency. Default: 64
  LABEL_LOOP_SLEEP                 Seconds between label passes. Default: 120
  STREAM_SHARD_SIZE                Canonical shard size. Default: 50000
  STREAM_MAX_ROWS                  Max rows per source; 0 means full stream. Default: 0
  FORCE_RESTREAM_SOURCE            Delete existing source .tmp dirs. Default: 0

Common first run:
  git clone https://github.com/firstbatchxyz/constellation.git /home/ubuntu/constellation
  cd /home/ubuntu/constellation
  scripts/resume_labeling_node.sh start

Watch:
  journalctl --user -u constellation-resume-sglang -f
  journalctl --user -u constellation-label-loop -f
  scripts/resume_labeling_node.sh status
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_repo_dir="$(cd "$script_dir/.." && pwd)"

REPO_DIR="${REPO_DIR:-$default_repo_dir}"
RUNS="${CONSTELLATION_RUNS_DIR:-$HOME/constellation-runs}"
HF_REPO="${HF_REPO:-driaforall/constellation-agenttrove-labeled}"
UV_BIN="${UV_BIN:-uv}"
SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen3.5-0.8B}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
SGLANG_LAUNCHER="${SGLANG_LAUNCHER:-auto}"
SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
ENSURE_CUDNN_VERSION="${ENSURE_CUDNN_VERSION:-9.16.0.29}"
LABEL_CONCURRENCY="${LABEL_CONCURRENCY:-64}"
LABEL_LOOP_SLEEP="${LABEL_LOOP_SLEEP:-120}"
STREAM_SHARD_SIZE="${STREAM_SHARD_SIZE:-50000}"
STREAM_MAX_ROWS="${STREAM_MAX_ROWS:-0}"
FORCE_RESTREAM_SOURCE="${FORCE_RESTREAM_SOURCE:-0}"
FINAL_HELPER="$REPO_DIR/scripts/final_dataset_systemd.sh"
RTX6000_HELPER="$REPO_DIR/scripts/run_sglang_rtx6000.sh"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export CONSTELLATION_RUNS_DIR="$RUNS"

require_final_helper() {
  if [[ ! -x "$FINAL_HELPER" ]]; then
    echo "missing executable final helper: $FINAL_HELPER" >&2
    exit 1
  fi
}

hf() {
  if command -v uvx >/dev/null 2>&1; then
    uvx hf "$@"
  else
    "$UV_BIN" tool run hf "$@"
  fi
}

enable_linger() {
  if command -v loginctl >/dev/null 2>&1; then
    sudo loginctl enable-linger "${USER:-$(id -un)}"
  fi
}

bootstrap_deps() {
  cd "$REPO_DIR"
  "$UV_BIN" sync
  "$UV_BIN" pip install -r requirements/train.txt
  "$UV_BIN" pip install "sglang" --prerelease=allow
  if [[ -n "$ENSURE_CUDNN_VERSION" ]]; then
    "$UV_BIN" pip install --upgrade "nvidia-cudnn-cu12==$ENSURE_CUDNN_VERSION"
  fi
}

seed_agenttrove_labels() {
  cd "$REPO_DIR"
  mkdir -p "$RUNS/final/labeled"

  hf auth whoami || hf auth login
  hf download "$HF_REPO" \
    --repo-type dataset \
    --include "agenttrove/agenttrove-*.jsonl" \
    --local-dir "$RUNS/final/labeled"

  shopt -s nullglob
  for shard in "$RUNS"/final/labeled/agenttrove/agenttrove-*.jsonl; do
    touch "$shard.done"
  done

  echo "Seeded AgentTrove labels:"
  wc -l "$RUNS"/final/labeled/agenttrove/agenttrove-*.jsonl 2>/dev/null || true
}

run_sglang() {
  require_final_helper
  local launcher="$SGLANG_LAUNCHER"
  if [[ "$launcher" == "auto" ]]; then
    if command -v docker >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qiE 'RTX.*6000|Blackwell'; then
      launcher="docker-rtx6000"
    else
      launcher="host"
    fi
  fi

  if [[ "$launcher" == "docker-rtx6000" ]]; then
    if [[ ! -x "$RTX6000_HELPER" ]]; then
      echo "missing executable RTX6000 helper: $RTX6000_HELPER" >&2
      exit 1
    fi
    exec "$RTX6000_HELPER" \
      --model "$SGLANG_MODEL" \
      --host 127.0.0.1 \
      --port "$SGLANG_PORT" \
      --detach \
      --stop-existing
  fi

  export SGLANG_MODEL
  export SGLANG_PORT
  export SGLANG_MEM_FRACTION_STATIC
  export SGLANG_ENABLE_JIT_DEEPGEMM
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE
  export ENSURE_CUDNN_VERSION
  exec "$FINAL_HELPER" run-sglang
}

start_sglang() {
  enable_linger
  if systemctl --user is-active --quiet constellation-resume-sglang 2>/dev/null; then
    echo "constellation-resume-sglang already active"
    return
  fi
  systemctl --user reset-failed constellation-resume-sglang 2>/dev/null || true
  systemd-run --user \
    --unit constellation-resume-sglang \
    --property=WorkingDirectory="$REPO_DIR" \
    --setenv=REPO_DIR="$REPO_DIR" \
    --setenv=CONSTELLATION_RUNS_DIR="$RUNS" \
    --setenv=SGLANG_MODEL="$SGLANG_MODEL" \
    --setenv=SGLANG_PORT="$SGLANG_PORT" \
    --setenv=SGLANG_MEM_FRACTION_STATIC="$SGLANG_MEM_FRACTION_STATIC" \
    --setenv=SGLANG_LAUNCHER="$SGLANG_LAUNCHER" \
    --setenv=SGLANG_ENABLE_JIT_DEEPGEMM="$SGLANG_ENABLE_JIT_DEEPGEMM" \
    --setenv=SGLANG_JIT_DEEPGEMM_PRECOMPILE="$SGLANG_JIT_DEEPGEMM_PRECOMPILE" \
    --setenv=ENSURE_CUDNN_VERSION="$ENSURE_CUDNN_VERSION" \
    "$REPO_DIR/scripts/resume_labeling_node.sh" run-sglang
}

wait_for_sglang() {
  echo "Waiting for SGLang on port $SGLANG_PORT..."
  until curl -fsS "http://127.0.0.1:$SGLANG_PORT/v1/models" >/dev/null; do
    sleep 5
  done
  echo "SGLang ready on port $SGLANG_PORT"
}

stream_source() {
  local source="$1"
  local dir="$2"
  local prefix="$3"
  local tmp_dir="$RUNS/final/canonical/$dir.tmp"
  local final_dir="$RUNS/final/canonical/$dir"

  cd "$REPO_DIR"
  mkdir -p "$RUNS/final/canonical"

  if [[ -d "$final_dir" && -f "$final_dir/.stream_done" ]]; then
    echo "canonical $source already done at $final_dir"
    return
  fi

  if [[ -d "$tmp_dir" && "$FORCE_RESTREAM_SOURCE" != "1" ]]; then
    echo "found existing $tmp_dir; preserving partial stream" >&2
    echo "set FORCE_RESTREAM_SOURCE=1 to delete it and restart $source" >&2
    exit 1
  fi

  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"

  "$UV_BIN" run python -m constellation.cli stream-convert \
    --source "$source" \
    --max-rows "$STREAM_MAX_ROWS" \
    --output-dir "$tmp_dir" \
    --shard-prefix "$prefix" \
    --shard-size "$STREAM_SHARD_SIZE" \
    --skip-errors \
    --no-hard-exit

  touch "$tmp_dir/.stream_done"
}

start_stream_source() {
  local source="$1"
  local dir="$2"
  local prefix="$3"
  local unit="constellation-stream-$dir"

  enable_linger
  if systemctl --user is-active --quiet "$unit" 2>/dev/null; then
    echo "$unit already active"
    return
  fi
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  systemd-run --user \
    --unit "$unit" \
    --property=WorkingDirectory="$REPO_DIR" \
    --setenv=REPO_DIR="$REPO_DIR" \
    --setenv=CONSTELLATION_RUNS_DIR="$RUNS" \
    --setenv=STREAM_MAX_ROWS="$STREAM_MAX_ROWS" \
    --setenv=STREAM_SHARD_SIZE="$STREAM_SHARD_SIZE" \
    --setenv=FORCE_RESTREAM_SOURCE="$FORCE_RESTREAM_SOURCE" \
    "$REPO_DIR/scripts/resume_labeling_node.sh" stream-source "$source" "$dir" "$prefix"
}

start_hermes_streams() {
  start_stream_source hermes-kimi hermes_kimi hermes_kimi
  start_stream_source hermes-glm hermes_glm hermes_glm
}

label_loop() {
  require_final_helper
  export SGLANG_PORT
  export LABEL_CONCURRENCY

  while true; do
    "$FINAL_HELPER" label-available-shards
    sleep "$LABEL_LOOP_SLEEP"
  done
}

start_label_loop() {
  enable_linger
  if systemctl --user is-active --quiet constellation-label-loop 2>/dev/null; then
    echo "constellation-label-loop already active"
    return
  fi
  systemctl --user reset-failed constellation-label-loop 2>/dev/null || true
  systemd-run --user \
    --unit constellation-label-loop \
    --property=WorkingDirectory="$REPO_DIR" \
    --setenv=REPO_DIR="$REPO_DIR" \
    --setenv=CONSTELLATION_RUNS_DIR="$RUNS" \
    --setenv=SGLANG_PORT="$SGLANG_PORT" \
    --setenv=LABEL_CONCURRENCY="$LABEL_CONCURRENCY" \
    --setenv=LABEL_LOOP_SLEEP="$LABEL_LOOP_SLEEP" \
    "$REPO_DIR/scripts/resume_labeling_node.sh" label-loop
}

start_all() {
  bootstrap_deps
  seed_agenttrove_labels
  start_sglang
  wait_for_sglang
  start_hermes_streams
  start_label_loop
  status
}

status() {
  echo "== services =="
  systemctl --user --no-pager --plain status constellation-resume-sglang 2>/dev/null || true
  systemctl --user --no-pager --plain status constellation-stream-hermes_kimi 2>/dev/null || true
  systemctl --user --no-pager --plain status constellation-stream-hermes_glm 2>/dev/null || true
  systemctl --user --no-pager --plain status constellation-label-loop 2>/dev/null || true

  echo
  echo "== processes =="
  pgrep -af 'sglang|stream-convert|llm-label|label-available-shards' || true
  docker ps --filter name=constellation-sglang --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true

  echo
  echo "== canonical shards =="
  wc -l "$RUNS"/final/canonical/*.tmp/*.jsonl "$RUNS"/final/canonical/*/*.jsonl 2>/dev/null || true

  echo
  echo "== labeled shards =="
  wc -l "$RUNS"/final/labeled/*/*.jsonl 2>/dev/null || true
  find "$RUNS/final/labeled" -name '*.tmp' -printf '%s %p\n' 2>/dev/null || true
  find "$RUNS/final/labeled" -name '*.done' 2>/dev/null | wc -l

  echo
  echo "== logs =="
  echo "journalctl --user -u constellation-resume-sglang -f"
  echo "journalctl --user -u constellation-stream-hermes_kimi -f"
  echo "journalctl --user -u constellation-stream-hermes_glm -f"
  echo "journalctl --user -u constellation-label-loop -f"
}

stop_all() {
  systemctl --user stop constellation-label-loop 2>/dev/null || true
  systemctl --user stop constellation-stream-hermes_kimi 2>/dev/null || true
  systemctl --user stop constellation-stream-hermes_glm 2>/dev/null || true
  systemctl --user stop constellation-resume-sglang 2>/dev/null || true
  docker rm -f constellation-sglang 2>/dev/null || true
  pgrep -af 'sglang|stream-convert|llm-label|label-available-shards' || true
}

command="${1:-start}"
case "$command" in
  start)
    start_all
    ;;
  bootstrap-deps)
    bootstrap_deps
    ;;
  seed-agenttrove-labels)
    seed_agenttrove_labels
    ;;
  run-sglang)
    run_sglang
    ;;
  start-sglang)
    start_sglang
    ;;
  wait-for-sglang)
    wait_for_sglang
    ;;
  stream-source)
    if [[ $# -ne 4 ]]; then
      echo "usage: $0 stream-source SOURCE DIR PREFIX" >&2
      exit 2
    fi
    stream_source "$2" "$3" "$4"
    ;;
  start-hermes-streams)
    start_hermes_streams
    ;;
  label-loop)
    label_loop
    ;;
  start-label-loop)
    start_label_loop
    ;;
  status)
    status
    ;;
  stop)
    stop_all
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
