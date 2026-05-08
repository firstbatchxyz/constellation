#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/final_dataset_systemd.sh setup
  scripts/final_dataset_systemd.sh run-sglang
  scripts/final_dataset_systemd.sh run-final-dataset
  scripts/final_dataset_systemd.sh label-available-shards

Environment overrides:
  REPO_DIR                         Repo checkout path.
  CONSTELLATION_RUNS_DIR           Artifact root. Default: ~/constellation-runs
  SGLANG_MODEL                     Label model. Default: Qwen/Qwen3.5-0.8B
  SGLANG_PORT                      SGLang port. Default: 30000
  SGLANG_MEM_FRACTION_STATIC       SGLang static memory fraction. Default: 0.75
  LABEL_CONCURRENCY                llm-label HTTP concurrency. Default: 16
  STREAM_SHARD_SIZE                Canonical shard size. Default: 50000
  STREAM_MAX_ROWS                  Max rows per source; 0 means full stream. Default: 0
  ENSURE_CUDNN_VERSION             Override CuDNN before SGLang. Default: 9.16.0.29
                                   Set empty to skip.
  UV_BIN                           uv binary. Default: uv
  FORCE_RESTREAM_SOURCE            Delete existing source .tmp dirs. Default: 0
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_repo_dir="$(cd "$script_dir/.." && pwd)"

REPO_DIR="${REPO_DIR:-$default_repo_dir}"
RUNS="${CONSTELLATION_RUNS_DIR:-$HOME/constellation-runs}"
SGLANG_MODEL="${SGLANG_MODEL:-Qwen/Qwen3.5-0.8B}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
LABEL_CONCURRENCY="${LABEL_CONCURRENCY:-16}"
STREAM_SHARD_SIZE="${STREAM_SHARD_SIZE:-50000}"
STREAM_MAX_ROWS="${STREAM_MAX_ROWS:-0}"
ENSURE_CUDNN_VERSION="${ENSURE_CUDNN_VERSION:-9.16.0.29}"
UV_BIN="${UV_BIN:-uv}"
FORCE_RESTREAM_SOURCE="${FORCE_RESTREAM_SOURCE:-0}"
SERVICE_DIR="$HOME/.config/systemd/user"
SCRIPT_PATH="$REPO_DIR/scripts/final_dataset_systemd.sh"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export CONSTELLATION_RUNS_DIR="$RUNS"

run_sglang() {
  cd "$REPO_DIR"
  export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-1}"
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-1}"

  if [[ -n "$ENSURE_CUDNN_VERSION" ]]; then
    "$UV_BIN" pip install --upgrade "nvidia-cudnn-cu12==$ENSURE_CUDNN_VERSION"
  fi

  exec "$UV_BIN" run python -m sglang.launch_server \
    --model-path "$SGLANG_MODEL" \
    --host 127.0.0.1 \
    --port "$SGLANG_PORT" \
    --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
}

wait_for_sglang() {
  echo "Waiting for SGLang on port $SGLANG_PORT..."
  until curl -fsS "http://127.0.0.1:$SGLANG_PORT/v1/models" >/dev/null; do
    sleep 10
  done
}

stream_source() {
  local source="$1"
  local dir="$2"
  local prefix="$3"
  local done="$RUNS/final/canonical/$dir/.done"
  local tmp_dir="$RUNS/final/canonical/$dir.tmp"

  if [[ -f "$done" ]]; then
    echo "canonical $source already done"
    return
  fi

  if [[ -d "$tmp_dir" && "$FORCE_RESTREAM_SOURCE" != "1" ]]; then
    echo "found existing $tmp_dir; preserving partial stream" >&2
    echo "set FORCE_RESTREAM_SOURCE=1 to delete it and restart $source" >&2
    return 1
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
  rm -rf "$RUNS/final/canonical/$dir"
  mv "$tmp_dir" "$RUNS/final/canonical/$dir"
  touch "$done"
}

label_one_shard() {
  local shard="$1"
  local source_dir
  local out
  local done
  local tmp

  source_dir="$(basename "$(dirname "$shard")")"
  source_dir="${source_dir%.tmp}"
  out="$RUNS/final/labeled/$source_dir/$(basename "$shard")"
  done="$out.done"
  tmp="$out.tmp"

  if [[ -f "$done" ]]; then
    echo "labeled $source_dir/$(basename "$shard") already done"
    return
  fi

  mkdir -p "$(dirname "$out")"
  rm -f "$tmp"

  "$UV_BIN" run python -m constellation.cli llm-label \
    --backend sglang \
    --api-base "http://127.0.0.1:$SGLANG_PORT/v1" \
    --concurrency "$LABEL_CONCURRENCY" \
    --input "$shard" \
    --output "$tmp"

  mv "$tmp" "$out"
  touch "$done"
}

label_available_shards() {
  local canonical_dir
  local source_dir
  local limit
  local shards

  shopt -s nullglob
  for canonical_dir in "$RUNS/final/canonical"/*; do
    [[ -d "$canonical_dir" ]] || continue
    source_dir="$(basename "$canonical_dir")"
    mapfile -t shards < <(find "$canonical_dir" -maxdepth 1 -type f -name '*.jsonl' | sort)
    (( ${#shards[@]} > 0 )) || continue

    limit="${#shards[@]}"
    if [[ "$source_dir" == *.tmp && ! -f "$canonical_dir/.stream_done" ]]; then
      # The newest shard is probably still being written by stream-convert.
      limit=$((limit - 1))
    fi

    for ((index = 0; index < limit; index++)); do
      label_one_shard "${shards[$index]}"
    done
  done
}

label_shards() {
  label_available_shards
}

run_source_with_labeling() {
  local source="$1"
  local dir="$2"
  local prefix="$3"
  local stream_pid

  if [[ -f "$RUNS/final/canonical/$dir/.done" ]]; then
    echo "canonical $source already done"
    label_available_shards
    return
  fi

  stream_source "$source" "$dir" "$prefix" &
  stream_pid="$!"
  while kill -0 "$stream_pid" >/dev/null 2>&1; do
    label_available_shards
    sleep 120
  done
  wait "$stream_pid"
  label_available_shards
}

write_reports() {
  local final="$RUNS/final/rollouts.qwen35_08_sglang_structured.final.jsonl"
  local tmp_final="$final.tmp"
  local labeled_shards=()

  mapfile -t labeled_shards < <(find "$RUNS/final/labeled" -name '*.jsonl' | sort)
  if (( ${#labeled_shards[@]} == 0 )); then
    echo "no labeled shards found under $RUNS/final/labeled" >&2
    exit 1
  fi

  cat "${labeled_shards[@]}" > "$tmp_final"
  mv "$tmp_final" "$final"

  "$UV_BIN" run python -m constellation.cli label-report \
    --input "$final" \
    --output "$RUNS/final/label-report.json" \
    --top-examples 3

  "$UV_BIN" run python -m constellation.cli target-report \
    --input "$final" \
    --top-examples 3 \
    --min-target-samples 25 \
    > "$RUNS/final/target-report.json"

  echo "DONE: $final"
}

run_final_dataset() {
  cd "$REPO_DIR"
  mkdir -p "$RUNS/final/canonical" "$RUNS/final/labeled" "$RUNS/final/logs"

  wait_for_sglang

  run_source_with_labeling agenttrove agenttrove agenttrove
  run_source_with_labeling hermes-kimi hermes_kimi hermes_kimi
  run_source_with_labeling hermes-glm hermes_glm hermes_glm

  write_reports
}

write_service_files() {
  mkdir -p "$SERVICE_DIR" "$RUNS/final" "$RUNS/logs"

  cat > "$SERVICE_DIR/constellation-sglang.service" <<SERVICE
[Unit]
Description=Constellation SGLang label server
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
Environment=PATH=$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin
Environment=REPO_DIR=$REPO_DIR
Environment=CONSTELLATION_RUNS_DIR=$RUNS
Environment=SGLANG_MODEL=$SGLANG_MODEL
Environment=SGLANG_PORT=$SGLANG_PORT
Environment=SGLANG_MEM_FRACTION_STATIC=$SGLANG_MEM_FRACTION_STATIC
Environment=ENSURE_CUDNN_VERSION=$ENSURE_CUDNN_VERSION
ExecStart=$SCRIPT_PATH run-sglang
Restart=on-failure
RestartSec=10
TimeoutStopSec=120

[Install]
WantedBy=default.target
SERVICE

  cat > "$SERVICE_DIR/constellation-final-dataset.service" <<SERVICE
[Unit]
Description=Constellation final dataset build
After=network-online.target constellation-sglang.service
Wants=constellation-sglang.service

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
Environment=PATH=$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin
Environment=REPO_DIR=$REPO_DIR
Environment=CONSTELLATION_RUNS_DIR=$RUNS
Environment=SGLANG_PORT=$SGLANG_PORT
Environment=LABEL_CONCURRENCY=$LABEL_CONCURRENCY
Environment=STREAM_SHARD_SIZE=$STREAM_SHARD_SIZE
Environment=STREAM_MAX_ROWS=$STREAM_MAX_ROWS
ExecStart=$SCRIPT_PATH run-final-dataset
Restart=no
TimeoutStopSec=120

[Install]
WantedBy=default.target
SERVICE
}

setup_services() {
  if command -v loginctl >/dev/null 2>&1; then
    sudo loginctl enable-linger "${USER:-$(id -un)}"
  fi

  write_service_files

  systemctl --user daemon-reload
  systemctl --user enable --now constellation-sglang.service
  systemctl --user restart constellation-final-dataset.service

  echo
  echo "Started services:"
  systemctl --user --no-pager --plain status constellation-sglang.service || true
  systemctl --user --no-pager --plain status constellation-final-dataset.service || true

  echo
  echo "Logs:"
  echo "  journalctl --user -u constellation-sglang -f"
  echo "  journalctl --user -u constellation-final-dataset -f"
}

command="${1:-setup}"
case "$command" in
  setup)
    setup_services
    ;;
  run-sglang)
    run_sglang
    ;;
  run-final-dataset)
    run_final_dataset
    ;;
  label-available-shards)
    wait_for_sglang
    label_available_shards
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
