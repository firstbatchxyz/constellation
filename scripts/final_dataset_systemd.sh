#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/final_dataset_systemd.sh setup
  scripts/final_dataset_systemd.sh run-sglang
  scripts/final_dataset_systemd.sh run-final-dataset

Environment overrides:
  REPO_DIR                         Repo checkout path.
  CONSTELLATION_RUNS_DIR           Artifact root. Default: ~/constellation-runs
  SGLANG_MODEL                     Label model. Default: Qwen/Qwen3.5-0.8B
  SGLANG_PORT                      SGLang port. Default: 30000
  SGLANG_MEM_FRACTION_STATIC       SGLang static memory fraction. Default: 0.75
  LABEL_CONCURRENCY                llm-label HTTP concurrency. Default: 16
  STREAM_SHARD_SIZE                Canonical shard size. Default: 50000
  STREAM_MAX_ROWS                  Max rows per source; 0 means full stream. Default: 0
  UV_BIN                           uv binary. Default: uv
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
UV_BIN="${UV_BIN:-uv}"
SERVICE_DIR="$HOME/.config/systemd/user"
SCRIPT_PATH="$REPO_DIR/scripts/final_dataset_systemd.sh"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export CONSTELLATION_RUNS_DIR="$RUNS"

run_sglang() {
  cd "$REPO_DIR"
  export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-1}"
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-1}"

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

  if [[ -f "$done" ]]; then
    echo "canonical $source already done"
    return
  fi

  rm -rf "$RUNS/final/canonical/$dir.tmp"
  mkdir -p "$RUNS/final/canonical/$dir.tmp"

  "$UV_BIN" run python -m constellation.cli stream-convert \
    --source "$source" \
    --max-rows "$STREAM_MAX_ROWS" \
    --output-dir "$RUNS/final/canonical/$dir.tmp" \
    --shard-prefix "$prefix" \
    --shard-size "$STREAM_SHARD_SIZE" \
    --skip-errors \
    --no-hard-exit

  rm -rf "$RUNS/final/canonical/$dir"
  mv "$RUNS/final/canonical/$dir.tmp" "$RUNS/final/canonical/$dir"
  touch "$done"
}

label_shards() {
  find "$RUNS/final/canonical" -name '*.jsonl' | sort | while read -r shard; do
    local rel="${shard#$RUNS/final/canonical/}"
    local out="$RUNS/final/labeled/$rel"
    local done="$out.done"
    local tmp="$out.tmp"

    if [[ -f "$done" ]]; then
      echo "labeled $rel already done"
      continue
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
  done
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

  stream_source agenttrove agenttrove agenttrove
  stream_source hermes-kimi hermes_kimi hermes_kimi
  stream_source hermes-glm hermes_glm hermes_glm

  label_shards
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
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
