#!/usr/bin/env bash
set -euo pipefail

# CPU-only canonical formatting node helper.
# Use this on a high-core CPU machine to stream/parse datasets into canonical
# JSONL shards, then move/upload closed shards to a GPU labeling node.

usage() {
  cat <<'USAGE'
Usage:
  scripts/format_cpu_node.sh start
  scripts/format_cpu_node.sh bootstrap
  scripts/format_cpu_node.sh start-hermes-streams
  scripts/format_cpu_node.sh stream-source SOURCE DIR PREFIX
  scripts/format_cpu_node.sh upload-complete
  scripts/format_cpu_node.sh status
  scripts/format_cpu_node.sh stop

Environment overrides:
  REPO_DIR                         Repo checkout path. Default: script parent.
  CONSTELLATION_RUNS_DIR           Artifact root. Default: ~/constellation-runs
  UV_BIN                           uv binary. Default: uv
  STREAM_SHARD_SIZE                Canonical shard size. Default: 50000
  STREAM_MAX_ROWS                  Max rows per source; 0 means full stream. Default: 0
  FORCE_RESTREAM_SOURCE            Delete existing source .tmp dirs. Default: 0
  HF_UPLOAD_REPO                   Optional HF dataset repo for canonical uploads.
                                   Example: driaforall/constellation-agenttrove-labeled
  HF_UPLOAD_PREFIX                 Remote folder for canonical uploads.
                                   Default: canonical
  CPU_CORES                        Informational/default thread count. Auto-detected.

Common first run on a fresh CPU node:
  git clone https://github.com/firstbatchxyz/constellation.git /home/ubuntu/constellation
  cd /home/ubuntu/constellation
  scripts/format_cpu_node.sh start

Watch:
  scripts/format_cpu_node.sh status
  journalctl --user -u constellation-format-hermes_kimi -f
  journalctl --user -u constellation-format-hermes_glm -f

Upload complete canonical shards:
  HF_UPLOAD_REPO=driaforall/constellation-agenttrove-labeled \
    scripts/format_cpu_node.sh upload-complete
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_repo_dir="$(cd "$script_dir/.." && pwd)"

REPO_DIR="${REPO_DIR:-$default_repo_dir}"
RUNS="${CONSTELLATION_RUNS_DIR:-$HOME/constellation-runs}"
UV_BIN="${UV_BIN:-uv}"
STREAM_SHARD_SIZE="${STREAM_SHARD_SIZE:-50000}"
STREAM_MAX_ROWS="${STREAM_MAX_ROWS:-0}"
FORCE_RESTREAM_SOURCE="${FORCE_RESTREAM_SOURCE:-0}"
HF_UPLOAD_REPO="${HF_UPLOAD_REPO:-}"
HF_UPLOAD_PREFIX="${HF_UPLOAD_PREFIX:-canonical}"
CPU_CORES="${CPU_CORES:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 64)}"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export CONSTELLATION_RUNS_DIR="$RUNS"
export HF_HOME="${HF_HOME:-$RUNS/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$RUNS/hf-datasets-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_CORES}"

install_uv() {
  if command -v "$UV_BIN" >/dev/null 2>&1; then
    return
  fi

  if [[ "$UV_BIN" != "uv" ]]; then
    echo "UV_BIN=$UV_BIN is not installed; install uv or set UV_BIN=uv" >&2
    exit 1
  fi

  echo "uv not found; installing with astral installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
  command -v uv >/dev/null 2>&1 || {
    echo "uv install completed but uv is still not on PATH" >&2
    exit 1
  }
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

bootstrap() {
  install_uv
  cd "$REPO_DIR"
  mkdir -p "$RUNS/final/canonical" "$HF_HOME" "$HF_DATASETS_CACHE"
  "$UV_BIN" sync
  "$UV_BIN" pip install --upgrade datasets huggingface_hub
}

stream_source() {
  local source="$1"
  local dir="$2"
  local prefix="$3"
  local tmp_dir="$RUNS/final/canonical/$dir.tmp"
  local final_dir="$RUNS/final/canonical/$dir"

  install_uv
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
  local unit="constellation-format-$dir"

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
    --setenv=HF_HOME="$HF_HOME" \
    --setenv=HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    --setenv=OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    "$REPO_DIR/scripts/format_cpu_node.sh" stream-source "$source" "$dir" "$prefix"
}

start_hermes_streams() {
  start_stream_source hermes-kimi hermes_kimi hermes_kimi
  start_stream_source hermes-glm hermes_glm hermes_glm
}

start_all() {
  bootstrap
  start_hermes_streams
  status
}

complete_shards() {
  local src_dir="$1"
  local expected="$STREAM_SHARD_SIZE"
  local shard

  shopt -s nullglob
  for shard in "$src_dir"/*.jsonl; do
    if [[ "$(wc -l < "$shard")" -eq "$expected" ]]; then
      printf '%s\n' "$shard"
    fi
  done
}

upload_complete() {
  if [[ -z "$HF_UPLOAD_REPO" ]]; then
    echo "set HF_UPLOAD_REPO to upload complete canonical shards" >&2
    exit 2
  fi

  install_uv
  hf auth whoami || hf auth login

  local upload_root="/tmp/constellation-canonical-upload-$$"
  local source_dir
  local source_name
  local shard
  mkdir -p "$upload_root"

  shopt -s nullglob
  for source_dir in "$RUNS"/final/canonical/*.tmp "$RUNS"/final/canonical/*; do
    [[ -d "$source_dir" ]] || continue
    source_name="$(basename "$source_dir")"
    source_name="${source_name%.tmp}"
    mkdir -p "$upload_root/$HF_UPLOAD_PREFIX/$source_name"
    while IFS= read -r shard; do
      ln -f "$shard" "$upload_root/$HF_UPLOAD_PREFIX/$source_name/$(basename "$shard")"
    done < <(complete_shards "$source_dir")
  done

  echo "Prepared complete canonical shards:"
  find "$upload_root" -type f -name '*.jsonl' -printf '%p\n' | sort

  if [[ -z "$(find "$upload_root" -type f -name '*.jsonl' -print -quit)" ]]; then
    echo "no complete canonical shards to upload"
    return
  fi

  hf upload "$HF_UPLOAD_REPO" "$upload_root" . \
    --repo-type dataset \
    --commit-message "Upload complete canonical Hermes shards"
}

status() {
  echo "== services =="
  systemctl --user --no-pager --plain status constellation-format-hermes_kimi 2>/dev/null || true
  systemctl --user --no-pager --plain status constellation-format-hermes_glm 2>/dev/null || true

  echo
  echo "== processes =="
  pgrep -af 'stream-convert|format_cpu_node' || true

  echo
  echo "== canonical shards =="
  wc -l "$RUNS"/final/canonical/*.tmp/*.jsonl "$RUNS"/final/canonical/*/*.jsonl 2>/dev/null || true

  echo
  echo "== recent canonical files =="
  find "$RUNS/final/canonical" -maxdepth 2 -type f -name '*.jsonl' \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -20 || true

  echo
  echo "== logs =="
  echo "journalctl --user -u constellation-format-hermes_kimi -f"
  echo "journalctl --user -u constellation-format-hermes_glm -f"
}

stop_all() {
  systemctl --user stop constellation-format-hermes_kimi 2>/dev/null || true
  systemctl --user stop constellation-format-hermes_glm 2>/dev/null || true
  pgrep -af 'stream-convert|format_cpu_node' || true
}

command="${1:-start}"
case "$command" in
  start)
    start_all
    ;;
  bootstrap)
    bootstrap
    ;;
  start-hermes-streams)
    start_hermes_streams
    ;;
  stream-source)
    if [[ $# -ne 4 ]]; then
      echo "usage: $0 stream-source SOURCE DIR PREFIX" >&2
      exit 2
    fi
    stream_source "$2" "$3" "$4"
    ;;
  upload-complete)
    upload_complete
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
