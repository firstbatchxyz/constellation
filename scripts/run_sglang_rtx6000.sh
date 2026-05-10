#!/usr/bin/env bash
set -euo pipefail

# RTX 6000 / Blackwell-friendly SGLang launcher.
# Defaults mirror the Docker/CUDA13 path that avoids local CUDA/JIT stack drift.

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-0.8B}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:dev-cu13}"
CONTAINER_NAME="${CONTAINER_NAME:-constellation-sglang}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SHM_SIZE="${SHM_SIZE:-32g}"

DETACH=0
DRY_RUN=0
PULL=0
RM=1
STOP_EXISTING=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Run SGLang in the CUDA13 Docker image tuned for RTX 6000 / Blackwell machines.

Usage:
  scripts/run_sglang_rtx6000.sh [options] [-- extra sglang args...]

Options:
  --model MODEL             HF repo or local model path.
                            Default: Qwen/Qwen3.5-0.8B
  --image IMAGE             Docker image.
                            Default: lmsysorg/sglang:dev-cu13
  --name NAME               Container name.
                            Default: constellation-sglang
  --host HOST               SGLang bind host.
                            Default: 127.0.0.1
  --port PORT               SGLang port.
                            Default: 30000
  --hf-cache PATH           Hugging Face cache mount.
                            Default: ~/.cache/huggingface
  --shm-size SIZE           Docker shared memory size.
                            Default: 32g
  --detach                  Run container in background.
  --pull                    Pull image before launching.
  --stop-existing           Stop/remove an existing container with --name first.
  --no-rm                   Do not pass docker --rm.
  --dry-run                 Print the docker command without running it.
  -h, --help                Show this help.

Environment:
  HF_TOKEN                  Forwarded into the container if set.
  HUGGING_FACE_HUB_TOKEN    Forwarded if set.
  CUDA_VISIBLE_DEVICES      Forwarded if set.
  SGLANG_IMAGE              Alternate default image.
  MODEL_PATH                Alternate default model.

Examples:
  scripts/run_sglang_rtx6000.sh

  scripts/run_sglang_rtx6000.sh --detach --pull --stop-existing

  scripts/run_sglang_rtx6000.sh --model Qwen/Qwen3.5-0.8B -- \
    --attention-backend triton \
    --sampling-backend pytorch
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --hf-cache)
      HF_CACHE="$2"
      shift 2
      ;;
    --shm-size)
      SHM_SIZE="$2"
      shift 2
      ;;
    --detach)
      DETACH=1
      shift
      ;;
    --pull)
      PULL=1
      shift
      ;;
    --stop-existing)
      STOP_EXISTING=1
      shift
      ;;
    --no-rm)
      RM=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but was not found on PATH" >&2
  exit 1
fi

mkdir -p "$HF_CACHE"

if [[ "$PULL" -eq 1 ]]; then
  docker pull "$IMAGE"
fi

if [[ "$STOP_EXISTING" -eq 1 ]]; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

docker_cmd=(docker run)

if [[ "$RM" -eq 1 ]]; then
  docker_cmd+=(--rm)
fi

if [[ "$DETACH" -eq 1 ]]; then
  docker_cmd+=(-d)
elif [[ -t 0 && -t 1 ]]; then
  docker_cmd+=(-it)
fi

docker_cmd+=(
  --name "$CONTAINER_NAME"
  --gpus all
  --network host
  --ipc=host
  --shm-size "$SHM_SIZE"
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "$HF_CACHE:/root/.cache/huggingface"
  -e "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_cmd+=(-e "HF_TOKEN=$HF_TOKEN")
fi

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  docker_cmd+=(-e "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN")
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  docker_cmd+=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

docker_cmd+=(
  "$IMAGE"
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
)

docker_cmd+=("${EXTRA_ARGS[@]}")

printf 'Launching SGLang:\n'
printf '  model: %s\n' "$MODEL_PATH"
printf '  image: %s\n' "$IMAGE"
printf '  host:  %s\n' "$HOST"
printf '  port:  %s\n' "$PORT"
printf '  name:  %s\n\n' "$CONTAINER_NAME"

printf 'Command:\n'
printf '  %q' "${docker_cmd[@]}"
printf '\n\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

exec "${docker_cmd[@]}"
