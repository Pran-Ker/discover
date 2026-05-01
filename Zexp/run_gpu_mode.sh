#!/usr/bin/env bash
# Launch a GPU-mode (CUDA kernel) discovery run.
# Usage:
#   bash Zexp/run_gpu_mode.sh
#   bash Zexp/run_gpu_mode.sh --problem_type mla_decode_nvidia --num_epochs 10
#
# Required env vars: TINKER_API_KEY, HF_TOKEN, WANDB_API_KEY

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python3"

source ~/.local/secrets
export HF_TOKEN="${HUGGING_FACE_TOKEN}"

export WANDB_ENTITY="hebbarpran-warping"
export WANDB_PROJECT="gpu-mode"
export HF_REPO="Pran-Ker/discover-gpu-mode"

: "${TINKER_API_KEY:?Need TINKER_API_KEY}"
: "${HF_TOKEN:?Need HF_TOKEN}"
: "${WANDB_API_KEY:?Need WANDB_API_KEY}"

PROBLEM_TYPE="trimul"
GROUP_SIZE=16
GROUPS_PER_BATCH=2
NUM_EPOCHS=50
EXPERIMENT_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --problem_type)     PROBLEM_TYPE="$2"; shift 2 ;;
    --group_size)       GROUP_SIZE="$2"; shift 2 ;;
    --groups_per_batch) GROUPS_PER_BATCH="$2"; shift 2 ;;
    --num_epochs)       NUM_EPOCHS="$2"; shift 2 ;;
    --experiment_name)  EXPERIMENT_NAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd "$REPO_ROOT"

EXTRA_ARGS=""
if [[ -n "$EXPERIMENT_NAME" ]]; then
  EXTRA_ARGS="--experiment_name $EXPERIMENT_NAME"
fi

"$PYTHON" -m examples.gpu_mode.env \
  --problem_type "$PROBLEM_TYPE" \
  --group_size "$GROUP_SIZE" \
  --groups_per_batch "$GROUPS_PER_BATCH" \
  --num_epochs "$NUM_EPOCHS" \
  --wandb_project "$WANDB_PROJECT" \
  $EXTRA_ARGS
