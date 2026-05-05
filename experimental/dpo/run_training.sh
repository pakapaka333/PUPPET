#!/bin/bash
set -e

CONFIG_PATH=$(realpath $1)
ROOT_DIR=$(grep -oP 'root_dir:\s*\K.*' "$CONFIG_PATH" | sed 's/^"//; s/"$//')
CUDA_VISIBLE_DEVICES=$2
RUN_DIR="dpo"

timestamp=$(date +"%Y_%m%d_%H%M%S")
export WANDB_RUN_NAME="$(grep -oP 'run_name:\s*\K.*' "$CONFIG_PATH" | sed 's/^"//; s/"$//')_${timestamp}"
export WANDB_PROJECT="$(grep -oP 'project_name:\s*\K.*' "$CONFIG_PATH" | sed 's/^"//; s/"$//')"

bash ${ROOT_DIR}/common/base_launcher.sh "${CONFIG_PATH}" "${CUDA_VISIBLE_DEVICES}" "${RUN_DIR}" "training" "${WANDB_RUN_NAME}"