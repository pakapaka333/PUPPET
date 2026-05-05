#!/bin/bash
set -e

CONFIG_PATH=$(realpath $1)
ROOT_DIR=$(grep -oP 'root_dir:\s*\K.*' "$CONFIG_PATH" | sed 's/^"//; s/"$//')
CUDA_VISIBLE_DEVICES=$2
RUN_DIR="watermark_det"

export SERVER_NAME=$(hostname)

bash ${ROOT_DIR}/common/base_launcher.sh "${CONFIG_PATH}" "${CUDA_VISIBLE_DEVICES}" "${RUN_DIR}" "detection"