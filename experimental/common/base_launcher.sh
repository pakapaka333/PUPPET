#!/bin/bash
set -e

REAL_CONFIG_PATH=$1
CUDA_VISIBLE_DEVICES=$2
RUN_DIR=$3
RUN_TYPE=$4
RUN_NAME=${5:-""}

# Setup
export ROOT_DIR=$(grep -oP 'root_dir:\s*\K.*' "$REAL_CONFIG_PATH" | sed 's/^"//; s/"$//')
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/submodules/MarkLLM:${ROOT_DIR}/submodules/ai-detection-paraphrases:${ROOT_DIR}/submodules/fast-detect-gpt/scripts:${PYTHONPATH}"
set -a; source "${ROOT_DIR}/.env"; set +a;
if [ -z "$RUN_NAME" ]; then
     timestamp=$(date +"%Y_%m%d_%H%M%S")
     export RUN_NAME="$(grep -oP 'run_name:\s*\K.*' "$REAL_CONFIG_PATH" | sed 's/^"//; s/"$//')_${timestamp}" 
else
     export RUN_NAME="${RUN_NAME}"
fi
exec > >(tee -a ${ROOT_DIR}/${RUN_DIR}/logs/${RUN_NAME}.o) 2> >(tee -a ${ROOT_DIR}/${RUN_DIR}/logs/${RUN_NAME}.e >&2)


# Run
echo "🚀 Starting ${RUN_TYPE}: ${RUN_NAME}"
cd ${ROOT_DIR}
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES &&
     uv run --isolated --locked --project ${RUN_DIR}/pyproject.toml -- python ${RUN_DIR}/${RUN_TYPE}.py --config "${REAL_CONFIG_PATH}"
echo "✅ ${RUN_TYPE} finished successfully: ${RUN_NAME}"