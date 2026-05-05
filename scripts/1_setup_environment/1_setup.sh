#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; source "${SCRIPTS_DIR}/.env"; set +a;

for dir in 2_build_dataset 3_train_model 4_evaluate_trained_model; do
    echo "==> Installing packages for $dir..."
    cd "$SCRIPTS_DIR/$dir"
    uv sync
done

echo "==> Done."
