#!/usr/bin/env bash
# Set up all external dependencies for the experimental code.
#
# Usage (run from the repository root):
#   bash experimental/setup/setup.sh
#
# What this script does:
#   [1/2] scripts/setup_submodules.sh — Add WaterBench, dipper, and MarkLLM
#                                       as submodule/subtrees and apply patches
#   [2/2] scripts/setup_eval.py       — Download evaluation materials
#                                       (IELTS prompt, MAGE utils, fakespot-ai utils)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SETUP_DIR="$REPO_ROOT/experimental/setup"

cd "$REPO_ROOT"

echo "==> [1/2] Setting up submodules..."
bash "$SETUP_DIR/scripts/setup_submodules.sh"

echo "==> [2/2] Downloading evaluation materials..."
python "$SETUP_DIR/scripts/setup_eval.py"

echo "==> Done."
