#!/usr/bin/env bash
# Set up external repositories (submodule and subtrees).
#
# Usage (run from the repository root):
#   bash experimental/setup/setup.sh
#
# Or directly (run from the repository root):
#   bash experimental/setup/scripts/setup_submodules.sh
#
# What this script does:
#   1. Add WaterBench as a submodule
#   2. Add ai-detection-paraphrases (dipper) as a subtree
#   3. Add MarkLLM as a subtree
#   4. Apply dipper.patch  (DipperParaphraser accepts model_kwargs)
#   5. Apply markllm.patch (avoid OOM; support chat-template)
#   6. Remove large files from MarkLLM (dataset/c4-train, watermark/xsir/dictionary/dictionary.txt)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SETUP_DIR="$REPO_ROOT/experimental/setup"

cd "$REPO_ROOT"

# ── 1. WaterBench (submodule) ─────────────────────────────────────────────────
echo "==> [1/6] Adding WaterBench as submodule..."
git submodule add https://github.com/THU-KEG/WaterBench.git experimental/submodules/WaterBench
git -C experimental/submodules/WaterBench checkout 8f3d779d66518a7b90ce1aad1fabaeb13cfca548
git add .gitmodules experimental/submodules/WaterBench
git commit -m "chore: add WaterBench as submodule"

# ── 2. ai-detection-paraphrases / dipper (subtree) ───────────────────────────
echo "==> [2/6] Adding ai-detection-paraphrases as subtree..."
git subtree add \
    --prefix experimental/submodules/ai-detection-paraphrases \
    https://github.com/martiansideofthemoon/ai-detection-paraphrases.git \
    95f3e2cb5e239929a1fc4bed26bf93f2c368da31 \
    --squash

# ── 3. MarkLLM (subtree) ──────────────────────────────────────────────────────
echo "==> [3/6] Adding MarkLLM as subtree..."
git subtree add \
    --prefix experimental/submodules/MarkLLM \
    https://github.com/THU-BPM/MarkLLM.git \
    c2b773d11dfc82ef6ec6704cfe17ebf01f63ef2b \
    --squash

# ── 4. Apply dipper.patch ─────────────────────────────────────────────────────
echo "==> [4/6] Applying dipper.patch..."
git apply --index --directory=experimental/submodules/ai-detection-paraphrases \
    "$SETUP_DIR/dipper.patch"
git commit -m "Apply dipper.patch: DipperParaphraser accepts model_kwargs"

# ── 5. Apply markllm.patch ────────────────────────────────────────────────────
echo "==> [5/6] Applying markllm.patch..."
git apply --index --directory=experimental/submodules/MarkLLM \
    "$SETUP_DIR/markllm.patch"
git commit -m "Apply markllm.patch: avoid OOM, support chat-template"

# ── 6. Remove large files from MarkLLM ───────────────────────────────────────
echo "==> [6/6] Removing large files from MarkLLM..."
rm -rf \
    experimental/submodules/MarkLLM/dataset/c4-train \
    experimental/submodules/MarkLLM/watermark/xsir/dictionary/dictionary.txt
git add experimental/submodules/MarkLLM
git commit -m "chore: remove large files from MarkLLM (dataset/c4-train, watermark/xsir/dictionary/dictionary.txt)"

echo "==> Done."
