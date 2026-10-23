#!/usr/bin/env bash
# Regenerate tests/fixtures/awq_reference/v1/qwen3_0_6b_awq_greedy.json.
#
# Run from WSL2 (any distro with CUDA passthrough -- generated against Ubuntu
# 24.04.5 LTS). autoawq has no Windows wheel and its `triton` dependency has
# no Windows PyPI wheel at all, so this step cannot run natively on this
# box -- see scripts/awq_reference/README.md for the full story.
#
# Usage, from a WSL2 shell with this repo's C: drive mounted under /mnt/c:
#   bash scripts/awq_reference/reproduce-reference.sh /mnt/c/Models/Qwen3-0.6B-AWQ
set -euo pipefail

MODEL_DIR="${1:?usage: reproduce-reference.sh <model_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV="$HERE/.venv"
OUT="$REPO_ROOT/tests/fixtures/awq_reference/v1/qwen3_0_6b_awq_greedy.json"

if [ ! -d "$VENV" ]; then
    python3.12 -m venv "$VENV"
fi
# Pinned, matching the fixture's own provenance block -- change BOTH together.
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$VENV/bin/python" -m pip install --quiet transformers==4.51.3 accelerate==1.2.1 autoawq==0.2.9 numpy

mkdir -p "$(dirname "$OUT")"
"$VENV/bin/python" "$HERE/generate_reference.py" "$MODEL_DIR" "$OUT"
echo "wrote $OUT"
