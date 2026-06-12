#!/usr/bin/env bash
# Apply all vllm patches inside the container.
#
# Usage (inside container):
#   cd /workspace/vllm
#   bash /path/to/verl-e2e-experiments/patches/apply.sh
#
# Or point at a different vllm tree:
#   VLLM_DIR=/workspace/vllm bash patches/apply.sh
set -euo pipefail

VLLM_DIR="${VLLM_DIR:-/workspace/vllm}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$VLLM_DIR"

for p in "$PATCH_DIR"/*.patch; do
    [ -e "$p" ] || continue
    name="$(basename "$p")"
    # Skip if already applied (reverse-check succeeds => patch is in place).
    if git apply --reverse --check "$p" >/dev/null 2>&1; then
        echo "[skip] $name (already applied)"
        continue
    fi
    if git apply --check "$p" >/dev/null 2>&1; then
        git apply "$p"
        echo "[ok]   $name"
    else
        # Fall back to fuzzy apply if line numbers drifted.
        if patch -p1 --forward --fuzz=3 <"$p" >/dev/null 2>&1; then
            echo "[ok]   $name (fuzzy)"
        else
            echo "[FAIL] $name — apply manually" >&2
            exit 1
        fi
    fi
done

echo "All patches applied."
