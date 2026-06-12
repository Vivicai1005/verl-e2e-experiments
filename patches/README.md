# vllm patches

Local edits to the in-container vllm tree (`/workspace/vllm`), stored as diffs so
they can be re-applied any time the container is recreated.

## Apply (inside the container)

```bash
cd /workspace/vllm
bash ~/verl-e2e-experiments/patches/apply.sh
```

Or point at a different tree:

```bash
VLLM_DIR=/some/other/vllm bash ~/verl-e2e-experiments/patches/apply.sh
```

`apply.sh` is idempotent: it skips patches already applied, uses `git apply`, and
falls back to fuzzy `patch -p1 --fuzz=3` if vllm's line numbers have drifted.

## Record a new change

Make the edit inside the container, then generate the diff straight from git so the
headers and hunk offsets are correct:

```bash
cd /workspace/vllm
# ...make your edit...
git diff path/to/file.py > ~/verl-e2e-experiments/patches/<short-name>.patch
```

One `.patch` file per change. `apply.sh` picks up every `*.patch` in this directory
automatically.

## Current patches

| File | What it does |
| --- | --- |
| `vllm-oracle-unquantized-no-contiguous.patch` | Drop the `.contiguous()` calls on the returned `w13_weight` / `w2_weight` in `fused_moe/oracle/unquantized.py` |
