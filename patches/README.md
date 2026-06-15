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

## Check if a patch is applied

Ask whether the patch could be *reversed* — if yes, it is already applied:

```bash
cd /workspace/vllm
git apply --reverse --check ~/verl-e2e-experiments/patches/<name>.patch \
  && echo "APPLIED" || echo "NOT applied"
```

- `git apply --reverse --check <patch>` succeeds → already applied
- `git apply --check <patch>` succeeds → not yet applied (can apply forward)

> ⚠️ **Caveat for fuzzy-applied patches.** `git apply` has **no fuzz factor** — it
> requires the patch's context lines to match the file *exactly*. If `apply.sh`
> reported `(fuzzy)`, the patch was applied by GNU `patch -p1 --fuzz=3`, which slid
> the hunk to a drifted offset that `git apply` will no longer recognize. In that
> case `git apply --reverse --check` reports **"NOT applied" even though it is** — a
> false negative. Re-check with the same tool that applied it, or just look at the
> file:
>
> ```bash
> cd /workspace/vllm
> # same tool, same fuzz, dry run:
> patch -p1 --reverse --fuzz=3 --dry-run < ~/verl-e2e-experiments/patches/<name>.patch \
>   && echo "APPLIED" || echo "NOT applied"
> # or just inspect the changed line(s):
> git diff --stat path/to/file.py
> ```

Or just re-run `apply.sh`: `[skip] ... (already applied)` means it is in place,
`[ok] ...` means it was just applied now. To see all local edits at once:

```bash
cd /workspace/vllm && git diff
```

> ⚠️ **Re-running after a fuzzy apply.** The `[skip]` guard in `apply.sh` also uses
> `git apply --reverse --check`, so it suffers the same false negative: a
> fuzzy-applied patch is **not** detected as already applied on the next run. GNU
> `patch --forward` still refuses to apply it twice ("previously applied patch
> detected"), but that returns non-zero, so the script would mislabel it `[FAIL]`.
> The durable fix is to record the patch from a clean tree so it applies cleanly
> (see below) — once it applies via `git apply`, the idempotency guard works.

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
