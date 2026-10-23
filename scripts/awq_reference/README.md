# AWQ reference fixture

An independent reference implementation's output for a real AWQ checkpoint
(`Orion-zhen/Qwen3-0.6B-AWQ`, revision `9f607bee5fca19cba0bd2b383ba804c89385ea18`),
generated once via HF `transformers` + `autoawq` — a quantizer's own output is
a claim until an independent comparison passes, not ground truth.

**Not currently consumed by any test.** This was originally built to back
`tests/awq_qwen3_reference.rs`, comparing against Lightbulb's own
`AwqQwen3`/`load_awq_llama` — both deleted in the candlelight removal
(#195, 0.16.0) before that comparison ever ran on this box. Kept per PM
direction (`docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entry #4, 2026-10-22):
it validates **any** AWQ implementation's correctness, not just the deleted
one, so it's ready for whichever of Fuel's `fuel_transformers::quant_w4a16`,
`baracuda`'s Marlin, or `mlmf-awq` eventually gets wired into Lightbulb and
verified AWQ-format-compatible (`ROADMAP.md`'s "AWQ validation: blocked on a
verified Fuel-side path" row tracks that decision).

## Why this is generated once and committed, not run at test time

`autoawq` has no prebuilt wheel for Windows + Python 3.12 (falls back to a
from-source build of its CUDA extension), and its `triton` dependency has **no
Windows PyPI wheel at all** — `triton-windows`, a community port, satisfies the
Python import but not pip's dependency resolution for the literal package name
`triton`. This box's native toolchain cannot run the reference implementation.

So the reference runs exactly once, in WSL2 (CUDA passthrough works there —
confirmed via `nvidia-smi -L` inside an `Ubuntu-24.04` WSL2 distro), and its
output is committed as a fixture:
`tests/fixtures/awq_reference/v1/qwen3_0_6b_awq_greedy.json`. Whatever future
test consumes it needs neither WSL2 nor Python at test time, only this file.

## Regenerating

From a WSL2 shell with any CUDA-passthrough-capable distro (generated against
`Ubuntu-24.04.5 LTS`; `wsl --install Ubuntu-24.04` if you don't have one) and
this repo's `C:` drive mounted under `/mnt/c` (WSL2's default):

```bash
bash scripts/awq_reference/reproduce-reference.sh /mnt/c/Models/Qwen3-0.6B-AWQ
```

This creates its own venv under `scripts/awq_reference/.venv` (gitignored),
installs the pinned stack (`torch==2.5.1+cu121`, `transformers==4.51.3`,
`accelerate==1.2.1`, `autoawq==0.2.9`), and writes the fixture. The checkpoint
itself is **not** fetched by this script — download it separately (`hf
download Orion-zhen/Qwen3-0.6B-AWQ --revision
9f607bee5fca19cba0bd2b383ba804c89385ea18`) and pass its directory.

## What to do if the output changes

A different `generated_token_ids` on regeneration is not automatically a bug
in Lightbulb *or* in the reference — it could be a version bump in
`autoawq`/`transformers` changing kernel numerics. Compare the new
`provenance` block against the committed one before concluding anything moved
for a reason that matters; if the pinned versions are unchanged and the output
still differs, that is non-determinism in the reference itself and worth its
own investigation before trusting either fixture.

## Licences

`Orion-zhen/Qwen3-0.6B-AWQ` is `gpl-3.0`; its base model `Qwen/Qwen3-0.6B` is
`apache-2.0`. Neither the checkpoint nor the fixture's generated text carries
any of the base model's weights or the AWQ repo's own code into this
repository — the fixture is sixteen integers (token ids) and a short decoded
string, used only as a test oracle.
