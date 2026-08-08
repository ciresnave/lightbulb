# Chat-template resolution — design

**Date:** 2026-08-06
**Status:** designed and approved; **not yet planned.** The sibling spec
(`2026-08-06-runner-result-metadata-design.md`) is planned and implemented
first.

**Dependency, stated precisely:** tiers 0–4 and the probe are fully independent
of the sibling spec. Only the runtime monitor (§3) needs `FinishReason`, which
the sibling introduces. So this spec could be built first *except* for §3 —
which is why it is sequenced second rather than being blocked.

---

## Summary

`/v1/chat/completions` builds its prompt as
`messages.map(|m| format!("{}: {}", m.role, m.content)).join("\n")`
(`chat.rs:184-188`). That is not any model's chat template. A chat model
prompted this way behaves like a base model: it continues text rather than
answering, and does not reliably emit EOS.

**Measured, not assumed:** on TinyLlama-1.1B-Chat with the current prefix, EOS
fired in **1 of 6 trials** across `max_tokens` ∈ {24, 64, 100}. Every other
request ran its full budget — latency and compute spent generating text nobody
asked for, on every request that does not happen to stop.

This spec resolves the correct template per model, records how it was chosen,
and notices when the choice was wrong.

---

## Context: the obvious approaches both fail on our own test model

**[verified 2026-08-06]** The TinyLlama-1.1B-Chat snapshot this project tests
against contains exactly three files:

```
config.json   model.safetensors   tokenizer.json
```

- **No `tokenizer_config.json`**, so no `chat_template` field. The authoritative
  source is absent.
- `tokenizer.json`'s `added_tokens` are `['<unk>', '<s>', '</s>']` only.
  TinyLlama-Chat uses Zephyr-style `<|user|>` / `<|assistant|>`, but those are
  **ordinary text**, not special tokens — so a vocabulary scan finds nothing.

A design resting on either "read the checkpoint's template" or "detect the
template from special tokens in the vocab" would fail on the first model we try
it against. Both are still worth having — they are free and correct when they
apply — but neither is sufficient, which is why a family registry is necessary
rather than a convenience.

---

## §1 — Five tiers, cheapest first

| Tier | Source | Cost | Confidence |
| --- | --- | --- | --- |
| **0** | Sidecar `lightbulb-chat-template.json` beside the model | free | whatever produced it — recorded |
| **1** | `chat_template` in `tokenizer_config.json` | free | authoritative |
| **2** | Vocabulary signature (`<\|im_start\|>` → ChatML, `[INST]` → Llama-2/Mistral) | free | high, when markers are added tokens |
| **3** | Family registry — `config.json` fields + model directory name | free | heuristic |
| **4** | Probe — generate with candidates, score by EOS rate | **generations** | empirical |

Tiers 0–3 run automatically at startup and are all free. **Tier 4 never runs
automatically** — see §4.

Resolution stops at the first tier that produces a template. The tier that
produced it is recorded and surfaced, so nobody has to guess later whether a
template was read or inferred.

## §2 — Rendering

HuggingFace `chat_template` values are Jinja. Rendered with **`minijinja`** —
pure Rust, no C dependencies.

Two things must be registered explicitly or real templates fail to render:

- **`raise_exception`** — templates call it to reject unsupported message
  orders. Without it, rendering errors with an unhelpful "unknown function".
- **`bos_token` / `eos_token`** — referenced as variables, sourced from the
  tokenizer.

A template that fails to render falls through to the next tier rather than
failing the request, and logs which template failed and why.

## §3 — Detecting a wrong choice at runtime

Tier 3 is a heuristic and can be wrong. The signal that it was wrong is the same
one that revealed the original defect: **EOS-fire rate.**

A rolling counter over recent completions tracks the fraction that finished with
`FinishReason::Stop` rather than `Length`.

**Concrete starting values, so this is implementable rather than aspirational:**
warn when fewer than **25%** of the last **20** completions finished with
`Stop`. Both are constants with a comment, tunable without a design change.

The reasoning behind 25%: the measured wrong-template rate on TinyLlama was
1-in-6 (≈17%), and a correctly-templated chat model answering short questions
should terminate nearly always. A threshold between those separates them with
room for models that legitimately produce long output. A 20-completion window
keeps a handful of long answers from tripping it.

These are a starting point chosen from one model's measurement, not a
calibrated value. If it proves noisy, raising the window is the first move —
the failure mode of a too-eager warning is a misleading log line pointing at
the wrong subsystem.

**Secondary signals**, weaker but cheap: the model emitting template markup as
literal text (`<|user|>`, `[/INST]`, `<|im_end|>`), or opening a new
conversational turn instead of answering.

**This monitors; it does not act.** It never silently switches templates — a
server that changes its prompting mid-flight based on a heuristic is harder to
debug than one that is consistently wrong and says so.

## §4 — The probe is an operator action

```
lightbulb chat-template probe <model-dir>
```

Renders a fixed prompt under each candidate template, generates with each, and
reports EOS-fire rate per candidate. Writes the sidecar **on confirmation**, not
automatically.

**Why not automatic**, when it would resolve our own test model unattended:

- It costs real generations. At startup it either delays serving or races with
  live traffic.
- A wrong conclusion gets **persisted**, and a cached wrong answer is worse than
  no cache: it stops anyone re-examining the question and carries the authority
  of a file on disk.
- Nobody would have looked at it. The probe's output is a small table that a
  human can sanity-check in seconds; skipping that step trades a large risk for
  a small convenience.

## §5 — Persistence records provenance, not just the answer

```json
{
  "template": "<jinja source>",
  "resolved_by": "probe",
  "evidence": "EOS 8/8 with zephyr, 1/8 with llama2, 0/8 with alpaca",
  "resolved_at": "2026-08-06T00:00:00Z",
  "model_fingerprint": "<hash of config.json>"
}
```

`resolved_by` and `evidence` are the load-bearing fields. Without them a
name-matched guess and a probed result are indistinguishable, and the first
person to doubt the template has to redo the work.

`model_fingerprint` invalidates the sidecar if the checkpoint changes
underneath it, so a directory reused for a different model does not silently
inherit the wrong template.

## §6 — Where it lives

| File | Responsibility |
| --- | --- |
| `src/api/chat_template.rs` **new** | Tier resolution, `minijinja` rendering, sidecar read/write |
| `src/api/mod.rs` *modify* | Resolve once at startup where the model path is known; store on `AppState` |
| `src/api/openai/chat.rs` *modify* | Render messages through the template instead of the ad-hoc join |
| `src/bin/lightbulb-cli.rs` *modify* | `chat-template probe` subcommand |

Resolution happens **once at startup**, not per request: it reads files, and a
per-request read would put filesystem I/O in the request path for a value that
cannot change while the process runs.

`/v1/completions` uses **no template** — it is OpenAI's raw-text endpoint.

## §7 — Testing

- **A known template renders to the expected string.** TinyLlama's Zephyr form
  produces `<|user|>\n…\n</s>\n<|assistant|>\n` for a fixed message list.
  Asserted against the literal expected output, not against "contains
  `<|user|>`" — a template that emitted the marker in the wrong place would pass
  the weaker check.
- **Tier order is honoured.** A model with both a `chat_template` and a
  registry entry resolves via tier 1. Fails if the tiers are reordered or if a
  later tier overwrites an earlier result.
- **A missing template falls through rather than failing.** A model with none of
  tiers 0–3 available serves with the documented fallback and logs it.
- **Sidecar round-trips with provenance.** Written and re-read, `resolved_by`
  and `evidence` survive. Fails if the writer drops them for brevity.
- **A stale sidecar is rejected.** Changing the fingerprint invalidates it.
  Fails if the fingerprint is written but never checked — which is the likely
  slip, and the reason this test exists.
- **`raise_exception` is registered.** A template calling it produces a template
  error, not an "unknown function" error. Distinguishes "we support this
  construct" from "it happened not to be used".

---

## §8 — Risks

| Risk | Assessment |
| --- | --- |
| **Registry rots** | A family table encodes knowledge that lives in checkpoints and changes as models ship. Mitigated by tier order: any model shipping its own template never consults the registry. The registry serves only models that omit one. |
| **`minijinja` divergence from Jinja2** | HF templates are authored against Python's Jinja2. Differences exist in filters and whitespace control. Rendering failure falls through to the next tier and logs; silent *mis*-rendering is the residual risk, and the render test guards the shape we ship against. |
| **EOS-rate monitor misfires** | A model legitimately generating long answers looks like a template failure. It only warns, never acts, so the cost is a log line. Threshold and window need to be conservative. |
| **Probe over-fits its prompt** | A single fixed prompt could favour one template by chance. Report per-candidate rates rather than a winner, and require confirmation, so a human sees a 5/8-vs-4/8 result for what it is. |

---

## Out of scope

- Multi-turn conversation state — the API is stateless; templates render the
  messages given.
- Tool/function-call templates. Some chat templates render tool definitions;
  nothing in Lightbulb produces them yet.
- Vision or multimodal message content.
- Auto-switching templates at runtime based on the monitor (§3) — explicitly
  rejected there.
