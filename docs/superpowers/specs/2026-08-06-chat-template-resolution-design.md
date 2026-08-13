# Chat-template resolution — design

**Date:** 2026-08-06
**Status:** designed, approved, and **planned** —
`docs/superpowers/plans/2026-08-08-chat-template-resolution.md` (6 tasks).
The sibling spec (`2026-08-06-runner-result-metadata-design.md`) was planned
and implemented first; it merged to `main` as PR #5 on 2026-08-08.

**Read the CORRECTED 2026-08-08 block in §6 before implementing.** The body of
this spec cites one prompt-construction site; there are three, and the third
lives in a module this spec's file table does not name.

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
lightbulb-probe <model-dir>
```

Renders a fixed prompt under each candidate template, generates with each, and
reports per candidate whether generation stopped on EOS. Writes the sidecar
**on confirmation**, not automatically.

> ### CORRECTED 2026-08-12 — as shipped, not as designed
>
> Two details above were wrong when implemented, and both were caught before
> implementation:
>
> - **The command is `lightbulb-probe`, a sibling binary**, not a subcommand of
>   `lightbulb-cli` (see §6's table row, corrected likewise). `grep -c
>   "lightbulb::" src/bin/lightbulb-cli.rs` returns **0**: that binary is a pure
>   HTTP client, and giving it in-process inference would put the engine's link
>   requirements on a tool whose job is to talk to a server that already has
>   them.
> - **There is no "rate".** A rate needs N trials that can differ, and the
>   default backend decodes greedily (`ParallelModelManager`,
>   `logits_slice.argmax(0)`) — only the `fuel-engine` path reads `temperature`
>   at all — so N trials of one prompt are the same generation N times and every
>   row could only read `0/N` or `N/N`. The probe generates **once** per
>   candidate, which makes each row deterministic and reproducible rather than a
>   sample of size one pretending otherwise. The confirmation gate is unchanged:
>   the risk it guards is a probe over-fitting its single prompt, which N never
>   addressed.
>
> §5's `evidence` example below still shows the old `8/8` shape. It is left as
> written because it illustrates the *field*, not the format; the shipped probe
> writes `probe: zephyr stopped on EOS in 8 tokens; chatml, llama2 did not`.

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
| `src/bin/lightbulb-probe.rs` **new** | The probe. A sibling binary, **not** a `lightbulb-cli` subcommand — see §4's CORRECTED block |

> ### CORRECTED 2026-08-08 — this table undercounts the work
>
> The summary cites `chat.rs:184-188` as though the ad-hoc join were one site.
> **It is three, and the table above omits the module containing the third.**
> Verified against `main` at `1506d64`:
>
> | Site | Reached by | Now |
> | --- | --- | --- |
> | `src/api/openai/chat.rs:186` | `create_chat_completion` — non-streaming | in the table |
> | `src/api/openai/chat.rs:409` | `create_chat_stream` — **streaming** | implied at best |
> | `src/contracts/validation.rs:146` | `messages_to_prompt`, called from `executor.rs:183` | **absent** |
>
> Fixing only the first leaves streaming and every contract request still
> prompting a chat model like a base model. Each unfixed site would read as
> correct in isolation, which is the failure shape the sibling branch hit four
> times.
>
> **`messages_to_prompt` is the worst of the three, because its doc comment
> presents the defect as a deliberate contract:** *"This matches the simple
> `role: content` format used by the existing `create_chat_completion`
> implementation."* A future reader has been told the duplication is intentional.
> It must be **deleted**, not re-pointed — leaving a second renderer behind is
> how the two drift apart again.
>
> ### The contract path cannot be fixed by rendering upstream
>
> The obvious fix — render once in `chat.rs` and hand `execute_contract` a
> string — **does not work**, and this is a design constraint rather than a
> detail. `execute_contract` *mutates the message list between attempts*:
> `inject_contract_instruction` (`executor.rs:170`) and `tightening_message`
> (`:176-179`) both push messages, and `:183` re-renders. The message list at
> attempt 3 is not the one the caller had.
>
> So the renderer must be reachable from inside the retry loop. The change:
>
> ```rust
> // now:  F: Fn(String) -> Fut
> // ->    F: Fn(Vec<RawMessage>) -> Fut
> ```
>
> The caller owns the template and renders each attempt's list; the contracts
> module stops owning prompt formatting entirely, which is what lets
> `messages_to_prompt` be deleted rather than duplicated. `Vec` rather than
> `&[RawMessage]` deliberately — a borrowing callback returning a `Future`
> needs an HRTB that buys nothing here, and the list is already cloned per
> contract at `:167`.
>
> **Consequence for `/v1/completions`:** unchanged. §6 already rules it takes no
> template, and it constructs no message list — the audit above found no fourth
> site.

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
