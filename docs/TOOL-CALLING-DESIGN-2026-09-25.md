# Tool-calling design — 2026-09-25

Design only, no code. Written in response to CireSnave's question (via the PM) about whether
OverMind can treat Lightbulb as an ordinary OpenAI-compatible provider — answered by
`LIGHTBULB-OPENAI-COMPAT-CHECKLIST.md` (OverMind PR #91) and Lightbulb's own scoring of it
(2026-09-24 peer report). Every row of that checklist passes or is non-blocking except one:
§6, tool-calling, which is a confirmed **total absence**, not a partial implementation —
`tool_calls` never appears in any response (verified by a live request and by a full grep of
`src/api/`), and the single other hit in the whole tree is `model_runner.rs:17-19`'s own
comment explaining why `FinishReason` deliberately has no `ToolCalls` variant: *"nothing in
Lightbulb produces either, and an enum carrying variants no code can construct is worse than
one that grows when a producer appears."*

This matters more than an ordinary missing feature because of what OverMind is: an agent
framework, not a chat client. OverMind's own fallback — `detect_smuggled_tool_call()`, which
regex-scans plain-text `content` for tool-call-shaped JSON a model wrote there instead of a
real `tool_calls` entry — cannot help here, because Lightbulb never tells the model any tools
exist in the first place. There is nothing to smuggle. Every task needing a tool call against
Lightbulb costs a re-prompt on first occurrence and a hard `PROTOCOL_FAILURE` on the second.

## Q1 — where does `tools` have to enter the prompt?

`src/api/chat_template.rs:231-236`, the `tmpl.render(minijinja::context!{...})` call. Today's
context is exactly `{messages, bos_token, eos_token, add_generation_prompt}` — no `tools` key.
This is a place to **add** a key, not invent new machinery: HF's chat-template convention —
which these templates literally are, since they're rendered verbatim from each checkpoint's own
`tokenizer_config.json` `chat_template` string — already defines a `tools` context variable
that template authors branch on with `{% if tools %}`. HF's own `apply_chat_template(tools=...)`
accepts tool definitions in essentially the OpenAI shape
(`{"type":"function","function":{"name","description","parameters"}}`), so little reshaping is
needed on the way in.

**But the file already names a landmine, unprompted, from earlier work on it.**
`chat_template.rs:172-179`'s own comment on Mistral-7B-Instruct-v0.3: `tool.items()` fails under
minijinja's stricter method coverage (`minijinja_contrib::pycompat`), *"in the branch a no-tools
message list never reaches."* This is not hypothetical risk — it's a known failure, already
discovered while building the template-resolution machinery, left latent because nothing
currently passes `tools`. Wiring §1 means immediately exercising code paths in at least one real,
already-supported template that are known not to work. Budget for `pycompat` gaps as part of this
work, not as a surprise found later — Mistral is a confirmed instance; there is no reason to
assume it's the only one among templates not yet exercised with `tools` set.

## Q2 — where does `tool_calls` come back out?

**Per-model-template problem, not a generic parsing problem — and the expensive half of this
job.** There is no universal wire format for how a model expresses "call this tool" in generated
text: Llama-3.1 uses `<|python_tag|>{json}`, Mistral uses `[TOOL_CALLS] [{json}]`, Qwen uses
`<tool_call>{json}</tool_call>` XML-ish tags, and other families differ again. Each needs its own
extraction parser, hand-written and hand-verified against real generated output — no shortcut
available.

This needs a keying mechanism analogous to, but **not reusable as**, `src/api/chat_template/registry.rs`.
That module's own header states its purpose precisely: *"Family table. Isolated from resolution
logic because this is the part that rots — it encodes knowledge that lives in checkpoints and
changes as models ship... Tier order limits the damage: any model shipping its own
`chat_template` never reaches here. The registry serves only models that omit one."* Its three
templates (`ZEPHYR`, `CHATML`, `LLAMA2`, lines 23-29) are last-resort *guesses* for checkpoints
that ship no template of their own, and none of them has any tool syntax at all. The checkpoints
that **do** ship a real template — the common `TokenizerConfig` tier, the majority case — are
exactly the ones that would need tool-output parsing, and that population never reaches this
registry. A tool-call-format table needs its own, parallel structure, keyed against the
`TokenizerConfig`-tier population instead of the registry's guess-of-last-resort population.
Getting this backwards — wiring a parser-selection table to the same population `registry.rs`
serves — would key it to exactly the wrong checkpoints.

Sized at roughly one parser per supported model family, plus that detection/keying mechanism.

## Q3 — sequencing `FinishReason::ToolCalls`

Land the first family's parser and the `FinishReason::ToolCalls` variant in the **same** change,
never the variant alone — exactly the hazard `model_runner.rs:17-19`'s comment exists to prevent.
Concretely: implement one family's output parser, prove it converts real generated text into a
real `Vec<ToolCall>` against a checkpoint from that family, and add the enum variant in the same
PR, wired to that one producer. Every site that matches on `FinishReason` — at least four in
`src/api/openai/chat.rs` by this reading (around lines 503, 527, 695, 857) — gets its new arm at
the same time, from a real value, never a placeholder.

**One asymmetry, worth acting on separately and not conflating with the above.**
`ChatMessage` (`chat.rs:55`) has no `tool_calls` field either, but adding
`tool_calls: Option<Vec<ToolCall>>` with `#[serde(skip_serializing_if = "Option::is_none")]` is
safe to land alone — an always-`None` optional field constructs nothing and lies to no one. The
struct field can land ahead of the first parser without recreating the enum-variant hazard. Do
not let that safety tempt landing `FinishReason::ToolCalls` early "to be ready" — that is the
exact case the existing comment warns against.

## Q4 — what does partial support look like, and how would it be known?

Per-model-knowable, not a coin flip, by construction if Q2/Q3 are sequenced as above: support is
a **named set** of model families with a working parser. Every request is knowably either a
member of that set (real `tool_calls` returned) or not (today's behavior — `tools` silently
dropped, no `tool_calls` ever) — no fuzzy middle state. A checkpoint outside the supported family
set gets exactly today's behavior, which OverMind's own checklist scoring already prices
correctly (one re-prompt, then `PROTOCOL_FAILURE`). The `⚠️ partial` label in OverMind's own §6
framing would describe *Lightbulb as a whole* (some families ✅, most still ❌) while every
individual request stays knowable, not fuzzy — which is the property OverMind's `Provider` table
would need if it ever wants to specify tool support per-model rather than per-server.

## Sizing, stated plainly

**Large, not small.** Q1 alone is genuinely small — a context key plus fixing the known Mistral
landmine before it becomes a live bug. Q2 is the real cost: one hand-written, hand-verified
parser per model family, with no generic shortcut. The **first** family end-to-end — parser,
`FinishReason::ToolCalls`, the four-plus `chat.rs` match sites, the `ChatMessage.tool_calls`
field, and a real test against a real checkpoint proving an actual `tool_calls` shape in a live
response — is expected to be a multi-day unit of work on its own. Every family after the first
adds a parser but not the surrounding machinery again.

**Recommended sequence:** prove the shape once, against a single family (Llama-3.1 or Mistral,
whichever verifies against a checkpoint actually on hand), before committing to a second.

## Two smaller defects found while scoring the checklist, recorded here, not fixed by this doc

1. `GET /v1/models` returns 405 (registered POST-only, `src/api/openai/mod.rs:19`); the POST
   handler returns a **hardcoded fake roster** (`["lightbulb-default","lightbulb-7b","lightbulb-13b"]`)
   that does not name the actually-loaded model. OverMind tolerates a missing `/models` endpoint
   entirely (`models_path` can be `None`), so this is not urgent, but an artifact that actively
   lies costs more than one that is simply absent. Separate fix from the `model`-field validation
   below — bundling them makes both harder to review, since the roster fix has its own open
   design question (what should it report with exactly one model loaded?).
2. The `model` request field is accepted and never validated or routed on — measured by sending
   `model: "nonexistent-model"` and receiving a normal 200 response served by whatever checkpoint
   is actually loaded. Harmless for a single-model server in isolation; a specific, real hazard
   for OverMind, which tries candidate models in sequence and records failures per-model name —
   every name serving the same model makes a 3-model failover record three distinct failures that
   were all the same checkpoint, an actively misleading diagnostic rather than a merely absent
   one. Addressed separately in this same batch (see the companion PR), scoped tightly: reject a
   `model` that does not name the loaded model, with an error naming what is actually loaded.
