# GGUF metadata as a chat-template source — design

**Status:** IMPLEMENTED — with its own root-cause claim falsified by the
implementation. §1 and §9 carry correction boxes; read those before trusting any
causal statement in this document.
**Supersedes nothing.** Extends `2026-08-06-chat-template-resolution-design.md`,
whose tier model this slots into.

## 1. The defect, measured

Serving a real GGUF through the shipped server returns garbage. Measured
2026-08-14, `--release`, default (candlelight) build, `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`
at revision `52e7645ba7c309695bec7ac98f4f005b139cf465`, file
`tinyllama-1.1b-chat-v1.0.Q4_0.gguf`, driven over HTTP with
`{"messages":[{"role":"user","content":"Name the capital of France."}],"max_tokens":24,"temperature":0.0}`:

```
content: "| ass istant | <0x0A> | ass istant | ass istant | | ass istant | ass | ass"
usage:   prompt_tokens 27, completion_tokens 24, finish_reason "length"
```

The startup log names the cause:

```
WARN  …declares no BOS or EOS token; rendering its chat template with an empty
      string there. A chat template that references the missing token will omit it.
      Chat template … resolved via Registry (bos "", eos "")
```

**Loading is fine.** The model loaded, prefilled 27 tokens and decoded 24.

~~What is broken is the prompt we hand it.~~ **Also falsified 2026-08-19** — see
the box below. The prompt was broken and is now fixed; the output is still
garbage, so "inference is fine" was an assumption carried by the same
correlation, not a measurement.

**Contributing defect — and NOT the root cause, measured 2026-08-19.** A
single-file `.gguf` has no companion JSON, so `special_tokens` returns empty
BOS/EOS and `resolve` falls through to a tier-3 family guess.
`registry::ZEPHYR` interpolates `eos_token`, so the rendered prompt carries **no
end-of-turn marker at all**.

> ### ⚠️ This section originally called that the root cause. That is falsified.
>
> This epic fixed the template. The prompt now renders correctly from the GGUF's
> own metadata:
>
> ```
> "<|user|>\nName the capital of France.</s>\n<|assistant|>\n"
> ```
>
> **and the model still emits garbage:**
>
> ```
> "| ass istant | i | user | user | ass istant | | | | | ass istant | < | user |"
> ```
>
> Byte-identical across four independent runs at `temperature: 0.0` against
> `tinyllama-1.1b-chat-v1.0.Q4_0.gguf`, with `bos "<s>"`, `eos "</s>"` and
> `Resolution::GgufMetadata` all asserted correct. So the missing end-of-turn
> marker was **a** defect, not **the** defect.
>
> **Why the original diagnosis looked solid and was not.** It rested on evidence
> **invariant to its own truth**: "broken template + garbage output" is equally
> consistent with *the template causes the garbage* and *the template is broken
> and something else is broken too*. Those two are indistinguishable until the
> template is fixed — which is what this epic did. Recorded rather than quietly
> amended, because the failure mode is the interesting part: a correlation with
> a plausible mechanism read as a cause.
>
> **What is now known, and it is more than before.** Chat-template resolution and
> rendering are correct, verified two independent ways (the served response's
> BOS/EOS/tier, and the rendered prompt string itself). The garbage is
> **downstream** of templating, in GGUF quantized generation.
>
> **Next diagnostic, deliberately not taken here:** a correct prompt *string* is
> not correct *token ids*. The previous epic shipped a BOS-doubling defect of
> exactly this shape — a template interpolating `bos_token` into prompt text
> while a `TemplateProcessing` post-processor prepended BOS again — which is why
> `BuiltPrompt` carries `add_special_tokens`. Dump the ids actually fed for this
> prompt and compare against llama.cpp's for the same string.

**The file already contains what we need.** Reading its header directly:

```
tokenizer.chat_template     = "{% for message in messages %}\n{% if message['role'] == 'user' %}…"
tokenizer.ggml.bos_token_id = 1  →  tokenizer.ggml.tokens[1] = "<s>"
tokenizer.ggml.eos_token_id = 2  →  tokenizer.ggml.tokens[2] = "</s>"
```

We never look inside the file. The previous epic taught `fingerprint`,
`sidecar_path` and `metadata_dir` that *companion files live beside a `.gguf`,
not inside a directory named after it* — correct, and one step short. For a
GGUF, the model author's declaration is **in the file's own metadata**.

## 2. Goal

A `.gguf` checkpoint resolves its chat template and special tokens from its own
embedded metadata, with the same authority a directory checkpoint's
`tokenizer_config.json` carries — so the served prompt is the one the model was
trained on.

Explicitly **not** a goal: serving GGUF on the Fuel backend. See §8.

## 3. Where it slots, and what outranks what

**Tier numbering does not change.** GGUF metadata becomes a second *source* within
existing tier 1 — "the checkpoint's own declaration" — rather than a new tier that
renumbers the rest. This is deliberate: tier numbers appear throughout the code
comments, the previous spec, and the implementation plan's Global Constraints
(e.g. "TinyLlama resolves at tier 3"), and a silent renumber would make all of that
prose wrong while every sentence still read as correct.

| tier | source | authority |
| --- | --- | --- |
| 0 | sidecar (`*.lightbulb-chat-template.json`) | operator decision, fingerprint-checked |
| **1** | **GGUF in-file metadata** *(`.gguf` only)*, **then** `tokenizer_config.json` | the author's declaration |
| 2 | vocabulary signature | inference |
| 3 | family registry | guess |
| 4 | probe | operator measurement (writes a sidecar) |

**Decision: for a single-file `.gguf`, in-file metadata outranks a companion
`tokenizer_config.json`.** The reason is the one that already shaped
`sidecar_path` in the previous epic: **a `.gguf` is file-scoped and a companion
JSON is directory-scoped.** Two `.gguf` files can sit in one directory — which
is why sidecars are named after the checkpoint rather than sharing one — and a
single `tokenizer_config.json` beside them cannot be authoritative for both. The
in-file metadata is unambiguously about *these* weights.

A directory checkpoint is unaffected: tier 1 simply does not apply to it.

## 4. Provenance: a new `Resolution` variant

Add `Resolution::GgufMetadata`. **Not** reuse of `TokenizerConfig` — this epic
exists to record *how* a template was chosen, and labelling a template read from
GGUF metadata as having come from a JSON file is exactly the kind of quiet lie
the sidecar's `evidence` field was introduced to prevent.

**This has a deliberate compile-time consequence.** `probe_override_check` matches
`Resolution` exhaustively with no `_` arm, precisely so a new variant is an error
rather than a silent "proceed". Adding this variant will break that match, and the
correct classification is **refuse** — alongside `TokenizerConfig`, for the same
reason: the checkpoint declares its own template, and a probe sidecar would
outrank it. `--force` still overrides.

## 5. Who reads the GGUF

**Fuel's `quantized::gguf_mmap::MmapedContent`**, which exposes
`pub fn metadata(&self) -> &HashMap<String, Value>`.

Rationale, and the alternatives rejected:

- **`fuel-core` is an unconditional dependency** (`Cargo.toml`, no `optional`),
  and `fuel-engine = []` enables no dependencies — it only switches which runner
  compiles. So Fuel's reader is available in **every** build, including the
  default candlelight one. This fix therefore lands once and serves both
  backends, needs no feature gate, and needs no Fuel pin bump — `metadata()`
  exists at our current pin `8771997`.
- **Not our `src/gguf/`.** It works, but it is candlelight-era and slated for
  retirement along with the three `src/model/custom_*` files that take
  `crate::gguf::Content` as a parameter. Extending it adds to what must later be
  removed.
- **Not MLMF.** Its `main` is coupled to `candlelight` (unconditional dep;
  `LoadOptions` carries `Device`/`DType`), so adopting it would tie this to the
  backend we are leaving. A backend-agnostic rebuild is in design; when
  `mlmf-meta` exists, swapping the reader behind this module's interface is a
  contained change. That is the point of putting the read behind one function.

`MmapedContent::from_path` mmaps the file. Mapping is virtual and this runs at
startup rather than per request — but two details of the earlier wording here
were wrong. `Content::read` does **not** touch only the header: it materialises
the entire metadata block eagerly, which on the real Q4_0 means
`tokenizer.ggml.tokens` as 32,000 heap `String`s. And it runs **twice per
start**, not once, because `resolve_for_model` (`chat_template.rs:770-775`)
calls both `resolve` and `special_tokens` and each reads the header. Neither is
a correctness problem; both are recorded so nobody re-derives the cost from an
optimistic sentence.

## 6. What is extracted, and one hard rule

From metadata:

- `tokenizer.chat_template` → the template source
- `tokenizer.ggml.bos_token_id` → index into `tokenizer.ggml.tokens` → BOS string
- `tokenizer.ggml.eos_token_id` → index into `tokenizer.ggml.tokens` → EOS string

**Byte-exact, no normalization, ever.** Special tokens are rendered into prompt
**text** and the tokenizer must then recognise the identical bytes. Any
normalization — Unicode, whitespace, case — between "what the file declared" and
"what we render" produces a prompt that reads correctly and tokenizes
differently, with no error. This rule comes from Unpopped, who enforce it for
kernel-dispatch tokens where `cuda:sm90` and `cuda:sm90a` are different targets a
prefix match silently collapses; token strings and merges have the same
"looks normalizable, isn't" property.

**A blank template is UNDECLARED, not declared-empty.** `trim().is_empty()`, not
`is_empty()` — consistent with the rule shipped in `030f1d5`, and for the same
reason: an empty source renders to an empty prompt, which reaches the model as a
request to continue nothing.

## 7. Failure modes

Every one falls through to the next tier and logs; none aborts startup. A GGUF
that declares nothing is a normal checkpoint, not an error.

| condition | behaviour |
| --- | --- |
| not a valid GGUF / header unreadable | fall through, `warn` naming the path |
| **valid GGUF, readable metadata, but Fuel refuses to open it** | fall through, `warn` naming the path **and the likely cause** — see below |
| no `tokenizer.chat_template` key | fall through, `debug` (common and benign) |
| template present but blank/whitespace | fall through, `warn` (malformed file) |
| **template present but not a string** (array, integer, …) | fall through, `warn` naming the key and the reason — **must not look like "absent"** |
| `bos/eos_token_id` absent | leave that token empty, existing `warn` applies |
| id out of range for `tokenizer.ggml.tokens` | leave empty, `warn` naming the id |
| `tokenizer.ggml.tokens` absent entirely | ids unresolvable, leave empty, `warn` |

### 7.1 The second row is not a malformed file, and the log must not say it is

**Measured 2026-08-15**, at our pinned Fuel rev `8771997e`:

```
fuel-formats/src/gguf.rs:389   Content::read
  ~405-412   metadata KV loop        <- tokenizer.chat_template parsed successfully HERE
   432-433   per-tensor GgmlDType::from_u32(..)?   <- aborts the entire read
```

`GgmlDType::from_u32` (`fuel-ir/src/quantized.rs:35`) accepts exactly
`{0,1,2,3,6,7,8,9,10,11,12,13,14,15,30}`. **Every IQ code is missing** — IQ4_NL
(20), IQ3_S (21), IQ4_XS (23), IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18),
IQ1_S (19), IQ2_S (22), IQ1_M, and the TQ family. These are ordinary
quantizations `llama.cpp` reads without complaint; MLMF's 29-file Hub corpus
carries 540 `IQ4_NL`, 30 `IQ3_S` and 30 `IQ4_XS` tensors.

MLMF's architect independently enumerated the live type space against
`llama.cpp` at its pinned commit — 27 quantized block structs cross-checked
against llama.cpp's own `static_assert`s, zero mismatches — and puts it at **35
codes**, adding `TQ1_0` (34), `TQ2_0` (35), `MXFP4` (39), `NVFP4` (40) and the
recent `Q1_0` (41) / `Q2_0` (42) to the list above. Their count and our reading
agree on the arithmetic: **Fuel accepts 15 and rejects 20.** The 15 is our
measurement of Fuel; the 35 is theirs of `llama.cpp`.

So on such a file **the chat template is decoded, held in memory, and then
discarded** because a later loop hit a tensor dtype we never needed.

**We could avoid this, and are choosing not to.** An earlier draft of this
section said *"there is no metadata-only entry point — `Content::read` is the
only `pub fn read`"*. That is false, and the falsehood was relayed to two other
projects before being caught. `fuel-core` re-exports the primitives
(`fuel-core/src/quantized/gguf_file.rs:19-21`): `VersionedMagic::read`
(`fuel-formats/src/gguf.rs:45`), `read_string` (`:68`), `ValueType::from_u32`
(`:124`) and `Value::read` (`:296`) are all `pub`. A metadata-only reader that
stops before the tensor directory is roughly 25 lines, constructible today at
pin `8771997e` with no upstream change.

We are not writing it, because our checkpoints do not hit the case and a
hand-rolled partial parser is a second implementation of the header layout —
the one nobody reviews. **What is missing upstream is a convenience function,
not a capability.** Recorded as a decision so the next person does not
re-derive it as an impossibility.

**Why this earns its own row rather than folding into the first.** The two
conditions arrive as the same `Err` *variant* from `from_path`, though not with
the same content — `fuel-formats/src/gguf.rs:31` and `:53` bail on magic and
version, `fuel-ir/src/quantized.rs:52` on `unknown dtype for tensor {u}` — and
the warn interpolates `{e}`, so the discriminating text does reach the operator.
Separating them structurally would be cheap too: read four bytes and compare
them to `GGUF` before calling `from_path`. We emit one message covering both
rows, hedged with "if the file is otherwise sound", rather than pay for that
split.

They stay separate rows because they are **opposite facts about the operator's
file**: one is corrupt, one is fine and we are the limitation. Logging both as
"not a valid GGUF" sends an operator hunting for corruption that does not exist.
Worse, the fall-through lands on a family-registry guess with no end-of-turn
marker — **the exact defect §1 measured**, reintroduced through a branch this
spec had classified as benign. A silent fall-through is what this epic exists to
eliminate; it must not be reintroduced by the error path of the fix.

**This does not block the epic.** Our checkpoint is Q4_0 (code 2, supported).
The remedy here is an honest log, not a workaround. Extending the type table is
Fuel's (filed with their architect); making metadata reads structurally immune
to it is MLMF's design — `mlmf-gguf` (container) and `mlmf-ggml` (block-quant
type table) are deliberately separate crates for this reason.

## 8. Out of scope

- **GGUF on the Fuel backend.** Needs a Fuel pin bump to `f1da2d94+` for
  `impl DecodeModel for QuantizedLlama3Model`, and is affected by the MLMF
  consolidation. Separate epic.
- **Adopting MLMF**, or retiring `src/gguf/` and the candlelight loaders. Those
  go with candlelight.
- **Sharded GGUF.** Out of scope here, and noted because Fuel's bench hardcodes
  the single-file name with no index handling — a shared limitation worth
  knowing before anyone points either project at a 7B.

## 9. Testing

**A synthetic GGUF builder in the test module**, not the 637,699,456-byte fixture. A valid
GGUF v3 header with metadata KV pairs and **zero tensors** is a few hundred
bytes and can be written by the test itself. This is the only way to exercise the
failure table in §7 — a real file has one metadata set, and the cases that matter
are the malformed ones.

Fixtures must be able to fail. The previous epic shipped a defect that every
TinyLlama-based test was structurally blind to, because its tier-3 template
never mentions `bos_token`; the guard tests had to build a synthetic
Llama-3-shaped checkpoint before they could discriminate anything. The same
applies here: assert the **observable consequence** (which template source
resolves, what BOS/EOS come back) rather than an internal flag, and cover both
polarities of every branch.

### ⚠️ The single-gate design below was falsified along with §1. Read that box first.

**As designed, this section specified ONE behavioural gate** — `#[ignore]`d and
checkpoint-gated against the real Q4_0 file — asserting all of:

- the completion **contains `"Paris"`**, and
- the resolved tier is `Resolution::GgufMetadata`, not `Registry`, and
- BOS/EOS come back as `"<s>"` / `"</s>"` rather than empty.

The reasoning was that the last two make the first non-accidental: a completion
could contain "Paris" for reasons unrelated to the template, and asserting the
resolved source plus the recovered tokens pins *why* it did.

**That reasoning was sound and its premise was wrong.** It assumed the three
assertions would pass or fail together, because it assumed the template was the
cause. Measured 2026-08-19: the last two pass, the first fails, and the rendered
prompt is correct. See §1's falsification box.

**What was actually built (`7b2d25d`), and why it is two tests:**

| test | state | asserts |
| --- | --- | --- |
| `a_gguf_is_served_with_its_own_template` | **passes** | BOS `"<s>"`, EOS `"</s>"`, `Resolution::GgufMetadata`, HTTP 200, **and the byte-exact rendered prompt** |
| `a_gguf_completion_is_still_garbage_after_correct_templating` | **`#[ignore]`d, expected to fail** | the completion contains `"Paris"` — a recorded downstream defect, not a regression from this epic |

Splitting rather than deleting is deliberate: the failing assertion is a
**measurement**, and deleting it would erase the evidence that isolates the
remaining bug. Splitting rather than leaving one red test is also deliberate:
the epic's deliverable is verified correct and should not be gated on a defect
one layer below it.

**The rendered-prompt assertion is the part worth keeping.** Nothing in the
suite asserted the string that actually reaches the model — the synthetic tests
assert *resolution*. That gap is why the false root-cause survived two audit
rounds. It is now asserted through `ResolvedTemplate::render`, the same call
`build_prompt_from_raw` makes internally.

Both tests mirror `tests/chat_template_e2e.rs`'s shape — its `post_raw` helper
and the `AppState` construction inside it, but **not** its
`#![cfg(feature = "fuel-engine")]` line, which is right for that file and wrong
for a backend-independent test.

## 10. Also in this change

Delete `src/loaders/mlmf_wrapper.rs` — 6,501 bytes, dated 2026-07-28, calling
`mlmf::prelude`, `mlmf::callbacks::default_progress_callback`,
`mlmf::load_safetensors` and `mlmf::load_gguf`. `grep -rn "mlmf_wrapper" src/`
returns nothing: it was never re-declared as a module after the MLMF integration
was abandoned, so it does not compile and is invisible. Found by MLMF's own agent
reading our tree. It is in `src/loaders/`, which this change touches, and its
`convert_mlmf_config_to_lightbulb()` is the artifact that argued MLMF should
expose raw JSON plus accessors rather than a normalized struct.

## 11. Open questions for review

**RULED 2026-08-15: keep `Option<String>` — but the justification given for
keeping it was wrong, and the thing it got wrong was a real defect.**

Raised by MLMF's architect as "absent is not empty". `GgufDeclaration.template`
is one `Option` collapsing three states: key absent, key present but wrong type,
key present but blank. The *outer* `Option<GgufDeclaration>` already separates
"not a GGUF / unreadable" from "valid GGUF declaring nothing".

The original answer here was: *behaviour is identical in all three cases, and §7
already gives them different log levels, so the collapse is harmless.* Tracing
the actual code paths shows that is true for only two of the three:

| state | path | log |
| --- | --- | --- |
| key absent | `md.get(..)` → `None` | `resolve`'s `debug!` only |
| **key present, wrong type** | `.and_then(\|v\| v.to_string().ok())` → `None` | **`resolve`'s `debug!` only — identical to absent** |
| key present, blank | `non_blank` → `None` | `non_blank`'s `warn!` **plus** the `debug!` |

So absent-vs-blank really are distinguished, and for those two the collapse is
fine: behaviour is identical, `non_blank` is the shared guard, and `resolve` is
the only consumer of `.template`. **The type stays as it is.**

But *present-but-undecodable* — the state actually raised — was indistinguishable
from absent in **both** the type and the log, and §7 had no row for it. A GGUF
declaring its template as an array or an integer would emit one `debug!` reading
"no usable tokenizer.chat_template" and fall to a family guess with no
end-of-turn marker. That is this epic's own failure mode, reproduced inside the
epic's fix.

**The remedy is not a type change**: it is a `warn!` on the `Err` branch of
`to_string()`, the §7 row added above, and a test. All three are in the plan.

The lesson worth keeping is not about `Option`. It is that *"these cases are
distinguished by logging"* is a claim about code, and it was accepted here
without tracing all three paths. Two were checked and the third was assumed to
behave like them.

**Should a companion `tokenizer_config.json` beside a `.gguf` ever win?** §3 says
no, on the file-scoped-versus-directory-scoped argument. The counter-case is an
operator who deliberately places a corrected `tokenizer_config.json` next to a
GGUF whose embedded template is wrong. Today that operator's remedy is the
sidecar, which outranks everything and records provenance — which is the
mechanism designed for exactly this. Flagged because it is the one precedence
decision here that a reasonable reviewer could take the other way.
