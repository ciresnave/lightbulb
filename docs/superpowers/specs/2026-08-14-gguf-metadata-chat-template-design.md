# GGUF metadata as a chat-template source — design

**Status:** proposed
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

**Loading and inference are fine.** The model loaded, prefilled 27 tokens and
decoded 24. What is broken is the prompt we hand it.

**Root cause.** A single-file `.gguf` has no companion JSON, so `special_tokens`
returns empty BOS/EOS and `resolve` falls through to a tier-3 family guess.
`registry::ZEPHYR` interpolates `eos_token`, so the rendered prompt carries **no
end-of-turn marker at all** and the model free-associates.

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

`MmapedContent::from_path` mmaps the file. Mapping is virtual, we touch only the
header, and this runs **once at startup** — not per request.

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
| no `tokenizer.chat_template` key | fall through, `debug` (common and benign) |
| template present but blank/whitespace | fall through, `warn` (malformed file) |
| `bos/eos_token_id` absent | leave that token empty, existing `warn` applies |
| id out of range for `tokenizer.ggml.tokens` | leave empty, `warn` naming the id |
| `tokenizer.ggml.tokens` absent entirely | ids unresolvable, leave empty, `warn` |

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

**A synthetic GGUF builder in the test module**, not the 640 MB fixture. A valid
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

**One behavioural gate**, `#[ignore]`d and checkpoint-gated, against the real
Q4_0 file: serve the §1 request over HTTP at `temperature: 0.0` and assert
**content**, not "coherence" — which is not assertable. Concretely, mirroring
`tests/fuel_engine_http.rs`'s existing shape:

- the completion **contains `"Paris"`**, and
- the resolved tier is `Resolution::GgufMetadata`, not `Registry`, and
- BOS/EOS come back as `"<s>"` / `"</s>"` rather than empty.

The last two are what make the first non-accidental: a completion could contain
"Paris" for reasons unrelated to the template, and asserting the resolved source
plus the recovered tokens pins *why* it did. The §1 measurement is the known-red
state, so this gate has a demonstrated failing baseline rather than an assumed
one — re-running it against `HEAD~` must produce the recorded garbage.

## 10. Also in this change

Delete `src/loaders/mlmf_wrapper.rs` — 6,501 bytes, dated 2026-07-28, calling
`mlmf::prelude`, `mlmf::callbacks::default_progress_callback`,
`mlmf::load_safetensors` and `mlmf::load_gguf`. `grep -rn "mlmf_wrapper" src/`
returns nothing: it was never re-declared as a module after the MLMF integration
was abandoned, so it does not compile and is invisible. Found by MLMF's own agent
reading our tree. It is in `src/loaders/`, which this change touches, and its
`convert_mlmf_config_to_lightbulb()` is the artifact that argued MLMF should
expose raw JSON plus accessors rather than a normalized struct.

## 11. Open question for review

**Should a companion `tokenizer_config.json` beside a `.gguf` ever win?** §3 says
no, on the file-scoped-versus-directory-scoped argument. The counter-case is an
operator who deliberately places a corrected `tokenizer_config.json` next to a
GGUF whose embedded template is wrong. Today that operator's remedy is the
sidecar, which outranks everything and records provenance — which is the
mechanism designed for exactly this. Flagged because it is the one precedence
decision here that a reasonable reviewer could take the other way.
