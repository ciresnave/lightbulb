# F3: `/v1/embeddings` design note

Status: DESIGN, no code. Decision: CireSnave board 505, 2026-11-26, verbatim: *"505: Yes for the embeddings endpoint.  No for streaming tool calls."* (`C:/Projects/CIRESNAVE-DECIDED-ARCHIVE.md` item 505).

## What is known and what is not

Source: Fuel lane `[FINDING] embeddings-paths` at fuel origin/main@dcf8c46a, **file-existence grep only, nothing run**.

- Encoder models exist in `fuel-transformers/src/models/`: `lazy_bert`, `lazy_distilbert`, `lazy_jina_bert`, `lazy_nomic_bert`, `lazy_modernbert`, `lazy_debertav2`, `lazy_xlm_roberta`, `lazy_stella_v5`, `lazy_nvembed_v2`; CLIP-style `lazy_clip`, `siglip`, `mobileclip`, `colpali` (out of scope here, image/text-pair models).
- `Tensor::l2_normalize` exists (`fuel-tensor/src/lazy.rs`, tested there). **Mean and CLS pooling are written per model, not a shared op.** attention_mask is handled in distilbert, jina_bert, debertav2 and the examples.
- Runnable references: `fuel-examples/examples/{bert,distilbert,jina-bert,nomic-bert,stella-en-v5,nvembed_v2,gte-qwen,splade,modernbert}`.
- **Confirmed by Fuel, source read (`[FINDING] embeddings-unknowns`, same ref, 2026-11-26):**
  - **No padded batch anywhere.** Every encoder forward takes ONE sequence (`&[u32]`); bert has no mask at all; nomic_bert, jina_bert, modernbert take an optional additive mask tensor; stella_v5 and nvembed_v2 take `&[u32]` masks with mask-weighted mean + L2; gte-qwen is last-token, one prompt at a time. Batching = a loop of single sequences.
  - **No GGUF encoder path** (0 hits in bert/jina/nomic/modernbert/stella/nvembed/xlm_roberta; the same grep finds GGUF in `lazy_quantized_qwen2`, so the query works). Encoders load from safetensors (`MmapedSafetensors`; read for bert, jina_bert, modernbert, others unread). **The Ollama GGUFs (`nomic-embed-text`, `mxbai-embed-large`, `bge-m3`, `all-minilm`, `snowflake-arctic-embed`) are not loadable today.** Needs a Fuel row if GGUF embedders are wanted; the v1 path below uses safetensors.
  - No shared Embedder/pooling API exists or is planned: pooling is Lightbulb-side per-family config.
  - `release_pooled_memory` is not built in Fuel (0 hits); no encoder-specific release.
- Lightbulb today serves GGUF `llama` and `qwen3` decoders only, so **every embedding path here is new to Lightbulb**, whatever Fuel has; v1's model source is Hugging Face safetensors, not the Ollama GGUF a user already holds.

## First models (proposal)

Ollama's embedding models, by what its users pull: `nomic-embed-text` (nomic_bert), `mxbai-embed-large` (bert-large), `bge-m3` (xlm_roberta), `all-minilm` (bert), `snowflake-arctic-embed` (bert-family). Evidence for this ranking is my recall of Ollama's library, **not measured**; the PM/CireSnave can reorder it. Names below are the Ollama names; **the v1 loader takes the safetensors repo of the same model** (e.g. `nomic-ai/nomic-embed-text-v1.5`), because Fuel has no GGUF encoder loader. Proposal:

1. **`nomic-embed-text`**: the Ollama default in most RAG guides; Fuel has `lazy_nomic_bert` and an example. Needs task prefixes (`search_document:`, `search_query:`) which the *client* sends; we must not add them.
2. **bert family** (`all-minilm`, `mxbai-embed-large`, `snowflake-arctic-embed`): one `lazy_bert` covers three popular names; cheapest breadth.
3. Later: `bge-m3` (xlm_roberta, 8k context, multilingual), `gte-qwen`/`stella` (decoder-backed, last-token pooling).

Do the first one end to end before adding a second; the second proves the per-model pooling table is real.

## Request and response (OpenAI shape)

`POST /v1/embeddings`, body `{model, input, encoding_format?, dimensions?, user?}`.

- `input`: a string, an array of strings, (OpenAI also allows token-id arrays: **reject with 400**, not silently wrong). Empty string: 400 like OpenAI. Cap array length and per-item tokens (config; over-length = 400, never silent truncation).
- `encoding_format`: `float` (default) or `base64` (little-endian f32, base64). Anything else 400. The OpenAI Python SDK requests `base64` by default, so this is not optional.
- `dimensions`: only for models trained for it (Matryoshka, e.g. nomic v1.5, mxbai): truncate then re-normalise. For other models 400 with the supported value. Never truncate a non-Matryoshka model.
- Response: `{object:"list", data:[{object:"embedding", index, embedding}], model, usage:{prompt_tokens,total_tokens}}`. `usage` counts the real tokenizer tokens including special tokens, per item summed; no completion tokens.
- Errors use the existing OpenAI error body: unknown model 400 (as chat); a loaded *chat* model named here, or an embedding model named in chat, 400 `model is not an embedding model`; OOM on load 503 with the existing load-failure path.
- `/v1/models` lists embedding models, marked so a client can tell (field `lightbulb.kind: "embedding"`, name is the ground truth; to be settled in review).

## Pooling and normalisation: a table, not a default

Wrong pooling returns plausible, silently bad vectors, so each model carries an explicit row, taken from the model's own reference config/card:

| model | pooling | normalise | prefix | notes |
|---|---|---|---|---|
| nomic-embed-text | mean (masked) | yes | client | Matryoshka |
| bert-family (minilm, mxbai, arctic) | per card: mean or CLS | per card | per card | read each |
| bge-m3 | CLS | yes | none | |
| gte-qwen / stella | last-token / per card | yes | per card | |

A model with no row is refused at load, not guessed. The row lives beside the loader, not in request handling. Mean pooling must use the attention mask (padding excluded), which is the trap in batching.

## Load/unload policy

One embedding model resident at a time (CireSnave's laptop, one GPU shared with the chat model). It goes through the existing model manager (lease, evict-idle) and F2 unload (`admin/models/{name}/unload`) rather than a parallel loader. Unload on idle after a configurable timeout (default shorter than chat since reload is cheap: proposal 60 s, review). A request that needs the embedding model while the chat model holds the device evicts the idle one if possible, else queues or 503s; **the memory behaviour with both resident is to be measured on the laptop, not assumed** (F2 two-model GPU measurement is the same test). Fuel has no encoder-specific release and `release_pooled_memory` is unbuilt, so v1 unload means dropping the model handle through the path F2 uses for decoders; **whether that returns the GPU memory is to be measured**.

## Batching

Fuel has no padded-batch forward (confirmed), so an input array is a loop of single-sequence forwards, in order; correct, not fast. The array-size cap bounds latency. No cross-request batching in v1; real batching is a Fuel feature to request only if throughput measures as a problem.

## `/api/embed` (Ollama-native): decision

Row 23 of the census says no Ollama-native routes exist and PM held "no Ollama-API code without a go". Lightbulb is meant to replace Ollama, but the board decision covers `/v1/embeddings` only. **Recommendation: not in F3.** It is a thin shim over the same core (`{model,input}` -> `{embeddings:[[...]]}`, always normalised, plus `/api/embeddings` legacy), so deferring costs little; ask CireSnave only if a client he names speaks the Ollama protocol. Left as an open question for review, not assumed.

## Test plan

1. **Reference vectors**: for each supported model, vectors from the model's reference implementation (sentence-transformers / the card's code) for 5 fixed sentences, stored as fixtures with the reference version and command recorded; assert cosine >= 0.999 (f32) per sentence. A weaker tolerance needs a stated reason.
2. **Pooling control**: the test must fail if mean is swapped for CLS (compute both, assert the wrong one is below tolerance; a test that cannot go red proves nothing).
3. Position invariance: a sentence alone vs inside an array gives the same vector (cosine >= 0.9999).
4. Unit norm for normalising models; `dimensions` truncation re-normalised; base64 round-trips to the float form.
5. Wire tests: array/string, bad `encoding_format`, token-id input 400, wrong-kind model 400, usage counts equal tokenizer count.
6. A synthetic tiny encoder fixture (like `src/test_models.rs`) so CI exercises the route without a downloaded model; the real-model reference test follows the existing real-checkpoint gating, with the self-hosted-runner rule (no `#[ignore]` without a census line).

## Open questions for review

1. First model order (above). 2. `dimensions` default behaviour on non-Matryoshka models (400 proposed). 3. Idle timeout default. 4. `/api/embed` (not in F3 proposed). 5. Whether to ask Fuel for a GGUF encoder loader (the Ollama-held files) or stay on safetensors for v1 (recommended: safetensors).
