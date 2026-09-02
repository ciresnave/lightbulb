//! Lightning GGUF loader with memory-mapped tensor access
//!
//! This module provides fast GGUF file loading using memory mapping for 2-10x speedup
//! over traditional seek+read approaches. Key features:
//!
//! - **Zero-copy tensor access**: Tensors are sliced directly from mmap (no copying)
//! - **Direct header parsing**: Parse GGUF v3 format directly from mmap bytes
//! - **Integrated tokenizer extraction**: Extracts tokenizer from GGUF metadata
//! - **Candle-compatible API**: Works alongside candle::quantized::gguf_file
//! - **Cross-platform**: Uses memmap2 for Windows/Linux/Mac compatibility
//!
//! Performance comparison (Phi-3 2GB model):
//! - Traditional (Candle): 3-8 seconds (200+ seek operations)
//! - Memory-mapped (Lightning): 0.5-2 seconds (1 mmap + pointer math)
//! - Speedup: 2-10x faster model loading

mod parser;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub use parser::{GGUFHeader, MetadataValue, TensorInfo as LightningTensorInfo};

// Re-export types from Candle for compatibility
pub use candlelight::core::quantized::gguf_file::{TensorInfo, Value};

/// Memory-mapped GGUF file content with zero-copy tensor access
///
/// This struct holds a memory-mapped view of a GGUF file, providing zero-copy
/// access to tensor data and metadata. The mmap is kept alive for the lifetime
/// of the Content struct.
pub struct Content {
    /// Memory-mapped file (must be kept alive for zero-copy access)
    mmap: Arc<Mmap>,

    /// Parsed GGUF header with metadata and tensor offsets
    header: GGUFHeader,

    /// Candle's parsed content (for backward compatibility)
    candle_content: candlelight::core::quantized::gguf_file::Content,
}

impl Content {
    /// Load a GGUF file using memory mapping
    ///
    /// This is the main entry point for loading GGUF files. It memory-maps the file
    /// and parses the header/metadata using Candle's proven parsing logic.
    ///
    /// # Arguments
    /// * `path` - Path to the GGUF file
    ///
    /// # Returns
    /// A Content struct with parsed metadata and ready for tensor access
    ///
    /// # Performance
    /// This method provides fast initial loading via mmap. The real performance gain
    /// comes from zero-copy tensor access (when we fully integrate with model loading).
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Open and memory-map the file
        let file = File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;

        // Safety: We're mapping a read-only file. The mmap will remain valid as long
        // as the Arc<Mmap> is alive, which we ensure by storing it in the struct.
        let mmap = unsafe {
            Mmap::map(&file)
                .with_context(|| format!("Failed to mmap GGUF file: {}", path.display()))?
        };

        let mmap = Arc::new(mmap);

        // Parse GGUF header directly from mmap (zero-copy)
        let header = parser::parse_gguf(&mmap)
            .with_context(|| format!("Failed to parse GGUF header from: {}", path.display()))?;

        // Also parse using Candle's API for backward compatibility
        // (Can be removed once all code uses Lightning GGUF)
        let mut file = File::open(path)?;
        let candle_content = candlelight::core::quantized::gguf_file::Content::read(&mut file)?;

        Ok(Self {
            mmap,
            header,
            candle_content,
        })
    }

    /// Get metadata from Lightning parser
    pub fn lightning_metadata(&self) -> &HashMap<String, parser::MetadataValue> {
        &self.header.metadata
    }

    /// Get tensor infos from Lightning parser
    pub fn lightning_tensor_infos(&self) -> &[parser::TensorInfo] {
        &self.header.tensor_infos
    }

    /// Get raw memory-mapped bytes (for low-level tensor access)
    pub fn raw_mmap(&self) -> &Arc<Mmap> {
        &self.mmap
    }

    /// Get tensor data offset (start of tensor data section)
    pub fn tensor_data_offset(&self) -> u64 {
        self.header.tensor_data_offset
    }

    /// Get metadata (Candle compatibility)
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.candle_content.metadata
    }

    /// Get all tensor infos
    pub fn tensor_infos(&self) -> &HashMap<String, TensorInfo> {
        &self.candle_content.tensor_infos
    }

    /// Extract tokenizer from GGUF metadata
    ///
    /// This method extracts tokenizer data from GGUF metadata fields and builds
    /// a tokenizers::Tokenizer compatible with the HuggingFace tokenizers library.
    ///
    /// Expected metadata fields:
    /// - tokenizer.ggml.tokens: Array of token strings
    /// - tokenizer.ggml.scores: Array of token scores (optional)
    /// - tokenizer.ggml.token_type: Array of token types (optional)
    /// - tokenizer.ggml.bos_token_id: Beginning-of-sequence token ID (optional)
    /// - tokenizer.ggml.eos_token_id: End-of-sequence token ID (optional)
    ///
    /// # Returns
    /// A tokenizers::Tokenizer instance ready for encoding/decoding
    /// Rebuild the checkpoint's own tokenizer from GGUF metadata.
    ///
    /// **A GGUF carries everything needed to reconstruct the reference
    /// tokenizer exactly, and an earlier version of this function threw all of
    /// it away.** It built a `Unigram` whose scores were INVENTED as
    /// `-(id as f64)` — the negative token index — while
    /// `tokenizer.ggml.scores` and `tokenizer.ggml.merges` sat unread. Unigram
    /// picks the segmentation maximising total score, so fabricated scores made
    /// short low-id pieces always win.
    ///
    /// Measured against `TinyLlama-1.1B-Chat-v1.0`'s own `tokenizer.json`, for
    /// the prompt this project's GGUF end-to-end test sends:
    ///
    /// | | old | reference |
    /// |---|---|---|
    /// | id count | 28 | 22 |
    /// | `capital` | `c`+`ap`+`it`+`al` | `capital` (7483) |
    /// | `France` | `F`+`ran`+`ce` | `France` (3444) |
    /// | newline | **id 0 — the UNK token** | `<0x0A>` (13) |
    /// | BOS with `add_special_tokens` | absent | `<s>` (1) |
    ///
    /// The model was fed UNK for every newline and shattered subwords
    /// throughout, which is the measured cause of the garbage completions in
    /// `tests/gguf_serving_e2e.rs`.
    ///
    /// **The reference is BPE, not Unigram.** This checkpoint's
    /// `tokenizer.ggml.merges` (61249) and `tokenizer.ggml.tokens` (32000) are
    /// BYTE-IDENTICAL to its `tokenizer.json` — verified by direct comparison —
    /// so this rebuilds BPE from them and mirrors the reference's normalizer,
    /// decoder and post-processor rather than approximating them.
    ///
    /// # `merges` is required, and a Unigram fallback was tried and rejected
    ///
    /// Two shapes of `llama` GGUF exist. One carries `tokenizer.ggml.merges`
    /// (converted from a HuggingFace `tokenizer.json`) and is rebuilt here
    /// exactly. The other carries `tokenizer.ggml.scores` and **no merges**,
    /// written by llama.cpp's own SentencePiece converter — measured locally,
    /// 3 of 4 `llama`-model files. **Those are refused.**
    ///
    /// A Unigram-from-real-scores path for them was implemented and then
    /// **removed after measuring it**. It builds, and it fixes byte fallback —
    /// newline stops being UNK — but the segmentation is still wrong:
    ///
    /// ```text
    /// unigram-from-scores  29 ids  us+er   c+ap+it+al   F+ran+ce
    /// reference            22 ids  user    capital      France
    /// ```
    ///
    /// **The scores are real; the algorithm is not the same one.** llama.cpp's
    /// SPM tokenizer is a scored bigram-merge; `tokenizers`' `Unigram` is
    /// Viterbi over unigram log-probabilities. Feeding SPM scores to Unigram
    /// produces plausible output that is not the checkpoint's own — the exact
    /// class of defect this function exists to remove, in a quieter form,
    /// because the words look almost right.
    ///
    /// So an unsupported shape is an ERROR rather than a fabrication. A wrong
    /// tokenizer produces fluent-looking nonsense with nothing in the logs,
    /// which is far worse to debug than a refusal to load. Supporting these
    /// files needs SPM's merge algorithm, not a different model with the same
    /// numbers in it.
    pub fn extract_tokenizer(&self) -> Result<tokenizers::Tokenizer> {
        use tokenizers::{
            AddedToken, Tokenizer,
            decoders::{
                byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence,
                strip::Strip,
            },
            models::bpe::BPE,
            normalizers::{Prepend, Replace, Sequence as NormalizerSequence},
            processors::template::TemplateProcessing,
        };

        const SPM: &str = "llama";
        const BPE_KIND: &str = "gpt2";
        let model_kind = self
            .metadata()
            .get("tokenizer.ggml.model")
            .and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("<absent>");
        if model_kind == BPE_KIND {
            return self.extract_byte_level_bpe_tokenizer();
        }
        if model_kind != SPM {
            let pre = self
                .metadata()
                .get("tokenizer.ggml.pre")
                .and_then(|v| match v {
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("<absent>");
            // Per-kind, because the gpt2 explanation is wrong for bert/t5/gemma4 --
            // and a refusal that confidently explains the wrong obstacle sends the
            // reader somewhere there is nothing to find.
            let detail = "Rebuilding it needs that tokenizer model's own construction, which this function does not implement. Only SentencePiece-lineage checkpoints carrying merges, and byte-level BPE (`gpt2`) whose `tokenizer.ggml.pre` names a verified splitting rule, are handled.";
            // ONE physical line per literal, deliberately. An earlier version used
            // `\` continuations and shipped a real newline into the message, so any
            // caller printing only the first line lost `pre` -- the most useful value
            // in it. The rendered string is what matters, not how the source looks.
            bail!(
                "GGUF tokenizer.ggml.model is {model_kind:?} (tokenizer.ggml.pre = {pre:?}); only {SPM:?} and {BPE_KIND:?} are supported. {detail} Refusing to approximate: a guessed tokenizer produces plausible nonsense with no error anywhere."
            );
        }

        let (tokens, vocab, merges) = self.vocab_and_merges(
            "GGUF has no tokenizer.ggml.merges. This is a SentencePiece-converted checkpoint, and rebuilding it needs SPM's scored bigram-merge algorithm. Building a Unigram from tokenizer.ggml.scores instead was measured and does NOT reproduce the checkpoint's segmentation (29 ids against the reference's 22: `capital` came out as c+ap+it+al), so it is refused rather than approximated.",
        )?;

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            // Load-bearing: without it every byte with no vocab entry becomes
            // UNK. That is what turned each newline in a chat prompt into id 0.
            .byte_fallback(true)
            .build()
            .map_err(|e| anyhow::anyhow!("building BPE from GGUF vocab and merges: {e}"))?;

        let mut tokenizer = Tokenizer::new(bpe);

        // Mirrors the reference tokenizer.json: Prepend then " " -> U+2581, and
        // NO pre-tokenizer. The old code's `Metaspace` pre-tokenizer was a
        // different mechanism reaching a similar-looking result.
        let replace_space = Replace::new(" ", "\u{2581}")
            .map_err(|e| anyhow::anyhow!("building the space normalizer: {e}"))?;
        tokenizer.with_normalizer(Some(NormalizerSequence::new(vec![
            Prepend::new("\u{2581}".to_string()).into(),
            replace_space.into(),
        ])));
        tokenizer.with_pre_tokenizer(None::<tokenizers::pre_tokenizers::PreTokenizerWrapper>);

        let replace_back = Replace::new("\u{2581}", " ")
            .map_err(|e| anyhow::anyhow!("building the space decoder: {e}"))?;
        tokenizer.with_decoder(Some(DecoderSequence::new(vec![
            replace_back.into(),
            ByteFallback::default().into(),
            Fuse::new().into(),
            Strip::new(' ', 1, 0).into(),
        ])));

        // Control tokens must be registered or they tokenize as ordinary text:
        // the EOS marker would become its individual characters.
        let unk_id = self.token_id("tokenizer.ggml.unknown_token_id");
        let bos_id = self.token_id("tokenizer.ggml.bos_token_id");
        let eos_id = self.token_id("tokenizer.ggml.eos_token_id");
        let specials: Vec<AddedToken> = [unk_id, bos_id, eos_id]
            .iter()
            .flatten()
            .filter_map(|&id| tokens.get(id as usize))
            .map(|t| AddedToken::from(t.clone(), true))
            .collect();
        if !specials.is_empty() {
            tokenizer.add_special_tokens(&specials);
        }

        // `tokenizer.ggml.add_bos_token` is a real per-model field and it
        // VARIES: true for llama-spm / gemma / phi-3 / deepseek, false for
        // every SmolLM2 build. It is ABSENT from this checkpoint, and
        // llama.cpp's default for a `llama` tokenizer is to add BOS, so absent
        // means true here rather than false.
        let add_bos = self
            .metadata()
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(true);
        if add_bos {
            if let (Some(bos), Some(id)) = (bos_id.and_then(|i| tokens.get(i as usize)), bos_id) {
                let processor = TemplateProcessing::builder()
                    .try_single(format!("{bos}:0 $A:0"))
                    .map_err(|e| anyhow::anyhow!("building the BOS post-processor: {e}"))?
                    .special_tokens(vec![(bos.clone(), id)])
                    .build()
                    .map_err(|e| anyhow::anyhow!("building the BOS post-processor: {e}"))?;
                tokenizer.with_post_processor(Some(processor));
            }
        }

        Ok(tokenizer)
    }

    /// Rebuild a BYTE-LEVEL BPE tokenizer (`tokenizer.ggml.model == "gpt2"`).
    ///
    /// Structurally different from the SentencePiece path above, not a variant
    /// of it: no normalizer, no `byte_fallback` (byte-level BPE encodes every
    /// byte through GPT-2's byte-to-unicode map, so there is nothing to fall
    /// back to), a `ByteLevel` decoder rather than the
    /// ByteFallback/Fuse/Strip sequence, and — the part that actually blocks
    /// generic support — a PRE-TOKENIZER that differs per checkpoint.
    ///
    /// # `tokenizer.ggml.pre` is the whole difficulty
    ///
    /// It names a splitting rule, and llama.cpp keeps a different regex per
    /// name. Measured over the local corpus: 18 `gpt2` files carrying 13
    /// distinct `pre` values. Picking one for all of them would reproduce this
    /// module's original defect in a quieter form — a tokenizer that is
    /// plausible and wrong.
    ///
    /// So [`Self::bpe_pre_tokenizer`] is a table of rules that have been
    /// VERIFIED against a reference, and anything absent from it is refused.
    fn extract_byte_level_bpe_tokenizer(&self) -> Result<tokenizers::Tokenizer> {
        use tokenizers::{
            AddedToken, Tokenizer, decoders::byte_level::ByteLevel as ByteLevelDecoder,
            models::bpe::BPE,
        };

        let pre = self
            .metadata()
            .get("tokenizer.ggml.pre")
            .and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("<absent>");

        let Some((pre_json, norm_json)) = Self::bpe_pre_tokenizer(pre) else {
            // Values that were investigated and deliberately NOT added get their
            // own reason. A generic "not verified" would send the next person
            // to repeat work that has already been done and come out negative.
            if let Some(why) = Self::bpe_refusal_reason(pre) {
                bail!(
                    "GGUF tokenizer.ggml.pre is {pre:?}, which this build refuses. {why} Verified values: {verified:?}.",
                    verified = Self::VERIFIED_PRE
                );
            }
            bail!(
                "GGUF tokenizer.ggml.pre is {pre:?}, which names a pre-tokenizer splitting rule this build has not verified. Byte-level BPE (`gpt2`) is supported, but only for `pre` values checked id-for-id against a reference: {verified:?}. Refusing to substitute a different rule: the vocab and merges would still load and the output would be plausible and wrong.",
                verified = Self::VERIFIED_PRE
            );
        };

        let (tokens, vocab, merges) = self.vocab_and_merges(
            "GGUF declares a `gpt2` tokenizer but carries no tokenizer.ggml.merges; byte-level BPE cannot be rebuilt without them",
        )?;

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            // FALSE, deliberately, and the opposite of the SPM path. Byte-level
            // BPE maps every byte into the vocab through GPT-2's byte-to-unicode
            // table, so there is no unrepresentable byte for a fallback to
            // catch. The reference `tokenizer.json` for this family agrees:
            // `byte_fallback: false`, `unk_token: null`.
            .byte_fallback(false)
            .build()
            .map_err(|e| {
                anyhow::anyhow!("building byte-level BPE from GGUF vocab and merges: {e}")
            })?;

        let mut tokenizer = Tokenizer::new(bpe);
        // The checkpoint's own declared normalizer and pre-tokenizer. Most are
        // `null`; `qwen2` declares NFC, and the SPM path's Prepend + U+2581
        // substitution would corrupt byte-level input for all of them.
        let normalizer: Option<tokenizers::normalizers::NormalizerWrapper> =
            serde_json::from_str(norm_json).map_err(|e| {
                anyhow::anyhow!(
                    "the recorded normalizer for tokenizer.ggml.pre={pre:?} is not valid JSON: {e}"
                )
            })?;
        tokenizer.with_normalizer(normalizer);
        let pre_tokenizer: tokenizers::pre_tokenizers::PreTokenizerWrapper = serde_json::from_str(
            pre_json,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "the recorded pre-tokenizer for tokenizer.ggml.pre={pre:?} is not valid JSON: {e}"
            )
        })?;
        tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

        // EVERY control token must be registered, not just the four named ids.
        //
        // This registered only `unknown`/`bos`/`eos`/`padding`, and anything
        // else the checkpoint marks as a control token then tokenized as
        // ORDINARY TEXT. Measured on SmolLM2-135M against its own
        // `tokenizer.json`: `"<repo_name>"` is token 3 for the reference and
        // came out as `[44, 22139, 79, 1245, 46]` — the characters — for us.
        // It went unnoticed because that checkpoint's `bos`/`eos` happen to be
        // `<|im_start|>`/`<|im_end|>`, so the tokens a chat prompt actually
        // contains were covered by the four-id list and the rest were not.
        //
        // `tokenizer.ggml.token_type` carries the answer per token:
        // 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 5 UNUSED, 6 BYTE.
        let mut specials: Vec<AddedToken> = Vec::new();
        let types = self.token_types();
        if types.len() == tokens.len() {
            for (id, ty) in types.iter().enumerate() {
                if matches!(ty, 3 | 4) {
                    if let Some(t) = tokens.get(id) {
                        specials.push(AddedToken::from(t.clone(), true));
                    }
                }
            }
        }
        // The four named ids as a floor, in case `token_type` is absent or
        // disagrees in length with the token list.
        for key in [
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
        ] {
            if let Some(t) = self.token_id(key).and_then(|id| tokens.get(id as usize)) {
                if !specials.iter().any(|a| a.content == *t) {
                    specials.push(AddedToken::from(t.clone(), true));
                }
            }
        }
        if !specials.is_empty() {
            tokenizer.add_special_tokens(&specials);
        }

        Ok(tokenizer)
    }

    /// The vocab and merge list, parsed once for both tokenizer families.
    ///
    /// Both paths read the SAME two metadata arrays into the SAME two shapes;
    /// only the message for a missing `merges` differs, which is why that is
    /// the parameter. This was written out twice, and two copies of a parser
    /// that must agree is the hazard this module keeps finding elsewhere.
    ///
    /// Returns the raw token list too: callers need it to resolve
    /// `tokenizer.ggml.*_token_id` indices back into token strings.
    fn vocab_and_merges(
        &self,
        missing_merges: &'static str,
    ) -> Result<(
        Vec<String>,
        tokenizers::models::bpe::Vocab,
        Vec<(String, String)>,
    )> {
        let tokens = self
            .get_metadata_string_array("tokenizer.ggml.tokens")
            .context("Missing tokenizer.ggml.tokens in GGUF metadata")?;
        let merges_raw = self
            .get_metadata_string_array("tokenizer.ggml.merges")
            .context(missing_merges)?;

        let vocab: tokenizers::models::bpe::Vocab = tokens
            .iter()
            .enumerate()
            .map(|(id, t)| (t.clone(), id as u32))
            .collect();

        // GGUF stores each merge as one space-separated pair, exactly as
        // `tokenizer.json` does.
        let merges: Vec<(String, String)> = merges_raw
            .iter()
            .map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .ok_or_else(|| anyhow::anyhow!("malformed merge entry {m:?}: no space"))
            })
            .collect::<Result<_>>()?;

        Ok((tokens, vocab, merges))
    }

    /// Why a specific `tokenizer.ggml.pre` was investigated and NOT added.
    ///
    /// Separate from the generic refusal so that work already done and found
    /// negative is not silently repeated. Both entries here reached "0 of 130
    /// cases disagree" against a candidate reference and were STILL refused,
    /// for reasons a score cannot express.
    fn bpe_refusal_reason(pre: &str) -> Option<&'static str> {
        match pre {
            "qwen35" => Some(
                r#"No admissible reference exists for it. The obvious candidate, `Qwen/Qwen3-8B`, turns out to declare a pre-tokenizer and vocab BYTE-IDENTICAL to `Qwen/Qwen2-7B`, so it is a reference for `qwen2` and not for this name. llama.cpp does define a distinct QWEN35 rule, matching `[\p{L}\p{M}]+` where the qwen2 rule matches `\p{L}+` — and this build's 130-case corpus CANNOT TELL THE TWO APART: measured, they differ on 0 of 130, because the qwen2 normalizer is NFC and composes away the combining marks that are the only thing the difference turns on. A 0-of-130 result here carries no information about which rule is correct, so adding it would be an entry with vacuous evidence."#,
            ),
            "<absent>" => Some(
                r#"An absent `tokenizer.ggml.pre` is not a name, it is the lack of one, and this table keys on names. The corpus file that omits it (gpt-neox) does verify 0 of 130 against `EleutherAI/gpt-neox-20b`, but keying on absence would apply that one checkpoint's rule to EVERY future GGUF that omits the field — which is precisely the one-rule-for-all-checkpoints failure this table exists to prevent. Re-export the checkpoint with `tokenizer.ggml.pre` set, or add its name here."#,
            ),
            _ => None,
        }
    }

    /// `tokenizer.ggml.pre` values whose splitting rule has been verified
    /// id-for-id against a reference. See [`Self::bpe_pre_tokenizer`].
    const VERIFIED_PRE: &'static [&'static str] = &[
        "smollm",
        "gpt-2",
        "falcon",
        "qwen2",
        "deepseek-coder",
        "refact",
        "deepseek-llm",
    ];

    /// The pre-tokenizer for a `tokenizer.ggml.pre` name, or `None` if this
    /// build has not verified that name.
    ///
    /// **A table, not a default.** Returning some general-purpose byte-level
    /// splitter for unknown names is exactly the failure this module exists to
    /// prevent: the vocab and merges load, encoding succeeds, and the ids are
    /// wrong in a way nothing reports.
    /// The declared pre-tokenizer and normalizer for a `tokenizer.ggml.pre`
    /// name, as JSON, or `None` if this build has not verified that name.
    ///
    /// **These are the checkpoints' OWN declarations, copied verbatim from
    /// their `tokenizer.json`.** Storing them as JSON rather than hand-building
    /// the equivalent Rust keeps provenance auditable — each string can be
    /// diffed against the model's published tokenizer — and removes a whole
    /// class of transcription error: `qwen2`'s rule is a 130-character regex,
    /// and one mangled backslash in a hand-written copy produces a tokenizer
    /// that is wrong in ways only a corpus catches.
    /// `every_verified_pre_spec_deserializes` gates them at test time.
    ///
    /// **A table, not a default.** Returning some general-purpose byte-level
    /// splitter for unknown names is exactly the failure this module exists to
    /// prevent: the vocab and merges load, encoding succeeds, and the ids are
    /// wrong with nothing reporting it.
    ///
    /// ⚠️ **Verified against each checkpoint's own `tokenizer.json`, NOT against
    /// llama.cpp.** llama.cpp is not ground truth here: measured over 130 cases
    /// on SmolLM2, ours vs the reference disagreed 0 times, ours vs llama.cpp 2,
    /// and the reference vs llama.cpp the SAME 2. Scoring against it would
    /// reproduce a reference that differs from the checkpoints.
    fn bpe_pre_tokenizer(pre: &str) -> Option<(&'static str, &'static str)> {
        match pre {
            // Verified against `SmolLM2-360M-Instruct/tokenizer.json`. See the
            // caveat below on the 360M/135M substitution.
            "smollm" => Some((
                r##"{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}"##,
                r##"null"##,
            )),
            // openai-community/gpt2 -- verified 0 of 130 cases against that checkpoint's own
            // `tokenizer.json`, whose vocab and merges match this GGUF.
            "gpt-2" => Some((
                r##"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true}"##,
                r##"null"##,
            )),
            // tiiuae/falcon-7b -- verified 0 of 130 cases against that checkpoint's own
            // `tokenizer.json`, whose vocab and merges match this GGUF.
            "falcon" => Some((
                r##"{"type":"Sequence","pretokenizers":[{"type":"Punctuation","behavior":"Contiguous"},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},{"type":"Digits","individual_digits":false},{"type":"Split","pattern":{"Regex":"[0-9][0-9][0-9]"},"behavior":"Isolated","invert":false}]}"##,
                r##"null"##,
            )),
            // Qwen/Qwen2-7B -- verified 0 of 130 cases against that checkpoint's own
            // `tokenizer.json`, whose vocab and merges match this GGUF.
            "qwen2" => Some((
                r##"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]}"##,
                r##"{"type":"NFC"}"##,
            )),
            // deepseek-ai/deepseek-coder-6.7b-instruct -- verified 0 of 130 cases against that checkpoint's own
            // `tokenizer.json`, whose vocab and merges match this GGUF.
            "deepseek-coder" => Some((
                r##"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[\r\n]"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"\\s?\\p{L}+"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"\\s?\\p{P}+"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"[一-龥ࠀ-一가-퟿]+"},"behavior":"Isolated","invert":false},{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"##,
                r##"{"type":"Sequence","normalizers":[]}"##,
            )),
            // smallcloudai/Refact-1_6B-fim -- verified 0 of 130 cases against that checkpoint's own
            // `tokenizer.json`, whose vocab matches this GGUF (extras at the tail).
            "refact" => Some((
                r##"{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}"##,
                r##"null"##,
            )),
            // deepseek-ai/deepseek-llm-7b-base -- verified 0 of 130 cases against that checkpoint's own
            // `tokenizer.json`, whose vocab matches this GGUF (extras at the tail).
            "deepseek-llm" => Some((
                r##"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[\r\n]"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"\\s?[!-/:-~！-／：-～‘-‟　-。]+"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"\\s+$"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"[一-龥ࠀ-一가-퟿]+"},"behavior":"Isolated","invert":false},{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"##,
                r##"{"type":"Sequence","normalizers":[]}"##,
            )),
            _ => None,
        }
    }

    /// `tokenizer.ggml.token_type`, one entry per vocab token.
    ///
    /// Empty when the key is absent or unreadable. **The element type is not
    /// fixed** — measured as `I32` in the local corpus, and `Value::to_i64()`
    /// returns `Err` for `I32`, so an accessor that only tries one width
    /// silently yields an empty list. That is how this array read as "no
    /// control tokens" the first time: the extraction failed and the failure
    /// looked like an answer.
    fn token_types(&self) -> Vec<i64> {
        let Some(Value::Array(a)) = self.metadata().get("tokenizer.ggml.token_type") else {
            return Vec::new();
        };
        a.iter()
            .filter_map(|v| match v {
                Value::I8(x) => Some(*x as i64),
                Value::U8(x) => Some(*x as i64),
                Value::I16(x) => Some(*x as i64),
                Value::U16(x) => Some(*x as i64),
                Value::I32(x) => Some(*x as i64),
                Value::U32(x) => Some(*x as i64),
                Value::I64(x) => Some(*x),
                Value::U64(x) => Some(*x as i64),
                _ => None,
            })
            .collect()
    }

    /// A `tokenizer.ggml.*_token_id` as a `u32`, or `None` if absent or not an
    /// integer.
    fn token_id(&self, key: &str) -> Option<u32> {
        self.metadata().get(key)?.to_u32().ok()
    }

    // Helper methods for metadata extraction

    fn get_metadata_string_array(&self, key: &str) -> Option<Vec<String>> {
        match self.metadata().get(key)? {
            Value::Array(values) => {
                let mut result = Vec::new();
                for v in values {
                    if let Value::String(s) = v {
                        result.push(s.clone());
                    } else {
                        return None;
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get zero-copy access to tensor data by name (Lightning GGUF)
    ///
    /// Returns a slice directly into the memory-mapped file for the specified tensor.
    /// This is the zero-copy path that provides 2-10x faster loading.
    ///
    /// # Arguments
    /// * `name` - Tensor name (e.g., "blk.0.attn_q.weight")
    ///
    /// # Returns
    /// A byte slice pointing to the tensor data in the mmap (zero-copy)
    ///
    /// # Example
    /// ```ignore
    /// let content = gguf::Content::read("model.gguf")?;
    /// let tensor_bytes = content.get_tensor_data("blk.0.attn_q.weight")?;
    /// // Parse quantized data from bytes (Q4_K, Q8_0, etc.)
    /// ```
    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8]> {
        // Find tensor index and info
        let (tensor_idx, tensor_info) = self
            .header
            .tensor_infos
            .iter()
            .enumerate()
            .find(|(_, ti)| ti.name == name)
            .with_context(|| format!("Tensor '{}' not found in GGUF file", name))?;

        // Calculate start offset (absolute position in file)
        let start = (self.header.tensor_data_offset + tensor_info.offset) as usize;

        // Calculate end offset:
        // If there's a next tensor, use its offset
        // Otherwise, use the file size
        let end = if tensor_idx + 1 < self.header.tensor_infos.len() {
            let next_tensor = &self.header.tensor_infos[tensor_idx + 1];
            (self.header.tensor_data_offset + next_tensor.offset) as usize
        } else {
            self.mmap.len()
        };

        // Validate bounds
        if start >= self.mmap.len() || end > self.mmap.len() || start >= end {
            anyhow::bail!(
                "Invalid tensor bounds for '{}' (start: {}, end: {}, file size: {})",
                name,
                start,
                end,
                self.mmap.len()
            );
        }

        // Return zero-copy slice
        Ok(&self.mmap[start..end])
    }

    /// Load a quantized tensor by name from the GGUF file
    ///
    /// This is the key method for loading quantized model weights! It reads the tensor
    /// data from the memory-mapped file and returns a QTensor ready for use with QMatMul.
    ///
    /// # Arguments
    /// * `reader` - A readable file handle (must be the same file that was memory-mapped)
    /// * `name` - Tensor name (e.g., "blk.0.attn_q.weight")
    /// * `device` - Device to load tensor on (CPU/CUDA)
    ///
    /// # Returns
    /// A QTensor containing the quantized weights
    ///
    /// # Example
    /// ```ignore
    /// let mut file = File::open("model.gguf")?;
    /// let content = gguf::Content::read("model.gguf")?;
    /// let q_tensor = content.tensor(&mut file, "blk.0.attn_q.weight", &device)?;
    /// let qmatmul = QMatMul::from_qtensor(q_tensor)?;
    /// ```
    pub fn tensor<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        name: &str,
        device: &candlelight::core::Device,
    ) -> candlelight::core::Result<candlelight::core::quantized::QTensor> {
        // Delegate to Candle's proven tensor loading logic
        self.candle_content.tensor(reader, name, device)
    }
}

#[cfg(test)]
mod bpe_spec_tests {
    use super::*;

    /// Every allowlisted `tokenizer.ggml.pre` has a spec, and every spec is
    /// valid JSON that `tokenizers` accepts.
    ///
    /// The specs are stored as the checkpoints' own declared JSON rather than
    /// hand-built Rust, which keeps their provenance auditable — each can be
    /// diffed against the published `tokenizer.json` — at the cost of moving a
    /// malformed one from a compile error to a runtime error. **This test is
    /// what pays that cost back.** It needs no checkpoint and no network, so it
    /// runs in the ordinary suite rather than behind `#[ignore]`.
    #[test]
    fn every_verified_pre_spec_deserializes() {
        for pre in Content::VERIFIED_PRE {
            let (pre_json, norm_json) = Content::bpe_pre_tokenizer(pre)
                .unwrap_or_else(|| panic!("{pre:?} is listed as verified but has no spec"));

            serde_json::from_str::<tokenizers::pre_tokenizers::PreTokenizerWrapper>(pre_json)
                .unwrap_or_else(|e| panic!("pre-tokenizer spec for {pre:?} does not parse: {e}"));

            serde_json::from_str::<Option<tokenizers::normalizers::NormalizerWrapper>>(norm_json)
                .unwrap_or_else(|e| panic!("normalizer spec for {pre:?} does not parse: {e}"));
        }
    }

    /// And a name that is NOT on the list has no spec.
    ///
    /// The necessary pair: without it, the test above is satisfied by a
    /// `bpe_pre_tokenizer` that returns the same spec for every input, which is
    /// precisely the "one rule for all checkpoints" failure the table exists to
    /// prevent.
    #[test]
    fn an_unlisted_pre_has_no_spec() {
        assert!(Content::bpe_pre_tokenizer("llama-bpe").is_none());
        assert!(Content::bpe_pre_tokenizer("command-r").is_none());
        assert!(Content::bpe_pre_tokenizer("<absent>").is_none());
        assert!(Content::bpe_pre_tokenizer("").is_none());
    }
}
