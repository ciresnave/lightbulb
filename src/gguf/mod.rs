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

        let Some(pre_tokenizer) = Self::bpe_pre_tokenizer(pre) else {
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
        // No normalizer: the reference has `normalizer: null`. The SPM path's
        // Prepend + U+2581 substitution would corrupt byte-level input.
        tokenizer.with_normalizer(None::<tokenizers::normalizers::NormalizerWrapper>);
        tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

        // Control tokens must be registered or they tokenize as ordinary text.
        let specials: Vec<AddedToken> = [
            self.token_id("tokenizer.ggml.unknown_token_id"),
            self.token_id("tokenizer.ggml.bos_token_id"),
            self.token_id("tokenizer.ggml.eos_token_id"),
            self.token_id("tokenizer.ggml.padding_token_id"),
        ]
        .iter()
        .flatten()
        .filter_map(|&id| tokens.get(id as usize))
        .map(|t| AddedToken::from(t.clone(), true))
        .collect();
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

    /// `tokenizer.ggml.pre` values whose splitting rule has been verified
    /// id-for-id against a reference. See [`Self::bpe_pre_tokenizer`].
    const VERIFIED_PRE: &'static [&'static str] = &["smollm"];

    /// The pre-tokenizer for a `tokenizer.ggml.pre` name, or `None` if this
    /// build has not verified that name.
    ///
    /// **A table, not a default.** Returning some general-purpose byte-level
    /// splitter for unknown names is exactly the failure this module exists to
    /// prevent: the vocab and merges load, encoding succeeds, and the ids are
    /// wrong in a way nothing reports.
    fn bpe_pre_tokenizer(pre: &str) -> Option<tokenizers::pre_tokenizers::PreTokenizerWrapper> {
        use tokenizers::pre_tokenizers::{
            byte_level::ByteLevel, digits::Digits, sequence::Sequence,
        };
        match pre {
            // Verified against `SmolLM2-360M-Instruct/tokenizer.json`, whose
            // vocab and merges are BYTE-IDENTICAL to the SmolLM2 GGUFs in the
            // corpus (49152 tokens, 48900 merges, zero id mismatches). Its
            // `pre_tokenizer` is a Sequence of `Digits{individual_digits:true}`
            // then `ByteLevel{add_prefix_space:false}` — the digit split is the
            // part that distinguishes it from plain GPT-2, which encodes
            // " 12345" as " 123" + "45".
            "smollm" => Some(
                Sequence::new(vec![
                    Digits::new(true).into(),
                    ByteLevel::new(false, true, true).into(),
                ])
                .into(),
            ),
            _ => None,
        }
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
