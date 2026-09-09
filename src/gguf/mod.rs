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

    /// Our own parser's header, ABSENT when our parser refuses the file.
    ///
    /// ⚠️ SYMMETRIC TO `candle_content`, AND FOR THE SAME REASON POINTED THE
    /// OTHER WAY. `parser::parse_gguf` reads GGUF v2/v3 only: v1 stores counts
    /// and string lengths as `u32` where v2/v3 use `u64`, so it is a different
    /// layout rather than one more accepted version number.
    ///
    /// Candle reads v1 (`VersionedMagic::GgufV1`). So the two parsers refuse
    /// DIFFERENT files, and before this was optional the call failed whenever
    /// EITHER refused -- an `AND` over two readers with complementary gaps,
    /// which throws away the union of their coverage and reports it as a shared
    /// limitation. Measured on `C:\Models`: candle reads the one v1 file with
    /// 18 metadata keys, 48 tensor infos and 512 tokenizer entries, and never
    /// got the chance because our parser had already failed the call.
    header: Option<GGUFHeader>,

    /// Why our parser refused, when it did.
    header_refusal: Option<String>,

    /// Candle's parsed content, ABSENT when candle refuses the file.
    ///
    /// ⚠️ Candle's `TensorInfo` embeds a `GgmlDType`, and `GgmlDType::from_u32`
    /// (candle-core 0.10.2, `src/quantized/mod.rs:293`) rejects the whole IQ
    /// codebook family — IQ4_NL (20), IQ3_S (21), IQ4_XS (23) among them. So a
    /// checkpoint carrying one cannot be represented in candle's types AT ALL,
    /// and this is `None` rather than a partially-filled value.
    ///
    /// It is needed only to LOAD TENSORS. Everything metadata-shaped is served
    /// from `header`, which our own parser produced before this was attempted.
    candle_content: Option<candlelight::core::quantized::gguf_file::Content>,

    /// Why candle refused, when it did. Carried so the error names the real
    /// cause at the point of use rather than "unavailable".
    candle_refusal: Option<String>,

    /// Our own parser's metadata, in candle's shape — populated ONLY when candle
    /// refused the file, and empty otherwise.
    ///
    /// ⚠️ EMPTY IS THE COMMON CASE AND IT IS NOT A GAP. When candle parsed, the
    /// map it already owns is returned directly and this stays empty. An earlier
    /// version of this field held `c.metadata.clone()` so one field could serve
    /// both paths — which cloned every model's metadata on load, and metadata
    /// includes `tokenizer.ggml.tokens`: 32 000 heap `String`s on TinyLlama.
    /// A convenience for the accessor, paid for on every load of every model.
    ///
    /// `reader_agreement_tests` proves the two sources agree on every corpus
    /// file both readers can read.
    fallback_metadata: HashMap<String, Value>,
}

/// A checkpoint's tokenizer vocabulary, and its merges if it declares any.
///
/// ⚠️ A NAMED TYPE RATHER THAN A TUPLE because the tuple tripped
/// `clippy::type_complexity` and the gate caught it -- 15 -> 17. Raising the
/// ceiling was the other option and would have been wrong: three anonymous
/// positions whose third is an `Option<Vec<(String, String)>>` is exactly as
/// hard to read as the lint says it is.
struct VocabAndMerges {
    /// The raw token list. Callers need it to resolve
    /// `tokenizer.ggml.*_token_id` indices back into token strings.
    tokens: Vec<String>,
    vocab: tokenizers::models::bpe::Vocab,
    /// `None` when the file declares no `tokenizer.ggml.merges`.
    merges: Option<Vec<(String, String)>>,
}

/// SHA-256 of the 32000-token Llama SentencePiece vocabulary.
///
/// Computed by `Content::vocab_sha256`, and independently by a Python script
/// implementing the same length-prefixed scheme, which agreed. Both
/// `ggml-vocab-llama-spm.gguf` and `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` produce
/// it -- their token lists are byte-identical -- which is exactly why one can
/// serve as an oracle for the other. `ggml-vocab-phi-3.gguf` (32064 tokens) and
/// `ggml-vocab-baichuan.gguf` (64000) produce different digests, as they must.
const LLAMA_SPM_VOCAB_SHA256: &str =
    "92cdbd78176976ed0c31897436a0b785cc99437d18cedb014044c4b64273ef70";

/// SHA-256 of Phi-3's 32064-token vocabulary.
///
/// ⚠️ A DIFFERENT DIGEST FROM `LLAMA_SPM_VOCAB_SHA256`, AND THE SAME VOCABULARY
/// UNDERNEATH. Phi-3's first 32000 ids are BYTE-IDENTICAL to the Llama
/// SentencePiece list; the extra 64 are its chat control tokens
/// (`<|endoftext|>`, `<|assistant|>`, `<|system|>`, `<|end|>` and placeholders),
/// none of which splits into two vocabulary tokens, so they contribute NO
/// merges — which is why the derived list is the same 61249 for both.
///
/// ⚠️ THAT RELATION WAS INVISIBLE TO THE SWEEP BUILT TO LOOK FOR IT. A digest
/// comparison answers `A == B` and is blind to `A ⊂ B`: hashing destroys
/// containment, since one appended token changes the whole value. The sweep
/// reported "phi-3 shares its vocabulary with nothing" and ITS POSITIVE CONTROL
/// PASSED — the control proved it could find EQUALITY, which is exactly the
/// relation that was not there. What caught it was an unexplained coincidence:
/// a 32064-token file producing exactly the 61249 merges of a 32000-token one.
const PHI3_VOCAB_SHA256: &str = "45715642b43ea2169115398dc8853cc6e8b70c969e42574417009b01724e2a2f";

/// SHA-256 of Baichuan's 64000-token vocabulary.
///
/// ⚠️ THIS ONE HAS NO ORACLE, AND ITS WARRANT IS WEAKER THAN THE OTHER TWO'S.
/// No corpus file shares this vocabulary — searched by digest across all 30,
/// and the search finds four genuinely shared groups, so it can detect sharing.
/// Nor is it a prefix of, or extended by, any of them. So there is no declared
/// merge list anywhere to compare the derived one against.
///
/// What stands in for it is stated at `spm_derivation_warrant`, and it is
/// evidence against gross failure rather than proof of correctness.
const BAICHUAN_VOCAB_SHA256: &str =
    "392ea9d92bd1c32d3dee2ce425d501eb9ab91eb27636a9e0a69af4651292034e";

/// Convert our own parser's metadata into candle's `Value` shape.
///
/// ⚠️ EXISTS BECAUSE CANDLE CANNOT ALWAYS PARSE A FILE WE CAN. Both enums are
/// the GGUF spec's thirteen metadata types, so this is a relabelling and not an
/// interpretation — every arm is total and there is no fallback.
///
/// ⚠️ THIS IS A SECOND IMPLEMENTATION OF A HEADER LAYOUT, WHICH IS THE THING
/// THE AUGUST SPEC DECLINED TO BUILD: "a hand-rolled partial parser is a second
/// implementation of the header layout -- the one nobody reviews."
///
/// The objection is real and it is answered by proof rather than by care:
/// `tests/gguf_metadata_reader_agreement.rs` runs BOTH readers over the corpus
/// and asserts key-for-key equality on every file candle can also read. A
/// second implementation that is continuously shown to agree with the first is
/// not the one nobody reviews; it is the one that reviews itself on every run.
fn to_candle_value(v: &parser::MetadataValue) -> Value {
    use parser::MetadataValue as M;
    match v {
        M::UInt8(x) => Value::U8(*x),
        M::Int8(x) => Value::I8(*x),
        M::UInt16(x) => Value::U16(*x),
        M::Int16(x) => Value::I16(*x),
        M::UInt32(x) => Value::U32(*x),
        M::Int32(x) => Value::I32(*x),
        M::Float32(x) => Value::F32(*x),
        M::Bool(x) => Value::Bool(*x),
        M::String(x) => Value::String(x.clone()),
        M::Array(xs) => Value::Array(xs.iter().map(to_candle_value).collect()),
        M::UInt64(x) => Value::U64(*x),
        M::Int64(x) => Value::I64(*x),
        M::Float64(x) => Value::F64(*x),
    }
}

/// Our parser's metadata, in candle's shape.
fn metadata_from_header(header: &GGUFHeader) -> HashMap<String, Value> {
    header
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), to_candle_value(v)))
        .collect()
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

        // Parse GGUF header directly from mmap (zero-copy). A refusal is
        // recorded, not propagated -- see the `header` field.
        let (header, header_refusal) = match parser::parse_gguf(&mmap) {
            Ok(h) => (Some(h), None),
            Err(e) => (None, Some(e.to_string())),
        };

        // ⚠️ CANDLE'S PARSE IS A SECOND, REDUNDANT READ OF THE SAME BYTES, AND
        // ITS FAILURE USED TO SINK THE WHOLE CALL.
        //
        // `parser::parse_gguf` above already produced the metadata AND the
        // tensor directory, and it stores `tensor_type` as a plain `u32`
        // without consulting any dtype table. Candle's read then re-parsed the
        // file and bailed on `GgmlDType::from_u32` for the IQ family.
        //
        // Measured 2026-09-06 on `C:\Models`: three SmolLM2 quantizations were
        // reported `unreadable` for this reason alone, while carrying complete
        // 49152-token vocabularies in the KV header -- which sits AHEAD of any
        // tensor. `unreadable` was a property of THIS CALL and was read as a
        // property of the FILE, understating the corpus in every census built
        // on it.
        //
        // So a refusal here is recorded, not propagated. Metadata comes from
        // our own parser; only tensor LOADING still requires candle, and a file
        // it cannot type is a file whose tensors it could not have loaded.
        let mut file = File::open(path)?;
        let (candle_content, candle_refusal) =
            match candlelight::core::quantized::gguf_file::Content::read(&mut file) {
                Ok(c) => (Some(c), None),
                Err(e) => (None, Some(e.to_string())),
            };

        // ⚠️ THE ONLY FATAL CASE IS BOTH READERS REFUSING. Either alone leaves
        // the file readable, and naming BOTH reasons matters: the two parsers
        // fail for unrelated causes, so one message would send the reader after
        // the wrong one.
        if header.is_none() && candle_content.is_none() {
            anyhow::bail!(
                "neither GGUF reader could parse {}. Ours: {}. Candle: {}.",
                path.display(),
                header_refusal.as_deref().unwrap_or("(no reason recorded)"),
                candle_refusal.as_deref().unwrap_or("(no reason recorded)")
            );
        }

        // Built only when candle refused AND ours succeeded. Nothing is copied
        // on the common path.
        let fallback_metadata = match (&candle_content, &header) {
            (Some(_), _) => HashMap::new(),
            (None, Some(h)) => metadata_from_header(h),
            (None, None) => unreachable!("the both-refused case bailed above"),
        };

        Ok(Self {
            mmap,
            header,
            header_refusal,
            candle_content,
            candle_refusal,
            fallback_metadata,
        })
    }

    /// Our parser's own header, or why it is absent.
    ///
    /// ⚠️ Errors rather than returning an empty value. Our parser reads GGUF
    /// v2/v3 only, so on a v1 file this is genuinely unavailable -- and an
    /// empty header would be indistinguishable from a file with no tensors.
    fn require_header(&self) -> Result<&GGUFHeader> {
        match &self.header {
            Some(h) => Ok(h),
            None => bail!(
                "this crate's own GGUF parser did not read this file: {}. Metadata IS available through `metadata()`, served by candle. Only the zero-copy mmap accessors need this parser.",
                self.header_refusal
                    .as_deref()
                    .unwrap_or("no reason recorded")
            ),
        }
    }

    /// Get metadata from Lightning parser
    pub fn lightning_metadata(&self) -> Result<&HashMap<String, parser::MetadataValue>> {
        Ok(&self.require_header()?.metadata)
    }

    /// Get tensor infos from Lightning parser
    pub fn lightning_tensor_infos(&self) -> Result<&[parser::TensorInfo]> {
        Ok(&self.require_header()?.tensor_infos)
    }

    /// Get raw memory-mapped bytes (for low-level tensor access)
    pub fn raw_mmap(&self) -> &Arc<Mmap> {
        &self.mmap
    }

    /// Get tensor data offset (start of tensor data section)
    pub fn tensor_data_offset(&self) -> Result<u64> {
        Ok(self.require_header()?.tensor_data_offset)
    }

    /// Get metadata (Candle-shaped, whichever parser produced it).
    ///
    /// Borrows candle's own map when candle parsed, so the common path copies
    /// nothing; falls back to ours only for a file candle refused.
    pub fn metadata(&self) -> &HashMap<String, Value> {
        match &self.candle_content {
            Some(c) => &c.metadata,
            None => &self.fallback_metadata,
        }
    }

    /// Get all tensor infos.
    ///
    /// ⚠️ Errors when candle refused the file, rather than returning an empty
    /// map. Candle's `TensorInfo` cannot represent an IQ-quantized tensor, so
    /// "no tensors" and "tensors this type cannot describe" would be the same
    /// value -- and the caller uses this to decide which weights to load.
    pub fn tensor_infos(&self) -> Result<&HashMap<String, TensorInfo>> {
        match &self.candle_content {
            Some(c) => Ok(&c.tensor_infos),
            None => bail!(
                "this GGUF's tensor directory cannot be represented: {}. Metadata IS available: the tokenizer, architecture and hyperparameters all read normally; only tensor loading is refused. This is analysed in docs/superpowers/specs/2026-08-14-gguf-metadata-chat-template-design.md, under the heading naming `GgmlDType::from_u32` -- the IQ codebook family is rejected by candle AND by fuel, so it is not a missing table entry. Read that before re-deriving it.",
                self.candle_refusal
                    .as_deref()
                    .unwrap_or("candle refused the file")
            ),
        }
    }

    /// Rebuild the checkpoint's own tokenizer from GGUF metadata.
    ///
    /// # Which `tokenizer.ggml.*` keys this reads, and which it deliberately does not
    ///
    /// ```text
    /// tokens             READ      the vocabulary
    /// merges             READ      REQUIRED -- see below; absence is a refusal
    /// model, pre         READ      select and gate the rebuild path
    /// token_type         READ      special-token registration
    /// bos/eos_token_id   READ      post-processor + special tokens
    /// add_bos_token      READ      whether to prepend BOS
    /// add_eos_token      READ      whether to append EOS
    ///
    /// scores             NOT READ  deliberately -- see "merges is required" below
    /// add_space_prefix   NOT READ  deliberately -- see below
    /// ```
    ///
    /// ⚠️ **A superseded doc block used to sit above this one listing `scores` as
    /// an "expected metadata field".** It was left behind when this function was
    /// rewritten, so the first thing a reader saw claimed a key was read that is
    /// deliberately refused — the correct account was thirty lines further down
    /// and lost to whichever came first.
    ///
    /// ## `add_space_prefix` is not read, and a fix could not be verified here
    ///
    /// Measured 2026-09-05 across the local corpus: 11 files declare it, and it
    /// **varies** (10 `false`, 1 `true`). That variation is not usable evidence.
    /// Every file where it could matter is either `gpt2` — byte-level BPE, where
    /// a SentencePiece space prefix is not a concept — or an architecture no
    /// rebuild path accepts (`gemma4`, `t5`). **The 5 `llama`-model files, the
    /// only ones where it would apply, do not declare it at all.**
    ///
    /// So implementing it would be a change no fixture in this corpus can
    /// distinguish from doing nothing. **Declined for that reason, recorded here
    /// rather than in a planning document, because a decline in a roadmap is
    /// invisible to the next person reading this code and wondering.**
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
    /// 4 of 5 `llama`-model files.
    ///
    /// ⚠️ **THE SECOND SHAPE IS NO LONGER UNIVERSALLY REFUSED, AND THE OLD TEXT
    /// HERE STATED A REQUIREMENT THAT IS NOT ONE.** It said rebuilding needed
    /// "SPM's scored bigram-merge algorithm". It needs the token list: for
    /// **llama.cpp's SentencePiece export specifically**, enumerating every
    /// split of a token into two vocabulary tokens reproduces the declared
    /// 61249-entry list exactly, reading no scores at all.
    ///
    /// ⚠️ **That is a property of THAT EXPORT FORMAT, not of vocabularies.** On
    /// byte-level BPE the same enumeration is ~2.2x too large — a learned merge
    /// list is a trained SEQUENCE, and most splits were never merges. Measured
    /// over six vocabularies in `derive_merges`, where the numbers are.
    ///
    /// What is still unknown is the ORDER, which BPE is sensitive to. Token-id
    /// order is empirically sufficient on the one vocabulary with an oracle and
    /// is not proven in general, so `spm_derivation_warrant` is an allowlist:
    /// vocabularies checked against a checkpoint that carries real merges are
    /// rebuilt, and the rest are refused with a message naming the digest a
    /// future oracle would have to match.
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
    /// which is far worse to debug than a refusal to load. **The Unigram path
    /// stays rejected for the reason above — it is a different algorithm on the
    /// same numbers.** Recovering the merges is a different thing entirely, and
    /// is what `derive_merges` does.
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

        let VocabAndMerges {
            tokens,
            vocab,
            merges: declared_merges,
        } = self.vocab_and_optional_merges()?;

        // A merge-less SentencePiece checkpoint can still be rebuilt IF this
        // exact vocabulary has been checked against a file that carries real
        // merges. See `derive_merges` for the algorithm and
        // `spm_derivation_warrant` for why it is an allowlist and not a rule.
        let merges = match declared_merges {
            Some(m) => m,
            None => {
                let digest = Self::vocab_sha256(&tokens);
                if Self::spm_derivation_warrant(&digest).is_some() {
                    Self::derive_merges(&tokens)
                } else {
                    bail!(
                        "GGUF has no tokenizer.ggml.merges, and this vocabulary's derived merges have not been checked against an oracle. Merges CAN sometimes be recovered from the token list alone by enumerating every split into two vocabulary tokens: for the 32000-token Llama vocabulary that reproduces a real list exactly (61249, 0 extra, 0 missing). It is not a general property -- measured over six vocabularies, the enumeration never MISSES a declared merge but on byte-level BPE it is ~2.2x too large, so it is exact only for llama.cpp's SentencePiece export. And BPE is ORDER-sensitive, where token-id order is only EMPIRICALLY sufficient on that one vocabulary. To retire this refusal, find a checkpoint with this same token list (sha256 {digest}) that DOES declare merges, compare the derived list to it, and add the digest to `spm_derivation_warrant`. Building a Unigram from tokenizer.ggml.scores instead was measured and does NOT reproduce the segmentation (29 ids against the reference's 22: `capital` came out as c+ap+it+al), so that is not the way round it."
                    );
                }
            }
        };

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
        // `tokenizer.ggml.add_eos_token` defaults to FALSE, unlike its BOS
        // sibling: llama.cpp appends BOS by default for a `llama` tokenizer and
        // does not append EOS.
        //
        // ⚠️ It was read NOWHERE until now, and the corpus is why that looked
        // correct. Of the six files that declare it, FIVE say `false` — and
        // reading nothing produces the same behaviour as reading `false`. The
        // code was ACCIDENTALLY CORRECT on five of six, so a test written
        // against those five would have passed against code that reads nothing
        // at all. The majority value in the population is what hid it.
        let add_eos = self
            .metadata()
            .get("tokenizer.ggml.add_eos_token")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(false);

        let named = |id: Option<u32>| -> Option<(String, u32)> {
            id.and_then(|i| tokens.get(i as usize).map(|t| (t.clone(), i)))
        };
        let prefix = named(bos_id).filter(|_| add_bos);
        let suffix = named(eos_id).filter(|_| add_eos);

        // ⚠️ Built when EITHER is wanted. The previous version nested the whole
        // construction inside `if add_bos`, so a checkpoint asking for EOS and
        // not BOS got NO post-processor at all — there was nowhere for an EOS to
        // go even once the key was read. Reading the key is only half the fix.
        if let Some((template, specials)) = post_processor_spec(prefix, suffix) {
            let processor = TemplateProcessing::builder()
                .try_single(template)
                .map_err(|e| anyhow::anyhow!("building the BOS/EOS post-processor: {e}"))?
                .special_tokens(specials)
                .build()
                .map_err(|e| anyhow::anyhow!("building the BOS/EOS post-processor: {e}"))?;
            tokenizer.with_post_processor(Some(processor));
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

    /// Tokens and vocab, with the DECLARED merges if the file carries any.
    ///
    /// Separate from `vocab_and_merges` because the SPM path can now proceed
    /// without them and the byte-level BPE path still cannot.
    fn vocab_and_optional_merges(&self) -> Result<VocabAndMerges> {
        let tokens = self
            .get_metadata_string_array("tokenizer.ggml.tokens")
            .context("Missing tokenizer.ggml.tokens in GGUF metadata")?;
        let vocab: tokenizers::models::bpe::Vocab = tokens
            .iter()
            .enumerate()
            .map(|(id, t)| (t.clone(), id as u32))
            .collect();
        let merges = match self.get_metadata_string_array("tokenizer.ggml.merges") {
            Some(raw) => Some(
                raw.iter()
                    .map(|m| {
                        m.split_once(' ')
                            .map(|(a, b)| (a.to_string(), b.to_string()))
                            .ok_or_else(|| anyhow::anyhow!("malformed merge entry {m:?}: no space"))
                    })
                    .collect::<Result<_>>()?,
            ),
            None => None,
        };
        Ok(VocabAndMerges {
            tokens,
            vocab,
            merges,
        })
    }

    /// A stable digest of a token list, for keying the derivation allowlist.
    ///
    /// ⚠️ SHA-256 RATHER THAN `DefaultHasher`, WHICH IS THE OBVIOUS CHOICE AND
    /// WOULD BE A LATENT BUG. `DefaultHasher`'s output is explicitly not stable
    /// across Rust releases, and `tests/gguf_corpus_vocab_census.rs` uses it
    /// correctly — it groups files within a single run and never persists a
    /// value. A digest baked into a source constant is the opposite case: a
    /// toolchain bump would silently stop matching, the allowlist entry would
    /// go dead, and the checkpoint would be refused again with no diagnostic.
    ///
    /// Length-prefixed so that no two distinct token lists can hash alike by
    /// concatenation — `["ab", "c"]` and `["a", "bc"]` must differ.
    fn vocab_sha256(tokens: &[String]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update((tokens.len() as u64).to_le_bytes());
        for t in tokens {
            h.update((t.len() as u64).to_le_bytes());
            h.update(t.as_bytes());
        }
        format!("{:x}", h.finalize())
    }

    /// Vocabularies whose derived merges have been checked against an ORACLE.
    ///
    /// ⚠️ AN ALLOWLIST, FOR THE SAME REASON `VERIFIED_PRE` IS ONE. The
    /// derivation below is an argument about vocabularies in general; this table
    /// records the ones where that argument has actually been MEASURED against a
    /// checkpoint that carries real merges. Keying on "is a SentencePiece file
    /// without merges" would apply one verified result to every future
    /// checkpoint of that shape, which is the failure `VERIFIED_PRE` exists to
    /// prevent.
    ///
    /// To add an entry you need a file with the SAME token list that DOES carry
    /// merges, so the derived list can be compared to a real one. Without that
    /// there is no oracle and confidence in the algorithm is not a substitute.
    fn spm_derivation_warrant(digest: &str) -> Option<&'static str> {
        match digest {
            LLAMA_SPM_VOCAB_SHA256 => Some(
                "the 32000-token Llama SentencePiece vocabulary. Verified against `tinyllama-1.1b-chat-v1.0.Q4_0.gguf`, whose token list is byte-identical and which carries 61249 real merges: the derived list is SET-IDENTICAL to it (0 extra, 0 missing), and both tokenizers agree on 26 varied inputs across both `add_special_tokens` arms. ⚠️ For comparison with the weaker warrants below, this vocabulary's score-order population splits 61248 adjacent pairs into 29611 INFORMATIVE and 31637 TIES (51.7% ties), and its 15 raw order violations are ALL transitions out of the -1e9 no-rank sentinel, so 0 are genuine. Stated so a reader comparing vocabularies is not weighing a corrected denominator against an uncorrected one. See `llama_spm_agrees_with_the_oracle_through_the_production_path`.",
            ),
            PHI3_VOCAB_SHA256 => Some(
                "Phi-3's 32064-token vocabulary, whose first 32000 ids are byte-identical to the Llama SentencePiece list above. The extra 64 are chat control tokens that split into nothing, so the derived merge list is the SAME 61249 and is SET-IDENTICAL to TinyLlama's declared one (0 extra, 0 missing). Same oracle, same proof, not a weaker one. See `phi3_extends_the_llama_vocabulary_and_inherits_its_oracle`.",
            ),
            BAICHUAN_VOCAB_SHA256 => Some(
                "Baichuan's 64000-token vocabulary. ⚠️ WEAKER WARRANT THAN THE TWO ABOVE, DELIBERATELY: no corpus file shares or extends this vocabulary, so NO DECLARED MERGE LIST EXISTS to compare against and nothing here is an oracle. What stands in its place: the derived merge order is checked against the ordering llama.cpp's own converter wrote into `tokenizer.ggml.scores` -- a separately-authored fact, not one this crate produced -- and contradicts it 0 times. ⚠️ STATE THE POPULATION HONESTLY: there are 54803 adjacent pairs, but 24347 of them are TIES -- equal scores, which no ordering can contradict -- so only 30456 pairs can discriminate anything and those are the evidence. Counting all 54803 would inflate it by ~80% with members structurally incapable of falsifying the claim. The comparator is FORCED rather than trusted: perturbing the first id in the measured sequence raises the count by exactly one, so 0 is a reading and not a silence. This is PROOF AGAINST GROSS FAILURE, NOT PROOF OF CORRECTNESS -- a merge list in the wrong ORDER could agree with those scores and still tokenize differently, and no measurement here would see it. See `baichuan_derived_order_agrees_with_the_converters_own_scores`.",
            ),
            _ => None,
        }
    }

    /// Every split of every vocab token into two vocab tokens, in token-id order.
    ///
    /// # It yields a SUPERSET of the declared merges, exact only for llama.cpp's SPM
    ///
    /// This reads **no scores at all**. Measured 2026-09-06 across every corpus
    /// vocabulary that declares a merge list, so the claim ranges over more than
    /// the one file this is used on:
    ///
    /// ```text
    /// file                          tokens  declared   derived  extra  MISSED
    /// tinyllama Q4_0    (llama/SPM)  32000     61249     61249      0       0
    /// SmolLM2 Q4_0      (gpt2)       49152     48900    107441  58541       0
    /// ggml-vocab-gpt-neox (gpt2)     50432     50009    117101  67092       0
    /// ggml-vocab-qwen2    (gpt2)    151936    151387    294166 142779       0
    /// ggml-vocab-starcoder(gpt2)     49152     48872    107610  58738       0
    /// ggml-vocab-falcon   (gpt2)     65024     64784    146829  82045       0
    /// ```
    ///
    /// ⚠️ **AN EARLIER VERSION OF THIS COMMENT SAID "a merge IS a split, so the
    /// merge SET is a function of the vocabulary". THAT IS FALSE IN GENERAL** and
    /// the table is why: on byte-level BPE the derivation is roughly 2.2x too
    /// large. A learned BPE merge list is a SEQUENCE that was trained; many
    /// splits into two vocab tokens were never merges. The structural-sounding
    /// argument happened to hold for **one export format** and was stated as a
    /// property of vocabularies.
    ///
    /// **What DOES hold across all six: the derivation never MISSES a declared
    /// merge — the `MISSED` column is 0 everywhere.** It over-generates, and it
    /// over-generates by nothing at all only for llama.cpp's SentencePiece
    /// export, which emits every valid split (hence 61249 for a 32000-token
    /// vocabulary, yielding only 29612 distinct products).
    ///
    /// ⚠️ So `spm_derivation_warrant` is NECESSARY, not merely prudent. The SPM
    /// path is already gated on `general.architecture`-style model kind, so a
    /// byte-level vocabulary cannot reach this function today — but the reason
    /// it must not is measured above rather than assumed.
    ///
    /// ## ⚠️ And the ORDER is not the SET either
    ///
    /// BPE merge priority is order-sensitive, so a correct set in the wrong
    /// order is still a wrong tokenizer. Token-id order is not exactly the
    /// declared order — the declared list is sorted non-decreasing by product id
    /// with EXACTLY ONE violation in 61248, Pearson 0.9994 over the first 5000,
    /// and matches this function's output at 1 position out of 61249.
    ///
    /// **It nonetheless produces identical tokenization on every input tried.**
    /// That is a measurement, not a proof: order-sensitivity did not bite on
    /// this vocabulary and nothing here shows it cannot on another.
    fn derive_merges(tokens: &[String]) -> Vec<(String, String)> {
        let index: std::collections::HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let mut out = Vec::new();
        for t in tokens {
            for c in 1..t.len() {
                if !t.is_char_boundary(c) {
                    continue;
                }
                let (a, b) = t.split_at(c);
                if index.contains(a) && index.contains(b) {
                    out.push((a.to_string(), b.to_string()));
                }
            }
        }
        out
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
                r#"No admissible reference exists for it, and NO CORPUS COULD DISCRIMINATE IT. `ggml-vocab-qwen2.gguf` and `ggml-vocab-qwen35.gguf` are 5928681 and 5928682 bytes, differ first at byte 539, and differ in size by exactly one — the length of "qwen35" over "qwen2". They are THE SAME FILE WITH A DIFFERENT LABEL: identical `general.architecture` (qwen2), identical `general.name`, identical token lists. So the string under test is the only difference between the two inputs, and a disagreement count between them is a logical necessity rather than a measurement. The obvious reference, `Qwen/Qwen3-8B`, declares a pre-tokenizer and vocab BYTE-IDENTICAL to `Qwen/Qwen2-7B`, so it is a reference for `qwen2` and not for this name. llama.cpp does define a distinct QWEN35 rule, matching `[\p{L}\p{M}]+` where the qwen2 rule matches `\p{L}+`. HISTORICAL, and kept because it is how this was first established: the 130-case corpus scored them 0 of 130 apart, because the qwen2 normalizer is NFC and composes away the combining marks the difference turns on. That was an empirical claim about one corpus; the byte-level fact above is structural and cannot rot."#,
            ),
            "mpt" => Some(
                r#"Investigated and refused: no reference exists for THIS checkpoint. Its vocab is byte-identical to this corpus's own gpt-neox vocab file (tokens and merges identical), and its own reference is gated on HuggingFace — `mosaicml/mpt-7b` returns HTTP 401. It EXISTS and is gated; an earlier note here said "not found", which was a connector rendering an authorization failure as absence. It scores 0 of 130 against the stand-in, and that is NOT sufficient: vocab identity does not imply RULE identity. Measured on this very corpus, `Qwen/Qwen3-8B` has a vocab and pre-tokenizer byte-identical to `Qwen/Qwen2-7B` while llama.cpp assigns those two `pre` names DIFFERENT rules — so an identical vocab is consistent with a different splitting rule, and only the checkpoint's own `tokenizer.json` settles it. RE-MEASURED 2026-09-03 and the premise HOLDS: `ggml-vocab-mpt.gguf` declares `general.architecture = mpt` and `general.name = mpt`, so it really is an MPT file and the gated repo really is the right one. A GGUF or GGML re-upload would not help — those carry the `pre` NAME, not the rule; only a `tokenizer.json` supplies the pre-tokenizer JSON, and no ungated re-upload carrying one has been found."#,
            ),
            "<absent>" => Some(
                r#"An absent `tokenizer.ggml.pre` is not a name, it is the lack of one, and this table keys on names. The corpus file that omits it (gpt-neox) does verify 0 of 130 against `EleutherAI/gpt-neox-20b`, but keying on absence would apply that one checkpoint's rule to EVERY future GGUF that omits the field — which is precisely the one-rule-for-all-checkpoints failure this table exists to prevent. Re-export the checkpoint with `tokenizer.ggml.pre` set, or add its name here. RE-MEASURED 2026-09-03 and the premise HOLDS: `ggml-vocab-gpt-neox.gguf` declares `general.architecture = gptneox`, `general.name = gpt-neox-20b`, and no `tokenizer.ggml.pre` at all. Note absence is model-dependent — an absent `pre` on a `llama`-model GGUF rebuilds fine, because the SentencePiece path does not consult it; only the byte-level BPE path needs it."#,
            ),
            _ => None,
        }
    }

    /// The `tokenizer.ggml.pre` values this build has verified.
    ///
    /// Public so `gguf_bpe_tokenizer_fidelity` can assert that its fixtures
    /// COVER every one of them. Without that, supplying fewer pairs than there
    /// are entries narrows the gate silently — and a table of seven entries
    /// exercised by one checkpoint is six entries wearing the seventh's
    /// evidence.
    pub fn verified_pre_values() -> &'static [&'static str] {
        Self::VERIFIED_PRE
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
        "llama-bpe",
        "command-r",
        "starcoder",
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
            // meta-llama/Meta-Llama-3-8B, fetched from the NousResearch mirror of
            // the SAME checkpoint (the canonical repo is gated). Verified 0 of 130;
            // the GGUF carries 128256 tokens against the reference model's 128000,
            // the extra 256 being its special tokens, confirmed at the tail.
            "llama-bpe" => Some((
                r##"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"##,
                r##"null"##,
            )),
            // CohereForAI/c4ai-command-r-v01, read from the ungated
            // `mlx-community/c4ai-command-r-v01-4bit` re-upload.
            //
            // THE REFERENCE IS A QUANTIZED RE-UPLOAD, admissible on a property
            // measured in this repo rather than on a judgement: a quantization
            // changes the WEIGHTS and leaves `tokenizer.ggml.tokens` untouched,
            // which is why six SmolLM2 quantizations are ONE vocabulary in
            // `tests/gguf_corpus_vocab_census.rs`. `llama-bpe` above is also a
            // third-party mirror; the two differ only in the variable that
            // measurement shows is irrelevant to the tokenizer.
            //
            // ⚠️ ITS EVIDENCE DIFFERS FROM THE OTHER EIGHT IN ONE RESPECT, STATED
            // RATHER THAN LEFT TO BE ASSUMED EQUAL. Those entries record "0 of
            // 130 cases": the repo's 30 plus 100 randomised strings from a
            // verification run that lives outside the repo. This entry was
            // measured at 0 of 30 -- the in-repo gate, every case, against the
            // reference's own tokenizer. The remaining 100 were NOT re-run,
            // because the generator that produced them is not in the repo and
            // reconstructing it from a seed and a prose description of its
            // alphabet would be a guess wearing a number.
            //
            // PROPERTIES, NOT A GRADE, so a later reader can re-check instead of
            // being told how much to trust it. From
            // `tests/gguf_reference_admissibility.rs`:
            //
            //   prefix identity  OK over ALL 255000 shared ids
            //   merges           253333 identical
            //   tail extras      1000, all special (<|START_OF_TURN_TOKEN|>,
            //                    <|END_OF_TURN_TOKEN|>, <|YES_TOKEN|>, ...)
            //   tokenizer.json   12777405 bytes, byte-size identical to
            //                    CohereLabs' own (gated) 4-bit repo
            //
            // THE GGUF IS THE CORROBORATION AGAINST THE ORIGINAL, and it is
            // better than reading the gated repo would have been. That repo is
            // unreachable even authenticated ("you are not in the authorized
            // list"), but `ggml-vocab-command-r.gguf` was converted from the
            // ORIGINAL checkpoint by llama.cpp -- a third party, a different
            // toolchain, no connection to this mirror. A different tokenizer
            // could not agree with it on 255000 ids and 253333 merges.
            //
            // NOTE the pre-tokenizer is byte-identical to `smollm` and `refact`
            // while the NORMALIZER differs (NFC where those are null). Three
            // `pre` names, one splitting rule, different pipelines -- which is
            // why entries are keyed on the name and not on the rule.
            "command-r" => Some((
                r##"{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}"##,
                r##"{"type":"NFC"}"##,
            )),
            // bigcode/starcoder2-7b, ungated (HTTP 200), verified against that
            // checkpoint's own `tokenizer.json`.
            //
            // ⚠️ THIS ENTRY EXISTS BECAUSE A REFUSAL'S PREMISE WAS WRONG, NOT
            // BECAUSE A NEW REFERENCE APPEARED. `starcoder` was refused with the
            // reasoning "no reference exists for THIS checkpoint; its vocab is
            // byte-identical to ANOTHER MODEL'S, `bigcode/starcoder2-7b`, and
            // vocab identity does not imply rule identity." Every inference in
            // that was sound. The premise was not.
            //
            // `ggml-vocab-starcoder.gguf` declares `general.architecture =
            // starcoder2`. IT IS A STARCODER2 FILE. So starcoder2-7b is not a
            // coincidental twin standing in for an unreachable original -- it is
            // the family the file was converted from, and the gated repo the
            // refusal was waiting on (bigcode/starcoder, StarCoder-1) is the
            // WRONG REPO.
            //
            // Measured, and the discriminator is the mismatched token NAMES
            // rather than the count -- a bare "48697 mismatches" would have read
            // as "wrong reference, refusal stands":
            //
            //   vs bigcode/starcoder2-7b   prefix identity over ALL 49152 ids,
            //                              48872 merges identical, NO tail extras
            //   vs StarCoder-1             48697 token / 47700 merge mismatches;
            //     (TheBloke/starcoder-GPTQ, id 5 gguf `<repo_name>` (SC2) against
            //      an ungated re-upload)   reference `<filename>` (SC1)
            //
            // RECORDED AS A PROPERTY, NOT AS A CHECKPOINT CLAIM: exact vocab and
            // merge identity does not pin WHICH starcoder2 (3b/7b/15b share a
            // tokenizer). What it establishes is that starcoder2-7b's
            // `tokenizer.json` IS this GGUF's tokenizer byte for byte, which is
            // what verifying a splitting rule needs.
            //
            // Its declared pre-tokenizer is byte-identical to `smollm`,
            // `refact` and `command-r` above; the normalizer is null, as
            // smollm's and refact's are, where command-r's is NFC. FOUR names,
            // one splitting rule, two pipelines -- which is why the table keys
            // on the name.
            "starcoder" => Some((
                r##"{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}"##,
                r##"null"##,
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
        // Zero-copy access is our parser's alone: the offsets come from ITS
        // header, so a file it did not read has no offsets to slice by.
        let header = self.require_header()?;

        // Find tensor index and info
        let (tensor_idx, tensor_info) = header
            .tensor_infos
            .iter()
            .enumerate()
            .find(|(_, ti)| ti.name == name)
            .with_context(|| format!("Tensor '{}' not found in GGUF file", name))?;

        // Calculate start offset (absolute position in file)
        let start = (header.tensor_data_offset + tensor_info.offset) as usize;

        // Calculate end offset:
        // If there's a next tensor, use its offset
        // Otherwise, use the file size
        let end = if tensor_idx + 1 < header.tensor_infos.len() {
            let next_tensor = &header.tensor_infos[tensor_idx + 1];
            (header.tensor_data_offset + next_tensor.offset) as usize
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
        // Delegate to Candle's proven tensor loading logic.
        match &self.candle_content {
            Some(c) => c.tensor(reader, name, device),
            None => Err(candlelight::core::Error::Msg(format!(
                "cannot load tensor {name:?}: {}",
                self.candle_refusal
                    .as_deref()
                    .unwrap_or("candle refused this file")
            ))),
        }
    }
}

/// The post-processor's template and special tokens, for whichever of BOS/EOS
/// the checkpoint asked for. `None` when it asked for neither.
///
/// ⚠️ PURE AND SEPARATE BECAUSE THE CORPUS CANNOT EXERCISE THE EOS ARM. Exactly
/// one file in the corpus declares `tokenizer.ggml.add_eos_token = true`
/// (`ggml-vocab-nomic-bert-moe.gguf`), and it declares
/// `tokenizer.ggml.model = "t5"` — which neither rebuild path accepts. So every
/// checkpoint we can actually load is one where EOS-appending code and
/// EOS-ignoring code behave identically, and an end-to-end test over the corpus
/// would pass against either. These unit tests are the only thing that can tell
/// them apart.
fn post_processor_spec(
    prefix: Option<(String, u32)>,
    suffix: Option<(String, u32)>,
) -> Option<(String, Vec<(String, u32)>)> {
    let template = match (&prefix, &suffix) {
        (Some((b, _)), Some((e, _))) => format!("{b}:0 $A:0 {e}:0"),
        (Some((b, _)), None) => format!("{b}:0 $A:0"),
        (None, Some((e, _))) => format!("$A:0 {e}:0"),
        (None, None) => return None,
    };
    // A checkpoint may use one token for both. Registering it twice is not an
    // error worth risking in a builder we do not own.
    let mut specials: Vec<(String, u32)> = Vec::new();
    for t in prefix.iter().chain(suffix.iter()) {
        if !specials.iter().any(|(_, id)| *id == t.1) {
            specials.push(t.clone());
        }
    }
    Some((template, specials))
}

#[cfg(test)]
mod post_processor_spec_tests {
    use super::post_processor_spec;

    fn bos() -> Option<(String, u32)> {
        Some(("<s>".to_string(), 1))
    }
    fn eos() -> Option<(String, u32)> {
        Some(("</s>".to_string(), 2))
    }

    /// The pre-existing behaviour, unchanged. Every checkpoint lightbulb can
    /// currently load lands here, so this is the arm the corpus does cover.
    #[test]
    fn bos_only_is_the_previous_template_exactly() {
        let (t, s) = post_processor_spec(bos(), None).expect("BOS alone must build");
        assert_eq!(t, "<s>:0 $A:0");
        assert_eq!(s, vec![("<s>".to_string(), 1)]);
    }

    /// ⚠️ THE ARM THE OLD CODE COULD NOT REACH AT ALL. It nested the whole
    /// construction inside `if add_bos`, so a checkpoint asking for EOS and not
    /// BOS got NO post-processor — there was nowhere for an EOS to go even once
    /// the key was read. Reading `add_eos_token` was only half the fix.
    #[test]
    fn eos_without_bos_still_builds_a_post_processor() {
        let (t, s) = post_processor_spec(None, eos())
            .expect("EOS alone must build -- the old code produced nothing here");
        assert_eq!(t, "$A:0 </s>:0");
        assert_eq!(s, vec![("</s>".to_string(), 2)]);
    }

    #[test]
    fn both_wrap_the_sequence() {
        let (t, s) = post_processor_spec(bos(), eos()).expect("both must build");
        assert_eq!(t, "<s>:0 $A:0 </s>:0");
        assert_eq!(s, vec![("<s>".to_string(), 1), ("</s>".to_string(), 2)]);
    }

    #[test]
    fn neither_builds_nothing() {
        assert!(
            post_processor_spec(None, None).is_none(),
            "a checkpoint wanting neither must get no post-processor, not an empty one"
        );
    }

    /// A checkpoint using one token for both must not register it twice.
    #[test]
    fn a_shared_token_is_registered_once() {
        let same = Some(("<|endoftext|>".to_string(), 0));
        let (t, s) = post_processor_spec(same.clone(), same).expect("must build");
        assert_eq!(t, "<|endoftext|>:0 $A:0 <|endoftext|>:0");
        assert_eq!(
            s,
            vec![("<|endoftext|>".to_string(), 0)],
            "the same id must appear once in the special-token list"
        );
    }
}

/// Name a metadata value's kind, so an error can say what was found rather than
/// only what was wanted.
fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::U8(_) => "u8",
        Value::I8(_) => "i8",
        Value::U16(_) => "u16",
        Value::I16(_) => "i16",
        Value::U32(_) => "u32",
        Value::I32(_) => "i32",
        Value::U64(_) => "u64",
        Value::I64(_) => "i64",
        Value::F32(_) => "f32",
        Value::F64(_) => "f64",
        Value::Bool(_) => "bool",
        Value::String(_) => "string",
        Value::Array(_) => "ARRAY",
    }
}

/// Read an integer-valued metadata key, distinguishing ABSENT from WRONG TYPE.
///
/// ⚠️ THE PREVIOUS FORM COLLAPSED THOSE TWO STATES into one message, at four
/// sites across two files:
///
/// ```text
/// _ => bail!("Missing or invalid metadata key: {key}")
/// ```
///
/// A key that is **present but the wrong type** is reported as missing, which
/// sends a reader looking for a truncated file. That is the same defect this
/// subsystem has now produced three times — #57's `llama.embedding_length`
/// naming a key when the cause was the architecture, and #61's version message
/// naming `expected 3` when the parser accepted 2 or 3. **A loud, specific,
/// correct-looking message that names the wrong cause.**
///
/// # ⚠️ The case that is not hypothetical: a per-layer ARRAY
///
/// `ggml-vocab-gemma-4.gguf` declares
///
/// ```text
/// gemma4.attention.head_count_kv = [8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 2, ...]
/// ```
///
/// **a 30-element per-layer array, not a scalar** — GQA grouping varies by
/// layer, aligned with that file's `attention.sliding_window_pattern`. Measured
/// by MLMF on their corpus and confirmed here.
///
/// So a reader holding `num_key_value_heads: usize` cannot represent it **at
/// all** — not wrong by a factor, *unrepresentable*. This does not fix that;
/// it makes the refusal say so instead of claiming the key is absent.
///
/// Latent today: `require_llama_architecture` refuses gemma4 before any of these
/// reads run.
pub(crate) fn metadata_u64(metadata: &HashMap<String, Value>, key: &str) -> Result<u64> {
    match metadata.get(key) {
        Some(Value::U64(v)) => Ok(*v),
        Some(Value::U32(v)) => Ok(u64::from(*v)),
        Some(Value::Array(a)) => bail!(
            "`{key}` is declared as an ARRAY of {} element(s), not a single integer. Some \
             checkpoints vary this per layer -- gemma4 declares \
             `attention.head_count_kv` as a 30-element array whose entries alternate with \
             its sliding-window pattern. A scalar cannot represent that, so this is a \
             limit of this reader rather than a malformed file.",
            a.len()
        ),
        Some(other) => bail!(
            "`{key}` is declared as {}, not an integer. The key is PRESENT -- this is a \
             type mismatch, not a missing key.",
            value_kind(other)
        ),
        None => bail!("`{key}` is not declared by this GGUF."),
    }
}

/// Read a float-valued metadata key, distinguishing ABSENT from WRONG TYPE.
///
/// Same rationale as [`metadata_u64`].
pub(crate) fn metadata_f32(metadata: &HashMap<String, Value>, key: &str) -> Result<f32> {
    match metadata.get(key) {
        Some(Value::F32(v)) => Ok(*v),
        Some(other) => bail!(
            "`{key}` is declared as {}, not an f32. The key is PRESENT -- this is a type \
             mismatch, not a missing key.",
            value_kind(other)
        ),
        None => bail!("`{key}` is not declared by this GGUF."),
    }
}

/// Refuse a GGUF whose declared architecture this project's loaders cannot read.
///
/// ⚠️ REFUSE ON THE DECLARATION, NOT ON A MISSING KEY. Every GGUF declares
/// `general.architecture` — 30 of 30 in the local corpus — and both GGUF config
/// readers ignored it, hardcoding the `llama.` prefix at seventeen literals with
/// no fallback. A qwen2 checkpoint declares `qwen2.embedding_length`, so it died
/// with:
///
/// ```text
/// Missing or invalid metadata key: llama.embedding_length
/// ```
///
/// which sends a reader hunting for a corrupt GGUF. **The refusal was loud and
/// specific, which is exactly what made it read as a considered decision rather
/// than an oversight** — the same shape as `if version != GGUF_VERSION`.
///
/// # Why this does not substitute the prefix
///
/// The `llama::Config` and the tensor mapping built downstream are llama-shaped.
/// Reading `qwen2.*` into them would trade a misleading error for a silently
/// wrong model, which is worse. Widening support needs a loadable non-llama
/// checkpoint, and the corpus does not have one — every file it holds with
/// tensors declares `llama`.
///
/// # ⚠️ TWO ATTENTION-GEOMETRY KEYS GO UNREAD, AND THIS REFUSAL IS WHY THAT IS SAFE
///
/// Whoever lifts this refusal must read them, because nothing else will notice.
/// Measured 2026-09-06 over the local corpus; re-derivable by walking the KV
/// headers for `<arch>.rope.dimension_count` and `<arch>.attention.key_length`.
///
/// **`<arch>.rope.dimension_count`** — declared by 20 files, read at zero sites.
/// RoPE is applied to this many dimensions, which is **not always the whole
/// head**:
///
/// ```text
/// gptneox   head_dim 96, dimension_count 24   = 0.25x  <- PARTIAL RoPE
/// every other file                            = head_dim exactly
/// ```
///
/// A reader assuming the full head rotates 96 dimensions where the checkpoint
/// says 24. That produces **wrong numbers, not an error** — the classic silent
/// case, and the same family as the `f16`/`bf16` swap `parse_dtype` guards.
///
/// **`<arch>.attention.key_length`** — declared by gemma4 and read at zero sites.
/// It gives head_dim **directly**, and where it is present `embedding_length /
/// head_count` is the wrong formula:
///
/// ```text
/// gemma4   key_length 512, but 2816 / 16 = 176   <- a 2.9x error, silently
/// ```
///
/// (That one caught me while measuring this: my first pass reported gemma4 as a
/// second partial-RoPE case. It is not — my *formula* was wrong, not the file.)
///
/// **Why this is currently harmless, stated as a scope rather than a reassurance:
/// 16 llama files declare these keys and ZERO have a `key_length` or a
/// `dimension_count` that differs from `embedding_length / head_count`.** So the
/// assumption is exactly right for every architecture this loader accepts —
/// *because* it accepts only `llama`.
///
/// ⚠️ **The hazard is that widening support looks like a prefix change.** Someone
/// replacing `llama.` with the declared architecture gets a config that loads and
/// a model that is quietly wrong on any gptneox-family checkpoint. The prefix is
/// the visible half; these two keys are not.
///
/// ## ⚠️ And the fix is not one line: FIVE production sites
///
/// `head_dim = hidden_size / num_heads` is computed independently at five
/// production sites, including `parallel_model_manager.rs:424` on the live
/// serving path:
///
/// ```text
/// PRODUCTION
///   model/awq_qwen3.rs:207                  fn new
///   model/custom_attention.rs:207           fn new
///   model/custom_attention.rs:274           fn from_gguf
///   model/custom_transformer.rs:391         fn from_gguf  (positional)
///   model/parallel_model_manager.rs:424     fn load_gguf  <- the live loader
///
/// UNDER #[cfg(test)] -- fixture arithmetic, not a checkpoint read
///   model/custom_attention.rs:1134          test_attention_dimensions
///   model/custom_transformer_block.rs:442   test_batched_transformer_block_shapes
///   model/custom_transformer_block.rs:510   ..._single_token
///   model/custom_transformer_block.rs:594   ..._dimension_validation
/// ```
///
/// ⚠️ **AN EARLIER VERSION OF THIS BLOCK SAID "NINE LIVE SITES" AND COUNTED THE
/// FOUR TEST FUNCTIONS AMONG THEM.** The grep that produced it excluded comments
/// and nothing else, so "live" was a word in the sentence rather than a measured
/// property — and it was the load-bearing word, since the whole point is how much
/// production code a fix has to reach. Corrected by resolving each line to its
/// enclosing `fn` and checking for a `#[cfg(test)]` above it.
///
/// **So this is not "read `key_length` instead of dividing" at one accessor.**
/// MLMF hit the same defect in their `config.rs` and theirs is a single
/// `head_dim()` method; ours is five, and a fix that misses one is silent
/// everywhere that site is used.
///
/// ## ⚠️ AND A SINGLE `head_dim` IS THE WRONG SHAPE FOR SOME ARCHITECTURES
///
/// gemma4 declares **two** attention geometries in one checkpoint — full layers
/// and sliding-window layers, with different dimensions:
///
/// ```text
/// gemma4.attention.key_length       512    gemma4.rope.dimension_count      512
/// gemma4.attention.key_length_swa   256    gemma4.rope.dimension_count_swa  256
/// ```
///
/// **A reader holding one `head_dim` cannot represent that model however it
/// derives the value** — not by division, and not by reading `key_length`
/// either. The shape is wrong, not just the arithmetic. Recorded because
/// "read the declared key" is the obvious remedy and it is insufficient here.
///
/// # One implementation because there are two callers
///
/// `loaders::load_gguf_llama` and
/// `model::parallel_model_manager::ParallelModelManager::load_gguf` both read
/// GGUF config, and only the second is reachable — `load_gguf_llama`'s own doc
/// comment says so. A check written into the first alone would pass its tests
/// and change nothing that runs, which is the failure that function's doc
/// comment warns about in as many words: *a correct fix applied to the wrong
/// caller is indistinguishable from a wrong fix.*
pub(crate) fn require_llama_architecture(metadata: &HashMap<String, Value>) -> Result<()> {
    let architecture = match metadata.get("general.architecture") {
        Some(Value::String(s)) => s.clone(),
        _ => bail!(
            "this GGUF declares no `general.architecture`, so the architecture cannot be \
             checked before reading llama-specific keys. Every GGUF in the reference corpus \
             declares it; a file without it is malformed or truncated."
        ),
    };
    if architecture != "llama" {
        bail!(
            "this GGUF declares `general.architecture = {architecture:?}`; this loader reads \
             `llama` only. Its hyperparameters are under the `{architecture}.` prefix, not \
             `llama.`, and the config and tensor mapping built here are llama-shaped, so \
             reading them would produce a wrong model rather than a missing key."
        );
    }
    Ok(())
}

/// Proof that our metadata reader agrees with candle's, on every corpus file
/// both can read.
///
/// ⚠️ THIS MODULE IS THE PRICE OF `metadata_from_header` EXISTING. The August
/// design spec declined to write a second header reader, and its reason has not
/// expired: *"a hand-rolled partial parser is a second implementation of the
/// header layout -- the one nobody reviews."*
///
/// The premise that justified declining DID expire — it said "our checkpoints
/// do not hit the case", and three corpus files now hit it — but the objection
/// is separate from the premise and survives it. A second implementation is
/// dangerous because it DIVERGES, and divergence is detectable on the OVERLAP:
/// 27 of 30 corpus files are read by both. So the answer is proof, not care.
#[cfg(test)]
mod reader_agreement_tests {
    use super::*;

    /// Structural equality through derived `Debug`, because candle's `Value`
    /// derives only `Debug, Clone` — there is no `PartialEq` to call.
    ///
    /// Debug is the right comparison rather than a weaker substitute: it is
    /// derived, so it prints the variant and its payload, and two values with
    /// the same Debug string are the same value. It also makes NaN compare
    /// EQUAL to NaN, which `==` would not — and structural identity is what
    /// this test is about, not IEEE semantics.
    fn same(a: &Value, b: &Value) -> bool {
        format!("{a:?}") == format!("{b:?}")
    }

    /// Every way two metadata maps can disagree, as reader-facing lines.
    ///
    /// ⚠️ BOTH DIRECTIONS ARE CHECKED ON PURPOSE. Comparing only over candle s
    /// keys would pass a reader that INVENTS keys candle never saw; comparing
    /// only over ours would pass a reader that silently DROPS them. The two
    /// failure modes are opposite and neither sweep alone sees both.
    fn disagreements(
        name: &str,
        ours: &HashMap<String, Value>,
        candle: &HashMap<String, Value>,
    ) -> Vec<String> {
        let mut out = Vec::new();
        for (k, cv) in candle {
            match ours.get(k) {
                None => out.push(format!("{name}: {k:?} MISSING from ours")),
                Some(ov) if !same(ov, cv) => out.push(format!(
                    "{name}: {k:?} differs\n     ours: {ov:?}\n   candle: {cv:?}"
                )),
                Some(_) => {}
            }
        }
        for k in ours.keys() {
            if !candle.contains_key(k) {
                out.push(format!("{name}: {k:?} EXTRA in ours"));
            }
        }
        out
    }

    fn corpus_files() -> Vec<std::path::PathBuf> {
        let Some(root) = std::env::var_os("LIGHTBULB_GGUF_CORPUS") else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut stack = vec![std::path::PathBuf::from(root)];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "gguf") {
                    out.push(p);
                }
            }
        }
        out.sort();
        out
    }

    /// What one pass over the corpus found.
    struct Sweep {
        compared: usize,
        skipped: usize,
        mismatches: Vec<String>,
    }

    /// Read every file with both parsers and collect where they disagree.
    ///
    /// `skipped` counts files only ONE reader could parse -- the three IQ
    /// checkpoints candle refuses, and the one GGUF v1 file ours refuses. They
    /// are the reason this code exists and they have no second opinion, so they
    /// cannot be compared and must not be counted as agreement.
    ///
    /// ⚠️ ALL FOUR ARE NOW READABLE THROUGH `Content::read`, AND STILL SKIPPED
    /// HERE. Readable and comparable are different properties: a file one reader
    /// refuses has a metadata map from the other and nothing to check it
    /// against. Counting them as agreement would let the corpus grow while the
    /// evidence stayed the same size.
    ///
    /// ⚠️ THE COMPARED COUNT IS 26 OF 30 AND PLANNED WORK DOES NOT RAISE IT.
    /// Letting candle serve the GGUF v1 file -- the next change to this area --
    /// gives CANDLE a reading and leaves OURS with none, so the overlap is
    /// unchanged. Stated because a figure that stays put for a non-obvious
    /// reason reads as stale: the next reader sees v1 land, expects 27,
    /// measures 26, and goes hunting a regression that is not there.
    /// Only implementing v1 widths in OUR parser would move it.
    fn sweep(files: &[std::path::PathBuf]) -> Sweep {
        let mut out = Sweep {
            compared: 0,
            skipped: 0,
            mismatches: Vec::new(),
        };
        for path in files {
            let Ok(content) = Content::read(path) else {
                out.skipped += 1;
                continue;
            };
            // A file needs BOTH readings to be comparable. Either one alone is
            // exactly the case this code exists for, and has no second opinion.
            let (Some(candle), Some(header)) =
                (content.candle_content.as_ref(), content.header.as_ref())
            else {
                out.skipped += 1;
                continue;
            };
            let ours = metadata_from_header(header);
            out.compared += 1;
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            out.mismatches
                .extend(disagreements(&name, &ours, &candle.metadata));
        }
        out
    }

    /// THE DIFFERENTIAL TEST. Both readers, every file both can read, key for key.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn our_metadata_agrees_with_candles_on_every_file_both_can_read() {
        let files = corpus_files();
        assert!(
            !files.is_empty(),
            "no .gguf files under LIGHTBULB_GGUF_CORPUS. An empty corpus makes this test pass while comparing nothing, which is indistinguishable from agreement."
        );
        eprintln!(
            "  SUBJECT: LIGHTBULB_GGUF_CORPUS={:?}  ({} .gguf files)",
            std::env::var("LIGHTBULB_GGUF_CORPUS").unwrap_or_default(),
            files.len()
        );

        let Sweep {
            compared,
            skipped,
            mismatches,
        } = sweep(&files);

        // ⚠️ POSITIVE CONTROL. Without it, a corpus where every file fails to
        // open reports zero mismatches, and zero mismatches reads as agreement.
        assert!(
            compared >= 2,
            "compared {compared} files (skipped {skipped}). This test proves nothing below two, and the overlap between the two readers is its whole purpose."
        );
        eprintln!(
            "  compared {compared} files, skipped {skipped}, mismatches {}",
            mismatches.len()
        );
        assert!(
            mismatches.is_empty(),
            "the two metadata readers DISAGREE on {} entries:\n  {}",
            mismatches.len(),
            mismatches.join("\n  ")
        );
    }

    /// ⚠️ BORN-RED ARM. Without this the agreement above is unfalsifiable: a
    /// comparator returning `true` for everything produces the same passing
    /// output as a correct one.
    #[test]
    fn the_comparator_detects_a_single_altered_field() {
        let a = Value::U32(4096);
        assert!(
            same(&a, &Value::U32(4096)),
            "identical values must compare equal"
        );
        assert!(
            !same(&a, &Value::U32(4097)),
            "one changed digit must be detected"
        );
        assert!(
            !same(&a, &Value::U64(4096)),
            "same number, different WIDTH must be detected"
        );
        assert!(
            !same(
                &Value::String("llama".into()),
                &Value::String("llama ".into())
            ),
            "a trailing space must be detected"
        );
        assert!(
            !same(
                &Value::Array(vec![Value::U32(1), Value::U32(2)]),
                &Value::Array(vec![Value::U32(1), Value::U32(3)])
            ),
            "a difference INSIDE an array must be detected -- vocabularies are arrays"
        );
    }

    /// The refusal message's citation still resolves.
    ///
    /// ⚠️ A REFERENCE FROM CODE INTO A DOC ROTS SILENTLY. `tensor_infos()` tells
    /// the reader where the IQ-dtype analysis lives, so that a refusal points at
    /// its own recorded decision instead of sending them to re-derive it — which
    /// is what happened on 2026-09-06, when that spec already held every answer
    /// and nothing pointed at it from the symptom.
    ///
    /// The citation names a HEADING rather than a line number on purpose: PR #71
    /// existed because four line references in a shipped rustdoc had drifted by
    /// exactly twenty.
    #[test]
    fn the_refusal_citation_resolves() {
        let spec = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("docs/superpowers/specs/2026-08-14-gguf-metadata-chat-template-design.md");
        let text = std::fs::read_to_string(&spec).unwrap_or_else(|e| {
            panic!(
                "the refusal message cites {}, which cannot be read: {e}",
                spec.display()
            )
        });
        assert!(
            text.contains("GgmlDType::from_u32"),
            "the refusal cites this spec for the heading naming `GgmlDType::from_u32`, and that anchor is no longer in the file. Either restore it or change the citation; a refusal pointing at nothing is worse than one pointing nowhere."
        );
    }

    /// The conversion is total: every one of the GGUF spec's thirteen metadata
    /// types converts into candle's shape with its value intact.
    #[test]
    fn every_metadata_type_converts_with_its_value_intact() {
        use parser::MetadataValue as M;
        let cases: Vec<(M, Value)> = vec![
            (M::UInt8(7), Value::U8(7)),
            (M::Int8(-7), Value::I8(-7)),
            (M::UInt16(700), Value::U16(700)),
            (M::Int16(-700), Value::I16(-700)),
            (M::UInt32(70000), Value::U32(70000)),
            (M::Int32(-70000), Value::I32(-70000)),
            (M::Float32(1.5), Value::F32(1.5)),
            (M::Bool(true), Value::Bool(true)),
            (M::String("llama".into()), Value::String("llama".into())),
            (M::UInt64(1 << 40), Value::U64(1 << 40)),
            (M::Int64(-(1 << 40)), Value::I64(-(1 << 40))),
            (M::Float64(2.5), Value::F64(2.5)),
            (
                M::Array(vec![M::UInt32(1), M::String("x".into())]),
                Value::Array(vec![Value::U32(1), Value::String("x".into())]),
            ),
        ];
        assert_eq!(cases.len(), 13, "the GGUF spec has thirteen metadata types");
        for (ours, expected) in &cases {
            let got = to_candle_value(ours);
            assert!(
                same(&got, expected),
                "converting {ours:?} produced {got:?}, expected {expected:?}"
            );
        }
    }
}

#[cfg(test)]
mod metadata_accessor_tests {
    use super::{Value, metadata_f32, metadata_u64};
    use std::collections::HashMap;

    fn with(key: &str, v: Value) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert(key.to_string(), v);
        m
    }

    /// The control: the accepted types still read, or every assertion below is
    /// satisfied by a function that refuses everything.
    #[test]
    fn the_accepted_types_still_read() {
        assert_eq!(metadata_u64(&with("k", Value::U32(7)), "k").unwrap(), 7);
        assert_eq!(metadata_u64(&with("k", Value::U64(9)), "k").unwrap(), 9);
        assert!(
            (metadata_f32(&with("k", Value::F32(1.5)), "k").unwrap() - 1.5).abs() < f32::EPSILON
        );
    }

    /// ABSENT and WRONG TYPE were one message. They are now two, and the
    /// wrong-type one must say the key is PRESENT — that is the whole fix.
    #[test]
    fn absent_and_wrong_type_are_distinguishable() {
        let absent = metadata_u64(&HashMap::new(), "llama.block_count")
            .expect_err("an absent key must fail")
            .to_string();
        let wrong = metadata_u64(
            &with("llama.block_count", Value::String("22".into())),
            "llama.block_count",
        )
        .expect_err("a string where an integer is wanted must fail")
        .to_string();

        assert!(
            absent.contains("not declared"),
            "an absent key must say so: {absent}"
        );
        assert!(
            wrong.contains("PRESENT") && wrong.contains("string"),
            "a wrong-type key must say it is PRESENT and name what was found: {wrong}"
        );
        assert_ne!(
            absent, wrong,
            "⚠️ the two states must not produce the same text — collapsing them is the \
             defect this fix removes, and identical messages would restore it"
        );
        for m in [&absent, &wrong] {
            assert!(
                !m.contains("Missing or invalid metadata key"),
                "the old collapsed wording must not survive: {m}"
            );
        }
    }

    /// ⚠️ The case that is not hypothetical. `ggml-vocab-gemma-4.gguf` declares
    /// `attention.head_count_kv` as a 30-element per-layer array, so a scalar
    /// read of it is UNREPRESENTABLE rather than merely wrong — and the message
    /// has to say that instead of claiming the key is absent.
    #[test]
    fn a_per_layer_array_is_named_as_an_array_with_its_length() {
        let layers: Vec<Value> = (0..30)
            .map(|i| Value::U32(if i % 6 == 5 { 2 } else { 8 }))
            .collect();
        let err = metadata_u64(
            &with("gemma4.attention.head_count_kv", Value::Array(layers)),
            "gemma4.attention.head_count_kv",
        )
        .expect_err("an array where a scalar is wanted must fail")
        .to_string();

        assert!(err.contains("ARRAY"), "must name the kind found: {err}");
        assert!(
            err.contains("30"),
            "must give the element count, so a reader knows it is per-layer: {err}"
        );
        assert!(
            err.contains("limit of this reader"),
            "an array-valued key is a limit of THIS READER, not a malformed file: {err}"
        );
    }
}

#[cfg(test)]
mod architecture_gate_tests {
    use super::{Value, require_llama_architecture};
    use std::collections::HashMap;

    fn declaring(arch: &str) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert(
            "general.architecture".to_string(),
            Value::String(arch.to_string()),
        );
        m
    }

    /// The control: llama must pass, or this is a permanent refusal rather than
    /// a gate.
    #[test]
    fn llama_is_accepted() {
        assert!(require_llama_architecture(&declaring("llama")).is_ok());
    }

    /// ⚠️ The thirteen architectures the LOCAL corpus actually declares, so a
    /// spec rename makes this stale visibly rather than leaving it passing
    /// against invented names.
    #[test]
    fn every_non_llama_architecture_is_refused_by_name() {
        for arch in [
            "qwen2",
            "phi3",
            "falcon",
            "command-r",
            "starcoder2",
            "gemma4",
            "baichuan",
            "refact",
            "mpt",
            "gptneox",
            "gpt2",
            "bert",
            "nomic-bert-moe",
        ] {
            let err = require_llama_architecture(&declaring(arch))
                .expect_err("a non-llama architecture must be refused")
                .to_string();
            assert!(
                err.contains(arch),
                "the refusal must NAME the declared architecture: {err}"
            );
            assert!(
                !err.contains("Missing or invalid metadata key"),
                "the refusal must not report a missing key -- the key is not missing, it is \
                 under the {arch}. prefix, and the key-shaped message IS the defect: {err}"
            );
        }
    }

    /// An absent declaration is a fact about the FILE, and is reported as one
    /// rather than as a missing hyperparameter.
    #[test]
    fn an_absent_declaration_says_so() {
        let err = require_llama_architecture(&HashMap::new())
            .expect_err("no architecture must be refused")
            .to_string();
        assert!(
            err.contains("general.architecture"),
            "the error must name the key that is genuinely absent: {err}"
        );
    }

    /// ⚠️ A DETECTOR FOR A DEFERRAL, WHICH REDDENS WHEN THE DEFERRAL GOES LIVE
    /// AND NAMES ITS OWN REMOVAL.
    ///
    /// `require_llama_architecture`'s doc records that
    /// `<arch>.rope.dimension_count` and `<arch>.attention.key_length` are read
    /// nowhere, and that this is safe **only** because every architecture where
    /// they differ is refused. **That safety is a CONJUNCTION, and a doc comment
    /// cannot enforce a conjunction** — a reader who lifts the refusal has no
    /// reason to open this file.
    ///
    /// So this asserts the conjunction directly:
    ///
    /// ```text
    /// EITHER we still refuse non-llama
    /// OR     src/ reads the geometry keys
    /// ```
    ///
    /// It is DORMANT while the refusal stands, and fires the moment someone
    /// widens architecture support without also reading the geometry — which is
    /// exactly the change that would otherwise produce silently wrong numbers on
    /// a gptneox-family checkpoint.
    ///
    /// **To remove this test:** read the geometry keys at all FIVE production
    /// `hidden_size / num_heads` sites (the other four are under `#[cfg(test)]`
    /// and compute fixture arithmetic), and note that gemma4 needs more than one
    /// head_dim because it declares separate sliding-window geometry. Then delete
    /// this test. It has no other purpose.
    /// Walk `src/` and report whether any Rust file contains one of `needles`.
    ///
    /// ⚠️ SEPARATED FROM THE DECISION so the decision is one line, and given a
    /// POSITIVE CONTROL at its call site — because a scan that finds nothing and
    /// a scan that never ran are indistinguishable. If `CARGO_MANIFEST_DIR` were
    /// wrong or `src/` unreadable, this returns `false`, the detector passes
    /// while the refusal stands, and it is silently blind until the day the
    /// refusal is lifted — when it would fire for the wrong reason.
    fn src_mentions(needles: &[&str]) -> bool {
        // ⚠️ A COMMENT MENTIONING THE CODE IS NOT A READ, AND TREATING IT AS
        // ONE SILENTLY DISARMS THIS GUARD. Measured 2026-09-06 against the
        // merged version: with the architecture refusal LIFTED and the geometry
        // keys still UNREAD, a single line
        //
        //     // DISCHARGED: this used to call metadata.get("...dimension_count")
        //
        // made this test PASS. The failure direction is the dangerous one — the
        // detector goes quiet rather than loud, and a DISCHARGED note quoting
        // retired code is exactly the thing this repo writes on purpose.
        //
        // So a line whose first non-whitespace is `//` (covering `//`, `///` and
        // `//!`) cannot count as a read. That also skips a literal sharing a line
        // with a trailing comment — UNDER-counting reads, which makes the guard
        // fire when it might not need to. That is the safe direction: a false
        // alarm is read; a false silence is not.
        //
        // Found by the Claim Auditor's polarity-blindness finding on their own
        // citation-anchored predicate, applied here rather than taken on trust.
        let is_reading_line = |line: &str| {
            let t = line.trim_start();
            !t.starts_with("//") && needles.iter().any(|n| line.contains(n))
        };
        let mut stack = vec![std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for e in entries.flatten() {
                let path = e.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|x| x == "rs")
                    && let Ok(text) = std::fs::read_to_string(&path)
                    && text.lines().any(is_reading_line)
                {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn the_head_dim_assumption_is_still_guarded_by_the_refusal() {
        // ⚠️ POSITIVE CONTROL FIRST. `general.architecture` is demonstrably read
        // by `require_llama_architecture` a few hundred lines above, so a scan
        // that cannot find it is broken rather than reporting an absence.
        assert!(
            src_mentions(&["general.architecture\")"]),
            "the source scan found no literal for a key this crate demonstrably reads, so              the scan itself is broken. Without this control a broken scan reports              \"geometry keys unread\" forever, which is indistinguishable from the truth              while the refusal stands."
        );

        let refuses_non_llama = require_llama_architecture(&declaring("gptneox")).is_err();
        // ⚠️ THE NEEDLES ARE SPLIT BECAUSE THIS SCAN READS THIS FILE.
        //
        // `src_mentions` walks `src/`, and `src/gguf/mod.rs` is in `src/`. A
        // needle written as one literal here sits on a NON-comment line, so the
        // comment-skip above does not reach it: the scan would find ITSELF,
        // report the geometry keys as read, and `refuses_non_llama ||
        // reads_geometry` would be `_ || true` — passing unconditionally,
        // forever, silently.
        //
        // Measured 2026-09-06 at `fb6af4be`. As merged, the literals were
        // `"rope.dimension_count\")"`, whose ON-DISK bytes carry a backslash
        // that the runtime needle does not:
        //
        //     in source   rope.dimension_count\")
        //     at runtime  rope.dimension_count")
        //
        // A BACKSLASH WAS THE ENTIRE GUARD. Respelling them as raw strings —
        // `r#"rope.dimension_count")"#`, which rustc treats as identical —
        // flipped `reads_geometry` false -> true. Confirmed by running the
        // test with the refusal lifted: it PASSED, when firing is its only
        // purpose. Nobody would flag that respelling in review.
        //
        // ⚠️ The obvious tidier fix REINTRODUCES the bug. A helper reads
        // `needle("rope.dimension_count")` — and the call's own closing `")`
        // completes the needle, so that line self-matches too. Verified.
        //
        // ⚠️ AND THIS COMMENT NAMES THE NEEDLE FOUR TIMES, SO IT IS ITSELF
        // DISARMING TEXT — safe only because #72 taught the scan to skip lines
        // beginning `//`. This fix DEPENDS on that one. Revert #72 and the
        // paragraph explaining the trap becomes the trap. Third time tonight
        // that prose about a hazard turned out to contain it.
        //
        // `concat!` assembles exactly the runtime needle while no contiguous run
        // of source bytes equals it.
        //
        // ⚠️ BE PRECISE ABOUT WHAT THIS BUYS, BECAUSE IT IS LESS THAN IT LOOKS.
        // It does NOT remove the hazard mechanically. Measured: rewriting these
        // two lines as `r#"rope.dimension_count")"#` still kills the guard, split
        // or no split. What it removes is the INNOCENT path — before, an ordinary
        // readability refactor of an escaped literal was enough; now it takes
        // deleting a `concat!` that this comment explicitly says not to delete.
        // The mechanical vulnerability is unchanged; the social one is not.
        //
        // A mechanical fix needs the scan to know which lines are test code, and
        // the parser that would do it is rejected below.
        const ROPE_DIMS: &str = concat!("rope.dimension_", "count\")");
        const KEY_LENGTH: &str = concat!("attention.key_", "length\")");

        // ⚠️ KNOWN AND MEASURED LIMITATION, LEFT OPEN DELIBERATELY: a mention
        // inside `#[cfg(test)]` still counts as a production read. So a future
        // test fixture that names `rope.dimension_count")` would disarm this
        // guard the same silent way. A brace-tracking `#[cfg(test)]` skipper was
        // written and REJECTED — it was defeated by its own explanatory comment,
        // which contained a `}` and closed the test region 48 lines early,
        // reclassifying the whole test module as production. Recorded rather
        // than half-fixed: a parser that mis-parses this file is worse than a
        // documented gap, because it fails in the same silent direction.
        let reads_geometry = src_mentions(&[ROPE_DIMS, KEY_LENGTH]);

        assert!(
            refuses_non_llama || reads_geometry,
            "THE ARCHITECTURE REFUSAL HAS BEEN LIFTED AND THE GEOMETRY KEYS ARE STILL              UNREAD.

             `<arch>.rope.dimension_count` says how many dimensions RoPE covers and it              is NOT always the whole head: gptneox declares 24 against a head_dim of 96.              `<arch>.attention.key_length` gives head_dim directly, and where present              `embedding_length / head_count` is the wrong formula (gemma4: 512 declared,              176 computed).

             Deferring these was safe only while every architecture that differs was              refused. That is no longer true, so a gptneox-family checkpoint now loads              and produces WRONG NUMBERS RATHER THAN AN ERROR.

             See `require_llama_architecture`: five production sites compute head_dim by division,              and gemma4 needs more than one head_dim because it declares separate              sliding-window geometry."
        );
    }
}

/// The merge derivation, and the oracle that licenses it.
#[cfg(test)]
mod spm_derivation_tests {
    use super::*;

    /// The prompt the reference fidelity gate uses, so a divergence here is
    /// comparable to one there.
    const PROMPT: &str = "<|user|>\nName the capital of France.</s>\n<|assistant|>\n";

    /// No corpus needed: the algorithm on a vocabulary small enough to check by
    /// hand. Runs in the ordinary suite, unlike everything else in this module.
    #[test]
    fn derive_merges_emits_every_split_into_two_vocab_tokens() {
        let vocab: Vec<String> = ["a", "b", "ab", "abb", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let got = Content::derive_merges(&vocab);
        // "a","b","c" have no split. "ab" splits as a|b. "abb" splits as a|bb
        // (bb absent) and ab|b (both present) -- so only the second survives.
        assert_eq!(
            got,
            vec![
                ("a".to_string(), "b".to_string()),
                ("ab".to_string(), "b".to_string()),
            ],
            "a merge is a split whose BOTH halves are in the vocabulary"
        );
    }

    /// ⚠️ The derivation must not fabricate a merge from a split that only
    /// exists at a non-character boundary.
    #[test]
    fn derive_merges_does_not_split_inside_a_character() {
        // U+00E9 is two UTF-8 bytes; splitting between them is not a boundary.
        let vocab: Vec<String> = ["\u{e9}", "e"].iter().map(|s| s.to_string()).collect();
        assert!(
            Content::derive_merges(&vocab).is_empty(),
            "no merge can be produced by cutting a multi-byte character in half"
        );
    }

    fn corpus_file(name: &str) -> Option<std::path::PathBuf> {
        let root = std::env::var_os("LIGHTBULB_GGUF_CORPUS")?;
        let mut stack = vec![std::path::PathBuf::from(root)];
        while let Some(dir) = stack.pop() {
            for e in std::fs::read_dir(&dir).ok()?.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().is_some_and(|f| f == name) {
                    return Some(p);
                }
            }
        }
        None
    }

    fn tokens_of(c: &Content) -> Vec<String> {
        c.get_metadata_string_array("tokenizer.ggml.tokens")
            .unwrap_or_default()
    }

    /// THE ORACLE. The derived list against a real one, on the same vocabulary.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn derived_merges_are_set_identical_to_a_real_declared_list() {
        let Some(tiny) = corpus_file("tinyllama-1.1b-chat-v1.0.Q4_0.gguf") else {
            lightbulb_skip();
            return;
        };
        let c = Content::read(&tiny).expect("read tinyllama");
        eprintln!("  SUBJECT: {}", tiny.display());
        let VocabAndMerges { tokens, merges, .. } = c
            .vocab_and_optional_merges()
            .expect("tinyllama vocab and merges");
        let declared = merges.expect("tinyllama declares merges; without them there is no oracle");

        // CONTROL: this really is the allowlisted vocabulary.
        assert_eq!(
            Content::vocab_sha256(&tokens),
            LLAMA_SPM_VOCAB_SHA256,
            "the oracle's vocabulary is not the one the allowlist names"
        );

        let derived = Content::derive_merges(&tokens);
        let d: std::collections::HashSet<_> = derived.iter().collect();
        let r: std::collections::HashSet<_> = declared.iter().collect();
        eprintln!(
            "  derived {} / declared {} / derived-only {} / declared-only {}",
            derived.len(),
            declared.len(),
            d.difference(&r).count(),
            r.difference(&d).count()
        );
        assert_eq!(d, r, "the derived merge SET differs from the declared one");
    }

    /// Rebuild two checkpoints' tokenizers and report where they disagree.
    ///
    /// ⚠️ BOTH SIDES GO THROUGH `extract_tokenizer`. An earlier experiment
    /// rebuilt one side by hand and twice reported a defect that was its own
    /// missing step — no `add_special_tokens`, then no post-processor. A partial
    /// reimplementation of the subject cannot distinguish its own gaps from the
    /// subject's, and the fix was not a better harness but NO harness.
    ///
    /// Returns the two `Content`s as well, because every caller then needs to
    /// ask something about the checkpoints to explain what it found.
    fn compare_rebuilt(a_name: &str, b_name: &str) -> Option<(Content, Content, Vec<String>)> {
        let (a_c, b_c) = (read_corpus(a_name)?, read_corpus(b_name)?);
        let a_tk = a_c
            .extract_tokenizer()
            .unwrap_or_else(|e| panic!("{a_name} does not rebuild: {e}"));
        let b_tk = b_c
            .extract_tokenizer()
            .unwrap_or_else(|e| panic!("{b_name} does not rebuild: {e}"));
        let inputs = tokenization_probe_inputs();
        let disagreements = disagreements_between(&a_tk, &b_tk, &inputs);
        eprintln!(
            "  {a_name} vs {b_name}: {} inputs x 2 arms, {} disagreements",
            inputs.len(),
            disagreements.len()
        );
        for d in disagreements.iter().take(6) {
            eprintln!("    {d}");
        }
        Some((a_c, b_c, disagreements))
    }

    /// Read one corpus checkpoint, or `None` when the corpus is absent.
    ///
    /// Shared so the comparison tests open their subjects identically — a
    /// per-test loader is a place for one of them to quietly read something
    /// else while still reporting agreement.
    fn read_corpus(name: &str) -> Option<Content> {
        let path = corpus_file(name)?;
        Some(Content::read(&path).unwrap_or_else(|e| panic!("reading {name}: {e}")))
    }

    /// Assert `shorter` is a byte-identical PREFIX of `longer`, returning the
    /// tail that extends it.
    ///
    /// ⚠️ This is the relation a digest comparison cannot express. Equality is
    /// what hashing tests; CONTAINMENT is what licenses one file's oracle to
    /// cover another, and the sweep that reported these two unrelated was
    /// answering the first question with its control passing.
    fn assert_prefix_and_tail<'a>(shorter: &[String], longer: &'a [String]) -> &'a [String] {
        assert!(
            longer.len() > shorter.len(),
            "the second list is not longer, so it cannot extend the first"
        );
        assert_eq!(
            &longer[..shorter.len()],
            shorter,
            "the leading ids are not byte-identical, so the shorter file's oracle does not carry"
        );
        &longer[shorter.len()..]
    }

    /// The special tokens a checkpoint actually registers, resolved from its
    /// declared `tokenizer.ggml.*_token_id` fields to the token strings.
    ///
    /// Shared, because two checkpoints over one vocabulary can register
    /// DIFFERENT specials — and the symmetric difference of these sets is the
    /// only admissible explanation for a tokenization disagreement between them.
    fn registered_specials(c: &Content) -> Vec<String> {
        let toks = tokens_of(c);
        [
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
        ]
        .iter()
        .filter_map(|k| match c.metadata().get(*k) {
            Some(Value::U32(v)) => Some(*v as usize),
            Some(Value::U64(v)) => Some(*v as usize),
            Some(Value::I32(v)) => Some(*v as usize),
            _ => None,
        })
        .filter_map(|id| toks.get(id).cloned())
        .collect()
    }

    /// A checkpoint's DECLARED merge list as a set, for use as an oracle.
    ///
    /// Panics rather than returning empty: a file with no declared merges is
    /// not a weak oracle, it is not an oracle, and an empty set would compare
    /// equal to nothing and silently pass a subset test.
    fn declared_merges(c: &Content) -> std::collections::HashSet<(String, String)> {
        let VocabAndMerges { merges, .. } =
            c.vocab_and_optional_merges().expect("vocab and merges");
        merges
            .expect("this checkpoint declares no merges, so it cannot serve as an oracle")
            .into_iter()
            .collect()
    }

    /// `tokenizer.ggml.scores` as `f32`s, or empty when the file declares none.
    ///
    /// ⚠️ TEST-ONLY, AND THAT BOUNDARY IS LOAD-BEARING. The production path
    /// reads NO scores — `derive_merges` works from the token list alone, and
    /// the module doc above records that `scores` is deliberately unread. These
    /// tests use the scores as an independent WITNESS to check the derived
    /// order against; they must never become an input to the derivation, or the
    /// witness and the subject stop being separate things.
    fn scores_of(c: &Content) -> Vec<f32> {
        match c.metadata().get("tokenizer.ggml.scores") {
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.to_f32().ok()).collect(),
            _ => Vec::new(),
        }
    }

    /// llama.cpp writes this where a token has no rank at all — byte tokens and
    /// unused slots. It is a MARKER, not a position.
    const SCORE_SENTINEL: f32 = -1_000_000_000.0;

    /// What a score-order comparison found, and over what.
    ///
    /// ⚠️ `pairs` IS NOT THE POPULATION. `informative` is. Equal-scored
    /// neighbours are ties, and a tie cannot contradict any ordering — so the
    /// pairs that could have falsified the claim are the ones with differing
    /// scores. Reporting `0 out of pairs` inflates the evidence by however many
    /// ties there are, which is 44% on one corpus vocabulary and 52% on another.
    struct Violations {
        /// Ascents, including those out of a sentinel.
        raw: usize,
        /// Ascents whose predecessor is a real rank — the ones that mean something.
        genuine: usize,
        /// Adjacent pairs with DIFFERING scores: the population that can falsify.
        informative: usize,
        /// All adjacent pairs, ties included. Kept only so the two can be shown
        /// side by side; it is not the denominator of any claim.
        pairs: usize,
    }

    /// Adjacent pairs where the derived merge order contradicts the order the
    /// scores imply, as `(raw, genuine)`.
    ///
    /// ⚠️ THE TWO NUMBERS DIFFER AND THE DIFFERENCE IS THE WHOLE POINT. Scores
    /// descend with rank, so an ASCENT between neighbours is a contradiction —
    /// unless the predecessor is `SCORE_SENTINEL`, which is not a rank and
    /// cannot be contradicted by one.
    ///
    /// Measured on `ggml-vocab-llama-spm.gguf`: 15 raw, and ALL 15 have a
    /// sentinel predecessor, so 0 genuine. An earlier version of this work read
    /// the raw 15 as that file's disagreement rate and was about to ship
    /// `99.976%` as the bar for other vocabularies to clear — a bar derived from
    /// a mismeasured reference, which is easier to clear than the true one and
    /// looks like success on both sides.
    fn score_order_violations(tokens: &[String], scores: &[f32]) -> Violations {
        let index: std::collections::HashMap<&str, usize> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();
        let mut product_ids = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            for c in 1..t.len() {
                if t.is_char_boundary(c) {
                    let (a, b) = t.split_at(c);
                    if index.contains_key(a) && index.contains_key(b) {
                        product_ids.push(i);
                    }
                }
            }
        }
        let seq: Vec<f32> = product_ids
            .iter()
            .filter_map(|&i| scores.get(i).copied())
            .collect();
        let mut raw = 0usize;
        let mut genuine = 0usize;
        let mut informative = 0usize;
        for k in 1..seq.len() {
            // ⚠️ A TIE CANNOT CONTRADICT ANY ORDERING, so it is not part of the
            // population this check ranges over. Counting ties in the
            // denominator inflates the evidence with members that are
            // structurally incapable of falsifying the claim.
            if seq[k] != seq[k - 1] {
                informative += 1;
            }
            if seq[k] > seq[k - 1] {
                raw += 1;
                if seq[k - 1] != SCORE_SENTINEL {
                    genuine += 1;
                }
            }
        }
        Violations {
            raw,
            genuine,
            informative,
            pairs: seq.len().saturating_sub(1),
        }
    }

    /// The token ids the order comparison actually ranges over — one entry per
    /// derived merge, in emission order.
    ///
    /// ⚠️ EXPOSED BECAUSE A FORCING ARM MUST PERTURB INSIDE THE MEASURED
    /// POPULATION. An earlier version set token 0's score below every other and
    /// the count did not move: token 0 is a control token that splits into
    /// nothing, so it never appears here at all. The perturbation was real and
    /// landed OUTSIDE the sequence being counted — a silence that looks exactly
    /// like a blind comparator, and the third distinct way a forcing arm failed
    /// on this measurement.
    fn derived_product_ids(tokens: &[String]) -> Vec<usize> {
        let index: std::collections::HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let mut out = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            for c in 1..t.len() {
                if t.is_char_boundary(c) {
                    let (a, b) = t.split_at(c);
                    if index.contains(a) && index.contains(b) {
                        out.push(i);
                    }
                }
            }
        }
        out
    }

    /// Ascents in a bare sequence — the counter `score_order_violations` uses,
    /// exposed so it can be checked against inputs whose answer is known.
    fn ascents(seq: &[f32]) -> usize {
        (1..seq.len()).filter(|&k| seq[k] > seq[k - 1]).count()
    }

    /// ⚠️ THE COUNTER, AGAINST ANSWERS KNOWN WITHOUT ANY FILE.
    ///
    /// Runs in the ordinary suite. Without this, every `0` below is a number
    /// from an instrument nothing has ever checked.
    #[test]
    fn the_ascent_counter_is_correct_on_known_sequences() {
        assert_eq!(ascents(&[5.0, 4.0, 3.0, 2.0]), 0, "strictly descending");
        assert_eq!(ascents(&[5.0, 6.0, 3.0, 2.0]), 1, "one ascent");
        assert_eq!(ascents(&[1.0, 2.0, 3.0, 4.0]), 3, "strictly ascending");
        assert_eq!(ascents(&[5.0, 5.0, 5.0]), 0, "ties are not ascents");
        assert_eq!(ascents(&[]), 0, "empty");
        assert_eq!(ascents(&[1.0]), 0, "single element has no pairs");
    }

    /// ⚠️ ONLY A STRICTLY-ADDITIVE PERTURBATION IS A VALID FORCING ARM.
    ///
    /// Two earlier attempts were rejected, and both reported the file whose
    /// answer is KNOWN as blind:
    ///
    /// ```text
    /// swap one adjacent pair            fixes one ascent, creates another -- nets to zero
    /// move the minimum to the front     adds one pair, REMOVES TWO        -- nets to zero
    /// prepend a lower value             adds one pair, removes nothing    -- always +1
    /// ```
    ///
    /// A perturbation that both ADDS and REMOVES can cancel. Believing either
    /// rejected arm would have produced "my instrument is broken" — a
    /// conclusion that stops work rather than yielding a wrong number, so
    /// nothing downstream would ever have contradicted it.
    #[test]
    fn prepending_a_lower_value_always_adds_exactly_one_ascent() {
        for seq in [
            vec![5.0, 4.0, 3.0],
            vec![5.0, 6.0, 3.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0],
        ] {
            let before = ascents(&seq);
            let lower = seq.iter().cloned().fold(f32::INFINITY, f32::min) - 1.0;
            let mut forced = vec![lower];
            forced.extend_from_slice(&seq);
            assert_eq!(
                ascents(&forced),
                before + 1,
                "prepending {lower} to {seq:?} must add exactly one ascent"
            );
        }
    }

    /// ⚠️ THE WEAKER WARRANT, MEASURED AND LABELLED AS WEAKER.
    ///
    /// Baichuan has no oracle: no corpus file shares or extends its vocabulary,
    /// so no declared merge list exists to compare the derived one against.
    /// What stands in its place is the ordering llama.cpp's own converter wrote
    /// into `tokenizer.ggml.scores` — a separately-authored fact this crate did
    /// not produce.
    ///
    /// ⚠️ AND THAT IS NOT PROOF OF CORRECTNESS. A merge list in the wrong ORDER
    /// could agree with those scores and still tokenize differently, and nothing
    /// measured here would see it. The claim is bounded on purpose.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn baichuan_derived_order_agrees_with_the_converters_own_scores() {
        let Some(c) = read_corpus("ggml-vocab-baichuan.gguf") else {
            lightbulb_skip();
            return;
        };
        let tokens = tokens_of(&c);
        let scores = scores_of(&c);

        // CONTROL: this is the vocabulary the allowlist names, and it declares
        // the scores the whole warrant rests on.
        assert_eq!(Content::vocab_sha256(&tokens), BAICHUAN_VOCAB_SHA256);
        assert_eq!(
            scores.len(),
            tokens.len(),
            "scores and tokens must be parallel, or the witness does not line up with the subject"
        );

        let v = score_order_violations(&tokens, &scores);
        let merges = Content::derive_merges(&tokens).len();
        eprintln!(
            "  baichuan: {} tokens, {merges} derived merges, {} adjacent pairs of which {} INFORMATIVE ({} ties), {} raw / {} genuine violations",
            tokens.len(),
            v.pairs,
            v.informative,
            v.pairs - v.informative,
            v.raw,
            v.genuine
        );

        // ⚠️ STATE THE POPULATION IN THE CLAIM. `0 violations` means something
        // very different over 54804 pairs than over 165, and the notation erases
        // the difference.
        // ⚠️ ASSERT ON THE INFORMATIVE COUNT, NOT THE RAW PAIR COUNT. An earlier
        // version guarded `merges > 50_000` and the warrant quoted 54804 -- but
        // 24347 of those pairs are TIES, which no ordering can contradict. The
        // claim's population is the pairs that could have falsified it.
        assert!(
            v.informative > 25_000,
            "the informative population collapsed to {} of {} pairs; a `0` over a small population is much weaker evidence and this assertion exists to make that visible",
            v.informative,
            v.pairs
        );
        assert_eq!(
            v.genuine, 0,
            "the derived order contradicts the converter's own score order {} times",
            v.genuine
        );

        // ⚠️ FORCE IT. Without this, `0` is indistinguishable from a comparator
        // that cannot report anything. Strictly additive: prepend a value lower
        // than every score, which adds exactly one ascent and removes nothing.
        // (A swap, or moving the minimum, can cancel — see
        // `prepending_a_lower_value_always_adds_exactly_one_ascent`.)
        let ids = derived_product_ids(&tokens);
        assert!(
            !ids.is_empty(),
            "no derived merges, so there is no population to perturb"
        );
        let mut forced_scores = scores.clone();
        let lowest = scores.iter().cloned().fold(f32::INFINITY, f32::min) - 1.0;
        // The first token in the MEASURED SEQUENCE, not the first token in the
        // vocabulary -- see `derived_product_ids`.
        forced_scores[ids[0]] = lowest;
        let forced_genuine = score_order_violations(&tokens, &forced_scores).genuine;
        eprintln!(
            "  forced (id {} = first in the derived sequence, scored below all others): genuine {forced_genuine}",
            ids[0]
        );
        assert!(
            forced_genuine > v.genuine,
            "the comparator did not register a deliberately introduced violation, so its {} above is a silence rather than a reading",
            v.genuine
        );
    }

    /// ⚠️ WHY `raw` AND `genuine` DIFFER, AND WHY A BASELINE WAS RETIRED.
    ///
    /// `-1000000000.0` is llama.cpp's marker for a token with no rank at all —
    /// byte tokens and unused slots. It is not a position and cannot be
    /// contradicted by one.
    ///
    /// On the VALIDATED file every raw violation has a sentinel predecessor, so
    /// its genuine count is 0. An earlier version of this work read the raw 15
    /// as that file's true disagreement rate and was about to ship `99.976%` as
    /// the bar for other vocabularies to clear. ⚠️ A BAR DERIVED FROM A
    /// MISMEASURED REFERENCE IS EASIER TO CLEAR THAN THE TRUE ONE, AND CLEARING
    /// IT LOOKS LIKE SUCCESS ON BOTH SIDES.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn every_raw_violation_on_the_validated_file_is_a_sentinel_artefact() {
        let Some(c) = read_corpus("ggml-vocab-llama-spm.gguf") else {
            lightbulb_skip();
            return;
        };
        let tokens = tokens_of(&c);
        let scores = scores_of(&c);
        let sentinels = scores.iter().filter(|&&s| s == SCORE_SENTINEL).count();
        let v = score_order_violations(&tokens, &scores);
        eprintln!(
            "  llama-spm: {sentinels} sentinel tokens, {} adjacent pairs of which {} INFORMATIVE ({} ties), {} raw, {} genuine",
            v.pairs,
            v.informative,
            v.pairs - v.informative,
            v.raw,
            v.genuine
        );

        // The file must actually CONTAIN sentinels, or this test proves nothing
        // about the distinction it exists to draw.
        assert!(
            sentinels > 0,
            "no sentinel scores here, so raw and genuine cannot differ and this test is vacuous"
        );
        assert!(
            v.raw > 0,
            "no raw violations, so nothing distinguishes the two counts"
        );
        assert_eq!(
            v.genuine, 0,
            "the validated file has genuine order violations, which would undermine using it as the reference at all"
        );
    }

    /// Baichuan rebuilds, and the allowlist is what permits it.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn baichuan_rebuilds_now_and_would_not_without_its_allowlist_entry() {
        let Some(c) = read_corpus("ggml-vocab-baichuan.gguf") else {
            lightbulb_skip();
            return;
        };
        let digest = Content::vocab_sha256(&tokens_of(&c));
        assert!(
            Content::spm_derivation_warrant(&digest).is_some(),
            "baichuan is not allowlisted, so the rebuild below cannot be attributed to this change"
        );
        let tk = c
            .extract_tokenizer()
            .expect("baichuan rebuilds from derived merges");

        // A rebuild that produces nothing usable is not a rebuild. Round-trip a
        // few inputs through it so the success is a reading, not a constructor
        // that happened to return Ok.
        for probe in ["hello", "\u{4E2D}\u{6587}", "a b c", "123"] {
            let enc = tk.encode(probe, false).expect("encode");
            assert!(
                !enc.get_ids().is_empty(),
                "{probe:?} encoded to nothing, so the tokenizer is not usable"
            );
        }
        eprintln!("  baichuan rebuilt and encodes; warrant: weaker, see spm_derivation_warrant");
    }

    /// ⚠️ THE ENABLING FACT, WHICH A DIGEST SWEEP CANNOT SEE.
    ///
    /// Phi-3's vocabulary EXTENDS the Llama SentencePiece one rather than
    /// differing from it, so the same oracle covers both. A whole-vocabulary
    /// digest comparison answers `A == B` and reported them unrelated with its
    /// own positive control passing.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn phi3_extends_the_llama_vocabulary_and_inherits_its_oracle() {
        let (Some(phi3_c), Some(spm_c), Some(tiny_c)) = (
            read_corpus("ggml-vocab-phi-3.gguf"),
            read_corpus("ggml-vocab-llama-spm.gguf"),
            read_corpus("tinyllama-1.1b-chat-v1.0.Q4_0.gguf"),
        ) else {
            lightbulb_skip();
            return;
        };
        let phi3_tokens = tokens_of(&phi3_c);
        let spm_tokens = tokens_of(&spm_c);

        // The digests DIFFER -- which is what hid the relation.
        assert_ne!(
            Content::vocab_sha256(&phi3_tokens),
            Content::vocab_sha256(&spm_tokens),
            "if these ever become equal the containment story below is the wrong explanation"
        );
        assert_eq!(Content::vocab_sha256(&phi3_tokens), PHI3_VOCAB_SHA256);

        // ...and phi-3 is a strict EXTENSION whose tail adds no merges.
        let extra = assert_prefix_and_tail(&spm_tokens, &phi3_tokens);
        eprintln!(
            "  phi-3 {} = llama-spm {} + {} extra: {:?}",
            phi3_tokens.len(),
            spm_tokens.len(),
            extra.len(),
            &extra[..4.min(extra.len())]
        );
        let derived_phi3 = Content::derive_merges(&phi3_tokens);
        let derived_spm = Content::derive_merges(&spm_tokens);
        assert_eq!(
            derived_phi3.len(),
            derived_spm.len(),
            "the extra tokens introduced merges, so phi-3 needs its own oracle after all"
        );

        // THE ORACLE: TinyLlama's DECLARED list, on a vocabulary it shares.
        let declared = declared_merges(&tiny_c);
        let derived: std::collections::HashSet<_> = derived_phi3.into_iter().collect();
        eprintln!(
            "  phi-3 derived {} vs TinyLlama declared {} -- extra {}, MISSED {}",
            derived.len(),
            declared.len(),
            derived.difference(&declared).count(),
            declared.difference(&derived).count()
        );
        assert_eq!(
            derived, declared,
            "phi-3's derived merges are not the declared list, so the shared-prefix oracle does not carry"
        );
    }

    /// phi-3 rebuilds, and agrees with the oracle over the shared probe set.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn phi3_agrees_with_the_oracle_through_the_production_path() {
        let Some((tiny_c, phi3_c, disagreements)) = compare_rebuilt(
            "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
            "ggml-vocab-phi-3.gguf",
        ) else {
            lightbulb_skip();
            return;
        };

        // ⚠️ NOT ASSERTED EQUAL TO ZERO, AND MY FIRST FILTER LOOKED THE WRONG WAY.
        //
        // The two checkpoints register DIFFERENT special tokens, so any input
        // containing a token special to exactly one of them legitimately differs.
        // Measured:
        //
        //     tinyllama   eos = id 2     '</s>'
        //     phi-3       eos = id 32000 '<|endoftext|>'   <- one of its extra 64
        //
        // So `</s>` is special to the ORACLE and ordinary text to phi-3, which is
        // phi-3's real configuration rather than a defect. An earlier version of
        // this test filtered on phi-3's EXTRA tokens and missed that the cause
        // was a token the oracle has and phi-3 does not.
        let a: std::collections::HashSet<String> =
            registered_specials(&tiny_c).into_iter().collect();
        let b: std::collections::HashSet<String> =
            registered_specials(&phi3_c).into_iter().collect();
        let only_one: Vec<&String> = a.symmetric_difference(&b).collect();
        eprintln!("  special to exactly one checkpoint: {only_one:?}");
        assert!(
            !only_one.is_empty(),
            "if the special sets are identical this filter is inert and the assertion below is vacuous"
        );

        let unexplained: Vec<&String> = disagreements
            .iter()
            .filter(|d| !only_one.iter().any(|e| d.contains(e.as_str())))
            .collect();
        assert!(
            unexplained.is_empty(),
            "{} disagreements involve no token special to exactly one checkpoint: {:?}",
            unexplained.len(),
            unexplained
        );
    }

    /// ⚠️ WHY THE ALLOWLIST IS NECESSARY AND NOT MERELY PRUDENT.
    ///
    /// The derivation is exact for llama.cpp's SentencePiece export and is a
    /// strict SUPERSET everywhere else. An earlier version of this work called
    /// that a structural property of vocabularies; it is a property of one
    /// export format, and this test is what makes the difference falsifiable
    /// rather than a sentence in a comment.
    ///
    /// It never MISSES a declared merge — that half does hold across every
    /// vocabulary measured — so the assertions below pin BOTH directions.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn the_derivation_over_generates_on_byte_level_bpe() {
        let mut checked = 0usize;
        for name in [
            "SmolLM2-135M-Instruct-Q4_0.gguf",
            "ggml-vocab-qwen2.gguf",
            "ggml-vocab-starcoder.gguf",
        ] {
            let Some(path) = corpus_file(name) else {
                continue;
            };
            let c = Content::read(&path).expect("read");
            let VocabAndMerges { tokens, merges, .. } =
                c.vocab_and_optional_merges().expect("vocab");
            let Some(declared) = merges else {
                panic!("{name} is a byte-level BPE file and must declare merges")
            };
            let derived: std::collections::HashSet<_> =
                Content::derive_merges(&tokens).into_iter().collect();
            let declared: std::collections::HashSet<_> = declared.into_iter().collect();

            let missed = declared.difference(&derived).count();
            let extra = derived.difference(&declared).count();
            eprintln!(
                "  {name:<38} declared {:>6}  derived {:>6}  extra {:>6}  MISSED {missed}",
                declared.len(),
                derived.len(),
                extra
            );

            // The half that HOLDS everywhere.
            assert_eq!(
                missed, 0,
                "{name}: the derivation missed a declared merge, which it has never done"
            );
            // The half that does NOT, and is the reason for the allowlist.
            assert!(
                extra > declared.len() / 2,
                "{name}: the derivation is no longer substantially over-generating on byte-level BPE ({extra} extra against {} declared). If that is a real change, the allowlist's justification needs re-deriving, not updating.",
                declared.len()
            );
            checked += 1;
        }
        // ⚠️ POSITIVE CONTROL: without it a corpus missing all three files
        // passes while asserting nothing.
        assert!(
            checked >= 2,
            "checked {checked} byte-level vocabularies; below two this test proves nothing"
        );
    }

    /// The digest the allowlist names is the one this vocabulary actually has.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn the_allowlisted_digest_belongs_to_llama_spm() {
        let Some(spm) = corpus_file("ggml-vocab-llama-spm.gguf") else {
            lightbulb_skip();
            return;
        };
        let c = Content::read(&spm).expect("read llama-spm");
        assert_eq!(
            Content::vocab_sha256(&tokens_of(&c)),
            LLAMA_SPM_VOCAB_SHA256
        );
        assert!(
            Content::spm_derivation_warrant(LLAMA_SPM_VOCAB_SHA256).is_some(),
            "the digest is allowlisted"
        );
    }

    /// ⚠️ THE GUARD. An unoracled vocabulary must still be refused, or the
    /// allowlist is decorative.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn a_vocabulary_with_no_oracle_is_still_refused() {
        // ⚠️ THIS WAS A LOOP OVER A LIST AND THE LIST HAS ONE ENTRY LEFT.
        //
        // phi-3 left it when it turned out to share the Llama vocabulary as a
        // prefix; baichuan left it when its weaker warrant was accepted. The
        // guard FAILED on each change, which is what it is for.
        //
        // The loop is gone rather than suppressed: `clippy::single_element_loop`
        // fired on the narrowed list and the gate caught it. A loop over one
        // element is a claim that there is a collection, and there is not.
        //
        // ⚠️ WHEN THIS LAST SUBJECT IS ALLOWLISTED, DELETE THIS TEST -- do not
        // rewrite it over an empty set. A guard that outlives its subject is a
        // green light with nothing behind it, and the loop form made that
        // failure silent: an empty `for` reaches no assertion and reports ok.
        let name = "tinyllamas-stories-260k-f32.gguf";
        let Some(p) = corpus_file(name) else {
            lightbulb_skip();
            return;
        };
        let c = Content::read(&p).expect("read");
        let digest = Content::vocab_sha256(&tokens_of(&c));
        assert_ne!(
            digest, LLAMA_SPM_VOCAB_SHA256,
            "{name} shares the oracle's vocabulary"
        );
        assert!(
            Content::spm_derivation_warrant(&digest).is_none(),
            "{name} must not be allowlisted"
        );
        let err = match c.extract_tokenizer() {
            Ok(_) => panic!("{name} rebuilt without an oracle for its vocabulary"),
            Err(e) => e.to_string(),
        };
        // The refusal must say what would RETIRE it, not merely that it happened.
        assert!(
            err.contains(&digest),
            "{name}: the refusal does not name the digest a future oracle must match: {err}"
        );
        eprintln!("  {name}: refused, and the message names its digest");
    }

    /// Inputs the tokenizer comparisons range over.
    ///
    /// ⚠️ NAMED AND SHARED so every vocabulary is probed with the SAME set. A
    /// per-test list drifts, and a comparison that agrees over a narrower set
    /// than its neighbour looks equally green while proving less.
    ///
    /// Covers: empty, bare space, newline, tab, CRLF, C0 controls and DEL, CJK,
    /// Greek, Cyrillic, emoji above the BMP, combining accents, a 200-character
    /// repeat, the SPM space marker, digits, decimals, identifier shapes, the
    /// special-token spellings, ASCII punctuation, and the reference prompt.
    fn tokenization_probe_inputs() -> Vec<String> {
        vec![
            String::new(),
            " ".into(),
            "\n".into(),
            "\n\n\t ".into(),
            "a".into(),
            PROMPT.into(),
            "The capital of France is Paris.".into(),
            "antidisestablishmentarianism".into(),
            "  leading and trailing  ".into(),
            "CamelCaseIdentifier".into(),
            "snake_case_name".into(),
            "1234567890".into(),
            "3.14159".into(),
            "<s></s><unk>".into(),
            "cafe\u{301} nai\u{308}ve".into(),
            "\u{65E5}\u{672C}\u{8A9E}".into(),
            "\u{395}\u{3BB}\u{3BB}\u{3B7}\u{3BD}\u{3B9}\u{3BA}\u{3AC}".into(),
            "\u{440}\u{443}\u{441}\u{441}\u{43A}\u{438}\u{439}".into(),
            "emoji \u{1F600}\u{1F680}".into(),
            "\u{0}\u{1}\u{7F}".into(),
            "tab\tsep".into(),
            "line1\nline2\r\nline3".into(),
            "a".repeat(200),
            "\u{2581}\u{2581}\u{2581}".into(),
            "!@#$%^&*()_+-=[]{}|;:',.<>?/".into(),
            "The quick brown fox jumps over the lazy dog".into(),
        ]
    }

    /// Every input on which two tokenizers produce different ids, both
    /// `add_special` arms, as reader-facing lines.
    fn disagreements_between(
        a_tk: &tokenizers::Tokenizer,
        b_tk: &tokenizers::Tokenizer,
        inputs: &[String],
    ) -> Vec<String> {
        let mut out = Vec::new();
        for inp in inputs {
            for add_special in [false, true] {
                let a = a_tk.encode(inp.as_str(), add_special).expect("encode a");
                let b = b_tk.encode(inp.as_str(), add_special).expect("encode b");
                if a.get_ids() != b.get_ids() {
                    out.push(format!(
                        "{inp:?} (add_special={add_special}) a {:?} b {:?}",
                        a.get_tokens(),
                        b.get_tokens()
                    ));
                }
            }
        }
        out
    }

    /// The whole point: llama-spm now rebuilds, and agrees with the oracle.
    ///
    /// ⚠️ BOTH SIDES GO THROUGH `extract_tokenizer`. An earlier version of this
    /// experiment rebuilt the tokenizer by hand and twice reported a defect that
    /// was its own missing step -- no `add_special_tokens`, then no
    /// post-processor. A partial reimplementation of the subject cannot
    /// distinguish its own gaps from the subject's.
    #[test]
    #[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
    fn llama_spm_agrees_with_the_oracle_through_the_production_path() {
        let Some((tiny_c, spm_c, disagreements)) = compare_rebuilt(
            "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
            "ggml-vocab-llama-spm.gguf",
        ) else {
            lightbulb_skip();
            return;
        };

        // CONTROL: same vocabulary, or the oracle does not range over the subject.
        assert_eq!(
            tokens_of(&spm_c),
            tokens_of(&tiny_c),
            "the vocabularies differ, so this is not an oracle"
        );

        // CONTROL: the oracle still produces the established figure.
        let oracle_tk = tiny_c.extract_tokenizer().expect("tinyllama rebuilds");
        assert_eq!(
            oracle_tk
                .encode(PROMPT, false)
                .expect("oracle encode")
                .get_ids()
                .len(),
            22,
            "the oracle no longer produces 22 ids, so there is no baseline"
        );

        // ⚠️ THE WARRANT STRING QUOTES THIS COUNT AND NOTHING CHECKED IT. It
        // said 28 while this test ran 26 -- the 28 came from a throwaway
        // experiment and never matched. A figure in prose beside a figure in
        // code drifts silently; this binds them.
        //
        // ⚠️ THIS IS A DETECTOR, NOT AN IMPOSSIBILITY, AND THAT WAS A DECISION.
        //
        // The stronger form would FORMAT the count into the warrant, so the two
        // could not disagree at all. It is declined deliberately: the warrant is
        // a production `&'static str` that a user reads in a REFUSAL MESSAGE,
        // and the count is a TEST-ONLY fact. Formatting one from the other would
        // make a shipped error message depend on a test fixture's length --
        // unreadable without running the test that defines it, and editable by
        // anyone changing the fixture without knowing they had touched a
        // user-facing string.
        //
        // Recorded here because a DECLINED CHECK HAS NO COMPLAINANT: no red, no
        // file, no row. Without this paragraph the missing binding reads as an
        // oversight in six weeks and someone "fixes" it.
        //
        // Forced rather than reasoned: changing the prose to "27" while leaving
        // this code at 26 turns the assertion below RED, and the failure message
        // quotes the offending warrant back.
        let inputs = tokenization_probe_inputs();
        let warrant = Content::spm_derivation_warrant(LLAMA_SPM_VOCAB_SHA256)
            .expect("the llama-spm vocabulary is allowlisted");
        assert!(
            warrant.contains(&format!("{} varied inputs", inputs.len())),
            "the warrant claims a different input count than this test runs ({} inputs). Warrant: {warrant}",
            inputs.len()
        );

        assert!(
            disagreements.is_empty(),
            "{} of {} comparisons disagree between the derived and declared tokenizers",
            disagreements.len(),
            inputs.len() * 2
        );
    }

    fn lightbulb_skip() {
        lightbulb_test_notice();
    }

    fn lightbulb_test_notice() {
        crate::test_notice::skip_unless_required(
            "LIGHTBULB_REQUIRE_CORPUS",
            "the SPM derivation tests need the local GGUF corpus",
        );
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
        // `llama-bpe` was here until it was verified against
        // `meta-llama/Meta-Llama-3-8B` and added, at which point this test went
        // red — which is the pair working: the allowlist grew and its negative
        // half noticed. `command-r` has now done the same, verified through the
        // ungated `mlx-community/c4ai-command-r-v01-4bit` re-upload.
        //
        // ⚠️ AND `starcoder` HAS NOW LEFT TOO, WHICH IS THE THIRD TIME AND THE
        // ONLY ONE WHERE THE TEST WAS PINNING A REFUSAL THAT WAS WRONG. Its
        // reasoning was sound and its PREMISE was false: the GGUF declares
        // `general.architecture = starcoder2`, so the gated repo the refusal
        // waited on was the wrong repo. A pinned refusal is only as good as the
        // fact it rests on, and this test cannot check that fact -- it checks
        // that the table says no, not that saying no is right.
        //
        // THE SURVIVORS ARE CHOSEN, NOT LEFTOVERS. Every name below is refused
        // FOR CAUSE rather than merely unexamined, and all three premises were
        // RE-MEASURED on 2026-09-03 rather than carried forward:
        //
        //   mpt        `general.architecture = mpt`, so it really is an MPT file
        //              and `mosaicml/mpt-7b` really is the right reference. It
        //              returns 401 -- gated, not absent. HOLDS.
        //   qwen35     `ggml-vocab-qwen2.gguf` and `ggml-vocab-qwen35.gguf` differ
        //              by ONE BYTE, at the offset where `pre` is stored. The
        //              string under test is the only difference between the two
        //              inputs, so no corpus can discriminate them. HOLDS, and is
        //              now structural rather than empirical.
        //   <absent>   `general.name = gpt-neox-20b`, no `pre` key at all; keying
        //              on ABSENCE would apply one checkpoint's rule to every
        //              future GGUF that omits the field. HOLDS.
        assert!(Content::bpe_pre_tokenizer("mpt").is_none());
        assert!(Content::bpe_pre_tokenizer("qwen35").is_none());
        assert!(Content::bpe_pre_tokenizer("<absent>").is_none());
        assert!(Content::bpe_pre_tokenizer("").is_none());
    }
}
