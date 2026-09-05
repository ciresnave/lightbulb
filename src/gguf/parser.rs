//! Low-level GGUF format parsing from memory-mapped bytes
//!
//! This module implements direct parsing of GGUF v3 file format according to:
//! https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//!
//! Key design decisions:
//! - Zero-copy: All parsing uses byte slices, no allocation until necessary
//! - Little-endian only: Models are little-endian by default (big-endian detection TBD)
//! - Minimal validation: Trust that files are well-formed (they come from llama.cpp)

use anyhow::{Context, Result, bail};
use std::collections::HashMap;

/// GGUF magic number: "GGUF" at byte level (0x47 0x47 0x55 0x46)
const GGUF_MAGIC: u32 = 0x46554747;

/// The GGUF container versions this parser reads. A SET, not an equality.
///
/// # Why v2 is in and v1 is out — they are refused for different reasons
///
/// **v2 and v3 share the field widths this parser already implements**: counts
/// and string lengths are `u64` in both. v3 adds big-endian support and changes
/// nothing this code reads. So v2 needed no new parsing at all — it was excluded
/// only because the check was written `version != 3`, an equality where a set
/// was meant, and a loud specific refusal reads as a considered scope decision.
///
/// **v1 is genuinely different**: its counts and string lengths are `u32`.
/// Measured 2026-09-05 by re-reading the local v1 file with `u64` widths — the
/// KV count comes out as `7308890738324930580`, and a v3-assuming parser dies
/// allocating against it.
///
/// # ⚠️ Refusing v1 is NOT a claim that the file is bad
///
/// `tinyllamas-stories-260k-f32.gguf` is a perfectly good GGUF: 48 tensors, 18
/// KV pairs, a 512-token vocabulary, `bos_token_id=1`, `eos_token_id=2`. It was
/// called "unreadable" by three separate lanes in one exchange, and it is only
/// unreadable *by builds that assume u64*. The distinction matters because
/// "1 file has no vocabulary" made the corpus census read 18 when it is 19.
///
/// # What supporting v1 would actually cost, and who asked
///
/// **Not a change to this constant.** Every count and every string length would
/// need width-switching at each read site, so the parser gains a second parse
/// path rather than one more accepted value. **Nobody has asked** — the only v1
/// file in the local corpus is a 260k-parameter toy, and mlmf refuses v1
/// deliberately for the same reason, so widening here would make two independent
/// readers agree by moving one of them rather than by finding anything.
///
/// Recorded because a bare constant cannot carry any of this, and a correct
/// constant with no reasoning beside it is indistinguishable from an
/// unconsidered one — which is precisely how `version != 3` survived.
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

/// Default alignment (used if general.alignment not specified)
const DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF metadata value types (enum gguf_metadata_value_type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataValueType {
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => bail!("Invalid metadata value type: {}", value),
        }
    }
}

/// GGUF metadata value (simplified for our needs)
#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

/// GGUF tensor type (enum ggml_type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(dead_code)]
pub enum TensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    // ... more quantization types (see spec for full list)
}

/// Parsed GGUF tensor information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32, // Raw type value (for now)
    pub offset: u64,      // Offset relative to tensor_data section
}

/// Parsed GGUF header and metadata
#[derive(Debug)]
pub struct GGUFHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensor_infos: Vec<TensorInfo>,
    pub tensor_data_offset: u64, // Absolute offset to tensor data in file
    pub alignment: u64,          // From general.alignment metadata
}

/// Parse GGUF file from memory-mapped bytes
///
/// This is the main entry point. It parses the entire GGUF structure:
/// 1. Header (magic, version, counts)
/// 2. Metadata key-value pairs
/// 3. Tensor infos
/// 4. Computes tensor_data_offset with alignment
///
/// # Arguments
/// * `data` - Memory-mapped file bytes (entire GGUF file)
///
/// # Returns
/// GGUFHeader with all metadata and tensor offsets ready for zero-copy access
pub fn parse_gguf(data: &[u8]) -> Result<GGUFHeader> {
    let mut offset = 0;

    // Parse header
    let magic = read_u32(data, &mut offset)?;
    if magic != GGUF_MAGIC {
        bail!(
            "Invalid GGUF magic number: 0x{:08X} (expected 0x{:08X})",
            magic,
            GGUF_MAGIC
        );
    }

    let version = read_u32(data, &mut offset)?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        // ⚠️ The message names what is ACCEPTED, derived from the constant. The
        // previous version read "(expected 3)" while the check accepted 2 or 3 —
        // stale from the moment v2 was added, and an error that misreports the
        // accepted set sends a reader to the wrong question.
        bail!(
            "Unsupported GGUF version: {version} (this parser reads {SUPPORTED_VERSIONS:?}). \
             v1 uses u32 counts and string lengths where v2/v3 use u64, so it needs a second \
             parse path rather than one more accepted value; see SUPPORTED_VERSIONS. This is \
             a limit of this reader, not a statement that the file is malformed."
        );
    }

    let tensor_count = read_u64(data, &mut offset)?;
    let metadata_kv_count = read_u64(data, &mut offset)?;

    // Parse metadata
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_string(data, &mut offset)?;
        let value_type = MetadataValueType::from_u32(read_u32(data, &mut offset)?)?;
        let value = read_metadata_value(data, &mut offset, value_type)?;
        metadata.insert(key, value);
    }

    // Extract alignment from metadata (default 32)
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| match v {
            MetadataValue::UInt32(a) => Some(*a as u64),
            _ => None,
        })
        .unwrap_or(DEFAULT_ALIGNMENT);

    // Parse tensor infos
    let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(data, &mut offset)?;
        let n_dimensions = read_u32(data, &mut offset)?;

        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(read_u64(data, &mut offset)?);
        }

        let tensor_type = read_u32(data, &mut offset)?;
        let tensor_offset = read_u64(data, &mut offset)?;

        tensor_infos.push(TensorInfo {
            name,
            dimensions,
            tensor_type,
            offset: tensor_offset,
        });
    }

    // Calculate tensor_data_offset with alignment
    // tensor_data starts after header + metadata + tensor_infos + padding
    let tensor_data_offset = align_offset(offset as u64, alignment);

    Ok(GGUFHeader {
        version,
        tensor_count,
        metadata,
        tensor_infos,
        tensor_data_offset,
        alignment,
    })
}

/// Align offset to next multiple of alignment
fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset + (alignment - (offset % alignment)) % alignment
}

/// Read u32 (little-endian)
fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset + 4 > data.len() {
        bail!("Unexpected EOF reading u32 at offset {}", *offset);
    }
    let bytes = &data[*offset..*offset + 4];
    *offset += 4;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read u64 (little-endian)
fn read_u64(data: &[u8], offset: &mut usize) -> Result<u64> {
    if *offset + 8 > data.len() {
        bail!("Unexpected EOF reading u64 at offset {}", *offset);
    }
    let bytes = &data[*offset..*offset + 8];
    *offset += 8;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Read f32 (little-endian)
fn read_f32(data: &[u8], offset: &mut usize) -> Result<f32> {
    if *offset + 4 > data.len() {
        bail!("Unexpected EOF reading f32 at offset {}", *offset);
    }
    let bytes = &data[*offset..*offset + 4];
    *offset += 4;
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read f64 (little-endian)
fn read_f64(data: &[u8], offset: &mut usize) -> Result<f64> {
    if *offset + 8 > data.len() {
        bail!("Unexpected EOF reading f64 at offset {}", *offset);
    }
    let bytes = &data[*offset..*offset + 8];
    *offset += 8;
    Ok(f64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Read string (uint64 length + UTF-8 bytes)
fn read_string(data: &[u8], offset: &mut usize) -> Result<String> {
    let len = read_u64(data, offset)? as usize;
    if *offset + len > data.len() {
        bail!("Unexpected EOF reading string at offset {}", *offset);
    }
    let bytes = &data[*offset..*offset + len];
    *offset += len;
    String::from_utf8(bytes.to_vec()).context("Invalid UTF-8 in GGUF string")
}

/// Read metadata value (recursive for arrays)
fn read_metadata_value(
    data: &[u8],
    offset: &mut usize,
    value_type: MetadataValueType,
) -> Result<MetadataValue> {
    match value_type {
        MetadataValueType::UInt8 => {
            if *offset + 1 > data.len() {
                bail!("Unexpected EOF reading UInt8");
            }
            let value = data[*offset];
            *offset += 1;
            Ok(MetadataValue::UInt8(value))
        }
        MetadataValueType::Int8 => {
            if *offset + 1 > data.len() {
                bail!("Unexpected EOF reading Int8");
            }
            let value = data[*offset] as i8;
            *offset += 1;
            Ok(MetadataValue::Int8(value))
        }
        MetadataValueType::UInt16 => {
            let bytes = &data[*offset..*offset + 2];
            *offset += 2;
            Ok(MetadataValue::UInt16(u16::from_le_bytes([
                bytes[0], bytes[1],
            ])))
        }
        MetadataValueType::Int16 => {
            let bytes = &data[*offset..*offset + 2];
            *offset += 2;
            Ok(MetadataValue::Int16(i16::from_le_bytes([
                bytes[0], bytes[1],
            ])))
        }
        MetadataValueType::UInt32 => Ok(MetadataValue::UInt32(read_u32(data, offset)?)),
        MetadataValueType::Int32 => {
            let value = read_u32(data, offset)?;
            Ok(MetadataValue::Int32(value as i32))
        }
        MetadataValueType::Float32 => Ok(MetadataValue::Float32(read_f32(data, offset)?)),
        MetadataValueType::Bool => {
            if *offset + 1 > data.len() {
                bail!("Unexpected EOF reading Bool");
            }
            let value = data[*offset];
            *offset += 1;
            if value > 1 {
                bail!("Invalid bool value: {} (must be 0 or 1)", value);
            }
            Ok(MetadataValue::Bool(value == 1))
        }
        MetadataValueType::String => Ok(MetadataValue::String(read_string(data, offset)?)),
        MetadataValueType::Array => {
            let element_type = MetadataValueType::from_u32(read_u32(data, offset)?)?;
            let len = read_u64(data, offset)? as usize;
            let mut array = Vec::with_capacity(len);
            for _ in 0..len {
                array.push(read_metadata_value(data, offset, element_type)?);
            }
            Ok(MetadataValue::Array(array))
        }
        MetadataValueType::UInt64 => Ok(MetadataValue::UInt64(read_u64(data, offset)?)),
        MetadataValueType::Int64 => {
            let value = read_u64(data, offset)?;
            Ok(MetadataValue::Int64(value as i64))
        }
        MetadataValueType::Float64 => Ok(MetadataValue::Float64(read_f64(data, offset)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn test_read_primitives() {
        let data = vec![
            0x12, 0x34, 0x56, 0x78, // u32: 0x78563412 (little-endian)
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // u64
        ];

        let mut offset = 0;
        assert_eq!(read_u32(&data, &mut offset).unwrap(), 0x78563412);
        assert_eq!(offset, 4);

        assert_eq!(read_u64(&data, &mut offset).unwrap(), 0x0807060504030201u64);
        assert_eq!(offset, 12);
    }

    #[test]
    fn test_read_string() {
        let mut data = vec![
            0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // length = 5
        ];
        data.extend_from_slice(b"hello");

        let mut offset = 0;
        assert_eq!(read_string(&data, &mut offset).unwrap(), "hello");
        assert_eq!(offset, 8 + 5);
    }
}

#[cfg(test)]
mod version_gate_tests {
    use super::{SUPPORTED_VERSIONS, parse_gguf};

    fn header_declaring(version: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&version.to_le_bytes());
        // Zero tensors and zero KV pairs: enough for an ACCEPTED version to get
        // past the gate, so the test discriminates the gate from a parse failure.
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b
    }

    /// ⚠️ THE MESSAGE MUST NAME THE ACCEPTED SET, AND MUST DERIVE IT.
    ///
    /// It previously read "(expected 3)" while the check accepted 2 or 3 — stale
    /// from the moment v2 was added, because the message named a constant that
    /// was no longer the rule. An error that misreports the accepted set sends a
    /// reader to the wrong question: they go looking for a corrupt file.
    #[test]
    fn a_refused_version_is_told_what_is_accepted() {
        let err = parse_gguf(&header_declaring(1))
            .expect_err("v1 must be refused")
            .to_string();
        assert!(
            err.contains('1'),
            "the error must name the version it saw: {err}"
        );
        for v in SUPPORTED_VERSIONS {
            assert!(
                err.contains(&v.to_string()),
                "the error must name every accepted version, so it cannot go stale                  against the constant: {v} missing from {err}"
            );
        }
        // Asserted POSITIVELY. The first version of this was
        // `!err.contains("malformed")`, which FAILS on a message that uses the
        // word in a negation: the assertion's FORM did not match its intent, and
        // a substring check cannot see a "not".
        assert!(
            err.contains("limit of this reader"),
            "refusing a version is a limit of THIS READER, and the message must say \n             so -- three lanes called this same file 'unreadable' in one exchange, \n             which is how a 512-token vocabulary went uncounted: {err}"
        );
    }

    /// The control. Without it the assertion above is satisfied by a parser that
    /// refuses everything, and "v1 is refused" would carry no information.
    #[test]
    fn every_supported_version_gets_past_the_gate() {
        for v in SUPPORTED_VERSIONS {
            let parsed = parse_gguf(&header_declaring(*v));
            assert!(
                parsed.is_ok(),
                "version {v} is in SUPPORTED_VERSIONS and must not be refused: {:?}",
                parsed.err()
            );
        }
    }
}
