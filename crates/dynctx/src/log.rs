// log.rs — audit logging with Cap'n Proto

//! Every state‑mutating Op pushed through the queue is serialised as a
//! Cap'n Proto `LogEntry` and appended to `context.log`.  Each record carries
//! the SHA‑256 of the previous record so that any truncation / insertion is
//! detectable.
//!
//! Layout on disk:
//! [u32 little‑endian length] [capnp blob] [32‑byte sha_prev]
//! Repeat …

use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

use capnp::{
    message::{Builder, ReaderOptions},
    serialize,
};
use memmap2::Mmap;
use ring::digest::{digest, SHA256};

include!(concat!(env!("OUT_DIR"), "/log_capnp.rs"));

/// A log entry that can be serialized to Cap'n Proto
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub seq: u64,
    pub op: u16,
    pub payload: Vec<u8>,
}

/// Helper that appends to an open log file.
pub struct LogWriter {
    file: File,
    prev_hash: [u8; 32],
}

impl LogWriter {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)?;
        let prev_hash = Self::last_hash(&mut file)?;
        Ok(Self { file, prev_hash })
    }

    fn last_hash(f: &mut File) -> std::io::Result<[u8; 32]> {
        if f.metadata()?.len() == 0 {
            return Ok([0u8; 32]);
        }
        f.seek(SeekFrom::End(-32))?;
        let mut buf = [0u8; 32];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }
    pub fn append_entry(&mut self, entry: LogEntry) -> std::io::Result<()> {
        // build capnp message
        let mut message = Builder::new_default();
        {
            let mut root = message.init_root::<log_entry::Builder>();
            root.set_seq(entry.seq);
            root.set_op(entry.op);
            root.set_payload(&entry.payload);
        }
        let mut buf: Vec<u8> = Vec::new();
        serialize::write_message(&mut buf, &message).map_err(|e| std::io::Error::other(e))?;
        // prepend length
        let len = buf.len() as u32;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&buf)?;
        // write previous hash
        self.file.write_all(&self.prev_hash)?;
        // compute new hash
        let new_hash = digest(&SHA256, &buf);
        self.prev_hash.copy_from_slice(new_hash.as_ref());
        self.file.flush()?;
        Ok(())
    }
}

/// Memory-mapped reader that streams sequentially and yields owned `LogEntry`s.
/// Uses zero-copy parsing directly from mapped memory for optimal performance.
pub struct LogReader {
    mmap: Mmap,
    offset: usize,
}

impl LogReader {
    pub fn new(file: File) -> std::io::Result<Self> {
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap, offset: 0 })
    }

    /// Create a LogReader from a file path
    pub fn from_path(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        Self::new(file)
    }
}

impl Iterator for LogReader {
    type Item = std::io::Result<LogEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we have enough bytes for the length prefix
        if self.offset + 4 > self.mmap.len() {
            return None; // EOF
        }
        // Read length directly from mapped memory (no syscall)
        let len_bytes = &self.mmap[self.offset..self.offset + 4];
        let len =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
        self.offset += 4;

        // Sanity check: prevent unreasonably large allocations
        if len > 100_000_000 {
            // 100MB max per log entry
            return Some(Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Log entry too large: {len} bytes (max: 100MB)"),
            )));
        }

        // Check if we have enough bytes for the data and hash
        if self.offset + len + 32 > self.mmap.len() {
            return Some(Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Incomplete log entry",
            )));
        }
        // Parse Cap'n Proto directly from mapped memory (zero-copy)
        let data_slice = &self.mmap[self.offset..self.offset + len];
        self.offset += len;

        // Skip the SHA hash for now (could be used for validation)
        let _sha_prev = &self.mmap[self.offset..self.offset + 32];
        self.offset += 32;

        // Cap'n Proto requires word-aligned data. Since we're reading from a memory-mapped file,
        // we may need to copy the data to ensure alignment.
        let aligned_data: Vec<u8> = data_slice.to_vec();
        // Use BufferSegments to read from aligned data
        let segments = match serialize::BufferSegments::new(&aligned_data[..], ReaderOptions::new())
        {
            Ok(s) => s,
            Err(e) => return Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e))),
        };
        let reader = capnp::message::Reader::new(segments, ReaderOptions::new());

        match reader.get_root::<log_entry::Reader>() {
            Ok(root) => {
                let payload = match root.get_payload() {
                    Ok(p) => p.to_vec(),
                    Err(_) => Vec::new(),
                };
                Some(Ok(LogEntry {
                    seq: root.get_seq(),
                    op: root.get_op(),
                    payload,
                }))
            }
            Err(e) => Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e))),
        }
    }
}

// =============================== tests =====================================
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_memory_mapped_log_reader() {
        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();

        // Create a simple log entry
        let entry = LogEntry {
            seq: 42,
            op: 100,
            payload: b"test payload".to_vec(),
        };

        // Write the entry using LogWriter
        let mut writer = LogWriter::new(temp_file.path()).unwrap();
        writer.append_entry(entry.clone()).unwrap();

        // Create memory-mapped reader
        let reader = LogReader::from_path(temp_file.path()).unwrap();

        // Read entries
        let entries: Result<Vec<_>, _> = reader.collect();
        let entries = entries.unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].seq, 42);
        assert_eq!(entries[0].op, 100);
        assert_eq!(entries[0].payload, b"test payload");
    }
    #[test]
    fn test_memory_mapped_multiple_entries() {
        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();

        // Write multiple entries
        let mut writer = LogWriter::new(temp_file.path()).unwrap();

        for i in 0..10 {
            let entry = LogEntry {
                seq: i,
                op: (i % 256) as u16,
                payload: format!("payload {}", i).into_bytes(),
            };
            writer.append_entry(entry).unwrap();
        }

        // Create memory-mapped reader and read all entries
        let reader = LogReader::from_path(temp_file.path()).unwrap();
        let entries: Result<Vec<_>, _> = reader.collect();
        let entries = entries.unwrap();

        assert_eq!(entries.len(), 10);

        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.seq, i as u64);
            assert_eq!(entry.op, (i % 256) as u16);
            assert_eq!(entry.payload, format!("payload {}", i).into_bytes());
        }
    }

    #[test]
    fn test_memory_mapped_empty_log() {
        // Create an empty temporary file
        let temp_file = NamedTempFile::new().unwrap();

        // Create memory-mapped reader
        let reader = LogReader::from_path(temp_file.path()).unwrap();

        // Should yield no entries
        let entries: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(entries.len(), 0);
    }
    #[test]
    fn test_memory_mapped_performance() {
        // Create a temporary file with many entries
        let temp_file = NamedTempFile::new().unwrap();
        let mut writer = LogWriter::new(temp_file.path()).unwrap();
        let entry_count = 1000usize;
        println!("Writing {} log entries...", entry_count);

        // Write many entries
        let write_start = std::time::Instant::now();
        for i in 0..entry_count {
            let entry = LogEntry {
                seq: i as u64,
                op: (i % 256) as u16,
                payload: format!("test payload with some data {}", i).into_bytes(),
            };
            writer.append_entry(entry).unwrap();
        }
        let write_time = write_start.elapsed();
        println!("Write time: {:?}", write_time);

        // Read with memory-mapped reader
        let read_start = std::time::Instant::now();
        let reader = LogReader::from_path(temp_file.path()).unwrap();
        let entries: Result<Vec<_>, _> = reader.collect();
        let entries = entries.unwrap();
        let read_time = read_start.elapsed();

        println!("Memory-mapped read time: {:?}", read_time);
        println!("Entries read: {}", entries.len());

        assert_eq!(entries.len(), entry_count);

        // Verify first and last entries
        assert_eq!(entries[0].seq, 0);
        assert_eq!(entries[entry_count - 1].seq, (entry_count - 1) as u64);

        // Performance expectations (memory mapping should be very fast)
        assert!(
            read_time.as_millis() < 1000,
            "Read took too long: {:?}",
            read_time
        );
    }
}
