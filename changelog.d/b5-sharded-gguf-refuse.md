kind: sharded GGUF refused up front; mlmf crates 0.11.29
---
### Changed
`mlmf-hf-layout` 0.7.26 -> 0.11.29 and new `mlmf-gguf` 0.11.29 (ROADMAP B5). A split GGUF (`<prefix>-00001-of-0000N.gguf`) given to either GGUF loader now fails immediately with an error naming the shard, the count and the missing siblings, instead of a late missing-tensor error. File names only (`model_fuel/gguf_shards.rs`, 4 unit tests incl. an on-disk case and a non-split control); no shard is opened or merged. Loading a shard set still needs Fuel to accept one: all its GGUF entry points take a single path (recorded in ROADMAP B5). The licence list is regenerated. Verified: `cargo test --lib gguf_shards` 4 passed.
