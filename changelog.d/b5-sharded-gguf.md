kind: sharded GGUF loads; Fuel 0.47.32 (ROADMAP B5)
---
### Added
- A split GGUF (`<prefix>-00001-of-0000N.gguf`, llama.cpp `gguf-split`) now loads as one model: both GGUF loaders resolve any named shard to shard 1 and Fuel opens the whole set from it (fuel#660). An incomplete set is still refused up front, naming the shard given and the missing siblings. `/v1/models` lists a complete set once, as its shard-1 file name, and not shards 2 and up; the memory preflight sums the on-disk size and the resident estimate over every shard.

### Changed
- Fuel dependencies 0.46.2 -> 0.47.32 (the first crates.io release carrying sharded GGUF); `mlmf-*` follow to 0.11.29 through Fuel.
