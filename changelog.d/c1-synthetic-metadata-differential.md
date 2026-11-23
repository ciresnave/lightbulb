kind: C1 the metadata differential runs on synthetic GGUFs
---
The differential test between our GGUF metadata reader and the loaders (`our_metadata_agrees_with_the_loaders_on_synthetic_files`) now has a counterpart that needs no corpus: it writes an F32, Q8_0, Q4_0, Q6_K, templated and chain-output synthetic file and requires both readers to read each and agree key for key (exact count, so an unreadable file fails instead of being skipped). Adding an unreadable file turns it red. The corpus version stays `#[ignore]`d for real files.
