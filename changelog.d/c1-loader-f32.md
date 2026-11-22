kind: C1 loader_f32 tests on the synthetic checkpoint
---
The three `#[ignore]`d `loader_f32` tests (`every_weight_is_f32`, `f32_and_stock_loaders_agree`, `loader_surfaces_eos_and_tokenizer`) now run on the synthetic checkpoint instead of needing a downloaded TinyLlama; they assert loader plumbing, not trained values. Test-only.
