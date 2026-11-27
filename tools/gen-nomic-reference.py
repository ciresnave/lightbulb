#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reference embeddings for the F3 acceptance gate (docs/F3-EMBEDDINGS-DESIGN.md, test plan 1).

Runs nomic-ai/nomic-embed-text-v1.5 (Apache-2.0, Nomic AI; https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
from its fp32 ONNX export with onnxruntime. NOT sentence-transformers with trust_remote_code: that would execute
third-party Hub code (pm-kernels condition 3). The model's own sentence-transformers pipeline is Transformer +
mean Pooling (modules.json, 1_Pooling/config.json), so the reference is: masked mean of last_hidden_state, then
L2 (cosine does not care about the norm; Lightbulb's spec normalises).

The token ids come from tokenizer.json via the `tokenizers` library, the same file Lightbulb loads, so tokenizer
agreement is NOT tested by this fixture (it shares the method); only the encoder and the pooling are.

    python tools/gen-nomic-reference.py C:/lightbulb-models/nomic-embed-text-v1.5 tests/fixtures/nomic-embed-text-v1.5-reference.json
"""
import json, sys
import numpy as np
import onnxruntime as ort
import tokenizers

SENTENCES = [
    "search_document: The quick brown fox jumps over the lazy dog.",
    "search_query: what is the capital of France?",
    "search_document: Paris is the capital and most populous city of France, with an estimated population of 2,102,650 residents in January 2023.",
    "search_query: Cafe au lait, naive resume, and Zurich: accents and CAPITALS are lower-cased by the tokenizer.",
    "search_document: fn main() { println!(\"hello, world\"); } // a line of Rust, with punctuation: {}[]()<>;",
]

def main(model_dir, out_path):
    tok = tokenizers.Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    sess = ort.InferenceSession(f"{model_dir}/onnx/model.onnx", providers=["CPUExecutionProvider"])
    rows = []
    for text in SENTENCES:
        ids = tok.encode(text, add_special_tokens=True).ids
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "token_type_ids": np.zeros((1, len(ids)), dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }
        hidden = sess.run(["last_hidden_state"], feed)[0][0]  # [seq, 768]
        mean = hidden.mean(axis=0)
        vec = mean / np.linalg.norm(mean)
        rows.append({"text": text, "tokens": len(ids), "ids": ids, "embedding": [float(x) for x in vec]})
    json.dump({
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "revision": "e9b6763023c676ca8431644204f50c2b100d9aab",
        "licence": "apache-2.0 (Nomic AI), https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
        "source": "onnx/model.onnx fp32 via onnxruntime; mean pooling over all tokens; L2-normalised",
        "versions": {"onnxruntime": ort.__version__, "tokenizers": tokenizers.__version__, "numpy": np.__version__},
        "command": "python tools/gen-nomic-reference.py <model dir> <out.json>",
        "dim": 768,
        "rows": rows,
    }, open(out_path, "w"), indent=1)
    print("wrote", out_path, [r["tokens"] for r in rows])

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
