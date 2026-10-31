# qwen35 chat template fixture

`chat_template.jinja` is the file of that name in
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B), downloaded with
`hf download Qwen/Qwen3.5-0.8B chat_template.jinja` on 2026-10-31, **unmodified**:
7755 bytes, sha256 `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`
(`tests/chat_template_qwen35.rs` pins both).

Licence: Apache-2.0, as declared by that repository (`LICENSE`, and `license: apache-2.0` in its
model card). Copyright belongs to the Qwen team; this crate claims none of it. It is carried only
to test our renderer, and is not compiled into any binary.

The expected renderings in `tests/chat_template_qwen35.rs` were produced by Python `jinja2` 3.1.6
using `transformers`' dialect (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`,
a `raise_exception` global) over the cases listed there. `.gitattributes` pins this directory to LF
so a CRLF checkout cannot change what the template renders.
