# MIRIX: Multi-Agent Memory System for LLM-Based Agents

**Original PDF:** [MIRIXMultiAgentMemorySystemForLLMBasedAgents.pdf](../../../../Desktop/books%20and%20courses/Machine%20Learning/MIRIXMultiAgentMemorySystemForLLMBasedAgents.pdf)
**Original Markdown:** [mirixmultiagentmemorysystemforllmbasedagents.md](../papers/markdown/mirixmultiagentmemorysystemforllmbasedagents.md)

---

## TL;DR

MIRIX introduces a modular, multi-agent memory architecture for LLM-based agents, featuring six specialized memory components and a meta-manager for routing and retrieval. It enables agents to efficiently store, retrieve, and reason over multimodal, long-term, and structured user data, outperforming existing memory systems in both accuracy and storage efficiency.

## Why it matters (for Candle and reproducible ML)

- Candle and similar Rust ML libraries benefit from robust, modular memory systems for agentic workflows, benchmarking, and telemetry.
- MIRIX's architecture supports reproducible experiments by enabling persistent, structured memory across sessions and modalities.
- The system's focus on compositional memory, privacy, and efficient retrieval aligns with the needs of scalable, open-source ML infrastructure.

## Key technical takeaways

- **Six memory components:** Core, Episodic, Semantic, Procedural, Resource, and Knowledge Vault, each managed by a dedicated agent.
- **Meta Memory Manager:** Orchestrates routing, updates, and retrieval across all components.
- **Active Retrieval:** Automatically generates topics from user queries to fetch relevant memories, reducing reliance on parametric LLM knowledge.
- **Multi-agent workflow:** Parallel updates and retrievals, modular design for extensibility and integration.
- **Benchmarks:** Outperforms RAG and long-context baselines on multimodal (ScreenshotVQA) and conversational (LOCOMO) datasets, with dramatic improvements in accuracy and storage.
- **Privacy and marketplace:** Envisions encrypted, user-controlled memory as a digital asset, supporting decentralized sharing and monetization.

## Implementation steps (for Candle or similar)

1. **Design modular memory classes** for each component, with hierarchical and structured fields (e.g., episodic events, semantic concepts, procedural steps).
2. **Implement a meta-manager** to route updates and retrievals, coordinating parallel memory agents.
3. **Integrate active retrieval**: Topic generation from queries, retrieval across all memory types, and injection into agent prompts.
4. **Support multimodal input** (text, images, documents) and efficient storage (e.g., SQLite, cloud hybrid).
5. **Build privacy controls**: Encryption, fine-grained permissions, and decentralized storage options.
6. **Benchmark** against RAG and long-context baselines using multimodal and conversational datasets.
7. **Expose APIs** for agent interaction, memory updates, and retrieval.

## Acceptance criteria

- Modular, extensible memory system with six components and meta-manager.
- Demonstrated active retrieval and parallel update workflow.
- Support for multimodal input and efficient storage.
- Privacy-preserving features and user-controlled sharing.
- Benchmark results showing improved accuracy and storage efficiency over RAG/long-context baselines.
- APIs for agent interaction and reproducible ML experiments.

---

**For Candle:** MIRIX provides a blueprint for building robust, agentic memory systems in Rust ML libraries, supporting reproducible research, benchmarking, and advanced telemetry in open-source environments.
