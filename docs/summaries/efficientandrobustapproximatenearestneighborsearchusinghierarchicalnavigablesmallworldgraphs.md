Title: Efficient and Robust Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graphs
Authors: Yu. A. Malkov, D. A. Yashunin

TL;DR
HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor search index that builds a multi-layer proximity graph with randomized layer assignment. It yields logarithmic scaling for search complexity in practice and provides strong empirical recall and robustness across datasets.

Why it matters
Nearest neighbor search is a core primitive for similarity search, retrieval, and many ML systems. HNSW delivers high recall and low latency at scale with a relatively simple algorithm and tunable parameters, making it a practical choice for production retrieval systems.

Key takeaways

- Structure: multiple levels of proximity graphs; higher levels are sparser and provide long-range navigation; search starts at top layer and greedily descends.  
- Algorithms: insert, search-layer, and neighbor selection heuristics (SELECT-NEIGHBORS) which balance connectivity vs efficiency.  
- Complexity: logarithmic search complexity observed in practice; construction and memory trade-offs controlled via parameters (M, efConstruction, etc.).  
- Robustness & performance: shows strong performance vs prior NSW and other baselines like Faiss across standard datasets (SIFT, GloVe, deep descriptors).

Implementation steps

1. Implement the HNSW index with layer assignment using geometric distribution for level selection.  
2. Implement neighbor selection heuristics that prefer diverse neighbors to preserve connectivity.  
3. Tune core parameters: M (max neighbors), efConstruction (construction ef), ef (query ef) for target recall/latency tradeoffs.  
4. Provide serialization and memory-efficient representations; consider disk-backed or sharded variants for very large datasets.  
5. Benchmark against Faiss indices on representative datasets and tune parameters for production constraints.

Acceptance criteria

- Achieve target recall (e.g., 0.95) at query latency and memory budget targets when compared to baseline Faiss indexes.  
- Demonstrated scaling behavior with dataset size and documented parameter guidance.  
- Reproducible benchmarks and integrated tests validating insertion, search, and persistence.

# Efficient — HNSW: Hierarchical Navigable Small World Graphs

TL;DR:

Why it matters:

Key takeaways:

Implementation steps:

Acceptance criteria:
