title: Rethinking Link Prediction for Directed Graphs
source_pdf: [RethinkingLinkPredictionForDirectedGraphs.pdf]("C:\Users\cires\OneDrive\Desktop\books and courses\Machine Learning\RethinkingLinkPredictionForDirectedGraphs.pdf")
source_markdown: [rethinkinglinkpredictionfordirectedgraphs.md](../papers/markdown/rethinkinglinkpredictionfordirectedgraphs.md)
[Rethinking Link Prediction for Directed Graphs]

**Links:**  
[Original PDF](c:/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/RethinkingLinkPredictionForDirectedGraphs.pdf)  
[Source Markdown](../papers/markdown/rethinkinglinkpredictionfordirectedgraphs.md)

---

## TL;DR

This paper introduces DirLinkBench, a comprehensive benchmark for evaluating directed link prediction methods, and proposes SDGAE, a novel spectral directed graph auto-encoder that achieves state-of-the-art results. Key findings include the importance of dual embeddings, decoder design, feature inputs, and negative sampling strategies for robust directed link prediction.

## Why it matters

Directed link prediction is crucial for understanding and modeling relationships in real-world graphs (e.g., citation, social, and web networks). Existing benchmarks and methods often fail to capture the unique challenges of directionality, leading to inconsistent and unreliable evaluations. DirLinkBench standardizes evaluation and highlights the need for expressive, theoretically grounded models that preserve graph asymmetry and degree distributions.

## Key technical takeaways

- **DirLinkBench Benchmark:**
  - Seven real-world directed graph datasets (citation, co-purchase, web, social).
  - 16 baselines, seven metrics, standardized splits, and modular extensibility (PyTorch Geometric).
  - Introduces ranking-based metrics (Hits@20/50/100, MRR) for directed link prediction.
- **Unified Framework:**
  - Binary classification setup for directed edges, standardized feature initialization, and negative sampling.
  - Reveals flaws in prior benchmarks (label leakage, class imbalance, poor metric choice).
- **SDGAE Model:**
  - Revisits DiGAE, showing its convolution is equivalent to GCN on undirected bipartite graphs.
  - SDGAE uses polynomial spectral filters, learns adaptive coefficients, and achieves lower time complexity.
  - Outperforms prior methods on most datasets and metrics, especially in preserving directed degree distributions.
- **Empirical Insights:**
  - Decoder and loss function choices (BCE > CE, MLP-based decoders) strongly affect performance.
  - Feature quality (original features, in/out degrees) is critical; in/out degrees often outperform random features.
  - Negative sampling strategy impacts results; shared random samples per run are preferable.

## Implementation steps (Candle/Rust context)

1. **Dataset Preparation:**

- Download and preprocess DirLinkBench datasets (remove duplicates, self-loops, isolated nodes; retain largest component).
- Standardize splits: 80% train, 5% validation, 15% test; sample negatives as described.

2. **Model Architecture:**

- Implement SDGAE encoder: polynomial spectral graph filters on undirected bipartite representation.
- Use two MLPs for source/target embedding initialization; iterative propagation for polynomial coefficients.
- Decoder: MLP-based or inner product; use BCE loss for training.

3. **Training & Evaluation:**

- Train for up to 2000 epochs with early stopping; tune hyperparameters as in the paper.
- Evaluate using ranking-based metrics (Hits@100 primary, others for comparison).
- Compare against baselines (DiGAE, STRAP, MagNet, etc.) using shared splits and metrics.

4. **Benchmarking:**

- Report results for all metrics and datasets; analyze degree distribution preservation and negative sampling effects.

## Acceptance criteria

- SDGAE implementation matches the paper's architecture and propagation equations.
- Evaluation uses DirLinkBench datasets, splits, and metrics as described.
- Results demonstrate competitive or superior performance to baselines, especially in Hits@100 and degree distribution preservation.
- Code is modular, extensible, and reproducible (preferably using Candle and PyTorch Geometric for reference).
- Summary links to both the original PDF and markdown source.

# TL;DR

This paper critically examines the problem of link prediction in directed graphs, revealing theoretical and empirical flaws in current benchmarks and methods. It introduces a unified framework for directed link prediction, exposes issues like label leakage and class imbalance in existing setups, and proposes DirLinkBench—a new, standardized benchmark for fair and robust evaluation.

# Why it matters

- Directed graphs are fundamental in many real-world applications (e.g., citation, social, and web networks), but most link prediction research focuses on undirected graphs.
- Existing methods and benchmarks for directed link prediction are inconsistent, often flawed, and may overstate the effectiveness of complex models due to experimental artifacts.
- A standardized, fair, and reproducible benchmark is essential for progress in this area and for the development of reliable graph learning systems.

# Key technical takeaways

- **Unified Framework:** The paper formalizes a general encoder-decoder framework for directed link prediction, categorizing methods as single, dual, complex-valued, or gravity-inspired embeddings, and analyzes their expressiveness for asymmetry and graph reconstruction.
- **Theoretical Analysis:** Dual methods (with separate source/target or complex embeddings) are strictly more expressive for directed graphs than single-embedding methods, but practical benchmarks often fail to reflect this due to experimental flaws.
- **Empirical Findings:** Simple baselines like MLPs can outperform or match state-of-the-art methods in many current benchmarks, challenging assumptions about model complexity.
- **Benchmark Issues:** The paper identifies four major issues in existing benchmarks: (1) neglect of strong baselines, (2) label leakage, (3) class imbalance and poor metrics, and (4) lack of standardization in splits and features.
- **DirLinkBench:** Introduces a new benchmark with seven real-world datasets, standardized splits, feature handling, and ranking-based metrics (e.g., Hits@K, MRR), enabling fair and reproducible evaluation.

# Implementation steps (for Candle or similar Rust ML frameworks)

1. **Data Preparation:** Use the DirLinkBench datasets, ensuring removal of duplicates, self-loops, and isolated nodes, and apply the standardized train/val/test splits.
2. **Modeling:** Implement the encoder-decoder framework for directed link prediction, supporting dual, complex, and gravity-inspired embeddings. Ensure the decoder can handle directionality.
3. **Baselines:** Include simple MLP and classic GNN baselines for comparison, as well as more complex models.
4. **Evaluation:** Use ranking-based metrics (Hits@K, MRR, AUC, etc.) for model assessment, and avoid label leakage by strictly separating test edges during training and feature computation.
5. **Reproducibility:** Ensure all experiments use fixed random seeds and shared splits for fair comparison.

# Acceptance criteria

- [ ] Implementation supports all DirLinkBench datasets and standardized splits.
- [ ] Encoder-decoder models can be configured for single, dual, complex, and gravity-inspired embeddings.
- [ ] Evaluation uses ranking-based metrics and avoids label leakage.
- [ ] Baseline MLP and GNNs are included for comparison.
- [ ] Results are reproducible and comparable to those reported in the paper.
