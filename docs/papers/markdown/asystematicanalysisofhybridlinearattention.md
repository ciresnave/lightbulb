5
2
0
2

l
u
J

8

]
L
C
.
s
c
[

1
v
7
5
4
6
0
.
7
0
5
2
:
v
i
X
r
a

A Systematic Analysis of Hybrid Linear Attention

Dustin Wang*1, Rui-Jie Zhu*1,2,
Steven Abreu3, Yong Shan2, Taylor Kergan1, Yuqi Pan2,4, Yuhong Chou5, Zheng Li2,
Ge Zhang2,6,�, Wenhao Huang2, Jason Eshraghian1,�

1UC Santa Cruz, 2ByteDance Seed, 3University of Groningen, 4CASIA, 5PolyU, 6M-A-P

?Equal Contribution, �Corresponding authors

Abstract

Transformers face quadratic complexity and memory issues with long sequences, prompting the
adoption of linear attention mechanisms using fixed-size hidden states. However, linear models
often suffer from limited recall performance, leading to hybrid architectures that combine linear and
full attention layers. Despite extensive hybrid architecture research, the choice of linear attention
component has not been deeply explored. We systematically evaluate various linear attention
models across generations�vector recurrences to advanced gating mechanisms�both standalone
and hybridized. To enable this comprehensive analysis, we trained and open-sourced 72 modelsa:
36 at 340M parameters (20B tokens) and 36 at 1.3B parameters (100B tokens), covering six linear
attention variants across five hybridization ratios. Benchmarking on standard language modeling
and recall tasks reveals that superior standalone linear models do not necessarily excel in hybrids.
While language modeling remains stable across linear-to-full attention ratios, recall significantly
improves with increased full attention layers, particularly below a 3:1 ratio. Our study highlights
selective gating, hierarchical recurrence, and controlled forgetting as critical for effective hybrid
models. We recommend architectures such as HGRN-2 or GatedDeltaNet with a linear-to-full
ratio between 3:1 and 6:1 to achieve Transformer-level recall efficiently.

Date: July 10, 2025
Correspondence: Ge Zhang at zhangge.eli@bytedance.com, Jason Eshraghian at jsn@ucsc.edu
Opensourced Models: Page

aOur model training and inference are both implemented using the flash-linear-attention library.

1

Introduction

The Transformer architecture [1] has become the dominant network design for Large Language Models (LLMs).
However, its quadratic complexity in sequence length for a single forward pass, O(L2), and the linearly
growing Key-Value (KV) cache in full attention mechanisms [1], become problematic as sequences lengthen.
To address this, numerous alternative approaches have emerged, with linear complexity models representing
one of the most promising directions [2]. These models compress the linearly growing KV cache into a static
vector or matrix�a hidden state�thereby achieving O(L) overall complexity [2]. This innovation tackles
both the quadratic complexity problem and the issue of an expanding KV cache.

To better utilize these fixed hidden states, linear complexity models have evolved through roughly three
generations. The first generation was characterized by vector-level hidden states derived from traditional

1

Recurrent Neural Networks (RNNs) [3]. The second generation advanced this by employing outer products
for state expansion, creating 2D hidden states [4]. The third generation further refined these models by
introducing delta rule state transitions [5], replacing the hidden state decay mechanisms used in the first
two generations. Each successive generation has generally yielded improved performance in both common
reasoning and retrieval benchmarks.

Despite these advancements, the same fixed hidden state that enables linear complexity also introduces
limitations in recall capabilities; linear attention models typically demonstrate poorer recall performance [6�8].
To mitigate this deficiency, Hybrid Linear Attention models have emerged [9]. This approach interleaves
full attention layers with linear attention layers. Although the overall complexity remains quadratic due to
the inclusion of full attention, these hybrid models deliver significantly better recall performance [10, 11]
and have successfully scaled to 400 billion parameters while maintaining competitive common reasoning
performance [11].

However, a critical observation in the development of hybrid models is that while the choice of the full
attention mechanism is generally consistent, the selection of the linear attention component appears relatively
arbitrary. Existing research predominantly focuses on ablation studies concerning the ratio of full to linear
attention layers, rather than on the specific architectural choices within the linear models themselves [10�13].
These studies often implicitly assume that a superior standalone linear model will inherently lead to a better-
performing hybrid model. We ask: is this assumption valid? Furthermore, the determination of the optimal
ratio in hybrid models is frequently guided by minimizing loss values, which primarily reflects performance
on short-text modeling tasks, potentially overlooking the more critical aspect of recall performance. Finally,
Ref. [14] has demonstrated that simply increasing the state size of linear attention models yields diminishing
returns. Given this limitation, we hypothesize that performance improvements may instead depend on a
model�s ability to effectively manage and retrieve information from its stored memory�a capability we examine
through our language modeling and recall evaluations.

In this work, we conduct a comprehensive investigation into a wide array of linear attention architectures
spanning all three generations, analyzing their language modeling and recall performance when hybridized.
We also benchmark pure linear attention models as a baseline and examine the impact of varying the ratio
between full and linear attention layers. Our research aims to answer three primary questions: 1) Does a
linear attention model with better standalone performance necessarily translate to a superior hybrid model
architecture? 2) What aspects are more significantly influenced by the ratio of full to linear attention layers:
overall language modeling (LM) performance or recall capabilities? 3) Which architectural components or
design principles within linear attention models are most crucial for optimizing both LM performance and
recall?

This paper makes three primary contributions.

1. We provide a comparative analysis of various linear attention architectures from all three recognized
generations, evaluating both their standalone language modeling and their recall performance and also
when integrated into hybrid models;

2. We examine the assumption that improvements in standalone linear attention models directly correlate
with enhanced performance in hybrid architectures, offering evidence-based insights into this relationship;

3. our work analyzes how the ratio of full attention to linear attention layers in hybrid models distinctly
affects overall language modeling capabilities versus specific recall performance, while also identifying
key architectural components within linear attention models that are crucial for these metrics.

2 Related work

Linear complexity language models. Early efforts to tame the quadratic cost of softmax attention rewrote it
as an associative recurrent update, giving linear attention with O(L) compute and memory [2]. Subsequent
fast-weight and kernel views improved stability and throughput [15]. A parallel line scales recurrent networks
themselves: RWKV introduces a receptance-weighted key�value recurrence that matches Transformer perplexity
while keeping a constant-size cache [16]; RetNet adds exponential decay for longer retention [17]; Mamba

2

and its successor Mamba-2 treat input-conditioned state-space models as RNNs, closing most perplexity gaps
on open benchmarks with linear-time generation [18]. Gated extensions (HGRN2, Gated DeltaNet) further
refine selective forgetting with per-token gates [4, 19]. Despite these gains, purely linear/recurrent models
still underperform Transformers on long-context retrieval and in-context learning [20�22], motivating hybrid
designs. See Table 1 for a more detailed overview of these models.

Hybrid architectures. The dominant remedy is to interleave a small number of full-attention layers with
many linear-time layers, yielding Transformer-like recall at a fraction of the KV-cache cost. Small-scale hybrid
models include Hybrid-H3 [23] using only two attention layers with up to 2.7B parameters, StripedHyena [13]
using a 1:1 ratio with 7B parameters, RecurrentGemma [24] using a 2:1 ratio and sliding window attention
with 2B and 9B parameters, and Mamba-2-Hybrid [20] using a 4:1 ratio with 8B parameters. Large-scale
demonstrations include Jamba (7:1 Mamba:Transformer) [9, 12] and production models at Character.ai and
MiniMax that converge on a 6�7:1 ratio [11, 25]. Variants trade capacity for memory: Zamba shares a single
global-attention block across many recurrent blocks [26, 27]; Samba keeps complexity strictly linear by pairing
Mamba with sliding-window attention (SWA) [28]. Beyond block-wise mixing, prefill�decode hybrids such
as YOCO compress all keys during an O(L) prefill pass and reuse them during decoding, slashing cache by
layer-count [29, 30]. A finer granularity is head-wise mixing: Hymba allocates some attention heads to
�
softmax and the rest to state-space updates within the same layer, halving cache while preserving accuracy
[10]. Empirically, long-context quality rises steeply once a few full-attention blocks or heads are present, after
which perplexity plateaus; thus the linear:full ratio primarily controls recall, whereas language modeling loss
is comparatively insensitive.

Principled hybrid architecture design While these empirical studies demonstrate the viability of hybrid
approaches, recent work has sought more principled design methodologies. The STAR framework [31] uses a
unified mathematical foundation for linear attention mechanisms to enable automated architecture synthesis
[32] propose using synthetic
through an evolutionary optimization method. Complementarily, Poli et al.
tasks to isolate specific capabilities and understand component interactions, finding that optimal architectures
leverage specialized layers through hybrid topology. These frameworks suggest that hybrid design can move
beyond empirical ratio exploration toward more grounded optimization of component synergies.

3 Method

3.1 Linear Attention and its variants

This section formalizes the three generations of linear�time attention mechanisms that motivate our study and
explains why we benchmark the specific representatives HGRN (first generation) and HGRN-2, GLA, RetNet
(second generation). For each generation we summarize the core update rule, highlight the architectural
property that distinguishes it from its predecessors, and discuss its implications for hybrid design. Table 1
gives a unified algebraic view of the mechanisms introduced below.

3.2 From vector recurrence to controlled forgetting
Generation 1 � gated vector recurrence. Early linear models collapsed the per-token key�value cache into
a single d-dimensional vector that is updated additively and modulated by a learned gate. Formally,
(0, 1)d enables
ht = ?t
selective retention without the vanishing gradients that plagued classical RNNs, yet the scalar state vector
still struggles to store multiple competing memories.

qt (HGRN; The element-wise gate ?t = f?(x1:t)

vt, ot = ht

ht?1 + (1

?t)

?

?

?

?

?

Generation 2 � outer-product state with decay. To increase capacity, second-generation architectures promote
Rd�n, typically n = d, accumulating rank-one outer products while applying
the hidden state to a matrix St
?
t , ot = Stqt. The form of the decay distinguishes three representative
a decay gate: St = St?1
t ) + vtk?
(1??
models. (i) HGRN-2 ties the gate across keys and values, i.e. ?t = ?t1, yielding a hierarchical separation
between slowly changing coarse memory and rapidly updated token detail [4].
(ii) GLA lets ?t be fully
data-dependent, learned per time-step and per channel, providing maximal flexibility but higher parameter

?

3

Model

Update rule (St)

Read-out (ot)

Vector-valued hidden state (classical / gated RNNs)

HGRN[4]
Hawk (RG-LRU)[33]

ht = ?t ? ht?1 + (1 ? ?t) ? vt
ht = rt ht?1 + it ? xt

ot = ht ? qt
ot = ht ? qt

Matrix-valued state via outer products

RetNet/Lightning [11, 17]
GLA[34]
Mamba-2[35]
RWKV-6[36]
HGRN-2/MetaLA[4, 37]

St = ? St?1 + vtk?
t
t ) + vtk?
St = St?1 ? (1??
t
St = ?t St?1 + vtk?
t
St = St?1Diag(?t) + vtk?
t
St = St?1Diag(?t) + vt(1 ? ?t)?

ot = Stqt
ot = Stqt
ot = Stqt
ot = (St?1 + (d ? vt)k?
ot = Stqt

t )qt

Delta-rule / controlled-forgetting family

DeltaNet[5]
Gated DeltaNet[19]

St = St?1(I ? ?tktk?
St = ?t St?1(I ? ?tktk?

t ) + ?tvtk?
t

t ) + ?tvtk?
t

ot = Stqt
ot = Stqt

Table 1 Unified comparison of linear-time attention mechanisms across three �generations�. 1st Gen keeps a vector ht ? Rd
with element-wise gating e.g. HGRN and Hawk, so the read-out is a Hadamard product (ht ? qt). 2nd Gen stores a matrix
St ? Rd�n via outer-product additions plus multiplicative decay, including GLA, RetNet, RWKV-6, HGRN-2 and Mamba. 3rd
Gen introduces delta-rule controlled forgetting that explicitly erases stale content (DeltaNet family). vt, kt, qt are value, key and
query projections; ?t, ?t, rt, it are gates; ? is the Hadamard product.

Figure 1 Three �generations� of linear-attention state updates. Generation 1 (left): Gated Vector Recurrence keeps a single
vector ht ? Rd. Each step first decays the previous state by an element-wise gate ?t (grey), then adds the gated input vt (pink).
Update and memory are both O(d). Generation 2 (centre): Outer-Product State with Decay stores a full matrix St ? Rd�n. A
diagonal decay mask (1 ??
t is written (pink/blue). Cost rises to
O(d2). Generation 3 (right): ?-Rule Controlled Forgetting first erases the slice of St?1 aligned with key kt using the projector
I ? ?tktk?
t . This selective forget-then-write preserves capacity while
matching the O(d2) compute and memory of Generation 2. The arrow at the bottom emphasizes the historical progression from
minimal to maximal recall capability.

t ) scales every column, after which a rank-1 outer product vtk?

t (cells with 1), then writes the new outer product ?tvtk?

cost [34]. (iii) RetNet fixes ?t = ? with ?
decay and the fastest inference but no content adaptivity [17].

?

(0, 1) learned once and shared across positions, giving the simplest

Generation 3 � delta-rule controlled forgetting. The third generation abandons per-channel (diagonal) decay
and instead applies a rank-1 dense transition that first erases any content aligned with the current key and

4

+1?+then writes the new association:

St = St?1

(cid:0)I

?tktk?
t

(cid:1) + ?tvtk?
t ,

?t

(0, 1).

?

?

?tktk?
t

?
couples all channels, information can flow across dimensions, alleviating
Because the projector I
state crowding and markedly boosting long-range recall. The update is mathematically identical to a single
stochastic-gradient step on an online least-squares objective�hence the name �delta rule��the hidden state
behaves similar to a fast, continually trained associative memory. This interpretation also clarifies why
models such as DeltaNet and Gated DeltaNet [5, 19] can match Transformer perplexity and exceed it on
recall-heavy tasks: each rank-1 correction injects quadratic feature interactions, giving the network a nonlinear
expressiveness that standard soft-max attention (a TC0 circuit) lacks, yet without sacrificing the O(L) time
and memory footprint.

3.3 Why these representatives?

Our main objective is to understand the roles of state size and state management in hybrid stacks. For
analysis, we pick one representative for each linear attention family and treat it as the �spokesperson" for a
cluster of closely related variants:

� HGRN (Gen-1) � the strongest vector-state model on language tasks, providing a clean, low-capacity

baseline against which all matrix variants can be compared.

� RetNet (Gen-2, fixed decay) � its single scalar decay factor ? is shared across positions and channels.
Lightning uses the identical mechanism, so RetNet serves as the representative for all fixed-gate
outer-product models.

� GLA (Gen-2, data-dependent gates) � applies a fully learned per-token diagonal gate. Mamba-2 reduces
this to a data-dependent scalar gate, and RWKV-6 modifies only the read-out while keeping the same
gated update; all three therefore live on the same design axis, with GLA chosen as the most expressive
exemplar.

� HGRN-2 (Gen-2, tied gate / hierarchical) � couples keys and values through a single gate, realizing a
two-scale hierarchy at minimal cost. MetaLA adopts the identical tied-gate update, so HGRN-2 stands
in for this class.

These four models span the spectrum from fixed through data-dependent to hierarchically shared gating,
giving us a controlled way to study how each strategy interacts with the sparse set of full-attention layers in a
hybrid model. Generation-3 delta-rule networks (DeltaNet, Gated DeltaNet) are surveyed only briefly, as they
are currently the sole public implementations of their family; this choice keeps our core comparison focused
on the gating�versus�state-size trade-offs within Generations 1�2.

3.4 Hybridization Strategy

Figure 2 sketches the hybrid stack used throughout our experiments. Each input sequence is first mapped to
embeddings, after which the network applies a repeating block of two conceptually different layers:

write St = St?1 + vtk?
t
sequence, these layers add

� Linear�attention layer. The layer carries a constant-size state St that is updated by an outer-product
(or the model-specific variant from Table 1). Because St does not grow with the
(Ld2)compute and a negligible amount of cache at inference time.
(cid:1)V refreshes

� Full-attention layer. A standard soft-max attention Atten(Q, K, V ) = softmax(cid:0)QK ?/?dk

(Ld) or

O

O

global token-to-token interaction and writes its keys and values to the KV cache.

We interleave the two kinds of layers in a fixed ratio r:1 (linear : full), repeating the composite block N times.
During training the stack is processed exactly like a Transformer; during autoregressive decoding only the
full-attention layers enlarge the cache, so the memory footprint is reduced by roughly a factor of r. In Section
4 we sweep r
(plus the pure cases) to quantify how this trade-off affects language-model loss
and recall capability.

24, 12, 6, 3

? {

}

5

Figure 2 Hybrid architecture: an embedding layer, N repetitions of linear-attention followed by full-attention, and a projection
head. Only the full-attention blocks grow the KV cache.

3.5 Benchmarking framework

For our comparative analysis, we benchmark a selection of hybridized linear attention mechanisms encompassing
all three generations on language understanding and recall tasks. Specifically, we evaluate model performance on
the following benchmarks: ARC-Challenge (ARC-c) [38], ARC-Easy (ARC-e) [38], HellaSwag [39], LAMBADA
(LMB) [40], OpenBookQA (OBQA) [41], and PIQA [42]. For recall-specific evaluation, we additionally use
RULER [43], a suite of benchmarks that assess a model�s performance on retrieval, multi-hop tracing,
aggregation, and question answering.

To systematically evaluate hybridization trade-offs, we test five configurations for each linear attention
mechanism: linear-to-full attention ratios of 24:1, 12:1, 6:1, 3:1, and a fully linear (pure) variant. As a baseline,
we also include a standard full-transformer model.

All models are pretrained on the fineweb-edu [44] dataset and using flash-linear-attention library [45]. The
340M-parameter models are trained on 20 billion tokens, and the 1.3B-parameter models are trained on 100
billion tokens. All models are optimized using the AdamW optimizer with a cosine learning rate schedule. For
the 340M-parameter models, we use a batch size of 50k tokens, accumulating a total of 20 billion training
tokens. For the 1.3B-parameter models, the batch size is set to 1 million tokens, reaching a total of 100 billion
training tokens. All evaluation benchmarks are conducted in a zero-shot manner without any task-specific
fine-tuning or prompt engineering.

4 Empirical Study

The empirical analysis then unfolds in three steps: First, we compare each linear-attention backbone in
isolation and in a hybridised form that interleaves full-attention layers, revealing whether strong standalone
results translate into strong hybrids. Next, keeping all other factors fixed, we vary the linear-to-full ratio (24:1,
12:1, 6:1, 3:1, and the pure linear case) and observe its effect on language-model accuracy and long-range recall
as measured by RULER. Finally, by contrasting model families that differ in gating, recurrence hierarchy,
and forgetting mechanisms, we identify the architectural ingredients that let hybrids match Transformer-level
recall with a much smaller KV cache.

6

EmbedLinear AttentionFull AttentionHead?�hidden layers?!=?!"#+?!?!$??????,?,?=?????????!?"V4.1 Standalone Performance versus Hybrid Performance

We investigate whether linear-attention architectures that excel in standalone evaluation maintain their
relative performance when combined with full attention in hybrid configurations. Tables 2 and 3 show that
this is not the case

At the 340M parameter scale, GatedDeltaNet achieves the highest standalone accuracy. However, when
hybridized, HGRN2 at a 6:1 linear-to-full attention ratio outperforms it, exceeding both the Transformer
baseline and the standalone leader by 1.2 percentage points.

Model

ARC-c ? ARC-e ? Hella ?

LMB ? OBQA ?

PIQA ? Avg ?

DeltaNet 6-1
GatedDeltaNet pure
GLA 6-1
HGRN 6-1
HGRN2 6-1
RetNet 24-1
Transformers

0.299
0.291
0.287
0.298
0.300
0.287
0.276

0.594
0.602
0.588
0.601
0.617
0.574
0.612

0.410
0.415
0.421
0.433
0.436
0.411
0.424

0.334
0.348
0.343
0.339
0.347
0.309
0.353

0.334
0.338
0.334
0.340
0.346
0.364
0.330

0.675
0.661
0.671
0.678
0.690
0.672
0.670

0.441
0.442
0.441
0.448
0.456
0.436
0.444

Table 2 340M model comparison of the best performing hybrid ratios from each linear attention variant as well as the transformer
for language modeling performance. Up arrows indicate that a higher number is better while down arrows indicate a lower
number is better. 340M models show insignificant recall capability due to their relatively small parameter count. We omit recall
benchmarks for this reason.

The same pattern emerges at 1.3B parameters: GatedDeltaNet leads in standalone evaluation but becomes
comparable with HGRN2 after hybridization. GatedDeltaNet at 24:1 and HGRN2 at 6:1 achieve equivalent
performance, with several other hybrid configurations within one percentage point.

Model

ARC-c ? ARC-e ? Hella ?

LMB ? OBQA ?

PIQA ? Avg ?

DeltaNet 6-1
GatedDeltaNet 24-1
GLA 3-1
HGRN 6-1
HGRN2 6-1
RetNet 6-1
Transformers

0.403
0.410
0.398
0.392
0.404
0.380
0.395

0.730
0.733
0.721
0.732
0.729
0.718
0.704

0.586
0.598
0.586
0.602
0.604
0.584
0.589

0.475
0.502
0.488
0.482
0.500
0.475
0.474

0.384
0.406
0.388
0.412
0.420
0.400
0.400

0.741
0.741
0.725
0.733
0.733
0.727
0.729

0.553
0.565
0.551
0.559
0.565
0.547
0.548

Table 3 1.3B parameter equivalent of Table 2

The relative performance of different architectures changes substantially when linear and full attention
mechanisms are combined, making standalone scores unreliable predictors of hybrid performance.

4.2 Impact of the Linear-to-Full Attention Ratio

We examine how the linear-to-full attention ratio affects language modeling and recall performance. Figure 3
illustrates the contrasting behavior of these two capabilities across different architectures and ratios. The left
panel shows that language modeling performance remains largely flat across all ratio configurations. Most
architectures cluster around 0.55-0.57 average LM score, with minimal variation.

The right panel reveals a markedly different pattern for recall performance. All architectures show a
clear upward trend as the proportion of full-attention layers increases. Performance rises from pure linear
configurations (around 0.1-0.35 RULER score) toward the full-attention baseline (dashed line at approximately
0.42). Notably, most architectures approach or exceed this baseline at the 3:1 ratio, with some like DeltaNet
and Gated-DeltaNet achieving their peak recall performance at this configuration. This trend demonstrates
that recall capability benefits substantially from increased full-attention allocation, unlike language modeling

7

Figure 3 Language performance and recall performance tasks are averaged and compared over varying ratios

performance. The reason behind this recall trend is likely because of the near infinite hidden state inherent in
the KV cache. With more full attention layers present, hybrid models perform better in recall tasks.

The behavior of these two capabilities has practical implications. Systems prioritizing language modeling
can operate efficiently with high linear-to-full ratios, while applications requiring long-range recall need more
balanced attention distributions.

We investigate the impact of the hybrid ratio further in Figure 4. Single-Key, Multi-key, and QA sub tasks
are noticeably affected by a changing ratio, with higher concentrations of full attention performing better
than lower concentrations. Common Word Extraction (CWE) and Frequent Word Extraction (FWE) does
not seem to correlate with the hybrid ratio.

These trends indicate that, apart from pure recall tasks, where the goal is to remember some information
presented in the past of a long sequence, the hybrid ratio does not make much difference. Practitioners can
freely optimize the linear to full attention ratio with minimal effect on language modeling performance.

4.3 Architectural Determinants of Hybrid Effectiveness

Our experiments indicate that hybrid performance hinges on three complementary architectural properties
rather than any single mechanism. Figure 3 and Tables 6 and 5 supply the quantitative backdrop; the analysis
below is necessarily observational. Ablations are left to future work.

Ratio DeltaNet GatedDeltaNet

GLA

HGRN HGRN2

RetNet Average

24-1
12-1
6-1
3-1
pure

0.472
0.476
0.478
0.475
0.472

0.468
0.478
0.481
0.478
0.487

0.470
0.474
0.471
0.478
0.459

0.488
0.482
0.486
0.483
0.459

0.488
0.488
0.494
0.489
0.486

0.472
0.472
0.472
0.468
0.471

0.476
0.478
0.480
0.479
0.472

Table 4 Language modeling performance results from each linear attention variant and ratio are averaged across 1.3B and 340M
parameter scales. Total average represents the aggregate score for each ratio. Pure refers to a pure Linear Attention model.

8

Figure 4 RULER sub task results based on ratio. RetNet and HGRN model families are omitted as their recall benchmark
results were insignificant.

Selective gating prevents catastrophic overwriting. Architectures that expose their hidden state to a learned,
token-wise gate�GatedDeltaNet and HGRN-2�consistently perform best in recall once hybridised, surpassing
the Transformer baseline by 2�5 percentage points at the optimal ratio. By contrast, RetNet�s fixed exponential
decay fails to protect long-range cues, yielding near-zero recall even when full-attention layers are added.

Hierarchical recurrence supplies multi-timescale context. The two-level pathway in HGRN-2 stores coarse
summaries at a slower update rate while the fast path handles token-level details. Relative to its single-path
ablation (HGRN), this hierarchy doubles recall and improves the LM�recall trade-off in Figure 3, suggesting
that widely spaced full-attention layers benefit from a recurrent hierarchy that can �latch� information between
them.

Controlled forgetting curbs state crowding. GatedDeltaNet realises controlled forgetting with an outer-product
delta rule, whereas HGRN-2 achieves the same goal through gated diagonal decay. Although only the former
subtracts stale content explicitly, both mechanisms prevent the unbounded accumulation that plagues purely
additive updates (e.g. GLA) and therefore attain strong recall scores.

Interaction with the linear to full ratio. Language-model accuracy varies by less than one percent across
ratios, but recall rises steadily as full-attention layers are added and saturates around 3:1. Architectures
lacking either gated or delta-style forgetting never reach Transformer-level recall, regardless of ratio, implying
that a suitable model architecture is necessary to attain results on par with the Transformer.

Practical guideline. A memory-constrained practitioner can select a gated, hierarchically recurrent model
with controlled forgetting�concretely, HGRN-2 or GatedDeltaNet�and devote roughly one quarter of the
layers to full attention. In our 1.3B-parameter setting this configuration achieved near-Transformer recall
while cutting KV-cache memory by a factor of four. Verifying the trend at larger scales remains future work.

9

Ratio DeltaNet GatedDeltaNet

GLA

HGRN2 Average

24:1
12:1
6:1
3:1
Pure

0.417
0.352
0.371
0.426
0.320

0.341
0.352
0.425
0.436
0.356

0.293
0.288
0.330
0.332
0.144

0.303
0.368
0.434
0.395
0.203

0.338
0.340
0.390
0.397
0.256

Table 5 Recall performance results from each linear attention variant and ratio are averaged. Total average represents the
aggregate score for each ratio.

4.4 Performance-Efficiency Pareto Front

We further explore the Pareto Front of performance and efficiency for all hybrid models considered herein. We
relate the FLOPs of the token mixers to the language modeling performance for each model and hybridization
setting (averaged over all benchmarks and both model sizes). Details for the FLOP calculation are presented
in Appendix A. Figure 5a shows this Pareto Front with the FLOPs calculated for the 1.3B parameter models
and a sequence length of 4,096. The Pareto Front of optimality is shown with a dashed line. It is immediately
evident that the pure HGRN model, due to its vector-sized state, uses orders of magnitude fewer FLOPs than
all other models. The full transformer model is on the opposite end of the Pareto Front, representing the
highest performance and lowest efficiency. The middle of the Pareto Front is occupied by the HGRN 24:1
model and the HGRN2 6:1 model.

(a) Sequence length L = 4096.

(b) Sequence length L = 32768.

Figure 5 Performance-Efficiency Pareto Front where FLOPs are calculated for the 1.3B parameter models with two different
sequence lengths. Note that the y-axis is inverted so that the Pareto Front can be clearly seen towards the top-right.

Figure 5b shows the same models with the FLOPs calculated for a sequence length of 32,768. This pushes the
efficiency of models with attention layers further down. The two ends of the spectrum�the pure HGRN and
the pure transformer�remain, but the lower-performance half of the spectrum is now also occupied by the
pure RetNet, pure HGRN2, and pure GatedDeltaNet (in increasing order of performance).

However, it is important to note that our efficiency metric�FLOPs in the forward pass�do not directly translate
into throughput or latency on conventional hardware. Specifically, the pure HGRN model is the most efficient
model in our analysis but this efficiency may not be reflected in throughput on modern GPUs. Although
the HGRN requires fewer operations in total, it still requires a comparable number of memory accesses and
its element-wise vector operations may underutilize the GPU in comparison to matrix operations in other

10

0.4600.4650.4700.4750.4800.4850.4900.495Avg. language modeling performance1012101110101091 / FLOPs for token mixingPerformance-Efficiency Pareto FrontModelsDeltaNetGatedDeltaNetHGRNHGRN2RetNetGLATransformerHybridization24:112:16:13:1PureSoftmaxHybridization24:112:16:13:1PureSoftmax0.4600.4650.4700.4750.4800.4850.4900.495Avg. language modeling performance101410131012101110101 / FLOPs for token mixingPerformance-Efficiency Pareto FrontModelsDeltaNetGatedDeltaNetHGRNHGRN2RetNetGLATransformerHybridization24:112:16:13:1PureSoftmaxHybridization24:112:16:13:1PureSoftmaxtoken mixers. However, the HGRN�s reduced computational footprint may be accelerated on smaller GPUs or
neuromorphic hardware [46, 47].

4.5 Summary of Findings and Practical Guidelines

This section condenses our empirical study into clear, actionable insights for building memory-efficient
long-context language models.

� Hybrid quality cannot be inferred from standalone performance. GatedDeltaNet is strongest in purely
linear form, yet HGRN-2 delivers the best results once full-attention layers are added, showing that
standalone benchmarks are insufficient when selecting a hybrid backbone.

� Recall�not perplexity�determines the optimal linear:full mix. Moving from a 24:1 to a 3:1 ratio nearly

doubles RULER recall while shifting language-model loss by less than 1,

� Three architectural ingredients underpin effective hybrids. Selective gating, hierarchical recurrence, and
controlled forgetting jointly enable Transformer-level recall with a small KV cache; omitting any one of
these components degrades retrieval markedly.

� Recommended deployment recipe. Employ a gated, hierarchical backbone (e.g., HGRN-2 or Gated-
DeltaNet) with one soft-max attention layer for every 3�6 linear layers. In our 1.3,B-parameter setting
this achieves near-Transformer recall while shrinking the KV cache by a factor of 4�7.

Model

Multikey Multiquery Multivalue

Singlevalue CWE+FWE

QA

VT

Avg

DeltaNet-hybrid-3-1
GLA-hybrid-3-1
GatedDeltaNet-hybrid-3-1
HGRN2-hybrid-6-1
Transformers

0.349
0.313
0.293
0.358
0.332

0.471
0.22
0.542
0.509
0.464

0.361
0.196
0.502
0.507
0.472

0.791
0.784
0.875
0.66
0.559

0.155
0.063
0.271
0.412
0.284

0.391
0.342
0.3
0.364
0.412

0.191
0.234
0.024
0.014
0.066

0.426
0.365
0.440
0.434
0.423

Table 6 Model ratios from each family with the best average RULER results are shown here.

5 Conclusion

We delivered the first systematic comparison of three generations of linear-time attention mechanisms,
evaluated both in isolation and within hybrid architectures that interleave a minority of full-attention layers.
The study shows that, when equipped with gating, hierarchical recurrence, and controlled forgetting, linear
backbones can match Transformer-level recall at a fraction of the KV-cache cost; a 3:1�6:1 (linear:full) ratio
provides a practical balance between memory and quality.

Limitations and future work. Our analysis is confined to models up to 1.3B parameters, a 2,048-token context
window, and block-wise mixing ratios. Whether the observed trade-offs persist at the 10B+ scale, with 128k
token contexts, or under instruction-tuning and multilingual data remains open. Exploring finer-grained
hybrids (e.g., head-wise or dynamic routing) and automated architecture search (e.g., the STAR framework)
constitute promising directions for extending this work.

References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and

Illia Polosukhin. Attention is all you need, 2023.

[2] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Fran�ois Fleuret. Transformers are rnns: Fast

autoregressive transformers with linear attention, 2020.

[3] Aaron Voelker, Ivana Kaji?, and Chris Eliasmith. Legendre memory units: Continuous-time representation in

recurrent neural networks. Advances in neural information processing systems, 32, 2019.

11

[4] Zhen Qin, Songlin Yang, Weixuan Sun, Xuyang Shen, Dong Li, Weigao Sun, and Yiran Zhong. Hgrn2: Gated

linear rnns with state expansion, 2024.

[5] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the

delta rule over sequence length, 2025.

[6] Xuyang Shen, Dong Li, Ruitao Leng, Zhen Qin, Weigao Sun, and Yiran Zhong. Scaling laws for linear complexity

language models. arXiv preprint arXiv:2406.16690, 2024.

[7] Samy Jelassi, David Brandfonbrener, Sham M. Kakade, and Eran Malach. Repeat after me: transformers are
better than state space models at copying. In Proceedings of the 41st International Conference on Machine
Learning, ICML�24. JMLR.org, 2024.

[8] Jerome Sieber, Carmen Amo Alonso, Alexandre Didier, Melanie Zeilinger, and Antonio Orvieto. Understanding
the differences in foundation models: Attention, state space models, and recurrent neural networks. In The
Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[9] Jamba Team. Jamba-1.5: Hybrid transformer�mamba models at scale, 2024.

[10] Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zijia Chen, et al. Hymba: A hybrid-head architecture for

small language models, 2024.

[11] MiniMax AI Team. Minimax-01: Scaling foundation models with lightning attention, 2025.

[12] Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Shaked Meirom, Yoav Shoham, et al. Jamba: A hybrid

transformer�mamba language model, 2024.

[13] Michael Poli, Jue Wang, Stefano Massaroli, Jeffrey Quesnelle, Ryan Carlow, Eric Nguyen, and Armin Thomas.

StripedHyena: Moving Beyond Transformers with Hybrid Signal Processing Models, 12 2023.

[14] Kai Liu, Jianfei Gao, and Kai Chen. Scaling laws for rnn llm in long-context scenarios with state size. In Proceedings
of the 63rd Annual Meeting of the Association for Computational Linguistics. Association for Computational
Linguistics, 2025.

[15] Imanol Schlag, Kazuki Irie, and J�rgen Schmidhuber. Linear transformers are secretly fast weight programmers,

2021.

[16] Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin
Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemyslaw
Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom,
Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang,
Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. Rwkv: Reinventing rnns
for the transformer era, 2023.

[17] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei.

Retentive network: A successor to transformer for large language models, 2023.

[18] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2024.

[19] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule, 2025.

[20] Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali
Hatamizadeh, Sudhakar Singh, Deepak Narayanan, et al. An empirical study of mamba-based language models.
arXiv preprint arXiv:2406.07887, 2024.

[21] Jongho Park, Jaeseung Park, Zheyang Xiong, Nayoung Lee, Jaewoong Cho, Samet Oymak, Kangwook Lee, and
Dimitris Papailiopoulos. Can mamba learn how to learn? a comparative study on in-context learning tasks. arXiv
preprint arXiv:2402.04248, 2024.

[22] Yingfa Chen, Xinrong Zhang, Shengding Hu, Xu Han, Zhiyuan Liu, and Maosong Sun. Stuffed mamba: State

collapse and state capacity of RNN-based long-context modeling, 2024.

[23] Daniel Y Fu, Tri Dao, Khaled Kamal Saab, Armin W Thomas, Atri Rudra, and Christopher Re. Hungry hungry
In The Eleventh International Conference on

hippos: Towards language modeling with state space models.
Learning Representations, 2023.

12

[24] Aleksandar Botev, Soham De, Samuel L. Smith, Anushan Fernando, et al. Recurrentgemma: Moving past

transformers for efficient open language models, 2024.

[25] Character.AI Research Blog. Optimizing ai inference at character.ai.

https://research.character.ai/

optimizing-inference/, 2024.

[26] Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, and Beren Millidge.

Zamba: A compact 7b ssm hybrid model, 2024.

[27] Paolo Glorioso, Quentin Anthony, Yury Tokpanov, Anna Golubeva, Vasudev Shyam, et al. The zamba2 suite:

Technical report, 2024.

[28] Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. Samba: Simple hybrid state

space models for efficient unlimited context language modeling, 2024.

[29] Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and

Furu Wei. You only cache once: Decoder-decoder architectures for language models, 2024.

[30] Elliott Goldstein, Michael Goldblum, Minshuo Chen, and Tom Goldstein. Goldfinch: Tokencat compression for

memory-efficient hybrid rnn�transformer models, 2024.

[31] Armin W Thomas, Rom Parnichkun, Alexander Amini, Stefano Massaroli, and Michael Poli. STAR: Synthesis of

tailored architectures. In The Thirteenth International Conference on Learning Representations, 2025.

[32] Michael Poli, Armin W Thomas, Eric Nguyen, Pragaash Ponnusamy, Bj�rn Deiseroth, Kristian Kersting, Taiji
Suzuki, Brian Hie, Stefano Ermon, Christopher R�, Ce Zhang, and Stefano Massaroli. Mechanistic design
and scaling of hybrid architectures. In Proceedings of the 41st International Conference on Machine Learning,
ICML�24, 2024.

[33] Soham De, Samuel L Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba
Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, et al. Griffin: Mixing gated linear recurrences with
local attention for efficient language models. arXiv preprint arXiv:2402.19427, 2024.

[34] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers
with hardware-efficient training. In Proceedings of the 41st International Conference on Machine Learning (ICML),
2024.

[35] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured

state space duality. arXiv preprint arXiv:2405.21060, 2024.

[36] Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah,
Xingjian Du, Teddy Ferdinan, Haowen Hou, Przemys?aw Kazienko, Kranthi Kiran GV, Jan Koco?, Bart?omiej
Koptyra, Satyapriya Krishna, Ronald McClelland Jr., Jiaju Lin, Niklas Muennighoff, Fares Obeid, Atsushi Saito,
Guangyu Song, Haoqin Tu, Cahya Wirawan, Stanis?aw Wo?niak, Ruichong Zhang, Bingchen Zhao, Qihang Zhao,
Peng Zhou, Jian Zhu, and Rui-Jie Zhu. Eagle and finch: Rwkv with matrix-valued states and dynamic recurrence,
2024.

[37] Yuhong Chou, Man Yao, Kexin Wang, Yuqi Pan, Rui-Jie Zhu, Jibin Wu, Yiran Zhong, Yu Qiao, Bo Xu, and Guoqi
Li. Metala: Unified optimal linear approximation to softmax attention map. Advances in Neural Information
Processing Systems, 37:71034�71067, 2024.

[38] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.

Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018.

[39] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish

your sentence?, 2019.

[40] Denis Paperno, Germ�n Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle,
Marco Baroni, Gemma Boleda, and Raquel Fern�ndez. The lambada dataset: Word prediction requiring a broad
discourse context, 2016.

[41] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a

new dataset for open book question answering, 2018.

[42] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical

commonsense in natural language, 2019.

13

[43] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and

Boris Ginsburg. Ruler: What�s the real context size of your long-context language models?, 2024.

[44] Guilherme Penedo, Hynek Kydl�?ek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra,
Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural
Information Processing Systems, 37:30811�30849, 2024.

[45] Songlin Yang and Yu Zhang. Fla: A triton-based library for hardware-efficient implementations of linear attention

mechanism, 2024.

[46] Rui-Jie Zhu, Yu Zhang, Ethan Sifferman, Tyler Sheaves, Yiqiao Wang, Dustin Richmond, Peng Zhou, and Jason K.

Eshraghian. Scalable matmul-free language modeling, 2024.

[47] Steven Abreu, Sumit Bam Shrestha, Rui-Jie Zhu, and Jason Eshraghian. Neuromorphic principles for efficient
large language models on intel loihi 2. In First Workshop on Scalable Optimization for Efficient and Adaptive
Foundation Models, 2025.

14

A Forward-Pass FLOP Analysis of Token Mixers

This section derives closed-form expressions for the floating-point operations (FLOPs) required by the different
recurrent and linear-attention token mixers that are presented in this paper, and contrasts them with classical
softmax self-attention. All counts are per layer, forward pass only, aggregated over all heads. Projection
layers, (un-)embedding, residual additions, normalisation, and other feed-forward blocks are not included so
that every token mixer is compared on equal footing.

Notation and assumptions

� L � causal sequence length (tokens seen by the layer).

� H - number of heads in the model (this applies to all trained models except the HGRN-1 model where

H = 1 by design).

� d � width of a single head. Vectors within a single head live in Rd. For a model dimension dmodel with

H heads we have d = dmodel/H.

� One FLOP is a single floating-point multiply or add.

� Only the token-mixer is costed. Projections, MLPs, norms, residuals are identical across models and

therefore omitted.

� All counts below are per token, per layer, summed over all heads. Multiply by L for a sequence, and by

the number of layers to estimate a full network.

Model

Total FLOPs / token Complexity in L

Softmax self-attention

HGRN-1 (vector, H = 1)
Hawk (RG-LRU, vector)

RetNet / Lightning

Mamba-2

GLA

RWKV-6

HGRN-2 / MetaLA

DeltaNet

Gated DeltaNet

2Ldmodel

5dmodel
5dmodel

5

5

7

7

7

8

8

d2
model
H
d2
model
H
d2
model
H
d2
model
H
d2
model
H
d2
model
H
d2
model
H

?(L2dmodel)

?(Ldmodel)
?(Ldmodel)

?(cid:0)Ld2

?(cid:0)Ld2

?(cid:0)Ld2

?(cid:0)Ld2

?(cid:0)Ld2

?(cid:0)Ld2

?(cid:0)Ld2

model/H(cid:1)
model/H(cid:1)
model/H(cid:1)
model/H(cid:1)
model/H(cid:1)
model/H(cid:1)
model/H(cid:1)

Table 7 Per-token forward FLOP counts summed over all heads. Multiply by L to obtain the per-layer sequence cost. For
HGRN-1 the mixer is single-head by design (H = 1); all other models allow arbitrary H. Softmax attention is quadratic in L,
whereas the linear/recurrent mixers are linear.

Baseline: softmax self-attention The dot-product matrix QK ? is strictly lower-triangular under causal
masking. Forming and applying it costs (H L(L + 1)dhead) multiplies plus the same number of adds. Using
dmodel = Hdhead gives

FLOPssoftmax
Importantly the head count cancels, so the quadratic term depends only on the model width.

?

2L2dmodel.

(1)

Vector-state mixers (classical / gated RNNs) A single hidden vector ht
head therefore needs 5dhead FLOPs per token. Aggregating over H heads:

?

Rdhead is maintained per head. Each

FLOPsvector = 5Hdhead = 5dmodel.

(2)

15

For HGRN-1 and Hawk, the reference implementation is single-head; we therefore take H = 1 and dhead =
dmodel.

Matrix-state mixers (outer-product family) Each head carries a dense state St
costs k d2

Rdhead�dhead . If an update
FLOPs per head (where k is a constant that varies between different models), the layer total is:

?

head

FLOPsmatrix = kHd2

head = k

d2
model
H

.

(3)

Hence adding more heads reduces the cost of these mixers at fixed dmodel. Each �pass� over the dhead
matrix either multiplies all entries or adds a rank�1 outer product, each costing d2
passes per token per head gives:

dhead
FLOPs. Summing the

head

�

� RetNet / Mamba�2 : five passes

k=5.

?

� GLA, RWKV�6, HGRN�2 : two extra gating passes (seven total)

� DeltaNet (& Gated): one further forget/restore pass (eight total)

k=7.

k=8.

?

?

Summary Table 7 shows a comparison between all models presented in Table 1. Once head aggregation is
included the classical vector mixers (HGRN-1, Hawk) remain linear in L and independent of H. Outer-product
and Delta-rule mixers grow as d2
model/H�wider models prefer more heads from a FLOP viewpoint�yet are
still asymptotically cheaper than softmax self-attention for long sequences. Figure 6 shows how the FLOPs
scale with sequence lengths for different pure (non-hybrid) token mixers.

Figure 6 Relationship between the sequence length and the number of FLOPs required by different token mixers. Note that the
HGRN-2 and GLA overlap, see analysis in the text.

16

1024204840968192163843276865536Sequence Length (L)1071081091010101110121013FLOPsFLOPs vs. Sequence Length for Different ModelsDeltaNetGated DeltaNetHGRN-1HGRN-2RetNetGLATransformer
