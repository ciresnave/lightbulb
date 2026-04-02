5
2
0
2

l
u
J

0
1

]

G
L
.
s
c
[

1
v
6
9
9
7
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

Skip a Layer or Loop it?
Test-Time Depth Adaptation of Pretrained LLMs

Ziyue Li1, Yang Li, Tianyi Zhou
1Department of Computer Science, University of Maryland, College Park
litzy619@umd.edu, yli.ml.research@gmail.com, tianyi.david.zhou@gmail.com

Abstract

Can a pretrained neural network adapt its architecture to different inputs without
any finetuning? Do we need all layers for simple tasks, and are they adequate
for challenging tasks? We found that the layers of a pretrained large language
model (LLM) can be manipulated as separate modules to build a better and
even shallower model customized for each test sample. In particular, each layer
from the pretrained model can be skipped/pruned or repeated multiple times as
recurrent neural networks (RNN), and stacked with others in arbitrary orders,
yielding a chain-of-layers (CoLa) per sample. This compositional space greatly
expands the scope of existing works on looped/recurrent pretrained modules, layer
pruning, or early-exit networks. We develop a Monte Carlo Tree Search (MCTS)
protocol to explore and identify the optimal CoLa for each sample from math and
commonsense reasoning benchmarks. Compared to a static model of a fixed depth,
CoLa allows shortcut paths (fast thinking), recurrence of the same layer(s) (slow
thinking), and combining both, offering more flexible, dynamic architectures for
different inputs. We conduct an extensive analysis of the MCTS-optimized CoLa,
which leads to two key findings: (1) For >75% of samples with correct predictions
by the original LLM, we can find shorter CoLa, suggesting a large space for
improving inference efficiency; (2) For >60% of samples with originally incorrect
predictions, we can identify CoLa achieving correct predictions, suggesting a large
space of performance enhancement. Our results highlight the shortcomings of
using a fixed architecture of pre-trained LLMs for inference on different samples
and pave the way to unlock the generalization power of test-time depth adaptation.

1

Introduction

Modern Transformer architectures, such as large language models (LLMs), have demonstrated
unprecedented generalization capabilities on diverse downstream tasks such as reading comprehension
and reasoning [18]. However, their architectures always stay the same during inference for all different
tasks and test samples, despite their difference in difficulty, complexity, and distribution gap from
the training task/data [17, 6]. Is it necessary to apply all layers for simple tasks? On the contrary, is
the pretrained model deep enough to address challenging tasks that require sophisticated reasoning?
These raise the question regarding a new dimension of model generalization: without any further
training, can a pretrained neural network�s layers adapt to each task or sample by composing a model
with a different depth? Answers to this question are also critical to investigating the human-like
fast-slow thinking capability of pretrained LLMs: By skipping, repeating, and rearranging their
layers, can we dynamically adjust the depth and architecture of the model for each task?

Existing works on test-time rearrangement of layers mainly focus on a limited scope of depth
adaptation, e.g., early-exit neural networks [20, 2, 16] or layer pruning [8] that aims to remove the
redundant layers. They are inspired and supported by the similarity analysis of representations across

Preprint. Under review.

layers, a potential effect of residual connections in Transformers. On the other hand, to extend
the Transformer architecture to recurrent neural networks (RNN) [5], recent works such as looped
transformer [9] and recurrent depth [10] explore the potential of each layer or the whole model as an
RNN cell. By repeating the same pretrained layer(s) during inference, the architecture may perform
slower and deeper thinking than on training data, or handle inputs of unseen lengths. In addition, it
has not been studied to change the order of selected layers for test-time adaptation, which can provide
a more flexible space for the architecture search.

Figure 1: Test-time layer composition search space for CoLa. Starting from the original forward
path, each input can dynamically skip or recurrently reuse any layer(s) to construct a custom Chain-of-
Layers (CoLa). This joint space enables both layer pruning and recurrence, supporting fast-slow depth
adaptation and dynamic architecture generalization from a pretrained model without any finetuning.

In this paper, we extend the scope of depth adaptation and architecture generalization to a search
space that allows selecting, skipping, and repeating arbitrary layers when constructing a customized
chain-of-layers (CoLa) model out of a pretrained LLM, as shown in Figure 1. The space enables us
to remove redundant layers for each sample and integrate Transformer and RNNs architectures in
one model. Hence, it offers more flexibility to process different samples with a varying number of
layers. Specifically, we conduct an extensive empirical study to verify that there often exists a better
(smaller or more accurate, or both) CoLa than the original one for many test samples/tasks. To this
end, we propose a principal and efficient Monte-Carlo Tree search (MCTS) protocol to search CoLa
with better prediction and/or shallower depth (i.e., minimum necessary) than the original model for
each sample. Specifically, MCTS starts from an initial CoLa and edits it by skipping or repeating
layers for multiple rounds to find CoLa that maximizes an upper confidence bound (UCB) objective,
which performs an exploitation-exploration trade-off with a depth penalty.

We apply the proposed MCTS procedure to diverse benchmarks of math and commonsense reasoning
tasks on both pretrained and instruction-finetuned LLMs. Surprisingly, the simple MCTS consistently
finds higher-quality and/or shallower CoLa for most samples, without training any parameters. This
demonstrates the broad existence of better CoLa architectures than the pretrained one for individual
samples, underscoring the unexplored, compositional generalization of layers from pretrained models
as separate modules. Moreover, we quantitatively evaluate their corrected errors and the reduced
depths, highlighting a significant improvement in the depth-quality trade-off space. Furthermore, we
conduct fine-grained analyses of the utilization of each layer in MCTS-optimized CoLa and the impact
of task difficulty on the depth of CoLa, which shed novel insights into the redundancy and alignment
of model architecture to specific tasks. Our key findings and main contributions can be summarized as:

� We introduce a new dimension of generalization that turns a static pretrained LLM into
dynamic architectures of adaptive depths without training any parameter: for different test
samples/tasks, the pretrained layers can be skipped, repeated, and assembled to create better (more
accurate and/or shallower) CoLa models without further training.

� We develop an MCTS protocol for efficient architecture search of CoLa with adaptive depth
for each sample. In-depth analysis of patterns in the achieved CoLa models sheds critical insights
into the importance and redundancy of layers at different depths of pretrained/finetuned models
of different sizes, which also vary for tasks at different difficulty levels.

2 Related Work

Layer Pruning and Early-Exit Neural Networks Many works aim to accelerate large Transformers
by statically pruning weights or dynamically halting computation. Static pruning typically removes re-
dundant neurons, heads, or layers after training. For example, [15] demonstrate that a significant frac-
tion of BERT�s attention heads can be dropped with negligible performance loss, and [11] investigate
fine-grained weight pruning in BERT. [8] introduce LayerDrop, a structured dropout technique that ef-

2

InputOutputLayer1Layer2LayerLLayer?Layer?+1�Layer?+2Skip:omitoneormorelayersRecurrent:reuseoneormorelayersforiterativerefinement�fectively trains models so arbitrary subsets of layers can be skipped during inference without requiring
fine-tuning. These methods produce smaller models that trade computation for a small accuracy loss.

By contrast, early-exit or input-adaptive methods add auxiliary classifiers at intermediate layers so
that "easy" inputs exit early. Notable examples include FastBERT [13] and DeeBERT [23], which
insert classifiers after each block and use confidence or entropy metrics to decide when to stop.
PABEE [26] employs a patience criterion to halt when predictions stabilize. DACT-BERT [7] adopts
a differentiable Adaptive Computation Time mechanism to learn how many Transformer layers to
run for each example. [14] estimate input "hardness" via mutual information or reconstruction error
to pre-determine the number of Transformer layers to use.

These early-exit networks achieve significant speedups on NLP tasks by adaptively reducing depth per
input. More recently, early-exit ideas have been extended to vision and multimodal Transformers. [24]
propose LGViT, which adds heterogeneous exit heads (local and global) to ViT so that vision trans-
formers can terminate early with minimal feature loss. [19] introduce MuE ("Multiple Exiting"), a
strategy for unified vision-language models that dynamically skips layers in both encoder and decoder
based on input similarity. These works demonstrate that later layers can be skipped to allow image
and vision-language models to adapt computation per sample with minimal accuracy drop. Our work
generalizes this approach by allowing skipping of arbitrary layers and enabling reuse of certain layers.

Looped Transformer and Recurrent Depth Another line of research makes Transformer depth
adaptive by looping or repeating layers. The Universal Transformer [5] was an early example: it
applies the same self-attention block recurrently and uses a halting mechanism to determine when
each position is "done" (adapting depth per token). Building on these ideas, recent work explicitly
introduces loops in model architectures. [9] demonstrate that a Looped Transformer � a single
Transformer block applied repeatedly � can achieve much better length generalization on algorithmic
tasks by adjusting the number of loops during inference. Similarly, [25] note that looped architectures
excel at learning algorithms by explicitly incorporating iterative characteristics into the transformer
architecture. More sophisticated variants like the Inner Thinking Transformer [3] interleave adaptive
loops with residual "thinking" connections and per-token routing, enabling the model to devote extra
computation only to particularly difficult tokens. In summary, these approaches explore recurrent
or elastic depth via explicit loops to tailor the number of applied layers to each input�s complexity.
Unlike our approach, they require special architecture design and training from scratch, whereas our
work focuses on pure test-time adaptation.

Dynamic Routing and Modular Inference A third theme treats networks as collections of modules
or experts with dynamically chosen pathways per sample. Mixture-of-Experts (MoE) Transformer
layers are a well-known example: they maintain multiple sub-networks ("experts") and route each
token to a subset. [22] introduce Routing Experts (RoE) for multimodal LLMs, retrofitting trained
models into a mixture-of-experts style by learning input-dependent shortcut routes through layers,
guided by sparsity regularizers. [12] present Mixture of Nested Experts (MoNE): experts organized
in a hierarchy of increasing capacity, where tokens are sent to smaller experts when sufficient. MoNE
learns to prioritize easy tokens through low-cost experts and reserve full models for hard cases,
halving inference compute on ImageNet/Video tasks.

These methods exemplify sample-wise routing: at inference time, the model conditionally activates
different sub-modules or experts for each input. Similarly, neural module networks [1] assemble
task-specific computation graphs from a library of modules. In modern LLMs/VLMs, these routing
approaches � whether through gating experts, skipping layers, or assembling modules � form a
spectrum of modular inference techniques that adapt the computation graph on a per-sample basis
to balance cost and accuracy. Interestingly, our work suggests that transformer layers can function
effectively as modules even without being specifically trained for that purpose.

3 MCTS for Optimizing CoLa with Adaptive Depths

Our approach, Chain-of-Layers (CoLa), treats a pretrained LLM�s layers as building blocks to be
composed dynamically per input. Formally, if the original model has layer [L1, . . . , LN ] in order,
CoLa seeks a new sequence (path) P = [Li, Lj, Lk, . . . ] (each L is one of the original layers) that
can replace the standard forward pass for a given input. The path can skip layers (omit some Lm) or
loop layers (repeat some Lm multiple times in succession, akin to unrolling an RNN). We restrict that
the path uses layers from the original model (no new weights) and each layer�s internal parameters

3

Algorithm 1 Adaptive MCTS for CoLa Optimization

1: Initialize root node P0 = [L1, L2, . . . , LN ]
2: for N = 1 to number of simulations do
3:
4:
5:
6:
7: end for
8: return Pareto-optimal paths (accuracy vs. length)

Selection: traverse tree maximizing UCB(P )
Expansion: generate skip/repeat candidates if node unexplored
Simulation: evaluate path accuracy on held-out input(s)
Backpropagation: update Q(P ) and v(P ) along trajectory

remain fixed � we only change the order and frequency of application. In effect, CoLa builds a custom
�sub-network� out of the existing layers for each query. This allows shallow execution for easy queries
and deeper or iterative execution for hard queries, all within the same model�s capacity.
Search via Monte Carlo Tree Search. The space of possible layer paths is combinatorially large,
especially when allowing both skips and repeats. To explore this space efficiently, we frame path
construction as a symbolic search problem and use MCTS to discover optimal execution plans per
input. Unlike greedy or beam search, MCTS better balances local exploitation with global explo-
ration�crucial when optimal paths are long or non-trivial. We formalize each MCTS game as follows:

State. A state is a partial or complete layer sequence, initialized as the standard forward path
P0 = [L1, L2, . . . , LN ]. At each step, the state evolves by applying a skip or repeat transformation.

Actions. Each action modifies a contiguous block of layers: skipping k layers or repeating a block of
k layers r times, where k, r ? {1, 2, 3, 4}. For example, �skip 2 layers� removes the next 2, while
�repeat 3 layers twice� inserts two additional copys of the next 3. We constrain the final path length to
avoid excessive computation.

Transition and Simulation. A path is complete once no further actions are allowed. The path is
executed on the input, yielding a prediction. A reward of 1 is assigned for a correct output, optionally
minus a penalty proportional to the normalized path length to encourage compact solutions.

Search Objective. We define the UCB score for node selection as:

UCB(P ) =

(cid:115)

+c

ln V
v(P )

Q(P )
v(P )
(cid:124) (cid:123)(cid:122) (cid:125)
Exploitation

? ?

||P ||
N
(cid:124) (cid:123)(cid:122) (cid:125)
Length Penalty

where Q(P ) is the cumulative reward for path P , v(P ) is the visit count, V is the total simulations, c
balances exploration, ? scales the path-length penalty, and ||P || is the number of layers used.

We run a fixed number of simulations per input to explore the space. After search, we return the
Pareto-optimal paths balancing accuracy and efficiency. The procedure is summarized in Algorithm 1.

4 Experimental Setup

We evaluate CoLa across diverse pretrained and instruction-tuned LLMs, assessing its impact on
generalization and computational efficiency. By enabling dynamic layer composition, CoLa provides
test-time flexibility beyond fixed-depth forward passes.

Models. We study three model families to examine how architecture and supervision influence
dynamic depth selection:

� LLaMA-3-3B, 8B: Standard dense decoder-only transformers trained on open-domain corpora.
� OLMoE-1B-7B: A Mixture-of-Experts transformer with conditional routing. Each MoE layer is

treated as a unit for skipping or repeating in CoLa.

� Instruction-tuned LLMs: For all models above, we include their instruction-tuned versions to

study the effect of supervised alignment on layer utility.

Datasets. We evaluate on two benchmark families that cover a broad spectrum of reasoning com-
plexity: 1) ARC-Easy and ARC-Challenge [4]: Commonsense reasoning tasks requiring shallow

4

Table 1: Comparing different architecture search spaces for layer composition. Accuracy (%)
of the original LLM and the one achieved by MCTS in three different search spaces: skip-only,
recurrence-only, or combining both (default of CoLa). Our skip+recurrence joint search space consis-
tently outperforms other two and the original LLM, especially on harder datasets such as DART-3/4/5.

Model

Variant

ARC-E ARC-C DART-1 DART-2 DART-3 DART-4 DART-5

LLaMA-3-3B-
Base

LLaMA-3-3B-
Instruct

LLaMA-3-8B-
Base

LLaMA-3-8B-
Instruct

OLMoE-1B-7B-
Base

OLMoE-1B-7B-
Instruct

Original
+Skip-only
+Recurrence-only
+CoLa (Skip+Recurrence)

Original
+Skip-only
+Recurrence-only
+CoLa (Skip+Recurrence)

Original
+Skip-only
+Recurrence-only
+CoLa (Skip+Recurrence)

Original
+Skip-only
+Recurrence-only
+CoLa (Skip+Recurrence)

Original
+Skip-only
+Recurrence-only
+CoLa (Skip+Recurrence)

Original
+Skip-only
+Recurrence-only
+CoLa (Skip+Recurrence)

27.80
75.40
65.40
95.80

39.80
86.80
76.20
95.80

45.60
89.20
73.20
95.80

76.00
92.80
89.20
95.80

24.80
76.80
49.80
95.80

53.20
89.40
81.40
95.80

20.80
75.60
54.80
98.20

37.40
86.60
72.60
98.20

42.60
86.60
69.00
98.20

69.00
93.20
87.20
98.20

21.40
77.60
51.40
97.40

41.60
85.40
71.20
98.00

7.00
27.80
26.60
63.60

13.00
30.20
40.60
81.00

13.40
39.80
33.40
79.80

10.00
44.20
30.00
84.20

12.60
26.00
30.20
57.60

14.80
29.80
36.80
63.80

5.40
18.40
16.80
46.80

7.40
22.20
25.20
64.00

6.80
29.00
17.80
58.00

6.00
25.20
20.80
66.20

3.40
15.20
15.80
41.20

7.20
20.40
23.60
48.00

3.00
18.40
11.20
34.80

5.80
19.60
21.00
47.60

4.20
20.00
8.40
42.80

6.80
25.00
24.20
54.60

3.00
15.60
8.00
32.60

3.00
18.60
11.60
36.80

2.60
13.00
8.20
26.00

9.80
16.20
18.20
42.40

2.20
16.40
8.80
31.80

12.00
17.80
24.40
49.40

1.80
8.80
5.40
23.00

1.20
9.00
9.60
28.00

1.40
8.20
5.00
25.60

14.40
13.20
23.80
44.00

0.80
12.20
5.60
29.20

12.40
13.40
21.60
47.80

1.80
4.20
3.60
16.80

0.60
5.60
5.00
22.00

symbolic or factual inference; 2) DART-Math [21]: a math benchmark stratified into five difficulty
levels (DART-1 to DART-5, from easiest to hardest), enabling analysis of how CoLa adapts execution
depth to task complexity.

Evaluation Protocol. All models are evaluated in a zero-shot setting with frozen weights. Each input
is prompted as a natural language question, and model predictions are compared against gold answers.
These outputs serve as feedback for MCTS to iteratively search optimized layer paths per input.

These experiments assess whether CoLa can discover more efficient or more accurate computation
paths than the default model, revealing the latent composability of pretrained LLM layers.

5 Experimental Analysis

To evaluate how compositional layer execution affects model behavior, we conduct a comprehensive
empirical analysis across datasets, model scales, and training configurations. Our experiments aim to
answer two core questions: (1) Does dynamic composition of pretrained layers improve generalization
and efficiency? and (2) How are individual layers engaged under adaptive execution paths?

5.1 Layer Composition in CoLa Enhances Both Generalization and Efficiency

A central hypothesis of this work is that fixed-depth computation can be a limiting factor for general-
ization and efficiency. By enabling test-time composition of layers�selectively skipping or repeating
layers from a pretrained backbone�CoLa offers a flexible alternative to the rigid forward pass of
standard transformers. This section demonstrates that such dynamic layer composition substantially
improves both prediction accuracy and computational efficiency across model scales and task types.

Finding 1

Joint search of layer-skip and layer-recurrence significantly improves generalization. While
layer-skip simplifies the architecture for easy inputs, and recurrence improves reasoning on
moderate tasks, their combination consistently performs the best on the hardest tasks.

To understand how different execution strategies impact generalization, we compare three search
space variants: skip-only (allowing layers to be bypassed but used only once), recurrence-only

5

(allowing repetition without skipping), and our full version (supporting both skipping and recurrence).
This combination is motivated by the complementary effects of the two operations: skipping allows
the model to compress its computation, effectively pruning away redundant or less informative layers,
while recurrence enables expansion of depth by revisiting high-utility layers for iterative refinement.
When used together, they grant the model both compression and expansion capabilities�allowing
it to adaptively match the depth and structure of reasoning to the input. This flexibility forms the
foundation of CoLa�s generalization gains across diverse tasks.

As shown in Table 1, both restricted variants improve performance over vanilla forward execution,
but in complementary and ultimately limited ways.

The skip-only variant provides strong gains on simpler tasks, particularly ARC-E and ARC-C,
suggesting that many inputs can be correctly processed with shallower depth. For example, on
LLaMA-3B, accuracy on ARC-E improves from 27.8% to 75.4%, showing that reducing computation
depth can be beneficial when the full model is over-parameterized for the task.

The recurrence-only variant tends to perform better on more complex tasks that benefit from repeated
application of informative layers. For instance, on DART-4 and DART-5, recurrence-only outperforms
significantly skip-only in several settings, including LLaMA-3-3B-Instr and LLaMA-3-8B-Instr,
where simply shortening the path is insufficient to capture task-specific dependencies.

However, neither skip-only nor recurrence-only suffices across the board. Our full method, which
searches over both skipping and recurrence decisions, consistently achieves the highest accuracy
on all datasets and models. These gains are especially large on harder DART-Math datasets: for
example, on DART-2, the LLaMA-3-8B-Instr model improves from 25.2% (skip-only) and 22.0%
(recurrence-only) to 66.2% with the full space�nearly a threefold increase.

Moreover, we observe a consistent pattern across the MoE model OLMoE as well. Although the
absolute improvement from CoLa is somewhat smaller compared to dense models, this is likely due
to the fact that OLMoE already employs sparse expert selection at each layer, leaving less room for
further compression or expansion. Nevertheless, CoLa is still able to discover substantially better
execution paths on OLMoE, highlighting that even sparse architectures benefit from dynamic layer
composition when equipped with flexible skipping and recurrence operations.

Finding 2

Reducing the depth helps correct errors. Various input tasks can be solved correctly using
shallower and highly compressed layer compositions. CoLa�s error correction often leads to even
fewer layers than keeping originally correct predictions, indicating severe �overthinking� in depth.

Beyond accuracy, we investigate whether these gains are achieved with lower inference cost. Fig-
ure 2 analyzes the average depth under CoLa across all tasks and models, revealing consistent and
substantial gains in inference efficiency.

Figure 2: Depth and non-recurrent depth (# unique layers) of CoLa on four models and seven
benchmarks. The average depth of CoLa for inputs whose predictions by the original model and
CoLa are both correct (C?C), and whose predictions by the original model are wrong but corrected
by CoLa (W?C). Both the depth and non-recurrent depth are effectively reduced in all cases.

We observe that, for both C?C (correct before and after) and W?C (corrected by CoLa) inputs,
the average depth is significantly lower than the full model depth. This confirms that many inputs

6

ARC-EasyARC-ChallengeDART-1DART-2DART-3DART-4DART-52425262728293B Depth3B-base CC3B-base WC3B-Instruct CC3B-Instruct WC8B-base CC8B-base WC8B-Instruct CC8B-Instruct WCFull Model DepthNon-recurrent depth (# unique layers)272829303132338B Depthcan be correctly processed without invoking the entire network�underscoring the potential for
computation-aware inference.

Moreover, the distinction between total depth and non-recurrent depth (i.e., the number of unique
layers used) reveals the extent of layer recurrence. In many cases, non-recurrent depth is markedly
lower than total depth, indicating that CoLa reactivates high-utility layers multiple times to achieve
correct answers more compactly. This recurrence-driven compression is most pronounced on simpler
datasets such as ARC-E, but remains visible even on harder DART tasks.

Interestingly, W?C paths tend to be shorter than C?C ones�both in total and non-recurrent depth.
This suggests that correcting an incorrect prediction does not require more computation, but rather a
reconfiguration that omits noisy or misleading layers present in the default forward pass.

Finally, we find that pretrained models generally yield more compressed execution paths than their
instruction-tuned counterparts. This is likely because instruction tuning calibrates more layers to be
relevant, leaving less opportunity for aggressive pruning or recurrence. In contrast, pretrained models
often contain layers that can be skipped or consolidated, allowing CoLa to discover leaner and more
effective execution patterns.

These patterns are further illustrated in Figure 3, which compares the actual depth and accuracy of
each strategy on DART-4 and DART-5. Recurrence-only strategies achieve moderate gains over
the pretrained baseline, but do so by increasing the overall depth�relying on repeated application of
helpful layers to compensate for rigid forward computation. In contrast, skip-only strategies operate
with reduced depth by omitting irrelevant layers, yielding marginal improvements in efficiency
and accuracy. Strikingly, CoLa achieves substantially higher accuracy than both skip-only and
recurrence-only strategies, despite not always using the fewest layer invocations. Its ability to balance
depth and performance highlights the strength of test-time dynamic layer composition�achieving
superior generalization with efficient yet targeted computation.

Finding 3

Keeping predictions correct does not require the full-depth model and all the layers. Even for
inputs whose predictions by the original model are already correct, CoLa can find shallower and
more effective architectures, revealing substantial redundancy in static transformer inference.

We further analyze how CoLa reshapes predic-
tion outcomes at the example level. Figure 4
categorizes test inputs into four transition types:
C?C, W?C, wrong before and after (W?W),
and cases where the original path remains optimal.

Across all models and datasets, the fraction of
inputs where the original path remains optimal
is nearly negligible�even among those correctly
solved without CoLa. This reveals that static
forward passes are rarely optimal, and that CoLa
consistently discovers alternative paths that lead
to the same or better outcomes. A substantial
portion of inputs fall into the W?C category,
demonstrating CoLa�s strong capacity to recover
from errors. While this corrective ability dimin-
ishes slightly on the hardest benchmarks (e.g.,
DART-5), gains remain significant across settings.

Figure 3: Accuracy�depth tradeoff on DART-
4/5 (hardest). Each point represents an architec-
ture search space (original, skip-only, recurrence-
only, CoLa) applied to pretrained or instruction-
tuned LLaMA-3-8B. CoLa consistently achieves
the best tradeoff: although its depth lies between
it
skip-only and recurrence-only strategies,
substantially improves accuracy, pushing the
accuracy�cost Pareto frontier forward.

These results show that CoLa not only improves
accuracy through path reconfiguration, but does
so more efficiently�often correcting errors or
simplifying computation on a per-input basis. This challenges the assumption that correct predictions
imply optimal computation, and underscores the benefit of flexible, structure-aware inference.

7

27.530.032.535.0Depth01020304050Accuracy (%)BetterDART-427.530.032.535.0Depth01020304050BetterDART-58B-Base8B-Base + Skip-only8B-Base + Recurrence-only8B-Base + CoLa8B-Instruct8B-Instruct + Skip-only8B-Instruct + Recurrence-only8B-Instruct + CoLaFigure 4: Prediction correctness transitions under CoLa. For each model and dataset, it reports the
proportion of four categories of test samples: original path remains optimal, correct?correct (C?C),
wrong?correct (W?C), and wrong?wrong (W?W). CoLa substantially improves prediction
outcomes. It rarely retains the original sub-optimal path.

5.2 Layer Engagement: Selection, Skipping, and Recurrence

To better understand how CoLa executes depth-adaptive computation, we study how different layers
are engaged across tasks and model scales. We organize the analysis into two parts: (1) which layers
are selected, and (2) how layers are recurred or skipped.

Finding 4

Harder tasks and larger models encourage more uniform layer engagement. As tasks become
more challenging and model capacity increases, CoLa distributes computation more evenly across
layers�moving from shallower specialization to deeper models involving more diverse layers.

Layer selection patterns reflect task complexity and depends on model scales. We analyze how
layer usage varies with model scale and task difficulty (Figure 5). Across all settings, early layers are
most frequently engaged, likely due to their broad utility in extracting low-level features. However,
deeper patterns diverge between models.

In the 3B model, layer usage follows a clear V-shape: early and late layers dominate, while middle
layers are suppressed. This suggests a tendency to compress computation into the extremes of the
network. In contrast, the 8B model exhibits a smoother decay from early to late layers, with middle
layers more consistently engaged�reflecting greater capacity to leverage intermediate represen-
tations. Larger models also exhibit more stable usage patterns, with reduced variance and fewer
outliers, especially after instruction tuning. Quantitatively, 8B models show greater diversity in layer
engagement, with higher entropy (3.46 v.s. 3.33) and lower maximum layer concentration (0.035 v.s.
0.040) compared to 3B. This indicates less reliance on any single layer and a more distributed use of
the model�s depth.

Task complexity further amplifies these differences. On harder datasets such as DART-5, layer usage
becomes increasingly uniform, especially in 8B models. Early, middle, and late layers are engaged
with comparable frequency, suggesting that complex reasoning benefits from activating a broader
slice of the model�s capacity.

Finding 5

Larger finetuned models adopt less stereotyped layer-usage patterns. While smaller models�
CoLa show skipping and recurrence patterns concentrated on layers at specific depths, larger and
instruction-tuned models engage layers more irregularly and adaptively regardless of their depths.

Fine-Grained Analysis of Skipping and Recurrence Patterns. To gain finer-grained insights into
how layers are used during inference, we examine not only whether a layer is selected, but how often
it is skipped entirely or recurred multiple times within a path. Figure 6 visualizes the per-layer skip
rate and repeat count per path, aggregated over all datasets and grouped by model variant.

Across all models, the earliest layers are consistently retained during inference, exhibiting near-
zero skip rates. Beyond this point, skip rates follow a non-monotonic profile: increasing across
intermediate layers before declining toward the deeper end. This pattern is most pronounced in smaller
pretrained models, where skipping is strongly concentrated in a narrow middle region. Instruction
tuning softens this behavior, resulting in a smoother and more dispersed skip distribution. Larger
models exhibit broader and less localized skipping patterns, with elevated skip rates spread across
both middle and late layers. While they lack the sharp spikes seen in smaller models, their skip
profiles still exhibit moderate fluctuations�indicating neither uniformity nor strict localization.

8

ARC-EARC-CDART-1DART-2DART-3DART-4DART-50.00.51.0Proportion3B-BaseARC-EARC-CDART-1DART-2DART-3DART-4DART-53B-Instruct ARC-EARC-CDART-1DART-2DART-3DART-4DART-58B-BaseARC-EARC-CDART-1DART-2DART-3DART-4DART-58B-InstructCase TypeOriginal pathremains optimalCCWCWWFigure 5: Layer selection patterns in CoLa. (a, b) Heatmaps show the frequency of each layer
being selected for 3B and 8B models, respectively, on each dataset, with darker shades indicating
higher usage. (c, d) Boxplots group layers into early, middle, and late segments, revealing systematic
variation in their usage across datasets and models. 3B models exhibit greater variability and more
aggressive pruning of mid-depth layers compared to 8B.

Recurrence distributions also show depth-dependent variation, though their profiles differ from skip
patterns in where repetition is concentrated. Smaller pretrained models concentrate repetition toward
late layers, while tuning promotes more even recurrence across the stack. In larger models, recurrence
counts fluctuate non-monotonically across depth, suggesting that layer recurrence is dynamically
adjusted based on context rather than fixed to particular depths.

Together, these results suggest that smaller models rely on stereotyped usage�amplifying or pruning
specific depths�while larger and instruction-tuned models adopt more distributed, context-sensitive
computation paths.

Figure 6: Skipping and recurrence rate of each layer on four models. Left: Skip rate�the
proportion of CoLa in which layer-i is skipped. Right: Averaged recurrence times of layer-i in CoLa.
CoLa models consistently keep early layers but exhibit an elevated skip rate of middle layers.

6 Conclusion

This work reframes model generalization through test-time architectural adaptation. By treating each
transformer layer as a reusable or skippable module, we introduce a flexible Chain-of-Layers (CoLa)
space that enables input-specific execution paths without additional training. Using Monte Carlo
Tree Search, CoLa discovers optimized layer compositions that improve both accuracy and efficiency
across diverse tasks and models. Empirically, it not only corrects original model errors but often does
so with shallower, more targeted computation. These results challenge the static nature of standard
forward passes and reveal the latent compositionality of pretrained layers�positioning test-time
depth adaptation as a promising step toward unifying fast and slow reasoning in LLM inference.

9

0.750.800.850.900.951.001.051.100369121518212427Layer(a) 3B Heatmap(Base) ARC-C(Base) ARC-E(Base) DART-1(Base) DART-2(Base) DART-3(Base) DART-4(Base) DART-5(Instruct) ARC-C(Instruct) ARC-E(Instruct) DART-1(Instruct) DART-2(Instruct) DART-3(Instruct) DART-4(Instruct) DART-5Dataset036912151821242730Layer(b) 8B Heatmap(Base) ARC-E(Base) ARC-C(Base) DART-1(Base) DART-2(Base) DART-3(Base) DART-4(Base) DART-5(Instruct) ARC-E(Instruct) ARC-C(Instruct) DART-1(Instruct) DART-2(Instruct) DART-3(Instruct) DART-4(Instruct) DART-5Dataset0.60.70.80.91.01.1Layer Frequency(c) 3B Boxplot(Base) ARC-E(Base) ARC-C(Base) DART-1(Base) DART-2(Base) DART-3(Base) DART-4(Base) DART-5(Instruct) ARC-E(Instruct) ARC-C(Instruct) DART-1(Instruct) DART-2(Instruct) DART-3(Instruct) DART-4(Instruct) DART-5Dataset(d) 8B BoxplotLayer Groupearlymiddlelate0.00.10.20.3LLaMA-3BSkip Rate3B-Base3B-Instruct0.00.10.20.3Repeat per Path3B-Base3B-Instruct0481216202428Layer0.00.10.2LLaMA-8B8B-Base8B-Instruct0481216202428Layer0.00.10.28B-Base8B-InstructReferences

[1] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In
Proceedings of the IEEE conference on computer vision and pattern recognition, pages 39�48,
2016.

[2] B Barla Cambazoglu, Hugo Zaragoza, Olivier Chapelle, Jiang Chen, Ciya Liao, Zhaohui Zheng,
and Jon Degenhardt. Early exit optimizations for additive machine learned ranking systems. In
Proceedings of the third ACM international conference on Web search and data mining, pages
411�420, 2010.

[3] Yilong Chen, Junyuan Shang, Zhenyu Zhang, Yanxi Xie, Jiawei Sheng, Tingwen Liu, Shuohuan
Wang, Yu Sun, Hua Wu, and Haifeng Wang. Inner thinking transformer: Leveraging dynamic
depth scaling to foster adaptive internal thinking. arXiv preprint arXiv:2502.13842, 2025.

[4] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,
and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning
challenge. arXiv preprint arXiv:1803.05457, 2018.

[5] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and ?ukasz Kaiser. Uni-

versal transformers. arXiv preprint arXiv:1807.03819, 2018.

[6] Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Sean
Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, et al. Faith and fate: Limits of
transformers on compositionality. Advances in Neural Information Processing Systems, 36:
70293�70332, 2023.

[7] Crist�bal Eyzaguirre, Felipe del Rio, Vladimir Araujo, and Alvaro Soto. Dact-bert: Differen-
tiable adaptive computation time for an efficient bert inference. arXiv preprint arXiv:2109.11745,
2021.

[8] Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with

structured dropout. arXiv preprint arXiv:1909.11556, 2019.

[9] Ying Fan, Yilun Du, Kannan Ramchandran, and Kangwook Lee. Looped transformers for

length generalization. arXiv preprint arXiv:2409.15647, 2024.

[10] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R Bartold-
son, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute
with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171, 2025.

[11] Mitchell A Gordon, Kevin Duh, and Nicholas Andrews. Compressing bert: Studying the effects

of weight pruning on transfer learning. arXiv preprint arXiv:2002.08307, 2020.

[12] Gagan Jain, Nidhi Hegde, Aditya Kusupati, Arsha Nagrani, Shyamal Buch, Prateek Jain, Anurag
Arnab, and Sujoy Paul. Mixture of nested experts: Adaptive processing of visual tokens. arXiv
preprint arXiv:2407.19985, 2024.

[13] Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Haotang Deng, and Qi Ju. Fastbert: a
self-distilling bert with adaptive inference time. arXiv preprint arXiv:2004.02178, 2020.

[14] Yijin Liu, Fandong Meng, Jie Zhou, Yufeng Chen, and Jinan Xu. Faster depth-adaptive
transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35,
pages 13424�13432, 2021.

[15] Zejian Liu, Fanrong Li, Gang Li, and Jian Cheng. Ebert: Efficient bert inference with dynamic
structured pruning. In Findings of the Association for Computational Linguistics: ACL-IJCNLP
2021, pages 4814�4823, 2021.

[16] Zhuang Liu, Zhiqiu Xu, Hung-Ju Wang, Trevor Darrell, and Evan Shelhamer. Anytime dense

prediction with confidence adaptivity. arXiv preprint arXiv:2104.00749, 2021.

[17] Binghui Peng, Srini Narayanan, and Christos Papadimitriou. On limitations of the transformer

architecture. In First Conference on Language Modeling, 2024.

10

[18] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song,
John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language
models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446,
2021.

[19] Shengkun Tang, Yaqing Wang, Zhenglun Kong, Tianchi Zhang, Yao Li, Caiwen Ding, Yanzhi
Wang, Yi Liang, and Dongkuan Xu. You need multiple exiting: Dynamic early exiting for
accelerating unified vision language model. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 10781�10791, 2023.

[20] Surat Teerapittayanon, Bradley McDanel, and Hsiang-Tsung Kung. Branchynet: Fast inference
via early exiting from deep neural networks. In 2016 23rd international conference on pattern
recognition (ICPR), pages 2464�2469. IEEE, 2016.

[21] Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. Dart-math: Difficulty-
aware rejection tuning for mathematical problem-solving. 2024. URL https://arxiv.org/
abs/2407.13690.

[22] Qiong Wu, Zhaoxi Ke, Yiyi Zhou, Xiaoshuai Sun, and Rongrong Ji. Routing experts: Learning to
route dynamic experts in multi-modal large language models. arXiv preprint arXiv:2407.14093,
2024.

[23] Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. Deebert: Dynamic early exiting

for accelerating bert inference. arXiv preprint arXiv:2004.12993, 2020.

[24] Guanyu Xu, Jiawei Hao, Li Shen, Han Hu, Yong Luo, Hui Lin, and Jialie Shen. Lgvit: Dynamic
early exiting for accelerating vision transformer. In Proceedings of the 31st ACM International
Conference on Multimedia, pages 9103�9114, 2023.

[25] Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers

are better at learning learning algorithms. arXiv preprint arXiv:2311.12424, 2023.

[26] Wangchunshu Zhou, Canwen Xu, Tao Ge, Julian McAuley, Ke Xu, and Furu Wei. Bert loses
patience: Fast and robust inference with early exit. Advances in Neural Information Processing
Systems, 33:18330�18341, 2020.

11

A Fine-Grained Depth Compression Analysis

Figure 7: Mean depth and non-recurrent depth of the shortest 5%, 10%, 20%, and 100% of
valid execution paths under CoLa.

To deepen our understanding beyond average-case trends, we conduct a percentile-based analysis
of inference path lengths under CoLa. Instead of aggregating over all inputs, we focus on the most
efficient cases by identifying the shortest valid execution paths and computing their mean total and
non-recurrent depth. For each model and dataset, we report the average depth among the shortest 5%,
10%, 20%, and 100% of CoLa paths for correctly solved inputs, including both C?C and W?C
transitions. The results, shown in Figure 7, highlight how much computation can be reduced in the
most efficient cases.

We analyze the mean total and non-recurrent depth of the shortest 5% and 20% of correct execution
paths under CoLa (Figure 7). For the top 5% most efficient cases, 3B models compress to 20�22
layers and 8B to 22.5�25, corresponding to up to 30% reduction from the full model depth. For the

12

2022242628303B Depth5th Percentile Depth3B-base CC3B-base WC3B-Instruct CC3B-Instruct WC8B-base CC8B-base WC8B-Instruct CC8B-Instruct WCFull Model DepthEffective Depth (unique layers)2022242628303B Depth10th Percentile Depth2022242628303B Depth20th Percentile DepthARC-EasyARC-ChallengeDART-1DART-2DART-3DART-4DART-5Tasks2022242628303B Depth100th Percentile Depth222426283032348B Depth222426283032348B Depth222426283032348B Depth222426283032348B Depth20th percentile, depths remain 12�23% lower than baseline, indicating that a substantial fraction of
inputs�especially corrected errors�can be processed with significantly fewer layers. Non-recurrent
depths are consistently even lower, confirming the presence of efficient recurrence patterns and layer
reuse in shallow yet effective execution paths.

B Implementation Details

Algorithm. We use 200 simulations per input in our MCTS implementation for optimizing CoLa,
balancing search quality and runtime. The UCB score incorporates a normalized path length penalty
with weight 5.0 to favor compact execution paths. To encourage exploration, the algorithm selects a
random unexplored child node with probability 0.1 instead of the one with the highest UCB score.
This behavior is hard-coded as a fixed conditional in the selection logic. All hyperparameters are
kept fixed across models and datasets to ensure consistency and reproducibility.

Datasets. We randomly sample 500 instances from each dataset for evaluation.

13


