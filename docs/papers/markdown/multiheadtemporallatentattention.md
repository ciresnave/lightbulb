5
2
0
2

y
a
M
1
2

]

G
L
.
s
c
[

2
v
4
4
5
3
1
.
5
0
5
2
:
v
i
X
r
a

Multi-head Temporal Latent Attention

Keqi Deng, Philip C. Woodland
Department of Engineering, University of Cambridge
Trumpington St., Cambridge, UK
kd502@cam.ac.uk

Abstract

While Transformer self-attention offers strong parallelism, the Key-Value (KV)
cache grows linearly with sequence length and becomes a bottleneck for inference
efficiency. Multi-head latent attention was recently developed to compress the
KV cache into a low-rank latent space. This paper proposes Multi-head Temporal
Latent Attention (MTLA), which further reduces the KV cache size along the tem-
poral dimension, greatly lowering the memory footprint of self-attention inference.
MTLA employs a hyper-network to dynamically merge temporally adjacent KV
cache vectors. To address the mismatch between the compressed KV cache and pro-
cessed sequence lengths, a stride-aware causal mask is proposed to ensure efficient
parallel training and consistency with inference behaviour. Experiments across
tasks, including speech translation, speech recognition, speech understanding and
text summarisation, demonstrate that MTLA achieves competitive performance
compared to standard Multi-Head Attention (MHA), while greatly improving infer-
ence speed and GPU memory usage. For example, on a English-German speech
translation task, MTLA achieves a 5.3� speedup and a reduction in GPU memory
usage by a factor of 8.3 compared to MHA, while maintaining translation quality.

1

Introduction

The Transformer [40] decoder has become increasingly important, particularly with the success
of large language models (LLMs) [6, 38]. As LLMs have been extended to other modalities such
as speech [10, 37, 13], this decoder-only architecture is gradually becoming a unified framework
for handling many tasks. For example, by placing an input speech sequence before the text and
modelling causal dependencies auto-regressively via self-attention, decoder-only models can naturally
handle speech tasks such as speech recognition and speech translation [43, 39]. However, during
auto-regressive inference, each decoding step requires loading the cached attention keys and values
to avoid re-encoding the history. This repeated memory access has emerged as a bottleneck, limiting
inference speed and constraining both the decoding batch size and sequence length [35, 33, 20]. As
model scales and application demands increase, reducing this memory bandwidth overhead is crucial
for efficient deployment.

To alleviate the memory bottleneck associated with the Key-Value (KV) cache during incremental
inference, several attention variants have been proposed. Multi-Query Attention (MQA) [35] reduces
the number of KV heads by sharing a single head of keys and values across all query heads, greatly
decreasing memory usage. Given that MQA can lead to quality degradation and training instability,
Grouped-Query Attention (GQA) [1] was proposed, which partitions query heads into groups, each
sharing a distinct head of keys and values. Despite these advancements, both MQA and GQA primarily
focus on reducing the number of KV cache heads, which can lead to performance degradation due
to limited representational capacity [28]. Recently, Multi-Head Latent Attention (MLA) [24] has
emerged as a more advanced approach. MLA reduces the KV cache size by lowering the latent
dimension of the saved KV vectors. [24, 28] show that MLA achieves higher model accuracy than

Preprint. Under review.

MQA and GQA, and can match or even surpass multi-head attention (MHA) [40]. However, existing
methods, including MQA, GQA, and MLA, have not explored compression along the temporal
dimension of the KV cache. Given that the KV cache size grows linearly with sequence length, there
is great potential for further KV cache compression, especially in long-context scenarios.

This paper proposes Multi-Head Temporal Latent Attention (MTLA), which builds on MLA but
further reduces the KV cache size along the temporal dimension. MTLA compresses the temporal
dimension by dynamically merging temporally adjacent KV cache vectors in a learnable manner.
Since the input sequence length varies across examples, this merging process cannot rely on static
parameters: instead, MTLA employs a hyper-network to generate the merging weights for the KV
cache. During inference the KV cache has fewer elements than the processed sequence and the most
recent KV cache vectors can be updated as processing proceeds. However, the correct KV cache
vectors must be used in training which is an issue for efficient parallel training. To address this issue,
this paper designs a stride-aware causal mask to ensure consistency between the attention behaviour
during parallel training and that during incremental inference. Following [24], decoupled rotary
position embedding is adopted to encode positional information, together with MTLA temporal
compression. Experiments on speech translation, speech recognition, speech understanding, and text
summarisation show that MTLA achieves competitive model accuracy compared to standard MHA,
while greatly improving inference speed and reducing GPU memory usage at inference.

The main contributions of this paper can be summarised in four main parts:

� MTLA is proposed, which is, to the best of our knowledge, the first work to compress the

temporal dimension of the self-attention KV cache.

� A hyper-network is used to dynamically generate weights for merging adjacent KV caches

along the temporal dimension.

� A stride-aware causal mask is designed for MTLA to achieve efficient parallel training,

simulating the attention behaviour during incremental decoding.

� MTLA matches MHA and MLA in accuracy across tasks while greatly increasing processing
speed and reducing GPU memory usage during inference. The code will be open-sourced:
https://github.com/D-Keqi/mtla

2 Related Work

Reducing the memory and computational overhead of the KV cache in Transformer decoders has
been a focal point of recent research. MQA [35] reduces KV cache size by sharing a single key and
value head across all query heads, while GQA [1] divides query heads into groups and each shares a
single key and value head. MLA [24] compresses KV representations into a lower-dimensional latent
space, offering better expressiveness than GQA and comparable or improved accuracy over MHA.
Additionally, techniques like MiniCache [26] and MLKV [45] reduce memory by sharing KV caches
across layers, though this may harm performance due to layer-specific attention patterns.

Another line of work explores linear attention models such as Linear Transformers [21, 41], RWKV
[32], and Mamba [17], which reduce memory via linear time complexity. However, they often struggle
with long-range dependencies, impacting tasks that rely on complex context. Recent theoretical
analysis [2] also proves that truly subquadratic inference time can not solve challenging tasks such
as document similarity. Despite the cost, quadratic attention remains crucial for fine-grained token
interactions, motivating our focus on Transformer attention.

Beyond architectural modifications, various engineering techniques have been proposed to optimise
Transformers. Dynamic token pruning methods, such as LazyLLM [15] and SnapKV [22], reduce
memory usage by selectively removing less important tokens from the KV cache. Pruning can also
be applied to attention heads or dimensions, though it may compromise contextual understanding
and complicate the pipeline [28]. In addition, KV quantisation [25] can further reduce memory
by lowering KV cache precision. Furthermore, FlashAttention [12, 11] restructures the attention
computation to minimise memory access overhead, enhancing both speed and efficiency. While these
engineering tricks enhance Transformer efficiency, this paper focuses on compressing the KV cache
along the temporal dimension, an under-explored direction which has potential for substantial savings
memory and computation for long-sequence tasks.

2

3 Preliminaries and Background

This section reviews some important background on the use of a KV-cache in auto-regressive inference
and the operation of standard multi-head attention. The approaches taken by the MQA, GQA and
MLA methods for reducing the size of the KV-cache are then outlined.

Key-Value Cache in Auto-regressive Inference At inference, the model generates one next token
xi at a time, using past tokens x1, � � � , xi?1. To reduce computation, Transformers cache previously
computed key and value vectors instead of re-computing the attention context for each step.
Given a query vector qi ? R1�d at step i, where d is the model dimension, and the cached key and
value matrices K<i ? R(i?1)�d and V<i ? R(i?1)�d, the attention output is computed as:

Attention(qi, K<i, V<i) = softmax

(cid:19)

(cid:18) qiK?
<i?
d

V<i

(1)

Here, qi is computed from xi, and K<i, V<i are cached from previous steps. Without caching, K<i
and V<i must be re-computed at every step, leading to redundant computation and quadratic time.

Multi-Head Attention (MHA) Given an input sequence X ? RT �d, where T denotes the sequence
length, MHA [40] projects it into query Q, key K, and value tensors V using learned weight matrices:

Q = XWQ ? RT �(nh�dh), K = XWK ? RT �(nh�dh), V = XWV ? RT �(nh�dh)

(2)

where WQ, WK, WV ? Rd�(nh�dh) are learned matrices, and nh is the number of attention heads.

Multi-Query Attention (MQA) MQA [35] shares key and value matrices across heads to reduce
memory. Each head h has its own query Q(h) = XW(h)

Q ? RT �dh , but all heads share:

K = XWK ? RT �dh , V = XWV ? RT �dh

(3)

Group-Query Attention (GQA) GQA [1] groups heads into g sets, each sharing a key and value.

K = XWK ? RT �(g�dh), V = XWV ? RT �(g�dh)

(4)

Heads in group i share K(i), V(i) ? RT �dh . Each head has independent queries as in MHA.

Multi-Head Latent Attention (MLA) MLA [24] compresses the key-value memory into a latent
sequence C ? RT �r with a smaller hidden dimension r < d. The attention computation becomes:

C = XWr ? RT �r

K = CWK ?RT �(nh�dh), V = CWV ? RT �(nh�dh)

(5)

(6)

where C is saved as KV cache and directly used for attention computation, avoiding explicit K and
V computation by absorbing WK into WQ and WV into the output projection.

4 Multi-head Temporal Latent Attention (MTLA)

This paper proposes Multi-head Temporal Latent Attention (MTLA), which, building upon compress-
ing the Key-Value (KV) cache into a low-rank latent space as in MLA, further compresses the KV
cache along the temporal dimension. Hence, MTLA can greatly reduce GPU memory usage and
accelerate inference. Meanwhile, MTLA addresses the challenge of mismatched KV cache length
and generated sequence length by introducing a stride-aware causal mask, enabling efficient parallel
training.

As illustrated in Fig. 1, unlike conventional Multi-Head Attention (MHA) that maintains separate
key and value cache vectors for each attention head, MTLA employs a shared low-rank latent vector
to compress key and value information across heads, following [24]. Furthermore, MTLA merges
adjacent latent vectors along the temporal dimension to store them as the KV cache.

3

Figure 1: Illustration of MTLA. Blue arrows denote transformations by linear layers, and the red
dashed lines indicate content attended to during attention. The example corresponds to 4 attention
heads. (a) Standard MHA; (b) MTLA with a temporal compression ratio of 2. ? denotes addition. The
transformation from compressed temporal-latent KV cache to multi-head KVs can be absorbed into
the query/output linear layers via matrix multiplication associativity, avoiding redundant computation.

Specifically, given an input sequence X ? RT �d, where T is the sequence length and d is the model
dimension, the multi-head queries Q = (q1, q2, � � � , qT ) are computed following standard MHA:
Q = XWQ ? RT �(nh�dh)
(7)
where WQ ? Rd�(nh�dh) are learned linear weight matrices. Following [24], low-rank compression
(dimension is r) is performed to obtain the low-rank latent vectors C = (c1, c2, � � � , cT ):

C = XWr ? RT �r
(8)
where Wr ? Rd�r is a trainable weight matrix. Layer normalisation [4] is then applied to C to
stabilise training, following the implementation in [24]. MTLA further applies learnable weights
(w1, w2, . . . , wT ) to compress the latent sequence C along the temporal dimension, yielding a shorter
compressed temporal-latent KV sequence �C = (�c1, �c2, � � � , �ct) ? Rt�r, where t = ?T /s? and s
denotes the temporal compression ratio.

As illustrated in Fig. 1, assuming s = 2, every 2 temporally adjacent latent vectors in C are merged
using the corresponding weights (w1, w2, . . . , wT ); for example, �c1 = w1 � c1 + w2 � c2, and
�c2 = w3 � c3 + w4 � c4. Since the length of (w1, w2, . . . , wT ) varies dynamically with the input and
cannot be handled using static parameters, MTLA utilises a hyper-network that takes C as input to
generate (w1, w2, . . . , wT ). Further details of this hyper-network are given in refer to Sections 4.1
and 4.2. The choice of s effectively controls the extent of KV cache compression in MTLA. However,
choosing too large a value can caused marked performance degradation.
With the cached �C ? Rt�r, the keys K and values V can be obtained through up-projection matrices
and used for attention computation:

K = �CWK ? Rt�(nh�dh),
V = �CWV ? Rt�(nh�dh),

(9)

(10)

(cid:18) QK?
?
dh
where WK, WV ? Rr�(nh�dh), and WO ? R(nh�dh)�d are are learned linear weight matrices. Note
that due to the associative property of matrix multiplication, Eq. 11 can be rewritten as:

VWO ? RT �d

Y = softmax

(11)

(cid:19)

softmax

(cid:19)

(cid:18) QK?
?
dh

VWO = softmax

(cid:32)

X(WQWK
dh

?

(cid:33)

?) �C?

�C(WV WO)

(12)

Therefore, the cached �C can be directly used for attention computation without explicitly computing
the keys and values, as WK and WV can be absorbed into WQ and WO, respectively.

4

w4w3w2w1UpdateUpdatew1w3Temporal DimensionAttention InputLow-rank Latent Vectors Compressed Temporal-Latent KV CacheMulti-head Keys & ValuesMulti-head QueriesTemporal DimensionAttention InputMulti-head KV CacheMulti-head Queries(a) MHA(b) MTLAFigure 2: Illustration of MTLA inference and training with temporal compression ratio 2. qi: query,
xi: attention input, �cj: compressed KV cache, �c
j: temporary version updated later. (a) Incremental
inference in MTLA, where at certain steps (e.g., 1, 3, 5), the model attends to the temporary �c
j. (b)
KV cache generated by simple pre-downsampling, which mismatches MTLA inference. (c) MTLA
training, where a stride-aware causal mask is used to match the inference condition.

?

?

4.1

Inference using MTLA

Fig. 2(a) illustrates inference using MTLA. Specifically, given a new input vector xi, the correspond-
ing low-rank latent vector ci is first computed following Eq. 8. Then, ci is fed into the hyper-network
to generate the corresponding weight wi. Specifically, the weight is computed as follows:

wi = Sigmoid (Linear(ci) � Linear(pej))

(13)

where j = ?i/s?, Linear denotes a linear layer transformation, pej is the positional embedding at
step j [40], and � denotes element-wise multiplication.

Once wi is obtained, the compressed temporal-latent KV cache can be updated. If the remainder
of i/s equals 1 (assuming i starts from 1), the cache is updated as �C = Concat( �C, wici) where
Concat denote concatenation; otherwise, the j-th cache vector is updated as �cj = �cj + wici. Note
that until the remainder of i/s equals 0, each �cj here actually corresponds to �c
j in Fig. 2, which will
be updated in later steps. Then, the attention output is computed following Eq. 12.

?

4.2 MTLA Training with Stride-aware Causal Mask

As shown in Fig. 2(a), during inference, queries at certain steps attend to the temporary �c
j. As
shown in Fig. 2(b), simply using pre-downsampling to obtain compressed KV vectors for attention
computation during training fails to match inference behaviour. Therefore, enabling efficient parallel
training poses a challenge. This paper proposes a stride-aware causal mask to address this issue.

?

During training, as shown in Fig. 2(c), MTLA computes the compressed temporal-latent KV sequence
as:

?

�C

?

?

= (�c
(cid:124)

1, . . . , �c1
, � � � , �c
(cid:124)
(cid:125)
(cid:123)(cid:122)
s

t, . . . , �ct
)
(cid:125)
(cid:123)(cid:122)
s

(14)

where s is the temporal compression ratio and t = ?T /s?. Therefore, this sequence length remains T
(only in training). To compute the sequence �C
, the compressed low-rank latent vectors C are first
passed through a hyper-network. To ensure parallel training efficiency, MTLA computes �C
using
matrix multiplication. Specifically, the hyper-network generates a weight matrix based on the input
C:

?

?

PE = (pe1, . . . , pe1
(cid:125)

(cid:124)

(cid:123)(cid:122)
s

)
, � � � , pet, . . . , pet
(cid:125)

(cid:124)

(cid:123)(cid:122)
s

W = Sigmoid(Linear(PE) � Linear(C)) ? RT �T

5

(15)

(16)

���w1w1w2w3w3w4w5w5w6w1w2w3w4w5w6(a) MTLA Inference(b) Simple KV Downsamplew1w2......w1w2............w3w4......w3w4c1c2c3c4w1000w1w20000w3000w3w4Chunk Maskingw1�c1=�w1�c1+w2�c2=w3�c3=�w3�c3+w4�c4=q1q2q3q4��q1��q2�q3��q4�q1��q2�q3��q4�q1��q2�q3��q4�q1��q2�q3��q4�q1��-?-?-?q2�-?-?-?q2�q3��-?-?q2�-?q4�Stride-aware Causal MaskingHypernetwork Output WeightsLow-rank Latent VectorsQueriesAttention Scores before Softmax(c) MTLA Training with Stride-aware Causal Mask-??

where PE consists of the replicated positional embedding vectors pej and � denotes matrix multi-
plication. As shown in the upper part of Fig. 2(c), after applying chunk masking (commonly used in
streaming Transformer encoders [9]) to the resulting W, it is multiplied with C to obtain �C
The resulting �C
is then used for attention computation as in Eq. 12 (serving as �C in Eq. 12). However,
instead of using a standard causal mask to prevent access to future information before the softmax, a
stride-aware causal mask is proposed, as shown in the lower part of Fig. 2(c), to match the attention
pattern of MTLA during incremental inference. Specifically, let m denote the row index and n the
column index; the stride-aware causal mask is zero only when n = m or n < m and n mod s = 0,
and ?? elsewhere. With this stride-aware causal mask, MTLA training retains the parallel efficiency
of standard attention.

.

?

4.3 Decoupled Rotary Position Embedding in MTLA

If Rotary Position Embedding (RoPE) [36] is to be used, similar to MLA [24], MTLA also requires
the use of decoupled RoPE [24]. A simple method is proposed in this paper to compress the cached
keys of decoupled RoPE along the temporal dimension. Specifically, the queries obtained from Eq. 7
are rotated with a position-dependent matrix to produce RoPE queries QR = (qR
T ) ?
RT �(nh�dR
h denotes per-head dimension for the decoupled RoPE. Similarly, the keys
can also be obtained as in Eq. 8 and rotated with a position-dependent matrix to obtain RoPE keys
KR = (kR

h ), where dR

2 , � � � , qR

1 , qR

1 , kR

2 , � � � , kR

T ) ? RT �dR
h .

Next, KR is compressed along the temporal dimension to obtain �KR = (�kR
t ) ? Rt�dR
h .
At inference, the most recent element in the RoPE key cache �KR can also be updated. If the remainder
of i/s equals 1, this cache is updated as �KR = Concat( �KR, kR
i ); otherwise, the j-th cache vector is
updated as �kR
i . Then, the RoPE query-key pairs are used to augment the attention computation
and Eq. 11 and Eq. 12 can be rewritten as:

2 , � � � , �kR

j = kR

1 , �kR

(cid:32)

X(WQWK

Y = softmax

?) �C? + QR( �KR)?

?

dh

(cid:33)

�C(WV WO)

(17)

where X ? R1�d in incremental inference, and when multiplying QR ? RT �(nh�dR
RdR
h �T , the head number of keys must first be repeated, following MQA [35].

h ) with ( �KR)? ?

This design of compressing decoupled RoPE keys along the temporal dimension simplifies the
training process: based on Eq. 17, the original KR ? RT �dR
h can be directly used in place of �KR
(also using �C
instead of �C as mentioned in Section 4.2), and the attention output can be computed
with the proposed stride-aware causal mask.

?

Assuming the number of self-attention layers is l, then for standard MHA, each token corresponds
to 2dhnhl elements in the KV cache. For MTLA, for simplicity, this paper follows the hyper-
parameter settings of [24], setting r = 4dh and dR
h = dh/2. Therefore, the average number of
KV cache elements per token in MTLA is 9dhl/(2s). The default value of s is set to 2, making
9dhl/(2s) = 2.5dhl close to the KV cache elements per token in MQA (i.e. 2dhl).

5 Experimental Setup

In this section, the proposed MTLA approach is evaluated on a range of tasks, including speech
translation (ST), text summarisation, automatic speech recognition (ASR), and spoken language
understanding (SLU), and is compared with standard MHA and advanced MLA. Since this work
focuses on self-attention, the experiments are conducted using a Transformer-based decoder-only
architecture, implemented within the Fairseq [30] toolkit.

5.1 Datasets

The ST task uses the MuST-C [14] v1.0 English-German (En-De) dataset, with data preprocessing
following the Fairseq example. The text summarisation task is conducted on the XSum [29] dataset.
For the ASR task, the AMI [7] dataset is employed. For the SLU task, the SLURP [5] dataset is used
to evaluate intent classification. More details of the datasets used are given in Appendix C.

6

5.2 Model Specifications

Since this paper focuses on self-attention, the model is built based on a Transformer decoder, where
the encoder output is prepended to the input of the self-attention module as a prompt, and the cross-
attention module is removed. This is sometimes referred to as a decoder-only structure. As a result,
the cached keys and values will contain information from the encoder output. The proposed MTLA,
along with the standard MHA and the MLA technique, are each used as the self-attention module to
build the model, while all other components are kept strictly identical. In the following sections, the
overall models built with MTLA, MHA, and MLA self-attention modules are referred to as MTLA,
MHA, and MLA for simplicity. The proposed MTLA, along with the standard MHA and the MLA
technique, are each used as the self-attention module to build the model, while all other components
are kept strictly identical. In the following sections, the overall models built with MTLA, MHA, and
MLA self-attention modules are referred to as MTLA, MHA, and MLA respectively.

The decoder used for all tasks shares the same configuration with 512 attention dimensions and 8
heads. For MTLA and MLA, r in Eq. 8 is set to 256 and dR
h is set to 32. In MTLA, the temporal
compression rate s is set to 2 by default unless otherwise specified. For the ST task, following the
Fairseq example, a Transformer encoder is used and initialised with ASR task weights. For the
text summarisation task, a standard Transformer encoder is used. For the ASR task, a Transformer
encoder is employed. For the SLU task, a Conformer [18] encoder is used. More details can be found
in Appendix D.

5.3 Metrics

All inference speed tests are conducted on the same NVidia RTX 6000 Ada GPU. To ensure a fair
comparison, all models used the same batch size and beam size during inference. Inference time
and the average GPU memory usage during inference are reported to evaluate efficiency. For the ST
task, case-sensitive detokenized BLEU [31] is reported. For the text summarisation task, ROUGE
[23] is used to evaluate summarisation quality, and ROUGE-1, ROUGE-2 (unigram and bigram
overlap), and ROUGE-L (longest common subsequence) scores are reported. For speech recognition,
word error rate (WER) results are reported. For the SLU task, accuracy is used to measure intent
classification (IC).

6 Experimental results

This paper evaluates the proposed MLTA across tasks, including ST, text summarisation, ASR, and
SLU, as both speech sequences and document texts are long sequences. Due to our computational
resource constraints that make large-scale pre-training infeasible, all experiments are conducted using
decoder-only architectures trained from scratch, allowing the effectiveness of MTLA to be assessed.
To ensure reproducibility, this paper builds upon standard open-source implementations, such as the
Transformer-based ST example in Fairseq. The goal is not to pursue task-specific state-of-the-art
results, but to systematically compare MTLA with MHA and MLA under consistent and general
model configurations. For each task, representative published results are reported to provide context.

6.1 ST Task Results

The ST results are shown in Table 1. Overall, the models built in this paper achieve competitive
performance on the MuST-C En-De benchmark dataset. The published results listed in Table 1
also use Transformer models, but based on an encoder-decoder architecture with cross-attention.
Table 1 results show that our built decoder-only architecture can achieve similar performance with
the same data and model scale. Comparing MHA and MLA, it is clear that MLA performs well:
MLA results in only a limited reduction in translation quality drop (by 0.19 BLEU points) and
offers improved inference speed and memory efficiency compared to MHA. Building upon MLA,
our proposed MTLA further improves the efficiency of the attention mechanism. With the default
temporal compression ratio (i.e., 2), MTLA even slightly outperforms MHA in translation quality,
suggesting that compressing redundant historical KV information may sometimes benefit model
performance. Compared to MHA, MTLA achieves 4.29� speedup in inference and reduces average
GPU memory consumption by a factor of 6.58.

7

Table 1: BLEU (?) results on the MuST-C En-De tst-COMMON set for multi-head attention (MHA),
multi-head latent attention (MLA), and multi-head temporal latent attention (MTLA). ESPnet-ST
[19] published results are broadly comparable (same data/scale; minor implementation differences).

ST Model

ESPnet-ST [19]
MHA
MLA
Proposed MTLA
Proposed MTLA w/ s = 3
Proposed MTLA w/ s = 4

Quality
(BLEU)
22.9
23.18
22.97
23.28
23.25
23.05

Inference
Time (s)
�
281.3
97.0
65.6
52.7
48.7

Speedup

Inference GPU Memory (MiB)
Avg. Usage Reduction Factor

�
1.00�
2.90�
4.29�
5.34�
5.78�

�
18646
5065
2835
2251
1921

�
1.00
3.68
6.58
8.28
9.71

Assuming the sequence length is T , MTLA reduces the per-token computational complexity during
decoding from O(T ) to O(T /s). Since self-attention is not the only component in the model (e.g.,
feed-forward networks also contribute), setting s = 2 does not directly halve the inference time.
Moreover, the reported GPU memory usage includes both activation memory and the storage of
KV Cache, so memory consumption is not halved either. Nevertheless, setting s = 2 already yields
substantial efficiency gains: MTLA achieves a 1.48� speedup in overall inference and reduces overall
GPU memory consumption by 1.79� compared to MLA. These gains become even more substantial
with larger s. For instance, with s = 4, GPU memory usage is reduced by 2.64�.

6.2 Results on Other Tasks

Experiment conclusions across text summarisation, ASR, and SLU tasks (Tables 2, 3, and 4) are
generally consistent with those from the ST experiments. First, our built models achieve competitive
performance across different tasks. Second, compared to MHA, MLA achieves competitive accuracy
(ROUGE scores, WER, and IC accuracy) and better inference efficiency. Our proposed MTLA
further improves inference efficiency. Compared to MHA, MTLA achieves up to 3.75� speedup and
reductions in GPU memory use by more than a factor of 7, while maintaining or even improving task
performance. These results highlight the broad applicability and practical benefits of our decoder-only
architecture and MTLA KV cache compression method across various sequence tasks.

Table 2: ROUGE (?) results on the XSum test set. ROUGE-1 (R1) (?), ROUGE-2 (R2) (?), and
ROUGE-L (RL) F1 (?) scores are reported. The published result of TransformerABS [27] is broadly
comparable to our results.

ST Model

R1

R2

RL

TransformerABS [27] 29.41 9.77 23.01
28.83 9.67 23.33
MHA
29.39 9.87 23.78
MLA
29.14 9.79 23.60
Proposed MTLA

Inference
Time (s)
�
352.3
141.1
105.2

Speedup

Inference GPU Memory (MiB)
Avg. Usage Reduction Factor

�
1.00�
2.50�
3.35�

�
16141
3746
2198

�
1.00
4.30
7.34

Table 3: WER (?) results on the AMI IHM test set for MHA, MLA, and the proposed MTLA. ESPnet
published [42] results are listed but not directly comparable to our built models.

ST Model

WER

16.49
ESPnet [42]
12.98
MHA
MLA
12.67
Proposed MTLA 12.66

Inference
Time (s)
�
269.4
105.3
71.8

Speedup

Inference GPU Memory (MiB)
Avg. Usage Reduction Factor

�
1.00�
2.56�
3.75�

�
17509
4415
2364

�
1.00
3.97
7.41

6.3 Comparisons with Related Work

This subsection further compares our work with other approaches, including MQA and GQA. First,
MLA and our MTLA follow the hyper-parameter settings of [24], as discussed in Section 4.3. Under

8

Table 4: Accuracy (?) results of intent classification (IC) on the SLURP test set for MHA, MLA, and
the proposed MTLA. ESPnet-SLU [3] published result is generally comparable to our built models.

ST Model

Accuracy

ESPnet-SLU [3]
MHA
MLA
Proposed MTLA

86.3
86.83
86.93
86.80

Inference
Time (s)
�
133.1
61.2
52.7

Speedup

Inference GPU Memory (MiB)
Avg. Usage Reduction Factor

�
1.00�
2.17�
2.53�

�
14370
3343
2051

�
1.00
4.30
7.01

Table 5: BLEU (?) results on the MuST-C En-De tst-COMMON set for related methods, including
Multi-Query Attention (MQA) and Group-Query Attention (GQA) with a group size of 2.

ST Model

MHA
MQA
GQA
MLA
Proposed MTLA
Proposed MTLA w/ s = 3
Proposed MTLA w/ s = 4

Quality
(BLEU)
23.18
22.70
22.75
22.97
23.28
23.25
23.05

Inference
Time (s)
281.3
168.1
190.6
97.0
65.6
52.7
48.7

Speedup

Inference GPU Memory (MiB)
Avg. Usage Reduction Factor

1.00�
1.67�
1.48�
2.90�
4.29�
5.34�
5.78�

18646
3074
5313
5065
2835
2251
1921

1.00
6.07
3.51
3.68
6.58
8.28
9.71

this configuration, each token in MLA results in a KV cache size equivalent to that of GQA with 2.25
groups. Therefore, the GPU memory usage for inference is similar between MLA and GQA. Note
that the GPU memory usage reported here includes both intermediate activations and the KV cache.

Importantly, MLA achieves faster inference than GQA and also outperforms MQA in speed, demon-
strating that storing KV information in low-rank latent vectors and directly using them in attention
reduces computation accelerates inference. Moreover, MLA also outperforms GQA in translation
quality, which is why this paper focuses comparisons to it.

For our proposed MTLA, with the default temporal compression rate s = 2, its pre-token KV cache
elements are equivalent to GQA with 2.25/2 = 1.125 groups. Since MQA corresponds to GQA with
1 group, the KV cache size of MTLA becomes roughly equivalent to that of MQA. This motivates
our choice of s = 2 as the default setting. As shown in Fig. 5, MTLA yields similar memory usage
as MQA while delivering 2.56� inference speedup. This is because MTLA inherits the low-rank
compression benefits of MLA and further reduces per-token complexity from O(T ) to O(T /s), with
T as the sequence length. In contrast, MQA and GQA offer limited speedups over MHA and mainly
reduce GPU memory usage.

As noted in Sec. 5.3, all inference speed tests use the same batch and beam size across models. MTLA
is a more advanced KV compression method than MQA (i.e., GQA with 1 group), which cannot
reduce group count further, while MTLA allows further compression by increasing s. For example,
with s = 4, MTLA significantly outperforms MQA in translation quality (p < 0.05, statistically
tested via SacreBLEU [34]), while also yielding greater inference speed and GPU memory reduction.

7 Conclusions

This paper proposes MTLA, the first work to compress the KV cache along the temporal dimen-
sion. Building upon the low-rank KV compression of MLA, MTLA employs a hyper-network to
dynamically merge adjacent KV caches, enabling effective temporal compression. A stride-aware
causal mask is proposed to ensure that MTLA maintains efficient parallel training while matching the
attention behaviour during incremental inference, addressing the mismatch between the compressed
KV cache length and the processed sequence length. Experiments across ST, text summarisation,
ASR, and SLU show that MTLA greatly accelerates inference and reduces GPU memory usage
at inference without sacrificing accuracy. With a temporal compression rate of 2, MTLA already
matches the KV cache compression level of MQA while delivering better accuracy and speed, and it
supports further compression, establishing itself as a more advanced KV cache compression method.

9

References

[1] Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebr�n, F., Sanghai, S., 2023. GQA:
training generalized multi-query transformer models from multi-head checkpoints, in: Proc.
EMNLP.

[2] Alman, J., Yu, H., 2025. Fundamental limitations on subquadratic alternatives to Transformers,

in: Proc. ICLR.

[3] Arora, S., Dalmia, S., Denisov, P., Chang, X., Ueda, Y., Peng, Y., Zhang, Y., Kumar, S., Ganesan,
K., Yan, B., Vu, N.T., Black, A.W., Watanabe, S., 2022. ESPnet-SLU: Advancing spoken
language understanding through ESPnet, in: Proc. ICASSP.

[4] Ba, J.L., Kiros, J.R., Hinton, G.E., 2016. Layer normalization. arXiv preprint arXiv:1607.06450

.

[5] Bastianelli, E., Vanzo, A., Swietojanski, P., Rieser, V., 2020. SLURP: A spoken language

understanding resource package, in: Proc. EMNLP (1).

[6] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A.,
Shyam, P., Sastry, G., Askell, A., et al., 2020. Language models are few-shot learners, in: Proc.
NeurIPS. URL: https://proceedings.neurips.cc/paper_files/paper/2020/file/
1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

[7] Carletta, J., Ashby, S., Bourban, S., Flynn, M., Guillemot, M., Hain, T., Kadlec, J., Karaiskos,
V., Kraaij, W., Kronenthal, M., Lathoud, G., Lincoln, M., Lisowska, A., McCowan, I., Post,
W.M., Reidsma, D., Wellner, P., 2005. The AMI meeting corpus: A pre-announcement, in: Proc.
MLMI.

[8] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao,
X., Wu, J., Zhou, L., Ren, S., Qian, Y., Qian, Y., Zeng, M., Wei, F., 2021a. WavLM: Large-scale
self-supervised pre-training for full stack speech processing. IEEE J. Sel. Top. Sig. Process. 16,
1505�1518.

[9] Chen, X., Wu, Y., Wang, Z., Liu, S., Li, J., 2021b. Developing real-time streaming Transformer

transducer for speech recognition on large-scale dataset, in: Proc. ICASSP.

[10] Chu, Y., Xu, J., Zhou, X., Yang, Q., Zhang, S., Yan, Z., Zhou, C., Zhou, J., 2023. Qwen-
audio: Advancing universal audio understanding via unified large-scale audio-language models.
arXiv preprint arXiv:2311.07919 URL: https://api.semanticscholar.org/CorpusID:
265157993.

[11] Dao, T., 2024. FlashAttention-2: Faster attention with better parallelism and work partitioning,

in: Proc. ICLR.

[12] Dao, T., Fu, D.Y., Ermon, S., Rudra, A., R�, C., 2022. FlashAttention: Fast and memory-efficient

exact attention with IO-awareness, in: Proc. NeurIPS.

[13] Deng, K., Sun, G., Woodland, P.C., 2025. Wav2Prompt: End-to-end speech prompt learning
and task-based fine-tuning for text-based LLMs, in: Proc. NAACL (Volume 1: Long Papers).

[14] Di Gangi, M.A., Cattoni, R., Bentivogli, L., Negri, M., Turchi, M., 2019. MuST-C: a multilingual
speech translation corpus, in: Proc. NAACL-HLT, Minneapolis, Minnesota. URL: https:
//aclanthology.org/N19-1202.

[15] Fu, Q., Cho, M., Merth, T., Mehta, S., Rastegari, M., Najibi, M., 2024. LazyLLM: Dynamic
token pruning for efficient long context LLM inference. arXiv preprint arXiv:2407.14057 .

[16] Graves, A., Fern�ndez, S., Gomez, F., Schmidhuber, J., 2006. Connectionist temporal classifica-
tion: Labelling unsegmented sequence data with recurrent neural networks, in: Proc. ICML.
URL: https://api.semanticscholar.org/CorpusID:9901844.

[17] Gu, A., Dao, T., 2024. Mamba: Linear-time sequence modeling with selective state spaces, in:

Proc. CoLM. URL: https://openreview.net/forum?id=tEYskw1VY2.

10

[18] Gulati, A., Qin, J., Chiu, C.C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu,
Y., Pang, R., 2020. Conformer: Convolution-augmented Transformer for speech recognition, in:
Proc. Interspeech. doi:10.21437/Interspeech.2020-3015.

[19] Inaguma, H., Kiyono, S., Duh, K., Karita, S., Yalta, N., Hayashi, T., Watanabe, S., 2020.

ESPnet-ST: All-in-one speech translation toolkit, in: Proc. ACL (demo).

[20] de Jong, M., Zemlyanskiy, Y., Ainslie, J., FitzGerald, N., Sanghai, S., Sha, F., Cohen, W.W.,
2023. FiDO: Fusion-in-decoder optimized for stronger performance and faster inference, in:
Proc. ACL (Findings).

[21] Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F., 2020. Transformers are RNNs: Fast

autoregressive transformers with linear attention, in: Proc. ICML.

[22] Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., Chen,
D., 2024. SnapKV: LLM knows what you are looking for before generation. arXiv preprint
arXiv:2404.14469 .

[23] Lin, C.Y., 2004. Rouge: A package for automatic evaluation of summaries, in: Text summariza-

tion branches out.

[24] Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D.,
et al., 2024a. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language
model. arXiv preprint arXiv:2405.04434 .

[25] Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C.,

et al., 2024b. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 .

[26] Liu, A., Liu, J., Pan, Z., He, Y., Haffari, R., Zhuang, B., 2024c. Minicache: KV cache

compression in depth dimension for large language models, in: Proc. NeurIPS.

[27] Liu, Y., Lapata, M., 2019.

Text summarization with pretrained encoders,

in: Proc.

EMNLP/IJCNLP (1).

[28] Meng, F., Tang, P., Yao, Z., Zhang, M., 2025. TransMLA: Multi-head latent attention is all you

need. arXiv preprint arXiv:2502.07864 .

[29] Narayan, S., Cohen, S.B., Lapata, M., 2018. Don�t give me the details, just the summary!
topic-aware convolutional neural networks for extreme summarization, in: Proc. EMNLP.

[30] Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., Grangier, D., Auli, M., 2019.
Fairseq: a fast, extensible toolkit for sequence modeling, in: Proc. NAACL-HLT (Demonstra-
tions). URL: https://aclanthology.org/N19-4009.

[31] Papineni, K., Roukos, S., Ward, T., Zhu, W.J., 2002. BLEU: a method for automatic evaluation
of machine translation, in: Proc. ACL. URL: https://aclanthology.org/P02-1040/.

[32] Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., Cao, H., Cheng,
X., Chung, M., Derczynski, L., Du, X., Grella, M., Gv, K., He, X., Hou, H., Kazienko,
P., Kocon, J., Kong, J., Koptyra, B., Lau, H., Lin, J., Mantri, K.S.I., Mom, F., Saito, A.,
Song, G., Tang, X., Wind, J., Wo�zniak, S., Zhang, Z., Zhou, Q., Zhu, J., Zhu, R.J., 2023.
RWKV: Reinventing RNNs for the transformer era, in: Proc. EMNLP (Findings). URL:
https://aclanthology.org/2023.findings-emnlp.936/.

[33] Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Heek, J., Xiao, K., Agrawal, S.,

Dean, J., 2023. Efficiently scaling transformer inference, in: Proc. MLSys.

[34] Post, M., 2018. A call for clarity in reporting BLEU scores, in: Proc. WMT.

[35] Shazeer, N., 2019. Fast transformer decoding: One write-head is all you need. arXiv preprint

arXiv:1911.02150 .

[36] Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y., 2024. Roformer: Enhanced Transformer

with rotary position embedding. Neurocomputing 568, 127063.

11

[37] Tang, C., Yu, W., Sun, G., Chen, X., Tan, T., Li, W., Lu, L., MA, Z., Zhang, C., 2024.
SALMONN: Towards generic hearing abilities for large language models, in: Proc. ICLR,
Vienna, Austria. URL: https://openreview.net/forum?id=14rn7HpKVk.

[38] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozi�re, B.,
Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., Lample, G., 2023.
LLaMA: Open and efficient foundation language models. arXiv:2302.13971.

[39] Tsunoo, E., Futami, H., Kashiwagi, Y., Arora, S., Watanabe, S., 2024. Decoder-only archi-
tecture for streaming end-to-end speech recognition, in: Proc. Interspeech. doi:10.21437/
Interspeech.2024-705.

[40] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.,

Polosukhin, I., 2017. Attention is all you need, in: Proc. NeurIPS.

[41] Wang, S., Li, B.Z., Khabsa, M., Fang, H., Ma, H., 2020. Linformer: Self-attention with linear

complexity. arXiv preprint arXiv:2006.04768 .

[42] Watanabe, S., Hori, T., Karita, S., Hayashi, T., Nishitoba, J., Unno, Y., Enrique Yalta Soplin, N.,
Heymann, J., Wiesner, M., Chen, N., Renduchintala, A., Ochiai, T., 2018. ESPnet: End-to-end
speech processing toolkit, in: Proc. Interspeech. doi:10.21437/Interspeech.2018-1456.

[43] Wu, J., Gaur, Y., Chen, Z., Zhou, L., Zhu, Y., Wang, T., Li, J., Liu, S., Ren, B., Liu, L., Wu, Y.,
2023. On decoder-only architecture for speech-to-text and large language model integration, in:
Proc. ASRU. doi:10.1109/ASRU57964.2023.10389705.

[44] Yang, S., Chi, P., Chuang, Y., Lai, C.J., Lakhotia, K., Lin, Y.Y., Liu, A.T., Shi, J., Chang, X.,
Lin, G., Huang, T., Tseng, W., Lee, K., Liu, D., Huang, Z., Dong, S., Li, S., Watanabe, S.,
Mohamed, A., Lee, H., 2021. SUPERB: Speech processing universal performance benchmark,
in: Proc. Interspeech.

[45] Zuhri, Z.M.K., Adilazuarda, M.F., Purwarianti, A., Aji, A.F., 2025. MLKV: Multi-layer key-
value heads for memory efficient transformer decoding, in: Proc. NAACL (Findings). URL:
https://aclanthology.org/2025.findings-naacl.305/.

12

A Limitations

Due to limited computational resources, this work does not investigate large language model (LLM)
pre-training. The proposed MTLA is designed specifically for decoder-only architectures and can
efficiently compress the KV cache. Standard text-based LLMs are successful examples of decoder-
only models. Recent studies have shown that pre-pending speech representations as prompts before
the self-attention input can extend text-based LLMs to speech tasks. However, building an LLM
based on MTLA or replacing self-attention in a pre-trained LLM with MTLA and re-training it
requires very substantial computational resources, which we do not possess. As a result, we are
unable to construct an LLM based on MTLA to verify its extension to other tasks, such as speech.
Instead, we construct decoder-only models and train them from scratch to evaluate MTLA across a
range of tasks.

Second, as Transformer-based models have been extensively developed by the community in recent
years, there is a large amount of related work. It is not feasible for us to implement and compare all
such approaches. In this work, we select and compare against the most relevant and representative KV
cache compression methods, including MQA, GQA, and MLA. Further comparisons are only feasible
through theoretical discussion, as presented in Section 2. Additionally, engineering techniques such
as FlashAttention are currently not directly compatible with MLA or MTLA. Although in theory they
could be integrated, doing so would require substantial engineering effort to extend FlashAttention,
and we leave this as future work. On the other hand, we are also limited by the type of GPU available.
For example, similar engineering tricks like FlashMLA rely on NVidia Hopper GPUs, while we only
have NVidia RTX 6000 Ada GPUs available.

This work focuses on long-sequence tasks, particularly speech, due to the naturally long sequence
length of speech inputs. We also conduct evaluations on a text summarisation task. While many addi-
tional tasks could be used to further evaluate its effectiveness of MTLA, we leave such investigations
for future work. Given the growing dimensionality of modern LLMs and the increasing use of long
reasoning chains to improve output quality, the MTLA, which compresses the KV cache along both
the latent and temporal dimensions, can be particularly valuable.

B Broader impact

Decoder-only architectures based on self-attention have become increasingly popular in recent years,
especially in the context of large language models (LLMs). However, due to their high dimensionality
and massive number of parameters, LLMs incur expensive inference costs and are heavily dependent
on GPUs. This problem is further exacerbated by the use of chain-of-thought, which enhances
reasoning ability but results in significantly longer output sequences, making inference even more
costly. Such inference consumes substantial energy from GPUs. By contrast, our proposed MTLA
compresses the Key-Value Cache in both latent and temporal dimensions, greatly improving inference
efficiency, which can be of great value to make LLMs more energy-efficient and environmentally
sustainable. Therefore, our work has the potential to generate a positive societal impact. We do not
know of any negative societal impact.

C Data Set Statistics

The ST task uses the MuST-C [14] v1.0 English-German (En-De) dataset, with data preprocessing
following the Fairseq example, using 8,000 unigrams as the target language modelling units and fbank
features as input. The text summarisation task is conducted on the XSum [29] dataset, where 30,000
BPE units are used. For the ASR task, the AMI [7] dataset is employed. Due to the challenging
nature of the data, fixed WavLM [8] Large features are extracted using the S3PRL [44] toolkit as
input. When measuring inference speed, this feature is pre-stored and 100 BPE units are used. For
the SLU task, the SLURP [5] dataset is used to evaluate intent classification, with fbank features as
input. Following [3], intent classification is performed by jointly predicting the transcription and the
intent to achieve better performance. A total of 500 BPE units are used for transcription modelling.

The data set statistics for the datasets used in the experiments are shown in Table 6. The MuST-C [14]
v1.0 En-De dataset comprises English-German speech translation data collected from TED Talks. The
Augmented Multi-Party Interaction (AMI) Meeting Corpus [7] offers 100 hours of English meeting

13

recordings captured in instrumented rooms, featuring multimodal data such as audio, video, and
whiteboard content, with annotations including speech transcriptions and dialogue acts. The Spoken
Language Understanding Resource Package (SLURP) [5] dataset is a comprehensive English spoken
language understanding resource encompassing 18 domains, designed to facilitate tasks like intent
classification and slot filling, with a diverse set of utterances. The XSum [29] dataset consists of BBC
news articles from 2010 to 2017, each paired with a single-sentence abstractive summary, totalling
over 226K document-summary pairs, and is widely used for evaluating summarisation models.

MuST-C v1.0 En-De
TED Talk
train
400.0 hours
3880K

dev
2.3 hours
26K

tst-COMMON
4.1 hours
44K

XSum Dataset
BBC News Articles
train
204K
431 words
23 words

test
11K
AMI Meeting Corpus
Meetings
train
100.0 hours
108K

test
12K

dev
11K

dev
13K

Domain
Train set
-Duration
-German words
Test sets
-Duration
-German words

Domain
Train set
-Documents
-Avg. article length
-Avg. summary length
Test sets
-Documents

Domain
Train set
-Duration
-Utterances
Test sets
-Utterances

Domain
Train set
-Duration
-Utterances
Test sets
-Duration
-Utterances

SLURP Dataset
Human-Computer Interaction (HCI) commands
train
83.7 hours
120K

dev
6.9 hours
9K

test
10.3 hours
13K

Table 6: Statistics of datasets used in this paper

D Hyper-parameter Details and Training

The decoder used for all tasks shares the same configuration: 9 layers, 512 attention dimensions, 2048
feed-forward dimensions, and 8 attention heads. The encoder for all tasks also uses this configuration,
except that the number of layers is increased to 12. For MTLA and MLA, r in Eq. 8 is set to 256
and dR
h is set to 32. In MTLA, the linear layers in Eq. 13 and Eq. 16 map the 256-dimensional input
to a 64-dimensional space. The temporal compression rate s is set to 2 by default unless otherwise
specified. Standard RoPE [36] is used in MHA to obtain positional information. Decoupled RoPE is
used in MLA, which is also employed in the proposed MTLA along with the temporal compression
described in Section 4.3.

For the ST task, following the Fairseq example, a Transformer encoder (including convolutional layers
for 4� downsampling) is used and initialised with ASR task weights. For the text summarisation task,

14

a standard Transformer encoder is used. For the ASR task, a Transformer encoder with convolutional
layers for 2� downsampling is employed. In addition to the cross-entropy loss, a connectionist
temporal classification (CTC) [16] auxiliary loss is computed with weight 1. For the SLU task, a
Conformer [18] encoder is used; beyond the configuration (e.g. 512 attention dimension) used in
Transformer, its depthwise convolutional layer has a kernel size of 31. Before entering the Conformer,
a convolutional layer with 2� downsampling is applied.

For the ST task, training follows the Fairseq example, using a learning rate of 2e-3, 10,000 warm-up
steps, and a maximum of 100,000 update steps. Each batch corresponds to 320,000 frames of Fbank
features, which is approximately 53 minutes of speech. The MHA, MLA, and MTLA models all
have 78M parameters. The GQA and MQA models constructed in Section 6.3 have 74M parameters.
During inference, each batch corresponds to 50,000 frames of Fbank features, and the beam size is
set to 50. For the text summarisation task, training uses a learning rate of 2e-4, 15,000 warm-up steps,
and a maximum of 60,000 update steps. Each batch corresponds to 40000 tokens. The MHA, MLA,
and MTLA models all have 79M parameters. During inference, each batch corresponds to 60,000
tokens, and the beam size is set to 10. For the ASR task, training uses a learning rate of 2e-4, 15,000
warm-up steps, and a maximum of 10,000 update steps. Each batch corresponds to approximately 16
minutes of speech. The MHA, MLA, and MTLA models all have 67M trainable parameters. During
inference, each batch corresponds to 20 minutes of speech, and the beam size is set to 20. For the
SLU task, training uses a learning rate of 2e-4, 50,000 warm-up steps, and a maximum of 30,000
update steps. Each batch corresponds to 18,000 frames of Fbank features. The MHA, MLA, and
MTLA models all have 103M parameters. During inference, each batch corresponds to 130,000
frames of Fbank features, and the beam size is set to 10.

Model training was performed on a single NVidia RTX 6000 Ada GPU with 48GB of memory. For
the ST task, each epoch took about 13 minutes. For the text summarisation, each epoch took about 20
minutes. For the ASR task, each epoch took about 50 minutes. For the SLU task, each epoch took
about 15 minutes.

E Assets and licenses

The following licenses apply to the datasets used in this paper:

� CC-BY-NC-ND-4.0: https://spdx.org/licenses/CC-BY-NC-ND-4.0 applies to

MuST-C data.

� CC-BY-SA-4.0: https://spdx.org/licenses/CC-BY-SA-4.0 applies to XSum data.
� CC BY 4.0: https://spdx.org/licenses/CC-BY-4.0 applies to AMI data.
� CC BY-NC 4.0: https://spdx.org/licenses/CC-BY-NC-4.0 applies to SLURP data.

The following license applies to the code and Python package used in this paper:

� Apache-2.0: applies to Fairseq (https://github.com/facebookresearch/fairseq/

blob/main/LICENSE).

15


