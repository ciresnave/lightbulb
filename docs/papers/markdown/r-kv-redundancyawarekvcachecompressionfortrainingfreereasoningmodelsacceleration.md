5
2
0
2

n
u
J

3

]
L
C
.
s
c
[

2
v
3
3
1
4
2
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

R-KV: Redundancy-aware KV Cache Compression for
Training-Free Reasoning Models Acceleration

Zefan Cai1, Wen Xiao2(cid:66), Hanshi Sun3, Cheng Luo4, Yikai Zhang1, Ke Wan5, Yucheng Li6,
Yeyang Zhou5, Li-Wen Chang, Jiuxiang Gu7, Zhen Dong8, Anima Anandkumar4,
Abedelkadir Asi2, Junjie Hu1(cid:66)
1University of Wisconsin - Madison 2Microsoft 3Carnegie Mellon University
4California Institute of Technology 5University of California - San Diego 6University of Surrey
7Adobe 8University of California - Berkeley
https://zefan-cai.github.io/R-KV.page/
https://github.com/Zefan-Cai/R-KV
https://github.com/Zefan-Cai/KVCache-Factory

Abstract

Reasoning models have demonstrated impressive performance in self-reflection
and chain-of-thought reasoning. However, they often produce excessively long
outputs, leading to prohibitively large key-value (KV) caches during inference.
While chain-of-thought inference significantly improves performance on complex
reasoning tasks, it can also lead to reasoning failures when deployed with existing
KV cache compression approaches. To address this, we propose Redundancy-
aware KV Cache Compression for Reasoning models (R-KV), a novel method
specifically targeting redundant tokens in reasoning models. Our method preserves
nearly 100% of the full KV cache performance using only 10% of the KV cache,
substantially outperforming existing KV cache baselines, which reaches only
60% of the performance. Remarkably, R-KV even achieves 105% of full KV
cache performance with 16% of the KV cache. This KV-cache reduction also
leads to a 90% memory saving and a 6.6� throughput over standard chain-of-
thought reasoning inference. Experimental results show that R-KV consistently
outperforms existing KV cache compression baselines across two mathematical
reasoning datasets.

1

Introduction

Recent advancements in large language models (LLMs) have demonstrated remarkable capabilities in
complex reasoning and self-reflection. However, reasoning models (e.g., DeepSeek-R1 [1]) exhibit a
critical deployment challenge: their tendency to produce excessively lengthy and redundant reasoning
traces results in unsustainable memory demands [2], primarily due to the rapid growth of the key-
value (KV) cache during autoregressive generation. For instance, a DeepSeek-R1-Distill-Llama-8B
model may generate 32K tokens to solve a complex math problem, consuming 15.5GB of memory
to load the model weight and 4.1GB of memory to store the KV cache. This paradigm of long
chain-of-thought (CoT) reasoning generation necessitates the development of KV cache compression.

Outputs from current reasoning models, especially during complex chain-of-thought generation,
are fundamentally marked by pervasive redundancy. This inherent characteristic means they are
often filled with superfluous content, including unnecessary reflections, iterative re-evaluations, and
verbose self-dialogue, all of which add little new semantic value while significantly inflating the

(cid:66) Corresponding to Wen Xiao wxiao@microsoft.com and Junjie Hu junjie.hu@wisc.edu

Preprint. Under review.

Figure 1: R-KV: (1) Decoding-Time Compression (�3.1); (2) KV Cache Selection with Importance
and Redundancy Estimation (�3.2, �3.3) ; (3) KV Cache Compression by joint selection (�3.4).

length of the generation beyond what is needed for concise, effective reasoning. Our analysis (�2.1)
shows that over half of the tokens in R1�s reasoning chains contribute minimally to task performance,
indicating that repetitive self-verification steps or intermediate calculations could be substantially
condensed by KV cache compression methods without compromising reasoning accuracy.

However, existing KV cache compression works [3, 4, 5, 6, 7] primarily handle long input prompts but
do not explore extensively for long generation outputs. Furthermore, based on our observation (�2.2),
standard KV-cache compression methods that rely on simple attention-based importance filtering
often fail because the repetitive sections generate high attention signals for themselves. Naively
pruning tokens with �low attention weight� may remove crucial but scattered bits of reasoning, or
over-retain duplicative self-reflections that appear to have high attention. This observation motivates
our exploration of redundancy-aware compression strategies, which selectively retain �important and
non-repetitive context� during decoding to preserve the model�s critical reasoning ability.

In this work, we propose Redundancy-aware KV cache compression for reasoning models (i.e.,
R-KV). Our approach consists of three key components: (1) an attention-based importance scoring
mechanism that selects critical tokens for retention, (2) a dynamic redundancy scoring mechanism
that identifies repetitive tokens through real-time analysis of key vectors, and (3) a joint eviction
mechanism that balances both redundancy and importance to optimize cache efficiency.

In our experiments on popular math reasoning benchmarks (�4), by selectively retaining only 10-34%
of the original KV cache, R-KV achieves comparable performance parity with the uncompressed
reasoning model, outperforming state-of-the-art compression baselines with only 60% of the perfor-
mance. Remarkably, R-KV even achieves 105% accuracy of the full KV baseline with around 16%
of the KV cache using DeepSeek-R1-Distill-Llama-8B on the AIME-24 dataset.

This advancement addresses a fundamental tension in deploying state-of-the-art LLMs�balancing
reasoning capabilities with practical memory constraints. Our contributions extend beyond technical
optimization: we provide systematic evidence that redundancy in CoT generation can be strategically
compressed without compromising reasoning abilities. As a training-free and model-agnostic method,
R-KV can be used in the rollout process in reinforcement learning (RL) and LLM serving.

2 Observation

2.1 Redundancy in Reasoning Models

As noted in [2], reasoning models often generate a detailed chain of thoughts and multiple reflec-
tion steps, resulting in significantly longer responses than standard models. Figure 2 shows that
both reasoning models (i.e., DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B and
DeepSeek-R1-Distill-Qwen-14B) generate more than 8� longer generation output compared to the
ground truth on two popular math reasoning datasets. However, not all of the additional tokens
contribute meaningful content, as much of the decoded context is dominated by repetition. Figure 2

2

Redundency EstimnationImportance ScoringAttentionSimilarity- (1 � ?)+ ?Joint Selection StrategyKey CacheQuery CacheLet ? = 0.5CosineKV Cache Compression0312131415161617KV Cache SelectionDecoding-Time CompressionBefore KV Cache CompressionAfter KV Cache Compression1617014161617031213141516R-KV KV Cache CompressionDecoding1213031220210121617181920Decoding0.20.10.20.10.10.10.20.30.10.10.10.10.10.20.20.10.20.10.10.10.20.30.10.10.10.10.10.20.050.-0.05000.05-0.05Key Cache031213141516Value Cache0312131415160141601416Figure 2: Comparison of generation length and average 1-/2-gram frequency for reasoning models and
ground truth of MATH-500 [8] and AIME 2024 [9]. . Reasoning models generate substantially longer
responses with 8-14� more tokens, and show higher word repetition with 5-7� higher frequency.

also shows that the average frequency of 1- to 2-grams is consistently higher in the generation output
of reasoning models than in ground truth, indicating greater repetitions in the generated outputs of
reasoning models.

2.2 Failure of Existing KV Compression Methods to Handle Redundancy

Most existing KV cache compression
methods prioritize token selection based
primarily on tokens� contextual impor-
tance, typically measured through atten-
tion scores between key and query to-
kens [3, 4]. While this approach effec-
tively retains critical context, it fails to ac-
count for redundancy�particularly prob-
lematic in reasoning models.
In such
models, we find that repetitive content
often receives disproportionately high at-
tention scores, as it closely mirrors pre-
viously generated repetitive text. As a
result, redundant tokens are excessively
retained, unnecessarily inflating the KV
cache size without providing additional
meaningful new information.
In Fig-
ure 3, we visualize the cached tokens
(inside red boxes) selected by a popular
attention-based KV cache method (i.e.,
SnapKV), showing many repetitions re-
lated to self-reflection and conclusion to
the final answer.

Figure 3: KV selected by SnapKV. SnapKV suffers from
redundancy in reasoning models. Black tokens are not se-
lected by SnapKV; brighter colors reflect higher attention
scores. Blue tokens are omitted output.

3 Redundancy-aware KV Cache Compression (R-KV)

To address the redundant thinking issue, we propose a redundancy-aware decoding-time KV cache
compression method (R-KV) that explicitly targets the compression of redundant tokens in reasoning
models. Our approach balances importance and non-redundancy in token selection, ensuring that
KV cache storage is allocated to both highly informative and diverse content. By incorporating

3

GTQwen-14BLlama-8BQwen-7B050010001500200025003000Average Length (tokens)209.72833.02979.02932.0Math-500  Average Length (tokens)GTQwen-14BLlama-8BQwen-7B05010015020031.2167.6214.4197.7Math-500  1-gram - Average FrequencyGTQwen-14BLlama-8BQwen-7B0.02.55.07.510.012.515.017.55.213.316.615.5Math-500  2-gram - Average FrequencyGTQwen-14BLlama-8BQwen-7B0250050007500100001250015000Average Length (tokens)1547.012402.015535.814592.0AIME24  Average Length (tokens)GTQwen-14BLlama-8BQwen-7B02040608010017.284.198.381.9AIME24  1-gram - Average FrequencyGTQwen-14BLlama-8BQwen-7B0246810123.59.811.710.2AIME24  2-gram - Average FrequencyGround TruthDeepSeek-R1-Distill-Qwen-14BDeepSeek-R1-Distill-Llama-8BDeepSeek-R1-Distill-Qwen-7BYou are given a math problem. Problem: In Mr. Roper's class of 30 students, [Question and Instruction - 102 words]First, the problem says that there are 30 students in total in the class. Out of [Think - 203 words]�[Reflection for 13 times and 581 words in total]�But wait, the � So, 10% of 30 is 3. So 3 students are leaving early.[Think - 36 words]But in the initial problem�So 3 students are leaving early.[Think - 42 words]But wait, the �30 is 3. So 3 students are leaving early.[Think - 36 words]But in the initial problem, the... So 3 students are leaving early.[Think - 42 words]But wait, the � early?" So, 10% of 30 is 3. So 3 students are leaving early.[Think - 36 words]But in the initial �" So, 10% of 30 is 3. So, 3 students are leaving early.[Think - 40 words]But wait, the user wrote: �10% of 30 is 3. So, 3 students are leaving early.[Think - 31 words]But in the initial � of 30 is 3. Therefore, 3 students are leaving early.[Think - 83 words]I think that's all. The calculation is straightforward: 10% of 30 is 3.[Conclusion- 11 words]redundancy estimation into the selection process, our method effectively mitigates unnecessary KV
cache growth while preserving the model�s reasoning capabilities.

Specifically, R-KV consists of three key components: (1) an importance scoring mechanism (�3.2)
leveraging attention weights, (2) a redundancy estimation mechanism (�3.3) based on semantic
similarity of key vectors, and (3) a joint selection strategy (�3.4) that optimizes cache efficiency by
balancing redundancy and importance.

3.1 Decoding-time Compression

Different from existing KV cache compression methods[3, 5, 4] that focus on the prefilling stage
to manage long-context inputs, our R-KV focuses on the decoding stage for reasoning models�a
distinctive setting where the generated output is significantly longer than the prompt.

Specifically, R-KV allocates memory for two components: a cache of budget size Bbudget to store
retained KV tokens, and a buffer of size Bbuffer for newly generated text tokens. The total memory
requirement is thus Btotal = Bbudget + Bbuffer. After the model generates each fixed-length text
segment in the buffer, R-KV performs KV cache compression. At the end of each text segment, the
last ? tokens are always retained in the cache as observation tokens, following prior work [3]. Next,
we concatenate the existing Bbudget tokens in the cache with the first Bbuffer ? ? tokens in the buffer,
resulting in n = Bbudget + Bbuffer ? ? candidate KV tokens. Each candidate is assigned a selection
score (�3.4), and we select the top k = Bbudget ? ? tokens to fit in the rest of the cache budget, in
addition to the ? observation tokens. This process compresses the KV cache while preserving critical
context, enabling efficient memory utilization during autoregressive decoding.

3.2

Importance Scoring via Attention Weights

Following attention-based methods (e.g., SnapKV [3], PyramidKV [5]), R-KV estimates token
importance using attention weights, leveraging the intuition that tokens receiving higher attention
contribute more to decoding and are thus more critical for preserving model performance. Specifically,
we compute each key token�s attention scores received from the last ? observation tokens during
decoding. In addition to the standard multi-head attention mainly adopted by the prior works [3], we
also propose the importance score estimation using the grouped-query attention. Below, we detail the
estimation on top of these two popular attention mechanisms used by current LLMs.

Multi-Head Attention (MHA). Given the last ? observation tokens as query Qh ? R?�d and n
key states Kh ? Rn�d for each attention head h, the attention scores Ah ? R?�n are computed as:

Ah = softmax(Qh � (Kh)?/

d).

(1)

?

In GQA, each key/value head h is shared among a group of
Grouped-Query Attention (GQA).
G distinct query heads indexed by g ? [0, G). Correspondingly, we denote the shared key/value
states as Kh, V h ? Rn�d, and the G query states as Qh,0, . . . , Qh,G?1 ? R?�d within the head
group indexed by h, where n is the number of key/value states, d is the head hidden dimension. The
attention score for each of the G query heads within the group is computed as:

Ah,g

group = Qh,g � (Kh)?/

d ? R?�n,

for g = 0, . . . , G ? 1.

(2)

?

These G individual matrices are then aggregated into a single consolidated matrix Ah
group for the head
group h using a max-pooling operation across the group dimension. The final attention weight Ah
for the head group h is then obtained by renormalizing Ah

group along the key token dimension.
(cid:1) ? R?�n

(cid:3)(cid:1) ? R?�n, Ah = softmax (cid:0)Ah

group

(3)

Ah

group = maxpool (cid:0)(cid:2)Ah,0

group, . . . , Ah,G?1

group

Stabilization and Importance Estimation. We use Ah hereafter to denote the attention weights
calculated using either MHA or GQA. Note that the per-token importance scores derived from Ah may
contain outliers with excessively high values, resulting in unstable estimation of importance scores.
To mitigate this influence, we follow the prior work [3] and apply a max-pooling operation to these
per-token importance scores over a sliding window of size 2W across recent tokens. Specifically, we
j,i as the attention score from the j-th query to the i-th key in Ah. We obtain the stabilized
denote Ah

4

attention score �Ah by computing its (i, j) entry, and finally obtain the importance score of retaining
the i-th token in the KV cache as Ih

i for each attention head h, as shown below:

j,i = max (cid:0)Ah
�Ah

j,i?W , . . . , Ah

j,i, . . . , Ah

j,i+W ?1

(cid:1) ,

I h
i =

1
?

??1
(cid:88)

j=0

�Ah

j,i ? R.

(4)

3.3 Redundancy Estimation via Semantic Similarity

To identify redundant tokens, we measure the semantic similarity between key states using cosine
similarity. Tokens with high similarity to others are considered potentially redundant and can be
selectively removed to optimize KV cache memory.

Cosine Similarity between Key Tokens: Given the key tokens Kh ? Rn�d for a specific head h,
h
We first normalize each key vector Kh
i , and then compute the cosine similarity
matrix Sh using the normalized key vectors.

i , ?i ? [0, 1) into K

K

h
i =

Kh
i
i ?2 + ?

?Kh

? Rd, Sh = K

h

h

(K

)? ? Rn�n, Sh

i,i ? 0, ?i ? [0, n),

(5)

where ? � ?2 is the L2 norm and ? is a small constant (e.g., 10?8) for numerical stability. To prevent
tokens from being marked as redundant with themselves, we zero out the diagonal elements Sh
i,i.

Enforce Retention of Recent Tokens. While redundant, such tokens may still carry meaningful
information. Thus, naively removing all redundant tokens can impair model performance. To address
this, we retain only the ? most recently generated tokens among those exhibiting high similarity,
as these later tokens tend to better support the model�s decoding than earlier ones. To enforce
this, we further zero out the similarity scores in Sh corresponding to these ? most recent similar
tokens. Formally, for each token i ? [0, n), we identify the set of indices of highly similar tokens:
I h
i = {j | Sh
j,i > T, j ? [0, n)}, where T is a fixed hyperparameter for similarity threshold. For this
set, we extract the subject I h
i , containing up to the ? largest indices�i.e., the ? most recent
similar tokens to token i, or fewer if not enough such tokens exist. We then suppress their influence
by zeroing out their similarity scores with token i in Sh, i.e., Sh
i,?. This modification
effectively nullifies the direct similarity links from token i to its ? most recent highly similar tokens.

j,i ? 0, ?j ? I h

i,? ? I h

i . Intuitively, �Sh

Redundancy Score Estimation: Finally, we compute normalized redundancy scores for all key
tokens in Eq. (6). First, for each key token i ? [0, n) in each head h, we compute its average similarity
score �Sh
i measures how similar token i is, on average, to all other key tokens in the
sequence. A high �Sh
i indicates that the semantic content of token i is largely shared with other tokens,
suggesting potential redundancy. Next, to obtain per-token redundancy scores Rh
i within a fixed
numerical range for each head h, we normalize �Sh
i using a softmax operation. The resulting score
Rh
i reflects the redundancy of token i for head h, with higher values indicating greater redundancy.

�Sh
i =

1
n

n?1
(cid:88)

j=0

j,i, Rh
Sh

i = (cid:0)softmax (cid:0)[ �Sh

0 , . . . , �Sh

n?1](cid:1)(cid:1)

i

(6)

3.4

Joint Selection Strategy for KV Cache Retention

To efficiently manage KV cache storage while retaining essential context, we employ a joint selection
strategy that integrates both importance and redundancy scores. Given a predefined token budget
Bbudget per attention head, our goal is to retain tokens that maximize information diversity while
minimizing redundancy. The final selection score Z h

i for each token i in head h is computed as:

Z h

i = ?I h

i ? (1 ? ?) Rh
i ,
i and the redundancy score Rh

where the importance score I h
respectively. A higher I h
i
while a higher Rh

i are computed in Eq. (4) and Eq. (6)
indicates that a token is more important and should ideally be retained,
i suggests higher token redundancy. The hyperparameter ? controls the trade-off

(7)

5

Figure 4: Results of R-KV compared with SnapKV and FullKV on the MATH-500 and AIME24
datasets for R1-Llama-8B (top) and R1-Qwen-14B (bottom). Results are reported as pass@1 based
on 64 generated responses per question.

between prioritizing important tokens and reducing redundant tokens. We discuss the rationale for
choosing ? through a sensitivity analysis in �5.1. This strategy ensures that the KV cache prioritizes
storing tokens that are both important and semantically diverse, thereby improving memory efficiency
without compromising model performance.

4 Experiment

4.1 Experimental Setup

Models and Datasets
In our experiments, we use variants of the DeepSeek-R1 distilled model:
DeepSeek-R1-Distill-Llama-8B, and DeepSeek-R1-Distill-Qwen-14B [1], which we refer to as
R1-Llama-8B and R1-Qwen-14B, respectively, for brevity throughout the paper.

We evaluate the models� mathematical reasoning capabilities using three benchmarks: MATH-500
[8] and AIME 2024 [9].

Hyperparameters We set Bbuffer = 128, ? = 8 and ? = 0.1, with an analysis of ? in �5.1.

Baselines We compare our method against SnapKV [3], originally designed for long prefilling.
To adapt it for decoding, we apply the same compression interval as our method, i.e., compressing
the KV cache every 128 decoding steps using identical Bbudget and Bbuffer. Our approach focuses
on improving KV cache eviction through a hybrid strategy, and we therefore restrict comparison to
state-of-the-art attention-based eviction methods. Budget allocation techniques (e.g., head-level [6]
and layer-level [5]) are orthogonal to our work and not included. We also report results for FullKV,
which retains the full KV cache and serves as the gold standard for decoding quality.

Evaluation Setup We set the maximum generation length to 16,384 tokens for MATH-500 and
32,768 tokens for AIME 2024 and AIME 2025, because further increasing the generation length
has shown no improvement on model performance on these datasets from our attempts. We find
that using greedy decoding to evaluate long-output reasoning models results in significant variability
across different setups. Following existing works [1], we utilize pass@k evaluation [10] and report

6

4%17%34%52%69%KV Cache Budget5001000150020004050607080Performance (Acc.)DeepSeekR1DistillLlama8B  MATH-500R-KVSnapKVFullKV2%3%5%7%10%13%16%20%KV Cache Budget05001000150020002500300001020304050Performance (Acc.)DeepSeekR1DistillLlama8B  AIME24RKVSnapKVFullKV5%9%18%27%36%54%72%KV Cache Budget500100015002000406080Performance (Acc.)DeepSeek-R1-Distill-Qwen-14B  MATH-500R-KVSnapKVFullKV2%4%8%12%17%25%33%KV Cache Budget010002000300040000204060Performance (Acc.)DeepSeekR1DistillQwen14B  AIME24RKVSnapKVFullKVpass@1 using a non-zero temperature. We use the recommended sampling temperature and top-p
value for each model, i.e., sampling temperature of 0.6 and a top-p value of 0.95 for DeepSeek-
R1 Distilled models. We generate 64 responses for each question. Pass@1 is then calculated as
Pass@1 = 1
i=1 pi, where pi denotes the correctness of the i-th response. This method provides
k
more reliable performance estimates.

(cid:80)k

4.2 Results

The accuracy performance of R-KV compared with all baselines is shown in Figure 4, with detailed
accuracy numbers in Appendix B.2. The KV cache budget ratio is calculated based on the KV cache
budget and the average generation length of tokens, i.e., R1-Llama-8B: 2, 979.1 on MATH-500
and 15, 535.8 on AIME24; R1-Qwen-14B: 2, 833.04 on MATH-500 and 12, 402 on AIME24. Our
method significantly outperforms the baseline SnapKV, achieving up to 40% Acc. improvement. We
provide two KV cache budget and performance analysis. Fixed budget analysis is more practical
because when the model outputs longer (i.e., from 2, 979.1 on MATH-500 to 15, 535.8 on AIME24),
the KV cache budget needed for lossless compression increases less (i.e., 512). In the KV cache
budget ratio perspective, the changes of lossless compression ratio is dominated by generation length.

For R1-Llama-8B, R-KV achieves lossless compression with 34% KV cache budget
Ratio Budget
on the MATH-500 dataset and with 10% KV cache budget on the AIME-2024 dataset. Given 16%
KV cache budget, our method even surpasses the FullKV baseline, reaching 105% of its accuracy.
Similarly, for R1-Qwen-14B, R-KV achieves lossless compression with 54% KV cache budget on
the MATH-500 dataset and with 25% KV cache budget on the AIME-2024 dataset. Given 33% KV
cache budget, our method achieves 105% of FullKV accuracy.

Fixed Budget For R1-Llama-8B, R-KV achieves lossless compression with 1024 KV cache budget
on the MATH-500 dataset and with 1536 KV cache budget on the AIME-2024 dataset. For R1-Llama-
8B, R-KV achieves lossless compression with 1536 KV cache budget on the MATH-500 dataset and
with 3072 KV cache budget on AIME-2024.

5 Discussion

5.1 How to Choose ??

Figure 5 shows the distributions of the Importance Score (Ih) and Redundancy Estimation (Rh) for
head h = 0 at the top layer (Nlayer = 31). The figure reveals that Ih is sparse and dominated by a
few outlier values, while the similarity distributions (which inform Rh) are relatively dense. When
? = 0, the token retention strategy is overned entirely by Redundancy Estimation (Rh). As shown in
Figure 5, the initial four tokens are not guaranteed to be preserved. As highlighted by prior work [7],
evicting these initial tokens can severely impair the generative capabilities of LLMs. Therefore, it is
crucial to select a ? value that starts from at least 0.01. On the other hand, as ? increases beyond 0.1,
the selection metric becomes increasingly dominated by attention scores. These observations suggest
that an optimal ? lies within the range of 0.01 ? ? ? 0.1, effectively balancing the contributions of
Importance Score and Redundancy Estimation.

Figure 6 presents the accuracy (Acc.) performance of R-KV on the DeepSeek-Distill-R1-Llama-8B
model using the MATH-500 dataset. The results further guide the choice of ? for optimal performance.
The figure demonstrates that ? = 0.1 yields the highest accuracy. In contrast, strategies relying solely
on redundancy (? = 0) or solely on attention (? = 1) exhibit the poorest performance, underscoring
the complementary nature of these two metrics and the importance of a balanced approach. Thus,
based on this finding, we select ? = 0.1 for all evaluations detailed in Figure 4.

5.2 Failure of Attention-Based Methods to Capture Redundancy

To thoroughly investigate the advantages of R-KV�s hybrid selection metrics (combining attention
and redundancy) over pure attention-based importance metrics, we compared the tokens selected by
R-KV against those chosen by a pure attention-based method (SnapKV). We present a case where
R-KV correctly completes the task while the comparison method fails. As illustrated in Figure 7,

7

Figure 5: KV sekection score comparison of attention-only metric v.s.
redundency-only metric v.s. R-KV with different ?. When ? ? 0.1,
the selection score starts to be dominated by attention score.

Figure 6: Performance Com-
parison of the same methods
as Figure 5.

(a) SnapKV

(b) R-KV

Figure 7: Comparison of selected key-value (KV) tokens for an example between SnapKV (left) and
R-KV (right). Grey tokens are unselected, while the gradient from light to dark red indicates the
number of attention heads selecting each token (darker = more heads). R-KV selects a more diverse
and broadly distributed set of tokens, capturing richer contextual information.

grey tokens represent unselected tokens, while the gradient from light orange to red indicates the
number of heads selecting each token, with darker red signifying selection by more heads.

When considering the tokens selected by all heads, we observe that R-KV selects a more diverse set
of tokens that cover a broader range and contain more effective information. These selections are
more evenly distributed throughout the decoded output, capturing a more comprehensive context
representation. In contrast, SnapKV�s selected tokens exhibit more limited coverage. It tends to favor
tokens positioned close to the query token, which are often selected multiple times by various heads,
indicating a concentration of attention in localized areas. Furthermore, SnapKV also selects tokens
that are not in close proximity to the query but still constitute largely redundant and unimportant
segments (i.e., �3 students are leaving early.� and �But in the initial�).

5.3 Efficiency Analysis

Memory Saving R-KV achieves improved memory efficiency by allocating fixed-size buffers for
both the retained KV cache and newly generated tokens. Unlike FullKV, which scales memory
linearly with sequence length, R-KV �s memory footprint remains constant, enabling substantial
savings during long-form generation. Detailed memory accounting is provided in Appendix C.1.

8

020406080100120Token Index1030103102101100Score Value (symlog)Score Comparison of Attention-only Metric v.s. Redundancy-only Metric v.s. R-KV with Different Redundancy-only (=0)R-KV (=0.01)R-KV (=0.1)R-KV (=0.2)R-KV (=0.4)Attention-only (=1)2004006008001000KV Cache Budget304050607080Performance (Acc.)Performance Comparisonalpha = 0alpha = 0.01alpha = 0.1alpha = 0.2alpha = 0.4alpha = 1[Question and Instruction - 65 words] � the format: Final answer: \boxed{}.[Think - 228 words] Okay, so I need to find out how many students that is. � So, there are 6 football players ... and band members. �12 students leaving early.[Reflection - 24 times and 1067 words] Wait, � Wait, the problem says, � Wait, �. "You are given a class of 30 students. � How many students are leaving early?� Wait, � Wait, � 20% + 25% + 15% = 60%. � So 12 students are leaving early. Wait, � . How many students are leaving early?� Wait, � So 12 students leave early. But ... Wait, � So, 20% of 30 is 6. So 6 students are leaving early. But wait, � So, 20% of 30 is 6. So 6 students are leaving early. But � "You are given a class of 30 students. � Wait, � Wait, � So 3 students are leaving early. But in the user's first message, � So 3 students are leaving early. But wait, � So 3 students are leaving early. But in the initial problem, � "You are given a class of 30 students. How many students are leaving early?" So 3 students are leaving early. But wait, the user wrote: �� So, 10% of 30 is 3. So 3 students are leaving early. But in the initial problem, � So 3 students are leaving early. But wait, � You are given a class of 30 students. How many students are leaving early?" So, 10% of 30 is 3. So 3 students are leaving early. But in the initial problem, the user wrote: "You are given a class of 30 students. 10% of them are leaving early. How many students are leaving early?" So, 10% of 30 is 3. So, 3 students are leaving early. But wait, the user wrote: "You are given a class of 30 students. 10% of them are leaving early. How many students are leaving early?" So, 10% of 30 is 3. So, 3 students are leaving early. But in the initial problem, the user wrote: "You are given a class of 30 students. 10% of them are leaving early. How many students are leaving early?" So, 10% of 30 is 3. Therefore, 3 students are leaving early. But wait, the user wrote: "You are given a class of 30 students. 10% of them are leaving early. How many students areleaving early?" So, 10% of 30 is 3. Therefore, 3 students are leaving early. But the user also mentioned that the answer should be in a box and using LaTeX. So, I need to write the final answer as \boxed{3}.[Conclusion- 11 words] I think that's all. The calculation is straightforward: 10% of 30 is 3.Pure attention-based metric selects redundant & unessential tokensIncorrect Answer[Question and Instruction - 65 words] You are given a math problem. Problem: In Mr. Roper's class of 30 students, 20% of the class are football players. Out of the remaining class, 25% of the students are cheerleaders or part of band. These 3 groups of students will need to leave early today to travel to an away game. How many students are leaving early? You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer. Provide the final answer in the format: Final answer: \boxed{}.[Think - 107 words] Okay, so I have this math problem here from Mr. Roper's class. Let me try to figure it out step by step. First, the problem says that there are 30 students in total in the class. Out of these, 20% are football players. Hmm, okay, so I need to find out how many students that is. To find 20% of 30, I can calculate 0.20 multiplied by 30. Let me do that: 0.20 * 30 equals 6. So, there are 6 football players in the class. Now, the problem says that out of the remaining class, 25% are cheerleaders or part of the band.[Reflextion - 292 words] Wait, so after taking out the football players, how many students are left? Let me subtract the football players from the total class. Total students: 30 Football players: 6 Remaining students: 30 - 6 = 24 So, there are 24 students left who are not football players. Of these 24, 25% are cheerleaders or part of the band. I need to find 25% of 24. Calculating 25% of 24: 0.25 * 24 = 6. So, 6 students are either cheerleaders or part of the band. The problem asks how many students are leaving early. These include the football players, cheerleaders, and band members. So, adding them up: Football players (6) + Cheerleaders/Band (6) = 12 students. Wait, let me double-check. 20% of 30 is indeed 6. Then 25% of the remaining 10 students (30 - 6 = 24? Wait, hold on, that doesn't add up. Wait, no, the remaining class after football players is 30 - 6 = 24. Then 25% of 24 is 6. So total leaving is 6 + 6 = 12. Yeah, that seems right. Alternatively, maybe I misread. It says "out of the remaining class." So after football players leave, 25% of the remaining are cheerleaders or band. So total students leaving: 20% of 30 is 6, then 25% of (30 - 6) is 25% of 24, which is 6. So total is 6 + 6 = 12. [Conclusion- 13 words] Yeah, that makes sense. I think that's correct. So the answer is 12 students leaving early.R-KV selects more diverse tokensCorrectGen. Length Method

Budget

Mem. Saving (%)

Batch

Throughput (tok/s) Tokens Gen. Dec. Time (s)

8K

16K

FullKV

R-KV

FullKV

R-KV

�
�

Fixed � 1024
Fixed � 1024
Fixed � 1536

Ratio � 10% � 819
Ratio � 34% � 2 785
Ratio � 54% � 4 423

�
�

Fixed � 1024
Fixed � 1024
Fixed � 1536

Ratio � 10% � 1 638
Ratio � 34% � 5 570
Ratio � 54% � 8 847

�
�

87.50
87.50
81.25

90.00
66.00
46.00

�
�

93.75
93.75
90.63

90.00
66.00
46.00

1
62 (max)

1
402 (max)
287 (max)

479 (max)
167 (max)
105 (max)

1
30 (max)

1
402 (max)
287 (max)

271 (max)
82 (max)
46 (max)

75.44
849.13

80.46
3 251.52
2 525.75

3 809.15
1 608.01
1 257.83

69.41
347.03

80.95
3 188.82
2 447.61

2 300.28
797.43
584.77

8 094
501 828

8 094
3 253 788
6 546 972

3 877 026
1 351 698
849 870

16 286
488 580

16 286
6 546 972
4 674 082

4 413 506
1 335 452
749 156

107.30
590.99

100.60
1 000.70
919.72

1 017.82
840.61
675.66

234.65
1 407.89

201.18
2 053.10
1 909.65

1 918.68
1 674.70
1 281.12

Table 1: Memory saving, throughput, and decoding-time comparison for Llama3-8B under various
generation length and KV cache compression budget settings.

Computation Overhead While R-KV introduces additional computation for importance and
redundancy scoring, the total overhead is modest and often outweighed by the reduced attention cost
over a compressed KV cache. This trade-off becomes increasingly favorable as sequence length
grows. Complexity comparisons can be found in Appendix C.1

Real-time analysis We present the real-time analysis of memory saving and end-to-end throughput
improvement in Table 1. When the batch size is 1, R-KV exhibits a slight throughput advantage over
FullKV. This suggests that the acceleration achieved by R-KV through reduced attention computation
outweighs computational overhead of R-KV. However, this direct speedup constitutes a minor
portion of the overall benefit. The primary throughput improvement from R-KV stems from enabling
significantly larger inference batch sizes due to KV cache compression.

We evaluate end-to-end throughput under both ratio-based and fixed KV cache budgets. R-KV
consistently enables much larger batch sizes and higher throughput than FullKV, with benefits
becoming more pronounced at longer sequence lengths. For example, at a sequence length of 16K,
R-KV achieves up to 9� larger batch sizes and over 6.6� higher throughput under a 10% compression
ratio, and 13.4� larger batch sizes with 9.2� throughput under a fixed budget of 1024. Detailed
analysis are provided in Appendix C.2.

6 Related Work

KV Cache Compression The optimization of KV cache memory efficiency in LLMs has garnered
increasing attention as model sizes and context windows expand. Existing approaches primarily fall
into three categories: dynamic token eviction[3, 11, 12], quantization[13, 14, 15], merging[16, 17, 18],
and low-rank decomposition[19, 20, 21]. Previous eviction methods like SnapKV[3], PyramidKV[5],
Ada-KV[22], HeadKV[6] dynamically prune tokens based on attention scores, but mainly focus
on evicting tokens for prefilling stage. StreamingLLM[7] and H2O[4] are proposed for decoding.
However, these general-purpose techniques often struggle with reasoning-intensive tasks, where
aggressive eviction risks disrupting critical intermediate steps in CoT, and suffers from reasoning
models� inherent redundency.

Efficient Reasoning Recent works in efficient reasoning focus on training the model to generate
less CoT without sacrificing performance. [23, 24, 25] use RL optimization with length penalty
rewards to encourage models to produce more concise chains-of-thought (CoT). [26, 27] employs
variable-length CoT datasets to supervised fine-tune (SFT) the LLM to reduce token usage while
preserving reasoning correctness. Both RL and SFT methods require additional training. [27, 28, 29]
use test-time prompting to reduce generation length, but these methods may hurt the performance. As
a KV cache compression work for reasoning models, R-KV is able to achieve lossless compression
without extensive training and prompting.

9

7 Conclusion

We introduced R-KV, a novel decoding-time KV cache compression method tailored to the challenges
of complex reasoning in large language models (LLMs). Reasoning models often generate long,
redundant outputs that impose substantial memory and computational burdens during inference.
R-KV addresses this by jointly scoring token importance and redundancy, enabling the retention of
essential reasoning content while discarding repetitive or uninformative tokens. This dynamic and
attention-guided strategy allows R-KV to preserve nearly full model performance using only 10�34%
of the original KV cache�substantially outperforming prior compression methods.

Extensive throughput and efficiency analysis demonstrate that R-KV enables up to 13� larger
batch sizes and over 9� speedup in long-sequence generation scenarios compared to FullKV, with
particularly strong gains under constrained memory budgets. With its training-free and model-
agnostic design, R-KV provides a scalable and deployment-ready solution for reasoning LLMs,
especially in streamlining the rollout phase of reinforcement learning workflows.

10

References

[1] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu,
Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan
Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang,
Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli
Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng
Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li,
Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian
Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean
Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan
Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian,
Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong
Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan
Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting
Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun,
T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu,
Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao
Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su,
Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang
Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X.
Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao
Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang
Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He,
Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong
Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha,
Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan
Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu,
Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang,
and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
learning, 2025.

[2] Mehdi Fatemi, Banafsheh Rafiee, Mingjie Tang, and Kartik Talamadupula. Concise reasoning

via reinforcement learning, 2025.

[3] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye,
T sianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for
before generation, 2024.

[4] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao
Song, Yuandong Tian, Christopher R�, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2o:
Heavy-hitter oracle for efficient generative inference of large language models, 2023.

[5] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue
Dong, Baobao Chang, Junjie Hu, et al. Pyramidkv: Dynamic kv cache compression based on
pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024.

[6] Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue Dong, and Wen Xiao. Not all heads
matter: A head-level kv cache compression method with integrated retrieval and reasoning,
2024.

[7] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming

language models with attention sinks, 2024.

[8] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn
Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset.
arXiv preprint arXiv:2103.03874, 2021.

[9] MAA. American invitational mathematics examination - aime.

In American Invitational

Mathematics Examination - AIME 2024, February 2024.

11

[10] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pond� de Oliveira Pinto, Jared
Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul
Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke
Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad
Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias
Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex
Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain,
William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant
Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie
Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and
Wojciech Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374,
2021.

[11] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells
you what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801,
2023.

[12] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios
Kyrillidis, and Anshumali Shrivastava. Scissorhands: Exploiting the persistence of impor-
tance hypothesis for llm kv cache compression at test time. Advances in Neural Information
Processing Systems, 36:52342�52364, 2023.

[13] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Sophia Shao,
Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference
with kv cache quantization. Advances in Neural Information Processing Systems, 37:1270�1303,
2024.

[14] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi
Chen, and Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. arXiv
preprint arXiv:2402.02750, 2024.

[15] Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan Zhou, Jianlong Wu, and Liqiang Nie.
Wkvquant: Quantizing weight and key/value cache for large language models gains more, 2024.

[16] Yuxin Zhang, Yuxuan Du, Gen Luo, Yunshan Zhong, Zhenyu Zhang, Shiwei Liu, and Rongrong
Ji. Cam: Cache merging for memory-efficient llms inference. In Forty-first International
Conference on Machine Learning, 2024.

[17] Jang-Hyun Kim, Junyoung Yeom, Sangdoo Yun, and Hyun Oh Song. Compressed context
memory for online language model interaction. arXiv preprint arXiv:2312.03414, 2023.

[18] Piotr Nawrot, Adrian ?a�ncucki, Marcin Chochowski, David Tarjan, and Edoardo M Ponti.
Dynamic memory compression: Retrofitting llms for accelerated inference. arXiv preprint
arXiv:2403.09636, 2024.

[19] Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong,
Yuejie Chi, and Beidi Chen. Shadowkv: Kv cache in shadows for high-throughput long-context
llm inference, 2025.

[20] Utkarsh Saxena, Gobinda Saha, Sakshi Choudhary, and Kaushik Roy. Eigen attention: Attention

in low-rank space for kv cache compression, 2024.

[21] Rongzhi Zhang, Kuang Wang, Liyuan Liu, Shuohang Wang, Hao Cheng, Chao Zhang, and
Yelong Shen. Lorc: Low-rank compression for llms kv cache with a progressive compression
strategy, 2024.

[22] Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S Kevin Zhou. Ada-kv: Optimizing
kv cache eviction by adaptive budget allocation for efficient llm inference. arXiv preprint
arXiv:2407.11550, 2024.

[23] Chen Li, Nazhou Liu, and Kai Yang. Adaptive group policy optimization: Towards stable

training and token-efficient reasoning, 2025.

12

[24] Junjie Yang, Ke Lin, and Xing Yu. Think when you need: Self-adaptive chain-of-thought

learning, 2025.

[25] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li,
Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang,
Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing
Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu
Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang
Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu,
Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan,
Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao
Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He,
Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan,
Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao
Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou,
Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang,
Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with
llms, 2025.

[26] Yingqian Cui, Pengfei He, Jingying Zeng, Hui Liu, Xianfeng Tang, Zhenwei Dai, Yan Han,
Chen Luo, Jing Huang, Zhen Li, Suhang Wang, Yue Xing, Jiliang Tang, and Qi He. Stepwise
perplexity-guided refinement for efficient chain-of-thought reasoning in large language models,
2025.

[27] Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen.

Token-budget-aware llm reasoning, 2025.

[28] Yule Liu, Jingyi Zheng, Zhen Sun, Zifan Peng, Wenhan Dong, Zeyang Sha, Shiwen Cui,
Weiqiang Wang, and Xinlei He. Thought manipulation: External thought can be efficient for
large reasoning models, 2025.

[29] Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning

models can be effective without thinking, 2025.

[30] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ah-
mad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela
Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem
Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson,
Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux,
Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret,
Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius,
Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary,
Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab
AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco
Guzm�n, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind
Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah
Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan
Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason
Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya
Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton,
Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Va-
suden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield,
Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal
Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz
Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke
de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin
Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kam-
badur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi,
Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne,
Onur �elebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal
Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao

13

Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert
Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre,
Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hos-
seini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov,
Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale,
Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane
Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha,
Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal
Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet,
Virginie Do, Vish Vogeti, V�tor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin
Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan,
Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine
Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert,
Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain,
Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay
Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit
Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu,
Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco,
Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe,
Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang,
Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock,
Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker,
Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester
Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon
Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine,
Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin
Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn,
Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers,
Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank
Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee,
Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan
Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison
Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj,
Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman,
James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff
Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin,
Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh
Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun
Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh,
Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro
Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt,
Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew
Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao
Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel
Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat,
Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White,
Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich
Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem
Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager,
Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang,
Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra,
Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ
Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh,
Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma,
Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao
Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang,
Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen
Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng,
Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez,

14

Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim
Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez,
Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu
Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Con-
stable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman,
Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin
Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary
DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3
herd of models, 2024.

[31] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li,
Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang,
Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren
Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang,
Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji
Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge,
Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren,
Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu
Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024.

[32] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and
Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head
checkpoints. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing, pages 4895�4901,
Singapore, December 2023. Association for Computational Linguistics.

A Method

A.1 Algorithm

The pseudo-code of the method is shown in Algorithm 1.

A.2

Implementation Details

Max Pooling of Attention Weights Latest open-source LLMs [30, 31] have widely adopted
Grouped-Query Attention (GQA) [32], where multiple query heads share a common pair of key-
value heads to substantially reduce memory access overhead during inference. In key-value (KV)
cache eviction strategies, it�s thus often necessary to downscale attention scores from (Q_head,
seq_len, seq_len) to (KV_head, seq_len, seq_len). While previous works such as SnapKV [3] have
predominantly employed mean pooling to aggregate attention scores across query head groups, we
hypothesize that max pooling could better preserve the most critical tokens for each query head. Our
empirical results demonstrate that max pooling leads to improved performance, and we adopt it for
all main experiments.

Calibration of SnapKV�s Observation Window Mask The official implementation of SnapKV
applied an upper triangular attention mask to the attention weights matrix to enforce causality. The
attention weights matrix is then processed with softmax, slicing, and summation to obtain observation
window scores for each prefix token.

They adopted an upper triangle to prevent tokens in the observation window from seeing tokens after
them, and then applied softmax and summation. However, their implementation does not account
for the fact that tokens within the observation window absorb part of the attention weight originally
assigned to prefix tokens, thereby disrupting the normalization property.

In our implementation, we address this issue by first slicing the attention weights before applying
softmax. This approach ensures proper normalization and leads to significantly better scores in our
tests.

15

return (Kfull, Vfull)

if Lfull ? Lbudget < Bbuffer then

Algorithm 1 R-KV: Qobs are query states for ? observation tokens, Kfull, Vfull are the full KV cache
states of length Lfull.
1: procedure R-KV((Kfull, Vfull), Lfull, Lbudget, Qobs, ?, Bbudget, Bbuffer, T, ?, ?, ?, H, dk)
2:
3:
4:
5:
6:
7:
8:
9:
10:
11:
12:

end if
(Kobs, Vobs) ? last ? tokens of (Kfull, Vfull)
(Kcand, Vcand) ? first (Lfull ? ?) tokens of (Kfull, Vfull)
Nc ? Lfull ? ?
if Nc ? Bbudget then

end if
for each head h = 0 . . . H ? 1 do

Compute attention matrix Ah ? R?�Nc using Qh

? Not enough candidates to prune beyond budget

? Check if compression is triggered

? Number of candidate tokens

return (Kfull, Vfull)

cand ? Handles MHA/GQA as

obs and Kh

per Eqs. (1)-(3) from text

? For each candidate token k
? q: observation token, k: candidate token

for k = 0 . . . Nc ? 1 do

(cid:80)??1

q=0 (Ah)qk

k,h ? 1
I ?
?
end for
{Ik,h}Nc?1

k=0 ? 1D-Pooling({I ?

k,h}Nc?1
k=0 )

end for
for each head h = 0 . . . H ? 1 do

Kh
Sh ? Kh

norm(Kh

norm)?

Sh ? RNc�Nc

norm ? RNc�dk ; For k = 0 . . . Nc ? 1, Kh

norm,k ? Kh

cand,k/(?Kh

cand,k?2 + ?)

? Cosine Similarity Matrix Computation, similarity matrix

? Prevent Self-Redundancy

uv ? ((Sh)uv > T ?1 : 0) for u, v ? {0, . . . , Nc ? 1} ? Identify Highly Similar Pairs
? Enforce Retention of Recent Tokens

uv = 1, v ? {0, . . . , Nc ? 1}}

u with up to ? largest indices v.

13:
14:
15:
16:
17:
18:
19:

20:

21:
22:
23:
24:
25:
26:
27:

28:

? Sh is now modified

(cid:80)Nc?1

v=0 (Sh)uv

for k = 0 . . . Nc ? 1 do

(Sh)kk ? 0

end for
Bh
for u = 0 . . . Nc ? 1 do
u ? {v | Bh
T h
u,? ? subset of T h
T h
for v? ? T h

u,? do
(Sh)u,v? ? 0

end for

end for
Let �Sh ? RNc where (�Sh)u ? 1
Nc
for u = 0 . . . Nc ? 1 do

Ru,h ? (softmax(�Sh))u

end for
for each head h = 0 . . . H ? 1 do
for k = 0 . . . Nc ? 1 do

Scorek,h ? ?Ik,h ? (1 ? ?)Rk,h

end for
Let AggScore ? RNc
for k = 0 . . . Nc ? 1 do

end for

29:
30:
31:
32:
33:
34:
35:
36:
37:
38:
39:
40:
41:
42:
43:
44:
45:
46:
47:
48:
49:
50:
51:
52: end procedure

end for

16

AggScorek ? meanh(Scorek,h)

? Aggregate scores across heads

end for
Idxsel ? indices of top-Bbudget tokens from {0, . . . , Nc ? 1} based on AggScore
Kcand_sel ? Kcand[Idxsel]; Vcand_sel ? Vcand[Idxsel]
Kcomp ? concatenate(Kcand_sel, Kobs)
Vcomp ? concatenate(Vcand_sel, Vobs)
Lprev_comp ? Bbudget + ?
return (Kcomp, Vcomp)

? Update length for next cycle

? Order might vary

Model

Benchmark Method

128

256

512

768

1 024

1 536

2 048

2 560

3 072

4 096

MATH

AIME24

MATH

Llama3-8B

Qwen-14B

82.38
FullKV
R-KV
51.08
SnapKV 32.53

FullKV
R-KV
SnapKV

49.79
0.42
0.16

94.58
FullKV
R-KV
56.21
SnapKV 26.32

82.38
67.39
50.07

49.79
10.21
0.94

94.58
73.33
43.93

82.38
76.92
64.03

49.79
29.48
4.53

94.58
84.77
77.93

82.38
80.21
70.81

49.79
40.31
11.20

94.58
88.79
82.52

82.38
81.34
74.43

49.79
45.26
15.73

94.58
90.72
86.63

82.38
82.34
78.43

49.79
51.56
26.04

94.58
92.72
90.86

82.38
82.65
80.50

49.79
52.29
32.76

94.58
93.62
92.73

�
�
�

�
�
�

49.79
53.85
39.43

49.79
53.13
41.93

�
�
�

�
�
�

�
�
�

�
�
�

�
�
�

AIME24

FullKV
R-KV
SnapKV

65.68
65.68
24.53
67.45
12.86
54.32
Table 2: Accuracy (%) of Llama3-8B and Qwen-14B on the MATH and AIME24 benchmarks
under different memory-optimization methods across context lengths. ��� denotes configurations that
were not evaluated.

65.68
7.92
2.86

65.68
42.66
25.00

65.68
56.09
46.56

65.68
0.57
0.26

65.68
36.25
16.30

65.68
55.00
36.41

65.68
64.32
52.86

�
�
�

B Experiment

B.1 Devices

We use NVIDIA A100 80G to finish all the experiments.

B.2 Main Results

See Table 2.

C Efficiency

C.1 Complexity Analysis of Memory and Computation

Memory Saving As discussed in �3.1, we need to allocate memory for the KV cache budget
Mbudget ? Rb�Bbudget�Nlayer�Nhead�d to retain Bbudget KV cache tokens, and for the buffer Mbuffer ?
Rb�Bbuffer�Nlayer�Nhead�d to store Bbuffer newly generated KV cache tokens during the generation of
a text segment. Here, b is the batch size, Nlayer is the number of Transformer layers, Nhead is the
number of attention heads, and d is the dimension of attention heads. In addition, we also need
to allocate memory for the model weight M?. During decoding, the previous query states are
typically discarded by default, so we use a query cache to store the last ? tokens in the query
state, consuming memory of M? ? Rb�?�Nlayer�Nhead�d. In summary, R-KV requires memory of
Mtotal = M? + Mbudget + Mbuffer + M? during generation. In comparison to FullKV without KV
cache compression, generating Bfull tokens requires memory of Mfull ? Rb�Bfull�Nlayer�Nhead�Dhead to
retain Bfull KV tokens, and memory of the model weight M0. Therefore, the memory saved by our
method w.r.t. FullKV is: Msaving = Mfull ? Mbudget ? Mbuffer ? M?.

Computation Overhead The computational complexity of importance scoring (See �3.2) is
O(?Bbudget) while redundancy estimation (see �3.3) has complexity O(B2
budget). Thus, the total
overhead incurred during each generation segment is O(?Bbudget + B2
budget). The generation com-
plexity without KV cache compression is O(BfullBbuffer), whereas the complexity with KV cache
compression is O((Bbudget + Bbuffer)Bbuffer). For reasoning models, Bfull tends to be large because
of the long generation length, and using a relatively small Bbudget value can efficiently reduce com-
putation cost. The effectiveness of this approach depends on depends on whether the speedup
gained by attending over a reduced KV cache outweighs the overhead of computing the compression
scores�i.e., the combined cost of importance and redundancy scores, (O(?Bbudget) + O(B2
budget)).

C.2 Detailed Analysis of Throughput Results

We analyze the end-to-end throughput from two perspectives: ratio budget and fixed budget.

17

Gen. Length Method

Budget

Mem. Saving (%)

Batch

Throughput (tok/s) Tokens Gen. Dec. Time (s)

8K

16K

FullKV

�
�

SnapKV

R-KV

FullKV

SnapKV

R-KV

Fixed � 1024
Fixed � 1024
Fixed � 1536
Fixed � 3072
Ratio � 10% � 819
Ratio � 34% � 2 785
Ratio � 54% � 4 423

Fixed � 1024
Fixed � 1024
Fixed � 1536
Fixed � 3072
Ratio � 10% � 819
Ratio � 34% � 2 785
Ratio � 54% � 4 423

�
�

Fixed � 1024
Fixed � 1024
Fixed � 1536
Fixed � 3072
Ratio � 10% � 1 638
Ratio � 34% � 5 570
Ratio � 54% � 8 847

Fixed � 1024
Fixed � 1024
Fixed � 1536
Fixed � 3072
Ratio � 10% � 1 638
Ratio � 34% � 5 570
Ratio � 54% � 8 847

�
�

87.50
87.50
81.25
62.50
90.00
66.00
46.00

87.50
87.50
81.25
62.50
90.00
66.00
46.00

�
�

87.50
87.50
81.25
81.25
90.00
66.00
46.00

93.75
93.75
90.63
81.25
90.00
66.00
46.00

1
62 (max)

1
402 (max)
287 (max)
150 (max)
479 (max)
167 (max)
105 (max)

1
402 (max)
287 (max)
150 (max)
479 (max)
167 (max)
105 (max)

1
30 (max)

1
402 (max)
287 (max)
150 (max)
271 (max)
82 (max)
46 (max)

1
402 (max)
287 (max)
150 (max)
271 (max)
82 (max)
46 (max)

75.44
849.13

81.26
3 253.93
2 525.25
1 527.67
3 808.81
1 625.46
1 269.68

80.46
3 251.52
2 525.75
1 520.99
3 809.15
1 608.01
1 257.83

69.41
347.03

81.03
3 202.17
2 449.02
1 413.84
2 306.26
798.42
586.43

80.95
3 188.82
2 447.61
1 406.28
2 300.28
797.43
584.77

8 094
501 828

8 094
3 253 788
2 322 978
1 214 100
3 877 026
1 351 698
849 870

8 094
3 253 788
6 546 972
1 214 100
3 877 026
1 351 698
849 870

16 286
488 580

16 286
6 546 972
4 674 082
2 442 900
4 413 506
1 335 452
749 156

16 286
6 546 972
4 674 082
2 442 900
4 413 506
1 335 452
749 156

107.30
590.99

99.60
999.96
919.90
794.74
1 017.91
831.58
669.36

100.60
1 000.70
919.72
798.23
1 017.82
840.61
675.66

234.65
1 407.89

200.99
2 044.54
1 908.56
1 727.84
1 913.71
1 672.61
1 277.48

201.18
2 053.10
1 909.65
1 737.13
1 918.68
1 674.70
1 281.12

Table 3: Memory-saving, throughput, and decoding-time comparison for LLAMA3-8B under various
generation lengths and KV-cache compression budgets.

Ratio Budget: section 4.2 indicates that for DeepSeek-R1-Distill-Llama-8B, lossless compression
(i.e., model performance equivalent to no KV compression) is achievable when the KV budget ratio,
relative to the output length, is between 10% and 34%. For DeepSeek-R1-Distill-Qwen-14B, this
range for lossless compression is 25% to 54% of the output length. Consequentlywe investigated the
maximum achievable batch size and corresponding throughput for R-KV at compression ratios of
10%, 34%, and 54%, comparing these against the maximum batch size and throughput of FullKV
using DeepSeek-R1-Distill-Llama-8B. In 8K sequence length setting, at a 54% compression ratio,
R-KV allows for a batch size 1.7 � larger than FullKV, resulting in 1.5 � the throughput. At a 10%
compression ratio, R-KV achieves a 7.7 � increase in batch size and a 4.5 � increase in throughput
compared to FullKV. For a 16K sequence length setting, at 54% compression, the batch size is 1.5 �
that of FullKV, and the throughput is 1.7 � higher. At 10% compression, R-KV supports a 9 � larger
batch size, delivering 6.6 � the throughput. We observe that for smaller batch sizes (e.g., less than
128), throughput scales nearly linearly with increasing batch size. However, for larger batch sizes this
linear scaling diminishes as inference on the NVIDIA A100 GPU becomes compute-bound.

Fixed Budget: We also conducted an analysis under a fixed KV cache budget. With an output length
of 8K and a fixed budget Bbudget = 1024, R-KV enables a batch size 6.48 � larger than FullKV,
yielding 3.8 � the throughput. At Bbudget = 1536, the batch size is 4.6 � larger, and throughput is 3
� that of FullKV. For an output length of 16K and Bbudget = 1024, R-KV achieves a 13.4 � increase
in batch size and a 9.19 � increase in throughput. With Bbudget = 1536, the batch size is 9.6 � larger,
and throughput is 7.1 � higher. In the fixed budget scenario, the advantage of R-KV becomes more
pronounced with longer generation lengths. This is because the KV cache size for R-KV under a
fixed budget does not increase with the sequence length, unlike FullKV where the memory footprint
grows linearly with the generation length, thus more severely limiting its maximum batch size.

C.3 Results

Full results could be found at Table 3. While R-KV incurs a minor computational overhead for
redundancy estimation compared with SnapKV, this results in a throughput that is only slightly lower,
with a negligible difference of less than 1%.

18

D Limitations

One limitation of our proposed KV cache compression method is its current compatibility with certain
advanced attention mechanisms, such as paged attention. Adapting our compression technique to
seamlessly integrate with such mechanisms presents a non-trivial challenge and may require further
investigation. Additionally, the implementation of KV cache compression within existing serving
frameworks can encounter practical difficulties, particularly if these frameworks lack native support
or flexible interfaces for KV cache compression. In serving frameworks that do not offer specialized
KV cache compression interfaces, the performance benefits of our method might be less pronounced.
Without such interfaces, implementing KV cache compression may necessitate reallocating memory
to store the compressed KV cache and subsequently deallocating the memory used for the original,
uncompressed cache. This process of memory reallocation can introduce significant overhead,
potentially offsetting some of the acceleration gains. In contrast, serving frameworks equipped with
dedicated KV compression interfaces can handle these operations much more efficiently, avoiding
such costly memory management tasks.

19

NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research,
addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
the checklist: The papers not including the checklist will be desk rejected. The checklist should
follow the references and follow the (optional) supplemental material. The checklist does NOT count
towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For
each question in the checklist:

� You should answer [Yes] , [No] , or [NA] .
� [NA] means either that the question is Not Applicable for that particular paper or the

relevant information is Not Available.

� Please provide a short (1�2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the
reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
(after eventual revisions) with the final version of your paper, and its final version will be published
with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
proper justification is given (e.g., "error bars are not reported because it would be too computationally
expensive" or "we were unable to find the license for the dataset we used"). In general, answering
"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
acknowledge that the true answer is often more nuanced, so please just use your best judgment and
write a justification to elaborate. All supporting evidence can appear either in the main paper or the
supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

� Delete this instruction block, but keep the section heading �NeurIPS Paper Checklist",
� Keep the checklist subsection headings, questions/answers and guidelines below.
� Do not modify the questions and only use the provided macros for your answers.

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper�s contributions and scope?
Answer: [Yes]
Justification: abstract and introduction
Guidelines:

� The answer NA means that the abstract and introduction do not include the claims

made in the paper.

� The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.

� The claims made should match theoretical and experimental results, and reflect how

much the results can be expected to generalize to other settings.

� It is fine to include aspirational goals as motivation as long as it is clear that these goals

are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: Limitation

20

Guidelines:

� The answer NA means that the paper has no limitation while the answer No means that

the paper has limitations, but those are not discussed in the paper.

� The authors are encouraged to create a separate "Limitations" section in their paper.
� The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.

� The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.

� The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.

� The authors should discuss the computational efficiency of the proposed algorithms

and how they scale with dataset size.

� If applicable, the authors should discuss possible limitations of their approach to

address problems of privacy and fairness.

� While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren�t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

Answer: [Yes]

Justification: 5.2 Failure of Attention-Based Methods to Capture Redundancy

Guidelines:

� The answer NA means that the paper does not include theoretical results.
� All the theorems, formulas, and proofs in the paper should be numbered and cross-

referenced.

� All assumptions should be clearly stated or referenced in the statement of any theorems.
� The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.

� Inversely, any informal proof provided in the core of the paper should be complemented

by formal proofs provided in appendix or supplemental material.

� Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification:

Guidelines:

� The answer NA means that the paper does not include experiments.

21

� If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.

� If the contribution is a dataset and/or model, the authors should describe the steps taken

to make their results reproducible or verifiable.

� Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.

� While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how

to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe

the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: code open-source

Guidelines:

� The answer NA means that paper does not include experiments requiring code.
� Please see the NeurIPS code and data submission guidelines (https://nips.cc/

public/guides/CodeSubmissionPolicy) for more details.

� While we encourage the release of code and data, we understand that this might not be
possible, so �No� is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).

� The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

� The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
� The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.

� At submission time, to preserve anonymity, the authors should release anonymized

versions (if applicable).

22

� Providing as much information as possible in supplemental material (appended to the

paper) is recommended, but including URLs to data and code is permitted.

6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

Justification: MATH-500 and AIME24

Guidelines:

� The answer NA means that the paper does not include experiments.
� The experimental setting should be presented in the core of the paper to a level of detail

that is necessary to appreciate the results and make sense of them.

� The full details can be provided either with the code, in appendix, or as supplemental

material.

7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Run experiments for 64 times and calculate averaged Pass@1.

Guidelines:

� The answer NA means that the paper does not include experiments.
� The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

� The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

� The method for calculating the error bars should be explained (closed form formula,

call to a library function, bootstrap, etc.)

� The assumptions made should be given (e.g., Normally distributed errors).
� It should be clear whether the error bar is the standard deviation or the standard error

of the mean.

� It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.

� For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).

� If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.

8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [Yes]

Justification: A100

Guidelines:

� The answer NA means that the paper does not include experiments.
� The paper should indicate the type of compute workers CPU or GPU, internal cluster,

or cloud provider, including relevant memory and storage.

23

� The paper should provide the amount of compute required for each of the individual

experimental runs as well as estimate the total compute.

� The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn�t make it into the paper).

9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification:
Guidelines:

� The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
� If the authors answer No, they should explain the special circumstances that require a

deviation from the Code of Ethics.

� The authors should make sure to preserve anonymity (e.g., if there is a special consid-

eration due to laws or regulations in their jurisdiction).

10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: Conclusion: As a training-free and model-agnostic solution, R-KV offers a
practical path to deploy advanced reasoning LLMs more efficiently and scalably.
Guidelines:

� The answer NA means that there is no societal impact of the work performed.
� If the authors answer NA or No, they should explain why their work has no societal

impact or why the paper does not address societal impact.

� Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

� The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.

� The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.

� If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification:
Guidelines:

24

� The answer NA means that the paper poses no such risks.
� Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.

� Datasets that have been scraped from the Internet could pose safety risks. The authors

should describe how they avoided releasing unsafe images.

� We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?

Answer: [Yes]

Justification:

Guidelines:

� The answer NA means that the paper does not use existing assets.
� The authors should cite the original paper that produced the code package or dataset.
� The authors should state which version of the asset is used and, if possible, include a

URL.

� The name of the license (e.g., CC-BY 4.0) should be included for each asset.
� For scraped data from a particular source (e.g., website), the copyright and terms of

service of that source should be provided.

� If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.

� For existing datasets that are re-packaged, both the original license and the license of

the derived asset (if it has changed) should be provided.

� If this information is not available online, the authors are encouraged to reach out to

the asset�s creators.

13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [Yes]

Justification:

Guidelines:

� The answer NA means that the paper does not release new assets.
� Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.

� The paper should discuss whether and how consent was obtained from people whose

asset is used.

� At submission time, remember to anonymize your assets (if applicable). You can either

create an anonymized URL or include an anonymized zip file.

14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?

Answer: [NA]

25

Justification:
Guidelines:

� The answer NA means that the paper does not involve crowdsourcing nor research with

human subjects.

� Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.

� According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.

15. Institutional review board (IRB) approvals or equivalent for research with human

subjects

Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification:
Guidelines:

� The answer NA means that the paper does not involve crowdsourcing nor research with

human subjects.

� Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

� We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.

� For initial submissions, do not include any information that would break anonymity (if

applicable), such as the institution conducting the review.

16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and does not impact the core methodology,
scientific rigorousness, or originality of the research, declaration is not required.
Answer: [NA]
Justification:
Guidelines:

� The answer NA means that the core method development in this research does not

involve LLMs as any important, original, or non-standard components.

� Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)

for what should or should not be described.

26


