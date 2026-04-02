5
2
0
2

y
a
M
3
2

]

G
L
.
s
c
[

3
v
6
6
0
5
0
.
3
0
5
2
:
v
i
X
r
a

Capacity-Aware Inference: Mitigating the Straggler
Effect in Mixture of Experts

Shwai He1 Weilin Cai2

Jiayi Huang2 Ang Li1

1University of Maryland, College Park
2The Hong Kong University of Science and Technology (Guangzhou)
shwaihe@umd.edu, angliece@umd.edu

Abstract

The Mixture of Experts (MoE) is an effective architecture for scaling large language
models by leveraging sparse expert activation to balance performance and efficiency.
However, under expert parallelism, MoE suffers from inference inefficiencies due
to imbalanced token-to-expert assignment, where underloaded experts complete
computations early but must wait for overloaded experts, leading to global delays.
We define this phenomenon as the Straggler Effect, as the most burdened experts
dictate the overall inference latency. To address this, we first propose Capacity-
Aware Token Drop, which enforces expert capacity limits by discarding excess
tokens from overloaded experts, effectively reducing load imbalance with minimal
performance impact (e.g., 30% speedup with only 0.9% degradation on OLMoE).
Next, given the presence of low-load experts remaining well below the capacity
threshold, we introduce Capacity-Aware Expanded Drop, which allows tokens
to include additional local experts in their candidate set before enforcing strict
local capacity constraints, thereby improving load balance and enhancing the
utilization of underused experts. Extensive experiments on both language and
multimodal MoE models demonstrate the effectiveness of our approach, yielding
substantial gains in expert utilization, model performance, and inference efficiency,
e.g., applying Expanded Drop to Mixtral-8�7B-Instruct yields a 0.2% average
performance improvement and a 1.85� inference speedup.

1

Introduction

In recent years, the rapid evolution of Large Lan-
guage Models (LLMs) [32, 38, 11] has driven a wave
of innovations, continuously expanding the frontiers
of AI research and applications. Among the model ar-
chitectural innovations, the Mixture of Experts (MoE)
framework has emerged as a pivotal technique for
optimizing the cost-performance trade-off in LLMs.
Specifically, MoE [35, 14] enhances scalability by
integrating multiple experts while activating only a
subset per input. This selective activation substan-
tially improves model performance without a corre-
sponding increase in computational cost, effectively
balancing efficiency and performance.

Figure 1: Illustration of the Straggler Effect
in MoE Inference, where the most burdened
experts dictate the overall latency.

Despite the success of MoE, a key efficiency challenge lies in the imbalanced token-to-expert
distribution, which results in some experts being overloaded while others remain underutilized
[26, 46]. In distributed GPU settings, experts are typically sharded across multiple devices, with each
GPU responsible for a subset of the experts. Under expert parallelism, low-load experts complete

102030405060Expert ID01234567Normalized LoadHigh-loadLow-load

their computations earlier but must wait for overloaded experts to finish, as synchronization barriers
are required before proceeding to the next stage. This expert-level straggler effect further propagates
to device-level delays, where GPUs hosting lighter expert workloads are stalled by GPUs hosting
heavier workloads, leading to inefficient resource utilization and increased end-to-end latency during
inference. As illustrated in Figure 1, this phenomenon is referred to as the Straggler Effect, where
the heavily loaded experts determine the overall latency of imbalanced MoE inference.

While auxiliary balance losses have been incorporated into the training process to alleviate imbalance
[35, 14, 11], these techniques remain ineffective in mitigating imbalance during inference. Specifi-
cally, as shown in Figure 1, our findings reveal a highly uneven token distribution among experts,
with the highest-load expert handling more than seven times the expected average load. Moreover,
managing such imbalance during inference often incurs additional resource overhead. For example,
DeepSeek-V3 mitigates this issue by duplicating high-load experts and deploying them redundantly
across devices [11]. This motivates us to explore efficient token-to-expert assignment by addressing
the key question: How can we prevent extreme overloading of heavily utilized experts?

We propose Capacity-Aware Inference to address this challenge. Specifically, for high-load experts,
we introduce Capacity-Aware Token Drop, which imposes a maximum capacity constraint and
discards excess tokens from overloaded experts. This approach alleviates severe load imbalance and
significantly improves efficiency, while maintaining model performance since the dropped tokens
represent only a small fraction of the total workload, e.g., OLMoE achieves a 30% speedup in MoE
layers with just a 0.9% performance degradation. After removing excess tokens from overloaded
experts, we observe that some low-load experts remain significantly underutilized relative to the
predefined capacity constraints, yet must still wait for other experts to complete their computations.
This leads us to a second key question: How can we effectively leverage the available capacity of
underutilized experts?

For low-load experts, we extend Token Drop with Capacity-Aware Expanded Drop, which further
utilizes the available capacity of underutilized experts to handle overflow tokens from high-load
experts. Specifically, under expert parallelism across multiple GPUs, Expanded Drop allows each
token to consider additional candidate experts on the same device while still enforcing strict local
capacity constraints. This expanded selection improves the utilization of low-load experts and
enhances the representational capacity of MoE models under capacity-constrained scenarios.

Extensive experimental results validate the effectiveness of our proposed techniques, demonstrating
significant improvements in both efficiency and performance, e.g., applying Expanded Drop to
Mixtral-8�7B-Instruct yields a 0.2% average performance improvement and a 1.85� inference
speedup. Moreover, in multimodal models, we identify redundancy among image tokens and show
that applying aggressive capacity constraints (e.g., setting the maximum to half of the average expert
load) can still maintain performance. In short, our contributions are in four folds:

� We identify the Straggler Effect caused by token imbalance at inference time in Mixture of

Experts, highlighting the optimization potential for reducing latency.

� Toward the high-load experts, we propose Capacity-Aware Token Drop, which enforces
capacity constraints by discarding excess tokens assigned to overloaded experts, thereby
mitigating extreme load imbalance.

� To better utilize underloaded experts, we introduce Capacity-Aware Expanded Drop,
which expands the candidate expert set to include additional local experts, further improving
load balance and model performance.

� Extensive experiments on both language and multimodal models validate the effectiveness
of our approach, demonstrating substantial improvements in inference efficiency with
comparable performance.

2 Related Works

Mixture of Experts Models The Mixture of Experts (MoE) is a kind of neural network architecture
with an extended set of parameters (referred to as �experts�) controlled by a router, which is first
introduced in the context of conditional computation [21, 23]. The potential of sparse activation
in MoE is subsequently exploited by [34] for efficient training and inference on pretrained models
with special designs, opening the door for MoE in various vision [33] and language [25, 13, 14]
scenarios. Attributed to its exceptional efficiency, MoE has been adopted as a foundational framework

2

in the designs of large language models [22, 9, 41, 45, 39], achieving superior scaling laws at low
computational costs. Despite these advancements, MoE still faces efficiency challenges in both
training and inference [3], and our work specifically focuses on enhancing inference-time efficiency.

Imbalance in Mixture of Experts The imbalance in token-to-expert assignments [44, 5] poses a
significant challenge to the deployment of MoE. This imbalance leads to inefficiencies in computation,
communication, and memory [18, 37, 42], making it a critical bottleneck for MoE scalability and
deployment. To mitigate this issue, an auxiliary balance loss [35] is incorporated into the training
process to encourage more uniform token distribution across experts. Additionally, various training
strategies have been introduced to further balance token assignments: Switch-Transformer [14] and
DeepSeek-V2 [10] implement Token Drop to alleviate expert overload, while DeepSeek-V3 [11]
introduces an additional sequence-level auxiliary loss to prevent severe token imbalance.

However, these techniques primarily focus on training and fail to ensure balanced token assignments
during inference. Instead, addressing token imbalance at inference often incurs additional resource
costs. For example, DeepSeek-V3 [11] mitigates this issue by duplicating high-load experts and
deploying them redundantly. In contrast, our approach effectively balances token assignments without
introducing additional computational overhead.

3 Background and Motivation

3.1 Extremely Imbalanced Expert Utilization

A Mixture of Experts (MoE) layer consists of a collection of n experts, {E1, E2, . . . , En} and a
router G that dynamically selects the most relevant experts for a given input x. The router computes
selection scores G(x), for all experts and selects the top k experts, resulting in a sparse activation:

K = TopK(Softmax(G(x)), k).

(1)

The input x is processed by the selected experts, and their outputs are combined into a weighted sum
based on the router�s scores. This process is mathematically expressed as:

y =

(cid:88)

i?K

G(x)i � Ei(x),

(2)

where K denotes the indices of selected experts, G(x)i represents the selection score for the i-th
expert, and Ei(x) is the output from the i-th expert. In transformer models, the MoE layer usually
replaces the feed-forward network (FFN) and only activates a subset of experts for each input.

While experts in MoE models are often deployed in parallel across distributed GPUs, imbalanced
token-to-expert assignments lead to varying levels of expert utilization and potential latency. Despite
the incorporation of balancing techniques during training, the load imbalance persists during inference.
To further investigate this issue, we conduct preliminary experiments to analyze expert-specific
utilization patterns and assess the impact of imbalance on practical latency.

To quantify expert utilization, we measure the
load across different experts. Given an input
batch x ? Rb�s�d with batch size b and se-
quence length s, the total number of tokens is
t = bs. Since each token selects k out of n
experts, the expected token count per expert is:

�N =

tk
n

.

(3)

However, due to imbalanced token assignments,
some experts may receive more or fewer tokens
than the expected value.

Figure 2: Expert-wise load, where each load value
is divided by �N for clarity. To ensure generality,
we visualize loads across different datasets.

Figure 2 illustrates the normalized peak tokens load for each expert to accommodate all tokens
within a single layer of OLMoE, where some experts receive an excessively large number of tokens
(e.g., more than seven times the average number of tokens), leading to severe load imbalance and,
consequently, significant latency. A detailed layer-by-layer analysis is provided in Appendix E.

3

0102030405060Expert ID012345678Normalized LoadRTEWinoGrandeOBQABoolQPIQA3.2 Motivation � the Straggler Effect

Under the expert parallelism scenario, where the number of assigned tokens dictates the processing
time of each expert, high-load experts become the bottleneck for overall latency within an MoE layer.
Specifically, low-load experts remain idle while waiting for high-load experts to complete, leading to
synchronization delays. Therefore, the latency of an MoE layer is given by:

L ? max({Ni}n

i=1),

(4)

where Ni represents the number of tokens assigned to the i-th expert, with the total token allocation
satisfying (cid:80)n
i=1 Ni = tk. According to Eq. 4, the latency follows the Straggler Effect: the most
burdened experts dictate the overall latency of the MoE layer. In the worst case, all tokens are
assigned to the same group of experts, underutilizing the parallel processing capability of MoE.
Conversely, distributing tokens evenly across experts maximizes computational efficiency and fully
leverages the parallelism of multiple experts. With the bounds of the ideal and worst cases, the range
of the highest load is given by:

max({Ni}n

i=1) ? [ �N ,

].

(5)

n �N
k

However, existing MoE models often adopt a dropless strategy during inference, which fails to
address token imbalance and can lead to significantly increased latency.

Given that the imbalance stems from excessively high- and low-load experts, we address this issue
by exploring the following questions: (1) For high-load experts, are there redundant tokens that
can be dropped without causing significant performance degradation? (2) For low-load experts that
must wait for high-load experts to complete forward passes, is there an opportunity to enhance their
utilization and improve performance without incurring substantial additional cost?

4 Methodology

Token Drop Regulates the Latency of High-Load Experts To address the question about over-
loaded experts, we first regulate their maximum utilization. Specifically, we introduce expert capacity
to control token allocation. Given a capacity factor ?, the maximum number of tokens assigned to
each expert (i.e., expert capacity) is defined as:

C = ? �N .

(6)

A higher ? allows more tokens to be retained, but experts handling excessive tokens may introduce
latency. Conversely, a lower ? enforces stricter capacity limits, reducing latency by discarding more
tokens, but at the risk of performance degradation. With the involvement of expert capacity ?, we
constrain the upper bound of latency as follows:

max({Ni}n

i=1) =

(cid:26)? �N

? < 1
within [ �N , ? �N ] ? ? 1

,

(7)

where ? is typically much smaller than n
k . This constraint ensures that no expert exceeds the specified
capacity limit, effectively mitigating severe load imbalances and reducing latency. Note that tokens
are distributed across devices under expert and data parallelism. To avoid additional communication
overhead, we apply capacity constraints to tokens within each local device, similar to the constraints
used during training [14]. This ensures that all devices respect the limits, maintaining strict control
over token flow to the experts.

Specifically, when a capacity constraint is imposed on each expert, experts must evaluate the volume
of assigned tokens before execution. For experts with a load below the predefined capacity, there is
no difference between capacity-constrained inference and traditional inference. However, when the
load exceeds the capacity, experts must discard excess tokens to adhere to the constraint. To address
this, we introduce a scoring function S to evaluate each token:

S(x) =

?

?
?
?

s11
s21
...
st1

. . .
. . .
...
. . .

?

?
?
?

s1n
s2n
...
stn

,

s12
s22
...
st2

4

(8)

where sij denotes the importance score of the mapping from the i-th token to the j-th expert. With
this score, each overflowed expert selectively discards those with lower scores. Let J be the set of
overflowed experts:

?J = KthValue(SJ , C),
(9)
where ?J represents the thresholds, i.e., C-th highest value in SJ , serving as a threshold to filter out
excess tokens:

TJ ? {(t, j) | t ? [1, . . . , N ], j ? J , S[t, j] ? ?J [j]}
SJ ? SJ ? MJ , where MJ ? 1 [SJ ? ?J ] ,

(10)

(11)

where TJ denotes the token indices retained by the experts indexed in J . The scores of rejected
tokens are masked to prevent them from being routed to their corresponding overflowed experts.

Regarding the specific scoring function, we explore multiple metrics and summarize them as follows:

Order: Discarding later tokens once earlier tokens have filled the expert capacity. This strategy was
first introduced in Switch-Transformer [14] during training, and we extend it to the inference phase.

Reverse Order: Instead of discarding later tokens, this approach removes earlier tokens to comply
with the expert capacity constraint.

Random: Dropping Excess tokens randomly to meet the predefined expert capacity constraints.

Score: Using the gating score G(x) as an importance indicator and discarding tokens.

Among these metrics, �Order� and �Reverse Order� are unstable, as shuffling sequences within a
batch may result in different tokens being dropped [17]. �Random� assumes all tokens have an equal
probability of being dropped. In contrast, �Score� is stable, unaffected by sequence order within a
batch. Notably, there is virtually no additional computational overhead associated with calculating
these metrics, and the dropping operation incurs minimal cost compared to the intensive computations
performed by the experts.

(a) Token Drop

(b) Expanded Drop

Figure 3: Illustration of Capacity-Aware Token Drop (a) and Expanded Drop (b). Both methods
first select experts based on gating scores. In Token Drop, tokens exceeding the local device capacity
are discarded prior to All-to-All communication. Expanded Drop enhances expert utilization by
allowing each token to consider additional m candidate experts on the same device while still
enforcing strict local capacity constraints.

Expanded Drop Enhances the Utilization of Low-load Experts Token Drop exclusively targets
overloaded experts by discarding overflowed tokens that exceed expert capacity but does not address
the underutilization of low-load experts. Next, we introduce Expanded Drop to ensure a more
balanced token-to-expert allocation.

A naive approach to rerouting under-selected tokens is to mask the mapping scores of overflowed
experts and then reselect experts for these tokens. However, the reselection may still result in
overflows, necessitating multiple rounds of selection and dropping, which increases latency. Moreover,
the repeated selection and dropping substantially raise the cost of token-to-expert mapping.

Expanded Drop adopts a simple yet effective strategy: for each token, it selects additional candidate
experts. Given m experts deployed on a single GPU, a token not only selects the top-k experts based
on gating scores, but also includes m local experts (e.g., 8 experts per device under 8-way expert
parallelism across 8 GPUs for a total of 64 experts) for substitution if the initially selected experts
are overflowed. As a result, each token may select up to m + k experts. The final selection is then
refined as experts drop tokens as needed to satisfy capacity constraints. This makes no change in the
token assignments in experts that are overflowed by the top-k experts. Meanwhile, for under-utilized
experts, the expanded top-k + m candidate pool increases the likelihood of receiving additional

5

Device 0Device 1Expert 1T2T5T6Expert 0T0T3T4T7Expert 3T0T3T5T7Expert 2T1T5T6T7T4GateT1T2T3T0GateToken Drop?=?.?????????=?AlltoAllDevice 0Device 1Token ExpandToken Drop?=?.?????????=?T5T6T7T4GateT1T2T3T0GateExpandExpert 0T0T3T4T7Expert 3T0T3T5T7Expert 2T1T5T6Expert 1T2T5T6T1AlltoAll?1?3?0?2tokens. After top-k + m selection and dropping, there might exists tokens that select more than k
experts. Through empirical analysis (Appendix D), we choose not to enforce a constraint that limits
each token to selecting at most k experts, thereby removing the need to explicitly retain the top-k
experts at the end.

Notably, the extra cost of token routing is minimal; the only difference lies in the negligible cost in the
concatenation of the gating scores from either the top-k or m experts on the local device. Moreover,
processing expanded tokens within local device eliminates inter-device communication.

5 Experiments

In this section, we conduct experiments under capacity-aware inference for MoE, with deployment
details provided in Appendix A.

5.1 Token Drop for High-load Experts

Table 1: Performance comparison across different capacity factors and selection metrics (i.e.,
Order, Reverse Order, Random, and Score). The baseline operates without capacity constraints,
represented as +?. We report the average performance over multiple random seeds.

Method

Baseline

Order
Reverse Order
Random
Score

Order
Reverse Order
Random
Score

Order
Reverse Order
Random
Score

?

OBQA PIQA RTE WinoGrande BoolQ ARC-C HellaSwag MMLU Avg.

+? 45.6

2.0

1.5

1.0

42.0
41.8
41.2
45.0

38.8
40.2
39.6
44.8

36.0
36.2
34.0
41.6

80.1

71.5
71.8
75.2
80.1

67.1
67.3
72.1
77.5

60.2
59.5
63.1
76.0

53.7

53.1
52.7
52.7
54.5

48.7
52.7
53.8
55.2

52.2
50.5
53.2
53.4

71.2

71.2
71.0
71.0
71.5

68.5
70.1
68.3
70.8

62.6
63.3
60.8
69.9

74.7

74.2
73.9
74.1
74.6

73.3
72.7
73.8
74.3

69.6
69.4
70.2
73.2

54.5

49.5
49.4
50.1
54.9

46.3
45.5
45.8
53.4

38.7
39.4
40.5
50.4

79.4

76.6
76.4
76.8
79.3

54.0
54.4
74.2
78.6

58.0
58.7
66.9
77.1

52.5

48.4
49.2
49.4
51.8

43.7
45.2
45.2
50.0

36.9
38.7
35.7
47.8

64.0

60.8
60.8
61.3
64.0

55.1
56.0
59.1
63.1

51.8
52.0
53.1
61.1

Investigation on Token Drop Metrics To assess the effectiveness of different metrics in regulating
token load to the target capacity, we compare various approaches on OLMoE by discarding excess
tokens and applying a range of capacity factors. As shown in Table 1, varying the dropping metrics
impacts performance at different levels. With higher capacities, the model maintains comparable
performance even when using naive selection methods like �Random�. However, as the capacity
factor decreases, performance degradation becomes more pronounced, particularly for �Order�,
�Reverse Order�, and �Random�. Notably, �Score� consistently outperforms other methods by a
large margin, demonstrating the effectiveness of leveraging gating scores as an importance measure.
Consequently, we adopt "Score" as the default metric.

Figure 4: Speedup of a single MoE layer compared to the baseline without capacity constraints,
achieved through two capacity-aware inference methods: Token Drop and Expanded Drop.

Efficiency Gains from Capacity-Constrained Inference We next explore the efficiency improve-
ments achieved by imposing expert capacity. Specifically, we employ distributed inference using
eight H20 GPUs, utilizing an 8-way Data Parallelism (DP) and 8-way Expert Parallelism (EP) strategy
through the Megatron-LM framework [36]. Notably, in Mixtral-8�7B-Instruct model, each GPU

6

64Experts on 8GPUs(8E / GPU)64 Experts + Shared Expert on 8 GPUs(8E & SE / GPU)8Experts on 8GPUs(1E / GPU)60 Experts + Shared Expert on 6 GPUs(10E&SE / GPU)Figure 5: End-to-End Model Speedup

Figure 6: Breakdown Analysis on OLMoE

hosts a single expert, whereas, in models like OLMoE-Instruct, multiple experts must be deployed on
a single GPU (e.g., eight experts per GPU) due to GPU resource constraints.

As illustrated in Figure 4, imposing constraints on expert capacity through Token Drop and Expanded
Drop, considerably accelerates inference across the four tested MoE models, in comparison to the
baseline model without capacity limitations. The enhanced efficiency of each MoE layer (Figure 4)
contributes to faster end-to-end model inference (Figure 5). Moreover, as the capacity factor ?
decreases, capacity-aware inference methods achieve significantly greater acceleration.

Notably, the efficacy of acceleration is influenced by the numerical relationship between the total
experts and the engaged GPUs in Expert Parallelism. As illustrated in Figure 5, for Mixtral-8�7B-
Instruct, deploying a single expert per GPU maximizes the effectiveness of capacity-aware inference.
In this configuration, Token Drop and Expanded Drop achieve end-to-end model speedups of 1.87 �
and 1.85�, respectively, with ? = 1.5. Conversely, deploying a greater number of experts on a single
GPU results in more modest acceleration gains, as evidenced by the �8E/GPU� (OLMoE-Instruct
and Deepseek-V2-Lite) the �10E/GPU� (Qwen1.5-MoE-Chat) in Figure 4 and Figure 5. This is
because the aggregated load from multiple experts diminishes the proportion of reduced load, which
is achieved by limiting the straggler expert. Therefore, it is anticipated that allocating more GPUs for
expert distribution, thereby reducing the number of experts per GPU, would enhance the acceleration
effect of capacity-aware inference.

The breakdown analysis presented in Figure 6 demonstrates that our proposed capacity-aware
inference methods substantially reduce the duration of expert computation, permutation and commu-
nication, while preserving a comparable cost for gate processing. Notably, the duration of permutation
and communication increases when tokens are expanded across a range of global experts. This is due
to the increased communication workload required to transmit expanded global tokens across various
GPU devices. Consequently, these results underscore the necessity of restricting the expanded tokens
to be processed by local experts.

Mitigating the Straggler Effect with Minimal Token
Discarding Given that expert capacity enforces MoE
layers to discard overflowed tokens, we next establish the
relationship between expert capacity and the correspond-
ing number of dropped tokens. For a capacity factor ?, the
total proportion of dropped tokens is given by:

DT =

(cid:80)n

i=1 ReLU(Ni ? ? �N )
i=1 Ni

(cid:80)n

,

(12)

where ReLU(Ni ? ? �N ) represent the number of dropped
tokens for the i-th expert.

Figure 7: Analysis of dropped tokens
with respect to capacity factors.

Figure 7 visualizes the number of dropped tokens across different capacity factors for various test
datasets, with a more detailed illustration provided in Appendix F. Although the most overloaded
expert receives much more tokens than the expected number of tokens �N , regulating the maximum
capacity has a limited impact on the overall number of accommodated tokens, thereby maintaining
competitive performance even after discarding overflow tokens. Moreover, dropping a small propor-
tion of overflowed tokens can significantly reduce the latency caused by overloaded experts (e.g.,
dropping 12% of overloaded tokens promotes the inference speed by 85% in Mixtral-8�7B-Instruct),
highlighting the efficacy of capacity-aware inference in improving both performance and efficiency.

7

             Multiple Experts per GPU        Single Expert per GPU1.01.21.41.61.8Speedup1.141.121.241.181.111.081.101.071.871.85OLMoE-2-DropOLMoE-2-Expand-DropOLMoE-0.5-DropOLMoE-0.5-Expand-DropDeepseek-2-DropDeepseek-2-Expand-DropQwen-1.5-DropQwen-1.5-Expand-DropMixtral-1.5-DropMixtral-1.5-Expand-DropBase1.01.52.01.01.52.01.01.52.002468Time (ms)Token DropExpanded Drop (Global))Expanded Drop (Local)GateExpert ComputationPermutation & Communication012345678Capacity Factor 020406080100Dropped Tokens (%)OBQAPIQARTEWinoGrandeBoolQTable 2: Comparison of Expert Drop, Token Drop and Expanded Drop. The capacity factor ? is
set to 2.0 for OLMoE and DeepSeek-V2-Lite, and 1.5 for Qwen1.5-MoE-Chat and Mixtral-8�7B-
Instruct. For Expert Drop, each forward pass skips one out of eight experts for Mixtral-8�7B-Instruct,
and the bottom 10% of lowest load experts for other models.

Model

OLMoE-Instruct

Qwen1.5-MoE-Chat

DeepSeek-V2-Lite-Chat

Mixtral-8�7B-Instruct

Method

Baseline

Expert Drop
Token Drop
Expanded Drop

Baseline

Expert Drop
Token Drop
Expanded Drop

Baseline

Expert Drop
Token Drop
Expanded Drop

Baseline

Expert Drop
Token Drop
Expanded Drop

OBQA PIQA RTE WinoGrande BoolQ ARC-C HellaSwag MMLU GSM8K Avg.

47.6

44.6
47.8
47.2

42.4

41.4
40.4
43.4

45.4

41.8
45.2
45.4

47.4

46.8
46.4
47.8

80.2

76.9
77.9
79.4

79.9

78.7
78.8
79.1

81.4

77.6
78.3
79.4

84.8

83.2
83.3
85.0

67.9

64.0
64.6
66.3

72.9

71.2
72.6
72.6

72.6

71.9
72.6
73.3

71.8

70.1
71.7
71.8

69.9

67.6
69.2
70.5

70.0

68.6
69.1
69.6

75.5

72.5
74.0
75.4

82.5

81.3
82.2
83.0

80.7

78.2
80.0
80.9

81.3

80.6
80.9
81.1

82.9

81.6
83.2
83.2

88.5

87.6
88.3
88.6

57.0

54.4
57.2
57.1

54.1

52.9
53.0
53.4

61.0

57.1
59.3
60.4

71.7

67.1
71.2
71.5

80.6

77.0
79.7
80.3

80.4

79.1
80.0
80.3

81.5

75.5
80.9
81.5

87.5

85.6
87.4
87.6

52.8

50.6
51.5
52.3

59.8

58.1
59.3
59.3

57.3

53.3
57.3
57.2

70.2

66.2
69.1
70.2

35.1

31.6
32.4
34.4

52.0

49.4
51.9
52.1

66.4

56.0
62.7
64.1

64.2

62.3
64.7
64.6

63.5

60.5
62.3
63.2

65.9

64.4
65.1
65.6

69.3

65.3
68.2
68.9

74.3

72.2
73.8
74.5

5.2 Expanded Drop to Low-load Experts

Besides the experts overloaded with tokens, some low-load experts receive only a few tokens,
raising important questions: Are these low-load experts redundant and removable, or should they
be leveraged to balance token allocation? Recent works [29, 19] remove less important experts to
improve efficiency, while our proposed Expanded Drop increases their utilization by redistributing
tokens for a more balanced assignment. Next, we investigate the significance of low-load experts and
validate the effectiveness of Expanded Drop.

The Critical Role of Low-Load Experts To explore the impact of low-load experts, we further
compare dropping tokens (i.e., Token Drop) with skipping experts (i.e., Expert Drop). For Expert
Drop, we adopt a conservative strategy that dynamically skips the 10% of experts with the lowest
token loads. Notably, the proportion of tokens removed in Expert Drop is significantly lower than in
Token Drop (2% in Expert Drop vs. 12% in Token Drop on OLMoE-Instruct).

Despite this, as shown in Table 2, Expert Drop experiences significant performance degradation
and is outperformed by Token Drop by a large margin. Moreover, due to the small proportion of
tokens assigned to low-load experts, removing these experts provides only marginal improvements in
inference speed (less than a 5% speedup). These findings indicate that retaining low-load experts
better preserves the performance of MoE models.

Effectiveness of Expanded Drop We examine the effec-
tiveness of utilizing low-load experts by Expanded Drop
instead of simply discarding these tokens to meet the target
capacity. Comparing Expanded Drop with Token Drop,
redistributing excess tokens to low-load experts enhances
performance, yielding a 0.9% improvement in the average
performance of Qwen1.5-MoE-Chat. Furthermore, con-
sidering the performance degradation observed in Expert
Drop, our findings highlight the crucial role of low-load
experts in maintaining model effectiveness.

Expanded Drop overselects experts for each token to ex-
pand the selection scope to prompt a more balanced token-
expert assignments. As shown in Figure 8, increasing the
overselection ratio m allows tokens to consider more candidate experts after being dropped from
overflowed ones, thereby improving low-load expert utilization and balancing the expert load.

Figure 8: Normalized expert load after
Token Drop and Expanded Drop, with
the capacity ratio set to ? = 1.5.

5.3 Extension to Multi-modal Mixture of Experts

In addition to applying capacity-aware inference to MoE models for language tasks, we also explore
its effectiveness in multi-modal MoE settings. Specifically, we evaluate the OLMoE based MolmoE
[12], across multi-modal benchmarks, including MME [15], MMBench [28], and SEED-Bench [27].

8

01020304050600.00.51.01.5LoadToken Drop0102030405060Expert ID0.00.51.01.5LoadExpanded DropFigure 9: Multi-modal token assignments across
different experts.

Figure 10: Comparison on MMBench across
six multimodal capabilities in Appendix B.

Given that the input sequence contains tokens from mul-
tiple modalities, we first investigate different token drop-
ping strategies. Specifically, we first treat all tokens
equally and drops those with the lowest scores (�Uni-
form�). Beyond this, considering the redundancy often
found in image tokens, we also experiment with a strat-
egy that prioritizes dropping image tokens before selec-
tively removing text tokens (�Image First�). For com-
parison, we also consider drop text tokens first (�Text
First�). As shown in Table 3, on the MME benchmark,
this image-first strategy yields improved performance,
highlighting the benefit of prioritizing dropping image
tokens for the load balance in multi-modal MoE models.

Table 3: Capacity-aware inference for
Multi-modality MoE models under dif-
ferent routing strategies. �Percep.� and
�Cognit.� denote Perception and Cognition,
respectively. ? is set to 1.0.

Method

Baseline

Token Drop
Expanded Drop

Token Drop
Expanded Drop

Token Drop
Expanded Drop

Strategy

Percep. Cognit.

�

Uniform

Text First

Image First

1358.1

1248.4
1307.6

1114.2
1163.6

1346.5
1362.1

269.6

245.4
273.6

214.4
241.3

288.9
297.1

Given the redundancy of image tokens and their large proportion in multi-modal tasks, we further
investigate more aggressive capacity factors for Token Drop and Expanded Drop using the �Image
First� strategy. Figure 10 demonstrates the effectiveness of Capacity-Aware Inference under low
capacity constraints (i.e., ? = 0.5). This is largely due to the high redundancy in image tokens
[4], which allows a higher dropping ratio without significantly affecting performance. Meanwhile,
the dominance of image tokens enables the use of very low capacity factors without significantly
affecting text token retention, as illustrated in Figure 9. Therefore, dropping image tokens at higher
ratios leads to more balanced token assignments and substantially improved inference efficiency.

5.4 Ablation Study

Model-Specific Imbalanced Property We explore the imbalance property in various models, such
as OLMoE, DeepSeek, Qwen and Mixtral, which differ in both architectures (e.g., depth and width)
and training strategies (e.g., training from scratch [31, 10] vs. training after upcycling [22, 39]).

On the one hand, our findings in Appendix E reveal different training strategies result in significantly
varying levels of imbalance. Specifically, MoE models trained from scratch exhibit a much higher
degree of imbalance. For instance, OLMoE and DeepSeek-V2-Lite experience peak expert-wise token
allocations exceeding 5 �N , whereas Qwen1.5-MoE and Mixtral are upcycled from dense language
models, maintain a more balanced distribution, with peak expert-wise allocations staying below 3 �N .
This is because upcycling initializes all experts with identical parameters [24], reducing divergence
and promoting balanced training in the early stages.

On the other hand, despite the widespread use of auxiliary balance loss in MoE training, it does not
guarantee balanced token assignments across experts, as token distribution still varies significantly
during inference on test data. This necessitates integrating expert capacity into the inference process.

(a) OLMoE

(b) OLMoE-Instruct

(c) Qwen1.5-MoE

(d) Qwen1.5-MoE-Chat

Figure 11: Performance change as capacity factors decrease from 3.0 to 0.0.

9

0102030405060Expert ID0123456Load=1.0=0.5Image TokensText TokensARCPFP-CFP-SLRRR01020304050607080Performance (%)69.378.739.265.539.860.066.877.739.362.536.455.769.278.039.663.838.157.4BaselineToken DropExpanded DropWinoGrandeARC-CHellaSwagMMLU3.02.62.21.81.41.00.60.2Capacity Factor304050607080Performance (%)3.02.62.21.81.41.00.60.2Capacity Factor304050607080Performance (%)3.02.62.21.81.41.00.60.2Capacity Factor20304050607080Performance (%)3.02.62.21.81.41.00.60.2Capacity Factor304050607080Performance (%)Capacity Factor Beyond the specific capacity values presented in Table 1, we further investigate a
wide range of capacity factors in Figure 11, spanning from 0.0 to 3.0. We exclude values exceeding 3.0,
as their performance closely aligns with capacity-agnostic scenarios. By analyzing the performance
changes when decreasing the capacity factor, we find that setting ? to 1.5 is sufficient to maintain
performance comparable to the original models. However, maintaining performance becomes
challenging under excessively low capacity factors, as high-load experts are forced to drop a significant
number of tokens.

6 Conclusion

In this paper, we identify the issue of imbalanced token-to-expert assignment in Mixture of Ex-
perts (MoE) models and introduce the Straggler Effect during inference, where high-load experts
become efficiency bottlenecks and dictate overall latency. To address this problem, we propose
Capacity-Aware Token Drop, which mitigates expert overload by enforcing strict capacity constraints.
Additionally, to better utilize underloaded experts, we present Capacity-Aware Expanded Drop, which
allows tokens to select additional experts on the same device while still respecting capacity limits,
thereby improving expert utilization. Our findings and proposed methods offer valuable insights and
effective strategies for improving MoE inference efficiency.

References

[1] Winogrande: An adversarial winograd schema challenge at scale. 2019.

[2] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning

about physical commonsense in natural language, 2019.

[3] Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. A survey on

mixture of experts, 2024. URL https://arxiv.org/abs/2407.06204.

[4] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao
Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for
large vision-language models, 2024.

[5] Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, and Yuanzhi Li. Towards understanding the
mixture-of-experts layer in deep learning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave,
and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL
https://openreview.net/forum?id=MaYzugDmQV.

[6] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and
Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions,
2019.

[7] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,
and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning
challenge, 2018.

[8] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John
Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168,
2021.

[9] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li,
Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization
in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.

[10] DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao,
Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji,
Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang,
Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li,
Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan,
Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang,

10

Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang,
Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng
Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan,
Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing
Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou,
Size Zheng, T. Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An,
Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang,
Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao
Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Liu, Xin Xie, Xingkai Yu, Xinnan Song, Xinyi Zhou,
Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y. K. Li, Y. X. Wei, Y. X. Zhu, Yanhong Xu,
Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Zheng, Yichao
Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Ying Tang, Yishi Piao, Yixin Dong, Yixuan Tan,
Yiyuan Liu, Yongji Wang, Yongqiang Guo, Yuchen Zhu, Yuduan Wang, Yuheng Zou, Yukun
Zha, Yunxian Ma, Yuting Yan, Yuxiang You, Yuxuan Liu, Z. Z. Ren, Zehui Ren, Zhangli Sha,
Zhe Fu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhiniu Wen,
Zhipeng Xu, Zhongyu Zhang, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, and Ziwei Xie.
Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.
URL https://arxiv.org/abs/2405.04434.

[11] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu,
Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian
Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao,
Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang,
Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo,
Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong
Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean
Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li,
Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian,
Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du,
R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu
Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu,
Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng
Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng,
Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang,
X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen,
Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang,
Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi
Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei,
Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng
Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying
He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo,
Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha,
Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou,
Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang,
Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong
Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu,
Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report,
2024. URL https://arxiv.org/abs/2412.19437.

[12] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park,
Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson,
Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris
Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert,
Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat,
Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon
Borchardt, Dirk Groeneveld, Jen Dumas, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa
Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi,
Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and
open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.

11

[13] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu,
Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of
language models with mixture-of-experts. In International Conference on Machine Learning,
pages 5547�5569. PMLR, 2022.

[14] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion
parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23
(120):1�39, 2022.

[15] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu,
Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive
evaluation benchmark for multimodal large language models. ArXiv, abs/2306.13394, 2023.
URL https://api.semanticscholar.org/CorpusID:259243928.

[16] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles
Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac�h, Haonan Li, Kyle McDonell, Niklas
Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron,
Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework
for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/
10256836.

[17] Jamie Hayes, Ilia Shumailov, and Itay Yona. Buffer overflow in mixture of experts.

In
Neurips Safe Generative AI Workshop 2024, 2024. URL https://openreview.net/forum?
id=SKWidEjUgU.

[18] Shwai He, Liang Ding, Daize Dong, Boan Liu, Fuqiang Yu, and Dacheng Tao. PAD-net:
An efficient framework for dynamic networks. In Anna Rogers, Jordan Boyd-Graber, and
Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 14354�14366, Toronto, Canada,
July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.803.
URL https://aclanthology.org/2023.acl-long.803.

[19] Shwai He, Daize Dong, Liang Ding, and Ang Li. Demystifying the compression of mixture-of-
experts through a unified framework, 2024. URL https://arxiv.org/abs/2406.02500.

[20] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and

Jacob Steinhardt. Measuring massive multitask language understanding, 2021.

[21] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures

of local experts. Neural computation, 3(1):79�87, 1991.

[22] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris
Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand,
et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

[23] Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm.

Neural computation, 6(2):181�214, 1994.

[24] Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa,
Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training
In The Eleventh International Conference on
mixture-of-experts from dense checkpoints.
Learning Representations, 2023. URL https://openreview.net/forum?id=T5nUQDrM4u.

[25] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang,
Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with condi-
tional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.

[26] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang,
Maxim Krikun, Noam Shazeer, and Zhifeng Chen. {GS}hard: Scaling giant models with
conditional computation and automatic sharding. In International Conference on Learning
Representations, 2021. URL https://openreview.net/forum?id=qrwe7XHTmYb.

12

[27] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying
Shan. Seed-bench: Benchmarking multimodal large language models. 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), pages 13299�13308, 2024.
URL https://api.semanticscholar.org/CorpusID:271963485.

[28] Yuanzhan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike
Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your
multi-modal model an all-around player? In European Conference on Computer Vision, 2023.
URL https://api.semanticscholar.org/CorpusID:259837088.

[29] Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and
Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-
experts large language models, 2024.

[30] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct

electricity? a new dataset for open book question answering, 2018.

[31] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min,
Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita
Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers,
Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh
Hajishirzi. Olmoe: Open mixture-of-experts language models, 2024. URL https://arxiv.
org/abs/2409.02060.

[32] OpenAI. Gpt-4 technical report, 2024.

[33] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andr�
Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts.
Advances in Neural Information Processing Systems, 34:8583�8595, 2021.

[34] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton,
and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts
layer. arXiv preprint arXiv:1701.06538, 2017.

[35] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton,
and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts
layer, 2017. URL https://arxiv.org/abs/1701.06538.

[36] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan
Catanzaro. Megatron-lm: Training multi-billion parameter language models using model
parallelism. arXiv preprint arXiv:1909.08053, 2019.

[37] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. Powerinfer: Fast large language model

serving with a consumer-grade gpu, 2023.

[38] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of

context, 2024.

[39] Qwen Team. Qwen1.5-moe: Matching 7b model performance with 1/3 activated parameters",

February 2024. URL https://qwenlm.github.io/blog/qwen-moe/.

[40] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman.
GLUE: A multi-task benchmark and analysis platform for natural language understanding. 2019.
In the Proceedings of ICLR.

[41] Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang
You. Openmoe: An early effort on open mixture-of-experts language models. arXiv preprint
arXiv:2402.01739, 2024.

[42] Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, and Mahesh Marina. Moe-infinity: Activation-aware

expert offloading for efficient moe serving, 2024.

[43] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a

machine really finish your sentence?, 2019.

13

[44] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Y Zhao, Andrew M. Dai,
Zhifeng Chen, Quoc V Le, and James Laudon. Mixture-of-experts with expert choice routing.
In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in
Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=
jdJo1HIVinI.

[45] Tong Zhu, Xiaoye Qu, Daize Dong, Jiacheng Ruan, Jingqi Tong, Conghui He, and Yu Cheng.
Llama-moe: Building mixture-of-experts from llama with continual pre-training. arXiv preprint
arXiv:2406.16554, 2024. URL https://arxiv.org/abs/2406.16554.

[46] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer,
and William Fedus. St-moe: Designing stable and transferable sparse expert models. arXiv
preprint arXiv:2202.08906, 2022.

14

A Implementation Details

Models We mainly focus on lightweight MoE models (less than 20B parameter budget). We
conduct experiments on OLMoE [31], Qwen1.5-MoE [39], DeepSeek-V2-Lite [10] and Mixtral [22],
due to their competitive performance and widespread adoption.

Datasets To evaluate model performance, we report normalized zero-shot or few-shot accuracy on
the LM-Harness benchmark. The number of shots for each task is detailed in Table 4, which includes
multiple tasks: ARC-C [7], BoolQ [6], HellaSwag [43], MMLU [20], OBQA [30], PIQA [2], RTE
[40], WinoGrande [1] and GSM8K [8]. The evaluation code is based on EleutherAI�s LM Harness
framework [16].

Table 4: Experimental settings for evaluation tasks. �Norm� refers to the normalization performed
with respect to the length of the input.

Task

Number of few-shot

Metric

BoolQ
RTE
OBQA
PIQA
MMLU
WinoGrande
GSM8K
HellaSwag
ARC-C

0
0
0
0
5
5
5
10
25

Accuracy
Accuracy
Accuracy (Norm)
Accuracy (Norm)
Accuracy
Accuracy
Exact Match
Accuracy (Norm)
Accuracy (Norm)

B Multi-Modal Tasks

In the scope of this paper, multimodal tasks refer to those involving both vision and language modali-
ties. We evaluate model performance using three representative benchmarks: MME, MMBench, and
SEED-Bench, each targeting different aspects of multimodal understanding and reasoning.

MME benchmark evaluates vision-language models along two dimensions: perception, which tests
visual grounding and recognition, and cognition, which assesses reasoning abilities such as counting
and relational understanding. It provides a fine-grained analysis of multimodal understanding.

MMBench is a comprehensive benchmark designed to assess the general multimodal understanding
ability of vision-language models. It evaluates model performance across six core capabilities: Coarse
Perception (CP), Fine-grained Perception�including single-instance (FP-S) and cross-instance (FP-
C), Attribute Reasoning (AR), Logical Reasoning (LR), and Relational Reasoning (RR). By covering
both perception and reasoning-oriented tasks, MMBench provides detailed insights into the strengths
and limitations of VLMs across diverse multimodal scenarios.

SEED-Bench is a large-scale benchmark for evaluating the generative comprehension of Multimodal
Large Language Models (MLLMs) across both image and video modalities. It includes 19K human-
annotated multiple-choice questions spanning 12 evaluation dimensions, enabling objective and
efficient assessment without human or GPT intervention. SEED-Bench reveals model limitations and
maintains a public leaderboard to support fair comparison and future research.

Component

Image
Text Prompt

Content

Token Count

Is this artwork titled virgin and child with sts catherine, cecilia, barbara, and ursula? Please answer yes or no.

576
31

Total

�
Table 5: An example multi-modal query in MME benchmark, showing the dominant proportion of
image tokens compared to text tokens.

607

As shown in Table 5, these tasks typically introduce a large number of image tokens. When faced with
imbalanced token-to-expert assignments, dropping redundant image tokens significantly improves

15

load balancing. Moreover, due to the high redundancy among image tokens, dropping a portion of
them has minimal impact on model performance.

As in MME and MMBench, the Image-First variants of Token Drop and Expanded Drop also exhibit
consistent effectiveness on SEED-Bench [27], maintaining strong performance even under low
capacity factors such as ? = 0.5. In addition to the redundancy in image tokens, Figure 9 shows that
text tokens constitute only a small portion of the total token assignments. This allows the regulation
of ? to retain almost all text tokens under the Image-First strategy.

Method

Baseline

Token Drop
Expanded Drop

Token Drop
Expanded Drop

Token Drop
Expanded Drop

?

?

0.5

1.0

1.5

Inst. Attr.

Inst. ID Inst. Interact.

Inst. Loc.

Inst. Count

Scene

Spatial Text Reasoning Overall

74.2

70.4
71.2

73.6
73.8

73.5
73.7

71.4

67.8
67.3

70.2
70.7

71.2
71.0

58.8

60.8
57.7

58.8
59.8

58.8
61.9

62.8

57.8
58.8

62.6
63.5

62.9
64.7

57.0

51.8
53.6

56.9
56.7

57.1
57.4

73.5

71.5
71.0

72.7
73.1

73.1
73.3

49.6

43.5
45.2

48.1
49.6

48.7
49.3

72.6

58.3
64.3

61.9
70.2

65.5
71.4

76.4

71.3
71.9

73.4
74.6

74.6
76.1

68.7

64.9
65.5

68.0
68.4

68.3
68.7

Table 6: Token Drop and Expanded Drop strategies for multi-modal MoE models evaluated on
SEED-Bench. Abbreviations: Inst. Attr. = Instance Attributes; Inst. ID = Instance Identification; Inst.
Interact. = Instance Interaction; Inst. Loc. = Instance Localization; Inst. Count = Instance Counting.

16

C Pseudocode for Token Drop and Expanded Drop

We present the detailed pseudo-code for the algorithm implementations, as shown in Algorithm 1 and
Algorithm 2. Both of these two methods adopt the selection metrics of �Score.�

Algorithm 1 Token Drop

Require: input_tokens, num_tokens, num_experts, k, ?

logits ? G(input_tokens)
scores ? Softmax(logits, dim = ?1)
topk_scores, topk_indices ? TopK(scores, k = k, dim = 1)
topk_masked_scores ? torch.zeros_like(logits).scatter(1, topk_indices, topk_scores)
topk_map ? torch.zeros_like(logits).int().scatter(1, topk_indices, 1).bool()

expert_capacity ? ? � num_tokens�k
num_experts
_, capacity_indices ? TopK(topk_masked_scores, k=expert_capacity, dim=0, sorted=False)
capacity_mask ? torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
final_map ? topk_map ? capacity_mask
final_scores ? topk_masked_gates � final_map
return final_scores, final_map

? Drop Token

Algorithm 2 Expanded Drop

Require: input_tokens, num_tokens, num_experts, k, ?, local_expert_id_list

logits ? G(input_tokens)
scores ? Softmax(logits, dim = ?1)
topk_scores, topk_indices ? TopK(scores, k = k, dim = 1)

? Expand

local_indices ? torch.tensor(local_expert_id_list).repeat(num_tokens, 1)
expanded_indices ? torch.cat((topk_indices, local_indices), dim = 1)
local_scores ? scores[:, local_expert_id_list]
expanded_scores ? torch.cat((topk_scores, local_scores), dim = 1)
expanded_masked_scores ? torch.zeros_like(logits).scatter(1, expanded_indices, expanded_scores)
expanded_map ? torch.zeros_like(logits).int().scatter(1, expanded_indices, 1).bool()

expert_capacity ? ? � num_tokens�k
num_experts
_, capacity_indices ? TopK(expanded_masked_scores, k=expert_capacity, dim=0, sorted=False)
capacity_mask ? torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
final_map ? expanded_map ? capacity_mask
final_scores ? expanded_masked_gates � final_map
return final_scores, final_map

? Drop Token

17

D Maximum Expert Selection

Expanded Drop employs overselection and dropping to not only ensure load balance but also improve
the utilization of underloaded experts. Although this process may allow some tokens to select more
than k experts, Table 7 shows that enforcing a strict k-expert limit is unnecessary: allowing additional
expert selections has the potential of enhancing representational capacity, while rigid constraints
introduce redundant computations.

Table 7: Ablation study on limiting the maximum number of k selected experts. �w/max� and
�w/o max� indicate runs with and without this constraint, respectively. ? is set to 1.0.

m Method OBQA PIQA RTE WinoGrande BoolQ ARC-C HellaSwag MMLU Avg.

2k

3k

w/ max
w/o max

w/ max
w/o max

42.4
42.2

42.0
42.0

75.8
75.9

75.6
75.6

53.2
53.4

53.4
53.8

68.6
69.7

69.6
69.8

72.7
73.1

72.7
72.9

50.3
50.3

50.3
50.3

77.1
77.0

77.0
77.1

47.6
47.8

47.4
47.6

61.0
61.2

61.0
61.1

E Layer-wise Expert Load

To analyze imbalanced token assignments, we measure the expert load for each expert by tracking the
peak expert load while running MoE models on various test datasets. Figure 12, 13, 14 and 15 present
the full results for the normalized layer-wise expert load for OLMoE, DeepSeek-V2, Qwen1.5-MoE,
and Mixtral-8�7B-Instruct, respectively.

F Calculation Dropped tokens

Based on Equation 12, we calculate the total number of dropped tokens across experts in each layer
under different capacity factors, as illustrated in Figures 16, 18, 17, and 19.

18

Figure 12: Layer-wise expert load in OLMoE-Instruct.

19

Expert ID02468LoadLayer 1Expert IDLayer 2Expert ID02468LoadLayer 3Expert IDLayer 4Expert ID02468LoadLayer 5Expert IDLayer 6Expert ID02468LoadLayer 7Expert IDLayer 8Expert ID02468LoadLayer 9Expert IDLayer 10Expert ID02468LoadLayer 11Expert IDLayer 12Expert ID02468LoadLayer 13Expert IDLayer 140102030405060Expert ID02468LoadLayer 150102030405060Expert IDLayer 16RTEWinoGrandeOBQABoolQPIQAFigure 13: Layer-wise expert load in Deepseek-V2-Lite.

20

Expert ID0510LoadLayer 2Expert IDLayer 3Expert ID0510LoadLayer 4Expert IDLayer 5Expert ID0510LoadLayer 6Expert IDLayer 7Expert ID0510LoadLayer 8Expert IDLayer 9Expert ID0510LoadLayer 10Expert IDLayer 11Expert ID0510LoadLayer 12Expert IDLayer 13Expert ID0510LoadLayer 14Expert IDLayer 15Expert ID0510LoadLayer 16Expert IDLayer 17Expert ID0510LoadLayer 18Expert IDLayer 19Expert ID0510LoadLayer 20Expert IDLayer 21Expert ID0510LoadLayer 22Expert IDLayer 23Expert ID0510LoadLayer 24Expert IDLayer 250102030405060Expert ID0510LoadLayer 260102030405060Expert IDLayer 27RTEWinoGrandeOBQABoolQPIQAFigure 14: Layer-wise expert load in Qwen1.5-MoE-Chat.

21

Expert ID05LoadLayer 1Expert IDLayer 2Expert ID05LoadLayer 3Expert IDLayer 4Expert ID05LoadLayer 5Expert IDLayer 6Expert ID05LoadLayer 7Expert IDLayer 8Expert ID05LoadLayer 9Expert IDLayer 10Expert ID05LoadLayer 11Expert IDLayer 12Expert ID05LoadLayer 13Expert IDLayer 14Expert ID05LoadLayer 15Expert IDLayer 16Expert ID05LoadLayer 17Expert IDLayer 18Expert ID05LoadLayer 19Expert IDLayer 20Expert ID05LoadLayer 21Expert IDLayer 220102030405060Expert ID05LoadLayer 230102030405060Expert IDLayer 24RTEWinoGrandeOBQABoolQPIQAFigure 15: Layer-wise expert load in Mixtral-8�7B-Instruct.

22

Expert ID0.02.5LoadLayer 1Expert IDLayer 2Expert ID0.02.5LoadLayer 3Expert IDLayer 4Expert ID0.02.5LoadLayer 5Expert IDLayer 6Expert ID0.02.5LoadLayer 7Expert IDLayer 8Expert ID0.02.5LoadLayer 9Expert IDLayer 10Expert ID0.02.5LoadLayer 11Expert IDLayer 12Expert ID0.02.5LoadLayer 13Expert IDLayer 14Expert ID0.02.5LoadLayer 15Expert IDLayer 16Expert ID0.02.5LoadLayer 17Expert IDLayer 18Expert ID0.02.5LoadLayer 19Expert IDLayer 20Expert ID0.02.5LoadLayer 21Expert IDLayer 22Expert ID0.02.5LoadLayer 23Expert IDLayer 24Expert ID0.02.5LoadLayer 25Expert IDLayer 26Expert ID0.02.5LoadLayer 27Expert IDLayer 28Expert ID0.02.5LoadLayer 29Expert IDLayer 3001234567Expert ID0.02.5LoadLayer 3101234567Expert IDLayer 32RTEWinoGrandeOBQABoolQPIQAFigure 16: Dropped tokens with respect to capacity factors in OLMoE-Instruct.

23

050100Dropped Tokens (%)Layer 1Layer 2050100Dropped Tokens (%)Layer 3Layer 4050100Dropped Tokens (%)Layer 5Layer 6050100Dropped Tokens (%)Layer 7Layer 8050100Dropped Tokens (%)Layer 9Layer 10050100Dropped Tokens (%)Layer 11Layer 12050100Dropped Tokens (%)Layer 13Layer 140.00.51.01.52.02.53.03.5050100Dropped Tokens (%)Layer 150.00.51.01.52.02.53.03.5Layer 16OBQAPIQARTEWinoGrandeBoolQFigure 17: Dropped tokens with respect to capacity factors in DeepSeek-V2-Chat.

24

050100DT (%)Layer 2Layer 3050100DT (%)Layer 4Layer 5050100DT (%)Layer 6Layer 7050100DT (%)Layer 8Layer 9050100DT (%)Layer 10Layer 11050100DT (%)Layer 12Layer 13050100DT (%)Layer 14Layer 15050100DT (%)Layer 16Layer 17050100DT (%)Layer 18Layer 19050100DT (%)Layer 20Layer 21050100DT (%)Layer 22Layer 23050100DT (%)Layer 24Layer 250.00.51.01.52.02.53.03.5050100DT (%)Layer 260.00.51.01.52.02.53.03.5Layer 27OBQAPIQARTEWinoGrandeBoolQFigure 18: Dropped tokens with respect to capacity factors in Qwen-1.5-MoE-Chat.

25

050100DT (%)Layer 1Layer 2050100DT (%)Layer 3Layer 4050100DT (%)Layer 5Layer 6050100DT (%)Layer 7Layer 8050100DT (%)Layer 9Layer 10050100DT (%)Layer 11Layer 12050100DT (%)Layer 13Layer 14050100DT (%)Layer 15Layer 16050100DT (%)Layer 17Layer 18050100DT (%)Layer 19Layer 20050100DT (%)Layer 21Layer 220.00.51.01.52.02.53.03.5050100DT (%)Layer 230.00.51.01.52.02.53.03.5Layer 24OBQAPIQARTEWinoGrandeBoolQFigure 19: Dropped tokens with respect to capacity factors in Mixtral-8�7B-Instruct.

26

050100DT (%)Layer 2Layer 3050100DT (%)Layer 4Layer 5050100DT (%)Layer 6Layer 7050100DT (%)Layer 8Layer 9050100DT (%)Layer 10Layer 11050100DT (%)Layer 12Layer 13050100DT (%)Layer 14Layer 15050100DT (%)Layer 16Layer 17050100DT (%)Layer 18Layer 19050100DT (%)Layer 20Layer 21050100DT (%)Layer 22Layer 23050100DT (%)Layer 24Layer 250.00.51.01.52.02.53.03.5050100DT (%)Layer 260.00.51.01.52.02.53.03.5Layer 27OBQAPIQARTEWinoGrandeBoolQ
