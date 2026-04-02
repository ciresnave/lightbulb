Preprint. Under review.

Reasoning Models Can Be Effective Without Thinking

Jingxuan He1 Charlie Snell1 Tyler Griggs1

Wenjie Ma1
Sewon Min1,2 Matei Zaharia1
1University of California, Berkeley
{windsey,jingxuan.he,csnell22,tgriggs,sewonm,matei}@berkeley.edu

2Allen Institute for AI

5
2
0
2

r
p
A
4
1

]
I

A
.
s
c
[

1
v
8
5
8
9
0
.
4
0
5
2
:
v
i
X
r
a

Abstract

Recent LLMs have significantly improved reasoning capabilities, primarily
by including an explicit, lengthy Thinking process as part of generation. In
this paper, we question whether this explicit thinking is necessary. Using
the state-of-the-art DeepSeek-R1-Distill-Qwen, we find that bypassing the
thinking process via simple prompting, denoted as NoThinking, can be sur-
prisingly effective. When controlling for the number of tokens, NoThinking
outperforms Thinking across a diverse set of seven challenging reasoning
datasets�including mathematical problem solving, formal theorem prov-
ing, and coding�especially in low-budget settings, e.g., 51.3 vs. 28.9 on
ACM 23 with 700 tokens. Notably, the performance of NoThinking becomes
more competitive with pass@k as k increases. Building on this observation,
we demonstrate that a parallel scaling approach that uses NoThinking to
generate N outputs independently and aggregates them is highly effective.
For aggregation, we use task-specific verifiers when available, or we ap-
ply simple best-of-N strategies such as confidence-based selection. Our
method outperforms a range of baselines with similar latency using Think-
ing, and is comparable to Thinking with significantly longer latency (up to
9x). Together, our research encourages a reconsideration of the necessity of
lengthy thinking processes, while also establishing a competitive reference
for achieving strong reasoning performance in low-budget settings or at
low latency using parallel scaling.

1

Introduction

Recent large reasoning models, such as DeepSeek-R1 (DeepSeek-AI et al., 2025), OpenAI
o1 (OpenAI, 2024), and others (Qwen Team, 2025; Muennighoff et al., 2025; Li et al., 2025a;
Ye et al., 2025), have significantly improved performance by increasing inference-time
compute (Snell et al., 2024; Brown et al., 2024). As shown in Figure 1, these models approach
complex tasks by first generating long chains of thought as a search process with reflection,
backtracking, and self-validation (�Thinking�, upper half of the blue box), followed by
final solution steps and a final answer (�Final Solution�, lower half). These reasoning
abilities are typically acquired through reinforcement learning with verified rewards or
fine-tuning on distilled reasoning traces. These models are generally believed to benefit from
this explicit reasoning process, which has led to the popular paradigm of inference-time
compute scaling�enhancing reasoning at the cost of significantly increased token usage and
latency. Recent work has explored efficient reasoning by reducing the token usage during
the thinking process, e.g., via reinforcement learning with length-based rewards (Aggarwal
& Welleck, 2025) or supervised fine-tuning on variable-length CoT traces (Sui et al., 2025).
However, most approaches still rely on the explicit Thinking process.

In this paper, we question whether the explicit Thinking process is necessary for advanced
reasoning. With an established reasoning model, DeepSeek-R1-Distill-Qwen, we disabled
the explicit thinking process through a simple prompting approach (orange box in Figure 1).
Our approach, called NoThinking, simply prefills the assistant response with a fabricated
Thinking block and has the model continue from there. We evaluate this approach extensively
on diverse benchmarks, including math problem solving (AIME 2024, AIME 2025, AMC

1

Preprint. Under review.

Figure 2: NoThinking out-
performs Thinking at a simi-
lar budget of 800 tokens.

Figure 1: Before producing the final step-wise solution, Think-
ing generates a long chain of thought in the dedicated think-
ing box, containing a search process with reflection, backtrack-
ing, and self-verification. In contrast, NoThinking directly
generates the solution after a prefilled dummy thinking box.

Figure 3: When control-
ling for latency, NoThinking
significantly outperforms
Thinking. Results are shown
with best-of-N sample se-
lection applied.

2023) (He et al., 2024), coding (Jain et al., 2025), and formal theorem proving (Zheng et al.,
2022; Azerbayev et al., 2023).

Our experimental results show that NoThinking is surprisingly effective, as evidenced by
pass@k metrics (1 ? k ? 64). Out of the box, NoThinking, using 2.0�5.1� fewer tokens, either
matches Thinking across all values of k or initially lags at k = 1 but quickly catches up and
sometimes surpasses Thinking as k increases. This pattern is not observed in the base models
used to train the reasoning model. Moreover, when controlling the token usage of both
approaches through the budget forcing technique (Muennighoff et al., 2025), NoThinking
often achieves higher pass@1 accuracy than Thinking in the low-budget regime, with the
performance gap expanding as k increases. This is exemplified in Figure 2. To systematically
assess efficiency, we examine the Pareto frontier of pass@k versus average token usage, and
find that NoThinking consistently outperforms Thinking. This suggests that even without the
explicit thinking process, NoThinking achieves significantly better accuracy-cost tradeoffs
across a wide range of budgets compared to the default prompting with Thinking.

The advantage of NoThinking in pass@k is so significant that it can enable new inference
strategies that achieve better accuracies and latencies than sequential inference with Think-
ing. To illustrate, we explore implementing parallel scaling with NoThinking, where multiple
responses are sampled in parallel and aggregated via best-of-N methods (Kang et al., 2025;
Wang et al., 2023). We consider both tasks with perfect verifiers (e.g., formal theorem
proving), where correctness can be automatically verified, and tasks without verifiers (e.g.,
general problem solving), where we rely on simple confidence-based selection strategies.
For tasks with verifiers, NoThinking with parallel scaling outperforms Thinking, both with
and without parallel scaling, achieving 7� lower latency and even 4� less total token usage.
For some tasks without verifiers, NoThinking also demonstrates strong performance: for
example, it surpasses Thinking with 9� lower latency and improved accuracy on Olympiad-
Bench (Math). As shown in Figure 3, we observe a consistent pattern across benchmarks:
NoThinking with parallel scaling offers a superior trade-off between accuracy and latency.
This illustrates that, for many tasks, reasoning model providers may be able to deliver better
latency and similar or improved quality through similar approach.

To summarize, our work is the first to show the surprising finding that the thinking process
of current reasoning models is not necessarily required for high performance, even for mod-
els that have been trained to follow this structured reasoning format. Moreover, we show
that this simple NoThinking prompting can be combined with parallel test-time compute to

2

Find        such that                                                 holds. Reasoning Task from AIME24Response from NoThinkingFinal Solution: We have two logarithmic equations: �. First, �. Now, �. Therefore, the result is               .Thinking: Okay, I think I have ?nished thinking.Response from ThinkingFinal Solution: To solve the problem, we ?rst � . Next, � . Finally, � . Therefore, the answer is               .Thinking: Let me parse the problem ?rst. ... Hmm, so we have two equations. � Maybe I can express one variable in terms of the other. � But wait, that seems complicated. Let me write it step by step. � Hmm, this is getting messy. Maybe I should take logarithms on both sides to make it easier. � Hmm, that�s still a complicated equation. � How about trying whether            works? � Alternatively, perhaps go back to the original equations.[...signi?cantly shortened trace�]So, the answer is               ? Let me verify the solution next. � Therefore, the answer is indeed correct.2021222324k020406080100Pass@k (%)OlympiadBench500100015002000250030003500Latency3540455055Accuracy (%)OlympiadBench with Best-of-16Preprint. Under review.

provide better latency-vs-accuracy than sequential inference. Overall, our work encourages
a reconsideration of the necessity of lengthy thinking processes, and provides a competitive
baseline for strong reasoning performance in low-budget or low-latency settings.

2 Related Work and Background

Test-Time Scaling for Language Models Scaling inference-time compute has emerged as
an effective strategy to enhance the reasoning capabilities of large language models (LLMs).
Broadly, existing methods fall into two categories. The first is sequential approaches ex-
emplified by models such as OpenAI�s o1, DeepSeek R1, and Qwen QwQ (DeepSeek-AI
et al., 2025; OpenAI, 2024; Qwen Team, 2025). These models produce long, structured
chain-of-thought (CoT) responses involving backtracking and verification in a single for-
ward pass. To gain such capabilities, costly training procedures�such as reinforcement
learning (RL) or iterative self-improvement�are required to bootstrap and refine reasoning
ability (Zelikman et al., 2022; Lambert et al., 2025; Yuan et al., 2024). For instance, DeepSeek
R1 is trained entirely via RL to generate extended CoTs, while RLVR (Lambert et al., 2025)
and PRIME (Yuan et al., 2024) leverage verifiable or latent supervision to improve reasoning
quality. The second category is parallel approaches that sample multiple outputs and aggre-
gate them. These include Best-of-N sampling and search-guided decoding techniques such
as Monte Carlo Tree Search (MCTS) (Snell et al., 2024; Brown et al., 2024), where multiple
candidate outputs are generated and selectively aggregated using search algorithms, reward
models, or external verifiers.

In contrast to prior work that assumes explicit thinking processes are essential for strong
performance, our work demonstrates that disabling explicit thinking is promising to im-
prove the accuracy-budget tradeoff. Our prompting-based method, simple yet overlooked,
achieves competitive performance without requiring additional training, reward signals, or
process supervision. While we utilize Best-of-N methods to illustrate the effectiveness of
our approach, our goal is not to innovate on sampling itself, but rather to present a strong,
cost-effective baseline for low-budget settings.

Efficient Reasoning Given the growing compute cost of sequential scaling in reasoning
models, recent work has explored various strategies to make reasoning in LLMs more
efficient. These approaches aim to promote smart and concise reasoning by optimizing the
length of generated reasoning sequences and eliminating unnecessary steps, as summarized
in a recent survey (Sui et al., 2025). Some methods continue to train reasoning models
for full-length CoT optimization, such as applying reinforcement learning with explicit
length-based rewards (Aggarwal & Welleck, 2025; Luo et al., 2025; Shen et al., 2025b; Arora
& Zanette, 2025; Qu et al., 2025), or fine-tuning on variable-length or concise reasoning
traces obtained from best-of-N sampling (Xia et al., 2025; Kang et al., 2024; Ma et al., 2025;
Munkhbat et al., 2025; Yu et al., 2024; Liu et al., 2024; Cui et al., 2025). Other approaches
modify the output paradigm itself to promote concise reasoning, for example by training
LLMs to leverage latent representations (Hao et al., 2024; Cheng & Durme, 2024; Shen et al.,
2025a; Xu et al., 2025b; Su et al., 2025; Saunshi et al., 2025; Shen et al., 2025c), or adopting
training-free techniques that select appropriate criteria to guide the reasoning strategy (Fu
et al., 2024; Wu et al., 2025b; Aytes et al., 2025; Liao et al., 2025; Sun et al., 2024; Li et al.,
2025b; Ding et al., 2025; Wang et al., 2025; Zhang et al., 2025; Yan et al., 2025). Additionally,
some methods explicitly instruct LLMs to generate fewer reasoning steps (Han et al., 2025;
Xu et al., 2025a; Lee et al., 2025; Renze & Guven, 2024; Wu et al., 2025a) (e.g., setting a token
budget in prompts), or dynamically route inputs based on task difficulty to control reasoning
complexity (Ong et al., 2025; Chuang et al., 2025b;a). While some of these methods are
effective, our approach is entirely orthogonal: we show that disabling the thinking process
can be surprisingly effective.

3 NoThinking Provides Better Accuracy-budget Tradeoffs than Thinking

In this section, we show that NoThinking can be surprisingly effective, outperforming
Thinking with budget forcing and offering better accuracy-budget tradeoffs. We first define

3

Preprint. Under review.

Thinking and NoThinking (Section 3.1), describe our experimental setup (Section 3.2), and
present experimental results (Section 3.3).

3.1 Method

Most modern reasoning models, such as R1 and R1-Distill-Qwen, follow a similar structure
in their generation process: the reasoning process within the thinking box, marked by
<|beginning of thinking|> and <|end of thinking|>, followed by the final answer. Based
on this structure, we define the two methods (Thinking and NoThinking) as follows.

Thinking refers to the default method of querying the reasoning models to produce the
following outputs: the reasoning process within the thinking box, the final solution, and the
final answer (Figure 1 (blue)).

NoThinking refers to a method that bypasses the explicit reasoning process through prompt-
ing, directly generating the final solution and answer. This is achieved by forcing the
thinking box to be empty during the decoding process (Figure 1 (orange)).

<|beginning of thinking|>
Okay, I think I have finished thinking.
<|end of thinking|>

The exact prompts we use can be found in Appendix C.

To control token usage in both methods, we adapt a budget forcing technique from Muen-
nighoff et al. (2025). When the model reaches the token budget, it is forced to generate
Final Answer:, ensuring an immediate final answer. If the model is still within the thinking
box at the token limit, <|end of thinking|> is appended before the final answer tag. To
implement budget forcing effectively, we slightly adjust the implementation for each task;
see Appendix A.1 for details.

3.2 Evaluation Setup

Models We use DeepSeek-R1-Distill-Qwen-32B (DeepSeek-AI et al., 2025) as our primary
model. It is a distilled variant of DeepSeek-R1, created by initializing the model with Qwen-
32B and training on the data generated by DeepSeek-R1. It is one of the state-of-the-art
reasoning models that use sequential test-time scaling, and is on par with a larger, DeepSeek-
R1-Distill-Llama-70B according to the report. As an additional baseline, we also include
Qwen-32B-Instruct (Qwen et al., 2025)�a general-purpose instruction-tuned model that
uses the same architecture but was not specifically designed for structured reasoning. We
have also experimented with R1-series models at smaller scales (7B and 14B) on AIME
tasks, and both exhibit similar behavior to the 32B model used in our main experiments.
The results have been added to Appendix B.2. We lack access for making interventions to
closed-source models. It is left to future work to explore models from other providers that
follow the Thinking-Solution format introduced in Section 1.

Tasks and Benchmarks We include a diverse set of challenging reasoning benchmarks,
covering tasks such as competitive math, coding, Olympiad problems, and theorem proving.

� Mathematical problem solving: We include both standard and Olympiad-level math
tasks. For standard problem solving, we use AIME 2024, AIME 2025, and AMC 2023,
which are widely used benchmarks for reasoning models (DeepSeek-AI et al., 2025;
OpenAI, 2024). For more advanced reasoning, we include the math subset of Olympiad-
Bench (He et al., 2024), which is much more challenging than AIME and AMC.

� Coding: We include LiveCodeBench (Jain et al., 2025) (release v2), a holistic and
contamination-free benchmark that continuously collects new problems over time.
� Formal theorem proving: We include MiniF2F (Zheng et al., 2022) for formal mathe-
matical reasoning, and ProofNet (Azerbayev et al., 2023) for logic and theorem proving.
These two datasets are among the most commonly used benchmarks for theorem proving
in formal languages.

4

Preprint. Under review.

Figure 4: Thinking vs. NoThinking vs. Qwen-Instruct without token budget controlled.
Blue, orange, and green indicate Thinking, NoThinking, and Qwen-Instruct, respectively. The
legend in each subfigure indicates the average number of tokens used by each approach.

Figure 5: Thinking vs. NoThinking with token budget controlled. Orange and blue
indicate NoThinking and Thinking, respectively. Under comparable token budget, NoThinking
consistently outperforms Thinking as k increases.

Metrics We report pass@k, which measures the probability of obtaining at least one correct
output among k randomly selected samples out of n generated completions per problem.
(cid:20)
It is formally defined as pass@k = E

, where n is the number of sampled

(cid:21)

problems

1 ? (n?c
k )
(n
k)

outputs per problem, and c is the number of correct outputs among them (Chen et al.,
2021). We used k = {1, 2, 4, 8, 16, 32} for theorem proving datasets (MiniF2F and ProofNet)
following the standard, k = {1, 2, 4, 8, 16, 32, 64} for smaller datasets (AIME24, AIME25,
AMC23) and k = {1, 2, 4, 8, 16} for larger datasets (OlympiaddBench, LiveCodeBench). For
formal theorem-proving benchmarks, pass@32 is the standard, while for math and coding
benchmarks, pass@1 (i.e., accuracy) is most commonly used.

3.3 Results

Thinking vs. NoThinking vs. Qwen Instruct without token budget controlled Figure 4
compares three models without budget forcing across different values of pass@k.

5

202122232425k020406080100pass@k (%)MiniF2F3049 tokens390 tokens850 tokens202122232425k020406080100ProofNet3250 tokens392 tokens889 tokens20212223242526k020406080100AIME 202410613 tokens4028 tokens955 tokens20212223242526k020406080100AIME 202512423 tokens3641 tokens890 tokensThinkingNoThinkingQwen-Instruct20212223242526k020406080100pass@k (%)AMC 20235828 tokens1154 tokens578 tokens2021222324k020406080100LiveCodeBench6702 tokens3444 tokens530 tokens2021222324k020406080100OlympiadBench-Math8615 tokens1960 tokens829 tokens20212223242526k020406080100Pass@k (%)AIME 2025 ( 1800 tokens)20212223242526k020406080100AMC 2023 ( 700 tokens)2021222324k020406080100LiveCodeBench ( 800 tokens)2021222324k020406080100OlympiadBench ( 800 tokens)20212223242526k020406080100Pass@k (%)AIME 2025 ( 3500 tokens)20212223242526k020406080100AMC 2023 ( 1700 tokens)2021222324k020406080100LiveCodeBench ( 3400 tokens)2021222324k020406080100OlympiadBench ( 1900 tokens)ThinkingNoThinkingPreprint. Under review.

Figure 6: Pass@k versus average token usage across different benchmarks. Each curve
represents a different method: blue for Thinking and orange for NoThinking. We use both
k = 1 and the largest available k for each benchmark.

First, on MiniF2F and ProofNet, NoThinking is comparable to Thinking across all values of
k (both significantly outperforming Qwen-Instruct). This is particularly surprising, given
that NoThinking is using 3.3�3.7x fewer tokens than Thinking. On other datasets, results
are more mixed. At k = 1, NoThinking lags behind Thinking, but the gap narrows as k
increases. Across all datasets, NoThinking eventually matches Thinking at the largest k,
while still using 2.0�5.1x fewer tokens. For AIME24, AIME25, and LiveCodeBench, both
Thinking and NoThinking significantly outperform Qwen-Instruct. However, on AMC23 and
OlympiadBench, Qwen-Instruct also closes the gap with Thinking and NoThinking. Example
inputs and outputs for different tasks and approaches are provided in Appendix C.

Thinking vs. NoThinking with token budget controlled NoThinking lags behind Thinking
on datasets other than MiniF2F and ProofNet, but it also uses significantly fewer tokens.
Therefore, we next compare Thinking and NoThinking with similar token budget on these
benchmarks, controlled by budget forcing described in Section 3.1.

As shown in Figure 5, when token usage is controlled, NoThinking generally outperforms
In particular, under a low-budget setting (e.g., fewer than ? 3, 000 tokens),
Thinking.
NoThinking consistently achieves better results than Thinking across all values of k, with
the performance gap widening as k increases. When token usage is higher (e.g., around
3, 500 tokens), Thinking performs better than NoThinking at pass@1, but NoThinking quickly
surpasses Thinking starting from k = 2.

Figure 6 further illustrates these results by plotting token usage on the x-axis, comparing
both pass@1 and pass@k with the largest available k. With pass@k, NoThinking consis-
tently outperforms Thinking across the entire budget range, without exception. For pass@1,
NoThinking is better in the low-budget regime, while worse in the high-budget regime.
LiveCodeBench is an exception, where Thinking outperforms NoThinking even in the low-
budget regime at pass@1. This is likely due to the fact that on LiveCodeBench, disabling the
thinking box does not significantly reduce token usage.

To mitigate the risk of data contamination, we include the newly released AIME 2025 which
is unlikely to have appeared in the pretraining data of existing models. Importantly, our
findings remain consistent across both new and established benchmarks, suggesting that the
observed trends are not artifacts of memorization but instead reflect generalizable model
behavior.

Summary Despite removing the thinking box�a core component of modern reasoning
models�NoThinking remains surprisingly effective. It matches Thinking while using 3.3�3.7x
fewer tokens on theorem-proving benchmarks and outperforms Thinking with a similar
token budget on other benchmarks, especially in low-budget settings�for instance, 51.3
vs. 28.9 on AMC23 with 700 tokens. Furthermore, NoThinking consistently outperforms

6

020004000600080001000012000Token Usage1020304050Accuracy (%)AIME 2025 (Pass@1)010002000300040005000Token Usage30405060708090AMC 2023 (Pass@1)0100020003000400050006000Token Usage525557606265677072LiveCodeBench (Pass@1)01000200030004000Token Usage303540455055OlympiadBench (Pass@1)020004000600080001000012000Token Usage20304050607080Accuracy (%)AIME 2025 (Pass@64)010002000300040005000Token Usage707580859095100AMC 2023 (Pass@64)0100020003000400050006000Token Usage80828486889092LiveCodeBench (Pass@16)01000200030004000Token Usage304050607080OlympiadBench (Pass@16)ThinkingNoThinkingPreprint. Under review.

Thinking across the entire budget range for pass@k with k > 1, with its advantage growing
as k increases.

3.4 Discussions and Analyses

Task-Specific Differences in NoThinking Performance Despite the consistent trends pre-
sented in Section 3.3, individual benchmarks reveal nuanced differences in how NoThinking
performs. In Figure 4, AMC 2023 shows a notable convergence across all three settings,
with minimal performance gaps�potentially suggesting saturation on this benchmark.
Moreover, MiniF2F and ProofNet exhibit surprising behavior: NoThinking achieves accuracy
comparable to Thinking at pass@1, while using significantly fewer tokens. However, this
should not be interpreted as task simplicity. In fact, our evaluation shows that even a strong
model like OpenAI�s o1 achieves only 30% accuracy on MiniF2F, and ProofNet consistently
yields low scores across all methods. These results emphasize the importance of consider-
ing task-specific variation in NoThinking effectiveness and that the strong performance of
NoThinking on certain benchmarks remains an open question for future work.

How Increasing k Affects NoThinking Performance We briefly explore potential rea-
sons for the observed changes in NoThinking performance as k increases, focusing on the
diversity of generated answers. We measure diversity by computing the entropy of the
answer distribution for each question. Specifically, for a question with answer distribution
{p1, p2, . . . , pn}, the entropy is defined as

H = ?

n
?
i=1
where pi is the empirical probability of the i-th unique answer. We then summarize diversity
using the mean and standard deviation of entropy across all questions. A higher mean
entropy indicates greater overall diversity, while a lower standard deviation suggests more
consistent diversity across questions. Our analysis is based on the Thinking vs. NoThinking
comparisons presented in Figure 5, excluding LiveCodeBench, which lacks exact answers.

pi log pi,

Task

Token Budget

Thinking

NoThinking

AIME 2025

AMC 2023

OlympiadBench

1800
3500

700
1700

800
1900

3.66 (�1.59)
3.03 (�2.06)
1.98 (�1.66)
1.36 (�1.52)
1.63 (�1.12)
1.16 (�1.13)

3.76 (�1.42)
3.43 (�1.37)
1.45 (�1.34)
1.34 (�1.23)
1.48 (�1.12)
1.42 (�1.12)

Table 1: Comparison of mean (� std) scores under different token budgets for Thinking and
NoThinking modes.

We observe no consistent advantage of either mode in terms of average diversity from Ta-
ble 1. In some settings, NoThinking yields higher mean entropy (AIME 2025); in others,
Thinking does (AMC 2023). However, NoThinking consistently exhibits lower variance
across questions, which suggests that NoThinking generates responses with more uniform
diversity across examples. We hypothesize that this increased consistency in diversity may
contribute to the improved pass@k performance observed with NoThinking as k increases,
though diversity alone does not fully explain the performance differences.

4 NoThinking Makes Parallel Test-Time Compute More Effective

Section 3 demonstrates that the advantage of NoThinking becomes more pronounced in
pass@k as k increases. This suggests that NoThinking can be effectively utilized with parallel
scaling methods�generating N samples independently in parallel and selecting the best-of-N
prediction. In this section, we show that such methods can outperform the configurations
possible through Thinking in terms of accuracy vs. latency.

7

Preprint. Under review.

Figure 7: Pass@1 results across all benchmarks, comparing NoThinking and Thinking, with
and without parallel scaling.

4.1 Motivation and Methods

Parallel Scaling v. Sequential Scaling Parallel scaling inherently provides low latency,
as multiple model invocations can be executed concurrently�either via API calls or local
model serving. This can be achieved through a multi-GPU setup or through batching on a
single GPU, resulting in higher GPU utilization compared to sequential scaling. The overall
latency is determined by the longest individual generation time. Motivated by our findings
that NoThinking generates more accurate solutions in the low-budget regime and becomes
increasingly effective in pass@k as k grows, we show that, when combined with simple
best-of-N methods, NoThinking with parallel sampling significantly improves accuracy. It
outperforms alternatives with comparable latency, such as Thinking with budget forcing and
parallel sampling, and even surpasses the pass@1 performance of full Thinking (Thinking
without budget forcing) under sequential scaling, despite incurring much lower latency.

Methods Parallel sampling requires a method that aggregates N independent solutions
to produce a single prediction, so-called best-of-N method. Formally, given N predictions
P = p1, � � � , pN, the best-of-N method selects p ? P as the final output. For tasks like
MiniF2F and ProofNet, we assume access to a perfect verifier f , which reliably determines
whether a given prediction p ? P is correct�the Lean (Moura & Ullrich, 2021) compiler. For
other tasks, where such a verifier is unavailable, we experiment with the following methods:

� Confidence-based. Following Kang et al. (2025), we adopt the self-certainty metric as a
lightweight, confidence-based selection method. Specifically, self-certainty quantifies
model confidence by computing the KL divergence between the predicted token distribu-
N
tion and a uniform distribution. After computing the self-certainty scores ci
i=1 for the set
of predictions P, we select the final answer using Borda voting, as described in the same
work. For benchmarks where equivalence checking is not applicable (LiveCodeBench),
we simply select the response with the highest self-certainty score.

� Majority voting. We report results based on consensus (majority vote) for tasks that have
exact answers (e.g., math problem solving and science), following prior work (DeepSeek-
AI et al., 2025; Wang et al., 2023). Specifically, given a set of answers {ai}n
i=1 extracted
1(ai = a).
?N
from all predictions in P, we define consensus as cons@n = argmaxa
i=1
For cases where k < N, we estimate cons@k by randomly sampling k answers from the
full set of N predictions, computing the majority vote, and averaging accuracy over
multiple trials using Monte Carlo simulation.

Metrics We define latency as the maximum number of tokens generated averaged across
the N trials and across the dataset. This metric is practically important, as lower latency
translates to faster response times for users, an essential goal in real-world applications.

8

02000400060008000Latency304050Pass@1 (%)MiniF2F0200040006000800010000Latency6810ProofNet01000200030004000Latency2040AIME 202401000200030004000Latency10203040AIME 2025Thinking with Parallel ScalingNoThinking with Parallel ScalingThinking without Parallel ScalingNoThinking without Parallel Scaling0500100015002000Latency304050607080Pass@1 (%)AMC 202301000200030004000Latency5456586062LiveCodeBench0100020003000Latency30354045505560OlympiadBench-MathPreprint. Under review.

Task

Thinking

BF (tokens)

Pass@K

Selection Methods (Pass@1)

Majority
Voting

Confidence
+ Highest

Confidence
+ Voting

AIME 2024

AIME 2025

AMC 2023

Thinking

NoThinking

Thinking

NoThinking

Thinking

NoThinking

3500

3500

3500

3500

2400

2400

73.33

77.30

40.00

53.73

92.50

95.00

43.33

46.67

30.00

33.33

77.50

77.50

40.00

20.00

30.00

20.00

65.00

57.50

46.67

50.00

30.00

33.33

77.50

85.00

Table 2: Comparison of Best-of-N selection methods�majority voting,
con-
fidence+highest, and confidence+voting�on selected experiments.
In �confi-
dence+highest�, we select the response with the highest model confidence.
In �confi-
dence+voting�, responses are ranked by confidence from high to low and assigned scores
using v(r) = (N ? r + 1)p, where r is the rank and p = 0.5; scores are then summed across
responses that yield the same answer. The best-performing method for each task is high-
lighted in bold. We also report pass@k as an upper bound, where k is the number of samples
used in Best-of-N selection.

4.2 Results

We present pass@1 results for Thinking and NoThinking across all benchmarks in Figure 7.
The performance of a single sampled response is referred to as pass@1 without parallel
scaling, while the accuracy of Best-of-N selection over multiple samples is considered
pass@1 with parallel scaling. For tasks without verifiers, we use confidence-based results
in the figure and present ablation results in Table 2 for selected experiments. This table
compares the Best-of-N methods discussed in Section 4.1. Confidence-based selection
generally outperforms majority voting. We also report pass@k accuracy as an upper bound
on pass@1 when using parallel scaling, which is included in Table 2. An extended version
of the table is available in Appendix B.3.

Perfect Verifiers NoThinking, when combined with parallel scaling, offers a highly efficient
alternative to traditional sequential approaches�achieving similar or better accuracy with
significantly lower latency and token usage. As shown in the first two plots of Figure 7,
NoThinking achieves comparable or even higher performance than Thinking, while incurring
much lower latency. Without parallel scaling, NoThinking closely matches Thinking in
accuracy at a fraction of the latency. If a perfect verifier is available, selecting the best
among k sampled responses would achieve pass@k accuracy. When combined with parallel
scaling, NoThinking matches the accuracy of Thinking without budget forcing and parallel
scaling�the representative sequential scaling baseline�while reducing latency by a factor
of 7. Moreover, on both the MiniF2F and ProofNet datasets, NoThinking achieves the
same accuracy using 4� fewer output tokens, highlighting its computational efficiency.
These results underscore the effectiveness of parallel sampling in verifier-available settings.
Additional details are provided in Appendix B.1.

Simple Best-of-N Methods NoThinking, when paired with parallel scaling and confidence-
based selection, consistently outperforms Thinking across most benchmarks under low token
budgets. Figure 7 (last five plots) presents confidence-based selection results across several
benchmarks, comparing Thinking and NoThinking under controlled token usage. We focus
on this low-budget regime for two reasons: (1) it aligns with our primary interest in efficient
inference, and (2) setting max tokens too high often results in excessively long and incoherent
outputs (�babbling�), which inflates latency and diminishes the value of comparison. As
expected, parallel scaling improves pass@1 performance for both Thinking and NoThinking.
However, across all math benchmarks, NoThinking consistently dominates the Pareto frontier

9

Preprint. Under review.

of Thinking with parallel scaling, demonstrating superior accuracy�budget tradeoffs. On
AMC 2023 and OlympiadBench, NoThinking outperforms Thinking consistently, regardless
of whether parallel scaling is used. Notably, even when compared to full Thinking (Thinking
without budget forcing), NoThinking achieves a higher pass@1 score (55.79 vs. 54.1) while
reducing latency by a factor of 9.

NoThinking is less effective on LiveCodeBench, which appears to be an outlier. This may be
due to the limitations of confidence-based selection for coding tasks, where voting strategies
cannot be applied without exact-match outputs. In these cases, we fall back to selecting
the highest-confidence response, which is less reliable. As shown in Table 2, this approach
consistently underperforms (often by a large margin) compared to voting-based methods
on tasks where voting is applicable.

Overall, these results highlight the effectiveness of NoThinking in verifier-free settings when
combined with parallel sampling and strong selection strategies.

Summary The surprising pass@k performance of NoThinking as k increases can be further
leveraged through parallel scaling, enabling improved pass@1 results at similar�or even
significantly lower�latency (up to 9� reduction). For tasks equipped with perfect verifiers,
this approach also reduces total token usage by up to 4x with similar or better accuracy.

5 Conclusion

Large language models generating a lengthy �thinking� process before producing solutions
have achieved strong results on reasoning tasks. In this paper, we question the necessity
of this process by introducing NoThinking, a simple yet effective prompting approach that
bypasses the explicit thinking process. We show that the same models, without long thinking
chains, can perform comparably to or better than their Thinking counterparts in pass@k
as k increases, while using significantly fewer tokens. Under comparable token budgets,
NoThinking consistently outperforms the traditional thinking results at most values of k.
Furthermore, we show that NoThinking can be combined with best-of-N selection methods
to achieve better accuracy�latency tradeoffs than are possible with standard Thinking.

We hope our research encourages a reconsideration of the necessity of lengthy thinking
processes, while providing a highly competitive reference for strong reasoning performance
in low-budget settings and at low latency.

Acknowledgements

This work was supported by the Sky Computing Lab at UC Berkeley. We are grateful to
Databricks for providing compute resources and thank Jonathan Frankle for facilitating
access. We also thank Kaylo Littlejohn (UC Berkeley) and Zhaoyu Li (University of Toronto)
for their valuable feedback on the draft.

References

Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks

with reinforcement learning, 2025. URL https://arxiv.org/abs/2503.04697.

Daman Arora and Andrea Zanette. Training language models to reason efficiently, 2025.

URL https://arxiv.org/abs/2502.04463.

Simon A. Aytes, Jinheon Baek, and Sung Ju Hwang. Sketch-of-thought: Efficient llm
reasoning with adaptive cognitive-inspired sketching, 2025. URL https://arxiv.org/
abs/2503.05179.

Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward W. Ayers, Dragomir
Proofnet: Autoformalizing and formally proving

Radev, and Jeremy Avigad.
undergraduate-level mathematics, 2023. URL https://arxiv.org/abs/2302.12433.

10

Preprint. Under review.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher
R�e, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with
repeated sampling, 2024. URL https://arxiv.org/abs/2407.21787.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto,
Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray,
Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin,
Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mo-
hammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings,
Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen
Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji,
Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh
Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage,
Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish,
Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code,
2021. URL https://arxiv.org/abs/2107.03374.

Jeffrey Cheng and Benjamin Van Durme. Compressed chain of thought: Efficient reasoning

through dense representations, 2024. URL https://arxiv.org/abs/2412.13171.

Yu-Neng Chuang, Leisheng Yu, Guanchu Wang, Lizhe Zhang, Zirui Liu, Xuanting Cai,
Yang Sui, Vladimir Braverman, and Xia Hu. Confident or seek stronger: Exploring
uncertainty-based on-device llm routing from benchmarking to generalization, 2025a.
URL https://arxiv.org/abs/2502.04428.

Yu-Neng Chuang, Helen Zhou, Prathusha Kameswara Sarma, Parikshit Gopalan, John
Boccio, Sara Bolouki, and Xia Hu. Learning to route llms with confidence tokens, 2025b.
URL https://arxiv.org/abs/2410.13284.

Yingqian Cui, Pengfei He, Jingying Zeng, Hui Liu, Xianfeng Tang, Zhenwei Dai, Yan
Han, Chen Luo, Jing Huang, Zhen Li, Suhang Wang, Yue Xing, Jiliang Tang, and Qi He.
Stepwise perplexity-guided refinement for efficient chain-of-thought reasoning in large
language models, 2025. URL https://arxiv.org/abs/2502.13260.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu,
Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan
Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu
Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong
Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu,
Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong
Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L.
Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin
Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang,
Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun
Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu
Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L.
Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu
Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu,
Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao,
Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An,
Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie,
Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin,
Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou,
Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao
Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong,
Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo,
Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo,
Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui
Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren,
Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao,

11

Preprint. Under review.

Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie,
Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang.
Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
URL https://arxiv.org/abs/2501.12948.

Yifu Ding, Wentao Jiang, Shunyu Liu, Yongcheng Jing, Jinyang Guo, Yingjie Wang, Jing
Zhang, Zengmao Wang, Ziwei Liu, Bo Du, Xianglong Liu, and Dacheng Tao. Dynamic
parallel tree search for efficient llm reasoning, 2025. URL https://arxiv.org/abs/2502.
16235.

Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Aurick Qiao, and
Hao Zhang. Efficiently serving llm reasoning programs with certaindex, 2024. URL
https://arxiv.org/abs/2412.20993.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen.
Token-budget-aware llm reasoning, 2025. URL https://arxiv.org/abs/2412.18547.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuan-
dong Tian. Training large language models to reason in a continuous latent space, 2024.
URL https://arxiv.org/abs/2412.06769.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi
Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong
Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-
level bilingual multimodal scientific problems, 2024. URL https://arxiv.org/abs/2402.
14008.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang,
Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and
In The Thirteenth
contamination free evaluation of large language models for code.
International Conference on Learning Representations, 2025. URL https://openreview.net/
forum?id=chfJJYC3iL.

Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. C3ot: Generating shorter chain-of-
thought without compromising effectiveness, 2024. URL https://arxiv.org/abs/2412.
11664.

Zhewei Kang, Xuandong Zhao, and Dawn Song. Scalable best-of-n selection for large
language models via self-certainty, 2025. URL https://arxiv.org/abs/2502.18581.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze
Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu,
Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind
Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi,
and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training,
2025. URL https://arxiv.org/abs/2411.15124.

Ayeong Lee, Ethan Che, and Tianyi Peng. How well do llms compress their own chain-of-
thought? a token complexity approach, 2025. URL https://arxiv.org/abs/2503.01141.

Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde,
Kourosh Hakhamaneshi, Shishir G. Patil, Matei Zaharia, Joseph E. Gonzalez, and Ion
Stoica. Llms can easily learn to reason from demonstrations structure, not content, is what
matters!, 2025a. URL https://arxiv.org/abs/2502.07374.

Peiji Li, Kai Lv, Yunfan Shao, Yichuan Ma, Linyang Li, Xiaoqing Zheng, Xipeng Qiu, and
Qipeng Guo. Fastmcts: A simple sampling strategy for data synthesis, 2025b. URL
https://arxiv.org/abs/2502.11476.

Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sa-
hoo, and Caiming Xiong. Reward-guided speculative decoding for efficient llm reasoning,
2025. URL https://arxiv.org/abs/2501.19324.

12

Preprint. Under review.

Tengxiao Liu, Qipeng Guo, Xiangkun Hu, Cheng Jiayang, Yue Zhang, Xipeng Qiu, and
Zheng Zhang. Can language models learn to skip steps?, 2024. URL https://arxiv.org/
abs/2411.01855.

Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun
Cao, and Dacheng Tao. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning
pruning, 2025. URL https://arxiv.org/abs/2501.12570.

Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. Cot-valve:
Length-compressible chain-of-thought tuning, 2025. URL https://arxiv.org/abs/2502.
09601.

Leonardo de Moura and Sebastian Ullrich. The lean 4 theorem prover and programming
language. In Automated Deduction � CADE 28: 28th International Conference on Automated
Deduction, Virtual Event, July 12�15, 2021, Proceedings, pp. 625�635, Berlin, Heidelberg,
2021. Springer-Verlag. ISBN 978-3-030-79875-8. doi: 10.1007/978-3-030-79876-5 37. URL
https://doi.org/10.1007/978-3-030-79876-5 37.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi,
Luke Zettlemoyer, Percy Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. s1: Simple
test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.

Tergel Munkhbat, Namgyu Ho, Seo Hyun Kim, Yongjin Yang, Yujin Kim, and Se-Young
Yun. Self-training elicits concise reasoning in large language models, 2025. URL https:
//arxiv.org/abs/2502.20122.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez,
M Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms with preference data,
2025. URL https://arxiv.org/abs/2406.18665.

OpenAI. Learning to reason with llms. OpenAI, 2024. URL https://openai.com/index/

learning-to-reason-with-llms/.

Yuxiao Qu, Matthew Y. R. Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beech-
ing, Ruslan Salakhutdinov, and Aviral Kumar. Optimizing test-time compute via meta
reinforcement fine-tuning, 2025. URL https://arxiv.org/abs/2503.07572.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu,
Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming
Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men,
Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang
Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan
Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Qwen Team.

Preview of qwen qwen1.5-32b.

qwq-32b-preview/, 2025. Accessed: 2025-03-20.

https://qwenlm.github.io/blog/

Matthew Renze and Erhan Guven. The benefits of a concise chain of thought on problem-
solving in large language models. In 2024 2nd International Conference on Foundation and
Large Language Models (FLLM), pp. 476�483. IEEE, November 2024. doi: 10.1109/fllm63129.
2024.10852493. URL http://dx.doi.org/10.1109/FLLM63129.2024.10852493.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi.
Reasoning with latent thoughts: On the power of looped transformers, 2025. URL
https://arxiv.org/abs/2502.17416.

Xuan Shen, Yizhou Wang, Xiangxi Shi, Yanzhi Wang, Pu Zhao, and Jiuxiang Gu. Efficient

reasoning with hidden thinking, 2025a. URL https://arxiv.org/abs/2501.19201.

Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang,
Kai Wang, and Shiguo Lian. Dast: Difficulty-adaptive slow-thinking for large reasoning
models, 2025b. URL https://arxiv.org/abs/2503.04472.

13

Preprint. Under review.

Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi:
Compressing chain-of-thought into continuous space via self-distillation, 2025c. URL
https://arxiv.org/abs/2502.21074.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute
optimally can be more effective than scaling model parameters, 2024. URL https://
arxiv.org/abs/2408.03314.

DiJia Su, Hanlin Zhu, Yingchen Xu, Jiantao Jiao, Yuandong Tian, and Qinqing Zheng. Token
assorted: Mixing latent and text tokens for improved language model reasoning, 2025.
URL https://arxiv.org/abs/2502.03275.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi
Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, and Xia Hu. Stop overthinking: A
survey on efficient reasoning for large language models, 2025. URL https://arxiv.org/
abs/2503.16419.

Hanshi Sun, Momin Haider, Ruiqi Zhang, Huitao Yang, Jiahao Qiu, Ming Yin, Mengdi Wang,
Peter Bartlett, and Andrea Zanette. Fast best-of-n decoding via speculative rejection, 2024.
URL https://arxiv.org/abs/2410.20290.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha
Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in
language models, 2023. URL https://arxiv.org/abs/2203.11171.

Yiming Wang, Pei Zhang, Siyuan Huang, Baosong Yang, Zhuosheng Zhang, Fei Huang, and
Rui Wang. Sampling-efficient test-time scaling: Self-estimating the best-of-n sampling in
early decoding, 2025. URL https://arxiv.org/abs/2503.01422.

Tong Wu, Chong Xiang, Jiachen T. Wang, and Prateek Mittal. Effectively controlling
reasoning models through thinking intervention, 2025a. URL https://arxiv.org/abs/
2503.24370.

Yuyang Wu, Yifei Wang, Tianqi Du, Stefanie Jegelka, and Yisen Wang. When more is less:
Understanding chain-of-thought length in llms, 2025b. URL https://arxiv.org/abs/
2502.07266.

Heming Xia, Yongqi Li, Chak Tou Leong, Wenjie Wang, and Wenjie Li. Tokenskip: Con-
trollable chain-of-thought compression in llms, 2025. URL https://arxiv.org/abs/2502.
12067.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. Chain of draft: Thinking faster by

writing less, 2025a. URL https://arxiv.org/abs/2502.18600.

Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. Softcot: Soft chain-of-thought for

efficient reasoning with llms, 2025b. URL https://arxiv.org/abs/2502.12134.

Yuchen Yan, Yongliang Shen, Yang Liu, Jin Jiang, Mengdi Zhang, Jian Shao, and Yueting
Zhuang. Inftythink: Breaking the length limits of long-context reasoning in large language
models, 2025. URL https://arxiv.org/abs/2503.06692.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is

more for reasoning, 2025. URL https://arxiv.org/abs/2502.03387.

Ping Yu, Jing Xu, Jason Weston, and Ilia Kulikov. Distilling system 2 into system 1, 2024.

URL https://arxiv.org/abs/2407.06023.

Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, Kaiyan Zhang, Bowen Zhou,
Zhiyuan Liu, and Hao Peng. Free process rewards without process labels, 2024. URL
https://arxiv.org/abs/2412.01981.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. Star: Bootstrapping reasoning

with reasoning, 2022. URL https://arxiv.org/abs/2203.14465.

14

Preprint. Under review.

Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun
Chen, and Ningyu Zhang. Lightthinker: Thinking step-by-step compression, 2025. URL
https://arxiv.org/abs/2502.15589.

Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. minif2f: a cross-system bench-
In International Conference on Learning

mark for formal olympiad-level mathematics.
Representations, 2022. URL https://openreview.net/forum?id=9ZPegFuFTFv.

15

Preprint. Under review.

A Method Details

A.1 Budget Forcing Details

To compare Thinking and NoThinking under similar token usage in an out-of-the-box manner,
we adopt a technique from Muennighoff et al. (2025), where a generation budget is enforced
by setting the max tokens parameter during the initial model call. If the generation reaches
this limit without producing a stop token (i.e., the reasoning is truncated), we prompt the
model to produce a final answer directly by appending an <|end of thinking|> tag (for
Thinking only) followed by a final answer prefix. For math and science tasks, we use the
prompt string �Final Answer:�. For coding tasks, we append code delimiters (i.e., ```) to
ensure the model generates code directly. For a given budget, we first run NoThinking and
record its average token usage, since its responses can vary in length and are not explicitly
constrained. We then apply a budget that is slightly lower than the average token usage
of NoThinking to Thinking because Thinking is more likely to be truncated under budget
constraints. This setup helps ensure that the total output length of Thinking (including
truncated reasoning and final answer) is roughly comparable to that of NoThinking.

B Additional Results

B.1 Results on Tasks with Verifiers

Full Thinking (Pass@1)

Task

Latency ?

Total token ? Accuracy

Latency ?

NoThinking (Pass@k)
Total Token ?

Accuracy

MiniF2F
Proofnet

7767
10178

743956
604507

34.84
5.52

1200 (-6.47x)
1413 (-7.20x)

190349 (-3.91x)
145921 (-4.14x)

36.38
6.90

k

2
2

Table 3: Costs for the NoThinking method to match full Thinking (pass@1) on tasks with
verifiers. The �Total tokens� column reports the cumulative number of tokens used across
all problems and samples to achieve the corresponding accuracy.

B.2 More Models

We experiment with smaller models (7B and 14B) from the R1-series on AIME tasks and
observe similar behavior to that discussed in Section 3. As shown in Figure 8, notably, on the
14B model, NoThinking initially lags behind but rapidly catches up as k increases, ultimately
approaching the performance of Thinking. NoThinking on the 7B model already produces
very few output tokens, so its performance does not increase as rapidly at k = 64 compared
to larger models. However, it still widens the gap with Qwen. We expect it to catch up to
Thinking with larger values of k.

We apply budget forcing to Thinking to match the token usage of NoThinking (Figure 9).
On the 7B model, NoThinking outperforms from the very beginning and continues to widen
the gap. On the 14B model, it initially lags behind but eventually catches up. Exploring
different budget settings would be an interesting direction for future work.

B.3 Best-of-N Methods Comparison

Table 4 presents the results of various best-of-N methods across all experiments. It is an
extended version of Table 2.

C Example Generations for Different Tasks

We present three examples of complete model traces that illustrate math problem solving,
coding, and formal theorem proving. Each example includes a comparison between Thinking
and NoThinking without budget forcing.

16

Preprint. Under review.

Figure 8: Thinking vs. NoThinking vs. Qwen-Instruct without token budget controlled on
DeepSeek-R1-Distill-Qwen-7B and -14B models. Blue, orange, and green indicate Thinking,
NoThinking, and Qwen-Instruct, respectively. We use a Qwen-Instruct model of the same
size as the reasoning model for comparison.

Figure 9: Thinking vs. NoThinking with token budget controlled on DeepSeek-R1-Distill-
Qwen-7B and -14B models. Orange and blue indicate NoThinking and Thinking, respectively.
Under comparable token budget, NoThinking consistently outperforms Thinking as k in-
creases.

17

20212223242526k020406080100pass@k (%)AIME 2024 (R1-7B)13118 tokens1408 tokens1177 tokens20212223242526k020406080100AIME 2024 (R1-14B)12035 tokens4025 tokens1044 tokens20212223242526k020406080100pass@k (%)AIME 2025 (R1-7B)14314 tokens1505 tokens1011 tokens20212223242526k020406080100AIME 2025 (R1-14B)13057 tokens4371 tokens901 tokensThinkingNoThinkingQwen-Instruct20212223242526k020406080100Pass@k (%)AIME 2024 (R1-7B) ( 1400 tokens)20212223242526k020406080100AIME 2024 (R1-14B) ( 4000 tokens)20212223242526k020406080100AIME 2025 (R1-7B) ( 1500 tokens)20212223242526k020406080100Pass@k (%)AIME 2025 (R1-7B) ( 900 tokens)20212223242526k020406080100AIME 2025 (R1-14B) ( 4200 tokens)20212223242526k020406080100AIME 2025 (R1-14B) ( 2200 tokens)ThinkingNoThinkingPreprint. Under review.

Task

Thinking

BF (tokens)

Pass@k

Selection Methods (Pass@1)

Majority
Voting

Confidence
+ Highest

Confidence
+ Voting

AIME 2024

AIME 2025

AMC 2023

OlympiadBench

Thinking

NoThinking

Thinking

NoThinking

Thinking

NoThinking

Thinking

NoThinking

4000

3500

2000

1800

1200

3500

1800

4000

3500

2000

1600

4000

3500

2400

1800

700

3000

2400

1600

3000

2000

800

3000

1800

1000

80.00

73.33

50.00

50.00

23.33

77.30

63.33

50.00

40.00

30.00

23.33

57.70

53.73

92.50

90.00

70.00

95.00

95.00

95.00

69.44

65.88

43.77

71.22

70.47

66.96

53.33

43.33

26.67

26.67

13.33

46.67

36.67

36.67

30.00

23.33

13.33

36.67

33.33

77.50

67.50

37.50

80.00

77.50

72.50

59.05

52.37

32.64

55.49

53.12

50.00

33.33

40.00

26.67

13.33

3.33

20.00

23.33

26.67

30.00

10.00

10.00

23.33

20.00

65.00

62.50

32.50

62.50

57.50

57.50

53.12

48.81

28.19

49.85

46.14

43.03

53.33

46.67

26.67

26.67

13.33

50.00

36.67

40.00

30.00

23.33

13.33

40.00

33.33

77.50

65.00

37.50

82.50

85.00

72.50

58.46

52.57

33.09

55.79

54.01

49.55

Table 4: Comparison of Best-of-N selection methods�majority voting, confi-
dence+highest, and confidence+voting�on benchmarks with exact-match answers (i.e.,
math benchmarks), evaluated under different token budgets for budget forcing (BF).
In �confidence+highest�, we select the response with the highest model confidence. In
�confidence+voting�, responses are ranked by confidence from high to low and assigned
scores using v(r) = (N ? r + 1)p, where r is the rank and p = 0.5; scores are then summed
across responses that yield the same answer. The best-performing method for each task is
highlighted in bold. We also report pass@k as an upper bound, where k is the number of
samples used in Best-of-N selection.

C.1 AIME 2024

Below, we present the full input and output traces for both Thinking and NoThinking for the
example shown in Figure 1.

Thinking (6099 Completion Tokens)

User: Return your final response within . There exist real numbers x and y, both greater
(cid:0)x4y(cid:1) = 10. Find xy. You should carefully think about the
than 1, such that logx (yx) = logy

18

Preprint. Under review.

problem and reason step by step.

Assistant:

<think >
Alright, so I have this logarithmic problem here, and I need to find the product xy where x
and y are real numbers greater than 1. The problem states that log base x of yx equals 10, and
log base y of x4y also equals 10. Hmm, okay, let�s try to break this down step by step.
First, let me recall the logarithm change of base formula because that seems relevant here. The
formula is logb(a) = (ln a)/(ln b), right? So, if I have a logarithm with some base, I can convert
it to natural logarithms or any other base. That might help me manipulate the equations.
So, the first equation is logx(yx) = 10. Let me rewrite that using the change of base formula. It
becomes

ln(yx)
ln x

= 10.

Similarly, the second equation is logy(x4y) = 10, which would be

ln(x4y)
ln y

= 10.

Wait, maybe I can simplify these expressions before applying the change of base. Remember,
logb(ac) = c � logb(a). So, for the first equation, logx(yx) is equal to x � logx(y). Similarly, the
second equation, logy(x4y) is equal to 4y � logy(x).
So, substituting back, the first equation becomes x � logx(y) = 10, and the second equation
becomes 4y � logy(x) = 10.
Hmm, now, do I remember any relationship between logx(y) and logy(x)? Yes, they are
reciprocals of each other, because logx(y) = 1
Let me denote logx(y) as some variable to make it easier. Let�s say a = logx(y). Then, logy(x)
would be 1/a because of the reciprocal relationship.
So substituting back into the equations, the first equation becomes x * a = 10, and the second
equation becomes 4y � (1/a) = 10. So now, I have two equations:

logy(x) . That might come in handy here.

� 1) x � a = 10
� 2) 4y/a = 10

Okay, so from the first equation, x = 10
which simplifies to y = 5a
2 .
a and y = 5a
So now, I have expressions for x and y in terms of a. So, x = 10
2 .
Wait, but I also know that a = logx(y). Let me write that in terms of x and y:

a , and from the second equation, 4y = 10a, so y = 10a
4 ,

logx(y) = a

But since x = 10

a and y = 5a

2 , substituting back:
(cid:18) 5a
2

log 10
a

(cid:19)

= a

Hmm, that seems a bit complicated, but maybe I can rewrite this using logarithm properties or
exponentials.
Remember that logb(c) = d is equivalent to bd = c. So, in this case,

(cid:19)a

(cid:18) 10
a

=

5a
2

So, (10/a)a = (5a)/2.
Hmm, that�s an equation in terms of a, which I can try to solve.
Let me write that down:

(cid:19)a

(cid:18) 10
a

=

5a
2

This seems tricky because the variable is both in the base and exponent. Maybe taking the
natural logarithm on both sides would help.

19

Preprint. Under review.

So, taking ln of both sides:

(cid:19)a(cid:21)

(cid:20)(cid:18) 10
a

ln

= ln

(cid:19)

(cid:18) 5a
2

Using the logarithmic power rule on the left side:

a � ln

(cid:19)

(cid:18) 10
a

= ln(5) + ln(a) ? ln(2)

Let me expand that:

a � ln(10) ? a � ln(a) = ln(5) + ln(a) ? ln(2)
Let me denote t = ln(a), then a = et. Let me try substituting that in.
So, equation becomes:

Hmm, still seems messy, but maybe I can factor out et on the left side:

et � ln(10) ? et � t = ln(5) + t ? ln(2)

et(ln(10) ? t) = ln(5) + t ? ln(2)
This is a transcendental equation, which might not have an algebraic solution. Maybe I need
to solve it numerically or make an intelligent guess.
Alternatively, perhaps there�s a substitution or some way to rearrange terms.
Wait, let�s step back and see if there�s another approach. Maybe instead of expressing x and y
in terms of a, I can express both equations in terms of a and then find a relationship between x
and y.
Wait, another thought: since I have x = 10/a and y = 5a/2, maybe I can express y in terms of
x.
So, from x = 10/a, we get a = 10/x. Then, y = 5a/2 = 5 ? (10/x)/2 = 50/(2x) = 25/x.
So, y = 25/x.
Therefore, the product xy = x � (25/x) = 25. Wait, so is it that simple?
Wait, hold on, if y = 25/x, then xy = 25. But before I conclude that, I need to make sure that
this is consistent with the original equations.
Because sometimes when you manipulate equations, you can introduce extraneous solutions
or lose constraints.
So, let me verify if this holds.
Given that xy = 25, so y = 25/x.
Then, plugging back into the original equations:
First equation: logx(yx) = 10.
Which is logx((25/x)x) = 10.
Simplify inside the log: (25/x)x = 25x/xx.
So, logx(25x/xx) = 10.
Using logarithm rules, logx(25x) ? logx(xx) = 10.
Which is x � logx(25) ? x � logx(x) = 10.
Simplify: x � logx(25) ? x � 1 = 10.
So, x � logx(25) = x + 10.
Similarly, logx(25) can be written as ln(25)/ ln(x).
So, x � (ln(25)/ ln(x)) = x + 10.
Hmm, that�s the same as (x/ ln(x)) � ln(25) = x + 10.
Not sure if that helps. Let me see if I can manipulate the other equation.
Second equation: logy(x4y) = 10.
Given that y = 25/x, so log25/x(x4�(25/x)) = 10.
Simplify exponent: 4 � (25/x) = 100/x.
So, log25/x(x100/x) = 10.
Again, using logarithm power rule: (100/x) ? log25/x(x) = 10.
So, (100/x) ? [ln(x)/ ln(25/x)] = 10.
Simplify denominator: ln(25/x) = ln(25) ? ln(x).
So, (100/x) ? [ln(x)/(ln(25) ? ln(x))] = 10.
Multiply both sides by x:

100 � [ln(x)/(ln(25) ? ln(x))] = 10x

20

Preprint. Under review.

Divide both sides by 10:

Hmm, so 10 � ln(x) = x � (ln(25) ? ln(x)).
Bring all terms to one side:

10 � [ln(x)/(ln(25) ? ln(x))] = x

10 ln(x) + x ln(x) ? x ln(25) = 0

Factor terms:

ln(x) � (10 + x) ? x ln(25) = 0
Hmm, this still seems complicated. Maybe I need to test if x = 5, because 25 is 5 squared, so
maybe x = 5 is a solution.
Let me test x = 5.
If x = 5, then y = 25/5 = 5.
So, x = 5 and y = 5.
Let me check the original equations:
First equation: log5(55) = log5(3125) = 5, but it�s supposed to be 10. Hmm, that doesn�t work.
Wait, so x = 5 and y = 5 gives us logx(yx) = 5, not 10. So, that�s not a solution.
Hmm, maybe x = something else.
Wait, maybe x = something else. Let me think.
Alternatively, perhaps I made a wrong assumption when I set y = 25/x. Let�s see, because
from the earlier step, I had x = 10/a and y = 5a/2, which leads to y = 25/x, but that may or
may not satisfy the original logarithmic equations.
Alternatively, maybe I can use the first approach where I had (10/a)a = (5a)/2.
Let me try to solve this equation numerically.
So, let me define f (a) = (10/a)a ? (5a)/2.
We can try plugging in some values for a to see where f (a) = 0.
First, let me note that a must be positive because it�s a logarithm.
Let me try a = 2.
f (2) = (10/2)2 ? (5 ? 2)/2 = 52 ? 10/2 = 25 ? 5 = 20 > 0.
a = 3: f (3) = (10/3)3 ? (15)/2 ? (3.333)3 ? 7.5 ? 37.037 ? 7.5 ? 29.537 > 0.
a = 4: (10/4)4 ? (20)/2 = (2.5)4 ? 10 ? 39.0625 ? 10 ? 29.0625 > 0.
a = 5: (10/5)5 ? 25/2 = 25 ? 12.5 = 32 ? 12.5 = 19.5 > 0.
a = 1: f (1) = 101 ? 5/2 = 10 ? 2.5 = 7.5 > 0.
a = 0.5: (10/0.5)0.5 ? (5 ? 0.5)/2 = 200.5 ? 2.5 ? 4.472 ? 2.5 ? 1.972 > 0.
a = 0.25: (10/0.25)0.25 ? (5 ? 0.25)/2 = 400.25 ? 0.625 ? 2.514 ? 0.625 ? 1.889 > 0.
Hmm, all positive. So f (a) is positive at all these points. Wait, that suggests that (10/a)a is
always greater than (5a)/2 for a > 0? But that can�t be, because when a approaches infinity,
10/a approaches 0, so (10/a)a approaches 0, and (5a)/2 approaches infinity. So, at some point,
f (a) must become negative.
Wait, but in the a approaches 0, 10/a approaches infinity, so (10/a)a approaches infinity0,
which is an indeterminate form, but often tends to 1. For example, (something large)0 = 1,
but (something large)something small can be eln(something large)�something small which could go to 1
or something else.
Wait, but when a approaches 0+, (10/a)a = ea ln(10/a) = ea(ln 10?ln a).
As a approaches 0+, a(ln 10 ? ln a) approaches 0 ? (ln 10 ? (?in f ty)) which is 0 � (?in f ty),
which is undefined, but the limit might be finite.
Wait, actually, the limit as a approaches 0+ of a ln(1/a) is 0, because ln(1/a) goes to infinity,
but a goes to zero, and the product goes to zero.
So, (10/a)a approaches e0 = 1 as a approaches 0+.
So, f (a) approaches 1 ? 0 = 1 > 0.
Similarly, as a approaches infinity, (10/a)a approaches 0, and (5a)/2 approaches infinity, so
f (a) approaches negative infinity.
Therefore, by the Intermediate Value Theorem, there must be some a > 0 where f (a) = 0.
So, let me try a = 6: (10/6)6 ? (5 ? 6)/2 ? (1.6667)6 ? 15 ? 17.79 ? 15 ? 2.79 > 0.
a = 7: (10/7)7 ? (1.4286)7 ?? Let me compute step by step:
1.42862 ? 2.0408
1.42864 ? (2.0408)2a ? 4.164
1.42866 ? (4.164) � (2.0408) ? 8.490

21

Preprint. Under review.

1.42867 ? 8.490 � 1.4286 ? 12.11
So, (10/7)7 ? 12.11, and (5 � 7)/2 = 17.5, so f (7) = 12.11 ? 17.5 ? ?5.39 < 0.
So, between a = 6 and a = 7, f (a) crosses zero.
Let me try a = 6.5:

(cid:19)6.5

(cid:18) 10
6.5

?

5 � 6.5
2

? (1.5385)6.5 ? 16.25

Compute 1.53856.5:
First, ln(1.5385) ? 0.433
Multiply by 6.5: 0.433 � 6.5 ? 2.8145
Exponentiate: e2.8145 ? 16.65

(cid:17)6.5

(cid:16) 10
6.5

? 16.65

So,
Then, subtract 16.25: f (6.5) = 16.65 ? 16.25 ? 0.4 > 0
So, f (6.5) = 0.4 > 0
Next, try a = 6.6:

(cid:19)6.6

(cid:18) 10
6.6

?

5 � 6.6
2

? (1.5152)6.6 ? 16.5

Compute ln(1.5152) ? 0.416
Multiply by 6.6: 0.416 � 6.6 ? 2.7456
Exponentiate: e2.7456 ? 15.63

(cid:17)6.6

(cid:16) 10
6.6

? 15.63

So,
Subtract 16.5: f (6.6) = 15.63 ? 16.5 ? ?0.87 < 0
So, f (6.6) ? ?0.87 < 0
So, between a = 6.5 and a = 6.6, f (a) crosses zero.
Try a = 6.55:

(cid:19)6.55

(cid:18) 10
6.55

?

5 � 6.55
2

? (1.5267)6.55 ? 16.375

Compute ln(1.5267) ? 0.423
Multiply by 6.55: 0.423 � 6.55 ? 2.771
Exponentiate: e2.771 ? 16.0

(cid:17)6.55

(cid:16) 10
6.55

? 16.0

So,
Subtract 16.375: f (6.55) = 16.0 ? 16.375 ? ?0.375 < 0
Earlier, at a = 6.5, f (a) = 0.4 > 0
Try linear approximation between a = 6.5 and a = 6.6:
- Change in a = 0.1 - Change in f (a) = ?1.27 - Solve 0.4 + (?1.27)(?a/0.1) = 0

0.4 ? 12.7 � ?a = 0 ? ?a =

0.4
12.7

? 0.0315

So, a ? 6.5 + 0.0315 ? 6.5315
Try a = 6.53:

(cid:19)6.53

(cid:18) 10
6.53

?

5 � 6.53
2

? (1.531)6.53 ? 16.325

ln(1.531) ? 0.427
0.427 � 6.53 ? 2.788
e2.788 ? 16.25
So, f (6.53) = 16.25 ? 16.325 ? ?0.075
Try a = 6.52:

(cid:19)6.52

(cid:18) 10
6.52

?

5 � 6.52
2

? (1.534)6.52 ? 16.3

ln(1.534) ? 0.429
0.429 � 6.52 ? 2.796
e2.796 ? 16.35

22

Preprint. Under review.

f (6.52) = 16.35 ? 16.3 = 0.05 > 0
So, between a = 6.52 and a = 6.53, f (a) crosses zero.
Use linear interpolation:

f (a) = f1 +

f2 ? f1
a2 ? a1

(a ? a1)

0 = 0.05 +

?0.075 ? 0.05
0.01

Thus, a ? 6.524, and then:

(a ? 6.52) ? ?0.05 = ?12.5(a ? 6.52) ? a =

81.55
12.5

? 6.524

x =

10
a

?

10
6.524

? 1.533,

y =

5a
2

?

5 � 6.524
2

? 16.31

Then:

Which confirms that the product is approximately 25.
Finally:

xy ? 1.533 � 16.31 ? 25

eu �

v
u

= 10,

4ev �

u
v

= 10

Multiply:

�

eu(cid:17)

(cid:16) v
u

� 4ev(cid:17)

= 100 ? 4eu+v = 100 ? eu+v = 25

(cid:16) u
v
Take natural log: u + v=ln(25).
But u=ln x, v=ln y, so ln x + ln y=ln(25).
Which is ln(xy)=ln(25), so xy=25.
Oh! So, despite the complicated a, the product xy is exactly 25.
So, the answer is 25.
Therefore, I can conclude that xy=25.
Final Answer: The value of xy is 25 .
</think >
Given the equations logx(yx) = 10 and logy(x4y) = 10, we need to find the product xy.
First, we rewrite the logarithmic equations using the change of base formula and properties of
logarithms:
1. logx(yx) = 10 can be rewritten as x logx(y) = 10. Let a = logx(y). Then, this equation
becomes x � a = 10. 2. logy(x4y) = 10 can be rewritten as 4y logy(x) = 10. Since logy(x) =
logx(y) = 1
From these, we derive: - x = 10
We then express y in terms of x: - Since x = 10
a , we have a = 10
2 = 5� 10
expression for y, we get y = 5a
2x = 25
x .
Thus, y = 25
x , which implies xy = x � 25
x = 25.
To verify, we use natural logarithms and properties of logarithms: - Let u = ln x and v = ln y.
The equations transform into: 1. eu � v
u = 10 2. 4ev � u
Multiplying these equations: - (cid:0) v
v 4ev(cid:1) = 100 - Simplifying, we get 4eu+v = 100, leading
u eu(cid:1) (cid:0) u
to eu+v = 25. - Taking the natural logarithm, we find u + v = ln(25), which means ln(x) +
ln(y) = ln(25), or ln(xy) = ln(25). Therefore, xy = 25.
Thus, the value of xy is 25 .

a , this equation becomes 4y � 1
a - y = 10a

x . - Substituting a into the

a = 10.
4 = 5a
2

2 = 50

v = 10

1

x

NoThinking (373 Completion Tokens)

User: Return your final response within . There exist real numbers x and y, both greater
(cid:0)x4y(cid:1) = 10. Find xy. Please write the answer for this math
than 1, such that logx (yx) = logy

23

Preprint. Under review.

problem directly without any thinking process.

Assistant:

<think >
Okay I have finished thinking.
</think >
Let�s solve the problem.
We are given that logx(yx) = 10 and logy(x4y) = 10. We need to find xy.
First, recall the logarithm power rule: logb(an) = n � logb(a).
Applying this rule to the first equation:

This implies:

logx(yx) = x � logx(y) = 10

logx(y) =

10
x

Let�s denote this as equation (1).
Now, apply the power rule to the second equation:

This implies:

logy(x4y) = 4y � logy(x) = 10

logy(x) =

10
4y

=

5
2y

Let�s denote this as equation (2).
Recall that logb(a) = 1

loga(b) . Using this property, we can relate equations (1) and (2):

logx(y) =

1
logy(x)

=?

10
x

=

2y
5

=? 10 � 5 = 2y � x =? 50 = 2xy =? xy = 25

Simplifying:

10
x

=

2y
5

Thus, the value of xy is 25.
25

C.2 LiveCodeBench

Thinking (3326 Completion Tokens)

User: Generate an executable Python function generated from the given prompt. The function
should take stdin as input and print the output. Simply call the function after the definition.
There are three cards with letters a, b, c placed in a row in some order. You can do the following
operation at most once:
� Pick two cards, and swap them. Is it possible that the row becomes abc after the operation?

Output �YES� if it is possible, and �NO� otherwise.

Input
The first line contains a single integer t (1 ? t ? 6) � the number of test cases.
The only line of each test case contains a single string consisting of each of the three characters
a, b, and c exactly once, representing the cards.
Output
For each test case, output �YES� if you can make the row abc with at most one operation, or
�NO� otherwise.
You can output the answer in any case (for example, the strings �yEs�, �yes�, �Yes� and �YES�
will be recognized as a positive answer).Sample Input 1: 6
abc
acb
bac

24

Preprint. Under review.

bca
cab
cba
Sample Output 1:
YES YES YES NO NO YES
Note
In the first test case, we don�t need to do any operations, since the row is already abc.
In the second test case, we can swap c and b: acb ? abc.
In the third test case, we can swap b and a: bac ? abc.
In the fourth test case, it is impossible to make abc using at most one operation. You should
carefully think about the problem and reason step by step.

Assistant:

<think >
Okay, I have to solve this programming problem. Let�s read the problem carefully.
So, the task is: Given a string of three distinct characters (a, b, c), determine if it�s possible to
turn it into �abc� by performing at most one swap of two characters. If it can be done, output
YES; otherwise, NO.
Hmm, right. So, each test case is a permutation of a, b, c. We need to check if either:

1. The string is already �abc� � then we can do nothing and it�s a YES. OR

2. There exists exactly one swap that can turn it into �abc�.

Wait, but how do I determine whether a single swap can achieve this?
Let me think about it. For each input string, I can consider all possible single swaps and see if
any of them result in �abc�.
But wait, the string is only three characters long. So, the number of possible swaps is limited.
Each swap can be between any two positions. For example, for a string s of length 3, the
possible swaps are (0,1), (0,2), (1,2). So three possible swaps.
So, for each test case, I can do the following:

� If the string is already �abc�, output YES.

� Else, check each possible pair of indices, swap them, and see if the result is �abc�. If any of

the swaps result in �abc�, output YES. Otherwise, NO.

That sounds manageable.
But wait, what about the case where the string is just one swap away from �abc�? Like, for
example, the input is �acb�. Swapping positions 1 and 2 gives �abc�, so YES.
Another example: input is �bca�. Let�s see. The desired output is �abc�. So let�s see what
swaps can do:
Original: b c a
Swap 0 and 1: c b a ? not �abc�.
Swap 0 and 2: a c b ? not �abc�.
Swap 1 and 2: b a c ? still not �abc�.
So in that case, it�s a NO.
Wait, but the sample input 4 is �bca� and the output is NO. So that�s correct.
Another example: input is �cba�. Let�s see:
Swap 0 and 2: a b c ? yes. So output is YES.
So, the approach is:
For each test case s:

� If s is �abc�, output YES.

� Else, for each possible pair of indices (i, j), i � j, create a new string where s[i] and s[j] are
swapped. Check if any of these new strings equals �abc�. If yes, output YES; else, NO.

Alternatively, since the string is small, I can also compute how many characters are in the
correct position. For example, the desired string is �abc�. So:
If the number of correctly placed characters is 2 or 3, then it�s possible with one swap or zero.
Wait, let�s think: If the string is �acb�, then two characters are correct (a and c are in correct
positions). Swapping the wrong ones (b and c) will fix it.
But wait, in �acb�, a is in position 0 (correct), c is in position 1 (wrong), b is in position 2
(correct). So two correct. So swapping positions 1 and 2 (c and b) will give �abc�.
So, another approach: count the number of mismatches between the input string and �abc�.
If the number of mismatches is 0: output YES.
If the number is 2: then swapping those two will fix it. So output YES.

25

Preprint. Under review.

If the number is 1 or 3: then it�s impossible with one swap.
Wait, that�s a clever insight. Because:

� 0 mismatches: already correct.

� 1 mismatch: you can�t fix it with one swap, because a swap affects two positions. If only

one character is wrong, you can�t swap to fix it without messing up another.

� 2 mismatches: you can swap the two wrong positions, which will fix both.

� 3 mismatches: for example, �cba� is �c�, �b�, �a�. So all positions are wrong. How? Wait,
�cba� has a in position 2, which is correct for �abc�? No, �abc� has a in 0, b in 1, c in
2. So �cba� has a in 2, which is wrong. So all three positions are wrong. So that�s three
mismatches. But wait, can a single swap fix this? Let�s see.

Wait, �cba� ? swapping 0 and 2 gives �a b c� ? yes. So in this case, with three mismatches, a
single swap can fix the problem. So, the earlier approach is incorrect.
Wait, so the number of mismatches isn�t a sufficient condition. Because in the case of three
mismatches, sometimes a single swap can fix it.
So, maybe the initial approach of checking all possible swaps is better.
Alternatively, perhaps the number of mismatches is 0, 2, or 3. Wait, because in the case of three
mismatches, swapping the first and last characters can fix it.
Hmm, so perhaps the condition is: if the number of mismatches is 0, or if the number of
mismatches is 2, or if the number of mismatches is 3 but the string is the reverse of �abc�.
Wait, but �cba� is the reverse. So, for example, in �cba�, swapping 0 and 2 gives �abc�. So in
this case, the number of mismatches is 3, but it�s fixable with one swap.
So, this suggests that the number of mismatches is not the only factor. So, perhaps the initial
approach of checking all possible swaps is more reliable.
So, for each test case, I can:

1. Check if the string is already �abc�. If yes, output YES.

Else:

2. For each possible pair of indices (i, j) where i � j, swap the characters at these positions and

see if the result is �abc�.

3. If any of these swaps result in �abc�, output YES.

4. Else, output NO.

Since the string is only three characters, there are only three possible swaps to check, so this
approach is feasible.
So, now, how to implement this in Python.
Let�s outline the steps:
Read t test cases.
For each test case:
s = input string.
if s == �abc�: print YES else: for each possible pair (i, j) in (0,1), (0,2), (1,2): create a list of the
characters. swap the i-th and j-th elements. check if the new string equals �abc�. if any do, set a
flag to True and break. if any swap worked, print YES, else NO.
Wait, but wait a second. For a string of 3 characters, the possible swap pairs are (0,1), (0,2),
(1,2), which are all possible unique pairs. So for each test case, we can generate all possible
swaps.
But how to generate all possible swaps?
In code:
for i in range(3):

for j in range(i+1, 3):
# swap i and j
lst = list(s)
lst[i], lst[j] = lst[j], lst[i]
new_s = ''.join(lst)
if new_s == 'abc':
# success
flag = True
break

if flag:

break

If any of the swaps result in �abc�, then output YES.

26

Preprint. Under review.

Yes, that should work.
Now, let�s test this logic against the sample input.
Sample Input 1:
6
abc ? yes
acb ? swap 1 and 2 ? abc ? yes.
bac ? swap 0 and 1 ? abc ? yes.
bca ? possible swaps:
swap 0 and 1: bac ? bca? No. Wait, let�s see. Original is bca. Swapping 0 and 1: cba ? not
abc. Swapping 0 and 2: a c b ? acb ? not abc. Swapping 1 and 2: b a c ? bac ? no. So none
of the swaps give abc. So output is NO.
cab ? swap 0 and 2 ? abc ? yes? Let�s see: original is cab. Swap 0 and 2: gives abc. So yes.
cba ? swap 0 and 2: a b c ? yes.
Wait, but in the sample output, the last test case is cba, which outputs YES.
Wait, the sample input is 6 test cases:
Test case 1: abc ? YES
Test case 2: acb ? YES
Test case3: bac ? YES
Test case4: bca ? output is NO
Test case5: cab ? YES
Test case6: cba ? output is YES.
So, according to the sample output, the fifth test case is cab, which swapping 0 and 2 gives abc.
So yes.
But according to the code, for cab, the code would check all swaps:
swap 0 and 1: �cba� ? not �abc�
swap 0 and 2: �abc� ? yes.
So the code would output YES.
Wait, but according to the sample output, the fifth test case�s output is NO. Wait no, let me
recheck the sample input and output.
Wait the sample input is:
6
abc
acb
bac
bca
cab
cba
Sample Output:
YES
YES
YES
NO
NO
YES
Wait, so test case 5 is �cab� ? output is NO. But according to the code, swapping 0 and 2 in
�cab� gives �abc�, so why the sample output is NO?
Wait, that�s conflicting. Wait, what�s �cab�?
Wait, �cab� is c a b.
If I swap 0 and 2: positions 0 is c, 2 is b. So after swap, it�s b a c ? bac ? which is not abc.
Wait wait, no. Wait �cab� is �c�, �a�, �b�. Swapping 0 and 2: �b�, �a�, �c� ? �bac� which is not �abc�.
Wait, but earlier I thought that swapping 0 and 2 in �cab� would give �abc�, but that�s incorrect.
Wait, maybe I�m making a mistake in the swap.
Wait �cab� is index 0: �c�, 1: �a�, 2: �b�.
If I swap 0 and 1: string becomes �a c b� ? �acb� ? not �abc�.
If I swap 0 and 2: string becomes �b a c� ? �bac� ? not �abc�.
If I swap 1 and 2: string becomes �c b a� ? �cba� ? not �abc�.
So, none of the swaps for �cab� results in �abc�. So, the correct output is NO.
But according to my earlier approach, for �cab� the code would generate all possible swaps
and find none result in �abc�, so output NO.
But in my initial analysis, I thought that swapping 0 and 2 would fix it. Oh no, I made a
mistake there.
So, the code is correct.
Wait, so how come the sample output for the fifth test case is NO?

27

Preprint. Under review.

Because for �cab�, we need to find if any swap can turn it into �abc�.
But in �cab�, the correct positions are a is in position 1 (should be 0), b is in position 2 (should
be 1), c is in position 0 (should be 2). So, the number of mismatches is 3.
But a single swap can�t fix all three. Because a swap affects two positions. So, it�s impossible.
So, the code will correctly output NO for �cab�.
So, the code seems correct.
Another test case: �cba�.
Swap 0 and 2: �a� �b� �c� ? yes. So output is YES.
So, the code handles that.
So, the code approach is correct.
Now, implementing this in Python.
The code structure is as follows:
Read t.
For each test case:
s = input().strip()
if s == �abc�: print(�YES�) else: possible = False for i in range(3): for j in range(i+1, 3): lst = list(s)
lst[i], lst[j] = lst[j], lst[i] if �.join(lst) == �abc�: possible = True break if possible: break print(�YES�
if possible else �NO�)
Yes.
Now, let�s test this code against the sample inputs.
Sample Input:
6
abc ? yes.
acb ? swap 1 and 2: yes.
bac ? swap 0 and 1: yes.
bca ? no.
cab ? no.
cba ? swap 0 and 2: yes.
Which matches the sample output.
So, the code should be correct.
So, the final code is as written.
</think >
To solve this problem, we need to determine if we can transform a given string of three distinct
characters (�a�, �b�, �c�) into the string �abc� by performing at most one swap of two characters.
If it is possible, we output �YES�; otherwise, we output �NO�.
Approach

1. Check if the string is already �abc�: If the input string is already �abc�, we can immediately

output �YES�.

2. Check all possible single swaps: For each input string that is not �abc�, we generate all
possible strings that can be formed by swapping exactly two characters. There are three
possible swaps for a string of length 3:

� Swap the first and second characters.
� Swap the first and third characters.
� Swap the second and third characters.

3. Evaluate each swap: For each swap, we check if the resulting string is �abc�. If any swap
results in �abc�, we output �YES�. If none of the swaps result in �abc�, we output �NO�.

Solution Code
```python
t = int(input())
for _ in range(t):

s = input().strip()
if s == 'abc':

print("YES")
continue
possible = False
for i in range(3):

for j in range(i + 1, 3):

lst = list(s)
lst[i], lst[j] = lst[j], lst[i]
if ''.join(lst) == 'abc':

28

Preprint. Under review.

possible = True
break
if possible:
break

print("YES" if possible else "NO")

```

Explanation

� Reading Input: We read the number of test cases and then process each string.

� Immediate Check: If the string is already �abc�, we print �YES�.

� Swapping Characters: For each string that is not �abc�, we generate all possible strings by
swapping each pair of characters. We check if any of these swapped strings equals �abc�.

� Output Result: Based on whether any valid swap results in �abc�, we print �YES� or

�NO�.

This approach ensures that we efficiently check all possible single swaps to determine if the
transformation is possible, leveraging the small size of the input string to keep the solution
simple and efficient.

NoThinking (256 Completion Tokens)

User: Generate an executable Python function generated from the given prompt. The function
should take stdin as input and print the output. Simply call the function after the definition.
There are three cards with letters a, b, c placed in a row in some order. You can do the following
operation at most once:
� Pick two cards, and swap them. Is it possible that the row becomes abc after the operation?

Output �YES� if it is possible, and �NO� otherwise.

Input
The first line contains a single integer t (1 ? t ? 6) � the number of test cases.
The only line of each test case contains a single string consisting of each of the three characters
a, b, and c exactly once, representing the cards.
Output
For each test case, output �YES� if you can make the row abc with at most one operation, or
�NO� otherwise.
You can output the answer in any case (for example, the strings �yEs�, �yes�, �Yes� and �YES�
will be recognized as a positive answer).Sample Input 1: 6
abc
acb
bac
bca
cab
cba
Sample Output 1:
YES YES YES NO NO YES
Note
In the first test case, we don�t need to do any operations, since the row is already abc.
In the second test case, we can swap c and b: acb ? abc.
In the third test case, we can swap b and a: bac ? abc.
In the fourth test case, it is impossible to make abc using at most one operation. Please solve
the above problem without the thinking process.

Assistant:

<think >
Okay, I have finished thinking.
</think >
The task is to determine if each given permutation of the letters �a�, �b�, and �c� can be trans-
formed into �abc� with at most one swap.
Here�s how to approach it:

1. Check if the string is already �abc�: If yes, output �YES�.

29

Preprint. Under review.

2. If not, check how many positions each character is away from its target position in �abc�:

� If exactly two characters are out of place, swapping them will make the string �abc�.
� If more than two are out of place, it�s impossible with one swap.

3. Return �YES� if any of the above conditions are met, else �NO�.

Here�s the Python code implementing this logic:
```python
def can_form_abc(s):

if s == 'abc':
return True
target = 'abc'
# Check if exactly two elements are different
diff = sum(1 for a, b in zip(s, target) if a != b)
return diff == 2

t = int(input())
for _ in range(t):

s = input().strip()
if can_form_abc(s):
print("YES")

else:

print("NO")

```

C.3 ProofNet

Thinking (1913 Completion Tokens)

User:

Assistant:

<think >
Alright, so I have this theorem to prove: if G is a finite group and a is an element of G, then
there exists a positive integer n, depending on a, such that a raised to the power of n equals
the identity element e in G. Okay, let�s break this down.
First, I need to recall what a group is. A group is a set equipped with an operation that combines
any two elements to form a third element, satisfying four conditions: closure, associativity, the
existence of an identity element, and the existence of inverse elements. Since G is finite, it has
a finite number of elements.
Now, the theorem is saying that for any element a in G, if I keep multiplying a by itself
repeatedly, at some point I�ll get back to the identity element e. That seems intuitive because
the group is finite, so the powers of a can�t keep producing new elements forever; they have to
eventually repeat or cycle back.
Let me think about an example. Suppose G is the group of integers modulo 5 under addition.
Wait, but addition is commutative, and this is a finite group. For any element a ? G, adding it
to itself enough times will eventually give me 0, which is the identity element in this case. For
instance, take a = 2 in Z/5Z. Then 2 + 2 = 4, 2 + 2 + 2 = 6 ? 1 (mod 5), and so on, until
adding 2 five times gives 10 ? 0 (mod 5). So, n = 5 works here.
But the theorem is about multiplicative notation, so let�s think of a multiplicative group. Maybe
the multiplicative group of complex roots of unity. For example, consider the group of 4th
roots of unity: {1, i, ?1, ?i}. Take a = i. Then i1 = i, i2 = ?1, i3 = ?i, i4 = 1. So, n = 4 works
here.
These examples suggest that such an n exists, but I need to prove it in general for any finite
group.
The proof provided earlier mentions that because G is finite, the set {a, a2, a3, . . .} must be
finite. Wait, is that always true? If G is finite, then the number of distinct powers of a must be
finite because there are only finitely many elements in G. So, the sequence a, a2, a3, . . . must
eventually repeat.

30

Preprint. Under review.

The proof then says that there exist i and j with i ?= j such that ai = aj. Without loss of
generality, assume i > j. Then, multiplying both sides by (aj)?1, which is a?j, gives ai?j = e.
So, n = i ? j is a positive integer such that an = e.
That seems straightforward, but let me make sure I understand why i and j must exist. Since
the set {a, a2, a3, . . .} is finite, by the pigeonhole principle, there must be some repetition. That
is, there must be distinct exponents i and j where ai = aj. This is because we have infinitely
many exponents but only finitely many elements, so some elements must repeat.
Wait, but the exponents are infinite, while the group is finite. So, the mapping from exponents
to group elements can�t be injective; it must eventually repeat. Therefore, there must be i and j
with i ?= j such that ai = aj.
Assuming i > j, then ai?j = e. That makes sense because ai = aj implies ai?j = e.
But what if i < j? Then, we could have aj?i = e, but since we assumed i > j, it�s okay. The
important thing is that the difference is positive.
So, this shows that there exists some positive integer n = |i ? j| such that an = e.
But the theorem says, �there is a positive integer n, depending on a, such that an = e.� So, this
n exists for each a ? G.
Wait, but in the examples I thought of earlier, like Z/5Z under addition, the n was the order
of the element. In multiplicative groups, it�s similar; n is the order of a.
So, this seems to be a general proof that in any finite group, every element has finite or-
der�meaning there exists some n such that an = e.
I think this makes sense. The key idea is the finiteness of the group, which forces the powers
of a to repeat, leading to an = e for some positive integer n.
But to make this rigorous, I should probably formalize the argument, maybe using the pigeon-
hole principle explicitly.
Also, I need to ensure that the inverse operation is well-defined in the group, which it is
because groups have inverses for each element.
So, in summary, because G is finite, the sequence of powers of a must eventually repeat, leading
to a power that equals the identity element. This establishes the existence of such an n for each
a in G.

</think >

Now, let�s provide a step-by-step explanation and proof of the theorem.
Step-by-Step Explanation and Proof:
Theorem: Let G be a finite group, and let a be an element of G. Then, there exists a positive
integer n, depending on a, such that an = e, where e is the identity element in G.
Proof:

1. Understanding the Group Structure:

� G is a finite group, meaning it has a finite number of elements.
� The operation in G is associative, there exists an identity element e, and every element

has an inverse in G.

2. Considering Powers of an Element:

� For any element a ? G, consider the sequence of its powers: a, a2, a3, . . ..
� Since G is finite, this sequence cannot contain infinitely many distinct elements; it

must eventually repeat.

3. Application of the Pigeonhole Principle:

� The pigeonhole principle states that if more objects are placed into containers than

there are containers, at least one container must contain more than one object.

� Here, the �objects� are the powers a, a2, a3, . . ., and the �containers� are the elements

of G.

� Since G is finite, there are only finitely many �containers.� Therefore, there must exist

distinct integers i and j with i > j such that ai = aj.

4. Deriving the Order of a:

� Given ai = aj with i > j, multiply both sides on the right by (aj)?1:

Simplifying, we get:

ai � (aj)?1 = aj � (aj)?1

ai?j = e

31

Preprint. Under review.

� Let n = i ? j. Since i > j, n is a positive integer.
� Therefore, an = e.

5. Conclusion:

� We have shown that for any element a in a finite group G, there exists a positive

integer n such that an = e.

� The integer n is the smallest such positive integer for which this holds, known as the

order of a in G.

**Thus, the theorem is proven.**
```lean4
import Mathlib

open Fintype Set Real Ideal Polynomial
open scoped BigOperators

theorem exercise_2_1_26 {G : Type*} [Group G]

[Fintype G] (a : G) : exists (n : nat), a � n = 1 :=
-- We use the fact that the order of an element in a finite group divides the
order of the group.
-- This is a direct consequence of Lagrange's theorem.
?Fintype.card G, by

-- We need to show that a raised to the power of the order of the group equals
the identity element.
apply Eq.symm
-- Since the order of the group is finite, we can use the fact that the powers
of `a` must eventually repeat.
-- This is a direct application of the pigeonhole principle.
rw [pow_card_eq_one]

?
```

NoThinking (163 Completion Tokens)

User: Prove the following theorem in Lean 4 by completing the following Lean 4 code which
contains a header, informal prefix, and formal statement. Enclose your final code within a
Lean 4 code block that starts with:
```lean4
import Mathlib

open Fintype Set Real Ideal Polynomial
open scoped BigOperators

/-- If $G$ is a finite group, prove that, given $a \in G$, there is a positive integer
$n$, depending on $a$, such that $a�n = e$.-/
theorem exercise_2_1_26 {G : Type*} [Group G]

[Fintype G] (a : G) : exists (n : nat), a � n = 1 :=

```

Make sure your code successfully proves the formal statement.

Assistant:

<think >
Okay, I have finished thinking.
</think >

The final Lean 4 code is:
```lean4
import Mathlib

open Fintype Set Real Ideal Polynomial
open scoped BigOperators

32

Preprint. Under review.

theorem exercise_2_1_26 {G : Type*} [Group G]

[Fintype G] (a : G) : exists (n : nat), a � n = 1 := by
-- Since G is a finite group, the order of any element a in G must divide the order of G.
-- The order of G is given by the cardinality of its underlying set.
use Fintype.card G
-- By Lagrange's theorem, the order of a divides the order of G.
-- Therefore, raising a to the power of the order of G will yield the identity element.
simp [pow_card_eq_one]

```

33


