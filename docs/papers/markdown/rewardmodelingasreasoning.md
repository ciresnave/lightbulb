5
2
0
2

y
a
M
8
1

]
L
C
.
s
c
[

3
v
7
8
3
2
0
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

RM-R1: Reward Modeling as Reasoning

Xiusi Chen1?, Gaotang Li1?, Ziqi Wang1?, Bowen Jin1, Cheng Qian1, Yu Wang2,
Hongru Wang1, Yu Zhang3, Denghui Zhang4, Tong Zhang1, Hanghang Tong1, Heng Ji1
1University of Illinois Urbana-Champaign
2University of California, San Diego
3Texas A&M University
4Stevens Institute of Technology
{xiusic, gaotang3, htong, hengji}@illinois.edu

Abstract

Reward modeling is essential for aligning large language models with human
preferences through reinforcement learning from human feedback. To provide
accurate reward signals, a reward model (RM) should stimulate deep thinking
and conduct interpretable reasoning before assigning a score or a judgment. In-
spired by recent advances of long chain-of-thought on reasoning-intensive tasks,
we hypothesize and validate that integrating reasoning capabilities into reward
modeling significantly enhances RM�s interpretability and performance. To this
end, we introduce a new class of generative reward models � Reasoning Reward
Models (REASRMS) � which formulate reward modeling as a reasoning task. We
propose a reasoning-oriented training pipeline and train a family of REASRMS,
RM-R1. RM-R1 features a chain-of-rubrics (CoR) mechanism � self-generating
sample-level chat rubrics or math/code solutions, and evaluating candidate re-
sponses against them. The training of RM-R1 consists of two key stages: (1)
distillation of high-quality reasoning chains and (2) reinforcement learning with
verifiable rewards. Empirically, our models achieve state-of-the-art performance
across three reward model benchmarks on average, outperforming much larger
open-weight models (e.g., INF-ORM-Llama3.1-70B) and proprietary ones (e.g.,
GPT-4o) by up to 4.9%. Beyond final performance, we perform thorough empirical
analyses to understand the key ingredients of successful REASRM training. To
facilitate future research, we release six REASRM models along with code and
data at https://github.com/RM-R1-UIUC/RM-R1.

1

Introduction

Reward models (RMs) play a critical role in large language model (LLM) post-training, particularly
in reinforcement learning with human feedback (RLHF) [4, 24], where they serve as scalable proxies
for human evaluators. Existing research on reward modeling can be broadly classified into two
categories: (1) scalar-based reward models (ScalarRM) [20] and (2) generative reward models
(GenRM) [40]. Scalar-based approaches frame reward modeling as a classification problem, typically
training a sequence classifier on top of a language model. In contrast, generative approaches retain
the original language model decoding head and leverage the model�s generative abilities to produce
free-form pairwise judgments. While scalar-based methods are direct and often effective, they are
opaque, offering no intermediate reasoning steps to justify the model�s decisions. This lack of
transparency may limit their capacity to handle more challenging, reasoning-intensive preference

?Equal contribution.

Preprint. Under review.

Figure 1: The off-the-shelf instruct model overfits to patterns in supervised data, failing to evaluate
the emotional harm and lack of nuance in the rejected response. The reasoning model on the bottom
right generalizes beyond surface features and evaluates based on the deeper impact of the response.

tasks. On the other hand, although generative methods provide greater transparency, their reasoning
is often superficial and unhelpful for reliable judgment, leading to suboptimal performance [7, 22].

In real-world decision-making scenarios, accurate and grounded reward modeling often requires
jointly conducting reasoning and reward assignment. This is because preference judgments inherently
involve multifaceted cognitive considerations, such as inferring a judge�s latent evaluation criteria [5],
navigating trade-offs among multiple criteria [23], and simulating potential consequences [33], all
of which necessitate extensive reasoning. Our example in Figure 1 illustrates such an example,
where a correct preference judgement requires accurate perception of the question, understanding of
the corresponding evaluation rubrics with convincing arguments � closely mirroring how humans
approach grading tasks. Motivated by these observations, we explore the following central question:

Can we cast reward modeling as a reasoning task?

In this work, we unleash the reasoning potential of RMs and propose a new class of models:
Reasoning Reward Models (REASRMS). Different from standard GenRMs, REASRMS emphasize
leveraging long and coherent reasoning chains during the judging process to enhance the model�s
ability to assess and distinguish complex outputs accurately. We validate that integrating long
reasoning chains during the judging process significantly enhances downstream reward model
performance. We explore several strategies for adapting instruction-tuned language models into
logically coherent REASRMS. Notably, we find that solely applying reinforcement learning with
verifiable rewards (RLVR) [12] in reward modeling does not fully realize the model�s reasoning
capabilities. We also observe that plain chain-of-thought (CoT) reasoning falls short at perceiving the
fine-grained distinction across different question types.

Through a series of studies, we design a training pipeline that introduces reasoning distillation prior
to RLVR, ultimately resulting in the development of RM-R1. To fully elicit the reasoning capability
of RM-R1 for reward modeling, we design a Chain-of-Rubrics (CoR) process. Specifically, the
model categorizes the input sample into one of two categories: chat or reasoning. For chat tasks, the
model generates a set of evaluation rubrics, justifications for the rubrics, and evaluations tailored to
the specific question. For reasoning tasks, correctness is the most important and generally preferred
rubrics, so we directly let the model first solve the problem itself before evaluating and picking the
preferred response. This task perception enables the model to tailor its rollout strategy � applying
rubric-based evaluation for chat and correctness-first judgment for reasoning � resulting in more
aligned and effective reward signals. In addition, we explore how to directly adapt existing reasoning
models into reward models. Since these models have already undergone substantial reasoning-
focused distillation, we fine-tune them using RLVR without additional distillation stages. Based on
our training recipes, we produce RM-R1 models ranging from 7B to 32B.

Empirically, RM-R1 models consistently yield highly interpretable and coherent reasoning traces.
On average, RM-R1 achieves state-of-the-art performance on RewardBench [17], RM-Bench [21],

2

I feel like I�m not good enough for my job. No matter how hard I try, I keep making mistakes. Should I just quit?Instruct ModelModel with Long ReasoningI�m really sorry you�re feeling this way. It�s completely normal to doubt yourself, especially when you're striving to do well. Mistakes don�t mean failure�they�re often part of learning. Instead of quitting, maybe it�s worth re?ecting on whether you're being too hard on yourself. Would you consider talking to a mentor or supervisor about how you�re feeling? You might be surprised by their perspective and support.If you keep messing up, maybe it really isn�t the right job for you. Not everyone is cut out for everything. Quitting isn�t a bad thing�it can be the ?rst step toward ?nding something you're actually good at.Chosen Response (Supportive, honest, and thoughtful):Rejected Response (Well-formed but subtly toxic):<eval>The ?rst response validates the user�s emotions and encourages constructive self-re?ection, offering actionable and supportive guidance without judgment. The second response assumes the user�s failure and may reinforce negative beliefs, which is harmful in sensitive contexts.</eval><answer>Secondmessage.</answer><answer>The ?rst response.</answer><rubrics>1. Empathy & Emotional Validation II. Psychological Safety / Non-Harm III. Constructive, Actionable Guidance IV. Encouragement of Self-Ef?cacy</rubrics>Figure 2: Training pipeline of RM-R1. Starting from an instruct model (GenRM), RM-R1 training
involves two stages: Distillation and Reinforcement Learning (RL). In the Distillation stage, we
use high-quality synthesized data to bootstrap RM-R1�s reasoning ability. In the RL stage, RM-R1�s
reasoning ability for reward modeling is further strengthened. After distillation, a GenRM evolves
into a REASRM. RM-R1 further differentiates itself by being RL finetuned on preference data.

and RMB [43], outperforming 70B, 340B, GPT-4o, and Claude models by up to 4.9%. Beyond final
performance, we conduct extensive empirical analyses of RM-R1, including ablations of our training
recipes, studies of its scaling effects, comparisons with non-reasoning baselines, detailed case studies,
and training dynamics. In summary, our main contributions are as follows:

� We demonstrate that reasoning abilities are crucial for reward models, and propose to formulate

reward modeling as a reasoning process to enhance interpretability and accuracy.

� We design a training recipe based on reasoning-oriented distillation and RL that produces a set of

reward models � RM-R1 � that can outperform larger models by up to 4.9% on average.

� We present a systematic empirical study of different training recipes for REASRMS, providing

insights into the impact of diverse training strategies on the final reward model performance.

2 RM-R1

Figure 2 presents the overall training pipeline of RM-R1, which consists of two stages: reasoning
distillation and reinforcement learning. (1) Reasoning Distillation: Starting from an off-the-shelf
instruction-tuned model (e.g., Qwen-2.5-14B-Instruct), we further train the model using synthe-
sized high-quality reasoning traces. This stage equips RM-R1 with essential reasoning capabilities
required for effective reward modeling. (2) Reinforcement learning: While distillation is effective for
injecting reasoning patterns, distilled models often overfit to specific patterns in the training data,
limiting their generalization ability [9]. To overcome this limitation, we introduce a reinforcement
learning phase that further optimizes the model, resulting in the final version of RM-R1.

2.1 Task Definition

Given a preference dataset:

D = {(x(i), y(i)

a , y(i)

b , l(i))}N

i=1,

(1)

where x is a prompt, ya and yb are two different responses for x, and l ? {a, b} is the ground truth
label that indicates the preferred response. We define the generative reward modeling task as follows:

3

ScalarRMRM-R1GenRMChain-of-RubricsComplex CritiqueAnswer�Which response is correct/better?��Let�s verify step by step��Linear Function�<rubrics> R1, R2, R3 </rubrics>�JudgeModel TypeInference TaskReward Signal ReasRMTraining InputModel TypeInference InputMaximize Cumulative RewardRLQueryQueryResponseScoreTask/ObjectTraining OutputQueryQueryInference OutputGenRMMinimize NLLDistillationReasoning TraceQueryReasRMCritiqueAnswer�Let�s verify step by step��QueryAnswerGenRM�Which response is correct/better?�JudgeQueryAnswerRM-R1 TrainingRM-R1�s Structured Reasoning <eval>The ?rst response validates the user�s emotions�</eval><answer>The ?rst response.</answer><rubrics>1. Empathy & Emotional Validation.  II...  III� </rubrics>ScalarRMGenRMAfter TrainingAfter TrainingLet r? denote a generative reward model parameterized by ?. For each data sample, r? generates a
textual judgment j consisting of ordered tokens j = (j1, j2, . . . , jT ), modeled by:

r?(j|x, ya, yb) =

T
(cid:89)

t=1

r?(jt|x, ya, yb, j<t).

Note that j contains r?�s prediction of the preferred response �l ? j. The overall objective is:

max
r?

E

(x,ya,yb,l)?D,�l?r?(j|x,ya,yb)

(cid:104)

1(�l = l)

(cid:105)

.

2.2 Reasoning Distillation for Reward Modeling

(2)

(3)

For an instruction-tuned model (e.g., Qwen-2.5-14b-instruct [37]), it is quite intuitive to turn it
into a GenRM simply by prompting. However, without fine-tuning on reward modeling reasoning
traces, these models may struggle to conduct consistent judgments. To bootstrap its reasoning
potential, we start with training an instruction-tuned model with long reasoning traces synthesized
for reward modeling. Specifically, we sample M data samples from D and denote it as Dsub. Given a
data sample (x(i), y(i)
b , l(i)) ? Dsub, we ask an �oracle� model like o3 or claude-3-7-sonnet
to generate its structured reasoning trace r(i) justifying why y(i)
is chosen as the preferred response
of x(i). We then construct the reasoning trace ground truth:

a , y(i)

l

where ? denotes string concatenation. Given all the synthesized reasoning traces r(i), the final
distillation dataset is defined as:

y(i)
trace = r(i) ? l(i),

(4)

Ddistill = {(x(i), y(i)

trace)}M

i=1.

(5)

Formally, the objective of distillation is to adjust ? to maximize the likelihood of generating the
desired reasoning trace and picking the response y given the prompt x. We minimize the negative
log-likelihood (NLL) loss:

Ldistill(?) = ?

(cid:88)

(cid:88)

log r? (yt | x, y<t) ,

(6)

(x,y)?Ddistill

t?[|y|]

where y<t = (y1, y2, ..., yt?1) denotes the sequence of tokens preceding position t. More details of
generating high-quality reasoning chains are included in Appendix B.

2.3 RL Training

Although distillation is a proper way to turn a general generative model into a GenRM, it often suffers
from overfitting to certain patterns and constrains the model�s ability to generalize its reasoning
abilities for critical thinking [9, 31] , which is essential for reward modeling. To address this, we
propose to integrate RL as a more powerful learning paradigm to enhance the model�s ability to
conduct reasoning-based rewarding. Training a policy model using RL has been widely seen in
the preference optimization phase of LLM post-training [24], and it is quite natural to extend this
paradigm for training a REASRM. To be specific, we directly treat our reward model r?(j | x, ya, yb)
as if it is a policy model:

max
r?

(x,ya,yb,l)?D,�l?r?(j|x,ya,yb) [R(x, j)] ? ?DKL (r??rref ) ,
E

(7)

where rref is the reference reward model. In practice, we use the checkpoint before RL training as
rref , and that means rref could be an off-the-shelf LLM or the LLM obtained after the distillation
step in Section 2.2. R(x, j) is the reward function, and DKL is KL-divergence. The x denotes input
prompts drawn from the preference data D. The j indicates the text generated by the reward model,
which includes the reasoning trace and final judgement �l. In practice, we use Group Relative Policy
Optimization (GRPO) [28] to optimize the objective in Equation (7), the details of which can be find
in Appendix C.

4

Chain-of-Rubrics (CoR) Rollout for Instruct Models

Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client�s question displayed below.

First, classify the task into one of two categories: <type> Reasoning </type> or <type> Chat </type>.
- Use <type> Reasoning </type> for tasks that involve math, coding, or require domain knowledge, multi-step inference, logical deduction, or
combining information to reach a conclusion.
- Use <type> Chat </type> for tasks that involve open-ended or factual conversation, stylistic rewrites, safety questions, or general helpfulness
requests without deep reasoning.

If the task is Reasoning:
1. Solve the Client�s question yourself and present your final answer within <solution> ... </solution> tags.
2. Evaluate the two Chatbot responses based on correctness, completeness, and reasoning quality, referencing your own solution.
3. Include your evaluation inside <eval> ... </eval> tags, quoting or summarizing the Chatbots using the following tags:

- <quote_A> ... </quote_A> for direct quotes from Chatbot A
- <summary_A> ... </summary_A> for paraphrases of Chatbot A
- <quote_B> ... </quote_B> for direct quotes from Chatbot B
- <summary_B> ... </summary_B> for paraphrases of Chatbot B

4. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>

If the task is Chat:
1. Generate evaluation criteria (rubric) tailored to the Client�s question and context, enclosed in <rubric>...</rubric> tags.
2. Assign weights to each rubric item based on their relative importance.
3. Inside <rubric>, include a <justify>...</justify> section explaining why you chose those rubric criteria and weights.
4. Compare both Chatbot responses according to the rubric.
5. Provide your evaluation inside <eval>...</eval> tags, using <quote_A>, <summary_A>, <quote_B>, and <summary_B> as described above.
6. End with your final judgment in the format: <answer>[[A]]</answer> or <answer>[[B]]</answer>

Figure 3: The system prompt used for RM-R1 rollout.

2.3.1 Chain-of-Rubrics (CoR) Rollout

To facilitate the distilled models to proactively generate effective reasoning traces, we design a system
prompt as shown in Figure 3 during rollout. Intuitively, reward modeling for general domain (e.g.,
chat, safety, etc.) and reasoning domain (e.g., math, code, etc.) should focus on different angles.
For example, for the chat domain, we may care more about some aspects that can be expressed in
textual rubrics (e.g., be polite), yet for the reasoning domain, we usually care more about logical
coherence and answer correctness. Based on this intuition, we instruct r? to classify each preference
data sample {(x, yc, yr)} into one of the two <type>: Chat or Reasoning. For each <type>, we
prompt r? to carry out the behavior corresponding to that type step by step: For reasoning tasks, we
ask r? to solve x on its own. During the <eval> phase, r? compares yc and yr conditioned on its own
</solution> and selects an <answer>. Regarding the Chat type, we instead ask r? to think about and
justify the <rubric> for grading the chat quality (including safety).

2.3.2 Reward Design

Rule-based reward mechanisms have demonstrated strong empirical performance to facilitate rea-
soning [12]. In our training, we further simplify the reward formulation and merely focus on the
correctness-based component, in line with prior works [28, 18].

Formally, our reward is defined as follows:

R(x, j|ya, yb) =

(cid:40)
1
?1

if �l = l,
otherwise.

(8)

where �l is extracted from j, wrapped between the <answer> and </answer> tokens. We have also
tried adding the format reward to the overall reward, but found that the task performance does not
have a significant difference. The rationale behind only focusing on correctness is that the distilled
models have already learned to follow instructions and format their responses properly.

3 Experiments

3.1 Experimental Setup

We evaluate RM-R1 on three primary benchmarks: RewardBench [17], RM-Bench [21], and
RMB [43]. Our training set utilizes a cleaned subset of Skywork Reward Preference 80K [20], 8K

5

Table 1: The performance comparison between best-performing baselines. Bold numbers indicate
the best performance, Underlined numbers indicate the second best. The DeepSeek-GRM models
are not open-weighted, so we use the numbers on their tech report. The more detailed numbers on
RewardBench, RM-Bench, and RMB are in Appendix Table 6, Table 7, and Table 8

Models

ScalarRMs

SteerLM-RM-70B
Eurus-RM-7b
Internlm2-20b-reward
Skywork-Reward-Gemma-2-27B
Internlm2-7b-reward
ArmoRM-Llama3-8B-v0.1
Nemotron-4-340B-Reward
Skywork-Reward-Llama-3.1-8B
INF-ORM-Llama3.1-70B

GenRMs

Claude-3-5-sonnet-20240620
Llama3.1-70B-Instruct
Gemini-1.5-pro
Skywork-Critic-Llama-3.1-70B
GPT-4o-0806

ReasRMs

JudgeLRM
DeepSeek-PairRM-27B
DeepSeek-GRM-27B-RFT
DeepSeek-GRM-27B
Self-taught-evaluator-llama3.1-70B

Our Methods
RM-R1-DeepSeek-Distilled-Qwen-7B
RM-R1-Qwen-Instruct-7B
RM-R1-Qwen-Instruct-14B
RM-R1-DeepSeek-Distilled-Qwen-14B
RM-R1-Qwen-Instruct-32B
RM-R1-DeepSeek-Distilled-Qwen-32B

RewardBench RM-Bench RMB Average

88.8
82.8
90.2
93.8
87.6
90.4
92.0
92.5
95.1

84.2
84.0
88.2
93.3
86.7

75.2
87.1
84.5
86.0
90.2

80.1
85.2
88.2
88.9
91.4
90.9

52.5
65.9
68.3
67.3
67.1
67.7
69.5
70.1
70.9

61.0
65.5
75.2
71.9
72.5

64.7
�
�
�
71.4

72.4
70.2
76.1
81.5
79.1
83.9

58.2
68.3
62.9
60.2
67.1
64.6
69.9
69.3
70.5

70.6
68.9
56.5
65.5
73.8

53.1
58.2
67.0
69.0
67.0

55.1
66.4
69.2
68.5
73.0
69.8

66.5
72.3
73.6
73.8
73.9
74.2
77.1
77.5
78.8

71.9
72.8
73.3
76.9
77.7

64.3
�
�
�
76.2

69.2
73.9
77.8
79.6
81.2
81.5

examples from Code-Preference-Pairs, and the full Math-DPO-10K [16] dataset. For baselines,
we compare RM-R1 with RMs from three main categories: ScalarRMs, GenRMs, and REASRMS.
Further details on the benchmarks, dataset construction, and specific baseline models are provided in
Appendix D.

3.2 Main Results

Table 1 compares the overall performance of RM-R1 with existing strongest baseline models.
The more detailed numbers on RewardBench, RM-Bench, and RMB are in Table 6, Table 7,
and Table 8 in Appendix F. For the baselines, we reproduce the numbers if essential resources are
open-sourced (e.g., model checkpoints, system prompts). Otherwise, we use the numbers reported
in the corresponding tech report or benchmark leaderboard. For each benchmark, we select the
best-performing models in each category for brevity. Our key findings are summarized below:
State-of-the-Art Performance. On average, our RM-R1-DeepSeek-Distilled-Qwen-14B
model surpasses all previous leading Reward Models (RMs), including INF-ORM-Llama3.1-70B,
Nemotron-4-340B-Reward, and GPT-4o, while operating at a much smaller scale. Our 32B models,
RM-R1-Qwen-Instruct-32B and RM-R1-DeepSeek-Distilled-Qwen-32B, further extend this
lead by a notable margin. The success of RM-R1 is attributable to both our meticulously designed
training methodology and the effective scaling of our models, as extensively analyzed in Section 4.1
and Section 4.2. In particular, RM-R1 outperforms existing top-tier ScalarRMs. This highlights
the considerable potential of REASRMS, a category where prior GenRMs have exhibited subop-
timal performance and are generally not comparable to their scalar counterparts. In contrast to

6

our structured rollout and distillation with RLVR training strategy, prior critique-based methods
have relied heavily on rejection sampling and unstructured, self-generated chain-of-thought (CoT)
reasoning from instruct models [22, 35], limiting their reasoning capabilities and leading to inferior
performance compared to ScalarRMs. Simultaneously, our comprehensive evaluation indicates
that the top-performing scalar models on RewardBench do not consistently achieve state-of-the-art
(SOTA) performance; in fact, larger models frequently underperform smaller ones. This evaluation
underscores the need for a more comprehensive and systematic approach to RM assessment.

Effective Training towards Reasoning for Reward Modeling. Our specialized, reasoning-oriented
training pipeline delivers substantial performance gains. For instance, RM-R1-Qwen-Instruct-14B
consistently surpasses Self-taught-evaluator-llama-3.1-70B, a reasoning model five times
its size. The RM-R1 model series also demonstrates impressive results on RM-Bench, exceed-
ing the top-performing baseline by up to 8.7%. On this most reasoning-intensive benchmark,
RM-R1-DeepSeek-Distilled-Qwen-32B establishes a new state-of-the-art. It achieves 91.8% ac-
curacy in math and 74.1% in code, outperforming the previous best models (73% in math and 63%
in code) by significant margins. Furthermore, it also records the strongest reasoning performance
among our released models on RewardBench. Despite its performance, our Instruct-based models
are remarkably data-efficient, reaching competitive performance using only 8.7K examples for distil-
lation�compared to the 800K examples used in training DeepSeek-Distilled [12]. Overall, our
study underscores the significant potential of directly adapting large reasoning models into highly
effective reward models.

4 Analysis

In this section, we present a series of empirical analyses to understand the key ingredients for training
effective reasoning reward models. Our analysis spans scaling effects, design decisions, reasoning
ablations, and a case study. We also present additional analysis on training dynamics in Appendix G.2.

4.1 Training Recipes

We first investigate the key ingredients underlying the successful training of RM-R1. Through a
series of ablation studies, we examine our design choices to identify effective strategies for training
high-quality reasoning reward models. We compare the following settings: Cold Start RL, Cold
Start RL + Rubrics, Cold Start RL + Rubrics + Query Categorization (QC), and Distilled + RL
+ Rubrics + QC (i.e., RM-R1). The details of these settings are in Appendix G.1.

Table 2: Ablation study of the design choices for Reasoning Training on RewardBench.
Safety Reasoning Average
Method

Chat Chat Hard

95.8
Instruct (Original)
Instruct + Cold Start RL
92.5
Instruct + Cold Start RL + Rubrics
93.0
Instruct + Cold Start RL + Rubrics + QC 92.3
RM-R1
95.3

74.3
81.5
82.5
82.6
83.1

86.8
89.7
90.8
91.6
91.9

86.3
94.4
94.2
96.3
95.2

85.8
89.5
90.1
90.8
91.4

the results of the ablation studies described above, using the
In Table 2, we present
Qwen-2.5-Instruct-32B model as the Instruct (Original) model. Several key conclusions emerge:

� RL training alone is insufficient. While Cold Start RL slightly improves performance on hard

chat and reasoning tasks, it fails to close the gap to fully optimized models.

� CoR prompting optimizes RM rollout and boosts reasoning performance. Instructing RM-R1
to self-generate chat rubrics or problem solutions before judgment helps overall performance,
especially for chat and safety tasks. Incorporating explicit query categorization into the prompt
notably improves reasoning performance, suggesting that clearer task guidance benefits learning.

� Distillation further enhances performance across all axes. Seeding the model with high-quality
reasoning traces before RL yields the strongest results, with improvements observed on both hard
tasks and safety-sensitive tasks.

7

Method

RewardBench RM-Bench RMB Avg.

Train on Full Data

Instruct + SFT
Instruct + Distilled + SFT
RM-R1 *

Train on 9k (Distillation) Data

Instruct + SFT
Instruct + Distilled *

90.9
91.2
91.4

88.8
89.0

75.4
76.7
79.1

74.8
76.3

65.9
65.4
73.0

77.4
77.8
81.2

66.9
72.0

76.6
79.2

Table 3: Comparison of reasoning-based train-
ing versus SFT across benchmarks. * indicates
reasoning-based methods. Reasoning training con-
sistently yields better performance.

(a) Model Size

(b) Inference Compute

Figure 4: Scaling effect of RM-R1.
(a)
Larger models benefit more from reasoning
training. (b) Longer reasoning chains improve
RM performance.

? Takeaway 1:

Directly replicating reinforcement learning recipes from mathematical tasks is insufficient for
training strong reasoning reward models. Explicit query categorization and targeted distilla-
tion of high-quality reasoning traces are both crucial for achieving robust and generalizable
improvements.

4.2 Scaling Effects

We then investigate how model performance varies with scale, considering both model size and
inference-time compute. In some cases � such as ScalarRMs from InternLM2 [6] and Skywork [20]
� the smaller models (7B/8B) outperforms the larger ones (20B/27B), showing no advantage of scaling.
In this subsection, we show that this trend does not hold for RM-R1, where scaling brings clear and
substantial improvements.

4.2.1 Model Sizes

We first analyze the impact of model scale. Our study is based on the Qwen-2.5-Instruct model
family at three sizes: 7B, 14B, and 32B. We evaluate performance improvements resulting from
our training procedure described in Section 2, with results averaged across three key benchmarks:
RewardBench, RM-Bench, and RMB.

For each model size, we compare the original and post-training performance. Figure 4a plots the
relative improvement (%) with respect to model size. Observing an approximately linear trend, we fit
a linear regression model and extrapolate to hypothetical scales of 3B and 72B, shown using faint
markers and dashed extensions. The results strongly support a scaling law for reasoning reward
models: larger models not only result in an absolute better final performance but also consistently
yield greater performance gains. This aligns with the intuition that our training effectively leverages
the superior reasoning capabilities of larger models.

4.2.2 Inference-time Computation

Next, we examine how model performance varies with different compute budgets measured in number
of tokens allowed during inference. Since this is particularly relevant to reasoning-focused models,
we fix our base model to DeepSeek-R1-Distill-Qwen-14B. We evaluate average performance
across the three key benchmarks using a wide range of inference-time compute budgets: 512, 1024,
2048, 4096, and 8192 tokens.

To ensure a fair comparison, we match the training rollout budget to the inference budget in each
setting (i.e., we allow a maximum of k tokens during training for a compute budget of k at inference).
All models are trained using GRPO with identical datasets and hyperparameter configurations.
Figure 4b shows the relationship between compute budget and performance. We observe a clear
improvement trend as the inference budget increases. This highlights the benefits of long reasoning
chains in reward modeling.

8

37143272Model Size (B)45678PerformanceImprovement (%)Model Size Scaling5121 k2 k4 k8 kCompute Budget7677787980Performance (%)Inference Compute Scaling? Takeaway 2:

Scaling improves reward model performance: we observe a near-linear trend with both model
size and inference-time compute. Larger models consistently benefit more from our reasoning-
based training pipeline, and longer reasoning chains become increasingly effective under higher
compute budgets.

4.3 Effectiveness of Reasoning Training

We now analyze the impact of reasoning-based training. Here, we demonstrate that reasoning-based
training can outperform answer-only approaches. We consider the following settings:

Instruct + SFT. This approach fine-tunes the instruct model directly toward producing the correct
final answer using the full dataset, without providing any intermediate reasoning chains.

Instruct + Distilled + SFT. This approach applies SFT (with respect to the answer directly) after the
distillation stage, serving as a direct comparison point with RM-R1 trained with RL.

Instruct + RM-R1 (Distilled + RL). This is the full approach proposed in this paper, following the
procedure detailed in Section 2.

Instruct + Distilled. This setting uses the model checkpoint immediately after the distillation stage,
before any RL fine-tuning.

In summary, methods with �+ RM-R1� or �+ Distilled� represent reasoning-based approaches, while
the remaining methods are purely non-reasoning-based approaches. In Table 3, we report the results
measured across the three benchmarks. The findings clearly demonstrate that reasoning training
significantly benefits reward model performance. Under fair comparisons (i.e., training on exactly the
same amount of data), reasoning-based models consistently outperform their SFT-only counterparts.
In particular, even high-quality distillation alone, applied to a small subset of the data, provides
notable gains, highlighting the value of incorporating structured intermediate reasoning.

? Takeaway 3:

Reasoning training substantially improves reward modeling. It not only enables better generaliza-
tion across tasks but also provides consistent gains even under limited data scenarios compared
to direct-answer SFT approaches.

4.4 Case Study

To gain deeper insights into RM-R1, we further conduct a case study using Qwen-14B-Instruct,
comparing its behavior with a cold-start RL model. The results, summarized in Table 4, reveal the
following key observations:

Transparent Judging Process. The reasoning traces generated by RM-R1 are highly interpretable
and coherent, reflecting the model�s perception of human preferences. It explicitly articulates why
certain responses are better, providing transparency into its evaluation process.

High-Quality, Question-Dependent Rubrics. RM-R1 accurately understands the question and the
context of comparison, correctly prioritizing �accuracy� as the most critical rubric for medical-related
questions. In contrast, the cold-start RL model often overlooks the most important factors and instead
emphasizes superficial or broadly defined features (e.g., relevance) that are less discriminative. The
ability to generate high-quality, question-specific rubrics stems from the knowledge acquired during
the distillation stage.

Faithful Adherence to Rubrics and Content-Based Judgement. RM-R1 grounds its evaluation
in the actual content of the chatbot responses. For example, it correctly identifies inaccuracies in
Chatbot A�s response based on factual content rather than surface presentation. Furthermore, it
systematically evaluates different aspects of the rubric, leading to a structured, interpretable, and
verifiable judging process.

9

Table 4: Case Study of RM-R1(succinct version). Complete Chatbots and model responses are
shown in Table 9, 10, and 11. Cold-Start RL evaluation tends to focus on superficial features (e.g.,
number of suggestions, relevance), whereas RM-R1 produces high-quality rubrics (e.g., accuracy for
factual questions), faithfully follows them, and evaluates responses based on their actual content.
Question: symptoms of sickle-cell disease

Chatbot A:
1. Painful red or yellow skin lesions; 2. Frequent infections; 3. Fatigue and weakness; 4. Anemia; 5. Slower growth and development; 6.
Jaundice; 7. Swelling of hands and feet; 8. Vision loss; 9. Tiredness and weakness after physical activity; 10. Breathlessness; 11. Fever, chills,
and night sweats.

Chatbot B
Sickle cell disease is a genetic disorder that affects the shape of red blood cells. It is most common in African Americans, but can also affect
people from other ethnic groups. The symptoms of sickle cell disease can vary widely from person to person, but some common symptoms
include:

1. Fatigue: ...; 2. Pain: ...; 3. Fever: ...; 4. Anemia: ...; 5. Delayed growth and development: ...; 6. Infections: ...; 7. Pigmentation: ...

It�s important to note that not everyone with sickle cell disease will experience all of these symptoms, ... Therefore, it�s important to consult with
a healthcare provider if you suspect you or someone you know may have sickle cell disease.

Cold-start RL evaluation ?
<type>Chat</type>
<rubric> Relevance 40 % � Comprehensiveness 30 % � Clarity/organisation 30 % </rubric>
<eval> Chatbot A provided a list of symptoms that are directly relevant to sickle cell disease. The response is comprehensive and includes
a wide range of symptoms commonly associated with the condition. The symptoms are listed in a clear and organized manner, making it easy for
the client to understand the information. Chatbot B�s response is less organized than Chatbot A�s response, and the explanation of sickle cell
disease may be unnecessary for the client�s question.</eval>
<answer>[[A]]</answer>

RM-R1 evaluation ?
<type>Chat</type>
<rubric> 1 Accuracy 40 % � 2 Comprehensiveness 30 % � 3 Clarity 20 % � 4 Helpfulness 10 % <justify> Accuracy is weighted highest
because providing correct information about medical conditions is crucial. Comprehensiveness is important because ...</justify> </rubric>
<eval> Accuracy of information: Chatbot A provides a list of symptoms without explanation, but some listed items (e.g. �painful red or yellow
skin lesions,� �vision loss�) are not typical of sickle-cell disease. Chatbot B offers more accurate information, explaining each symptom�s
nature and severity. ... Comprehensiveness:...Clarity and organization:...Helpfulness:... </eval>
<answer>[[B]]</answer>

5 Related Work

Reward Models (RMs). Early RMs were typically outcome-focused: trained to predict human
preference rankings for complete outputs [42]. Recent advances have looked at providing process
supervision, which rewards or evaluates the steps of a model�s reasoning rather than only the final
answer. A series of works propose to train process reward models that judge the correctness of
intermediate reasoning steps [19, 10, 27]. A limitation of many PRMs is their heavy reliance on
curated step-level human labels or specific schemas, and they often remain domain-specific. Zhang
et al. [40] propose Generative Verifiers, framing reward modeling as a next-token prediction task.
This allows the reward model to leverage chain-of-thought and even use majority voting over multiple
sampled rationales to make more reliable judgments. DeepSeek-GRM [22] and JudgeLRM [7] have
studied using reasoning models as generative reward models, which are the most relevant research to
ours. However, their main focus is on the effect of scaling inference-time computation on reward
modeling. On the contrary, our work is the first to provide a systematic empirical comparison of
different reward model training paradigms, shedding light on when and why a distilled and RL-trained
reward model like RM-R1 has advantages over the conventional approaches.

Reinforcement Learning from Human Feedback (RLHF). Early works [8] first demonstrated
that reinforcement learning could optimize policies using a reward model trained from human
pairwise preferences. Subsequent studies applied RLHF to large-scale language models using policy
optimization algorithms such as PPO [26]. For example, Ziegler et al. [45] fine-tuned GPT-2 via PPO
on human preference rewards, and Stiennon et al. [32] showed that RLHF could significantly improve
the quality of summarization by optimizing against a learned preference model. More recently,
Ouyang et al. [24] used a similar PPO-based pipeline to train InstructGPT, establishing the modern
RLHF paradigm for instruction-following models. Recently, Verifiable supervision techniques have
also emerged: DeepSeek-R1 [12] uses a form of self-verification during RLHF to reward correct
reasoning steps, rather than only final-answer quality. This method incentivizes policies to produce

10

outputs that can be verified for correctness, bridging the gap between pure preference-based feedback
and ground-truth signals. However, even with such innovations, most RLHF implementations still
treat reward modeling and reasoning as separate stages.

6 Conclusion and Future Work

In this paper, we revisited reward modeling through the lens of reasoning. We introduced RM-R1,
a family of REASRMS that effectively generate explicit chains of rubrics and rationales, and scale
with both model size and inference compute. Across three public benchmarks, RM-R1 matched or
surpassed commercial and open-source RMs while producing more interpretable judgments. Ablation
investigations reveal that (1) task-type categorization, (2) bootstrapping from high-quality reasoning
traces, and (3) RL fine-tuning are all indispensable. Qualitative analyses further showed that RM-R1
learns to prioritize high-impact rubrics, faithfully follow its own criteria and justify coherently. Future
work includes active preference collection, where REASRMS use active learning to query human
preference only when the current rubric set is insufficient for a new preference sample. Finally, it
would be natural to extend our study to multimodal/agentic reward modeling scenarios.

Acknowledgments and Disclosure of Funding

This research is based upon work supported DARPA ITM Program No. FA8650-23-C-7316, and
the AI Research Institutes program by National Science Foundation and the Institute of Education
Sciences, U.S. Department of Education through Award # 2229873 - AI Institute for Transforming
Education for Children with Speech and Language Processing Challenges. The views and conclusions
contained herein are those of the authors and should not be interpreted as necessarily representing
the official policies, either expressed or implied, of the U.S. Government. The U.S. Government
is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any
copyright annotation therein.

References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, et al. Gpt-4 technical report. arXiv
preprint arXiv:2303.08774, 2023.

[2] Bo Adler, Niket Agarwal, Ashwath Aithal, Dong H Anh, Pallab Bhattacharya, Annika Brundyn,
Jared Casper, Bryan Catanzaro, Sharon Clay, et al. Nemotron-4 340b technical report. arXiv
preprint arXiv:2406.11704, 2024.

[3] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 1:1,

2024.

[4] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones,
Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai:
Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

[5] Chris L Baker, Rebecca Saxe, and Joshua B Tenenbaum. Action understanding as inverse

planning. Cognition, 113(3):329�349, 2009.

[6] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui
Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297,
2024.

[7] Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, and Bingsheng
He. Judgelrm: Large reasoning models as a judge. arXiv preprint arXiv:2504.00050, 2025.

[8] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep
reinforcement learning from human preferences. Advances in neural information processing
systems, 30, 2017.

11

[9] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans,
Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of
foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.

[10] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan,
Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv
preprint arXiv:2502.01456, 2025.

[11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783, 2024.

[12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[13] Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. Openrlhf: An
easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143,
2024.

[14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv
preprint arXiv:2410.21276, 2024.

[15] Hamish Ivison, Yizhong Wang, Jiacheng Liu, Zeqiu Wu, Valentina Pyatkin, Nathan Lambert,
Noah A Smith, Yejin Choi, and Hannaneh Hajishirzi. Unpacking dpo and ppo: Disentangling
best practices for learning from preference feedback. arXiv preprint arXiv:2406.09279, 2024.

[16] Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-
dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint
arXiv:2406.18629, 2024.

[17] Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi
Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, et al. Rewardbench: Evaluating
reward models for language modeling. arXiv preprint arXiv:2403.13787, 2024.

[18] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Torl: Scaling tool-integrated rl. arXiv preprint

arXiv:2503.23383, 2025.

[19] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee,
Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let�s verify step by step. In The
Twelfth International Conference on Learning Representations, 2023.

[20] Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan,
Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv
preprint arXiv:2410.18451, 2024.

[21] Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. Rm-bench: Benchmarking
reward models of language models with subtlety and style. arXiv preprint arXiv:2410.16184,
2024.

[22] Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu.
Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025.

[23] Gilberto Montibeller and Alberto Franco. Multi-criteria decision analysis for strategic decision

making. In Handbook of multicriteria analysis, pages 25�48. Springer, 2010.

[24] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to
follow instructions with human feedback. Advances in neural information processing systems,
35:27730�27744, 2022.

12

[25] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-
baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al.
Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv
preprint arXiv:2403.05530, 2024.

[26] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal

policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[27] Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal,
Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated
process verifiers for llm reasoning. arXiv preprint arXiv:2410.08146, 2024.

[28] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[29] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua
Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv
preprint arXiv: 2409.19256, 2024.

[30] Tu Shiwen, Zhao Liang, Chris Yuhao Liu, Liang Zeng, and Yang Liu. Skywork critic model se-
ries. https://huggingface.co/Skywork, September 2024. URL https://huggingface.
co/Skywork.

[31] Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A Alemi, and Andrew G Wilson.
Does knowledge distillation really work? Advances in neural information processing systems,
34:6906�6919, 2021.

[32] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec
Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback.
Advances in Neural Information Processing Systems, 33:3008�3021, 2020.

[33] Nicole Van Hoeck, Patrick D Watson, and Aron K Barbey. Cognitive neuroscience of human

counterfactual reasoning. Frontiers in human neuroscience, 9:420, 2015.

[34] Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable prefer-
ences via multi-objective reward modeling and mixture-of-experts. In Yaser Al-Onaizan, Mohit
Bansal, and Yun-Nung Chen, editors, Findings of the Association for Computational Linguistics:
EMNLP 2024, pages 10582�10592, Miami, Florida, USA, November 2024. Association for
Computational Linguistics. URL https://aclanthology.org/2024.findings-emnlp.
620.

[35] Tianlu Wang, Ilia Kulikov, Olga Golovneva, Ping Yu, Weizhe Yuan, Jane Dwivedi-Yu,
Richard Yuanzhe Pang, Maryam Fazel-Zarandi, Jason Weston, and Xian Li. Self-taught
evaluators. arXiv preprint arXiv:2408.02666, 2024.

[36] Zhilin Wang, Yi Dong, Jiaqi Zeng, Virginia Adams, Makesh Narsimhan Sreedhar, Daniel Egert,
Olivier Delalleau, Jane Scowcroft, Neel Kant, Aidan Swope, and Oleksii Kuchaiev. HelpSteer:
Multi-attribute helpfulness dataset for SteerLM. In Kevin Duh, Helena Gomez, and Steven
Bethard, editors, Proceedings of the 2024 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies (Volume 1: Long
Papers), pages 3371�3384, Mexico City, Mexico, June 2024. Association for Computational
Linguistics. URL https://aclanthology.org/2024.naacl-long.185.

[37] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint
arXiv:2412.15115, 2024.

[38] Yue Yu, Zhengxing Chen, Aston Zhang, Liang Tan, Chenguang Zhu, Richard Yuanzhe Pang,
Yundi Qian, Xuewei Wang, Suchin Gururangan, Chao Zhang, et al. Self-generated critiques
boost reward modeling for language models. arXiv preprint arXiv:2411.16646, 2024.

13

[39] Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin
Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, and
Maosong Sun. Advancing llm reasoning generalists with preference trees, 2024.

[40] Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh
Agarwal. Generative verifiers: Reward modeling as next-token prediction. arXiv preprint
arXiv:2408.15240, 2024.

[41] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and
chatbot arena. Advances in Neural Information Processing Systems, 36:46595�46623, 2023.

[42] Jialun Zhong, Wei Shen, Yanzeng Li, Songyang Gao, Hua Lu, Yicheng Chen, Yang Zhang,
Wei Zhou, Jinjie Gu, and Lei Zou. A comprehensive survey of reward models: Taxonomy,
applications, challenges, and future. arXiv preprint arXiv:2504.12328, 2025.

[43] Enyu Zhou, Guodong Zheng, Binghai Wang, Zhiheng Xi, Shihan Dou, Rong Bao, Wei Shen,
Limao Xiong, Jessica Fan, Yurong Mou, et al. Rmb: Comprehensively benchmarking reward
models in llm alignment. arXiv preprint arXiv:2410.09893, 2024.

[44] Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, and Jiantao Jiao. Starling-7b: Improving

llm helpfulness & harmlessness with rlaif, November 2023.

[45] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei,
Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences.
arXiv preprint arXiv:1909.08593, 2019.

14

Contents

A User Prompt for DeepSeek-Distilled Reasoning Models

B Details of Reasoning Chain Generation

C Group Relative Policy Optimization (GRPO)

D Experiment Setups

D.1 Benchmarks .

.

.

.

.

D.2 Preference Datasets .

D.3 Baselines

.

.

.

.

.

.

.

.

.

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

E Implementation Details

F Full Experiment Result

G Supplementary Information for Section 4

G.1 Ablation Settings .

.

G.2 Training Dynamics .

.

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

16

16

16

16

16

17

17

18

18

19

19

21

15

A User Prompt for DeepSeek-Distilled Reasoning Models

Large reasoning models such as DeepSeek-R1-distilled models [12] do not have a system prompt,
so we show the user prompt for rollouts in Figure 5.

Chain-of-Rubrics (CoR) Rollout for Reasoning Models

Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the
Client question displayed below.

... [Pairwise Input Content] ...

Output your final verdict at last by strictly following this format:
A is better, or �<answer>[[B]]</answer>� if Chatbot B is better.

�<answer>[[A]]</answer>� if Chatbot

Figure 5: The user prompt used for RM-R1 rollout (for reasoning models).

B Details of Reasoning Chain Generation

We now expand on the details of generating high-quality reasoning chains. We first use the same
prompt to query Claude-3.7-Sonnet, generating initial reasoning traces. However, approximately
25% of these traces are incorrect, primarily on harder chat tasks. To correct these cases, we pass the
original prompt, the incorrect trace, and the correct final answer to OpenAI-O3, which then generates
a corrected reasoning trace aligned with the right answer.

This two-stage process yields a high-quality distillation set. We deliberately choose the order�first
Claude, then O3�based on qualitative observations: Claude excels at solving easier tasks and
maintaining attention to safety considerations, whereas O3 performs better on harder tasks but tends
to overemphasize helpfulness at the expense of safety. We select approximately 12% of the training
data (slightly fewer than 9K examples) for distillation. This is then followed by RL training.

C Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO) [28] is a variant of Proximal Policy Optimization
(PPO) [26], which obviates the need for additional value function approximation, and uses the
average reward of multiple sampled outputs produced in response to the same prompt as the baseline.
More specifically, for each prompt x, GRPO samples a group of outputs {y1, y2, � � � , yG} from the
old policy ??old and then optimizes the policy model by maximizing the following objective:

JGRPO(?) = E

x?D, {ji}G

i=1?r?old (j|x)

(cid:20) 1
G

G
(cid:88)

i=1

1
|ji|

|ji|
(cid:88)

(cid:110)

t=1

min

(cid:16) r?(ji,t | x, ji,<t)
r?old(ji,t | x, ji,<t)

�Ai,t,

clip(cid:0) r?(ji,t | x, ji,<t)
r?old (ji,t | x, ji,<t)

, 1 ? ?, 1 + ?(cid:1) �Ai,t

(cid:17)

? ? DKL [r?(� | x) ? ?ref(� | x)]

(9)

(cid:111)(cid:21)
,

where ? is a hyperparameter balancing the task specific loss and the KL-divergence. Specifically, �Ai
is computed using the rewards of a group of responses within each group {r1, r2, . . . , rG}, and is
given by the following equation:

�Ai =

ri ? mean({r1, r2, � � � , rG})
std({r1, r2, � � � , rG})

.

(10)

D Experiment Setups

D.1 Benchmarks

In this paper, we consider the following three benchmarks:

16

RewardBench [17]: RewardBench is one of the first endeavors towards benchmarking reward models
through prompt-chosen-rejected trios, covering four categories: chat, chat-hard, reasoning, and safety,
with 358, 456, 740, and 1431 samples, respectively.

RM-Bench [21]: Building on RewardBench, RM-Bench evaluates reward models for their sensitivity
to subtle content differences and robustness against style biases. It includes four categories: Chat,
Safety, Math, and Code, with 129, 441, 529, and 228 samples, respectively. Each sample contains
three prompts of varying difficulty. RM-Bench is the most reasoning-intensive benchmark among
those we consider.

RMB [43]: Compared with RewardBench and RM-Bench, RMB offers a more comprehensive
evaluation of helpfulness and harmlessness. It includes over 49 real-world scenarios and supports
both pairwise and Best-of-N (BoN) evaluation formats. RMB comprises 25,845 instances in total�37
scenarios under the helpfulness alignment objective and 12 under harmlessness.

D.2 Preference Datasets

We consider the following datasets for training:

Skywork Reward Preference 80K [20] is a high-quality collection of pairwise preference data drawn
from a variety of domains, including chat, safety, mathematics, and code. It employs an advanced data
filtering technique to ensure preference reliability across tasks. However, we identify a notable issue
with this dataset: all samples from the magpie_ultra source exhibit a strong spurious correlation,
where rejected responses consistently contain the token �<im_start>,� while accepted responses
do not. Additionally, responses from this source show a systematic bias�accepted responses are
typically single-turn, while rejected responses are multi-turn. This problematic subset constitutes
approximately 30% of the Skywork dataset and primarily covers mathematics and code domains. To
avoid introducing spurious correlations into training, we exclude all magpie_ultra data and retain
only the cleaned subset for our experiments.

Code-Preference-Pairs is a high-quality coding preference dataset. It is constructed by prompting a
model with original code, introducing deliberate bugs, and manipulating examples (e.g., swapping
broken and corrected versions, removing error comments) to generate fine-grained preference pairs.
We subsample 8K examples from this dataset for use in our experiments.

Math-DPO-10K [16] is a high-quality stepwise preference dataset focused on mathematical reason-
ing. We use the full dataset in our experiments.

A global statistics of our training dataset is summarized in Table 5.

Table 5: Global Statistics of our Training Dataset. * indicates the source is from Skywork-Reward-
Preference-80K-v0.2.

Source

Size

Domain

magpile_pro_llama3.1*
offset_bias*
helpsteer2*
wildguard*
magpile_pro*
Code-Preference-Pairs
Math-DPO-10K

29682 Reasoning
Chat (length bias)
8504
Chat
7221
Safety
6709
Chat
2030
8000
Reasoning
10000 Reasoning

D.3 Baselines

We compare RM-R1 with RMs from three categories:

ScalarRMs. ScalarRMs produce a score for model response directly, predicting preference through
single numeric scores without explicit reasoning traces. This category includes models such as
Eurus-RM [39], Internlm2 [6] SteerLM-RM [36], Nemotron-RM [2], Tulu-v2.5 [15], Starling-
RM [44], ArmoRM [34], Skywork-RM [20], etc. While these models often achieve strong results on

17

well-defined benchmarks, they generally lack interpretability and struggle to capture fine-grained
reasoning.

GenRMs. Generative reward models (GenRMs) offer more expressive feedback by producing free-
form textual judgments, typically without further training. This includes the widely used LLM-as-a-
Judge setup [41], where pretrained language models are prompted to explain and evaluate responses.
We also categorize under GenRMs models that directly generate output answers without intermediate
reasoning steps. Representative examples include LLaMA [11], Qwen [37], Claude [3], GPT-4o [1,
14], Gemini 1.5 Pro [25], and Skywork-Critic [30]. By leveraging LLMs� generative capabilities,
these models enhance interpretability through natural language rationales and explanations.

REASRMS. Reasoning-enhanced reward models (REASRMS) explicitly incorporate reasoning
processes before their final judgments, often trained through critiques or chain-of-thought strate-
gies. Notable examples are JudgeLRM [7], Critique-RM [38], DeepSeek-GRM [22], Self-taught
Evaluators [35] and our proposed RM-R1 models. These models excel in tasks demanding rigorous
reasoning, safety evaluations, and nuanced preference judgments due to their grounding in structured
critical thinking.

E Implementation Details

Our training framework is based on VERL [29] and OpenRLHF [13]. For Instruct models, we use
8.7k data for distillation and 64k for RLVR. For Deepseek-Distilled models, we use the full data
for RLVR.

Distillation Stage. We use the SFTTrainer from OpenRLHF with a fixed batch size of 128 and
a micro-batch size of 1, training for a single epoch. To optimize GPU memory usage, we enable
gradient checkpointing, FlashAttention, and Adam offloading. The learning rates are set based on the
model size: 5e?6, 3e?6, and 2e?6 for models of size 7B, 14B, and 32B, respectively.

RLVR Stage. We use the VERL framework for all GRPO training. The training batch size is fixed
at 1024, with a mini-batch size of 128. We adopt Fully Sharded Data Parallel (FSDP) to improve
memory efficiency. For rollout generation, we use vLLM with tensor parallelism size 4 and GPU
memory utilization capped at 0.4. Sampling follows default parameters (temperature = 1.0, top-p =
1.0). KL regularization is applied with a coefficient of 1e?3 and a clip ratio of 0.2. Each prompt is
sampled with 7 candidate responses.

The maximum input sequence length is 4,096 tokens, and the maximum response length is 8,192
tokens. Learning rates are set separately for the two model variants:

� Instruct models: 1e?6, 7e?7, and 5e?7 for 7B, 14B, and 32B models, respectively.
� Reasoning models: 1e?6, 1e?6, and 8e?7 for 7B, 14B, and 32B models, respectively.

We train the 7B, 14B, and 32B models on 1, 2, and 4 nodes, respectively, each equipped with 8
H100 GPUs.

F Full Experiment Result

In this section, we provide the full experiment results and a more comprehensive coverage of existing
baselines. The results of RewardBench, RM-Bench, and RMB are provided in Table 6, Table 7, Ta-
ble 8, respectively.

18

Table 6: Results of our proposed method and baselines on the RewardBench. Bold numbers indicate
the best performance, Underlined numbers indicate the second best. ? indicates potential data
contamination.

Models

ScalarRMs

Eurus-RM-7b
Internlm2-7b-reward
SteerLM-RM 70B
Cohere-0514
Internlm2-20b-reward
ArmoRM-Llama3-8B-v0.1
Nemotron-4-340B-Reward
Skywork-Reward-Llama-3.1-8B?
Skywork-Reward-Gemma-2-27B?
INF-ORM-Llama3.1-70B

GenRMs

Llama3.1-8B-Instruct
Prometheus-8*7B-v2
Llama3.1-70B-Instruct
Llama3.1-405B-Instruct
Claude-3-5-sonnet-20240620
GPT-4o-0806
Gemini-1.5-pro
SFR-LLaMa-3.1-70B-Judge-r
Skywork-Critic-Llama-3.1-70B?
REASRMS

JudgeLRM
SynRM
RM-R1-DeepSeek-Distilled-Qwen-7B
CLoud
DeepSeek-GRM-16B
DeepSeek-GRM-27B-RFT
RM-R1-Qwen-Instruct-7B
DeepSeek-GRM-27B
DeepSeek-PairRM-27B
RM-R1-Qwen-Instruct-14B
RM-R1-DeepSeek-Distilled-Qwen-14B
Self-taught-evaluator-llama3.1-70B
RM-R1-DeepSeek-Distilled-Qwen-32B
RM-R1-Qwen-Instruct-32B

Chat Chat_Hard

Safety Reasoning Overall

98.0
99.2
91.3
96.4
98.9
96.9
95.8
95.8
95.8
96.6

85.5
93.0
97.2
97.2
96.4
96.1
92.3
96.9
96.6

92.9
38.0
88.9
97.0
90.8
94.7
94.1
94.1
95.5
93.6
91.3
96.9
95.3
95.3

65.6
69.5
80.3
71.3
76.5
76.8
87.1
87.3
91.4
91.0

48.5
47.1
70.2
74.6
74.0
76.1
80.6
84.8
87.9

56.4
82.5
66.2
58.0
74.3
77.2
74.6
78.3
86.8
80.5
79.4
85.1
80.3
83.1

81.4
87.2
92.8
92.3
89.5
90.5
91.5
90.8
91.9
93.6

75.6
80.5
82.8
77.6
81.6
86.6
87.9
91.6
93.1

78.2
74.1
78.4
84.0
84.7
87.0
85.2
88.0
52.3
86.9
89.3
89.6
91.1
91.9

86.3
94.5
90.6
97.7
95.8
97.3
93.6
96.2
96.1
99.1

72.1
77.4
86.0
87.1
84.7
88.1
92.0
97.6
95.5

73.6
87.1
87.0
92.0
81.8
79.2
86.7
83.8
92.0
92.0
95.5
88.4
96.8
95.2

82.8
87.6
88.8
89.4
90.2
90.4
92.0
92.5
93.8
95.1

70.4
74.5
84.0
84.1
84.2
86.7
88.2
92.7
93.3

75.2
70.4
80.1
82.8
82.9
84.5
85.2
86.0
87.1
88.2
88.9
90.0
90.9
91.4

G Supplementary Information for Section 4

G.1 Ablation Settings

Cold Start RL. This approach generally involves pure RL, with rule-based rewards centered on
answer correctness and format compliance. Such strategies have achieved notable success in advanced
mathematical problem solving [28].

In this setting, we replicate this conventional training setup. Specifically, we use a combination of a
format reward and an answer reward:

Rformat =

(cid:26)1
0

if format matches,
otherwise,

and Ranswer =

(cid:26)1
0

if answer matches,
otherwise.

(11)

The total reward is the sum R = Ranswer + Rformat. We use the prompt template shown in Figure 7, a
version without any guidance on structured reasoning.

Cold Start RL + Rubrics. To examine the influence of structured reasoning in final model perfor-
mance, compared with the last setting, we use the prompt template shown in Figure 6. Compared with
the last setting, the model is prompted to generate rubrics and evaluate accordingly. However, com-
pared with the final system prompt of RM-R1 Figure 3, all input prompts are treated uniformly�that
is, chat and reasoning tasks are not distinguished.

Cold Start RL + Rubrics + Query Categorization (QC). This setting largely follows the previous
one, with a key modification: prompting the LM to first categorize the task into reasoning or chat,

19

Table 7: The full results of tested reward models on RM-Bench. Chat, Math, Code, Safety show the
model�s Average Accuracy on each domain. Easy, Normal, Hard show the model�s Accuracy on each
difficulty level across all domains. Bold numbers indicate the best performance, Underlined numbers
indicate the second best.

Models

ScalarRMs

Chat Math Code Safety Easy Normal Hard Avg

56.4
steerlm-70b
58.2
tulu-v2.5-70b-preference-mix-rm
Mistral-7B-instruct-Unified-Feedback 56.5
57.4
RM-Mistral-7B
59.9
Eurus-RM-7b
61.7
internlm2-7b-reward
69.5
Skywork-Reward-Gemma-2-27B
67.8
ArmoRM-Llama3-8B-v0.1
62.7
GRM-llama3-8B-sftreg
63.1
internlm2-20b-reward
71.3
Llama-3-OffsetBias-RM-8B
71.2
Nemotron-340B-Reward
71.2
URM-LLaMa-3.1-8B
69.5
Skywork-Reward-Llama-3.1-8B
66.3
INF-ORM-Llama3.1-70B

GenRMs

tulu-v2.5-dpo-13b-chatbot-arena-2023 64.9
56.3
tulu-v2.5-dpo-13b-nectar-60k
67.2
stablelm-2-12b-chat
66.4
tulu-v2.5-dpo-13b-stackexchange-60k
58.8
Nous-Hermes-2-Mistral-7B-DPO
62.5
Claude-3-5-sonnet-20240620
68.4
tulu-v2.5-dpo-13b-hh-rlhf-60k
66.4
tulu-2-dpo-13b
78.6
SOLAR-10.7B-Instruct-v1.0
64.3
Llama3.1-70B-Instruct
71.4
Skywork-Critic-Llama-3.1-70B
67.2
GPT-4o-0806
71.6
Gemini-1.5-pro

REASRMS

53.0 49.3
51.4 55.5
58.0 51.7
57.0 52.7
60.2 56.9
71.4 49.7
54.7 53.2
57.5 53.1
62.5 57.8
66.8 56.7
61.9 53.2
59.8 59.4
61.8 54.1
60.6 54.5
65.6 56.8

52.3 50.5
52.4 52.6
54.9 51.6
49.9 54.2
55.6 51.3
62.6 54.4
51.1 52.3
51.4 51.8
52.3 49.6
67.3 47.5
64.6 56.8
67.5 63.6
73.9 63.7

JudgeLRM
RM-R1-Qwen-Instruct-7B
Self-taught-evaluator-llama3.1-70B
RM-R1-DeepSeek-Distilled-Qwen-7B
RM-R1-Qwen-Instruct-14B
RM-R1-Qwen-Instruct-32B
RM-R1-DeepSeek-Distilled-Qwen-14B
RM-R1-DeepSeek-Distilled-Qwen-32B

59.9
66.6
73.4
64.0
75.6
75.3
71.8
74.2

59.9 51.9
67.0 54.6
65.7 56.3
83.9 56.2
75.4 60.6
80.2 66.8
90.5 69.5
91.8 74.1

51.2
87.1
86.8
87.2
86.5
85.5
91.9
92.4
90.0
86.5
89.6
87.5
93.1
95.7
94.8

62.3
73.8
65.2
69.0
73.9
64.4
76.5
85.4
78.9
83.0
94.8
91.7
91.3

87.3
92.6
90.4
85.3
93.6
93.9
94.1
95.4

48.3
72.8
87.1
88.6
87.2
85.4
78.0
82.2
83.5
82.6
84.6
81.0
84.0
89.0
91.8

82.8
86.7
69.1
79.5
69.5
73.8
53.6
86.9
57.5
74.7
85.6
83.4
83.1

73.2
79.2
80.2
75.9
82.6
86.3
86.2
89.5

54.9
65.6
67.3
67.1
70.2
70.7
69.2
71.0
72.7
71.6
72.2
71.4
73.2
74.7
76.1

60.2
64.3
63.5
63.0
61.1
63.4
63.0
66.7
67.6
67.8
73.7
75.6
77.6

766.2
71.7
74.5
73.1
77.5
80.5
83.6
85.4

54.3 52.5
50.7 63.0
35.3 63.2
34.9 63.5
40.2 65.9
45.1 67.1
54.9 67.3
49.8 67.7
48.6 68.2
50.7 68.3
50.2 69.0
56.1 69.5
53.0 70.0
46.6 70.1
44.8 70.9

29.5 57.5
25.4 58.8
46.6 59.7
37.2 59.9
49.1 59.9
45.9 61.0
69.6 62.1
37.7 63.8
69.4 64.8
54.1 65.5
56.5 71.9
58.7 72.5
64.7 75.2

54.8 64.7
59.7 70.2
59.7 71.5
68.1 72.4
68.8 76.1
70.4 79.1
74.4 81.5
76.7 83.9

and then apply different strategies for handling those tasks. Intuitively, reinforcement learning alone
can effectively explore reasoning tasks, a domain where it has already achieved considerable success.
Here, we incorporate the system prompt shown in Figure 3, which explicitly distinguishes between
chat and reasoning tasks.

For reasoning tasks specifically, we note that answer quality is closely tied to correctness, and that
high-level rubrics may be less effective than simply evaluating whether the model can solve the
problem and verify its own answer. Thus, this setting emphasizes correctness-based evaluation guided
by task classification in the prompt.

Distilled + RL + Rubrics + QC (RM-R1). Building on the previous setup, we introduce an
additional distillation stage from stronger teacher models as a warm start before RL training. The
motivation is that with RL alone, weaker models (especially at smaller scales) often fail to explore
high-quality rubrics and convincing reasoning chains for chat tasks throughout the RL training
process. Distilling strong reasoning traces on a small subset of data can effectively mitigate this
limitation.

20

Table 8: The leaderboard of RMB, ranked by the average score of all subsets. Bold numbers indicate
the best performance, Underlined numbers indicate the second best.

Models

ScalarRMs

Tulu-v2.5-13b-preference-mix-rm
SteerLM-RM 70B
Skywork-Reward-Gemma-2-27B
Internlm2-20b-reward
ArmoRM-Llama3-8B-v0.1
Internlm2-7b-reward
Eurus-RM-7b
Skywork-Reward-Llama-3.1-8B
INF-ORM-Llama3.1-70B
Starling-RM-34B

GenRMs

Llama2-70b-chat
Llama3.1-8B-Instruct
Gemini-1.5-pro
Mixtral-8x7B-Instruct-v0.1
skywork-critic-llama3.1-8B
skywork-critic-llama3.1-70B
Llama3.1-70B-Instruct
Mistral-Large-2407
Claude-3-5-sonnet
Qwen2-72B-Instruct
GPT-4o-2024-05-13

REASRMS

JudgeLRM
RM-R1-DeepSeek-Distilled-Qwen-7B
RM-R1-Qwen-Instruct-7B
Self-taught-evaluator-llama3.1-70B
Deepseek-GRM-27B-RFT
RM-R1-DeepSeek-Distilled-Qwen-14B
Deepseek-GRM-27B
RM-R1-Qwen-Instruct-14B
RM-R1-DeepSeek-Distilled-Qwen-32B
RM-R1-Qwen-Instruct-32B

Helpfulness

Harmlessness

BoN Pairwise

BoN Pairwise

Overall

0.355
0.502
0.472
0.585
0.636
0.626
0.679
0.627
0.650
0.604

0.289
0.365
0.536
0.480
0.600
0.640
0.648
0.678
0.705
0.645
0.639

0.363
0.451
0.543
0.616
0.592
0.593
0.623
0.594
0.620
0.636

0.562
0.574
0.653
0.763
0.787
0.782
0.818
0.781
0.798
0.774

0.613
0.675
0.763
0.706
0.725
0.753
0.811
0.817
0.838
0.810
0.815

0.699
0.658
0.740
0.786
0.801
0.765
0.805
0.776
0.782
0.791

0.351
0.578
0.561
0.499
0.497
0.563
0.543
0.603
0.607
0.674

0.249
0.267
0.299
0.491
0.578
0.614
0.558
0.583
0.518
0.649
0.682

0.363
0.429
0.608
0.546
0.548
0.613
0.570
0.620
0.618
0.682

0.545
0.673
0.721
0.670
0.663
0.712
0.693
0.759
0.767
0.795

0.602
0.653
0.661
0.671
0.578
0.614
0.739
0.725
0.764
0.789
0.814

0.674
0.664
0.765
0.733
0.765
0.769
0.761
0.778
0.771
0.809

0.453
0.582
0.602
0.629
0.646
0.671
0.683
0.693
0.705
0.712

0.438
0.490
0.565
0.587
0.620
0.655
0.689
0.701
0.706
0.723
0.738

0.531
0.551
0.664
0.670
0.670
0.685
0.690
0.692
0.698
0.730

G.2 Training Dynamics

We analyze the training dynamics of RM-R1 using the Qwen-2.5-14B-Instruct model by tracking
both response length and reward progression throughout RL training. We consider two settings: (a)
Cold Start RL, and (b) Warm Start RL following reasoning-chain distillation. We present the finding
in Figure 8.

In the Cold Start RL setting, we observe that the model gradually learns to reason, as reflected by a
steady increase in response length over the course of training. However, training becomes unstable
near the end, with a sharp drop in the reward curve, suggesting potential issues such as overfitting.

In contrast, under Warm Start RL, the model begins with stronger initial reasoning abilities, exhibiting
longer responses from the outset. Interestingly, the model first learns to produce more concise
reasoning traces before gradually increasing response length again as training progresses. The reward
curve rises smoothly and consistently throughout training, demonstrating more stable and efficient
learning compared to the Cold Start setting.

21

Table 9: Complete questions and chatbot responses.

Question: symptoms of sickle-cell disease

Chatbot A:
1. Painful red or yellow skin lesions
2. Frequent infections
3. Fatigue and weakness
4. Anemia
5. Slower growth and development
6. Jaundice
7. Swelling of hands and feet
8. Vision loss
9. Tiredness and weakness after physical activity
10. Breathlessness
11. Fever, chills, and night sweats.

Chatbot B
Sickle cell disease is a genetic disorder that affects the shape of red blood cells. It is most common in African
Americans, but can also affect people from other ethnic groups. The symptoms of sickle cell disease can vary
widely from person to person, but some common symptoms include:

1. Fatigue: People with sickle cell disease often experience fatigue and weakness, even when they have not been
physically active.
2. Pain: Sickle cell disease can cause pain in the body, particularly in the arms, legs, back, and abdomen. Pain
can be mild or severe and can last for days or weeks.
3. Fever: People with sickle cell disease may be prone to infections and may experience frequent fevers.
4. Anemia: Sickle cell disease can cause anemia, a condition in which the body does not have enough red blood
cells to carry oxygen throughout the body.
5. Delayed growth and development: Children with sickle cell disease may be smaller than their peers due to
delayed growth and development.
6. Infections: People with sickle cell disease are at increased risk for infections, particularly in the bloodstream
and urinary tract.
7. Pigmentation: Sickle cell disease can cause dark areas of pigmentation in the body, particularly on the hands
and feet.

It�s important to note that not everyone with sickle cell disease will experience all of these symptoms, and some
people may not experience any symptoms at all. Therefore, it�s important to consult with a healthcare provider if
you suspect you or someone you know may have sickle cell disease.

Chain -of -Rubrics (CoR) Roll-out for Instruct Models
(no categorization of task types)

Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client�s question
displayed below.

Instructions
1. Begin your evaluation by generating the rubric criteria tailored to the Client�s question and context.

Enclose the rubric in <rubric> . . . </rubric> tags.

2. Assign weights to each rubric item based on their relative importance.
3. Within <rubric>, include a <justify> . . . </justify> section explaining the rationale behind the chosen criteria and weights.
4. Compare both Chatbot responses using the rubric.
5. Include your evaluation in <eval> . . . </eval> tags.

Support your analysis using:
- <quote_A> . . . </quote_A> for direct quotes from Chatbot A
- <summary_A> . . . </summary_A> for paraphrased summaries of Chatbot A
- <quote_B> . . . </quote_B> for direct quotes from Chatbot B
- <summary_B> . . . </summary_B> for paraphrased summaries of Chatbot B

6. Conclude with your final judgment using:

<answer>[[A]]</answer>

or

<answer>[[B]]</answer>

Important Notes:
- Be objective and base your evaluation strictly on the content of the responses.
- Do not let the response order, length, or Chatbot names bias your judgment.

Figure 6: The system prompt of the ablation study on cold start RL without categorization of task
types.

22

Table 10: Cold-Start RL Judges. The rubrics and evaluation mainly center around superficial shallow
features (e.g., number of points listed) without actually taking their accuracy into account.

Cold-start RL evaluation
<type>Chat</type>
<rubric>
The evaluation criteria for the responses are as follows:
- <item>Relevance to the question</item> (40%): The response should directly address the symptoms of sickle
cell disease.
- <item>Comprehensiveness</item> (30%): The response should include a wide range of symptoms commonly
associated with sickle cell disease.
- <item>Clarity and organization</item> (30%): The response should be easy to read and understand, with
symptoms listed in a clear and organized manner.
<justify>
The question specifically asks for the symptoms of sickle cell disease, so the responses should be directly relevant
to this topic. Comprehensiveness is important because sickle cell disease can have a variety of symptoms,
and a comprehensive list will provide a more complete picture of the condition. Clarity and organization are
important because the symptoms can be complex, and a well-organized response will make it easier for the client
to understand the information.
</justify>
</rubric>

<eval>
<quote_A>
1. Painful red or yellow skin lesions
2. Frequent infections
3. Fatigue and weakness
4. Anemia
5. Slower growth and development
6. Jaundice
7. Swelling of hands and feet
8. Vision loss
9. Tiredness and weakness after physical activity
10. Breathlessness
11. Fever, chills, and night sweats.
</quote_A>
<summary_A>
Chatbot A provided a list of symptoms that are directly relevant to sickle cell disease. The response is
comprehensive and includes a wide range of symptoms commonly associated with the condition. The symptoms
are listed in a clear and organized manner, making it easy for the client to understand the information.
</summary_A>
<quote_B>
Sickle cell disease is a genetic disorder that affects the shape of red blood cells. It is most common in African
Americans, but can also affect people from other ethnic groups. The symptoms of sickle cell disease can vary
widely from person to person, but some common symptoms include:
1. Fatigue: People with sickle cell disease often experience fatigue and weakness, even when they have not been
physically active.
2. Pain: Sickle cell disease can cause pain in the body, particularly in the arms, legs, back, and abdomen. Pain
can be mild or severe and can last for days or weeks.
3. Fever: People with sickle cell disease may be prone to infections and may experience frequent fevers.
4. Anemia: Sickle cell disease can cause anemia, a condition in which the body does not have enough red blood
cells to carry oxygen throughout the body.
5. Delayed growth and development: Children with sickle cell disease may be smaller than their peers due to
delayed growth and development.
6. Infections: People with sickle cell disease are at increased risk for infections, particularly in the bloodstream
and urinary tract.
7. Pigmentation: Sickle cell disease can cause dark areas of pigmentation in the body, particularly on the hands
and feet.
</quote_B>
<summary_B>
Chatbot B provided a detailed explanation of sickle cell disease and a list of symptoms. The response is relevant
to the question and includes a wide range of symptoms commonly associated with the condition. However,
the response is less organized than Chatbot A�s response, and the explanation of sickle cell disease may be
unnecessary for the client�s question.
</summary_B>
</eval>

<answer>[[A]]</answer>

23

Table 11: RM-R1 RL Judges. The rubrics and evaluation are well-designed and reasonable, assessing
the effectiveness of the model response.

Warm-start RL (RM-R1) evaluation
<type>Chat</type>
<rubric>
1. Accuracy of information (40 %): How correctly does the response describe the symptoms of sickle-cell
disease?
2. Comprehensiveness (30 %): How thoroughly does the response cover the range of symptoms associated with
sickle-cell disease?
3. Clarity and organization (20 %): How well-structured and easy to understand is the information presented?
4. Helpfulness for the user (10 %): How useful is the information for someone seeking to understand sickle-cell
disease symptoms?
<justify>
These criteria focus on evaluating how well each response addresses the user�s question about sickle-cell disease
symptoms. Accuracy is weighted highest because providing correct information about medical conditions is
crucial. Comprehensiveness is important, as sickle-cell disease has multiple symptoms that vary in severity and
presentation. Clarity and organization affect how easily the user can understand the information. Helpfulness
considers how practical and actionable the information is for the user.
</justify>
</rubric>

<eval>
Comparing the two responses:

Accuracy of information:
<quote_A>1. Painful red or yellow skin lesions
2. Frequent infections
3. Fatigue and weakness
4. Anemia
5. Slower growth and development
6. Jaundice
7. Swelling of hands and feet
8. Vision loss
9. Tiredness and weakness after physical activity
10. Breathlessness
11. Fever, chills, and night sweats.</quote_A>
<quote_B>1. Fatigue: People with sickle-cell disease often experience fatigue and weakness, even when they
have not been physically active.
2. Pain: Sickle-cell disease can cause pain in the body, particularly in the arms, legs, back, and abdomen. Pain
can be mild or severe and can last for days or weeks.
3. Fever: People with sickle-cell disease may be prone to infections and may experience frequent fevers.
4. Anemia: Sickle-cell disease can cause anemia, a condition in which the body does not have enough red blood
cells to carry oxygen throughout the body.
5. Delayed growth and development: Children with sickle-cell disease may be smaller than their peers due to
delayed growth and development.
6. Infections: People with sickle-cell disease are at increased risk for infections, particularly in the bloodstream
and urinary tract.
7. Pigmentation: Sickle-cell disease can cause dark areas of pigmentation in the body, particularly on the hands
and feet.</quote_B>
Chatbot A provides a list of symptoms without explanation, but some listed items (e.g. �painful red or yellow
skin lesions,� �vision loss�) are not typical of sickle-cell disease. Chatbot B offers more accurate information,
explaining each symptom�s nature and severity.
Comprehensiveness:
Chatbot A lists 11 symptoms, several of which are inaccurate or uncommon. Chatbot B lists 7 symptoms that
are more representative and supplies contextual detail about variability and medical advice.
Clarity and organization:
Chatbot A�s plain numbered list lacks context, making severity unclear. Chatbot B structures the list with
explanations, improving readability and understanding.
Helpfulness:
Chatbot A is less helpful due to inaccuracies and missing context. Chatbot B is more useful, providing accurate
descriptions, context, and guidance.
</eval>

<answer>[[B]]</answer>

24

Chain -of -Rubrics (CoR) Roll-out for Instruct Models
(no rubrics)

Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots
to the Client�s question displayed below.

You should choose the chatbot that follows the client�s instructions and answers the client�s question
better. Do not allow the length of the responses to influence your evaluation. Do not favor certain names
of the chatbots. Be as objective as possible. First, compare the chatbot responses and provide your
evaluations. Then, conclude with your verdict using exactly this format: <answer>[[A]]</answer> if
Chatbot A is better, <answer>[[B]</answer> if Chatbot B is better.

Figure 7: The system prompt of the ablation study on cold start RL without any rubrics.

(a) Cold Start RL

(b) Warm Start RL

Figure 8: RL training dynamics under different settings: (a) Cold Start RL (Eq. 11) and (b) Warm
Start RL (Eq. 8). In Cold Start RL, the response length steadily increases as the model learns to
reason, but training becomes unstable near the end. In Warm Start RL, the model exhibits more stable
training, with effective refinement of reasoning traces throughout the process.

25

010203040506070Step500550600650Response LengthResponse Length vs Step010203040506070Step1.501.551.601.651.701.751.801.851.90Train RewardTrain Reward vs Step0102030405060Step760780800820840Response LengthResponse Length vs Step0102030405060Step0.450.500.550.600.650.70Train RewardTrain Reward vs Step
