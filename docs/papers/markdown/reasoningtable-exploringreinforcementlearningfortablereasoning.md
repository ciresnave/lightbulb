5
2
0
2

n
u
J

2

]
L
C
.
s
c
[

1
v
0
1
7
1
0
.
6
0
5
2
:
v
i
X
r
a

Preprint.

Reasoning-Table: Exploring Reinforcement Learning for Ta-
ble Reasoning

Fangyu Lei1,2?, Jinxiang Meng1,2?, Yiming Huang,
Tinghong Chen2, Yun Zhang1, Shizhu He1,2, Jun Zhao1,2, Kang Liu1,2
1 Institute of Automation, CAS 2 University of Chinese Academy of Sciences

Datasets: https://huggingface.co/datasets/TableQAKit/Reasoning-Table
Code: https://github.com/MJinXiang/Reasoning-Table

Abstract

Table reasoning, encompassing tasks such as table question answering, fact
verification, and text-to-SQL, requires precise understanding of structured
tabular data, coupled with numerical computation and code manipula-
tion for effective inference. Supervised fine-tuning (SFT) approaches have
achieved notable success but often struggle with generalization and robust-
ness due to biases inherent in imitative learning. We introduce Reasoning-
Table, the first application of reinforcement learning (RL) to table reasoning,
achieving state-of-the-art performance. Through rigorous data preprocess-
ing, reward design, and tailored training strategies, our method leverages
simple rule-based outcome rewards to outperform SFT across multiple
benchmarks. Unified training across diverse tasks enables Reasoning-Table
to emerge as a robust table reasoning large language model, surpassing
larger proprietary models like Claude-3.7-Sonnet by 4.0% on table reason-
ing benchmarks. The approach also achieves excellent performance on
text-to-SQL tasks, reaching 68.3% performance on the BIRD dev dataset
with a 7B model. Further experiments demonstrate that Reasoning-Table
enhances the model�s generalization capabilities and robustness.

1

Introduction

Tabular data provide a foundational structure for storing, organizing, and presenting
information, underpinning data analysis and data engineering, which are essential in fields
like business intelligence and scientific research (Borisov et al., 2022).

Table reasoning, recognized as a critical task, involves extracting insights, performing logical
inferences, and generating precise responses from structured tabular data (Pasupat & Liang,
2015b; Zhu et al., 2021). This task involves various task types, such as table question answer-
ing (Zhong et al., 2017; Pasupat & Liang, 2015b), fact verification (Chen et al., 2019), and
text-to-SQL (Li et al., 2024; Lei et al., 2024), and handles various table formats. The advent of
Large Language Models (LLMs) (OpenAI, 2023b; Anthropic, 2024) has markedly enhanced
table reasoning capabilities, with cutting-edge frameworks exhibiting an increasing reliance
on these models.

Prompt engineering crafts task-specific pipelines for table reasoning (Cheng et al., 2022b;
Wang et al., 2024; Ye et al., 2023; Cao et al., 2023; Deng et al., 2025), achieving strong
performance on targeted tasks. However, it struggles with scalability and generalizability,
requiring significant manual effort. Alternatively, supervised fine-tuning (SFT) enhances
LLMs� understanding of tables Zhang et al. (2024a); Zha et al. (2023); Zhang et al. (2024b),
yet struggles with generalization and adaptability due to imitation learning biases Chu
et al. (2025); Guo et al. (2025a), often underperforming in tasks requiring complex reasoning,
novel table structures, or unseen task types, leading to poor performance.

?Equal contribution. Authors in alphabetical order.

1

Preprint.

Figure 1: Reasoning-Table encompasses all table-related tasks and diverse table types.
Compared to SFT, using the GPRO RL method, the reasoning model achieves stronger
reasoning capabilities, generalization, and robustness.

Recent advanced LLMs (Jaech et al., 2024; Anthropic, 2024; Guo et al., 2025b) demonstrate
that deeper reasoning via increased computation improves response quality. This has driven
significant RL-based progress in domains like mathematical reasoning (Yu et al., 2025) and
code generation (Luo et al., 2025a). RL offers the potential to acquire high-quality reasoning
paths through independent exploration of the solution space, instead of relying solely on
teacher model distillation. This self-directed learning can help mitigate bias and allows for
a deeper discovery of effective solution patterns. We explore applying RL to table reasoning
tasks, aiming to enhance LLMs� capability in handling complex tables and questions by
training deep thinking models without needing to distill from proprietary sources.

In this work, we introduce Reasoning-Table, the first LLM reinforcement learning method
specifically designed for almost all table reasoning tasks. We develop a comprehensive
pipeline encompassing dataset generation, data filtering strategies, and reinforcement train-
ing methods. Reasoning-Table leverages RL to significantly boost performance on diverse
table reasoning tasks, including table question answering (tableQA), table fact verification
(tableFV), table-to-text generation, and text-to-SQL tasks. Reasoning-Table, which utilizes
simple rule-based output reward and requires neither costly reasoning trace distillation nor
as much data as SFT, substantially surpasses the pure SFT baseline by an average of 17.36%.
We also propose table-specific reward mechanisms to better leverage the table structure,
particularly a position evidence consistency reward that addresses the reward sparsity
inherent in RL paradigms applied to table and database interactions. Notably, unified rein-
forcement learning training across 7 tableQA datasets enables Reasoning-Table to surpass
larger proprietary models, such as Claude-3.7-Sonnet, showcasing outstanding effectiveness
and robustness. Training on text-to-SQL datasets also demonstrates excellent performance,
achieving a strong result of 68.3% on the BIRD SQL leaderboard among models of the same
size using a 7B model.

Further experiments demonstrate the superior potential of RL training over SFT for table rea-
soning tasks, as shown by more convenient training data acquisition, enhanced performance,
stronger generalization capabilities, and improved robustness under perturbations. This
superiority stems from RL�s ability to improve reasoning processes through autonomous
exploration and optimization based on reward signals, rather than being limited by biases
inherited from teacher models. Extensive empirical analyses reveal valuable insights into
table-specific reward design, unified task training strategies, and data selection for table rea-
soning. All our curated high-quality datasets, training configurations, and implementations
will be publicly released, laying a solid foundation for future research and scaling in table
reasoning tasks.

2

Question AnsweringFact Veri?cationTable-to-TextText-to-SQLTablesTasksSupervised Fine-TuningReasoning-TableSFTCold StartRollout ModuleReward ModuleReference LLM� Rule-based outcome� Format reward � Position rewardGroupComputationA1A2AnPolicy LLMReject SamplingEvidence ConsistencyImitation Learning:1. Limited generalization 2. Low robustness3. Lack of in-depth thinkingGRPO TrainingKLReasoning TraceQualityFilteringTrainingReasoning dataPreprint.

2 Table Reasoning

Task types. Table 12 summarizes the 15 datasets we used for table reasoning. The tasks
are primarily divided into 4 major categories: tableQA, tableFV, table-to-text and text-to-
SQL. Text-to-SQL is a significant branch of tableQA, classified separately due to its distinct
characteristics.
Task formulation. Given a natural language query Q and table(s) T = {T1, T2, . . . , Tn}
(n ? 1), the goal is to derive an answer A. In tableQA, A is a string or number extracted
from T . In tableFV, A is a boolean (1 or 0) that verifies Q against T . In table-to-text, A
is a text sequence generated from T . In text-to-SQL, A is a SQL query derived from Q to
retrieve or manipulate data from database T .

Table types. Tables are diverse and include: (1) Simple Column-Header Tables: Basic CSV-
or HTML-like tables with only column headers. (2) Spreadsheet Tables: Feature hierarchical
or multi-level headers, row and column headers, and mixed data types (text, numbers,
formulas). (3) Database Tables: Stored in relational databases, often with multiple interrelated
tables, foreign keys, and indexes. (4) Embedded/Semi-Structured Tables: Found in documents
(e.g., PDFs, Word, webpages), with text interspersed around or within the table.

3 Methodology

3.1 SFT Dataset Preprocessing

Most table reasoning datasets lack annotations for rationales. We generate two types of SFT
data: No-Reason SFT and Reason-SFT.

No-Reason SFT data. We use the full original datasets to construct No-Reason SFT, with
the input data format as shown in Table 13. The total data volume is provided in Table 8.

Reason-SFT data sampling. To create high-quality long chain-of-thought data, we use
proprietary LLMs (Guo et al., 2025b; Anthropic, 2024; OpenAI, 2023a) to generate reasoning
traces from existing table reasoning datasets. Each example is processed 5 times to produce
multiple outputs. For tableQA and tableFV, we retain instances with correct responses
based on the EM metric. For table-to-text generation tasks, we use the BLEU score as the
evaluation criterion, keeping samples with scores above 0.3. For text-to-SQL tasks, we select
trajectories from samples with an execution accuracy of 1. The specific sample counts can be
found in Table 8. We found this approach to be costly and dependent on LLM performance,
and we cannot generate reasoning traces for the entire dataset, as model performance cannot
reach 100%.

Reason-SFT data quality filtering. To ensure high-quality SFT data, a redundancy detection
algorithm refines reasoning processes within <think> tags from Deepseek-R1. The process:
(1) segments text into sentences, discarding those with fewer than five words; (2) computes
TF-IDF-based cosine similarity; (3) adjusts similarity scores with penalties for question
mismatch and modal verb disparity; and (4) removes examples with more than two sentence
pairs exceeding a similarity threshold (? = 0.7). This filters out approximately 90.3% of
trajectories, producing refined datasets detailed in Table 8, with further details in App.A.1.

3.2 RL Dataset Preprocessing

Difficulty control. For table reasoning tasks, RL excels because dataset answers are care-
fully human-annotated, unlike LLM-generated rationales. Notably, controlling QA pair
alignment and data difficulty is also essential. We used Qwen2.5-32B to generate 8 answers
per example, with results reported in Table 9. The results are used for subsequent data
ablation study (Sec.5.2). Initially, we used the full dataset.

Position evidence. The column-row structure is a distinguishing feature of tabular data
compared to other data types. To leverage this, we introduce a table-specific reasoning
content called position trace. We employ a prompting strategy (see Table 15) with two
modes: (1) Column localization: Identifying relevant columns. (2) Column and Cell localization:

3

Preprint.

Identifying both columns and specific cells. For position annotation, we use LLMs to sample
each example 5 times. We filter for samples with correct final answers and compute the
union of their position annotations, i.e., (cid:84)5
i=1 Pi, where Pi denotes the position set from the
i-th correct sample. This ensures robust and comprehensive position evidence. We extract
the position evidence for three datasets, with 71.17% yielding valid evidence, statistic details
are shown in App.A.2. For some datasets where the rows and columns containing the
answers have already been annotated, we directly utilize this information (Zhu et al., 2021;
Cheng et al., 2022a). For text-to-SQL tasks, we use the column names that appear in the
gold SQL as the position reward, which is also commonly referred to as schema linking (Yu
et al., 2018; Li et al., 2024).

3.3 Training

Supervised fine-tuning. Let {(xi, yi)}N
to optimize the model parameters ? by minimizing the standard cross-entropy loss:

i=1 be a labeled dataset of input-target pairs. We seek

N
?
i=1
where ??(� | x) denotes the model�s predicted probability distribution for the target given
the input x.

LSFT(?) = ?

(cid:0)yi | xi

log ??

(cid:1),

Reinforcement training. Inspired by recent advanced reasoning models, We use GRPO to
train the reasoning model. GRPO uses relative performance within a group of candidates to
compute an advantage Ai for policy updates based on the provided objective function.

JGRPO(?) = E

x?P(X)

clip(cid:0) ??(oi|x)
?old(oi|x)

(cid:104) 1
G

Eoi??old

G
?
i=1
(cid:17)
, 1 ? ?, 1 + ?(cid:1)Ai

(cid:16)

min

(cid:16) ??(oi|x)
?old(oi|x)

Ai,

? ?DKL(????ref)

(cid:17)(cid:105)

= ?? (oi|x)

i

where rratio
?old(oi|x) is the importance sampling ratio, and Ai denotes the group-relative
advantage. The hyperparameters ? and ? control the update range and divergence from the
reference policy.

3.4 Reward Design

Rule-based outcome reward. (Rans). The model�s final answer must be enclosed within an
<answer> tag. The reward function evaluates the predicted answer A against the reference
answer Aref, producing a binary reward Rans ? {0, 1}, where 1 indicates a correct or
equivalent answer, and 0 indicates an incorrect answer. The rule-based reward uses 4
evaluation metrics: (1) Exact match: Checks if A = Aref. (2) Execution result match: For tasks
like SQL, checks if Exec(A) = Exec(Aref). (3) F1 thresholding: Assigns 1 if the token-level
F1 score F1(A, Aref) ? ?, where ? is a threshold, and 0 otherwise. (4) BLEU thresholding:
Assigns 1 if BLEU(A, Aref) ? ?, where ? is a threshold, and 0 otherwise. Calculation details
are provided in App.B.4. The rule-based reward is defined as:

Rans =

?

??????????????
??????????????

1

1

1

1

if A = Aref
(Short string or number answer)
if F1(A, Aref) ? ?
(Long string answer)
if Exec(A) = Exec(Aref)
(Text-to-SQL)
if BLEU(A, Aref) ? ?
(Long-Form QA, Table-to-text)

0 otherwise

4

Preprint.

Format reward (Rfmt). To ensure structural consistency and output readability, we enforce
the presence of both <think> and <answer> tags in the model�s output.

Position reward (Rpos). To encourage faithful reasoning grounded in the table, the model is
required to annotate table cells with <|cell content|><|column name|> tags when directly
referencing them in reasoning. Each annotated cell provides evidence for the reasoning
process. The position reward evaluates how well the model identifies and annotates the
relevant cells during reasoning:Rpos = |P?G|
|P?G| where P denotes the set of cells annotated
by the model and G represents the ground-truth set of cells that should be referenced to
answer the question. This encourages models to precisely identify the relevant information
in the table rather than over-annotating or missing critical values. The model�s final result
must be consistent with its predicted position. This reward function ensures that the final
answer is directly derived from the model�s attention positions, rather than being generated
in isolation from the reasoning steps.

Final reward (R). Simply summing these rewards could allow the model to achieve a
high score by prioritizing consistency (e.g., correct cell annotations or format) even when
the answer is incorrect. To address this, we design the final reward to prioritize answer
correctness while rewarding proper reasoning and formatting:

R(oi) = Rans(oi) � (1 + ?1Rpos(oi)) + ?2Rfmt(oi)

where ?1 and ?2 are tunable hyperparameters. The multiplication of Rans with the position
reward term ensures that Rpos contributes only when the answer is correct, preventing
the model from being overly rewarded for consistent but incorrect reasoning. The format
reward, weighted by ?2, encourages structural compliance independently.

4 Experiments

4.1 Setup

Benchmarks. We use several table reasoning datasets, covering tableQA, tableFV, table-
to-text, and text-to-SQL, with format prompt for inference (See Tab.14). We compare our
method against large open-source LLMs, proprietary LLMs (Claude-3.7-Sonnet, GPT-4o),
and the table-specific domain model TableGPT2 (Su et al., 2024).

Single dataset training settings. We use Qwen2.5-7B models as the base model. We employ
four training settings for each dataset: (1) No-Reason SFT: We use the input and output
data from the dataset to train the model, but not all data includes high-quality reasoning
traces. (2) Reason-SFT Start: We use the chain-of-thought reasoning traces generated through
rejection sampling (Sec.3.1) to train the model. (3) RL-zero: We use the GRPO algorithm
Shao et al. (2024) for RL on the data. The learning rate is 1e ? 6, rollout number is 16, batch
size is 32, and epochs are 3. We set ? = 1e ? 3 for KL regularization and apply a generation
temperature of 1.0. We utilize the rule-based outcome reward, set ?1 = 0, ?2 = 0.2. (4)
Reason SFT+RL: We use Reason-SFT data for a cold start to initialize the model to learn the
reasoning trace format, followed by GRPO training on the fine-tuned checkpoints.

Unified training settings. Validating performance on individual dataset tasks is insufficient
for assessing the method�s comprehensive capabilities, making unified training critical. We
conduct unified fine-tuning and reinforcement learning across all tableQA datasets. To
address excessively large training sets in some datasets, we adjust data quantities to balance
the model�s reward, with details provided in Table 8. For SFT, we use a batch size of 64,
a learning rate of 5e-6, and train for 5 epochs. For RL, we employ a learning rate of 1e-6,
utilize rule-based rewards, and set ?1 = 0 and ?2 = 0.2, indicating that both answer reward
and format reward are used simultaneously.

Metrics. We use Exact Match(EM) for tableQA and tableFV, Execution Accuracy (EX) for
text-to-SQL, BLEU for table-to-text task (Sec.B.4). For the base model and the unified training
model, we also use the arithmetic mean score to evaluate the overall performance on the
tableQA task. Details are provided in App.C.

5

Preprint.

Table 1: Performance on tableQA tasks. The metric is EM or Accuracy. AVG is the arithmetic
mean of seven tasks. We only use the train set for training and the test set for evaluation,
without using the validation set.

Method

WikiTQ HybridQA MultiHiertt OTT-QA FinQA TAT-QA HiTab AVG

Claude
GPT-4o
TableGPT2-7B
Qwen2.5-32B-Inst
Qwen2.5-7B-Inst

No-Reason SFT
Reason-SFT
RL-zero
Reason-SFT+RL

No-Reason SFT
Reason-SFT
RL-zero
Reason SFT+RL

82.02
81.19
63.70
79.65
57.27

58.19
74.88
76.35
80.07

57.21
72.35
72.75
75.46

39.36
39.30
30.03
38.2
31.84

36.35
37.18
43.68
44.64

42.36
35.17
41.37
42.83

40.98
40.86
25.12
37.74
27.54

62.69
66.35
48.87
56.5
50.50

Single Dataset Training

15.38
36.61
37.80
40.97

Unified Training

15.92
38.50
37.23
39.56

42.52
55.94
65.13
68.27

42.44
54.40
67.67
68.68

57.45
57.63
38.36
59.2
52.40

57.36
63.38
66.87
68.32

58.41
60.42
60.41
64.46

53.09
53.45
55.12
67.29
49.79

54.46
71.59
74.38
79.79

53.57
63.45
69.87
73.75

75.96
73.92
63.89
73.29
57.19

78.28
74.43
80.56
81.94

78.72
72.72
71.53
73.61

58.79
58.96
46.44
58.84
46.65

/
/
/
/

49.80
56.72
60.12
62.62

Table 2: Performance on text-to-SQL (using Qwen2.5-Coder), long-form QA, tableFV and
table-to-text tasks.

Method

Claude-3.7-Sonnet
GPT-4o
TableGPT2-7B
Qwen2.5-7B-Inst
Qwen2.5-32B-Inst

No-Reason SFT
Reason-SFT
RL-zero
Reason SFT+RL

Long-form QA

Fact Verification

Table-to-Text

Text-to-SQL

FetaQA (BLEU) TabFact (Acc)

FEVEROUS (Acc) ToTTo (BLEU)

Spider (EX) BIRD (EX)

25.57
28.47
13.80
24.96
27.75

36.08
37.81
42.96
43.18

91.12
92.09
83.92
82.82
89.48

71.84
78.31
68.57
63.62
74.26

Single Dataset Training

88.15
90.18
91.46
92.77

86.84
79.48
86.53
85.67

15.90
17.19
13.20
13.92
14.46

16.30
19.55
23.71
28.39

80.66
77.95
75.53
77.75
84.91

79.13
82.92
83.18
84.20

59.13
63.56
47.39
50.72
64.41

55.38
59.58
62.41
63.16

4.2 Main Results

RL training exhibiting superior results compared to SFT across all datasets. As shown in
Table 1 and Table 2, we conduct experiments with both SFT and RL across all datasets. In
single-dataset settings, RL-zero consistently outperforms Reason-SFT and No-Reason SFT. In
the unified training setting, RL-zero surpasses Reason-SFT by 10.0% and No-Reason SFT by
3.1%, while using less data and requiring no reasoning traces from a teacher model. This
demonstrates RL�s strong exploration capability, enabling the model to learn effectively
from its own rollouts for enhanced reasoning performance.

SFT cold start enhances RL performance across all tasks. Our experiments demonstrate
that initializing with SFT before RL yields significant performance improvements across
nearly all tasks. In the unified training setting, Reason-SFT+RL outperforms RL-zero by
an average of 2.79%, underscoring the critical role of SFT initialization in model train-
ing. Furthermore, on the TableQA task, Reason-SFT+RL surpasses Claude-3.5-Sonnet and
Qwen2.5-32B by approximately 2.5%, highlighting the potential of Reasoning-Table�s data
and methods for scaling training.

Reasoning-intensive tasks benefit more from Reasoning-Table. More challenging tasks,
such as BIRD text-to-SQL, exhibit greater performance improvements from our methods,
whereas tasks like table-to-text and tableFV show smaller gains. On the TabFact dataset,
RL-zero outperforms No-Reason SFT by only 3%, reflecting the influence of task difficulty on
performance gains.

6

Preprint.

(a) Response length with train-
ing steps.

(b) Reward with training steps.

(c) EM score with training steps.

Figure 2: Training metrics for the WikiTQ dataset, showing the performance of RL and
SFT+RL under both the single-dataset and unified-dataset settings.

Training progress shows consistent improvement in reward and performance. In Figure
2, we illustrate the trends of the reward score, model response length, and evaluation
score over the training steps. We observe that the response length of models with RL-Zero
increases gradually with training steps. In contrast, the response length of models with
SFT initially decreases and then stabilizes within a consistent range. Both reward and
performance steadily improve with training steps.

5 Analysis

5.1 Training Config Ablation Study

The format reward has a minor impact on the model�s performance. From Table 3 and
Table 4, comparing ?1 = 0, ?2 = 0 with ?1 = 0, ?2 = 0.2 , we observe that the format reward
has a minimal impact on both text-to-SQL and overall experimental performance.

RL enhances the text-to-SQL capabilities across different base models. For text-to-SQL
task, we observe that even with a strong base model, the Reasoning-Table approach still
effectively improves performance. When we apply RL using OmniSQL Li et al. (2025)�an
SQL LLM trained on a high-quality SFT dataset comprising 200M data entries�its perfor-
mance shows significant improvement compared to using Qwen-2.5-Coder (Hui et al., 2024).
We also employed an additional technique by incorporating an extra reward based on N-
gram similarity (App.C.2). This involves matching the similarity between the predicted SQL
and the gold SQL, which enables Reasoning-Table to nearly achieve the SOTA performance
for 7B models on the BIRD dataset, achieving a performance of 68.3 on the dev set.

Position reward is beneficial for reinforcement learning in table reasoning. As shown in
Table 4, we conduct experiments using widely used datasets, Spider and BIRD. Comparing
?1 = 0, ?2 = 0 with ?1 = 0.2, ?2 = 0, we find that the position reward (similar to schema
linking here) brings performance gains, improving the performance by 1.1% on the BIRD
dataset. As shown in Table 3, we analyze the impact of the position reward for three
tableQA tasks. The configuration ?1 = 0, ?2 = 0 indicates that no position reward is
used, while ?1 = 0.2, ?2 = 0 incorporates the position reward to ensure consistency. We
observed that incorporating the position reward further improves the model�s performance
by approximately 0.6% on WikiTQ and TAT-QA. This suggests that the position reward
helps the model better focus on the core content of the table and ensures consistency in its
reasoning process and results. We believe that this type of reward still has room for further
exploration to fully unlock its potential.

LLM-as-a-judge can serve as a complement to rule-based string matching. Datasets like
WikiTQ often contain multiple valid answers (e.g., �2023-10-01� vs. �October 1, 2023�,
or �7 days� vs. �one week�), which rule-based evaluation methods, such as exact match
(EM) or F1, struggle to handle due to their reliance on string matching. To address this
limitation, we employ an LLM-as-a-judge approach, using an LLM M to assess the semantic
equivalence or correctness of the predicted answer A relative to the reference answer Aref,
producing a binary reward BM(A, Aref) ? {0, 1}. We use Qwen2.5-32B as the judge. The

7

020406080100120140Steps020040060080010001200Response LengthRLSFT+RLUni_RLUni_SFT+RL020406080100120140Steps0.00.20.40.60.81.0Reward ScoreRLSFT+RLUni_RLUni_SFT+RL020406080100120140Steps0.500.550.600.650.700.750.800.85AccuracyRLSFT+RLUni-RLUni-SFT-RLPreprint.

reward is defined as: Rans = BM(A, Aref). The reward BM is computed by prompting M
with the prompt P, as detailed in Table 16. This approach effectively captures semantic
equivalence (e.g., �25�C� vs. �77�F�), overcoming the limitations of rule-based method.
In our experiments (Table 3), we keep the reward weighting configuration constant (?1 =
0, ?2 = 0) and vary the method for obtaining Rans, comparing rule-based reward and LLM-
judged reward, with prompts specified in Table 16. Results show that LLM-judged rewards
improve performance on WikiTQ but have little effect on other datasets.

Table 3: Performance across different reward
configurations, all using the Reason SFT+RL
setting.

Reward
?1 = 0, ?2 = 0
(Rule-based Rans )
?1 = 0, ?2 = 0.2
(Rule-based Rans )
?1 = 0.2, ?2 = 0
(Rule-based Rans )
?1 = 0, ?2 = 0
(LLM-judged Rans )

WikiTQ TAT-QA HiTab

80.35

79.62

81.38

80.07

79.79

81.94

80.95

79.47

81.60

81.23

79.96

81.04

Table 4: Performance across datasets with
different models and reward configurations
for text-to-SQL tasks.

Base Model + Reward

Spider BIRD

Qwen2.5-Coder-Inst-7B
+ Spider/BIRD train set SFT
+ ?1 = 0, ?2 = 0
+ ?1 = 0, ?2 = 0.2
+ ?1 = 0.2, ?2 = 0
OmniSQL-7B
+ ?1 = 0, ?2 = 0
+ ?1 = 0.2, ?2 = 0
+ ?1 = 0.2, ?2 = 0 + N-gram

82.9

83.3
83.4
84.2

84.8
87.0
87.4
87.6

59.5

62.1
62.7
63.1

64.1
66.8
67.7
68.3

5.2 Data Ablation Study

We conduct data ablation experiments in the RL-zero setting, based on 8 sampling runs with
a temperature of 0.8, as outlined in Section 3.2 and summarized in Table 9. We evaluate
three data configurations. All data: the original, unfiltered dataset containing all examples;
Challenging data: we filter out examples that were consistently answered correctly, retaining
only those that presented challenges; Variable-performance data: we remove both examples
that were consistently correct and those that were consistently incorrect, retaining only
examples with variable performance. We report the average performance across tableQA
and text-to-SQL tasks for benchmarking, as shown in Figure 3. Notably, the model achieves
the best performance when trained on Challenging data.

5.3 Advanced Evaluation and Analysis

Table 5: Out-of-domain performance
comparison between RL and SFT meth-
ods.

Figure 3: Effect of dif-
ferent data selection
strategy.

Figure 4:
Impact
of model size on
performance
(Avg.
tableQA).

Model

AIT-QA TableBench

Qwen2.5-Inst-7B
Uni No-Reason SFT
Uni Reason SFT
Uni Zero-RL
Uni Reason SFT+RL

86.21
61.94
62.33
89.32
91.33

51.22
32.11
51.80
58.33
60.72

Robustness evaluation. Reasoning-Table enhances the model�s ability to handle table
perturbations, including (1) input format perturbation and (2) column order perturbation. As
shown in Table 6, we use different table serialization methods as input (markdown was
used during training). RL models exhibit robust and superior performance under various
input format perturbations. We keep the question unchanged while randomly altering the
row and column order of the table. According to Table 7, SFT models are highly sensitive

8

Table QAText-to-SQL5055606570Avg score of tasks58.570.860.172.759.672.2Data versionAll dataChallenging dataVariable-performance data1.5B3B7B14BModel Size (Billion Parameters)304050607080Performance Score28.536.846.6550.733.648.462.670.8Performance increaseswith model sizeQwen2.5-InstructReasoning-TablePreprint.

to column and row perturbations, exhibiting a performance drop of 15.22%. In contrast,
RL models adapt effectively to these changes, as they develop strong table reasoning skills
through reinforcement learning, rather than relying solely on imitation.

Table 6: Different input formats of WikiTQ
Task. We using the single dataset training
setting.

Table 7: Model performance under pertur-
bations, with % drops from original setting.
Higher scores, lower drops show better ro-
bustness.

Model

Markdown CSV DataFrame

Qwen2.5-Inst-7B
No-Reason SFT
Reason SFT
Zero-RL
Reason SFT+RL

57.27
58.19
74.88
76.35
80.07

59.23
56.97
69.44
76.35
80.45

58.72
55.29
70.26
72.26
78.95

Model

Origin Column

Row

Both

Qwen2.5-Inst-7B

Uni No-Reason SFT
Uni Reason SFT
Uni Zero-RL
Uni SFT+RL

57.27

57.21
72.35
72.75
75.46

56.80(-0.47)

51.56(-5.71)

47.26(-10.01)

53.61(-3.60)
68.46(-3.89)
70.31(-2.44)
74.50(-0.96)

48.82(-8.39)
64.70(-7.65)
68.58(-4.17)
70.80(-4.66)

46.38(-10.83)
57.66(-14.69)
66.80(-5.95)
68.20(-7.26)

Generalization assessment. We evaluate the generalization of Reasoning-Table on out-of-
domain datasets, featuring task and table types not encountered during training. Specifically,
we use AIT-QA (Katsis et al., 2022) and TableBench (Wu et al., 2025a) (see App.B.3 for
details), with results presented in Table 5. The results show that the RL method outperforms
the SFT method by approximately 30% on unseen datasets. This indicates that SFT�s
imitation learning often limits the model�s generalization ability, thereby impairing its
general reasoning capabilities. In contrast, Reasoning-Table, by optimizing rewards through
exploration, significantly enhances the model�s adaptability and robustness across diverse
table reasoning tasks.

Scaling training analysis. The scaling law, which describes how model size impacts
performance, is essential for evaluating the efficacy of training approaches. We examine
the scaling properties of the proposed training method by testing models of varying sizes
from the Qwen2.5-Instruct series, specifically 1.5B, 3B, 7B, and 14B parameters. We measure
average performance on the tableQA benchmarks, with comprehensive results displayed
in Figure 4. Larger models consistently achieve better results than smaller ones across
both tasks. However, post-training performance gains are minimal for smaller models
(1.5B), while larger models show significant improvements. These results indicate that the
R1-paradigm training approach becomes increasingly effective as model size grows.

5.4 Test time scaling analysis.

As shown in Figure 5, we eval-
uate the impact of RL train-
ing on test-time scaling in the
text-to-SQL task, using Qwen2.5-
7B-Coder, Qwen2.5-7B-Coder-SFT,
and Reasoning-Table-7B-SFT-RL,
with rollouts of 1, 2, 4, 8, 16, and
32, respectively. The line graph
illustrates pass rate and majority
voting performance across these
rollouts, demonstrating that RL
training significantly enhances the
model�s test-time scaling capabil-
ity. Using the strongest OmniSQL
model as the base, Reasoning-
Table achieves a performance of
69.0 in the majority voting setting,
approaching single-model state-of-the-art results on the BIRD-Bench leaderboard (https:
//bird-bench.github.io).

Figure 5: Test-Time Scaling. For the BIRD dataset, we
perform multiple samplings and report the pass@n and
majority voting EX scores.

9

12481632Number of Rollouts50556065707580BIRD EX(%)79.179.279.362.369.064.8Model Performance MetricsQwen2.5-7B-Coder (Pass Rate)Qwen2.5-7B-Coder-SFT (Pass Rate)Reasoning-Table-7B-SFT-RL (Pass Rate)Qwen2.5-7B-Coder (Majority Voting)Qwen2.5-7B-Coder-SFT (Majority Voting)Reasoning-Table-7B-SFT-RL (Majority Voting)Preprint.

6 Related Work

Table reasoning. Table reasoning focuses on enabling models to understand and reason
over structured tabular data to perform tasks such as question answering, fact verification,
and text generation. Several datasets have been proposed, including those for table question
answering Zhong et al. (2017); Pasupat & Liang (2015b); Zhu et al. (2021); Chen et al. (2020;
2021b), fact verification (Chen et al., 2019; Aly et al., 2021), table-to-text (Parikh et al., 2020),
and text-to-SQL Yu et al. (2018); Li et al. (2024), providing a wide range of datasets for the
field. There are two main approaches: one involves carefully designed prompt engineering,
leveraging large language models combined with tool calls like Python, SQL to address
tasks (Cheng et al., 2022b; Cao et al., 2023; Jiang et al., 2023; Wang et al., 2024; Ye et al., 2023;
Cao & Liu, 2025). The other approach uses fine-tuning with table-specific data to enable
models to understand tables Xie et al. (2022); Zhang et al. (2024a;b); Su et al. (2024); Zhu
et al. (2024). Reasoning-Table not only uses large amounts of table data for training but
also incorporates RL training, making it the first work to experiment and analyze across all
table reasoning tasks. Reasoning-Table shares the same motivation as Table-R1 (Yang et al.,
2025; Wu et al., 2025b), but our work differs in data and experimental settings. We believe
that these studies can jointly contribute to advancing table reasoning using reinforcement
learning.

RL for reasoning. The emergence of DeepSeek-R1 (Guo et al., 2025a) shows RL�s ability
to guide models in generating high-quality reasoning trajectories. This has led to studies
enhancing reasoning in mathematics (e.g., DeepScaleR (Luo et al., 2025b), AReaL (Mei et al.,
2025), Light-R1 (Wen et al., 2025), DAPO (Yu et al., 2025)), VAPO (Yuan et al., 2025), and code
generation (e.g., DeepCoder (Luo et al., 2025a)), with significant progress. The R1-paradigm
has made breakthroughs in various NLP domains, such as tool use (Feng et al., 2025; Qian
et al., 2025; Zhang et al., 2025), RAG (Jin et al., 2025), and SQL generation (Pourreza et al.,
2025; Ma et al., 2025). Our work focuses on table reasoning, conducting comprehensive
training and analysis.

7 Conclusion

We propose Reasoning-Table, the first work to apply the LLM reinforcement learning
with the rule-based outcome reward for training on table reasoning tasks, covering a
diverse range of task types and tables. Experiments on tasks such as tableQA and text-to-
SQL demonstrate that Reasoning-Table significantly enhances table reasoning capabilities,
achieving performance comparable to proprietary LLMs. Additionally, Reasoning-Table
improves model robustness and generalization. It also provides high-quality reasoning
datasets, detailed table-processing solutions, data selection strategies, and evaluation scripts,
laying a foundation for future scaling of table-reasoning training.

Limitations

Reasoning-Table is the first work to apply reinforcement learning to table reasoning tasks
for large language models. We identify two limitations in our work, which also point to our
future directions:

Lack of a code execution module. While we used an SQL executor for text-to-SQL tasks,
we did not integrate a code execution module for other tasks such as tableQA, which can
introduce potential hallucinations and computational errors. In the future, we will explore
table RL based on code execution and tool integration.

Lack of fine-grained control over data proportions. As shown in Table 1, when we perform
combined training using all table QA datasets, the model�s performance is worse than that
of single-dataset training. This is caused by differences in the distribution of training data
and rewards. We believe that more effective data combination strategies could help alleviate
this issue. Looking ahead, exploring how to achieve truly unified training across all table
reasoning tasks will be an important research direction.

10

Preprint.

Lack of data scaling across diverse sources. Although we utilized nearly all available table
reasoning datasets for unified training, this approach remains somewhat in-domain and
lacks scalability. In the future, we will investigate a pipeline for automatically collecting
and synthesizing table reasoning data from the Internet to enable scalable RL training for
table reasoning.

References

Rami Aly, Zhijiang Guo, Michael Schlichtkrull, James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, Oana Cocarascu, and Arpit Mittal. Feverous: Fact extraction and
verification over unstructured and structured information. arXiv preprint arXiv:2106.05707,
2021.

AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 1:1,

2024.

Vadim Borisov, Tobias Leemann, Kathrin Se�ler, Johannes Haug, Martin Pawelczyk, and
Gjergji Kasneci. Deep neural networks and tabular data: A survey. IEEE transactions on
neural networks and learning systems, 2022.

Lang Cao and Hanbing Liu. Tablemaster: A recipe to advance table understanding with

language models. arXiv preprint arXiv:2501.19378, 2025.

Yihan Cao, Shuyi Chen, Ryan Liu, Zhiruo Wang, and Daniel Fried. Api-assisted code
generation for question answering on varied table structures. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, pp. 14536�14548, 2023.

Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li,
Xiyou Zhou, and William Yang Wang. Tabfact: A large-scale dataset for table-based fact
verification. In International Conference on Learning Representations, 2019.

Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang
Wang. Hybridqa: A dataset of multi-hop question answering over tabular and textual data.
In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1026�1036,
2020.

Wenhu Chen, Ming-Wei Chang, Eva Schlinger, William Yang Wang, and William W Cohen.
Open question answering over tables and text. In International Conference on Learning
Representations, 2021a.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon,
Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al. Finqa: A dataset
of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical
Methods in Natural Language Processing, pp. 3697�3711, 2021b.

Zhoujun Cheng, Haoyu Dong, Zhiruo Wang, Ran Jia, Jiaqi Guo, Yan Gao, Shi Han, Jian-
Guang Lou, and Dongmei Zhang. Hitab: A hierarchical table dataset for question
answering and natural language generation. In Proceedings of the 60th Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1094�1110, 2022a.

Zhoujun Cheng, Tianbao Xie, Peng Shi, Chengzu Li, Rahul Nadkarni, Yushi Hu, Caiming
Xiong, Dragomir Radev, Mari Ostendorf, Luke Zettlemoyer, et al. Binding language
models in symbolic languages. arXiv preprint arXiv:2210.02875, 2022b.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans,
Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study
of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.

Minghang Deng, Ashwin Ramachandran, Canwen Xu, Lanxiang Hu, Zhewei Yao, Anupam
Datta, and Hao Zhang. Reforce: A text-to-sql agent with self-refinement, format restriction,
and column exploration. arXiv preprint arXiv:2502.00675, 2025.

11

Preprint.

Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan
Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool
use in llms. arXiv preprint arXiv:2504.11536, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025b.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun
Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint
arXiv:2409.12186, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low,
Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card.
arXiv preprint arXiv:2412.16720, 2024.

Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Xin Zhao, and Ji-Rong Wen. StructGPT: A
general framework for large language model to reason over structured data. In Houda
Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empir-
ical Methods in Natural Language Processing, pp. 9237�9251, Singapore, December 2023.
Association for Computational Linguistics.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed
Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines
with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Yannis Katsis, Saneem Chemmengath, Vishwajeet Kumar, Samarth Bharadwaj, Mustafa
Canim, Michael Glass, Alfio Gliozzo, Feifei Pan, Jaydeep Sen, Karthik Sankaranarayanan,
et al. Ait-qa: Question answering dataset over complex tables in the airline industry.
In Proceedings of the 2022 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies: Industry Track, pp. 305�314, 2022.

Fangyu Lei, Tongxu Luo, Pengqi Yang, Weihao Liu, Hanwen Liu, Jiahe Lei, Yiming Huang,
Yifan Wei, Shizhu He, Jun Zhao, et al. Tableqakit: A comprehensive and practical toolkit
for table-based question answering. arXiv preprint arXiv:2310.15075, 2023.

Fangyu Lei, Jixuan Chen, Yuxiao Ye, Ruisheng Cao, Dongchan Shin, SU Hongjin, ZHAO-
QING SUO, Hongcheng Gao, Wenjing Hu, Pengcheng Yin, et al. Spider 2.0: Evaluating
language models on real-world enterprise text-to-sql workflows. In The Thirteenth Interna-
tional Conference on Learning Representations, 2024.

Haoyang Li, Shang Wu, Xiaokang Zhang, Xinmei Huang, Jing Zhang, Fuxin Jiang, Shuai
Wang, Tieying Zhang, Jianjun Chen, Rui Shi, et al. Omnisql: Synthesizing high-quality
text-to-sql data at scale. arXiv preprint arXiv:2503.02240, 2025.

Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin,
Ruiying Geng, Nan Huo, et al. Can llm already serve as a database interface? a big bench
for large-scale database grounded text-to-sqls. Advances in Neural Information Processing
Systems, 36, 2024.

Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang
Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, and
Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini level, 2025a. Notion
Blog.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin
Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing
o1-preview with a 1.5b model by scaling rl, 2025b. Notion Blog.

12

Preprint.

Peixian Ma, Xialie Zhuang, Chengjin Xu, Xuhui Jiang, Ran Chen, and Jian Guo. Sql-r1:
Training natural language to sql reasoning model by reinforcement learning. arXiv preprint
arXiv:2504.08600, 2025.

Zhiyu Mei, Wei Fu, Kaiwei Li, Guangju Wang, Huanchen Zhang, and Yi Wu. Real: Efficient
rlhf training of large language models with parameter reallocation. In Proceedings of the
Eighth Conference on Machine Learning and Systems,MLSys 2025, Santa Clara, CA, USA, May
12-15, 2025. mlsys.org, 2025.

Linyong Nan, Chiachun Hsieh, Ziming Mao, Xi Victoria Lin, Neha Verma, Rui Zhang,
Wojciech Kry�sci �nski, Hailey Schoelkopf, Riley Kong, Xiangru Tang, et al. Fetaqa: Free-
form table question answering. Transactions of the Association for Computational Linguistics,
10:35�49, 2022.

OpenAI. Gpt-4 technical report, 2023a.

R OpenAI. Gpt-4 technical report. arxiv 2303.08774. View in Article, 2:13, 2023b.

Ankur Parikh, Xuezhi Wang, Sebastian Gehrmann, Manaal Faruqui, Bhuwan Dhingra,
Diyi Yang, and Dipanjan Das. Totto: A controlled table-to-text generation dataset. In
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pp. 1173�1186, 2020.

Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured
tables. In Proceedings of the 53rd Annual Meeting of the Association for Computational Lin-
guistics and the 7th International Joint Conference on Natural Language Processing of the Asian
Federation of Natural Language Processing, ACL 2015, July 26-31, 2015, Beijing, China, Volume
1: Long Papers, pp. 1470�1480. The Association for Computer Linguistics, 2015a. doi:
10.3115/v1/p15-1142. URL https://doi.org/10.3115/v1/p15-1142.

Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured
tables. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguis-
tics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long
Papers), pp. 1470�1480, 2015b.

Mohammadreza Pourreza, Shayan Talaei, Ruoxi Sun, Xingchen Wan, Hailong Li, Aza-
lia Mirhoseini, Amin Saberi, Sercan Arik, et al. Reasoning-sql: Reinforcement learn-
ing with sql tailored partial rewards for reasoning-enhanced text-to-sql. arXiv preprint
arXiv:2503.23157, 2025.

Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-T �ur,
Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs. arXiv preprint
arXiv:2504.13958, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Aofeng Su, Aowen Wang, Chao Ye, Chen Zhou, Ga Zhang, Gang Chen, Guangcheng Zhu,
Haobo Wang, Haokai Xu, Hao Chen, et al. Tablegpt2: A large multimodal model with
tabular data integration. CoRR, 2024.

Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent Perot, Zifeng
Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, et al. Chain-of-table:
Evolving tables in the reasoning chain for table understanding. In The Twelfth International
Conference on Learning Representations, 2024.

Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu,
Lifu Tang, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch
and beyond. arXiv preprint arXiv:2503.10460, 2025.

13

Preprint.

Xianjie Wu, Jian Yang, Linzheng Chai, Ge Zhang, Jiaheng Liu, Xeron Du, Di Liang, Daixin
Shu, Xianfu Cheng, Tianzhen Sun, et al. Tablebench: A comprehensive and complex
benchmark for table question answering. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 39, pp. 25497�25506, 2025a.

Zhenhe Wu, Jian Yang, Jiaheng Liu, Xianjie Wu, Changzai Pan, Jie Zhang, Yu Zhao, Shuangy-
ong Song, Yongxiang Li, and Zhoujun Li. Table-r1: Region-based reinforcement learning
for table understanding. arXiv preprint arXiv:2505.12415, 2025b.

Tianbao Xie, Chen Henry Wu, Peng Shi, Ruiqi Zhong, Torsten Scholak, Michihiro Yasunaga,
Chien-Sheng Wu, Ming Zhong, Pengcheng Yin, Sida I Wang, et al. Unifiedskg: Unifying
and multi-tasking structured knowledge grounding with text-to-text language models. In
Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp.
602�631, 2022.

Zheyuan Yang, Lyuhao Chen, Arman Cohan, and Yilun Zhao. Table-r1: Inference-time

scaling for table reasoning. arXiv preprint arXiv:2505.23621, 2025.

Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, and Yongbin Li. Large language
models are versatile decomposers: Decomposing evidence and questions for table-based
reasoning. In Proceedings of the 46th international ACM SIGIR conference on research and
development in information retrieval, pp. 174�184, 2023.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan,
Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement
learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene
Li, Qingning Yao, Shanelle Roman, et al. Spider: A large-scale human-labeled dataset
for complex and cross-domain semantic parsing and text-to-sql task. In Proceedings of the
2018 Conference on Empirical Methods in Natural Language Processing, pp. 3911�3921, 2018.

Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi
Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, et al. Vapo: Efficient and reliable
reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118,
2025.

Liangyu Zha, Junlin Zhou, Liyao Li, Rui Wang, Qingyi Huang, Saisai Yang, Jing Yuan,
Changbao Su, Xiang Li, Aofeng Su, et al. Tablegpt: Towards unifying tables, nature
language and commands into one gpt. arXiv preprint arXiv:2307.08674, 2023.

Shaokun Zhang, Yi Dong, Jieyu Zhang, Jan Kautz, Bryan Catanzaro, Andrew Tao, Qingyun
Wu, Zhiding Yu, and Guilin Liu. Nemotron-research-tool-n1: Tool-using language models
with reinforced reasoning. arXiv preprint arXiv:2505.00024, 2025.

Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. TableLlama: Towards open large general-
ist models for tables. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings
of the 2024 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 6024�6044, Mexico
City, Mexico, June 2024a. Association for Computational Linguistics.

Xiaokang Zhang, Sijia Luo, Bohan Zhang, Zeyao Ma, Jing Zhang, Yang Li, Guanlin Li, Zijun
Yao, Kangli Xu, Jinchang Zhou, et al. Tablellm: Enabling tabular data manipulation by
llms in real office usage scenarios. arXiv preprint arXiv:2403.19318, 2024b.

Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang. Multihiertt: Numerical reasoning
over multi hierarchical tabular and textual data. In Proceedings of the 60th Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6588�6600, 2022.

Victor Zhong, Caiming Xiong, and Richard Socher. Seq2sql: Generating structured queries
from natural language using reinforcement learning. arXiv preprint arXiv:1709.00103, 2017.

14

Preprint.

Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv,
Fuli Feng, and Tat-Seng Chua. Tat-qa: A question answering benchmark on a hybrid
of tabular and textual content in finance. In Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International Joint Conference on Natural
Language Processing (Volume 1: Long Papers), pp. 3277�3287, 2021.

Fengbin Zhu, Ziyang Liu, Fuli Feng, Chao Wang, Moxin Li, and Tat Seng Chua. Tat-llm: A
specialized language model for discrete reasoning over financial tabular and textual data.
In Proceedings of the 5th ACM International Conference on AI in Finance, pp. 310�318, 2024.

15

Preprint.

A Data preprocess details

A.1 Reason-SFT Data Quailty Control.

Since reasoning process generated by Deepseek-R1 may include redundant or contradictory
steps (e.g., the model revises earlier correct reasoning unnecessarily), we introduce a redun-
dancy detection algorithm to eliminate such low-quality samples. Specifically, we focus on
the text within <think> tags and apply the following steps:
(1) Sentence Segmentation and Filtering: Each reasoning segment is split into individual sen-
tences. Sentences with fewer than five words are discarded to remove trivial or malformed
content.
(2) TF-IDF and Cosine Similarity Computation: Sentences are vectorized using TF-IDF, and
pairwise cosine similarity scores are computed for all sentence pairs within each <think>
segment.
(3)Semantic Similarity Penalty Adjustments: If one sentence is a question and the other is not,
the similarity score is down-weighted (Question Mismatch Penalty). If two similar sentences
differ significantly in modal certainty (e.g., use of might, must, will), their similarity score
is penalized (Modal Verb Disparity Penalty). The final adjusted similarity score between
sentence pair (si, sj) is computed as:

�sim(si, sj) = simcos(si, sj) � PQM(si, sj) � PMV(si, sj)
where PQM ? [0, 1] and PMV ? [0, 1] are penalty factors for question mismatch and modal
(4) Redundancy-Based Filtering Criterion: If more than two
verb disparity, respectively.
sentence pairs within a <think> block exhibit high adjusted similarity (threshold ? = 0.7),
the entire example is marked as redundant and removed. Formally, let S = {s1, s2, . . . , sn}
be the set of valid sentences in a <think> segment. Define:

R = (cid:8)(si, sj) |
If |R| > 2, the entire example is marked as redundant and removed.

�sim(si, sj) > ?, i < j(cid:9)

A.2 SFT Data Quailty Control

The data processing pipeline shown in Table 8 presents statistics on the number of samples
retained during SFT data collection.

Table 8: Filtered table reasoning datasets. The Origin column represents the amount of
No-Reason Data;; After Stage1 is the amount of Reason-SFT data synthesized by RFT on
the training set; After Stage2 is the amount of Reason-SFT data finally retained after our
filtering algorithm

Dataset

WikiTQ
HybridQA
MultiHiertt
OTT-QA
FinQA
FeTaQA
TAT-QA
HiTab
ToTTo
TabFact
FEVEROUS
Total

Origin
(No-Reason Data)

After Stage1

After Stage2
(Reason-SFT Data)

Unified
SFT Data

14,152
62,682
7,830
41,469/10,000
6,251
7,326
13,210
7,417
120,761/6,500
92,283/24,101
71,291/4,000
444,672

11,306
41,734
1,879
6,031
3,774
2,819
8,553
4,887
2,097
21,238
3,024
107,342

10,446
38,544
1,475
5,357
3,321
2,663
7,559
4,380
1,896
19,029
2,894
97,564

10,446
5,000
1,475
5,357
3,321
2,663
7,559
4,380
0
3,000
0
43,201

A.3 RL Data Quailty Control

Difficulty control. We use Qwen2.5-32B-Instruct to answer each question in the training set
8 times to classify the difficulty of each example. This was done only for the tableQA and
text2sql tasks. The experimental results are shown in Table 9.
Position evidence control. We extract position evidence for three datasets, WikiTQ, TAT-QA
and HiTab, the statistic results are shown in Table 10.

16

Preprint.

Table 9: Table reasoning datasets with Pass@n performance.

Dataset

Pass@1 Pass@2 Pass@4 Pass@8

WikiTQ
HybridQA
MultiHiertt
OTT-QA
FinQA
TAT-QA
HiTab
Spider
BIRD

76.80
38.26
30.54
31.26
61.29
57.94
73.29
83.36
62.19

77.03
40.80
34.61
34.03
66.17
64.95
77.15
88.29
69.17

78.89
42.36
36.56
37.14
68.96
69.44
80.18
89.65
72.29

79.72
43.72
39.15
39.24
71.14
73.76
81.94
91.49
75.10

Table 10: Position evidence generation control. Solve Rate indicates that the model predicts
the correct answer, we use Pass@10; Evidence-Containing Responses indicates the number
of responses that contain evidence in the reasoning process; Valid Evidence Responses
indicates that the evidence in the reasoning process is true from the table; Overall Correct
denotes the accuracy of both correct answers and evidence from the table.

Dataset

Solve Rate (Pass@10) Evidence-Containing Responses (%) Valid Evidence Responses (%) Overall Correct (%)

WikiTQ
TAT-QA
HiTab

Average

86.66
74.00
71.35

78.53

95.80
89.95
99.58

94.53

98.59
72.57
99.91

91.28

79.35
67.29
70.58

71.17

A.4 Example of Filtered Data

Our filtering strategy effectively removes redundant reasoning processes. In Table 11,
we present an example that was filtered out due to excessive repetitive verification and
ineffective reasoning.

Table 11: A filtered case. The reasoning process is wrapped in <think> </think>, and one
can clearly see that this reasoning process is redundant.

Question: How many people entered office in 1970?
Ground Truth: 10
<think> Alright, let�s tackle this question. The task is to find out how many people
entered office in 1970 based on the provided table. Let me start by understanding the
table structure and then check each entry carefully.
First, the table has four columns: State, Title, Name, and Entered office. The �Entered
office� column has years, some with exact dates and others just the year. The question
specifically asks for entries in 1970. So, I need to scan through the �Entered office�
column and count all instances where the year is 1970.
Let me go row by row.
1. Barbados - Sir Stanley Robinson entered in 1966. Not 1970.
2. Barbados - Sir Theodore Brancker entered in 1961. Nope.
3. Bolivia - President of the Senate, entered 1970. Name is missing, but the year is 1970.
So that�s one.
. . .
Wait, let me recount:
- Bolivia: 2 entries (President of the Senate and Chamber of Deputies) both 1970. Total
2.
- Brazil: 2 entries (President of the Senate and Chamber of Deputies) both 1970. Now
total 4.
- El Salvador: 1970 (with ?). Total 8.
- Guatemala: 1970. Total 9.
- Uruguay: President of the Chamber of Deputies entered March 1, 1970. Total 10.
Wait, that�s 10? Let me check again.
Wait, the entries are:

17

Preprint.

1. Bolivia President of the Senate: 1970 (1)
2. Bolivia President of the Chamber of Deputies: 1970 (2)
10. Uruguay President of the Chamber of Deputies: March 1, 1970 (10)
Additionally, Nicaragua has entries with �None� so excluded. But the entry for
Uruguay�s Fernando Elichirigoity is March 1, 1970, which counts.
Wait, also, what about Venezuela? Their entries are 1968. No. Mexico�s entries are 1967.
So the count is 10?
Wait, but let me check the exact entries again:
Looking at the table:
- Bolivia: two entries with 1970. So two.
- Brazil: two entries 1970. So two more, total 4.
- Uruguay: Chamber of Deputies entered March 1, 1970. 10.
Is there any other entry? Let�s check again:
Looking at the list:
- Ecuador: both entries have ?. So no.
- Puerto Rico: 1969. No.
- Others are all prior years.
So total of 10 entries. But wait, let me check again the entries:
Wait, the Uruguay entry for Fernando Elichirigoity is March 1, 1970. So that�s definitely
1970. So yes, that�s 10.
But wait, the El Salvador entry has a question mark. Should that be counted?
The original question says �how many people entered office in 1970� based on the table.
The table lists some entries with a question mark. For example, El Salvador�s entry is
�1970 ?�. The question mark here might indicate uncertainty in the data. However, the
problem statement says to answer based on the table as presented, so even if there�s a
question mark, if the year is written as 1970, we should count it. Similarly, entries like
�1966 ?� (Dominican Republic) would not be counted as 1970.
Therefore, El Salvador�s entry with �1970 ?� is considered as 1970. So that�s included,
making the total 10.
Wait, but let�s recount:
1. Bolivia President of the Senate: 1970 (1)
4. Brazil President of the Chamber of Deputies: 1970 (4)
10. Uruguay President of the Chamber of Deputies: 1970 (10)
Yes, that�s 10. Therefore, the answer is 10.
But wait, I need to check if any of the entries with exact dates (e.g., July 20, 1970) are
counted. Since the year is 1970, they should be counted regardless of the month. So
even if they entered in July, it�s still 1970. So those are valid.
Now, cross-checking again, the entries are:
- Bolivia: 2
- Brazil: 2
- Chile: 1
- Uruguay: 1
Total: 2 + 2 = 4, plus 1 = 5, plus 1 = 6, plus 1 = 7, plus 1 = 8, plus 1 = 9, plus 1 = 10.
Yes, that�s 10. So the answer should be 10. </think>
<answer> 10 </answer>

B Evaluation Details

B.1 Datasets details

For OTT-QA (Chen et al., 2021a), which is an open retrieval QA dataset, we use the oracle
setting. This means that we only use the provided table containing the correct answer as
input. The focus of our paper is on table reasoning, rather than retrieval. For HybridQA
(Chen et al., 2020) and MultiHiertt (Zhao et al., 2022), we use the data processed by Table-
QAKit (Lei et al., 2023). They performed preprocessing, filtering out irrelevant long passages
and retaining only the tables and some core text, which is beneficial for our study of table
reasoning.

18

Preprint.

B.2 Tasks and Metrics

The tasks and metrics are shown in Table 12.

Table 12: All table reasoning tasks. Tasks marked with * indicate out-of-domain evaluation.

Task Type

Task

Table Type

Output

Metrics

Question Answering

WikiTQ (Pasupat & Liang, 2015a)
HybridQA (Chen et al., 2020)
MultiHiertt (Zhao et al., 2022)
OTT-QA (Chen et al., 2021a)
FinQA (Chen et al., 2021b)
FeTaQA (Nan et al., 2022)
TAT-QA (Nan et al., 2022)
HiTab (Cheng et al., 2022a)
AIT-QA* (Nan et al., 2022)
TableBench* (Wu et al., 2025a)

Simple Table
Semi-Structured Table
Spreadsheet Table
Semi-Structured Table
Spreadsheet Table
Spreadsheet Table
Spreadsheet Table
Spreadsheet Table
Simple Table
Simple Table

Answer
Answer
Answer
Answer
Answer
Sentence Answer
Free-Form Answer
Answer
Answer
Answer

Text-to-SQL

Spider Yu et al. (2018)
BIRD Li et al. (2024)

Table-to-Text

ToTTo (Parikh et al., 2020)

Fact Verification

TabFact (Chen et al., 2019)
FEVEROUS (Aly et al., 2021)

Database Table
Database Table

Simple Table

Simple Table
Simple Table

SQL
SQL

Text

Boolean
Boolean

Acc
EM,F1
EM,F1
EM,F1
Acc
BLEU
EM,F1
Acc
EM
EM

EX
EX

BLEU

Acc
Acc

B.3 Out-of-domain Evaluation

We evaluate out-of-domain generalization on TableBench (Wu et al., 2025a) and AIT-
QA (Katsis et al., 2022). For TableBench, we use two subsets: NUMERICALREASONING and
FACTCHECKING, with exact match (EM) as the evaluation metric. For AIT-QA, we report
execution accuracy (EX). All rollouts are generated using vLLM v0.8.2 with a maximum
generation length of 4096 tokens.

B.4 Metrics

Exact Match (EM). This metric returns 1 if the predicted answer exactly matches the reference
answer, and 0 otherwise:

EM =

(cid:26)1

if �y = y
0 otherwise

where �y is the predicted answer and y is the ground-truth answer.
BLEU Score. BLEU measures the n-gram precision with a brevity penalty (BP) to penalize
short candidates:

BLEU = BP � exp

wn log pn

(cid:33)

(cid:32) N
?
n=1

where pn is the modified n-gram precision, wn is the weight (typically uniform), and BP is
defined as:

(cid:26)1

BP =

exp (cid:0)1 ? r
c

(cid:1)

if c > r
if c ? r

with c the length of the candidate and r the length of the reference.
Execution Accuracy (EX). This metric evaluates whether the execution result of the predicted
logical form equals the ground-truth execution result:

EX =

(cid:26)1

if Execute( �P) = Execute(P)

0 otherwise

where �P and P are the predicted and gold programs/logical forms respectively, and Execute
denotes the result of executing the program on the input table.
F1 Score. The F1 score is computed based on the precision and recall of tokens in A relative
to Aref, where precision is the fraction of tokens in A that appear in Aref, and recall is the
fraction of tokens in Aref that appear in A.

Precision =

|Tokens(A) ? Tokens(Aref)|
|Tokens(A)|

, Recall =

|Tokens(A) ? Tokens(Aref)|
|Tokens(Aref)|

19

Preprint.

F1 = 2 �

Precision � Recall
Precision + Recall

Tokens are obtained by splitting A and Aref into words or subwords, ignoring case and
punctuation, unless specified otherwise.

C Experiments Details

C.1 Setup

SFT settings. For single-dataset SFT, We perform full fine-tuning separately on the no-
reason and reason data for each task. we use a batch size of 64, a learning rate of 5e-6, and
train for 5 epochs.

For uni-dataset SFT, We merge the no-reasoning and reasoning data from all tasks in the
Table 12 and perform full fine-tuning on each. We use a batch size of 64, a learning rate of
5e-6, and train for 5 epochs.

RL setting. We adopt the GRPO algorithm as our core reinforcement learning method and
enhance it with several modifications. We discard the entropy loss entirely. To prevent
rapid entropy collapse during training, we follow DAPO Yu et al. (2025) and apply the
Clip-Higher strategy by setting the clip ratio upper bound to 0.28. To further reduce the
noise introduced by excessively long sequences, we mask the loss of truncated responses.
The rollout temperature is fixed at 1.0 for all experiments, and the learning rate is set to
1 � 10?6.

For SQL RL, we remove the KL divergence loss term. We use a rollout batch size of 64 and
an update batch size of 16. For each prompt, we sample 16 trajectories during rollout. The
maximum response length is set to 8192 tokens.

For single-dataset RL and uni-dataset RL, we retain the KL loss with a small coefficient of
0.001. We use a rollout batch size of 256 and an update batch size of 64. Each prompt is
sampled with 5 trajectories during rollout. The maximum response length is set to 4096
tokens.

C.2 N-Gram similarity for SQL.

For predicted SQL Sp and gold SQL Sg:

1. Tokenize: Split into Tp = {tp,1, . . . , tp,m}, Tg = {tg,1, . . . , tg,n}, normalizing key-

words.

2. N-grams: For n:

3. Similarity:

Np = {(tp,i, . . . , tp,i+n?1) | 1 ? i ? m ? n + 1}

Ng = {(tg,j, . . . , tg,j+n?1) | 1 ? j ? n ? n + 1}

simn =

(cid:40) |Np?Ng|
|Np?Ng|
0

if |Np ? Ng| > 0
otherwise

Use n = 2; if tokens are insufficient, set n = 1.

D Other train results

We also present the metric changes during the training process on the HiTab dataset. As
shown in Figure 6, the model trained with our carefully crafted reasoning data for cold start
followed by RL training achieves the best performance.

20

Preprint.

(a) Response length with train-
ing steps.

(b) Reward with training steps.

(c) EM score with training steps.

Figure 6: We present the training metrics for the HiTab dataset, showing the performance of
RL and SFT+RL under both the single-dataset setting and the unified-dataset setting.

E Prompts

In this section, we present the prompts used in Reasoning-Table. To ensure evaluation
consistency, the prompt format is uniform across all LLMs. Table 13 shows the reason-SFT
reject sampling prompt. Table 14 shows the generation prompts. Table 15 presents the
prompt for getting the position of tables. Table 16 shows the llm-as-a-judge prompt for
LLM-based reward.

Table 13: The prompt of reason-SFT generation. question and table will be replaced with the
specific question and table.

## Objective: You are provided with a table and a question related to the table.
Your task is to answer the question step by step based on the table.
Table: table
Question: question
The answer should be short and simple. It can be a number, a word, or a phrase in the
table, but not a full sentence.
Your response should end with �Answer: xxx� (answer to the question).
Now, answer the question step by step:

Table 14: Input format of tableQA examples. question and table will be replaced. We use
this template during training.

Based on the table below, please answer the question, the answer should be short and
simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
## Notes:
- Output the reasoning process inside a single pair of <think> ... </think> tags.
- Output the final answer inside <answer> ... </answer> tags.
- Think through your reasoning process carefully, solve Question step by step.
## Output Format Example
<think> Step-by-step reasoning here... </think>
<answer> here is your answer </answer>
#Task:
Table: table
Question: question

F Examples

We demonstrate Reasoning-Table �s reasoning process using representative examples from
the training phase. Table 17 traces a complete reasoning trajectory with explicit position
annotations, revealing the provenance of each evidence fragment and the step-wise infer-
ence leading to the correct answer. Table 18 provides an example from the WikiTQ dataset,

21

020406080100120140Steps020040060080010001200Response LengthRLSFT+RLUni_RLUni_SFT+RL020406080100120140Steps0.00.20.40.60.81.0Reward ScoreRLSFT+RLUni_RLUni_SFT+RL020406080100120140Steps0.500.550.600.650.700.750.800.85AccuracyRLSFT+RLUni-RLUni-SFT-RLPreprint.

Table 15: Prompt for getting the position of tables. question and table will be replaced
with the specific question and table during training and inference. With two modes: (1)
Column localization: Identifying relevant columns, only use <|column name|> (2) Column and
Cell localization: Identifying both columns and specific cells, use <|cell content|><|column
name|>.

You are an expert in table-based reasoning. You are given a table and a question.
Your task is to answer the question based on the table.
# Guidelines
1. To answer the question, you should first think step-by-step by placing your reasoning
trace inside <think> ... </think> tags based on the table.
2. In your step-by-step reasoning inside <think> ... </think> , ONLY annotate a table
cell with <|cell content|><|column name|> WHEN you are actually using that value in
reasoning. Do NOT include or list all table cells�only those you are directly referencing.
3. Please use <answer> ... </answer> to enclose the final answer. The answer should be
short and simple. It can be a number, a word, or a phrase in the table, but not a full
sentence. It must be consistent with the question.
4. You must include <think> ... </think> and <answer> ... </answer> tags in your
response.
# Task
Table: table
Question: question

demonstrating the model�s ability to pinpoint and leverage critical table entries. Table 19
highlights a case from the TAT-QA benchmark, showcasing Reasoning-Table �s proficiency
in numerical calculation and internal consistency checks. In Table 20, we present a FEVER-
OUS example that demonstrates the model�s fact-verification capabilities over structured
tabular data. Table 21 presents a HiTab example, underscoring the model�s effectiveness in
extracting relevant information from vertically oriented tables. Finally, Table 22 shows a
HybridQA case that combines tabular and textual inputs, illustrating Reasoning-Table �s
proficiency in integrating and reasoning over structurally diverse information sources.

22

Preprint.

Table 16: The prompt for the LLM-as-a-judge reward: Note that this reward is only used
as a supplement and analysis for our method, and we don�t use this reward on the full
experiment.

Here is the original question, the correct answer, and the candidate answer. Please
evaluate whether the correct answer and the candidate answer are consistent.
# Examples:
-
Question: What is the distance from Paris to London?
Candidate Answer: 5 km
Correct Answer: 5000 m
Consistent: Yes
-
Question: How many people live in the city?
Candidate Answer: 1 million
Correct Answer: 1000000
Consistent: Yes
-
Question: What is the date today?
Candidate Answer: 2023-10-01
Correct Answer: October 1, 2023
Consistent: Yes
-
Question: What is the temperature in Paris?
Candidate Answer: 25�C
Correct Answer: 77�F
Consistent: No
-
Question: What is the distance from Paris to London?
Candidate Answer: 5 km
Correct Answer: 10 km
Consistent: No
-
# YOUR TASK
Respond with only Yes or No. Do not include a rationale.
Question: {question}
Candidate Answer: {candidate answer}
Correct Answer: {correct answer}
Consistent:

23

Preprint.

Table 17: A case with position sources. <|cell content|><|column name|> indicates position
source.

Question: braden gellenthien won the men�s competition in 2006. who won the next year?
Ground Truth: Jedd Greschock
Table:
| Year | Location | Men�s Individual | Women�s Individual | Men�s Team | Women�s Team | Mixed
Team | Ref |
| 1996 | Vaulx-en-Velin | Franck Dauphin (FRA) | Anna Campagnoli (ITA) | None | None | None |
[226] |
| 1998 | Taoyuan | John Blaschke (USA) | Shih Ya-Ping (TPE) | United States | Chinese Taipei |
None | [227] |
| 2000 | Madrid | Tsai Sung-Pin (TPE) | Jamie van Natta (USA) | United States | United States |
None | [228] |
| 2002 | Chonburi | Adam Wheatcroft (USA) | Mary Zorn (USA) | United States | United States |
None | [229] |
| 2004 | Madrid | Choi Yong-Hee (KOR) | Mary Zorn (USA) | South Korea | United States | None |
[230] |
| 2006 | Vini?cn�e | Braden Gellenthien (USA) | Amandine Bouillot (FRA) | United States | France |
United States | [231] |
| 2008 | Tainan | Jedd Greschock (USA) | Erika Anschutz (USA) | Mexico | South Korea | United
States | [232] |
| 2010 | Shenzhen | Adam Gallant (USA) | Vida Halimian (IRI) | United States | United States |
United States | [233] |
| 2012 | C �ordoba | Daniel Suter (USA) | Kristina Berger (GER) | Italy | United States | United States
| [234] |
| 2014 | Legnica | None | None | None | None | None | None |
Response:
<think> Let me analyze this step by step:
1. The question states that Braden Gellenthien won the men�s competition in 2006.
2. Looking at the table, I can verify that <|Braden Gellenthien (USA)|><|Men�s Individual|>
indeed won in <|2006|><|Year|>.
3. The question asks who won the next year after 2006, which would be 2007.
4. Looking at the table, I don�t see any entry for 2007. The years jump from 2006 to 2008.
5. In <|2008|><|Year|>, I can see that <|Jedd Greschock (USA)|><|Men�s Individual|> won the
men�s individual competition.
So the next winner after Braden Gellenthien that appears in the table is Jedd Greschock in 2008.
However, there is no data for 2007 in the table.
Wait, I need to be careful here. The question specifically asks about �the next year� which would
be 2007. But since there�s no 2007 entry in the table, I should clarify that the next year shown in
the data after 2006 is 2008, when Jedd Greschock won. </think>
<answer> Jedd Greschock </answer>

24

Preprint.

Table 18: A task example of WikiTQ.

Question: which stadium can hold more people than ballymena showgrounds, but less than
windsor park?
Ground Truth: The Oval
Table:
Table:
# | Stadium | Capacity | City | Home team
1 | Windsor Park | 24,734 | Belfast | Linfield, Northern Ireland national team
2 | The Oval | 15,000 | Belfast | Glentoran
3= | Ballymena Showgrounds | 8,000 | Ballymena | Ballymena United
3= | New Grosvenor Stadium | 8,000 | Ballyskeagh | Lisburn Distillery
3= | Shamrock Park | 8,000 | Portadown | Portadown
6 | Brandywell Stadium | 7,700 | Derry | Derry City
7= | Seaview | 6,500 | Belfast | Crusaders
7= | The Showgrounds | 6,500 | Coleraine | Coleraine
7= | The Showgrounds | 6,500 | Newry | Newry City
10= | Inver Park | 6,000 | Larne | Larne
10= | Solitude | 6,000 | Belfast | Cliftonville
13 | Dixon Park | 5,333 | Ballyclare | Ballyclare Comrades
14 | Mourneview Park | 5,000 | Lurgan | Glenavon
15 | Donegal Celtic Park | 4,200 | Belfast | Donegal Celtic
16= | Holm Park | 3,000 | Armagh | Armagh City
16= | Lakeview Park | 3,000 | Loughgall | Loughgall
16= | Tillysburn Park | 3,000 | Belfast | Harland & Wolff Welders
21 | Clandeboye Park | 2,850 | Bangor | Bangor, Ards
22 | Wilgar Park | 2,500 | Belfast | Dundela
23= | Allen Park | 2,000 | Antrim | Chimney Corner
23= | Hagan Park | 2,000 | Coagh | Coagh United
23= | Mill Meadow | 2,000 | Castledawson | Wakehurst, Moyola Park
23= | Riada Stadium | 2,000 | Ballymoney | Glebe Rangers, Ballymoney United
28= | Crystal Park | 1,500 | Banbridge | Banbridge Town
28= | Fortwilliam Park | 1,500 | Tobermore | Tobermore United
28= | Newforge Lane | 1,500 | Belfast | PSNI
28= | The Showgrounds (Limavady) | 1,500 | Limavady | Limavady United
37 | Glen Road Heights | 400 | Belfast | Sport & Leisure Swifts
38 | Upper Braniel | 300 | Belfast | Knockbreda
39 | Milltown Playing Fields | 280 | Warrenpoint | Warrenpoint Town
? | Upper Malone | ? | Belfast | Queen�s University
Response:
<think> Okay, let�s see. The question is asking for a stadium that can hold more people than
Ballymena Showgrounds but less than Windsor Park. So first, I need to find the capacities of
those two stadiums from the table.
Looking at the table, Ballymena Showgrounds (row 3=) has a capacity of 8,000. Windsor Park
(row 1) has a capacity of 24,734. So we need stadiums with a capacity greater than 8,000 but less
than 24,734.
Now, let�s go through the list and check each stadium�s capacity.
Stadium 2 is The Oval with 15,000. That�s more than 8,000 and less than 24,734. So that�s a
candidate.
Stadium 3= has Ballymena Showgrounds at 8,000, so the next one after that is Stadium 6,
Brandywell Stadium with 7,700. Wait, 7,700 is less than 8,000, so that�s out.
Then Stadium 7= Seaview is 6,500, which is also under. The next is Stadium 10= Inver Park at
6,000. Still under. Solitude is 6,000 as well. Taylors Avenue is 6,000. Then Stadium 13 Dixon Park
is 5,333. All those are below 8,000.
Stadium 14 Mourneview Park is 5,000. Then Stadium 15 Donegal Celtic Park is 4,200. The next
set is from 16= onwards with 3,000, 2,850, etc. All those are way below 8,000.
So after Ballymena Showgrounds (8,000), the next stadium with a higher capacity is The Oval at
15,000. Then Stadium 6 is 7,700, which is below 8,000. So The Oval is the only one that�s between
8,000 and 24,734. Wait, wait, let me check again.
Wait, The Oval is 15,000. That�s definitely more than 8,000 and less than 24,734. So that�s the
answer. Are there any others? Let me check the list again.
Stadium 3= are all 8,000. Then Stadium 6 is 7,700 (below). Then the next stadiums after that are
all below 8,000. So The Oval is the only one that fits. So the answer should be The Oval. </think>
<answer> The Oval </answer>

25

Preprint.

Table 19: A task example of TAT-QA.

Question: What was the percentage change in equity settled share plans in 2019 from 2018?
Ground Truth: 8.77%
Table:
| 2019 | 2018
| �m | �m
Adjusted operating profit | 282.7 | 264.9
Depreciation and amortisation (excluding IFRS16 depreciation) | 34.3 | 32.9
Cash payments to pension schemes in excess of charge to P&L | (5.2) | (4.6)
Equity settled share plans | 6.2 | 5.7
Working capital changes | (21.4) | (22.5)
Cash generation | 296.6 | 276.4
Response:
<think> Okay, let�s see. The question is asking for the percentage change in equity settled share
plans from 2018 to 2019. Alright, so first, I need to find the values for those two years from the
table.
Looking at the table, the row for Equity settled share plans has values of 6.2 for 2019 and 5.7 for
2018. So the 2018 value is 5.7 million pounds, and the 2019 value is 6.2 million.
To calculate the percentage change, the formula is [(New Value - Old Value)/Old Value] * 100.
Here, the new value is 2019�s 6.2, and the old value is 2018�s 5.7.
Subtracting 5.7 from 6.2 gives 0.5. Then, divide that by the old value, 5.7. So 0.5 divided by
5.7. Let me compute that. 0.5 / 5.7 equals approximately 0.0877. Multiply by 100 to get the
percentage, which is about 8.77%.
Wait, let me check the calculation again. 0.5 divided by 5.7. Let�s do 0.5 � 5.7. 5.7 times 0.08 is
0.456, and 0.5 minus that is 0.044. So 0.044 divided by 5.7 is approximately 0.0077. So total is
0.0877, which is 8.77%. Rounding to two decimal places, that�s 8.77%.
But sometimes percentage changes are rounded to one decimal. Let me confirm. The problem
doesn�t specify, so maybe keep two decimals. So the answer would be approximately 8.77%. Let
me make sure I didn�t mix up the years. 2019 is the new value, 2018 the old. Yes, correctly. So the
percentage increase is 8.77%. </think>
<answer> 8.77% </answer>

26

Preprint.

Table 20: A task example of FEVEROUS.

Claim: Pescara Calcio a 5, which is based in Pescara, Italy, a city that was founded in 1811, won
two games and lost six games in the 2015/2016 UEFA Futsal Cup.
Ground Truth: REFUTES
Table:
Season | Competition | Round | Opponent | Result
2015/16 | UEFA Futsal Cup | Main Round | Romania City�US T�argu Mures, | Win 8�3
- | - | - | Germany Hamburg Panthers | Win 11�0
- | - | - | Ukraine FC Lokomotiv Kharkiv | Win 5�1
- | - | Elite Round | Belgium FP Halle-Gooik | Win 4�2
- | - | - | Kazakhstan Tulpar Karagandy | Win 4�0
- | - | - | North Macedonia KMF Zelezarec Skopje | Win 2�0
- | - | Final Four | Spain Inter FS | Loss 2�4
- | - | - | Portugal SL Benfica | Loss (p) 2�2 (0�2)
2017/18 | UEFA Futsal Cup | Main Round | Kazakhstan Kairat Almaty | Loss 1�5
- | - | - | Belarus Stalitsa Minsk | Loss 4�6
- | - | - | Russia Dinamo Moskva | Win 5�3
- | - | Elite Round | Serbia Ekonomac | Loss 2�3
- | - | - | Netherlands ZVV �t Knooppunt | Win 9�1
- | - | - | Spain Barcelona | Loss 1�3
Response:
<think> Okay, let�s try to figure out the correct label for this claim. The claim is about Pescara
Calcio a 5 winning two games and losing six in the 20152016 UEFA Futsal Cup. The table
provided is for table 1, which lists their competitions.
First, let me look at the 201516 UEFA Futsal Cup season in the table. The main round shows
they played Romania City�US T�argu Mures, and won 8-3. Then they played Germany Hamburg
Panthers and won 11-0, and Ukraine FC Lokomotiv Kharkiv and won 5-1. That�s three wins
so far. Then in the Elite Round, they played Belgium FP Halle-Gooik and won 4-2, Kazakhstan
Tulpar Karagandy 4-0, and North Macedonia KMF Zelezarec Skopje 2-0. That�s three more wins.
Then in the Final Four, they lost to Spain Inter FS 2-4 and Portugal SL Benfica, with a loss (p) 2-2
(0-2). So in the Final Four, they had two losses.
So tallying the wins: main round 3, elite round 3, total 6 wins. Losses: Final Four 2. But the claim
says they won two games and lost six. Wait, that�s the opposite of what the table shows. The
table shows 6 wins and 2 losses in the 201516 UEFA Futsal Cup. The claim states the opposite.
But wait, the user also mentioned the text cell (Pescara Calcio a 5 R1C1): Win 8�3. Looking at
the table, the main round�s first game against Romania City�US T�argu Mures, was a win 8-3. So
maybe the text cell is referring to that same game? But the claim�s numbers are reversed. If the
actual results were 6 wins and 2 losses, but the claim says 2 wins and 6 losses, then that would be
REFUTES.
Another thing to check: Does the text�s mention of �Win 8�3� match any of their games? Yes, in
the main round against Romania City�US T�argu Mures, , they did win 8-3. But the claim is about
the total wins and losses in the season, not a single game. So even if the team did have some
games with high scores, the overall record is six wins and two losses. Therefore, the claim is
REFUTED by the table. </think>
<answer> REFUTES </answer>

27

Preprint.

Table 21: A task example of HiTab.

Question: what was the percentage of acquaintances to their victims among all females accused
of sexual assault in 2015?
Ground Truth: 53.5
Table:
Table title: violent crime perpetrated by females, by relationship of the accused to the victim,
2015
Type of Violent Crime | Current/Former Spouse | % | Intimate Relationship | % | Other Family |
% | Acquaintance | % | Stranger | % | Total | %
Homicide | 10 | 23.8 | 0 | 0.0 | 16 | 38.1 | 13 | 31.0 | 3 | 7.1 | 42 | 100.0
Attempted Murder | 9 | 23.1 | 10 | 25.6 | 6 | 15.4 | 8 | 20.5 | 6 | 15.4 | 39 | 100.0
Other Violations Causing Death | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 2 | 40.0 | 3 | 60.0 | 5 | 100.0
Sexual Assaults - All Levels | 8 | 4.0 | 23 | 11.6 | 43 | 21.7 | 106 | 53.5 | 18 | 9.1 | 198 | 100.0
Other Sexual Violations | 0 | 0.0 | 9 | 10.1 | 28 | 31.5 | 42 | 47.2 | 10 | 11.2 | 89 | 100.0
Sexual Violations Specific to Children | 0 | 0.0 | 8 | 12.9 | 18 | 29.0 | 27 | 43.5 | 9 | 14.5 | 62 | 100.0
Sexual Violations Not Specific to Children | 0 | 0.0 | 1 | 3.7 | 10 | 37.0 | 15 | 55.6 | 1 | 3.7 | 27 | 100.0
Assault - Level 3 - Aggravated | 45 | 16.7 | 93 | 34.4 | 42 | 15.6 | 68 | 25.2 | 22 | 8.1 | 270 | 100.0
Assault - Level 2 - Weapon/Bodily Harm | 1310 | 22.9 | 1476 | 25.8 | 985 | 17.2 | 1452 | 25.3 | 506 |
8.8 | 5729 | 100.0
Assault - Level 1 - Common | 4227 | 18.3 | 4910 | 21.3 | 4400 | 19.1 | 7540 | 32.7 | 1993 | 8.6 | 23070 |
100.0
Other Assaults | 11 | 0.7 | 13 | 0.9 | 18 | 1.2 | 439 | 28.9 | 1038 | 68.3 | 1519 | 100.0
Forcible ConfinementKidnapping | 19 | 13.7 | 25 | 18.0 | 64 | 46.0 | 18 | 12.9 | 13 | 9.4 | 139 | 100.0
Robbery | 1 | 0.3 | 18 | 4.6 | 9 | 2.3 | 109 | 27.9 | 254 | 65.0 | 391 | 100.0
Total | 6227 | 16.5 | 7343 | 19.5 | 6401 | 17.0 | 13340* | 35.3 | 4426 | 11.7 | 37737 | 100.0
Response:
<think> Okay, let�s try to figure this out. The question is asking for the percentage of acquain-
tances among all females accused of sexual assault in 2015. Looking at the table, I need to find
the row that corresponds to sexual assault levels.
First, I�ll scan through the �type of violent crime� column to find entries related to sexual
assault. There are several entries here. Let me check each one. The first sexual-related crime is
�sexual assaults - all levels�. Then there�s �other sexual violations�, �sexual violations specific to
children�, and �sexual violations not specific to children�. Then there�s �assault...� which might
not be sexual, so probably not relevant here. The question specifically mentions �sexual assault�,
so I should focus on �sexual assaults - all levels�.
In that row, the columns are divided by relationships: currentformer spouse, intimate relationship,
other family, acquaintance, stranger, and total. Each has a number and a percent. The question
is about the percentage of acquaintances. So for �sexual assaults - all levels�, the acquaintance
percentage would be the value in that column.
Looking at the acquaintance column for that row: the number is 106, and the percent is 53.5. The
total for that row is 198, which adds up (106 is half of 198). So the answer should be 53.5% here.
But wait, let me make sure there�s no other sexual assault categories. The next entries are other
sexual violations, which might be separate. But since the question says �sexual assault�, maybe
only the �sexual assaults - all levels� is the correct row. The other entries under sexual violations
might be subcategories, so including them would change the total. But the question is about all
females accused of sexual assault, so the main category here is �sexual assaults - all levels�, and
the percentage for acquaintances is 53.5%.
So the answer should be 53.5%. </think>
<answer> 53.5 </answer>

28

Preprint.

Table 22: A case of HybridQA.

Question: What is the team most recently managed by the 2012 manager of Real Garcilaso ?
Ground Truth: Alfonso Ugarte
Table:
Table Introduction: The 2012 Torneo Descentralizado de F �utbol Profesional (known as the 2012
Copa Movistar for sponsorship reasons) is the ninety-sixth season of Peruvian football. A total of
16 teams are competing in the tournament, with Juan Aurich as the defending champion. The
Torneo Descentralizado began on February 19 and is scheduled to end on December 9, 2012.
Team | Manager | Captain | Kit manufacturer | Shirt sponsor
Alianza Lima | Jos�e Soto | Juan Jayo | Nike |
Cienciano | Raul Arias | Julio Garcia | Aries | Movistar
Cobresol | Octavio Vidales | Gregorio Bernales | Loma �s | Universidad Alas Peruanas
Inti Gas | Edgar Ospina | Nick Montalva | Walon | Inti Gas
Jos�e G�alvez | Javier Arce | Marco Ruiz | Real | SIDERPERU Megaplaza
Juan Aurich | Franco Navarro | Luis Guadalupe | Walon | Grupo Oviedo
Le �on de Hu�anuco | Jean Ferrari | Carlos Zegarra | Walon | Roky �s
Melgar | Julio Zamora | Antonio Meza Cuadra | Marathon |
Real Garcilaso | Freddy Garc�?a | Ram �on Rodr�?guez | Walon | I-RUN
Sport Boys | Jorge Espejo | Jorge Huam�an | Triathlon | Gobierno Regional del Callao
Sport Huancayo | Wilmar Valencia | Sergio Ibarra | Manchete | Caja Municipal de Huancayo
Sporting Cristal | Roberto Mosquera | Erick Delgado | Umbro | Cerveza Cristal Volkswagen
Uni �on Comercio | Mario Viera | H�ector Sosa | Real | New Holland
Universidad C�esar Vallejo | V�?ctor Rivera | Carlos Galv�an | Walon | Universidad C�esar Vallejo
Universidad San Mart�?n | Angel Cappa | Leao Butr �on | Umbro | Herbalife
Universitario | Nolberto Solano | John Galliquio | Umbro | Radio Exitosa
Related Text Information: Real Garcilaso: Cusco FC ( known as Real Garcilaso until 2019
) is a Peruvian football club based in the city of Cusco . Freddy Garc�?a (football manager):
Freddy Manuel Petroleo Garc�?a Loayza ( born 22 November 1959 ) is a Peruvian foot-
ball manager who most recently was the manager of Alfonso Ugarte in the Copa Per �u .
Ram �on Rodr�?guez (footballer): Ram �on Rodr�?guez del Solar ( born 8 September 1977 in Pil-
copata , Cuzco ) is a Peruvian footballer who plays as a striker for Santa Rosa in the Torneo
Descentralizado . In his long career , Rodr�?guez has played for the likes of Cienciano , FBC
Melgar , Total Clean , Deportivo Municipal , Sport Boys , Alianza Atl�etico , Inti Gas Deportes
, and Cobresol and Real Garcilaso . His nicknamed is El Rat �on ( The Mouse ) Walon Sport:
Walon Sport , or simply Walon , is a multinational textile business founded in 1989 in Peru . It
is one of the principal sportswear manufacturers in Peru , and it provides the kits for various
association football clubs in the Liga 1 ( Peruvian first division ) . The company further expanded
its production when it acquired deals with clubs in the Colombian Categor�?a Primera A. Walon
Sport has also been the sponsor of the Peru national football team from 1998 to 2010 .
Response:
<think> Okay, let�s see. The question is asking: What is the team most recently managed by the
2012 manager of Real Garcilaso?
First, I need to figure out who the manager of Real Garcilaso was in 2012. Looking at the table
provided under the 2012 Torneo Descentralizado section, Real Garcilaso�s manager is listed as
Freddy Garc�?a. So the 2012 manager of Real Garcilaso is Freddy Garc�?a.
The related text in-
Next, I need to find out what team he most recently managed.
formation includes a section about Freddy Garc�?a. Let me check that. The text says:
�Freddy Garc�?a (football manager): ... most recently was the manager of Alfonso Ugarte in
the Copa Per �u.� So according to this, after his time with Real Garcilaso in 2012, his most recent
managerial position was with Alfonso Ugarte, and that was in the Copa Per �u. The Copa Per �u is
a different league, probably a lower division or a regional league. Since the question is about
the most recent team he managed, it would be Alfonso Ugarte. But wait, I need to make sure
there�s no other team mentioned after Real Garcilaso in his managerial history. The text states
�most recently was the manager of Alfonso Ugarte�, which implies that�s the latest one before the
information was written. The text might not list all his past teams, just the latest. So the answer
should be Alfonso Ugarte. </think>
<answer> Alfonso Ugarte </answer>

29


