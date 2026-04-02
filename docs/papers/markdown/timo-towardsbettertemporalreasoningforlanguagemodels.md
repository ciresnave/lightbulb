4
2
0
2

g
u
A
9
1

]
L
C
.
s
c
[

2
v
2
9
1
4
1
.
6
0
4
2
:
v
i
X
r
a

Published as a conference paper at COLM 2024

TIMO: Towards Better Temporal Reasoning for Language Models

Zhaochen Su1?, Jun Zhang1, Tong Zhu1, Xiaoye Qu2, Juntao Li1�, Min Zhang1, Yu Cheng3
1Institute of Computer Science and Technology, Soochow University, China
2Shanghai AI Laboratory
3The Chinese University of Hong Kong
{suzhaochen0110,junzhang20030309}@gmail.com;
{ljt,minzhang}@suda.edu.cn; quxiaoye@pjlab.org.cn;
tzhu1997@outlook.com; chengyu@cse.cuhk.edu.hk

Abstract

Reasoning about time is essential for Large Language Models (LLMs) to under-
stand the world. Previous works focus on solving specific tasks, primarily on
time-sensitive question answering. While these methods have proven effective,
they cannot generalize to a wider spectrum of temporal reasoning tasks. Therefore,
we propose a crucial question: Can we build a universal framework to handle
a variety of temporal reasoning tasks? To that end, we systematically study 38
temporal reasoning tasks. Based on the observation that 19 tasks are directly
related to mathematics, we first leverage the available mathematical dataset to set
a solid foundation for temporal reasoning. However, the in-depth study indicates
that focusing solely on mathematical enhancement falls short of addressing pure
temporal reasoning tasks. To mitigate this limitation, we propose a simple but
effective self-critic temporal optimization method to enhance the model�s tempo-
ral reasoning capabilities without sacrificing general task abilities. Finally, we
develop TIMO, a model designed to excel in temporal reasoning at the 7B and
13B scales. Notably, TIMO outperforms the counterpart LLMs by 10.0 and 7.6 in
average accuracy scores and achieves the new state-of-the-art (SOTA) performance
of comparable size. Extensive experiments further validate our framework�s effec-
tiveness and its generalization across diverse temporal tasks. The code is available
at https://github.com/zhaochen0110/Timo.

1

Introduction

Large Language Models (LLMs) have achieved remarkable success in various reasoning tasks (Zhou
et al., 2022; Zhao et al., 2023; Chang et al., 2023), such as mathematical, commonsense, and symbolic
reasoning. Despite these advances, LLMs face significant challenges in temporal reasoning (Chen
et al., 2021; Tan et al., 2023a), which is crucial in human perception. Compared to other reasoning
tasks that focus solely on one specific reasoning ability, temporal reasoning is an integrated task that
requires arithmetic (Zhu et al., 2023a), logic (Mishra et al., 2022a) and world knowledge (Wei et al.,
2022).

Prior efforts to improve the temporal reasoning capacity of LLMs focus mainly on time-sensitive
question-answering (Chen et al., 2021), and utilize methods such as step-by-step reasoning (Zhu et al.,
2023b; Li et al., 2023) and ruled-based supervised fine-tuning (SFT) (Tan et al., 2023a; Yuan et al.,
2023b). More recent studies expand the scope of temporal tasks to include basic temporal concepts
understanding (e.g., duration), intricate temporal interpretations (e.g., relation) and computations (e.g.,
arithmetic) (Wang & Zhao, 2023). Due to their task-specific nature, the aforementioned methods
exhibit limited generalization across the wider spectrum of temporal tasks.

To address these limitations, we explore a crucial question: Can we build a universal framework
to handle a variety of temporal reasoning tasks? To tackle this question, we face the following
challenges: (1) integrating different temporal reasoning tasks into a unified framework; (2) generating

?

Work was done during the internship at Shanghai AI lab.
�Juntao Li is the Corresponding Author.

1

Published as a conference paper at COLM 2024

Figure 1: The detailed classification of 38 tempo-
ral reasoning tasks. 19 tasks are directly related
to mathematics (i.e., Math-time tasks).
and selecting the high-quality training dataset automatically; (3) improving the comprehensive
temporal reasoning abilities while maintaining its general performance.

Figure 2: TIMO outperforms LLAMA in all
temporal tasks and is the current state-of-the-
art (SOTA) model of comparable size.

In response to these challenges, we first systematically study 38 subtasks within the temporal reason-
ing benchmark proposed by Wang & Zhao (2023). As shown in Figure 1, our analysis reveals that 19
tasks are directly related to mathematical reasoning (i.e., mathematical time tasks). For example, when
�identifies the next leap year following 2024�, mathematical skills are required to calculate the results.
The rest are categorized as pure temporal reasoning tasks, focusing solely on temporal reasoning
without using mathematical abilities. Meanwhile, mathematical reasoning stands out with its diverse
and rich instruction tuning datasets compared to temporal reasoning (Cobbe et al., 2021; Mishra et al.,
2022b; Yue et al., 2023). Therefore, it is intuitive to build a generalist temporal reasoning framework
based on math-enhanced LLMs, setting a solid foundation for temporal reasoning skills. However,
our in-depth study indicates that focusing solely on mathematical enhancement through supervised
fine-tuning falls short of addressing pure-time tasks. To bridge this gap, we further introduce a simple
but effective method to obtain comprehensive temporal reasoning abilities. Specifically, we propose a
self-critic method to generate and select the high-quality temporal preference pairs, which are then
utilized for enhancing model temporal capabilities through preference optimization. Finally, we
propose a unified temporal reasoning framework, namely TIMO. With this framework, our model
achieves superior performance among 38 temporal tasks, as depicted in Figure 2.

In our experiments, we train LLAMA2 models at both 7B and 13B scales with our framework, which
results in TIMO-7B and TIMO-13B. These two models demonstrate a substantial improvement of
10.0 and 7.6 in average accuracy scores over the base models, respectively. Our comprehensive
analysis indicates that our framework successfully integrates substantial mathematical knowledge
along with temporal information. Extensive experiments further verify the effectiveness of our method
in preserving general task capabilities and maintaining robustness under different scenarios. To sum
up, our contributions are shown below: (1) we systematically study diverse temporal reasoning tasks
and discover the inner correlation between time and mathematics, where temporal reasoning could
benefit from math instructions; (2) we make the first attempt to build a unified framework to address
38 temporal tasks. Specifically, upon mastering mathematical reasoning capabilities, we propose a
simple but effective self-critic temporal optimization method to strengthen the temporal reasoning
capabilities comprehensively; (3) the proposed framework outperforms 10.0 and 7.6 scores over the
baselines, establishing as the new SOTA model of comparable sizes. Besides, our models consistently
enhance the temporal reasoning capabilities without sacrificing general task performance.

2 Revealing the Correlation between Mathematics and Temporal Reasoning

2.1 Analysis on Temporal Reasoning Benchmark

Wang & Zhao (2023) provides a comprehensive collection of 38 subtasks centered around temporal
reasoning tasks. It is widely observed that a substantial portion of these tasks relies on mathe-

2

Math-time tasksPure-time tasksAmbiguityArithmeticDurationFrequencyAmbiguityCausalityDurationFrequencyNliOrderingRelationStorytellingTypical timeShift-LTShift-MTShift-CalendarShift-STHour Adjust(24h)Time Zone ConverTime ComputMonth ShiftWeek IdentiHour Adjust(12h)Year ShiftApplicationDate ComputMulti-Step ComparComputationFactsComparisonApplicationComputationInterpretationEffectCauseCommonsenseFactsDirect ComparReadingAnalogy InferCommonsenseReadingNliFactsCommonsenseRelationStorytellingReadingFactsCommonsenseComparison86.077.487.595.555.044.044.071.061.363.958.581.082.092.078.290.597.587.079.578.083.065.066.376.585.589.5AmbiguityDurationFrequencyCausalityNliOrderingRelationStorytellingAmbiguity (Math)Arithmetic (Math)Duration (Math)Frequency (Math)Typical timeLLama-13BTimo-13BPublished as a conference paper at COLM 2024

Figure 3: Performance comparison with
MATH-COT and traditional prompting meth-
ods in math-time tasks.

Figure 4: Comparisons on temporal tasks
with models trained on different numbers of
math instructions.

matical skills for calculating and reasoning about time. For example, within the Frequency
category, the Computation subtask requires the calculation of event frequencies or intervals. In
the Ambiguity Resolution task, mathematics provides a standardized method of time repre-
sentation, such as the 24-hour format and date calculation formulas, enabling different temporal
expressions to be accurately understood and converted. Based on these observations, we categorize
temporal tasks into two categories. The specific subtasks under each category are shown in Figure 1.
Below is our classification:

� Mathematical Time Tasks (Math-time tasks): These are temporal reasoning tasks that necessitate
mathematical skills, such as calculating time frequencies, converting time shifts, comparing time
sequences and so on. This category encompasses a total of 19 subtasks.

� Pure Time Tasks (Pure-time tasks): These tasks require only temporal reasoning abilities for
resolution and include reasoning about temporal commonsense, applications in real-world scenarios,
temporal natural language inference (NLI) and so on. This category also contains 19 subtasks.

2.2 Bridging Mathematics and Temporal Tasks

Inspired by Wei et al. (2022), we construct MATH-COT for each temporal task to establish a
connection between mathematics and temporal tasks. We utilize the MathInstruct dataset (Yue
et al., 2023), which comprises a diversified collection of mathematical problems with detailed
rationales. From this dataset, we select five mathematical question-CoT pairs and employ GPT-4 to
generate MATH-COT rationales by mimicking mathematical reasoning. Since pure-time questions
lack mathematical rationales, MATH-COT is specifically designed for math-time tasks. We compare
MATH-COT with two prompting methods: (1) FEW-SHOT, which samples five question-answer pairs
per task, and (2) COT (Wei et al., 2022), where GPT-4 is used to generate step-by-step rationales
for each task. We conduct the experiments using LLAMA2-7B under the 5-shot setting and report
the accuracy for each task. As shown in Figure 3, integrating mathematical reasoning into temporal
tasks leads to a significant enhancement in model performance, with MATH-COT outperforming
traditional prompting methods in all math-time tasks.

2.3 Mathematical Reasoning as a Foundation for Temporal Understanding

Given the established correlation between mathematics and temporal reasoning, it is intuitive to
instruct models in mastering mathematical reasoning to establish a solid foundation for advanced
temporal reasoning abilities. This connection motivates our investigation into how varying degrees of
mathematical instruction influence model performance. Specifically, we select 180k mathematical
CoT rationales from the MathInstruct and perform scaling experiments by fine-tuning the
LLAMA2-7B with different volumes of math instructions (i.e., 0, 50k, 100k, 150k, and 180k). We
evaluate the models on both math-time tasks and pure-time tasks under the 5-shot setting. The results
are shown in Figure 4. After supervised fine-tuning on 50k math instruction tuning instances, the
model exhibits a notable improvement in performing math-time tasks, with accuracy increasing from
56.4 to 63.3. However, It is worth noting that this enhancement in mathematical skills has a minimal
impact on pure-time tasks, with a maximum enhancement of 1.9. Additionally, our analysis indicates

3

Fs.CoTMATH-CoTAmb.30.5 46.8 55.0 Arith.23.7 46.2 52.1 Dur.30.5 42.5 54.0 Freq.46.0 67.0 69.5 0k50k100k150k180kMath-time tasks56.4 63.3 64.6 64.8 64.4 Pure-time tasks68.8 69.8 70.7 70.1 70.5 62.6 66.6 67.7 67.5 67.5 0k50k100k150k180kambiguity0.6040.670.6640.6580.658arithmetic0.5211111110.6133333330.641111110.62333330.6causality0.9350.9450.9450.950.955duration0.6385714290.6257142860.690.69142860.66frequency0.7383333330.760.780.7850.76333333nli0.440.50.520.430.52ordering0.420.40.450.4550.425relation0.540.540.490.490.58storytelling0.660.720.720.690.76typical_time0.72750.7650.7650.72750.715Published as a conference paper at COLM 2024

Figure 5: The pipeline of our self-critic temporal task optimization method. Based on the generated
responses by mathematical models (MathLLM), we classify correct and wrong sets using golden
answers. From these two sets, we further select the high-quality pairs with our proposed hierarchical
scoring method. Finally, the chosen pairs are utilized for DPO training.

a declining trend in performance across both task categories as the volume of math instructions
increases. We believe this decline results from overfitting to mathematical tasks due to excessive data,
adversely impacting the model�s temporal reasoning capability (Mishra et al., 2022a).

3 Self-critic Temporal Task Optimization

In the previous section, we discovered that focusing solely on mathematical enhancement falls short of
addressing pure-time tasks. To mitigate this limitation, we introduce a simple but effective self-critic
optimization framework to equip the model with comprehensive temporal reasoning abilities. The
pipeline of our proposed framework is detailed in Figure 5.
Given the mathematical model L, we start by generating a set of N candidate responses Yi =
{y1
} for each input prompt xi. Given the golden label gi for each prompt xi, we divide
Yi into the correct response set R
+
i = {yn
i

+
i and the incorrect response set R
? Yi ? align(yn

?
i :
?
i = Yi ? R

i , . . . , yN
i

i , y2

+
i ,

(1)

R

i , gi) = true}, R
i , gi) is a function that returns true if the response yn

where align(yn
i aligns with the golden label gi, and
false otherwise. Inspired by the LLM-as-a-Judge mechanism (Zheng et al., 2023; Yuan et al., 2024;
Qu et al., 2024b), we utilize mathematical model L directly as a reward model to identify high-quality
response pairs. Notably, we introduce a novel hierarchical scoring method, which is specifically
designed for evaluating responses to temporal tasks and contains five key aspects: (1) relevance and
basic temporal reasoning; (2) understanding of temporal aspects; (3) application of internal temporal
knowledge; (4) direct and well-organized addressing of the question; (5) insightfulness and advanced
?
reasoning. To choose the higher quality pair from the correct set R
i , we prioritize
the response that utilizes the model�s temporal reasoning to the fullest extent. The criteria for our
evaluation prompts are illustrated in Figure 10 and 11. For each criterion a response meets, a point is
awarded. We prompt the model L to assign a score rn
i , quantifying its
i
quality across the above dimensions.

? [0, 5] to each response yn

+
i and wrong set R

The temporal preference pair (y
+
correct set R
i as y
preference optimization (DPO) by optimizing the following loss function:

) is formed by selecting the top-scoring response from the
?
i . We then utilize these pairs to perform direct

+
i and from the incorrect set R

?
i as y

+
i , y

?
i

LDPO(??; ?ref) = ?E(x,y+

i ,y?

i

)?D [log ? (? log

?
i , and ? is a hyperparameter.

+
??(y
i
?ref(y+
i

? x)
? x)

? ? log

?
??(y
i
?ref(y?
i

? x)
? x)

)] ,

(2)

where y

+
i

is favored over y

4 Experiments

4.1 Experimental Setup

Training Setup We use LLAMA2 7B and 13B (Touvron et al., 2023) as our base pre-trained model.
For SFT, we select 100k instances from MathInstruct (Yue et al., 2023), the most representative

4

Golden AnswerCorrect SetWrong SetTemporalPreference PairsTimoDPOTrainingSelectMathLLMPure-timeTasksGenerateResponsesPairsAct as a Reward ModelPublished as a conference paper at COLM 2024

Model

Math-time Tasks

Pure-time Tasks

Average

AMB. ARITH. DUR.

FREQ. AMB. DUR.

FREQ. CAUS. NLI ORDER REL.

STORY TYP.

LLAMA2
TIMELLAMA
WIZARDCODER
CODELLAMA
WIZARDMATH
TORA
MAMMOTH

TIMO

LLAMA2
WIZARDCODER
CODELLAMA
WIZARDMATH
TORA
MAMMOTH

TIMO

55.0
52.5
53.8
44.5
63.3
45.5
62.0

65.3

61.3
58.5
57.8
58.8
56.8
64.9

65.0

52.1
42.7
51.9
55.2
52.9
44.8
52.3

60.8

63.9
60.1
60.6
58.3
50.8
67.0

66.3

54.0
42.5
40.5
50.5
45.0
44.0
54.5

59.5

58.5
55.5
61.0
62.0
48.0
71.0

76.5

69.5
55.5
66.0
68.0
74.0
69.8
59.5

72.0

81.0
72.3
74.8
75.5
75.8
79.8

85.5

7B Parameter Model

68.2
34.8
58.6
62.0
56.6
63.8
62.6

77.2

77.0
11.5
74.0
77.0
68.0
75.0
69.5

90.0

13B Parameter Model

77.4
69.0
69.4
77.2
75.8
73.0

78.2

87.5
85.0
76.0
84.0
80.5
81.5

90.5

85.0
71.0
74.0
73.0
73.0
74.0
67.0

83.0

86.0
82.0
82.0
81.0
79.0
86.0

92.0

93.0
42.5
86.0
86.0
94.0
92.0
90.5

95.0

95.5
92.0
90.5
95.0
97.5
97.0

97.5

44.0
15.0
43.0
55.0
49.0
48.0
39.0

74.0

55.0
54.0
54.0
58.0
56.0
62.0

87.0

43.0
7.5
38.0
44.0
39.0
39.5
43.0

71.5

44.0
49.0
46.5
43.0
38.5
48.0

79.5

54.0
48.0
45.0
45.0
36.0
46.0
51.0

70.0

44.0
51.0
53.0
54.0
64.0
57.0

78.0

66.0
5.0
54.0
55.0
63.0
68.0
69.0

87.0

71.0
59.0
58.0
82.0
80.0
77.0

83.0

72.5
32.0
65.8
68.3
64.3
73.3
67.3

83.8

82.0
71.3
69.5
77.3
79.5
78.8

89.5

62.7
38.6
57.8
59.8
59.9
58.2
60.0

72.7

70.7
65.9
65.7
68.4
65.6
72.1

78.3

Table 1: Results on 38 temporal tasks. For clarity, these tasks are grouped and displayed based on
their associated time-related domains. The abbreviations AMB., ARITH., DUR., FREQ., CAUS.,
REL., TYP. refer to ambiguity resolution, arithmetic, duration, frequency, causality, relation, and
typical time. All values are percentages. Best results are in bold and the second results are underlined.

dataset for mathematical reasoning instruction tuning. For self-critic temporal optimization, we
focus on pure temporal reasoning tasks, which encompass 19 subtasks. We reserve 100 instances for
evaluation and utilize the remaining data for training. If a subtask contains fewer than 5,000 samples,
we maintain all of them. Otherwise, we randomly select 5,000 instances. In total, we use 35,655
instances for optimization.

Evaluation Setup We conduct a comprehensive evaluation across all temporal reasoning tasks,
encompassing a total of 38 tasks. Following Tan et al. (2023a), we assess the model performance on
100 examples for each task, amounting to a total of 3,800 instances. Consistent with prior work (Qu
et al., 2024a; Xia et al., 2024), we evaluate the model�s temporal abilities under the 5-shot setting
and utilize greedy decoding (i.e., temperature = 0) for generating model�s responses. We extract the
prediction from the response and calculate the accuracy of each subtask.

Implementation Details We utilize four/eight NVIDIA Tesla A100 GPUs to train models. To
facilitate parallel training, we employ DeepSpeed Zero-Stage 3 (Ren et al., 2021) and FlashAtten-
tion2 (Dao, 2023). For SFT, we use a learning rate of 2e-5, a batch size of 128, and a cosine scheduler
with a 3% warm-up period for 2 epochs. For candidate response generation, we sample N = 5
candidate responses with temperature T = 0.8, p = 0.95. When evaluating candidate responses, as
there is variance to these scores, in our experiments we also use sampled decoding (with the same
parameters) and generate these evaluations multiple (3) times and take the average. For DPO, we
follow the hyper-parameters from Tunstall et al. (2023) with a batch size 32, learning rate 5e-7, a
warm ratio of 0.1 using linear warmup scheduler for 9 epochs.

4.2 Baselines

To ensure the fairness of the experiments, we select baseline models built upon the foundational
model LLAMA2. The baselines are selected based on the following dimensions:

� LLMs for Temporal Reasoning: TIMELLAMA (Yuan et al., 2023b) is currently the only open-
source model that is specifically designed for temporal reasoning. It is developed to make temporal
predictions and generate time-related explanations.

� LLMs for Mathematical Reasoning: TIMO is trained through temporal optimization based on
mathematical models. Here, we compare the following mainstream mathematical models: (1)
MAMMOTH (Yue et al., 2023) is designed for general mathematics problem-solving and is trained
on the MathInstruct dataset. (2) WIZARDMATH (Luo et al., 2023a) utilizes the proposed
Reinforcement Learning from Evol-Instruct Feedback (RLEIF) (Xu et al., 2023) to enhance its
mathematical reasoning capabilities. (3) TORA (Gou et al., 2024), a series of Tool-integrated
Reasoning LLM Agents, is designed to solve challenging mathematical reasoning problems.

5

Published as a conference paper at COLM 2024

Figure 6: Performance of GPT series and our framework�s models. MATHLLAMA is based on
mathematical instruction tuning and TIMO is our final model. Math-time tasks are marked with �,
while others are pure-time tasks. We highlight our model�s achievements: a green star (?) where
TIMO beats GPT-3.5, and a red star (?) for surpassing GPT-4.

� LLMs for Code Generation: Previous work indicates that the usage of code enhances the model�s
ability to solve reasoning tasks (Gao et al., 2023). We select the following popular code models as
our baselines: (1) CODELLAMA (Roziere et al., 2023), a family of LLMs for code generation
and programming-related tasks. (2) WIZARDCODER (Luo et al., 2023b) is similar to WizardMath
and adapts the RLEIF method within the domain of coding.

4.3 Main Results

Table 1 presents the results of TIMO across 38 temporal tasks. From the results, we observe: (1)
TIMO surpasses counterpart LLMs in average accuracy of 10.0 and 7.6 scores, and outperforms other
competitive math-solving and code-solving models with a clear margin, achieving the SOTA results on
average. We also discover TIMO-7B consistently outperforms all 13B models in average performance,
achieving a maximum performance gain of 7.1. (2) Mathematical models do not show significant
advantages in solving math-related tasks. This phenomenon is also observed in LLMs enhanced for
coding abilities and temporal prediction capabilities. It indicates that excessive training on specific
abilities leads the model to overfit in task-centric enhancements, diminishing its performance in
other areas (Jha et al., 2023). (3) It is worth noting that TIMO underperforms MAMMOTH in the
Arithmetic task (i.e., scoring 66.3 vs 67.0) when evaluated under the 13B model size parameter.
The superior performance of MAMMOTH can be attributed to its advanced general math-solving
abilities, which facilitate more accurate computations in time-related scenarios. However, other
mathematical models like TORA and WIZARDMATH do not achieve the same effectiveness in
handling the Arithmetic task. A detailed case study for illustration is in Appendix B.

5 Further Analysis on our Framework

In our framework, we initially train a mathematical model, i.e., MATHLLAMA. Then, we optimize
its pure temporal reasoning abilities to derive the final TIMO model. In this section, we first compare
the performance of these two stages. Then, we delve into these models through the lens of token
distribution shift and detailed case analysis.

Ablation Analysis of Framework We compare the model�s performance on both math-time
tasks and pure-time tasks. The results are shown in Figure 6. Compared to the foundational
model LLAMA, MATHLLAMA demonstrates superior performance in math-related tasks and
surpasses the LLAMA in the majority of pure-time tasks, achieving higher scores in 6 out of
9 tasks. This improvement is attributed to the advanced logical and reasoning skills developed
through mathematical instruction tuning (Mishra et al., 2022a). When compared to TIMO and
MATHLLAMA, our framework demonstrates strong generalization capabilities, achieving significant
improvement in pure-time tasks, with only minimal performance degradation in the arithmetic

6

Published as a conference paper at COLM 2024

Figure 7: Token distribution shift on different stages of our framework. The ratios of unshifted,
marginal, and shifted tokens are colored (%). Frequently shifted tokens are shown below, where
math-related tokens are labeled in red and time-related tokens are shown in green.

task. Additionally, it is worth noting that TIMO outperforms MATHLLAMA in various math-time
tasks (i.e., Ambiguity Resolution, Duration and Frequency). This improvement is
attributed to our framework�s ability to learn generalized temporal features.

Token Distribution Shift Analysis To understand the learning process and the differences between
the different stages of our framework, we follow the methodology proposed by Lin et al. (2024) to
analyze through the lens of token distribution shift. We analyze three pairs of models at the 7B scale:
LLAMA vs MATHLLAMA, MATHLLAMA vs TIMO, and LLAMA vs TIMO. The results are shown
in Figure 7. Notably, we observe the largest token distribution shift when transitioning from LLAMA
to TIMO. Furthermore, we investigate the top 200 most frequently shifted tokens, labeling math-
related tokens in red and time-related tokens in green. The transition from LLAMA to MATHLLAMA
primarily features changes in math-related tokens. Conversely, the switch from MATHLLAMA to
TIMO is characterized by the presence of time-related tokens. When compared to LLAMA, TIMO
exhibits shifts in both math-related and time-related tokens, demonstrating a profound capacity to
integrate substantial mathematical knowledge along with the temporal information.

Case Analysis As shown in Table 2, we present a case analysis to provide a clear and intuitive
demonstration of TIMO�s superior performance. In math-time tasks, both MATHLLAMA and TIMO
effectively integrate temporal knowledge with computational capabilities to give the correct CoT and
answer. However, LLAMA produces an incorrect result due to the error in time calculation, which
indicates the importance of mathematical skills in solving math-time tasks. In our provided case of
the pure-time tasks, both MATHLLAMA and LLAMA fail to grasp the sequence of events, i.e., the
timing of Amy�s laundry activities. On the other hand, TIMO demonstrates a strong understanding and
application of temporal reasoning, accurately tracking the sequence and timing of Amy�s activities
and giving the correct answer. Overall, these cases vividly demonstrate TIMO�s comprehensive
capabilities in temporal reasoning across different temporal task types.

6 More Detailed Study

Performance Comparison between TIMO and OpenAI GPT Models We compare TIMO-
13B with the current most powerful LLMs, i.e., GPT-3.5 and GPT-4. Specifically, we use the
gpt-3.5-turbo-1106 and gpt-4-1106-preview and set the temperature to 0 for consis-
tent evaluation. The results are shown in Figure 6. Despite its relatively small size of 13B parameters,
TIMO demonstrates impressive performance on pure-time tasks, outperforming GPT-3.5 in 7 out of
9 tasks and surpassing GPT-4 in 5 out of 9 tasks. Notably, TIMO exceeds GPT-4 by a significant
margin of 38 accuracy scores in the Relation task. Although there has been a significant improve-
ment in pure-time tasks, the performance on math-time tasks suggests that there is still room for
further enhancement. This is attributed to the foundational model�s weaker mathematical reasoning
capabilities. We leave it as future work to further improve the model�s temporal reasoning abilities by
better integrating mathematics capabilities.

7

  Unshifted (68.87%)  Marginal (15.62%)  Shifted (15.51%)    'The',  'answer',  'Therefore',  �So�,  �Original�,  'calculation',  'add',  'subtract',  'frequency',  'days',  'calculate',  'approximately',  'total',  'count',  �ג,  'Added',  �match�,   'convert',   �Sub�,  'Adding',  �equals�,  �exceed�,  �*�,  ...�The',  'answer',  'calculation',  add', 'duration',  'frequency',  'day',  �days�,  'year',  'total',  �Since�,  'hours',  'subtract',  �ג,  �AM�, 'approximately', 'weeks',  �months�,  'minutes',  �-�,  �equals�,  'adding',  �=�,  �convert�,  'sum' ...�calculation',  'duration',  'frequency',  'align',  �count�,  �total�,  'hours',  �once�,  'days',  �day�,  'time',  'years', 'context',  'minutes',  �subtract�,  'weeks',  'AM',  'july',  'months',  �Since�,  'next',  �President�,  'PM',  �following�,  �clock�...      LLaMA           TimoUnshifted (66.68%)  Marginal (16.22%)  Shifted (17.10%)       MathLLaMA          TimoUnshifted (68.62%)  Marginal (15.32%)  Shifted (16.06%)       LLaMA         MathLLaMAPublished as a conference paper at COLM 2024

Questions

LLAMA

MATHLLAMA

TIMO

Questions

LLAMA

MATHLLAMA

TIMO

In which week of year 1652 does the date 10-27-1652 occur?
(A) Week 43; (B) Week 39; (C) Week 46; (D) Week 30

Mathematical Time Tasks

Calculating the weeks from January 1st to October 27th, it�s 27 days. Dividing 27 by 7 places October 27th in the 4th week.
However, depending on the week starting day and system used, it can be considered as the 39th week. The answer is (B). (cid:37)
We calculate the total days from January 1st to October 27th, which amounts to
31 (Jan.) + 28 (Feb.) + 31 (Mar.) + 30 (Apr.) + 31 (May.) + 30 (Jun.) + 31 (Jul.) + 31 (Aug.) + 30 (Sept.) + 27 (Oct.),
which equals 300 days. Dividing 300 by 7, we find that October 27th falls into the 43rd week of the year. The answer is (A).(cid:33)
Calculating the weeks from January 1st to October 27th, the total days are
31 (Jan.) + 28 (Feb.) + 31 (Mar.) + 30 (Apr.) + 31 (May.) + 30 (Jun.) + 31 (Jul.) + 31 (Aug.) + 30 (Sept.) + 27 (Oct.),
summing up to 300 days. Dividing 300 by 7 places October 27th in the 43rd week of the year. The answer is (A). (cid:33)

Pure Time Tasks

According to the Premise: Amy started doing her laundry early in the morning every weekend. answer the question:
What�s the more plausible CAUSE?
(A) It allowed her to have the washing machines to herself as most people do laundry later in the day.
(B) Amy found more lost socks in the laundry room.

The more plausible cause is that she found more lost socks in the laundry room.
This explanation aligns with the purpose of doing laundry,
which is to clean and organize clothing, and is a logical response to Amy�s need to find lost socks. The answer is (B). (cid:37)
The more plausible cause is that she found more lost socks in the laundry room.
This explanation aligns with the purpose of doing laundry, which is to clean and organize clothes,
and the additional socks found in the laundry room would be a logical outcome. The answer is (B). (cid:37)
The reason is that it allowed her to have the washing machines to herself as most people do laundry later in the day.
This explanation aligns with the practicality of early morning laundry,
as it minimizes the likelihood of encountering other people using the laundry facilities. The answer is (A). (cid:33)

Table 2: A case study to intuitively show the effectiveness of Timo with temporal direct preference
optimization. The right and wrong steps are colored by blue and red, respectively.

MATH-TIME

Performance Comparison among Differ-
ent Rewarding Strategies
In our frame-
work, we design a series of criteria to
assess the standard of responses and ob-
tain high-quality temporal preference pairs.
To verify the effectiveness of our criteria,
we compare our prompting approach with
the widely adopted self-rewarding strat-
egy (Yuan et al., 2024) and the random selection strategy. As shown in Table 3, our strategy
outperforms others in both math-time and pure-time tasks, highlighting its superiority in evaluating
the quality of generated responses across different types of temporal challenges.

Table 3: Performance on different rewarding methods

RANDOM
LLM-JUDGE
TIMO

61.5
61.3
63.9

79.8
80.2
81.5

PURE-TIME

Robustness across Mathematical Models With TIMO being derived from a mathematical model
trained with 100k math instructions, we validate the robustness and adaptability of our framework
across different mathematical models, which is achieved by implementing self-critic temporal task
optimization in models supervised fine-tuned with different volumes of instruction dataset (i.e.,
50k, 100k, 150k, 180k). The results are shown in Figure 8. The experiments show that the trained
models consistently outperform in handling time-related tasks compared to their corresponding
mathematical models, highlighting our method�s capability to enhance temporal reasoning across
different mathematical training backgrounds.

General Tasks Capability Assessment To verify the model�s ability to retain its original capa-
bilities, we utilize the lm-evaluation-harness (Gao et al., 2021) to evaluate its performance on six
typical downstream tasks: 5-shot MMLU (Hendrycks et al., 2020), 25-shot ARC Challenge (Clark
et al., 2018), 5-shot GSM8K (Cobbe et al., 2021), 10-shot HellaSwag (Zellers et al., 2019), 5-shot
Winogrande (Sakaguchi et al., 2021) and 0-shot TruthfulQA (Lin et al., 2022). In addition to com-
paring with LLAMA and MATHLLAMA, we introduce TIMO-SFT, which mirrors our framework
in all aspects except for its training methodology. Specifically, TIMO-SFT is supervised fine-tuned
with the chosen responses in the selected preference pairs. The results are shown in Figure 9. We
surprisingly discover that TIMO outperforms other baselines in the reasoning and general knowledge
ability tasks. Error analysis shows that our model aligns with the base model for 97% of the correct
responses. This consistency indicates that our TIMO effectively preserves general task knowledge,
demonstrating remarkable generalization capabilities.

8

Published as a conference paper at COLM 2024

Figure 8: Results of TIMO trained on the math
dataset of different sizes, demonstrating con-
sistent improvements across models.

Figure 9: Reasoning and general knowledge
performance comparison under current main-
stream benchmarks.

7 Related Work

Temporal Reasoning in LLMs Time is a crucial dimension in our physical world (Lazaridou
et al., 2021; Su et al., 2022; 2023; Zhao et al., 2024). Despite the advanced capabilities of LLMs
in various tasks, their reasoning abilities are still underdeveloped (Su et al., 2024; Qiao et al., 2023;
Huang & Chang, 2023; Sun et al.; Chu et al., 2023). Temporal reasoning, which is fundamental
for humans to understand the world, is an important task in reasoning and has gained substantial
research focus (Pustejovsky, 2003; UzZaman et al., 2012; Huang et al., 2024). However, existing
works often specialize in limited aspects of temporal reasoning, such as frequency (Zhou et al., 2019),
duration (Zhang & Choi, 2021), or event-time relations (Chen et al., 2021; Tan et al., 2023a). In
our work, we address a comprehensive scope of temporal reasoning, including various levels and
dimensions of understanding about time (Wang & Zhao, 2023). Differing from prior approaches
that rely on external knowledge (Yuan et al., 2023a; Tan et al., 2023b; Xiong et al., 2024) or impose
temporal constraints (Li et al., 2023; Zhu et al., 2023b) within a narrow sub-scope of tasks, we
propose a unified framework designed to generalize across different temporal reasoning scenarios.

Preference Optimization for LLMs Preference optimization approaches typically involve training
a fixed reward model based on preference data, and then utilizing the reward model to train via
reinforcement learning (RL) (Schulman et al., 2017; Ziegler et al., 2019; Stiennon et al., 2020;
Bai et al., 2022). To simplify this process, Direct Preference Optimization (DPO) (Rafailov et al.,
2023) is introduced to avoid training the reward model entirely, and instead directly train the LLM
using preference pairs. Building on this approach, recent works explore automatic optimization and
self-correction in LLMs (Pan et al., 2023; Ji et al., 2024). This involves two key steps: instructing
LLMs to self-generate their training dataset (Wang et al., 2023; Taori et al., 2023; Tunstall et al.,
2023; Liu et al.) and serving LLMs as reward models (Fernandes et al., 2023; Saha et al., 2023;
Dubois et al., 2024) to select high-quality data. The self-generated data optimization enables models
to iteratively improve their performance through a self-rewarding mechanism (Yuan et al., 2024).
Inspired by the above works, we introduce a self-critic temporal optimization method that leverages
the inherent capabilities of the model itself to achieve significant improvements in all temporal tasks.

8 Conclusion

In this paper, we consider the problem of building a universal framework to strengthen the temporal
reasoning capabilities of LLMs. Through systematic investigation, we discover a close relationship
between mathematics and temporal reasoning. Building upon this insight, we introduce a self-critic
temporal optimization method to equip the model with comprehensive temporal reasoning abilities.
The TIMO model, trained within our proposed framework, indicates significant generalizability across
38 temporal tasks, establishing as the new SOTA model of comparable sizes. Extensive experiments
further demonstrate the effectiveness of our framework in maintaining general task abilities.

9

MathLLaMA-7BTimo-7B50k66.670.7100k67.772.7150k67.472.5180k67.572.3+4.1+5.0+5.1+4.846.652.514.778.639.074.270.746.253.924.578.941.674.372.946.353.021.578.542.774.269.947.257.925.379.846.875.878.3MMLUARCGSM8KHellaSwagTruthfulQAWinoGranadeTemporalLLaMAMathLLaMATimo-SFTTimoPublished as a conference paper at COLM 2024

Acknowledgement

We want to thank all the anonymous reviewers for their valuable comments. We are also grateful to
Wei Liu for his insightful suggestions during the mathematical reasoning experiments. This work was
supported by the National Science Foundation of China (NSFC No. 62206194), the Priority Academic
Program Development of Jiangsu Higher Education Institutions, the Natural Science Foundation of
Jiangsu Province, China (Grant No. BK20220488), and Young Elite Scientists Sponsorship Program
by CAST (2023QNRC001).

References

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain,
Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with
reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner,
Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jeff Wu. Weak-to-
strong generalization: Eliciting strong capabilities with weak supervision, 2023. URL https:
//arxiv.org/abs/2312.09390.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan
Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and
Xing Xie. A survey on evaluation of large language models, 2023.

Wenhu Chen, Xinyi Wang, and William Yang Wang. A dataset for answering time-sensitive questions.
In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks
Track (Round 2), 2021. URL https://openreview.net/forum?id=9-LSfSU74n-.

Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng,
Ming Liu, Bing Qin, and Ting Liu. A survey of chain of thought reasoning: Advances, frontiers
and future. arXiv preprint arXiv:2309.15402, 2023.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge,
2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John
Schulman. Training verifiers to solve math word problems, 2021.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. 2023.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos
Guestrin, Percy S Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for
methods that learn from human feedback. Advances in Neural Information Processing Systems,
36, 2024.

Patrick Fernandes, Daniel Deutsch, Mara Finkelstein, Parker Riley, Andr� FT Martins, Graham
Neubig, Ankush Garg, Jonathan H Clark, Markus Freitag, and Orhan Firat. The devil is in the
errors: Leveraging large language models for fine-grained machine translation evaluation. arXiv
preprint arXiv:2308.07286, 2023.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster,
Laurence Golding, Jeffrey Hsu, Alain Le Noac�h, Haonan Li, Kyle McDonell, Niklas Muennighoff,
Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika,
Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot
language model evaluation, September 2021. URL https://doi.org/10.5281/zenodo.
5371628.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and

Graham Neubig. Pal: Program-aided language models, 2023.

10

Published as a conference paper at COLM 2024

Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Minlie Huang, Nan Duan,
and Weizhu Chen. ToRA: A tool-integrated reasoning agent for mathematical problem solving.
In The Twelfth International Conference on Learning Representations, 2024. URL https://
openreview.net/forum?id=Ep0TtjVoap.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob
Steinhardt. Measuring massive multitask language understanding. In International Conference on
Learning Representations, 2020.

Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A
survey.
In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of the
Association for Computational Linguistics: ACL 2023, pp. 1049�1065, Toronto, Canada, July
2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.67. URL
https://aclanthology.org/2023.findings-acl.67.

Rikui Huang, Wei Wei, Xiaoye Qu, Wenfeng Xie, Xianling Mao, and Dangyang Chen. Joint
multi-facts reasoning network for complex temporal question answering over knowledge graph,
2024.

Aditi Jha, Sam Havens, Jeremy Dohmann, Alex Trott, and Jacob Portes. Limit: Less is more for

instruction tuning across evaluation paradigms, 2023.

Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan,
Zhonghao He, Jiayi Zhou, Zhaowei Zhang, Fanzhi Zeng, Kwan Yee Ng, Juntao Dai, Xuehai Pan,
Aidan O�Gara, Yingshan Lei, Hua Xu, Brian Tse, Jie Fu, Stephen McAleer, Yaodong Yang, Yizhou
Wang, Song-Chun Zhu, Yike Guo, and Wen Gao. Ai alignment: A comprehensive survey, 2024.

Angeliki Lazaridou, Adhi Kuncoro, Elena Gribovskaya, Devang Agrawal, Adam Liska, Tayfun Terzi,
Mai Gimenez, Cyprien de Masson d�Autume, Tomas Kocisky, Sebastian Ruder, et al. Mind the gap:
Assessing temporal generalization in neural language models. Advances in Neural Information
Processing Systems, 34:29348�29363, 2021.

Xingxuan Li, Liying Cheng, Qingyu Tan, Hwee Tou Ng, Shafiq Joty, and Lidong Bing. Unlocking

temporal question answering for large language models using code execution, 2023.

Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu,
Chandra Bhagavatula, and Yejin Choi. Urial: Aligning untuned LLMs with just the �write� amount
of in-context learning.
In The Twelfth International Conference on Learning Representations,
2024. URL https://openreview.net/forum?id=wxJ0eXwwda.

Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human
falsehoods. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings
of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pp. 3214�3252, Dublin, Ireland, May 2022. Association for Computational Linguis-
tics. doi: 10.18653/v1/2022.acl-long.229. URL https://aclanthology.org/2022.
acl-long.229.

Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. What makes good data for
In The

alignment? a comprehensive study of automatic data selection in instruction tuning.
Twelfth International Conference on Learning Representations.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng,
Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical
reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583,
2023a.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing
Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with
evol-instruct, 2023b.

Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tan-
may Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, and Ashwin Kalyan. LILA:
In Yoav Goldberg, Zornitsa Kozareva,
A unified benchmark for mathematical reasoning.

11

Published as a conference paper at COLM 2024

and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural
Language Processing, pp. 5807�5832, Abu Dhabi, United Arab Emirates, December 2022a.
Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.392. URL
https://aclanthology.org/2022.emnlp-main.392.

Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral,
and Ashwin Kalyan. NumGLUE: A suite of fundamental yet challenging mathematical reasoning
tasks. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the
60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 3505�3523, Dublin, Ireland, May 2022b. Association for Computational Linguistics. doi:
10.18653/v1/2022.acl-long.246. URL https://aclanthology.org/2022.acl-long.
246.

Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang.
Automatically correcting large language models: Surveying the landscape of diverse self-correction
strategies. arXiv preprint arXiv:2308.03188, 2023.

J Pustejovsky. The timebank corpus. In Proceedings of Corpus Linguistics 2003, pp. 647�656, 2003.

Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei
Huang, and Huajun Chen. Reasoning with language model prompting: A survey. In Anna Rogers,
Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), pp. 5368�5393, Toronto,
Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.
294. URL https://aclanthology.org/2023.acl-long.294.

Xiaoye Qu, Qiyuan Chen, Wei Wei, Jishuo Sun, and Jianfeng Dong. Alleviating hallucination in
large vision-language models with active retrieval augmentation, 2024a. URL https://arxiv.
org/abs/2408.00555.

Xiaoye Qu, Mingyang Song, Wei Wei, Jianfeng Dong, and Yu Cheng. Mitigating multilingual
hallucination in large vision-language models, 2024b. URL https://arxiv.org/abs/
2408.00550.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea
Finn. Direct preference optimization: Your language model is secretly a reward model.
In
Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://
openreview.net/forum?id=HPuSIXJaa9.

Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia
Zhang, Dong Li, and Yuxiong He. {ZeRO-Offload}: Democratizing {Billion-Scale} model
training. In 2021 USENIX Annual Technical Conference (USENIX ATC 21), pp. 551�564, 2021.

Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi
Adi, Jingyu Liu, Tal Remez, J�r�my Rapin, et al. Code llama: Open foundation models for code.
arXiv preprint arXiv:2308.12950, 2023.

Swarnadeep Saha, Omer Levy, Asli Celikyilmaz, Mohit Bansal, Jason Weston, and Xian Li.
Branch-solve-merge improves large language model evaluation and generation. arXiv preprint
arXiv:2310.15123, 2023.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An
adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99�106,
2021.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy

optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford,
Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in
Neural Information Processing Systems, 33:3008�3021, 2020.

12

Published as a conference paper at COLM 2024

Zhaochen Su, Zecheng Tang, Xinyan Guan, Lijun Wu, Min Zhang, and Juntao Li. Improving temporal
generalization of pre-trained language models with lexical semantic change. In Proceedings of the
2022 Conference on Empirical Methods in Natural Language Processing, pp. 6380�6393, 2022.

Zhaochen Su, Juntao Li, Zikang Zhang, Zihan Zhou, and Min Zhang. Efficient continue train-
ing of temporal language model with structural information. In Houda Bouamor, Juan Pino,
and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP
2023, pp. 6315�6329, Singapore, December 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.findings-emnlp.418. URL https://aclanthology.org/2023.
findings-emnlp.418.

Zhaochen Su, Juntao Li, Jun Zhang, Tong Zhu, Xiaoye Qu, Pan Zhou, Yan Bowen, Yu Cheng, et al.
Living in the moment: Can large language models grasp co-temporal reasoning? arXiv preprint
arXiv:2406.09072, 2024.

Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jiaqi Liu, Jiaqi Xu, Mingyu
Ding, Hongyang Li, Mengzhe Geng, et al. A survey of reasoning with foundation models:
Concepts, methodologies, and outlook.

Qingyu Tan, Hwee Tou Ng, and Lidong Bing. Towards benchmarking and improving the tem-
poral reasoning capability of large language models.
In Anna Rogers, Jordan Boyd-Graber,
and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp. 14820�14835, Toronto, Canada, July
2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.828. URL
https://aclanthology.org/2023.acl-long.828.

Qingyu Tan, Hwee Tou Ng, and Lidong Bing. Towards robust temporal reasoning of large language
models via a multi-hop qa dataset and pseudo-instruction tuning. arXiv preprint arXiv:2311.09821,
2023b.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy
Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model.
https://github.com/tatsu-lab/stanford_alpaca, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation
and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada,
Shengyi Huang, Leandro von Werra, Cl�mentine Fourrier, Nathan Habib, et al. Zephyr: Direct
distillation of lm alignment. arXiv preprint arXiv:2310.16944, 2023.

Naushad UzZaman, Hector Llorens, James F. Allen, Leon Derczynski, Marc Verhagen, and James
Pustejovsky. Tempeval-3: Evaluating events, time expressions, and temporal relations. CoRR,
abs/1206.5333, 2012. URL http://arxiv.org/abs/1206.5333.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and
Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In
Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13484�
13508, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/
2023.acl-long.754. URL https://aclanthology.org/2023.acl-long.754.

Yuqing Wang and Yun Zhao. Tram: Benchmarking temporal reasoning for large language models.

arXiv preprint arXiv:2310.00835, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed H Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In Advances
in Neural Information Processing Systems, 2022.

Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, and Huaxiu Yao.
Rule: Reliable multimodal rag for factuality in medical vision language models, 2024. URL
https://arxiv.org/abs/2407.05131.

13

Published as a conference paper at COLM 2024

Siheng Xiong, Ali Payani, Ramana Kompella, and Faramarz Fekri. Large language models can learn

temporal reasoning. arXiv preprint arXiv:2401.06853, 2024.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin
Jiang. Wizardlm: Empowering large language models to follow complex instructions, 2023.

Chenhan Yuan, Qianqian Xie, Jimin Huang, and Sophia Ananiadou. Back to the future: Towards
explainable temporal reasoning with large language models. arXiv preprint arXiv:2310.01074,
2023a.

Chenhan Yuan, Qianqian Xie, Jimin Huang, and Sophia Ananiadou. Back to the future: Towards

explainable temporal reasoning with large language models, 2023b.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason

Weston. Self-rewarding language models, 2024.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen.
Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint
arXiv:2309.05653, 2023.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a
machine really finish your sentence?
In Anna Korhonen, David Traum, and Llu�s M�rquez
(eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics,
pp. 4791�4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi:
10.18653/v1/P19-1472. URL https://aclanthology.org/P19-1472.

Michael Zhang and Eunsol Choi. SituatedQA: Incorporating extra-linguistic contexts into QA. In
Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), Proceedings
of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7371�7387,
Online and Punta Cana, Dominican Republic, November 2021. Association for Computational
Linguistics. doi: 10.18653/v1/2021.emnlp-main.586. URL https://aclanthology.org/
2021.emnlp-main.586.

Bowen Zhao, Zander Brumbaugh, Yizhong Wang, Hannaneh Hajishirzi, and Noah A. Smith. Set the

clock: Temporal alignment of pretrained language models, 2024.

Jinman Zhao and Xueyan Zhang. Exploring the limitations of large language models in compositional

relation reasoning, 2024. URL https://arxiv.org/abs/2403.02615.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min,
Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv
preprint arXiv:2303.18223, 2023.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on
Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https:
//openreview.net/forum?id=uccHPGDlao.

Ben Zhou, Daniel Khashabi, Qiang Ning, and Dan Roth. �going on a vacation� takes longer than
�going for a walk�: A study of temporal commonsense understanding. In Kentaro Inui, Jing Jiang,
Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods
in Natural Language Processing and the 9th International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), pp. 3363�3369, Hong Kong, China, November 2019. Association
for Computational Linguistics. doi: 10.18653/v1/D19-1332. URL https://aclanthology.
org/D19-1332.

Yucheng Zhou, Xiubo Geng, Tao Shen, Guodong Long, and Daxin Jiang. Eventbert: A pre-trained
model for event correlation reasoning. In Proceedings of the ACM Web Conference 2022, pp.
850�859, 2022.

14

Published as a conference paper at COLM 2024

Xinyu Zhu, Junjie Wang, Lin Zhang, Yuxiang Zhang, Yongfeng Huang, Ruyi Gan, Jiaxing Zhang,
and Yujiu Yang. Solving math word problems via cooperative reasoning induced language models.
In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4471�
4485, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/
2023.acl-long.245. URL https://aclanthology.org/2023.acl-long.245.

Xinyu Zhu, Cheng Yang, Bei Chen, Siheng Li, Jian-Guang Lou, and Yujiu Yang. Question answering
as programming for solving time-sensitive questions. arXiv preprint arXiv:2305.14221, 2023b.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul
Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv
preprint arXiv:1909.08593, 2019.

15

Published as a conference paper at COLM 2024

A Prompt

Our rewarding prompts are shown in Figure 10 and 11. The prompts for different temporal tasks can
be found in our public GitHub repository: https://github.com/zhaochen0110/Timo.

B Comparative Analysis of Mathematical Models on Arithmetic Tasks

We observe that task-specific fine-tuning approaches compromise the LLMs� original abilities.
We conduct a case study to delve deeper into this phenomenon. As shown in Table 4, TORA
and WIZARDMATH have difficulty grasping basic concepts of time. Specifically, these models
demonstrate challenges in accurately converting between the 12-hour and 24-hour time formats, a
fundamental aspect of temporal understanding. This case study serves as a clear illustration of how
specialized fine-tuning can compromise the LLMs� inherent ability to perform fundamental reasoning,
underscoring the need for a balanced approach in model training.

Questions

TIMO

MAMMOTH

WIZARDMATH

TORA

What is 02:45 PM + 10:44?
(A) 2:39 AM (B) 3:41 AM (C) 1:29 AM (D) 11:20 PM
The calculation is 14 ? 45 + 10 ? 44 = 25 ? 29, which is 1:29 AM. The answer is (C).
The calculation is 14 ? 45 + 10 ? 44 = 25 ? 29, which is 1:29 AM. The answer is (C).
The calculation is 15 ? 45 + 10 ? 44 = 26 ? 29, which is 3:41 AM. The answer is (B).
The calculation is 14 ? 45 + 10 ? 44 = 25 ? 29, which is 3:41 AM. The answer is (B).

Table 4: The case chosen from the Hour Adjustment (12h) task. TORA and WIZARDMATH
fall short in time calculation and converting between 12-hour and 24-hour formats.

C Iterative Optimization Study

Recent work (Touvron et al., 2023; Yuan et al., 2024) suggests that updating preference data through
multiple iterative rounds enhances the performance of preference optimization. Therefore, we explore
Iterative DPO to refine alignments across temporal reasoning tasks. The results are shown in Table 5.
However, we do not observe a significant improvement from iterative training. The reason might
be due to the efficiency of our method, where a single iteration is sufficient for robust learning, and
excessive training could instead diminish performance in temporal reasoning tasks.

MATH-TIME

PURE-TIME

1 ITER.
2 ITERS
3 ITERS

63.9
62.1
57.8

81.5
80.9
80.1

Table 5: Comparison on different iteration settings

D Validating TIMO on LLAMA3-8B

To further validate the effectiveness of TIMO in enhancing temporal reasoning across different LLMs,
we conducted additional experiments using the LLAMA3-8B model. The results are shown in Table 6.
Compared to vanilla LLAMA3-8B, TIMO shows an average improvement of 5.1 scores, with 1.2
scores in math-related tasks and 9 scores in time-related tasks. These consistent improvements across
both the LLAMA2 and LLAMA3 series demonstrate TIMO�s strong generalization capabilities
across different model series, enhancing its applicability and effectiveness in diverse settings.

E Evaluating the Impact of Math LLM on Temporal Reasoning

Existing work on weak-to-strong generalization suggests that distilling data from a weaker or
equivalent LLM benefits a stronger LLM (Burns et al., 2023). To address concerns regarding the

16

Published as a conference paper at COLM 2024

MATH-TIME

PURE-TIME AVERAGE

LLAMA3-8B
+TIMO

81.4
82.6

79.6
88.6

80.5
85.6

Table 6: Performance Comparison of LLAMA3-8B with and without TIMO enhancement

influence of the LLM-as-Judge framework compared to the use of a specialized math LLM, we
conducted experiments using vanilla LLAMA2-7B and LLAMA2-7B-CHAT, representing general
SFT LLAMA models. As presented in Table 7, our results demonstrate that incorporating a math
LLM yields significant improvements in temporal reasoning tasks. Specifically, the math LLM
outperforms the vanilla LLAMA2-7B and LLAMA2-7B-CHAT models by an average of 3.6 and
7 scores, respectively. The performance gains are especially notable in math-related tasks, where
the math LLM achieves scores 5.5 and 10.5 scores higher than those of the other two models.
These results indicate that the math LLM component is crucial for enhancing temporal reasoning
capabilities, outperforming the self-critic temporal optimization (i.e., LLM-as-Judge) framework
alone. The results indicate that math-specific training plays a pivotal role in reasoning over time,
confirming the value of specialized LLMs in complex reasoning tasks.

MATH-TIME

PURE-TIME AVERAGE

TIMO (LLAMA2-7B)
TIMO (LLAMA2-7B-CHAT)
TIMO (MATHLLAMA-7B)

58.4
53.4
63.9

79.7
78.1
81.5

69.1
65.7
72.7

Table 7: Comparison of temporal reasoning performance across different based LLM, with TIMO
applied for temporal optimization on LLAMA2-7B, LLAMA2-7B-CHAT, and MATHLLAMA-7B.

F Further Evaluation of TIMO on Temporal Reasoning Datasets

To further assess TIMO�s improvements in temporal reasoning, we extended our evaluation to
additional temporal reasoning datasets, i.e., MCTACO (Zhou et al., 2019) and TempReason (Tan
et al., 2023a). These datasets were selected to validate TIMO�s effectiveness across a broader range
of temporal reasoning tasks.

� MCTACO: This dataset evaluates a wide range of commonsense knowledge related to events,

including the duration, frequency, order, stationary nature, and typical timing of events.

� TempReason: This dataset emphasizes implicit temporal reasoning in structured facts, focusing on

both event-time reasoning and event-event reasoning.

The results are shown in Table 8. TIMO achieves scores 6.2 and 15.5 points higher than LLAMA2-7B
and WIZARDMATH-7B on the TempReason task. Additionally, TIMO surpasses MAMMOTH-7B by
19.3 points on the MCTACO task. These results indicate that TIMO excels across various temporal
reasoning datasets, demonstrating its robust general temporal reasoning abilities. In future work, We
will further explore the generalization of TIMO across more reasoning tasks, such as commonsense
reasoning (Sakaguchi et al., 2021), and composition relations reasoning (Zhao & Zhang, 2024).

MCTACO TempReason

LLAMA2-7B
MAMMOTH-7B
WIZARDMATH-7B
TIMO-7B

50.3
37.0
12.7
56.3

46.6
52.8
37.3
52.8

Table 8: Results on the MCTACO and TempReason datasets. TIMO-7B outperforms its counterparts,
demonstrating superior general temporal reasoning abilities.

17

Published as a conference paper at COLM 2024

Figure 10: The prompt for our LLM to act as a chosen reward model

Figure 11: The prompt for our LLM to act as a rejected reward model

18

Use the following 5-point scoring system to review the user's question and the corresponding correct answer:- Add 1 point if the response is relevant to the user's inquiry and shows some temporal reasoning ability, even if it is incomplete or contains irrelevant content.- Add another point if the response correctly understands the temporal aspects of the user's question but doesn't fully resolve the query or provide a direct answer.- Award a third point if the response accurately uses internal temporal knowledge to effectively answer the basic elements of the user's question.- Grant a fourth point if the response not only effectively uses temporal reasoning but is also clearly written from an AI Assistant�s perspective, addressing the user's question directly and comprehensively, and is well-organized.- Bestow a fifth point for a response that excellently applies temporal reasoning, reflecting expert knowledge in time-based queries, and demonstrates a high-quality, engaging, and insightful answer.User: { prompt }<response> { response } </response>After examining the user�s instruction and the response:- Conclude with the score using the format: �Score: <total points>�.Use the following 5-point scoring system to review the user's question and the corresponding incorrect answer that attempts to use temporal knowledge but fails to correctly solve the problem:- Add 1 point if the response attempts to be relevant to the user�s inquiry and shows an attempt at temporal reasoning, even if the information is incorrect.- Add another point if the response partially correctly attempts to handle the temporal aspects of the user's question but includes errors or misconceptions.- Award a third point if the response clearly tries to use temporal knowledge but fails to accurately address the basic elements of the user�s question.- Grant a fourth point if the response, while attempting to reason temporally from an AI Assistant�s perspective, shows some logic but contains errors or misses key information.- Bestow a fifth point for those answers that make an effort in temporal reasoning but are incorrect, potentially misleading the user but not completely deviating from the topic of the question.User: { prompt }<response> { response } </response>After examining the user�s instruction and the response:- Conclude with the score using the format: �Score: <total points>�.
