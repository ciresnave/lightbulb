Preprint

A Sober Look at Progress in Language Model Reasoning:
Pitfalls and Paths to Reproducibility

Andreas Hochlehnert1? Hardik Bhatnagar1? Vishaal Udandarao1,2?
Samuel Albanie Ameya Prabhu1� Matthias Bethge1�

1T�bingen AI Center, University of T�bingen

2 University of Cambridge

(cid:128) Leaderboard

� Code

? Eval Logs

Abstract

Reasoning has emerged as the next major frontier for language models
(LMs), with rapid advances from both academic and industrial labs. How-
ever, this progress often outpaces methodological rigor, with many evalua-
tions relying on benchmarking practices that lack transparency, robustness,
or statistical grounding. In this work, we conduct a comprehensive em-
pirical study and find that current mathematical reasoning benchmarks
are highly sensitive to subtle implementation choices�including decod-
ing parameters, random seeds, prompt formatting, and even hardware
and software-framework configurations. Performance gains reported in
recent studies frequently hinge on unclear comparisons or unreported
sources of variance. To address these issues, we propose a standardized
evaluation framework with clearly defined best practices and reporting
standards. Using this framework, we reassess recent methods and find
that reinforcement learning (RL) approaches yield only modest improve-
ments�far below prior claims�and are prone to overfitting, especially on
small-scale benchmarks like AIME�24. In contrast, supervised finetuning
(SFT) methods show consistently stronger generalization. To foster repro-
ducibility, we release all code, prompts, and model outputs, for reasoning
benchmarks, establishing more rigorous foundations for future work.

5
2
0
2

r
p
A
9

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
8
0
7
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

Figure 1: The Sombre State of LM Reasoning for Math. (left) when re-evaluating recent 1.5B
reasoning-enhanced models on AIME-24 using a standardized framework (see Section 4),
we find substantial drops to reported results in the original papers, (right) the observed
improvements from recent methods (gray highlighted area) fall entirely within the variance
range (orange box plots) of DeepSeek-R1 1.5B model performance. This suggests that these
methods do not significantly outperform the base model�underscoring the importance of
rigorous, multi-seed evaluation protocols for obtaining reliable performance estimates.

?equal contribution, ? core contributor, �equal advising

1

DeepScaleR-1.5BOpenRS3-1.5BII-1.5BFastCuRL-1.5B01020304050Accuracy (%)-6.1%-17.0%-2.2%-6.8%43.137.046.729.734.232.043.136.3Reported vs Measured ResultsSeedTop PTemperatureRange ofImprovementVariance DS-R1-1.5B

Preprint

1

Introduction

�The first principle is that you must not fool yourself, and you are the easiest person to fool.�

�Richard Feynman

Reasoning has become central to recent advances in large language models (LLMs), playing a
key role in nearly all frontier systems (Jaech et al., 2024; Anthropic, 2025; OpenAI, 2025; xAI,
2025; Meta-AI, 2025; DeepMind, 2025). Recent months have seen a surge of research focused
on understanding and improving LLM reasoning, accompanied by several open-source
tools and training strategies (see Li et al. (2025b) for a survey). This momentum has sparked
optimism that building capable, competitive reasoning models may soon be within reach.

However, as evaluation practices shape the direction and perceived progress of the field
(Liao et al., 2021), concerns around methodological rigor are growing. Non-reproducible or
inconclusive evaluations can distort scientific understanding, misguide adoption, and skew
future research priorities (Henderson et al., 2018; Marie et al., 2021; Musgrave et al., 2020;
Prabhu et al., 2020; Andrychowicz et al., 2020; Colas et al., 2018). In the fast-moving area
of LLM reasoning, where rapid publication cycles and benchmarking races are common,
methodological shortcuts can quietly undermine progress. While concerns about repro-
ducibility in LLM evaluations are well-documented (Reuel et al., 2024; Biderman et al., 2024),
their persistence�especially in reasoning�calls for renewed scrutiny and higher standards.

Motivated by a growing number of inconsistent empirical claims across the reasoning land-
scape, we conduct a rigorous investigation into the current state of reasoning benchmarks,
focusing specifically on mathematical reasoning�one of the most widely used testbeds for
evaluating algorithmic advances in this space (HuggingFaceH4, 2024; AI-MO).

Our main finding is that many recent empirical conclusions may be overly optimistic and fail
to generalize under careful re-evaluation. We identify a surprising degree of sensitivity in
LLM-based reasoning pipelines to seemingly minor design choices�ranging from decoding
parameters, prompt formatting, and random seeds to the hardware and software stacks used
during evaluation (see Table 1). Particularly concerning is the instability introduced by small
benchmark sizes: for example, AIME�24 and AMC�23 each contain only 30�40 examples,
making performance metrics highly volatile�where even one question can shift Pass@1
by over 3 percentage points. This leads to substantial variance across seeds, often resulting
in double-digit performance swings that challenge the reliability of published results. In
Section 3, we systematically analyze the root causes of this instability, including sampling
variance, decoding configurations, evaluation frameworks, and hardware heterogeneity. We
show that these factors can significantly distort conclusions if not carefully controlled.

In Section 4, we propose a set of best practices aimed at improving reproducibility and rigor
in reasoning benchmarks. We also re-evaluate recent techniques using a standardized and re-
producible evaluation stack. Our findings are sobering�reinforcement learning (RL) applied
to distillation-based models such as DeepSeek-R1 yields little to no statistically significant
gains. Some methods, such as OpenRS, show promising results in original reports, but fail
to hold up under repeated evaluation. RL training on base models like Qwen2.5 Math does
show stronger performance, but still often underperforms instruction-tuned counterparts.1
Furthermore, RL-trained models exhibit significant performance drops on newer bench-
marks such as AIME�25, echoing patterns of test set overfitting or �hill-climbing� observed
in prior work (Golchin & Surdeanu, 2023; Roberts et al., 2023; Dominguez-Olmedo et al.,
2024). In contrast, supervised fine-tuning (SFT) continues to deliver stable, generalizable
improvements across benchmarks, underscoring its maturity as a training paradigm. These
observations point to a critical need for more reliable and standardized evaluation protocols.

Taken together, in this work, we aim to provide not only a clearer assessment of where
current methods stand, but also the tools and practices needed to make reasoning evaluation
more transparent, robust, and reproducible. To this end, we open-source all code, prompts,
and outputs to facilitate fair and accountable progress in this increasingly important area.

1We note that OpenReasoner-Zero is a consistent exception, achieving competitive performance.

2

Preprint

2 Related Works

Language Model Reasoning (for Math). The recent releases of OpenAI-O1 (Jaech et al.,
2024) (in September 2024), OpenAI-O3 (OpenAI, 2025) (in December 2024) and DeepSeek-
R1 (DeepSeek-AI, 2025) (in January 2025), have spurred the language modelling community
to work on improving the reasoning capabilites of language models. Several popular meth-
ods for improving those capabilites have emerged with supervised fine-tuning (SFT) and
reinforcement learning (RL) being the two primary methods of interest (Uesato et al., 2022;
Lightman et al., 2023; Lyu et al., 2025; Team, 2025). Recent works have built upon the
DeepSeek-R1 recipe by proposing newer RL algorithms, including LCPO (Aggarwal &
Welleck, 2025), REINFORCE++ (Hu, 2025), DAPO (Yu et al., 2025), DPO-VP (Tu et al., 2025),
VinePPO (Kazemnejad et al., 2024), CPPO (Lin et al., 2025a), VAPO (Yue et al., 2025) and
GRO (Cai, 2025). To gain a stronger understanding of how to induce mathematical capabili-
ties, other works have conducted significant empirical studies exploring the design space
of RL methods (Zeng et al., 2025b; Liu et al., 2025b; Team et al., 2025; Shao et al., 2024),
including data scaling trends (Shen et al., 2025), curriculums (Wen et al., 2025b; Roux et al.,
2025) and reward design (Gao et al., 2024a; Cui et al., 2025; Ma et al., 2023). Based on the
success of these methods, there have also been recent efforts into scaling up reinforcement
learning approaches to induce reasoning in domains beyond math, including code (Liu &
Zhang, 2025; Xie et al., 2025; Jha et al., 2024; Yu et al., 2024), medicine (Zhang et al., 2025;
Sim & Chen, 2024) and other sciences (Su et al., 2025; Yuan et al., 2025; Zeng et al., 2025a).
Further, some works also explored scaling up RL-based approaches to modalities beyond
just language, including vision (Ma et al., 2025; Meng et al., 2025; Huang et al., 2025; Peng
et al., 2025; Chen et al.; Deng et al., 2025; Liu et al., 2025c; Feng et al., 2025; Lin et al., 2025b).
In our work, we objectively re-evaluate the claims made by several of these recent works
under a standardized lens, and find that many of the reported gains do not hold up strongly
when pitted on a level-playing field against well-tuned baselines.

Sobering Studies on ML Progress. Machine learning is a field of rapid progress. Due to the
lightning speed of papers coming out across the various sub-fields of machine learning, prac-
titioners and researchers often fail to rigorously evaluate algorithmic progress (Hutchinson
et al., 2022; Dehghani et al., 2021; Machado et al., 2018; Ghosh et al., 2024; Balduzzi et al., 2018;
Liao et al., 2021; Cawley & Talbot, 2010; Lipton & Steinhardt, 2019; Prabhu et al., 2024b). This
has led to several papers showing that simple well-tuned baselines outperform months of
progress on a specific sub-field in machine learning, including in continual learning (Prabhu
et al., 2024a; 2020), active learning (Cawley, 2011) and test-time adaptation (Press et al.,
2023). With the rapid influx of reasoning-based LMs, such statistically rigorous comparisons
of models are ever more important�yet, despite the heavy use of RL-algorithms for driving
progress in reasoning, there is very little mention of how different methods standardize their
evaluations across different factors of variability. RL-algorithms themselves are known to
be quite fickle to extremely minor variations including random seeds (Agarwal et al., 2021;
Gorsane et al., 2022; Chan et al., 2019; Jordan et al., 2020; Patterson et al., 2024). Some works
have even gone as far as suggesting that reliable benchmarking of RL-based methods is com-
putationally infeasible (Jordan et al., 2024). Additionally, other works have demonstrated
critical reliability issues in the generalization of frontier models to minor perturbations in
the question inputs (Mirzadeh et al., 2024; Nezhurina et al., 2024; Srivastava et al., 2024),
the type of tasks tested (Yan et al., 2025; Petrov et al., 2025; Dominguez-Olmedo et al., 2024;
Roberts et al., 2025), metrics used (Liu et al., 2024) and in data-scarce scenarios (Udandarao
et al., 2024; Kandpal et al., 2023; Parashar et al., 2024). Given such a volatile landscape,
in this work, we aim to level the playing field across recent LM-methods that have been
released and provide an objective look on the progress that the reasoning community has
made. Our findings, which we discuss in the rest of the paper, are sobering at best.

3 Exploring the Design Space of Reasoning: What Matters Most?

Recent reasoning-focused language models are evaluated under highly heterogeneous
conditions�including differences in evaluation frameworks and hardware, number of
random seeds, temperature, and nucleus sampling parameters (top_p) (see Table 1). While
prior work has examined the effect of sampling parameters in multiple-choice (Renze, 2024)

3

Preprint

Table 1: Taxonomy of current open-weight reasoning models. For each model, we report
the base model it was post-trained from and the exact type of post-training algorithm applied
(RL vs SFT). Further, we note the evaluation framework that the original paper uses for
reporting results along with the exact temperature, generation sequence length, and top_p
sampling parameters used for AIME-24 evaluation, with the number of generations used
for computing Pass@1 (K). It is evident that there is no clear standardization across different
models with respect to evaluation frameworks used and the sampling parameters. This
motivates the need to closely scrutinize the evaluations of current reasoning models.

Model

Algorithm

Base

Framework Temp Top_p Seq. Len K

DeepSeek-R1-Distill-1.5B
DeepSeek-R1-Distill-7B
DeepSeek-R1-Distill-14B
DeepSeek-R1-Distill-32B
OpenThinker-32B
Bespoke-Stratos-32B
Bespoke-Stratos-7B
s1.1-7B
s1.1-32B
LIMO
MiniMath-R1-1.5B

DeepScaleR-1.5B-Preview
Open-RS1
Open-RS2
Open-RS3
II-Thought-1.5B-Preview
Oat-Zero-1.5B
Oat-Zero-7B
STILL-3-1.5B-preview
FastCurl-1.5B-Preview
LIMR

SFT
SFT
SFT
SFT
SFT
SFT
SFT
SFT
SFT
SFT
SFT DeepSeek-R1-Distill-1.5B

Qwen2.5-Math-1.5B
Qwen2.5-Math-7B
Qwen2.5-14B
Qwen2.5-32B
Qwen2.5-32B-Instruct
Qwen2.5-32B-Instruct
Qwen2.5-7B-Instruct
lm-eval-harness
Qwen2.5-7B-Instruct
lm-eval-harness
Qwen2.5-32B-Instruct
Qwen2.5-32B-Instruct math-eval-harness
oumi-ai

RL DeepSeek-R1-Distill-1.5B
RL DeepSeek-R1-Distill-1.5B
RL DeepSeek-R1-Distill-1.5B
RL DeepSeek-R1-Distill-1.5B
RL DeepSeek-R1-Distill-1.5B
Qwen2.5-Math-1.5B
RL
RL
Qwen2.5-Math-7B
RL DeepSeek-R1-Distill-1.5B
RL DeepSeek-R1-Distill-1.5B
Qwen2.5-Math-7B
RL

� 0.6
� 0.6
� 0.6
� 0.6
evalchemy 0.7
evalchemy 0.7
evalchemy 0.7
0
0
0
�
verl 0.6
lighteval 0.6
lighteval 0.6
lighteval 0.6
evalscope 0.6
custom
0
custom
0
custom 0.6
verl 0.6
custom 0.4

0.95
0.95
0.95
0.95
0.8
0.8
0.8
�
�
1
�

0.95
0.95
0.95
0.95
0.95
1
1
0.95
0.95
0.95

32,768 64
32,768 64
32,768 64
32,768 64
5
32,768
5
32,768
32,768
5
32,768 64
32,768 64
1
32,768
�
�

32,768 16
32,768 32
32,768 32
32,768 32
32,768 64
1
3,000
1
3,000
32,768
5
32,768 16
4
3,072

and coding tasks (Arora et al., 2024), the influence of these choices remains underexplored
for open-ended reasoning models�particularly those trained with reinforcement learning.
In this section, we systematically assess how these evaluation design choices affect reported
performance, and highlight the sources of variance that most impact the reliability of results.

3.1 Experimental Setup

We adopt a consistent experimental setup throughout this section, unless otherwise stated.
Our analysis includes nine widely used models grouped into two commonly benchmarked
size classes: 1.5B and 7B parameters. For the 1.5B class, we evaluate: DeepSeek-R1-Distill-
1.5B (DeepSeek-AI, 2025), DeepScaleR-1.5B (Luo et al., 2025), II-1.5B-Preview (Intelligent
Internet, 2025) , OpenRS1-1.5B, OpenRS2-1.5B, and OpenRS3-1.5B (Dang & Ngo, 2025).
Note that DeepScaleR-1.5B, II-1.5B-Preview, and the OpenRS models are all initialized
from DeepSeek-R1-Distill-1.5B and subsequently finetuned via reinforcement learning (e.g.,
GRPO (Shao et al., 2024)) to enhance mathematical reasoning capabilities. For the 7B class,
we evaluate: DeepSeek-R1-Distill-7B, S1.1-7B (Muennighoff et al., 2025), and OpenThinker-
7B (Team, 2025). Both S1.1-7B and OpenThinker-7B are finetuned Qwen2.5-7B-Instruct
models (Yang et al., 2024a), trained using supervised learning on reasoning traces derived
from DeepSeek-R1. All models are benchmarked on three widely used datasets: AIME�24 (AI-
MO), AMC�23 (AI-MO, 2024), and MATH500 (Hendrycks et al., 2021), using the Pass@1
metric. Each result is averaged over multiple seeds and obtained on a standardized software
stack (throguh a Docker image), and hardware with the following configuration: one 40
GB A100 GPU, an AMD 7302 32-core CPU, and 1TB RAM. All experiments were run using
lighteval (Fourrier et al., 2023) with the vllm backend (Kwon et al., 2023).

Sampling Parameters: To systematically compare the impact of sampling parameters on
accuracy, all experiments in this section were performed with a standardized configuration:
temperature=0.8, top_p=0.9, and both max_model_len and max_new_tokens set to
32,768 tokens. This context length matches the limits of models such as OpenThinker-7B
and S1.1-7B, although certain models (e.g., DeepSeek) support longer sequences of up to

4

Preprint

131,072 tokens. We chose this standardized evaluation length to ensure comparability, with
a detailed analysis of the influence of completion length presented in Figure 9. Unless
otherwise specified, results in this section are averaged over 10 random seeds for AIME�24
and AMC�23, and 3 seeds for MATH500, following the recommendations from Section 3.2.1.

3.2 Seed Variance in Evaluation

Figure 2: Accuracy varies significantly across random seeds. We find significantly high
Pass@1 variation across 20 different random seeds for nine models on AIME�24, AMC�23,
and MATH500. Variance is particularly high on AIME�24 (upto 15%) and AMC�23 (upto 13%)
due to the small number of test samples, highlighting instability of single-seed evaluations.

We begin by analyzing the variance induced purely by the random seed used during
evaluation�an aspect often neglected in benchmarking practices. While recent work calls
for statistical rigor (e.g., using error bars and multiple runs) (Bowyer et al., 2025; Biderman
et al., 2024; Madaan et al.), evaluations frequently rely on single-seed runs, obscuring
potential variability. We assess the seed-induced variance across 20 independent evaluation
runs for each of the nine models. Results are shown in Figure 2.

Key Insight. Pass@1 values show surprisingly high standard deviation�ranging from 5 to
15 percentage points across seeds. This issue is particularly severe for AIME�24 and AMC�23,
which have only 30 and 40 test samples respectively. A change in just one question shifts
Pass@1 by 2.5�3.3 percentage points.

Takeaway 1 Single-seed evaluations on small datasets are highly unstable. Accurate
reporting requires averaging over multiple seeds.

Takeaway 2 Small datasets such as AIME24 (30 samples) make model comparisons
unreliable, as solving just one extra question already shifts pass@1 by 3%. Variance
from sampling parameters or random seeds can easily cause fluctuations of 1�2 correct
answers, leading to unstable rankings � especially when models cluster around 30%
performance.

3.2.1 Can Bootstrapping Improve Mean Estimates?

Figure 3: Bootstrapped seed averaging is reliable only beyond a threshold. We plot the
variance of Mean Pass@1 scores on AIME�24 when averaging over K = 1 to K = 10 seed
runs, finding that the variance is extremely high for small K and significantly reduced by
K = 10. This suggests that using multi-seed evaluations (K ? 10) would yield more stable
estimates. For results on AMC23 and MATH500 see Figures 12 and 13 respectively.

5

0.10.20.30.40.50.6AccuracyAIME240.60.70.80.9AMC230.820.840.860.880.900.920.940.96MATH500DeepScaleR-1.5BDeepSeek-R1-Distill-1.5BOpenRS1-1.5BOpenRS2-1.5BOpenRS3-1.5BII-1.5BOpenThinker-7BS1.1-7BDeepSeek-R1-Distill-7B2.55.07.510.0010Variance of MeansDeepSeek-R1-Distill-1.5B2.55.07.510.00510OpenRS3-1.5B2.55.07.510.002040DeepScaleR-1.5BPreprint

To mitigate high variance, recent work has adopted bootstrapping�averaging multiple
evaluation runs to stabilize results. For example, DeepSeek reports Pass@1 over 64 runs,
while DeepScaleR uses 16. We study the effectiveness of this approach by bootstrapping
estimates for AIME�24 using 1 to 10 evaluation runs. Figure 3 shows that while variance is
extreme for K = 1 and still large for K = 2, it reduces sharply for K ? 10. Further analysis
of variance across additional datasets is presented Figures 12 and 13.

Takeaway 3 Bootstrapping over 10 runs substantially stabilizes Pass@1 estimates and
should be considered a minimal standard for reliable evaluation.

3.2.2 Variance from Sampling Parameters: Temperature and top-p

Figure 4: Higher temperatures yield better accuracies. We find across all three datasets,
higher temperatures produce better peak accuracy but introduce instability, revealing a
tradeoff between performance and reproducibility. Results obtained by varying temperature
from 0 to 1 in increments of 0.1, while keeping top_p fixed at 0.9.

Figure 5: Higher top_p values improve performance at no cost to stability. Across all
datasets, we find that higher top_p values generally improve performance while preserving
similar amounts of variance as lower top_p values. Results were obtained by varying
top_p from 0 to 1 in increments of 0.1, while holding the temperature constant at 0.8.

Reducing the temperature or increasing the nucleus sampling parameter (top_p) improves
the accuracy of performance estimates without incurring additional computational cost.
Figure 4 illustrate the impact of temperature and Figure 5 show that of top_p across
multiple models and datasets. Notably, a more reproducible estimate is associated with
significant drops in measured performance, highlighting a consistent tradeoff between
reproducibility and high performance. We recommend optimizing the temperature for
performance, and comparing the best parameter per model.

Additionally, we investigate the impact of the temperature and top_p hyperparameter as
prior works often employ different temperature and top_p settings when comparing the
same model. To isolate the impact of varying temperature and top_p, we averaged pass@1
across seeds and compute variation of this estimate across temperature and top_p in a

6

0.00.20.40.60.81.00.10.20.30.40.50.6accuracyAIME240.00.20.40.60.81.00.50.60.70.80.9accuracyAMC230.00.20.40.60.81.00.7500.7750.8000.8250.8500.8750.9000.9250.950accuracyMATH500DeepScaleR-1.5BDeepSeek-R1-Distill-1.5BOpenRS1-1.5BOpenRS2-1.5BOpenRS3-1.5BII-1.5BOpenThinker-7BS1.1-7BDeepSeek-R1-Distill-7B0.40.50.60.70.80.91.00.20.30.40.50.6accuracyAIME240.40.50.60.70.80.91.00.50.60.70.80.9accuracyAMC230.50.60.70.80.91.00.8000.8250.8500.8750.9000.9250.950accuracyMATH500DeepScaleR-1.5BDeepSeek-R1-Distill-1.5BOpenRS1-1.5BOpenRS2-1.5BOpenRS3-1.5BII-1.5BOpenThinker-7BS1.1-7BDeepSeek-R1-Distill-7BPreprint

boxplot. Figure 6 and 7 show the performance variation. We see that temperature-induced
and top_p-induced fluctuations not only affect performance estimates but also introduce
substantial variability in performance itself, which can lead to unfair comparisons when
evaluating the same model across different temperatures.

Figure 6: Accuracies vary significantly across temperature values. Across nine different
models and three datasets, we observe consistently large variations in performance (upto
15%) induced by changing the temperature. Results were obtained by varying the tempera-
ture from 0 to 1 in increments of 0.1, while holding top_p constant at 0.9.

Figure 7: Accuracies vary significantly across top_p values. Across nine different models
and three datasets, we observe consistently large variations in performance (upto 8%)
induced by changing the top_p value. Results were obtained by varying top_p from 0 to 1
in increments of 0.1, while holding the temperature constant at 0.8.

Takeaway 4 Temperature and top_p can introduce substantial performance varia-
tion�especially on small benchmarks�and should be set to each model�s optimal values
to ensure fair and stable evaluation.

3.3 Variance from Hardware and Software Factors

Performance can also vary due to non-obvious factors like hardware and evaluation frame-
work�yet this is rarely acknowledged. Models are often tested on heterogeneous systems
and evaluated using different toolchains.For example, S-1.1 (Muennighoff et al., 2025) uses
lm-evaluation-harness (Gao et al., 2024b), the OpenRS model suite uses lighte-
val (Fourrier et al., 2023), and II-1.5B-Preview uses evalscope (Alibaba ModelScope
Community) for evaluation.

Hardware Variation. We evaluated the same model across five different compute clus-
ters, each with varying GPU types and memory configurations. As shown in Figure 8,
performance varied by up to 8% for OpenRS-1.5B and 6% for DeepSeek-R1-Distill-7B on
AIME�24, with similar trends observed on AMC�23. While it is known that inference engines
such as vLLM can be sensitive to hardware differences (vLLM Contributors, 2024)�and

7

0.20.30.40.5AccuracyAIME240.50.60.70.80.9AMC230.7500.7750.8000.8250.8500.8750.9000.9250.950MATH500DeepScaleR-1.5BDeepSeek-R1-Distill-1.5BOpenRS1-1.5BOpenRS2-1.5BOpenRS3-1.5BII-1.5BOpenThinker-7BS1.1-7BDeepSeek-R1-Distill-7B0.150.200.250.300.350.400.450.500.55AccuracyAIME240.550.600.650.700.750.800.850.90AMC230.800.820.840.860.880.900.920.94MATH500DeepScaleR-1.5BDeepSeek-R1-Distill-1.5BOpenRS1-1.5BOpenRS2-1.5BOpenRS3-1.5BII-1.5BOpenThinker-7BS1.1-7BDeepSeek-R1-Distill-7BPreprint

(a) AIME24. Significant differences are observed
in model performance across compute clusters.

(b) AMC23. Similar variability is seen across
hardware in AMC23 results.

Figure 8: Performance variation across compute clusters. Accuracy differences emerge
when the same models are evaluated across compute clusters for both AIME24 and AMC23
datasets�these large differences in performance also persist when evaluating 7B models.

that low-level optimizations in PyTorch or CUDA (PyTorch Contributors, 2024) may intro-
duce non-determinism�our results demonstrate that these effects can measurably impact
benchmark accuracy, even when averaging over multiple seeds.

Evaluation across different Python frameworks. Evaluation results can vary based on
the framework used, due to differences in prompt templates, inference engines (e.g.,
vLLM (Kwon et al., 2023)), and response extraction strategies (e.g., MathVerify). For ex-
ample: lighteval is used by OpenRS (Dang & Ngo, 2025), evalchemy (Guha et al.,
2024) is used by models like OpenThinker and Bespoke-Stratos, other frameworks include
lm-evaluation-harness (Gao et al., 2024b) and evalscope (Alibaba ModelScope Com-
munity).

To assess this impact, we compare lighteval and evalchemy, keeping all other variables
fixed: model, dataset, hardware, decoding parameters, and random seeds (3 per model). For
a fair comparison, we evaluated two models, DeepSeek-R1-Distill-1.5B and S1.1-7B, at their
default temperature and top_p parameter values on a single GPU. We present results
averaged over three seeds for higher robustness. As shown in Table 2, framework-induced
differences are generally small (1�2pp) but can still affect model rankings in tightly clustered
scenarios.

Overall, our findings underscore that significant per-
formance variations can arise solely from differences
in hardware and software configurations, emphasiz-
ing the need to standardize for reliable evaluations.

Model

lighteval evalchemy

R1-Distill-1.5B
S1.1-7B

26.6
22.2

26.6
17.7

Table 2: AIME24 across frameworks.

Takeaway 5 Re-running the exact same experimental configurations across compute
clusters and evaluation frameworks yields notably different results.

3.4 Effect of Prompt Format and Context Length

Maximum Output Tokens. Figure 9 shows that reducing max_new_tokens harms perfor-
mance�especially on long-form problems. This sensitivity varies by model and dataset.
Although reducing this setting lowers cost, it may induce premature stopping, leading to
incorrect answers.

Prompt Format. Prompt formatting has a measurable impact on accuracy. As shown in
Figure 10, models perform best when using math-specific prompts and their native chat
templates. Omitting templates leads to performance drops, particularly for instruction-
tuned models. We compare accuracy under three different prompt settings (see Table 5): (1)
a math-specific prompt formatted using the model�s chat template, (2) only the model�s chat

8

DeepScaleR-1.5BDeepSeek-R1-Distill-1.5BDeepSeek-R1-Distill-7BOpenRS1-1.5BOpenRS2-1.5BOpenRS3-1.5BOpenThinker-7BS1.1-7BII-1.5B0.200.250.300.350.400.450.500.550.60AccuracyAIME24Cluster ACluster BCluster CCluster DCluster EDeepScaleR-1.5BDeepSeek-R1-Distill-1.5BDeepSeek-R1-Distill-7BOpenRS1-1.5BOpenRS2-1.5BOpenRS3-1.5BOpenThinker-7BS1.1-7BII-1.5B0.600.650.700.750.800.850.900.95AMC23Preprint

Figure 9: Models are extremely sensitive to output token lengths. We sweep across different
max_new_tokens (number of tokens that models are allowed to generate) for DeepScaleR-
1.5B and DeepSeek-R1-Distill-1.5B/7B on three datasets and find that they are heavily
sensitive to output length limits, with premature truncation degrading the performance.

Figure 10: Using no prompt templates yields worse performance. We compare Pass@1
scores across three prompt formats: (1) math-specific prompt with chat template, (2) default
chat template only, and (3) no template. Instruction-tuned models perform best with struc-
tured prompts and templates; omitting templates leads to consistent performance drops.

template with no additional prompt, and (3) no template at all, i.e., the question without
any special tokens or instructions. Interestingly, while base models like Qwen2.5-Math may
benefit from prompt-free setups (Liu et al., 2025b), instruction-tuned models rely heavily on
format alignment. Thus, maintaining consistent and format-aware prompting is essential
for maximizing instruction-tuned model performance.

Takeaway 6 It is critical to use large generation context lengths to avoid output truncation
which can degrade performance; further, using correct prompt formats and chat templates
is important for extracting best model performance.

4 Way Forward: Standardization in Evaluations

In this section, we standardize evaluation frameworks, propose best practices, and compre-
hensively evaluate existing methods.

4.1 Recommendations: Which practices to adopt?

We propose a set of best practices informed by our experiments and guided with current
research insights:

� Hardware and Software Stack Standardization: To promote reproducibility and facilitate
future work, we release all code within a Docker container, along with step-by-step
instructions for running experiments on Runpod�s publicly accessible, on-demand GPU
instances. This setup allows any researcher to replicate and extend our results under
identical conditions.

9

40968192163843276865536131072Max New Tokens0.10.20.30.40.50.6AccuracyAIME24DeepScaleR-1.5BDeepSeek-R1-Distill-1.5BDeepSeek-R1-Distill-7B40968192163843276865536131072Max New Tokens0.40.50.60.70.80.9AMC2340968192163843276865536131072Max New Tokens0.750.800.850.900.95MATH500MathDefaultNo Template0.20.30.40.50.60.70.80.9AccuracyDeepScaleR-1.5B-PreviewAMC23MATH500AIME24MathDefaultNo Template0.20.40.60.8DeepSeek-R1-Distill-Qwen-1.5BMathDefaultNo Template0.40.50.60.70.80.91.0DeepSeek-R1-Distill-Qwen-7BPreprint

� Variance Estimates: For small benchmarks (e.g., AIME�24), run evaluations with at least
ten random seeds. Report the mean and standard deviation to quantify uncertainty and
assess the statistical significance of performance differences.

� Model-Specific Hyperparameter Optimization: Tune hyperparameters (such as tem-
perature and top_p) separately for each model, then fix them across tasks to ensure
consistency and fair comparisons.

� Context Length and Prompt Template Selection: Ensure the context length is sufficiently
large�especially for models with long reasoning chains�to avoid premature truncation
and under-reported accuracy. For instruction-tuned models, always use the appropriate
chat template to match the expected input format.

� Robust Answer Matching: We strongly recommend using a resilient answer extraction
pipeline that handles parsing issues and evaluates expression equivalence, rather than
relying on exact string matching. This reduces the likelihood of spurious gains from
formatting artifacts.

� Transparent Evaluation Protocols: We recommend to release code, prompts, and model
outputs, and clearly document the evaluation stack. Report uncertainties (e.g., via stan-
dard deviations) and include both quantitative and qualitative analyses to enable thor-
ough and reproducible comparisons.

4.2 Standardization Procedure

We adopt a largely consistent experimental setup with prior work, with the key difference be-
ing our use of publicly accessible cloud instances from Runpod2. Each instance is equipped
with a single A100 PCIe GPU, 8 vCPUs, and 128 GB of RAM. We evaluate all models listed
in Table 3 across six benchmarks: AIME�24 (AI-MO), AIME�25 (Lin, 2025), AMC�23 (Knovel
Engineering, 2025), MATH500 (HuggingFaceH4, 2024), Minerva (Lewkowycz et al., 2022),
and OlympiadBench (He et al., 2024). All experiments are conducted using the LightEval
framework (Fourrier et al., 2023) (0.8.1) with a vLLM backend, repeated across ten random
seeds for AIME�24, AIME�25, AMC�23 and three random seeds for the rest. Depending
on the base model architecture, we set the maximum number of new tokens (e.g., 4096
for QwenMath-based models), apply optimal hyperparameters, and use the appropriate
chat template. LightEval�s LaTeX-based answer extraction and evaluation pipeline ensures
reliable and consistent result parsing and correctness matching, similar to math-verify.

4.3 A Sober Look: Results

We present experimental results in Table 3, and analyze different aspects of the results.

RL-training on R1-Distill We evaluated several reinforcement learning (RL) approaches
(e.g., GRPO) using the DeepSeek R1-Distill-1.5B model. We first observe that none of the
L1 models (Aggarwal & Welleck, 2025) outperformed the original DeepSeek R1-Distill
baseline � an expected outcome given that L1 training prioritized smaller output length
over accuracy. OpenRS (Dang & Ngo, 2025) reported strong gains (10�15%) on AIME,
AMC, and OlympiadBench. However, our replication showed no statistically significant
improvements over the R1 - Distill baseline. Same case held for Still-3 and Light-R1 model,
which showed no significant improvement over the R1-Distill baseline. II-Thought and
FastCurl yield modest improvements across benchmarks, especially over AIME�24 but
the observed gains did not carry over significantly to AIME�25 indicating overfitting to
existing benchmarks. Only DeepscaleR demonstrated robust, significant improvements
across benchmarks.

Takeaway 1 Most RL-trained variants of the DeepSeek R1-Distill model do not yield
meaningful performance improvements (except DeepscaleR), suggesting that a reliable
and scalable RL training recipes are still lacking.

2https://www.runpod.io/pricing

10

Preprint

Model

AIME�24 AIME�25 AMC�23 MATH500 Minerva Olympiad

Based on: Deepseek R1 Distill Qwen 1.5B (RL)

R1-Distill (DeepSeek-AI, 2025)
28.7�4.8
L1-Exact (Aggarwal & Welleck, 2025) 24.4�3.3
27.7�4.2
L1-Max (Aggarwal & Welleck, 2025)
28.9�6.0
Open-RS1 (Dang & Ngo, 2025)
31.3�7.7
Open-RS2 (Dang & Ngo, 2025)
29.7�4.6
Open-RS3 (Dang & Ngo, 2025)
STILL-3 (Min et al., 2024)
34.7�5.5
II-Thought (Intelligent Internet, 2025) 32.0�5.9
36.3�4.3
FastCuRL (Song et al., 2025)
37.0�6.6
DeepScaleR (Luo et al., 2025)

22.3�5.2 71.5�3.9
22.3�4.2 70.5�3.7
21.0�5.0 73.2�6.0
21.3�4.2 75.0�3.3
22.7�5.6 73.0�5.7
24.7�6.5 69.2�5.5
24.0�6.4 72.5�5.4
24.0�4.1 79.5�5.1
27.0�3.7 78.8�4.1
30.3�4.3 76.2�4.6

84.9�0.3
86.6�0.8
84.7�0.1
85.1�0.8
84.1�0.2
84.2�1.1
86.6�1.9
86.6�0.6
87.9�1.2
87.8�1.0

30.5�1.0
31.5�1.7
33.3�0.9
30.4�0.2
29.2�1.1
28.6�2.3
30.0�0.6
31.7�0.6
30.8�1.4
31.0�1.5

52.4�0.4
52.5�1.3
52.3�0.6
53.2�1.9
53.7�0.6
51.8�0.8
53.9�1.5
54.9�0.4
56.5�0.6
55.5�1.1

Based on: Deepseek R1 Distill Qwen 7B (RL)

R1-Distill (DeepSeek-AI, 2025)
Light-R1 (Wen et al., 2025a)

52.3�6.3
53.0�4.8

39.0�5.9 91.5�2.7
41.0�3.5 90.0�3.1

94.1�0.3
93.5�0.5

40.1�0.4
41.3�1.3

67.3�0.1
68.0�1.2

Based on: Qwen2.5 Math 1.5B (RL)

Math (Base) (Yang et al., 2024b)
Oat-Zero (Liu et al., 2025a)
Math (Instruct) (Yang et al., 2024b)

11.3�3.6
16.0�3.2
12.0�1.7

44.0�4.9
5.7�2.7
6.7�3.4
52.5�2.9
11.7�5.7 54.8�5.3

51.7�5.5
73.5�1.7
74.7�0.5

11.3�2.2
26.3�0.8
26.7�1.8

26.0�0.6
37.2�1.3
37.9�0.2

Based on: Qwen2.5 Math 7B (RL)

Math (Base) (Yang et al., 2024b)
SimpleRL-Zoo (Zeng et al., 2025b)
LIMR (Li et al., 2025a)
Oat-Zero (Liu et al., 2025a)
Math (Instruct) (Yang et al., 2024b)

20.7�3.8
22.7�5.2
30.7�3.2
28.0�3.1
15.7�3.9

8.7�3.9
56.2�5.7
10.7�3.4 62.2�3.6
62.2�3.4
7.8�3.3
8.8�2.5
66.2�3.6
10.7�3.8 67.0�3.9

64.3�0.5
76.9�1.8
76.5�0.4
79.4�0.3
82.9�0.1

17.3�1.9
30.1�2.8
34.9�1.3
34.4�1.4
35.0�0.6

29.0�0.5
39.3�0.6
39.3�0.9
43.8�1.1
41.3�0.9

Based on: Qwen2.5 1.5B (RL)

Qwen (Base) (Yang et al., 2024a)
SimpleRL-Zoo (Zeng et al., 2025b)
Qwen (Instruct) (Yang et al., 2024a)

0.0�0.0
0.3�1.1
1.3�1.7

0.0�0.0
0.3�1.1
0.7�1.4

2.5�2.5
13.2�4.7
26.2�4.8

3.3�1.5
12.0�6.5
57.5�1.1

1.8�0.4
4.0�2.4
19.4�1.3

1.5�0.5
4.2�2.0
20.3�1.1

Based on: Qwen2.5 7B (RL)

3.3�3.3
Qwen (Base) (Yang et al., 2024a)
SimpleRL-Zoo (Zeng et al., 2025b)
14.0�2.1
Open Reasoner Zero (Hu et al., 2025) 19.7�2.9
12.3�3.2
Qwen (Instruct)

30.0�9.0
0.0�0.0
4.3�2.7
58.0�1.6
15.7�2.7 59.5�4.5
52.8�4.8
7.3�3.4

64.6�1.0
77.9�0.8
83.9�1.1
77.1�1.2

25.7�0.9
33.0�0.2
31.6�1.3
34.9�1.0

30.1�1.2
39.0�0.1
47.6�1.7
38.7�1.0

Based on: Qwen2.5 7B (SFT)

12.3�3.2
Qwen (Instruct) (Yang et al., 2024a)
17.8�2.2
Eurus2 Prime (Cui et al., 2025)
s1.1 (Muennighoff et al., 2025)
19.0�3.2
Bespoke Stratos (Bespoke Labs, 2024) 20.3�4.3
30.5�6.2
OpenThinker (Team, 2025)
48.3�8.9
OpenR1 (Face, 2025)
53.0�4.6
OpenThinker2 (Team, 2025)

7.3�3.4
52.8�4.8
14.0�1.7 63.0�3.9
21.0�5.5 59.5�3.7
18.0�4.8 60.2�4.9
26.0�4.4 71.4�3.9
35.5�4.2 86.0�4.5
41.0�5.0 87.0�3.5

77.1�1.2
80.1�0.1
80.8�0.6
84.7�0.5
88.3�1.4
�
81.6�0.7

34.9�1.0
37.5�1.0
37.5�1.1
39.1�1.3
37.9�3.8
�
33.9�0.2

38.7�1.0
43.9�0.3
48.2�1.4
51.9�1.1
55.6�1.4
�
46.9�1.3

Table 3: A Standardized and Sober Compilation of LM-Reasoning Results. We report
Pass@1 accuracy (mean � std) of all models across six math reasoning benchmarks under a
standardized evaluation setup�results are averaged over ten seeds for AIME and AMC, and
three seeds for the rest, using the LightEval framework with best hyperparameters tuned
per method, 32,768 context lengths for all except 4,096 for Math models, and appropriate
prompt templates. RL- and SFT-based variants are evaluated relative to their respective
base or instruction-tuned models. Main takeaways�(1) RL-trained methods do not yield
meaningful performance gains, (2) SFT on reasoning traces yields significant generalization.

RL Training on Qwen2.5 Math and Base Models: We next analyze RL training applied
to the Qwen2.5 Base and Qwen2.5 Math Base models, a trend trying to replicate gains by
Deepseek-R1 Zero. Unlike the R1-Distill results, RL training with Oat-Zero, LIMR, and
SimpleRL-Zoo consistently produced statistically significant gains over the base model,

11

Preprint

especially across Math500, Minerva and OlympiadBench benchmarks. This indicates that
RL-based approaches can indeed offer substantial improvements given a base model instead
of a distilled R1 model. However, these gains remained smaller than those achieved via
instruction tuning in the original Qwen papers, suggesting that instruction tuning alone
may be sufficient to far surpass current gains from RL methods in this setting. We also
observed that the improvements on AIME�24 were also significant, but did not carry over
to AIME�25 indicating a troubling overfitting trend. Notably, Open Reasoner-Zero-7B was
the only RL-trained model to consistently outperform the instruct-tuned baseline by large
margins across all benchmarks.

Takeaway 2 While RL-trained methods can often substantially improve base model per-
formance, instruction tuning remains superior (except Open Reasoner Zero), suggesting
again that a reliable and scalable RL training recipes are still lacking.

Effectiveness of Supervised Finetuning. We assessed supervised finetuning methods like
s1.1, Eurus2 Prime, Bespoke Stratos, OpenR1 and OpenThinker models, which further refine
instruction-tuned models using reasoning traces. Supervised methods consistently outper-
formed the instruct-tuned baseline across all benchmarks (even Minerva) and generalized
comparatively well to AIME�25. The performance improvements from OpenThinker were
especially notable. These results underscore the maturity and effectiveness of SFT when
training recipes are scaled to large datasets.

Takeaway 3 Supervised finetuning on reasoning traces from larger models yields sig-
nificant, generalizable gains across benchmarks with progress over time successfully
replicated � highlighting its robustness and maturity as a training paradigm.

Overfitting and Generalization We now examine the overfitting by comparing performance
on AIME�24 versus the more challenging AIME�25. RL-trained models showed a pronounced
performance drop between the two, indicating overfitting to the training distribution.
In contrast, supervised fine-tuning (SFT) models maintained consistent improvements,
suggesting better generalization. Openthinker2 showed significant degradation compared
to Openthinker across benchmarks not provided in their blogpost, indicating overfitting via
data-curation. This highlights a gap in current evaluation protocols, and a need to assess
out-of-distribution generalization for reasoning models.

Takeaway 4 Current RL-based approaches are very susceptible to overfitting, empha-
sizing the need for more rigorous out-of-distribution benchmarks. By comparison, SFT
models exhibit stronger generalization and resilience.

4.4 Do Discovered Phenomena Replicate? A Detailed Analysis.

We further investigate two recently noted phenomena to see if they replicate in our experi-
ments: (1) how response length correlates with performance, and (2) the decline in response
diversity following reasoning-focused training.

4.4.1 Are Incorrect Responses Longer?

Recent research (Wang et al., 2025) suggests that incorrect answers often have dispropor-
tionately long reasoning chains. We first verify whether this finding holds in our setting,
and then we explore possible explanations behind the observed variations.

Do longer responses indicate a higher likelihood of an incorrect answer? We compare the
distribution of response lengths for correct and incorrect answers across 6 datasets (AIME24,
AIME25, AMC23, MATH500, Minerva and OlympiadBench) averaged across random seeds
for each model. Figure 11 shows histograms of the average number of responses per seed,
binned by response length. A clear trend emerges: shorter responses are significantly more

12

Preprint

Figure 11: Response Length vs. Accuracy. Histogram of correct vs. incorrect responses by
response length, averaged over random seeds across AIME24, AIME25, AMC23, MATH500,
Minerva and OlympiadBench benchmarks. Longer outputs tend to be more error-prone,
even in complete responses not close to the maximum sequence length.

likely to be correct, while longer responses become progressively more error-prone. This
pattern is consistent across all seeds and is especially pronounced for responses exceeding
10,000 tokens. We now address two questions:

Q1. Does this pattern hold for both RL- and SFT-trained models? Yes. We find the
trend is consistent across both RL- and SFT-trained models (additional figures provided in
Appendix figures 17 and 18 ). We consistently observe that the effect is more pronounced
in RL-trained models (displayed on the left) than in SFT-trained models (displayed on the
right). As detailed in the Appendix, both the Qwen 2.5 Math base exhibit a slight shift in
length, though this shift is notably more evident in R1-distill and subsequent RL-trained
models.

Q2. Is this primarily because of truncated or incomplete responses? Although responses
nearing the 32,000-token limit are almost always incorrect (due to limited context-length),
this trend persists even for complete responses which are shorter� Longer responses are
associated with a higher likelihood of being incorrect.

Takeaway 5 Longer responses correlate with a greater chance of error, response length is
a practical heuristic for consensus@k, identifying low-confidence or failed generations.

4.4.2 Is There Diversity Collapse in Reasoning Training?

Model

Baseline

AIME�24

AIME�25

AMC�23

?@5

?@10

?@1

?@5

?@10

?@1

?@5

?@10

R1-Distill
Open-RS3
DeepScaleR R1-Distill
S1.1-7B
II-Thought

+0.4
-1.7
Qwen-Instruct +5.7 +10.9 +13.5 +11.9 +10.5 +10.4 +5.8 +9.6 +9.7
+1.2 +6.3 +0.7 +0.2
R1-Distill

-1.0
+1.5
+0.2 +4.4

-0.9
+1.4
+2.2 +6.1

+0.4
+3.6

-0.6
+0.6

-0.2
-1.8

+2.5

+0.8

+0.5

-3.5

-3.6

?@1
+1.5
+9.0

Table 4: RL-trained models do not show a diversity collapse (Dang et al.). We report the
delta between Pass@k of RL-trained models and their corresponding baselines. Unlike
reported in prior work, we observe no significant phenomenon of diversity collapse: ?@5
and ?@10 are largely positive, and are negative at similar rates as ?@1.

Dang et al. has reported a counterintuitive phenomenon in reasoning models: improvements
in Pass@1 achieved through supervised fine-tuning or RL can reduce Pass@k performance
due to diminished output diversity�a phenomenon termed diversity collapse. Theoretical

13

050001000015000200002500030000Response Length0100200300400500600Average Count per SeedLight-R1-7B-DSCorrect (1.0)Incorrect (0.0)050001000015000200002500030000Response Length025050075010001250150017502000OpenThinker-7BCorrect (1.0)Incorrect (0.0)Preprint

analyses attribute this collapse to the model concentrating too much probability mass on a
single reasoning path, while current decoding strategies fail to recover the lost diversity.

To examine these claims, we compare the Pass@k performance (for k ? 1, 5, 10) of RL-trained
models against their corresponding base models (e.g., DeepSeek-R1-Distill-Qwen-1.5B)
across all datasets. Table 4 shows the delta in Pass@k relative to each method�s base model.

Findings. We do not observe a consistent diversity collapse. Gains in Pass@1 generally come
with improvements in Pass@k, though the magnitude of these gains varies. When Pass@k
performance does drop, it does so alongside (rather than independently of) occasional
declines in Pass@1, providing no support for the diversity collapse hypothesis.

Takeaway 6 Standard decoding strategies appear sufficient to capture the model�s full
distribution over valid reasoning paths, counter to the diversity collapse hypothesis.

5 Conclusion

Our study shows that much of the perceived progress in LLM-based reasoning, particularly
in mathematical benchmarks, rests on unstable and often non-reproducible foundations.
We find that minor differences in sampling parameters, prompt formatting, hardware, and
software configurations can lead to major shifts in reported performance�casting doubt
on many recent empirical claims. Reinforcement learning methods, while promising in
theory, offer at best modest gains in practice and are prone to overfitting, especially on small
benchmarks like AIME�24. In contrast, supervised finetuning continues to deliver consistent,
generalizable improvements across a wide range of benchmarks and model sizes.

To address these challenges, we advocate for standardized, transparent evaluation protocols.
Our open-sourced framework, complete with Dockerized environments, seed-averaged met-
rics, and robust answer matching, provides reproducible foundations for future research. We
hope this work shifts the focus from leaderboard chasing to methodological rigor�ensuring
that future claims of progress in reasoning are both meaningful and measurable.

Author Contributions

Andreas, Vishaal and Ameya conceived the project. Andreas and Hardik co-led the exper-
iments, with Vishaal and Ameya advising the experimental design. The manuscript was
written by Andreas, Hardik, Vishaal and Ameya. Matthias and Samuel provided helpful
feedback and advice throughout the project.

Acknowledgments

The authors would like to thank (in alphabetical order): Matteo Farina, Shyamgopal Karthik,
Nikhil Parthasarathy, Shiven Sinha, Joschka Str�ber, Thadd�us Wiedemer for helpful feed-
back on the draft. AH acknowledges funding by the Federal Ministry of Education and
Research (BMBF), FKZ: 01IS24079A. HB has received funding from the Digital Europe
Programme under grant agreement No 101195233 (OpenEuroLLM). AH, HB and VU thank
the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for support.
VU also thanks the European Laboratory for Learning and Intelligent Systems (ELLIS) PhD
program for support. VU was supported by a Google PhD Fellowship in Machine Intelli-
gence. AP and MB acknowledge financial support by the Federal Ministry of Education
and Research (BMBF), FKZ: 011524085B and Open Philanthropy Foundation funded by the
Good Ventures Foundation. This work was supported by the Digital Europe Programme
under grant agreement No 101195233 (OpenEuroLLM).

14

Preprint

References

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C Courville, and Marc
Bellemare. Deep reinforcement learning at the edge of the statistical precipice. Advances
in neural information processing systems, 34:29304�29320, 2021.

Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks

with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.

AI-MO. AIMO Validation AIME Dataset.

AI-MO. AIMO Validation AMC Dataset. https://huggingface.co/datasets/

AI-MO/aimo-validation-amc, 2024. Accessed: 2025-03-29.

Alibaba ModelScope Community. Evalscope documentation. https://evalscope.

readthedocs.io/en/latest/. Accessed: 2025-03-29.

Marcin Andrychowicz, Anton Raichuk, Piotr Sta �nczyk, Manu Orsini, Sertan Girgin, Raphael
Marinier, L�onard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, et al.
What matters in on-policy reinforcement learning? a large-scale empirical study. arXiv
preprint arXiv:2006.05990, 2020.

Anthropic. Claude 3.7 Sonnet System Card, 2025. URL https://assets.anthropic.
com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.
pdf. Accessed: 2025-03-29.

Chetan Arora, Ahnaf Ibn Sayeed, Sherlock Licorish, Fanyu Wang, and Christoph Treude.
Optimizing large language model hyperparameters for code generation. arXiv preprint
arXiv:2408.10577, 2024.

David Balduzzi, Karl Tuyls, Julien Perolat, and Thore Graepel. Re-evaluating evaluation.

Advances in Neural Information Processing Systems, 31, 2018.

Bespoke Labs.

Bespoke-stratos-7b.
Bespoke-Stratos-7B, 2024. Accessed: 2025-03-29.

https://huggingface.co/bespokelabs/

Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, Leo Gao, Jonathan Tow, Baber
Abbasi, Alham Fikri Aji, Pawan Sasanka Ammanamanchi, Sidney Black, Jordan Clive,
et al. Lessons from the trenches on reproducible evaluation of language models. arXiv
preprint arXiv:2405.14782, 2024.

Sam Bowyer, Laurence Aitchison, and Desi R Ivanova. Position: Don�t use the CLT in LLM
evals with fewer than a few hundred datapoints. arXiv preprint arXiv:2503.01747, 2025.

Xin Cai. One framework to rule them all: Unifying rl-based and rl-free methods in rlhf.

arXiv preprint arXiv:2503.19523, 2025.

Gavin C Cawley. Baseline methods for active learning. In Active Learning and Experimen-
tal Design workshop In conjunction with AISTATS 2010, pp. 47�57. JMLR Workshop and
Conference Proceedings, 2011.

Gavin C Cawley and Nicola LC Talbot. On over-fitting in model selection and subsequent
selection bias in performance evaluation. The Journal of Machine Learning Research, 11:
2079�2107, 2010.

Stephanie CY Chan, Samuel Fishman, John Canny, Anoop Korattikara, and Sergio Guadar-
rama. Measuring the reliability of reinforcement learning algorithms. arXiv preprint
arXiv:1912.05663, 2019.

Liang Chen, Lei Li, Haozhe Zhao, and Yifan Song. Vinci. r1-v: Reinforcing super generaliza-

tion ability in vision-language models with less than 3 dollars.

C�dric Colas, Olivier Sigaud, and Pierre-Yves Oudeyer. How many random seeds? sta-
arXiv preprint

tistical power analysis in deep reinforcement learning experiments.
arXiv:1806.08295, 2018.

15

Preprint

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan,
Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards.
arXiv preprint arXiv:2502.01456, 2025.

Quy-Anh Dang and Chris Ngo. Reinforcement learning for reasoning in small llms: What

works and what doesn�t, 2025. URL https://arxiv.org/abs/2503.16219.

Xingyu Dang, Christina Baek, J Zico Kolter, and Aditi Raghunathan. Assessing diver-
sity collapse in reasoning. In Scaling Self-Improving Foundation Models without Human
Supervision.

Google DeepMind.
URL

2025.
gemini-model-thinking-updates-march-2025/. Accessed: 2025-04-07.

ai model,
https://blog.google/technology/google-deepmind/

2.5: Our most

intelligent

Gemini

DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement

Learning, 2025. URL https://arxiv.org/abs/2501.12948.

Mostafa Dehghani, Yi Tay, Alexey A Gritsenko, Zhe Zhao, Neil Houlsby, Fernando Diaz,
Donald Metzler, and Oriol Vinyals. The benchmark lottery. arXiv preprint arXiv:2107.07002,
2021.

Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Open-
vlthinker: An early exploration to complex vision-language reasoning via iterative self-
improvement. arXiv preprint arXiv:2503.17352, 2025.

Ricardo Dominguez-Olmedo, Florian E Dorner, and Moritz Hardt. Training on the test task

confounds evaluation and emergence. arXiv preprint arXiv:2407.07890, 2024.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL

https://github.com/huggingface/open-r1.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou
Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint
arXiv:2503.21776, 2025.

Cl�mentine Fourrier, Nathan Habib, Hynek Kydl�?cek, Thomas Wolf, and Lewis Tunstall.
LightEval: A lightweight framework for LLM evaluation, 2023. URL https://github.
com/huggingface/lighteval.

Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weilin Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju
Wang, and Yi Wu. On designing effective rl reward at training time for llm reasoning.
arXiv preprint arXiv:2410.15115, 2024a.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles
Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac�h, Haonan Li, Kyle McDonell,
Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf,
Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and
Andy Zou. A framework for few-shot language model evaluation, 07 2024b. URL
https://zenodo.org/records/12608602.

Adhiraj Ghosh, Sebastian Dziadzio, Ameya Prabhu, Vishaal Udandarao, Samuel Albanie,
and Matthias Bethge. Onebench to test them all: Sample-level benchmarking over open-
ended capabilities. arXiv preprint arXiv:2412.06745, 2024.

Shahriar Golchin and Mihai Surdeanu. Time travel in llms: Tracing data contamination in

large language models. arXiv preprint arXiv:2308.08493, 2023.

Rihab Gorsane, Omayma Mahjoub, Ruan John de Kock, Roland Dubb, Siddarth Singh, and
Arnu Pretorius. Towards a standardised performance evaluation protocol for cooperative
marl. Advances in Neural Information Processing Systems, 35:5510�5521, 2022.

16

Preprint

Etash Guha, Negin Raoof, Jean Mercat, Ryan Marten, Eric Frankel, Sedrick Keh, Sachin
Grover, George Smyrnis, Trung Vu, Jon Saad-Falcon, Caroline Choi, Kushal Arora, Mike
Merrill, Yichuan Deng, Ashima Suvarna, Hritik Bansal, Marianna Nezhurina, Yejin Choi,
Reinhard Heckel, Seewong Oh, Tatsunori Hashimoto, Jenia Jitsev, Vaishaal Shankar, Alex
Dimakis, Mahesh Sathiamoorthy, and Ludwig Schmidt, November 2024.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi
Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong
Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level
bilingual multimodal scientific problems, 2024. URL https://arxiv.org/abs/2402.
14008.

Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David
Meger. Deep reinforcement learning that matters. In Proceedings of the AAAI conference on
artificial intelligence, volume 32, 2018.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang,
Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the
math dataset. arXiv preprint arXiv:2103.03874, 2021.

Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models.

arXiv preprint arXiv:2501.03262, 2025.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xi-
angyu Zhang. Open-reasoner-zero: An open source approach to scaling reinforce-
ment learning on the base model. https://github.com/Open-Reasoner-Zero/
Open-Reasoner-Zero, 2025.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, and
Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language
models. arXiv preprint arXiv:2503.06749, 2025.

HuggingFaceH4.

Math-500 dataset.

https://huggingface.co/datasets/

HuggingFaceH4/MATH-500/blob/main/README.md, 2024. Accessed: 2025-03-29.

Ben Hutchinson, Negar Rostamzadeh, Christina Greer, Katherine Heller, and Vinodkumar
Prabhakaran. Evaluation gaps in machine learning practice. In Proceedings of the 2022
ACM conference on fairness, accountability, and transparency, pp. 1859�1876, 2022.

Intelligent Internet. II-Thought : A Large-Scale, High-Quality Reasoning Dataset, 2025.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low,
Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card.
arXiv preprint arXiv:2412.16720, 2024.

Piyush Jha, Prithwish Jana, Pranavkrishna Suresh, Arnav Arora, and Vijay Ganesh. Rlsf:
Reinforcement learning via symbolic feedback. arXiv preprint arXiv:2405.16661, 2024.

Scott Jordan, Yash Chandak, Daniel Cohen, Mengxue Zhang, and Philip Thomas. Evaluating
In International Conference on

the performance of reinforcement learning algorithms.
Machine Learning, pp. 4962�4973. PMLR, 2020.

Scott M Jordan, Adam White, Bruno Castro Da Silva, Martha White, and Philip S Thomas.
Position: Benchmarking is limited in reinforcement learning research. arXiv preprint
arXiv:2406.16241, 2024.

Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large
language models struggle to learn long-tail knowledge. In International Conference on
Machine Learning, pp. 15696�15707. PMLR, 2023.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva
Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Unlocking rl potential for llm
reasoning through refined credit assignment. arXiv preprint arXiv:2410.01679, 2024.

17

Preprint

Knovel Engineering. Amc-23 dataset, 2025. URL https://huggingface.co/

datasets/knoveleng/AMC-23.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu,
Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large
language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th
Symposium on Operating Systems Principles, 2023.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski,
Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo,
Solving quan-
Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra.
In S. Koyejo, S. Mohamed,
titative reasoning problems with language models.
A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural
In-
formation Processing Systems, volume 35, pp. 3843�3857. Curran Associates, Inc.,
2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/
file/18abbeef8cfe9203fdf9053c9c4fe191-Paper-Conference.pdf.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. LIMR: Less is More for RL Scaling. arXiv

preprint arXiv:2502.11886, 2025a.

Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao,
Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, et al. From system 1 to system 2: A
survey of reasoning large language models. arXiv preprint arXiv:2502.17419, 2025b.

Thomas Liao, Rohan Taori, Inioluwa Deborah Raji, and Ludwig Schmidt. Are we learning
yet? A meta review of evaluation failures across machine learning. In Thirty-fifth Conference
on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy
Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let�s verify step by step.
In The Twelfth International Conference on Learning Representations, 2023.

Yen-Ting Lin. Aime 2025 dataset, 2025. URL https://huggingface.co/datasets/

yentinglin/aime_2025. Accessed: 2025-03-29.

Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji. Cppo: Accelerating the train-
arXiv preprint

ing of group relative policy optimization-based reasoning models.
arXiv:2503.22342, 2025a.

Zhiyu Lin, Yifei Gao, Xian Zhao, Yunfan Yang, and Jitao Sang. Mind with eyes: from
language reasoning to multimodal reasoning. arXiv preprint arXiv:2503.18071, 2025b.

Zachary C Lipton and Jacob Steinhardt. Troubling trends in machine learning scholarship:
Some ml papers suffer from flaws that could mislead the public and stymie future research.
Queue, 17(1):45�77, 2019.

Jiawei Liu and Lingming Zhang. Code-r1: Reproducing r1 for code with reliable rewards.

2025.

Junnan Liu, Hongwei Liu, Linchen Xiao, Ziyi Wang, Kuikun Liu, Songyang Gao, Wenwei
Zhang, Songyang Zhang, and Kai Chen. Are your llms capable of stable reasoning? arXiv
preprint arXiv:2412.13147, 2024.

Zichen Liu, Changyu Chen, Wenjun Li, Tianyu Pang, Chao Du, and Min Lin. There may
not be aha moment in r1-zero-like training � a pilot study. https://oatllm.notion.
site/oat-zero, 2025a. Notion Blog.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee,
and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025b. URL
https://arxiv.org/abs/2503.20783.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua
arXiv preprint

Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning.
arXiv:2503.01785, 2025c.

18

Preprint

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin
Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. DeepScaleR:
Surpassing O1-Preview with a 1.5B Model by Scaling RL, 2025. Notion Blog.

Chengqi Lyu, Songyang Gao, Yuzhe Gu, Wenwei Zhang, Jianfei Gao, Kuikun Liu, Ziyi
Wang, Shuaibin Li, Qian Zhao, Haian Huang, et al. Exploring the limit of outcome reward
for learning mathematical reasoning. arXiv preprint arXiv:2502.06781, 2025.

Yan Ma, Steffi Chern, Xuyang Shen, Yiran Zhong, and Pengfei Liu. Rethinking rl scaling
for vision language models: A transparent, from-scratch framework and comprehensive
evaluation scheme. arXiv preprint arXiv:2504.02587, 2025.

Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh
Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward
design via coding large language models. arXiv preprint arXiv:2310.12931, 2023.

Marlos C Machado, Marc G Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and
Michael Bowling. Revisiting the arcade learning environment: Evaluation protocols and
open problems for general agents. Journal of Artificial Intelligence Research, 61:523�562,
2018.

Lovish Madaan, Aaditya K Singh, Rylan Schaeffer, Andrew Poulton, Sanmi Koyejo, Pontus
Stenetorp, Sharan Narang, and Dieuwke Hupkes. Quantifying variance in evaluation
benchmarks, 2024. URL https://arxiv. org/abs/2406.10229.

Benjamin Marie, Atsushi Fujita, and Raphael Rubino. Scientific credibility of machine
translation research: A meta-evaluation of 769 papers. arXiv preprint arXiv:2106.15195,
2021.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu,
Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring
visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint
arXiv:2503.07365, 2025.

Meta-AI.

The llama 4 herd: The beginning of a new era of natively
URL https://ai.meta.com/blog/

multimodal
llama-4-multimodal-intelligence/. Accessed: 2025-04-07.

innovation,

2025.

ai

Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng
Wang, Xiaoxue Cheng, Huatong Song, Wayne Xin Zhao, Zheng Liu, Zhongyuan Wang,
and Ji-Rong Wen. Imitate, explore, and self-improve: A reproduction report on slow-
thinking reasoning systems, 2024. URL https://arxiv.org/abs/2412.09413.

Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and
Mehrdad Farajtabar. Gsm-symbolic: Understanding the limitations of mathematical
reasoning in large language models. arXiv preprint arXiv:2410.05229, 2024.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi,
Luke Zettlemoyer, Percy Liang, Emmanuel Cand�s, and Tatsunori Hashimoto. s1: Simple
test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.

Kevin Musgrave, Serge Belongie, and Ser-Nam Lim. A metric learning reality check. In
Computer Vision�ECCV 2020: 16th European Conference, Glasgow, UK, August 23�28, 2020,
Proceedings, Part XXV 16, pp. 681�699. Springer, 2020.

Marianna Nezhurina, Lucia Cipolina-Kun, Mehdi Cherti, and Jenia Jitsev. Alice in won-
derland: Simple tasks showing complete reasoning breakdown in state-of-the-art large
language models. arXiv preprint arXiv:2406.02061, 2024.

OpenAI. OpenAI o3-mini System Card, January 2025. URL https://cdn.openai.com/

o3-mini-system-card-feb10.pdf.

19

Preprint

Shubham Parashar, Zhiqiu Lin, Tian Liu, Xiangjue Dong, Yanan Li, Deva Ramanan, James
Caverlee, and Shu Kong. The neglected tails in vision-language models. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12988�12997, 2024.

Andrew Patterson, Samuel Neumann, Martha White, and Adam White. Empirical design

in reinforcement learning. Journal of Machine Learning Research, 25(318):1�63, 2024.

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai
Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong
reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.

Ivo Petrov, Jasper Dekoninck, Lyuben Baltadzhiev, Maria Drencheva, Kristian Minchev,
Mislav Balunovi�c, Nikola Jovanovi�c, and Martin Vechev. Proof or bluff? evaluating llms
on 2025 usa math olympiad. arXiv preprint arXiv:2503.21934, 2025.

Ameya Prabhu, Philip HS Torr, and Puneet K Dokania. Gdumb: A simple approach that
questions our progress in continual learning. In Computer Vision�ECCV 2020: 16th European
Conference, Glasgow, UK, August 23�28, 2020, Proceedings, Part II 16, pp. 524�540. Springer,
2020.

Ameya Prabhu, Shiven Sinha, Ponnurangam Kumaraguru, Philip HS Torr, Ozan Sener, and
Puneet K Dokania. Randumb: A simple approach that questions the efficacy of continual
representation learning. arXiv e-prints, pp. arXiv�2402, 2024a.

Ameya Prabhu, Vishaal Udandarao, Philip Torr, Matthias Bethge, Adel Bibi, and Samuel
Albanie. Efficient lifelong model evaluation in an era of rapid progress. arXiv preprint
arXiv:2402.19472, 2024b.

Ori Press, Steffen Schneider, Matthias K�mmerer, and Matthias Bethge. Rdumb: A simple
approach that questions our progress in continual test-time adaptation. Advances in Neural
Information Processing Systems, 36:39915�39935, 2023.

PyTorch Contributors. Reproducibility � pytorch documentation. https://pytorch.

org/docs/stable/notes/randomness.html, 2024. Accessed: 2025-04-09.

Matthew Renze. The effect of sampling temperature on problem solving in large language
In Findings of the Association for Computational Linguistics: EMNLP 2024, pp.

models.
7346�7356, 2024.

Anka Reuel, Amelia Hardy, Chandler Smith, Max Lamparth, Malcolm Hardy, and Mykel J
Kochenderfer. BetterBench: Assessing AI Benchmarks, Uncovering Issues, and Establish-
ing Best Practices. arXiv preprint arXiv:2411.12990, 2024.

Jonathan Roberts, Mohammad Reza Taesiri, Ansh Sharma, Akash Gupta, Samuel Roberts,
Ioana Croitoru, Simion-Vlad Bogolin, Jialu Tang, Florian Langer, Vyas Raina, et al. Ze-
robench: An impossible visual benchmark for contemporary large multimodal models.
arXiv preprint arXiv:2502.09696, 2025.

Manley Roberts, Himanshu Thakur, Christine Herlihy, Colin White, and Samuel Dooley. To
the cutoff... and beyond? a longitudinal perspective on llm data contamination. In The
Twelfth International Conference on Learning Representations, 2023.

Nicolas Le Roux, Marc G Bellemare, Jonathan Lebensold, Arnaud Bergeron, Joshua Greaves,
Alex Fr�chette, Carolyne Pelletier, Eric Thibodeau-Laufer, S�ndor Toth, and Sam Work.
Tapered off-policy reinforce: Stable and efficient reinforcement learning for llms. arXiv
preprint arXiv:2503.14286, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Wei Shen, Guanlin Liu, Zheng Wu, Ruofei Zhu, Qingping Yang, Chao Xin, Yu Yue, and Lin
Yan. Exploring data scaling trends and effects in reinforcement learning from human
feedback. arXiv preprint arXiv:2503.22230, 2025.

20

Preprint

Shamus Sim and Tyrone Chen. Critique of impure reason: Unveiling the reasoning behaviour

of medical large language models. arXiv preprint arXiv:2412.15748, 2024.

Mingyang Song, Mao Zheng, Zheng Li, Wenjie Yang, Xuan Luo, Yue Pan, and Feng Zhang.
FastCuRL: Curriculum Reinforcement Learning with Progressive Context Extension for
Efficient Training R1-like Reasoning Models, 2025. URL https://arxiv.org/abs/
2503.17287.

Saurabh Srivastava, Anto PV, Shashank Menon, Ajay Sukumar, Alan Philipose, Stevin
Prince, Sooraj Thomas, et al. Functional benchmarks for robust evaluation of reasoning
performance, and the reasoning gap. arXiv preprint arXiv:2402.19450, 2024.

Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and
Dong Yu. Expanding rl with verifiable rewards across diverse domains. arXiv preprint
arXiv:2503.23829, 2025.

Team. Open Thoughts. https://open-thoughts.ai, January 2025.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li,
Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement
learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Songjun Tu, Jiahao Lin, Xiangyu Tian, Qichao Zhang, Linjing Li, Yuqian Fu, Nan Xu, Wei
He, Xiangyuan Lan, Dongmei Jiang, et al. Enhancing llm reasoning with iterative dpo: A
comprehensive empirical investigation. arXiv preprint arXiv:2503.12854, 2025.

Vishaal Udandarao, Ameya Prabhu, Adhiraj Ghosh, Yash Sharma, Philip Torr, Adel Bibi,
Samuel Albanie, and Matthias Bethge. No" zero-shot" without exponential data: Pretrain-
ing concept frequency determines multimodal model performance. In The Thirty-eighth
Annual Conference on Neural Information Processing Systems, 2024.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang,
Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with
process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

vLLM Contributors.

https://github.com/
vllm-project/vllm/blob/098900d7c2b53324687977eece400f634755cf51/
examples/offline_inference/reproduciblity.py, 2024. Accessed: 2025-04-09.

Inference reproducibility script.

Yue Wang, Qiuzhi Liu, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Linfeng Song,
Dian Yu, Juntao Li, Zhuosheng Zhang, et al. Thoughts are all over the place: On the
underthinking of o1-like llms. arXiv preprint arXiv:2501.18585, 2025.

Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu,
Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, and Xiangzheng
Zhang. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond, 2025a.
URL https://arxiv.org/abs/2503.10460.

Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu,
Lifu Tang, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch
and beyond. arXiv preprint arXiv:2503.10460, 2025b.

xAI. Grok 3 beta � the age of reasoning agents. February 2025. URL https://x.ai/

news/grok-3. Accessed: 2025-03-29.

Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai
Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based
reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.

Kai Yan, Yufei Xu, Zhengyin Du, Xuesong Yao, Zheyu Wang, Xiaowen Guo, and Jiecao Chen.
Recitation over reasoning: How cutting-edge language models can fail on elementary
school-level reasoning problems? arXiv preprint arXiv:2504.00509, 2025.

21

Preprint

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint
arXiv:2412.15115, 2024a.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu,
Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2.5-math technical report: Toward
mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024b.

Huimu Yu, Xing Wu, Weidong Yin, Debing Zhang, and Songlin Hu. Codepmp: Scal-
able preference model pretraining for large language model reasoning. arXiv preprint
arXiv:2410.02229, 2024.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan,
Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement
learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Weizhe Yuan, Jane Yu, Song Jiang, Karthik Padthe, Yang Li, Dong Wang, Ilia Kulikov,
Kyunghyun Cho, Yuandong Tian, Jason E Weston, et al. Naturalreasoning: Reasoning in
the wild with 2.8 m challenging questions. arXiv preprint arXiv:2502.13124, 2025.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, et al. Vapo:
Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint
arXiv:2504.05118, 2025.

Thomas Zeng, Shuibai Zhang, Shutong Wu, Christian Classen, Daewon Chae, Ethan Ewer,
Minjae Lee, Heeju Kim, Wonjun Kang, Jackson Kunde, et al. Versaprm: Multi-domain
process reward model via synthetic reasoning data. arXiv preprint arXiv:2502.06737, 2025a.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He.
SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base
Models in the Wild. arXiv preprint arXiv:2503.18892, 2025b.

Sheng Zhang, Qianchu Liu, Guanghui Qin, Tristan Naumann, and Hoifung Poon. Med-rlvr:
Emerging medical reasoning from a 3b base model via reinforcement learning. arXiv
preprint arXiv:2502.19655, 2025.

22

Preprint

A Appendix

A.1 Bootstrapping Results on Additional Datasets

To complement our analysis in 3, we present bootstrapped variance results on two additional
datasets: AMC�23 and MATH500. As shown in Figures 12 and 13, high variance in Pass@1
persists even when averaging over multiple seeds (K = 5), mirroring the trends observed on
AIME�24. These results reinforce our conclusion that small benchmark sizes yield unstable
estimates and that robust performance reporting requires multiple seed runs.

Figure 12: Variance of mean Pass@1 on AMC�23. Bootstrapped estimates show substantial
variance even with K = 5 evaluation runs, highlighting the instability of single-seed evalua-
tions.

Figure 13: Variance of mean Pass@1 on MATH500. Similar to AIME�24 and AMC�23,
the estimates remain volatile across seeds. Even K = 5 runs do not eliminate variance,
underscoring the need for larger K.

23

2468100510152025Variance of MeansDeepSeek-R1-Distill-1.5B2468100510152025OpenRS3-1.5B24681005101520DeepScaleR-1.5B246810K012345Variance of MeansDeepSeek-R1-Distill-7B246810K0.02.55.07.510.012.5OpenThinker-7B246810K05101520S1.1-7B2468100.00.10.20.30.40.5Variance of MeansDeepSeek-R1-Distill-1.5B2468100.00.20.40.60.81.0OpenRS3-1.5B2468100.00.20.40.60.8DeepScaleR-1.5B246810K0.00.20.40.6Variance of MeansDeepSeek-R1-Distill-7B246810K0.00.10.20.30.4OpenThinker-7B246810K0.00.10.20.3S1.1-7BPreprint

A.2 Hardware & Software Variations

In Figure 14, we show that the model performance variation due to hardware configuration
is not limited to AIME�24 and AMC�23. Similar discrepancies are observed on MATH500,
where different compute clusters yield different accuracy scores�even when model, seeds,
and decoding parameters are held constant. This further emphasizes the need for hardware
and software standardization when reporting benchmark results.

Figure 14: Performance variation across compute clusters on MATH500. Differences in GPU
type and environment lead to non-trivial shifts in performance, reinforcing the importance
of hardware standardization.

A.3 Prompt Variants and Template Settings

We provide the exact templates used for our three prompt settings in Table 5: Math, Default,
and No Template. These formats are based on the DeepSeek tokenizer but adapted for each
model�s specific chat template. Our results (in 3.4) indicate that instruction-tuned models
are highly sensitive to prompt formatting, with performance degrading significantly when
prompts deviate from their training-time structure.

A.4 Effect of Output Length Limits

We further explore how varying max_new_tokens impacts model accuracy. Figures be-
low compare OpenRS-series models (with 131,072-token context windows) and Open-
Thinker/S1.1 models (with 32,768-token limits).

Figure 15 shows that OpenRS models are highly sensitive to this parameter�shortening
outputs results in clear accuracy drops. Similarly, Figure 16 reveals the same pattern for
OpenThinker-7B and S1.1-7B, despite their smaller context lengths. In both cases, premature
truncation leads to incomplete reasoning chains and incorrect answers, confirming the
importance of setting appropriate generation limits.

24

DeepScaleR-1.5BDeepSeek-R1-Distill-1.5BDeepSeek-R1-Distill-7BOpenRS1-1.5BOpenRS2-1.5BOpenRS3-1.5BOpenThinker-7BS1.1-7BII-1.5B0.820.840.860.880.900.920.94MATH500Preprint

Prompt
Math

Default

Example

<|begin_of_sentence|><|User|>Solve the follow-
ing math problem efficiently and clearly. The
last line of your response should be of the
following format: �Therefore, the final answer
is: $\boxed{ANSWER}$. I hope it is correct�
(without quotes) where ANSWER is just the final
number or expression that solves the problem.
Think step by step before answering.\n <|Assis-
tant|><think>\n{Question}
<|begin_of_sentence|><|User|>{Question} <|Assis-
tant|><think>\n

No Template{Question}

Table 5: Prompt templates used in our evaluation. The inclusion or exclusion of structured
prompt tokens significantly impacts performance for instruction-tuned models.

Figure 15: Impact of max_new_tokens on OpenRS models. Models with long context
support (131,072 tokens) experience degraded performance when max_new_tokens is set
too low.

Figure 16: Impact of max_new_tokens on OpenThinker and S1.1 models. Despite shorter
context limits (32,768 tokens), performance still degrades noticeably when output length is
constrained.

A.5 Response Length vs. Accuracy � Per-Model Breakdown

To supplement the aggregated results shown in Figure 11, we include detailed histograms
for each individual model in the appendix. These plots show the distribution of correct
and incorrect responses across response lengths, averaged over random seeds. Due to the
number of models analyzed, we split the results into two figures for clarity.

Figures 17 and 18 reveal that the overall trend observed in the main paper holds consistently
across nearly all models: incorrect responses tend to be longer than correct ones.

These results reinforce the idea that excessively long outputs often indicate failure modes
such as hallucinated reasoning, verbose overthinking, or degenerate loops. Importantly, this

25

40968192163843276865536131072Max New Tokens0.10.20.30.4AccuracyAIME24OpenRS1-1.5BOpenRS2-1.5BOpenRS3-1.5B40968192163843276865536131072Max New Tokens0.40.50.60.70.8AMC2340968192163843276865536131072Max New Tokens0.740.760.780.800.820.840.86MATH500409681921638432768Max New Tokens0.10.20.30.40.50.6AccuracyAIME24OpenThinker-7BS1.1-7BOpenThinker2-7B409681921638432768Max New Tokens0.30.40.50.60.70.80.9AMC23409681921638432768Max New Tokens0.700.750.800.850.900.95MATH500Preprint

correlation persists well below the maximum sequence length, ruling out truncation as the
sole cause.

Figure 17: Response Length vs. Correctness � Models (1/2). Average number of correct
and incorrect responses across response length bins for a subset of models. Longer responses
consistently correlate with incorrect predictions.

Across all models, longer responses are a consistent marker of incorrect outputs, making
response length a useful signal for detecting low-confidence or erroneous reasoning chains.

26

05001000150020002500300035004000Response Length0500100015002000Avg Count per SeedQwen2.5-Math-1.5BCorrect (1.0)Incorrect (0.0)05001000150020002500300035004000Response Length01000200030004000500060007000Qwen2.5-Math-1.5B-Instruct05001000150020002500300035004000Response Length025050075010001250150017502000Qwen2.5-Math-7B05001000150020002500300035004000Response Length01000200030004000500060007000Avg Count per SeedQwen2.5-Math-7B-Instruct050001000015000200002500030000Response Length050001000015000200002500030000Qwen2.5-1.5B050001000015000200002500030000Response Length01000200030004000DeepSeek-R1-Distill-Qwen-1.5B050001000015000200002500030000Response Length0100200300400500600Avg Count per SeedDeepSeek-R1-Distill-Qwen-7B050001000015000200002500030000Response Length0500100015002000Qwen2.5-1.5B-Instruct050001000015000200002500030000Response Length025050075010001250150017502000Qwen2.5-7B-Instruct050001000015000200002500030000Response Length02004006008001000120014001600Avg Count per SeedDeepScaleR-1.5B-Preview050001000015000200002500030000Response Length025050075010001250150017502000OpenThinker-7B050001000015000200002500030000Response Length050100150200250300350Open-RS1050001000015000200002500030000Response Length050100150200250300350Avg Count per SeedOpen-RS2050001000015000200002500030000Response Length0500100015002000250030003500Open-RS3Preprint

Figure 18: Response Length vs. Correctness � Models (2/2). Continuation of model-wise
response length analysis. The same trend holds across the remaining models, with incorrect
answers being disproportionately long.

27

050001000015000200002500030000Response Length02505007501000125015001750Avg Count per Seeds1.1-7BCorrect (1.0)Incorrect (0.0)050001000015000200002500030000Response Length0500100015002000II-Thought-1.5B-Preview05001000150020002500300035004000Response Length020406080100Qwen2.5-Math-1.5B-Oat-Zero05001000150020002500300035004000Response Length020406080100120Avg Count per SeedQwen2.5-Math-7B-Oat-Zero050001000015000200002500030000Response Length05001000150020002500300035004000STILL-3-1.5B-preview050001000015000200002500030000Response Length01000200030004000500060007000Bespoke-Stratos-7B050001000015000200002500030000Response Length050010001500200025003000Avg Count per SeedFastCuRL-1.5B-Preview05001000150020002500300035004000Response Length020406080100120140160LIMR050001000015000200002500030000Response Length02500500075001000012500150001750020000OpenR1-Qwen-7B05001000150020002500300035004000Response Length0100020003000400050006000Avg Count per SeedQwen-2.5-Math-7B-SimpleRL-Zoo050001000015000200002500030000Response Length05001000150020002500Qwen-2.5-1.5B-SimpleRL-Zoo050001000015000200002500030000Response Length0200040006000800010000Qwen-2.5-7B-SimpleRL-Zoo050001000015000200002500030000Response Length01000200030004000500060007000Avg Count per SeedL1-Qwen-1.5B-Max050001000015000200002500030000Response Length01000200030004000500060007000L1-Qwen-1.5B-Exact050001000015000200002500030000Response Length010002000300040005000600070008000Open-Reasoner-Zero-7B
