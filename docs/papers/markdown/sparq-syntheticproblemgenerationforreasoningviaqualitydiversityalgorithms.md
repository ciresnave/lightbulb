5
2
0
2

n
u
J

7
1

]

G
L
.
s
c
[

2
v
9
9
4
6
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

SPARQ: Synthetic Problem Generation for Reasoning
via Quality-Diversity Algorithms

Alex Havrilla1,3,?

Edward Hughes2 Mikayel Samvelyan2

Jacob Abernethy1,3

Google Research1 Google DeepMind2 Georgia Tech3

Abstract

Large language model (LLM) driven synthetic data generation has emerged as
a powerful method for improving model reasoning capabilities. However, most
methods either distill large state-of-the-art models into small students or use natural
ground-truth problem statements to guarantee problem statement quality. This
limits the scalability of these approaches to more complex and diverse problem
domains. To address this, we present Synthetic Problem Generation for Reasoning
via Quality-Diversity Algorithms (SPARQ), a novel approach for generating high-
quality and diverse synthetic math problem and solution pairs using only a single
model (Gemma-2-9b) by measuring a problem�s solve-rate: a proxy for problem
difficulty. Starting from a seed dataset of 7.5K samples, we generate over 20
million new problem-solution pairs. We show that filtering the generated data
by difficulty and then fine-tuning the same model on the resulting data improves
relative model performance by up to 24%. Additionally, we conduct ablations
studying the impact of synthetic data quantity, quality and diversity on model
generalization. We find that higher quality, as measured by problem difficulty,
facilitates better in-distribution performance. Further, while generating diverse
synthetic data does not as strongly benefit in-distribution performance, filtering for
more diverse data facilitates more robust OOD generalization. We also confirm
the existence of model and data scaling laws for synthetically generated problems,
which positively benefit downstream model generalization.

1

Introduction

The quantity of high-quality problem statements is one of the most impactful factors affecting the
downstream performance of both supervised fine-tuned (SFT) and RL fine-tuned models [Toshniwal
et al., 2024b, Singh et al., 2024]. However, the vast majority of synthetic data generation approaches
either (1) generate only new solutions to a fixed problem set [Toshniwal et al., 2024b,a, Havrilla et al.,
2024b] or (2) generate new problems using a large state-of-the-art oracle without carefully filtering
the resulting problem data for quality and correctness [Yue et al., 2023, Yu et al., 2024]. These
approaches sidestep a key difficulty in problem generation: the lack of a ground truth signal which
can be used to filter out illogical/invalid problems. As a result, these approaches are not scalable to
generating new problem-solution pairs whose difficulty exceeds the capabilities of existing SOTA
models. Additionally, the restriction to natural data limits problem-solution diversity, especially in
the increasingly complex domains in which models are applied.

In this work we present Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms
(SPARQ), a novel synthetic data generation algorithm producing high-quality and diverse problem-
solution pairs using a single student model S?2 (Gemma-2-9b [Team et al., 2024]) and seed dataset
?Work done during an internship at Google Research. Correspondence to ahavrilla3@gatech.edu.

Preprint. Under review.

D. The key idea behind SPARQ is to use monte-carlo rollouts from S?2 to estimate the solve-rate
(i.e., "difficulty") of a problem-solution pair (Q, A) for S?2. We then use the solve-rate as a proxy for
problem quality, allowing us to filter out low-quality generations which are either too hard/impossible
or too easy. We demonstrate that generating and filtering data based on this quality score significantly
improves the pass@1 accuracy of the MATH [Hendrycks et al., 2021] SFT Gemma-2-9b student
from 38% to 47%. An ablation confirms that increased average train problem quality correlates well
with improved downstream performance.

We then investigate the role of problem diversity in problem generation and downstream generalization
using SPARQ. Problem diversity is measured first by annotating problem-solution pairs (Q, A)
with the skill/technique set used in solving Q. For example, Q may require a combination of the
pigeonhole principle and algebra in the correct solution. Representing problems with their
most relevant skills allows us to measure the overall diversity of a set of problems by examining
the coverage (number of unique skills) and redundancy (i.e., number of repeated skills). Given
a fixed sample budget N , we find increasing problem diversity does not improve in-distribution
generalization relative to a randomly selected training data baseline. However, we find models trained
on more diverse synthetic data generalize better to out-of-distribution (OOD) tasks as the amount of
compute used to solve the task increases.

In summary, we make the following contributions:

1. We introduce the solve-rate of a problem Q with respect to S?2 as an effective way of

measuring and filtering by synthetic problem quality.

2. We present SPARQ, a new approach for synthetic problem generation that directly optimizes
for the data quality and diversity. Training on the resulting data leads to an absolute
downstream improvement over the baseline of 9%.

3. We carefully ablate the impact of quality and diversity of synthetic data generated with
SPARQ on model generalization, revealing increased quality to correlate well with better
in-distribution generalization and diversity to benefit OOD generalization.

2 Synthetic Problem Generation for Reasoning via Quality-Diversity

Algorithms

Setup and notation Let ?1 ? Rd1 , ?2 ? Rd2 be optionally distinct model parameters. Suppose
we are given a generating language model G?1 and student model S?2. Fix a tokenized vocabulary
V = {1, ..., V } ? N for some V ? N. Let the task ? be a distribution of problem-solution pairs
(Q, A) ? V LQ �V LA where LQ, LA ? N are the maximum context lengths of problems and solutions
respectively. Let D be a seed dataset of n ? N problem-solution pairs (Qi, Ai) ? ? .

A high-level synthetic data generation recipe Given D we initialize a working set set W (0) of
mutation candidates meant to hold high-quality samples for future mutation. We also initialize an
archive A(0) which accumulates all generated data regardless of quality. Our synthetic data generation
algorithm then proceeds in two phases: data generation and data filtration. The data generation phase
proceeds iteratively at round t by sampling a batch of problem-solution pairs (Q(t)
i ) ? C of size
i
b ? N from the working set W (t) using the mutation selection distribution C(t) ? ?(W (t)). Muta-
tions (Q?(t)
, A(t)
i )
i
i ) the child or mutation of (Q(t)
the parent and (Q?(t)
, A(t)
i ). We then measure the QUALITY
i
of the new pairs (Q?(t)
i ) and update the working set W (t) ? W (t+1) via the update function
i
U(t). The update function U(t) (cid:16)
1 , A?(t)
? W (t+1) inserts the
high-quality subset of new mutations into W (t) and removes excess low-quality data in W (t) to pro-
b )} (cid:83) A(t)
duce W (t+1). Finally, we take the union of new mutations {(Q?(t)
to produce A(t+1). Note: when it is clear we will suppress the notational dependence on the round t.

i ) ? G?1 (�|(Qi, Ai)) are then sampled using the generator G?1 . We call (Q(t)

1 ), ..., (Q?(t)

1 ), ..., (Q?(t)

W (t), {(Q?(t)

b , A?(t)

b , A?(t)

1 , A?(t)

(cid:17)
b )}

, A?(t)

, A?(t)

, A?(t)

, A(t)

i

i

Once all rounds of data generation conclude, the data filtration phase begins. During this phase,
every synthetic pair (Q, A) ? A in the archive A resulting from the data generation phase is filtered
by evaluating whether QUALITY((Q, A)) > 0. The remaining data is used to construct a new training
dataset D?. Figure 1 shows a diagram of the entire pipeline.

2

Figure 1: Diagram of the synthetic training data generation pipeline with SPARQ. Phase 1: Data
generation begins with Dseed initializing the working set. Samples (Q, A) are iteratively selected
from the working set W via the selection distribution C and mutated via the generator G?1. All
new mutations are stored in the archive A. The quality of new samples (Q?, A?) is assessed via the
difficulty student model S?2 has solving for A? given Q?. The working set update function U then
updates W with new sample (Q?, A?). Phase 2: After data generation, data filtration proceeds by
removing all low-quality samples from the archive A. Each remaining question Q is then paired with
its successful verifications Sk forming synthetic training tuples (Q, Sk).

Measuring data quality Given an arbitrary problem-solution pair (Q, A) the key idea behind our
approach is to use the average score the student model S?2 receives when solving Q. We call this
quantity the solve-rate of S?2 on Q which is computed via K many rollouts S1, . . . , SK ? S?2 (�|Q)
i.e.

SOLVERATE((Q, A), S?) :=

is_correct(Q, A, Sk)

1
K

K
(cid:88)

k=1

where is_correct(Q, A, Sk) is 1 if Si agrees with the intended solution A (in practice we check for
the same numerical final solution) else 0; it is for this reason we call each rollout Sk ? S?2 (�|Q) a
verification of Q. Intuitively, the solve-rate of (Q, A) measures the difficulty of Q for student S?2.
We use the solve-rate of a problem Q to define a quality score for Q as

QUALITY((Q, A), S?) =

(cid:26)1 ? SOLVERATE((Q, A), S?), Tl ? SOLVERATE((Q, A), S?) ? Tu

0

otherwise

which defines the quality of a problem as one minus its solve-rate for problems with a solve-rate
between Tl and Tu and 0 otherwise. We introduce the thresholds 0 < Tl < Tu < 1 for three reasons:
(a) to remove impossible problems i.e. those which are never verified by S?2; (b) to attempt to remove
problems which are sometimes verified by S?2 but contain reasoning errors in their intended solution
A; c) to remove trivial pairs too easy for the student. Note: we call such pairs (Q, A) containing a
reasoning error in A invalid.

Evaluating the quality of a pair (Q, A) with its solve-rate comes with the added benefit that successful
verifications Sk of Q produced by S?2 can be added to the training data for free. Thus our final
training samples are of the form (Q, S) where Q is a high-quality problem generated by G?1 and S is
a successful verification of Q via S?2 . Since we will be training S?2 with this new D?
train this makes
all training solutions entirely self-generated.
Measuring data diversity In addition to data quality, we are also interested in the diversity of
synthetically generated data. Abstractly, the diversity measure DIV : P(support(? )) ? R is a
function from sample subsets of the task ? to R measuring some notion of the subset�s coverage of

3

? or self-redundancy. Concretely, we propose to measure the diversity of a set of problem-solution
pairs {(Q1, A1), ..., (Qn, An)} via their representation as skill-sets. Informally, the skill-set of a pair
(Q, A) is the alphabetically ordered k-tuple of semantic skills ?((Q, A)) = ? = (?1, ..., ?k) which
are used in A to solve Q. For example, Q might require a combination of algebra (?1) and polynomial
factoring (?2) for a correct solution. This representation enables a number of diversity measures. For
this work we simply count the number of unique skills in a problem set {(Q1, A1), ..., (Qn, An)},
i.e.

DIV({(Q1, A1), ..., (Qn, An)}) = |{?j : ? = ?((Qi, Ai)), 1 ? j ? k}|

We limit the set of all possible skills as {?1, ..., ?M } with fixed size M ? M.

2.1 Variations on the high-level data generation recipe

In Section 2 we defined a novel high-level data generation pipeline for producing high-quality
synthetic data. However, the choices of sample selection distribution C and working set update
function U are left undefined. Here we detail four different implementations of the above pipeline
with different choices for C and U affecting the resulting quality and diversity of generated data.
Static uniform data generation Our simplest method is static uniform data generation. Mutation
parent samples are selected uniformly at random from the working set W, that is, the sample selection
function C is uniform. The update function U keeps the working set W fixed. As a result, only
samples from the seed-dataset are sampled for mutation.
Static diverse data generation We modify the above implementation slightly by partitioning the
working set W into equivalence classes [Q, A]? by identifying samples with the same skill-sets
?. Formally, [Q, A]? = {(Q, A) : ?((Q, A)) = ?}. In a slight abuse of notation, we also call
each equivalence class a skill-set. C then samples uniformly first over skill-sets and then over class
elements with the skill-set.
Dynamic uniform data generation We again slightly modify the static uniform generation proce-
dure by iteratively updating the working set W to contain the T ? N highest-quality samples in the
archive A. This is done by inserting high-quality mutations and removing the lowest-quality samples
from W (t) to produce W (t+1). Samples for mutation are selected uniformly from W as in static
uniform generation.
Dynamic diverse data generation Finally, inspired by algorithms from the quality-diversity
literature [Mouret and Clune, 2015, Pourcel et al., 2024, Samvelyan et al., 2024b], we combine the
modifications proposed in the above two methods to propose dynamic diverse data generation. As
with static diverse generation, we partition the working set W (t) into skill-set based equivalence
classes which are sampled uniformly to produce mutation parents. After mutation, we assign the
skill-sets ?1, ..., ?n to new sample mutations and insert high-quality samples (Q, A) into their
corresponding skill-set class [Q, A]?((Q,A)). For each class we enforce a uniform population limit
T? = T by removing the lowest-quality samples in a class when new high-quality samples are added.
Intuitively, the goal of this quality-diversity driven data generation algorithm is to generate maximally
diverse, high-quality problems with as many unique skill-sets and difficulty levels as possible.

3 Experiments
Setup We apply SPARQ to improve the math reasoning abilities of LLMs. As our seed dataset D
we use the 7.5K train set from MATH [Hendrycks et al., 2021]. We utilize the Gemma-2 model series
[Team et al., 2024], always taking Gemma-2-9b as our student model S?2 . To produce a strong student
model S?2 we fine-tune the pre-trained Gemma-2-9b on D for 3 epochs. Unless otherwise specified,
we use the instruction tuned Gemma-2-27b-it as our problem generator G?1 (see Subsection 3.3 for
recursive self-improvement results). K = 16 rollouts are used to compute the solve-rate. A solve-rate
thresholding between Tl = 0.1 and Tu = 0.9 is used. In the diversity-driven methods, we use the
M = 100 most commonly occurring skills in D with combinations of size at most k = 3 to measure
diversity. Gemma-2-9b-it is used to identify the problem skill-sets. Prompts for problem generation
and skill classification are shared in Appendix F.

We run each data generation method with a mutation batch size of 64 for a maximum of 5K steps.
This results in an archive A with 320K synthetic problem-solution pairs. Note: because each problem
requires K = 16 verifications for the quality evaluation, a total of 5 million synthetic solutions are

4

Figure 2: Performance of the downstream Gemma-2-9B models trained on synthetic data generated
with SPARQ. Left: In-distribution performance of students on MATH test. On the x-axis is the
number of synthetically generated problem-solution pairs. On the y-axis is MATH performance.
Each curve plots the performance of a different data generation strategy. Right: OOD performance
of downstream students on the AIME benchmark after training with 100K generated problems. On
the x-axis is the number of inference-time solutions. On the y-axis is AIME pass@n accuracy.

generated (per method) in addition to problem statements and skill classifications. We find that the
vast majority of generated problems are too difficult for S?2 to solve, resulting in a SOLVERATE
of 0. As a result, the average final size of a synthetically generated training dataset Dtrain comes
out to around 80K unique problems with QUALITY(Q) > 0 and a corresponding 500K (Q, S)
problem-verification pairs. Figure 10 in the Appendix shows a histogram of the distribution of
problem solve-rates. See Appendix A for a description of training hyperparameters.

3.1 Main Results for Problem Generation
Self-synthetically generated data significantly improves over the SFT baseline Figure 2 il-
lustrates the in-distrbution and OOD performance of downstream models trained on the resulting
data from each variant of SPARQ. We find training on data generated by every method improves
over the SFT baseline. In particular, the static uniform method improves by up to an absolute 9%,
increasing from 38% to 47%. Downstream performance also benefits from scaling the size of the
problem set: tripling the number of generated problems leads to a roughly 1.5% performance increase.
When scaling the inference compute during OOD evaluation, the static uniform method improves
over the SFT baseline from 20% to 25% (with K = 96 inference samples). The OOD performance
improvement can be seen at all inference compute budgets, increasing as the budget increases. This
demonstrates the problems generated by G?1 do not overfit to the initial seed dataset D, allowing for
an OOD improvement.

Dynamic diverse (QD) methods produce the most diverse data Figure 3 plots the diversity, as
measured by number of unique problem skill-sets, of the unfiltered data archives and filtered train
datasets produced by each method against the number of generated problems. All methods discover
novel combinations of skills to produce a more diverse archive as the number of generated problems
increases. The dynamic diverse method, combining dynamic updates to the working set with a
diversity-focused partitioning of the working set, produces the most diverse data by discovering 5000
new skill combinations after 100K problems. In contrast, the static uniform method produces the
least diverse data, discovering 4250 combinations. This gap becomes even larger when restricting to
the training subset: dynamic diverse training data contains 3000 skill-sets, whereas static uniform
training data contains only 2000. As a result, the training data generated by dynamic diverse methods
is significantly more diverse than data generated by static uniform methods.

Static uniform models perform best downstream Despite the more diverse nature of dynamic
diversely generated data, the static uniformly generated data leads to the best downstream models
both in-distribution and OOD. This gap persists across all problem generation sizes, with the dynamic
diverse method reaching a final in-distribution performance of 44% vs. 47% with static uniform
generation. This relatively small difference in in-distribution performance is perhaps unsurprising,
as our seed dataset D is already closely aligned with the test set. Generating more diverse data that
covers a wider range of problems does not appear to be as beneficial for improving performance on

5

Figure 3: Coverage (number of unique skill-sets discovered) vs. number of problems generated. The
QD algorithm achieves consistently higher coverage than static generation. Note: the train subset
considers only generated problems with quality greater than 0.

Figure 4: % of easy verifications kept versus downstream math accuracy. Filtering out easy verifica-
tions in training positively benefits lower data regimes but negatively affects higher data regimes.

the distribution from which D was generated. This underscores a critical point: selecting the right
notion of diversity when generating data depends crucially on the downstream tasks of interest.

Perhaps more surprisingly, the static uniform method also shows superior performance in the OOD
setting, once again outperforming the other approaches. The dynamic uniform approach, which
iteratively updates the working set with the highest-quality mutations, comes close but still under-
performs. The gap between these two methods could be due to a key difficulty the dynamic method
faces, namely the reliability of the QUALITY measure. We have no guarantee that the problem-
solution pair (Q, A) with a low solve-rate (and thus high-quality) contains logically valid reasoning.
As a result, it becomes difficult to filter out high-quality but logically-invalid problems from the
working set. This has the following unintended negative effect on our working set: when such a high-
quality invalid sample is selected for mutation, it becomes likely to generate yet more high-quality
but logically invalid data. This allows for dynamic generation methods in particular to reward hack
the QUALITY measure. In the next paragraph we conduct an initial investigation into this relationship,
confirming that difficult but solvable problems are more likely to be invalid. However, in Section 3.2
we also confirm that training on higher-quality samples leads to better performance, suggesting that
invalid samples can still be helpful at train time (while unhelpful during problem generation).

Filtering easy verifications gives mixed results By default the construction of our training dataset
in Section 3 is biased towards easier problems. This is because for a problem Q with quality above the
threshold Tq we construct training samples (Q, Sk) by pairing Q with all successful verifications Sk.
Easier problems will be verified more often (by definition) and as a result contribute more samples per
problem to the training data. This results in a bias towards easy problems in the training data which
may not be desirable. To account for this, we experiment with removing 75% of the easy (Q, Sk)
pairs per easy Q. Note: we define Q as easy if SOLVERATE((Q, A), S?2 ) ? 0.5. Figure 4 plots the
resulting change in performance for N=100K and N=300K generated problems. Filtering the 300K
generated problem set significantly reduces the size of the training dataset from 500K (Q, Sk) pairs
to 350K.

6

Figure 5: Average problem solve-rate versus problem validity. A higher solve-rate strongly correlates
with validity, suggesting harder problems are less likely to be logically valid.

We find filtering has mixed results. For a smaller amount of data filtering has a positive impact,
increasing performance by .075%. However, for a larger amount of data the effect is reversed. This
indicates that the performance increase from harder problems may be stable once a critical mass is
reached. In contrast, with less data there are less difficult problems available and thus the effect of
training on many easy problems is damaging.

High-quality problems can be noisy Our results demonstrate the solve-rate based quality measure
for filtering synthetically generated problem-solution pairs positively impacts model performance.
However, we are by no means guaranteed a high-quality pair (Q, A) is a logically valid pair, i.e.
the problem and intended solution do not contain reasoning errors (instead we simply have (Q, A)
is difficult for S?2 to solve according to the intended solution A). Intuitively, we might expect
problems with a higher solve-rate (and thus lower quality) to have higher chance of being valid:
simply because S?2 consistently arrives at the intended solution. We investigate this relationship
empirically by labeling the correctness of synthetic (Q, A) pairs by using a SOTA reasoning model
(Gemini-2.5-flash) to label Q with alternative solutions A?. The final solutions of A and A? are
then compared and used to label (Q, A) as valid if A, A? agree and invalid otherwise.

In Appendix Figure 5 we bin these labeled problems into varying solve-rate levels and plot the average
validity of samples in each bin. The plot demonstrates a strong correlation between the likely validity
of a problem and its solve-rate, and thus a strongly inverse relationship between validity and quality.
Specifically, the harder a problem is for S?2 to solve, the more likely it is to be invalid. Surprisingly,
as demonstrated above, this does not result in decreased fine-tuning performance when training on
higher quality problems. This suggests fine-tuning to be robust to potentially high levels of invalid
reasoning/noise in training data. For more investigation into the effect of the quality measure, see
Section 3.2.

3.2 Results for Problem Filtering with a Fixed Training Sample Budget

In the previous section, we examined the downstream performance of four variants of SPARQ
while keeping the phase 2 filtering strategy constant. We showed the static uniform synthetic data
generation method improves over baseline SFT performance. However, the dynamic uniform method
designed to jointly improve both the quality and diversity of data under-performs naive static uniform
generation.

In an attempt to better understand how training data quality and diversity affect model performance,
we now fix a single data generation method and explore different filtering approaches that result in
different levels quality and diversity. In particular, we conduct a series of ablations on the archive Asu
of 300K problems generated via the static uniform method.

We pre-process the archive by removing all pairs with QUALITY(Q, A) = 0, leaving approximately
150K (Q, A) pairs. We then select a single successful verifying rollout SQ for each Q. This gives
us a train sample pool D?
train = {(Q, SQ) : (Q, �) ? Asu, SQ verifies Q} with no repeated questions.
Finally, for each experiment, we will fix a sample budget N ? N and filter D?
train to produce a smaller

7

Figure 6: Left: In-distribution data scaling curves for various filtering strategies. On the x-axis is
the number of training samples N (distinct from the number of generated problems in Figure 2). On
the y-axis is MATH performance of fine-students. Each curve plots the performance of a different
data filtering strategy. Right: OOD inference scaling curves for filtering strategies trained with
N = 215 samples. On the x-axis is the number of inference-time samples. On the y-axis is pass@n
performance on AIME. Each curve plots the performance of a different data filtering strategy.

training dataset with target levels of data quality and diversity. We then train S?2 on the subset and
evaluate its in-distribution and OOD performance.
Fix a sample budget 212 ? N ? 215. We construct N ?sample training mixtures in the following
ways:

� Quality: To ablate different levels of quality with the same fixed N , we sample training
pairs (Q, Sk) from D?
train from a Gaussian distribution with mean quality m and standard
deviation 0.1. We sample using four different m ranging from 0.2 to 0.8 and ensure that
each selected Q is unique. Max quality refers to the subset sampled with mean quality
m = 0.8.

� Diversity: We sample a maximally diverse N ?sample subset by partitioning samples into
their respective skill-sets/niches. We then select the subset of skill-sets S = {?i} which
maximizes the number of unique skills ?i
j in S. Representative samples are uniformly
selected from each skill-set and added to the training subset.

� QD: We jointly optimize data quality and diversity by selecting a diverse set of skill-sets

and then sampling high-quality problems within each problem skill-set.

� Random: As a baseline we uniformly randomly sample a subset of size N from D?

train.

Jointly filtering for quality and diversity performs best Figure 6 shows the in-distribution and
OOD performance of the filtered data subsets. We find that the mixture filtered to optimize both
data quality and diversity performs best both in-distribution and OOD at nearly all training sample
budgets. The best performing QD filtered model achieves 45.5% accuracy on the MATH test set with
just 32K training samples: within 1.5% of the best static uniform model trained on 500K samples.
QD filtering also consistently improves over the randomly filtered baseline, at times by up to 3%.
Filtering for quality benefits in-distribution performance, filtering for diversity benefits OOD
Contrary to our findings in Section 3.1, here we see filtering to maximize data diversity (even without
quality) leads to superior OOD performance comparable to models trained on the QD subset. With
K = 96 samples the diversely trained model improves over the random baseline by 6% (from 19%
to 25%). Notably, the superiority of the diversely trained model is not clear until at least K = 16
samples, after which the gap with Quality and Random continues to widen. This suggests that
diversely trained models may scale better with inference compute. Finally, the left side of Figure
7 illustrates the OOD performance of each method with K = 96 inference samples against the size
of training data. We find the OOD performance comparisons for each method stay consistent across
different amounts of training data. However, for in-distribution performance, a model trained on only
high-quality (but not diverse) data performs better. As in Section 3.1, this suggests quality is more
beneficial than diversity for in-distribution performance.
Higher-quality data correlates with better performance So far we have yet to confirm training
on higher-quality data yields better downstream model performance. We do so by sub-sampling D?
train
via a Gaussian distribution with quality mean m one of 0.2, 0.4, 0.6, 0.8. The right side of Figure 7

8

Figure 7: Left: AIME pass@96 performance of filter methods versus number of training samples.
Right: Mean training sample quality versus MATH test performance.

plots the MATH test performance of models trained on subsets with different mean levels of quality.
For training sets with size greater than or equal to 8192, we see quality has a positive impact on
performance. This confirms our choice of quality measure via the student S?2 solve-rate to be a viable
proxy for data quality. When selecting N = 215 samples, filtering for data with a mean quality of 0.8.
vs. 0.2 can lead to over 2.5% absolute improvement. Note this occurs even in spite of our findings in
Section 3.1 showing harder problems are likely to contain logical mistakes. This suggests that S?2
benefits from harder problems during training even with moderate levels of noise.
Towards more performant methods for QD-driven synthetic data generation In Section 3.1 we
found that directly optimizing for higher quality samples (as done in dynamic generation methods) or
more diverse samples (as done diverse methods) failed to improve over the simpler static uniform
approach. Yet, our current investigation in Section 3.2 demonstrates benefits when directly filtering
for quality and diversity given the larger dataset D?
train. This suggests several avenues for improving
QD-driven variants of SPARQ. Firstly, the algorithms in Section 3.1 are compute equalized (i.e.,
the same number of problems are generated) but not training sample equalized (i.e., one algorithm
may generate more viable training samples than another). This translates into a difference in training
sample yield-rate where the static uniform algorithm produces viable samples for training more
consistently. Secondly, the hackablity of our proposed Quality measure becomes an important
concern. Dynamic generation methods will insert high-quality but logically invalid samples into
the working set, resulting in a higher percentage of high-quality but logically invalid samples in the
future. In Appendix B we explore an attempt to detect and mitigate these samples by looking at the
QUALITY distributions of their children.

3.3 Towards Recursive Self-Improvement

Figure 8: Left:% improvement of end-to-end static self-problem generation (same model as generator
and verifier). Right: Improvement from scaling problem generator model size. Gemma-2-9B acts as
the student S?2.

In our previous experiments we used Gemma-2-27B-it as the generator G?1 and a fine-tuned
Gemma-2-9B as S?2 . As a result, while all solutions in the training data are self-generated by the
student S?2 , the problems are not. Now, we apply our methods so that the same models (Gemma-2-2B,

Note that in most self-improvement style works the problems are fixed and thus not self-generated.

9

Gemma-2-9B) act uniquely as both the generator G?1 and student S?2. Results for the % relative
improvement of each model over its respective SFT baseline via static uniform data generation are
reported in Figure 8.

Models can self-improve by generating their own problems After generating 300K problem-
solution pairs, we find the 9B model self-improves by a relative 20% and the 2B model self-improves
by a relative 15%. Smaller improvements persist when generating less data. On the right side of Figure
8 we also plot the absolute performance of the 9B student when trained using problems generated by
a generator of varying size. This shows larger models generate problems more useful for the self-
improvement of S?2. For example, using a small 2B generator results in only 5% improvement over
the baseline. Scaling up the generator G?1 adds another 2.5% and 1.5% improvement cumulatively.
These results suggest that applying static uniform data generation to larger models benefits both from
an increase in the model�s problem solving ability and the model�s ability to generate good questions.

4 Related Works
Synthetic data generation for reasoning Many works have shown the benefit of synthetic data
generation for reasoning by distilling from a large teacher model to a smaller student [Yu et al., 2024,
Yue et al., 2023, Li et al., 2024, Liu et al., 2024, Luo et al., 2025]. Other works (many using RL
for LLMs) generate only novel solutions to a fixed problem set [Havrilla et al., 2024b, Singh et al.,
2024, DeepSeek-AI et al., 2025]. More recently, Dong and Ma [2025] and Poesia et al. [2024] used
small-sized LLMs to generate novel problem and solution pairs to reasoning problems in a formal
environment. This differs from our work where we do not rely on any ground truth environment (e.g.
Lean) to evaluate the quality of novel problems. Lin et al. [2025] used small-sized LLMs to jointly
generate and verify novel code problems. Related ideas to the solve-rate based quality measure used
in this work have been used in open-ended reinforcement learning [Team et al., 2021] to prioritize
efficient level sampling. Most related, Pourcel et al. [2024] uses a similar metric with a quality-
diversity algorithm to generate difficult programming puzzles for a novel benchmark programming
benchmark. In contrast, our work focuses on applying QD inspired ideas for training data generation
and thoroughly ablating the effects of our quality and diversity measures on model performance.

QD x LLMs The number of works at the intersection of Quality-Diversity methods has been
increasing rapidly over the last several years [Lehman et al., 2022, Bradley et al., 2023, Meyerson
et al., 2023, Zhang et al., 2023, Samvelyan et al., 2024a, Wu et al., 2024, Chao et al., 2024, Samvelyan
et al., 2024a, Havrilla et al., 2024a]. Lehman et al. [2022] were the first to utilize LLMs in evolutionary
loop by evolving racing agents. Bradley et al. [2023] utilized AI feedback to iteratively synthesize
high-quality poetry. Zhang et al. [2023] utilized powerful LLMs to generate a diverse set of RL
environments for training open-ended agents. See Havrilla et al. [2024a] for an in-depth review of
the intersection of QD with LLMs for synthetic data generation.

5 Conclusion and Limitations

In this work, we presented SPARQ, a new approach for generating high-quality and diverse synthetic
math problem by optimizing both data quality and diversity. We find that training on the resulting data
with SPARQ gave absolute improvements of up to 9% over an SFT baseline and scaled with both
the size of the problem generator and the amount of generated problem data. Further, we conducted
thorough ablations into the effects of data quality and diversity, finding that training on high-quality
data leads to better in-distribution generalization and training on more diverse data can lead to better
OOD generalization. Future work might address some of the current method�s limitations by focusing
on designing better QD inspired data generation algorithms with improved training sample yield rates
and mitigating over-optimization of our quality measure.

References

Herbie Bradley, Andrew Dai, Hannah Teufel, Jenny Zhang, Koen Oostermeijer, Marco Bellagente,
Jeff Clune, Kenneth Stanley, Gr�gory Schott, and Joel Lehman. Quality-diversity through ai
feedback, 2023. URL https://arxiv.org/abs/2310.13032.

Wang Chao, Jiaxuan Zhao, Licheng Jiao, Lingling Li, Fang Liu, and Shuyuan Yang. A match made
in consistency heaven: when large language models meet evolutionary algorithms. arXiv preprint
arXiv:2401.10510, 2024.

10

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu,
Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu,
Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao
Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao,
Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding,
Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang
Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong,
Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao,
Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang,
Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang,
Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L.
Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang,
Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng
Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng
Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan
Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang,
Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen,
Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li,
Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang,
Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan,
Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia
He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong
Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha,
Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang,
Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li,
Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen
Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
URL https://arxiv.org/abs/2501.12948.

Kefan Dong and Tengyu Ma. Stp: Self-play llm theorem provers with iterative conjecturing and

proving, 2025. URL https://arxiv.org/abs/2502.00212.

Alex Havrilla, Andrew Dai, Laura O�Mahony, Koen Oostermeijer, Vera Zisler, Alon Albalak, Fabrizio
Milo, Sharath Chandra Raparthy, Kanishk Gandhi, Baber Abbasi, Duy Phung, Maia Iyer, Dakota
Mahan, Chase Blagden, Srishti Gureja, Mohammed Hamdy, Wen-Ding Li, Giovanni Paolini,
Pawan Sasanka Ammanamanchi, and Elliot Meyerson. Surveying the effects of quality, diversity,
and complexity in synthetic data from large language models, 2024a. URL https://arxiv.org/
abs/2412.02980.

Alex Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu,
Maksym Zhuravinskyi, Eric Hambro, Sainbayar Sukhbaatar, and Roberta Raileanu. Teaching large
language models to reason with reinforcement learning, 2024b. URL https://arxiv.org/abs/
2403.04642.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL
https://arxiv.org/abs/2103.03874.

Joel Lehman, Jonathan Gordon, Shawn Jain, Kamal Ndousse, Cathy Yeh, and Kenneth O. Stanley.

Evolution through large models, 2022. URL https://arxiv.org/abs/2206.08896.

Chengpeng Li, Zheng Yuan, Hongyi Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi
Tan, Xiang Wang, and Chang Zhou. Mugglemath: Assessing the impact of query and response
augmentation on math reasoning, 2024. URL https://arxiv.org/abs/2310.05506.

Zi Lin, Sheng Shen, Jingbo Shang, Jason Weston, and Yixin Nie. Learning to solve and verify: A
self-play framework for code and test generation, 2025. URL https://arxiv.org/abs/2502.
14948.

Haoxiong Liu, Yifan Zhang, Yifan Luo, and Andrew Chi-Chih Yao. Augmenting math word problems

via iterative question composing, 2024. URL https://arxiv.org/abs/2401.09003.

11

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng,
Qingwei Lin, Shifeng Chen, Yansong Tang, and Dongmei Zhang. Wizardmath: Empowering
mathematical reasoning for large language models via reinforced evol-instruct, 2025. URL
https://arxiv.org/abs/2308.09583.

Elliot Meyerson, Mark J Nelson, Herbie Bradley, Arash Moradi, Amy K Hoover, and Joel
Lehman. Language model crossover: Variation through few-shot prompting. arXiv preprint
arXiv:2302.12170, 2023.

Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites, 2015. URL

https://arxiv.org/abs/1504.04909.

Gabriel Poesia, David Broman, Nick Haber, and Noah D. Goodman. Learning formal mathematics

from intrinsic motivation, 2024. URL https://arxiv.org/abs/2407.00695.

Julien Pourcel, C�dric Colas, Gaia Molinaro, Pierre-Yves Oudeyer, and Laetitia Teodorescu. Aces:
Generating diverse programming puzzles with with autotelic generative models, 2024. URL
https://arxiv.org/abs/2310.10692.

Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan,
Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rockt�schel,
and Roberta Raileanu. Rainbow teaming: Open-ended generation of diverse adversarial prompts.
In Advances in Neural Information Processing Systems, volume 37, pages 69747�69786, 2024a.

Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan,
Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rockt�schel,
and Roberta Raileanu. Rainbow teaming: Open-ended generation of diverse adversarial prompts,
2024b. URL https://arxiv.org/abs/2402.16822.

Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J.
Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, Abhishek Kumar, Alex Alemi,
Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Elsayed, Hanie Sedghi,
Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron,
Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura Culp, Lechao Xiao, Maxwell L.
Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yundi Qian, Yamini
Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, and Noah Fiedel. Beyond
human data: Scaling self-training for problem-solving with language models, 2024. URL https:
//arxiv.org/abs/2312.06585.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya
Bhupatiraju, L�onard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram�, Johan
Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar,
Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin,
Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur,
Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison,
Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia
Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris
Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger,
Dimple Vijaykumar, Dominika Rogozi�nska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric
Noland, Erica Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary
Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Pluci�nska, Harleen Batra,
Harsh Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway, Jetha
Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost
van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mohamed,
Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia,
Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago,
Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, Machel
Reid, Manvinder Singh, Mark Iverson, Martin G�rner, Mat Velloso, Mateo Wirth, Matt Davidow,
Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan,
Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad
Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda,
Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep

12

Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh
Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, S�bastien
M. R. Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom, Susan
Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi,
Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei,
Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei,
Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins,
Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav
Petrov, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena
Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi,
and Alek Andreev. Gemma 2: Improving open language models at a practical size, 2024. URL
https://arxiv.org/abs/2408.00118.

Open Ended Learning Team, Adam Stooke, Anuj Mahajan, Catarina Barros, Charlie Deck, Jakob
Bauer, Jakub Sygnowski, Maja Trebacz, Max Jaderberg, Michael Mathieu, Nat McAleese, Nathalie
Bradley-Schmieg, Nathaniel Wong, Nicolas Porcel, Roberta Raileanu, Steph Hughes-Fitt, Valentin
Dalibard, and Wojciech Marian Czarnecki. Open-ended learning leads to generally capable agents,
2021. URL https://arxiv.org/abs/2107.12808.

Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor
Gitman. Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data,
2024a. URL https://arxiv.org/abs/2410.01560.

Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman.
Openmathinstruct-1: A 1.8 million math instruction tuning dataset, 2024b. URL https://arxiv.
org/abs/2402.10176.

Xingyu Wu, Sheng hao Wu, Jibin Wu, Liang Feng, and Kay Chen Tan. Evolutionary computation in
the era of large language model: Survey and roadmap, 2024. URL https://arxiv.org/abs/
2401.10034.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok,
Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical
questions for large language models, 2024. URL https://arxiv.org/abs/2309.12284.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen.
Mammoth: Building math generalist models through hybrid instruction tuning, 2023. URL
https://arxiv.org/abs/2309.05653.

Jenny Zhang, Joel Lehman, Kenneth Stanley, and Jeff Clune. Omni: Open-endedness via models of

human notions of interestingness. arXiv preprint arXiv:2306.01711, 2023.

A Training Hyperparameters

We fine-tune the pre-trained version of S?2 for a single epoch on the resulting Dtrain. We use learning
rate lr = 2e ? 6 with a cosine schedule decaying to 2e ? 7 with a batch size of 16. Training is done
on a slice of 8x8 TPUv4s.

B Measuring Noisy Problem Quality via Perturbative Verification

In Section 2 we use a reference model S?2 to introduce the solve-rate as a proxy to measure a
problem-solution pair�s quality/difficulty. Figure 4 shows that harder problems (those successfully
verified less often than easy problems) more often contain logical inaccuracies leading to an incorrect
solution. As a result, many (Q, A) pairs evaluated as high-quality contain mistakes depending in
some way on the behavior of the verifier S?2. This makes the score rate a noisy proxy of quality and
leads to the following question: how can we efficiently filter out (Q, A) pairs with low solve-rate
which are logically invalid? One potential solution is to evaluate the validity of a sample (Q, A) via
the measured quality of its downstream mutations/children (Q?

1, A?

1), ..., (Q?

n, A?

n).

13

Figure 9: Child score distributions for valid and invalid samples.

I , A1

I ), ..., (Qn

The intuition goes as follows: suppose we have two pairs (QV , AV ), (QI , AI ) , with identical corre-
sponding quality scores q = qV = qI , which are valid/invalid respectively. Now suppose we sample
V , ..., An
n mutations of both pairs using generator G?1 to produce child sets (Q1
V ) and
(Q1
V , ..., qn
I . In order to
identify the invalid pair (QI , AI ) we might hope that the invalid sample generates more invalid
children than the valid sample. We investigate two possible measurements to detect whether this
is the case: 1) the mean difference of the scores of a parent with its children 2) the % of children
of a parent with quality q = 0. Formally, we define the mean difference as
and the

I ) with corresponding quality scores q1

V ), ..., (Qn
I , ..., qn

V , A1
V and q1

I , ..., An

(cid:80)n

i=1(q?qi)
n

|{qi:qi=0}|
n

child failure rate as
. To examine the quality distributions of children of valid and invalid
samples we take the 1000 annotated (Q, A) pairs in Section 3 and generate n = 16 mutations of each.
We then compute quality scores for each mutation via the score rate based on Gemma-2-9b as S?2 .
Figure 9 plots the distribution of mean child scores, differences between mean child score and parent
score, and % of children with a score of 0. The distribution of mean child scores for all samples is
left skewed towards 0 since the majority of parents have a score of 0. The mean scores of children
concentrate around their parent�s score with gaussian-like decay in the tails. When plotting the
distributions for valid/invalid samples we keep only the samples with solve-rate between l = 0.1 and
u = 0.5. This is done i) to remove samples with a score of 0 and ii) remove easy samples skewing the
scores of valid problems (since easy problems are more likely to be valid). Surprisingly, after filtering
in this way, we find valid and invalid samples have similar distributions of child scores. Both have a
mean child parent score difference of 0.03 and around 28% children with 0 score. This suggests a
different mutation mechanism more sensitive to the validity/invalidity of parents may be needed to
find a difference between respective child scores or a more reliable oracle may be needed.

C The Impact of Seed-Dataset Size

To investigate whether the seed dataset size impacts the relative performances of static uniform and
dynamic diverse data generation algorithms we randomly sample small seed dataset with 700 samples

14

Figure 10: Left: Solve-rate distribution of statically generated problems. Right: Solve-rate distribu-
tion of QD generated problems. The vast majority of generated problems are either too hard (SR = 0)
or too easy (SR = 1).

from D to produce Dsmall. We run both algorithms to produce 10,000 samples each and fine-tune
S?2 on the resulting data. We find the model trained on static uniformly generated data gets 40% on
MATH test whereas the model trained on dynamic diverse data gets 39% on MATH test.

D A Skill-Unbounded QD Algorithm

In Section 3 when implementing the dynamic diverse algorithm we limit the diversity of problems by
only considering the top M = 100 most commonly occurring skills. Additionally, when selecting
samples for mutation, we uniformly sample skill-sets. This selection procedures does not take into
account inter-skill-set similarities where one skill (e.g. Algebra) may be repeated across many
skill-sets. To promote even more sample diversity we propose the following modifications to the
dynamic diverse data generation algorithm:

� We allow for an unbounded set of possible skills generated by the skill-classifier LLM.
However, we still restrict the skill-set description of problem to the three most relevant
skills.

� We sample a set of skill-sets for mutation by solving the skill-sets optimization prob-
lem argmaxS DIV(S) where the diversity measure DIV of a collection of skill-sets
S = {?1, ..., ?n} is the number of unique skills ?i in S.

We use this algorithm to generate 100,000 synthetic problem-solution pairs and train a model on
the resulting data. The model gets 43% accuracy on MATH test and 21% accuracy on AIME with
K = 96 samples. This is slightly worse than the unmodified dynamic diverse method.

E Solve-rate Distributions of Synthetically Generated Problems

See Figure 10 for a histogram showing the solve-rate distributions of static uniformly and dynamic
diversely generated problems. The vast majority are never solved by S?2 (SOLVERATE = 0) or are
too easy (SOLVERATE = 1).

15

F Prompts

Mutation Prompt

You are tasked with generating a mutation conditioned on a set of input problems. You will
be shown the problems below.
{problem}
{solution}
Now generate a novel problem and solution. Enclose the problem in <problem>...</problem>
tags and the solution in <solution>...</solution> tags. Make sure to include the intended
final answer in the solution enclosed in the ...
latex style. If there are multiple numerical
answers, write them as a comma separated list (n1, n2, ...) .

Skill Classification Prompt

You will be a shown a reasoning problem below and solution below. Your job is to list the
relevant skills/lemmas used in solving the problem. Enclose all skills in a comma separated
list enclosed in the tags <skills>...</skills>. For example, if the problem is solved using a
combination of arithmetic and pigeonhole-principle, write <skills>arithmetic,pigeonhole-
principle</skills>. Only include up to {k} relevant skills.
{problem}
{solution}

G Synthetic problem samples

Table 1: Hard, valid sample (SR = 0.25).

Problem

What is the smallest base b for which the base b representation of 100010 requires
exactly 4 digits?

Solution

For a base b representation of 100010 to have exactly 4 digits, the largest power of b
that is less than 1000 must be 3. Therefore, we have the requirement that b3
le1000 < b4. We then realize that b = 6 satisfies this requirement since 63 <
1000 < 64. We also realize that this is the smallest possible value of b since if b
were less than 6, b4 would be less than 1000. Thus, our only solution is b = 6 .

Table 2: Easy, invalid sample (SR = 0.5625).

Problem

Solution

We have that 2x2f ?(x) ? 3xg(x) = h?(x) where f (x), g(x), and h(x) are all
polynomials in x and f ?(x) and h?(x) denote the derivatives of f (x) and h(x)
respectively. If the degree of h(x) is 6, what is the minimum possible degree of
g(x)?

Since h?(x) has degree 5 (one less than the degree of h(x)), the degree of the
left-hand side of the equation must also be 5. The term involving f ?(x) will have a
degree that is one less than the degree of f (x). Therefore, to achieve a degree of 5
on the left-hand side, the term ?3xg(x) must have a degree of 5. This means the
minimum degree of g(x) is 4 .

16

Table 3: Hard, invalid sample (SR = 0.125).

Problem

Find the quadratic polynomial, with real coefficients, which has ?2i as a root, and
where the constant term is 5.

Since the polynomial has real coefficients, the other root must be 2i. Thus, the
polynomial is

(x ? (?2i))(x ? 2i) = (x + 2i)(x ? 2i)

Solution

= x2 ? (2i)2
= x2 + 4

= x2 + 4 .

We then add 5 to get a constant term of 5:

x2 + 4 + 5 = x2 + 9 .

17


