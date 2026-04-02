From Complex to Simple: Unraveling the Cognitive Tree for Reasoning
with Small Language Models
Junbing Yan1, Chengyu Wang2, Taolin Zhang2, Xiaofeng He1,
Jun Huang2, Wei Zhang1,3?
1 School of Computer Science and Technology, East China Normal University
2 Alibaba Group
3 Shanghai Institute for AI Education
{junbingyan531,zhangwei.thu2011}@gmail.com, hexf@cs.ecnu.edu.cn
{chengyu.wcy,zhangtaolin.ztl,huangjun.hj}@alibaba-inc.com

Abstract

Reasoning is a distinctive human capacity, en-
abling us to address complex problems by
breaking them down into a series of manage-
able cognitive steps. Yet, complex logical rea-
soning is still cumbersome for language models.
Based on the dual process theory in cognitive
science, we are the first to unravel the cognitive
reasoning abilities of language models. Our
framework employs an iterative methodology
to construct a Cognitive Tree (CogTree). The
root node of this tree represents the initial query,
while the leaf nodes consist of straightforward
questions that can be answered directly. This
construction involves two main components:
the implicit extraction module (referred to as
the intuitive system) and the explicit reason-
ing module (referred to as the reflective sys-
tem). The intuitive system rapidly generates
multiple responses by utilizing in-context ex-
amples, while the reflective system scores these
responses using comparative learning. The
scores guide the intuitive system in its subse-
quent generation step. Our experimental re-
sults on two popular and challenging reasoning
tasks indicate that it is possible to achieve a per-
formance level comparable to that of GPT-3.5
(with 175B parameters), using a significantly
smaller language model that contains fewer pa-
rameters (<=7B) than 5% of GPT-3.5. 1

1

Introduction

The human brain is akin to a garden, where in-
stincts are seeds that sprout and grow, while rea-
son acts as the gardener, pruning and nurturing
the plants of knowledge to bloom into the flowers
of enlightenment. For machines, recently, Large
Language Models (LLMs) have demonstrated their
abilities to tackle diverse tasks through instanta-
neous question answering, exhibiting some levels

? Correspondence to Wei Zhang.
1The source code will be released in the EasyNLP frame-
work (Wang et al., 2022a). URL: https://github.com/
alibaba/EasyNLP

of intelligence (Ouyang et al., 2022; Wei et al.,
2022a; Wang et al., 2022b).

However, to cross the chasm between machines
and humans, three main challenges still lie ahead:
1) Reasoning ability. When it comes to mathemat-
ical and reasoning problems, the performance of
LMs is still not satisfactory (Koyejo et al., 2022;
Cobbe et al., 2021b). 2) Cognition capacity. The
evaluation and decision-making process regarding
the problem and its current state is of paramount
importance, especially when dealing with problems
that involve lengthy reasoning chains or multi-step
solutions. However, current methods (Wei et al.,
2022b; Yao et al., 2023) often lack comprehensive
validation and tend to focus on verifying intermedi-
ate results (Imani et al., 2023). 3) Efficiency. The
deployment and inference costs of LLMs are rel-
atively high, especially when utilizing parameter-
free inference enhancement techniques (Wei et al.,
2022b; Yao et al., 2023). These techniques require
extensive contexts and multiple steps of answer
generation, leading to a further increase in infer-
ence costs and time.

We suggest that valuable insights into address-
ing these challenges can be derived from the cog-
nitive processes of humans. In cognitive science,
the dual process theory (Evans, 1984, 2003, 2008;
Sloman, 1996) states that our brain initially em-
ploys an implicit, unconscious, and intuitive pro-
cess known as the Intuitive System, which re-
trieves relevant information. This is followed by
an explicit, conscious, and controllable reasoning
process called the Reflective System. The Intu-
itive System is capable of providing resources in
response to requests, while the Reflective System
facilitates a deeper exploration of relational infor-
mation through sequential thinking in the working
memory. Although slower, the Reflective System
possesses a unique human rationality (Baddeley,
2010). In complex reasoning tasks (including log-
ical reasoning and mathematical reasoning tasks),

3
2
0
2

v
o
N
2
1

]
L
C
.
s
c
[

1
v
4
5
7
6
0
.
1
1
3
2
:
v
i
X
r
a

solving complex reasoning problems.

� Improved Validation Capabilities. By ex-
posing the model to contrastive examples of
correct decisions versus incorrect or ambigu-
ous ones, we can improve the model�s ability
to make decisions (cognition ability). Addi-
tionally, apart from evaluating the model�s
judgment of intermediate results, we have also
integrated the model�s assessment of the over-
all correctness of the reasoning process.

� Efficient Framework. The combination of
the Intuitive System and the Reflective Sys-
tem can be applied to various reasoning tasks,
i.e., both logical reasoning and mathematical
reasoning. Notably, we are able to attain com-
parable reasoning performance to models with
substantially larger parameter sizes, such as
GPT-3.5 with 175B parameters, while utiliz-
ing relatively small language models (with
1.5B and 7B parameters). This allows the
trained small models to be deployed online
for efficient inference.

2 Cognitive Tree (CogTree) Framework

The reasoning ability of humans primarily arises
from acquiring pertinent information from the envi-
ronment and subconscious processing (Prystawski
and Goodman, 2023).
In our approach, we in-
corporate a tree structure to systematically tackle
reasoning problems, taking inspiration from hu-
man problem-solving procedures. This methodol-
ogy aligns with the planning processes analyzed
by Newell et al., 1959; Newell and Simon, 1972.
Newell and his colleagues defined problem solv-
ing (Newell et al., 1959) as the exploration of a
combinatorial problem space, represented as a tree.
In our mathematical and logical reasoning set-
ting, each node n in the cognitive tree T represents
either a theory in a logical set or the solution to a
sub-problem in a mathematical question. An edge
e of the tree corresponds to the evaluation of the
current node�s state s, which can be a confidence
score or a classification result. The problem decom-
position module (i.e., the Intuitive System) receives
the theory and the original problem as input, and
generates the decomposition of the original prob-
lem. Next, the newly generated nodes are used to
expand the tree, providing the reasoning module
(i.e., the Reflective System) with the information
that needs to be verified.

Figure 1: A schematic illustration of the proposed frame-
work named CogTree. An intuitive system is employed
to generate candidate plans, while a reflective system
verifies the plausibility of each plan to guide the next
generation of the intuitive system. This iterative process
is repeated to create the tree structure for reasoning.

these two systems coordinate with each other, en-
gaging in iterative cycles of fast and slow think-
ing (Daniel, 2017).

In this paper, we propose the Cognitive Tree
(CogTree) framework to address the aforemen-
tioned issues. Inspired by the dual process theory,
our system consists of the Intuitive System and the
Reflective System. In our implementation, the In-
tuitive System and the Reflective System are both
generative models, albeit with distinct objectives.
The Intuitive System employs in-context examples
to dissect intricate problems into sub-problems and
produce responses to the query. Conversely, the Re-
flective System evaluates the outcomes generated
by the Intuitive System and chooses the most likely
solution to provide guidance for the next generation
step. The aforementioned process is an iterative
tree generation process, which continues until the
original problem is decomposed into manageable
sub-problems (corresponding to nodes on the tree)
that can be easily solved.

Our main contributions are as follows:

� Problem Decomposition Paradigm. We pro-
pose a novel framework based on human cog-
nition called Cognitive Tree (CogTree) for

QueryLogicalReasoningMath ProblemPlanPlanPlanPlanTaskGenerationScoresReflectiveSystemIntuitiveSystemFigure 2: An illustration of how the Intuitive System and the Reflective System work to incrementally produce a
mathematical reasoning problem. At each step, the Intuitive System generates a set of decompositions based on
the query. The Reflective System then scores the candidate decompositions and returns the top-ranked ones. The
process terminates when the decomposition successfully matches the target answer.

The discernment capacity of the Reflective Sys-
tem plays a pivotal role in enhancing the overall
efficacy of the model (Imani et al., 2023). In par-
ticular, we utilize the cross-checking technique to
not only verify the precision of intermediate out-
comes but also validate the accuracy of the entire
reasoning process upon its completion. However,
relying solely on individual decision judgments
for training the model is inadequate (Imani et al.,
2023). To augment the model�s ability to evalu-
ate the state s, we propose the implementation of
a comparative reinforcement approach. This ap-
proach entails introducing a new training objective,
whereby the model is tasked with maximizing the
disparity in vector space between representations of
correct decisions and representations of incorrect
or ambiguous decisions.

In addition, efficiency pertains to the efficacy of
the problem decomposition and verification pro-
cess. Generating lengthy texts using a model as
massive as 175B entails substantial time and finan-
cial expenses. To tackle this issue, our method
can be implemented by simply fine-tuning a com-
paratively smaller model (1.5B or 7B) exclusively
for reasoning tasks. This enables us to deploy the
model for these tasks with minimal costs.

3

Implementation

In this section, we describe the implementations of
CogTree in detail.

3.1

Intuitive System

The generative capability of the Intuitive System
serves as the foundation for constructing the Cog-
nitive Tree. Thus, we choose decoder-only models
(e.g., GPT2-XL (Radford et al., 2019) or LLaMA-
7B (Touvron et al., 2023)) as the Intuitive System.
To enhance the effectiveness of the Intuitive Sys-
tem, we employ an in-context approach. Let us de-
fine the Query (Q) as the ultimate goal for logical
reasoning problems or the question to be answered
in mathematical problems. In the case of logical
reasoning problems, the Decomposition (D) in-
volves further breaking down the goal into smaller
components, where reasoning through this decom-
position enables the attainment of the goal. For
mathematical problems, it refers to one of the sub-
problems derived from the original problem, and
solving this sub-problem contributes to resolving
the original problem as a whole. The Decomposi-
tion set (Z) represents the collection of decompo-
sitions for all examples in the training set. In our
approach, we retrieve K examples (e.g., Query: Q;

Task:Math ProblemQuery:Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?Target Answer:Yesterday Weng earned $10. Intuitive SystemReflective SystemStep=1:Step=2:Sub-question: How much does Weng earn per minute?Answer: Weng earns 12/60 = $0.2 per minute.Sub-question: How much does babysitting charging per hour?Answer: Babysitting cost $12 per hour.Sub-question: How many hours is 50 minutes equivalent to?Answer: 50 minutes equals 1hours.1a. 1b. 1c. Sub-question: How long did Weng babysit yesterday?Answer: Weng works an hour.Sub-question: How much did Weng earn?Answer: Working 50 minutes, she earned 0.2 x 50 =$10.Sub-question: How many minutes did Weng babysit yesterday?Answer: 50 minutes.2a. 2b. 2c. SureImpossibleLikelySureLikelyImpossibleQuery: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?Decomposition: Sub-question: How much does Weng earn per minute? Answer: Weng earns 12/60 = $0.2 per minute.QueryDecomposition1Decomposition2GenerationP (Score1| Decomposition1)P (Score2| Decomposition2)ValidationGiven the current state s (Query: Q with De-
composition: D), we utilize the Reflective System,
which shares the same model architecture as the In-
tuitive System, to generate a score v that validates
the current state. This is represented by V (f?, s) ?
f?(v|s). Additionally, based on the complete rea-
soning chain S = {s1, � � � , si, � � � , sn}, we em-
ploy the Reflective System to produce an overall
score o, which can be expressed as O(f?, S) ?
f?(o|S).

To generate the scores v and o, we utilize the
model to produce a classification result. Since
the answers generated by the Intuitive System can
sometimes be misleading and cannot be accurately
assessed at this stage, we adopt a prompt-based
approach and treat it as a classification problem,
where the model outputs one of three categories:
sure, impossible, or likely. The likely response sig-
nifies that the generated answer is plausible but
requires further verification.

3.3 Training

Intuitive System. Supervised Fine-tuning (SFT)
has demonstrated its effectiveness in aligning with
human intentions (Ouyang et al., 2022). In our
approach, the Intuitive System is designed to de-
compose the queries (i.e., complex problems) into
sub-problems by leveraging in-context examples.
Since we employ generative models as our Intuitive
System, the loss calculation is only necessary for
the generative text (without the given context) dur-
ing auto-regressive computation. Given a sample
of tokens with a length of N denoted as X, where
X = {x1, � � � , xi, � � � , xn}. Furthermore, we de-
fine the sequence length of in-context examples as
M . We use a standard language modeling objective
to maximize the following likelihood function:

LIS =

N
(cid:88)

i>M

log P (xi|x1, � � � , xi?1; ?)

(1)

Reflective System. The acquisition of the Reflec-
tive System can be achieved through the same train-
ing approach as the Intuitive System, which in-
volves utilizing positive and negative samples to
obtain classification results from the model. Since
the Reflective System is primarily focused on gen-
erating judgments for a given state s, the loss func-
tion can be defined as follows:

Figure 3: Example of Query and Decomposition for
logical reasoning.

Figure 4: Example of Query and Decomposition for
mathematical problems.

Decomposition: D) from the inference decomposi-
tion set (Z), which are then utilized as the context
for the model�s input. Detailed examples are shown
in Figure 3 and Figure 4.

Then, the output can be generated as y ?
f?(y|x, z1���K). Here, z represents the K exam-
ples2 recalled from the decompositions set Z,
where Z = {z1, � � � , zL}.
In practice, we use
the Intuitive System to obtain the representation
of the current query (final transformer block�s ac-
tivation) and calculate the cosine similarity with
the representations of other queries in set Z. We
then retrieve the K most similar queries from the
set. Moreover, [y] ? f?(y|x, z1���K) is sampled as
a continuous language sequence.

3.2 Reflective System

The Reflective System differs from the Intuitive
System in terms of its approach of generating in-
sights. While the Intuitive System relies on quick
intuition, the Reflective System�s role is to evaluate
the decompositions to determine their acceptability.
In practice, we employ two methods to verify the
results: the verification of intermediate processes
and the verification of the entire reasoning chain.

2In the experiment, we set K = 5.

LRS = log P (v|s; ?)

(2)

Step=1:Query1: earth rotating on its axis causes cycles of day and night on earthDecomposition1: [ earth is a planet that rotates on its tilted axis ] + [ a planet rotating causes cycles of day and night on that planet ]Step=2:Query2: earth is a planet that rotates on its tilted axisDecomposition2: [ the earth rotates on its tilted axis ] + [ earth is a kind of planet ]Step=1:Query1: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?Decomposition1: How many clips did Natalia sell in May? Natalia sold 48/2 = 24 clips in May.Step=2:Query2: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Decomposition1: How many clips did Natalia sell in May? Natalia sold 48/2 = 24 clips in May.Decomposition2: How many clips did Natalia sell altogether in April and May? Natalia sold 48+24 = 72 clips altogether in April and May.Dataset

Input

Output

EB

A hypothesis that needs to be proven and a set of theory.
(Hypo: Phobos is a kind of moon. Theory: [Mars is
a kind of planet; moons orbit planets; Phobos orbits
Mars.])

GSM8K A math word problem (Natalia sold clips to 48 of her
friends in April, and then she sold half as many clips
in May. How many clips did Natalia sell altogether in
April and May?)

Yes or No to indicate whether or not the hypothesis
can be proven based on the theory. (Yes)

A number denoting the solution to the mathematical
problem. (72)

Table 1: Task overview. Examples of input and output are printed in blue.

Dataset

Split

# Samples Max Steps Avg Steps

EB

Train
Dev
Test

GSM8K Train
Dev
Test

1313
187
340

6726
747
1319

17
15
11

9
8
11

3.2
3.2
3.3

3.6
3.5
3.7

i, � � � , v?

1, � � � , v?

{v?
n}. By maximizing the distance
between v? and the correct answers v, we can en-
hance the learning process. Let g(v?, �)3 be a match-
ing function between negative sample v? and the
positive sample v. The loss function for contrastive
learning can be expressed as follows:

Table 2: Data statistics of EB and GSM8K.

LCL =

However,

the effectiveness of this training
In cog-
method is found to be unsatisfactory.
nitive theory, human decision-making behavior
arises from the comparative analysis of various op-
tions (Festinger, 1957). Drawing inspiration from
the cognitive theory, we adopt a contrastive learn-
ing approach to enhance the model�s ability to dis-
tinguish between different states. The fundamental
concept of contrastive learning is to learn represen-
tations of positive and negative samples by max-
imizing their distance in the sample space (Chen
et al., 2020). Consequently, the selection of nega-
tive samples plays a critical role in determining the
effectiveness of contrastive learning.

For logic reasoning datasets, one approach to
generate more challenging negative examples is to
replace one of the theories in decomposition with
another theory from the current theory set. This
negative example is more challenging for the model
to distinguish because the theories within the same
theory set are more similar.

For mathematical problems, since our exper-
imental dataset GSM8K (Cobbe et al., 2021a)
only provides the correct answer itself, it does
not offer incorrect solutions. We use the dataset
PRM800K (Lightman et al., 2023) to enhance
the learning process, where there are ambigu-
ous responses S? = {s?
n} (seem-
ingly correct but actually incorrect). The judg-
ment generated by Reflective System is V ? =

1, � � � , s?

i, � � � , s?

exp(f (v, y))

exp(f (v, y)) + (cid:80)

v??V ? exp(f (v, v?))

(3)
Hence, the total loss function for the Reflective

System is given by:

Ltotal = ? � LRS + (1 ? ?) � LCL

(4)

Here, ? is the hyper-parameter. We conduct ex-
periments on ? and choose ? = 0.5 as the best
setting.4

4 Experiments

We perform an extensive evaluation of our method
utilizing two well-established benchmark datasets:
the Entailment Bank (EB) (Dalvi et al., 2021) and
GSM8K (Cobbe et al., 2021a). EB consists of
human-annotated tuples containing information
about theories, provable goals, and correspond-
ing reasoning paths. GSM8K, on the other hand,
presents a challenging arithmetic reasoning task
that language models frequently find difficult to
tackle (Hendrycks et al., 2021; Cobbe et al., 2021b).
Examples of these two datasets are in Table 1. The
dataset statistics are shown in Table 2.

4.1 Experimental Setup

In our experiments, we use GPT2-XL (Radford
et al., 2019) and LLaMA-7B (?) from the Hugging-
face transformers (Wolf et al., 2020) library as the
underlying models for the Intuitive System and the
Reflective System.

3In the experiment, we use cosine similarity as g(v?, �)
4For specific details, please refer to the Section 4.7.

Model

#Params. Accuracy (%) ? (%) Accuracy (%) ? (%)

EB

GSM8K

Comparative Systems

GPT-3.5 (code-davinci-002)

+ Standard prompt
+ Chain-of-thought prompt
+ Tree-of-thought prompt

Our Models (CogTree)

GPT2-XL (Intuitive System only)

+ GPT2-XL (as Reflective System)
+ LLaMA (as Reflective System)

LLaMA (Intuitive System only)

+ GPT2-XL (as Reflective System)
+ LLaMA (as Reflective System)

175B
175B
175B
175B

1.5B
1.5B
7B

7B
1.5B
7B

80.76
84.23
92.45
93.31

82.37
92.63
93.16

86.14
93.25
94.25

-
+3.47
+11.69
+12.55

-
+10.26
+10.79

-
+ 7.11
+8.11

16.17
17.03
60.27
61.39

23.53
35.84
34.68

43.52
47.80
61.28

-
+0.86
+44.10
+45.22

-
+12.31
+11.15

-
+4.28
+17.76

Table 3: Overall test set performance in terms of accuracy and relative improvement.

During training, we use the Adam optimizer with
?1 = 0.9, ?2 = 0.999, ? = 1e ? 8. We use the
learning rate ? = 1e ? 4 for both Systems. We
use a batch size of 4 and set a large epoch number
(i.e., 100) and use the validation set to do early
stopping. In practice, the best epoch is often within
50. During the inference stage, in each step, we use
the Intuitive System to generate the top_beam=3
answers and then let the Reflective System select
the most probable answer to continue generating
for the next step. We add an �end� marker. The
inference process stops when the Intuitive System
generates the �end� marker or when the maximum
number of inferences, which is 20, is reached. For
each Query, we perform 5 complete reasoning pro-
cess.5

Following Zhou et al., 2022, we use code-
davinci-0026 (code-davinci-002 is constructed on
the foundation of the GPT-3.5 architecture) for
comparative experiment due to its strong reason-
ing abilitiy and employ various prompt strate-
gies to conduct our experiments on GPT2-XL and
LLaMA-7B. For each example, we sample Stan-
dard prompting (detailed cases to be described in
Section 4.2) and Chain-of-thought (CoT) prompt-
ing for 100 times for average performance. For
Tree-of-thought (ToT), at each step we generate 5
candidate answers and sample values 3 times for
each example.

4.2 Results on Entailment Bank

On EB, followed by Zhao et al., 2023, we assess the
capabilities of systems in distinguishing between

5Detailed examples are shown in Figure 7 and Figure 8.
6https://openai.com/

provable and non-provable goals. To accomplish
this, we assign a non-provable goal to each devel-
opment and testing theories by selecting it from
other (theory, goal, reasoning path) samples. The
selection is adversarial: We input all the goals in
the set into our pre-trained model separately, ob-
taining the last output of the last transformer block
as the representation of each goal. We proceed by
computing the cosine similarity between all non-
provable goals and the provable goal. Based on
this computation, we identify the hard negative
example with the highest similarity. For a given
theory T and a query Q, we allow the system to
generate a reasoning path S and obtain the proof
value o = f?(o|S) for that path. Given the choices
�Sure/Likely/Impossible�, we say �Q is provable�
when the value o is �sure� and not provable other-
wise.

Baselines. Vanilla: The raw input to the model
is as follows: Query: Q; Theory Set: T , followed
by a question: Based on the theory, can the goal
be approved? Please answer with yes or no. Stan-
dard prompt: We use an input-output (IO) prompt
with 5 in-context examples (e.g. Query: Q; Theory
Set: T ; Answer: A.). Chain-of-though prompt:
For chain-of- thought (CoT) prompting, we aug-
ment each input-output pair using examples with
complete reasoning chain (e.g. Query: Q; The-
ory Set: T ; Reasoning Chain: S; Answer: A.).
Tree-of-thought prompt: In each step, we provide
the model with the theory set and ask the model
to select two theories from the set to generate a
new inference, which is added to the theory set
while removing the two selected theories. Each
step generates five candidates, and the model gen-

answer each sub-question sequentially, ultimately
deriving the final answer.
Baselines. Vanilla: Directly input the question and
let the model to answer (e.g. Query: Q). Standard
prompt: We use an input-output (IO) prompt with
5 in-context examples (e.g. Query: Q; Answer: A).
Chain-of-though prompt: For CoT, we use the
step-by-step solution of mathematical problems as
enhanced input-output pair (e.g. Query: Q; Rea-
soning Chain: S; Answer: A). Tree-of-though
prompt: We adopt the methodology described in
ToT (Yao et al., 2023). Our implementation in-
volves the incremental generation of the answer,
with the model producing one step at a time. We
sample the results five times for each step and em-
ploy the model to assess the generated outcomes
until the final answer is obtained7.
Main Results. Experimental results are shown
in Table 3. It is evident that the direct question-
answering accuracy of GPT-3.5 is merely 16%.
The traditional input-output approach did not im-
prove its effectiveness. In contrast, CoT and ToT
demonstrate a significant enhancement in solving
mathematical problems, with an improvement of
approximately 44%. This indicates the crucial role
of providing the model with example reasoning
chains for tasks involving multi-step inference. Our
SFT-improved GPT2 achieves only 23.5% accu-
racy, which may be attributed to the scale low (Ka-
plan et al., 2020) caused by model parameters. The
performance of our LLaMA-7B and GPT-3.5 mod-
els is nearly indistinguishable, suggesting that fine-
tuning small models using our provided method
can approach the performance of larger models.
Notably, the inclusion of the Reflective System on
GSM8K leads to a greater overall improvement
compared to EB, indicating that using the Reflec-
tive System can yield better results, particularly in
more complex problems.
Error Analysis. We have found that the examples
where the model fails to answer can be divided into
two categories. The first category is the failure in
decomposing the question into subproblems (De-
tailed examples are shown in Figure 5), which is
consistent with the findings of Zhou et al. (2022).
Such failures can be resolved through manual de-
composition. The second category of failure is

7It is worth noting that our tree construction method differs
significantly from ToT (Yao et al., 2023). ToT employs a
bottom-up approach, whereas we utilize explicit questioning
to break down the problem for the model and address it in a
top-down fashion, step by step.

Figure 5: Samples failed at each step on EB (left) and
GSM8K (right).

erates subsequent inferences based on the highest
likelihood until the set ultimately contains only
one inference, which is compared with the goal to
determine consistency (e.g., theoryi + theoryj ->
inference (theory set without theoryi&j)).
Main results. Table 3 presents the classification ac-
curacy on EB. The results demonstrate that GPT2-
XL (1.5B), trained exclusively on in-context exam-
ples, outperforms GPT-3.5 (175B) despite having
fewer than 1% of the model�s parameters. By in-
corporating CoT and ToT augmentation methods,
the accuracy of GPT-3.5 is substantially enhanced,
reaching 92-93%. Furthermore, when our approach
is combined with the Reflective System for result
verification, even higher performance is achieved
(94% by LLaMA-7B), surpassing the prompt aug-
mentation method employed by GPT-3.5.
Error Analysis. Although our model has achieved
satisfactory results, it still exhibits deficiencies in
certain cases. We have conducted an examination
of instances where the model�s inference has failed
(Detailed examples are shown in Figure 5). Specif-
ically, we observed a decline in the model�s accu-
racy when the length of the inference chain exceeds
10 steps. Our model struggles to determine the ap-
propriate decomposition of the goal in such cases.
This limitation may arise from the excessively long
and divergent nature of the chain required to reach
the goal, which surpasses the model�s current ca-
pacity for abstraction in this regard. It is worth
noting that even for humans, this task can be com-
plex, often requiring multiple attempts to arrive at
the final inference chain.

4.3 Results on GSM8K

On GSM8K, we employ in-context examples to
enable the model to generate sub-questions for a
given problem. In the implementation, we conduct
five sampling iterations at each decomposition step
and choose the one with the highest probability for
generating the next step. Once all the sub-questions
have been generated, we proceed to have the model

<=2345>=601234Error Rate (%)GPT2-XLLLaMA-7B<=2345>=6691215Error Rate (%)GPT2-XLLLaMA-7BEB

GSM8K

4.5 Compared with Other Finetune SOTAs

Model

GPT-XL

+Specialize
+DecomP
+Self-Ask
+CogTree

81.54% 21.43%
83.27% 24.35%
82.69% 25.46%
93.16% 34.68%

LLaMA-7B

+Specialize
+DecomP
+Self-Ask
+CogTree

84.32% 34.21%
82.43% 39.75%
83.98% 38.35%
94.25% 61.28%

Table 4: Performance metrics for different methods with
EB and GSM8K.

Model

GPT-XL

- decomposition

EB

GSM8K

92.63% 35.84%
62.38% 18.94%

LLaMA-7B

- decomposition

93.25% 61.28%
69.79% 27.16%

Table 5: Ablation study in terms of F1.

when the model provides inaccurate answers to
the subproblems. These failures are frequently ob-
served in GPT2-XL and are the main reason for the
unsatisfactory performance of GPT2. One possible
reason for these failures is the inadequacy of the
model�s parameter size, which hinders its ability to
acquire fundamental mathematical capabilities.

4.4 Backbone Modifications

We conduct experiments employing different back-
bones for the Intuitive System and the Reflective
System. The results presented in Table 3 demon-
strate that when GPT2-XL is utilized as the In-
tuitive System and LLaMA-7B as the Reflective
System, there is an observed improvement in over-
all performance on the EB dataset, compared to
using GPT2-XL as the Reflective System. How-
ever, when a more powerful model is employed
as the Reflective System on the GSM8K dataset,
there is no noticeable performance enhancement.
This finding suggests that the Intuitive System re-
stricts the system�s performance on the GSM8K
dataset. Conversely, when LLaMA-7B serves as
the Intuitive System and GPT2-XL as the Reflec-
tive System, the performance improvement on the
GSM8K dataset, in comparison to using LLaMA-
7B as the Reflective System, is not substantial. This
indicates that in this particular case, the Reflective
System limits the overall system�s performance.

In pursuit of fairness, we conducted a comparison
of state-of-the-art methods that had been fine-tuned
on identical datasets, in contrast to our prior eval-
uation of methods relying on GPT-3.5, which is
parameter-free.
DecomP (Khot et al., 2023) solves complex tasks
by decomposing them (via prompting) into simpler
sub-tasks. Specialize (Fu et al., 2023) proposes
small model specialization to enhance performance
by focusing model capacity on a specific target task.
Self-Ask (Press et al., 2022) explicitly asks the
model itself (and then answers) follow-up questions
before answering the initial question.

It can be observed in Table 4, when compared
with the finetune-based methods, Cogtree still
achieves state-of-the-art results. This is attributed
to not only decomposing the problem but also in-
corporating result validation in subsequent steps.

4.6 Ablation Study

We conducted further experiments to demonstrate
the effectiveness of decomposing step by step. "w/
decomposition" indicates that we used the method
from our paper to decompose and sequentially an-
swer the original problem. On the other hand, "w/o
decomposition" means that we did not use the in-
termediate problem decomposition and directly an-
swered the original problem, relying on System 1
to generate the answer.

As we can see in the Table 5, the accuracy of di-
rectly answering the original problem is low, espe-
cially when the original problem is complex. This
is also in line with human cognition. When solving
math problems, we also solve intermediate prob-
lems first and then obtain the final answer, which
improves our accuracy.

4.7 Hyper-parameter Analysis

We vary one hyper-parameter with others fixed.
From Figure 6, as ? increases, the performance
first increases and then drops, and it can achieve
the best result when ? = 0.5.

5 Related Work

Multi-step Reasoning. Reasoning has long been a
key focus in natural language processing research.
Initially, most studies concentrated on basic tasks
like single-sentence language inference (Zamansky
et al., 2006; MacCartney and Manning, 2009; An-
geli et al., 2016; Hu et al., 2020; Chen et al., 2021)

problems into sub-problems, and the edges rep-
resent judgments regarding the correctness of the
decomposition. Based on the implementation of
our approach over GPT2 and LLaMA, we have
achieved results comparable to GPT-3.5 on EB and
GSM8K datasets, indicating the effectiveness of
our framework.

Limitations

Due to the limitation of computational resources,
we did not test our method on larger scale models.
As the model size increases, using our approach
may lead to further improvement in the accuracy
of answers to these questions. Another direction
worth exploring is diversifying the validation meth-
ods for the Reflective System, such as using multi-
verification to compare the generated results and
select the optimal answer.

Acknowledgements

This work was supported in part by National Natu-
ral Science Foundation of China under Grant (No.
62072182, No. 92270119), the Fundamental Re-
search Funds for the Central Universities, and by
Alibaba Group through Alibaba Research Intern
Program.

References

Gabor Angeli, Neha Nayak, and Christopher D. Man-
ning. 2016. Combining natural logic and shallow
reasoning for question answering. In Proceedings
of the 54th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 442�452, Berlin, Germany. Association for
Computational Linguistics.

Alan Baddeley. 2010. Working memory. Scholarpedia,

5(2):3015.

Kaj Bostrom, Zayne Sprague, Swarat Chaudhuri, and
Greg Durrett. 2022. Natural language deduction
through search over statement compositions. In Find-
ings of the Association for Computational Linguistics:
EMNLP 2022, Abu Dhabi, United Arab Emirates, De-
cember 7-11, 2022, pages 4871�4883. Association
for Computational Linguistics.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and
Geoffrey E. Hinton. 2020. A simple framework for
contrastive learning of visual representations. In Pro-
ceedings of the 37th International Conference on
Machine Learning, ICML 2020, 13-18 July 2020, Vir-
tual Event, volume 119 of Proceedings of Machine
Learning Research, pages 1597�1607. PMLR.

Figure 6: The impact of the hyper-parameter ? on EB
(left) and GSM8K (right).

and commonsense inference (Rajani et al., 2019;
Latcinnik and Berant, 2020; Shwartz et al., 2020)
in a single step. Lately, there has been a surge
in research interest regarding multi-step reasoning
and mathematical problem solving. The search
space for a correct reasoning path is extremely vast
and complex. Previous approaches (Bostrom et al.,
2022; Creswell et al., 2022; Zhao et al., 2023) pre-
dominantly emphasized a bottom-up reasoning ap-
proach, with a strong focus on system design. In
contrast, our approach to constructing the cognitive
tree adopts a top-down methodology, reducing the
burden during the era of generative models. This
top-down approach offers greater flexibility for ap-
plication across models of varying scales.
LLM as Evaluation. The utilization of Language
Models (LLMs) to evaluate the validity of their
own predictions is gaining significance as a proce-
dure in problem-solving. The introduction of the
self-reflection mechanism by Shinn et al., 2023;
Madaan et al., 2023; Paul et al., 2023 involves LMs
providing feedback to their generated candidates.
Tree of Thought (Yao et al., 2023) and our approach
share a common utilization of a tree-based struc-
ture for problem-solving. However, ToT primarily
concentrates on tree construction and limited self-
validation of intermediate results. According to the
dual process theory, which suggests that validation
requires deeper levels of thinking, our approach
incorporates a contrastive learning method to en-
hance the model�s ability to distinguish accurate
results and facilitate comprehensive global valida-
tion of the generated outcomes.

6 Conclusion

In this paper, we proposed a new framework named
CogTree to address complex logical reasoning and
mathematical problems. The process of reason-
ing involves constructing a Cognitive Tree, where
the nodes represent the decomposition of complex

0.10.30.50.70.9919293949596Accuracy (%)LLaMAGPT2-XL0.10.30.50.70.930405060Accuracy (%)LLaMAGPT2-XLZeming Chen, Qiyue Gao, and Lawrence S. Moss. 2021.
NeuralLog: Natural language inference with joint
In Proceedings of
neural and logical reasoning.
*SEM 2021: The Tenth Joint Conference on Lexical
and Computational Semantics, pages 78�88, Online.
Association for Computational Linguistics.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021a. Training verifiers to solve math word prob-
lems. CoRR, abs/2110.14168.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Jacob Hilton, Reiichiro Nakano, Christopher Hesse,
and John Schulman. 2021b. Training verifiers to
solve math word problems. CoRR, abs/2110.14168.

Antonia Creswell, Murray Shanahan, and Irina Higgins.
2022. Selection-inference: Exploiting large language
models for interpretable logical reasoning. CoRR,
abs/2205.09712.

Bhavana Dalvi, Peter Jansen, Oyvind Tafjord, Zhengnan
Xie, Hannah Smith, Leighanna Pipatanangkura, and
Peter Clark. 2021. Explaining answers with entail-
ment trees. In Proceedings of the 2021 Conference
on Empirical Methods in Natural Language Process-
ing, pages 7358�7370, Online and Punta Cana, Do-
minican Republic. Association for Computational
Linguistics.

Kahneman Daniel. 2017. Thinking, fast and slow.

Jonathan Evans. 2008. Dual-processing accounts of
reasoning, judgment, and social cognition. Annual
review of psychology, 59:255�78.

Jonathan St B. T. Evans. 1984. Heuristic and analytic
processes in reasoning*. British Journal of Psychol-
ogy, 75(4):451�468.

Jonathan St.B.T. Evans. 2003.

In two minds: dual-
process accounts of reasoning. Trends in Cognitive
Sciences, 7(10):454�459.

Leon Festinger. 1957. Cognitive Dissonance.

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and
Tushar Khot. 2023. Specializing smaller language
In Interna-
models towards multi-step reasoning.
tional Conference on Machine Learning, ICML 2023,
23-29 July 2023, Honolulu, Hawaii, USA, volume
202 of Proceedings of Machine Learning Research,
pages 10421�10430. PMLR.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul
Arora, Steven Basart, Eric Tang, Dawn Song, and Ja-
cob Steinhardt. 2021. Measuring mathematical prob-
lem solving with the MATH dataset. In Proceedings
of the Neural Information Processing Systems Track
on Datasets and Benchmarks 1, NeurIPS Datasets
and Benchmarks 2021, December 2021, virtual.

Hai Hu, Qi Chen, Kyle Richardson, Atreyee Mukher-
jee, Lawrence S. Moss, and Sandra Kuebler. 2020.
MonaLog: a lightweight system for natural language
In Proceedings
inference based on monotonicity.
of the Society for Computation in Linguistics 2020,
pages 334�344, New York, New York. Association
for Computational Linguistics.

Shima Imani, Liang Du, and Harsh Shrivastava. 2023.
Mathprompter: Mathematical reasoning using large
language models. CoRR, abs/2303.05398.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B.
Brown, Benjamin Chess, Rewon Child, Scott Gray,
Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.
Scaling laws for neural language models. CoRR,
abs/2001.08361.

Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao
Fu, Kyle Richardson, Peter Clark, and Ashish Sab-
harwal. 2023. Decomposed prompting: A modular
approach for solving complex tasks. In The Eleventh
International Conference on Learning Representa-
tions, ICLR 2023, Kigali, Rwanda, May 1-5, 2023.
OpenReview.net.

Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Bel-
grave, K. Cho, and A. Oh, editors. 2022. Advances
in Neural Information.

Veronica Latcinnik and Jonathan Berant. 2020. Explain-
ing question answering models through text genera-
tion. arXiv preprint arXiv:2004.05569.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Har-
rison Edwards, Bowen Baker, Teddy Lee, Jan
Leike, John Schulman, Ilya Sutskever, and Karl
Cobbe. 2023. Let�s verify step by step. CoRR,
abs/2305.20050.

Bill MacCartney and Christopher D. Manning. 2009.
In Proceed-
An extended model of natural logic.
ings of the Eight International Conference on Com-
putational Semantics, pages 140�156, Tilburg, The
Netherlands. Association for Computational Linguis-
tics.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Sean Welleck, Bodhisattwa Prasad Majumder,
Shashank Gupta, Amir Yazdanbakhsh, and Peter
Clark. 2023. Self-refine: Iterative refinement with
self-feedback. CoRR, abs/2303.17651.

A. Newell and H.A. Simon. 1972. Human problem

solving. UNESCO (Paris).

Allen Newell, J. C. Shaw, and Herbert A. Simon. 1959.
Report on a general problem-solving program. In
Information Processing, Proceedings of the 1st In-
ternational Conference on Information Processing,
UNESCO, Paris 15-20 June 1959, pages 256�264.
UNESCO (Paris).

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll L. Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray,
John Schulman, Jacob Hilton, Fraser Kelton, Luke
Miller, Maddie Simens, Amanda Askell, Peter Welin-
der, Paul F. Christiano, Jan Leike, and Ryan Lowe.
2022. Training language models to follow instruc-
tions with human feedback. In NeurIPS.

Debjit Paul, Mete Ismayilzada, Maxime Peyrard, Beat-
riz Borges, Antoine Bosselut, Robert West, and Boi
Faltings. 2023. REFINER: reasoning feedback on in-
termediate representations. CoRR, abs/2304.01904.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2022. Measuring
and narrowing the compositionality gap in language
models. CoRR, abs/2210.03350.

Ben Prystawski and Noah D. Goodman. 2023. Why
think step-by-step? reasoning emerges from the lo-
cality of experience. CoRR, abs/2304.03843.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
Dario Amodei, Ilya Sutskever, et al. 2019. Language
models are unsupervised multitask learners. OpenAI
blog, 1(8):9.

Nazneen Fatema Rajani, Bryan McCann, Caiming
Xiong, and Richard Socher. 2019. Explain your-
self! leveraging language models for commonsense
reasoning. In Proceedings of the 57th Annual Meet-
ing of the Association for Computational Linguistics,
pages 4932�4942, Florence, Italy. Association for
Computational Linguistics.

Noah Shinn, Beck Labash, and Ashwin Gopinath. 2023.
Reflexion: an autonomous agent with dynamic mem-
ory and self-reflection. CoRR, abs/2303.11366.

Vered Shwartz, Peter West, Ronan Le Bras, Chandra
Bhagavatula, and Yejin Choi. 2020. Unsupervised
commonsense question answering with self-talk. In
Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pages 4615�4629, Online. Association for Computa-
tional Linguistics.

Steven Sloman. 1996. The empirical case for two sys-
tems of reasoning. Psychological Bulletin, 119:3�.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timoth�e Lacroix,
Baptiste Rozi�re, Naman Goyal, Eric Hambro, Faisal
Azhar, Aur�lien Rodriguez, Armand Joulin, Edouard
Grave, and Guillaume Lample. 2023. Llama: Open
and efficient foundation language models. CoRR,
abs/2302.13971.

Chengyu Wang, Minghui Qiu, Taolin Zhang, Tingting
Liu, Lei Li, Jianing Wang, Ming Wang, Jun Huang,
and Wei Lin. 2022a. Easynlp: A comprehensive and
easy-to-use toolkit for natural language processing.
In Proceedings of the The 2022 Conference on Empir-
ical Methods in Natural Language Processing, pages
22�29. Association for Computational Linguistics.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Al-
isa Liu, Noah A. Smith, Daniel Khashabi, and Han-
naneh Hajishirzi. 2022b. Self-instruct: Aligning lan-
guage model with self generated instructions. CoRR,
abs/2212.10560.

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin
Guu, Adams Wei Yu, Brian Lester, Nan Du, An-
drew M. Dai, and Quoc V. Le. 2022a. Finetuned
language models are zero-shot learners. In The Tenth
International Conference on Learning Representa-
tions, ICLR 2022, Virtual Event, April 25-29, 2022.
OpenReview.net.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le,
and Denny Zhou. 2022b. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
NeurIPS.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, R�mi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le
Scao, Sylvain Gugger, Mariama Drame, Quentin
Lhoest, and Alexander M. Rush. 2020. Transformers:
State-of-the-art natural language processing. In Pro-
ceedings of the 2020 Conference on Empirical Meth-
ods in Natural Language Processing: System Demon-
strations, EMNLP 2020 - Demos, Online, November
16-20, 2020, pages 38�45. Association for Computa-
tional Linguistics.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Thomas L. Griffiths, Yuan Cao, and Karthik
Narasimhan. 2023. Tree of thoughts: Deliberate
problem solving with large language models. CoRR,
abs/2305.10601.

Anna Zamansky, Nissim Francez, and Yoad Winter.
2006. A �natural logic�inference system using the
lambek calculus. Journal of Logic, Language and
Information.

Hongyu Zhao, Kangrui Wang, Mo Yu, and Hongyuan
Mei. 2023. Explicit planning helps language models
in logical reasoning. CoRR, abs/2303.15714.

Denny Zhou, Nathanael Sch�rli, Le Hou, Jason Wei,
Nathan Scales, Xuezhi Wang, Dale Schuurmans,
Olivier Bousquet, Quoc Le, and Ed H. Chi. 2022.
Least-to-most prompting enables complex reasoning
in large language models. CoRR, abs/2205.10625.

A Appendix

Below we present some successful and failed cases
of CogTree for analysis.

Figure 7: A successful case in EB of 3 total steps. Green represents answers judged as �sure� by the Reflective
System. Gray represents answers judged as �likely� by the Reflective System, and red represents answers judged as
�impossible� by the Reflective System.

Figure 8: A successful case in GSM8K of 2 total steps. Green represents answers judged as �sure� by the Reflective
System. Gray represents answers judged as �likely� by the Reflective System, and red represents answers judged as
�impossible� by the Reflective System.

Step=1:Query1: the sun will be the star that appears the brightest to the earthDecomposition1-1: [ as the stars become closer, the light of the stars will appear brighter] + [ the sun is the star that is closest to earth]Decomposition1-2: [ as the stars become closer, the light of the stars will appear brighter] + [ a source of something produces that something ]Decomposition1-3: [ a star produces light] + [ the sun is the star that is closest to earth]Step=2:Query2: as the stars become closer, the light of the stars will appear brighterDecomposition2-1: [ stars are a source of light ] + [ as a source of light becomes closer , the light will appear brighter ]Decomposition2-2: [ a source of something produces that something] + [ stars are a source of light]Decomposition2-3: [ a star produces light ] + [ a source of something produces that something ] Step=3:Query3: stars are a source of lightDecomposition3-1: [ a star produces light ] + [ a source of something produces that something ]Decomposition3-2: [ a source of something produces that something] + [ a star produces light]Decomposition3-3: [ the sun is a star ] + [ a star produces light ]Step=1:Query1: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?Decomposition1-1: How many clips did Natalia sell in May? Natalia sold 48/2 = 24 clips in May.Decomposition1-2: How many clips did Natalia sell inApril? Natalia sold 48clips in April.Decomposition1-3: How many clips did Natalia sell in May? Natalia sold 48clips in May.Step=2:Query2: : Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Decomposition1: How many clips did Natalia sell in May? Natalia sold 48/2 = 24 clips in May.Decomposition2-1: How many clips did Natalia sell altogether in April and May? Natalia sold 48+24 = 72 clips altogether in April and May.Decomposition2-2: How many clips did Natalia sell altogether in April and May? Natalia sold 48+48= 96in April and May.Decomposition2-3: How many clips did Natalia sell altogether in April and May? Natalia sold 48in April and May.Figure 9: A failed case in EB of 11 total steps. Our system stops generating after Step 5.

Figure 10: A failed case in GSM8K of 6 total steps. Our system stops generating after Step 4.

Step=1:Query1: two tectonic plates pushing toward each other changes the earth's crust material into metamorphic rockDecomposition1: [ exposure to intense heat and pressure changes the material of earth's crust into metamorphic rock] +[ two tectonic plates pushing toward each other cause intense heat and pressure in the earth's crust]Step=2:Query2: exposure to intense heat and pressure changes the material of earth's crust into metamorphic rockDecomposition2: [ Exposure to extreme heat and pressure can transform igneous rock into metamorphic rock ]+ [ the continental crust mainly is made of granite ]Step=3:Query3: exposure to extreme heat and pressure changes the material of earth's crust into metamorphic rockDecomposition3: [ exposure to extreme heat and pressure changes the material of earth's crust ] + [ the crust is made of different kinds of crust ]Step=4:Query4: exposure to extreme heat and pressure changes the material of earth's crust Decomposition4: [ exposure to extreme heat and pressure changes igneous rock into metamorphic rock ] +[ rock is a kind of material ]Step=5:Query5: exposure to extreme heat and pressure changes igneous rock into metamorphic rockDecomposition5: [ two tectonic plates pushing toward each other cause intense heat and pressure in earth's crust ] +[ rock is a kind of material ]Step=1:Query: John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 hours but spends the first 2 hours in standstill traffic.  He spends the next half-hour driving at a speed of 30mph, before being able to drive the remaining time of the 4 hours going at 80 mph.  How far is he from home at the end of those 4 hours?Decomposition1: How far was John during the first 3 hours of driving?John drives at a speed of 60 mph for 3 hours, so the distance covered is calculated as follows: Distance = 60 �3  = 180 milesStep=2:Decomposition2: How far did John driving in the next half hour?John drives at a speed of 30 mph for half an hour, so the distance covered is calculated as follows:Distance = 30  �0.5  = 15 milesStep=3:Decomposition3:How long is thedistance covered during the remaining time of 4 hours?John spends 2 hours in standstill traffic, so he has 4 hours -2 hours = 2 hours remaining to drive at a speed of 80 mph. Distance = 80 �2 = 160 milesStep=4:Decomposition4: How long is the total distance covered during the 4 hours?To find the total distance covered, we add up the distances calculated in the previous steps: Total Distance = 180 + 15  + 160 = 355 miles
