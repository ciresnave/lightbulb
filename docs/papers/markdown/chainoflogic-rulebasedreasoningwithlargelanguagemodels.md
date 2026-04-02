Chain of Logic: Rule-Based Reasoning with Large Language Models

Sergio Servantez�

Joe Barrow?� Kristian Hammond� Rajiv Jain�

�Northwestern University

� Adobe Research

? Pattern Data

servantez@u.northwestern.edu

joe.barrow@patterndataworks.com

Kristian.Hammond@northwestern.edu

rajijain@adobe.com

4
2
0
2

b
e
F
3
2

]
L
C
.
s
c
[

2
v
0
0
4
0
1
.
2
0
4
2
:
v
i
X
r
a

Abstract

Rule-based reasoning, a fundamental type of
legal reasoning, enables us to draw conclusions
by accurately applying a rule to a set of facts.
We explore causal language models as rule-
based reasoners, specifically with respect to
compositional rules - rules consisting of mul-
tiple elements which form a complex logical
expression. Reasoning about compositional
rules is challenging because it requires multi-
ple reasoning steps, and attending to the logi-
cal relationships between elements. We intro-
duce a new prompting method, Chain of Logic,
which elicits rule-based reasoning through de-
composition (solving elements as independent
threads of logic), and recomposition (recombin-
ing these sub-answers to resolve the underlying
logical expression). This method was inspired
by the IRAC (Issue, Rule, Application, Con-
clusion) framework, a sequential reasoning ap-
proach used by lawyers. We evaluate chain of
logic across eight rule-based reasoning tasks in-
volving three distinct compositional rules from
the LegalBench benchmark and demonstrate it
consistently outperforms other prompting meth-
ods, including chain of thought and self-ask,
using open-source and commercial language
models.

1

Introduction

The surge in reasoning capabilities of language
models (LMs) has the potential to transform the
legal industry, a sector inherently reliant on sophis-
ticated reasoning. The development of models that
can assist and validate the reasoning processes of
legal practitioners promises to usher in a new legal
era. Such advancements would not only enhance
the efficiency of legal services but also expand the
capacity of professionals to service more clients,
thereby broadening access to justice. At present,
language models are prone to hallucinations in a
legal setting (Dahl et al., 2024), and even powerful
LMs like GPT-4 struggle to perform basic legal
tasks (Blair-Stanek et al., 2023a).

Figure 1: Example showing compositional structure
of rule for Personal Jurisdiction task from LegalBench.
Color coding is used to identify rule elements and illus-
trate how these elements form a complex logical expres-
sion. Reasoning about compositional rules requires not
only correctly applying each element to a fact pattern,
but also resolving the logical expression. If the logical
expression evaluates to true, it triggers a consequence
(personal jurisdiction exists).

Legal tasks typically require sophisticated rule-
based reasoning. These rules are written in natural
language and expressed in many forms, including
statutes, judicial holdings and even contract provi-
sions. Similar to an if/then statement, a rule has
an antecedent (a condition that can be evaluated
to true or false) and a consequent (the outcome
triggered if the antecedent is satisfied). Even the
earliest recorded laws, written in Sumerian on clay
tablets, have used this conditional structure (Roth,
1995). Rule-based reasoning allows us to draw
conclusions by applying a rule to a set of facts to
determine if these preconditions are satisfied. For
example, if parking is prohibited (consequent) be-
tween 2pm and 4pm (antecedent), and we know it
is currently 3pm, then we can conclude that parking
is currently prohibited. Often rules, especially com-
plex rules, are compositional in nature meaning the
antecedent consists of multiple conditions joined
by and and or operators forming a complex logical
expression (see Figure 1). In law, these constituent

conditions are called rule elements. Recent work
demonstrates that models can struggle to reason
about even basic compositional rules (Blair-Stanek
et al., 2023b).

We explore large language models as rule-based
reasoners, evaluating several common in-context
learning methods across 8 tasks and 3 distinct rules
from LegalBench, a collaboratively constructed
legal reasoning benchmark (Guha et al., 2023).
LegalBench tasks were designed by subject mat-
ter experts to evaluate useful legal reasoning skills.
Prior work on LegalBench has focused on evaluat-
ing the capacity of models to perform rule-based
reasoning given in-context demonstrations of the
same rule being applied to varying fact patterns.
These few-shot prompts are valuable in law where
annotated data is scarce, limiting the ability to fine-
tune models for specific legal tasks. Yet requiring
several reasoning examples for each rule in a real
world setting poses a challenge in terms of both
cost and scalability. Additionally, a same-rule set-
ting allows the model to largely mimic the reason-
ing path of the in-context examples. In contrast,
a different-rule setting enables us to examine a
model�s ability to dynamically generate this rea-
soning path - the same way a lawyer would reason.
Our work shifts focus to evaluating and improving
a model�s ability to perform rule-based reasoning
given only a single in-context demonstration of a
different rule application, thus removing the need
to store examples for each rule.

In this work, we introduce Chain of Logic, a
new prompting approach to guide LMs in reason-
ing about compositional rules by decomposing rule
elements into a series of logical statements (eval-
uate to true or false), before explicitly recompos-
ing these element answers to resolve the under-
lying logical expression. Understanding natural
language questions requires understanding how to
break down a question into a set of steps yielding an
answer (Wolfson et al., 2020). The inspiration for
our method is rooted in the IRAC Framework, an
approach to legal reasoning where lawyers break
down the reasoning process into four sequential
steps: issue spotting, rule identification, rule appli-
cation and conclusion. During the rule application
step, rules are typically divided into their elements
and addressed separately before recombining these
threads to arrive at a conclusion. In devising our
chain of logic approach, our interdisciplinary team
combined this domain knowledge with a variety of

insights from prior work on in-context learning.

Our experiments show chain of logic outper-
forms other prompting methods, including chain
of thought (Wei et al., 2023) and self-ask (Press
et al., 2023), across a variety of rule-based reason-
ing tasks using both open-source and commercial
models. We demonstrate that given a single exam-
ple of chain of logic, a model can learn to general-
ize this approach to a different rule and fact pattern,
and thereby improve its reasoning capabilities.

2 Background

Large language models have demonstrated strong
capabilities as zero-shot (Kojima et al., 2023) and
few-shot (Brown et al., 2020) reasoners. Chain
of thought prompting enhanced problem solving
performance even further by eliciting models to
produce intermediate reasoning steps (Wei et al.,
2023). This approach proved effective at solving
complex tasks, including arithmetic and common-
sense reasoning tasks. Many other prompting meth-
ods followed with a similar aim of decomposing a
complex task into a sequence of simpler subtasks.
Self-ask improved multi-hop question answering
performance beyond chain of thought prompting by
guiding the model to explicitly pose and answer in-
termediate questions (Press et al., 2023). Similar to
decomposing a multi-hop question into a sequence
of intermediate questions, compositional rules can
be decomposed into rule elements. However, rea-
soning over these rules requires not only answering
each element correctly, but also understanding the
logical relationships that exists between these el-
ements. For example, if a rule with two elements
resolves to (true, false), then the final answer de-
pends on whether the boolean relationship between
those elements is and or or. In Section x, we show
existing prompting methods can resolve rule ele-
ments correctly while still getting the final answer
incorrect. This motivates the need for a more ro-
bust prompting method which attends to both the
element level answers and the underlying logical
structure. Further motivating this work, zero-shot
methods have been observed outperforming one-
shot and few-shot approaches on legal reasoning
tasks (Yu et al., 2022; Blair-Stanek et al., 2023b),
suggesting models can struggle with in-context
learning in a legal setting.

Figure 2: Comparing one-shot examples demonstrating chain of thought, self-ask and our chain of logic method
on the Personal Jurisdiction task. Through a sequence of reasoning steps, chain of logic decomposes a rule into
elements which are solved independently, before recomposing sub-answers to arrive at a final conclusion. See
Section 3 for a detailed discussion on the chain of logic approach.

3 Chain of Logic

We introduce chain of logic, a new prompting
approach to elicit rule-based reasoning in LMs
through a series of instructive reasoning steps.
Each step in this series helps inform the next, and
enables the model to unravel the many reasoning
tasks needed to arrive at the right conclusion. Our
method builds on chain of thought and self-ask by
not only considering problem decomposition, but
also attending to relationships between subtasks
that can dictate the final answer, which we later
show improves performance. Our prompt begins
with a single example demonstrating the chain of
logic approach, appended with the inference-time
problem. Figure 2 contains a one-shot example of
our approach depicting the sequence of reasoning
steps which we describe below:

� Step 1: Structured Input. Clearly label the
task inputs to identify rule, facts and issue,
similar to the IRAC Framework.

� Step 2: Rule Decomposition. Decompose
the rule into elements by identifying relevant
text spans and assigning elements to variables.
Similar to the least-to-most approach (Zhou
et al., 2023), our method first identifies the
subtasks before trying to solve them.

� Step 3: Logical Expression. Construct a com-
plex logical expression representing the logi-
cal relationships that exist between these ele-
ment variables.

� Step 4: Question Answering. Iterate through
each rule element, rephrasing the element as
a question before providing a rationale and
answer. Like self-ask, we dynamically con-
struct this series of sub-questions, but for our
approach this construction is informed by the
elements identified in step 2. The model need
only rephrase each element as question.

� Step 5: Element Recomposition. Replace el-
ement variables in the logical expression with
the corresponding sub-answers from the pre-
vious step.

� Step 6: Resolve Expression. Resolve the log-
ical expression populated with sub-answers to
arrive at the final solution.

During inference, the model performs each step
in this process automatically given only a one-shot
example and the inference task inputs (formatted
as described in step 1). Since the one-shot task
inputs (rule, facts, issue) are different from the task
inputs being tested, the model is reasoning on its
own at each step, including deciding how many
rule elements exist, the text span of each element
and the logical relationships between them. This
more closely resembles a real world setting where
lawyers apply the IRAC framework to a rule and
fact pattern being observed for the first time.

The intermediate steps in our approach provide
a detailed accounting of each factor contributing to
the final conclusion (see Figure 3 for an example
of model output). This allows for transparent and
explainable decision-making, which is particularly

Figure 3: Comparing GPT-3.5 output for chain of thought, self-ask and our chain of logic method on the same
Personal Jurisdiction task. Chain of logic prompting elicits large language models to reason about complex rules
while also constructing an interpretable reasoning path. The prompts here are abridged, omitting a one-shot example
for each method from the Diversity Jurisdiction task (see Section 4.1).

vital in the legal domain where the justification
for an answer can be as important as the answer
itself. Chain of logic also enables us to debug
inaccurate conclusions by retracing the reasoning
path to locate errors in rule application and logic.

4 Experiments

We conduct a series of experiments to compare
our proposed chain of logic approach with existing
prompting methods, and to evaluate the rule-based
reasoning capabilities of LMs in a different-rule
setting. While we sometimes observe an inverse
relationship between prompt complexity and per-
formance (simpler prompts perform better), we find
that chain of logic consistently outperforms other
prompting methods across a range of rule-based
reasoning tasks.

4.1 Tasks

LegalBench (Guha et al., 2023) is a legal reasoning
benchmark designed and constructed by an inter-
disciplinary team of computer scientists and legal
professionals. The tasks in this benchmark have
been designed by subject matter experts to measure
legal reasoning capabilities that are both interest-
ing and useful. We consider 8 rule-based reasoning
tasks involving 3 distinct compositional rules from
this benchmark:

� Personal Jurisdiction (1 task): This task in-
volves determining whether a court has the

authority to preside over a dispute based on
where the defendant is domiciled, and whether
the defendant had sufficient contacts with the
forum state and the claim arose out of the
nexus of those contacts. There are 50 test
samples for this task.

� Diversity Jurisdiction (6 tasks): This task
involves determining whether a federal court
can preside over a lawsuit pertaining to state
law based on whether complete diversity ex-
ists (no plaintiff and defendant are citizens
of the same state) and the amount in contro-
versy exceeding $75,000. There are 6 diver-
sity jurisdiction tasks with increasingly com-
plex fact patterns, from Diversity Jurisdiction
1 (easiest) to Diversity Jurisdiction 6 (hardest).
There are 300 test samples for each diversity
jurisdiction task.

� J.Crew Blocker (1 task): This task involves
determining whether a provision from a loan
agreement contains a J.Crew Blocker restric-
tive covenant based on whether the provision
prohibits transferring IP to an unrestricted sub-
sidiary or requires lender consent for IP trans-
fers to a subsidiary. There are 54 test samples
for this task.

Each sample from all tasks contains a rule, fact
pattern and question. To correctly answer the ques-
tion, the rule must be applied to the fact pattern.

GPT-3.5

GPT-4

Llama-2

Mistral Average

Zero-Shot
Zero-Shot-LR (Yu et al., 2022)
Zero-Shot-LS (Jiang and Yang, 2023)

Standard Prompting (Brown et al., 2020)
Chain of Thought (Wei et al., 2023)
Self-Ask (Press et al., 2023)
Chain of Logic (ours)

76.3
57.2
75.5

72.6
70.1
68.2
87.0

90.4
90.6
90.1

88.6
89.1
86.3
92.3

74.2
65.3
58.4

65.1
71.8
67.5
74.6

60.8
62.8
52.2

45.5
46.4
46.5
63.1

75.4
69.0
69.1

68.0
69.4
67.1
79.3

Table 1: Aggregated performance (accuracy) across all rules, including Personal Jurisdiction, Diversity Jurisdiction
and J. Crew Blocker, using both open-source and commercial language models. Chain of logic consistently
outperforms other prompting methods. All methods are one-shot, except zero-shot approaches.

The prompt for each task contains a one-shot exam-
ple from another rule, followed by the test sample.
The benchmark expects an answer of yes or no for
each question. For our method we reformat the
answer to true or false to more closely resemble
formal logic. If the model�s response is ambigu-
ous, we use a second prompt "Therefore the answer
(true or false) is", following Kojima et al. (Kojima
et al., 2023).

4.2 Baseline Methods

In addition to our chain of logic approach (see Sec-
tion 3), we examine six prompting methods. We
first explore zero-shot prompting where the prompt
includes only the test sample (rules, facts, question)
with no in-context demonstrations. Blair-Stanek
et al. showed that zero-shot prompting can outper-
form few-shot methods for some legal reasoning
tasks, even with chain of thought reasoning (Blair-
Stanek et al., 2023b). This suggests some models
could have difficulty learning through in-context
demonstrations in a legal setting. We include zero-
shot prompting in our experiments to further ex-
plore this hypothesis. For completeness, we also
include two zero-shot methods designed for legal
reasoning tasks from prior work. Legal syllogism
(LS) prompting is a zero-shot approach to legal
judgement prediction where the prompt first de-
fines legal syllogism before instructing the model
to perform syllogistic reasoning (Jiang and Yang,
2023). Legal reasoning (LR) prompting (Yu et al.,
2022) is also a zero-shot method where the prompt
includes a simple approach description: "Approach:
Issue, rule, application, conclusion".

For the remaining methods, we use a one-shot
approach where a single demonstration with cor-
rect answer is included in the prompt. First, we use

standard prompting (Brown et al., 2020) where the
in-context demonstration includes only the sample
and answer. Second, we explore chain of thought
(Wei et al., 2023) prompting which includes a ra-
tional written by a legal professional explaining
the relevant reasoning before the demonstration an-
swer. Last, we include the self-ask approach (Press
et al., 2023) where the one-shot example demon-
strates explicitly asking and answering an interme-
diate question for each rule element before the final
answer. See Figure 2 for an illustration of these
one-shot examples. While self-ask was originally
evaluated on another compositional reasoning task,
multi-hop question answering, its demonstrated
ability to increase reasoning capacity by decom-
posing complex problems makes it a compelling
method to explore in this work.

4.3 Language Models

We experiment with two commercial models, GPT-
3.5-Turbo and GPT-4 (OpenAI, 2023), and two
leading open-source models, Llama-2-70b-chat
(Touvron et al., 2023) and Mistral-7B-OpenOrca,
which is a Mistral-7b (Jiang et al., 2023) model
fine-tuned with Orca (Mukherjee et al., 2023). We
select open-source models of disparate size to in-
vestigate the relationship between model size and
rule-based reasoning abilities. For all language
models we set the temperature to 0.0 to make the
output more focused and reproducible, and other-
wise use default settings.

5 Results and Discussion

Table 1 presents results for baseline methods and
our chain of logic approach macro-averaged across
all rule sets (Personal Jurisdiction, Diversity Ju-
risdiction and J.Crew Blocker), using commercial

Figure 4: Accuracy (%) across all 6 Diversity Jurisdiction tasks using GPT-3.5. The fact patterns in these tasks are
increasingly complex, from DJ1 (easiest) to DJ6 (hardest). Chain of logic particularly outperforms other prompting
methods for tasks requiring arithmetic operations (DJ3, DJ5, DJ6). See Section 5.1 for a detailed discussion.

and open-source models. We report the macro-
average here to provide a balanced view across rule
sets, though as shown later in Figure 5.1 and the
appendix, Chain of logic performs best on the Di-
versity Jurisdiction task, which is downweighted
with this metric. See the Appendix for more de-
tailed model performance by rule. For each model,
a zero-shot method is the best performing base-
line. This is consistent with prior work finding that
zero-shot methods can outperform one-shot and
few-shot prompting on legal reasoning tasks (Yu
et al., 2022; Blair-Stanek et al., 2023b). Chain of
logic significantly improves rule-based reasoning
through in-context learning.

For each model, chain of logic outperforms
all baseline methods, including zero-shot meth-
ods. The performance gap between chain of logic
and the best performing baseline is 3.9% on aver-
age. This gap is smaller for open source models
(0.3% and 0.4%) and larger for commercial mod-
els, with GPT-3.5 having the largest performance
improvement of 10.7%. For this work we observe a
large performance difference between the commer-
cial and open source models, and hypothesize the
performance gains may be larger for the stronger
commercial models which can more easily follow
the longer prompts. This can also be seen in the
performance differences between the baseline zero
and one-shot methods. For GPT-4 we believe the
absolute performance gain may be lower due to
ceiling effects of the model. The specialized zero-
shot legal baselines demonstrate comparable per-
formance to zero-shot for some models, but are
also inconsistent. LR prompting slightly outper-
forms zero-shot using Mistral-7b and GPT-4, but

also underperforms by 19.1% for GPT-3.5.

The one-shot baselines demonstrate the chal-
lenges to in-context learning in this setting by con-
sistently underperforming zero-shot methods. The
model output depicted in Figures 3 and 5 illustrate
common errors we observe in rule-based reasoning
using chain of thought and self-ask. First, chain
of thought often generates fluent but illogical rea-
soning paths. The output in Figure 5 shows the
model incorrectly conclude that personal jurisdic-
tion does not exist because Dustin plans to move to
New York, even though future residency is not an
element of the rule. Second, the question/answer
pairs generated by the self-ask approach commonly
decompose the rule incorrectly (Figure 3) or incom-
pletely (Figure 5). The output in the latter figure
shows the model only asks intermediate questions
for two out of three rule elements, omitting whether
the defendant had sufficient contact with the forum
state. Chain of logic prevents these omissions by
first restating the rule elements (step 2 in the ap-
proach), similar to how System 2 Attention (We-
ston and Sukhbaatar, 2023) first restates only the
relevant portions of a context before attending to
a task. In the ablation study, we show that remov-
ing this step significantly decreases performance.
Last, Figure 5 also demonstrates how not recogniz-
ing logical relationships can lead to an incorrect
solution, even when sub-questions are answered
correctly. The model correctly determines the de-
fendant is domiciled in the forum state, which is
sufficient to conclude personal jurisdiction exists,
but still reaches the wrong final answer.

Reasoning Step

Step 1: Structured Input
Step 2: Rule Decomposition
Step 3: Logical Expression
Step 4: Question Answering
Step 5: Element Recomposition
Step 6: Resolve Expression

PJ

-25.9
-12.9
-14.8
-49.3
-5.5
-3.7

Table 2: Ablations. Reduction in performance (abso-
lute percentage) observed when removing the specified
reasoning step from the chain of logic approach using
GPT-3.5 on the Personal Jurisdiction task.

5.2 Ablation Study

Our experiments show chain of logic improves in-
context learning for rule-based reasoning tasks. In
this section, we assess the contribution of each rea-
soning step to the overall performance as shown
in Table 2. We isolate these effects by removing
a single reasoning step at a time. We evaluate on
the most complex rule, personal jurisdiction, using
the model with the best comparable performance,
GPT-3.5. Unsurprisingly, removing step 4 drasti-
cally reduces overall performance since there is no
clear path between the logical expressions in steps
3 and 5. Notably, the removal of step 1, switching
from structured to unstructured task inputs, leads
to a 25.9 absolute percentage point performance
decrease. We believe this structured content is use-
ful for the rule decomposition step that follows.
Similarly, the ablations show that decomposing the
rule into elements (step 2: -12.9%) and generating
a logical expression representing the relationships
between elements (step 3: -14.8%) are both criti-
cal to the performance gains we observe for this
approach.

6 Related Work

There has been extensive work on improving the
problem solving capabilities of LMs by enabling a
model to use more computation for more complex
tasks. Graves introduces an adaptive algorithm for
dynamically selecting the number of computation
steps a recurrent neural network should take based
on the task complexity (Graves, 2017). Ling et
al. first demonstrated the use of answer rationales,
natural language explanations generated before the
solution, to solve algebraic word problems (Ling
et al., 2017). Several approaches followed this
work in exploring the use of intermediate reasoning

Figure 5: GPT-4 output demonstrating common errors
in one-shot methods.

5.1 Diversity Jurisdiction Series

The Diversity Jurisdiction series gives us a unique
opportunity to examine how performance is af-
fected by task complexity. Recall there are 6 tasks
involving the diversity jurisdiction rule. The fact
patterns in these tasks increase in complexity across
two dimensions: 1) the number of parties (plain-
tiffs and defendants), and the number of claims
per plaintiff-defendant pair. The simplest task, Di-
versity Jurisdiction 1 (DJ1), contains one plaintiff,
one defendant and one claim. The most complex
task, Diversity Jurisdiction 6 (DJ6), contains two
plaintiffs, two defendants, and 4 total claims.

Figure 4 shows results across all 6 Diversity Ju-
risdiction tasks using GPT-3.5. All methods, ex-
cept zero-shot-LR, perform above 95% accuracy on
DJ1, yet even the best performing method, chain of
logic, only scores a 61.7% accuracy on DJ6. Chain
of logic outperforms the baseline methods for 5 out
of 6 tasks, with standard prompting outperforming
our method by 2.3% for DJ2. Most notably, chain
of logic significantly outperforms other methods
on tasks requiring arithmetic reasoning to add dol-
lar amounts from multiple claims (DJ3, DJ5, DJ6).
The performance gap between chain of logic and
zero-shot, the best performing baseline, is 36% for
DJ5. All baselines perform near random chance for
this task. For tasks involving arithmetic reasoning,
the baselines also demonstrate an inverse relation-
ship between prompt complexity and performance.
That is to say, simpler prompts tend to perform
better. Chain of logic is the clear exception to this
trend being both the most complex prompt and the
best performing model.

steps (Nye et al., 2021; Wei et al., 2023; Wang et al.,
2023). More similar to our work, other methods
have explored decomposing compositional ques-
tions into simpler sub-questions using supervised
models (Qi et al., 2019; Min et al., 2019; Khot
et al., 2021) and prompting techniques (Mishra
et al., 2022; Press et al., 2023). Talmor and Berant
also recognized the importance of explicitly iden-
tifying relationships between sub-questions (Tal-
mor and Berant, 2018). The authors train a su-
pervised model to translate questions into a com-
putation tree and answer sub-questions by query-
ing a search engine. We extend these works with
our decomposition-recomposition approach, ex-
ploring simple questions involving compositional
rules where the solution depends not only on rule
element answers, but also the logical relationships
between elements.

Prior work in natural language processing has
explored a broad range of legal tasks, including
clause classification (Hendrycks et al., 2021) and
recommendation (Aggarwal et al., 2021), contract
summarization (Manor and Li, 2019), legal judg-
ment prediction (Chalkidis et al., 2019) and even
answering bar exam questions (au2 and Katz, 2022;
Zhong et al., 2019). Rule-based reasoning can in-
volve rules from many sources. Saeidi et al. ex-
plored conversational question answering where
the solution required rule-based reasoning about a
collection of regulations (Saeidi et al., 2018). Ser-
vantez et al. captured rules in contract text through
graph-based extraction and converted them into
code (Servantez et al., 2023). Similar to our work,
Holzenberger and Van Durme introduced an ap-
proach to reasoning about tax statutes by decom-
posing the reasoning process (Holzenberger and
Van Durme, 2021). This approach first extracts key
arguments (entities, dates, dollar amounts) from
the statute text and fact pattern using fine-tuned
BERT models, before arriving at a final answer.
We do not explore this method since it requires a
fine-tuned model and does not easily translate to
the tasks in this work.

7 Limitations and Future Work

While LegalBench tasks were crafted by subject
matter experts, the rules have been simplified to en-
sure answers are objectively correct. This greatly
improves evaluation, but also means the scope of
complexity is narrower than many real world rules.
Additionally, our current approach only addresses

rules where the solution is based on whether the
antecedent has been triggered. Real world rules can
contain complex consequences which themselves
require some form of reasoning. For example, cal-
culating tax liability after determining the applica-
ble tax bracket. In future work, we hope to build
on our current approach in several directions. First,
for simplicity chain of logic performs all reasoning
step in a single forward pass. However, a multi-
ple pass approach would allow us to incorporate
other reasoning tasks like rule identification which
is not addressed here. This moves us toward more
real world scenarios where the applicable rule is
not known a priori. Second, investigating whether
rule-based reasoning could be improved further
by dynamically sampling multiple reasoning paths.
And third, incorporating retrieval augmented gen-
eration (Lewis et al., 2021) by allowing the model
to access external sources of knowledge like term
definitions. This is particularly useful in the legal
domain where terms or concepts have a distinct
meaning based on the jurisdiction or contract.

8 Conclusion

In this work, we explore causal language models as
rule-based reasoners, and show these models can
have difficulty learning from in-context demonstra-
tions using common prompting approaches. We
present a new prompting method, chain of logic,
to elicit rule-based reasoning in language models
through decomposition and recomposition. Our ex-
periments show chain of logic consistently outper-
forms other prompting methods, including chain of
thought and self-ask, across a variety of rule-based
reasoning tasks using both open-source and com-
mercial language models. We show how chain of
logic creates a coherent and interpretable reasoning
path by unraveling the many reasoning steps re-
quired by compositional rules. These findings may
also be a useful signal for future instruction tuning
of language models to imbue them with reasoning.
By enhancing the rule-based reasoning capabili-
ties of LMs through in-context learning, chain of
logic also reduces the need for annotated legal data,
which has historically been a bottleneck for the
legal domain.

Acknowledgements

This work was supported in part by Adobe Re-
search. The authors thank Neel Guha for helpful
discussions related to LegalBench.

References

Vinay Aggarwal, Aparna Garimella, Balaji Vasan
Srinivasan, Anandhavelu N, and Rajiv Jain. 2021.
ClauseRec: A clause recommendation framework for
AI-aided contract authoring. In Proceedings of the
2021 Conference on Empirical Methods in Natural
Language Processing, pages 8770�8776, Online and
Punta Cana, Dominican Republic. Association for
Computational Linguistics.

Michael Bommarito II au2 and Daniel Martin Katz.

2022. Gpt takes the bar exam.

Andrew Blair-Stanek, Nils Holzenberger, and Ben-
jamin Van Durme. 2023a. Blt: Can large language
models handle basic legal text?

Andrew Blair-Stanek, Nils Holzenberger, and Benjamin
Van Durme. 2023b. Can gpt-3 perform statutory
reasoning? In Proceedings of the Nineteenth Inter-
national Conference on Artificial Intelligence and
Law, ICAIL �23, page 22�31, New York, NY, USA.
Association for Computing Machinery.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, Christopher Hesse, Mark Chen, Eric
Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess,
Jack Clark, Christopher Berner, Sam McCandlish,
Alec Radford, Ilya Sutskever, and Dario Amodei.
2020. Language models are few-shot learners.

Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Ale-
tras. 2019. Neural legal judgment prediction in en-
glish.

Matthew Dahl, Varun Magesh, Mirac Suzgun, and
Daniel E. Ho. 2024. Large legal fictions: Profiling
legal hallucinations in large language models.

Alex Graves. 2017. Adaptive computation time for

recurrent neural networks.

Neel Guha, Julian Nyarko, Daniel E. Ho, Christo-
pher R�, Adam Chilton, Aditya Narayana, Alex
Chohlas-Wood, Austin Peters, Brandon Waldon,
Daniel N. Rockmore, Diego Zambrano, Dmitry Tal-
isman, Enam Hoque, Faiz Surani, Frank Fagan, Galit
Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason
Hegland, Jessica Wu, Joe Nudell, Joel Niklaus, John
Nay, Jonathan H. Choi, Kevin Tobia, Margaret Hagan,
Megan Ma, Michael Livermore, Nikon Rasumov-
Rahe, Nils Holzenberger, Noam Kolt, Peter Hender-
son, Sean Rehaag, Sharad Goel, Shang Gao, Spencer
Williams, Sunny Gandhi, Tom Zur, Varun Iyer, and
Zehua Li. 2023. Legalbench: A collaboratively built
benchmark for measuring legal reasoning in large
language models.

Nils Holzenberger and Benjamin Van Durme. 2021.
Factoring statutory reasoning as language understand-
ing challenges. In Proceedings of the 59th Annual
Meeting of the Association for Computational Lin-
guistics and the 11th International Joint Conference
on Natural Language Processing (Volume 1: Long
Papers), pages 2742�2758, Online. Association for
Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, L�lio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timoth�e Lacroix,
and William El Sayed. 2023. Mistral 7b.

Cong Jiang and Xiaolei Yang. 2023. Legal syllogism
prompting: Teaching large language models for legal
judgment prediction.

Tushar Khot, Daniel Khashabi, Kyle Richardson, Peter
Clark, and Ashish Sabharwal. 2021. Text modular
networks: Learning to decompose tasks in the lan-
guage of existing models.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yu-
taka Matsuo, and Yusuke Iwasawa. 2023. Large lan-
guage models are zero-shot reasoners.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich K�ttler, Mike Lewis, Wen tau Yih, Tim Rock-
t�schel, Sebastian Riedel, and Douwe Kiela. 2021.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blun-
som. 2017. Program induction by rationale genera-
tion : Learning to solve and explain algebraic word
problems.

Laura Manor and Junyi Jessy Li. 2019. Plain English
summarization of contracts. In Proceedings of the
Natural Legal Language Processing Workshop 2019,
pages 1�11, Minneapolis, Minnesota. Association for
Computational Linguistics.

Sewon Min, Victor Zhong, Luke Zettlemoyer, and Han-
naneh Hajishirzi. 2019. Multi-hop reading compre-
hension through question decomposition and rescor-
ing.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin
Choi, and Hannaneh Hajishirzi. 2022. Reframing
instructional prompts to gptk�s language.

Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawa-
har, Sahaj Agarwal, Hamid Palangi, and Ahmed
Awadallah. 2023. Orca: Progressive learning from
complex explanation traces of gpt-4. arXiv preprint
arXiv:2306.02707.

Dan Hendrycks, Collin Burns, Anya Chen, and Spencer
Ball. 2021. Cuad: An expert-annotated nlp dataset
for legal contract review.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari,
Henryk Michalewski, Jacob Austin, David Bieber,
David Dohan, Aitor Lewkowycz, Maarten Bosma,

David Luan, Charles Sutton, and Augustus Odena.
2021. Show your work: Scratchpads for intermediate
computation with language models.

OpenAI. 2023. Gpt-4 technical report. View in Article,

2:13.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models.

Peng Qi, Xiaowen Lin, Leo Mehr, Zijian Wang, and
Christopher D. Manning. 2019. Answering complex
open-domain questions through iterative query gen-
eration.

Martha T Roth. 1995.

Law collections from
Mesopotamia and Asia minor, volume 6. Scholars
Press.

Marzieh Saeidi, Max Bartolo, Patrick Lewis, Sameer
Singh, Tim Rockt�schel, Mike Sheldon, Guillaume
Bouchard, and Sebastian Riedel. 2018. Interpretation
of natural language rules in conversational machine
reading.

Sergio Servantez, Nedim Lipka, Alexa Siu, Milan Ag-
garwal, Balaji Krishnamurthy, Aparna Garimella,
Kristian Hammond, and Rajiv Jain. 2023. Com-
putable contracts by extracting obligation logic
graphs. In Proceedings of the Nineteenth Interna-
tional Conference on Artificial Intelligence and Law,
ICAIL �23, page 267�276, New York, NY, USA. As-
sociation for Computing Machinery.

Alon Talmor and Jonathan Berant. 2018. The web as a
knowledge-base for answering complex questions.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton
Ferrer, Moya Chen, Guillem Cucurull, David Esiobu,
Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller,
Cynthia Gao, Vedanuj Goswami, Naman Goyal, An-
thony Hartshorn, Saghar Hosseini, Rui Hou, Hakan
Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa,
Isabel Kloumann, Artem Korenev, Punit Singh Koura,
Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Di-
ana Liskovich, Yinghai Lu, Yuning Mao, Xavier Mar-
tinet, Todor Mihaylov, Pushkar Mishra, Igor Moly-
bog, Yixin Nie, Andrew Poulton, Jeremy Reizen-
stein, Rashi Rungta, Kalyan Saladi, Alan Schelten,
Ruan Silva, Eric Michael Smith, Ranjan Subrama-
nian, Xiaoqing Ellen Tan, Binh Tang, Ross Tay-
lor, Adina Williams, Jian Xiang Kuan, Puxin Xu,
Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan,
Melanie Kambadur, Sharan Narang, Aurelien Ro-
driguez, Robert Stojnic, Sergey Edunov, and Thomas
Scialom. 2023. Llama 2: Open foundation and fine-
tuned chat models.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le,
Ed Chi, Sharan Narang, Aakanksha Chowdhery, and

Denny Zhou. 2023. Self-consistency improves chain
of thought reasoning in language models.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and
Denny Zhou. 2023. Chain-of-thought prompting elic-
its reasoning in large language models.

Jason Weston and Sainbayar Sukhbaatar. 2023. System

2 attention (is something you might need too).

Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner,
Yoav Goldberg, Daniel Deutch, and Jonathan Berant.
2020. Break it down: A question understanding
benchmark.

Fangyi Yu, Lee Quartey, and Frank Schilder. 2022. Le-
gal prompting: Teaching a language model to think
like a lawyer.

Haoxi Zhong, Chaojun Xiao, Cunchao Tu, Tianyang
Zhang, Zhiyuan Liu, and Maosong Sun. 2019. Jec-
qa: A legal-domain question answering dataset.

Denny Zhou, Nathanael Sch�rli, Le Hou, Jason Wei,
Nathan Scales, Xuezhi Wang, Dale Schuurmans,
Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi.
2023. Least-to-most prompting enables complex rea-
soning in large language models.

A Appendix

A.1 LegalBench Task Examples

Personal Jurisdiction (Figure 3)

� Rule: There is personal jurisdiction over a
defendant in the state where the defendant
is domiciled, or when (1) the defendant has
sufficient contacts with the state, such that
they have availed itself of the privileges of the
state and (2) the claim arises out of the nexus
of the defendant�s contacts with the state.

� Fact Pattern: Dustin is a repairman who lives
in Arizona and repairs computers in Califor-
nia, Oregon, and Washington. While trav-
elling to repair a computer in Washington,
Dustin is involved in a car crash in Oregon
with Laura, a citizen of Texas. After the ac-
cident, Dustin returns to Arizona. Laura sues
him in Oregon.

� Issue: Is there personal jurisdiction?

Diversity Jurisdiction 1

� Rule: Diversity jurisdiction exists when
there is (1) complete diversity between plain-
tiffs and defendants, and (2) the amount-in-
controversy (AiC) is greater than $75k.

� Fact Pattern: James is from Arizona. Lucas
is from Arizona. James sues Lucas for negli-
gence for $64,000.

� Issue: Is there diversity jurisdiction?

Diversity Jurisdiction 6

� Rule: Diversity jurisdiction exists when
there is (1) complete diversity between plain-
tiffs and defendants, and (2) the amount-in-
controversy (AiC) is greater than $75k.

� Fact Pattern: Theodore is from North Dakota.
Amelia is from Georgia. Benjamin is from
Delaware. Mia is from Illinois. Theodore and
Amelia both sue Benjamin for trademark in-
fringement for $42,000 and copyright infringe-
ment for $71,000. Theodore and Amelia both
sue Mia for securities fraud for $45,000 and
medical malpractice for $57,000.

� Issue: Is there diversity jurisdiction?

J.Crew Blocker

� Rule: The JCrew Blocker is a provision that
typically includes (1) a prohibition on the bor-
rower from transferring IP to an unrestricted
subsidiary, and (2) a requirement that the bor-
rower obtains the consent of its agent/lenders
before transferring IP to any subsidiary.

� Fact Pattern: Notwithstanding anything to
foregoing, no Intellectual Property that is ma-
terial to the Borrower and its Restricted Sub-
sidiaries, taken as a whole (as reasonably de-
termined by the Borrower), shall be owned
by or licensed, contributed or otherwise trans-
ferred to any Unrestricted Subsidiary.

� Issue: Do the following provisions contain

JCrew Blockers?

A.2 Model Performance by Rule

Model performance (accuracy) by rule for all rea-
soning tasks. Diversity Jurisdiction results are ag-
gregated across all six datasets.

� Issue: Is there diversity jurisdiction?

Diversity Jurisdiction 2

� Rule: Diversity jurisdiction exists when
there is (1) complete diversity between plain-
tiffs and defendants, and (2) the amount-in-
controversy (AiC) is greater than $75k.

� Fact Pattern: Sophia is from Arkansas. Ben-
jamin is from Hawaii. Noah is from Arkansas.
Sophia sues Benjamin and Noah each for
defamation for $24,000.

� Issue: Is there diversity jurisdiction?

Diversity Jurisdiction 3

� Rule: Diversity jurisdiction exists when
there is (1) complete diversity between plain-
tiffs and defendants, and (2) the amount-in-
controversy (AiC) is greater than $75k.

� Fact Pattern: William is from Montana.
Theodore is from Connecticut. William sues
Theodore for medical malpractice for $9,000
and negligence for $35,000.

� Issue: Is there diversity jurisdiction?

Diversity Jurisdiction 4

� Rule: Diversity jurisdiction exists when
there is (1) complete diversity between plain-
tiffs and defendants, and (2) the amount-in-
controversy (AiC) is greater than $75k.

� Fact Pattern: Emma is from New Hampshire.
Mia is from Wisconsin. Evelyn is from Cal-
ifornia. Emma and Mia both sue Evelyn for
copyright infringement for $5,400,000.

� Issue: Is there diversity jurisdiction?

Diversity Jurisdiction 5

� Rule: Diversity jurisdiction exists when
there is (1) complete diversity between plain-
tiffs and defendants, and (2) the amount-in-
controversy (AiC) is greater than $75k.

� Fact Pattern: Elijah is from Hawaii. Ava is
from Oklahoma. Amelia is from Minnesota.
Elijah and Ava both sue Amelia for defama-
tion for $3,000 and copyright infringement for
$80,000.

Zero-Shot
Zero-Shot-LR
Zero-Shot-LS

Standard Prompting
Chain of Thought
Self-Ask
Chain of Logic (ours)

Zero-Shot
Zero-Shot-LR
Zero-Shot-LS

Standard Prompting
Chain of Thought
Self-Ask
Chain of Logic (ours)

Zero-Shot
Zero-Shot-LR
Zero-Shot-LS

Standard Prompting
Chain of Thought
Self-Ask
Chain of Logic (ours)

Zero-Shot
Zero-Shot-LR
Zero-Shot-LS

Standard Prompting
Chain of Thought
Self-Ask
Chain of Logic (ours)

Personal Jurisdiction Diversity Jurisdiction

J.Crew Blocker

68.0
60.0
70.0

72.0
74.0
60.0
78.0

73.9
69.2
69.5

73.5
69.7
70.7
88.6

87.0
42.6
87.0

72.2
66.7
74.0
94.4

Table 3: GPT-3.5 Performance

Personal Jurisdiction Diversity Jurisdiction

J.Crew Blocker

84.0
86.0
78.0

76.0
82.0
84.0
90.0

87.1
87.8
94.2

93.6
89.2
78.5
94.3

100
98.1
98.1

96.3
96.3
96.3
92.6

Table 4: GPT-4 Performance

Personal Jurisdiction Diversity Jurisdiction

J.Crew Blocker

66.0
52.0
42.0

56.0
62.0
70.0
72.0

65.9
66.0
50.0

57.7
60.7
52.8
66.6

90.7
77.8
83.3

81.5
92.6
79.6
85.2

Table 5: Llama-2-70b Performance

Personal Jurisdiction Diversity Jurisdiction

J.Crew Blocker

56.0
50.0
48.0

44.0
36.0
52.0
54.0

65.4
66.1
62.4

46.3
66.2
70.8
72.4

61.1
72.2
46.3

46.3
37.0
16.7
63.0

Table 6: MistralOrca-7b Performance


