4
2
0
2

l
u
J

2

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
7
6
2
0
.
7
0
4
2
:
v
i
X
r
a

Reasoning in Large Language Models: A Geometric
Perspective

Romain Cosentino?, Sarath Shekkizhar?
Tenyx
{romain,sarath}@tenyx.com

Abstract

The advancement of large language models (LLMs) for real-world applications
hinges critically on enhancing their reasoning capabilities. In this work, we explore
the reasoning abilities of large language models (LLMs) through their geometrical
understanding. We establish a connection between the expressive power of LLMs
and the density of their self-attention graphs. Our analysis demonstrates that the
density of these graphs defines the intrinsic dimension of the inputs to the MLP
blocks. We demonstrate through theoretical analysis and toy examples that a
higher intrinsic dimension implies a greater expressive capacity of the LLM. We
further provide empirical evidence linking this geometric framework to recent
advancements in methods aimed at enhancing the reasoning capabilities of LLMs.

1

Introduction

Large language models (LLMs), such as GPT-4 [1], Llama 3 [2], have achieved impressive per-
formance on a wide range of tasks. The search for better LLMs hinges critically on the reasoning
performance of these models. However, it is unclear what aspects of the language models are essential
for achieving this goal. Today the predominant approach, considered by the community, to advance
reasoning involves (i) increased model size (where larger models have resulted in better reasoning
capabilities) [3�5] and (ii) increased context length [6], more tokens or text as input to the LLM,
through chain of thought [7], retrieval augmented generation [8], or prompting with examples [9].

While these approaches have been sufficient, they represent only part of the potential avenues for
improvement. Moreover, longer inputs and bigger models correspond to increased computational
cost and inference latency for real-world use cases. In this work, we take a principled approach to
understand and elucidate the properties of LLMs that allow for improved and better reasoning. Our
study leverages the geometry of the transformer layer [10], a key component in LLMs, with empirical
evidence on simulated as well as Llama 3 family of models [2] to justify our claims.

In particular, we characterize key properties of the transformer layer that are correlated with its
capacity or expressive power. We show that the (i) density of interaction between the tokens in the
self-attention or multi-head attention (MHA) module of a transformer exemplifies the complexity of
function representation achievable by the multi-layer perceptron (MLP) layer that follows it, and (ii)
increased model size and context length facilitates higher attention density and consequently better
reasoning. Our analysis presents a path toward improving reasoning and advancing LLMs while
deepening our understanding of the models and their behaviors. We note that our accompanying work
[11], presented an analysis as in this work where we showed the brittleness of toxicity guardrails
obtained via RLHF through the lens of LLM geometry.

In this work, we are specifically interested in understanding how the geometry of the LLM is
correlated with its reasoning capabilities. Besides, we are investigating how increasing the input

?Equal Contribution

sequence length as well as the number of attention heads affects the geometry of the LLM. In fact,
it has been empirically demonstrated that these are critical factors increasing LLMs� reasoning
capabilities.

We will start in section 2 with a brief detour and highlight some important Deep Neural Networks
(DNNs) geometric notions (subsection 2.1): (i) how they are partitioning their input space and (ii)
how such a partitioning is related to their approximation capabilities. In this section, we will also
show that increasing the intrinsic dimension of the DNN input affects its partitioning.

After this necessary detour, we will extrapolate these notions to LLMs (subsection 2.2). We will
highlight how one can capture their expressive power by inspecting the intrinsic dimension of the
self-attention block. Specifically by analyzing the graph density of each attention head. We show
how this intrinsic dimension is related to context length and the number of heads.

In section 3 we introduce a series of experiments designed to analyze the correlation between these
geometrical properties and the LLM reasoning capabilities. Our findings reveal that as the number of
examples provided in the prompt increases, the intrinsic dimension of the LLM also rises. Notably,
while the increase in the intrinsic dimension at the first layer is not indicative of the accuracy of the
model�s responses, a significant rise in the intrinsic dimension at the final layer strongly correlates
with enhanced reasoning performance. This suggests that the geometry of the LLM�s internal
representations plays a crucial role in its ability to reason effectively.

2

Input Space Partitioning and Expressive Power

In this section, we delve into the geometrical intuitions that underpin a fundamental aspect of Deep
Neural Networks (DNNs): the adaptive partitioning of the DNN input space. This process leads to the
formation of regions within the input space, each associated with an affine map that characterizes how
the network processes the inputs in that region. We then leverage this perspective in conjunction with
the multi-head attention (MHA) layer in the transformer module to develop a novel geometric view
of LLMs. This perspective allows us to hypothesize about the role of model size and context length
in modern LLMs and presents a path toward alternate ideas that can lead to improved reasoning
capabilities.

2.1 Deep Neural Networks

We describe the continuous piecewise affine formulation of DNNs to elucidate the concept of
their induced local linear mappings. In particular, we focus on the simple case of the multilayer
perceptron (MLP) consisting of one hidden layer, typically employed in a transformer, from a spline
geometric viewpoint. Subsequently, we provide an intuitive depiction through simulated experiments
of their approximation capabilities, emphasizing the significance of the adaptive partitioning property,
and the role of input space dimension.

Continuous Piece-wise Affine Formulation of DNNs: The geometric characterization of MLPs
employing nonlinearities, such as (leaky-)ReLU, absolute value, and max-pooling, have been exten-
sively studied from the lens of a continuous piecewise linear operator, resulting in a partition ? of the
input space [12�14]. As such, a DNN defined as f? with parameters ? can be re-written as

f?(x) =

(cid:88)

???

1{x??} (A?x + B?) ,

(1)

where 1 defines the indicator function, A? and B? the per region affine parameters associated with
the DNN layer, and x the input to the network. The indicator function is data dependent and subsumes
the affine parameters and the nonlinearity of the region ? ? ?. A depiction of the regions and the
partition induced by an MLP having a 2-dimensional input is given in Figure 1.

Partitioning, Number of regions, and Function approximation: The approximation capability of
a DNN for a given interval in the input space is directly proportional to the number of regions and the
mapping associated with that input space interval. As per the continuous piece-wise affine property
of DNNs defined in Equation 1, consider the two possible scenarios in terms of approximation: (i)
the target function is linear in a given interval, in which case a single region is sufficient enough

2

Figure 1: Continuous Piece-wise Affine view of MLP. 2-dimensional visualization of the input
space partitioning induced by a one hidden layer MLP randomly initialized using standard with
bias (Left) and zero bias (Right). Each region, depicted by a particular color and bounded by black
lines, has a set of CPA parameters A?, B? described in Equation 1. These parameters depend on the
per-layer affine parameters and the state of the nonlinearities of the region ?.

to approximate it; the DNN is only required to adjust its slope and bias for the interval, or (ii) the
function is non-linear in the interval, in which case the DNN would require multiple regions to
approximate the curvature of the target function; the more regions in the interval corresponds in turn
to better function approximation.

In Figure 2, we validate the above claim and present a visualization of such phenomena in DNN. The
target function to be approximated is a simple sin function with input space ? [?2?, 2?]. First, the
higher the number of neurons, the higher the approximation power. In particular, with enough regions,
the DNN can approximate arbitrarily complex functions within an input space. Theoretically, we
know that a DNN with an infinite number of neurons is a universal approximator and the geometric
view presents a different view of the same theorem. Second, the approximation error associated with
each interval locally is directly proportional to the number of regions available to the DNN in that
interval. Finally, the positioning of these regions is data-driven, albeit architectural changes induce a
bias, DNNs can densify with more or less partitions in their input space that require more curvature
based on the uniformity and size of training data.

Number hidden neurons: 50

Number hidden neurons: 500

Figure 2: DNN approximation & induced number of input space regions. The ground truth and
approximation of a sine function by an MLP ( (Top)), the number of associated regions the MLP
induces in its input space (Middle), and the approximation error (Bottom). We present results for a
1-hidden layer MLP with 50 neurons (Left) and 500 neurons (Right). We note that the model breaks
from its linear behavior with the DNN introducing a new region whenever a change of direction in
the MLP mapping occurs. Subsequently, we obtain a new affine mapping as per Equation 1 for each
new region created by the model with finer approximation in spaces where the number of regions is
higher, as seen in the wider MLP with 500 neurons. The crucial advantage of DNNs is their ability to
adapt the positioning of these regions and learn data-driven partitions.

3

When adding neurons there is an increase in
the number of regions, thus the approximation
power of the DNN does increase. We ask now
the question of whether there is another way to
increase DNN capacity without affecting the ar-
chitecture. In particular, we investigate how the
number of regions interacts with the intrinsic
dimension (see subsection 2.2 for definition) of
the input space. In Figure 3, we show for differ-
ent sizes of 1-hidden layer MLP that the number
of regions scales exponentially with the intrinsic
dimension.

In the following section, we make use of the
geometrical aspects of MLPs, i.e., the approxi-
mation, expressivity, and dimensionality, in con-
junction with a multi-head attention layer to un-
derstand the geometry of transformer modules in
LLMs. In particular, we present a framework for
understanding LLMs through these geometrical
concepts, both from a theoretical and empirical
standpoint.

2.2 Large Language Models

Figure 3: Number of regions as a function of
input dimension - Upper bound of number of re-
gions spanned by a 1-hidden layer MLP (50, 100,
and 500 neurons) concerning the input space in-
trinsic dimension. We observe that increasing the
intrinsic dimension affect exponentially the num-
ber of regions. As such, for a given number of
neurons, one can artificially increase the number
of regions by increasing the intrinsic dimension
of the input space. This will be a crucial compo-
nent to understanding why increasing the size of
the prompt via many-shot or CoT induces better
reasoning capabilities in LLMs. This will be the
central point of subsection 2.2 as well as section 3.

In this section, we interpret the architectural
components of an LLM and its variations that
can help improve the expressive power of the
LLMs. Concretely, we will study the impact
of the LLM-induced partition concerning an in-
crease in the number of attention heads as well
as the context length (the sequence of tokens
passed as input). To do so, we will exploit re-
sults from [11], showing that the expressive power of an LLM increases as the intrinsic dimension of
the self-attention layer increases.

Intrinsic dimension ? Multi-Head Attention graph density: We begin by introducing notation
through the definition of a transformer layer in a causal LLM, as follows

Head(?)

h (X) ? softmaxcausal

(cid:18)

XQ(?)
h

(cid:16)

XK(?)
h

(cid:17)?(cid:19)

XV (?)
h ,

(single-head mapping of X)

MHA(?)(X) ?

H
(cid:88)

h=1

Head(?)

h (X)O(?)
h ,

Layer(?)(X) ? MLP(?) (cid:16)
(cid:16)

LayerNorm(?) (cid:16)
Layer(L) ? � � � ? Layer(1)(cid:17)

LLM(X) ?

(X),

MHA(?)(X) + X

(2)

(combination of H heads)

(cid:17)(cid:17)

+ X,

(single layer)

(3)

(4)

(compose L layers) (5)

where we denote the attention map as follows

Attn(?)

h (X) ? softmaxcausal

(cid:18)

XQ(?)

h K(?)

h

?

X ?

(cid:19)

.

(6)

It is evident from Equation 6, that the output of an attention layer is a right stochastic matrix that
defines a graph where the nodes of the graph are the sequence of tokens and the edges (weights)
are defined by the attention values. We will usually refer to density of the self-attention graph when
expressing the level of connectivity of the graph, i.e., the number of tokens that have an edge.

4

In Theorem 2.1, we capture explicitly the relationship between the output of the multi-head attention
layer as defined in Equation 5 and the intrinsic dimension driven by the sum of the dimensions
induced in each individual attention layer.
Theorem 2.1 (causal multi-head Minkowski sum ([11]) ). The ith row of the MHA mapping output
(Equation 3) lives in the Minkowski sum of single-head convex hulls as (MHA(?)(X)i,.)? ? H(?)
1 (i)+
� � �+H(?)
with effective dimension at most

(V (?)

h )?xj, j = 1, . . . , i

H (i) where H(?)

h (i) ? Hull

h O(?)

(cid:110)

(cid:111)

H
(cid:88)

h=1

#

(cid:110)

Attn(?)

(cid:111)
h (X (?))i,j > 0, j = {1, 2, . . . , i}

.

(7)

From Equation 7, it is clear that the intrinsic dimension can be increased by either (i) enforcing a
highly connected attention graph, or (ii) adding more attention heads. We will now exploit such a
property and connect it to the expressive power of LLMs.

Intrinsic Dimension (ID): The ID of an embedding space refers to the minimum number of
parameters required for its characterization while maintaining its structure [15]. Approaches for ID
estimation [16, 17] often rely on the construction of similarity-based graphs [18]. However, in LLMs,
the similarity graph is readily available in the form of attention values. We define a soft notion of
intrinsic dimension, equivalent to the definition in Theorem 2.1, namely,

ID?

?(i):=#

(cid:110)

Attn(?)

(cid:111)
h (X (?))i,j > ?, j = {1, 2, . . . , i}

.

(8)

Intuitively, ID?
?(i) is the number of tokens that are influential, beyond a threshold ?, in defining the
ith embedding. In practice, we set the threshold ? based on the statistics and the distribution of the
attention values across several examples (0.1 in all our experiments).

LLM expressive power ? intrinsic dimension: Theorem 2.1 is consequential, specifically when
we consider subsection 2.1, and in particular with Figure 3. We showed that: (i) the higher the
number of regions, the higher the approximation capability of DNNs, and (ii) the number of regions
can be increased by, not only having more neurons but by increasing the ID of the MLP�s input.

We also know from the transformer architecture described in Equation 2 through Equation 5 and
Theorem 2.1 that the intrinsic dimension of the input to the MLP is driven by the attention maps.
Therefore, the higher the density of the attention graph, the higher the number of regions that will be
induced by the MLP, and thus, the higher its expressive power.

It is now clear that one can increase the expressive power of an LLM by (i) increasing the number of
heads as per the additive nature of Equation 7, (ii) performing prompt modifications as to increase
the density of the attention graph. Note that both these approaches have commonly been employed in
various aspects in the last couple of years.

In Figure 4, we propose to re-use our sine function toy-example. Specifically, we show the number
of regions induced by the MLP for different context lengths and number of heads. We consider
a one-layer LLM, i.e., embedding, self-attention, and then 1-hidden layer MLP. To encode the 1-
dimensional time dimension into a higher dimensional space, we consider the embedding layer a
"positional encoding". Specifically, each time bin t is mapped to a sinusoid which frequency depends
on the context length as well as the position. We observe that the number of regions induced by the
MLP in the input space increases with both the context length and the number of heads. Similarly
as with the MLP example in subsection 2.1, the capabilities of the LLM are tied to the number of
regions, that is, the more populated a region of the input space, the better the approximation.

We provide in Figure 5 a more quantitative experiment regarding the number of regions induced by
the MLP concerning context length and number of attention heads. Here again, we observe that to
increase the number of regions and therefore improve the approximation capabilities of LLMs, one
can increase the number of heads in the self-attention block or increase the context length.

It is now clear that these correlations are the result of Theorem 2.1 together with the hyperplane
arrangement result displayed in Figure 3. That is, the number of regions induced by hyperplane
arrangement exponentially increases with high intrinsic dimensional spaces. In LLMs we identified

5

Context Length: 10 - Numbers of Heads: 1

Context Length: 10 - Numbers of Heads: 10

Context Length: 100 - Numbers of Heads: 1

Context Length: 100 - Numbers of Heads: 10

Figure 4: LLM approximation & induced number of input space regions - Visualization of
sin(t) (1000 time bins) approximation by a 1-block LLM, i.e., embedding ? attention block (as in
Equation 3) ? 1-hidden layer MLP. We display the approximation of the sin function together with
the number of regions induced by the MLP in the input space for different numbers of heads and
context lengths (Top Left) context length: 10 and number of heads: 1, (Top Right) context length: 10
and number of heads: 10, (Bottom Left) context length: 100 and number of heads: 1, (Bottom Right)
context length: 100 and number of heads: 10. We observe that both context length and number of
heads are inducing an increase in the number of regions spanned by the MLP in the input space, which
improves the approximation capabilities of the LLM. This result coincides with our geometrical
description.

that the number of heads as well as the context length are ways to increase the intrinsic dimension of
the MLP input, therefore increasing its capability to generate dense partitions.

We now propose to analyze how using this geometrical relationship as a tool to increase the expressive
power of LLM can lead to better reasoning capabilities.

3 Experiment: Increasing LLM expressive power does improve its reasoning

ability

In this section, we are analyzing the capabilities of LLMs to answer reasoning questions through the
lens of the aforementioned geometric analysis. Specifically, we are questioning how the increase
in a number of regions induced by the MLP can lead to better reasoning capabilities. In fact, it is
clear that approximation capabilities and generalization are not equivalent notions. However, it is
not yet determined that the reasoning capabilities of LLMs are tied to their generalization. While

6

Figure 5: LLM input space regions - (Left) Depiction of the number of regions induced by the MLP
block in the input space of the LLM concerning the number of attention heads and context length.
(Right) Zoom in on two rows of the left figure, specifically for several attention heads: 5, 10. We
observe that increasing both attention heads and context length does increase the number of regions,
which as mentioned, leads to better approximation properties. It is important to note that, while
changing the number of attention heads can be tedious and require pre-training or fine-tuning, one
can seamlessly vary the context length. There is therefore a way to improve LLM approximation
capability without interacting with the weights of the model.

these notions are still hard to pinpoint, we will focus in this experimental section on the relationship
between intrinsic dimension, thus expressive power, and reasoning capabilities.

We propose two experiments to demonstrate that there is an intriguing correlation between them. For
our experiments, we utilized the GSM8K-Zero dataset to assess the model�s performance in generating
correct answers across different few-shot scenarios, ranging from 0 to 10 shots. Specifically, for
each sample and each 1 to 10-shot condition, we examined how the intrinsic dimension of the model
varied across different layers when compared to the 0-shot baseline. Additionally, we evaluated how
these variations influenced the quality of the model�s responses. In the first experiment reported in
Figure 6, the few shot examples are question-answer pairs randomly sampled from the GSM8K-Zero
training set. For the second experiment reported in Figure 7, these few shot examples are random
tokens.

From these experiments, we make the following observations: (i) pre-pending the question at hand
with any type of token does increase the intrinsic dimension at the first layer. In fact, the first layer
attention graph behaves as a uniform distribution over the tokens, however, this increase is not
necessarily correlated with the reasoning capability of the model as the random token experiment
demonstrates Figure 7. (ii) We observe that when the pre-pended tokens lead to an increase in the
intrinsic dimension at the final layer of the model, the reasoning capabilities of the LLM improve
significantly. This improvement is reflected in a higher percentage of questions being answered
correctly.

In Figure 8, we display the variation in intrinsic dimension of the 1 to 10 shots sampled with respect
to 0 for each layer. We clearly see that no matter the size of the model, the last layers ID are highly
informative regarding the correctness of the response. While the first layers seem to have a huge
variation in ID whether the output is correct or not, the variance is too large to be significant and
reliable.

These experiments highlight the correlation between a model�s expressive power and its reasoning
capabilities. As discussed in section 2, enhancing this expressive power can be achieved by increasing
the dimension of the input to the MLP blocks. This relationship suggests that more complex input
contributes to the improved reasoning performance of the model.

In LLMs, adding context to the prompt can increase the ID (depending on how related is the context
to the question), and therefore increase the number of piece-wise affine maps produced by the MLP.
One should note that, for an LLM, each token output by the self-attention head is independently
transformed by the MLP. Thus, an MLP with a finer partition will have a more adaptive affine map
for each token. If we think about this from an approximation standpoint, as the tokens are linearly
combined to produce their predictions, the approximation error that is independently applied to each
of them by the MLP can compound easily, and therefore, the more precise the partitioning around

7

Llama3 8B

Llama3 70B

Figure 6: Reasoning vs ID increase. Percentage of correct responses, i.e., reasoning or extraction,
concerning relative ID change for Llama3 8B (Left) and 70B (Right) Instruct models. The actual
number of correct responses and the number of examples associated with each bin are denoted above
each histogram for reference. We consider as input base prompt examples with incorrect responses
from the GSM8K-Zero dataset (approx. 300 samples), along with their prepended variants where
1 to 10 fixed few-shot examples are used. For each input, we collect (i) the change in the intrinsic
dimension of the input concerning the base prompt, where the ID is computed at the final layer,
and (ii) the correctness in the output generated by the LLM. We evaluate the response generated by
prompting a Mixtral 8 � 22B Instruct model. We observe that the higher the ID change, the higher
the probability of obtaining a correct response from the LLM.

Llama3 8B

Llama3 8B

Figure 7: Ablation with random tokens. Percentage of correct responses, i.e., reasoning or
extraction, concerning relative ID change for Llama3 8B Instruct model with random (Left) and
shuffled few-shot example text (Right). As in Figure 6, we consider as input base prompt examples
with incorrect responses from the GSM8K-Zero dataset (approx. 300 samples), along with their
prepended variants obtained through randomly sampled tokens or permuted text in the few-shot
examples. We observe that the increase in ID is limited in the examples (< 60) and even negative for
the random token case. Consequently, obtaining a correct response is saturated and averages out to
around 40%, which is similar to the case with the 8B model and few-shot examples.

8

these tokens, the less the approximation error in the prediction. An aspect that has not been explored
here as well as in most work is how these notions are tied to the generalization capabilities, if any, of
LLMs.

In LLMs, incorporating additional context into the prompt can increase the intrinsic dimension of
the model, particularly if the context is closely related to the question. This increase in ID leads
to a greater number of piece-wise affine maps produced by the MLP. It�s important to note that in
LLMs, each token output by the self-attention mechanism is independently transformed by the MLP.
Consequently, an MLP with a more refined partitioning scheme will apply a more adaptive affine
map to each token.

Llama3 8B

Llama3 70B

Figure 8: Reasoning vs ID across layers. Correct vs Incorrect response with respect to relative ID
change for Llama3 8B (Left) and 70B (Right) Instruct models across each layer. We consider as
input base prompt examples with incorrect responses from the GSM8K-Zero dataset (approx. 300
samples), along with their prepended variants where 1 to 10 fixed few-shot examples are used. For
each input, we collect (i) the change in the intrinsic dimension of the input with respect to the base
prompt, where the ID is computed at the final layer, and (ii) the correctness in the output generated
by the LLM. We evaluate the response generated by prompting a Mixtral 8 � 22B Instruct model.
We observe that the higher the ID change, the higher the probability of obtaining a correct response
from the LLM.

From an approximation perspective, since the model�s predictions are formed by linearly combining
these embedded tokens, the approximation error can accumulate across tokens. Therefore, finer
partitioning around the tokens reduces the approximation error in the final prediction.

An intriguing aspect that remains largely unexplored in this work, as well as in most related research,
is how these geometric insights into intrinsic dimension and affine map partitioning relate to the
generalization capabilities of LLMs. This connection could offer valuable insights into the robustness
and adaptability of these models in various contexts.

4 Related Work

The success of transformer-based models [10] across various input modalities has spurred significant
research into the understanding of their internal mechanisms. Our work follows the lead of several
key works on this topic. The difference, however, between these previous works and ours is the
lens of analysis: we focus, fundamentally, on an end-to-end geometric perspective rather than a
mechanistic framework [19] or pattern analysis through empirical results [20�22]. Our work is also
different from these prior works in that we study the impact of model size and context length in
transformer models and their role in reasoning capabilities, a critical aspect of modern LLMs whose
understanding is largely absent.

9

Theoretical works for understanding the reasoning capabilities of LLMs make use of input-output
relationships through different frameworks in a domain-specific manner.
[23�25] make use of
graph problems to understand the expressiveness of LLMs and associate them with the algorithmic
complexity of the graph problem. [26�28] use algorithmic reasoning as a way to understand the
limitations of LLMs reasoning abilities. [29] investigate arithmetic learning and the impact of input
formatting on LLM reasoning. Closely related, [30] investigates the ability of LLMs to learn group
actions. [31] consider a two-layer causal transformer and evaluate its generalization capability for
copying, reversing, and sorting operations.

Other studies on transformers focus on initialization and training dynamics [32�35]. Albeit resorting
to simplifying assumptions, these works shed light on the role of different components, such as
the residual connection. The embedding geometry in the intermediate and last layers has also been
explored previously. [36] provides empirical insights about the position and context embeddings, [36]
presents an asymptotic (both in data and model) analysis to explain the emergent abilities of LLMs
through latent space modeling, and [37] identifies linear subspaces in contextualized embeddings to
demonstrate geometric structure in LLMs.

Other works [38�40] have studied the role of capacity in understanding LLMs and their transfer
performance. In particular, [39] empirically observed the role of intrinsic dimension (embedding
dimension) in LLMs and its impact on generalization and downstream task representations. We note
that our approach generalizes these observations while accommodating the sequence dimension, i.e.,
unlike previous works that relied on the dimension of entire sentences or tasks for their study, our
geometric study presents a context-dependent analysis of LLMs.

Our work makes use of several mathematical tools developed with deep neural networks, in general, to
understand transformer architecture. These observations, individually, may not be novel or have been
implicitly noted in the literature. Notably, the spline view of neural networks was previously presented
[41], which considered a partitioning of a fixed dimensional input space by the non-linearities in
the network. Moreover, we note that the mathematical ideas presented in this work are likely
implicitly known to researchers and practitioners familiar with transformers, and our contribution lies
in leveraging this understanding to build a geometric interpretation of transformers.

5 Discussion and Open Questions

We presented here some aspects of DNNs and LLMs geometry, where in particular, we show the
importance of the input space partitioning induced by the MLPs exploiting their piece-wise affine
formulation. The adaptive partitioning of DNN in general plays a huge role in their approximation
capability. In fact, as opposed to traditional spline, the regions induced by the MLP in their input space
are data-dependent, and henceforth determined during training. We showed how such an interplay
between approximation and the number of regions impacts the ability of LLMs to approximate
functions. Then, we show that, while approximation power is not equivalent to generalization, it
seems to be highly correlated to the reasoning capabilities of LLMs. In this work, we provided a brief
overview of the underlying theory and a limited set of experiments related to these concepts. We
believe that further exploration of this phenomenon is crucial to enhancing the reasoning capabilities
of LLMs. Our hope is that through this, smaller LLMs can soon bridge the performance gap with
their larger counterparts.

References

[1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida,
J. Altenschmidt, S. Altman, S. Anadkat, et al., �Gpt-4 technical report,� arXiv preprint
arXiv:2303.08774, 2023.

[2] AI@Meta, �Llama 3 model card,� 2024.

[3] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Rad-
ford, J. Wu, and D. Amodei, �Scaling laws for neural language models,� arXiv preprint
arXiv:2001.08361, 2020.

10

[4] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas,
L. A. Hendricks, J. Welbl, A. Clark, et al., �Training compute-optimal large language models,�
arXiv preprint arXiv:2203.15556, 2022.

[5] D. Hernandez, J. Kaplan, T. Henighan, and S. McCandlish, �Scaling laws for transfer,� arXiv

preprint arXiv:2102.01293, 2021.

[6] J. Pfau, W. Merrill, and S. R. Bowman, �Let�s think dot by dot: Hidden computation in

transformer language models,� arXiv preprint arXiv:2404.15758, 2024.

[7] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al., �Chain-of-
thought prompting elicits reasoning in large language models,� Advances in Neural Information
Processing Systems, vol. 35, pp. 24824�24837, 2022.

[8] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang,

�Retrieval-augmented generation for large language models: A survey,� 2024.

[9] R. Agarwal, A. Singh, L. M. Zhang, B. Bohnet, S. Chan, A. Anand, Z. Abbas, A. Nova, J. D.
Co-Reyes, E. Chu, et al., �Many-shot in-context learning,� arXiv preprint arXiv:2404.11018,
2024.

[10] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, ?. Kaiser, and
I. Polosukhin, �Attention is all you need,� Advances in neural information processing systems,
vol. 30, 2017.

[11] R. Balestriero, R. Cosentino, and S. Shekkizhar, �Characterizing large language model geometry

solves toxicity detection and generation,� arXiv preprint arXiv:2312.01648, 2023.

[12] R. Balestriero and R. Baraniuk, �A spline theory of deep learning,� in International Conference

on Machine Learning, pp. 374�383, 2018.

[13] R. Balestriero and R. G. Baraniuk, �Mad max: Affine spline insights into deep learning,�

Proceedings of the IEEE, vol. 109, no. 5, pp. 704�727, 2020.

[14] R. Balestriero, R. Cosentino, B. Aazhang, and R. Baraniuk, �The geometry of deep networks:
Power diagram subdivision,� Advances in Neural Information Processing Systems, vol. 32,
2019.

[15] R. Bennett, �The intrinsic dimensionality of signal collections,� IEEE Transactions on Informa-

tion Theory, vol. 15, no. 5, pp. 517�525, 1969.

[16] P. Campadelli, E. Casiraghi, C. Ceruti, and A. Rozza, �Intrinsic dimension estimation: Relevant
techniques and a benchmark framework,� Mathematical Problems in Engineering, 2015.

[17] P. Pope, C. Zhu, A. Abdelkader, M. Goldblum, and T. Goldstein, �The intrinsic dimension of

images and its impact on learning,� arXiv preprint arXiv:2104.08894, 2021.

[18] S. Shekkizhar and A. Ortega, �Graph construction from data by non-negative kernel regression,�
in Intl. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pp. 3892�3896, IEEE,
2020.

[19] N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen,
T. Conerly, et al., �A mathematical framework for transformer circuits,� Transformer Circuits
Thread, vol. 1, 2021.

[20] E. Voita, D. Talbot, F. Moiseev, R. Sennrich, and I. Titov, �Analyzing multi-head self-attention:
Specialized heads do the heavy lifting, the rest can be pruned,� arXiv preprint arXiv:1905.09418,
2019.

[21] Z. Niu, G. Zhong, and H. Yu, �A review on the attention mechanism of deep learning,� Neuro-

computing, vol. 452, pp. 48�62, 2021.

[22] A. Panigrahi, N. Saunshi, H. Zhao, and S. Arora, �Task-specific skill localization in fine-tuned
language models,� in International Conference on Machine Learning, pp. 27011�27033, PMLR,
2023.

11

[23] J. Kim, D. Nguyen, S. Min, S. Cho, M. Lee, H. Lee, and S. Hong, �Pure transformers are pow-
erful graph learners,� Advances in Neural Information Processing Systems, vol. 35, pp. 14582�
14595, 2022.

[24] C. Sanford, B. Fatemi, E. Hall, A. Tsitsulin, M. Kazemi, J. Halcrow, B. Perozzi, and V. Mir-
rokni, �Understanding transformer reasoning capabilities via graph algorithms,� arXiv preprint
arXiv:2405.18512, 2024.

[25] C. Sanford, D. Hsu, and M. Telgarsky, �Transformers, parallel computation, and logarithmic

depth,� arXiv preprint arXiv:2402.09268, 2024.

[26] H. Zhou, A. Nova, H. Larochelle, A. Courville, B. Neyshabur, and H. Sedghi, �Teaching

algorithmic reasoning via in-context learning,� 2022.

[27] E. Zelikman, Q. Huang, G. Poesia, N. Goodman, and N. Haber, �Parsel: Algorithmic reasoning
with language models by composing decompositions,� in Advances in Neural Information
Processing Systems (A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine,
eds.), vol. 36, pp. 31466�31523, Curran Associates, Inc., 2023.

[28] B. Liu, J. Ash, S. Goel, A. Krishnamurthy, and C. Zhang, �Exposing attention glitches with
flip-flop language modeling,� Advances in Neural Information Processing Systems, vol. 36,
2024.

[29] N. Lee, K. Sreenivasan, J. D. Lee, K. Lee, and D. Papailiopoulos, �Teaching arithmetic to small

transformers,� arXiv preprint arXiv:2307.03381, 2023.

[30] Y. Zhang, A. Backurs, S. Bubeck, R. Eldan, S. Gunasekar, and T. Wagner, �Unveiling trans-
formers with lego: a synthetic reasoning task,� arXiv preprint arXiv:2206.04301, 2022.

[31] Y. Li and J. L. McClelland, �Systematic generalization and emergent structures in transformers

trained on structured tasks,� arXiv preprint arXiv:2210.00400, 2022.

[32] Y. Dong, J.-B. Cordonnier, and A. Loukas, �Attention is not all you need: Pure attention loses
rank doubly exponentially with depth,� in International Conference on Machine Learning,
pp. 2793�2803, PMLR, 2021.

[33] L. Noci, S. Anagnostidis, L. Biggio, A. Orvieto, S. P. Singh, and A. Lucchi, �Signal propagation
in transformers: Theoretical perspectives and the role of rank collapse,� Advances in Neural
Information Processing Systems, vol. 35, pp. 27198�27211, 2022.

[34] E. Boix-Adsera, E. Littwin, E. Abbe, S. Bengio, and J. Susskind, �Transformers learn through

gradual rank increase,� arXiv preprint arXiv:2306.07042, 2023.

[35] A. Trockman and J. Z. Kolter, �Mimetic initialization of self-attention layers,� arXiv preprint

arXiv:2305.09828, 2023.

[36] J. Song and Y. Zhong, �Uncovering hidden geometry in transformers via disentangling position

and context,� arXiv preprint arXiv:2310.04861, 2023.

[37] E. Hernandez and J. Andreas, �The low-dimensional linear geometry of contextualized word

representations,� arXiv preprint arXiv:2105.07109, 2021.

[38] A. Aghajanyan, A. Shrivastava, A. Gupta, N. Goyal, L. Zettlemoyer, and S. Gupta, �Better
fine-tuning by reducing representational collapse,� arXiv preprint arXiv:2008.03156, 2020.

[39] A. Aghajanyan, L. Zettlemoyer, and S. Gupta, �Intrinsic dimensionality explains the effective-

ness of language model fine-tuning,� arXiv preprint arXiv:2012.13255, 2020.

[40] T. Chen, J. Frankle, S. Chang, S. Liu, Y. Zhang, Z. Wang, and M. Carbin, �The lottery ticket
hypothesis for pre-trained bert networks,� Advances in neural information processing systems,
vol. 33, pp. 15834�15846, 2020.

[41] R. Balestriero et al., �A spline theory of deep learning,� in International Conference on Machine

Learning, pp. 374�383, PMLR, 2018.

12


