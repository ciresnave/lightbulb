3
2
0
2

y
a
M
7
1

]
E
N
.
s
c
[

2
v
6
7
8
8
0
.
5
0
3
2
:
v
i
X
r
a

NEUROSYMBOLIC AI AND ITS TAXONOMY: A SURVEY

?

Wandemberg Gibaut
Eldorado Institute of Technology
Campinas, S�o Paulo
wandemberg.gibaut@eldorado.org.br

Leonardo Pereira
Eldorado Institute of Technology
Campinas, S�o Paulo
leonardo.pereira@eldorado.org.br

Fabio Grassiotto
Eldorado Institute of Technology
Campinas, S�o Paulo
fabio.grassiotto@eldorado.org.br

Alexandre Osorio
Eldorado Institute of Technology
Campinas, S�o Paulo
lalexandre.osorio@eldorado.org.br

Eder Gadioli
Eldorado Institute of Technology
Campinas, S�o Paulo
eder.gadioli@eldorado.org.br

Amparo Munoz
Eldorado Institute of Technology
Campinas, S�o Paulo
amparo.munoz@eldorado.org.br

Sildolfo Gomes
Eldorado Institute of Technology
Campinas, S�o Paulo
sildolfo.gomes@eldorado.org.br

Claudio Filipi Goncalves do Santos
Eldorado Institute of Technology
Campinas, S�o Paulo
claudio.santos@eldorado.org.br

ABSTRACT

Neurosymbolic AI deals with models that combine symbolic processing, like classic AI,
and neural networks, as it�s a very established area. These models are emerging as an
effort toward Arti?cial General Intelligence (AGI) by both exploring an alternative to just
increasing datasets� and models� sizes and combining Learning over the data distribution,
Reasoning on prior and learned knowledge, and by symbiotically using them. This survey
investigates research papers in this area during the recent years and brings classi?cation
and comparison between the presented models as well as applications.

Keywords Neurosymbolic AI � Deep Learning � Reasoning

1

Introduction

As Arti?cial Intelligence, and Deep Learning in particular, reach impressive results, it gains also unprece-
dented popularity not only in academics and industry but also in popular culture and society in general.
This increasingly ubiquitous AI presence has arisen several concerns about its impacts on humanity and
the planet, with some well-known scientists like Stephen Hawking having spoken concerns about AI�s
accountability [1].

Despite achieving outstanding results in Computer Vision, Natural Language Processing and Game Playing
[2, 3], tasks in which AIs formerly have poor performance compared to humans, those concerns about
AI triggered debates among research communities, including those discussed by Gary Marcus [4] and on
AAAI-2020 debate with Geoffrey Hinton, Yoshua Bengio and Yann LeCun [5].

?Citation: Gibaut et al. Neurosymbolic AI and its Taxonomy: a Survey DOI:10.48550/arXiv.2305.08876.

Neurosymbolic AI and its Taxonomy: a Survey

These discussions have a common ground about the future of AI: the aim during the next decades is to
investigate and build richer AI systems that are explainable, trustworthy, and based on solid principles that
unify the ability to learn from experience and to reason from what has been learned [6]. As discussed by
Daniel Kahneman[7, 5], there is a need for a "System 2" for Language (and symbol) manipulation.

Here, Neurosymbolic Computing appears as a strong candidate to ?ll those gaps by integrating learning
from the environment in a connectionist fashion and reasoning from what has been learned using symbolic
processing and representation. Table 1 shows a summary of the surveyed methods that are talked about in
this survey while Table 2 shows the papers we analyzed, also with a brief description.

Method

Description

Table 1: Brief explanation of surveyed methods

Logic Tensor Networks (LTN) [8]

Neural Logic Machine (NLM) [9]

Logical Boltzmann Machines (LBM) [10]

Logic Neural Networks (LNN) [11]

Neuro-Symbolic Concept Learner (NS-CL) [12]

LTN is a Neurosymbolic computational model and
formalism. It supports learning and reasoning through
Real Logic - a many-valued, end-to-end differentiable
?rst-order logic - as a representation language for
deep learning [8].

NLM is a Neurosymbolic architecture for inductive
learning and logic reasoning. By using neural networks
and a symbolic processor [9].

LBMs is a Neurosymbolic system capable of representing
any propositional logic formulae in strict disjunctive normal
form [10].

LNN is a novel framework that provides neural networks for
learning and symbolic logic for knowledge and reasoning.
Every neuron has a meaning as a component in a weighted
real-valued logic formulae [11].

NS-CL is a model that is capable of learning visual concepts,
words, and semantic parsing of sentences without explicit
supervision. The model learns by "looking" at images and
reading paired questions and answers, thus building an
object-based scene representation and translating sentences
into symbolic programs that can be executed [12].

Generative Neurosymbolic Machines (GNM) [13]

GNMs is a generative model that combines the bene?ts of distributed
and symbolic representations to support both structured
representations of symbolic components and
density-based generation [13].

A brief intermission by de?ning what symbols are is a good place to start, with the following subsection.

1.1 Symbols

In the classical article �Computer science as empirical inquiry: Symbols and search�, Alan Newell and Herbert Simon
introduced the concept of Physical Symbol Systems as the required building blocks for intelligent systems. Symbolic AI
had the initial intent of modeling specialist systems through the processing of symbolic inputs in what is known as
inference engines [15].

In the article, it is hypothesized that symbols used within a computational system are the very same representations
humans use for daily life, with the logical implication that we humans are instances of physical symbol systems. This
concept, familiar to anyone involved with Computing Science, comes from the formalization of logic.

The Physical Symbol Hyphothesis states, then, that for a physical system to show general intelligence action, it must be
a physical symbol system. The hypothesis can be explained by three main pillars: Necessity, Suf?ciency, and General
Intelligent Action.

Necessity can be explained in the sense that any physical system that exhibits general intelligence will be one implemen-
tation of a physical symbol system.

Suf?ciency in the sense that all physical symbol systems can be further organized to support general intelligence.

General Intelligent Action is supposed to be the scope of intelligence as can be seen in humans, i.e., the behavior that is
capable of supporting adaptation to an environment.

2

Neurosymbolic AI and its Taxonomy: a Survey

Table 2: Surveyed papers and their descriptions

Paper Name

Logic Tensor Networks

Description
"...a neurosymbolic framework that
supports querying, learning and
reasoning with both rich data and
abstract knowledge about
the world." [8]

Neural-Symbolic Integration for Fairness in AI
Logic Tensor Networks for
Semantic Image Interpretation

An application of LTN for AI Fairness
An application of LTN for
Semantic Image Interpretation

Neurosymbolic AI for Situated Language Understanding

An application of LTN with multimodal data
fo Language Understanding

Neural-Symbolic Reasoning
Under Open-World and Closed-World Assumptions

A study comparing LTN with purely
symbolix reasoning

Neural Logic Machines

Logical Neural Networks

LOA: Logical Optimal Actions for
Text-based Interaction Games

"... a neural-symbolic architecture for both
inductive learning and logic reasoning." [9]

"... a novel framework seamlessly providing
key properties of both neural nets (learning)
and symbolic logic
(knowledge and reasoning)." [11]

An application of LNN to Language
Interaction Game with Reinforcement
Learning

LNN-EL: A neuro-symbolic approach
to short-text entity linking.

An application of LNN to Natural Language
Processing

Leveraging abstract meaning representation
for knowledge base question answering

Logical Boltzmann Machines

"... a modular KBQA system, that
leverages (1) Abstract Meaning
Representation (AMR) parses for
task-independent question
understanding ..." [14]

"... a neurosymbolic system that can
represent any propositional logic
formula in strict disjunctive normal form." [10]

Expert Knowledge Induced Logic Tensor Networks:
A Bearing Fault Diagnosis Case Study

An application of LTN to bearing fault
diagnosis

Generative Neurosymbolic Machines

The Neuro-Symbolic Concept Learner:
Interpreting Scenes, Words, and Sentences
From Natural Supervision

A generative model that combines symbolic
and neural processing to support both
structured representations of
symbolic components and
density-based generation

"... a model that learns visual concepts,
words, and semantic parsing of sentences
without explicit supervision on any of
them; instead..." [12]

1.2 Methodoly and Scope of this Work

The objective of this survey is to provide an overview of the state-of-the-art of Neurosymbolic AI, which seeks to combine
the reasoning and explainability of symbolic representations of network models with the robust learning that can be
achieved with neural networks. Also, this paper focuses on works published from 2019 until now. Previous surveys
can also be found [16, 17, 18]. Compared to [16] our work brings papers published since 2019, while their work was
published in 2019 analysing works prior to that, our work is also focus on applications of neurosymbolic system which
they did not. The work made in [17] limits itself to a logical and probabilistic perspective, not focusing in frameworks or
applications. In [18] their focus was mainly in neuro-symbolic AI and graph combination and, while a few frameworks

3

Neurosymbolic AI and its Taxonomy: a Survey

analysed overlap with our analysis they did not metioned Neural Logic Machine (NLM) [9], Logical Boltzmann Machines
(LBM) [10], Logic Neural Networks (LNN) [11] or Generative Neurosymbolic Machines (GNM) [13]. Furthermore our
survey is the only survey among them that ties everything together, such as logic, frameworks, applications, etc...

The rest of this work is divided as follows: Section 2 discusses how each of the models we analyze in this paper deals with
the Knowledge Representations in their systems. Section 3 concerns a core topic on Neurosymbolic AI: how each system
implements its learning processes and its relations with pure Symbolic and pure connectionist systems. Section 4 shows
how Neurosymbolic systems reason about their knowledge and data extrapolation. Section 5 discusses Explainability
and Trustworthiness in Neurosymbolic systems, while Section 6 presents some interesting applications for those systems.
Finally, Section 7 wraps everything we discussed, presenting some conclusions and a table where one can visualize how
each model is located in the presented taxonomy. In this here survey we will be using existing taxonomy in the ?eld.

2 Knowledge Representation in Neurosymbolic Systems

This section discusses how knowledge is represented in neurosymbolic systems. [19] argues that logical calculus can
be carried out exactly or approximately by a neural network, so the knowledge representation would be the basis of a
neural-symbolic system that provides a mapping mechanism between symbolism and connectionism. On one hand, a
symbolic system knowledge is commonly represented as a discrete set of symbols [20]. Take for example a multi-class
classi?cation problem where we have four classes. These four classes are represented either as binary values or as a
4-state variable. On the other hand, neuron-like systems often adopt continuous output space as a standard. This can be
seen in deep learning systems such as regression tasks.

Both the LTN[8] and the LNN[11] frameworks analyzed in this section use First Order Logic (FOL). FOL enables the
representation of a certain sphere in terms of objects that already exist in that sphere and relations that hold between
certain objects in that sphere. As FOL can be treated as a language its vocabulary includes constant symbols, predicate
symbols, and functional symbols. In this domain, constants refer to objects, while predicates and functions refer to
relations.

First, it is analyzed how Logic Tensor Networks (LTN) [8] deal with knowledge. The concept behind LTN�s vision of
logic is Real Logic. Real Logic, simply put, is a way of representing symbolic knowledge so that it can be applied in a
Machine Learning environment [21]. Its main goal is to achieve a vector-based representation technique which should
be shown adequately for integrating machine learning and symbolic reasoning in a grounded way. To achieve this they
use fuzzy semantics (the truth value of a logical formula is between 0 and 1 instead of being just true or false).

The ?rst step to this system is a language L, which in this case would be First Order Logic. Once the language has been
established we need the constant symbols C to write with a given language, an example of these symbols would be ?, /? ,
etc... It also needs functional symbols F such as � or ?. Finally, predicate symbols P should be used, which are an
amalgamation of the two, meaning a set of relational symbols.

The semantics of this language in the "real world" comes in the form of groundings G. Groundings are the way Real
Logic maps the language L into the "real world" with real numbers in the form of n-tuples. Grounding also means that
domains D are interpreted as tensors in the "real world" if one so desires. To ground an atomic formula quanti?ers
and connectives are needed, those being �, ? and ?. Attention is needed to the fact that the term grounding does not
take this meaning in other contexts but in Real Logic the term is used to drive home the point that these semantics are
grounded in real values.

symbol C,

a better grasp of

con-
For
can write a predicate P such as:
stant
? x ? Class(Positive).. With this equation, we now ground these symbols with our grounding G for a real
value representation. This real representation could be a tensor which would represent the degree of truth of this
statement in our dataset or neural network.

an example
symbols F , ?.

is presented.
We

and functional

Given a

concept,

this

?,

language L,

Logic Neural Networks (LNN) [11] also use Real-Valued Logic when representing symbolic knowledge, but instead
of static values, it uses truth bounds. LNN also can not use functions or equality symbols and, since equality symbols
are not handled, it makes the the unique-names assumption, predicating that every constant symbol refers to a distinct
object in the domain. One distinct aspect of LNNs is that they introduce two quanti?ers when dealing with FOL, those
being universal and existential.

In LNNs there is an atomic neuron for each predicate symbol, a neuron for logical connectivity, and a connective neuron.
Each neuron carries a table whose columns are sorted by unique variables appearing in the represented sub-formulae,
and rows sorted by a set of n-element tuples of groundings, where the tuple size n corresponds to the arity of the neuron.
The contents of the tables are the bounds - as stated before - of each grounding when substituted for the variables.

When dealing with bounds they can be, if well explored, more interesting than static values: "The bounds of a universal
quanti?er node are set to the bounds of the grounding with the lowest upper bound, so that if all of its groundings are
True then the universal statement is also True. And the bounds of the existential quanti?er are set to the bounds of the
grounding with the highest lower bound so that if this lower bound is True the existential statement will be True when
at least one of its groundings is True" [11]. Dealing with LNNs also allows the exploration of the concept of inference

4

Neurosymbolic AI and its Taxonomy: a Survey

Figure 1: An illustration of Neural Logic Machines (NLM). During forward propagation, NLM takes object
properties and relations as input, performs sequential logic deduction, and outputs conclusive properties or
relations of the objects. Adapted from [9]

for First Order Logic. When exploring inference for FOL all neurons in the LNN return tables of groundings and their
corresponding bounds. Neural activation functions are then mutated to perform agglutination over columns that share
variables while still computing truth value bounds associated with certain groundings. Inverse activation functions are
modi?ed similarly but are then ?ltered, reducing results over non-signi?cant variables so that we can have the tightest
bounds.

[22] explain that LTNs are instances of the Differentiable Fuzzy Logic (DFL) method, which employs fuzzy logic with
differentiable logical operators. Loss functions that maximize the satis?ability of the symbolic knowledge base are built
into this method. Since the fuzzy logical operators are differentiable, well-known methods like gradient descent can be
employed in this optimization.

Neural Logic Machines (NLM) are presented by [9] as Neurosymbolic models designed to address some challenges that,
allegedly, pure connectionist Neural Network models cannot address. According to the authors, those challenges are:

� The learning system should recover a set of lifted rules (i.e., rules that apply to objects uniformly instead of

being tied with speci?c ones) and generalize to scenarios with different data than the presented.

� The learning system should deal with high-order relational data and quanti?ers, which go beyond the scope of
typical graph-structured neural networks like, jointly inspecting a set of objects to apply transitivity rules.

� The learning system should scale up regarding the complexity of the rules. The existing logic-driven approaches
suffer an exponential computational complexity as the number of logic rules to be learned increases (also
known as �the Curse of Dimensionality�).

� The learning system should recover rules based on a minimal set of learning priors.

Concerning the Knowledge Representation in NLMs, the model adopts probabilistic tensor representation for the logic
predicates, that are grounded in a set of objects. Also, NLMs have a � breath� property that is intrinsically related to the
concept of arity, which represents how many objects are considered in the logical rule (i.e. if a rule is about an object
itself, about a relationship between two of them, and so on). The total �width� of the model will also take into account
the number of prepositions to be learned. Figure illustrates this.

Logical Boltzmann Machines (LBM) appear ?rst in [10] as a Neurosymbolic system that can represent any propositional
logic formula in strict disjunctive form. The system relies on the equivalence between the logical satis?ability of formulas
and the energy minimization in Restricted Boltzmann Machines. An LBM system is built upon a weighted knowledge
base, with its topology re?ecting that prior knowledge. Here, each propositional logic formula is mapped into a Strict
Disjunctive Normal Form (SDNF), that is, a disjunction of conjunctions with at most one conjunctive clause that maps to
True for any assignment of truth-values x. An example of LBM for the Nixon diamond problem is shown in Figure 2.
The weighted knowledge base used to construct this network was:

� 1000 : n ? r Nixon is a Republican.
� 1000 : n ? q Nixon is also a Quaker.
� 10 : r ? �p Republicans tend not to be Paci?sts.
� 10 : q ? p Quakers tend to be Paci?sts.

5

Neurosymbolic AI and its Taxonomy: a Survey

Another Neurosymbolic system we will explore is the NeuroSymbolic Concept Learner (NS-CL) presented by [12]. This
model consists of an object-based scene representation and sentence translation into executable symbolic programs. They
use neurosymbolic reasoning to execute these programs on the latent scene representation. The logic behind the project
is to emulate human concept learning by allowing the perception module to learn visual concepts based on the language
description of the object referred to.

Just as humans are capable of learning visual concepts by jointly understanding vision and a given language, the authors
propose the NS-CL a system that jointly learns visual perception, words, and a semantic language parsing from images
and pairs of question-answer. The NS-CL consists of three modules those being: a neural-based perception module (this
module is used to extract object-level representations from a given scene), a visually- grounded semantic parser (this
module translates questions into executable programs) and a symbolic program executor (the module that will read out
the perceptual representation of objects, classi?es their attributes, and relations, and executes the program to obtain an
answer).

The NS-CL learns from natural supervision, for example, images and question-answer pairs. This means that it requires
no annotations on images or semantic programs for sentences. Just as a human would, this system learns via curriculum
learning. NS-CL will start by learning concepts or representations of individual objects from short questions on simple
scenes, for example,� What color is the cube?�. The ?nal result is learning object-based concepts such as colors and shapes.
The next step is learning relational concepts by leveraging object-based concepts to interpret object referrals, for example,�
Is there a sphere next to the cube?�. By default, the model will iteratively adapt to more complex scenes and highly
compositional questions. The authors argue that NS-CL�s modularized designenables interpretable, robust, and accurate
visual reasoning. As evidence, they claim that it achieves state-of-the-art performance on the CLEVR dataset [23].

3 Learning in Neurosymbolic Systems

There are some conceptually different learning processes in neurosymbolic systems regarding the integration of prior
knowledge and data generalization. [19] lists methods that approach learning and neurosymbolic systems that were
tested and proved. This section explains a few of those methods.

3.1

Inductive Logic Programming

First we have Inductive Logic Programming (ILP). In an ILP system, we take advantage of the learning capability of
neurosymbolic computing to generate a logic program automatically from examples. We can approach ILP as bottom-up
or top-down. In a bottom-up approach, a logic program is built by extracting speci?c clauses from examples and, after
that, generalization procedures are applied in search of more general clauses [19]. The disadvantage of this approach is
that it highly depends on how well the generalization can be. In the ?eld of neural-symbolic computing, bottom clause

Figure 2: An example of RBM for the Nixon diamond problem. With = 0.5, this RBM has energy function:
E = -h1(1000n+1000r-1500)-h2(-2000n+1000)-h3(1000n+1000q-1500)-h4(10r-10p-5)-h5(-10r+5)-h6(10q+10p-15)-
h7(-10q+5). Adapted from [10]

6

Neurosymbolic AI and its Taxonomy: a Survey

propositionalisation is a popular approach because bottom clause literals can be encoded into the neural network directly
as data features while presenting their semantic meaning.

On the other hand, in a top-down approach, a logic program is built by extracting general clauses from examples. The
resulting program is then extended to speci?c problems [19]. The most popular idea concerning this approach is to
take advantage of neural networks� inference and learning capabilities to ?ne-tune and test the quality of our rules by
replacing logical operations with differentiable operations. Starting with the most general clauses, a top-down based
system creates rules from facts. We can cite Neural Logic Programming (NLP) [24] as an example. In NLP learning
of rules are based on the differentiable inference of TensorLogs. In this application matrix computations are used to
soften logic operators where con?dence of conjuctions and con?dence of disjunctions are computed as product and sum,
respectively.

In Neural Logic Machines (NLM) [9], the learning process is given by inputting tensors representing predicates of
different arities (�facts�, considering the Closed-World Assumption [25]) from a knowledge base and the model outputs
tensors representing new predicates. For example, in the Family Tree example presented by the authors, the input
predicates are IsSon, IsDaughter, IsFather and IsMother and the outputs (conclusions) are HasFather, HasSister, IsGrandparent,
IsUncle and IsMGUncle.

3.2 Hybrid Learning

Another important approach to learning in a neurosymbolic context is Horizontal Hybrid Learning (HHL), which is a
neurosymbolic answer to data problems. Techniques such as deep learning usually require large amounts of data to ?nd
statistical regularities. However, the access to large datasets can be dif?cult and, as we know, a small dataset can make
models susceptible to over?tting. On the other hand, a neurosymbolic system has the advantage of generalization if
prior knowledge is provided. This characteristic is achieved by combining logical formulas with data during the learning
process while using data to ?ne-tune our knowledge at the same time. We can improve the ef?ciency and effectiveness of
a neural network by encoding logical knowledge as controlled parameters while training models [26, 19]. When lacking
prior knowledge the ideia of neural-symbolic integration for knowledge transfer learning can be applied. This solution
aims to extract symbolic knowledge from a related domain and transfer it to improve the learning in another domain,
starting from a network that does not need to be instilled with background knowledge.

The next learning approach is Vertical Hybrid Learning (VHL) [19], which employs an idea similar to how human
brains work. Neuroscienti?c studies show that different areas of the brain process different input signals such as logical
thinking and emotion processing [27]. In VHL we place a logic network on top of a deep neural network to learn what is
happening in a black-box system that has a high-level abstraction from complex inputs such as audio, images and text.
As an example in [28] a Fast-RCNN [29] is used for bounding-box detection of parts of objects in combination with a
Logic Tensor Network, which is used to reason about relations between parts of objects and types of objects, meaning
that the perception part (Fast-RCNN) is ?xed and learning is carried out in the reasoning part (LTN).
In LTN [8], learning refers to the satis?ability of a triple T = (cid:104)K, G(�|?), ?(cid:105) where K is a set of closed ?rst-order logic
formulas G(�|?) is a parametric grounding for all the symbols S = D ? X ? C ? F ? P and all the logical operators;
and ? = {?s} for s ? S is the hypothesis space for each set of parameters ?s associated with symbol s. The learning
processes may occur on the grounding of constants (embedding), grounding of functions (regression or generative task)
or grounding of predicates (classi?cation). The idea in such systems is to embed prior knowledge in the network as
axioms.

Although LBM systems [10] are necessarily built using a weighted knowledge base, they can further improve their
performance in certain tasks by learning over data. This process is pretty straightforward with train and validation
separation and the target variable left out. The inference is performed using a conditional distribution on the target
variable.

The NS-CL learning process consists of its semantic parser generating the hierarchies of latent programs in a sequence
to tree manager [30]. They use a bidirectional GRU [31] to encode an input question which outputs a ?xed-lenght
embedding of the question. After that a decoder based on GRU cells is applied to the embedding and recovers the
hierarchy of operations as the latent program.

Given the latent program recovered from the question in natural language a symbolic program executor is enabled and
derives the answer based on the object-based visual representation. This program executor is a collection of deterministic
functional modules which were designed to execute all logic operations speci?ed in the domain-speci?c language (DSL).

To make the execution differentiable from the visual representations, the authors represent the intermediate result in a
probabilistic manner. A set of objects is represented by a vector, just as the attention mask over all objects in the scene.
Each element Maski denotes the probability of the n-th object of the scene belonging to the set. The ?rst Filter operation
will output a mask with size 4, since there are 4 objects in the scene. Each element will represent the probability that the
corresponding object is selected. The output mask on the object will be then fed into the next module as input and the
execution continues normally.

7

Neurosymbolic AI and its Taxonomy: a Survey

The NS-CL training process is split into four stages: learning object-level visual concepts, learning relational questions,
learning more complex questions with perception modules ?xed and joint ?ne-tuning of all modules. The authors found
out that the joint ?ne-tuning is essential to their neurosymbolic concept learner.

4 Reasoning in Neurosymbolic Systems

From a purely mathematical perspective, Reasoning is the task of verifying if a certain conclusion is a logical consequence
of a set of premises, both interpreted as symbols [8]. Although there is other ways, a model-based approach is usually
the focus of Neurosymbolic systems, as there is a need to integrate Learning and Reasoning.

This section explores how reasoning happens in a neurosymbolic context. Reasoning being such an important feature in
a neural network system, various approaches have been studied and tested in this context. These approaches can be
model-based or theorem proving. Given that in a neurosymbolic enviroment the main focus lies in the integration of
reasoning and learning the preferred approach is model-based.

4.1 Forward and Backward chaining

The ?rst approach to model-based reasoning is forward and backward chaining. This approach consists of two popular
inference techniques for logic programs and systems. For the speci?c case - this being neurosymbolic systems - both
forward and backward chaining are implemented by feedforward inference. As implied by their names, forward and
backward chaining are opposites. In a forward chaining context new facts are generated from the head literals of the rules
using know facts that exist in the knowledge base. For example, imagine a dataset where we have famous individuals
and their contribution to society, we know that Machado de Assis was a writer and that implies he can read so it infers
that Frank Herbert could read because he also was a writer.

In contrast, backward chaining does a backward search from a goal in the knowledge base in order to determine if
a certain query is possible. Continuing the example given above, in the context of a backward chaining we query if
Frank Herbert could read and than ?nd in the knowledge base if another person with the same profession as him could
also read. It is important to acknowledge that backward reasoning is expressively harder to implement than forward
reasoning.

Inside this classi?cation, NLM [9] can be considered systems as a model that performs reasoning by forward chaining,
since it presents the inputs and outputs of neural networks as grounding tensors of predicates for existing facts and new
facts respectively.

4.2 Approximate Satis?ability

The second approach to model-based reasoning consists of approximate satisfability. When dealing with arbitrary
formulas, inference can be more complex, given that a way to traverse this is to search over our hypothesis space for a
solution that maximizes satisfability in the formulas and facts of knowledge base.

Reasoning with maximum satisfability is NP-hard and for this reason neurosymbolic systems offer approximate
satisfability. These systems are trained to approximate the best possible satisfability so inference is ef?cient with
feedforward propagation. This approach can be thought as a method that travels in the hypothesis space searching for
the point where the restrictions (formulas and facts) are in their most interesting form.

In LTN [8], the reasoning process is given by an approximate satis?ability of a relation of logical consequence between a
knowledge-base and a grounded theory for a closed formula ?. That can be done either by querrying after learning - that
is, by considering only the grounded theories that maximize the satis?ability level - or by searching a counter-example
to the refered logical consequence.

Since the level of satis?ability of a symbolically generated set of formulas that belong to the axiom to be checked on
a LTN gives a measure of the reasoning capability of a neural network, the reasoning capability of a LTN network is
obtained by calculating the grounding of any formula whose predicates are already grounded in the NN or by de?ning
a new predicate in terms of existing predicates [22]. Also, in [22] it is proved that the iterative provision of negative
information can improve reasoning capabilities dramatically.

As Logical Boltzmann Machines [10] relies on the equivalence between logical Satisfabiality and network energy
minimization, these systems see Reasoning as the mentioned energy minimization or an appoximation of it. Here, to
Reason about the satisfability of a prepostion, the system takes a set of assigned inputs xb and must perform a search for
an assingment of truth-values for xb that satis?es the formula from what the LBM is constructed. The method normally
used here is the Gibbs sampling in which the process starts with a random initialisation of xB, and proceeds to infer
values for the hidden units hj and then the unassigned variables xb in the visible layer of the RBM, using the conditional
distributions. If the number of unassigned variables is not too large such that the partition function can be calculated
directly, the Reasoning process may be performed by lowering the free energy.

8

Neurosymbolic AI and its Taxonomy: a Survey

4.3 Relationship reasoning

The next approach is the use of relationship reasoning. This method is farly known and consists of reasoning about
relationships between entities. The NS-CL [12], in contrast with other approaches analysed in this survey, uses visual
reasoning in order to determine an object�s attributes. In this system, visual attributes are implemented as neural
operators, mapping the object representation into an attribute-speci?c embbeding space. These visual concepts belonging
to the shape attribute are later represented as vectors, which are also learned along the process, in the shape embedding
space. They later measure the cosine distance between these vectors in order to compute the probability that an object is
a cube by using a prede?ned function. They classify relational concepts, for example Left and Right, between a paior of
objects similarly, except that they concatenate the visual representations of these objects to form a new representation (of
their relation).

The NS-CL [12] learning and reasoning heavly envolves it semantic parsing. The semantic parsing module translates
a natural language question into an executable program with a hiearchy based of primitive operations which are
represented in a DSL. This DSL takes care of a set of fundamental operations for visual reasoning, these being ?ltering
out objects with certain concepts, querying the attribute of an object, etc... The operation shares, of course, the same
input and output interface and, as a result, can be compositionally combined to form programs of any complexity.

4.4 Exploration of Practical Reasoning in DFL

In [32] they argue that practical reasoning in a LTN is considered a task similar to inference in traditional Machine
Learning, in which conventional statistical evaluation methods are considered to select the best grounding available.
The satis?ability level of a set of formulas, symbolically generated, that belong to the axiom to be checked on a LTN
gives a measure of the reasoning capability of a neural network. In this way, the reasoning capability of a LTN network
can be obtained by calculating the grounding of any formula whose predicates are already grounded in the NN or by
de?ning a set of new predicates in terms of other existing predicates.

In LTN, a quanti?ed formula can perform a kind of semi-supervised learning by expanding the set of examples in an
existing domain. On the other hand, it can help with transfer learning from a data-rich domain (de?ned by a set of
predicates) to domain with insuf?cient data (e.g. a new predicate). In [32] it is proved that the iterative provision of
negative information can improve reasoning capabilities dramatically.

5 Explainability and Trustworthiness in Neurosymbolic Systems

One of the biggest criticisms of Deep Learning techniques is the lack of explainability or the �BlackBox effect�. Most of
the current state-of-the-art models are built in such a way that they are mostly opaque: the user/programmer only may
gather knowledge about input-output pairing not knowing, for example, how a given feature impacts a classi?cation
decision. Even more problematic: the ubiquitous presence of AI systems in society raises questions about its negative
impacts on people�s lives, how much �trustworthy� is a system, and how we can increase it to have systems that took
good, fair decisions.

Many attempts have been made for solving the �BlackBox effect� based on rules extraction methods [33, 34, 35]. The
main goal of these attempts was to - based on accuracy, ?delity, consistency, and comprehensibility - search for logic
rules from a given trained network [33]. There were successful attempts to implement these ideas [36]. However, those
approaches were combinatorial and do not scale well when dealing with the dimensions of current neural networks
(deep learning) resulting in the idea beginning to cool off, until recently when the idea of neurosymbolic systems
reemerged as a combination of a global and local approach.

LTN systems [8] approach to this is by increasing user and ML model communication through querying and answering.
Here there are three types of queries: truth queries, value queries, and generalization truth queries.

One of the key features of neurosymbolic AI is its explainability. Given its connectionist network nature, a neurosymbolic
system can be represented in a set of readable expressions. Natural language generation can be mentioned as an approach
in which a user couples a deep network with sequence models to extract natural language knowledge. As can be seen
in [37] one can, instead of querying parameters of a trained model, extract relational knowledge where, by performing
inference of a trained embedding network on a text data, predicates are obtained.

Another area of interest in neurosymbolic AI is program synthesis. The use of neurosymbolic AI has been proposed to
build computer programs on an incremental approach based on large amounts of input-output samples [38]. This is
achieved by employing a neural network to represent partial trees. In a domain-speci?c language, there are tree nodes,
symbols, and rules represented as vectors. It can, of course, achieve explainability through the tree-based structure of
this here network.

Although nothing about explainability is directly mentioned in LBM [10], a Knowledge Extraction may be performed on
the system by querying it with appropriated unassigned variables.

9

Neurosymbolic AI and its Taxonomy: a Survey

Figure 3: Disparity impact extracted with SHAP before and after LTN learning of fairness constraints in
real-world data. Extracted from [40].

Following up on the need to de?ne queries for ML explainability, in [39] the authors propose the development of a
declarative language to specify explainability queries, named FOIL. The computational complexity of FOIL queries
over classes of models such as decision trees and general decision diagrams was studied, with the conclusion of this
experiment being that the tractability of the FOIL evaluation was achievable by restricting the structure of the models or
the portion of the FOIL language under study, deemed that such a language as proposed could be used in practice.

6 Applications of neuro-symbolic Systems

Besides the main characteristics of the neuro-symbolic Systems, we argue that is also important to highlight applications
using them. In the LTN main paper [8] the authors show some toy-problems applications with their system. Between
what is shown, there are some interesting applications like multilabel classi?cation, semi-supervised pattern recognition,
learning embeddings, and reasoning.

[40] shows an application of an LTN for fairness in AI. Here, the authors use First Order Logic (FOL) to input fairness
constraints through a group of axioms that imputes the same treatment both to a protected and unprotected group. They
discuss the results regarding the SHAP values [41, 42] on the impact of four observable values on the Demographic
Parity metric. Then, in a second experiment, they show how LTN systems perform better regarding accuracy even with
some fairness constraints on three popular datasets.

Those datasets were the Adult dataset [43] for predicting income, the German [43] for credit risk, and the COMPAS [44]
for recidivism. They used the same experimental setup as [45] but with tuning simpli?cations. A comparison is proposed
between their results, the one proposed in [45] - another neural network-based approach that applies fairness constraints
into the loss function by mean of the Lagrange multipliers - and the one proposed in [46] - a naive-Bayes classi?er -.
They use gender as the protected feature in both the Adult and German datasets and race in the COMPAS [44] dataset.
Their next step was to divide our data into ?ve arbitrary subgroups for querying, these subgroups are not to be confused
with the protected and unprotected groups.

First, the network is trained without fairness constraints and they proceed to query the network to return the truth value
of the LTN predicate used for classi?cation. The output of this query helps to determine, as a proxy of similarity, their
fairness constraints. Those fairness constraints were logical axioms, following LTN philosophy, it being "For all data in
the subgroup equivalent result should derive from our classi?er with no difference if our input is in the protected or
unprotected group". With these logical axioms, one could measure how well our trained model satis?es them, giving us
a better understanding of how our model perpetuates biases.

They then proceeded to use the aforementioned SHAP values to observe the demographic disparity between the
protected and unprotected sections in those sub-groups. The demographic disparity is, in summation, a form of
measuring if our protected and unprotected groups are receiving the same outcome from our model at equal proportions.
With demographic disparity, one could measure if our algorithm, for example, bene?ts men over women when deciding
if one should get a loan.

In their paper [40], they use a classical example of the COMPAS [44] dataset. They compare the demographic disparity
by calculating their SHAP values. Without anti-bias axioms, the trained model exhibits, as can be seen in ?gure 3,
a relevant disparity between the protected and unprotected groups. Finally, they applied fairness constraints using
the same axiom mentioned before by incorporating it into the network training axioms. This resulted in a signi?cant
reduction in demographic disparity as can also be seen in ?gure 3.

The NS-CL was applied, as stated earlier, in the CLEVR [23] dataset. They claim that by applying their system to
this dataset the NS-CL learns visual concepts with remarkable accuracy, allows data-ef?cient visual reasoning, and
generalizes well to new attributes, visual composition, and language domains.

10

Neurosymbolic AI and its Taxonomy: a Survey

They introduce a video reasoning benchmark for the complex temporal and causal structure behind the interacting
objects, drawing inspiration from developmental psychology. The limitations of various current visual reasoning models
on the benchmark are evaluated too.

The CLEVR [23] dataset consists of videos of shapes colliding or static. The authors use their framework (NS-CL) to
answer questions concerning which shapes are present, which color they are, if they collide or not, etc... The article
shows that this neuro-symbolic approach outperforms all other state-of-the-art methods by a considerable margin.

When comparing models in the generalizing to new visual compositions the NS-CL outperforms all baseline models that
do not use annotations and achieves comparable results with models trained by full program annotations. Comparing
now in the generalizing to new visual concepts the NS-CL also outperforms the baseline models.

The authors then extend their testing into other program domains by conducting experiments on MS-COCO ?? images.
Their models do not outperform baseline models but show competitive performance on QA (Question Answering)
accuracy. Beyond question answering the authors conclude that the NS-CL effectively learns visual concepts from data.

Another possible application of neurosymbolic Computing is using it in combination with Graph Neural Networks
(GNN), as proposed by [47]. Attention is brought to the fact that the authors only suggested this system in theory
and argue that such a system, to the best of their knowledge, does not exist yet. The authors bring up attention as an
example of how these two concepts can be connected. The core building blocks of GNN are graph convolution operation,
which enables one to perform learning over graph inputs. As stated in [48] graph convolutions, with slight variations,
can be seen as an attention mechanism. Neurosymbolic systems, such as Pointer Networks [49] implement attention
mechanisms over their inputs.

The author further exercises this idea with an example related to biology and neurosymbolic Computing. Graphs are
natural representations of proteins, among other molecules. In [50] the authors have conceived the ?rst IA-created
antibiotic ("halcin") by training a GNN to "predict the probability that a given input molecule has a growth inhibition effect on
the bacterium E. coli and using it to rank randomly-generated molecules" [47]. The use of symbolic knowledge and restrictions
could surely decrease the number of combinations as simulated scenarios in this context.

Neuro-symbolic AI, more precisely LTNs, can also be used for bearing fault diagnostics with constant shaft speed [51].
The authors incorporated knowledge into deep learning in this case by proposing a new scoring function based on
physical knowledge for classifying bearing faults. This scoring function later acts as domain expertise for the LTN loss
function which would be later injected into the deep learning model. The authors show that their approach outperforms
the baseline methods in test accuracy.

The authors used weighted axioms to input knowledge into a neural network (NN) classi?er. The parameters of the NN
are optimized to maximize satis?ability. In the article, they also incorporate knowledge with a method other than LTNs
Real Logic by extending the feature space with inputs describing this knowledge as in [52].

Regarding experiments the authors used a pure knowledge-based classi?cation, they did this by diving the number of
correctly identi?ed labels of a class through the number of all labels - these labels being identi?ed using knowledge-
based axioms - of a class. Real Logic in the loss function was also used, they evaluated the performance of LTNs, given
the proposed knowledge base, against pure Deep Learning (DL) approaches with Multilayer Perceptrons (MLP) or
Convolutional Neural Networks (CNN). The study concludes that despite little performance enhancement when looking
at isolated models, a combination of neuro-symbolic and deep learning approaches does result in an accuracy increase.
In summation, the articles conclude that inducting knowledge increased performance, particularly with fewer data
available. They also conclude that a neurosymbolic approach helps with getting a better understanding of the dataset.
The counterpoint to this is that, for them, performance gains of LTNs are not completely intuitive.

[13] show in their work the Generative neurosymbolic Machines (GNM), a probabilistic generative model, an approach
to generate images based on both statistical proprieties inferred over the images� spatial distributions and explicitly
structured entity-based representations. They achieve this in GNM via a two-layer latent hierarchy: the top layer
generates the global distributed latent representation for ?exible density modeling and the bottom layer yields from the
global latent the latent structure map for entity-based and symbolic representations. Figure 4 illustrates the difference
between GNN, Distributed latent variable models (D-LVM), and symbolic latent variable models (S-LVM).

The authors evaluate the quality and properties of the generated images on three datasets (ARROW [53], MNIST-4 [54],
and MNIST-10 [54]) in terms of clarity and scene structure, key factors on the dataset, and the model itself by using three
metrics:

� Scene structure accuracy (S-Acc): they manually classi?ed the 250 generated images per model into success or
failure based on the correctness of the scene structure in the image without considering generation quality

� Discriminability score (D-Steps), they measure how dif?cult it is for a binary classi?er to discriminate the
generated images from the real images. Here, they considered the number of training steps required for the
binary classi?er to reach 90% classi?cation accuracy.

� Log-likelihood (LL) using importance sampling with 100 posterior samples

11

Neurosymbolic AI and its Taxonomy: a Survey

Figure 4: Graphical models of D-LVM, S-LVM, and GNM. zg is the global distributed latent representation,
zs is the symbolic structured representation, and x is an observation. Adapted from [13].

Figure 5: Datasets and generation examples. MNIST-4 (left), MNIST-10 (middle), and Arrow room (right).
Extracted from [13].

The model achieves almost perfect accuracy for ARROW [53] and MNIST-4 [54] (some generated images may be
seen in ?gure 5) and GNM samples are signi?cantly more dif?cult for a discriminator to distinguish from the real
images than those generated by the baselines. Also, GNMs can be used to generate novel images by controlling an
object�s structured representation, such as the position, independently of other components or by traversing the globally
distributed representation and generating images, which also makes the generated objects re?ect the correlation between
components.

LTNs are also used in [28] for Semantic Image Interpretation in two important tasks: the classi?cation of bounding
boxes, and the detection of the part-of relation between any two bounding boxes. Both tasks are evaluated using
the PASCAL-PART dataset [55]. In [56] a model is created allowing reincorporating some ideas of classic AI into a
framework of neurosymbolic intelligence using multimodal contextual modeling of interactive situations, events, and
object properties. They discuss how situated grounding provides diverse data and multiple levels of modeling for a

12

Neurosymbolic AI and its Taxonomy: a Survey

Figure 6: An architecture overview for LOA.

variety of AI learning challenges, including learning how to interact with object affordances, learning semantics for
novel structures and con?gurations, and transferring such learned knowledge to new objects and situations.

Another noteworthy application of a neurosymbolic framework is the LOA (Logical Optimal Actions) [57] an architecture
of action decision-making based on reinforcement learning (RL) and neurosymbolic AI. It uses LNNs (Logic Neural
Networks) to bridge natural language and interaction with a set of games.

As we discussed earlier, LNNs can train the constraints and rules with logical functions in a neural network and,
because every neuron in the network has a component for a formula weighted real valued logic, it is able to compute
the probability and contradiction loss for each given proposition. As a neuro-symbolic framework LNNs also follow
symbolic rules, meaning that they yield interpratable and explanable representations. Figure 6 shows an overview
architecture for LOA.

Using those tools the authors propose a neuro-symbolic Reinforcement Learning (RL) method that uses external
knowledge given prior in logical networks. They use a text-based game in order to test their new framework. The
text-base game learning environment used is called TextWorld [58]. They use it as a small-scale example of a natural
language-based interactive environment. The demonstration provided the authors is a web-based user interface for
visualizing the game�s interaction with an display for the natural text observation from the environment, typing the
action sentence and showing the reward value from the each taken action. The LOA in this demonstration also takes
advantage of visualizing trained and pre-de?ned logical rules in the LNN via the same interface. The authors argue that
this will help the human user understand the bene?ts of introducing the logical rules via neuro-symbolic frameworks.
The LOA model initially receives logic state value as logical fact from the language understanding component. This
component, by turn, receives raw natural language state value from the environment. The next step for the model is the
LNN. In order for the input to get optimal action for it, the action goes into the environment to execute the commanded
action and is later given an reward. This reward is inputed to the LOA agent which will be trained by the action decision
network (LNN) using the acquired reward value and chosen action from the network.

The LOA demonstration proposed by the authors supports two functionalities: playing the text-based game through
human interactions and visualizing the trained and pre-de?ned LNN to increase its interpretability for rule-acquiring.
We can choose the game from some existing text-based interaction games such as the TextWorld Coin-Collector [58].
In these types of games a human player can input any action by natural language then the demonstration framework
displays the raw observation output from the given environment.

The LNN contains simple rules for the TextWorld Coin-Collector game. For example, the rule is that the player should
take the �go west� action, after he ?nds the east room ("found west" then "go west") in case of a more complex LNN that
supposes no repetition. As the authors themselves put: "The round box explains the proposition from the given observation
inputs, the circle with a logical function means a logical function node of LNN, and the rectangle box explains an action candidate
for the agent. The highlighted nodes (red node) have �true� value, and non-highlighted nodes (white node) have �false� value. [...], the
agent found the north exit from the given observation (�Observation (t=1)�) by using semantic parser 2, then the going north room
action (�go north�) is activated. [...], if the user clicks the selectable box, the LOA recommends only one action which is �go north�..."

13

Neurosymbolic AI and its Taxonomy: a Survey

Figure 7: (a) Question with 2 mentions that shall be disambiguated against DBpedia. (b) For each mentioned
candidate entity pair, the character-level Jaccard similarity along with the in-degree of the entity in the KG is
shown. (c) (Partial) Ego networks for entities RoderickCameron and JamesCameron. Adapted from [59].

After the LNN selected the "go north" action, the player found two doors: east and south doors. However, the south door
is connected to the previous room because the player took going north as an action in the previous step. In this case, the
authors are using a simple LNN, thus the "go south" action is also recommended. However, the complex LNN, which
has the functionality of avoiding already visited rooms, happens due to the contradiction loss in this LNN setting. This
is one of the bene?ts of introducing neurosymbolic frameworks that human users can easily understand. The authors
show in this demonstration that introducing the LNN into an RL agent brings signi?cant bene?ts such as converging
faster than other nonsymbolic and neurosymbolic methods while also adding more explainability to the process.

For yet another application of neurosymbolic frameworks, we can highlight the LNN-EL which is a neurosymbolic
approach to short-text entity linking using Logic Neural Networks (LNN-EL) [59]. In this system, the authors use
LNNs to perform short-text entity linking. Entity linking (EL) is the task related to disambiguating textual mentions
by linking them and canonical entities provided by a knowledge graph (KG) such as DBpedia. The particular type of
EL discussed in this article (short text) has attracted attention due to its relevance for downstream applications such
as question-answering, conversational systems, etc... The authors argue that short-text EL is particularly challenging
because of the limited context surrounding mentions, thus resulting in greater ambiguity.

The authors exemplify the task of short-text entity linking with the question in Figure 7. The question contains mention1
(Cameron) and mention2(Titanic) but DBpedia contains several person entities whose last name matches Cameron as can
be seen in Figure 7(b). On one hand, given the higher in-degree - which is a result of using the more popular candidate
entity in the KG, one can link mention1, correctly, to James Cameron. In terms of mention2 on the other hand the correct
entry is Titanic(1997 f ilm) as opposed to Titanic the ship which has higher string similarity. To link to the correct entity,
one needs to exploit the fact that James Cameron has an edge connection with Titanic(1997 f ilm) in the KG as shown in
Figure 7(c). This example is meant to provide intuition as to how priors, local features (string similarity), and collective
entity linking can be exploited to overcome the limited context in short-text EL.

Figure 8 shows an overview of the authors� neurosymbolic approach for short-text EL. Given the input text T, together
with the labeled data in the form (mi,Ci,Li), where mi is a mention in T, Ci is a list of candidate entities ei j for the mention
mi and where each li j denotes a link or not-link label for the pair (mi,ei j). The authors ?rst generate a set Fi j of features
for each pair (mi,ei j). After the feature generation, the newly labeled data with features is inputted into the LNN.

In this neurosymbolic approach to short-text EL, the EL rules are a restricted form of Fist Order Logic comprising a set of
Boolean predicates connected via logical operators. The authors use the LNN logical approach to evaluate if a given
mention is related to an entity. Rules can be disjuncted together to form a larger EL algorithm, thus creating a function
(e.g.: Links(mi,eei j)). By using FOL the authors create an EL algorithm that can be easily understood and manipulated by
users. However, the authors highlight that to obtain competitive performance against the best deep learning approaches,
this system requires a signi?cant amount of manual effort to ?ne-tune certain aspects.

Although the pure LNN model underperforms compared to black-box deep learning methods it is competitive and
outperforms other logic-based approaches. Furthermore, LNN-ELens, which combines the core LNN-EL with deep
learning approaches, easily beats those same deep learning approaches in almost all datasets. The authors argue that
the leverage this model has over other models, these being deep learning or other techniques, is its explainability,
interpretability, and transferability.

They test the model transferability by training the model on one dataset and evaluating it on the other two datasets. They
observe that the model transfers reasonably well even if the training is done on a very small dataset. Benchmark deep
learning models outperform the LNN-EL in transferability but the LNN-EL needs signi?cantly less data for training to
achieve reasonable performance.

14

Neurosymbolic AI and its Taxonomy: a Survey

Figure 8: Overview of the authors� approach. Extracted from [59].

Our last application also makes use of LNNs. In this article, [14] the authors use LNNs as a reasoner for knowledge base
question answering (KBQA). Knowledge base question answering - a sub-?eld of question answering as a whole - is
a very important task in natural language processing (NLP). KBQA demands that a system answer natural language
questions based on facts available in a knowledge base (KB). This system retrieves facts from a KB through structured
queries, these queries often contain multiple triples that represent the steps or antecedents required for obtaining the
answer, thus enabling a transparent and self-explanatory form of QA.

In the ?eld of KBQA, neural networks are the most common approach to this type of problem. The authors propose
neuro-symbolic Question Answering (NSQA), a modular knowledge base question answering system that would:
a) delegate the complexity of understanding natural language questions to AMR (Abstract Meaning Representation)
parsers [60, 61] - which is one of the benchmark parsers types for semantic representation; b) reduce the need for
end-to-end training data using a pipeline architecture where each module is trained for its speci?c task; c) facilitate the
use of an independent reasoner via an intermediate logic form (LNNs).

Suppose that a question, given in natural language, is input into the system; it ?rstly parsers the question into an abstract
meaning representation (AMR) graph, then transforms this graph into a set of candidate KB-aligned logical queries; the
system then uses an LNN to reason over KB facts, producing answers to KB-aligned logical queries. In this survey, the
actual parsing or the AMR to KG Logic process is not explored. However, it focuses on the LNN reasoner.

Given the KB-aligned logical queries, the NSQA uses LNNs as a ?rst-order logic neuro-symbolic reasoner. The system
uses LNNs type-based reasoning to eliminate queries based on inconsistencies making use of the type hierarchy in the
KB. The system also uses the LNNs geographic reasoning to answer questions such as "Was Cartola born in Salvador?",
this happens because the entities related to dbo : birthplace are generally cities, but the question requires a comparison of
countries. The authors address this manually by adding logic axioms to perform the required transitive reasoning for
dbo : birthplace.

For their experimental evaluation, the authors use 4 datasets. They ?rst evaluate NSQA against four systems, two being
graph-driven approaches, one being a KB agnostic approach and the last being an ensemble of entity and relation linking
modules and train a Tree-LTSM model for query ranking. The NSQA system outperforms three of the four systems in
two of the four datasets.

This survey shall highlight the section in which the authors analyze the LNN reasoner results.

The authors evaluate the performance of NSQAs under a LNN reasoner and a deterministic translation of query graphs
to SPARQL. They tested it in one dataset and the LNN approach outperforms the deterministic translation by a 2.9%
margin. Table 3 shows the type of question the NSQA system is capable of answering.

7 Conclusion

Neurosymbolic Arti?cial Intelligence is a prominent approach that combines learning over data distribution and
reasoning on prior and learned knowledge. The present survey compiles the most recent works on neurosymbolic AI and

15

Neurosymbolic AI and its Taxonomy: a Survey

Question Type/Reasoning Example
Simple
Multi-Relational
Count-based
Superlative
Comparative
Geographic
Temporal

Who is the president of Brazil
Give me all actors starring in movies directed by Glauber Rocha
How many books did Machado de Assis write?
What is the highest mountain in Chile?
Does A Grande Familia have more episodes than Grey�s Anatomy?
Was Cartola born in Argentina?
When will start the ?nal match of the football world cup 2022

Supported
(cid:88)
(cid:88)
(cid:88)
(cid:88)

(cid:88)

Table 3: Question NSQA system is capable of answering.

their applications. Also, it provides a taxonomy of approaches on each aspect of the area, like Knowledge Representation,
Learning and Reasoning, as well as a brief historical introduction.

About 14 papers were reviewed with a focus on building systems with prior knowledge - or premises - that would
signi?cantly reduce the amount of data needed to achieve good performance concerning traditional metrics like accuracy.
Table 4 shows a taxonomic comparison between the surveyed models.

In order to further enrich the discussion and ?eld of neurosymbolic AI we will now highlight blindspots we found con-
cerning the articles we read. This highlighted points could be a jumping point for future work conserning neurosymbolic
AI:

� Ability to save models in LTNs: When creating a model sometimes it can takes hours or even days for it to

run completely and in LTNs there is no feature to save this model for further use without re-training.

� Explanability: Although the models try (and succeed mostly) in being explanable this explanation is mostly

aimed at the scientist or specialist, thus not being user-friendly.

� Comparison: When comparing models we felt a lack of more robust comparisons between neurosymbolic

approaches and non-logic/symbolic approaches such as deep learning, decision trees and so on.

� Fairness and algorithmic bias: In our tests and read articles we came across the use of neurosymbolic AI for
fairness and bias mitigation. We found out they mix very well and there is a lack of artiles exploring this
relation.

While these systems often underperform traditional, pure Deep Learning techniques, the amount of data, computational
costs, and the possibility to have a better understanding and guidance over the system justify further research in this
direction as an alternative to evergrowing DL models.

Acknowledgments

The authors are grateful to the Eldorado Research Institute.

References

[1] Stuart Russell, Sabine Hauert, Russ Altman, and Manuela Veloso. Ethics of arti?cial intelligence. Nature,

521(7553):415�416, 2015.

[2] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436�444, 2015.

[3] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert,
Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature,
550(7676):354�359, 2017.

[4] Gary Marcus. The next decade in ai: four steps towards robust arti?cial intelligence. arXiv preprint arXiv:2002.06177,

2020.

[5] Daniel Kahneman, Francesca Rossi, Geoffrey Hinton, Yoshua Bengio, and Yann LeCun. Aaai20 ?reside chat with

daniel kahneman, 2020.

[6] Leslie G Valiant. Three problems in computer science. Journal of the ACM (JACM), 50(1):96�99, 2003.

[7] Daniel Kahneman. Thinking, fast and slow. new york: Farrar, straus and giroux, 2011. 2011.

[8] Samy Badreddine, Artur d'Avila Garcez, Luciano Sera?ni, and Michael Spranger. Logic tensor networks. Arti?cial

Intelligence, 303:103649, feb 2022.

[9] Honghua Dong, Jiayuan Mao, Tian Lin, Chong Wang, Lihong Li, and Denny Zhou. Neural logic machines. In

International Conference on Learning Representations, 2018.

16

Neurosymbolic AI and its Taxonomy: a Survey

Table 4: Comparison between the models a

Model

Knowledge
Representation

Learning

Reasoning

Explainabilty Applications

LTN [8]

Tensorisation

Vertical Hybrid
Learning

Approximate
Sat

Knowledge
Extraction

NLM [9]

Tensorisation

Inductive Logic
Programming

Forward
Chaining

-

LBM [10]

Propositionalization

Inductive Logic
Programming

Approximate
Sat

Knowledge
Extraction

Fairness
Constrains [40],
Semantic Image
Interpretation [28] ,
Situated Language
Understanding [56],
Neural-Symbolic
Reasoning[22]

Family Tree
Reasoning,
Block World, sorting,
path ?ding [9]

Satisfying set
search,
Classi?cation [10]

LNN [11]

Tensorisation

Vertical Hybrid
Learning

Forward/Backwards
Chaining

Knowledge
Extraction

LOA [57],
EL [59], KBQA [14]

NS-CL [12] Propositionalization

GNM [13]

Tensorisation

Vertical Hybrid
Learning
Evidence
Lower Bound

Relationship
Reasoning
Approximate
Sat

-

-

CLEVRER [12]

Image Generation [13]

[10] Son N Tran and Artur d�Avila Garcez. Logical boltzmann machines. arXiv preprint arXiv:2112.05841, 2021.

[11] Ryan Riegel, Alexander G Gray, Francois PS Luus, Naweed Khan, Ndivhuwo Makondo, Ismail Yunus Akhalwaya,

Haifeng Qian, Ronald Fagin, Francisco Barahona, Udit Sharma, et al. Logical neural networks. 2020.

[12] Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B Tenenbaum, and Jiajun Wu. The neuro-symbolic concept
learner: Interpreting scenes, words, and sentences from natural supervision. In International Conference on Learning
Representations. International Conference on Learning Representations, ICLR, 2019.

[13] Jindong Jiang and Sungjin Ahn. Generative neurosymbolic machines. Advances in Neural Information Processing

Systems, 33:12572�12582, 2020.

[14] Pavan Kapanipathi, Ibrahim Abdelaziz, Srinivas Ravishankar, Salim Roukos, Alexander Gray, Ram�n Fernandez
Astudillo, Maria Chang, Cristina Cornelio, Saswati Dana, Achille Fokoue, et al. Leveraging abstract meaning
representation for knowledge base question answering. In Annual Meeting of the Association for Computational
Linguistics, 2021.

[15] Allen Newell and Herbert A Simon. Computer science as empirical inquiry: Symbols and search. In ACM Turing

award lectures, page 1975. 2007.

[16] A Garcez, M Gori, LC Lamb, L Sera?ni, M Spranger, and SN Tran. Neural-symbolic computing: An effective
methodology for principled integration of machine learning and reasoning. Journal of Applied Logics, 6(4):611�632,
2019.

[17] Luc De Raedt, Sebastijan Dumancic, Robin Manhaeve, and Giuseppe Marra. From statistical relational to neuro-
symbolic arti?cial intelligence. In Proceedings of the Twenty-Ninth International Joint Conference on Arti?cial Intelligence,
IJCAI 2020, pages 4943�4950. ijcai. org, 2020.

[18] Lu�s C Lamb, Artur d�Avila Garcez, Marco Gori, Marcelo OR Prates, Pedro HC Avelar, and Moshe Y Vardi. Graph
neural networks meet neural-symbolic computing: a survey and perspective. In Proceedings of the Twenty-Ninth
International Conference on International Joint Conferences on Arti?cial Intelligence, pages 4877�4884, 2021.

[19] AD Garcez, M Gori, LC Lamb, L Sera?ni, M Spranger, and SN Tran. Neural-symbolic computing: An effective
methodology for principled integration of machine learning and reasoning. Journal of Applied Logics, 6(4):611�631,
2019.

[20] Hector J Levesque. Knowledge representation and reasoning. Annual review of computer science, 1(1):255�287, 1986.

[21] Luciano Sera?ni and Artur S. d�Avila Garcez. Logic tensor networks: Deep learning and logical reasoning from

data and knowledge. CoRR, abs/1606.04422, 2016.

[22] Benedikt Wagner and AS d�Avila Garcez. Neural-symbolic reasoning under open-world and closed-world assump-

tions. In CEUR Workshop Proceedings, volume 3121. CEUR, 2022.

17

Neurosymbolic AI and its Taxonomy: a Survey

[23] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick.
Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE
conference on computer vision and pattern recognition, pages 2901�2910, 2017.

[24] Fan Yang, Zhilin Yang, and William W Cohen. Differentiable learning of logical rules for knowledge base reasoning.

Advances in neural information processing systems, 30, 2017.

[25] Jack Minker. On inde?nite databases and the closed world assumption. In International Conference on Automated

Deduction, pages 292�308. Springer, 1982.

[26] Artur Garcez and Gerson Zaverucha. The connectionist inductive learning and logic programming system. Appl.

Intell., 11:59�77, 07 1999.

[27] Kalanit Grill-Spector and Rafael Malach. The human visual cortex. Annual review of neuroscience, 27:649�77, 02 2004.

[28] Ivan Donadello, Luciano Sera?ni, and Artur D�Avila Garcez. Logic tensor networks for semantic image interpreta-

tion. In Proceedings of the 26th International Joint Conference on Arti?cial Intelligence, pages 1596�1602, 2017.

[29] Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 1440�1448, 2015.

[30] Li Dong and Mirella Lapata. Language to logical form with neural attention. In Proceedings of the 54th Annual

Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 33�43, 2016.

[31] Kyunghyun Cho, Bart van Merrienboer, �aglar G�l�ehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk,
and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation.
In EMNLP, 2014.

[32] Benedikt Wagner and AS d�Avila Garcez. Neural-symbolic reasoning under open-world and closed-world assump-

tions. In CEUR Workshop Proceedings, volume 3121. CEUR, 2022.

[33] Robert Andrews, Joachim Diederich, and Alan Tickle. Survey and critique of techniques for extracting rules from

trained arti?cial neural networks. Knowledge-Based Systems, 6:373�389, 12 1995.

[34] Henrik Jacobsson. Rule extraction from recurrent neural networks: A taxonomy and review. Neural Computation,

17:1223�1263, 06 2005.

[35] Qinglong Wang, Kaixuan Zhang, Alexander G. Ororbia II, Xinyu Xing, Xue Liu, and C. Lee Giles. An empirical

evaluation of rule extraction from recurrent neural networks. Neural Computation, 30(9):2568�2591, 2018.

[36] Artur S. d�Avila Garcez, Krysia Broda, and Dov M. Gabbay. Neural-symbolic learning systems - foundations and

applications. 2002.

[37] Antoine Bordes, Jason Weston, Ronan Collobert, and Yoshua Bengio. Learning structured embeddings of knowledge
bases. In Proceedings of the Twenty-Fifth AAAI Conference on Arti?cial Intelligence, AAAI�11, page 301�306. AAAI
Press, 2011.

[38] Emilio Parisotto, Abdel-rahman Mohamed, Rishabh Singh, Lihong Li, Dengyong Zhou, and Pushmeet Kohli.

Neuro-symbolic program synthesis. In ICLR (Poster), 2017.

[39] Marcelo Arenas, Daniel Baez, Pablo Barcel�, Jorge P�rez, and Bernardo Subercaseaux. Foundations of symbolic
languages for model interpretability. Advances in Neural Information Processing Systems, 34:11690�11701, 2021.

[40] Benedikt Wagner and AS d�Avila Garcez. Neural-symbolic integration for fairness in ai.

In CEUR Workshop

Proceedings, volume 2846, 2021.

[41] Erik �trumbelj and Igor Kononenko. Explaining prediction models and individual predictions with feature

contributions. Knowledge and information systems, 41(3):647�665, 2014.

[42] LS Shapley. A value for n-person games: contributions to the theory of games (am 28), volume ii, 1953.

[43] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017.

[44] Jeff Larson, Julia Angwin, Lauren Kirchner, and Surya Mattu. How we analyzed the compas recidivism algorithm,

May 2016.

[45] Manisha Padala and Sujit Gujar. Fnnc: Achieving fairness through neural networks. In Christian Bessiere, editor,
Proceedings of the Twenty-Ninth International Joint Conference on Arti?cial Intelligence, IJCAI-20, pages 2277�2283.
International Joint Conferences on Arti?cial Intelligence Organization, 7 2020. Main track.

[46] YooJung Choi, Golnoosh Farnadi, Behrouz Babaki, and Guy Van den Broeck. Learning fair naive bayes classi?ers
by discovering and eliminating discrimination patterns. In Proceedings of the AAAI Conference on Arti?cial Intelligence,
volume 34, pages 10077�10084, 2020.

[47] Lu�s C Lamb, Artur d�Avila Garcez, Marco Gori, Marcelo OR Prates, Pedro HC Avelar, and Moshe Y Vardi. Graph
neural networks meet neural-symbolic computing: a survey and perspective. In Proceedings of the Twenty-Ninth
International Conference on International Joint Conferences on Arti?cial Intelligence, pages 4877�4884, 2021.

[48] Victor Garcia and Joan Bruna. Few-shot learning with graph neural networks. In 6th International Conference on

Learning Representations, ICLR 2018, 2018.

18

Neurosymbolic AI and its Taxonomy: a Survey

[49] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. Advances in neural information processing

systems, 28, 2015.

[50] Jonathan M. Stokes, Kevin Yang, Kyle Swanson, Wengong Jin, Andres Cubillos-Ruiz, Nina M. Donghia, Craig R.
MacNair, Shawn French, Lindsey A. Carfrae, Zohar Bloom-Ackermann, Victoria M. Tran, Anush Chiappino-Pepe,
Ahmed H. Badran, Ian W. Andrews, Emma J. Chory, George M. Church, Eric D. Brown, Tommi S. Jaakkola, Regina
Barzilay, and James J. Collins. A deep learning approach to antibiotic discovery. Cell, 180(4):688�702.e13, 2020.

[51] Dhiraj Neupane and Jong-Hoon Seok. Bearing fault detection and diagnosis using case western reserve university

dataset with deep learning approaches: A review. IEEE Access, 8:93155�93178, 2020.

[52] Manuel Arias Chao, Chetan Kulkarni, Kai Goebel, and Olga Fink. Fusing physics-based and deep learning models

for prognostics. Reliability Engineering and System Safety, 217(C), 2022.

[53] Apache. Apache arrow: A cross-language development platform for in-memory data, 2019.

[54] Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing

Magazine, 29(6):141�142, 2012.

[55] Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan Yuille. Detect what you can:
Detecting and representing objects using holistic models and body parts. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 1971�1978, 2014.

[56] Nikhil Krishnaswamy and James Pustejovsky. Neurosymbolic ai for situated language understanding, 2020.

[57] Daiki Kimura, Subhajit Chaudhury, Masaki Ono, Michiaki Tatsubori, Don Joven Agravante, Asim Munawar,
Akifumi Wachi, Ryosuke Kohita, and Alexander Gray. Loa: Logical optimal actions for text-based interaction games.
In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint
Conference on Natural Language Processing: System Demonstrations, pages 227�231, 2021.

[58] Marc-Alexandre C�t�, �kos K�d�r, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew
Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. Textworld: A learning environment
for text-based games. In Tristan Cazenave, Abdallah Saf?dine, and Nathan Sturtevant, editors, Computer Games,
pages 41�75, Cham, 2019. Springer International Publishing.

[59] Hang Jiang, Sairam Gurajada, Qiuhao Lu, Sumit Neelam, Lucian Popa, Prithviraj Sen, Yunyao Li, and Alexander G

Gray. Lnn-el: A neuro-symbolic approach to short-text entity linking. In ACL/IJCNLP (1), 2021.

[60] Laura Banarescu, Claire Bonial, Shu Cai, Madalina Georgescu, Kira Grif?tt, Ulf Hermjakob, Kevin Knight, Philipp
Koehn, Martha Palmer, and Nathan Schneider. Abstract meaning representation for sembanking. In Proceedings of
the 7th linguistic annotation workshop and interoperability with discourse, pages 178�186, 2013.

[61] Bonnie Dorr, Nizar Habash, and David Traum. A thematic hierarchy for ef?cient generation from lexical-conceptual
structure. In Conference of the Association for Machine Translation in the Americas, pages 333�343. Springer, 1998.

19


