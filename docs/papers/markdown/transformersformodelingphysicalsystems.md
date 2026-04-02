Version of Record: https://www.sciencedirect.com/science/article/pii/S0893608021004500
Manuscript_e417ab0edd3f9111d35a327b4311e0b0

Transformers for Modeling Physical Systems

Nicholas Genevaa, Nicholas Zabarasa,?

aScienti?c Computing and Arti?cial Intelligence (SCAI) Laboratory, University of Notre Dame,
311 Cushing Hall, Notre Dame, IN 46556, USA

Abstract

Transformers are widely used in natural language processing due to their ability to
model longer-term dependencies in text. Although these models achieve state-of-the-
art performance for many language related tasks, their applicability outside of the
natural language processing ?eld has been minimal. In this work, we propose the
use of transformer models for the prediction of dynamical systems representative of
physical phenomena. The use of Koopman based embeddings provide a unique and
powerful method for projecting any dynamical system into a vector representation
which can then be predicted by a transformer model. The proposed model is able
to accurately predict various dynamical systems and outperform classical methods
that are commonly used in the scienti?c machine learning literature.1

Keywords: Transformers, Deep Learning, Self-Attention, Physics, Koopman,
Surrogate Modeling

1. Introduction

The transformer model [1], built on self-attention, has largely become the state-of-
the-art approach for a large set of natural language processing (NLP) tasks including
language modeling, text classi?cation, question answering, etc. Although more recent
transformer work is focused on unsupervised pre-training of extremely large mod-
els [2, 3, 4, 5], the original transformer model garnered attention due to its ability
to out-perform other state-of-the-art methods by learning longer-term dependencies
without recurrent connections. Given that the transformer model was originally de-
veloped for NLP, nearly all related work has been rightfully con?ned within this ?eld
with only a few exceptions. Here, we focus on the development of transformers to

?Corresponding author
1Code available at: https://github.com/zabaras/transformer-physx.

Preprint submitted to Neural Networks

November 8, 2021

� 2021 published by Elsevier. This manuscript is made available under the Elsevier user license
https://www.elsevier.com/open-access/userlicense/1.0/

model dynamical systems that can replace otherwise expensive numerical solvers. In
other words, we are interested in using transformers to learn the language of physics.
The surrogate modeling of physical systems is a research ?eld that has ex-
isted for several decades and is a large ongoing e?ort in scienti?c machine learn-
ing. Past literature has explored multiple surrogate approaches including Gaussian
processes [6, 7, 8], polynomial chaos expansions [9], reduced-order models [10, 11],
reservoir computing [12] and deep neural networks [13, 14, 15]. A surrogate model
is de?ned as a computationally inexpensive approximate model of a physical phe-
nomenon that is designed to replace an expensive computational solver that would
otherwise be needed to resolve the system of interest. An important characteristic
of surrogate models is their ability to model a distribution of initial or boundary
conditions rather than learning just one solution. This is arguably essential for the
justi?cation of training a deep learning model versus using a standard numerical
solver, particularly in the context of utilizing deep learning methods which tend to
have expensive training procedures. The most tangible applications of surrogates are
for optimization, design and inverse problems where many repeated simulations are
typically needed.

Standard deep neural network architectures such as auto-regressive [16, 15], resid-
ual/Euler [17, 18], recurrent and LSTM based models [16, 19, 20, 21] have been
largely demonstrated to be e?ective at modeling various physical dynamics. Such
models generally rely on the most recent time-steps to provide complete information
on the current and past state of the system�s evolution. Approaches that meld nu-
merical time�integration methods with neural networks have also proven to be fairly
successful, e.g. [22, 23, 24], but have a ?xed temporal window from which information
is provided. Present machine learning models lack generalizable time cognizant capa-
bilities to predict multi-time-scale phenomena present in systems including turbulent
?uid ?ow, multi-scale materials modeling, molecular dynamics, chemical processes,
etc. Much work is needed to scale such deep learning models to complex physical
systems that are of scienti?c and industrial interest. This work deviates from this
pre-existing literature by investigating the use of transformers for the prediction of
physical systems, relying entirely on self-attention to surrogate model dynamics. In
the recent work of [25], such self-attention models were tested to learn single solutions
of several low-dimensional ordinary di?erential equations.
The novel contributions of this paper are as follows:

(a) The application of
self-attention transformer models for modeling physical dynamics; (b) The use of
Koopman dynamics for developing physics inspired embeddings of high-dimensional
systems with connections to embedding methods seen in NLP; (c) Discussion of
the relations between self-attention with traditional numerical time-integration; (d)

2

Demonstration of our model on high-dimensional partial di?erential equation prob-
lems that include chaotic dynamics, ?uid ?ows and reaction-di?usion systems. To
the authors best knowledge, this is the ?rst work to explore transformer NLP archi-
tectures for the surrogate modeling of physical systems. The remainder of this paper
is as follows: In Section 2, the machine learning methodology is discussed including
the transformer decoder model in Section 2.1 and the Koopman embedding model
in Section 2.2. Following in Section 3, the proposed model is implemented for a set
of numerical examples of di?erent dynamical nature. This includes classical chaotic
dynamics in Section 3.1, periodic ?uid dynamics in Section 3.2 and three-dimensional
reaction-di?usion dynamics in Section 3.3. Lastly, concluding discussion and future
directions are given in Section 4.

2. Methods

We are interested in systems that can be described through a dynamical ordinary or
partial di?erential equation of the form:

?t = F (cid:0)x, ?(t, x, ?), ?x?, ?2

x?, ? � ?x?, . . .(cid:1) ,

t ? T ? R+, x ? ? ? Rm,

(1)

in which ? ? Rn is the solution of this di?erential equation of n state variables
with parameters ?, in the time interval T and spatial domain ? with a boundary
? ? ?. This general form can embody a vast spectrum of physical phenomena
including ?uid ?ow and transport processes, mechanics and materials physics, and
molecular dynamics. In this work, we are interested in learning the set of solutions
for a distribution of initial conditions ?0 ? p(?0), boundary conditions B(?) ?
p (B) ?x ? ? or equation parameters ? ? p (?). This accounts for modeling initial
value, boundary value and stochastic problems.

To make the modeling of such dynamical systems applicable to the use of modern
machine learning architectures, the continuous solution is discretized in both the
spatial and temporal domains such that the solution of the di?erential equation is
? = (?0, ?1, . . . ?T ) ; ?i ? Rn�d, for which ?i has been discretized by d points in
?. We assume an initial state ?0 and that the time interval T is discretized by
T time-steps with a time-step size ?t. Hence, we pose the modeling a dynamical
system as a time-series problem. The proposed machine learning methodology has
two core components: the transformer for modeling dynamics and the embedding
network for projecting physical states into a vector representation. Similar to NLP,
the embedding model is trained prior to the transformer. This embedding model is
then frozen and the entire data-set is converted to the embedded space in which the

3

transformer is then trained as illustrated in Fig. 1. During testing, the embedding
decoder is used to reconstruct the physical states from the transformer�s predictions.

Figure 1: The two training stages for modeling physical dynamics using transformers. (Left to right)
The embedding model is ?rst trained using Koopman based dynamics. The embedding model is
then frozen (?xed), all training data is embedded and the transformer is trained in the embedded
space.

2.1. Transformer
The transformer model was originally designed with NLP as the sole application with
word vector embeddings of a passage of text being the primary input [1]. However,
recent works have explored using attention mechanisms for di?erent machine learning
tasks [26, 27, 28] and a few investigate the use of transformers for applications outside
of the NLP ?eld [29]. This suggests that self-attention and in particular transformer
models may work well for any problem that can be posed as a sequence of vectors.

2.1.1. Transformer Decoder
In this work, the primary input to the transformer will be an embedded dynamical
system, ? = (?0, ?1, . . . .?T ), where the embedded state at time-step i is denoted as
?i ? Re. Given that we are interested in the prediction of a physical time series,
this motivates the usage of a language modeling architecture that is designed for the
sequential prediction of words in a body of text. We select the transformer decoder
architecture used in the Generative Pre-trained Transformer (GPT) models [30, 3].
Our model follows the GPT-2 architecture based on the implementation in the Hug-
ging Face transformer repository [31], but is signi?cantly smaller in size than these
modern NLP transformers. This model consists of a stack of transformer decoder
layers that use masked attention, as depicted in Fig. 2. The input to the transformer
is the embedded representation of the physical system from the embedding model and
a sinusoidal positional encoding proposed in the original transformer [1]. Since the

4

Figure 2: The transformer decoder model used for the prediction of physical dynamics.

transformer does not contain any recurrent or convolutional operation, information
regarding the relative position of the embedded input sequence must be provided.
The positional embedding is de?ned by the following sine and cosine functions:

P Epos,2j = sin (cid:0)pos/100002j/k(cid:1) , P Epos,2j+1 = cos (cid:0)pos/100002j/k(cid:1) ,

(2)

for the 2j and 2j + 1 elements in the embedded vector, ?i. pos is the embedded
vector�s relative or global position in the input time series and k is the transformer�s
context window. To train the model, consider a data set of D embedded i.i.d. time-
series D = {?i}D
i=1 for which we can use the standard time-series Markov model
(language modeling) log-likelihood:

LD =

D
(cid:88)

T
(cid:88)

i

j

? log p (cid:0)?i

j|?i

j?k, . . . , ?i

j?1, ?(cid:1) ,

(3)

where ? are the model�s parameters. Contrary to the standard NLP approach which
poses the likelihood as a softmax over a dictionary of tokens, the likelihood here is
taken as a standard Gaussian between the transformer�s prediction and the target
embedded value resulting in a L2 loss. This is due to the fact that the solution to
most physical systems cannot be condensed to a discrete ?nite set. Thus tokenization
into a ?nite dictionary not possible and a softmax approach not applicable. Training
is the standard auto-regressive method used in GPT [30], as opposed to the word
masking [2], constrained to the embedded space. The physical states, ?i, have the
potential to be very high-dimensional and training the transformer in the lower-
dimensional embedded space can signi?cantly lower training costs.

2.1.2. Self-Attention
Self-attention is the main mechanism that allows the transformer to learn temporal
dependencies. Prior to the seminal transformer paper, several works had already

5

proposed a handful of di?erent attention models typically integrated into recurrent
models for language processing [32, 33, 34]. With the popularization of the trans-
former model [1], the most commonly used attention model is the scaled-dot product
attention:

ki = Fk(xi),

qi = Fq(xi),

vi = Fv(xi)

cn =

k
(cid:88)

i=1

?n,ivi,

?n,i =

exp(qT
n ki/
j=1 exp(qT

(cid:80)k

?

dk)
?

n kj/

dk)

(4)

(5)

,

in which we use xi ? Rd and ci ? Rdv to denote an arbitrary input and context
output, respectively. k ? Rdk, q ? Rdk and v ? Rdv are referred to as the key, query,
and value vectors, respectively, calculated using neural networks Fk, Fq and Fv. The
attention score, ?n,i, is calculated by the soft-max of the dot product between the
query and key vectors scaled by the dimension dk. Due to the soft-max calculation
note that the attention scores always sum to one, (cid:80)k

i=1 ?n,i = 1, for every input.

?

dk

The attention calculation can be condensed for the entire context length of the
model by a matrix representation C = sof tmax (cid:0)QK T /
(cid:1) V , where Q ? Rk�dk,
K ? Rk�dk and V ? Rk�dv . As illustrated in Fig. 2, self-attention is typically imple-
mented with a residual connection with multiple independent attention calculations
referred to as attention heads [1]. While computationally inexpensive to evaluate,
the memory requirement of scaled-dot product attention can become increasingly
cumbersome as the context length, k, increases. Hence, several methods for approx-
imating this calculation have been proposed which include the use of kernels and
random projections in an attempt to lower the dimensionality of the self-attention
calculation without loss of predictive accuracy [35, 36, 37].

While designed for natural language processing, in the context of dynamical sys-
tems, self-attention bears a very similar form to numerical time-integration meth-
ods. The use of time-integration methods such as Runge-Kutta schemes or Euler
methods can be found in Neural ODEs [38, 39] and Res-Net based models [17, 18].
Self-attention o?ers a much larger learning capacity than using these traditional nu-
merical methods for learning a unique latent time-integration method. The function
space of a self-attention layer with a residual connection contains the space of ex-
plicit linear time-integration methods within an arbitrarily small non-zero amount
of error.

Theorem 1. Consider a dynamical system of the form, d?/dt = f (t, ?), ?(t) ? Rd,
where f : R � Rd ? Rd is Lipschitz continuous with respect to ? and continuous in
t. Let the function A? (t, ?t?k?t:t) = �?t+?t be a self-attention layer with a residual

6

connection, of context length k and containing learnable parameters ? ? Rd?.Suppose
A := (cid:8)A? (t, ?) | ?? ? Rd?(cid:9) be the set of all possible self-attention calculations. Let
(cid:1) = ?t+?t be the ith order explicit Adams method time-integrator.
Mi
The set of functions up to kth order M := {Mi | 1 ? i ? k} is a subset of A such
that ? A?i ? A s.t. ||Mi ? A?i||? < O((cid:15)) for any Mi ? M and (cid:15) > 0.

(cid:0)t, ?t?(i?1)?t:t

Proof. The state variables are discretized w.r.t. time such that ?i = ?(i?t); ti =
i?t. The general de?nition for linear multi-step methods follows:

?n+s + as?1 � ?n+s?1 + ... + a0 � ?n

= ?t � (bs � f (tn+s, ?n+s) + bs?1 � f (tn+s?1, ?n+s?1) + ... + b0 � f (tn, ?n)) ,

(6)

in which ai and bi are coe?cients determined by the integration method. For explicit
Adams methods, the ?n+s is directly computed with bs = 0, as?1 = ?1 and a0:s?2 =
0. The generalized formula for s-step explicit Adams methods can be represented as
the following linear combination:

Ms = ?n+s = ?n+s?1 + ?t

s?1
(cid:88)

j=0

bjf (tn+j, ?n+j) ,

(7)

which encapsulates up to and including sth order time integration [40]. Consider
a residual scaled dot-product self-attention calculation with context length, s, for
output prediction �?n+s, input states ?n:n+s?1 and time t:

A?i = �?n+s = ?n+s?1 +

s?1
(cid:88)

i=0

?n+s?1,i vi, ?n+s?1,i =

n+s?1ki/

exp(qT
j=0 exp(qT

(cid:80)s?1

n+s?1kj/

dk)

?

dk)
?

, (8)

for which ki = Fk(tn+i, ?n+i), qi = Fq(tn+i, ?n+i) and vi = Fv(tn+i, ?n+i) vectors
are outputs of di?erentiable functions parameterized by neural networks:

Fk : R � Rd ? Rdk, Fq : R � Rd ? Rdk, Fv : R � Rd ? Rdv .

(9)

It su?ces to show that there exists a form of Eq. (8) that is equal to Eq. (7) within
a error order O((cid:15)) such that

(cid:12)
(cid:12)
(cid:12)

(cid:12)
(cid:12)?n+s ? �?n+s
(cid:12)

(cid:12)
(cid:12)
(cid:12)

(cid:12)
(cid:12)
(cid:12)?

< O((cid:15)).

(10)

To this end, although Fq and Fk are fully-trainable neural networks, herein we con-
sider a ?xed single-entry query and key vectors resulting in the following simpli?ed

7

self-attention scores:

qi = Fq(tn+i, ?n+i) =

(cid:112)

dk em, ki = Fk(tn+i, ?n+i) = log(bi) em,

(cid:80)s?1

?n+s?1,i =

exp(log(bi))
j=0 exp(log(bj))
where em ? Rdk is a unit vector with all elements set to zero except the m-th element
that is set to 1. By the universal approximation theorem [41, 42, 43], the neural
network, Fv, is assumed to be of su?cient capacity such that it can approximate the
r.h.s. of the governing equation:

bi
(cid:80)s?1
j=0 bj

(11)

=

,

vi = Fv(tn+i, ?n+i) = c f (tn+i, ?n+i) + O((cid:15)),

(cid:15) > 0,

(12)

where the constant c is taken as c = ?t (cid:80)s?1
be combined leading to the following:

j=0 bj. Equations (11) and (12) can then

?n+s?1 +

s?1
(cid:88)

i=0

?n+s?1,i vi = ?n+s?1 +

s?1
(cid:88)

i=0

bi
(cid:80)s?1
j=0 bj

c f (tn+i, ?n+i) + O((cid:15)),

= ?n+s?1 + ?t

s?1
(cid:88)

i=0

bif (tn+i, ?n+i) + O((cid:15)).

(13)

For a large capacity neural network, we can assume that the error can be made
arbitrarily small ((cid:15) ? 0) and Eq. (10) is proved. Therefore, the form of the explicit
Adams multi-step methods of order ? s can be captured by a residual self-attention
transformer layer.

A single transformer layer is signi?cantly more expressive than any numerical
time integration scheme, enabling it to learn more complex temporal dependen-
cies. However, the linear combination of weighted features from past time-steps
is a commonality between self-attention and numerical time-integration methods.
This mathematical similarity indicates self-attention models may be better suited
for learning dynamical systems than alternative deep learning approaches. In par-
ticular traditional Euler time integration present in various models for modeling
dynamics [17, 18, 44] is a speci?c case of this single layer attention calculation.

Remark. The inclusion of time, t, as an input is required for non-autonomous sys-
tems with explicit time-dependent terms. This has surprising connections to the
implementation of the transformer which uses a positional embedding. Although the
positional embedding is included for the sake of denoting the order of the input to
the transformer, using current time of the given input state can accomplish the same
goal.

8

2.2. Embedding Model
The second major component of the machine learning methodology is the embed-
ding model responsible for projecting the discretized physical state space into a 1D
vector representation.
In NLP, the standard approach is to tokenize then embed
a ?nite vocabulary of words, syllables or characters using methods such as n-gram
models, Byte Pair Encoding [45], Word2Vec [46, 47], GloVe [48], etc. These methods
allow language to be represented by a series of 1D vectors that serve as the input
to the transformer. Clearly a ?nite tokenization and such NLP embeddings are not
directly applicable to a physical system, thus we propose our own embedding method
designed speci?cally for dynamical systems. Consider learning the generalized map-
ping between the system�s state space and embedded space: F : Rn�d ? Re and
G : Re ? Rn�d. Naturally, multiple approaches can be used especially if the dimen-
sionality of the embedded space is less than that of the state-space but this is not
always the case.

The primary approach that we will propose is a Koopman observable embedding
which is a technique that can be applied universally to all dynamical systems. Con-
sidering the discrete time form of the dynamical system in Eq. (1), the evolution of
the state variables can be abstracted by ?i+1 = F (?i) for which F is the dynamic
map from one time-step to the next. The foundation of Koopman theory states
any dynamical system can be represented in terms of an in?nite dimensional linear
operator acting on an in?nite set of state observable functions, g (?i), such that:

Kg (?i) (cid:44) g ? F (?i) ,

(14)

where K is the in?nite-dimensional linear operator referred to as the Koopman oper-
ator [49]. This implies that the system of observables can be evolved in time through
repeated application of the Koopman operator:

g (?i+1) = Kg (?i) ,

g (?i+2) = K2g (?i) ,

g (?i+3) = K3g (?i) , . . .

(15)

in which Kn denotes a n-fold composition, e.g. K3(g) = K(K(K(g))). Modeling the
dynamics of a system through the linear Koopman space can be attractive due to
its simpli?cation of the dynamics but also the potential physical insights it brings
along with it. Spectral analysis of the Koopman operator can reveal fundamental
dynamical modes that drive the system�s evolution in time.

Koopman theory can be viewed as a trade o? between lifting the state space
into observable space with more complex states but simpler dynamics. In practice,
the Koopman operator must be ?nitely approximated. This ?nite approximation
requires the identi?cation of the essential measurement functions that govern the

9

system�s dynamics and the respective approximate Koopman operator. Data-driven
machine learning has proven to be an e?ective approach for learning key Koopman
observables for modeling, control and dynamical mode analysis of many physical
systems [50, 51, 52, 53]. In recent years, the use of deep neural networks for learning
Koopman dynamics has proven to be successful [54, 55, 56, 57]. While deep learning
methods have enabled greater success with discovering Koopman observables and
operators such approaches have yet to be demonstrated for the long time prediction
of high-dimensional systems. This is likely due to the approximation of the ?nite-
dimensional Koopman observables, limited data and complete dependence on the
discovered Koopman operator K to model the dynamics. Suggesting the prediction
of a system through a single linear transform clearly has signi?cant limitations and
is fundamentally a naive approach from a machine learning perspective.

Figure 3: Example of a Koopman embedding for a two-dimensional system using a convolution
encoder-decoder model. The encoder model, F, projects the physical states into the approximate
Koopman observable embedding. The decoder model, G recovers the physical states from the
embedding.

In this work, we propose using approximate Koopman dynamics as a methodology
to develop embeddings for the transformer model such that F(?i) (cid:44) g(?i). As
seen in Fig. 3, the embedding model follows a standard encoder-decoder model with
the middle latent variables being the Koopman observables.
In this model, the
Koopman operator assumes the form of a learnable banded matrix that is optimized
Imposing some level of inductive bias on the form of the
with the auto-encoder.
Koopman matrix is fairly common in similar deep Koopman works [55, 56, 58]. We
found that this reduction of learnable parameters helped encourage the model to
discover better dynamical modes, preventing the model from over?tting to high-
frequency ?uctuations. Additionally, this form requires signi?cantly less memory to
store allowing models with embedding vectors of higher-dimensionality to be trained.
This learned Koopman operator is disposed of once training of the embedding model
is complete. Given the data set of physical state time-series, D? = {?i}D
i , the

10

Koopman embedding model is trained with the following loss:

D
(cid:88)

T
(cid:88)

LD? =

?0 M SE (cid:0)?i

j, G ? F (cid:0)?i
(cid:123)(cid:122)
Reconstruction

j

(cid:1)(cid:1)

(cid:125)

+?1 M SE (cid:0)?i

j, G ? KjF (cid:0)?i

0

(cid:124)

(cid:123)(cid:122)
Dynamics

(cid:1)(cid:1)

(cid:125)

+?2 (cid:107)K(cid:107)2
2
(cid:124) (cid:123)(cid:122) (cid:125)
Decay

.

i=1

j=0

(cid:124)

(16)
This loss function consists of three components: the ?rst is a reconstruction loss
which ensures a consistent mapping to and from the embedded representation. The
second is the Koopman dynamics loss which pushes ?j to follow linear dynamics.
The last term decays the Koopman operator�s parameters to help force the model to
discover meaningful dynamical modes and further prevent over?tting.

In reference to traditional NLP embeddings, we believe our Koopman observ-
able embedding has a motivation similar to Word2Vec [46] as well as more recent
embedding methods such as context2vec [59], ELMo [60], etc. These methods are
based on word context and association to develop a map where words that are re-
lated or synonymous to each other have similar embedded vectors. The Koopman
embedding model has a similar objective encouraging physical realizations contain-
ing similar dynamical modes to also have similar embeddings. This is because the
Koopman operator is time-invariant which means the embedded states must share
the same basis functions that govern their evolution. As a result, the loss func-
tion of the embedding model rewards time-steps that are near each other in time
or have the same underlying dynamics to have similar embeddings. Hence, our goal
with the embedding model is to not ?nd the true Koopman observables or operator,
but rather leverage Koopman to enforce physical context and association using the
learned dynamical modes.

3. Experiments and Results

The proposed transformer model is implemented for three dynamical systems. The
classical Lorenz system is used as a baseline test case in Section 3.1 to compare
the proposed model to alternative machine learning approaches due to the Lorenz
system�s numerical sensitivity. Following in Section 3.2, we consider the modeling of
two-dimensional Navier-Stokes ?uid ?ow and compare di?erent embedding methods
for the transformer. Lastly, to demonstrate the models scalability, we demonstrate
in Section 3.3 the transformer for the prediction of a three-dimensional reaction-
di?usion system. These examples encompass surrogate modeling physical systems
with stochastic initial conditions and parameters.

11

3.1. Chaotic Dynamics
As a foundational numerical example to rigorously compare the proposed model to
other classical machine learning techniques, we will ?rst look at surrogate modeling
of the Lorenz system governed by:

dx
dt

= ? (y ? x) ,

dy
dt

= x (? ? z) ? y,

dz
dt

= xy ? ?z.

(17)

We use the classical parameters of ? = 28, ? = 10, ? = 8/3. For this numerical
example, we wish to develop a surrogate model for predicting the Lorenz system
given a random initial state x0 ? U(?20, 20), y0 ? U(?20, 20) and z0 ? U(10, 40).
In other words, we wish to surrogate model various initial value problems for this
system of ODEs. The Lorenz system is used because of its well known chaotic
dynamics which make it extremely sensitive to numerical perturbations and thus an
excellent benchmark for assessing a machine learning model�s accuracy.

A total of four alternative machine learning models are implemented: a fully-
connected auto-regressive model, a fully-connected LSTM model, a deep neural net-
work Koopman model and lastly an echo-state model. All of these types of models
have been proposed in past literature for predicting various physical systems (e.g.
auto-regressive [15], fully-connected LSTMs [61], deep Koopman [55, 56] and echo-
state models [62, 63]). Each are provided the same training, validation and testing
data sets containing 2048, 64 and 256 time-series at a time-step size of ?t = 0.01
solved using a Runge-Kutta numerical solver, respectively. The training data set
contains time-series of 256 time-steps while the validation and testing data sets have
1024 time-steps. Each model is allowed to train for 500 epochs if applicable. The
proposed transformer and embedding model are trained for 200 and 300 epochs, re-
spectively, with an embedding dimension of 32. The embedding model is a simple
fully-connected encoder-decoder model, F : R3 ? R32; G : R32 ? R3, illustrated in
Fig. 4. The transformer is trained with a context length of 64 with 4 transformer
decoder layers. Both the training and validation data sets were chunked into a set
of 64 steps for the alternative models, when applicable, to train on the same context
length.

We plot four separate test cases in Fig. 5 for which only the initial state is
provided and the transformer model predicts 320 time-steps. Several test predictions
for alternative models are provided in Appendix A. In general, we can see that the
transformer model is able to yield extremely accurate predictions even beyond its
context length. Additionally, we plot the Lorenz solution for 25k time-steps from a
numerical solver and predicted from the transformer model in Fig. 6. Note that both
have the same structure, which qualitatively indicates that the transformer indeed
maintains physical dynamics.

12

Figure 4: Fully-connected embedding network with ReLU activation functions for the Lorenz sys-
tem.

Figure 5: Four test case predictions using the transformer model for 320 time-steps.

The proposed transformer and alternative models� relative mean squared errors
for the test set are plotted in Fig. 7a as a function of time and listed in Table 1 seg-
mented into several intervals based on the transformer�s context length. In general,
we can see all deep learning models perform well in the time-series length for which
they were trained with the deep Koopman model performing the best followed by
the transformer. As we extrapolate our predictions past the trained context range,
the bene?ts of the transformer become apparent with it achieving the best accuracy
for later times. To quantify accuracy of the chaotic dynamics of each model, the
Lorenz map is plotted in Fig. 7b which is a well-de?ned relation between successive
z local maxima despite the Lorenz�s chaotic nature. Calculated using 25k time-step
predictions from each model, again we can see that the transformer model agrees
the best with the numerical solver indicating that it has learned the best physical
dynamics of all the tested models.

Additionally, we test the transformer�s sensitivity to contaminated data by adding
white noise to the training observations scaled by the magnitude of the state vari-
ables. Each model tested with clean data is retrained with data perturbed by 1%
and 5% noise. The e?ects of these two di?erent noise levels is qualitatively illus-

13

(a) Numerical Solver

(b) Transformer

Figure 6: Lorenz solution of 25k time-steps with ?t = 0.01.

(a)

(b)

Figure 7: (a) The test relative mean-squared-error (MSE) with respect to time. (b) The Lorenz
map produced by each model.

trated in Appendix A. The errors are listed in Table 2. In general, we can indeed
see that the transformer can still perform adequately with noisy data by being the
best performing model for 1% noise and still being competitive, particularly at later
time-steps, with 5% noise.

The self-attention vectors, ?i ? R64, for a single time-series prediction of 512
steps are plotted in Fig. 8. For time-steps over i ? 64, the full context length of the
transformer is used with the attention weight ?i,64 corresponding to the most recent
time-step. Note that the transformer has learned multi-scale temporal dependencies
not achievable with other machine learning architectures. Additionally, this veri?es
that each attention head is learning di?erent dependencies suggesting that each head
may be a particular component of the transformer�s latent dynamics. This aligns
with what is observed in NLP when using multi-head attention which increases the
transformer�s predictive accuracy [1].

14

Table 1: Test set relative mean-squared-error (MSE) for surrogate modeling the Lorenz system at
several time-step intervals.

Model
Transformer
LSTM
Autoregressive
Echo State
Koopman
� Learnable parameters for the embedding/transformer model.
� Learnable output parameters/?xed input and reservoir parameters.

Parameters
36k/54k�
103k
92k
7.5k/6.3m�
108k

[0 ? 64)
0.0003
0.0041
0.0057
0.1026
0.0001

[128 ? 192)
0.0221
0.0369
0.0485
0.2209
2.0315

Relative MSE
[64 ? 128)
0.0060
0.0175
0.0253
0.1917
0.0962

Table 2: Test set relative mean-squared-error (MSE) for surrogate modeling the Lorenz system at
several time-step intervals with noisy data.

Relative MSE 1% Noise

Relative MSE 5% Noise

Model
Transformer
LSTM
Autoregressive
Echo State
Koopman

[0 ? 64)
0.0021
0.0045
0.0114
0.0859
0.0047

[64 ? 128)
0.0216
0.0218
0.0417
0.1686
0.1192

[128 ? 192)
0.0429
0.0437
0.0901
0.2102
0.1597

[0 ? 64)
0.0210
0.0212
0.0760
0.1000
0.0200

[64 ? 128)
0.0759
0.0758
0.2060
0.1581
0.0787

[128 ? 192)
0.1292
0.1324
0.2065
0.2051
0.1906

3.2. 2D Fluid Dynamics
The next dynamical system we will test is transient 2D ?uid ?ow governed by the
Navier-Stokes equations:

?ui
?t

+ uj

?ui
?xj

= ?

1
?

?p
?xi

+ ?

?2ui
?xj?xj

,

(18)

in which ui and p are the velocity and pressure, respectively. ? is the viscosity of
the ?uid. We consider modeling the classical problem of ?ow around a cylinder at
various Reynolds number de?ned by Re = uind/? in which uin = 1 and d = 2 are
In this work, we choose to
the inlet velocity and cylinder diameter, respectively.
develop a surrogate model to predict the solution at any Reynolds number between
Re ? U (100, 750). This problem is a fairly classical ?ow to investigate in scienti?c
machine learning with various levels of di?culty [64, 65, 55, 66, 67, 21]. Here we
choose one of the more di?cult forms: model the ?ow starting at a steady state ?ow

15

Figure 8: Attention vectors of the transformer model for the prediction of a single test case of the
Lorenz system. The top contours illustrate the attention weights at each time-step; the bottom
shows the respective state variables.

?eld at t = 0. Meaning the model is provided zero information on the structure of
the cylinder wake during testing other than the viscosity.

Training, validation and testing data is obtained using the OpenFOAM simula-
tor [68], from which a rectangular structured sub-domain is sampled centered around
the cylinder�s wake. Given that the ?ow is two-dimensional, our model will predict
the x-velocity, y-velocity and pressure ?elds, (ux, uy, p) ? R3�128�64. The training,
validation and test data sets consist of 27, 6 and 7 ?uid ?ow simulations, respec-
tively with 400 time-steps each at a physical time-step size of ?t = 0.5. As a base
line model, we train a convolutional encoder-decoder model with a stack of three
convolutional LSTMs [69] in the center. The input for this model consists of the
velocity ?elds, pressure ?eld and viscosity of the ?uid. We note that convolutional
LSTMs have been used extensively in recent scienti?c machine learning literature
for modeling various physical systems including ?uid dynamics [66, 70, 19, 21, 71],
thus can be considered a state-of-the-art approach. The convolutional LSTM model
is trained for 500 epochs.

Additionally, three di?erent embedding methods are implemented: the ?rst is
the proposed Koopman observable embedding using a convolutional auto-encoder

16

illustrated in Fig. 9. This model encodes the ?uid x-velocity, y-velocity, pressure and
viscosity ?elds to an embedded dimension of 128, F : R4�128�64 ? R128; G : R128 ?
R3�128�64. The second embedding method is using the same convolutional auto-

Figure 9: 2D convolutional embedding network with ReLU activation functions for the ?ow around a
cylinder system consisting of 5 convolutional encoding/decoding layers. Each convolutional operator
In the decoder, the feature maps are up-sampled before applying a
has a kernel size of (3, 3).
standard convolution. Additionally, two auxiliary fully-connected networks are used to predict the
diagonal and o?-diagonal elements of the Koopman operator for each viscosity ?.

encoder model but without the enforcement of Koopman dynamics on the embedded
variables. The third embedding method tested was principal component analysis
(PCA) as a classical baseline. For each embedding method an identical transformer
model with a context length of 128 and 4 transformer decoder layers is trained.
Similar to the previous example, the embedding models are trained for 300 epochs
when applicable and the transformer is trained for 200. To further isolate the impact
of the transformer model from the embedding model, we also train a fully-connected
model with 4 LSTM cells using the Koopman embedding as an input for 200 epochs.
Each trained model is tested on the test set by providing the initial laminar state
at t = 0 with the ?uid viscosity and allowing the model to predict 400 time-steps
into the future. Two test predictions using the proposed transformer model with
Koopman embeddings are plotted in Fig. 10 in which the predicted vorticity ?elds
are in good agreement with the true solution. The test set relative mean square
error for each output ?eld for each model is plotted in Fig. 11. The errors of each
?eld over the entire time-series are listed in Table 3. Additional results are provided
in Appendix B.

For all alternative models, a rapid error increase can be seen between t = [0, 100]
which is due to the transition from the laminar ?ow into vortex shedding. This
error then plateaus since each model is able to produce stable vortex shedding, as

17

Figure 10: Vorticity, ? = ?xuy ??yux, of two test case predictions using the proposed transformer
with Koopman embeddings at Reynolds numbers 233 (top) and 633 (bottom).

Figure 11: Test set relative mean-squared-error (MSE) of the transformer with Koopman (KM),
auto-encoder (AE) and PCA embedding methods, the convolutional LSTM model (ConvLSTM)
and fully-connected LSTM with Koopman embeddings (LSTM-KM).

illustrated in Figs. B.18 & B.19 in Appendix B. The proposed transformer with
Koopman embeddings is the only model that can accurately match the instantaneous
states from the numerical solution. These results indicate that the performance of
the transformer is highly dependent on the embedding method used, which should be
expected. However, the transformer with self-attention is equally important to the
model�s success as indicated by the performance of the Koopman embedding LSTM
model. Compared to the widely used ConvLSTM model, the proposed transformer
o?ers more reliable predictions for this ?uid ?ow with less learnable parameters.

To gain a greater understanding of why the Koopman embedding performs better
than the alternatives, we perform linear dimensionality reduction of the embedded
states using PCA into two principle components, (cid:101)?1, (cid:101)?2, in Fig. 12. For all embed-
ding methods, a circular structure is present re?ecting the periodic dynamics of the

18

Table 3: Test set relative mean-squared-error (MSE) of each output ?eld for surrogate modeling
2D ?uid ?ow past a cylinder. Models listed include the transformer with Koopman (KM), auto-
encoder (AE) and PCA embedding methods, the convolutional LSTM model (ConvLSTM) and
fully-connected LSTM with Koopman embeddings (LSTM-KM).

Model

Parameters

Relative MSE [0 ? 400]
uy
ux

p

Transformer-KM 224k/628k� 0.0042 0.0198 0.0021
224k/628k�
0.0161
Transformer-AE
Transformer-PCA 3.1m/628k�
0.0117
0.0103
0.0094

ConvLSTM
LSTM-KM
� Learnable parameters for the embedding/ transformer model.
� Learnable parameters for the embedding/ LSTM model.

0.1068
0.0984
0.0938
0.1155

0.0288
0.0247
0.0240
0.0266

934k
224k/759k�

vortex shedding. Compared to the auto-encoder, we note that the Koopman prin-
ciple subspace has a more consistent structure for later time-steps indicating that
the Koopman loss term does indeed encourage the discovery of common dynamical
modes. Yet for early time-steps, the Koopman model has a unique trajectory be-
tween Reynolds numbers. This lack of uniqueness with the PCA embedding at early
time-steps results in the transformer not being able to di?erentiate between Reynolds
numbers, lowering predictive accuracy with this embedding method. The Koopman
approach strikes a balance between structure but uniqueness between ?ows for the
transformer to learn with.

Figure 12: The two-dimensional principle subspace of the embedded vectors, ?i, from each tested
embedding model for two di?erent Reynolds numbers.

19

3.3. 3D Reaction-Di?usion Dynamics
The ?nal numerical example to demonstrate the proposed transformer model is a
3D reaction-di?usion system governed by the Gray-Scott model:

?u
?t

= ru

?2u
?x2
i

? uv2 + f (1 ? u) ,

?v
?t

= rv

?2v
?x2
i

+ vu2 ? (f + k) v,

(19)

in which u and v are the concentration of two species, ru and rv are their respective
di?usion rates, k is the kill rate and f is the feed rate. This is a classical system
of particular application to chemical processes as it models the following reaction:
U + 2V ? 3V ; V ? P . For a set region of feed and kill rates, this seemingly simple
system can yield a wide range of complex dynamics [72, 73]. Hence, under the right
settings, this system is an excellent case study to push the proposed methodology to
its predictive limits. In this work, we will use the parameters: ru = 0.2, rv = 0.1,
k = 0.055 and f = 0.025 which results in a complex dynamical reaction. Akin to
the ?rst numerical example, the initial condition of this system is stochastic such
that the system is seeded with 3 randomly placed perturbations within the periodic
domain.

Training, validation and testing data are obtained from a Runge-Kutta ?nite
di?erence simulation on a structured grid, (u, v) ? R2�32�32�32. The training, val-
idation and test data sets consist of 512, 16 and 56 time-series, respectively, with
200 time-steps each at a physical time-step size of ?t = 5. A 3D convolutional
encoder-decoder is used to embed the two species into a 512 embedding dimension,
F : R2�32�32�32 ? R512; G : R512 ? R2�32�32�32, illustrated in Fig. 13.

Transformer models with varying depth are trained all with a context length
of 128. All other model and training parameters for the transformer models are
consistent. A test prediction using the transformer model is shown in Fig. 14 and
the errors for each trained transformer are listed in Table 4. Despite this system
having complex dynamics in 3D space, the transformer is able to produce acceptable
predictions with very similar structures as the numerical solver. To increase the
model�s predictive accuracy we believe the limitation here is not in the transformer,
but rather the number of training data and the inaccuracies of the embedding model
due to the dimensionality reduction needed. This is supported by the fact that
increasing the transformer�s depth does not yield considerable improvements for the
test errors in Table 4. Additional, results are provided in Appendix C.

4. Conclusion

While transformers and self-attention models have been established as a powerful
framework for NLP tasks, the adoption of such methodologies has yet to fully per-

20

Figure 13: 3D convolutional embedding network with leaky ReLU activation functions for the Gray-
Scott system. Batch-normalization used between each of the convolutional layers. In the decoder,
the feature maps are up-sampled before applying a standard 3D convolution.

mute other ?elds. In this work, we have demonstrated the potential transformers
have for modeling dynamics of physical phenomena. The transformer architecture
allows the model to learn longer and more complex temporal dependencies than al-
ternative machine learning methods. Such models can be particularly bene?cial for
physical systems that exhibit dynamics that evolve on multiple time scales or consist
of multiple phases such as turbulent ?uid ?ow, chemical reactions or molecular dy-
namics. This can be attributed to the transformer�s ability to draw information from
many past time-steps directly with self-attention, learning accurate time integration
without computationally expensive recurrent connections.

The key challenge of using the transformer model is identifying appropriate em-
beddings to represent the physical state of the system, for which we propose the use
of Koopman dynamics to enforce dynamical context. Using the proposed methods,
our transformer model can outperform alternative models widely seen in recent sci-

21

(a) u target (top) and transformer prediction (bottom)

(b) v target (top) and transformer prediction (bottom)

Figure 14: Test case volume plots for the Gray-Scott system. Isosurfaces displayed span the range
u, v = [0.3, 0.5] to show the inner structure.

enti?c machine learning literature. The investigation of unsupervised pre-training of
such transformer models as well as gaining a better understanding of what atten-
tion mechanisms imply for physical systems will be the subject of works in the near
future.

Acknowledgments
The anonymous reviewers are thanked for their signi?cant e?ort in improving and
clarifying this manuscript. The work reported here was initiated from the Defense
Advanced Research Projects Agency (DARPA) under the Physics of Arti?cial Intel-
ligence (PAI) program (contract HR00111890034). The authors acknowledge com-
puting resources provided by the AFOSR O?ce of Scienti?c Research through the
DURIP program and by the University of Notre Dame�s Center for Research Com-
puting (CRC). The work of NG was supported by the National Science Foundation
(NSF) Graduate Research Fellowship Program grant No. DGE-1313583.

22

Table 4: Test set relative mean-squared-error (MSE) for surrogate modeling 3D Gray-Scott system.

Relative MSE [0 ? 200]

Model
Transformer
Transformer
Transformer

Parameters
6.2m/6.6m�
6.2m/12.9m�
6.2m/25.5m�
� Learnable parameters for the embedding/ transformer model.

u
0.0159
0.0154
0.0125

Layers
2
4
8

v
0.0120
0.0130
0.0101

References

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
I. Guyon, U. V.
L. u. Kaiser, I. Polosukhin, Attention is all you need, in:
Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, R. Garnett
(Eds.), Advances in Neural Information Processing Systems, Vol. 30, Curran
Associates, Inc., 2017.
URL
3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

https://proceedings.neurips.cc/paper/2017/file/

[2] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep
bidirectional transformers for language understanding, in: NAACL-HLT, Asso-
ciation for Computational Linguistics, 2019, pp. 4171�4186. doi:10.18653/v1/
N19-1423.
URL https://aclanthology.org/N19-1423

[3] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, Language

models are unsupervised multitask learners, OpenAI Blog.
URL
models_are_unsupervised_multitask_learners.pdf

https://cdn.openai.com/better-language-models/language_

[4] Z. Dai, Z. Yang, Y. Yang, J. G. Carbonell, Q. Le, R. Salakhutdinov,
Transformer-XL: Attentive language models beyond a ?xed-length context, in:
Proceedings of the 57th Annual Meeting of the Association for Computational
Linguistics, 2019, pp. 2978�2988. doi:10.18653/v1/P19-1285.
URL https://aclanthology.org/P19-1285

[5] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettle-
moyer, V. Stoyanov, Roberta: A robustly optimized bert pretraining approach,
arXiv preprint arXiv:1907.11692.

23

[6] I. Bilionis, N. Zabaras, B. A. Konomi, G. Lin, Multi-output separable Gaus-
sian process: Towards an e?cient, fully Bayesian paradigm for uncertainty
quanti?cation, Journal of Computational Physics 241 (2013) 212 � 239.
doi:10.1016/j.jcp.2013.01.011.
URL
S0021999113000417

http://www.sciencedirect.com/science/article/pii/

[7] I. Bilionis, N. Zabaras, Bayesian Uncertainty Propagation Using Gaussian
Processes, Springer International Publishing, 2016, pp. 1�45. doi:10.1007/
978-3-319-11259-6_16-1.
URL https://doi.org/10.1007/978-3-319-11259-6_16-1

[8] S. Atkinson, N. Zabaras, Structured Bayesian Gaussian process latent variable
model: Applications to data-driven dimensionality reduction and high-
dimensional inversion, Journal of Computational Physics 383 (2019) 166 � 195.
doi:10.1016/j.jcp.2018.12.037.
URL
S0021999119300397

http://www.sciencedirect.com/science/article/pii/

[9] D. Xiu, G. E. Karniadakis, The Wiener-Askey polynomial chaos for stochastic
di?erential equations, SIAM journal on scienti?c computing 24 (2) (2002) 619�
644. doi:10.1137/S1064827501387826.
URL https://doi.org/10.1137/S1064827501387826

[10] S. Chakraborty, N. Zabaras, E?cient data-driven reduced-order models for
high-dimensional multiscale dynamical systems, Computer Physics Communi-
cations 230 (2018) 70 � 88. doi:10.1016/j.cpc.2018.04.007.
URL
S0010465518301176

http://www.sciencedirect.com/science/article/pii/

[11] H. Gao, J.-X. Wang, M. J. Zahr, Non-intrusive model reduction of large-scale,
nonlinear dynamical systems using deep learning, Physica D: Nonlinear
Phenomena 412 (2020) 132614. doi:10.1016/j.physd.2020.132614.
URL
S0167278919305573

http://www.sciencedirect.com/science/article/pii/

[12] G. Tanaka, T. Yamane, J. B. H�eroux, R. Nakane, N. Kanazawa, S. Takeda,
H. Numata, D. Nakano, A. Hirose, Recent advances in physical reser-
voir computing: A review, Neural Networks 115 (2019) 100 � 123.
doi:10.1016/j.neunet.2019.03.005.

24

URL
S0893608019300784

http://www.sciencedirect.com/science/article/pii/

[13] Y. Zhu, N. Zabaras, Bayesian deep convolutional encoder�decoder networks for
surrogate modeling and uncertainty quanti?cation, Journal of Computational
Physics 366 (2018) 415 � 447. doi:10.1016/j.jcp.2018.04.018.
URL
S0021999118302341

http://www.sciencedirect.com/science/article/pii/

[14] R. K. Tripathy, I. Bilionis, Deep UQ: Learning deep neural network surrogate
models for high dimensional uncertainty quanti?cation, Journal of Computa-
tional Physics 375 (2018) 565 � 588. doi:10.1016/j.jcp.2018.08.036.
URL
S0021999118305655

http://www.sciencedirect.com/science/article/pii/

[15] N. Geneva, N. Zabaras, Modeling the dynamics of PDE systems with physics�
constrained deep auto�regressive networks, Journal of Computational Physics
403 (2020) 109056. doi:10.1016/j.jcp.2019.109056.
URL
S0021999119307612

https://www.sciencedirect.com/science/article/pii/

[16] S. Mo, Y. Zhu, N. Zabaras, X. Shi, J. Wu, Deep convolutional encoder-
decoder networks for uncertainty quanti?cation of dynamic multiphase ?ow
in heterogeneous media, Water Resources Research 55 (1) (2019) 703�728.
doi:10.1029/2018WR023528.
URL https://doi.org/10.1029/2018WR023528

[17] R. Gonz�alez-Garc�?a, R. Rico-Mart�?nez, I. Kevrekidis, Identi?cation of dis-
tributed parameter systems: A neural net based approach, Computers &
Chemical Engineering 22 (1998) S965 � S968, european Symposium on Com-
puter Aided Process Engineering-8. doi:10.1016/S0098-1354(98)00191-4.
URL
S0098135498001914

http://www.sciencedirect.com/science/article/pii/

[18] A. Sanchez-Gonzalez, J. Godwin, T. Pfa?, R. Ying, J. Leskovec, P. Battaglia,
in: H. D. III,
Learning to simulate complex physics with graph networks,
A. Singh (Eds.), Proceedings of the 37th International Conference on Machine
Learning, Vol. 119 of Proceedings of Machine Learning Research, PMLR, 2020,
pp. 8459�8468.
URL http://proceedings.mlr.press/v119/sanchez-gonzalez20a.html

25

[19] M. Tang, Y. Liu, L. J. Durlofsky, A deep-learning-based surrogate model for
data assimilation in dynamic subsurface ?ow problems, Journal of Computa-
tional Physics 413 (2020) 109456. doi:10.1016/j.jcp.2020.109456.
URL
S0021999120302308

https://www.sciencedirect.com/science/article/pii/

[20] R. Maulik, R. Egele, B. Lusch, P. Balaprakash, Recurrent neural network ar-
chitecture search for geophysical emulation (2020) 1�14doi:10.1109/SC41405.
2020.00012.
URL https://doi.org/10.1109/SC41405.2020.00012

[21] N. Geneva, N. Zabaras, Multi-?delity generative deep learning turbulent ?ows,
Foundations of Data Science 2 (2020) 391. doi:10.3934/fods.2020019.
URL http://aimsciences.org/article/id/3a9f3d14-3421-4947-a45f-a9cc74edd097

[22] Yi-Jen Wang, Chin-Teng Lin, Runge-kutta neural network for identi?cation of
dynamical systems in high accuracy, IEEE Transactions on Neural Networks
9 (2) (1998) 294�307. doi:10.1109/72.661124.

[23] M. Zhu, B. Chang, C. Fu, Convolutional neural networks combined with Runge-

Kutta methods, arXiv preprint arXiv:1802.08831.

[24] H. Wessels, C. Wei�enfels, P. Wriggers, The neural particle method � an
updated Lagrangian physics informed neural network for computational ?uid
dynamics, Computer Methods in Applied Mechanics and Engineering 368
(2020) 113127. doi:10.1016/j.cma.2020.113127.
URL
S0045782520303121

https://www.sciencedirect.com/science/article/pii/

[25] A. Shalova, I. Oseledets, Tensorized transformer for dynamical systems model-

ing, arXiv preprint arXiv:2006.03445.

[26] P. Veli?ckovi�c, G. Cucurull, A. Casanova, A. Romero, P. Li`o, Y. Bengio, Graph

attention networks.
URL https://openreview.net/forum?id=rJXMpikCZ

[27] H. Zhang, I. Goodfellow, D. Metaxas, A. Odena, Self-attention generative ad-
versarial networks, in: K. Chaudhuri, R. Salakhutdinov (Eds.), Proceedings of
the 36th International Conference on Machine Learning, Vol. 97 of Proceedings
of Machine Learning Research, PMLR, 2019, pp. 7354�7363.
URL http://proceedings.mlr.press/v97/zhang19d.html

26

[28] J. Fu, J. Liu, H. Tian, Y. Li, Y. Bao, Z. Fang, H. Lu, Dual attention network
for scene segmentation, in: Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, 2019, pp. 3146�3154.
URL
https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_
Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf

[29] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, I. Sutskever, Genera-
tive pretraining from pixels, in: H. D. III, A. Singh (Eds.), Proceedings of the
37th International Conference on Machine Learning, Vol. 119 of Proceedings of
Machine Learning Research, PMLR, 2020, pp. 1691�1703.
URL http://proceedings.mlr.press/v119/chen20s.html

[30] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, Improving language

understanding by generative pre-training, OpenAI Blog.
URL
language-unsupervised/language_understanding_paper.pdf

https://openai-assets.s3.amazonaws.com/research-covers/

[31] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac,
T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma,
Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger, M. Drame, Q. Lhoest, A. M.
Rush, Huggingface�s transformers: State-of-the-art natural language processing,
arXiv preprint arXiv:1910.03771.

[32] A. Graves, G. Wayne, I. Danihelka, Neural turing machines, arXiv preprint

arXiv:1410.5401.

[33] D. Bahdanau, K. Cho, Y. Bengio, Neural machine translation by jointly learning
to align and translate, in: 3rd International Conference on Learning Represen-
tations, 2015. arXiv:1409.0473.

[34] T. Luong, H. Pham, C. D. Manning, E?ective approaches to attention-based
neural machine translation, in: EMNLP, 2015, pp. 1412�1421. arXiv:508.
04025.
URL http://aclweb.org/anthology/D/D15/D15-1166.pdf

[35] S. Sukhbaatar, E. Grave, P. Bojanowski, A. Joulin, Adaptive attention span in
transformers, in: Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, Association for Computational Linguistics, Flo-
rence, Italy, 2019, pp. 331�335. doi:10.18653/v1/P19-1032.
URL https://aclanthology.org/P19-1032

27

[36] S. Sukhbaatar, E. Grave, G. Lample, H. Jegou, A. Joulin, Augmenting self-

attention with persistent memory, arXiv preprint arXiv:1907.01470.

[37] N. Kitaev, L. Kaiser, A. Levskaya, Reformer: The e?cient transformer, in:

International Conference on Learning Representations, 2020.
URL https://openreview.net/forum?id=rkgNKkHtvB

[38] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. K. Duvenaud, Neural ordinary
di?erential equations, in: S. Bengio, H. Wallach, H. Larochelle, K. Grauman,
N. Cesa-Bianchi, R. Garnett (Eds.), Advances in Neural Information Processing
Systems, Vol. 31, Curran Associates, Inc., 2018, pp. 6571�6583.
URL
69386f6bb1dfed68692a24c8686939b9-Paper.pdf

https://proceedings.neurips.cc/paper/2018/file/

[39] E. Dupont, A. Doucet, Y. W. Teh, Augmented neural odes, in: H. Wallach,
H. Larochelle, A. Beygelzimer, F. d'Alch�e-Buc, E. Fox, R. Garnett (Eds.), Ad-
vances in Neural Information Processing Systems, Vol. 32, Curran Associates,
Inc., 2019, pp. 3140�3150.
URL
21be9a4bd4f81549a9d1d241981cec3c-Paper.pdf

https://proceedings.neurips.cc/paper/2019/file/

[40] J. Stoer, R. Bulirsch, Introduction to numerical analysis, Vol. 12, Springer, 2013.

doi:10.1007/978-1-4757-2272-7.
URL https://doi.org/10.1007/978-1-4757-2272-7

[41] K. Hornik, M. Stinchcombe, H. White, Multilayer feedforward networks
(1989) 359�366.

are universal approximators, Neural Networks 2 (5)
doi:10.1016/0893-6080(89)90020-8.
URL
0893608089900208

https://www.sciencedirect.com/science/article/pii/

[42] G. Cybenko, Approximation by superpositions of a sigmoidal function, Math-
ematics of control, signals and systems 2 (4) (1989) 303�314. doi:10.1007/
BF02551274.
URL https://doi.org/10.1007/BF02551274

[43] K. Hornik, Approximation capabilities of multilayer feedforward networks,
Neural Networks 4 (2) (1991) 251�257. doi:10.1016/0893-6080(91)90009-T.
URL
https://www.sciencedirect.com/science/article/pii/
089360809190009T

28

[44] Y. Lu, A. Zhong, Q. Li, B. Dong, Beyond ?nite layer neural networks: Bridg-
ing deep architectures and numerical di?erential equations, in: International
Conference on Machine Learning, PMLR, 2018, pp. 3276�3285.

[45] P. Gage, A new algorithm for data compression, Vol. 12, McPherson, KS: R &

D Publications, c1987-1994., 1994, pp. 23�38.
URL https://www.derczynski.com/papers/archive/BPE_Gage.pdf

[46] T. Mikolov, K. Chen, G. Corrado, J. Dean, E?cient estimation of word repre-
sentations in vector space, in: Workshop Proceedings International Conference
on Learning Representations, 2013.

[47] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, J. Dean, Distributed
representations of words and phrases and their compositionality, in: Advances
in neural information processing systems, 2013, pp. 3111�3119.
URL
9aa42b31882ec039965f3c4923ce901b-Paper.pdf

https://proceedings.neurips.cc/paper/2013/file/

[48] J. Pennington, R. Socher, C. D. Manning, Glove: Global vectors for word rep-
resentation, in: Proceedings of the 2014 conference on empirical methods in
natural language processing (EMNLP), 2014, pp. 1532�1543.

[49] B. O. Koopman, Hamiltonian systems and transformation in Hilbert space,
Proceedings of the national academy of sciences of the United States of America
17 (5) (1931) 315. doi:10.1073/pnas.17.5.315.

[50] Q. Li, F. Dietrich, E. M. Bollt, I. G. Kevrekidis, Extended dynamic mode decom-
position with dictionary learning: A data-driven adaptive spectral decomposi-
tion of the Koopman operator, Chaos: An Interdisciplinary Journal of Nonlinear
Science 27 (10) (2017) 103111. arXiv:https://doi.org/10.1063/1.4993854,
doi:10.1063/1.4993854.
URL https://doi.org/10.1063/1.4993854

[51] M. Korda, I. Mezi�c, Linear predictors for nonlinear dynamical systems: Koop-
man operator meets model predictive control, Automatica 93 (2018) 149 � 160.
doi:10.1016/j.automatica.2018.03.046.
URL
S000510981830133X

http://www.sciencedirect.com/science/article/pii/

[52] M. Korda, M. Putinar, I. Mezi�c, Data-driven spectral analysis of the Koopman
operator, Applied and Computational Harmonic Analysis 48 (2) (2020) 599 �

29

629. doi:10.1016/j.acha.2018.08.002.
URL
S1063520318300988

http://www.sciencedirect.com/science/article/pii/

[53] I. Mezic, On numerical approximations of the Koopman operator, arXiv preprint

arXiv:2009.05883.

[54] N. Takeishi, Y. Kawahara, T. Yairi, Learning Koopman invariant subspaces
for dynamic mode decomposition, in:
I. Guyon, U. V. Luxburg, S. Bengio,
H. Wallach, R. Fergus, S. Vishwanathan, R. Garnett (Eds.), Advances in
Neural Information Processing Systems, Vol. 30, Curran Associates, Inc., 2017.
URL
https://proceedings.neurips.cc/paper/2017/file/
3a835d3215755c435ef4fe9965a3f2a0-Paper.pdf

[55] B. Lusch, J. N. Kutz, S. L. Brunton, Deep learning for universal linear em-
beddings of nonlinear dynamics, Nature communications 9 (1) (2018) 1�10.
doi:10.1038/s41467-018-07210-0.
URL https://doi.org/10.1038/s41467-018-07210-0

[56] S. E. Otto, C. W. Rowley, Linearly recurrent autoencoder networks for learning
dynamics, SIAM Journal on Applied Dynamical Systems 18 (1) (2019) 558�593.
doi:10.1137/18M1177846.
URL https://doi.org/10.1137/18M1177846

[57] S. L. Brunton, M. Budi?si�c, E. Kaiser, J. N. Kutz, Modern Koopman theory for

dynamical systems, arXiv preprint arXiv:2102.12086.

[58] Y. Li, H. He, J. Wu, D. Katabi, A. Torralba, Learning compositional Koopman
operators for model-based control, in: International Conference on Learning
Representations, 2020.
URL https://openreview.net/forum?id=H1ldzA4tPr

[59] O. Melamud, J. Goldberger, I. Dagan, context2vec: Learning generic context
embedding with bidirectional LSTM, in: Proceedings of the 20th SIGNLL con-
ference on computational natural language learning, 2016, pp. 51�61.
URL https://aclanthology.org/K16-1006.pdf

[60] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, L. Zettle-
moyer, Deep contextualized word representations, in: Proceedings of the 2018

30

Conference of the North American Chapter of the Association for Compu-
tational Linguistics: Human Language Technologies, Volume 1 (Long Pa-
pers), Association for Computational Linguistics, 2018, pp. 2227�2237. doi:
10.18653/v1/N18-1202.
URL https://aclanthology.org/N18-1202

[61] J. Zhao, F. Deng, Y. Cai, J. Chen, Long short-term memory - fully connected
(LSTM-FC) neural network for PM2.5 concentration prediction, Chemosphere
220 (2019) 486�492. doi:10.1016/j.chemosphere.2018.12.128.
URL
S0045653518324639

https://www.sciencedirect.com/science/article/pii/

[62] A. Chattopadhyay, P. Hassanzadeh, D. Subramanian, K. Palem, Data-driven
prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep
learning methods: Reservoir computing, ANN, and RNN-LSTM, Nonlinear Pro-
cesses in Geophysics 27 (2020) 373�389. doi:10.5194/npg-27-373-2020.
URL https://doi.org/10.5194/npg-27-373-2020

[63] M. Luko?sevi?cius, A practical guide to applying echo state networks, in: Neural
networks: Tricks of the trade, Springer, 2012, pp. 659�686. doi:10.1007/
978-3-642-35289-8_36.
URL https://doi.org/10.1007/978-3-642-35289-8_36

[64] S. Lee, D. You, Prediction of laminar vortex shedding over a cylinder using deep

learning, arXiv preprint arXiv:1712.07854.

[65] J. Morton, A. Jameson, M. J. Kochenderfer, F. Witherden, Deep dynamical
modeling and control of unsteady ?uid ?ows,
in: S. Bengio, H. Wallach,
H. Larochelle, K. Grauman, N. Cesa-Bianchi, R. Garnett (Eds.), Advances in
Neural Information Processing Systems, Vol. 31, Curran Associates, Inc., 2018.
URL
https://proceedings.neurips.cc/paper/2018/file/
2b0aa0d9e30ea3a55fc271ced8364536-Paper.pdf

[66] R. Han, Y. Wang, Y. Zhang, G. Chen, A novel spatial-temporal prediction
method for unsteady wake ?ows based on hybrid deep neural network, Physics
of Fluids 31 (12) (2019) 127101. doi:10.1063/1.5127247.
URL https://doi.org/10.1063/1.5127247

[67] J. Xu, K. Duraisamy, Multi-level convolutional autoencoder networks
Computer
spatio-temporal
in Applied Mechanics and Engineering 372 (2020) 113379.

for
Methods

parametric

prediction

dynamics,

of

31

doi:10.1016/j.cma.2020.113379.
URL
S0045782520305648

https://www.sciencedirect.com/science/article/pii/

[68] H. Jasak, A. Jemcov, Z. Tukovic, et al., Openfoam: A C++ library for complex
physics simulations, in: International workshop on coupled methods in numeri-
cal dynamics, Vol. 1000, IUC Dubrovnik Croatia, 2007, pp. 1�20.

[69] X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-k. Wong, W.-c. WOO, Convolu-
tional lstm network: A machine learning approach for precipitation nowcasting,
in: C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, R. Garnett (Eds.), Advances
in Neural Information Processing Systems, Vol. 28, Curran Associates, Inc.,
2015.
URL
07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf

https://proceedings.neurips.cc/paper/2015/file/

[70] S. Wiewel, M. Becher, N. Thuerey, Latent space physics: Towards learning the
temporal evolution of ?uid ?ow, Computer Graphics Forum 38 (2) (2019) 71�82.
doi:10.1111/cgf.13620.
URL https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13620

[71] R. Maulik, B. Lusch, P. Balaprakash, Reduced-order modeling of advection-
dominated systems with recurrent neural networks and convolutional autoen-
coders, Physics of Fluids 33 (3) (2021) 037106. arXiv:https://doi.org/10.
1063/5.0039986, doi:10.1063/5.0039986.
URL https://doi.org/10.1063/5.0039986

[72] J. E. Pearson, Complex patterns in a simple system, Science 261 (5118) (1993)

189�192. doi:10.1126/science.261.5118.189.
URL https://science.sciencemag.org/content/261/5118/189

[73] K. J. Lee, W. D. McCormick, Q. Ouyang, H. L. Swinney, Pattern formation by
interacting chemical fronts, Science 261 (5118) (1993) 192�194. doi:10.1126/
science.261.5118.192.
URL https://science.sciencemag.org/content/261/5118/192

32

Appendix A. Lorenz Supplementary Results

Predictions of the tested models for three test cases are plotted in Fig. A.15. All
machine learning methods are accurate for the ?rst several time-steps, but begin
to quickly deviate from the numerical solution in subsequent time-steps. The trans-
former model is consistently accurate within the plotted time frame, typically outper-
forming alternative methods at later time-steps. The predictions of the transformer
model are also compared to the solution from a less accurate Euler time-integration
method in Fig. A.16. Due to the numerical di?erences between the Runge-Kutta
ground truth and the Euler methods, the solutions are quick to deviate and the
transformer clearly outperforms the Euler numerical simulator. Lastly, in Fig. A.17
we illustrate the two di?erent noise levels used to contaminate the training data for
the results listed in Table 2.

Figure A.15: Three Lorenz test case predictions using each tested model for 128 time-steps.

Figure A.16: Three Lorenz test solutions solved using Runge-Kutta and Euler time-integration
methods with the prediction of the transformer for 256 time-steps.

33

Figure A.17: Comparison between the clean and contaminated (noisy) training data.

Appendix B. Cylinder Supplementary Results

Prediction ?elds of all the tested models for a single test case are plotted in Figs. B.18
and B.19. While all models are able to perform adequately with qualitatively good
results, the transformer model with Koopman embeddings (Transformer-KM) out-
performs alternatives. Additionally, the training and validation errors during training
are plotted in Fig. B.20 where all models have similar training and validation error
indicating minimal over?tting.

The evolution of the ?ow ?eld projected onto the dominant eigenvectors of the
learned Koopman operator for two Reynolds numbers is plotted in Fig. B.21. This
re?ects the dynamical modes that were learned by the embedding model to impose
physical �context� on the embeddings. Given that the eigenvectors are complex, we
plot both the magnitude, |?|, and angle, ??. For both Reynolds numbers, it is easy
to see the initial transition region, t < 100, before the system enters the periodic
vortex shedding. Once in the periodic region, we can see that the higher Reynolds
number has a higher frequency which re?ects the increased vortex shedding speed.

Appendix C. Gray-Scott Supplementary Results

Volume plots of two test cases for both species are provided in Figs. C.22 & C.23. In
addition to better visualize the accuracy of the trained transformer model, contour
plots for three test cases are also provided in Fig. C.24. The transformer model
is able to reliably predict the earlier time-steps with reasonable accuracy. As the
system continues to react and the reaction fronts begin to interact resulting in the
well-known complex structures of the Gray-Scott system [72, 73]. During these later

34

Figure B.18: Velocity magnitude predictions of a test case at Re = 633 using the transformer with
Koopman (KM), auto-encoder (AE) and PCA embedding methods, the convolutional LSTM model
(ConvLSTM) and fully-connected LSTM with Koopman embeddings (LSTM-KM).

times, the transformer begins to degrade in accuracy yet does maintain notable
consistency with the true solution.

35

Figure B.19: Pressure predictions of a test case at Re = 633 using the transformer with Koopman
(KM), auto-encoder (AE) and PCA embedding methods, the convolutional LSTM model (ConvL-
STM) and fully-connected LSTM with Koopman embeddings (LSTM-KM).

Figure B.20: The training and validation mean square error (MSE) of the transformer with Koop-
man (KM), auto-encoder (AE) and PCA embedding methods, the convolutional LSTM model
(ConvLSTM) and fully-connected LSTM with Koopman embeddings (LSTM-KM) during training.

36

(a) Re = 233

(b) Re = 633

Figure B.21: The dynamics of the ?uid ?ow around a cylinder projected onto the 8 most dominant
eigenvectors of the learned Koopman operator, K, in the embedding model.

37

(a) u target (top) and transformer prediction (bottom)

(b) v target (top) and transformer prediction (bottom)

Figure C.22: Test case volume plots for the Gray-Scott system.
range u, v = [0.3, 0.5] to show the inner structure.

Isosurfaces displayed span the

(a) u target (top) and transformer prediction (bottom)

(b) v target (top) and transformer prediction (bottom)

Figure C.23: Test case volume plots for the Gray-Scott system.
range u, v = [0.3, 0.5] to show the inner structure.

Isosurfaces displayed span the

38

Figure C.24: x ? y plane contour plots of three Gray-Scott test cases sliced at z = 16.

39


