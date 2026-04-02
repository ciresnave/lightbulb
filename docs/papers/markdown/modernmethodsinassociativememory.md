Modern Methods in Associative Memory

Dmitry Krotov1,2, Benjamin Hoover1,3, Parikshit Ram1, Bao Pham1,4

1IBM Research, 2MIT, 3Georgia Tech, 4RPI

Date: July 14, 2025

Tutorial: ICML 2025, Vancouver, BC, Canada

Website: https://tutorial.amemory.net

Abstract
Associative Memories like the famous Hopfield Networks are elegant models for describing fully

recurrent neural networks whose fundamental job is to store and retrieve information. In the

past few years they experienced a surge of interest due to novel theoretical results pertaining to

their information storage capabilities, and their relationship with SOTA AI architectures, such

as Transformers and Diffusion Models. These connections open up possibilities for interpreting

the computation of traditional AI networks through the theoretical lens of Associative Memories.

Additionally, novel Lagrangian formulations of these networks make it possible to design powerful

distributed models that learn useful representations and inform the design of novel architectures.

This tutorial provides an approachable introduction to Associative Memories, emphasizing the

modern language and methods used in this area of research, with practical hands-on mathematical

derivations and coding notebooks.

5
2
0
2

l
u
J

8

]

G
L
.
s
c
[

1
v
1
1
2
6
0
.
7
0
5
2
:
v
i
X
r
a

Contents

1 Introduction

2 Dense Associative Memory: Discrete State Vector

2.1

Information Storage Capacity . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

2.2 Limiting Cases

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

1

5
6

9

2.3 General Dense Associative Memory with Binary State Variables . . . . . . . . . .

10

3 General Dense Associative Memory

3.1 Building Blocks of AMs with Modular Energies . . . . . . . . . . . . . . . . . . .

3.2 Dynamical Neurons and their Lagrangians . . . . . . . . . . . . . . . . . . . . . .

3.3 Hypersynapses

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

3.4 Energy Descent Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

3.5

Implementing AMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

3.5.1 Energy Transformer Block . . . . . . . . . . . . . . . . . . . . . . . . . . .

3.6 Bridging Energy Minimization and Feedforward Prediction . . . . . . . . . . . . .

4 Failure of Memory and Generative AI

4.1 Diffusion Models

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

4.2 Diffusion Models from Associative Memories . . . . . . . . . . . . . . . . . . . . .

4.3 Memorization - Spurious - Generalization Transition . . . . . . . . . . . . . . . .

5 Associative Memory: A Machine Learning Model

5.1 Machine Learning Modeling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

5.2 Associative Memory Network as a Model . . . . . . . . . . . . . . . . . . . . . . .

5.2.1 Memory Capacity and Expressivity . . . . . . . . . . . . . . . . . . . . . .

5.2.2

Supervised Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

5.2.3 Nonparametric vs Parametric Models . . . . . . . . . . . . . . . . . . . . .

5.3 Clustering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

5.3.1 Euclidean Clustering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

5.3.2 Deep Clustering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

5.4 Kernel Machines

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

5.4.1 Random Features . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

5.4.2 Novel Energy Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . .

6 Conclusion

12
12

14

15

17

17

18

23

25
26

27

30

33
33

36

38

39

40

40

41

44

46

47

49

54

Chapter 1

Introduction

Associative Memory (AM) is a core concept in psychology responsible for linking related items

[1]. For instance, if one is shown an image of a strawberry, it is likely that they can recall the

smell and taste of this fruit; or, in the case of an image of a person, an acquaintance of them

would be able to name them, see Fig. (1.1) for the demonstration of AM. These are examples of

input-output pairs that are associated in our memory, where prompting for an element of the

pair results in content-addressable retrieval of the other element.

Figure 1.1: The form of Associative Memory discussed in this tutorial uses an energy function
to unify three important aspects of human cognition: association, memory, and error correction.
We are capable of associating images, sights, sounds, smells, and symbols with each other. These
associations allow us to retrieve memories using partial or corrupted information, making it a
content-addressable memory with error-correction capabilities. The functionality of Associative
Memory is modeled by an energy function, where low values of the energy correspond to stored
memories and constitute the most likely states of the system.

1

CHAPTER 1.

INTRODUCTION

2

Another important aspect of Associative Memory is the notion of error correction. You can

easily read the text in Fig. (1.1) without much difficulty, despite the fact that almost no words in

that paragraph are proper English words. The reason why you are able to comprehend this text

is because there are powerful error correction mechanisms that are constantly working in your

brain that associate imperfect inputs with the proper semantic meaning of individual words. The

same applies to the example above: the image of the strawberry can be presented with all kinds

of distortions and imperfections. Despite all that variability in the input, Associative Memory

manages to link those inputs with the proper smell and taste.

Thus, Associative Memory is a content addressable information storage system that
is capable of error correction.

Associative Memory plays a major role in the history of AI. Following the 1943 model of artificial

neuron by McCulloch and Pitts [2], and a body of work [3] on artificial neural networks (ANNs) �

Perceptrons � by Frank Rosenblatt in the 1950-1960s, the public community at large has been

extremely enthusiastic about the future. Popular media outlets from that period promised that

the Perceptron �will be able to walk, talk, see, write, reproduce itself and be conscious of its

existence� [4], similar to what we read in popular press about AI today. However, in 1969, Minsky

and Papert demonstrated that simple Perceptrons (without hidden layers) could not compute

even the simplest logical gates, e.g., XOR [5]. The public perception of this result led to the drop

of enthusiasm in ANNs. Most of the computer science community at that time left the field of

ANNs � triggering what historians of science later called the �AI Winter� [6].

John Hopfield�s seminal paper of 1982 [7] on what is now called the Hopfield network of Associative

Memory was the major driving force that ended that period. In his paper, Hopfield connected

computational aspects of Associative Memory with collective properties of Ising [8] magnets in

condensed matter physics, which were a �hot topic� at the time. Specifically, Hopfield posed a
simple quantifiable problem: Given a network of D neurons, how much information (or memories)
can such a network store and retrieve? Content-addressable Associative Memory retrieval was a

sufficiently non-trivial problem to illustrate the potential of ANNs� computational abilities. At

the same time, it was simple enough to be analytically solvable using powerful methods developed

in statistical physics. The confluence of these two aspects created a �harmonic oscillator� level

abstraction for ANN computation, and set the grounds for many extensions and generalizations

that followed.

Associative memory has been a prominent theme of ANN research in the 1960s-1980s. A highly

incomplete and subjective list of seminal papers from that period includes: Anderson [9], Willshow,

Buneman, Longuet-Higgins [10], Amari [11; 12], Cohen and Grossberg [13], Hopfield [14], Amit,

Gutfreund, Sompolinsky [15], and many others.

The main focus of this tutorial is on the Energy-based Associative Memories. This is the
class of ANNs which are recurrent neural networks that can be described by a state vector, which

evolves in time according to some non-linear rule. This state vector can be either continuous

or discrete. The update rule can be written either in terms of continuous time (differential

equation), or discrete set of update steps (which we will usually treat as a discretization of that

differential equation). Finally, there exist multiple options for how one updates the state vector.

CHAPTER 1.

INTRODUCTION

3

The most common choices are: synchronous � all compute elements of the state vector are
updated simultaneously, or asynchronous � at any given time a subset of all elements of the state
vector are updated, i.e., a random element of the state vector is updated while the remaining

elements are kept intact. For the purposes of this tutorial, we will most work with continuous
states, continuous time, and synchronous updates. Thus, the state vector of the network x
D,
which has individual elements xi (index i runs from 1 to D), evolves according to the following
differential equation:

R

?

dxi
dt

= fi(x, t)

(1.1)

where the functions fi(x, t) represent the vector field that defines the dynamics. We will refer
to individual elements of this vector as �neurons� although in many situations these variables

may describe a different biological structure, e.g., an astrocyte or their processes (long tentacles

originating from the astrocyte�s cell body).

A general system of coupled non-linear differential equations may have many complex behaviors:

fixed points, limit cycles, strange attractors, or chaotic behavior. Energy-based AMs are a special

subclass of general systems (1.1) that have the notion of an energy function (sometimes also

referred to as a Lyapunov function). You can think of the temporal evolution of the state vector

as a ball rolling downhill in a sophisticated energy landscape as seen in Fig. (1.1). The energy

is bounded from below, and the ball is only allowed to move in a way that decreases its energy.

Because of these restrictions, eventually, the ball must either stop at one of the local minima

or reach a manifold that corresponds to flat energy. In the latter case, the ball may continue to

move along that manifold as long as the energy does not increase.

The local minima of the energy (which can be point-like attractors � zero-dimensional manifolds
� or alternatively, manifolds of higher dimension) are called memories. The process of shaping
the energy landscape corresponds to writing information into the AM network or learning. The

dynamical trajectory of energy descent, illustrated by Eq. (1.1), corresponds to memory recall, or
inference. Association happens between the initial state of the network x(t = 0) and the final
). Finally, the asymptotic states of the dynamics are
asymptotic state of the network x(t
typically stable (unless they lie on the flat portions of the energy landscape). Intuitively, this

? ?

means that small perturbations that do not push the state vector outside the basin of the fixed

point�s attraction gets auto-corrected by the network itself. For these reasons, this network is an

AM system.

In some settings, memories in this system may correspond to individual instances of the training

data. Alternatively, they may correspond to emergent attracting manifolds that are shaped by

the learning algorithm (e.g., backpropagation [16; 17], Hebbian learning [18], contrastive training

[19; 20], etc.). In the latter case, the memories do not typically correspond to individual instances

of the training data, but rather describe consolidated memories � �knowledge� � that the

network acquires through synergy of AM architecture, a specific learning algorithm, and the

choice of training data.

Intuitively, you can think about the initial state x(t = 0) as a �question� that you ask the neural
network. This question positions the state vector at some high-energy location on the energy

landscape. The network will perform computation by moving that state to a local minimum

CHAPTER 1.

INTRODUCTION

4

(or a metastable state) � the process of �thinking.� Once the local minimum is reached, the

computation stops and the network�s state does not evolve in time anymore. You can read out
that final state x(t
) and convert it to the answer to the posed question. Importantly, this
computation is very different from conventional feed-forward architectures, e.g., feed-forward

? ?

convolutional neural networks, transformers, or large language models without chain of thought.

These conventional architectures are described by a computational graph with a finite number of

steps. This means that if the network has 10 layers, it must produce some kind of an answer

to the question after exactly 10 steps. This happens regardless of the complexity of the posed

question. AM architectures are very different. They can dynamically adapt the computational

graph based on the complexity of the posed question. For simple questions the network may

produce an answer in 5 steps, but for more complicated questions the network may need to �think�

longer.

Finally, because of the network�s energy-based architecture, the final answer is asymptotically
stable. This means that once the computation or �thinking� stopped and the network converged
to an answer, the precise timing of the output�s readout doesn�t matter. Assuming that the
readout time T is large enough, we can use x(t = T ) or x(t = T + 0.5 seconds) as the final answer,
and the two must be identical. This property of asymptotic stability makes AM framework

extremely appealing for neuromorphic devices, where hardware imperfections may prevent the

ability to read the network�s state at a precise timing.

In the past few years there has been significant advances in the field of AMs. These advances
pertain to the development of Dense Associative Memories or DenseAMs [21]. They are
flexible energy-based AM architectures that are capable of storing large amounts of information,

enable incorporation of many useful inductive biases (e.g., convolutions [22], attention [23], etc.)

in their architecture, and have mathematically controllable properties of emergent local minima.

DenseAM ideas have triggered a large amount of innovative ideas about the potential use cases of

AMs and we believe they will enable a new frontier for AM research [24]. In this tutorial, we

will cover many of these new developments from both the theoretical perspective and practical

implementations. The tutorial is supplemented with a collection of notebooks and suggested

problems that the readers can explore on their own to better understand the core ideas and

methods, and to get hands-on experience coding an AM network suitable for their own use case.

We intentionally designed these problems so that they are simple but still illustrate a useful

mathematical concept and the wonderful idea of AM. We hope that you enjoy this learning

experience.

Chapter 2

Dense Associative Memory: Discrete State
Vector

This chapter introduces a popular class of AMs � Dense Associative Memory (DenseAM). This

family of models is a generalization of celebrated Hopfield networks. While Hopfield networks are

very elegant mathematical models that satisfy all of the AM requirements, they are known to

have a very small information storage capacity. As a result, DenseAMs are specifically designed

to retain all of the benefits within Hopfield networks, but rectify their small information storage

issue [21; 25].

As discussed earlier, AMs can be formulated both in discrete and continuous variables, and in

discrete or continuous time. In this chapter, we focus on DenseAMs with discrete state vector,

1

and discrete asynchronous updates. Specifically, we will be working with a set of discrete variables
, index i = 1, ..., D which compose a state vector ?. In addition to that, the network
?i =
will have K memory vectors ?� with index � = 1, ..., K. Each memory is a D-dimensional vector
with individual elements denoted by ?�
i

{�

}

.

The energy function is defined as:

E =

?

K

D

F

?�
i ?i

.

�=1
(cid:88)

(cid:16)

i=1
(cid:88)

(cid:17)

(2.1)

The goal of the network is to start at some initial state ?
high-energy state, and lower the energy by flipping the elements of the state vector. The dynamics

, which typically corresponds to a

(t=0)
i

of flipping stops when no further single element flip can reduce the energy. At that point, the

network has reached a local minimum of the energy. As usual, we will refer to the individual

elements of the state vector are neurons or spins.

In order to formalize this intuitive dynamical equation, pick a single neuron and define its state

5

CHAPTER 2. DENSE ASSOCIATIVE MEMORY: DISCRETE STATE VECTOR

6

at the next iteration as:

?

(t+1)
i

= Sign
(cid:20)
= Sign
(cid:20)

E

?i =

1, ?j?=i = ?

?

(cid:16)
K

F

?�
i +

�=1
(cid:88)
K

(cid:16)

D

j?=i
(cid:88)
D

(t)
j

?

(cid:17)

?�
j ?

(t)
j

?

(cid:17)

(cid:16)

K

F

�=1
(cid:88)

(cid:16)

?�
i +

?

D

j?=i
(cid:88)

(t)
?�
j ?
j

(cid:17)(cid:21)

(cid:17)(cid:21)

(2.2)

E

?i = +1, ?j?=i = ?

(t)
j

?

Sign

2 ?�

i F ?

(t)
?�
j ?
j

(cid:20)

�=1
(cid:88)

(cid:16)

j?=i
(cid:88)

(cid:17)

+ higher order subleading terms
(cid:21)

.

1 with states of all other neurons
This update rule compares the energies of two states: ?i =
clamped to their current values, and ?i = +1 with all the other neurons clamped. The Sign[
]
�
function assignes the state of the ith neuron to the one corresponding to the lowest energy among
these two possibilities. Finally, in the last line of Eq. (2.2) we have used the Taylor series to
F (x) + ?F ?(x) + higher order terms. It is legitimate to terminate
expand the function F (x + ?)
the expansion after the first term since ? =
1 is much smaller than the overlap between the
clamped part of the state vector and the memories.

?

?

�

This update rule is typically written in the following form:

?

(t+1)
i

= Sign
(cid:20)

K

D

?�
i f

�=1
(cid:88)

(cid:16)

j?=i
(cid:88)

(t)
?�
j ?
j

,

(cid:17)(cid:21)

(2.3)

where we dropped the factor of 2 in the argument of the sign function (it doesn�t play any role
), which is a derivative of the function
there) and introduced an activation function f (
) = F ?(
�
�
F (

) defining the energy.
�

The energy function (2.1) is a finite sum of smooth functions (we assume that the function F (
)
�
does not have singularities - infinite values for finite arguments) that depend on the finite number

of discrete variables. Thus, the energy is finite and bounded from below. Additionally, the

dynamical equations (2.2) and (2.3) decrease the value of energy at each iteration. Thus, if we

keep applying these update equations to the state vector for a long time, eventually the system

will reach a steady state � no single neuron flip can further reduce the energy.

2.1

Information Storage Capacity

How many memories or local minima can such a system store and successfully retrieve? The
network, specified by Eq. (2.1), can be defined for any number K of memories. But, it turns out,
if you pack too many of such vectors inside the D-dimensional discrete space, the local minima of
the energy will no longer correspond to the stored patterns. In what follows we will compute the
largest value of K that permits successful remembering of the stored patterns.

In general, this maximal value Kmax will depend on the specific choices for the stored memories.
We will derive a statistical scaling law for this memory capacity assuming that the patterns are

CHAPTER 2. DENSE ASSOCIATIVE MEMORY: DISCRETE STATE VECTOR

7

drawn at random from the following distribution:

?�
i =

+1, with probability 1
2
1, with probability 1
2

?
?

?

(2.4)

With this distribution, it is easy to compute the correlation functions for these variables. The

?

one-point and two-point correlation functions are given by:

?�
i ?

?

= 0,

?�
i ??

j ?

?

= ?�??ij

(2.5)

In order to quantify the information storage capacity of this network we will use the following
trick. We will initialize the network in the state corresponding to one of the memories, say ?1
i
and let it evolve in time according to the update rule. If the pattern ?1
corresponds to a local
i
minimum, that state must be stable. In other words, the dynamics should not change that initial

,

state. Mathematically, this means that

(t+1)
?
i

?1
i f

= Sign
(cid:20)

(cid:16)

j?=i
(cid:88)

= Sign
(cid:20)

?1
i f

D

?
(cid:16)
signal

(cid:124)

(cid:123)(cid:122)

D

K

D

j ?1
?1
j

+

?�
i f

?�
j ?1
j

(cid:17)

+

�=2
(cid:88)
K
?�
i f

(cid:16)

j?=i
(cid:88)
D
?�
j ?1
j

(cid:17)(cid:21)

?
= ?1
i

�=2
(cid:88)

(cid:124)

j?=i
(cid:16)
(cid:88)
noise

(cid:123)(cid:122)

(cid:21)

(cid:17)

(cid:125)

1

(cid:17)

(cid:125)

(2.6)

Derivation of the generating function

It is helpful to introduce a new variable

j=2
(cid:88)
and compute the generating function defined as a statistical average of the exponent of

D

?1
j

? =

(2.7)

that variable

Since ?1
j
and computed explicitly

are independent for different indices j, the statistical average can be factorized

M (? ) =

e? ?

?

?

(2.8)

M (? ) =

1
2D?1

...

e? ?2e? ?3...e? ?D = cosh(? )D?1

(2.9)

?D=�1
(cid:88)
All correlation function can be computed by taking derivatives of the generating function.

?3=�1
(cid:88)

?2=�1
(cid:88)

For instance

?2p

?

=

?

?2pM
?? 2p

= (2p

?

1)!!Dp

(cid:12)
? =0
(cid:12)
(cid:12)
(cid:12)
(cid:12)

(2.10)

CHAPTER 2. DENSE ASSOCIATIVE MEMORY: DISCRETE STATE VECTOR

8

) is non-negative, the signal term pushes the argument of the Sign
Assuming that the function f (
�
function towards aligning it with the desired pattern ?1
. The noise term generally pushes that
i
argument away from the desired pattern and in some situations may outweigh the signal term.

Below, we will compute the characteristic magnitude of the noise term and determine when it

becomes dominant and destroys the stability of the target memory. Specifically, we can compute

the mean and variance of the noise term. The mean

noise

=

?

?

K

D

?�
i f

?�
j ?1
j

= 0

(cid:28)

�=2
(cid:88)

(cid:16)

j?=i
(cid:88)

(cid:17) (cid:29)

(2.11)

is equal to zero since index i appears only once in the correlator, see Eq. (2.5). The variance is
given by

noise2

?

=

?

K

D

K

D

?�
i f

?�
j ?1
j

??
i f

k ?1
??
k

�=2
(cid:88)

(cid:28)
K

=

f

(cid:16)
D

j?=i
(cid:88)
?�
j ?1
j

�=2 (cid:28)
(cid:88)

(cid:16)

j?=i
(cid:88)

(cid:17)

?=2
(cid:88)
D

(cid:16)

k?=i
(cid:88)

(cid:17) (cid:29)
K

i.d.
=

D

f

?�
j

f

D

?�
k

(2.12)

(cid:17) (cid:29)

�=2 (cid:28)
(cid:88)

(cid:16)

j?=i
(cid:88)

(cid:17)

(cid:16)

k?=i
(cid:88)

(cid:17) (cid:29)

?�
k ?1

k

f

2

(cid:16)

k?=i
(cid:88)

,

D

(cid:17)

?�
j

= (K

?

1)

f

(cid:68)

(cid:16)

j?=i
(cid:88)

(cid:17)

(cid:69)

where we used that

?�
i ??

i ?

?

= ?�? and the property that in distribution ?�

j ?1
j

i.d.
= ?�
j

.

Now, it is instructive to restrict our calculation to the class of power energy functions so that

) =

F (
�

1
n

)n,
(
�

)n?1,
) = (
f (
�
�

where n is an integer.

(2.13)

In this case, the variance of the noise can be computed exactly (through the generating function)

and is equal to1 [26]:

?2 =

?

noise2

= (2n

?

?

3)!!KDn?1 .

(2.14)

Figure 2.1: Gaussian probability distribution function. Shaded area indicates the probability of
an error or spin flip.

Now we are ready to compute the probability of an error. The noise term in Eq. (2.6) is a sum

1We assume that K is large so that K

1

?

?

K.

CHAPTER 2. DENSE ASSOCIATIVE MEMORY: DISCRETE STATE VECTOR

9

over many independent random variables. When K and D are large, this noise term behaves
approximately as a Gaussian random variable. When the sign of the noise term is the same as

the sign of the signal, the noise term pushes the update in the right direction and does not cause

issues. The problem arises when the noise is large and its sign is opposite to that of the signal.

In this situation, it is possible that the noise can outweigh the signal and flip the spin of interest.

The probability of this event is given by the area under a Gaussian distribution, as shown in

Fig. (2.1):

P (error) =

?

(cid:90)f (D?1)

dx
?2??2

e? x2

2?2 =

?

(cid:90)f (D?1)
?

e? y2

2 = g

dy
?2?

1)

f (D

?
?

(cid:17)

(cid:16)

< 1% .

(2.15)

Thus, if we want the probability of error be smaller than a certain value the following inequality

must be satisfied :

where ? is a numerical constant independent of K, D, and n (for 1% error ?
translates into the following bound for the number of memories :

?

2.576). This

f (D

?

1) > ??,

(2.16)

K < Kmax =

1
?2(2n

?

Dn?1 .

3)!!

(2.17)

Thus, as long as the number of memories is smaller than Kmax, the network initialized in one of
the memories remains there and the dynamics does not flow away from it. It turns out, this is

precisely the point when associative memory recall breaks. If the number of memories is smaller
than Kmax our network works as intended. Once K exceeds Kmax, reliable recall breaks. This
does not mean that the network becomes useless in that regime. In fact, it instead becomes a

generative model. We will discuss this aspect later.

What have we learned so far?

� The number of memories K is upper bounded.

� The Memory storage capacity heavily depends on the shape of the energy function

) and the shape of the activation function f (
F (
�

).
�

� The sharper the energy peaks around memories � the larger the memory storage

capacity.

2.2 Limiting Cases

It is instructive to study a few limiting cases of the general family Eq. (2.1). Each of these models

are frequently studied in the literature and have distinct properties.

The Hopfield Model n = 2. The simplest, and the most popular, example of the Dense
Associative Memory is the Hopfield model. One can obtain it from the general form Eq. (2.1)

CHAPTER 2. DENSE ASSOCIATIVE MEMORY: DISCRETE STATE VECTOR

10

choosing the function as F (
�

)2. The energy function can be written as
) = 1
2 (
�

E =

1
2

?

K

D

?�
i ?i

�=1 (cid:16)
(cid:88)

i=1
(cid:88)

(cid:17)

2

=

1
2

?

D

i,j=1
(cid:88)

?iTij?j,

where

Tij =

i ?�
?�
j .

(2.18)

K

�=1
(cid:88)

In this case, according to the general result Eq. (2.17), the memory storage capacity scales linearly

with the size of the network:

Kmax

D .

?

(2.19)

This is the famous Kmax
0.14D scaling law from the Hopfield�s 1982 paper [7], derived by [15]
using tools from statistical mechanics. While this model is appealing from the perspective of

?

mathematical elegance and simplicity, this scaling law presents a major practical limitation. In

the end, the hallmark of modern AI applications is the ability to store and process large amounts

of information, a property severely limited by this scaling law.

DenseAM with n = 3. Fortunately, this problem disappears for a more rapidly peaking energy
function (obtained via an alternative activation function). For F (
)3, for example, the
energy is given by

) = 1
3 (
�
�

E =

1
3

?

K

D

?�
i ?i

3

=

1
3

?

i=1
(cid:88)
and the memory storage capacity scales as:

i,j,k=1
(cid:88)

�=1 (cid:16)
(cid:88)

(cid:17)

D

Tijk?i?j?k,

where

Tijk =

i ?�
?�

j ?�
k ,

(2.20)

K

�=1
(cid:88)

which is significantly faster than linearly.

Kmax

D2,

?

(2.21)

DenseAM with F (
�
memory storage capacity. For exponential function F (
this DenseAM can store and retrieve scale as:

). It turns out that one can even achieve the exponentially large
�
) [27; 28], the number of memories that
�

) = exp(

Kmax

D
2 ,

2

?

(2.22)

which is more than sufficient for storing any practically relevant amount of information. Note,

this number is the square root of the total number of binary states of the network. Despite its

huge memory storage capacity, this model retains strong error correcting capabilities and has

large size basins of attraction around each stored memories.

2.3 General Dense Associative Memory with Binary State Vari-

ables

Although simple models represented by Eq. (2.1) illustrate the computational capacilities of Dense

Associative Memories, more general energy functions are also frequently studied. For binary

CHAPTER 2. DENSE ASSOCIATIVE MEMORY: DISCRETE STATE VECTOR

11

DenseAM models, the general form of the energy function is given by

K

F

S

?�, ?

,

(2.23)

E =

Q

?

(cid:104)

�=1
(cid:88)

(cid:16)

(cid:3)(cid:17)(cid:105)

(cid:2)
where the function F (
)n or
) is a rapidly growing separation function (e.g., power F (
�
exponent), S[x, x?] is a similarity function (e.g., a dot product or a Euclidean distance), and Q is
a scalar monotone function (e.g., linear or logarithm). There are many possible combinations
of various functions F (
) that lead to different models from the DenseAM
), and Q(
), S(
�
�
family [21; 27; 29; 30; 31; 32]. We will discuss the relationship between these binary models and

) = (
�
�

,
�

�

DenseAMs with continuous states in the next Chapter.

Notebook 2.1: Storage and recovery of memories in DenseAM

In this notebook, we offer the reader the possibility to experience storage and retrieval of

patterns in DenseAM models. A set of simple Pokemon images can be embedded in the

memory pool of the model. The model can then be queried by a corrupted version of a

memory. The dynamical trajectory of the recall process retrieves the desired memory. By
varying parameter n the reader can experience both successful recovery of the memories
and memory failures, when the recovered image does not correspond to the desired memory.

All these numerical results are to illustrate the general theory discussed in this chapter.

Checkout the notebook as a blog post, a colab notebook or as a raw .ipynb file.

Minimize energyHow many patterns can we store?Chapter 3

General Dense Associative Memory

In the previous Chapter, we introduced Dense Associative Memories with discrete state vectors.

While mathematically aesthetic and simple to analyze, such models do not allow backpropagation

training � due to their discrete nature. It turns out, most of the desired properties of Dense

Associative Memories with discrete states are inherited by models with continuous variables.

Moreover, the discrete state models can be derived as limiting cases of the continuous models.

There are two other limitations of the models represented by Eq. (2.1). First, they do not have

the hierarchical structure of representations, a crucial aspect which limits their ability to handle

complex patterns from real-world datasets. Second, they have a rigid energy function � although
the energy depends on the learnable parameters ?�
i
of patterns and relationships the network can model.

� its specific form may constrain the types

In this Chapter, we introduce �building blocks� of Dense Associative Memories. Specifically, we
develop a modular energy perspective where the energy of any model from this family can be
decomposed into standardized components: neuron layers that encode dynamic variables and
hypersynapses that encode their interactions. The total energy of the system is the sum of the
individual component energies subtracted by the energies of all the layers and all the interactions
between those layers. This framework of energy-based building blocks for memory not only clarifies
how existing methods relate to each other, but also provides a systematic language for designing

new architectures. We demonstrate the flexibility of this abstraction by showing how it helps to

formulate all of the known models from this family, including Hierarchical Associative Memories

[33], Energy Transformers [34], neuron-astrocyte networks [35], and many others.

We refer to this generalized abstraction of Energy-based AMs as HAMUX [36] after the software
library that introduced it (here, HAMUX stands for �Hierarchical Associative Memory User
eXperience�). We emphasize, however, that the abstraction is more fundamental than its specific
software implementation.

3.1 Building Blocks of AMs with Modular Energies

HAMUX builds deep AMs by summing the modular energies of neuron layers and hypersynapses.

A neuron layer captures a non-linearity in the network (e.g., ReLU, sigmoid, tanh, softmax,
layernorm, etc.). We call these non-linearities activations, and their inputs or pre-activations
serve as the dynamic variables of the system. For example, a neuron layer can capture the

12

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

13

Figure 3.1: HAMUX hypergraph diagrams are a graphical depiction of an AM whose
total energy is the sum of the neuron layer (node) and hypersynapse (hyperedge) energies.
Inference is done recurrently, modeled by a system of differential equations where each neuron
layer�s hidden state updates to minimize the total energy. When all non-linearities are captured in
the dynamic neurons, inference becomes a local computation that avoids differentiating through
non-linearities.

computation �x = ReLU(x), which has activations �x and pre-activations x which serve as the
dynamic internal state for this neuron layer. Structurally, neuron layers are the nodes of our
energy-based computation graph.

A hypersynapse is a parameterized energy function that captures how similar or aligned the
activations of its connected neuron layers are. For example, a simple hypersynapse may take the
form ES(�x, �y; ?) = �x??�y, where ? is a synaptic weight matrix. The gradient of ES w.r.t. �x or
�y looks like a Dense linear transformation, though more complex synaptic energies can be chosen
to look like Conv, Pooling, or even Attention layers. Hypersynapses define the interactions
between neurons and are the hyperedges of our energy-based computation graph.

For a system of L neuron layers and S hypersynapses, the total energy of the system is

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

Etotal =

L

?=1
(cid:88)

S

Eneuron

?

+

Esynapse

s

.

s=1
(cid:88)

14

(3.1)

The total energy is structured such that the activations of a neuron layer affect only connected
hypersynapses and itself. Let �x? and x? represent the activations and internal states of neuron
layer ?, and let N(?) represent the set of hypersynapses that connect to neuron layer ?. The
following update rule describes how neuron internal states x? minimize the total energy using
only local signals

??

dx?
dt

=

?Etotal
? �x?

?

=

?Esynapse
s
? �x? ?

?

?Eneuron
?
? �x?

=

x? ?
I

x?,

(3.2)

? ?

?

(cid:88)s?N(?)

?

where

is the total synaptic input current to neuron layer ?, which is
fundamentally local and serves to minimize the energy of connected hypersynapses. See sections

x? :=
I

s?N(?) ?

�x?Esynapse

?

s

(3.2) and (3.3) to understand the above equation in more detail. The time constant for neurons
in layer i is denoted by ??. When the activations �x? are bounded, the above system is guaranteed
to converge for any choice of hypersynapse energies.

(cid:80)

3.2 Dynamical Neurons and their Lagrangians

A neuron layer or node is a fancy term to describe the dynamic variables in AM. Each neuron
layer has an internal state x which evolves over time and an activation �x that forwards a signal
to the rest of the network. Think of neurons like the activation functions of standard neural
networks, where x are the �pre-activations� and �x are the outputs e.g., �x = ReLU(x).

In order to define neuron�s layer energy, AMs employ two mathematical tools from physics: convex
Lagrangian functions and the Legendre transform. For each neuron layer, we define a convex,
scalar-valued Lagrangian
of this Lagrangian produces the dual
variable �x (our activations) and the dual energy Ex(�x) (our new energy) as in:

x(x). The Legendre transform

L

T

�x =

Ex(�x) =

(

T

x) =

L

?L
x, �x
?

x(x)

(activation function)

x(x)

? ? L

(dual energy)

(3.3)

where

,
?�
2

x is convex, the Jacobian of the activations
is the element-wise inner product. Because
L
�?
x(x) (i.e., the Hessian of the Lagrangian) is positive definite. This important point is

? �x
?x =
summarized in Fig. (3.1).

?

L

The dual energy Ex(�x) has another nice property: its gradient equals the hidden states. Thus,
when we minimize the energy of our neurons (in the absence of any other signal), we observe

exponential decay. This is nice to keep the dynamic behavior of our system bounded and
well-behaved, especially for very large values of x.

dx
dt

=

??

�xEx(�x) =

x.

?

(3.4)

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

15

Summary For non-physicists, the terminology used in this section can be daunting. The key
insight is simple: a neuron layer is just a convex function
x (the Lagrangian) applied to an
internal state x. The Legendre transform of this Lagrangian then automatically provides two
x(x), and (2) the dual energy representation Ex(�x).
things: (1) the activation function �x =
This mathematical machinery abstracts away some of the complexity of non-linearities and gives

?L

L

us a simpler system to work with.

Proof: Energy gradient equals hidden states

Show that ?Ex(�x)

? �x = x.

?Ex(�x)
? �x

=

?
? �x

x, �x
(
?

x(x))

? ? L
?

?x
? �x ?
?x
? �x ?

x(x)
L
?x
?x
? �x

�x

?x
? �x

= x + �x

= x + �x

= x

3.3 Hypersynapses

The activations of one neuron layer are sent to other neurons via communication channels called
hypersynapses. At its most general, a hypersynapse is a scalar valued energy function defined on
top of the activations of connected neuron layers. For example, a hypersynapse connecting neuron
layers X and Y has an interaction energy Exy(�x, �y; ?), where ? represents the synaptic weights
or learnable parameters. Exy(�x, �y; ?) encodes the desired relationship between activations �x
and �y: when this energy is low, the activations satisfy the relationship encoded by the synaptic
weights ?. During energy minimization, the system adjusts the activations to reduce all energy
terms, which means synapses effectively �pull� the connected neuron layers toward configurations

encoded in the parameters that minimize their interaction energy.

Hypersynapses in the HAMUX framework differ from biological synapses in two fundamental

ways:

1. Hypersynapses can connect any number of layers simultaneously, while biological
synapses connect only two neurons. This officially makes hypersynapses �hyperedges� in

graph theory terms.

2. Hypersynapses are undirected, meaning that all connected layers influence each other
bidirectionally during energy minimization. Meanwhile, biological synapses are unidirec-

tional, meaning signal flows from a presynaptic to postsynaptic neuron.

Because of these differences, we choose the distinct term �hypersynapses� to distinguish them

from biological synapses.

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

16

Figure 3.2: Hypersynapses are represented as undirected (hyper)edges in a hyper-
graph. Shown is an example pairwise synapse, which is a single energy function Exy(�x, �y; ?)
defined on the activations �x and �y from connected nodes, which necessarily propagate signal to
both connected nodes. Here, signal is defined as the negative gradient of the interaction energy
�xExy(�x, �y; ?)).
w.r.t. the connected layer�s activations (e.g., layer X receives signal
This is in contrast to biological synapses which are directional and only propagate signal in one
direction from layer X to Y, needing a separate synapse to bring information back from Y to X.

x =

??

I

Hypersynapse notation conventions

For synapses connecting multiple layers, we subscript with the identifiers of all connected

layers. For example:

� Exy � synapse connecting layers X and Y

� Exyz � synapse connecting layers X, Y, and Z.

� Exyz... � synapses connecting more than three layers are possible, but rare.

However, synapses can also connect a layer to itself (self-connections). To avoid confusion
with neuron layer energy Ex, we use curly brackets for synaptic self-connections. For
example, E{x} represents the interaction energy of a synapse that connects layer X to itself.

Because almost every interaction energy is parameterized in some way, we generally omit
? from the notation in subsequent sections when it�s not central to the discussion

The undirected nature of hypersynapses fundamentally distinguishes AM from traditional neural

networks. Whereas feed-forward networks follow a directed computational graph with clear

input-to-output flow, AMs have no inherent concept of �forward� or �backward� directions. All

connected layers influence each other bidirectionally during energy minimization, with information

propagating from deeper layers to shallower layers as readily as the other way around. See

Fig. (3.1) for a visual illustration.

Unlike the neuron layer�s energies, the interaction energies of the hypersynapses are completely
unconstrained: any function that takes activations as input and returns a scalar is admissable and
will have well-behaved dynamics1. The interaction energy of a synapse may choose to introduce

1Some energies could be more meaningful than the others.

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

17

its own non-linearities beyond those handled by the neuron layers. When this occurs, the energy

minimization dynamics must compute gradients through these �synaptic non-linearities�, unlike

the case where all non-linearities are abstracted into the neuron layer Lagrangians.

3.4 Energy Descent Dynamics

The central result is that dynamical equations Eq. (3.2) decrease the global energy of the network

Eq. (3.1). In order to demostrate this, consider the total time derivative of the energy

dEtotal
dt

=

L

i=1
(cid:88)

?Etotal
? �xi

? �xi
?xi

dxi
dt

=

?

?i

dxi
dt

?2
x
L
?xi?xi

dxi
dt ?

0,

L

i=1
(cid:88)

(3.5)

where we expressed the partial of the energy w.r.t. the activations through the velocity of the
neuron�s internal states Eq. (3.2). The Hessian matrix ?2Lx
has the size number of neurons
?xi?xi
in layer i multiplies by the number of neurons in layer i. As long as this matrix is positive
semi-definite, a property resulting from the convexity of the Lagrangian, the total energy of the

network is guaranteed to either decrease or stay constant � increase of the energy is not allowed.

Additionally, if the energy of the network is bounded from below, the dynamics in Eq. (3.2) are

guaranteed to lead the trajectories to fixed manifolds corresponding to local minima of the energy.

If the fixed manifolds have zero-dimension, i.e., they are fixed point attractors, the velocity field

will vanish once the network arrives at the local minimum. This correspondes to Hessians being

strictly positive definite. Alternatively, if the Lagrangians have zero modes, resulting in existence

of zero eigenvalues of the Hessian matrices, the network may converge to the fixed manifolds, but

the velocity fields may stay non-zero, while the network�s state moves along that manifold.

3.5

Implementing AMs

We have established how the computational graph is built and the rules for how neuron layers

and hypersynapses are constructed. We now discuss how the above mathematical framework can

be used to recreate some of the commonly used AM models.

Exercise 3.1: Designing the energy for a custom DenseAM

Problem Consider a DenseAM model consisting of D neurons with the following activa-
tion function �xi = tanh(?xi). Design the synaptic energy and the global energy to recreate
.
DenseAM with discrete variables discussed in Eqs. (2.1) and (2.3), in the limit ?
? ?

Solution

First, define the Lagrangian for this network so that its partial gives the desired activation

=

L

1
?

D

log

cosh(?xi)

,

resulting in

�xi = tanh(?xi)

(3.6)

i=1
(cid:88)

(cid:16)

(cid:17)

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

18

The synapse connects neuron layer to itself and its synaptic energy is given by

Esynapse =

K

D

F

?�
j �xj

�=1
(cid:88)

(cid:16)

j=1
(cid:88)

(cid:17)

?

The total energy of the network is

Etotal = Eneuron + Esynapse =

D

�xixi

(cid:104)

i=1
(cid:88)

K

D

F

?�
j �xj

�=1
(cid:88)

(cid:16)

j=1
(cid:88)

(cid:17)

? L

?

(cid:105)

The dynamical update equation Eq. (3.2) is given by

(3.7)

(3.8)

K

D

=

?�
i f

?�
j �xj

?

dxi
dt

?

xi

(3.9)

(cid:16)
Now, let�s discretize time. Set ? = 1 and write the above equation in finite differences
(dt = 1). The result is

(cid:17)

�=1
(cid:88)

j=1
(cid:88)

which leads to

xt
i

xt+1
i ?
dt

=

K

D

?�
i f

�=1
(cid:88)

(cid:16)

j=1
(cid:88)

?�
j �xt
j

xt
i

?

(cid:17)

K

D

xt+1
i =

?�
i f

?�
j �xt
j

(3.10)

(3.11)

Finally, express everything through the activations �xi and take the limit ?
. In this
limit �xi = ?i = Sign(xi) and the energy of the layer vanishes, resulting in the total energy

? ?

�=1
(cid:88)

(cid:16)

j=1
(cid:88)

(cid:17)

Etotal =

?

K

D

F

?�
j ?t
j

�=1
(cid:88)

(cid:16)

j=1
(cid:88)

(cid:17)

(3.12)

The discrete update equation can be obtained by acting with the Sign(
sides of Eq. (3.11), resulting in Eq. (2.3).

) function on both
�

3.5.1 Energy Transformer Block

We now explain how the techniques developed above can be used for building the Energy

Transformer (ET) architecture [34]. For clarity of presentation, we use language associated with

the image domain, although this architecture can also be used for language or graphs with minimal

modifications.

The overall pipeline is similar to the Vision Transformer networks (ViTs) [37]. An input image is

split into non-overlapping patches. After passing these patches through the encoder and adding the

positional information, the semantic content of each patch and its position is encoded in the token
xiA. In the following the indices i, j, k = 1, ..., D are used to denote the token vector�s elements,

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

19

Figure 3.3: Inside the ET block. The input tokens x(t) passes through a sequence of operations
and gets updated to produce the output tokens x(t+1). The operations inside the ET block are
carefully engineered so that the entire network has a global energy function, which decreases with
time and is bounded from below. In contrast to conventional transformers, the ET-based analogs
of the attention module and the feed-forward MLP module are applied in parallel as opposed
to consecutively. Right: The ET block recurrently minimizes the energy of a corrupted image
represented by a collection of tokens, where 50% of the tokens are occluded. Shown is an image
not seen when training the ET block.

indices A, B, C = 1, ..., N are used to enumerate the patches and their corresponding tokens. It is
helpful to think about each image patch as a physical particle, which has a complicated internal
state described by a D-dimensional vector xA. This internal state describes the identity of the
particle (representing the pixels of each patch), and the particle�s positional embedding (the

patch�s location within the image). The ET block is described by a continuous time differential
equation, which describes interactions between these particles. Initially, at t = 1 the network is
given a set containing two groups of particles corresponding to open and masked patches. The

�open� particles know their identity and location in the image. The �masked� particles only know

where in the image they are located, but are not provided the information about what image

patch they represent. The goal of ET�s non-linear dynamics is to allow the masked particles to

find an identity consistent with their locations and the identities of open particles. This dynamical

evolution is designed so that it minimizes a global energy function. The identities of the masked

particles are considered to be revealed when the dynamical trajectory reaches the fixed point.

Thus, the central question is: how can we design the energy function that accurately captures the

task that the Energy Transformer needs to solve?

The masked particles� search for identity is guided by two pieces of information: identities of the

open particles, and the general knowledge about what patches are in principle possible in the

space of all possible images. These two pieces of information are described by two contributions

to the ET�s energy function: the energy based attention and the Hopfield Network. Below we

define each element of the ET block in the order they appear in Fig. (3.3).

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

20

Layer-Norm

Each token, or a particle, is represented by a vector x
operations inside the ET block are defined using a layer-normalized token representation:

D. At the same time, most of the

R

?

�xi = ?

xi

�x

?
xj

?

1
D

(cid:114)

j
(cid:80)

(cid:0)

+ ?i,

where

�x =

1
D

D

k=1
(cid:88)

xk

2

�x

+ ?

(cid:1)

(3.13)

The scalar ? and the vector elements ?i are learnable parameters, ? is a small regularization
constant. Following the general recipe of HAMUX, this operation can be viewed as an activation

function for the neural layer and can be derived as a partial derivative of the Lagrangian function:

(x) = D?

L

1
D

(cid:115)

2

�x

+ ? +

xj

?

(cid:1)

j
(cid:88)

(cid:0)

j
(cid:88)

?jxj,

so that

�xi =

?

(x)

L
?xi

(3.14)

See [30; 38; 33] for the discussion of this property.

Multi-Head Energy Attention

The first contribution to the ET�s energy function is responsible for exchanging information

between the particles (tokens). Similarly to the conventional attention mechanism, each token

generates a pair of queries and keys (ET does not have a separate value matrix; instead the value

matrix is a function of keys and queries). The goal of the energy based attention is to evolve

the tokens in such a way that the keys of the open patches are aligned with the queries of the
masked patches in the internal space of the attention operation. Below we use index ? = 1, ..., Y
to denote elements of this internal space, and index h = 1, ..., H to denote different heads of
this operation. With these notations the energy-based attention operation is described by the

following energy function:

EATT =

1
?

?

H

N

log

?

exp (?AhBC)

?

C=1
(cid:88)
where the attention matrix AhBC is computed from query and key tensors as follows:

B?=C
(cid:88)

h=1
(cid:88)

?

?

AhBC =

K?hB Q?hC,

K?hB =

Q?hC =

?
(cid:88)

j
(cid:88)

W K

?hj �xjB,

W Q

?hj �xjC,

j
(cid:88)
Y �H�D and WQ

H�N �N

R

A

?

Y �H�N

K

R

?

Y �H�N

Q

R

?

and the tensors WK
R
perspective of HAMUX, Eq. (3.15) is the energy of the synapse, which mixes layers of neurons

Y �H�D are learnable parameters. From the

R

?

?

or tokens.

From the computational perspective each patch generates two representations: query (given the

(3.15)

(3.16)

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

21

position of the patch and its current content, where in the image should it look for the prompts

on how to evolve in time?), and key (given the current content of the patch and its position, what

should be the contents of the patches that attend to it?). The log-sum energy function (3.15)

is minimal when for every patch in the image its queries are aligned with the keys of a small
number of other patches connected by the attention map. Different heads (index h) contribute to
the energy additively.

Hopfield Network Module

The next step of the ET block, which we call the Hopfield Network (HN), is responsible for

ensuring that the token representations are consistent with what one expects to see in realistic

images. The energy of this sub-block is defined as:

N

K

D

EHN =

?

G

?�j �xjB

,

B=1
(cid:88)

�=1
(cid:88)

(cid:16)

j=1
(cid:88)

(cid:17)

K�D

?

R

?

(3.17)

) is an integral
where ?�j is a set of learnable weights (memories in the Hopfield Network), and G(
�
of the activation function r(
). This formula is identical to the energy
), so that G(
)? = r(
�
�
�
Eq. (3.7). Depending on the choice of the activation function this step can be viewed either as
a classical continuous Hopfield Network [14] if the activation function grows slowly (e.g., r(
) =
�
ReLU), or as a Dense Associative Memory [21; 30] if the activation function is sharply peaked
around the memories (e.g., r(
) = power or softmax). The HN sub-block is analogous to the
�
feed-forward MLP step in the conventional transformer block but requires that the weights of

the projection from the token space to the hidden neurons� space to be the same (transposed

matrix) as the weights of the subsequent projection from the hidden space to the token space.
Thus, the HN module here is an MLP with shared weights that is applied recurrently. The energy
contribution of this block is low when the token representations are aligned with some rows of
the matrix ?, which represent memories, and high otherwise.

Dynamics of Token Updates

The inference pass of the ET network is described by the continuous time differential equation,

which minimizes the sum of the two energies described above. The whole ET network contains of

layers of tokens coupled through two types of synapses, attention synapse and Hopfield Network

synapse, so that

Etotal = Eneuron +

Esynapse

?

2

N

D

?=1
(cid:88)

xiA �xiA

i=1
(cid:88)

A=1
(cid:88)

(cid:104)
EATT + EHN + O(?)

A=1
(cid:88)

N

?

+ EATT + EHN

(xA)

L

(cid:105)

(3.18)

=

?

We work in the regime when the parameter ? in the definition of the layer-norm Lagrangian is
small � it only serves as a regularization to prevent the division by zero. In this limit, neuron

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

22

layer energy vanishes, and the total HAMUX energy is the sum of EATT and EHN

?

dxiA
dt

=

?Etotal
? �xiA

,

?

where

Etotal = EATT + EHN

(3.19)

Here xiA is the token representation (input and output from the ET block), and �xiA is its
layer-normalized version. The first energy is low when each patch�s queries are aligned with the

keys of its neighbors. The second energy is low when each patch has content consistent with the
general expectations about what an image patch should look like (memory slots of the matrix ?).
The dynamical system, represented by Eq. (3.19), finds a trade-off between these two desirable

properties of each token�s representation. For numerical evaluations, Eq. (3.19) is discretized in

time.

To demonstrate that the dynamical system (3.19) minimizes the energy, consider the temporal

derivative

dEtotal
dt

=

i,j,A
(cid:88)

?Etotal
? �xiA

? �xiA
?xjA

dxjA
dt

=

1
?

?

i,j,A
(cid:88)

?Etotal
? �xiA

M A
ij

?Etotal
? �xjA ?

0

(3.20)

The last inequality sign holds if the symmetric part of the matrix

M A

ij =

? �xiA
?xjA

=

?2
?xiA?xjA

L

(3.21)

is positive semi-definite (for each value of index A). The Lagrangian (3.14) satisfies this condition.

Notebook 3.1: Energy Transformer

In this notebook, we offer the reader the possibility to build the ET block in code following

the general rules of HAMUX. We have pre-trained this network on ImageNet and loaded

the weights of the model, so that the reader can quickly play with parameters and visualize

energy decent dynamics and learned representations at inference time. All these numerical

results are to illustrate the general theory discussed in this chapter.

Checkout the notebook as a blog post, a colab notebook or as a raw .ipynb file.

Learned patch patternsLearned position correlationsPatch ColumnPatch Row12345678910111213141234567891011121314Build and interpret the trained Energy TransformerCHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

23

3.6 Bridging Energy Minimization and Feedforward Prediction

Although associative memories fundamentally differ from feedforward networks, they can both be
used to solve the same tasks. Let ? describe the network parameters. Traditional feedforward
networks transform input tensors x
, where
y? = f?(x) represents the model�s prediction.
computational graph that maps input tensors z

In contrast, an AM builds an energy-based

to a scalar energy value via E? :

to output tensors y

via f? :

X (cid:55)? Y

? X

? Y

R.

Z (cid:55)?

? Z

Figure 3.4: Associative Memory is fully compatible with traditional prediction tasks.
By fixing a subset of variables (input) and minimizing the energy with respect to the remaining
variables, we can predict optimal values for the output variables. Left: a 2D energy landscape
used for the general Associative Memory task of cleaning an input. Right: a slice through the
total energy landscape represents the energy objective for a prediction task.

How can we use an AM for prediction? When input space
are distinct (e.g.,
in classification or segmentation), the energy function takes both spaces as input: E? :
R.
for which we want to know the best output y?
Given an input x
(as described in
Eq. (3.22)), prediction or inference becomes a coordinate-wise constrained energy minimization
problem where we fix one of the variables and minimize the energy with respect to the other.

and output space

X �Y (cid:55)?

? X

? Y

X

Y

y? = arg min
y?Y

E?(x, y)

(3.22)

of a feedforward network represents a noiseless, inpainted version

Sometimes the output space

Y

X

of

, as in masked-token prediction where x? = f?(x(0)). In this case, f? :

is doing

a proxy of �error minimization� inside its computation graph (which likely has many residual
connections), and the energy function is E? :

X (cid:55)?
The inference process of Eq. (3.22) is flexible and can be adapted for different prediction tasks.
Let�s view the optimization objective of Eq. (3.22) as a higher order function
applied to energy
function E?. In this way, we can describe different input-output mappings from the energy. For
instance, Eq. (3.22) is a mapping
that performs a global search of E? over
possible y given a clamped x. We can define other useful inference objectives. For example,

y(E?) :

X (cid:55)? Y

R.

x
(cid:1)

F

F

X (cid:55)? X

CHAPTER 3. GENERAL DENSE ASSOCIATIVE MEMORY

24

we can invert the search to instead search over x in the region of some initial guess x(0) given
a clamped y. This would be represented via
. Or we could jointly
optimize both variables given only an initial guess for both, as in

F(x,y)
(cid:1)

x(E?) :

X �Y (cid:55)? X �Y
The Hopfield Network generally considers this last scenario, but each inference process represents

X � Y (cid:55)? X
F(x,y)
(cid:1)

(x,y)(E?) :

.

a different way to extract solutions from the same underlying energy landscape.

Energy function vs. Loss function

The key idea in AMs is that every computation serves to optimize some objective. However,
we choose to distinguish between two types of objectives: the energy function and the
loss function. We say the energy function governs the dynamics of neuron states during
inference, while the loss function governs the dynamics of model parameters during training.
The primary difference is in whether the gradient is taken with respect to the states of the

network, or the parameters.

Chapter 4

Failure of Memory and Generative AI

In 1977, psychologists Roger Brown and James Kulik described a famous experiment, in which

respondents were asked to self-report the circumstances in which they found out about the

highly surprising and consequential news of the President John F. Kennedy assassination [39].

Among many insightful findings from this study, peculiar responses have been recorded containing

detailed, emotional, highly realistic, and convincing descriptions of learning about this news for

the first time that were factually inaccurate. For instance, one of the respondents (person A)

vividly describes how person B came down the stairs to the first floor of the house, while person

A was focusing on work and told person A that she heard about the assassination on the news.

This recollection is detailed enough to include specific phrases and portions of the conversation

between persons A and B during this recalled event. Although both persons are well familiar

with each other and their recollections are plausible, documented evidence suggests that person

A and person B could not be present in the same location after the JFK�s assassination [40; 41].

This is an example of misremembering, a phenomenon that is general and can be observed in

many other situations. For instance, during a crime investigation two eyewitnesses can give

mutually contradictory accounts of what they saw. Sometimes, both accounts can be different

from what has actually happened. These examples of misremembering highlight a failure mode

of human memory in which multiple observed events (training data) blend together and form

novel memories, which are different from any of the observed events (training data points).

Misremembering leads to creation of novel hypothetical memories, which in certain aspects share

a degree of similarity (correlated) to the training data, but are distinct from individual training

instances. Thus, misremembering can lead to creativity.

In generative AI, creativity is a key objective. For instance, diffusion models, being trained on
a sufficiently large set of training images, can generate genuinely novel photorealistic images of

previously unseen events [42; 43; 44; 45]. A typical diffusion model training pipeline contains of

two phases: the forward process when the noise is injected into the training samples, and the

reverse process when a neural network is used to predict how much noise should be removed

from the noisy sample with the goal to reconstruct the original uncorrupted training data point.

At runtime, random noise is passed through the reverse process and converted into generated

samples. The training pipeline of diffusion models can be conceptualized as the process of writing

the training data into a memory network [46; 47]. By doing so, the information about training

samples is written into the synaptic weights of the neural network that is used for denoising.

The reverse process can be conceptualized as an attempt of memory recall � that memorized

25

CHAPTER 4. FAILURE OF MEMORY AND GENERATIVE AI

26

information should be retrieved from the synaptic weights and turned into a generated sample.

It is well established that the memory recall from the diffusion models can be successful; in

that case the generated sample matches exactly at least one of the training samples. It can also

be unsuccessful; in that case the generated sample will be novel and will not match any of the

training samples. This is what is called the memorization-to-generalization transition in diffusion

models [48; 49; 50], which occurs when the size of the training set is increased. Successful memory

retrieval is typically viewed as a negative outcome in diffusion models and can often lead to

privacy and copyright violations. Similarly, in LLMs training samples can often be extracted

verbatim from the synaptic weights of the neural network, a property that has been a subject of

intense discussions in the research community and public discourse [48]. At the same time, LLMs

can also generate novel previously unseen responses. Importantly for us, in all these examples
creativity arises as a result of a failure of memory recall.

In Energy-based Associative Memory networks, memories are identified as local minima of an

energy landscape, and the process of memory recall is conceptualized as a dynamical trajectory

starting at a high energy state (corrupted memory) and leading to the best matching local minimum

(recovered memory). Misremembering arises when the recovered memory (local minimum) of the

energy function is different from any of the training data. These misremembered local minima
are called spurious states, see the below Fig. (4.2) for their illustration. In AM literature, they
are typically viewed as an obstacle to the faithful memory recall. For this reason, researchers in

this field typically aim to either remove them entirely from the energy landscape, or mitigate

their contribution to computation (e.g., by raising their energy) [20; 51].

In this Chapter, we discuss the emergence of spurious states in Dense Associative Memories, and
the general relationship between these memory models and diffusion models, popular in generative

AI. In previous chapters, AM models were studied in two situations. First, when the models

have small memory capacity and are trained on a small amount of data, e.g., classical Hopfield

Networks. Second, in situations when the models have large memory storage capacity, but still

are trained on a small amount of data. The focus of this Chapter are settings in which the models

are big (large information storage capabilities), and the amount of training data is even bigger

(exceeds the critical memory storage capacity of the model). In this regime, DenseAMs turn into

generative AI models.

4.1 Diffusion Models

Diffusion models have recently gained popularity, due to their flexibility and accuracy in modeling

high-dimensional distributions for a variety of domains, including image generation [43; 45; 44],

audio [52; 53; 54], video synthesis [55; 56; 57; 58], and other scientific applications. However, these

powerful and flexible models pose great challenges related to privacy and security, as concerns

grow about their tendency to generate their training data [49; 59; 48]. Such matters consequently

emphasize the need for further understanding of memorization and generalization behaviors in

diffusion models.

There are two fundamental processes which govern the aspects of diffusion models. Firstly, the
forward process typically described by the following It� Stochastic Differential Equation (SDE)

CHAPTER 4. FAILURE OF MEMORY AND GENERATIVE AI

27

Figure 4.1: A general illustration of diffusion models. Addition of noise transforms the complex
data distribution into a simple distribution � an isotropic Gaussian. The reverse process removes
the noise and transforms a noise sample into a sample from the data distribution.

[45]:

dxt = f (xt, t)dt + g(t)dwt,

(4.1)

transforms the given data distribution (x0 = y) into a simpler distribution1, e.g., an isotropic
Gaussian distribution. Here, wt is the standard Wiener process (or Brownian motion) and f (xt, t)
denotes the drift term that guides the diffusion process, which we will assume to be zero for the
most part of this section. Meanwhile, g(t) represents the diffusion coefficient that controls the
noise at each time step t
T . Secondly, the reverse process removes the injected noise at each
?
step t and it is described as

dxt = [f (xt, t)

g(t)2

?

?

xt log pt(xt)]dt + g(t)d �wt,

(4.2)

where �wt is the standard Wiener process. To effectively solve Eq. (4.2), one must reliably estimate
x log pt(x) via training a neural network s?(x, t). The learned weights ?? are obtained
the score
using methods for denoising score matching across multiple times steps [42; 43; 45]. The general

?

description of this optimization problem, given by [45], is formulated as

?? = arg min

?

Et,y,xt

?(t)

?

s?(xt, t)

? ?

xt log pt(xt

2

,

y)
?
|

(4.3)

? U

(t0, T ) is sampled from the uniform distribution

where t
a small time t0
forward process and ?(t) is a positive weighting function.

0 to a larger time T , while y

?

?

p(y) and xt

U

over the set of times ranging from
y) is the
y). Here, p(xt

p(xt

?

|

|

(cid:2)

(cid:3)

4.2 Diffusion Models from Associative Memories

A fascinating aspect about diffusion models is their process of shaping their energy landscape.
Instead of directly learning their energy function E?(xt, t), diffusion models learn the negative
gradient of their energy function or the score function:

1Assume that x0 = y

R

?

D are i.i.d samples coming from a data distribution p(y).

xt log pt(xt) =

xtE?(xt, t),

??

?

(4.4)

CHAPTER 4. FAILURE OF MEMORY AND GENERATIVE AI

28

using the process denoted in Eq. (4.3). However, this particular process does not explain how

generalization happens in such models. Instead, it tells a story of diffusion models behaving like

AM systems. Specifically, during training, diffusion models are learning how to remove noise from

a perturbed memory cue (or query) to obtain a clean memory accordingly to Eq. (4.3). At some

point, these models must behave like AM systems where they can effectively recover memories (or

stored training data points) from noises. But once a certain threshold (memorization capacity) is

exceeded, diffusion models can no longer act like effective denoisers or AM systems, the successive

failure in memory recall of these models must facilitate and signal their transition to generative

modeling.

Consequently, following the derivation done in [50], we can establish a fundamental connection

between diffusion and AM models. Consider the training data distribution in the variance-
exploding (VE) setting of f (xt, t) = 0 and g(t) = ?. In this case, the marginal probability
distribution of new samples can be computed exactly as

p(xt, t) = E

y?data(cid:34)

1
(2??2t)

D
2

exp

?

xt

y
?
2?2t

2
2
?

.

(cid:35)

(cid:17)

?

(cid:16)

(4.5)

Assuming the empirical distribution of the data p(y) = 1
K

?(D)(y

?

?�), where ?� represents

an individual data point (with data size K), this marginal distribution can be written as

K

�=1
(cid:80)

p(xt, t) =

1
K

K

�=1
(cid:88)

1
(2??2t)

D
2

exp

xt
?

?�
?
2?2t

2
2
?

(cid:17)

?

(cid:16)

def
?

exp

(cid:32) ?

EDM(xt, t)

,
2?2t (cid:33)

(4.6)

where we also defined the energy EDM of diffusion model, which up to state� or x� independent
terms is equal to

(4.7)

EDM(xt, t) =

2?2t log

?

K

exp

xt
?

?�
?
2?2t

2
2

?

?

.
(cid:17)(cid:21)

�=1
(cid:88)
As already observed in [47], the above energy function (4.7) is closely related to that of DenseAMs,

(cid:16)

(cid:20)

which are large memory storage variants of classical Hopfield networks, see Chapter (2).

The core idea behind DenseAMs is to design an energy function that peaks very sharply around

the intended memory patterns to prevent the overlapping (or cross talk) between them. Hence,

such networks can store and retrieve a much larger number of patterns, compared to the classical

Hopfield networks, and scales super-linearly (and possibly exponentially) to the size of the network,

allowing the decoupling of information storage capacity from the dimensionality of the data

[21; 27]. Of particular interest here is the DenseAM model studied in [60] (see also [31]), which

bears strong resemblance to Eq. (4.7):

EAM(x) =

??1 log

?

?

x
?

?

?

?

?�

2
2

,

(4.8)

�=1
(cid:88)
where ? is the inverse �temperature�, which controls the steepness of the energy landscape around
the memories ?�.

(cid:16)

(cid:17)(cid:21)

(cid:20)

K

exp

CHAPTER 4. FAILURE OF MEMORY AND GENERATIVE AI

29

Figure 4.2: A simple illustration depicting the change in the energy landscape as the size of
the training dataset is increased. In the small data regime, the diffusion model memorizes the
training data points as local minima of the energy. When the amount of training data exceeds
the memory capacity of the model, spurious patterns are formed and training data points are no
longer energy minima. Subsequent increase of the training set size leads to the generalization
phase, which is defined by the formation of continuous manifold of the low energy states. Figure
is obtained from [50].

Notice that Eqs. (4.7) and (4.8) are identical if ? = 1/(2?2t). The two systems described above
have important differences and similarities. In typical AM tasks, the inverse temperature ? is
kept constant (and is typically large). At the same time, the diffusion energy EDM(xt, t) describes
an intrinsically non-equilibrium system, since the effective temperature explicitly depends on

time. However, notice that since the reverse process (4.2) is guaranteed to invert the forward

step (4.1), the fixed points of the denoising process are guaranteed to coincide with the original

data points. Specifically, both of the above energy functions, Eqs. (4.7) and (4.8), express a
competition among the stored data points y
p(y) to see which one is closer to the query xt at
time t. Hence, although there might be differences in dynamical trajectories for EDM and EAM,
their fixed points must be the same2.

?

Specifically, the manifolds of the data in diffusion models must emerge from the point-like memory

storing systems, like AMs, in the limit when they are overloaded with amount of data above the

critical memory capacity. In this regime, distinct basins of attraction corresponding to separate

memories merge, forming the manifolds of the data. At the boundary of this transition, a separate
�phase� corresponding to spurious states, which is ubiquitous in AMs around the critical memory
load, appears and signals the onset of generalization, see Fig. (4.2) for the simple illustration of
this memorization-generalization transition. It is worth noting that DenseAMs typically have
an exponentially large memory storage capacity (in the number of neurons D) for uncorrelated
patterns. However, in the cases of real data, due to the high correlation of samples, the critical
memory load is much lower than the exponentially large capacity of uncorrelated data � a
well-known fact in associative memories [61; 62; 63; 64; 65; 66].

2We remind the reader that the fixed points retrieved from the reverse process correspond to t = 0.

CHAPTER 4. FAILURE OF MEMORY AND GENERATIVE AI

30

4.3 Memorization - Spurious - Generalization Transition

To better illustrate the connection between diffusion models and DenseAMs, we can investigate a

simple 2-dimensional toy model, see Fig. (4.3), that exhibits many aspects of the memorization-

generalization transition in these two types of models. Specifically, imagine that the training data

lies on a unit circle. We are interested in exploring how the shape of the energy function (4.8)

changes as the number of training data points, used in training a diffusion model, increases.

Specifically, in the trivial case of a single training point (K = 1), there exists only a single memory
?1 on the energy landscape of Eq. (4.8), making it independent from the inverse temperature or
sharpness value ?. In contrast, when there exists two training points (K = 2), there exists two
corresponding minima (or memories) ?1 and ?2:

EAM(x) =

??1 log

exp

?

?

x
?

?

?

?

?1

2
2

+ exp

?

x

?

?

?

?

?2

2
2

,

(4.9)

(cid:17)
. However, for finite values of ?, there exists a configuration which yields a minimum:

(cid:17)(cid:105)

(cid:16)

(cid:16)

(cid:104)

when ?

? ?

? = arg min

x

EAM(x),

(4.10)

= ?1
= ?2. This �novel� local minimum of the energy is the spurious state illustrated in the

that does not correspond with any of the two training data points or stored patterns, i.e., ?
and ?
cartoonish Fig. (4.2).

When the training data size K
described as a continuous density of states:

? ?

, the empirical data distribution of the toy model can be

(cid:0)
The probability of the generated data is proportional (up to terms independent of the state x) to

(cid:1)

p(y) =

1
?

?

1 + y2
y2

2 ?

1

.

(4.11)

+?

dy1dy2 p(y) e???x?y?2

2 = e??(R2+1)I0(2?R),

(4.12)

p(x)

?

(cid:90)??

where I0(
model is given by

) is a modified Bessel function of the first kind3. Thus, the energy of the 2D circle
�

EAM(R, ?) = R2 + 1

1
?

?

log

I0(2?R)

????

(R

?

1)2,

(4.13)

3In order to obtain Eq. (4.12), it is easiest to introduce polar coordinates for both the state vector x and the

(cid:2)

(cid:3)

training data y:

(cid:40)

x1 = R cos(?)
x2 = R sin(?)

(cid:40)

y1 = r cos(?)
y2 = r sin(?)

The integral (4.12) can then be written as

p(x)

?

?
(cid:90)

2?(cid:90)

d?

0

0

rdr

1
?

?(r2

?

1)e??[R2+r2?2Rr cos(???)] = e??(R2+1)I0(2?R)

and explicitly computed using the definition of the modified Bessel functions [67].

?
?
CHAPTER 4. FAILURE OF MEMORY AND GENERATIVE AI

31

? {

Figure 4.3: Energy landscape evolution for the 2D toy model as training data size K increases.
use standard VE-SDE based diffusion pipeline with training
Models trained at K
2, 9, 1000
}
data sampled from the unit circle, shown in black for K
. Generated samples are shown
alongside the learned score field s?(xt, t) done via a neural network, aligned with the negative
gradient of the energy Eq. (4.7). Hierarchical clustering identifies structure within the generations,
with cluster centroid energies visualized by color and numerical value. The rightmost panel shows
the exact solution as K
derived in Eq. (4.13). As K grows, the model initially memorizes
individual data points, forming isolated basins. Around K = 9, spurious patterns emerge �
distinct low-energy attractors not present in the data � which mark the onset of generalization.
At large K, the model enters a fully generalized regime, where low-energy states lie on a flat,
continuous manifold shown in Fig. (4.2). Top-row figure is retrieved from [50].

2, 9
}

? ?

? {

where R is the radius of the unit circle and ? is the polar angle. Keep in mind, the dependence
on ? in Eq. (4.13) disappears for the final result.

For our diffusion model, we consider the following forward process which describes the VE-setting:

dxt = ?dwt,

(4.14)

where the drift term f (xt, t) = 0 and wt is Brownian motion. The corresponding reverse process
is described as

dxt =

?2

?

?

xt log pt(xt)

dt + ?2dwt,

(4.15)

where the diffusion coefficient g(t) = ? is fixed as 1, matching the radius of the unit circle. Using
, over the time
these SDEs, we trained a set of SDE-based diffusion models, for K
[?, 1] where ? = 10?5, using the objective (4.3). Then, we visualize the energy
domain of t
landscape of each model and record our results in Fig. (4.3).

2, 9, 1000
}

? {

?

(cid:3)

(cid:2)

As expected, the local minima of the resulting EAM in Eq. (4.13) form a continuous manifold,
corresponding to R = 1. The data samples from the model in Fig. (4.3) occupy the vicinity of
that manifold. This behavior describes the fully generalized phase. In the limit of large ?, the
energy landscape is described by a parabola centered around R = 1. For our trained diffusion

CHAPTER 4. FAILURE OF MEMORY AND GENERATIVE AI

32

model at K = 1000, we see that the exact energy and approximated energy, obtained from the
diffusion model, are very much aligned to one another in Fig. (4.3).

Meanwhile, for small number of data points (K = 2), the diffusion model exhibits memorization.
The local minima of the energy correspond to the training data points. Importantly, at K = 9,
we are able to observe the first signs of spurious states. At this stage, the model begins to learn
emergent (different from the training data) local minima of the energy. Subsequent increase of
the size of the training set leads to fully generalized behavior, which is illustrated for K = 1000.
At that stage, all of the samples from the model live in close proximity of the exact data manifold.

The right panel shows the analytical expression for the energy landscape, defined by Eq. (4.13).

Thus, the conventional diffusion modeling pipeline, following Eq. (4.3), agrees very well with the

theoretical prediction of the empirical energy (4.13) and the cartoonish illustration in Fig. (4.2).

From the perspective of DenseAMs, we can see a novel phase also exists in diffusion models �
spurious states � previously overlooked in the memorization-generalization literature of these
models [49; 59; 68; 69; 70; 71; 72; 73; 74]. As demonstrated in [50], diffusion models trained on real

and high dimensional datasets also follow the same trend illustrated in Fig. (4.3): transitioning
from memorization to spurious phase to generalization as the training data size K increases.
Hence, by viewing the problem of generalization as a failure of storing all of the data points

as memories, we can provide a novel understanding of the memorization and generalization in

generative diffusion models and interestingly, demonstrate the existence of spurious states in such

models. This illustrates that diffusion models behave as AM systems in the small data regime,

and as generative models in the large data regime.

Notebook 4.1: Comparison of diffusion energy and DenseAM energy

In this notebook, we offer the reader the possibility to train a simple diffusion model using

data from the 2-D circle as an example. The reader can reconstruct the energy profile of

the diffusion model by integrating the score function and compare this energy profile with

the energy of DenseAM model.

Checkout the notebook as a blog post, a colab notebook or as a raw .ipynb file.

Chapter 5

Associative Memory:
A Machine Learning Model

In this Chapter, we will view Associative Memory networks through the lens of machine learning

modeling. After presenting a brief discussion on machine learning modeling, and the sources of

error in (machine) learning Section (5.1), we present Associative Memory network as a machine

learning model that can be used much like other models in learning, highlighting its inference

process, expressivity, application to supervised learning, and its parametric and nonparametric

forms, see Section (5.2). Then we discuss how this model can be used for the unsupervised

learning task of clustering in Section (5.3). Finally, we elaborate on the connection between

Associative Memory and Kernel Machines, and discuss novel Associative Memory models that

emerge from this connection in Section (5.4).

5.1 Machine Learning Modeling

The purpose of learning is to obtain a version of the ground-truth distribution (or equivalently the
data-generating function) given (potentially noisy) samples from the ground-truth distribution.

The first step is data acquisition. Given data, we choose a model or function class
which
corresponds to not just a method (such as Support Vector Machines [75], Generalized Linear
Models [76], Decision trees [77], etc.), but their specific configuration governed by their respective
hyperparameters such as regularization forms and penalties, trees depth, network architecture and
, the learning
activations, optimization configurations. Given our choice of the function class
process searches (optimizes) for the function �f
that (approximately) minimizes the empirical
? F
risk � the sample loss computed over the training data � or some surrogate of it which better
represents the true risk � the population loss � or is easier to optimize, such as some continuous
version of a discrete loss and/or some form of regularization that mitigates overfitting such as

F

F

weight penalty or decay, dropout and such.

We currently have an understanding of the factors [78; 79; 80; 81] affecting the excess risk of
this chosen/learned function � the difference between the true risk of this learned function �f
and the best possible function f ?. At a high level, these factor depend on (i) the choice of the
function class and its capacity to model the data generating process, (ii) the use of an empirical
risk estimate instead of the true risk to learn this function, and (iii) the approximation in the
empirical risk minimization (ERM) over the class of functions g

.

? F

33

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

34

For a particular method (decision trees, linear models, neural networks), let
denote the function
class for some fixed hyperparameter ?
? (tree depth, number of trees for tree ensembles;
regularization parameter for linear and nonlinear models, activation functions, batch size in
stochastic gradient descent or SGD, etc.) in the space of valid hyperparameters ?.

F

?

Focusing on supervised learning, for any model or function f :
generated from a distribution pdata over
and the empirical risk Rm(f ) of f with m samples

X � Y

, and a loss function
(xi, yi)
}

m
i=1 ?

{

X ? Y
:

with (x, y), x

, y

? Y
? X
, the true risk R(f )

L
Y � Y
pdata is given by

R(f ) = E(x,y)?pdata [

L

(y, f (x))] =

(y, f (x))dpdata,

L

Rm(f ) = Em [

L

(y, f (x))] =

1
m

(cid:90)

m

(yi, f (xi)).

L

i=1
(cid:88)

We denote the Bayes optimal model as f ? where, for any (x, y)

pdata,

?

f ?(x) = arg min

�y?Y

E [

L

(y, �y)

x] .
|

We denote with the following:

�f = arg min

R(f ),

f ?F

�fm = arg min

f ?F

Rm(f ),

(5.1)

(5.2)

(5.3)

as the true risk minimizer �f
model class

respectively.

F

and the empirical risk minimizer �fm

(with m samples) in

? F

? F

When performing empirical risk minimization or ERM over

, the excess risk is given by

F

= R( �fm)

?

E

R(f ?) = R( �fm)

?
Eest

R( �f )

+ R( �f )

R(f ?)
,

?
Eapp

(5.4)

(cid:124)

(cid:123)(cid:122)

est(m) = R( �fm)

(cid:124)
which decomposes into two terms: (i) the approximation risk
R(f ?), and (ii) the
R( �f ). For limited number of samples m, there is a tradeoff
estimation risk
between
app but increases
E
est(m) [78; 79]. Roughly speaking, methods are termed universal approximators if there is some
E
app can be made arbitrarily small.
hyperparameter which ensures that the approximation error
E
very large,

Of course, the flip side is that this can make the corresponding class of functions

est, where a larger function class
E

(cid:123)(cid:122)
(cid:125)
app = R( �f )

usually reduces

E
app and

?

?

F

E

E

(cid:125)

F

often increasing the estimation error

est(m) for a fixed m.

E

Bottou and Bosquet [80] study the tradeoffs in a �large-scale� setting where the learning is
compute bound (in addition to the limited number of samples m). Given any computational
budget T , they consider the learning setting �small-scale� when the number of samples m is small
enough to allow for the ERM to be performed to arbitrary precision. In this case, the tradeoff
est terms (as above). They consider the large scale setting where the
E
ERM needs to be approximated given the computational budget and discuss the tradeoffs in the
excess risk of an approximate empirical risk minimizer �fm
est, they
R( �fm) � the excess risk incurred due to
introduce the optimization risk term

. In addition to

is between the

app and

app and

? F

E

E

E

opt = R( �fm)
E

?

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

35

approximate ERM � and argue that, in compute-bound large-scale learning, approximate ERM
on all the samples m can achieve lower excess risk than high precision ERM on a subsample of
size m?

m. Fig. (5.1) provides a visual representation of this excess risk decomposition.

?

f ?

app

E

E

�f

�fm

est

E

�fm

opt

E

app
E
f ?

0
?
�f
?

�fm

est

E

opt

E

E

�fm

E

F

F

E
? F

? F
? F

. We depict the decomposition of

F
incurred by
Figure 5.1: Decompositions of excess risk
the approximate empirical risk minimizer �fm
found (usually) with a scalable optimization
with respect to the Bayes optimal model f ?. The true risk
algorithm in the model class
minimizer �f
is the best approximator of the optimal f ?, while the exact empirical risk
minimizer �fm
can be distinct from the true risk minimizer �f since we are using the empirical
risk (obtained with a finite training set) for our learning instead of the true risk. This figure is
can be such that the
partially replicated from Ram et al., 2023 [81]. Left: The model class
F
Bayes optimal f ?
app � the difference
, hence we would have a nonzero approximation risk
?? F
in the true risk (defined in Eq. (5.1)) between the Bayes optimal f ? and the true risk minimizer
�f in our model class
such
or can be approximated to arbitrary precision
that the Bayes optimal f ? is in the model class
with a model in the function class. In this case, the true risk minimizer �f
f ? will have (almost)
0. However, it is important to note that the estimation risk
zero approximation risk with
est � the difference in the risk between the true risk minimizer and the empirical risk minimizer
E
(for some notion of size), with larger model
� is often related to the size of the model class
classes incurring larger estimation risk. The optimization risk
opt � the difference between the
risk of the exact and approximate empirical risk minimizers � can also be affected implicitly by
the size of the model class
, where larger classes require more computational resources for the
learning optimization to achieve any specific level of empirical risk approximation; conversely, for
fixed computational resources, larger function classes can incur larger optimization risk.

app > 0. Right: We can also select a large model class
E

� with

app

?

?

F

F

F

F

F

E

E

E

For parametric models, the functions f
,
?
}
where we explicitly denote the dependence of ? in f?. Thus, the true risk minimizer, and the
empirical risk minimizer can be respectively written as:

are parameterized with ?

? where

f?, ?
{

? F

?

F

?

?

�f ? f�?,

�? = arg min

???

R(f?) = arg min

???

E(x,y)?pdata [

(y, f?(x))]

L

�fm ? f�?m

,

�?m = arg min

???

Rm(f?) = arg min

???

1
m

m

i=1
(cid:88)

(yi, f?(xi)).

L

(5.5)

(5.6)

The approximate empirical risk minimizer would be denoted with f�?m
with corresponding model
parameters �?m. The parameters ?
? corresponds to model/function specific parameters �
weights and biases for linear models and neural networks; split dimensions and thresholds, and
leaf node values for univariate decision trees. Once these parameters �?m are learned from the
without having to keep the training
data, one can make predictions f�?m
data

? X
around anymore. For nonparametric models such as nearest neighbor models

(x) on new inputs x

?

m
(xi, yi)
i=1
}

{

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

36

and (kernel) density estimation based models (such as the Nadaraya-Watson estimator), the

training data is also required for making predictions, and thus are considered as part of the

�model parameters�.

5.2 Associative Memory Network as a Model

The previously discussed Associative Memory networks can be viewed as a parameterized model

f? :
the D-dimensional Euclidean space. The interpretation is detailed in the following:

with parameters ?. For the sake of simplicity, let us for now assume that

X ? X

D,

R

X ?

Model parameters are stored patterns

The model parameters ? can be reshaped as a (D
the K stored patterns
?�
R
{
vector. Note that these model parameters can be learned.

D, �

[[K]]

�

?

?

}

K) matrix, and is usually termed as
� each stored pattern ?� is a D dimensional

Energy function

Given these model parameters ?, we have an energy function of a state v
usually of the following general form:

D,

R

? X ?

E?(v; ?) =

Q

?

K

?

?

�=1
(cid:88)

F (? S [?(v), ?�])

?

,

?

(5.7)

� One can view the v as the internal state and ?(v) as its activation. See Section (3.2) of

Chapter (3) for a discussion on states and activations.

� The S :

Euclidean distance.

X � X ?

R denotes a notion of similarity such as a dot-product or negative squared

� ? > 0 is the inverse temperature and controls how much high similarities are magnified and
low similarities are diminished. This inverse temperature ? controls the sharpness of the
energy around the memories, with larger values of ? inducing sharper energy landscapes
while smaller values generating smoother ones. See Notebook 2.1 in Section (2.3), Chapter (2)

for a demonstration.

� The separation function F : R

F (z) = exp(z) for some z

?
� The scaling function Q : R

R.

?

R is a fast-growing function such as F (z) = zp or

?

R is a monotonic non-decreasing function such as identity

Q(z) = z or logarithm Q(z) = log z for some z

R.

?

Inference via energy descent

Given a learning rate ? > 0, number of steps T and a clamping mask m
for some x

D, f?(x)
}
is computed as follows via (clamped) coordinate gradient descent over the

? {

0, 1

? X

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

37

energy function, with the descent initialized as the input x:

?

v(0)
v(t)

x,
v(t?1)
?
? (x) ? v(T ),
f m

? m

?

vE?(v; ?)

|v=v(t?1) ,

? ?

[[T ]] ,

t

?

(5.8)

(5.9)

(5.10)

denotes the element-wise multiplication of vectors. In the absence of the clamping

where
mask, we drop the m subscript and use f?.

?

� As the learning rate ?

?

0, this energy gradient descent is defined by the following dynamics:

dv
dt

=

m

?

? ?

vE?(v; ?).

(5.11)

� The learning rate ? does not need to be fixed, and can also vary with time. For example,

the learning rate can decay with time.

? ?

� When the number of steps T (also referred to as the number of layers in the Associative
Memory network) goes to infinity (that is, T
), for an appropriately set learning rate
? (sufficiently small or appropriately scheduled), f?(x) will be one of the local minima of
0 at v = f?(x). These
the energy function � that is,
fixed points (local minima) are often termed as the retrieved memories. Note that we are
seeking local minima and not saddle points which correspond to meta-stable states where it
is hard to decrease the energy but it is not a local minima. See Demircigil et al., 2017 [27]
for a discussion of the energy landscape and the basins of attraction for the different local
minima.

vE?(v; ?) = 0,

2
vE?(v; ?)

?

?

?

� The gradient clamping mask m enables clamping of a subset of the state variables. When
m is the D-dimensional all-one vector 1N , the complete state vector v is modified in
, then only the first N ? < N entries of the state vector
Eq. (5.9). If m =
v are allowed to be modified, while the remaining (N
N ?) entries of v are clamped to
their initial values obtained from the input x. This clamping and coordinate-wise gradient
descent is discussed in Section (3.6) of Chapter (3), see also Fig. (3.4) for a visualization of

N ?, 0?
1?

(N ?N ?)

?

(cid:104)

(cid:105)

?

the clamped energy-descent.

� If the learning rate ? is small, and T is not too large, the input would not be modified
x. If ? is large (and not decayed appropriately), we might never

significantly, and f?(x)
arrive at a local minima of the energy.

?

Given the model parameters ? (that can be interpreted as K stored patterns), the various
hyperparameters � the functions Q, F, S in the energy function, the inverse temperature ?, the
learning rate ? > 0, the number of layers/steps T � define the inference process with this model
f?. There is a tight relationship between the energy function and probability density through
the Boltzmann distribution � that is, the density p(v) of a state v is tied its energy E(v) as
E(v)). Given this interpretation, the inference with the described Associative
p(v)
Memory network amounts to a form of likelihood maximization via gradient descent.

exp(

?

?

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

38

Classic energy for binary patterns

As an example, if ?�
, F (z) = z2,
?
? = 1 and Q is the identity function, then the corresponding energy function is that of the
Classic Hopfield Network (CHN):

[[K]], ? : R

, S[v, v?] =

v, v?
?

? {?

? {?

1, +1

D
}

1, 1

?

�

?

}

E(v; ?) =

K

?

�=1
(cid:88)

?(v), ?�

(
?

)2 .
?

(5.12)

Log-sum-exp energy with real valued patterns

With ?�
Q(z) = log z, we obtain the widely used log-sum-exp or LSE energy:

[[K]], ? as identity, S[v, v?] =

v
?

1/2

v?

?

?

R

?

?

�

?

D

2, F (z) = exp(z) and
?

E?(v; ?) =

log

?

K

exp

�=1
(cid:88)

(cid:16)

?/2

?

v

?

?

?�

2
?

.

(cid:17)

(5.13)

Note that common representations of the LSE energy contains a preceding (1/?) term on
the right-hand-side to cancel out the ?-scaling in the gradient. However, we are removing
this here since we only care about the direction of the gradient, and not the magnitude.

5.2.1 Memory Capacity and Expressivity

Given the above energy function, we can now consider the set

energy function in Eq. (5.7) defined as:

of local minima of the

M ? X

=

v

:

?

? X

M

vE?(v; ?) = 0,

2
vE?(v; ?)

?

(cid:8)

0

.

?

(cid:9)

(5.14)

Note that this set of local minima will depend on all but two of the various hyperparameters
previously discussed � this set does not depend on the learning rate ? and the number of steps
T . In the scenario where the learning rate ?
, for any
, the output f?(x)
input x
? X
making f? :
may not be a surjection with a finite T especially when the value of T is small.

as the output is one of the local minima, thereby essentially
a surjective function (many-to-one mapping); note that f? for the same ?

0 and the number of layers/steps T

X ? M

? ?

? M

?

? M

such that v

?�, then the model
For a stored pattern ?�, if there exists a local minima v
has memorized a stored pattern and is able to approximately retrieve it (given an appropriate
input initializing the energy descent). The memory capacity of an Associative Memory network
is informally defined as the largest number Kmax of randomly generated stored patterns ?
such that each stored pattern can be (approximately) retrieved. For example, with the classic
O(D); with the log-sum-exp energy in Eq. (5.13), there exists
energy in Eq. (5.12), Kmax
?
O(exp(D)). See Chapter (2) for
hyperparameters (specifically, values of ?) such that Kmax
further discussion of memory capacity.

?

?

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

39

X ? M

. We can increase the cardinality of

Given that, under appropriate configurations, the Associative Memory network operates as a
, the expressivity or the approximation ability of the model f? is related
surjection f? :
to the size (cardinality) of
to up to Kmax by increasing
the model size (in terms of the number of model parameters) to up to DKmax corresponding to
? containing Kmax stored patterns in D-dimensions. Beyond that point, increasing the number
of (randomly generated) stored patterns in Kmax would not increase the cardinality of
. There
are ways to carefully design the stored patterns such that the cardinality of
Kmax.

can go beyond

M

M

M

M

5.2.2 Supervised Learning

X

X

to

, in the previously discussed supervised learning setup of Section (5.1), we usually

While we have discussed the Associative Memory network as a model f? :
from
consider models of the form f? :
One way to handle that with an Associative Memory network is to consider a model f? :
is a d-dimensional feature space, and
where

,
Z ? Z
is a k-dimensional output space,
would be a D = (d + k)-dimensional space with the features and output concatenated into the

mapping from a feature space

to an output space

that maps

X ? X

X ? Y

X � Y

. If

?

Z

X

X

Y

Y

.

Z
state vector. This concept is also described in Section (3.6) of Chapter (3).

k ]?
k�D. Also consider an uninformative default (potentially learnable) prediction y0

Consider the clamping vector m = [0?
?
� as
0, 1
{
}
an example, y0 = 0k for regression or y0 = (1/k)1k for k-class classification. Then we can define
a function g? :

mapping features x to predictions using parameters ? as follows:

D, and the matrix M = [0k�d Ik]
}

d 1?

? {

? Y

0, 1

X ? Y

0 ]?,

?

v(0)
v(t)

[x?, y?
v(t?1)
?
?
g?(x) ? M v(T ).

? m

vE?(v; ?)

? ?

|v=v(t?1) ,

[[T ]] ,

t

?

(5.15)

(5.16)

(5.17)

See Fig. (3.4) for a visualization this energy minimization based inference process for a supervised

learning problem. There are a few important things to note here:

� The energy E?(v, ?) now depends on a similarity function S :

similarity between the features in
function S can be defined as S[z, z?] = ?SX [x, x?] + (1
defined accordingly with x?, y?), and SX :
output specific similarity functions.

X � X ?

X

and the outputs in

, which can incorporate

Y

Z � Z
. For example, the similarity
?)SY [y, y?] where z = [x?y?]? (z?
R are feature and

?
R, SY :

Y � Y ?

� During the energy descent, the features values x

provide the initialization v(0) =

[x?y?

0 ]?, and then are not modified at all because of the clamping mask m.

? X

� The final output of g? is obtained by extracting the last k-dimensions of the final state v(T )

for the energy descent with the element selecting matrix M .

� We are considering a gradient descent over the energy E?(v; ?), which would define a
to the

). However, we are clamping the state variables corresponding

density over (
input x. This roughly corresponds to a conditional density over
descent corresponds to a conditional likelihood maximization.

X � Y

, and the clamped energy

X

Y

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

40

5.2.3 Nonparametric vs Parametric Models

Nonparametric Models

An important question with an Associative Memory model f? is the process of obtaining the
model parameters ? (also known as the �stored patterns�). Given a set of patterns
[[K]]
}
pdata), we can consider a nonparametric
(possibly from an unknown distribution pdata, that is, ?�
, with the size of the model
form of the model where ? is just all of the data
}
(which would be O(KD) when each stored pattern ?� is of size D) growing with the number of
stored patterns (which is K). As discussed previously in Section (5.2.1), the corresponding energy
) local minima, where Kmax is the capacity of
function E?(
the model. Note that if the stored patterns have a specific structure, then the number of local
minima can be higher than Kmax. For the supervised learning setup discussed in Section (5.2.2),
the stored patterns ?� would be feature-output pairs (x, y), x

; ?) can have up to O(min
K, Kmax
{
�

in the training set.

?�, �
{

?
?�, �

[[K]]

, y

?

?

{

}

? X

? Y

The single-step retrieval dynamics (that is, number of layers T = 1) of the nonparametric
Associative Memory networks has recently also been recently interpreted as the solution of a

specific nonparametric support vector regression problem [82]. Different choices of kernel functions
and training data preprocessing result in different energy functions E?(

; ?).
�

Parametric Models

One can also consider a parametric form of the model f?, where the size of ? is pre-specified,
and the parameters are learned using the data. As an example, we can say that the size of ? is
[[K]] of size D. However, here we are allowed to
such that it can store only K patterns ?�, �
learn these patterns ?�. Given a set of training examples S =
and a
regularizer R, at a high level, we can learn ? by solving the following problem:

, a loss function

m
i=1

zi

L

?

{

}

min
?

R(?) +

m

1
m

L

(zi, f?(zi)).

(5.18)

i=1
(cid:88)
If the training example zi is a feature-label pair (xi, yi), and the loss is the negative cross-entropy
loss between the labels, and the Associative Memory model is as defined in Section (5.2.2), then
(zi, f?(zi)) would simplify to NegativeCrossEntropy(yi, f?(xi)) as in a standard classification
L
problem. The regularization R(?) can be utilized to avoid overfitting. For example, R(?) =
penalizes the norm of the learnable stored patterns, scaled with a
?
?2,1 = ?
?
?
R+. One can also consider a regularization that enforces the learnable stored patterns to be
?
well separated so as to make the memory retrieval process more efficient and robust [83; 84]. The
loss
and the regularization R can be specified in a problem dependent manner. As the model
f? corresponds to a T -layer recursive network, we can learn ? with (stochastic) gradient descent
given that the loss

and the regularization R are differentiable.

�?[[K]] ?

?�

(cid:80)

L

?

?

L

5.3 Clustering

Consider an Associative Memory network based model with parameters ? =
and an energy function E?(
D

D, �
?
R as defined in Section (5.2). With any input x

?�
{

R

?

[[K]]

}
D,

R

; ?) : R
�

?

?

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

41

the energy descent involved in the model output f?(x) intuitively would move x towards the local
energy minimum closest to it. When the number of local minima is relatively small, for all input,

the outputs will contract towards this small set of local minima. An example of this behaviour is

shown in Fig. (5.2). One way of thinking about this contraction effect is the following � the

Associative Memory model moves relatively close-by points closer together, potentially moving

far-away points even farther. What is interesting is that this is a collective contraction effect on

the whole set of inputs even though the model operates on all the input independently.

Figure 5.2: Contraction of points over the energy landscape. Here we demonstrate
how the Associative Memory model can contract a set of points via gradient descent over the
energy landscape. Column 1: The initial set of 9 points. Column 2: The initial set of points
are overlaid on an energy landscape with 3 local minima � lighter colour denotes lower energy.
Columns 3-5: Each point separately undergoes the energy descent for 3 steps corresponding to
a 3-layer model. Column 6: The outputs of the model applied separately to each point in the
set are now a contracted version of the initial set (column 1).

If the set of input points are all close-by to begin with, and the energy local minima are relatively

more spread out, then all points could potentially contract towards the same local minima.

However, if the input points are as spread out as the energy local minima, then the contraction
effect would lead to input points getting more clustered � a subset of the points getting closer
to each other while each subset getting farther away from each other. This capability of the

Associative Memory network makes it quite useful for the classical problem of clustering.

5.3.1 Euclidean Clustering

Given a set of points S =
in a D-dimensional Euclidean space, a commonly
studied clustering problem is the k-means clustering problem, which seeks to solve the following
discrete optimization problem:

xi
{

D, i

[[m]]

R

?

?

}

min
c1,...,ck?RD

m

i=1
(cid:88)

(cid:13)
(cid:13)

(cid:13)
(cid:13)

min
j?[[k]]

xi

?

cj

2

.

(5.19)

[[k]], with a
This problem seeks to partition the set of points S into k disjoint subsets Cj, j
prototype or center cj
d for each subset Cj, ensuring that squared Euclidean distance between
a point in the subset and the corresponding center is small. This is a NP-hard problem even for
k = 2 [85], and Lloyd�s algorithm [86] is the most commonly used approximate algorithm though
many more efficient algorithms with improved approximation guarantees have been developed.

R

?

?

The hardness of this problem is partially due to the discrete nature of the objective in Eq. (5.19),

and thus usually requires discrete algorithms. This objective in its current form is not conducive

to gradient descent based solutions prevalent in modern machine learning. One can modify this
discrete objective into a continuous one by replacing the minj?[[k]] in Eq. (5.19) with a soft-min

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

42

function, leading to soft or fuzzy k-means clustering [87; 88]:

min
c1,...,ck?RD

m

i=1
(cid:88)

(cid:88)j?[[k]]

exp(

?

xi

?
?
j??[[k]] exp(
(cid:13)
(cid:13)

2

cj

?
(cid:13)
?
(cid:13)

?

)
xi
(cid:13)
(cid:13)

xi

?

?
cj?

2

cj
2)
(cid:13)
?
(cid:13)

,

(5.20)

where ? > 0 is a hyperparameter. The above objective is an upper bound of the k-means objective
in Eq. (5.19) and we would be minimizing the upper bound. Larger values of ? make the upper
bound tighter.

(cid:80)

2, one can
Instead of relaxing the discrete assignments of points to clusters minj?[[k]]
emulate the discrete assignments by �moving� a point xi to its closest cluster cj?(xi) with
2 using the contraction capability of Associative Memory networks
j?(xi) ? arg minj?[[k]]
2, the amount by which the point was �moved� [60]. Thus, the
and using the term
k-means objective in Eq. (5.19) can be re-written as:

cj
?
cj?(xi)
(cid:13)
(cid:13)

xi
(cid:13)
(cid:13)

xi

xi

cj

(cid:13)
(cid:13)

(cid:13)
(cid:13)

?

?

(cid:13)
(cid:13)
m

(cid:13)
(cid:13)

min
c1,...,ck?RD

i=1
(cid:88)

min
j?[[k]]

xi

?

2

cj

?

(cid:13)
(cid:13)

(cid:13)
(cid:13)

2

cj?(xi)

m

xi

?

i=1 (cid:13)
(cid:88)
(cid:13)
(cid:13)
f?(xi)
xi

?

min
c1,...,ck?RD
m

min
?

?

i=1
(cid:88)

(cid:13)
(cid:13)

(5.21)

(cid:13)
(cid:13)
(cid:13)
2 if f?(xi)

cj?(xi).

?

(cid:13)
(cid:13)

This distinction and equivalence is visualized in Fig. (5.3).

Figure 5.3: Computing the k-means objective with points and cluster centers. The
computation of the k-means objective requires us to implicitly or explicitly assign points to
and centers ?. Column 2: The k-means
clusters. Column 1: We are given a set of points
objective in Eq. (5.19) assigns each point to its closest center (
?), and then sums this distance-
to-closest-center over all points. Column 3: Instead of relaxing the discreteness in the k-means
objective as in soft k-means in Eq. (5.20), ClAM [60] uses an Associative Memory network
with (learnable) parameters ? to effectively relocate each point to its closest center, and then
considers the sum of these per-point-relocation in Eq. (5.21) as a surrogate for the k-means
objective. Column 4: Instead of complete contraction to the cluster centers, sometimes it might
be beneficial to partially contract to the cluster centers [89].

�?

�

cj?(xi).
What we need then is an Associative Memory model f? : R
, then the condition
If use the k cluster centers as the stored patterns, that is ? ?
cj?(xi) would be satisfied if the basins of attraction around each cluster center (which
f?(xi)
is also the stored pattern) matches the Voronoi partition of the input space given the cluster
centers. Given k centers, the Voronoi partition of the space is (i) a k-partition of space with
each partition corresponding to a specific center, and (ii) any point in a specific partition has its

R
c1, . . . , ck
{

D such that f?(xi)

?

?

?

}

D

corresponding center as its closest center. Given 3 centers, Fig. (5.4) shows the Voronoi partition

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

43

of the input space.

With ? ?
, and the following energy function E?(
�
inverse temperature ? > 0 and number of layers T , we can get the desired behaviour:

c1, . . . , ck

; ?) for an appropriately large

}

{

E?(v; ?) =

1
?

?

log

k

j=1
(cid:88)

exp(

?

?

cj

2

).

v

?

(5.22)

(cid:13)
(cid:13)

(cid:13)
(cid:13)

cj?(xi) corresponds to the basins of attraction of each
The desired behavior of having f?(xi)
[[k]] matching Voronoi partition of the space given the memories/centers
stored pattern ?�, �
?
. The dependence on ? for T = 10 layers of the DenseAM is visualized
?� = c�, �
[[k]]
{
in Fig. (5.4). This allows us to solve the discrete clustering problem in Eq. (5.19) with the

?

?

}

�

Figure 5.4: Basins of attraction vs Voronoi partition. The basins of attraction of the
given memories/centers (black dots
) for different ? values with a 10-layer Associative Memory
network f? (T = 10) are shown by the colored regions. Dashed lines show the desired Voronoi
partition. As the value of ? increases, the basins of attraction start aligning with the desired
Voronoi partitions. Column 1: For a small inverse-temperature ? = 0.001, the Voronoi partition
does not align with the basins of attraction of the Associative Memory network. Column 2: With
a higher inverse-temperature ? = 10, the basins of attraction partially align with the Voronoi
partition. Column 3: With a high inverse-temperature ? = 100, the basins of attraction and
the Voronoi partition are practically indistinguishable. This figure is replicated from Saha et al.,
2023 [60].

re-written objective (5.21) completely with gradient descent, since we can differentiate through
the T layers of the Associative Memory network. Additionally, we can leverage the clamped
inference procedure in Associative Memory networks to extend the standard clustering objective
over the training set S to effectively give us a self-supervised clustering loss by creating multiple
S � thereby enabling self-supervision in clustering. Concretely, we
versions of each point x
can mask each input x with a random m
x to the Associative
d to form the input m
}
x) with the clamping mask
Memory network. Then, we can perform clamped inference f �m
�m which is the complement of the mask m. Thus, our standard clustering loss in Eq. (5.21)
would be extended as following from the standard clustering loss on the left to the self-supervised

? (m

0, 1

? {

?

?

?

clustering loss on the right:

x

?

f?(x)
?

?

2

self-supervision
??????????

min
?

min
?

x?S
(cid:88)

x?S
(cid:88)

Em

m
?

?

(x

?

f �m
? (m

2

x)
?

?

(5.23)

The overall clustering process, which involves the learning of the stored patterns ?, is visualized
in Fig. (5.5).

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

44

? {

0, 1
}

S, we first apply a mask (in
Figure 5.5: Euclidean clustering with DenseAM. For x
purple) m
D to x to get the initial iterate v(0) for the AM recursion. With T recursions,
we have a completed version v(T ). The use of the mask m is optional, and allows for a semi-
supervised clustering loss by leveraging the clamped inference f �m
? (x) in Associative Memory
networks; see Eq. (5.23) and Saha et al., 2023 [60, Section 3.4] for details. In the limiting case,
x and we do the unclamped inference f?(x). The
there is no mask (that is, m = 1D) and v(0)
stored patterns ? are updated with the gradient
on the loss in Eq. (5.21). This figure is
replicated from Saha et al., 2023 [60].

?

?

L

?

?

5.3.2 Deep Clustering

For data modalities, such as image or text, it is often necessary to first learn information-preserving

Euclidean representations before clustering these learned representations. This problem of jointly
learning representations and clustering is often referred to as deep clustering [90; 91; 92]. One
common way to learn information-preserving representations is to use an auto-encoder and
minimize the reconstruction error, where e? :
D is a domain-specific encoder (parameterized
X ?
with ?) that maps the input (images, text) into a latent Euclidean space, which is then used to
(parameterized with ?) which often
reconstruct the original data using a decoder d? : R
r(x, d?(e?(x))) for a
mirrors the domain-specific encoder, and the reconstruction loss defined as
:

R, giving us the following learning problem given a dataset S

? X

loss

r :

R

L

D

L

X � X ?

? X

min
?,?

x?S
(cid:88)

r(x, d?(e?(x)))

L

(5.24)

A simple but useful baseline is to just solve the above problem, and then perform k-means
clustering on latent representations e?(S) =
. However, as we are already learning
e?(x), x
representations, it is beneficial to steer the learned representations to already have a favourable

?

S

{

}

clustered structure.

This is often obtained by augmenting the reconstruction loss

r(x, d?(e?(x))) with some form of
2, where
a clustering loss
e?(x)
[[K]] are learnable cluster centers in the latent space; see Eq. (5.20) as an example of
D, j
cj
a continuous clustering loss. Thus, the overall learning problem can be written as following for a

), like a relaxed version of minj?[[k]]

c1, . . . , ck

c(e?(x),

cj

(cid:13)
(cid:13)

(cid:13)
(cid:13)

?

R

L

L

?

?

{

}

?={?1,?2,?3}?1?2?3v(0)?m?xv(1)v(T)v(2)v(t)xmSelf-supervisedlossL??LUpdatedprototypes?AMrecursionCHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

45

regularization hyperparameter ?

[0, 1]:

?

min
?,?,{c1,...,ck}?RD

(1

?

x?S
(cid:88)

?)

r(x, d?(e?(x))) + ?

L

c(e?(x),

L

c1, . . . , ck
{

).
}

(5.25)

The hyperparameter ? balances the reconstruction and clustering loss as there is an inherent
tradeoff between (i) preserving information in the latent space with e?(x)
= x?
� thus low reconstruction loss, and (ii) forming tight clusters in the latent space by mapping
cj?(x) where
all points within a cluster to almost the same representation, that is e?(x)
2 � giving us low clustering loss. The goal is to find the
j?(x) ? arg minj?[[k]]
sweet spot, which allows us to have low reconstruction loss (preserving necessary information),

e?(x?) for x

e?(x)

cj

?

??

?

(cid:13)
(cid:13)

(cid:13)
(cid:13)

while forming a clustered structure in the latent space by pushing both the representations and

the cluster centers to have low clustering loss. There is also an implicit objective of forming
well-balanced clusters and avoiding representation collapse, where all points end up in the same
cluster in the latent space.

By viewing Associative Memory network as a contractive layer (see Fig. (5.2)), we can introduce a

clustered structure in the learnable latent space in an alternate manner. Instead of maintaining an
c for clustering that pushes the latent representations to be clustered when minimized,
we can employ the contractive nature of the Associative Memory networks to directly have a

objective

L

clustered structure in the latent space, and just focus on optimizing the reconstruction loss of

this structured latent space by contracting the latent space before reconstructing [89].

Given a domain-specific encoder e?, a corresponding decoder d?, and an Associative Memory
model f? serving as a contraction layer, we can solve the following optimization problem:

min
?,?,?

x?S
(cid:88)

r(x, d?(f?(e?(x)))).

L

(5.26)

Here, the input x is first encoded to the latent space as e?(x)
D, and passed through the
contraction layer to get f?(e?(x)). Then the original input is reconstructed with the decoder to
get d?(f?(e?(x))). In contrast to the use of Associative Memory networks in vanilla Euclidean
clustering [60] where we want complete contraction � the points are relocated to the closest

R

?

cluster center � in this setup, it is beneficial to only consider partial contraction � the points
are modified to have a more clustered structure, but the output of the model f? are still distinct
for distinct models. This is visualized in Fig. (5.3) with Fig. (5.3) (Column 3) showing complete

contraction, while Fig. (5.3) (Column 4) visualizing partial contraction.

The overall learning procedure is shown in Fig. (5.6). This method provides a single loss function

that simultaneously ensures that the loss of information is minimized, while pushing the latent

representations to have a clustered structure through the Associative Memory network; no separate

clustering loss is utilized here. This loss function, and thus the resulting deep clustering scheme,

is agnostic to the data modality (images or text or something else) and the corresponding encoder
and decoder architectures. Given that the Associative Memory network model f? is differentiable,
with respect to its learnable parameters ? and its output, we can perform deep clustering by the
solving Eq. (5.26) with (stochastic) gradient descent provided the encoder e? and decoder d? are

?
CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

46

, the encoder e? maps x to the latent space to get e?(x)

Figure 5.6: Deep clustering with DenseAM. Given an input x

in the ambient space
D. Then we use the (partial)
X
contraction capability of a Associative Memory model f? to move the latent representation from
e?(x) to f?
e?(x) towards one of the memories. This contracted representation is then mapped
?
e?(x). For the purposes of learning
back to the ambient space
?
the encoder, decoder and Associative Memory network parameters, we utilize the reconstruction
e?(x)) and backpropagate the gradients with respect to the parameters ?, ?, ?.
loss
The solid arrows denote the forward-pass x
e?(x) to compute the
f?(x)
single loss term in Eq. (5.26). The dashed arrows denote the backward pass showing the single
loss driving all updates. This figure is replicated from Saha et al., 2024 [89].

with the decoder d? to get d?

r(x, d?

e?(x)

? X

f?

f?

f?

d?

?

?

?

R

X

L

?

?

?

?

?

?

differentiable with respect to their respective parameters and output.

5.4 Kernel Machines

Revisiting the energy function of an Associative Memory network f? with parameters ? in
Eq. (5.7), and denoting the F (?S[v, ?�]) term with ?(v, ?�), we can write the energy function as:

E?(v; ?) =

Q

?

?

?(v, ?�)

,

?

(5.27)

(cid:88)�?[[K]]

where the

?
R,
� ?(v, ?�) term can be interpreted as a kernel sum with the kernel ? :
the core computation in kernel machines [93]. This simple observation allows us to leverage the

X � X ?

?

(cid:80)

AmbientSpaceXxd?(f?(e?(x)))LatentSpaceRDEncodere?DeepclusteringlossL=PxLr(x,d??f??e?(x))??LDecoderd?e?(x)?1?2?3?4?5?6f?(e?(x))AMrec.??L??LCHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

47

rich literature on kernel machine for the development of novel Associative Memory networks with

unique capabilities. Two main areas of research in kernel machines focus on the following:

� A lot of research focused on the development and use of expressive domain-specific kernels
?, and the understanding of their properties such as expressivity and generalization. In
the context of Associative Memory networks, this corresponds to the development of novel

domain-specific energy functions, since one can create an energy given a kernel function

through Eq. (5.27), thereby expanding the applicability of these models to new domains.

� ?(v, ?�) which (i) implies that we need to keep the set

� Every inference in vanilla kernel machines requires the computation of the kernel sum
}�?[[K]] even for inference, and
(ii) leads to extremely expensive training and inference as each inference is naively O(K),
(cid:80)
the number of memories (or terms in the kernel sum). A lot of research focused on improving

?�
{

the computational time and space complexity of these kernel-sum computations. This

corresponds to improving the computational time and space complexity of the computation

of the energy and thus the energy gradient � this would speed up each energy descent step
and thus the overall inference with a model f?.

5.4.1 Random Features

Roughly speaking, for a symmetric positive definite kernel ? :
, where
feature map ? :
, ?(x, x?) =
x, x?
?(x), ?(x?)
?
?
simplified as follows:

R, there exists an implicit
X � X ?
is a Reproducing Kernel Hilbert space, such that for any
H
. If an explicit feature map ? is available, a kernel sum can be

X ? H

? X

?(v, ?�) =

K

�=1
(cid:88)

K

�=1
(cid:88)

?(v), ?(?�)
?

?

=

?(v),

(cid:42)

K

�=1
(cid:88)

?(?�)

(cid:43)

,

(5.28)

K
where we would just need to compute the
�=1 ?(?�) term once, and use it for any subsequent
inference, thereby removing the O(K) dependence both from the time and space complexity of
the inference. However, note that the computational complexities now depends on the size of the
feature map ?(v) and ?(?�).

(cid:80)

A commonly used and expressive kernel is the RBF (radial basis function) kernel ? : R
D
R+
2) for a scaling parameter ? > 0. This kernel possesses an infinite
with ?(x, x?) = exp(
?
dimensional feature map ? : R
?, thus making the explicit feature map practically
D
unusable. Various indexing schemes have been developed and analysed [94; 95; 96] to speed up

?

?

x?

?

?

�

R

R

x

?

?

D

the computation of kernel sums..

As an alternate seminal approach [97], random Fourier features were used to develop approximate
Y for the RBF kernel such that ?(x, x?)
feature maps ? : R
D
precisely, with high probability,

?(x), ?(x?)
?

[97]. More

? ?

?

R

x, x?
?

?

D,

R

?(x, x?)

?

?(x), ?(x?)

O

?

D/Y

,

(5.29)

implying that an ? approximation guarantee requires Y

O(D/?2). The first set of random

(cid:12)
(cid:12)

(cid:10)

(cid:16)(cid:112)

(cid:17)

(cid:11)(cid:12)
(cid:12)
?

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

48

feature maps for the RBF kernel were defined as follows using trigonometric functions:

?(x) =

1
?Y

cos(

cos(

(cid:10)

?1, x
?2, x

(cid:10)
� � �
?Y , x

cos(

(cid:10)

+ b1)
+ b2)

+ bY )

(cid:11)

(cid:11)

(cid:11)

?

?
?
?
?
?

?

?
?
?
?
?

, and ?(x) =

1
?Y

)

sin(

cos(

?1, x
?1, x
(cid:10)
?2, x
cos(
(cid:11)
(cid:10)
?2, x
)
(cid:11)
(cid:10)

sin(

)
(cid:11)

)

(cid:10)
(cid:11)
� � �
?Y , x
?Y , x
(cid:10)

)
(cid:11)

)

cos(

sin(

?

?
?
?
?
?
?
?
?
?
?
?
?

?

?
?
?
?
?
?
?
?
?
?
?
?

,

(0, ID),

?i
bi
i
?

? N

(0, 1),

? U

?

[[Y ]] .

(5.30)
Note that the first set of random features produces a Y -dimensional feature map with Y random
features while the second set produces a 2Y -dimensional feature map using Y random features,
(0, ID) denotes the D-dimensional isotropic
but provides better approximation guarantees. The
(0, 1) denotes the univariate uniform distribution over
standard normal distribution, and the
the range [0, 1]. Since then, various other random features have been developed for the RBF
kernel [98; 99; 100; 101], and for various other kernels [102; 103]. Liu et al., 2021 [104] provide a

N

U

(cid:11)

(cid:10)

comprehensive survey of random feature for kernel approximation.

In the context of Associative Memory, this allows us to approximate the energy function of the

form in Eq. (5.27) as follows:

E?(v; ?) =

Q

?

?

?(v, ?�)

?

? ?

?

Q

K

?

?(v),

?(?�)

= �E?(v; T ),

(5.31)

(cid:43)

(cid:42)

?

?

�=1
(cid:88)

(cid:88)�?[[K]]

?
?
?
?
?
?
; ?) (and its gradient) requires us to have access to all
where the computation of the energy E?(
�
of size KD, while the computation of the approximate
the stored patterns ? =
K
energy �E?(
�=1 ?(?�) of size
(which can be generated on the fly given the Y
Y and the random features
random seeds and thus do not need to be stored explicitly) [105]. We can now perform inference

; T ) (and its gradient) only requires us to have access to the T ?
�
[[Y ]]

(?i, bi), i
{

?
?
?
?
?
?

?�, �

[[K]]

(cid:80)

?T

(cid:123)(cid:122)

?

?

{

}

}

(cid:124)

(cid:125)

via gradient descent on this approximate energy, providing an unique (Dense) Associative Memory
model that does not require the stored patterns ? for inference. It has been shown that the
approximation in the energy translates to approximation in the inference � the inference f?(x)
by minimizing the exact energy E?(
; ?) is approximated with the inference fT (x) by minimizing
�
the approximate energy �E?(
; T ). This appproximation is affected by the following factors [105]:
�

� The approximation depends on the kernel approximation introduced by the random features
N/Y ), with larger number of random features Y improving the

with a factor of O(
approximation.

(cid:112)

� The approximation also depends on the initial energy E?(x; ?) of the input x � the
initial state for the energy descent � with larger initial energy leading to higher levels of

approximation.

� The hyperparameter ? which corresponds to the step-size (or learning rate) of the energy

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

49

gradient descent, with smaller ? implying lower levels of approximation.

Of course, if we are already considering a kernel function ? in Eq. (5.27), which has an explicit
)2), then we can directly use
feature map ? (for example if ?(x, x?) =
the explicit exact feature map to simplify the exact energy, and incur no approximation in the

or ?(x, x?) = (
?

x, x?
?

x, x?

?

?

kernel function evaluation, and thus in the Associative Memory model inference.

Notebook 5.1: Distributed Representation for Dense Associative Memory

In this notebook, we demonstrate how we utilize random features to disentangle the size of

the Dense Associative Memory network from the number of memories to be stored. Given
the standard log-sum-exp energy E?(
; ?), corresponding to a model f? of size O(DK), we
�
demonstrate how we can use the trigonometric random features to develop an approximate
energy �E?(
; T ) using a distributed representation T of the memories ? =
�
thus giving us a model fT of size O(Y ).

?�, �
{

[[k]]

?

}

,

Checkout the notebook as a blog post, a colab notebook or as a raw .ipynb file.

E?(v; ?)

=

log

?

f??O(DK) size

(cid:124)

(cid:123)(cid:122)

(cid:125)

exp(

?

?/2

v
?

?

?�

2)

?

K

�=1
(cid:88)
K

?

log

? ?

?(

?v), ?(

??�)

�=1 (cid:68)
(cid:88)

(cid:112)

K

(cid:112)

(cid:69)

?(x) =

=

=

?

?

log

?(

?v),

?(

??�)

(cid:42)

(cid:112)

log

?(

?v, T

(cid:43)

�=1
(cid:88)

(cid:112)
= �E?(v; T )

(cid:68)

(cid:112)

(cid:69)

fT ?O(Y ) size

(cid:124)

(cid:123)(cid:122)

(cid:125)

5.4.2 Novel Energy Functions

?

?1, x
)
cos(
?1, x
)
sin(
(cid:11)
(cid:10)
?2, x
cos(
)
(cid:11)
(cid:10)
?2, x
)
sin(
(cid:11)
(cid:10)
(cid:10)
(cid:11)
� � �
?Y , x
?Y , x
(cid:10)
(cid:10)

cos(
sin(

?
?
?
?
?
?
?
)
?
?
)
?
(cid:11)
?
?
[[Y ]]
(cid:11)
?

i
?

(0, ID),

1
?Y

?
?
?
?
?
?
?
?
?
?
?
?
? N

?i

? ?
?�

As discussed earlier in Section (5.2), given an energy function, we can define a probability density
through the Boltzmann distribution. Alternately, given a density function p :
R?0, we can
define a energy function E(v)

log p(v) through the same relationship.

X ?

R

Given a set of samples ? =

from an unknown distribution pdata over
}
D, one way to define a density function �p is through kernel density estimation, where the
X ?
goal is to devise a �p that closely approximates the unknown pdata. A kernel density estimate or
KDE at any point v

is defined as:

pdata, �

[[K]]

?

?

{

? X

�ph(v; ?) =

1
Kh

K

?

�=1
(cid:88)

(cid:18)

?�

v

?
h

,

(cid:19)

(5.32)

where ? : R
d
valid density, the kernel function needs to satisfy the following conditions:

R?0 is the kernel function, and h > 0 is the kernel bandwidth. For this to be a

?

� Symmetry: ?(x) = ?(

x)

?

x

?

? X

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

50

� Nonnegativity: ?(x)

0

� Normalization:

x

? X

?

?
x ?(x) dx = 1.
(cid:82)
xi

D
i=1 ?1(
|

), where ?1 : R
|
R

For multivariate data (that is D > 1), the kernel ? has been defined both as ?(x) = c?1(
)
?
?
R?0 is an univariate kernel function, xi denotes the
or ?(x) = c?
i-th coordinate of x for any x
D, and c, c? are positive constants ensuring that normalization
(cid:81)
condition for ? is satisfied. Note that, with the RBF kernel (which becomes the Gaussian kernel
with proper scaling for normalization) with ?1(z) = exp(
) =
|
2). We will consider the univariate case from hereon for the ease of exposition with D =
exp(
x
? ?
1, where ? : R
R?0 satisfying the aforementioned symmetry, nonnegativity and normalization
conditions.

R, ?1(
?

i ?1(
|

) =
?

z
?|

2), z

xi

?

?

(cid:81)

?

?

x

x

?

|

),
For the purpose of KDE, the scale of the kernel function is not unique. That is, for a given ?(
�
/b), for some b > 0. Then, one obtains the same KDE by rescaling the
we can define �?(
�
choice of h. Hence, the shape of the kernel function plays a more important role in determining
the choice of the kernel. We now introduce two parameters associated with the kernel � the scale
�? and the regularity ?? defined as:

) = b?1?(
�

�? ?

x2?(x) dx,

?? ?

(?(x))2 dx

(cid:90)x

(cid:90)x

(5.33)

The quality or generalization of KDE depends on these two properties of the kernel. The
generalization error of �ph(
; ?) is measured by the Mean Integrated Squared Error or MISE, and is
�
given by

(cid:21)
where the expectation is over the K random samples ? from pdata.

(cid:20)(cid:90)v

?

MISE(h) = E

(�ph(v; ?)

pdata(v))2dv

,

(5.34)

Assuming that the ground-truth density pdata is twice continuously differentiable, a second-order
Taylor expansion gives the leading terms of the MISE(h), which decomposes into squared bias
and variance terms [106, Section 2.5]:

MISE(h)

�2
?
4

h4

?

p??
data(v)

2

dv

+

(cid:90)v

(cid:12)
bias-squared term
(cid:12)

(cid:12)
(cid:12)

.

??
Kh
variance

(5.35)

(cid:123)(cid:122)
Thus, reducing the bandwidth h decreases bias but increase variance, and vice verse for increasing
h, thereby highlighting the bias-variance tradeoff. Balancing the bias-squared and variance terms,
we can have kernel-specific optimal choice h?
?

for the bandwidth

(cid:125)

(cid:124)

(cid:124)(cid:123)(cid:122)(cid:125)

h?
?

?

??
K�2
?

(cid:32)

4
p??
data(v)

v

1/5

2

dv

,

(cid:33)

.

Plugging this into Eq. (5.35) gives us the best possible MISE:

(cid:82)

(cid:12)
(cid:12)

(cid:12)
(cid:12)

MISE(h?
?)

?�???

5
4 (cid:32)

?

v |

p??
data(v)
|
K

(cid:82)

4/5

2 dv

(cid:33)

,

(5.36)

(5.37)

CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

51

where the choice of the kernel function ? affects the MISE through its scale �? and regularity
??. Thus, it is intuitive to select the kernel function ? based on the optimal MISE(h?
?). As
discussed above, the scale of the kernel function is non-unique, and can be fixed to �? = 1 by
appropriately scaling the kernel function. Hence, the kernel with the smallest regularity ??,
subject to �? = 1 (without loss of generality), over the class of normalized, symmetric, and
positive kernels is most desirable. This is a well-studied problem [107; 108] [106, Section 2.7],
and the Epanechnikov kernel ?epan(z) = max
achieves the optimal
1
? |
{
is known as the efficiency of any kernel relative to
?). The quantity, Eff(?) ? ??/??epan
MISE(h?
the Epanechnikov kernel. Various kernels with varying levels of efficiencies have been developed,

2, 0
}
|

= ReLU

z
? |

2
|

1

z

(cid:0)

(cid:1)

and we present a representative subset of these kernel functions in Fig. (5.7). Similar analysis

and guarantees can be established for multivariate KDE.

Figure 5.7: Different kernels used for Kernel Density Estimation. We show the shapes,
expression and KDE efficiency relative to the Epanechnikov kernel (higher is better) for 8 kernels.
The center of each kernel is marked with a red ?. To highlight the shape of the kernel, we have
removed any scaling in the kernel expression. Note that all above kernels except Gaussian have
finite support. The Epanechnikov kernel has the highest efficiency (100%). The Gaussian kernel
is extremely popular, and it is more efficient (95.1%) than the Uniform kernel (92.9%). However,
there are various other kernels (such as the Triangle kernel) with better efficiency. This image is
replicated from Hoover et al., 2025 [109].

The rich literature of KDE and its suite of well-studied kernel functions opens up the path to the

development of various energy functions for Associative Memory networks � one for each kernel

function � which have not been considered previously. As a natural first choice, one can select

the optimal Epanechnikov kernel, which leads to the following novel energy function [109]:

E?(v; ?) =

log

?

K

�=1
(cid:88)

ReLU

1

(cid:16)

?/2

v
?

?

?�

2
?

?

.

(cid:17)

(5.38)

This makes use of a shifted-ReLU operation, and thus is termed the log-sum-ReLU or LSR energy.

x0.000.250.500.751.00(x)Gaussianexp(x2)Eff(): 95.1%xTriangleReLU(1x)Eff(): 98.6%xUniformI(1x2)Eff(): 92.9%xEpanechnikovReLU(1x2)Eff(): 100.0%3210123x0.000.250.500.751.00(x)Cosinecos(2min(x,1))Eff(): 99.9%3210123xQuarticReLU(1x2)2Eff(): 99.4%3210123xTriweightReLU(1x2)3Eff(): 98.7%3210123xTricubeReLU(1x3)3Eff(): 99.8%CHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

52

This can be contrasted with the popular LSE or log-sum-exp energy shown in Eq. (5.13), where
we replace the exponential separation function F (z) = exp(z) with the shifted-ReLU separation
function F (z) = ReLU (1 + z) with the negative squared Euclidean distance based similarity
function S[x, x?] =

1/2

x?

x

.

?

?

?

?

Figure 5.8: Emergence of novel energy local minima. LSR energy can create more
memories than there are stored patterns under critical regimes of ?. Left: 1D LSR vs LSE energy
landscape. Note that LSE is never capable of having more local minima than the number of
stored patterns. Right: 2D LSR energy landscape, where increasing ? creates novel local minima
where basins intersect. Unsupported regions are shaded gray. This image is replicated from
Hoover et al., 2025 [109].

This novel energy function has various desirable properties:

Exact single-step retrieval

For an Associative Memory network with the LSR energy and appropriate hyperparameters,

it is possible to have exact retrieval of stored patterns in a single energy gradient step.

This is in contrast to LSE where only approximate retrieval is possible unless the inverse-
temperature ?

.
? ?

Exponential memory capacity without exponential separation function

This Associative Memory network equipped with the LSR energy has exponential memory
capacity � that is, the number of stored patterns that are retrievable is O(exp(D)). This
is similar to the LSE energy.

Generation of a multitude of novel memories

Finally, the LSE energy can introduce numerous novel energy local minima to the energy

landscape, while also maintaining local minima around the stored patterns, enabling

simultaneous retrieval of stored patterns and retrieval of novel memory, providing a path to

data generation in Associative Memory networks with energy descent. This phenomena is

visualized for data in one and two dimensions in Fig. (5.8), and has been utilized to create

LSR memories while creating  ones.preservesnovelLSE can do only one or the other.? = 0.6? = 0.3? = 1.1? = 2.11 novel  0 preserved1 novel  3 preserved4 novel  3 preserved0 novel  3 preservedLow CriticalHighLSR (ours)LSECHAPTER 5. ASSOCIATIVE MEMORY: A MACHINE LEARNING MODEL

53

novel samples from an approximation of the underlying data distribution pdata. Such a
phenomena has not previously been seen in literature.

However, this novel LSR energy can pose certain novel challenges:

Regions of infinite energy

For a given configuration of an Associative Memory network, with LSR energy, there exists

such that E?(v; ?) =

v
is visualized in Fig. (5.8) (Right) as the gray shaded region for two dimensional data.

given the finite support of the Epanechnikov kernel. This

? X

?

Chapter 6

Conclusion

In this tutorial we have covered recent advances in energy-based Associative Memory, including

information storage capacity calculations (Chapter 2), relationship to transformers (Chapter

3) and diffusion models (Chapter 4), and connection to non-neural network machine learning

(Chapter 5) and other ideas. In the past few years, Associative Memory became an active area

of research with many lines of exploration coexisting and branching into several disciplines.

Since this tutorial was prepared for ICML audience, we focused mostly on explaining the core

ideas with only minimal derivations necessary to understand those ideas. We also prepared

coding notebooks to help AI practitioners gain hands-on experience with AM basics. Inevitably,

with this strategy in mind, many important and exciting aspects of AMs remained outside

the scope of this tutorial. For instance, we did not discuss the biological implementations of

DenseAMs [30; 110; 111; 35; 112; 113; 114]. Several valuable trends in AM-inspired statistical

physics [115; 116; 117; 118; 28; 119; 120; 121; 122] have also only been briefly mentioned. Memory

augmentation of large language models (LLMs) [123; 124; 125; 126] is becoming an active area

of research with clever ideas on how memory models can be used synergistically with feed-

forward architectures. There are exciting ideas around novel neural architectures inspired by

AMs [29; 33; 127; 128; 26; 129; 130], and domain specific applications [131; 132] that have

not been covered with sufficient detail either. Quantum DenseAMs is an emerging topic [133].

Neuromorphic hardware based on DenseAMs [134] is becoming a promising area of research too.

We expect these trends to grow and new trends to appear. We hope that this introductory

tutorial may provide an entry point for new researchers in this exciting field.

54

Bibliography

[1] Endel Tulving. Episodic and semantic memory. Organization of memory, 1972.

[2] Warren S McCulloch and Walter Pitts. A logical calculus of the ideas immanent in nervous

activity. The bulletin of mathematical biophysics, 5:115�133, 1943.

[3] Frank Rosenblatt. Principles of neurodynamics. Perceptrons and the theory of brain

mechanisms, 1962.

[4] The New York Times. New navy device learns by doing; psychologist shows embryo of

computer designed to read and grow wiser. The New York Times, 1958.

[5] Minsky Marvin and A Papert Seymour. Perceptrons. Cambridge, MA: MIT Press, 6(318-

362):7, 1969.

[6] The Royal Swedish Academy of Sciences. Nobel prize in physics 2024. Nobel Prize Outreach,

2024.

[7] John J Hopfield. Neural networks and physical systems with emergent collective com-
putational abilities. Proceedings of the national academy of sciences, 79(8):2554�2558,
1982.

[8] Stephen G. Brush. History of the lenz-ising model. Rev. Mod. Phys., 39:883�893, 10 1967.

[9] James A Anderson. A memory storage model utilizing spatial correlation functions. Kyber-

netik, 5(3):113�119, 1968.

[10] David J Willshaw, O Peter Buneman, and Hugh Christopher Longuet-Higgins. Non-

holographic associative memory. Nature, 222(5197):960�962, 1969.

[11] S-I Amari. Learning patterns and pattern sequences by self-organizing nets of threshold

elements. IEEE Transactions on computers, 100(11):1197�1206, 1972.

[12] S-I Amari. Neural theory of association and concept-formation. Biological cybernetics,

26(3):175�185, 1977.

[13] Michael A Cohen and Stephen Grossberg. Absolute stability of global pattern formation
and parallel memory storage by competitive neural networks. IEEE transactions on systems,
man, and cybernetics, 5:815�826, 1983.

[14] John Hopfield. Neurons With Graded Response Have Collective Computational Properties
Like Those of Two-State Neurons. Proceedings of the National Academy of Sciences of the
United States of America, 81:3088�92, June 1984.

55

BIBLIOGRAPHY

56

[15] Daniel J Amit, Hanoch Gutfreund, and Haim Sompolinsky. Storing infinite numbers of
patterns in a spin-glass model of neural networks. Physical Review Letters, 55(14):1530,
1985.

[16] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations

by back-propagating errors. nature, 323(6088):533�536, 1986.

[17] Paul J. Werbos. Generalization of backpropagation with application to a recurrent gas

market model. Neural Networks, 1(4):339�356, 1988.

[18] Donald O. Hebb. (1949) donald o. hebb, the organization of behavior, new york: Wiley,

introduction and chapter 4, "the first stage of perception: growth of the assembly," pp. xi-
xix, 60-78. In Neurocomputing, Volume 1: Foundations of Research. The MIT Press, 04
1949.

[19] Geoffrey Hinton and Terrence Sejnowski. Optimal perceptual inference. In Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition, pages 448�453, 01 1983.

[20] John J. Hopfield, David I. Feinstein, and Richard G. Palmer. �unlearning� has a stabilizing

effect in collective memories. Nature, 304:158�159, 1983.

[21] Dmitry Krotov and John J Hopfield. Dense associative memory for pattern recognition.

Advances in neural information processing systems, 29, 2016.

[22] Kunihiko Fukushima. Neocognitron: A self-organizing neural network model for a mechanism
of pattern recognition unaffected by shift in position. Biological Cybernetics, 36:193�202,
1980.

[23] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
Gomez, ?ukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural
information processing systems, 30, 2017.

[24] Dmitry Krotov. A new frontier for hopfield networks. Nature Reviews Physics, 5(7):366�367,

2023.

[25] Dmitry Krotov and John Hopfield. Dense associative memory is robust to adversarial inputs.

Neural computation, 30(12):3151�3167, 2018.

[26] Hamza Chaudhry, Jacob Zavatone-Veth, Dmitry Krotov, and Cengiz Pehlevan. Long
sequence hopfield memory. Advances in Neural Information Processing Systems, 36:54300�
54340, 2023.

[27] Mete Demircigil, Judith Heusel, Matthias L�we, Sven Upgang, and Franck Vermet. On
a model of associative memory with huge storage capacity. Journal of Statistical Physics,
168(2):288�299, 2017.

[28] Carlo Lucibello and Marc M�zard. Exponential capacity of dense associative memories.

Physical Review Letters, 132(7):077301, 2024.

[29] Hubert Ramsauer, Bernhard Sch�fl, Johannes Lehner, Philipp Seidl, Michael Widrich,

Lukas Gruber, Markus Holzleitner, Thomas Adler, David Kreil, Michael K Kopp, G�nter

BIBLIOGRAPHY

57

Klambauer, Johannes Brandstetter, and Sepp Hochreiter. Hopfield networks is all you need.
In International Conference on Learning Representations, 2021.

[30] Dmitry Krotov and John J Hopfield. Large associative memory problem in neurobiology
and machine learning. In International Conference on Learning Representations, 2021.

[31] Beren Millidge, Tommaso Salvatori, Yuhang Song, Thomas Lukasiewicz, and Rafal Bogacz.

Universal hopfield networks: A general framework for single-shot associative memory models.
In International Conference on Machine Learning, pages 15561�15583. PMLR, 2022.

[32] Thomas F Burns and Tomoki Fukai. Simplicial hopfield networks.

In The Eleventh

International Conference on Learning Representations, 2022.

[33] Dmitry Krotov. Hierarchical associative memory, 2021.

[34] Benjamin Hoover, Yuchen Liang, Bao Pham, Rameswar Panda, Hendrik Strobelt,
Duen Horng Chau, Mohammed Zaki, and Dmitry Krotov. Energy transformer. Advances
in Neural Information Processing Systems, 36, 2024.

[35] Leo Kozachkov, Jean-Jacques Slotine, and Dmitry Krotov. Neuron�astrocyte associative
memory. Proceedings of the National Academy of Sciences, 122(21):e2417788122, 2025.

[36] Benjamin Hoover, Duen Horng Chau, Hendrik Strobelt, and Dmitry Krotov. A univer-
sal abstraction for hierarchical hopfield networks. The Symbiosis of Deep Learning and
Differential Equations II, 2022.

[37] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,

Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,

Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image
recognition at scale. In International Conference on Learning Representations, 2021.

[38] Fei Tang and Michael Kopp. A remark on a paper of krotov and hopfield. arXiv preprint

arXiv:2105.15034, 2021.

[39] Roger Brown and James Kulik. Flashbulb memories. Cognition, 5(1):73�99, 1977.

[40] Marigold Linton. I remember it well. Psychology Today, 13(2):81, 1979.

[41] Elizabeth F Loftus. Memory. Rowman & Littlefield Publishers, 1988.

[42] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep

unsupervised learning using nonequilibrium thermodynamics. In Francis Bach and David
Blei, editors, Proceedings of the 32nd International Conference on Machine Learning,
volume 37 of Proceedings of Machine Learning Research, pages 2256�2265, Lille, France, 7
2015. PMLR.

[43] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models.

Advances in neural information processing systems, 33:6840�6851, 2020.

[44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models.

arXiv preprint arXiv:2010.02502, 2020.

BIBLIOGRAPHY

58

[45] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon,

and Ben Poole. Score-based generative modeling through stochastic differential equations.
In International Conference on Learning Representations, 2021.

[46] Benjamin Hoover, Hendrik Strobelt, Dmitry Krotov, Judy Hoffman, Zsolt Kira, and

Duen Horng Chau. Memory in Plain Sight: Surveying the Uncanny Resemblances of

Associative Memories and Diffusion Models, 2023.

[47] Luca Ambrogioni. In Search of Dispersed Memories: Generative Diffusion Models Are

Associative Memory Networks. Entropy, 26(5), 2024.

[48] Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian

Tramer, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from
diffusion models. In 32nd USENIX Security Symposium (USENIX Security 23), pages
5253�5270, 2023.

[49] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein.

Diffusion art or digital forgery? investigating data replication in diffusion models. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 6048�6058, 2023.

[50] Bao Pham, Gabriel Raya, Matteo Negri, Mohammed J Zaki, Luca Ambrogioni, and Dmitry

Krotov. Memorization to generalization: Emergence of diffusion models from associative
memory. arXiv preprint arXiv:2505.21777, 2025.

[51] John Hertz, Anders Krogh, and Richard G Palmer. Introduction to the theory of neural

computation. Addison Wesley Longman, 1991.

[52] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan.
Wavegrad: Estimating gradients for waveform generation. arXiv preprint arXiv:2009.00713,
2020.

[53] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A
versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020.

[54] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang,

and Mark D Plumbley. AudioLDM: Text-to-audio generation with latent diffusion models.
In Proceedings of the 40th International Conference on Machine Learning, volume 202,
pages 21450�21474, 2023.

[55] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and

David J Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022.

[56] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu,

Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation
without text-video data. arXiv preprint arXiv:2209.14792, 2022.

[57] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja

Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent

BIBLIOGRAPHY

59

diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
2023.

[58] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr,

Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh.
Video generation models as world simulators. openai.com, 2024.

[59] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein.
Understanding and mitigating copying in diffusion models. Advances in Neural Information
Processing Systems, 36:47783�47803, 2023.

[60] Bishwajit Saha, Dmitry Krotov, Mohammed J Zaki, and Parikshit Ram. End-to-end
differentiable clustering with associative memories. In Proceedings of the 40th International
Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research,
pages 29649�29670. PMLR, 23�29 Jul 2023.

[61] C Cortes, A Krogh, and JA Hertz. Hierarchical associative networks. Journal of Physics A:

Mathematical and General, 20(13):4449, 1987.

[62] Hanoch Gutfreund. Neural networks with hierarchically correlated patterns. Physical

Review A, 37(2):570, 1988.

[63] A Krogh and JA Hertz. Mean-field analysis of hierarchical associative networks
with�magnetisation�. Journal of Physics A: Mathematical and General, 21(9):2211, 1988.

[64] I Kanter and Haim Sompolinsky. Associative recall of memory without errors. Physical

Review A, 35(1):380, 1987.

[65] JL Van Hemmen. Hebbian learning, its correlation catastrophe, and unlearning. Network:

Computation in Neural Systems, 8(3):V1, 1997.

[66] Aditya Cowsik and Adithya Sriram. Dense hopfield networks with hierarchical memories.

In New Frontiers in Associative Memories, 2025.

[67] Izrail Solomonovich Gradshteyn and Iosif Moiseevich Ryzhik. Table of integrals, series, and

products. Academic press, 2014.

[68] Casey Meehan, Kamalika Chaudhuri, and Sanjoy Dasgupta. A non-parametric test to detect
data-copying in generative models. In International Conference on Artificial Intelligence
and Statistics, 2020.

[69] Gerrit J.J. Van den Burg and Chris Williams. On memorization in probabilistic deep

generative models. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan,
editors, Advances in Neural Information Processing Systems, 2021.

[70] TaeHo Yoon, Joo Young Choi, Sehyun Kwon, and Ernest K Ryu. Diffusion probabilistic
models generalize when they fail to memorize. In ICML 2023 Workshop on Structured
Probabilistic Inference Generative Modeling, 2023.

[71] Xiangming Gu, Chao Du, Tianyu Pang, Chongxuan Li, Min Lin, and Ye Wang. On

memorization in diffusion models. arXiv preprint arXiv:2310.02664, 2023.

BIBLIOGRAPHY

60

[72] Nicholas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian

Tram�r, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from
diffusion models. In Proceedings of the 32nd USENIX Conference on Security Symposium,
SEC �23, USA, 2023. USENIX Association.

[73] Beatrice Achilli, Enrico Ventura, Gianluigi Silvestri, Bao Pham, Gabriel Raya, Dmitry

Krotov, Carlo Lucibello, and Luca Ambrogioni. Losing dimensions: Geometric memorization
in generative diffusion. arXiv preprint arXiv:2410.08727, 2024.

[74] Giulio Biroli, Tony Bonnaire, Valentin de Bortoli, and Marc M�zard. Dynamical regimes of

diffusion models. Nature Communications, 15(1), November 2024.

[75] Corinna Cortes and Vladimir Vapnik. Support-vector networks. Mach. Learn., 20(3):273�297,

September 1995.

[76] J. A. Nelder and R. W. M. Wedderburn. Generalized linear models. Journal of the Royal

Statistical Society. Series A (General), 135(3):370�384, 1972.

[77] J. R. Quinlan. Induction of decision trees. Machine Learning, 1:81�106, 1986.

[78] Vladimir Vapnik. Estimation of dependences based on empirical data. Springer Science &

Business Media, 2006.

[79] Luc Devroye, L�szl� Gy�rfi, and G�bor Lugosi. A probabilistic theory of pattern recognition,

volume 31. Springer Science & Business Media, 2013.

[80] L�on Bottou and Olivier Bousquet. The tradeoffs of large scale learning. In Advances in

neural information processing systems, pages 161�168, 2008.

[81] Parikshit Ram, Alexander G Gray, Horst C Samulowitz, and Gregory Bramble. Toward

theoretical guidance for two common questions in practical cross-validation based hyper-
parameter selection. In Proceedings of the 2023 SIAM International Conference on Data
Mining (SDM), pages 802�810. SIAM, 2023.

[82] Jerry Yao-Chieh Hu, Bo-Yu Chen, Dennis Wu, Feng Ruan, and Han Liu. Nonparametric

modern hopfield models. arXiv preprint arXiv:2404.03900, 2024.

[83] Dennis Wu, Jerry Yao-Chieh Hu, Teng-Yun Hsiao, and Han Liu. Uniform memory retrieval
with larger capacity for modern hopfield models. arXiv preprint arXiv:2404.03900, 2024.

[84] Jerry Yao-Chieh Hu, Dennis Wu, and Han Liu. Provably optimal memory capacity for

modern hopfield models: Transformer-compatible dense associative memories as spherical
codes. In The Thirty-eighth Annual Conference on Neural Information Processing Systems,
2024.

[85] Sanjoy Dasgupta. The hardness of k-means clustering. UCSD Technical Report, 2008.

[86] Stuart Lloyd. Least squares quantization in PCM. IEEE transactions on information theory,

28(2):129�137, 1982.

BIBLIOGRAPHY

61

[87] J. C. Dunn. A fuzzy relative of the isodata process and its use in detecting compact

well-separated clusters. Journal of Cybernetics, 3:32�57, 1974.

[88] James C Bezdek. Pattern recognition with fuzzy objective function algorithms. Springer

Science & Business Media, 2013.

[89] Bishwajit Saha, Dmitry Krotov, Mohammed J Zaki, and Parikshit Ram. Deep clustering
with associative memories. In NeurIPS Workshop on Machine Learning and Compression,
2024.

[90] Erxue Min, Xifeng Guo, Qiang Liu, Gen Zhang, Jianjing Cui, and Jun Long. A survey of
clustering with deep learning: From the perspective of network architecture. IEEE Access,
6:39501�39514, 2018.

[91] Yazhou Ren, Jingyu Pu, Zhimeng Yang, Jie Xu, Guofeng Li, Xiaorong Pu, S Yu Philip,
and Lifang He. Deep clustering: A comprehensive survey. IEEE Transactions on Neural
Networks and Learning Systems, 2024.

[92] Sheng Zhou, Hongjia Xu, Zhuonan Zheng, Jiawei Chen, Zhao Li, Jiajun Bu, Jia Wu,

Xin Wang, Wenwu Zhu, and Martin Ester. A comprehensive survey on deep clustering:
Taxonomy, challenges, and future directions. ACM Computing Surveys, 57(3), November
2024.

[93] L�on Bottou. Large-scale kernel machines. MIT press, 2007.

[94] Parikshit Ram, Dongryeol Lee, William March, and Alexander Gray. Linear-time algorithms
for pairwise statistical problems. Advances in Neural Information Processing Systems, 22,
2009.

[95] Ryan Curtin, William March, Parikshit Ram, David Anderson, Alexander Gray, and Charles
Isbell. Tree-independent dual-tree algorithms. In International Conference on Machine
Learning, pages 1435�1443. PMLR, 2013.

[96] Ryan R Curtin, Dongryeol Lee, William B March, and Parikshit Ram. Plug-and-play

dual-tree algorithm runtime analysis. J. Mach. Learn. Res., 16:3269�3297, 2015.

[97] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. Advances

in neural information processing systems, 2007.

[98] Felix Xinnan X Yu, Ananda Theertha Suresh, Krzysztof M Choromanski, Daniel N
Holtmann-Rice, and Sanjiv Kumar. Orthogonal random features. Advances in neural
information processing systems, 29, 2016.

[99] Krzysztof M Choromanski, Mark Rowland, and Adrian Weller. The unreasonable effec-
tiveness of structured random orthogonal embeddings. Advances in neural information
processing systems, 30, 2017.

[100] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea

Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al.
Rethinking attention with performers. Proceedings of ICLR, 2020.

BIBLIOGRAPHY

62

[101] Valerii Likhosherstov, Krzysztof Marcin Choromanski, Kumar Avinava Dubey, Frederick

Liu, Tamas Sarlos, and Adrian Weller. Dense-exponential random features: Sharp positive
estimators of the gaussian kernel. In Thirty-seventh Conference on Neural Information
Processing Systems, 2023.

[102] Purushottam Kar and Harish Karnick. Random feature maps for dot product kernels. In

Artificial intelligence and statistics, pages 583�591. PMLR, 2012.

[103] Raffay Hamid, Ying Xiao, Alex Gittens, and Dennis DeCoste. Compact random feature
maps. In International conference on machine learning, pages 19�27. PMLR, 2014.

[104] Fanghui Liu, Xiaolin Huang, Yudong Chen, and Johan AK Suykens. Random features for
kernel approximation: A survey on algorithms, theory, and beyond. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 44(10):7128�7148, 2021.

[105] Benjamin Hoover, Duen Horng Chau, Hendrik Strobelt, Parikshit Ram, and Dmitry Krotov.
Dense associative memory through the lens of random features. In The Thirty-eighth Annual
Conference on Neural Information Processing Systems, 2024.

[106] Matt P Wand and M Chris Jones. Kernel smoothing. CRC press, 1994.

[107] Vassiliy A Epanechnikov. Non-parametric estimation of a multivariate probability density.

Theory of Probability & Its Applications, 14(1):153�158, 1969.

[108] Hans-Georg M�ller. Smooth optimum kernel estimators of densities, regression curves and

modes. The Annals of Statistics, pages 766�774, 1984.

[109] Benjamin Hoover, Krishna Balasubramanian, Dmitry Krotov, and Parikshit Ram. Dense
associative memory with epanechnikov energy. In New Frontiers in Associative Memories,
2025.

[110] Mallory A Snow and Jeff Orchard. Biological softmax: Demonstrated in modern hopfield
networks. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 44,
2022.

[111] Danil Tyulmankov, Kim Stachenfeld, Dmitry Krotov, and Larry Abbott. Memorization

and consolidation in associative memory networks. In Associative Memory
Networks in 2023, 2023.

&

{\

}

Hopfield

[112] Sarthak Chandra, Sugandha Sharma, Rishidev Chaudhuri, and Ila Fiete. Episodic and
associative memory from spatial scaffolds in the hippocampus. Nature, pages 1�13, 2025.

[113] Mohadeseh Shafiei Kafraj, Dmitry Krotov, Brendan A Bicknell, and Peter E Latham.
In New Frontiers in Associative

A biologically plausible associative memory network.
Memories, 2025.

[114] Kaining Zhang and Gaia Tavoni. Maximizing memory capacity in heterogeneous networks.

PRX Life, 3(2):023016, 2025.

[115] Elena Agliari, Francesco Alemanno, Adriano Barra, Martino Centonze, and Alberto Fachechi.

BIBLIOGRAPHY

63

Neural networks with a redundant representation: Detecting the undetectable. Physical
review letters, 124(2):028301, 2020.

[116] Elena Agliari and Giordano De Marzo. Tolerance versus synaptic noise in dense associative

memories. The European Physical Journal Plus, 135(11):1�22, 2020.

[117] Linda Albanese, Francesco Alemanno, Andrea Alessandrelli, and Adriano Barra. Replica
symmetry breaking in dense hebbian neural networks. Journal of Statistical Physics,
189(2):24, 2022.

[118] Elena Agliari, Linda Albanese, Francesco Alemanno, Andrea Alessandrelli, Adriano Barra,

Fosca Giannotti, Daniele Lotito, and Dino Pedreschi. Dense hebbian neural networks: a
replica symmetric picture of supervised learning. Physica A: Statistical Mechanics and its
Applications, 626:129076, 2023.

[119] Robin Th�riault and Daniele Tantari. Dense hopfield networks in the teacher-student

setting. SciPost Physics, 17(2):040, 2024.

[120] David G Clark. Transient dynamics of associative memory models. arXiv preprint

arXiv:2506.05303, 2025.

[121] Kazushi Mimura, Jun�ichi Takeuchi, Yuto Sumikawa, Yoshiyuki Kabashima, and An-
thony CC Coolen. Dynamical properties of dense associative memory. arXiv preprint
arXiv:2506.00851, 2025.

[122] Flavio Nicoletti, Francesco D�Amico, and Matteo Negri. Statistical mechanics of vector
hopfield network near and above saturation. arXiv preprint arXiv:2507.02586, 2025.

[123] Mikhail S Burtsev, Yuri Kuratov, Anton Peganov, and Grigory V Sapunov. Memory

transformer. arXiv preprint arXiv:2006.11527, 2020.

[124] Zexue He, Leonid Karlinsky, Donghyun Kim, Julian McAuley, Dmitry Krotov, and Rogerio

Feris. Camelot: Towards large language models with training-free consolidated associative
memory. arXiv preprint arXiv:2402.13449, 2024.

[125] Ivan Rodkin, Yuri Kuratov, Aydar Bulatov, and Mikhail Burtsev. Associative recurrent

memory transformer. arXiv preprint arXiv:2407.04841, 2024.

[126] Yu Wang, Dmitry Krotov, Yuanzhe Hu, Yifan Gao, Wangchunshu Zhou, Julian McAuley,

Dan Gutfreund, Rogerio Feris, and Zexue He. M+: Extending memoryllm with scalable
long-term memory. arXiv preprint arXiv:2502.00592, 2025.

[127] Andreas F�rst, Elisabeth Rumetshofer, Johannes Lehner, Viet T Tran, Fei Tang, Hubert

Ramsauer, David Kreil, Michael Kopp, G�nter Klambauer, Angela Bitto, et al. Cloob:
Modern hopfield networks with infoloob outperform clip. Advances in neural information
processing systems, 35:20450�20468, 2022.

[128] Yuchen Liang, Dmitry Krotov, and Mohammed J Zaki. Modern hopfield networks for graph

embedding. Frontiers in big Data, 5:1044709, 2022.

BIBLIOGRAPHY

64

[129] Ryo Karakida, Toshihiro Ota, and Masato Taki. Hierarchical associative memory, parallelized

mlp-mixer, and symmetry breaking. arXiv preprint arXiv:2406.12220, 2024.

[130] Xueyan Niu, Bo Bai, Lei Deng, and Wei Han. Beyond scaling laws: Understanding
transformer performance with associative memory. arXiv preprint arXiv:2405.08707, 2024.

[131] Michael Widrich, Bernhard Sch�fl, Milena Pavlovi?, Hubert Ramsauer, Lukas Gruber,

Markus Holzleitner, Johannes Brandstetter, Geir Kjetil Sandve, Victor Greiff, Sepp Hochre-

iter, et al. Modern hopfield networks and attention for immune repertoire classification.
Advances in neural information processing systems, 33:18832�18845, 2020.

[132] Qian Zhang, Dmitry Krotov, and George Em Karniadakis. Operator learning for recon-
structing flow fields from sparse measurements: an energy transformer approach. arXiv
preprint arXiv:2501.08339, 2025.

[133] Takeshi Kimura and Kohtaro Kato. Analysis of discrete modern hopfield networks in open

quantum system. arXiv preprint arXiv:2411.02883, 2024.

[134] Khalid Musa, Santosh Kumar, Michael Katidis, and Yu-Ping Huang. Dense associative
memory in a nonlinear optical hopfield neural network. arXiv preprint arXiv:2506.07849,
2025.


