Article

https://doi.org/10.1038/s41467-025-61475-w

Explosive neural networks via higher-order
interactions in curved statistical manifolds

Received: 2 September 2024

Accepted: 23 June 2025

Check for updates

;
,
:
)
(

0
9
8
7
6
5
4
3
2
1

;
,
:
)
(

0
9
8
7
6
5
4
3
2
1

Miguel Aguilera 1,2
Hideaki Shimazaki

, Pablo A. Morales3,4, Fernando E. Rosas

5,6,7,8 &

9,10

Higher-order interactions underlie complex phenomena in systems such as
biological and arti?cial neural networks, but their study is challenging due to
the scarcity of tractable models. By leveraging a generalisation of the maximum
entropy principle, we introduce curved neural networks as a class of models
with a limited number of parameters that are particularly well-suited for
studying higher-order phenomena. Through exact mean-?eld descriptions, we
show that these curved neural networks implement a self-regulating annealing
process that can accelerate memory retrieval, leading to explosive order-
disorder phase transitions with multi-stability and hysteresis effects. Moreover,
by analytically exploring their memory-retrieval capacity using the replica trick,
we demonstrate that these networks can enhance memory capacity and
robustness of retrieval over classical associative-memory networks. Overall, the
proposed framework provides parsimonious models amenable to analytical
study, revealing higher-order phenomena in complex networks.

Complex physical, biological, and social systems often exhibit higher-
order interdependencies that cannot be reduced to pairwise interac-
tions between their components1,2. Recent studies suggest that higher-
order organisation is not the exception but the norm, providing var-
ious mechanisms for its emergence3�6. Modelling studies have revealed
that higher-order interactions (HOIs) underlie collective activities such
as bistability, hysteresis, and �explosive� phase transitions associated
with abrupt discontinuities in order parameters4,7�11.

HOIs are particularly important for the functioning of biological
and arti?cial neural systems. For instance, they shape the collective
activity of biological neurons12,13, being directly responsible for their
inherent sparsity5,13�15 and possibly underlying critical dynamics16,17.
HOIs have also been shown to enhance the computational capacity of
arti?cial recurrent neural networks18,19. More speci?cally, �dense asso-
ciative memories� with extended memory capacity20�23 are realised by
speci?c non-linear activation functions, which effectively incorporate
HOIs. These non-linear functions are related to attention mechanisms

of transformer neural networks24 and the energy landscape of diffusion
models25,26, leading to the conjecture that HOIs underlie the success of
these state-of-the-art deep learning models.

Despite their importance, existent studies of HOIs face signi?cant
computational challenges. Analytically tractable models that incor-
porate HOIs typically limit interactions to a single order (e.g., p-spin
models22,27,28). Otherwise, attempting to represent diverse HOIs
exhaustively results in a combinatorial explosion29. This issue is per-
vasive, restricting investigations of high-order interaction models�
such as contagion9, Ising19, or Kuramoto30 models�to highly homo-
geneous scenarios3,16 or to models of relatively low-order9,11,31. While
attempts have been made to model all orders of HOIs and perform
theoretical analyses20�23,32�37, it is currently unclear how to construct
parsimonious models to address the diverse effects of HOIs in a
principled manner.

To address this challenge, here we employ an extension of the
maximum entropy principle to capture HOIs through the deformation

1BCAM � Basque Center for Applied Mathematics, Bilbao, Spain. 2IKERBASQUE, Basque Foundation for Science, Bilbao, Spain. 3Research Division, Araya Inc.,
Tokyo, Japan. 4Centre for Complexity Science, Imperial College London, London, UK. 5Sussex AI and Sussex Centre for Consciousness Science, Department
of Informatics, University of Sussex, Brighton, UK. 6Department of Brain Sciences and Centre for Complexity Science, Imperial College London, London, UK.
7Center for Eudaimonia and Human Flourishing, University of Oxford, Oxford, UK. 8Principles of Intelligent Behavior in Biological and Social Systems (PIBBSS),
Prague, Czech Republic. 9Graduate School of Informatics, Kyoto University, Kyoto, Japan. 10Center for Human Nature, Arti?cial Intelligence, and Neuroscience
(CHAIN), Hokkaido University, Sapporo, Japan.

e-mail: maguilera@bcamath.org

Nature Communications |

 (2025) 16:6511

1

Article

https://doi.org/10.1038/s41467-025-61475-w

of the space of statistical models. When applied to neural networks,
our approach generalises classical neural network models to yield a
family of curved neural networks that effectively incorporate HOIs of all
orders. The resulting models have rich connections with the literature
on the statistical physics of neural networks21,22,27,34. These features
enable the exploration of various aspects of HOIs using techniques
including mean-?eld approximations, quenched disorder analyses,
and path integrals.

Our analyses reveal how relatively simple curved neural networks
exhibit some of the hallmark characteristics of higher-order phenom-
ena, such as explosive phase transitions, arising both in mean-?eld
models and in more complex transitions to spin-glass states. These
phenomena are driven by a self-regulated annealing process, which
accelerates memory retrieval through positive feedback between
energy and an �effective� temperature�a perspective that can also
explain memory-retrieval dynamics in other modern arti?cial net-
works. Furthermore, we show�both analytically and experimentally�
that this mechanism can lead to an increase in the memory capacity or
robustness of memory retrieval in these neural networks. Overall, the
core contributions of this work are (i) the development of a parsimo-
nious neural network model based on the maximum entropy principle
that captures interactions of all orders, (ii) the discovery of a self-
regulated annealing mechanism that can drive explosive phase tran-
sitions, and (iii) the demonstration of enhanced memory capacity
resulting from this mechanism.

Results
High-order interactions in curved manifolds
The maximum entropy principle (MEP) is a general modelling frame-
work based on the principle of adopting the model with maximal
entropy compatible with a given set of observations, under the ratio-
nale that one should not assume any structure beyond what is speci-
?ed by the assumptions or features selected from the data38,39. The
traditional formulation of the MEP is based on Shannon�s entropy40,
and the resulting models correspond to Boltzmann distributions of the
, where x = (x1, �, xn), ? is a nor-
form p�x� = exp
malising potential, and ?a are parameters constraining the average
value of observables h f a�x�i. While observables are often set to low
orders (e.g. fi(x) = xi, fij(x) = xixj, corresponding to ?rst and second
order statistics), higher-order interdependencies can be included by
considering observables of the type fI(x) = ?i?Ixi, where I is a set of
indices of order k = ?I?. Unfortunately, an exhaustive description of
interactions up to order k ? 1 becomes unfeasible in practice due to an
exponential number of terms (for more details on the MEP, see Sup-
plementary Note 1).

af a�x� (cid:2) ?
?

P

(cid:3)

(cid:2)

a

The MEP can be expanded to include other entropy functionals
such as Tsallis�41 and R�nyi�s42. Concretely, maximising the R�nyi
entropy (with the scaling parameter ? ? ?1)43

H?�p� = (cid:2)

1
? ln

X

x

p�x�1 + ?

�1�

while constraining h f a�x�i (i.e., the expectation of features by p(x))
results in models of the form (see Supplementary Note 1):

"

p?�x� = exp�(cid:2)??� 1 + ??

#

1=?

?

af a�x�

,

+

X

a

where ?? is a normalising constant given by

?? = ln

"

X

x

1 + ??

X

a

#
1=?

?

af a�x�

:

+

�2�

�3�

Above, the square bracket operator sets negative values to zero,
x�
(cid:3) + = maxf0, xg. We refer to distributions following (2) as the
deformed exponential family, which maximises both R�nyi and Tsallis
entropies44,45. When ? ? 0, R�nyi�s entropy tends to Shannon�s and (2)
to the standard exponential family42.

A fundamental insight explored in this study is that higher-order
interdependencies can be ef?ciently captured by deformed exponen-
tial family distributions46,47. Starting from a standard Shannon�s MEP
model with low-order interactions, it can be shown that varying ? in (2)
results in a deformation of the statistical manifold which, in turn,
enhances the capability of p?(x) to account for higher-order inter-
dependencies. In effect, the consequence of deformation can be
investigated by rewriting (2) via Taylor expansion of the exponent

0

p?�x� = exp

@

X1

k = 1

(cid:2)1
k?

(cid:2)??

X

a

!
k

1

?

af a�x�

(cid:2) ??

A,

�4�

which is valid for the case 1 + ??a?afa(x) > 0, and otherwise p?(x) = 0.
This shows that the deformed manifold contains interactions of all
orders even if fa(x) is restricted to lower orders while establishing a
speci?c dependency structure across the orders, thereby avoiding a
combinatorial explosion of the number of required parameters. The
deformation resulting from the maximisation of a non-Shannon
entropy has been shown to re?ect a curvature of the space of possible
models in information geometry42,45,48,49. This leads to a particular
foliation of the space of possible models50 (an �onion-like� manifold
structure, Fig. 1), which has properties that allow to re-derive the MEP
from fundamental geometric properties�for technical details, see
Supplementary Note 1.

Curved neural networks
Several well-known neural network models adhere to the MEP, such as
Ising-like models51 and Boltzmann machines52. Interestingly, these
models can encode patterns in their weights in the form of �associative
memories� as in Nakano-Amari-Hop?eld networks53�55, being amenable
for investigations using tools from equilibrium and nonequilibrium
statistical physics literature56�59. Following the principles laid down in
the previous section, we now introduce a family of recurrent neural
networks that we call curved neural networks.

For this purpose, let us consider N binary variables x1, �, xN taking

values in {1, ?1} following a joint probability distribution

p?�x� = exp�(cid:2)??� 1 (cid:2) ??E�x�

�

(cid:3)1=?
+ ,

�5�

where ?? is a normalising constant. Above, we call E(x) and ? the
(stochastic) energy function (i.e., Hamiltonian) and the inverse tem-
perature, due to their similarity with the Gibbs distribution in statistical
physics when ? ? 0. Note that, unlike exponential families, these
models do not exhibit energy invariance under constant shifts. How-
ever, as demonstrated in Ref. 41, deformed exponential models can be
related to energy-invariant models by rescaling their temperature,
which can be seen as maximising entropy with respect to escort
statistics rather than the original natural statistics.

Neural network models are typically de?ned by considering p?(x)

as de?ned in (5) with an energy function of the form

E�x� = (cid:2)

XN

i = 1

Hixi (cid:2)

1
N

X

i < j

Jijxixj,

�6�

where Jij is the coupling strength between neurons xi and xj, and Hi are
bias terms. In the limit ? ? 0, p0(x) recovers the Ising model. Emulating
classical associative memories, the weights Jij can be made to encode a
collection of M neural patterns ? a = f? a
1 = �1 and a = 1, �, M

1 , . . . ? a

Ng, ? a

Nature Communications |

 (2025) 16:6511

2

Article

https://doi.org/10.1038/s41467-025-61475-w

Mean-?eld behaviour of curved associative-memory networks
As with regular associative memories58, one can solve the behaviour of
curved associative-memory networks through mean-?eld methods in
the thermodynamic limit N ? ? (Supplementary Note 3). Here the
energy is extensive, meaning that it scales with the system�s size N. To
ensure the deformation parameter remains independent of system
properties such as size or temperature, we scale it as follows:

? =

?0
N?

:

�10�

Fig. 1 | Higher-order decomposition resulting from the foliation of a statistical
manifold. Illustration of a family of standard MEP models (right) and its deformed
counterpart (bottom left). The space of MEP distributions with constraints of dif-
ferent orders constitute nested sub-manifolds29, giving rise to a hierarchy of sub-
families of models of the form E?
E?
1 (cid:4) E?
E?
?E0
k
higher-order terms in (4), and therefore certain subsets of E?
imate E0
r .

(cid:4)
? �x� = e(cid:2)?? 1 (cid:2) ??Ek �x�
42. The foliation depends on the curvature ?, and in general
k \ E0

? ? for k < r. For small values of ???, it is possible to neglect
k effectively approx-

2 (cid:4) (cid:5) (cid:5) (cid:5) (cid:4) E?
k but rather E?

k = fp�k�

g such that

1=?
+

(cid:5)

n

r

by using the well-known Hebbian rule55,56

Jij = J

XM

a = 1

? a
i

? a
j ,

�7�

where J is a scaling parameter.

Before proceeding with our main analysis, one can gain insights
into the effect of the curvature ? from the dynamics of a recurrent
neural network that behaves as a sampler of the equilibrium distribu-
tion described by (5). For this, we adapt the classic Glauber dynamics
to curved neural networks (see Supplementary Note 2) to obtain

(cid:6)

(cid:4)

(cid:5)
1=?
ni� = 1 + 1 (cid:2) ??0�x�?E�x�
+

(cid:7)

(cid:2)1

,

�8�

p�xijx

P

the

where x\i denotes
xi,
?E�x� = 2xi�Hi + 1
j Jijxj� is the energy difference associated with
N
detailed balance, and ?0�x� is an effective inverse temperature given
by

all neurons

state of

except

?0�x� =

?
1 (cid:2) ??E�x�
�

(cid:3) +

:

�9�

Again, ? ? 0 recovers the classic Glauber dynamics and ?0�x� = ?. Thus,
the curvature affects the dynamics through the deformed nonlinear
activation function (8) and the state-dependent effective temperature
?0�x� (9), with higher ?0�x� inducing lower degrees of randomness in
the transitions. The effect of E(x) on ?0�x� depends then on the sign of
?. A negative ? increases ?0�x� during relaxation, reducing the
stochasticity of the dynamics and accelerating convergence to a low-
energy state. This, in turn, raises ?0, creating a positive feedback loop
between energy and effective temperature. The effect is similar to
simulated annealing, but the coupling of the energy and effective
inverse temperature lets the annealing scheduling self-regulate to
In contrast, positive ? decelerates the
accelerate convergence.
dynamics through negative feedback. Such accelerating or decelerat-
ing dynamics underlie non-trivial complex collective behaviours of the
curved neural networks, which will be examined in the subsequent
sections.

Under this condition, we calculate the normalising potential ?? by
introducing a delta integral and calculating a saddle-node solution,
? a
resulting in a set of order parameters m = {m1, �, mM}, ma = 1
i hxii
N
in the limit of size N ? ?. This calculation assumes 1 ? ??E(x) > 0 so that
�(cid:3) + operators can be omitted and ?? is differentiable. The solution
results in (for Hi = 0):

P

i

?? = N

?
?0 ln

?0
?

(cid:2)

XM

a = 1

?0NJm2
a

+

XN

i = 1

ln 2 cosh ?0J

where ?0 is given by

!

!

? a
i ma

,

XM

a = 1

?0 =

?
P

1 + ?0 1
2 J

,

am2
a

�11�

�12�

and the values of the mean-?eld variables ma are found from the
following self-consistent equations:

ma =

XN

i = 1

? a
i
N

tanh ?0J

!

? b
i mb

:

XM

b = 1

�13�

Similarly, using a generating functional approach59, we use the
Glauber rule in (8) to derive a dynamical mean-?eld given by path
integral methods (see Supplementary Note 4). This yields

_ma = (cid:2) ma +

XN

i = 1

? a
i
N

tanh ?0J

!

? b
i mb

,

XM

b = 1

�14�

where ?0 is de?ned as in (12) for each m. Note that in large systems, we
recover the classical nonlinear activation function, and the deforma-
tion affects the dynamics only through the effective temperature ?0.

Explosive phase transitions
To illustrate these ?ndings, let us focus on a neural network with a
single associative pattern (M = 1), which is similar to the Mattis model60
and equivalent to a homogeneous mean-?eld Ising model61 (with
i < jxixj) by changing a variable xi ? ?ixi. Rewriting
energy E�x� = (cid:2) 1
N J
(13), we ?nd that a one-pattern curved neural network follows a mean-
?eld model given by

P

(cid:3)
m = tanh ?0Jm

(cid:2)

,

?0 =

?
1 + ?0 1
2 Jm2

:

�15�

�16�

This result generalises the well-known Ising mean-?eld solution
m = tanh ?Jm�

�, which is recovered for ? = 0.

By evaluating these equations, one ?nds that the model exhibits
the usual order-disorder phase transition for positive and small

Nature Communications |

 (2025) 16:6511

3

Article

https://doi.org/10.1038/s41467-025-61475-w

Fig. 2 | Explosive phase transitions in curved neural networks. a Phase transi-
tions of the curved neural network with one associative memory, for J = 1 and values
of ?0 = (cid:2)0:5 (top, displaying a second-order phase transition) and ?0 = (cid:2)1:5 (bot-
tom, displaying an explosive phase transition). Solid lines represent the stable ?xed
points, and dotted lines correspond to unstable ?xed points. b Phase diagram of
the system. The areas indicated by P and M refer to the usual paramagnetic (dis-
ordered) and magnetic (ordered) phases, respectively. The area indicated by Exp
represents a phase where ordered and disordered states coexist in an explosive

phase transition characterised by a hysteresis loop. (c) Solutions of (15)-(16) for
?0, m, ? (black line) for ?0 = (cid:2)1:2, and projections to the plane m = 0, ? = 0 and ?0 = 0,
obtaining respectively the relation between ?, ?0 and solutions of the ?at and the
deformed models respectively (grey lines). (d) Mean-?eld dynamics of the single-
pattern neural network for ? = 1.001 (near criticality from the ordered phase) for
some values of ?0 in [ ?1.5, 0]. For large negative ?0 the dynamics �explodes�, with m
(top) and ?0 (bottom) converging abruptly.

negative values of ?0 (Fig. 2a top). However, for large negative values of
?0, a different behaviour emerges: an explosive phase transition8 that
displays hysteresis due to HOIs (Fig. 2a bottom). The resulting phase
diagram (Fig. 2b) closely resembles phase transitions in higher-order
contagion models9,11 and higher-order synchronisation observed in
Kuramoto models30.

One can intuitively interpret the effect of the deformation para-
meter ?0 by noticing that, for a ?xed ?0, m is the solution of a function of
?0. For ?0 = 0, this results in the mean-?eld behaviour of the regular
exponential model, which assigns a value of m to each inverse tem-
perature ? = ?0. In the case of the deformed model, the possible pairs of
solutions �m, ?0� are the same, but their mapping to the inverse tem-
peratures ? changes. Namely, this deformation can be interpreted as a
stretching (or contraction) of the effective temperature, which maps
each pair �m, ?0� to an inverse temperature ? = ?0�1 + 1
?0Jm2� according
2
to (16). Thus, one can obtain the mean-?eld solutions of the deformed
patterns as mappings of the solutions of the original model. This is
illustrated in Fig. 2c, where the solution of ?0, m, ? is projected to the
planes ? = 0 and ?0 = 0, obtaining the solutions for the ?at (?0 = 0) and
the deformed (?0 = (cid:2) 1:2) models respectively.

In order to gain a deeper understanding of the explosive nature of
this phase transition, we study the dynamics of the single-pattern
neural network. By rewriting (14) for M = 1, and under the change of
variables mentioned above to remove ?, the dynamical mean-?eld
equation of the system reduces to

saturates, creating a positive feedback loop between ?0 and m that
gives rise to the explosive nature of the phase transition. This positive
loop occurs only if ?0 is negative; otherwise, negative feedback simply
makes the convergence of m slower.

Overlaps between memory basins of attraction
A key property of associative-memory networks is their ability to
retrieve patterns in different contexts. In the case of one-pattern asso-
ciative-memory networks, the energy function E�x� = (cid:2) J
jxj is
N
a quadratic function with two minima at x = � ?, which con?gure global
attractors. Instead, a two-pattern associative-memory network has an
energy function with four minima (if suf?ciently separated), but their
attraction basins can overlap when the patterns are correlated.

i < jxi

P

?

?

i

To study the degree of the overlap between pairs of patterns, we
analyse solutions of (13) for a network with two patterns with corre-
lation h? 1
? 2
i i = C (see Supplementary Note 3.3 for details). In this sce-
i
nario, the system is described by two mean-?eld patterns:

ma =

1
2

(cid:3)
(cid:2)
�1 + C� tanh ?0J�m1 + m2�

(cid:3)
(cid:2)
�1 (cid:2) C� tanh ?0J�m1 (cid:2) m2�

+ w

1
2

with w = 3 ? 2a = � 1 for a = 1, 2, and

�18�

�19�

_m = (cid:2) m + tanh ?0Jm

(cid:2)

(cid:3)

,

�17�

?0 =

?
2 J�m2

1 + ? 1

1 + m2
2�

:

where ?0 is calculated as in (16). Simulations of the dynamical mean-
?eld equations for values of ? just above the critical point are depicted
in Fig. 2d. Trajectories with strongly negative ?0 saturate earlier than
smaller negative ?0, con?rming accelerated convergence. During this
process, the effective inverse temperature ?0 rapidly increases until it

Figure 3 shows how the hysteresis effect and explosive phase
transitions persist in the case of two patterns for C = 0.2 with negative
?0. This example shows two consecutive, overlapping explosive bifur-
cations (going from 1 to 2, and then to 4 ?xed points), creating a
hysteresis involving 7 ?xed points within a more compressed

Nature Communications |

 (2025) 16:6511

4

Article

https://doi.org/10.1038/s41467-025-61475-w

Fig. 3 | Interaction between two encoded memories. a Values of ?? for different
mean-?eld values m1, m2, indicating the attractor structure of the network for
different values of ? with J = 1, C = 0.2 for ?0 = 0 (top row) and ?0 = (cid:2) 1:2 (bottom
row). b Bifurcations of the order parameters m1, m2. For ?0 = 0 we observe an

attractor bifurcating into two and then into four. For ?0 = (cid:2) 1:2, we observe the
same sequence, but with a coexistence hysteresis regime in which 7 attractors are
possible.

parameter range of ? than the classical case. Consequently, the
memory-retrieval region for the four embedded memories expands.
These results illustrate complex hysteresis cycles as well as an
increased memory capacity for ?nite temperatures by negative values
of ?0. This enhanced capability for memory retrieval is further inves-
tigated through the replica analyses in the next section.

P

P

? a
i

M
a = 1

i < jxi

Memory retrieval with an extensive number of patterns
Next, we investigate how the deformation related to ? impacts the
memory-storage capacity of associative memories. In classical asso-
ciative networks of N neurons, the energy function is de?ned as
? a
j xj with M = ?N. As the number of patterns
E�x� = (cid:2) J
N
learned by the network increases, the system transitions to a dis-
ordered spin-glass state in the thermodynamic limit. Furthermore, one
can analytically solve this model62�65. For example, using the replica-
trick method can determine the memory capacity of the system62, and
theoretically identify the critical value of ? at which memory retrieval
becomes impossible�leading to a disordered spin-glass phase. Here,
we apply a similar approach to reveal how deformed associative
memory networks afford an enhanced memory capacity.

Applying the replica trick in conjunction with the methods out-
lined in previous sections allows us to solve the system (see Supple-
mentary Note 5). This method entails computing a mean-?eld variable
m corresponding to one of the patterns ? a and averaging over the
others. For simplicity, a pattern with all positive unity values
? a = (1, 1, �, 1) is considered, which is equivalent to any other single
pattern just by a series of sign ?ip variable changes. The degree of
similarity or overlap of this pattern with other patterns in the system
introduces a new order parameter q, which contributes to measuring
disorder in the system. After introducing the relevant order para-
meters and solving under a replica-symmetry assumption, the nor-
malising potential is derived as

?? = N

?
?0 ln
1
(cid:2) N
2
Z

+ N

?�?0J�2�r + R (cid:2) 2qr�

?
?0 (cid:2) N?0Jm2 (cid:2) N
(cid:2)
? ln 1 (cid:2) ?0J�1 (cid:2) q�

1
2

(cid:2)

(cid:2)

(cid:2)

Dz ln 2 cosh ?0Jm + ?0J

p

(cid:3)

(cid:2) ?0J

p

(cid:3)
?????
rq
??????
?r
z

�20�

(cid:3)

,

(cid:3)

where J is a scaling factor, and the order parameters are de?ned as

Z

Z

m =

q =

(cid:2)
Dz tanh ?0Jm + ?0J

p

(cid:3)

??????
?r
z

,

(cid:2)
Dz tanh2 ?0Jm + ?0J

p

(cid:3)

??????
?r
z

,

with

r =

q

�1 (cid:2) ?0J�1 (cid:2) q��2 , R =

�?0J�(cid:2)1 (cid:2) �1 (cid:2) 2q�
�1 (cid:2) ?0J�1 (cid:2) q��2

:

�21�

�22�

�23�

As in previous cases, the model is governed by an effective tempera-
ture

?0 =

(cid:2)

1 + ?0 1
2

Jm2 + ?J�?0�R (cid:2) qr� (cid:2) 1�

(cid:3) :

�24�

?

This solution differs from the models in previous sections by the self-
dependence of ?0.

To obtain a phase diagram, we solved (21)-(22) numerically for
given ?, ?0 at ?0 = 0, and rescaled the inverse temperature as in the
previous section to obtain the corresponding values of ? for each ?0.
Using the resulting order parameters and calculating the free energy
for each ?, ?, ?0, we constructed the phase diagram of the system
(similarly to regular associative memories58,62) characterised by the
following distinct phases (Fig. 4):

(cid:129) A paramagnetic phase (P), corresponding to disordered solutions
with m = q = 0, where memory-retrieval fails due to the dominance
of ?uctuations.

(cid:129) A ferromagnetic phase (F), corresponding to stable memory-

retrieval solutions with m > 0 and q > 0.

(cid:129) A spin-glass phase (SG), exhibiting spurious-retrieval solutions

with m = 0 and q > 0.

(cid:129) A mixed phase (M), where F and SG types of solutions coexist,
being the spin-glass solutions a global minimum of the normal-
ising potential ??.

Nature Communications |

 (2025) 16:6511

5

Article

https://doi.org/10.1038/s41467-025-61475-w

100 dataset, which comprises 60,000 32 � 32 colour images66. To
adapt the dataset to binary patterns suitable for storage in an asso-
ciative memory, we processed each RGB channel by assigning a value
of 1 to pixels with values greater than the channel�s median value and
?1 otherwise (Fig. 5a). The resulting array of N = 32 ? 32 ? 3 binary values
for each image was assigned to patterns ? a. Note that associative
memories (as well as our theory above) usually assume that patterns
are relatively uncorrelated, and speci?c methods are required to adapt
them to correlated patterns67,68. To simplify the problem, we con-
ducted experiments using a selection of 100 images with covariance
values smaller than 10=
(the standard deviation of the covariance
values for uncorrelated patterns is 1=
). We used a random search to
select patterns with low correlations: we randomly picked an image
and replaced it if its correlation exceeded the threshold, repeating
until all correlations were below it.

????
N

????
N

p

p

P

ixi

We evaluated the memory retrieval capacity of networks with
various degrees of curvature ? by encoding different numbers of
memories, as described in (7). As a measure of performance, we eval-
uated the stability of the network by assigning an initial state x = ? a and
? a
i after T = 30N Glauber updates for
calculating the overlap o =
? = 2, J = 1. The process was repeated R = 500 times from different initial
conditions (different encoded patterns and different initial states) to
estimate the value of m in (21). Experimental outcomes con?rm our
theoretical results, revealing that memory capacity increases with
negative values of ?0, while positive values reduce the memory capacity
(Fig. 5b), but reduce the extent and magnitude of the high variability
region in pattern retrieval (Fig. 5c), which is consistent with the
reduction of the mixed phase. Note that the resulting memory capacity
of the system observed in our experiments (i.e., the value of ? at which
the transition happens) is diminished due to the presence of correla-
tions among some of the memorised patterns.

Finally, we investigated transitions near the spin-glass phase
boundaries. First, we note that, for J ? 0 and ? = J?2, the model in (21)-
(22) converges to (see Supplementary Note 5)

Z

q =

(cid:2)

Dz tanh2 ?0

p

???
z
q

(cid:3)

,

?0 =

?

1 + 1
2

?0?0�1 (cid:2) q2�

,

�26�

�27�

Fig. 4 | Memory capacity is enhanced by geometric deformation. Phase diagram
of a curved associative memory with an extensive number of encoded patterns
M = ?N and J = 1 for (a) different T = 1/? at ?0 = 0 (black dashed lines), 0.8, ? 0.8 (solid
lines), and for (b) different ?0 at ? = 2. F indicates the ferromagnetic (i.e., memory
retrieval) phase, SG the spin-glass phase (where saturation makes memory retrieval
inviable), M a mixed phase, and P the paramagnetic region. Both in F and M,
ferromagnetic and spin-glass solutions coexist, but we differentiate these by cal-
culating respectively whether memory-retrieval or spin-glass solutions are the
global minimum of the normalising potential ??. The dotted lines in (a) near T = 0
indicate the AT lines, below which the replica-symmetric solution is not valid.
Increasing ?0 to larger negative values extends the retrieval phase into larger values
of ?, indicating an increased memory capacity, while larger positive values reduce
the extension of the mixed phase, increasing robustness of memory retrieval.

For ?0 = 0 (black dashed lines), the phase transition re?ects the
behaviour of associative memories near saturation58,62. With negative ?0
(red lines), we observe an expansion of the ferromagnetic and mixed
phases,
indicating an enhanced memory-storage capacity by the
deformation. Conversely, a positive value of ?0 (yellow lines) decreases
the memory capacity but reduces the extent of the mixed phase. In the
mixed phase, retrieved memories (m > 0) are represented at a local�
but not global�minimum of the normalising potential ?? in (20),
indicating a larger probability of observing spurious patterns. Thus, we
expect positive values of ?0 to result in more robust memory retrieval.
The stability of the replica symmetry solution is given by the

condition

(cid:2)

(cid:3)
1 + ?0�1 (cid:2) q�

2

Z

> ??02

Dz cosh(cid:2)4?0

(cid:2)

p

??????
?r
z

(cid:3)

,

Jm + J

�25�

which is captured by the dotted lines near zero temperature in Fig. 4a.
Note that all solutions in Fig. 4b are stable under the replica symmetry
assumption.

We complement the analysis from the previous section with an
experimental study of a system encoding patterns from an image
classi?cation benchmark. The patterns are sourced from the CIFAR-

which at ? = 0 recovers the well-known Sherrington-Kirkpatric model69
(see Supplementary Note 6). While in the classical case, a phase
transition occurs from a paramagnetic to a spin-glass phase, the
curvature effect of ?0?0 modi?es the nature of this transition. For small
values of ?0, the system exhibits a continuous phase transition akin to
the Sherrington�Kirkpatrick spin-glass, where dq
d? shows a cusp (Fig. 6a).
However, for ?0 = (cid:2)1 the phase transition becomes second-order,
displaying a divergence of dq
d? at the critical point (Fig. 6b). Moreover,
increasing the magnitude of negative ?0 leads to a ?rst-order phase
transition with hysteresis (Fig. 6c), resembling the explosive phase
transition observed in the single-pattern associative-memory network.
This hybrid phase transition combines the typical critical divergence of
a second-order phase transition with a genuine discontinuity, similar
to �type V� explosive phase transitions8.

We analytically calculated the properties of these phase transi-
tions (see Supplementary Note 6). By computing the solution at ?0 = 0
and rescaling ?0, we determined that the critical point is located at
?0 (consistent with Fig. 6a�c). The slope of the order para-
?
c = 1 + 1
2
meter around the critical point is, for ?0 ? (cid:2)1, equal to �1 + ?0�(cid:2)1, indi-
cating the onset of a second-order phase transition as depicted in
Fig. 6b. The resulting phase diagram of the curved Sherrington-
Kirkpatrick model is shown in Fig. 6d.

Nature Communications |

 (2025) 16:6511

6

Article

https://doi.org/10.1038/s41467-025-61475-w

Fig. 6 | Explosive spin glasses. Phase transitions for order parameter q for replica-
symmetric disordered spin models displaying (a) a cusp phase transition for
?0 = (cid:2)0:5, (b) a second-order phase transition for ?0 = (cid:2)1:0 and (c) an explosive
phase transition for ?0 = (cid:2)1:2. d Phase diagram of the explosive spin glass, dis-
playing a paramagnetic (P), spin-glass (SG) and an explosive phase (Exp).

Fig. 5 | Simulation study for the effect of deformation on image encoding.
a Examples of CIFAR-100 images (top) and their RGB binarised versions (bottom).
Every 32 � 32 � 3 binary RGB pixel value for each image a is assigned to the value of
one position of pattern ? a
obtained in experiments, measured by the overlap between the ?nal state of the
network and the encoded pattern.

i . b, c Mean and variance of pattern retrieval values

Comparison with other dense associative memory models
Although our primary objective is to develop a parsimonious model of
HOIs to explain higher-order phenomena, our framework can also be
used to explain the behaviour of modern networks with HOIs,
including the recently proposed relativistic Hop?eld model32�34 and
dense associative memories20,21. For this, let us consider the energy
F �E(cid:3) of the exponential family distribution p�x� (cid:6) e(cid:2)?F �E(cid:3) given by the
nonlinear transformation (denoted by F ) of the classical energy E(x).
The deformed exponential models in this study correspond to
?0 ln�1 (cid:2) ?0E=N�, while the relativistic model corresponds to
F �E(cid:3) = (cid:2) N
. For the deformed exponential, the term F �E(cid:3)
F �E(cid:3) = (cid:2) N
?0
can be expanded as

?????????????????????
1 (cid:2) ?0E=N

p

F �E(cid:3) = E +

?0

2N

E2 +

?02

3N2 E3 + . . .

�28�

i

a

(cid:2)

1
N

P

(cid:3)2

? a
i xi

P
i

P
i

? a
i xi. For ?0 < 0, all coef?cients of

When E depends on the quadratic Mattis magnetisation (i.e.,
), then F �E(cid:3) expands in terms of even-order HOIs
E = (cid:2)
P
? a
i xi in the expansion are
of
negative, indicating that embedded memories have deeper energy
minima than in the classical case. The same signs appear for each order
in the relativistic energy with ?0 < 0. We also note that ? in the free
energy of both the deformed exponential and relativistic models in the
limit of large N appears scaled according to an effective temperature
given by ?0 = ??
F �E(cid:3) (e.g., (11) and Eq. (6.2) in Ref. 34). Moreover, the
E
input in the Glauber dynamics is approximated for large sizes as

??F �E(cid:3) (cid:7) ??

E

F �E(cid:3) ?E�x� = ?0?E�x�:

�29�

for

The effective inverse temperatures ?0 = ?�1 (cid:2) ?0E=N�(cid:2)1
the
deformed exponential and ?0 = 2(cid:2)1�1 (cid:2) ?0E=N�(cid:2)1=2 for the relativistic
models are decreasing functions of E when ?0 < 0, resulting in an
acceleration of memory retrieval�with lower energy E resulting in
higher ?0 (lower temperature). While the relativistic model has been
studied for ?0 > 032�34, we conjecture it may exhibit explosive phase
transitions if ?0 < 0. Conversely, a positive ?0 introduces alternating
? a
i xi, and a shallower energy landscape
signs in even-order terms of
due to a reduction in ?0. This shallower energy landscape reduces the
memory capacity of the deformed exponential networks by expanding
the spin-glass phases (Fig. 4), but also enlarges the recall (ferromag-
netic) region by mitigating the formation of spurious memories given

P
i

Nature Communications |

 (2025) 16:6511

7

Article

https://doi.org/10.1038/s41467-025-61475-w

by overlapping patterns in the mixed phase (in alignment with
previous work32 on mitigation of spurious memories in the relativistic
model).

P

aF�

This perspective on accelerated memory retrieval by nonlinearity
extends to dense associative memories20,21, which achieve supralinear
memory capacities through nonlinear pattern encoding. Speci?cally,
P
? a
their energy function is given by F = (cid:2)
i xi� with F being e.g.,
i
function20, F�z� = z� (cid:3)p
+ or an exponential
a thresholded power
nonlinearity21 F(z) = ez at zero temperature. These nonlinearities narrow
basins of attraction, reducing memory overlap and preventing transi-
tions to the spin-glass phase. The jumps in the Glauber dynamics of
such systems are weighed by an accelerating function. Namely, from
our perspective, the dynamics of such systems can be described via
positive feedback on weights linked to a speci?c memory, which
increase during memory retrieval. This follows from the fact that,
k (cid:8) 2? a
relating the linear difference in Mattis terms ??a
k xk with the
(cid:3)
P
? a
i xi (cid:2) ??a
nonlinear difference ?F a
k (cid:8) F
, the update
i
of the kth neuron is determined by the sign of

P
i

? a
i xi

(cid:2) F

(cid:2)

(cid:2)

(cid:3)

k

?F �x� =

X

a

?F a
k
??a
k

??a

k =

X

a

wa
k

??a
k

:

�30�

?F a
k
??a
k

i

P

P
i

Here, we show that the effective weight wa
k (cid:8)
becomes an
P
? a
i xi when F is the power, exponential, or
increasing function of
i
more generally, a convex function (See Supplementary Note 7). Thus,
? a
i xi as pattern ? a is retrieved strengthens its basin of
increasing
attraction and ensures positive feedback. Meanwhile, retrieval of ? a
? b
i xi for orthogonal patterns ? b, lowering their weights,
reduces
suppressing their recall to minimise interference. This competitive
mechanism highlights the higher memory capacity of these models
compared to curved neural networks with uniform temperature scal-
ing. Unlike the effective inverse temperature in curved networks,
which depends only on the system�s state or energy, the effective
weight in updating the k-th neuron additionally depends on the neu-
ron�s state xk, thus no longer representing a global modulation of the
energy.

Discussion
HOIs play a critical role in enabling emergent collective phenomena in
natural and arti?cial systems. Modelling HOIs is, however, highly non-
trivial, often requiring advanced analytic tools (such as simplicial
complexes or hypergraphs) that entail an exponential increase in
parameters for large systems. In this paper, we addressed this issue by
leveraging the maximum entropy principle to effectively capture HOIs
in models via a deformation parameter ?, which is associated with the
R�nyi entropy. Given their close connection with statistical physics,
this family of models provides a useful setup to investigate the effect of
HOIs on spin systems, including explosive ferromagnetic and spin-
glass phase transitions, extending studies on anomalous phase tran-
sitions found in other systems2,7�9,11, and the capability of networks to
store memories.

The observed effects in curved neural networks can be explained
via an effective temperature, inducing a positive or negative feedback
effect in memory retrieval. As we discussed above, this effect is present
in different forms across other dense associative memories20,21,34. A
similar argument may apply to diffusion models framed within dense
associative memories25,26, where the energy follows a log-sum-exp
nonlinearity. Thus, the accelerated mechanism found in this study
clari?es memory retrieval in advanced associative networks, providing
an important step toward designing extended memory capacities and
improved noise scheduling.

Curved neural networks also provide insights into biological
neural systems, where evidence suggests the presence of alternating
positive and negative HOIs for even and odd orders, respectively. This
alternation leads to sparse neuronal activity, which has been shown to

be instrumental for enabling extended periods of total silence5,13�15,35.
Interestingly, such sparse activity patterns may coexist with the
accelerated memory retrieval dynamics, as both involve positive even-
order HOIs. The attainment of enhanced memory, combined with
sparse activity, presents a promising direction for understanding
energy-ef?cient biological neuronal networks35,36. Future work may
investigate how curved neural networks might support both energy
ef?ciency and high memory capacities, potentially by adopting a
thresholded, supralinear neuronal activation function20,35. Addition-
ally, developing statistical methods for ?tting these models to
experimental data (i.e., theories for learning) represents an important,
yet largely unexplored, research avenue. Together, these research
directions offer a compelling path to uncover the principles of ef?cient
information coding in biological neural systems.

Overall, our results demonstrate the bene?ts of considering the
maximum entropy principle, emergent HOIs, and nonlinear network
dynamics as theoretically intertwined notions. As showcased here,
such an integrated framework reveals how information encoding,
retrieval dynamics, and memory capacity in neural networks are
mediated by HOIs, providing principled, analytically tractable tools
and insights from statistical mechanics and nonlinear dynamics. More
generally, the framework presented in this work extends beyond
neural networks and contributes to a general theory of HOIs, paving
the road toward a principled study of higher-order phenomena in
complex networks.

Data availability
The CIFAR-100 dataset used in this study is available at https://www.cs.
toronto.edu/~kriz/cifar.html.

Code availability
The code generated in this study is available in the GitHub repository,
https://github.com/MiguelAguilera/explosive-neural-networks.

References
1.

Lambiotte, R., Rosvall, M. & Scholtes, I. From networks to optimal
higher-order models of complex systems. Nat. Phys. 15,
313�320 (2019).
Battiston, F. et al. The physics of higher-order interactions in com-
plex systems. Nat. Phys. 17, 1093�1098 (2021).

2.

5.

4.

3. Amari, S.-i, Nakahara, H., Wu, S. & Sakai, Y. Synchronous ?ring and
higher-order interactions in neuron pool. Neural Comput. 15,
127�142 (2003).
Kuehn, C. & Bick, C. A universal route to explosive phenomena. Sci.
Adv. 7, eabe3824 (2021).
Shomali, S. R., Rasuli, S. N., Ahmadabadi, M. N. & Shimazaki, H.
Uncovering hidden network architecture from spiking activities
using an exact statistical input-output relation of neurons. Com-
mun. Biol. 6, 169 (2023).
Thibeault, V., Allard, A. & Desrosiers, P. The low-rank hypothesis of
complex systems. Nat. Phys. 20, 294�302 (2024).
Angst, S., Dahmen, S. R., Hinrichsen, H., Hucht, A. & Magiera, M. P.
Explosive ising. J. Stat. Mech.: Theory Exp. 2012, L06002 (2012).

6.

7.

8. D�Souza, R. M., G�mez-Gardenes, J., Nagler, J. & Arenas, A. Explo-

sive phenomena in complex networks. Adv. Phys. 68,
123�223 (2019).
Iacopini, I., Petri, G., Barrat, A. & Latora, V. Simplicial models of
social contagion. Nat. Commun. 10, 2485 (2019).

9.

10. Mill�n, A. P., Torres, J. J. & Bianconi, G. Explosive higher-order

Kuramoto dynamics on simplicial complexes. Phys. Rev. Lett. 124,
218301 (2020).
Landry, N. W. & Restrepo, J. G. The effect of heterogeneity on
hypergraph contagion models. Chaos 30 (2020).

11.

12. Montani, F. et al. The impact of high-order interactions on the rate of

synchronous discharge and information transmission in

Nature Communications |

 (2025) 16:6511

8

Article

https://doi.org/10.1038/s41467-025-61475-w

somatosensory cortex. Philos. Trans. R. Soc. A: Math. Phys. Eng. Sci.
367, 3297�3310 (2009).

13. Tka?ik, G. et al. Searching for collective behavior in a large network
of sensory neurons. PLoS Comput. Biol. 10, e1003408 (2014).
14. Ohiorhenuan, I. E. et al. Sparse coding and high-order correlations
in ?ne-scale cortical networks. Nature 466, 617�621 (2010).
15. Shimazaki, H., Sadeghi, K., Ishikawa, T., Ikegaya, Y. & Toyoizumi, T.

Simultaneous silence organizes structured higher-order interac-
tions in neural populations. Sci. Rep. 5, 9821 (2015).

16. Tka?ik, G. et al. The simplest maximum entropy model for collective
behavior in a neural network. J. Stat. Mech.: Theory Exp. 2013,
P03011 (2013).

17. Tka?ik, G. et al. Thermodynamics and signatures of criticality in a
network of neurons. Proc. Natl Acad. Sci. 112, 11508�11513 (2015).

18. Burns, T. F. & Fukai, T. Simplicial Hop?eld networks. In: The Eleventh
International Conference on Learning Representations (2022).
19. Bybee, C. et al. Ef?cient optimization with higher-order Ising

37. Hoover, B., Chau, D. H., Strobelt, H., Ram, P. & Krotov, D. Dense

associative memory through the lens of random features. Adv.
Neural Inform. Process. Syst. 38 (2024).

38. Jaynes, E. T. Probability Theory: The Logic of Science (Cambridge

University Press, 2003).

39. Cofr�, R., Herzog, R., Corcoran, D. & Rosas, F. E. A comparison of the
maximum entropy principle across biological spatial scales.
Entropy 21, 1009 (2019).

40. Jaynes, E. T. Information theory and statistical mechanics. Phys. Rev.

106, 620 (1957).

41. Tsallis, C., Mendes, R. & Plastino, A. R. The role of constraints within
generalized nonextensive statistics. Phys. A: Stat. Mech. Appl. 261,
534�554 (1998).

42. Morales, P. A. & Rosas, F. E. Generalization of the maximum entropy

principle for curved statistical manifolds. Phys. Rev. Res. 3,
033216 (2021).

43. Valverde-Albacete, F. & Pel�ez-Moreno, C. The case for shifting the

machines. Nat. Commun. 14, 6033 (2023).

R�nyi entropy. Entropy 21, 46 (2019).

20. Krotov, D. & Hop?eld, J. J. Dense associative memory for pattern

recognition. Adv. Neural Inform. Process. Syst. 29 (2016).

21. Demircigil, M., Heusel, J., L�we, M., Upgang, S. & Vermet, F. On a
model of associative memory with huge storage capacity. J. Stat.
Phys. 168, 288�299 (2017).

22. Agliari, E. et al. Dense Hebbian neural networks: a replica symmetric

picture of unsupervised learning. Phys. A: Stat. Mech. Appl. 627,
129143 (2023).

23. Lucibello, C. & M�zard, M. Exponential capacity of dense associa-

tive memories. Phys. Rev. Lett. 132, 077301 (2024).

24. Krotov, D. A new frontier for Hop?eld networks. Nat. Rev. Phys. 5,

366�367 (2023).

44. Umarov, S., Tsallis, C. & Steinberg, S. On aq-central limit theorem
consistent with nonextensive statistical mechanics. Milan. J. Math.
76, 307�328 (2008).

45. Wong, T.-K. L. & Zhang, J. Tsallis and r�nyi deformations linked via a
new ?-duality. IEEE Trans. Inf. Theory 68, 5353�5373 (2022).
46. Guisande, N. & Montani, F. R�nyi entropy-complexity causality

space: a novel neurocomputational tool for detecting scale-free
features in EEG/iEEG data. Front. Comput. Neurosci. 18,
1342985 (2024).

47. Jauregui, M., Zunino, L., Lenzi, E. K., Mendes, R. S. & Ribeiro, H. V.
Characterization of time series via r�nyi complexity�entropy curves.
Phys. A: Stat. Mech. Appl. 498, 74�85 (2018).

25. Ambrogioni, L. In search of dispersed memories: Generative diffu-

48. Wong, T.-K. L. Logarithmic divergences from optimal transport and

sion models are associative memory networks. Entropy 26,
381 (2024).

26. Ambrogioni, L. The statistical thermodynamics of generative diffu-
sion models: Phase transitions, symmetry breaking, and critical
instability. Entropy 27, 291 (2025).

27. Bovier, A. & Niederhauser, B. The spin-glass phase-transition in the
Hop?eld model with p-spin interactions. Adv. Theor. Math. Phys. 5,
1001�1046 (2001).

28. Agliari, E., Fachechi, A. & Marullo, C. Nonlinear PDEs approach to

statistical mechanics of dense associative memories. J. Math. Phys.
63 (2022).

29. Amari, S.-i. Information geometry on hierarchy of probability dis-

tributions. IEEE Trans. Inf. theory 47, 1701�1711 (2001).

r�nyi geometry. Inf. Geom. 1, 39�78 (2018).

49. Vigelis, R. F., De Andrade, L. H. & Cavalcante, C. C. Properties of a
generalized divergence related to Tsallis generalized divergence.
IEEE Trans. Inf. Theory 66, 2891�2897 (2019).

50. Amari, S.-I. Information Geometry and its Applications Vol. 194

(Springer, 2016).

51. Roudi, Y., Dunn, B. & Hertz, J. Multi-neuronal activity and functional

connectivity in cell assemblies. Curr. Opin. Neurobiol. 32, 38�44
(2015).

52. Mont�far, G. in Information Geometry and Its Applications: On the
Occasion of Shun-ichi Amari�s 80th Birthday, IGAIA IV Liblice, Czech
Republic, June 2016, (eds Ay, N., Gibilisco, P. & Mat��, F.) 75�115
(Springer, 2018).

30. Skardal, P. S. & Arenas, A. Higher order interactions in complex

53. Nakano, K. Associatron-a model of associative memory. IEEE Trans.

networks of phase oscillators promote abrupt synchronization
switching. Commun. Phys. 3, 218 (2020).

31. Ganmor, E., Segev, R. & Schneidman, E. Sparse low-order interac-
tion network underlies a highly correlated and learnable neural
population code. Proc. Natl Acad. Sci. 108, 9679�9684 (2011).
32. Barra, A., Beccaria, M. & Fachechi, A. A new mechanical approach to
handle generalized Hop?eld neural networks. Neural Netw. 106,
205�222 (2018).

33. Agliari, E., Barra, A. & Notarnicola, M. The relativistic Hop?eld net-

Syst. Man Cybern. 3, 380�388 (1972).

54. Amari, S.-I. Learning patterns and pattern sequences by self-

organizing nets of threshold elements. IEEE Trans. Comput. 100,
1197�1206 (1972).

55. Hop?eld, J. J. Neural networks and physical systems with emergent

collective computational abilities. Proc. Natl Acad. Sci. 79,
2554�2558 (1982).

56. Amit, D. J. Modeling Brain Function: the World of Attractor Neural

Networks (Cambridge University Press, 1989).

work: rigorous results. J. Math. Phys. 60 (2019).

57. Coolen, A. C., K�hn, R. & Sollich, P. Theory of Neural Information

34. Agliari, E., Alemanno, F., Barra, A. & Fachechi, A. Generalized

Processing Systems (OUP Oxford, 2005).

guerra�s interpolation schemes for dense associative neural net-
works. Neural Netw. 128, 254�267 (2020).

58. Coolen, A. In Handbook of Biological Physics (eds Moss, F. &

Gielen, S.) Vol. 4, 553�618 (Elsevier, 2001).

35. Rodr�guez-Dom�nguez, U. & Shimazaki, H. Alternating shrinking
higher-order interactions for sparse neural population activity.
Preprint at https://arxiv.org/abs/2308.13257 (2023).

36. Santos, S., Niculae, V., McNamee, D. & Martins, A. F. Hop?eld-

fenchel-young networks: a uni?ed framework for associative
memory retrieval. Preprint at https://arxiv.org/abs/2411.08590
(2024).

59. Coolen, A. In Handbook of Biological Physics (eds Moss, F. &

Gielen, S.) Vol. 4, 619�684 (Elsevier, 2001).

60. Mattis, D. Solvable spin systems with random interactions. Phys.

Lett. A 56, 421�422 (1976).

61. Kochma?ski, M., Paszkiewicz, T. & Wolski, S. Curie�Weiss magnet�
a simple model of phase transition. Eur. J. Phys. 34, 1555 (2013).

Nature Communications |

 (2025) 16:6511

9

Article

https://doi.org/10.1038/s41467-025-61475-w

62. Amit, D. J., Gutfreund, H. & Sompolinsky, H. Storing in?nite num-
bers of patterns in a spin-glass model of neural networks. Phys. Rev.
Lett. 55, 1530 (1985).

63. Bovier, A., Gayrard, V. & Picco, P. Gibbs states of the Hop?eld model

with extensively many patterns. J. Stat. Phys. 79, 395�414 (1995).
64. Talagrand, M. Rigorous results for the Hop?eld model with many

patterns. Probab. theory Relat. ?elds 110, 177�275 (1998).

65. Shcherbina, M. & Tirozzi, B. The free energy of a class of Hop?eld

models. J. Stat. Phys. 72, 113�125 (1993).

66. Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images

(University of Toronto, 2009).

Competing interests
The authors declare no competing interests.

Additional information
Supplementary information The online version contains
supplementary material available at
https://doi.org/10.1038/s41467-025-61475-w.

Correspondence and requests for materials should be addressed to
Miguel Aguilera.

67. Fontanari, J. F. & Theumann, W. On the storage of correlated pat-

terns in Hop?eld�s model. J. Phys. 51, 375�386 (1990).

68. Agliari, E., Barra, A., De Antoni, A. & Galluzzi, A. Parallel retrieval of

Peer review information Nature Communications thanks Luca Ambro-
gioni, and the other, anonymous, reviewer(s) for their contribution to the
peer review of this work. A peer review ?le is available.

correlated patterns: from Hop?eld networks to Boltzmann
machines. Neural Netw. 38, 52�63 (2013).

69. Sherrington, D. & Kirkpatrick, S. Solvable model of a spin-glass.

Phys. Rev. Lett. 35, 1792 (1975).

Acknowledgements
The authors thank Ulises Rodriguez Dominguez for valuable discus-
sions on this manuscript. M.A. is funded by a Junior Leader fellowship
from �la Caixa� Foundation (ID 100010434, code LCF/BQ/PI23/
11970024), John Templeton Foundation (grant 62828), Basque Gov-
ernment ELKARTEK funding (code KK-2023/00085) and Grant
PID2023-146869NA-I00 funded by MICIU/AEI/10.13039/
501100011033 and cofunded by the European Union, and supported
by the Basque Government through the BERC 2022-2025 program and
by the Spanish State Research Agency through BCAM Severo Ochoa
excellence accreditation CEX2021-01142-S funded by MICIU/AEI/
10.13039/501100011033. P.A.M. acknowledges support by JSPS
KAKENHI Grant Number 23K16855, 24K21518. F.R. is supported by the
UK ARIA Safeguarded AI programme and the PIBBSS Af?liatership
programme. H.S. is supported by JSPS KAKENHI Grant Number JP
20K11709, 21H05246, 24K21518, 25K03085.

Author contributions
M.A., P.A.M., F.E.R., and H.S. designed and reviewed the research and
wrote the paper. M.A. contributed the analytical and numerical results.
P.A.M. contributed part of the analytical results of the replica analysis.

Reprints and permissions information is available at
http://www.nature.com/reprints

Publisher�s note Springer Nature remains neutral with regard to jur-
isdictional claims in published maps and institutional af?liations.

Open Access This article is licensed under a Creative Commons
Attribution 4.0 International License, which permits use, sharing,
adaptation, distribution and reproduction in any medium or format, as
long as you give appropriate credit to the original author(s) and the
source, provide a link to the Creative Commons licence, and indicate if
changes were made. The images or other third party material in this
article are included in the article's Creative Commons licence, unless
indicated otherwise in a credit line to the material. If material is not
included in the article's Creative Commons licence and your intended
use is not permitted by statutory regulation or exceeds the permitted
use, you will need to obtain permission directly from the copyright
holder. To view a copy of this licence, visit http://creativecommons.org/
licenses/by/4.0/.

� The Author(s) 2025

Nature Communications |

 (2025) 16:6511

10


