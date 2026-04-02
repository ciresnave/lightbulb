Learning Stable Koopman Embeddings

Fletcher Fan, Bowen Yi, David Rye, Guodong Shi and Ian R. Manchester

1
2
0
2

t
c
O
3
1

]

G
L
.
s
c
[

1
v
9
0
5
6
0
.
0
1
1
2
:
v
i
X
r
a

Abstract� In this paper, we present a new data-driven
method for learning stable models of nonlinear systems. Our
model
lifts the original state space to a higher-dimensional
linear manifold using Koopman embeddings. Interestingly, we
prove that every discrete-time nonlinear contracting model can
be learnt in our framework. Another signi?cant merit of the
proposed approach is that it allows for unconstrained optimiza-
tion over the Koopman embedding and operator jointly while
enforcing stability of the model, via a direct parameterization
of stable linear systems, greatly simplifying the computations
involved. We validate our method on a simulated system and
analyze the advantages of our parameterization compared to
alternatives.

I. INTRODUCTION

The problem of ?tting models to data generated from
dynamical systems, known as system identi?cation, is ubiq-
uitous in science and engineering. One important considera-
tion in system identi?cation is the stability of the model. In
many applications, the ?tted model is used for prediction of
future behaviors of the system, and an unstable model would
erroneously produce unbounded predictions.

There are many different forms of stability for nonlinear
systems. In this paper, we consider contraction, also known
as incremental stability, which can be viewed as a �strong�
type of stability for dynamical systems that studies the
convergence between any two trajectories of the given system
[10]. There has been much prior work on learning contracting
models in system identi?cation for various model classes,
including polynomial models [26], [27], Gaussian mixture
models [20] and neural network models [21], [22], [13]. In
this paper, we propose a new class of contracting nonlinear
models that combines the expressiveness of neural networks
with the strong stability guarantees associated with linear
systems. Furthermore, we propose a learning framework
that ?ts this class of models to data via an unconstrained
optimization problem.

Our work bridges the gap between learning stable non-
linear models and approximating the Koopman operator [7],
an in?nite-dimensional linear operator that can describe the
dynamics of any nonlinear system by embedding it in a
higher-dimensional space. There has been growing interest
in data-driven methods that estimate ?nite-dimensional ap-
proximations of the Koopman operator and its eigenfunctions
[24], [28], [4], motivated by the appeal of being able to apply
linear systems analysis to complex nonlinear systems. Due

This work was supported by the Australian Research Council.
The authors are with the Australian Center for Field Robotics and
Sydney Institute for Robotics and Intelligent Systems, The Univer-
sity of Sydney, NSW 2006, Australia. Corresponding author�s email:
f.fan@acfr.usyd.edu.au

to this merit, Koopman-based methods have been developed
for system identi?cation [15], state observation and control
[8] of nonlinear systems.

In Koopman identi?cation approaches, a central problem is
how to learn the Koopman embedding from data. Recently
many methods [9], [11], [14], [18], [19], [25], [29] have
been proposed to address this problem, however most of
them do not consider the stability of the learned model.
Unstable learned models may have serious robustness issues,
particularly when applied to dissipative physical systems,
making them unsuitable for practical use. We aim to address
this issue by imposing stability constraints on the Koopman
model.

Our model class is motivated by recent work [30] showing
that, for continuous-time (CT) nonlinear systems, there is
an equivalence between the Koopman and contraction ap-
proaches for stability analysis under some mild technical
assumptions. We extend this equivalence result to discrete-
time (DT) systems in this paper, and provide an algorithmic
framework for learning Koopman models with contracting
properties.

The main contributions of this paper are threefold:

C1 We propose a novel Koopman learning framework
jointly models the Koopman operator and em-
that
bedding from data, while imposing the model stabil-
ity/contraction constraint.

C2 We prove that every nonlinear discrete-time contracting
model can be learnt in our framework in an arbitrarily
large compact set, which may be viewed as the exten-
sion of [30] from CT to DT; see Theorem 1.

C3 Our work builds on the contracting model class iden-
ti?ed in [22], which allows for unconstrained opti-
mization of objective functions, unlike some existing
parameterizations of Koopman operators, e.g. [12]. As
a result, it signi?cantly simpli?es the implementation of
optimization algorithms for learning the model param-
eters.

The rest of the paper is organized as follows. Section II
de?nes the system identi?cation problem and provides the
background on Koopman operator theory and data-driven
methods for estimating the Koopman operator, and also
restates the main result in [30]. Section III extends [30,
Theorem 1] to DT systems. Section IV de?nes the model set
used in our learning framework, and Section V de?nes the
optimization problem. Section VII provides some numerical
validations of our framework on a handwriting dataset.

Notation. All mappings and functions are assumed suf-
?ciently smooth. Given f : Rn ? Rm, we denote the
gradient operator ?f := (?f /?x)(cid:62). Let F be the space

of smooth real-valued scalar functions Rn ? R. We use
(�)� to denote the Moore-Penrose pseudoinverse of a matrix.
?min(�) and ?max(�) respectively represent the smallest and
largest eigenvalue of a square matrix. We use |�| to denote
vector norms, i.e. |�|2 is the vector 2-norm. Sometimes, we
may simply write x(t) as xt.

II. PROBLEM DEFINITION AND BACKGROUND

In this paper, we consider the identi?cation of a Koopman
embedding and operator for a discrete-time (DT) autonomous
state-space system:

x(t + 1) = f (x(t)),

(1)

where x ? Rn and t is the timestep. We assume the system
(1) has a single equilibrium at x(cid:63), i.e. f (x(cid:63)) = x(cid:63). Further,
we assume the dynamics f (x) are unknown, but we have
access to full-state trajectory data {�xt}T
t=0, generated by
system (1). We are concerned with learning a function ?(x)
(i.e. the Koopman embedding) that smoothly maps from the
original state space Rn to a possibly higher-dimensional
space RN (N ? n), as well as a linear matrix A ? RN �N
(i.e. a ?nite-dimensional approximation of the Koopman
operator) that describes the evolution of ?(x) over time.

The Koopman embedding and the matrix A in fact form
a predictive model of the system (1), which we de?ne as a
Koopman model.

De?nition 1 (DT Koopman model): Given a Koopman
embedding ?(x) and matrix A, the corresponding Koopman
model is:

x(t) = a(x0, t) = ?L(At?(x0)),
where ?L : RN ? Rn is a left-inverse of ?(x) such that
?L(?(x)) = x, and x0 is an initial condition.

(2)

The problem of learning ?(x) and A can be treated as a
minimization of the prediction error of the Koopman model
on the given data {�xt}T

t=0.

A. Koopman Operator Theory

Before presenting our theoretical contributions, we provide
some background on the Koopman operator. The Koopman
operator was proposed in [7] for CT dynamical models. First,
let us recall the de?nition of its variant for DT systems.

De?nition 2: (Koopman operator) For the DT dynamical
model (1), the Koopman operator K : F ? F is de?ned by

K[?(x)] := ? ? f (x)

(3)

for ? ? F, assuming that the system has a unique solution
?t ? N. We term the scalar real-valued function ? : Rn ? R
an observable.

Since the Koopman operator is de?ned on the functional
space, it is in?nite-dimensional. It is also easy to verify
that the Koopman operator is linear, i.e. K[k1?1 + k2?2] =
k1K[?1]+k2K[?2] for any k1, k2 ? R and ?1, ?2 ? F. This
property makes Koopman methods widely popular in the
analysis of dynamical models. Despite the in?nite dimension
of the Koopman operator, some key properties of a given
nonlinear dynamical model�e.g. stability and dynamical

behaviours�can be captured by a few particular functions,
i.e. the Koopman eigenfunctions.

De?nition 3: (Koopman

eigenfunction) A Koopman
eigenfunction is a non-zero observable ?? ? F/{0}
satisfying

K[??(x)] = ???(x)

(4)

for some ? ? C, which is the associated Koopman eigen-
value.

A Koopman eigenfunction de?nes a coordinate in which
the system trajectories behave as a linear system. To be
precise, de?ne a coordinate z? = ??(x), the dynamics of
which are given by

z?(t + 1) = ?z?(t),

with the initial condition z?(0) = ??(x(0)). Indeed, the
de?nition (4) is equivalent to solving the algebraic equation

??(f (x)) = ???(x),

?x ? Rn

if the DT dynamical model (1) is prior.

B. Dynamic Mode Decomposition

In the problem of system identi?cation, we are more inter-
ested in ?nding the Koopman eigenfunctions and eigenvalues
only from the collected data set {�xt}T
t=0, for which dynamic
mode decomposition (DMD) provides an ef?cient data-
driven approach to approximating the Koopman operator
[24].

In DMD, usually some heuristically predetermined, suf?-
ciently rich observables ?1, . . . ?N (N (cid:29) n)�rather than
Koopman eigenfunctions�are involved to learn the non-
linearity in the dynamical model. The task in the DMD
method is to seek a matrix A ? RN �N in order to obtain a
?nite-dimensional approximation of K, which minimizes the
following:

T
(cid:88)

j=0

|?(x(t + 1)) ? A?(x(t)))|2
2,

(5)

in which we have de?ned ? := [?1, . . . , ?N ](cid:62). The least
square problem (5) has a unique solution

A = Y1Y �
2

(6)

with

Y1 := (cid:2)?(x(1))
Y2 := (cid:2)?(x(0))

. . . ?(x(T ))(cid:3)
. . . ?(x(T ? 1))(cid:3)

if Y2 is full row rank. DMD is a simple, ef?cient method to
approximate the Koopman operator, but two issues arise:

1) In the DMD method, the observables ? are predeter-
mined, which signi?cantly affects the learning accuracy,
but in the literature the selection of observables usually
done in a heuristic manner. Since these observables are
closely connected to the Koopman eigenfunctions for a
given dynamical model, a natural question is: can the
observables and the matrix A be learnt concurrently to
improve accuracy?

2) For a stable dynamical model, the above least square
solution may yield an unstable model due to various
kinds of perturbations in the data set {�xt}T
t=0, which
would be unacceptable in many applications. Hence,
imposing stability constraints is an important consid-
eration in learning algorithms.

The main motivation of the paper is to address the above

issues and present a novel Koopman learning framework.

C. Contraction analysis

In this paper, we are interested in stable nonlinear mod-
els. Indeed, there are many different forms of stability for
nonlinear systems; we focus on contracting systems [10].

Contraction analysis provides another way to study non-
linear systems by means of linear systems theory exactly and
globally. In contraction analysis we are concerned with the
differential dynamics of a given system, which is indeed a
linear time-varying (LTV) system. The differential dynamics
of the model (1) are given by

(7)

(x(t))?x(t),

?x(t + 1) =

?f
?x
with ?x ? Rn representing the in?nitesimal displacement.
Informally, if the LTV system (7) is exponentially stable
along any feasible trajectories x(t), we can say the system
(1) is contracting. Its formal de?nition is given as follows.
De?nition 4: Given the DT system (1), if there exists a
uniformly bounded metric M (x), i.e. a1In (cid:22) M (x) (cid:22) a2In
for some a2 ? a1 > 0, guaranteeing
?f
?x

(x(t))?M (x(t)) (cid:22) ??M (x(t)),
(8)
with 0 < ? < 1, then we say that the given system is
contracting.

(x(t))(cid:62)M (x(t+1))

?f
?x

A central result of contraction analysis is that, for con-
tracting systems, all trajectories converge exponentially to a
single trajectory, i.e., for any two trajectories xa and xb

|xa(t) ? xb(t)| ? a0?t|xa(0) ? xb(0)|

for some a0 > 0.

In this paper, we propose an algorithm to learn dynamical
models which are contracting in the sense of De?nition 4.
Generally speaking, verifying or guaranteeing contraction for
a nonlinear model is non-trivial. Motivated by the Koopman
approach, we instead consider a transformation of the state
space into a higher-dimensional manifold on which the
dynamics are linear. A natural question that arises is the
conservativeness of such an approach for verifying stability.
However, it was recently shown in [30] that the Koopman
and contraction approaches are equivalent to each other for
stability analysis when considering CT dynamical models.

III. STABILITY CRITERION FOR DISCRETE-TIME
KOOPMAN MODELS

the model set proposed here can provide suf?cient degrees
of freedom for learning nonlinear DT models.

Theorem 1: Consider the system (1). Suppose that there

exists a mapping ? : Rn ? RN with N ? n such that
D1 There exists Schur stable matrix A ? RN �N satisfying

?(f (x)) = A?(x).

(9)

D2 ?(x) := ??(x)(cid:62) has full column rank, and ?(x)(cid:62)?(x)

is uniformly bounded.

Then system (1) is contracting with the contraction metric
?(x)(cid:62)P ?(x), where P is any positive-de?nite matrix sat-
isfying P ? A(cid:62)P A (cid:31) 0. Conversely, if the system (1) is
contracting with the metric M (x) ? Rn�n
(cid:31)0 , and assuming
that f is invertible and its inverse f ?1 is continuous. Then, in
any invariant compact set X ? Rn, there exists a continuous
Koopman mapping ? : Rn ? Rn verifying D1 and D2.

Proof: (?) From D2 there exists a matrix P = P (cid:62) (cid:31) 0

satisfying the Lyapunov condition

P ? A(cid:62)P A (cid:31) Q,

(10)

for some constant positive de?nite matrix Q (cid:31) 0 without
loss of generality. We de?ne a new coordinate z := ?(x) in
which the in?nitesimal displacement ?zt ? RN at time t is
given by

?zt = ?(xt)?xt,

(11)

where ?xt
coordinate.

is an in?nitesimal displacement

in the x-

The DT differential dynamics of x can be written as

?xt+1 = F (xt)?xt,

(12)

where F (x) := ?f (x)(cid:62).

Similarly, for z we have

?zt+1 = A?(xt)?xt = ?(xt+1)F (xt)?xt,

(13)

where we have used the relations zt+1 = Azt and zt+1 =
?(f (xt)) in their differential forms. Hence, we obtain

?(xt+1)F (xt) = A?(xt).

Due to the full column rank of ?(x) and (10), it follows that

?(xt)(cid:62)(P ? A(cid:62)P A)?(xt) (cid:31) ?(xt)(cid:62)Q?(xt).

(14)

Then, by substituting (13), we have

?(xt)(cid:62)P ?(xt) ? F (xt)(cid:62)?(xt+1)(cid:62)P ?(xt+1)F (xt)

(13)
(cid:31) ?(cid:62)Q? (cid:23)

?min(Q)
?max(P )

?(cid:62)P ?.

(15)
Now since ? has full column rank and P (cid:31) 0, we have
M (x) := ?(cid:62)P ? (cid:31) 0. Substituting into (15):

M (xt) ? F (xt)(cid:62)M (xt+1)F (xt) (cid:31) ?M (xt),

(16)

In this section, we extend the main result in [30] to DT
systems, i.e. the Koopman and contraction approaches are
equivalent for nonlinear stability analysis. We will show that

with ? := ?min(Q)/?max(P ). By selecting Q = ?P with
? ? (0, 1), we have ? ? (0, 1). This is exactly the contraction
condition for the system (1) with respect to the metric M .

(?) For the given DT system, from directly applying the
Banach ?xed-point theorem we conclude that there exists a
unique ?xed-point x(cid:63) ? X , i.e. f (x(cid:63)) = x(cid:63).

First, we parameterise the unknown mapping ?(x) as
?(x) := x + T (x), with a new mapping T (x) to be searched
for. Then, the algebraic equation (9) becomes T (f (x)) +
f (x) = Ax + AT (x). By ?xing A = ?f (x(cid:63))(cid:62), from the
contraction assumption, we have M (x(cid:63)) ? A(cid:62)M (x(cid:63))A (cid:23)
?M (x(cid:63)), thus A being Schur stable. It yields

T (f (x)) = AT (x) + H(x),

(17)

in which we have de?ned H(x) := Ax ? f (x). We make
the key observation that the algebraic equation (17) exactly
coincides with the one in the formulation of the Kazantzis-
Kravaris-Luenberger observer for nonlinear DT systems [2,
Eq. (7)]. In our case, the function H(x) is continuous and,
following [2, Theorem 2], we have a feasible solution to (17)
as follows: 1

T (x) =

+?
(cid:88)

j=0

AiH(X(x, ?j + 1)),

(18)

with the de?nition

X(x, j) = f ? f ? � � � ? f
(cid:125)
(cid:124)

(cid:123)(cid:122)
j times

(x), X(x, ?j) = (f ?1)j(x)

for i ? N+.

Although ?0(x) := x + T (x) with T de?ned above
satis?es D1 in the entire set X , the condition D2 may be
not true. Hence, we need to modify the obtained ?0(x). By
considering the evolution of the trajectories in the x- and
z := ?(x)-coordinates respectively, we have

z(tx) = ?0(x(tx)) = ?0(X(x, tx)) = Atx?0(x),
with tx ? N+, thus satisfying ?0(x) = A?tx ?0(X(x, tx)).
Then, we modify ?0(x) into

?(x) := A?tx [X(x, tx) + T (X(x, tx))]

(19)

with a suf?ciently large tx ? N+.

Finally,

let us check conditions D1 and D2. For the

algebraic condition, we have

?(f (x)) = A?tx ?0(X(f (x), tx))
= A?tx ?0(f (X(x, tx)))
= A?tx � A?0(X(x, tx))
= A?(x)

where we have used the fact

X(f (x), tx) = f ? f ? � � � ? f
(cid:125)
(cid:124)

(cid:123)(cid:122)
(j+1) times

= f (X(x, tx))

1The second assumption in [2] holds true in any backward invariant
compact set. Since contracting systems generally cannot guarantee such
invariance, we may modify the dynamics as xt+1 = ?(xt)f (xt) with

?(x) =

(cid:40)

1,

0,

if x ? cl(X )
if x /? X (cid:48)

with X ? X (cid:48), and then continue the analysis.

in the second equation. Therefore, ?(x) de?ned in (19)
satis?es the algebraic equation (9). Regarding D2, let us
study the Jacobian of ?(x) in (19), which is given by

??
?x

(x) = A?tx

(cid:20)

I +

?T
?x

(X(x, tx))

(cid:21) ?X
?x

(x).

On the other hand, we have that ?xX is full rank and

H(x(cid:63)) = 0,

?H
?x

(x(cid:63)) = 0,

as a result ?T (x(cid:63)) = 0. If tx ? N+ is suf?ciently large, the
largest singular value of ?T (X(x, tx)) would be very small,
and then the identity part of ?(x) will dominate ??(x).
Hence, ?(x) is an injection for a large tx ? N+.

IV. MODEL SET

In this section, we de?ne the model set that we optimize
over in our learning framework. We parameterize both the
Koopman observables ?(x) and the matrix A in our model
and train them jointly. To the best of our knowledge, the joint
learning of ? and A with the model stability constraint�as
done in the paper�has not been previously considered in
the literature.

Recall our de?nition of the Koopman model (2). By
Theorem 1, the model (2) is guaranteed to be contracting
if Conditions D1 and D2 are satis?ed. In the following, we
propose parameterizations of ?(x) and A that satisfy these
conditions.

A. Parameterization of observables

We propose to parameterize the observables as:

?(x) = Cx + ?(x, ?N N ),

(20)

where C = [In, 0n�(N ?n)](cid:62). The nonlinear part ?(x) can
be any differentiable function approximator, parameterized
by ?N N . For brevity, we drop the dependence on ?N N in our
notation. In this paper, we consider ?(x) as a feedforward
neural network due to its scalability, but any differentiable
function approximator can be used.

The dimensionality of the observables N is a hyperpa-
rameter chosen by the user. For N = n, the observables will
be of the same form as the constructive mapping ?0(x) =
x + T (x) in Theorem 1.

In order to reconstruct

the original state x from the
observables, we need to train a separate function ?L(z) to
compute the left-inverse of ?(x). Indeed, the left invertibility
of ? is necessary for condition D2. We propose to simply
parameterize this left
inverse function as another neural
network ?L(z, ?L).

Remark 1: There are many possible parameterizations of
the observables that are compatible with our framework,
with Equation (20) being just the one chosen to mimic the
constructive mapping from Theorem 1. For some parameter-
izations, the left inverse may be computed analytically and
does not have to be modelled as a separate function. For
example, if ?(x) = [x(cid:62), ?(x)(cid:62)](cid:62), then the left inverse is
simply x = C?(x), where C = [I, 0].

B. Parameterization of the Koopman operator

embedding space:

The Koopman matrix A has to satisfy Condition D1 of
Theorem 1, i.e. it must be Schur stable. There are many
equivalent conditions for enforcing stability of linear sys-
tems, including the well-known Lyapunov inequality P ?
A(cid:62)P A (cid:31) 0 for some P (cid:31) 0, and the recently proposed
parameterization in [3], which was used to train stable
Koopman operators for ?xed observables in [12]. However,
solving optimization problems with these constraints in
an ef?cient manner is non-trivial, especially when jointly
searching for the observables.

In the following, we present an unconstrained param-
eterization of A, which is a special case of the direct
parameterization approach proposed in [22].

Proposition 1: Consider the parametric matrix A(L, R)

de?ned as:

A(L, R) = 2(M11 + M22 + R ? R(cid:62))?1M21,

(21)

where

M :=

(cid:20)M11 M12
M21 M22

(cid:21)

= LL(cid:62) + (cid:15)I,

(22)

with (cid:15) a small positive constant. Then for any real-valued
L ? R2N �2N and R ? RN �N , A0 = A(L, R) is a necessary
and suf?cient condition for A0 to be Schur stable.

Proof: Let E = (M11 + M22 + R ? R(cid:62))/2, F = M21

and P = M22. Then we have A(L, R) = E?1F and

M =

(cid:20)E + E(cid:62) ? P F (cid:62)
P

F

(cid:21)

.

(23)

It has been shown that M (cid:31) 0 is necessary and suf?cient for
E?1F to be Schur stable [26]. Since our parameterization
M = LL(cid:62) + (cid:15)I is positive de?nite by construction, this
proves suf?ciency for A(L, R) to be Schur stable. Addition-
ally, all M can be constructed from L, e.g. via Cholesky
factorization, and by extension, all E, F and P can be
constructed from L and R. This completes the proof.

C. Overall Koopman Model

It may be helpful to think of our model as a linear system

with an output �x that is an estimate of the original state:

z(t) = Az(t ? 1),
�x(t) = ?L(z(t)),

(24)

where z(0) = ?(x0). This system is equivalent to (2). In
this form, it is clear that as long as A is stable and ?L is
uniformly bounded, then the output �x will always converge
to a single equilibrium.

Jse :=

1
T

T
(cid:88)

|�zt ? zt|2
2,

(26)

t=0
where �zt = ?(�xt), and zt = At?(�x0). While we could also
minimize the simulation error in x, in practice we found this
produced poor results.

The complete optimization problem is:

min
???

1
T

T
(cid:88)

t=0

(cid:12)?(�xt) ? A(L, R)t?(�x0)(cid:12)
(cid:12)
2
2 + ?Jrec.
(cid:12)

(27)

The reconstruction loss Jrec is de?ned as

Jrec =

1
T

T
(cid:88)

t=0

(cid:12)�xt ? ?L(?(�xt))(cid:12)
(cid:12)
(cid:12)

2
2.

(28)

Minimizing Jrec gives us an approximate left-inverse ?L for
the Koopman mapping. The loss Jrec can be thought of as
a penalty term that relaxes the constraint

x = ?L(?(x)) ?x,

and the constant ? is a hyperparameter that determines the
weighting of the penalty.

We emphasize two important properties of Problem (27).
First, it is an unconstrained optimization problem. The pa-
rameter set ? is the space of real numbers of the appropriate
dimensionality. Second, there exists a differentiable mapping
from the parameters ? to the objective for any choice of
differentiable mapping ??, e.g. using our parameterization
(20) with ?? as a neural network.

These two properties enable us to ?nd a local optimum
to Problem (27) using any off-the-shelf ?rst-order optimizer
in conjunction with an automatic differentiation (autodiff)
toolbox. This signi?cantly simpli?es the implementation of
our framework. Using an autodiff software package, one only
needs to write code that evaluates the objective function at
each iteration of the optimization process, and the gradients
w.r.t. ? are automatically computed via the chain rule. In
contrast, constrained problems such as the one proposed in
[12] require specialized algorithms to solve. Although the
objective (27) is nonconvex, deep learning methods have
been shown to be effective at ?nding approximate global
minima for such problems; see [23, Chapter 21] for example.
is worth noting that our model class is
agnostic to the optimization problem. In fact, the model can
be optimized for any differentiable objective function. This
is another advantage of an unconstrained parameterization.

Remark 2: It

To summarize, our model parameters consist of:

B. Implementation Details

? = {?N N , ?L, L, R}.

(25)

V. LEARNING FRAMEWORK

A. Optimization Problem

To ?t the model parameters (25) to data, we consider
the problem of minimizing the simulation error in the

We implemented our learning framework in PyTorch2
and used the Adam optimizer [6] to solve Problem (27).
The neural network parameters ?N N and ?L are initialized
using the default scheme in PyTorch, while L, R, and b are
initialized randomly from a uniform distribution.

2https://github.com/pytorch/pytorch

1) Fast matrix power computation: As explained in Sec-
tion V-A, the only code we need to implement for solving
Problem (27) is the evaluation of the objective function,
which is also the main computational bottleneck. In par-
ticular, repeatedly computing the matrix power At for the
same A and many t�s can be computationally inef?cient.
Here we describe a simple trick to speed up matrix power
computations. Consider the eigendecomposition of A given
by V ?V ?1, where the columns of V are the eigenvectors
and ? is a diagonal matrix of the eigenvalues. Then it is
clear that

At = (V ?V ?1)t = V ?tV ?1

(29)

for integer t. Notice that ?t can be computed element-
wisely for each eigenvalue on the diagonal, which offers a
signi?cant speed-up over computing a matrix power. This
trick assumes A is diagonalizable, but this can easily be
veri?ed in code and, if the condition is not satis?ed, the
original matrix power computation can be performed instead.

VI. CONTINUOUS-TIME CASE

In this section, we brie?y present the CT formulation of
our learning framework. For an autonomous system governed
by an ordinary differential equation (ODE) ?x = f (x), there
exists a semigroup of Koopman operator Kt associated with
the ?ow map X(x, t) of the system, de?ned as:

Kt?(x(t)) = ?(X(x, t)).

(30)

We refer to the in?nitesimal generator of this semigroup as
the continuous-time Koopman operator �K [28]:
d
dt

?(x(t)) = ?? � f (x(t)).

�K?(x(t)) =

(31)

De?nition 5 (CT Koopman model): The continuous-time

Koopman model is given by:

x(t) = ?L(exp(A?t)?(x0)),

(32)

where the Koopman mapping ? is parameterized as in (20),
and the ?nite-dimensional matrix A? is parameterized as:

A? = (N N (cid:62) + (cid:15)I)?1(?QQ(cid:62) ? (cid:15)I +

1
2

(R ? R(cid:62))),

(33)

with parameters N , Q and R. This is an unconstrained
parameterization of all CT stable (Hurwitz) matrices.

Given full-state trajectory data {�xk}K

k=0 with correspond-
ing time {tk}K
k=0, we would like to minimize the simulation
error of the Koopman model. The optimization problem is

min
???

1
K

K
(cid:88)

k=0

|??(�xk) ? exp(A?tk)??(�x0)|2

2 + ?Jrec,

(34)

where Jrec is as de?ned in Equation (28). Problem (34) is
an unconstrained optimization problem just like the discrete-
time problem, hence a local minimum can be obtained using
a ?rst-order optimizer and an autodiff software package. A
trick similar to that described in Section V-B.1 can be used
to compute the matrix exponential in the objective in (34).
Note that in terms of the data required, the only difference
between the DT and CT learning frameworks is that the

TABLE I: Comparison of model sets for our method and
prior works.

Method

SOC [12]
LKIS [25]
[11]
[19]
Ours

Learns
observables or
eigenfunctions
Neither
Observables
Eigenfunctions
Eigenfunctions
Observables

Continuous or
discrete time

Stability
constraint

Discrete
Discrete
Discrete
Continuous
Both

(cid:51)
(cid:55)
(cid:55)
(cid:51)
(cid:51)

CT case requires the time corresponding to each data point.
The CT problem can be useful to consider when the data is
sampled at non-uniform time intervals, or when the sampling
rate differs between the training and test scenarios.

VII. NUMERICAL EXAMPLES

We validated our framework on the LASA handwriting
dataset [5], which consists of human-drawn trajectories of
various letters and shapes3. It has been widely used as a
benchmark for learning contracting dynamics in continuous-
time [1], [5], [16], [17], [20]. In our results, we trained
discrete-time models in order to compare them with existing
DT Koopman learning frameworks. Contraction is an impor-
tant constraint for this data set as unconstrained models can
have spurious attractors [5], leading to poor generalization
to unseen initial conditions.

For each shape in the dataset, we attempted to train a
discrete-time model that would regulate to the desired equi-
librium point from any initial condition. To prepare the data
for learning DT models, we ?tted splines to the trajectories
and re-sampled the datapoints at a uniform time interval. The
t ](cid:62) ? R4, where
state vector was chosen to be �xt = [y(cid:62)
yt and ?yt are the position and velocity vectors at time t.
All data was scaled to the range [?1, 1] before training. For
each shape in the dataset, we performed leave-one-out cross
validation. Test trajectories are plotted in Figure 3 as solid
black lines for a subset of the shapes in the dataset.

t , ?y(cid:62)

The metric we used to compare different methods was

normalized simulation error (NSE), de?ned as:

(cid:80)T

,

2

2

(35)

N SE =

t=0|�xt ? �xt|2
(cid:80)T
t=0|�xt|2
where {�x}T
t=0 is the simulated trajectory using the learned
model, and {�x}T
t=0 is the true trajectory. The aim of our
comparisons was to evaluate our framework against prior
methods for learning Koopman models. The key differences
of some recent frameworks are summarized in Table I.
We did not compare against [11] as they assume some
prior knowledge about the spectrum of the system, which
differs from the problem setting we consider. Due to space
constraints, we leave comparisons of CT frameworks to
future work.

In the following, we refer to our framework as SKEL

(Stable Koopman Embedding Learning).

3https://cs.stanford.edu/people/khansari/download.html

A. Comparison with other Koopman matrix parameteriza-
tions

We compared SKEL against two recently-proposed Koop-
man learning frameworks, namely SOC [12] and LKIS
[25]. In particular, we compared our unconstrained stable
parameterization of the Koopman operator against a con-
strained stable parameterization (SOC), and a unconstrained
parameterization without stability guarantees (LKIS).

The SOC parameterization is given by A = S?1OCS,
where O is orthogonal and C is positive-semide?nite with
(cid:107)C(cid:107) ? 1. A projected gradient descent method was used to
solve the optimization problem. The LKIS parameterization
is A = Y1Y �
2 , where Y1 and Y2 are as de?ned in Equation
(6), with parametric ?(x).

To make it a fair comparison, we kept all other aspects
of the optimization problem the same, i.e. using simulation
error as the optimization objective and using parametric
observables of the form (20). We were interested mainly in
comparing parameterizations of the Koopman operator as our
framework is agnostic to choice of objective and observables,
and these choices often depend on the particular application.
All instances of ?(x) were fully-connected feedforward
neural networks with ReLU (recti?ed linear units) activation
functions, 2 hidden layers with 50 nodes each and an output
dimensionality of 20. Hyperparameter values were chosen to
be ? = 103 and (cid:15) = 10?8.

A boxplot of the normalized simulation error for the three
methods is shown in Figure 1. It is clear that SKEL achieves
the lowest median NSE on the test set with 95% con?dence.
From Figure 2, it can be seen that LKIS actually attains the
lowest training error, but does not generalize to the test set as
well as SKEL. This can be seen as a symptom of over?tting,
and shows that the stability guarantees of SKEL have a
regularizing effect on the model. With regards to SOC, we
observed that the constrained optimization problem would
often converge to poor local minima, which is re?ected in
the relatively high training and test errors.

B. Robustness to perturbations in initial condition

We performed a qualitative evaluation of the robustness
of the models to small perturbations in the initial condition
of the test trajectory. We compared only SKEL and LKIS
as it was clear from Figure 1 that SOC underperformed in
this setting. The results are plotted in Figure 3. It can be
seen that the SKEL models produce trajectories that converge
to each other due to their contracting property, whereas the
LKIS models behave unpredictably, indicating instability of
the learned model.

VIII. CONCLUSION

We have presented a novel Koopman learning framework
that jointly models the Koopman operator and observables
while guaranteeing model stability, via an unconstrained
optimization problem. We showed that our framework out-
performs existing Koopman methods on a real-world hand-
writing problem and achieves the lowest median simulation

Fig. 1: Comparison of SKEL with other Koopman learning
methods. Outliers were clipped for better visibility of boxes.
Number of outliers with NSE > 1 from left to right: 1
(SKEL), 15 (LKIS), 0 (SOC).

Fig. 2: Training loss (Eq. (27)) for each method

error. Further work can be done to extend this framework to
controlled systems.

REFERENCES

[1] C. Blocher, M. Saveriano, and D. Lee. Learning stable dynamical
In 14th Int. Conf. on Ubiquitous

systems using contraction theory.
Robots and Ambient Intell. (URAI), pages 124�129, 2017.

[2] L. Brivadis, V. Andrieu, and U. Serres. Luenberger observers for
discrete-time nonlinear systems. In the 58th IEEE Conf. on Decision
and Control (CDC), pages 3435�3440. IEEE, 2019.

[3] N. Gillis, M. Karow, and P. Sharma. A note on approximating the
nearest stable discrete-time descriptor systems with ?xed rank. Appl.
Numerical Math., 148:131�139, 2020.

[4] M. Haseli and J. Cortes. Learning koopman eigenfunctions and
invariant subspaces from data: Symmetric subspace decomposition.
IEEE Trans. on Automatic Control, pages 1�1, 2021.

[5] S. M. Khansari-Zadeh and A. Billard. Learning stable nonlinear
IEEE Trans. on

dynamical systems with gaussian mixture models.
Robotics, 27(5):943�957, 2011.

[6] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization.

arXiv preprint arXiv:1412.6980, 2014.

[7] B. O. Koopman. Hamiltonian systems and transformation in Hilbert
space. Proc. of the National Academy of Sciences, 17(5):315�318,
May 1931.

SKELLKIS [24]SOC [12]Test Normalized Simulation Error00.10.20.30.40.50.60.70.80.91SKELLKIS [24]SOC [12]Training Loss�10-300.511.522.533.544.5(a) SKEL (ours)

(b) LKIS [25]

Fig. 3: Simulations of SKEL and LKIS models on test data. Trajectories from the models are shown as red dotted lines,
while the true trajectory is shown as a solid black line. Initial conditions were sampled from a square region of width 2mm
centered at the start point of the true trajectory. The target point is marked by a black star.

[8] M. Korda and I. Mezi�c. Linear predictors for nonlinear dynamical sys-
tems: Koopman operator meets model predictive control. Automatica,
93:149 � 160, 2018.

[9] Q. Li, F. Dietrich, E. M. Bollt, and I. G. Kevrekidis. Extended
dynamic mode decomposition with dictionary learning: A data-driven
adaptive spectral decomposition of the koopman operator. Chaos: An
Interdisciplinary J. of Nonlinear Science, 27(10):103111, 2017.
[10] W. Lohmiller and J.-J. E. Slotine. On contraction analysis for non-

linear systems. Automatica, 34(6):683�696, 1998.

[11] B. Lusch, J. N. Kutz, and S. L. Brunton. Deep learning for universal
linear embeddings of nonlinear dynamics. Nature Comm., 9(1):4950,
Nov. 2018.

[12] G. Mamakoukas, O. Xherija, and T. Murphey. Memory-ef?cient
learning of stable linear dynamical systems for prediction and control.
In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin,
editors, Advances in Neural Information Processing Syst., volume 33,
pages 13527�13538. Curran Associates, Inc., 2020.

[13] I. R. Manchester, M. Revay, and R. Wang. Contraction-based methods
for stable identi?cation and robust machine learning: a tutorial. In the
60th IEEE Conf. on Decision and Control (CDC), 2021.

[14] A. Mardt, L. Pasquali, H. Wu, and F. No�e. Vampnets for deep learning

of molecular kinetics. Nature Comm., 9(1):5, Jan. 2018.

[15] A. Mauroy and J. Goncalves. Koopman-based lifting techniques for
nonlinear systems identi?cation. IEEE Trans. on Automatic Control,
65(6):2550�2565, 2020.

[16] S. Mohammad Khansari-Zadeh and A. Billard. Learning control
lyapunov function to ensure stability of dynamical system-based robot
reaching motions. Robotics and Autonomous Syst., 62(6):752�765,
2014.

[17] K. Neumann, A. Lemme, and J. J. Steil. Neural learning of stable
dynamical systems based on data-driven lyapunov candidates.
In
IEEE/RSJ Int. Conf. on Intell. Robots and Syst., pages 1216�1222,
2013.

[18] S. E. Otto and C. W. Rowley.

Linearly recurrent autoencoder
networks for learning dynamics. SIAM J. on Applied Dynamical Syst.,
18(1):558�593, 2019.

[19] S. Pan and K. Duraisamy. Physics-informed probabilistic learning of
linear embeddings of nonlinear dynamics with guaranteed stability.
SIAM J. on Appl. Dynamical Syst., 19(1):480�509, 2020.

[20] H. Ravichandar, I. Salehi, and A. Dani. Learning partially contracting

dynamical systems from demonstrations. In S. Levine, V. Vanhoucke,
and K. Goldberg, editors, Proc. of the 1st Annual Conf. on Robot
Learning, volume 78 of Proceedings of Machine Learning Research,
pages 369�378. PMLR, 13�15 Nov 2017.

[21] M. Revay, R. Wang, and I. R. Manchester. A convex parameteriza-
IEEE Control Syst. Lett.,

tion of robust recurrent neural networks.
5(4):1363�1368, 2021.

[22] M. Revay, R. Wang, and I. R. Manchester. Recurrent equilibrium
networks: Unconstrained learning of stable and robust dynamical
models. 2021. arXiv:2104.05942.

[23] T. Roughgarden. Beyond the Worst-case Analysis of Algorithms.

Cambridge University Press, 2020.

[24] P. J. Schmid. Dynamic mode decomposition of numerical and

experimental data. J. of Fluid Mechanics, 656:5�28, 2010.

[25] N. Takeishi, Y. Kawahara, and T. Yairi. Learning koopman invariant
subspaces for dynamic mode decomposition.
In I. Guyon, U. V.
Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and
R. Garnett, editors, Advances in Neural Information Processing Sys-
tems 30, pages 1130�1140. Curran Associates, Inc., 2017.

[26] M. M. Tobenkin, I. R. Manchester, and A. Megretski. Convex parame-
terizations and ?delity bounds for nonlinear identi?cation and reduced-
order modelling. IEEE Trans. on Automatic Control, 62(7):3679�3686,
2017.

[27] J. Umenberger and I. R. Manchester. Convex bounds for equation error
in stable nonlinear identi?cation. IEEE Control Syst. Lett., 3(1):73�78,
2019.

[28] M. O. Williams, I. G. Kevrekidis, and C. W. Rowley. A data�driven
approximation of the koopman operator: Extending dynamic mode
decomposition. J. of Nonlinear Science, 25(6):1307�1346, Dec 2015.
[29] E. Yeung, S. Kundu, and N. Hodas. Learning deep neural network rep-
resentations for koopman operators of nonlinear dynamical systems.
In American Control Conf. (ACC), pages 4832�4839, 2019.

[30] B. Yi and I. R. Manchester. On the equivalence of contraction
and Koopman approaches for nonlinear stability and control.
In
the 60th IEEE Conf. on Decision and Control (CDC), 2021.
(See
ArXiv:2103.15033 for an extended version).

?40?200?10010203040?40?200?10010203040?20020?30?20?1001020?100102030?20?100102030?30?20?100?1001020304050?40?200?50?40?30?20?100?20020?20?10010203002040010203040500204001020304050x (mm)y (mm)?40?200?10010203040?40?200?10010203040?20020?30?20?1001020?100102030?20?100102030?30?20?100?1001020304050?40?200?50?40?30?20?100?20020?20?10010203002040010203040500204001020304050x (mm)y (mm)
