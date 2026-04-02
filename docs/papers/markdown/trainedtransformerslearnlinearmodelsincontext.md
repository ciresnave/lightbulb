3
2
0
2

t
c
O
9
1

]
L
M

.
t
a
t
s
[

3
v
7
2
9
9
0
.
6
0
3
2
:
v
i
X
r
a

Trained Transformers Learn Linear Models In-Context

Ruiqi Zhang
UC Berkeley
rqzhang@berkeley.edu

Spencer Frei
UC Berkeley
frei@berkeley.edu

Peter L. Bartlett
UC Berkeley and Google DeepMind
peter@berkeley.edu

October 23, 2023

Abstract

Attention-based neural networks such as transformers have demonstrated a remarkable ability to ex-
hibit in-context learning (ICL): Given a short prompt sequence of tokens from an unseen task, they can
formulate relevant per-token and next-token predictions without any parameter updates. By embedding a
sequence of labeled training data and unlabeled test data as a prompt, this allows for transformers to be-
have like supervised learning algorithms. Indeed, recent work has shown that when training transformer
architectures over random instances of linear regression problems, these models� predictions mimic those
of ordinary least squares.

Towards understanding the mechanisms underlying this phenomenon, we investigate the dynamics of
ICL in transformers with a single linear self-attention layer trained by gradient flow on linear regression
tasks. We show that despite non-convexity, gradient flow with a suitable random initialization finds a
global minimum of the objective function. At this global minimum, when given a test prompt of labeled
examples from a new prediction task, the transformer achieves prediction error competitive with the
best linear predictor over the test prompt distribution. We additionally characterize the robustness of
the trained transformer to a variety of distribution shifts and show that although a number of shifts are
tolerated, shifts in the covariate distribution of the prompts are not. Motivated by this, we consider
a generalized ICL setting where the covariate distributions can vary across prompts. We show that
although gradient flow succeeds at finding a global minimum in this setting, the trained transformer is
still brittle under mild covariate shifts. We complement this finding with experiments on large, nonlinear
transformer architectures which we show are more robust under covariate shifts.

1 Introduction

Transformer-based neural networks have quickly become the default machine learning model for problems
in natural language processing, forming the basis of chatbots like ChatGPT [Ope23], and are increasingly
popular in computer vision [Dos+21]. These models can take as input sequences of tokens and return
relevant next-token predictions. When trained on sufficiently large and diverse datasets, these models are
often able to perform in-context learning (ICL): when given a short sequence of input-output pairs (called a
prompt) from a particular task as input, the model can formulate predictions on test examples without having
to make any updates to the parameters in the model.

1

Recently, Garg et al. [Gar+22] initiated the investigation of ICL from the perspective of learning partic-
ular function classes. At a high-level, this refers to when the model has access to instances of prompts of the
form (x1, h(x1), . . . , xN , h(xN ), xquery) where xi, xquery are sampled i.i.d. from a distribution Dx and h is
sampled independently from a distribution over functions in a function class H. The transformer succeeds
1), . . . , x?
at in-context learning if when given a new prompt (x?
query) corresponding to an
independently sampled h? it is able to formulate a prediction for x?
query) given a
sufficiently large number of examples N . The authors showed that when transformer models are trained
on prompts corresponding to instances of training data from a particular function class (e.g., linear models,
neural networks, or decision trees), they succeed at in-context learning, and moreover the behavior of the
trained transformers can mimic those of familiar learning algorithms like ordinary least squares.

N , h?(x?
query that is close to h?(x?

1, h?(x?

N ), x?

Following this, a number of follow-up works provided constructions of transformer-based neural net-
work architectures which are capable of achieving small prediction error for query examples when the
prompt takes the form (x1, ?w, x1?, . . . , xN , ?w, xN ?, xquery) where xi, xquery, w i.i.d.? N(0, Id) [Osw+22;
Aky+22]. However, this leaves open the question of how it is that gradient-based optimization algorithms
over transformer architectures produce models which are capable of in-context learning.1

In this work, we investigate the learning dynamics of gradient flow in a simplified transformer archi-
tecture when the training prompts consists of random instances of linear regression datasets. Our main
contributions are as follows.

� We establish that for a class of transformers with a single layer and with a linear self-attention module
(LSAs), gradient flow on the population loss with a suitable random initialization converges to a global
minimum of the population objective, despite the non-convexity of the underlying objective function.

� We characterize the learning algorithm that is encoded by the transformer at convergence, as well
as the prediction error achieved when the model is given a test prompt corresponding to a new (and
possibly nonlinear) prediction task.

� We use this to conclude that transformers trained by gradient flow indeed in-context learn the class
of linear models. Moreover, we characterize the robustness of the trained transformer to a variety of
distribution shifts. We show that although a number of shifts can be tolerated, shifts in the covariate
distribution of the features xi can not.

� Motivated by this failure under covariate shift, we consider a generalized setting of in-context learning
where the covariate distribution can vary across prompts. We provide global convergence guarantees
for LSAs trained by gradient flow in this setting and show that even when trained on a variety of
covariate distributions, LSAs still fail under covariate shift.

� We then empirically investigate the behavior of large, nonlinear transformers when trained on linear
regression prompts. We find that these more complex models are able to generalize better under
covariate shift, especially when trained on prompts with varying covariate distributions.

2 Additional Related Work

The literature on transformers and non-convex optimization in machine learning is vast. In this section, we
will focus on those works most closely related to theoretical understanding of in-context learning of function

1We note a concurrent work also explores the optimization question we consider here [Ahn+23]; we shall provide a more

detailed comparison to this work in Section 2.

2

classes.

As mentioned previously, Garg et al. [Gar+22] empirically investigated the ability for transformer archi-
tectures to in-context learn a variety of function classes. They showed that when trained on random instances
of linear regression, the models� predictions are very similar to those of ordinary least squares. Additionally,
they showed that transformers can in-context learn two-layer ReLU networks and decision trees, showing
that by training on differently-structured data, the transformers learn to implement distinct learning algo-
rithms. A number of works further investigated the types of algorithms implemented by transformers trained
on in-context examples of linear models [APG23; AL23].

Aky�urek et al. [Aky+22] and Oswald et al. [Osw+22] examined the behavior of transformers when
trained on random instances of linear regression, as we do in this work. They considered the setting of
isotropic Gaussian data with isotropic Gaussian weight vectors, and showed that the trained transformer�s
predictions mimic those of a single step of gradient descent. They also provided a construction of trans-
formers which implement this single step of gradient descent. By contrast, we explicitly show that gradient
flow provably converges to transformers which learn linear models in-context. Moreover, our analysis holds
when the covariates are anisotropic Gaussians, for which a single step of vanilla gradient descent is unable
to achieve small prediction error.2

Let us briefly mention a number of other works on understanding in-context learning in transform-
ers and other sequence-based models. Han et al. [Han+23] suggests that Bayesian inference on prompts
can be asymptotically interpreted as kernel regression. Dai et al. [Dai+22] interprets ICL as implicit fine-
tuning, viewing large language models as meta-optimizers performing gradient-based optimization. Xie et
al. [Xie+21] regards ICL as implicit Bayesian inference, with transformers learning a shared latent concept
between prompts and test data, and they prove the ICL property when the training distribution is a mixture
of HMMs. Similarly, Wang, Zhu, and Wang [WZW23] perceives ICL as a Bayesian selection process, im-
plicitly inferring information pertinent to the designated tasks. Li et al. [Li+23a] explores the functional
resemblance between a single layer of self-attention and gradient descent on a softmax regression problem,
offering upper bounds on their difference. Min et al. [Min+22] notes that the alteration of label parts in
prompts does not drastically impair the ICL ability. They contend that ICL is invoked when prompts reveal
information about the label space, input distribution, and sequence structure.

Another collection of works have sought to understand transformers from an approximation-theoretic
perspective. Yun et al. [Yun+19; Yun+20] established that transformers can universally approximate any
sequence-to-sequence function under some assumptions. Investigations by Edelman et al. [Ede+22] and
Likhosherstov, Choromanski, and Weller [LCW21] indicate that a single-layer self-attention can learn sparse
functions of the input sequence, where sample complexity and hidden size are only logarithmic relative to the
sequence length. Further studies by P�erez, Marinkovi�c, and Barcel�o [PMB19], Dehghani et al. [Deh+19],
and Bhattamishra, Patel, and Goyal [BPG20] indicate that the vanilla transformer and its variants exhibit
Turing completeness. Liu et al. [Liu+23] showed that transformers can approximate finite-state automata
with few layers. Bai et al. [Bai+23] showed that transformers can implement a variety of statistical machine
learning algorithms as well as model selection procedures. Abernethy et al. [Abe+23] showed that a pre-
trained transformer can be used to define a transformer that segments a prompt into examples and labels and
learns to solve a sparse retrieval task. Zhang et al. [Zha+23] interpreted in-context learning via a Bayesian
model averaging process.

A handful of recent works have developed provable guarantees for transformers trained with gradient-

2To see this, suppose (xi, yi) are i.i.d. with x ? N(0, ?) and y = ?w, x?. A single step of gradient descent under the squared
(cid:1) w ? x??w. Clearly, this is not

(cid:1) = x? (cid:0) 1

(cid:80)n

(cid:80)n

i=1 xix?

i

n

loss from a zero initialization yields the predictor x (cid:55)? x? (cid:0) 1
close to x?w when ? ?= Id.

n

i=1 yixi

3

based optimization. Jelassi, Sander, and Li [JSL22] analyzed the dynamics of gradient descent in vision
transformers for data with spatial structure. Li, Li, and Risteski [LLR23] demonstrated that a single-layer
transformer trained by a gradient method could learn a topic model, treating learning semantic structure
as detecting co-occurrence between words and theoretically analyzing the two-stage dynamics during the
training process.

Finally, we note a concurrent work by Ahn et al. [Ahn+23] on the optimization landscape of single
layer transformers with linear self-attention layers as we do in this work. They show that there exist global
minima of the population objective of the transformer that can achieve small prediction error with anisotropic
Gaussian data, and they characterize some critical points of deep linear self-attention networks.
In this
work, we show that despite nonconvexity, gradient flow with a suitable random initialization converges to a
global minimum that achieves small prediction error for anistropic Gaussian data. We also characterize the
prediction error when test prompts come from a new (possibly nonlinear) task, when there is distribution
shift, and when transformers are trained on prompts with possibly different covariate distributions across
prompts.

3 Preliminaries

(cid:17)

(cid:16)1 2
3 4

Notation We first describe the notation we use in the paper. We write [n] = {1, 2, ..., n}. We use
? to denote the Kronecker product, and Vec the vectorization operator in column-wise order. For ex-
= (1, 3, 2, 4)?. We write the inner product of two matrices A, B ? Rm�n as
ample, Vec
?A, B? = tr(AB?). We use 0n and 0m�n to denote the zero vector and zero matrix of size n and m � n,
respectively. For a general matrix A, Ak: and A:k denote the k-th row and k-th column, respectively. We de-
note the matrix operator norm and Frobenius norm as ?�?op and ?�?F . We use Id to denote the d-dimensional
identity matrix and sometimes we also use I when the dimension is clear from the context. For a positive
semi-definite matrix A, we write ?x?2
A := x?Ax. Unless otherwise defined, we use lower case letters for
scalars and vectors, and use upper case letters for matrices.

3.1

In-context learning

We begin by describing a framework for in-context learning of function classes, as initiated by Garg et al.
[Gar+22]. In-context learning refers to the behavior of models that operate on sequences, called prompts,
of input-output pairs (x1, y1, . . . , xN , yN , xquery), where yi = h(xi) for some (unknown) function h and
examples xi and query xquery. The goal for an in-context learner is to use the prompt to form a prediction
(cid:98)y(xquery) for the query such that (cid:98)y(xquery) ? h(xquery).

From this high-level description, one can see that at a surface level, the behavior of in-context learning
is no different than that of a standard learning algorithm: the learner takes as input a training dataset and
returns predictions on test examples. For instance, one can view ordinary least squares as an �in-context
learner� for linear models. However, the rather unique feature of in-context learners is that these learning
algorithms can be the solutions to stochastic optimization problems defined over a distribution of prompts.
We formalize this notion in the following definition.
Definition 3.1 (Trained on in-context examples). Let Dx be a distribution over an input space X , H ? Y X
a set of functions X ? Y, and DH a distribution over functions in H. Let ? : Y � Y ? R be a loss function.
Let S = ?n?N{(x1, y1, . . . , xn, yn) : xi ? X , yi ? Y} be the set of finite-length sequences of (x, y) pairs
and let

F? = {f? : S � X ? Y, ? ? ?}

4

be a class of functions parameterized by ? in some set ?. For N > 0, we say that a model f : S � X ? Y
is trained on in-context examples of functions in H under loss ? w.r.t. (DH, Dx) if f = f?? where ?? ? ?
satisfies

?? ? argmin???
EP =(x1,h(x1),...,xN ,h(xN ),xquery) [? (f?(P ), h(xquery))] ,
i.i.d.? Dx and h ? DH are independent. We call N the length of the prompts seen during

(3.1)

where xi, xquery
training.

As mentioned above, this definition naturally leads to a method for learning a learning algorithm
from data: Sample independent prompts by sampling a random function h ? DH and feature vectors
i.i.d.? Dx, and then minimize the objective function appearing in (3.1) using stochastic gradient
xi, xquery
descent or other stochastic optimization algorithms. This procedure returns a model that is learned from
in-context examples and can form predictions for test (query) examples given a sequence of training data.
This leads to the following natural definition that quantifies how well such a model performs on in-context
examples corresponding to a particular hypothesis class.

Definition 3.2 (In-context learning of a hypothesis class). Let Dx be a distribution over an input space
X , H ? Y X a class of functions X ? Y, and DH a distribution over functions in H. Let ? : Y �
Y ? R be a loss function. Let S = ?n?N{(x1, y1, . . . , xn, yn) : xi ? X , yi ? Y} be the set of finite-
length sequences of (x, y) pairs. We say that a model f : S � X ? Y defined on prompts of the form
P = (x1, h(x1), . . . , xM , h(xM ), xquery) in-context learns a hypothesis class H under loss ? with respect to
(DH, Dx) up to error ? ? R if there exists a function MDH,Dx(?) : (0, 1) ? N such that for every ? ? (0, 1),
and for every prompt P of length M ? MDH,Dx(?),

EP =(x1,h(x1),...,xM ,h(xM ),xquery)

(cid:20)

(cid:16)
?

f (P ), h (xquery)

(cid:17)(cid:21)

? ? + ?,

(3.2)

where the expectation is over the randomness in xi, xquery

i.i.d.? Dx and h ? DH.

The additive error term ? in Definition 3.2 above allows for the possibility that the model does not
achieve arbitrarily small error. This error could come from using a model which is not complex enough
to learn functions in H or from considering a non-realizable setting where it is not possible to achieve
arbitrarily small error.

With these two definitions in hand, we can formulate the following questions: suppose a function class
F? is given and DH corresponds to random instances of hypotheses in a hypothesis class H. Can a model
from F? that is trained on in-context examples of functions in H w.r.t. (DH, Dx) in-context learn the
hypothesis class H w.r.t. (DH, Dx) with small prediction error? Do standard gradient-based optimization
algorithms suffice for training the model from in-context examples? How long must the contexts be during
training and at test time to achieve small prediction error? In the remaining sections, we shall answer these
questions for the case of one-layer transformers with linear self-attention modules when the hypothesis class
is linear models, the loss of interest is the squared loss, and the marginals are (possibly anisotropic) Gaussian
marginals.

3.2 Linear self-attention networks

Before describing the particular transformer models we analyze in this work, we first recall the definition
of the softmax-based single-head self-attention module [Vas+17]. Let E ? Rde�dN be an embedding ma-
trix that is formed using a prompt (x1, y1, . . . , xN , yN , xquery) of length N . The user has the freedom to

5

determine how this embedding matrix is formed from the prompt. One natural way to form E is to stack
(xi, yi)? ? Rd+1 as the first N columns of E and to let the final column be (xquery, 0)?; if xi ? Rd, yi ? R,
we would then have de = d + 1 and dN = N + 1. Let W K, W Q ? Rdk�de and W V ? Rdv�de be the key,
query, and value weight matrices, W P ? Rde�dv the projection matrix, and ? > 0 a normalization factor.
The softmax self-attention module takes as input an embedding matrix E of width dN and outputs a matrix
of the same size,

fAttn(E; W K, W Q, W V , W P ) = E + W P W V E � softmax

(cid:18) (W KE)?W QE
?

(cid:19)

,

where softmax is applied column-wise and, given a vector input of v, the i-th entry of softmax(v) is given
by exp(vi)/ (cid:80)
s exp(vs). The dN �dN matrix appearing inside the softmax is referred to as the self-attention
matrix. Note that fAttn can take as its input a sequence of arbitrary length.

In this work, we consider a simplified version of the single-layer self-attention module, which is more
amenable to theoretical analysis and yet is still capable of in-context learning linear models. In particular,
we consider a single-layer linear self-attention (LSA) model, which is a modified version of fAttn where
we remove the softmax nonlinearity, merge the projection and value matrices into a single matrix W P V ?
Rde�de, and merge the query and key matrices into a single matrix W KQ ? Rde�de. We concatenate these
matrices into ? = (W KQ, W P V ) and denote

fLSA(E; ?) = E + W P V E �

E?W KQE
?

.

(3.3)

We note that recent theoretical works on understanding transformers looked at identical models [Osw+22;
Li+23b; Ahn+23]. It is noteworthy that recent empirical work has shown that state-of-the-art trained vision
transformers with standard softmax-based attention modules are such that (W K)?W Q and W P W V are
nearly multiples of the identity matrix [TK23], which can be represented under the parameterization we
consider.

The user has the flexibility to determine the method for constructing the embedding matrix from a
prompt P = (x1, y1, . . . , xN , yN , xquery). In this work, for a prompt of length N, we shall use the following
embedding, which stacks (xi, yi)? ? Rd+1 into the first N columns with (xquery, 0)? ? Rd+1 as the last
column:

E = E(P ) =

(cid:18)x1 x2
y2
y1

� � � xN xquery
� � � yN

0

(cid:19)

? R(d+1)�(N +1).

(3.4)

We take the normalization factor ? to be the width of embedding matrix E minus one, i.e., ? = dN ?1, since
each element in E � E? is a inner product of two vectors of length dN . Under the above token embedding,
we take ? = N. We note that there are alternative ways to form the embedding matrix with this data, e.g.
by padding all inputs and labels into vectors of equal length and arranging them into a matrix [Aky+22],
or by stacking columns that are linear transformations of the concatenation (xi, yi) [Gar+22], although the
dynamics of in-context learning will differ under alternative parameterizations.

The network�s prediction for the token xquery will be the bottom-right entry of matrix output by fLSA,

namely,

(cid:98)yquery = (cid:98)yquery(E; ?) = [fLSA(E; ?)](d+1),(N +1).
Here and after, we may occasionally suppress dependence on ? and write (cid:98)yquery(E; ?) as (cid:98)yquery. Since the
prediction takes only the right-bottom entry of the token matrix output by the LSA layer, actually only part

6

of W P V and W KQ affect the prediction. To see how, let us denote

W P V =

?

?

W P V
wP V
11
12
21 )? wP V
(wP V
22

?
? ? R(d+1)�(d+1), W KQ =

?

?

W KQ
wKQ
12
11
21 )? wKQ
(wKQ

22

?
? ? R(d+1)�(d+1),

(3.5)

11 ? Rd�d; wP V

where W P V
the prediction (cid:98)yquery is

12 , wP V

21 ? Rd; wP V

22 ? R; and W KQ

11 ? Rd�d; wKQ

12 , wKQ

21 ? Rd; wKQ

22 ? R. Then,

(cid:16)

(cid:98)yquery =

(wP V

21 )? wP V
22

(cid:17)

�

(cid:19)

(cid:18) EE?
N

?

?

W KQ
11
(wKQ
21 )?

?

? xquery,

(3.6)

since only the last row of W P V and the first d columns of W KQ affects the prediction, which means we can
simply take all other entries zero in the following sections.

3.3 Training procedure

In this work, we will consider the task of in-context learning linear predictors. We will assume training
prompts are sampled as follows. Let ? be a positive definite covariance matrix. Each training prompt,
indexed by ? ? N, takes the form of P? = (x?,1, h? (x?1), . . . , x?,N , h? (x?,N ), x?,query), where task weights
w?

i.i.d.? N(0, Id), inputs x?,i, x?,query
Each prompt corresponds to an embedding matrix E? , formed using the transformation (3.4):

i.i.d.? N(0, ?), and labels h? (x) = ?w? , x?.

E? :=

(cid:18) x?,1
?w? , x?,1?

x?,2
?w? , x?,2?

� � �
� � �

x?,N
?w? , x?,N ?

x?,query
0

(cid:19)

? R(d+1)�(N +1).

We denote the prediction of the LSA model on the query label in the task ? as (cid:98)y?,query, which is the bottom-
right element of fLSA(E? ), where fLSA is the linear self-attention model defined in (3.3). The empirical risk
over B independent prompts is defined as

(cid:98)L(?) =

1
2B

B
(cid:88)

(cid:18)

? =1

(cid:98)y?,query ? ?w? , x?,query?

(cid:19)2

.

(3.7)

We shall consider the behavior of gradient flow-trained networks over the population loss induced by the
limit of infinite training tasks/prompts B ? ?:

L(?) = lim
B??

(cid:98)L(?) =

1
2

Ew? ,x?,1,��� ,x?,N ,x?,query

(cid:2)((cid:98)y?,query ? ?w? , x?,query?)2(cid:3)

(3.8)

Above, the expectation is taken w.r.t. the covariates {x?,i}N
w? , i.e. over x?,i, xquery
descent with infinitesimal step size and has dynamics given by the following differential equation:

i=1 ? {xquery} in the prompt and the weight vector
i.i.d.? N(0, ?) and w? ? N(0, Id). Gradient flow captures the behavior of gradient

d
dt

? = ??L(?).

(3.9)

We will consider gradient flow with an initialization that satisfies the following.

7

Assumption 3.3 (Initialization). Let ? > 0 be a parameter, and let ? ? Rd�d be any matrix satisfying
?????F = 1 and ?? ?= 0d�d. We assume

W P V (0) = ?

(cid:19)

(cid:18)0d�d 0d
0?
1
d

, W KQ(0) = ?

(cid:18)??? 0d
0?
0
d

(cid:19)

.

(3.10)

This initialization is satisfied for a particular class of random initialization schemes: if M has i.i.d. en-
tries from a continuous distribution, then by setting ??? = M M ?/?M M ??F , the assumption is satisfied
almost surely. The reason we use this particular initialization scheme will be made more clear in Section 5
when we describe the proof, but at a high-level this is due to the fact that the predictions (3.6) can be viewed
as the output of a two-layer linear network, and initializations satisfying Assumption 3.3 allow for the layers
to be �balanced� throughout the gradient flow trajectory. Random initializations that induce this balancedness
condition have been utilized in a number of theoretical works on deep linear networks [DHL18; ACH18;
Aro+19; Azu+21]. We leave the question of convergence under alternative random initialization schemes
for future work.

4 Main results

In this section, we present the main results of this paper. First, in Section 4.1, we prove the gradient flow
on the population loss will converge to a specific global optimum. We characterize the prediction error
of the trained transformer at this global minimum when given a prompt from a new prediction task. Our
characterization allows for the possibility that this new prompt comes from a nonlinear prediction task.
We then instantiate our results for well-specified linear regression prompts and characterize the number
of samples needed to achieve small prediction error, showing that transformers can in-context learn linear
models when trained on in-context examples of linear models.

Next, in Section 4.2, we analyze the behavior of the trained transformer under a variety of distribution
shifts. We show the transformer is robust to a number of distribution shifts, including task shift (when the
labels in the prompt are not deterministic linear functions of their input) and query shift (when the query
example xquery has a possibly different distribution than the test prompt). On the other hand, we show that
the transformer suffers from covariate distribution shifts, i.e. when the training prompt covariate distribution
differs from the test prompt covariate distribution.

Finally, motivated by the failure of the trained transformer under covariate distribution shift, we con-
sider in Section 4.3 the setting of training on in-context examples with varying covariate distributions across
prompts. We prove that transformers with a single linear self-attention layer trained by gradient flow con-
verge to a global minimum of the population objective, but that the trained transformer still fails to perform
well on new prompts. We complement our proof in the linear self-attention case with experiments on large,
nonlinear transformer architectures which we show are more robust under covariate shifts.

4.1 Convergence of gradient flow and prediction error for new tasks

First, we prove that under suitable initialization, gradient flow will converge to a global optimum.

Theorem 4.1 (Convergence and limits). Consider gradient flow of the linear self-attention network fLSA
defined in (3.3) over the population loss (3.8). Suppose the initialization satisfies Assumption 3.3 with
initialization scale ? > 0 satisfying ?2???op

?

d < 2 where we have defined
(cid:19)

tr(?)Id ? Rd�d.

(cid:18)

? :=

1 +

1
N

1
N

? +

8

Then gradient flow converges to a global minimum of the population loss (3.8). Moreover, W P V and W KQ
converge to W P V

respectively, where

and W KQ

?

?

W KQ

? = (cid:2)tr (cid:0)??2(cid:1)(cid:3)? 1

4

?

?

??1 0d
0?
d

0

?

? ,

W P V

? = (cid:2)tr (cid:0)??2(cid:1)(cid:3) 1

4

?

?

0d�d 0d

0?
d

1

?

? .

(4.1)

The full proof of this theorem appears in Appendix A. We note that if we restrict our setting to ? = Id,
then the limiting solution described found by gradient flow is quite similar to the construction of Oswald
et al. [Osw+22]. Since the prediction of the transformer is the same if we multiply W P V by a constant c ?= 0
and simultaneously multiply W KQ by c?1, the only difference (up to scaling) is that the top-left entry of
their W KQ matrix is Id rather than the (1 + (d + 1)/N )?1Id that we find for the case ? = Id.

Next, we would like to characterize the prediction error of the trained network described above when the
network is given a new prompt. Let us consider a prompt of the form (x1, ?w, x1?, . . . , xM , ?w, xM ?, xquery)
i.i.d.? N(0, ?). A simple calculation shows that the prediction (cid:98)yquery at the global
where w ? Rd and xi, xquery
optimum with parameters W KQ

and W P V

is given by

?

(cid:98)yquery =

(cid:16)

0?
d

?

?

1
M

(cid:17)

1

?

(cid:80)M

= x?

query??1

(cid:32)

1
M

M
(cid:88)

i=1

xix?
i

w.

M xqueryx?

query

i=1 xix?
(cid:80)M
1
M

i + 1
i=1 w?xix?
i
(cid:33)

(cid:80)M

1
M
(cid:80)M

i=1 xix?
i w
i=1 w?xix?

i w

1
M

?

?

?

?

?

?

??1 0d
0?
d

0

?

?

xquery

0

?

?

(4.2)

When the length of prompts seen during training N is large, ??1 ? ??1, and when the test prompt length
M is large, 1
queryw. Thus, for sufficiently large prompt lengths, the
M
trained transformer indeed in-context learns the class of linear predictors.

i ? ?, so that (cid:98)yquery ? x?

i=1 xix?

(cid:80)M

In fact, we can generalize the above calculation for test prompts which could take a significantly different
form than the training prompts. Consider prompts that are of the form (x1, y1, . . . , xn, yn, xquery) where, for
some joint distribution D over (x, y) pairs with marginal distribution x ? N(0, ?), we have (xi, yi) i.i.d.? D
and xquery ? N(0, ?) independently. Note that this allows for a label yi to be a nonlinear function of the
input xi. The prediction of the trained transformer for this prompt is then

(cid:98)yquery =

(cid:16)

0?
d

?

?

1
M

(cid:17)
1

(cid:80)M

i=1 xix?
1
M

i + 1
(cid:80)M
i=1 x?

M xqueryx?
i yi

query

= x?

query??1

(cid:32)

1
M

M
(cid:88)

i=1

(cid:33)

yixi

.

1
M

1
M

(cid:80)M

i=1 xiyi
i=1 y2
i

(cid:80)M

?

?

?

?

??1 0d
0?
d

0

?

?

?

?

?

?

xquery

0

(4.3)

Just as before, when N is large we have ??1 ? ??1, and so when M is large as well this implies

(cid:98)yquery ? x?

query??1E(x,y)?D[yx] = x?

query

(cid:32)

argmin
w?Rd

E(x,y)?D[(y ? ?w, x?)2]

.

(4.4)

(cid:33)

This suggests that trained transformers in-context learn the best linear predictor over a distribution when the
test prompt consists of i.i.d. samples from a joint distribution over feature-response pairs. In the following
theorem, we formalize the above and characterize the prediction error when prompts take this form.

9

Theorem 4.2. Let D be a distribution over (x, y) ? Rd � R, whose marginal distribution on x is Dx =
N(0, ?). Assume ED[y], ED[xy], ED[y2xx?] exist and are finite. Assume the test prompt is of the form
P = (x1, y1, . . . , xM , yM , xquery), where (xi, yi), (xquery, yquery) i.i.d.? D. Let f ?
LSA be the LSA model with
parameters W P V
in (4.1), and (cid:98)yquery is the prediction for xquery given the prompt. If we define

and W KQ

?

?

a := ??1E(x,y)?D [xy] ,

? := E(x,y)?D

(cid:104)(cid:0)xy ? E (xy) (cid:1)(cid:0)xy ? E (xy) (cid:1)?(cid:105)

,

(4.5)

then, for ? = ? + 1

N ? + 1

N tr(?)Id. we have,

E ((cid:98)yquery ? yquery)2 = min
w?Rd
(cid:124)

E (?w, xquery? ? yquery)2
(cid:125)

(cid:123)(cid:122)
Error of best linear predictor
tr (cid:2)???2?(cid:3) +

(cid:104)
?a?2

1
N 2

+

1
M

??2?3 + 2 tr(?) ?a?2

??2?2 + tr(?)2 ?a?2

??2?

(cid:105)

,

(4.6)

where the expectation is over (xi, yi), (xquery, yquery) i.i.d.? D.

The full proof is deferred to Appendix B. Let us now make a few remarks on the above theorem before

considering particular instances of D where we may provide more explicit bounds on the prediction error.

First, this theorem shows that, provided the length of prompts seen during training (N ) and the length
of the test prompt (M ) is large enough, a transformer trained by gradient flow from in-context examples
achieves prediction error competitive with the best linear model. Next, our bound shows that the length
of prompts seen during training and the length of prompts seen at test-time have different effects on the
prediction error:
ignoring dimension and covariance-dependent factors, the prediction error is at most
O(1/M + 1/N 2), decreasing more rapidly as a function of the training prompt length N compared to
the test prompt length M . Additionally, it is worth noting that even if M ? ?, the gap between the predic-
tion error of the transformer with that of the best linear predictor does not vanish unless N ? ? as well.
Thus, the transformer is inherently limited by training on finite-length prompts.

Let us now consider when D corresponds to noiseless linear models, so that for some w ? Rd, we have
(x, y) = (x, ?w, x?), in which case the prediction of the trained transformer is given by (4.2). Moreover,
?? + ?ww??. Hence
a simple calculation shows that the ? from Theorem 4.2 takes the form ? = ?w?2
Theorem 4.2 implies the prediction error for the prompt P = (x1, ?w, x1?, . . . , xM , ?w, xM ?, xquery) is

(cid:110)

Ex1,...,xM ,xquery ((cid:98)yquery ? ?w, xquery?)2
(cid:111)
1
??2?3 + tr(??2?2) ?w?2
=
M
d + 1
M

? + 2 ?w?2

(cid:104)
?w?2

1
N 2

?w?2

?w?2

? +

?

?

(cid:110)

+

?w?2

1
N 2
2 tr(?) + ?w?2

??2?3 + 2 ?w?2
??1 tr(?)2(cid:105)

,

??2?2 tr(?) + ?w?2

??2? tr(?)2(cid:111)

The inequality above uses that ? ? ?. Finally, if we assume that w ? N(0, Id) and denote ? as the condition
number of ?, then by taking expectations over w we get the following:

Ex1,...,xM ,xquery,w ((cid:98)yquery ? ?w, xquery?)2 ?

?

(d + 1) tr(?)
M
(d + 1) tr(?)
M

+

+

10

(cid:2)tr(?) + 2d tr(?) + tr(??1) tr(?)2(cid:3)

1
N 2
(1 + 2d + d2?) tr(?)
N 2

,

From the upper bound above, we can see the rate w.r.t M and N are still at most O(1/M ) and O(1/N 2)
respectively. Moreover, the generalization error also scales with dimension d, tr(?) and the condition
number ?. This suggests that for in-context examples involving covariates of greater variance, or a more
ill-conditioned covariance matrix, the generalization error will be higher for the same lengths of training and
testing prompts. Putting the above together with Theorem 4.2, Definition 3.1 and Definition 3.2, we get the
following corollary.

Corollary 4.3. The transformer fLSA trained on length-N prompts of in-context examples of functions in
{x (cid:55)? ?w, x?} w.r.t. w ? N(0, Id) and Dx = N(0, ?) by gradient flow on the population loss (3.8) for
initializations satisfying Assumption 3.3 converges to the model fLSA(� ; W KQ
). This model takes a
prompt P = (x1, y1, . . . , xM , yM , xquery) and returns a prediction (cid:98)yquery for xquery given by

, W P V
?

?

(cid:98)yquery = [fLSA(P ; W KQ

?

, W P V
?

)]d+1,M +1 = x?

query

(cid:18)

? +

1
N

? +

tr(?)
N

Id

(cid:19)?1 (cid:32)

(cid:33)

yixi

.

1
M

M
(cid:88)

i=1

This model in-context learns the class of linear models {x (cid:55)? ?w, x?} with respect to w ? N(0, Id) and
Dx = N(0, ?) up to error ? := (1 + 2d + d2?) tr(?)/N 2 (where ? is the condition number of ?): provided
M ? (d + 1) tr(?)??1, the model achieves prediction error at most ? + ?.

It is worth emphasizing that the transformer fLSA(� ; W KQ

) only learns the function class up to
error ? = O(1/N 2) in the sense of Definition 3.2. In particular, training on finite-length prompts leads to
prediction error bounded away from zero.

, W P V
?

?

4.2 Behavior of trained transformer under distribution shifts

Using the identity (4.3), it is straightforward to characterize the behavior of the trained transformer under
a variety of distribution shifts. In this section, we shall examine a number of shifts that were first explored
empirically for transformer architectures by Garg et al. [Gar+22]. Although their experiments were for
transformers trained by gradient descent, we find that (in the case of linear models) many of the behaviors
of the trained transformers under distribution shift are identical to those predicted by our theoretical char-
acterizations of the performance of transformers with a single linear self-attention layer trained by gradient
flow on the population.

Following Garg et al. [Gar+22], for training prompts of the form (x1, h(x1), . . . , xN , h(xN ), xquery),
i.i.d.? Dtest
x ,

H , while for test prompts let us assume xi

and h ? Dtrain

i.i.d.? Dtrain

x

let us assume xi, xquery
xquery ? Dtest

query, and h ? Dtest

H . We will consider the following distinct categories of shifts:

� Task shifts: Dtrain

� Query shifts: Dtest

H ?= Dtest
H .
query ?= Dtest
x .

� Covariate shifts: Dtrain

x

?= Dtest
x .

In the following, we shall fix Dtrain

x = N(0, ?) and vary the other distributions. Recall from (4.3) that

the prediction for a test prompt (x1, y1, . . . , xN , yN , xquery) is given by (for N large),

(cid:98)yquery = x?

query??1

(cid:32)

1
M

M
(cid:88)

i=1

(cid:33)

yixi

? x?

query??1

(cid:32)

1
M

M
(cid:88)

i=1

(cid:33)

yixi

.

(4.7)

11

Task shifts. These shifts are tolerated easily by the trained transformer. As Theorem 4.2 shows, the trained
transformer is competitive with the best linear model provided the prompt length during training and at test
time is large enough. In particular, even if the prompt is such that the labels yi are not given by ?w, xi? for
some w ? N(0, Id), the trained transformer will compute a prediction which has error competitive with the
best linear model that fits the test prompt.

For example, consider a prompt corresponding to a noisy linear model, so that the prompt consists of
a sequence of (xi, yi) pairs where yi = ?w, xi? + ?i for some arbitrary vector w ? Rd and independent
sub-Gaussian noise ?i. Then from (4.7), the prediction of the transformer on query examples is

(cid:98)yquery ? x?

query??1

(cid:32)

1
M

M
(cid:88)

i=1

(cid:33)

yixi

= x?

query??1

(cid:32)

1
M

M
(cid:88)

i=1

(cid:33)

xix?
i

w + x?

query??1

(cid:32)

1
M

M
(cid:88)

i=1

(cid:33)

?ixi

.

Since ?i is mean zero and independent of xi, this is approximately x?
queryw when M is large. And note that
this calculation holds for an arbitrary vector w, not just those which are sampled from an isotropic Gaussian
or those with a particular norm. This behavior coincides with that of the trained transformers observed
by Garg et al. [Gar+22].

Query shifts. Continuing from (4.7), since yi = ?w, xi?,

(cid:98)yquery ? x?

query??1

(cid:32)

1
M

M
(cid:88)

i=1

(cid:33)

xix?
i

w.

From this we see that whether query shifts can be tolerated hinges upon the distribution of the xi�s. Since
Dtrain

x , if M is large then

x = Dtest

(cid:98)yquery ? x?

query??1?w = x?

queryw.

(4.8)

Thus, very general shifts in the query distribution can be tolerated. On the other hand, very different behavior
can be expected if M is not large and the query example depends on the training data. For example, if the
query example is orthogonal to the subspace spanned by the xi�s, the prediction will be zero, as was observed
with transformer architectures by Garg et al. [Gar+22].

Covariate shifts.
transformer. This can be easily seen due to the identity (4.3): when Dtrain
in (4.8) does not hold as 1
i=1 xix?
M
consider test prompts where the covariates are scaled by a constant c ?= 1, then

In contrast to task and query shifts, covariate shifts cannot be fully tolerated in the
x , then the approximation
i will not cancel ??1 when M and N are large. For instance, if we

?= Dtest

(cid:80)M

x

(cid:98)yquery ? x?

query??1

(cid:32)

1
M

M
(cid:88)

i=1

(cid:33)

xix?
i

? x?

query??1c2?w = c2x?

queryw ?= x?

queryw.

This failure mode of the trained transformer with linear self-attention was also observed in the trained trans-
former architectures by Garg et al. [Gar+22]. This suggests that although the predictions of the transformer
may look similar to those of ordinary least squares in some settings, the algorithm implemented by the
transformer is not the same since ordinary least squares is robust to scaling of the features by a constant.

It may seem surprising that a transformer trained on linear regression tasks fails in settings where ordi-
nary least squares performs well. However, both the linear self-attention transformer we consider and the

12

transformers considered by Garg et al. [Gar+22] were trained on instances of linear regression when the
covariate distribution Dx over the features was fixed across instances. This leads to the natural question
of what happens if the transformers instead are trained on prompts where the covariate distribution varies
across instances, which we explore in the following section.

4.3 Transformers trained on prompts with random covariate distributions

In this section, we will consider a variant of training on in-context examples (in the sense of Definition 3.1)
where the distibution Dx is itself sampled randomly from a distribution, and training prompts are of the
i.i.d.? Dx and h ? DH. More formally, we can
form (x1, h(x1), . . . , xN , h(xN ), xquery) where xi, xquery
generalize Definition 3.1 as follows.

Definition 4.4 (Trained on in-context examples with random covariate distributions). Let ? be a distribution
over distributions Dx defined on an input space X , H ? Y X a set of functions X ? Y, and DH a
distribution over functions in H. Let ? : Y �Y ? R be a loss function. Let S = ?n?N{(x1, y1, . . . , xn, yn) :
xi ? X , yi ? Y} be the set of finite-length sequences of (x, y) pairs and let

F? = {f? : S � X ? Y, ? ? ?}

be a class of functions parameterized by some set ?. We say that a model f : S � X ? Y is trained on
in-context examples of functions in H under loss ? w.r.t. DH and distribution over covariate distributions ?
if f = f?? where ?? ? ? satisfies

?? ? argmin???

EP =(x1,h(x1),...,xN ,h(xN ),xquery) [? (f?(P ), h(xquery))] ,

(4.9)

where Dx ? ?, xi, xquery

i.i.d.? Dx and h ? DH.

We recover the previous definition of training on in-context examples by taking ? to be concentrated
on a singleton, supp(?) = {Dx}. The natural question is then, if a model f is trained on in-context
examples from a function class H w.r.t. DH and a distribution ? over covariate distributions, and if one
then samples some covariate distribution Dx ? ?, does f in-context learn H w.r.t. (DH, Dx) for that Dx
(cf. Definition 3.2) with small prediction error? Since Dx is random, we can hope that this may hold in
expectation or with high probability over the sampling of the covariate distribution. In the remainder of this
section, we will explore this question for transformers with a linear self-attention layer trained by gradient
flow on the population loss.

We shall again consider the case where the covariates have Gaussian marginals, xi ? N(0, ?), but we
shall now assume that within each prompt we first sample a random covariance matrix ?. For simplicity, we
will restrict our attention to the case where ? is diagonal. More formally, we shall assume training prompts
are sampled as follows. For each independent task indexed by ? ? [B], we first sample w? ? N(0, Id).
Then, for each task ? and coordinate i ? [d], we sample ??,i independently such that the distribution of each
??,i is fixed and has finite third moments and is strictly positive almost surely. We then form a diagonal
matrix

?? = diag(??,1, . . . , ??,d).

Thus the diagonal entries of ?? are independent but could have different distributions, and ?? is iden-
tically distributed for ? = 1, . . . , B. Then, conditional on ?? , we sample independent and iden-
tically distributed x?,1, . . . , x?,N , x?,query ? N(0, ?? ). A training prompt is then given by P? =

13

(x?,1, ?w? , x?,1?, . . . , x?,N , ?w? , x?,N ?, x?,query) Notice that here, x?,i, x?,query are conditionally indepen-
dent given the covariance matrix ?? , but not independent in general. We consider the same token embed-
ding matrix as (3.4) and linear self-attention network, which forms the prediction (cid:98)yquery,? as in (3.6). The
empirical risk is the same as before (see (3.7)), and as in (3.8), we then take B ? ? and consider the
gradient flow on the population loss. The population loss now includes an expectation over the distribution
of the covariance matrices in addition to the task weight w? and covariate distributions, and is given by

L(?) =

1
2

Ew? ,?? ,x?,1,��� ,x?,N ,x?,query

(cid:2)((cid:98)y?,query ? ?w? , x?,query?)2(cid:3) .

(4.10)

In the main result for this section, we show that gradient flow with a suitable initialization converges to

a global minimum, and we characterize the limiting solution. The proof will be deferred to Appendix C.

Theorem 4.5 (Global convergence in random covariance case). Consider gradient flow of the linear self-
attention network fLSA defined in (3.3) over the population loss (4.10), where ?? are diagonal with indepen-
dent diagonal entries which are strictly positive a.s. and have finite third moments. Suppose the initialization
satisfies Assumption 3.3, ?E?? ??F ?= 0, with initialization scale ? > 0 satisfying

?2 <

2 ?E?? ??2
F
E ??? ?op ??? ?2

F

(cid:105) .

?

(cid:104)

d

(4.11)

Then gradient flow converges to a global minimum of the population loss (4.10). Moreover, W P V and W KQ
converge to W P V

respectively, where

and W KQ

?

?

W KQ

? =

(cid:13)
(cid:2)E?? ?2
(cid:13)
(cid:13)

?

(cid:3)?1 E (cid:2)?2
?

? 1
2

(cid:3)(cid:13)
(cid:13)
(cid:13)
F

�

W P V

? =

(cid:13)
(cid:2)E?? ?2
(cid:13)
(cid:13)

?

(cid:3)?1 E (cid:2)?2
?

1
2

(cid:3)(cid:13)
(cid:13)
(cid:13)
F

�

?

?

0d�d 0d

0?
d

1

0?
d
?

? ,

(cid:2)E?? ?2

?

(cid:3)?1 (cid:2)E?2
?

?

?

?

? ,

(cid:3) 0d
0

(4.12)

where ?? = N +1

N ?? + 1

N tr(?? )Id ? Rd�d and the expectations above are over the distribution of ?? .

From this result, we can see why the trained transformer fails in the random covariance case. Suppose
we have a new prompt corresponding to a weight matrix w ? Rd and covariance matrix ?new, sampled
from the same distribution as the covariance matrices for training prompts, so that conditionally on ?new
i.i.d.? N(0, ?new). The ground-truth labels are given by yi = ?w, xi?, i ? [M ] and
we have xi, xquery
yquery = ?w, xquery?. At convergence, the prediction by the trained transformer on the new task will be

(cid:98)yquery =

(cid:16)

0?
d

?

?

1
M

(cid:17)

1

(cid:80)M

query

i=1 xix?
1
M

i + 1
(cid:80)M
i=1 x?
(cid:34)

M xqueryx?
i yi
M
(cid:88)

1
M

i=1

xix?
i

1
M

(cid:35)

w

= x?

query � (cid:2)E?2

?

(cid:3) (cid:2)E?? ?2

?

(cid:3)?1

�

1
M

(cid:80)M

i=1 xiyi
i=1 y2
i

(cid:80)M

?

?

?

?

(cid:2)E?? ?2

?

(cid:3)?1 (cid:2)E?2
?

0?
d

?

?

?

?

(cid:3) 0d
0

xquery

0

?

?

? x?

query � (cid:2)E?2

?

(cid:3) (cid:2)E?? ?2

?

(cid:3)?1

� ?neww almost surely when M ? ?.

(4.13)

14

The last line comes from the strong law of large numbers. Thus, in order for the prediction on the query
(cid:3)?1�?new to be close to the identity.
example to be close to the ground-truth x?
When ?? ? ?new is deterministic, this indeed is the case as we know from Theorem 4.2. However, this
clearly does not hold in general when ?? is random.

queryw, we need (cid:2)E?2

(cid:3) (cid:2)E?? ?2

?

?

To make things concrete, let us assume for simplicity that M, N ? ? so that ?? ? ?? and the

identity (4.13) holds (conditionally on ?new). Then, taking expectation over ?new in (4.13), we obtain

If we consider the case ??,i

E [ (cid:98)yquery| xquery, w] ? x?

query � (cid:2)E?2
i.i.d.? Exponential(1), so that E[?? ] = Id, E[?2

(cid:3) (cid:2)E?3
?

(cid:3)?1

� [E?? ] w.

?

? ] = 2Id, and E[?3

? ] = 6Id,

we get

1
3
This shows that for transformers with a single linear self-attention layer, training on in-context examples
with random covariate distributions does not allow for in-context learning of a hypothesis class with varying
covariate distributions.

E
(cid:98)yquery ?

?w, xquery?.

Experiments with large, nonlinear transformers. We have shown that even when trained on prompts
with random covariance matrices, transformers with a single linear self-attention layer fail to in-context learn
linear models with random covariance matrices. We now investigate the behavior of more complex trans-
former architectures that are trained on in-context examples of linear models, both in the fixed-covariance
case and in the random-covariance case.

We examine the performance of transformers with a GPT2 architecture [Rad+19] that are trained on
linear regression tasks with mean-zero Gaussian features with either a fixed covariance matrix or random
covariance matrices. For the fixed covariance case, the covariance matrix is fixed to the identity matrix
across prompts. For the random covariance case, covariates are drawn from x ? N(0, c?) where ? is
i.i.d.? Exponential(1) and c > 0 is a scaling factor. We set c = 1 during training and
diagonal with ?i
vary this value at test time. The transformer is trained using the procedure of Garg et al. [Gar+22] (see
Appendix E for more details). We consider linear models in d = 20 dimensions and we train on prompt
lengths of N = 40, 70, 100 with either fixed or random covariance matrices. The performance of these
trained models, when tested on new data with fixed covariance or random covariance matrices (c = 1, 4, 9),
is represented in six curves in Figure 1. Using the calculation (4.13), we can compare the prediction error
for the linear self-attention networks in the M ? ?, N ? ? limit (the black dash line) to those of GPT2
architectures. We additionally compare these models to the ordinary least-squares solution which is optimal
for this task.

From the figure, we can see that the GPT2 model trained on fixed covariance succeeds in the random
covariance setting if the variance is not too large, which shows that the larger nonlinear model is able to
generalize better than the model with a single linear self-attention layer. However, when the variance is large
(c = 4, 9 for the bottom two figures), the GPT2 model trained with fixed covariance is unsuccessful. When
trained on random covariance, the model performs better for test prompts from higher-variance random
covariance matrices, but still fails to match least squares when the scaling is largest (c = 9).

Furthermore, we notice some surprising behaviors when the test prompt length exceeds the training
prompt length (i.e., M > N ): there is an evident spike in prediction error, regardless of whether training
and testing were performed on fixed or random covariance, and the spike appears to decrease when evaluated
on prompts with higher variance. Although we are unsure of why the spike should decrease with higher-
variance prompts, the failure of large language models to generalize to larger contexts than they were trained

15

Figure 1: Normalized prediction error for transformers with GPT2 architectures as a function of the number
of in-context test examples M when trained on in-context examples of linear models in d = 20 dimensions.
Colored lines correspond to different training context lengths (N ? {40, 70, 100}) and different training
procedures (either a fixed identity covariance matrix or random diagonal covariance matrices with each
diagonal element sampled i.i.d. from the standard exponential distribution). The four figures correspond to
evaluating on either fixed covariance or random covariance matrices of different scales. The gray dashed
line shows the prediction error of zero estimator and the black dashed line the prediction error of LSA model
when M, N ? ?. The GPT2 models achieve smaller error when they are trained on random covariance
matrices with larger contexts, but their prediction error spikes when evaluated on contexts larger than those
they were trained on.

on is a well-known problem [Dai+19; Ani+22]. In our setting, we conjecture that this spike in error comes
from the absolute positional encodings in the GPT2 architecture. The positional encodings are randomly-
initialized and are learnable parameters but the encoding for position i is only updated if the transformer
encounters a prompt which has a context of length i. Thus, when evaluating on prompts of length M > N ,
the model is relying upon random positional encodings for M ? N samples. We note that a concurrent
work has explored the performance of transformers with GPT2 architectures for in-context learning of lin-
ear models and found that removing positional encoders improves performance when evaluating on larger
contexts [APG23]. We leave further investigation of this behavior for future work.

16

020406080100in-context examples0.00.20.40.60.81.0squared errorTest on Fixed Covariance020406080100in-context examples0.00.20.40.60.81.0squared errorTest on Random Covariance, Scale = 1.0020406080100in-context examples01234squared errorTest on Random Covariance, Scale = 4.0020406080100in-context examples02468squared errorTest on Random Covariance, Scale = 9.0Zero EstimatorLSA Limitfixedcov_N40fixedcov_N70fixedcov_N100randomcov_N40randomcov_N70randomcov_N100Least Squares5 Proof ideas

In this section, we briefly outline the proof sketch of Theorem 4.1. The full proof of this theorem is left for
Appendix A.

5.1 Equivalence to a quadratic optimization problem

We recall each task ? corresponds to a weight vector w? ? N(0, Id). The prompt inputs for this task are
i.i.d.? N(0, ?), which are also independent of w? . The corresponding labels are y?,j = ?w? , x?,j?. For
x?,j
each task ?, we can form the prompt into a token matrix E? ? R(d+1)�(N +1) as in (3.4), with the right-
bottom entry being zero.

The first key step in our proof is to recognize that the prediction (cid:98)yquery(E? ; ?) in the linear self-attention
model can be written as the output of a quadratic function u?H? u for some matrix H? depending on the
token embedding matrix E? and for some vector u depending on ? = (W KQ, W P V ). This is shown in the
following lemma, the proof of which is provided in Appendix A.1.

Lemma 5.1. Let E? ? R(d+1)�(N +1) be an embedding matrix corresponding to a prompt of length N
and weight w? . Then the prediction (cid:98)yquery(E? ; ?) for the query covariate can be written as the output of a
quadratic function,

where the matrix H? is defined as,

(cid:98)yquery(E? ; ?) = u?H? u,

H? =

1
2

X? ?

(cid:19)

(cid:18) E? E?
?
N

? R(d+1)2�(d+1)2

, X? =

?

?

0d�d
(x?,query)?

x?,query

0

?
? ? R(d+1)�(d+1)

(5.1)

and

u = Vec(U ) ? R(d+1)2

, U =

?

?

U11

u12
(u21)? u?1

?
? ? R(d+1)�(d+1),

where U11 = W KQ
11 ? Rd�d, u12 = wP V
particular components of W P V and W KQ, defined in (3.5).

21 ? Rd�1, u21 = wKQ

21 ? Rd�1, u?1 = wP V

22 ? R correspond to

This implies that we can write the original loss function (3.7) as

(cid:98)L =

1
2B

B
(cid:88)

(cid:16)

? =1

u?H? u ? w?

? x?,query

(cid:17)2

.

(5.2)

Thus, our problem is reduced to understanding the dynamics of an optimization algorithm defined in
terms of a quadratic function. We also note that this quadratic optimization problem is an instance of a rank-
one matrix factorization problem, a problem well-studied in the deep learning theory literature [Gun+17;
Aro+19; LMZ18; CLC19; Bel20; LLL20; Jin+23; SSX23].

Note, however, this quadratic function is non-convex. To see this, we will show that H? has negative
(cid:17)

eigenvalues. By standard properties of the Kronecker product, the eigenvalues of H? = 1
are the products of the eigenvalues of 1

2 X? and the eigenvalues of E? E?

N . Since E? E?

?

2 X? ?

(cid:16) E? E?
N

?

? is symmetric and

17

positive semi-definite, all of its eigenvalues are nonnegative. Since E? E?
? is nonzero almost surely, it thus
has at least one strictly positive eigenvalue. Thus, if X? has any negative eigenvalues, H? does as well. The
characteristic polynomial of X? is given by,

det(�I ? X? ) = det

?

?

�Id

?x?,query

?x?

?,query

�

?
? = �d?1 (cid:16)

�2 ? ?x?,query?2
2

(cid:17)

.

Therefore, we know almost surely, X? has one negative eigenvalue. Thus H? has at least d + 1 negative
eigenvalues, and hence the quadratic form u?H? u is non-convex.

5.2 Dynamical system of gradient flow

We now describe the dynamical system for the coordinates of u above. We prove the following lemma in
Appendix A.2.

Lemma 5.2. Let u = Vec (U ) := Vec

?

?

U11

u12
(u21)? u?1

?

? as in Lemma 5.1. Consider gradient flow over

L :=

(cid:16)

E

1
2

u?H? u ? w?

? x?,query

(cid:17)2

(5.3)

with respect to u starting from an initial value satisfying Assumption 3.3. Then the dynamics of U follows

d
dt
d
dt

U11(t) = ?u2

?1??U11? + u?1?2
(cid:104)

u?1(t) = ? tr

u?1??U11?(U11)? ? ?2(U11)?(cid:105)

,

(5.4)

and u12(t) = 0d, u21(t) = 0d for all t ? 0, where ? = (cid:0)1 + 1

N

(cid:1) ? + 1

N tr(?)Id ? Rd�d.

We see that the dynamics are governed by a complex system of d2 + 1 coupled differential equations.
Moreover, basic calculus (for details, see Lemma A.1) shows that these dynamics are the same as those of
gradient flow on the following objective function:

�? : Rd�d � R ? R,

�? (U11, u?1) = tr

?1??U11?(U11)? ? u?1?2(U11)?
u2

(cid:21)

.

(5.5)

(cid:20) 1
2

Actually, the loss function �? is simply the loss function L in (5.3) plus some constants that do not depend on
the parameter u. Therefore our problem is reduced to studying the dynamics of gradient flow on the above
objective function.

Our next key observation is that the set of global minima for �? satisfies the condition u?1U11 = ??1.
Thus, if we can establish global convergence of gradient flow over the above objective function �?, then we
have that u?1(t)U11(t) ? ??1 ?N ?? ??1.

Lemma 5.3. For any global minimum of �?, we have

u?1U11 = ??1.

18

(5.6)

Putting this together with Lemma 5.2, we see that at those global minima of the population objective
satisfying U11 = (c?)?1, u?1 = c and u12 = u21 = 0d, the transformer�s predictions for a new linear
regression task prompt are given by

(cid:98)yquery(E; ?) =

1
M

M
(cid:88)

i=1

yix?

i ??1xquery = w?

(cid:32)

1
M

M
(cid:88)

i=1

(cid:33)

xix?
i

??1xquery ? w?xquery.

Thus, the only remaining task is to show global convergence when gradient flow has an initialization satis-
fying Assumption 3.3.

5.3 PL inequality and global convergence

We now show that although the optimization problem is non-convex, a Polyak-?ojasiewicz (PL) inequality
holds, which implies that gradient flow converges to a global minimum. Moreover, we can exactly calculate
the limiting value of U11 and u?1.

Lemma 5.4. Suppose the initialization of gradient flow satisfies Assumption 3.3 with initialization scale
satisfying ?2 <

for ? = (1 + 1

2?

N )? + tr(?)

N Id. If we define

d???op

� :=

?

?2

d ???2

op tr (??1??1) tr (??1)

????2
F

?

(cid:104)
2 ?

d?2 ???op

(cid:105)

> 0,

(5.7)

then gradient flow on �? with respect to U11 and u?1 satisfies, for any t ? 0,

(cid:13)
(cid:13)
2
(cid:13)?�?(U11(t), u?1(t))
(cid:13)
(cid:13)
(cid:13)
2

:=

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

? �?
?U11

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
F

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

? �?
?u?1

(cid:12)
2
(cid:12)
(cid:12)
(cid:12)
(cid:12)

+

? �

(cid:18)

�?(U11(t), u?1(t)) ?

min
U11?Rd�d,u?1?R

(cid:19)

�?(U11, u?1)

.

(5.8)
Moreover, gradient flow converges to the global minimum of �?, and U11 and u?1 converge to the following,

lim
t??

u?1(t) = (cid:13)

1
2

(cid:13)??1(cid:13)
F and lim
(cid:13)
t??

U11(t) = (cid:13)

? 1
(cid:13)??1(cid:13)
F ??1.
2
(cid:13)

(5.9)

With these observations, proving Theorem 4.1 becomes a direct application of Lemma 5.1, 5.2, 5.3, and
Lemma 5.4. It then only requires translating U11 and u?1 back to the original parameterization using W P V
and W KQ.

6 Conclusion and future work

In this work, we investigated the dynamics of in-context learning of transformers with a single linear self-
attention layer under gradient flow on the population loss.
In particular, we analyzed the dynamics of
these transformers when trained on prompts consisting of random instances of noiseless linear models over
anisotropic Gaussian marginals. We showed that despite non-convexity, gradient flow from a suitable ran-
dom initialization converges to a global minimum of the population objective. We characterized the pre-
diction error of the trained transformer when given a new prompt that consists of a training dataset where
the responses are a nonlinear function of the inputs. We showed how the trained transformer is naturally

19

robust to shifts in the task and query distributions but is brittle to distribution shifts between the covariates
seen during training and the covariates seen at test time, matching the empirical observations on trained
transformer models of Garg et al. [Gar+22].

There are a number of natural directions for future research. First, our results hold for gradient flow on
the population loss with a particular class of random initialization schemes. It is a natural question if similar
results would hold for stochastic gradient descent with finite step sizes and for more general initializations.
Further, we restricted our attention to transformers with a single linear self-attention layer. Although this
model class is rich enough to allow for in-context learning of linear predictors, we are particularly interested
in understanding the dynamics of in-context learning in nonlinear and deep transformers.

Finally, the framework of in-context learning introduced in prior work was restricted to the setting where
the marginal distribution over the covariates (Dx) was fixed across prompts. This allows for guarantees
akin to distribution-specific PAC learning, where the trained transformer is able to achieve small prediction
error when given a test prompt consisting of linear regression data when the marginals over the covariates
are fixed. However, other learning algorithms (such as ordinary least squares) are able to achieve small
prediction error for prompts corresponding to well-specified linear regression tasks for very general classes
of distributions over the covariates. As we showed in Section 4.3, when transformers with a single linear
self-attention layer are trained on prompts where the covariate distributions are themselves sampled from
a distribution, they do not succeed on test prompts with covariate distributions sampled from the same
distribution. By contrast, we demonstrated with experiments that larger, nonlinear transformer architectures
appear to be more successful in this setting but are still sub-optimal. Developing a better understanding of
the dynamics of in-context learning when the covariate distribution varies across prompts is an intriguing
direction for future research.

Acknowledgements

We gratefully acknowledge the support of the NSF and the Simons Foundation for the Collaboration on the
Theoretical Foundations of Deep Learning through awards DMS-2031883 and #814639, and of the NSF
through grant DMS-2023505.

20

Contents

1

Introduction

2 Additional Related Work

3 Preliminaries

In-context learning .

3.1
.
3.2 Linear self-attention networks
.
3.3 Training procedure .

.

.

.

.

.

.

.

.
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
.
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

1

2

4
4
5
7

4 Main results

8
8
4.1 Convergence of gradient flow and prediction error for new tasks
4.2 Behavior of trained transformer under distribution shifts
. . . . . . . . . . . . . . . . . . . 11
4.3 Transformers trained on prompts with random covariate distributions . . . . . . . . . . . . . 13

. . . . . . . . . . . . . . .

5 Proof ideas

17
5.1 Equivalence to a quadratic optimization problem . . . . . . . . . . . . . . . . . . . . . . . 17
5.2 Dynamical system of gradient flow . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
5.3 PL inequality and global convergence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

6 Conclusion and future work

19

A Proof of Theorem 4.1

A.1 Proof of Lemma 5.1 .
A.2 Proof of Lemma 5.2 .
A.3 Proof of Lemma 5.3 .
A.4 Proof of Lemma 5.4 .

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

.
.
.
.

22
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

B Proof of Theorem 4.2

35

C Proof of Theorem 4.5

37
C.1 Dynamical system .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
C.2 Loss function and global minima . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
C.3 PL Inequality and global convergence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40

.

.

.

.

.

D Technical lemmas

E Experiment details

44

46

21

A Proof of Theorem 4.1

In this section, we prove Lemma 5.1, Lemma 5.2, Lemma 5.3 and Lemma 5.4. Theorem 4.1 is a natural
corollary of these four lemmas when we translate u?1 and U11 back to W P V and W KQ.

A.1 Proof of Lemma 5.1

For the reader�s convenience, we restate the lemma below.

Lemma 5.1. Let E? ? R(d+1)�(N +1) be an embedding matrix corresponding to a prompt of length N
and weight w? . Then the prediction (cid:98)yquery(E? ; ?) for the query covariate can be written as the output of a
quadratic function,

where the matrix H? is defined as,

(cid:98)yquery(E? ; ?) = u?H? u,

H? =

1
2

X? ?

(cid:19)

(cid:18) E? E?
?
N

? R(d+1)2�(d+1)2

, X? =

?

?

0d�d
(x?,query)?

x?,query

0

?
? ? R(d+1)�(d+1)

(5.1)

and

u = Vec(U ) ? R(d+1)2

, U =

?

?

U11

u12
(u21)? u?1

?
? ? R(d+1)�(d+1),

where U11 = W KQ
11 ? Rd�d, u12 = wP V
particular components of W P V and W KQ, defined in (3.5).

21 ? Rd�1, u21 = wKQ

21 ? Rd�1, u?1 = wP V

22 ? R correspond to

Proof. First, we decompose WP V and WKQ in the way above. From the definition, we know (cid:98)y?,query is the
right-bottom entry of fLSA(E? ), which is

(cid:16)

(cid:98)y?,query =

(u12)? u?1

(cid:17) (cid:18) E? E?
?

(cid:19)

N

?

?

U11
(u21)?

?

? x?,query.

We denote ui ? Rd+1 as the i-th column of
Then, we have

(cid:17)

(cid:16) U11
(u21)?

and xi

?,query as the i-th entry of x?,query for i ? [d].

(cid:98)y?,query =

d
(cid:88)

(cid:16)

xi
?,query

(u12)? u?1

(cid:17) (cid:18) E? E?
?

(cid:19)

N

d
(cid:88)

(cid:20)
ui

(cid:16)

tr

ui =

(u12)? u?1

(cid:17)

� xi

?,query

i=1

(cid:19)(cid:21)

(cid:18) E? E?
?
N

?

?

?

?

(cid:16)

(u12)? u?1

(cid:17)

� x?

?,query ?

(cid:18) E? E?
?
N

?

(cid:19)

?

i=1
?

?

?

= tr

?Vec

?

?

U11
(u21)?
?

?

?

tr

?Vec

?

?

(cid:20)
uu? � X? ?

tr

U11

u12
(u21)? u?1
(cid:18) E? E?
?
N

(cid:19)(cid:21)

=

=

1
2

1
2

22

?

?

?
? Vec?

?

?

?

?

U11

u12
(u21)? u?1

?

?

?

?

? �

?

0d(d+1)�d(d+1)
(cid:16) E? E?
N

x?
?,query ?

?

(cid:17)

x?,query ?

(cid:16) E? E?
N
0(d+1)�(d+1)

?

(cid:17)

?

?

?

?

H? , uu?(cid:69)
(cid:68)

.

=

Here, we use some algebraic facts about matrix vectorization, Kronecker product and trace. For reference,
we refer to [PP+08].

A.2 Proof of Lemma 5.2

For the reader�s convenience, we restate the lemma below.

Lemma 5.2. Let u = Vec (U ) := Vec

?

?

U11

u12
(u21)? u?1

?

? as in Lemma 5.1. Consider gradient flow over

with respect to u starting from an initial value satisfying Assumption 3.3. Then the dynamics of U follows

L :=

(cid:16)

E

1
2

u?H? u ? w?

? x?,query

(cid:17)2

(5.3)

d
dt
d
dt

U11(t) = ?u2

?1??U11? + u?1?2
(cid:104)

u?1(t) = ? tr

u?1??U11?(U11)? ? ?2(U11)?(cid:105)

,

(5.4)

and u12(t) = 0d, u21(t) = 0d for all t ? 0, where ? = (cid:0)1 + 1

N

(cid:1) ? + 1

N tr(?)Id ? Rd�d.

Proof. From the definition of L in (5.3) and the dynamics of gradient flow, we calculate the derivatives of
u. Here, we use the chain rule and some facts about matrix derivatives. See Lemma D.1 for reference.

(cid:16)

= ?2E

?H? , uu??H?

(cid:17)

(cid:16)

u + 2E

w?

? x?,queryH?

(cid:17)

u.

du
dt

(A.1)

Step One: Calculate the Second Term We first calculate the second term. From the definition of H? , we
have

(cid:104)

E

w?

? x?,queryH?

(cid:105)

=

1
2

d
(cid:88)

E

i=1

(cid:20)
(cid:0)xi

?,queryX?

(cid:1) ?

(cid:18)

wi
?

E? E?
?
N

(cid:19)(cid:21)

.

For ease of notation, we denote

(cid:98)?? :=

1
N

N
(cid:88)

i=1

x?,ix?
?,i.

(A.2)

Then, from the definition of E? E?

?

N , we know
?

E? E?
?
N

=

?

(cid:98)?? + 1

N x?,query � x?
w? (cid:98)??

?,query

?

? .

(cid:98)?? w?

w?

? (cid:98)?? w?

Since w? ? N(0, Id) is independent of all prompt inputs and query input, we have

1
2

d
(cid:88)

E

i=1

(cid:20)
(cid:0)xi

?,queryX?

(cid:1) ?

(cid:18) wi
?
N

(cid:18)x?,query � x?

(cid:19)(cid:19)(cid:21)

?,query 0
0

0

23

=

=

1
2

1
2

d
(cid:88)

(cid:20)

(cid:20)

E

E

(cid:0)xi

?,queryX?

(cid:34)

E

i=1
d
(cid:88)

i=1

(cid:0)xi

?,queryX?

(cid:1) ?

(cid:1) ?

(cid:18) wi
?
N
(cid:32) E (cid:2)wi

(cid:18)x?,query � x?

?,query 0
0

(cid:19)(cid:19)(cid:21) (cid:12)
(cid:12)
(cid:12)
(cid:12)

(cid:21)

x?,query

? | x?,query

(cid:18)x?,query � x?

N

0

?,query 0
0

(cid:19)(cid:33)(cid:35)

= 0.

0

(cid:3)

Therefore, we have

E

(cid:104)
w?

? x?,queryH?

(cid:105)

=

1
2

d
(cid:88)

E

i=1

?
(cid:0)xi
?

?,queryX?

?
?wi
?

?

?

(cid:1) ?

(cid:98)?? w?

(cid:98)??
? (cid:98)?? w?
w?

? (cid:98)?? w? .

?

?

?

?

?

? .

Since X? only depends on x?,query by definition, and x?,query is independent of w? and x?,i, i = 1, 2, ..., N,
we have

E

(cid:104)
w?

? x?,queryH?

(cid:105)

=

=

=

1
2

1
2

1
2

d
(cid:88)

i=1

d
(cid:88)

i=1

d
(cid:88)

i=1

?
?E (cid:0)xi

?,queryX?

?
?wi
?

(cid:1) ? E

?

?

(cid:98)?? w?

(cid:98)??
? (cid:98)?? w?
w?

? (cid:98)?? w? .

?

?

?

?

?

?

E(wi

E(wi

? )?
? )? E (cid:0)wi
? w?
?

?E(wi

? w? )

? w?

? ?w?

?

?

?

?

(cid:1)

?

?

?

?

?

?

0d�d ?i

??
i

0d�d ?i

??
i

0

?

? ?

?

?

0
?

?

? ?

?

0d�d ?i

??
i

0

? ,

where ?i denotes ?:i. Here, the second line comes from the fact that E(cid:98)?? = ?, and that w? is independent
of all prompt input and query input. The last line comes from the fact that w? ? N(0, Id). Therefore, simple
computation shows that

(cid:104)

E

w?

? x?,queryH?

(cid:105)

u =

1
2

?

?

0d(d+1)�d(d+1)
A?

?

? � u,

A

0(d+1)�(d+1)

(A.3)

where

A =

?

?
?
?
?
?
?
?

V1 + V ?
1
V2 + V ?
2

...

Vd + V ?
d

?

?
?
?
?
?
?
?

? Rd(d+1)�(d+1),

Vj =

?

?

0d�d

0

(cid:80)d

i=1 ?ij?i
0

?

? =

?

?

0d�d ??j

0

0

?
? ? R(d+1)�(d+1).

(A.4)

Step Two: Calculate the First Term Next, we compute the first term in (A.1), namely

D := 2E

(cid:16)

(cid:17)
?H? , uu??H? u

.

24

For simplicity, we denote Z? := 1

N E? E?

? . Using the definition of H? in (5.1) and Lemma D.1, we have

D = 2E

(cid:16)

(cid:17)
?H? , uu??H? u
(cid:104)

(cid:16)

E

tr

X? ? Z? Vec (U ) Vec (U )?(cid:17)

(cid:105)
(X? ? Z? ) Vec (U )

(definition of H? in (5.1) and u = Vec(U ))

(definition)

(cid:104)

E

tr

(cid:16)

Vec (Z? U X? ) Vec (U )?(cid:17)

Vec (Z? U X? )

(cid:105)

(Vec(AXB) = (B? ? A) Vec(X) in Lemma D.1)
(cid:105)
Vec (U )? � Vec (Z? U X? ) � Vec (Z? U X? )

(cid:104)

E

(property of trace operator)

?

?

E

d+1
(cid:88)

(cid:16)

i,j=1

(Z? U X? )ij Uij

(cid:17)

?

Vec (Z? U X? )

? .

=

=

=

=

1
2

1
2

1
2

1
2

Step Three: u12 and u21 Vanish We first prove that if u12 = u21 = 0d, then d
dt u21 = 0d.
If this is true, then these two blocks will be zero all the time since we assume they are zero at initial time in
Assumption 3.3. We denote Ak: and A:k as the k-th row and k-th column of matrix A, respectively.

dt u12 = 0d and d

Under the assumption that u12 = u21 = 0d, we first compute

(Z? U X? ) =

?

?

w?
?

(cid:98)?? w? u?1x?
(cid:17)
(cid:16)

?,query
w? u?1x?

(cid:98)??

?,query

(cid:16)

Written in an entry-wise manner, it will be

(cid:98)?? + 1

N x?,query � x?
(cid:16)
w?
(cid:98)??
?

(cid:17)

U11x?,query

(cid:17)

U11x?,query

?,query

?

? .

(Z? U X? )kl =

?

???????
???????

(cid:16)

(cid:17)

(cid:16)

w? u?1xl
(cid:98)??
?,query
k:
(cid:98)?? + 1
N x?,query � x?
(cid:17)
(cid:16)
w? u?1xl
w?
?
w?
?

U11x?,query

(cid:98)??

(cid:98)??

(cid:16)

(cid:17)

?,query

?,query

k, l ? [d]

(cid:17)

k:

U11x?,query

k ? [d], l = d + 1

l ? [d], k = d + 1

k = l = d + 1

.

(A.5)

We use Dij to denote the (i, j)-th entry of the (d + 1) � (d + 1) matrix �D such that Vec( �D) = D. Now

we fix a k ? [d], then

Dk,d+1 =

=

?

?

E

d+1
(cid:88)

(cid:16)

i,j=1

?

?

E

d
(cid:88)

(cid:16)

i,j=1

1
2

1
2

(Z? U X? )ij Uij

(Z? U X? )ij Uij

(cid:17)

(cid:17)

?

(Z? U X? )k,d+1

?

?

(Z? U X? )k,d+1

? +

(cid:104)(cid:16)

E

1
2

(Z? U X? )d+1,d+1 u?1

(cid:17)

(Z? U X? )k,d+1

(cid:105)

,

(A.6)

since Ui,d+1 = Ud+1,i = 0 for any i ? [d]. For the first term in the right hand side of last equation, we fix
i, j ? [d] and have
(cid:16)

(cid:17)

E

(Z? U X? )ij Uij

(Z? U X? )k,d+1

25

(cid:18)

Uij

=E

(cid:17)

(cid:16)

(cid:98)??

i:

w? u?1xj

?,query �

(cid:18)

(cid:98)?? +

1
N

x?,query � x?

?,query

(cid:19)

U11x?,query

= 0,

(cid:19)

k:

since w? is independent with all prompt input and query input, namely all x?,i for i ? [query], and w? is
mean zero. Similarly, for the second term of (A.6), we have

(cid:16)

E

=E

(Z? U X? )d+1,d+1 u?1
(cid:18)

(cid:16)

(cid:17)

(cid:18)

u?1w?
?

(cid:98)??

U11x?,query �

(cid:98)?? +

(cid:17)

(Z? U X? )k,d+1

1
N

x?,query � x?,query

U11x?,query

= 0

(cid:19)

(cid:19)

k:

since E (cid:0)w?
?
k ? [d]. Similar calculation shows that Dd+1,k = 0 for k ? [d].

(cid:1) = 0 and w? is independent of all x?,i for i ? [query]. Therefore, we have Dk,d+1 = 0 for

For k ? [d], to calculate the derivative of Uk,d+1, it suffices to further calculate the inner product of the

d(d + 1) + k th row of E (cid:2)w?

? x?,queryH?

(cid:3) and u. From (A.3), we know this is

1
2

d
(cid:88)

j=1

??

k ?jUd+1,j = 0

given that u12 = u21 = 0d. Therefore, we conclude that the derivative of Uk,d+1 will vanish given u12 =
u21 = 0d. Similarly, we conclude the same result for Ud+1,k for k ? [d]. Therefore, we know u12 = 0d and
u21 = 0d for all time t ? 0.

Step Four: Dynamics of U11 Next, we calculate the derivatives of U11 given u12 = u21 = 0d. For a fixed
pair of k, l ? [d], we have

Dkl =

E

1
2

?

?

d
(cid:88)

(cid:16)

i,j=1

(Z? U X? )ij Uij

(cid:17)

?

(Z? U X? )kl

? +

(cid:104)(cid:16)

E

1
2

(Z? U X? )d+1,d+1 u?1

(cid:17)

(Z? U X? )kl

(cid:105)

.

For fixed i, j ? [d], we have

E

(cid:104)(cid:16)

(Z? U X? )ij Uij

(cid:17)

(Z? U X? )kl

(cid:105)

= Uiju2
?1

E

(cid:104)(cid:16)

(cid:17)

(cid:98)??

w? xj

?,queryxl
(cid:105)

?,queryw?
?
(cid:16)
(cid:104)(cid:16)

(cid:17)

= Uiju2
?1

E

(cid:104)

i:
?,queryxl
xj
?,query
(cid:16)
(cid:17)
(cid:104)(cid:16)

= Uiju2

?1??,jlE

(cid:98)??

(cid:98)??

i:

(cid:98)??

i:

� E
(cid:17)

:k

(cid:105)

.

(cid:16)

(cid:17)

(cid:105)

(cid:98)??
(cid:17)

(cid:98)??

:k
(cid:105)

:k

Therefore, we sum over i, j ? [d] to get

E

1
2

?

?

d
(cid:88)

(cid:16)

i,j=1

(Z? U X? )ij Uij

(cid:17)

?

(Z? U X? )kl

? =

For the last term, we have

(cid:104)(cid:16)

E

1
2

(Z? U X? )d+1,d+1 u?1

(cid:17)

(Z? U X? )kl

(cid:105)

=

26

1
2

1
2

u2
?1

E

(cid:16)(cid:16)

(cid:98)??

(cid:17)

(cid:16)

k:

(cid:17)(cid:17)

(cid:98)??

U11?l

u2
?1

E

(cid:16)(cid:16)

(cid:98)??

(cid:17)

(cid:16)

k:

(cid:17)(cid:17)

(cid:98)??

U11?l.

So we have

Additionally, we have

Dkl = u2
?1

E

(cid:16)(cid:16)

(cid:98)??

(cid:17)

(cid:16)

k:

(cid:17)(cid:17)

(cid:98)??

U11?l.

(cid:16)

(cid:104)
E

2

w?

? x?,queryH?

(cid:17)

(cid:105)
u

(l?1)(d+1)+k

=

?

?

?

?

0d(d+1)�d(d+1)
A?

A

0(d+1)�(d+1)

?

?

? � u

?

(l?1)(d+1)+k

(definition)

(cid:16)

=

0(d+1)�d(d+1) Vl + V ?
l

(cid:17)

k:

= ??

k ?lu?1.

� U

(definition of A in (A.4))

(definition of Vi in (A.4))

Therefore, we have that for k, l ? [d], the dynamics of Ukl is

which implies

d
dt

Ukl = ?u2
?1

E

(cid:16)(cid:16)

(cid:98)??

(cid:17)

(cid:16)

k:

(cid:17)(cid:17)

(cid:98)??

U11?l + u?1??

k ?l,

U11 = ?u2
?1

E

(cid:17)2(cid:19)

(cid:18)(cid:16)

(cid:98)??

d
dt

U11? + u?1?2.

From the definition of (cid:98)?? (equation (A.2)), the independence and Gaussianity of x?,i and Lemma D.2,

we compute

(cid:18)(cid:16)

E

(cid:98)??

(cid:17)2(cid:19)

?

(cid:32)

= E

?

1
N

=

(cid:104)

N ? 1
N

N
(cid:88)

x?,ix?
?,i

(cid:33)2?
?

(definition (A.2))

i=1
(cid:16)

E

x?,1x?
?,1

(cid:17)(cid:105)2

+

(cid:16)

E

1
N

(cid:17)

x?,1x?

?,1x?,1x?
?,1
(independence between prompt input)

=

N + 1
N

?2 +

1
N

tr(?)?.

We define

? :=

N + 1
N

? +

1
N

tr(?)Id.

Then, from (A.1), we know the dynamics of U11 is

d
dt

U11 = ?u2

?1??U11? + u?1?2.

(Lemma D.2)

(A.7)

(A.8)

Step Five: Dynamics of u?1 Finally, we compute the dynamics of u?1. We have

Dd+1,d+1 =

E

1
2

?

?

d
(cid:88)

(cid:16)

i,j=1

(Z? U X? )ij Uij

(cid:17)

?

(Z? U X? )d+1,d+1

? +

(cid:104)(cid:16)

E

1
2

(Z? U X? )d+1,d+1 u?1

(cid:17)

(Z? U X? )d+1,d+1

(cid:105)

.

(A.9)

27

For the first term above, we have

?

?

E

d
(cid:88)

(cid:16)

i,j=1

(Z? U X? )ij Uij

(cid:17)

?

(Z? U X? )d+1,d+1

?

d
(cid:88)

i,j=1

d
(cid:88)

i,j=1

d
(cid:88)

=u?1

=u?1

=u?1

UijE

(cid:104)(cid:16)

(cid:17)

(cid:98)??

UijE

(cid:104)(cid:16)

(cid:17)

(cid:98)??

UijE

(cid:104)(cid:16)

(cid:17)

(cid:98)??

i,j=1
?

=u?1E tr

?

d
(cid:88)

?jUij

� w? w?
? �

(cid:17)

(cid:16)

(cid:98)??

� U11x?,queryxj

?,query

(cid:105)

(from (A.5))

(cid:17)

(cid:16)

(cid:98)??

� U11x?,queryxj

?,query

(cid:105)

(independence and distribution of w? )

(cid:17)

(cid:16)

(cid:98)??

(cid:105)

� U11?j

(independence between prompt covariates)

�

�

i:

i:

i:

(cid:17)

(cid:16)

(cid:98)??

i:

(cid:17)

(cid:16)

�

(cid:98)??

U11

?
? = u?1E tr

(cid:20)
?(U11)? (cid:16)

(cid:98)??

(cid:17)2

U11

(cid:21)

i,j=1

(cid:20)

(cid:16)

E

(cid:98)??

(cid:17)2

=u?1 tr

U11?(U11)?

(cid:21)

.

For the second term in (A.9), we have

(cid:104)(cid:16)

E

(Z? U X? )d+1,d+1 u?1

(cid:17)

(Z? U X? )d+1,d+1

(cid:105)

= u?1E

(cid:16)

(cid:104)

w?
?

(cid:98)??

(cid:17)

U11x?,queryx?

?,query(U11)? (cid:16)

(cid:17)

(cid:105)

w?

(cid:98)??

(from (A.5))
(cid:17)(cid:105)

?,query(U11)? (cid:16)
U11x?,queryx?
(cid:17)(cid:105)

(cid:98)??

= u?1E tr

= u?1E tr
(cid:20)

= u?1 tr

(cid:104)
w? w?
?
(cid:17)
(cid:104)(cid:16)

(cid:98)??

(cid:16)

(cid:17)

(cid:98)??
U11?(U11)? (cid:16)
(cid:98)??
(cid:21)
(cid:17)2

U11?(U11)?

.

(cid:16)

E

(cid:98)??

Therefore, we know

Additionally, we have

Dd+1,d+1 = u?1 tr

(cid:20)

(cid:16)

E

(cid:98)??

(cid:17)2

U11?(U11)?

(cid:21)

.

(cid:16)

(cid:104)

E

2

w?

? x?,queryH?

(cid:17)

(cid:105)
u

(d+1)2

?

?

=

?

?

0d(d+1)�d(d+1)
A?

A

0(d+1)�(d+1)

?

?

? � u

?

(d+1)2

(cid:16)

=

V1 + V ?
1

... Vd + V ?
d

0(d+1)�(d+1)

(cid:17)

� U

d+1:

(definition of A in (A.4))

(from (A.3))

=

d
(cid:88)

i,j=1

??

i ?jUji = tr

(cid:16)

?(U11)??

(cid:17)

.

28

Then, from (A.1), we have the dynamics of u?1 is

u?1 = ? tr

(cid:104)

u?1??U11?(U11)? ? ?2(U11)?(cid:105)

.

d
dt

(A.10)

A.3 Proof of Lemma 5.3

Lemma 5.3 gives the form of global minima of an equivalent loss function. First, we prove that gradient
flow on L defined in (3.8) from the initial values satisfying Assumption 3.3 is equivalent to gradient flow
on another loss function �? defined below. Then, we derive an expression for the global minima of this loss
function.

First, from the dynamics of gradient flow, we can actually recover the loss function up to a constant. We

have the following lemma.

Lemma A.1 (Loss Function). Consider gradient flow over L in (5.3) with respect to u starting from an
initial value satisfying Assumption 3.3. This is equivalent to doing gradient flow with respect to U11 and
u?1 on the loss function

�? (U11, u?1) = tr

(cid:20) 1
2

?1??U11?(U11)? ? u?1?2(U11)?
u2

(cid:21)

.

(A.11)

Proof. The proof is simply by taking gradient of the loss function in (A.11). For techniques in matrix
derivatives, see Lemma D.1. We take the gradient of �? on U11 to obtain

? �?
?U11

=

1
2

?1????U11?? +
u2

1
2

?1??U11? ? u?1?2 = u2
u2

?1??U11? ? u?1?2,

since ? and ? are commutable. We take derivatives w.r.t. u?1 to get

? �?
?u?1

= tr

(cid:104)

u?1??U11?(U11)? ? ?2(U11)?(cid:105)

.

Combining this with Lemma 5.2, we have

d
dt

U11(t) = ?

? �?
?U11

,

d
dt

u?1(t) = ?

? �?
?u?1

.

We remark that actually this is the loss function L up to some constant. This loss function �? can be

negative. But we can still compute its global minima as follows.
Corollary A.2 (Minimum of Loss Function). The loss function �? in Lemma A.1 satisfies

min
U11?Rd�d,u?1?R

�? (U11, u?1) = ?

tr (cid:2)?2??1(cid:3)

1
2

and

�? (U11, u?1) ?

min
U11?Rd�d,u?1?R

�? (U11, u?1) =

(cid:16)

(cid:13)
(cid:13)
(cid:13)?

1
2

1
2

u?1?

1

2 U11?

1

2 ? ???1(cid:17)(cid:13)
2
(cid:13)
(cid:13)
F

.

29

Proof. First, we claim that

�? (U11, u?1) =

(cid:20)

tr

? �

(cid:16)

1
2

u?1?

1

2 U11?

1

2 ? ???1(cid:17) (cid:16)

u?1?

1

2 U11?

1

2 ? ???1(cid:17)?(cid:21)

?

tr (cid:2)?2??1(cid:3) .

1
2

To calculate this, we just need to expand the terms in the brackets and notice that ? and ? are commutable:

(cid:20)

tr

? �

(cid:16)

u?1?

1

2 U11?

1

2 ? ???1(cid:17) (cid:16)

u?1?

1

2 U11?

1

2 ? ???1(cid:17)?(cid:21)

? tr (cid:2)?2??1(cid:3)

1

2 U11?(U11)??1/2 ? u?1???1?

1

2 U11?

1

2 ? u?1?

1

2 U11?

? tr[?2??1]

1

2 U11?(U11)??1/2 ? u?1???1?

1

2 U11?

1

2 ? u?1?

1

2 U11?

1

(cid:104)
2 U11?(U11)??
??
??U11?(U11)?(cid:105)
(cid:104)

(cid:105)

1
2

? u?1 tr
(cid:104)
?2U11?

? 2u?1 tr

(cid:105)

1
2

(cid:104)

????1?

1

2 U11?

1
2 ? ??

1

2 U11?

3

3

2 ??1 + ??2?2(cid:17)(cid:105)
2 ??1(cid:17)(cid:105)
2 ??1(cid:105)

3

Equations (i) and (ii) use that ? and ? commute.
2 ? ???1(cid:17) (cid:16)

Since ? ? 0 and

2 U11?

u?1?

(cid:16)

1

1

u?1?

1

2 U11?

1

2 ? ???1(cid:17)?

? 0, we know from Lemma

(cid:20)

tr

? �

(cid:16)

1
2

u?1?

1

2 U11?

1

2 ? ???1(cid:17) (cid:16)

u?1?

1

2 U11?

1

2 ? ???1(cid:17)?(cid:21)

? 0,

(cid:104)

(i)
= tr

? �

(cid:16)

u2
?1?

(cid:16)

(cid:104)
? �

= tr

u2
?1?

= u2

?1 tr

(ii)
= u2

?1 tr
= 2�? (U11, u?1) .

D.4 that

which implies

Equality holds when

�? (U11, u?1) ? ?

tr (cid:2)?2??1(cid:3) .

1
2

U11 = ??1,

u?1 = 1,

so the minimum of �? must be ? 1
from the fact that tr(A?A) = ?A?2

2 tr (cid:2)?2??1(cid:3) . The expression for �? (U11, u?1) ? min �? (U11, u?1) comes
F for any matrix A.

Lemma 5.3 is an immediate consequence of CorollaryA.2, since the loss will keep the same when we

replace (U11, u?1) by (cU11, c?1u?1) for any non-zero constant c.

A.4 Proof of Lemma 5.4

In this section, we prove that the dynamical system in Lemma 5.2 satisfies a PL inequality. Then, the PL
inequality naturally leads to the global convergence of this dynamical system. First, we prove a simple
lemma, which says the parameters in the LSA model will keep �balanced� in the whole trajectory. From the
proof of this lemma, we can understand why we assume a balanced parameter at the initial time.

Lemma A.3 (Balanced Parameters). Consider gradient flow over L in (5.3) with respect to u starting from
an initial value satisfying Assumption 3.3. For any t ? 0, it holds that

u2
?1 = tr

U11(U11)?(cid:105)
(cid:104)

.

(A.12)

30

Proof. From Lemma 5.2, we multiply the first equation in (5.4) by (U11)? from the right to get

(cid:18) d
dt

(cid:19)

U11(t)

(U11(t))? = ?u2

?1??U11?(U11)? + u?1?2(U11)?.

Also we multiply the second equation in Lemma 5.2 by u?1 to obtain

(cid:18) d
dt

(cid:19)

u?1(t)

u?1(t) = tr

(cid:104)
?u2

?1??U11?(U11)? + u?1?2(U11)?(cid:105)

.

Therefore, we have

(cid:20)(cid:18) d
dt

tr

(cid:19)

U11(t)

(U11(t))?

(cid:21)

=

(cid:18) d
dt

(cid:19)

u?1(t)

u?1(t).

Taking the transpose of the equation above and adding to itself gives

(cid:104)

U11(t)(U11(t))?(cid:105)

tr

=

d
dt

d
dt

(cid:0)u?1(t)2(cid:1) .

Notice that from Assumption 3.3, we know that at t = 0,
??????(cid:105)

u?1(0)2 = ?2 = ?2 tr

(cid:104)

= tr

U11(0)(U11(0))?(cid:105)
(cid:104)

.

So for any time t ? 0, the equation holds.

In order to prove the PL inequality, we first prove an important property which says the trajectories of
u?1(t) stay away from saddle point at origin. First, we prove that u?1(t) will stay positive along the whole
trajectory.

Lemma A.4. Consider gradient flow over L in (5.3) with respect to u starting from an initial value satisfying
Assumption 3.3. If the initial scale satisfies

then, for any t ? 0, it holds that

0 < ? <

(cid:115)

?

2
d ???op

,

u?1 > 0.

(A.13)

Proof. From Lemma A.1, we are actually doing gradient flow on the loss �?. The loss function is non-
increasing, because

(cid:42)

d�?
dt

=

dU11
dt

,

? �?
?U11

(cid:43)

(cid:42)

+

du?1
dt

,

? �?
?u?1

(cid:43)

= ?

(cid:13)
(cid:13)
(cid:13)
(cid:13)

dU11
dt

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
F

?

(cid:13)
(cid:13)
(cid:13)
(cid:13)

du?1
dt

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
F

? 0.

We notice that when u?1 = 0, the loss function �? = 0. Therefore, as long as �?(U11(0), u?1(0)) < 0, then for
any time, u?1 will be non-zero. Further, since u?1(0) > 0 and the trajectory of u?1(t) must be continuous,
we know u?1(t) > 0 for any t ? 0.

31

Then, it suffices to prove when 0 < ? <

, it holds that �?(U11(0), u?1(0)) < 0. From As-

(cid:113) 2?

d???op

sumption 3.3, we can calculate the loss function at the initial time:

�?(U11(0), u?1(0)) =

?4
2

(cid:104)

?????????(cid:105)

tr

? ?2 tr

?2???(cid:105)
(cid:104)

.

From the property of trace, we know

(cid:104)

?2???(cid:105)

tr

??????(cid:105)
(cid:104)

= tr

= ????2

F .

From Von-Neumann�s trace inequality (Lemma D.3) and the fact that (cid:13)
(cid:13)????????(cid:13)
(cid:13)
(cid:13)F

?????????(cid:105)

d ????2
F

�???op ?

(cid:13)???(cid:13)
(cid:13)???(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)F

(cid:13)
(cid:13)

?

?

tr

?

d

(cid:104)

(cid:13)F = 1, we know

?

???op =

d ????2

F ???op .

Therefore, we have

�?(U11(0), u?1(0)) ?

=

?

d?4
2
?2
2

????2

F ???op ? ?2 ????2
(cid:104)?

(cid:105)

F

d?2 ???op ? 2

.

????2
F

From Assumption 3.3, we know ????F ?= 0. From (A.7), we know ???op > 0. Therefore, when

we have

0 < ? <

(cid:115)

?

2
d ???op

,

�?(U11(0), u?1(0)) < 0.

From the lemma above, we can actually further prove that the u?1(t) can be lower bounded by a positive
constant for any t ? 0. This will be a critical property to prove the PL inequality. We have the following
lemma.

Lemma A.5. Consider gradient flow over L in (5.3) with respect to u starting from an initial value satisfying
Assumption 3.3 with initial scale 0 < ? <

. For any t ? 0, it holds that

(cid:113) 2?

d???op

u?1 ?

(cid:115)

?2
d ???2
op

?
2

????2
F

?

(cid:104)

2 ?

d?2 ???op

(cid:105)

> 0.

(A.14)

Proof. We prove by contradiction. Suppose the claim does not hold. From Lemma A.3, we know u2
tr (cid:2)U11(U11)?(cid:3) = ?U11?2
function:

?1 =
F . From Lemma A.4, we know u?1 = ?U11?F . Recall the definition of loss

�?(U11, u?1) = tr

(cid:20) 1
2

?1??U11?(U11)? ? u?1?2(U11)?
u2

(cid:21)

.

32

Since ? ? 0, ? ? 0, and they commute, we know from Lemma D.4 that ?? ? 0. Again, since
?1??U11?(U11)?(cid:3) ? 0.
U11?(U11)? =
So

? 0, from Lemma D.4 we have tr (cid:2) 1

U11?

U11?

2 u2

(cid:17) (cid:16)

(cid:17)?

(cid:16)

1
2

1
2

�?(U11, u?1) ? ? tr

u?1?2(U11)?(cid:105)
(cid:104)

.

From Von-Neumann�s trace inequality, we know for any t ? 0,
u?1?2(U11)?(cid:105)

du?1

(cid:13)?2(cid:13)
(cid:13)

? ?

? tr

?

(cid:104)

(cid:13)op ?U11?F = ?

du2

?1 ???2

op .

?

Therefore, under our assumption that the claim does not hold, we have

�?(U11, u?1) ? ?

?

du2

?1 ???2

op > ?

????2
F

?

(cid:104)

2 ?

(cid:105)

d?2 ???op

?2
2

? �?(U11(0), u?1(0)).

Here, the last inequality comes from the proof of Lemma A.4. This contradicts the non-increasing property
of the loss function in gradient flow.

Finally, let�s prove the PL inequality and further, the global convergence of gradent flow on the loss

function �?. We recall the stated lemma from the main text.

Lemma 5.4. Suppose the initialization of gradient flow satisfies Assumption 3.3 with initialization scale
satisfying ?2 <

for ? = (1 + 1

2?

N )? + tr(?)

N Id. If we define

d???op

� :=

?

?2

d ???2

op tr (??1??1) tr (??1)

????2
F

?

(cid:104)
2 ?

d?2 ???op

(cid:105)

> 0,

(5.7)

then gradient flow on �? with respect to U11 and u?1 satisfies, for any t ? 0,

(cid:13)
(cid:13)
2
(cid:13)?�?(U11(t), u?1(t))
(cid:13)
(cid:13)
(cid:13)
2

:=

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

? �?
?U11

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
F

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

? �?
?u?1

(cid:12)
2
(cid:12)
(cid:12)
(cid:12)
(cid:12)

+

? �

(cid:18)

�?(U11(t), u?1(t)) ?

min
U11?Rd�d,u?1?R

(cid:19)

�?(U11, u?1)

.

(5.8)
Moreover, gradient flow converges to the global minimum of �?, and U11 and u?1 converge to the following,

lim
t??

u?1(t) = (cid:13)

1
2

(cid:13)??1(cid:13)
F and lim
(cid:13)
t??

U11(t) = (cid:13)

? 1
(cid:13)??1(cid:13)
F ??1.
2
(cid:13)

(5.9)

Proof. From the definition and Lemma A.5, we have

???(U11, u?1)?2

2 ?

(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
F

??
?U11
(cid:13)
(cid:13)
(cid:13)??
?2
d ???2
op

1
2

?

?1

= u2

?

2

= (cid:13)

(cid:13)u2

(cid:16)

u?1?

?1??U11? ? u?1?2(cid:13)
2
(cid:13)
F
2 ? ???1(cid:17)
?

2 U11?

?

1

1

????2
F

(cid:104)
2 ?

d?2 ???op

1
2

(cid:13)
2
(cid:13)
(cid:13)
F
(cid:105) (cid:13)
(cid:13)
(cid:13)??

(cid:16)

1
2

u?1?

1

2 U11?

1

2 ? ???1(cid:17)

1
2

?

(cid:13)
2
(cid:13)
(cid:13)
F

.

(A.15)

33

To see why the second line is true, recall that u?1 ? R and ? and ? commute. The last line comes from the
lower bound of u?1 in Lemma A.5. From Corollary A.2, we know

? ?

min
U11?Rd�d,u?1?R

?(U11, u?1) =

=

1
2
1
2

(cid:20)

tr

?

(cid:16)

u?1?

1

2 U11?

1

2 ? ???1(cid:17) (cid:16)

u?1?

1

2 U11?

1

2 ? ???1(cid:17)?(cid:21)

(cid:16)

(cid:13)
(cid:13)
(cid:13)?

1
2

u?1?

1

2 U11?

1

2 ? ???1(cid:17)(cid:13)
2
(cid:13)
(cid:13)
F

.

Therefore, we know that

? ?

min
U11?Rd�d,u?1?R

?(U11, u?1) ?

=

(cid:16)

(cid:16)

(cid:13)
(cid:13)
(cid:13)??
(cid:13)
(cid:13)
(cid:13)??

1
2

1
2

1
2
1
2

u?1?

1

2 U11?

1

2 ? ???1(cid:17)

1
2

?

u?1?

1

2 U11?

1

2 ? ???1(cid:17)

1
2

?

(cid:13)
2
(cid:13)
(cid:13)
F
(cid:13)
2
(cid:13)
(cid:13)
F

2

�

(cid:13)
2
(cid:13)
(cid:13)
F

2 ?? 1

(cid:13)
(cid:13)?? 1
(cid:13)

(cid:13)
(cid:13)
2
(cid:13)?? 1
(cid:13)
(cid:13)
(cid:13)
F
� tr (cid:0)??1??1(cid:1) tr (cid:0)??1(cid:1)

2

(A.16)

We compare (A.15) and (A.16) to obtain that in order to make the PL condition hold, one needs to let

� :=

?

?2

d ???2

op tr (??1??1) tr (??1)

????2
F

?

(cid:104)
2 ?

d?2 ???op

(cid:105)

> 0.

Once we set this �, we get the PL inequality. The � is positive due to the assumption for ? in the lemma.

From the dynamics of gradient flow and the PL condition, we know

(cid:18)

�? ?

d
dt

min
U11?Rd�d,u?1?R

(cid:19)

�?(U11, u?1)

(cid:42)

=

(cid:43)

(cid:42)

+

,

? �?
?U11

du?1
dt

,

? �?
?u?1

(cid:43)

= ?

(cid:13)
(cid:13)
(cid:13)
(cid:13)

dU11
dt

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
F

?

(cid:12)
(cid:12)
(cid:12)
(cid:12)

du?1
dt

(cid:12)
2
(cid:12)
(cid:12)
(cid:12)

dU11
dt
(cid:18)

? ?�

�? ?

min
U11?Rd�d,u?1?R

(cid:19)

�?(U11, u?1)

.

Therefore, we have when t ? ?,

0 ? �??

min
U11?Rd�d,u?1?R

�?(U11, u?1) ? exp (?�t)

(cid:20)
�?(U11(0), u?1(0)) ?

min
U11?Rd�d,u?1?R

(cid:21)
�?(U11, u?1)

? 0,

which implies

(cid:20)

�? ?

lim
t??

min
U11?Rd�d,u?1?R

�?(U11, u?1)

(cid:21)

= 0.

From Corollary A.2, we know this is

(cid:16)

(cid:13)
(cid:13)
(cid:13)?

1
2

u?1?

1

2 U11?

1

2 ? ???1(cid:17)(cid:13)
2
(cid:13)
(cid:13)
F

? 0.

Since ? and ? are non-singular and positive definite, and they commute, we know
(cid:13)
(cid:13)?? 1
(cid:13)

2 ? ???1(cid:17)(cid:13)
2
(cid:13)
(cid:13)
F
This implies u?1U11 ? ??1 ? 0d�d entry-wise. Since u?1 = ?U11?F , we know

(cid:13)u?1U11 ? ??1(cid:13)
(cid:13)
2
F ?
(cid:13)

(cid:13)
(cid:13)?? 1
(cid:13)

2 ?? 1

2 U11?

(cid:13)
(cid:13)
(cid:13)?

u?1?

(cid:13)
2
(cid:13)
(cid:13)
F

(cid:16)

1
2

2

1

2

1

(cid:13)
2
(cid:13)
(cid:13)
F

? 0.

?1 = ?u?1U11?F ? (cid:13)
u2

(cid:13)??1(cid:13)

(cid:13)F .

Therefore, we know

lim
t??

u?1(t) = (cid:13)

1
2

(cid:13)??1(cid:13)
F and lim
(cid:13)
t??

U11(t) = (cid:13)

? 1
(cid:13)??1(cid:13)
F ??1.
2
(cid:13)

34

B Proof of Theorem 4.2

In this section, we prove Theorem 4.2, which characterizes the excess risk of the prediction of a trained LSA
layer with respect to the risk of best linear predictor, on a new task which is possibly non-linear. First, we
restate the theorem.

Theorem 4.2. Let D be a distribution over (x, y) ? Rd � R, whose marginal distribution on x is Dx =
N(0, ?). Assume ED[y], ED[xy], ED[y2xx?] exist and are finite. Assume the test prompt is of the form
P = (x1, y1, . . . , xM , yM , xquery), where (xi, yi), (xquery, yquery) i.i.d.? D. Let f ?
LSA be the LSA model with
parameters W P V
in (4.1), and (cid:98)yquery is the prediction for xquery given the prompt. If we define

and W KQ

?

?

a := ??1E(x,y)?D [xy] ,

? := E(x,y)?D

(cid:104)(cid:0)xy ? E (xy) (cid:1)(cid:0)xy ? E (xy) (cid:1)?(cid:105)

,

(4.5)

then, for ? = ? + 1

N ? + 1

N tr(?)Id. we have,

E ((cid:98)yquery ? yquery)2 = min
w?Rd
(cid:124)

E (?w, xquery? ? yquery)2
(cid:125)

(cid:123)(cid:122)
Error of best linear predictor
tr (cid:2)???2?(cid:3) +

(cid:104)
?a?2

1
N 2

+

1
M

??2?3 + 2 tr(?) ?a?2

??2?2 + tr(?)2 ?a?2

??2?

(cid:105)

,

(4.6)

where the expectation is over (xi, yi), (xquery, yquery) i.i.d.? D.

Proof. Unless otherwise specified, we denote E as the expectation over (xi, yi), (xquery, yquery) i.i.d.?
D. Since when (x, y) ? D, we assume E[x], E[y], E[xy], E[xx?], E[y2xx?] exist, we know that
E (?w, xquery? ? yquery)2 exists for each w ? Rd. We denote

a := arg min

w?Rd

E (?w, xquery? ? yquery)2

as the weight of the best linear approximator. Actually, if we denote the function inside the minimum above
as R(w), we can write it as

R(w) = w??w ? 2E

(cid:16)

yquery � x?

query

(cid:17)

w + Ey2

query.

Since the Hessian matrix
convex and hence, the global minimum can be achieved at the unique first-order stationary point. This is

?w?w? R(w) is ?, which is positive definitive, we know that this function is strictly

?2

a = ??1E (yquery � xquery) .

We also define a similar vector for ease of computation:

b = ??1E (yquery � xquery) .

Therefore, we can decompose the error as

E ((cid:98)yquery ? yquery)2 = E (?a, xquery? ? yquery)2
(cid:124)
(cid:125)

(cid:123)(cid:122)
I

+ E ((cid:98)yquery ? ?b, xquery?)2
(cid:125)
(cid:123)(cid:122)
II

(cid:124)

35

(B.1)

(B.2)

+ E (?b, xquery? ? ?a, xquery?)2
(cid:123)(cid:122)
(cid:125)
III

(cid:124)

+ 2E ((cid:98)yquery ? ?b, xquery?) (?b, xquery? ? ?a, xquery?)
(cid:123)(cid:122)
(cid:125)
V

(cid:124)

(cid:124)

+ 2E ((cid:98)yquery ? ?b, xquery?) (?a, xquery? ? yquery)
(cid:125)
(cid:123)(cid:122)
IV
+ 2E (?b, xquery? ? ?a, xquery?) (?a, xquery? ? yquery)
(cid:123)(cid:122)
(cid:125)
VI

(cid:124)

The term I is the first term on the right hand side of (4.6). So it suffices to calculate II to VI.

First, from the tower property of conditional expectation, we have

V = 2E

= 2E

(cid:20)

E

(cid:20)

E

(cid:18)

((cid:98)yquery ? ?b, xquery?) (?b, xquery? ? ?a, xquery?)

(cid:18)

(cid:12)
(cid:12)
(cid:98)yquery ? ?b, xquery?
(cid:12)
(cid:12)

(cid:19)

xquery

(?b, xquery? ? ?a, xquery?)

= 0,

(cid:19)(cid:21)

(cid:12)
(cid:12)
xquery
(cid:12)
(cid:12)

(cid:21)

since

E

(cid:18)

(cid:12)
(cid:12)
(cid:98)yquery ? ?b, xquery?
(cid:12)
(cid:12)

(cid:19)

xquery

=

(cid:32)

E 1
M

M
(cid:88)

i=1

(cid:33)?

yi??1xi ? b

xquery = 0.

Similarly, for IV, we have

IV = 2E ((cid:98)yquery ? ?b, xquery?) (?a, xquery? ? yquery)

(cid:20)

(cid:18)

E

((cid:98)yquery ? ?b, xquery?) (?a, xquery? ? yquery)

(cid:20)

E

(cid:18)

(cid:12)
(cid:12)
(cid:98)yquery ? ?b, xquery?
(cid:12)
(cid:12)

(cid:19)

xquery, yquery

(?a, xquery? ? yquery)

(cid:12)
(cid:12)
xquery, yquery
(cid:12)
(cid:12)

(cid:19)(cid:21)

(cid:21)

= 2E

= 2E

= 0.

For VI, we have

(cid:104)

VI = 2E tr
(cid:105)
(cid:104)
(b ? a)a??

(b ? a) (?a, xquery? ? yquery) x?
(cid:16)

(b ? a)E

= 2 tr

? 2 tr

(cid:104)

(cid:105)

query

yqueryx?

query

(cid:17)(cid:105)

= 0,

where the last line comes from the definition of a. Therefore, all cross terms vanish and it suffices to consider
II and III.

For II, from the definition we have

II

(cid:32)

=E

1
M

M
(cid:88)

i=1

yixi ? E (yquery � xquery)

??1xqueryx?

query??1

(cid:33)?

(cid:32)

1
M

M
(cid:88)

i=1

yixi ? E (yquery � xquery)

(cid:33)

=E tr

(cid:32)

1
M

M
(cid:88)

i=1

yixi ? E (yquery � xquery)

(cid:33) (cid:32)

1
M

M
(cid:88)

yixi ? E (yquery � xquery)

??2?

(cid:33)?

i=1
(property of trace and the fact that ? and ? commute)

36

M
(cid:88)

E tr

(cid:110)
(yixi ? E (yquery � xquery)) (yjxj ? E (yquery � xquery))? ??2?

(cid:111)

i,j=1
(cid:110)
(y1x1 ? E (yquery � xquery)) (y1x1 ? E (yquery � xquery))? ??2?

E tr

(cid:111)

tr (cid:2)???2?(cid:3) .

(all cross terms vanish due to the independence of xi)

=

1
M 2

=

=

1
M

1
M

The last line comes from the definition of ?.

For III, we have

III = E(b ? a)?xqueryx?

query(b ? a) = a??(??1 ? ??1)?(??1 ? ??1)?a
??2?3aa?(cid:105)

(cid:104)(cid:0)I ? ???1(cid:1)2
1
N 2 tr
(cid:104)
1
tr(??2?3aa?) + 2 tr(?) tr(??2?2aa?) + tr(?)2 tr(??2?aa?)
N 2

(cid:104)(cid:0)Id + tr(?)??1(cid:1)2

??2?3aa?(cid:105)

= tr

=

=

(cid:105)

.

(property of trace and the fact that ? and ? commute)

Combining all terms above, we conclude.

C Proof of Theorem 4.5

The proof of Theorem 4.5 is very similar to that of Theorem 4.1. The first step is to explicitly write out
the dynamical system. In order to do so, we notice that the Lemma 5.1 does not depend on the training
data and data-generaing distribution and hence, it still holds in the case of a random covariance matrix.
Therefore, we know when we input the embedding matrix E? to the linear self-attention layer with parameter
? = (W KQ, W P V ), the prediction will be

(cid:98)yquery(E? ; ?) = u?H? u,

where the matrix H? is defined as,

H? =

1
2

X? ?

(cid:19)

(cid:18) E? E?
?
N

? R(d+1)2�(d+1)2

, X? =

?

?

0d�d
(x?,query)?

?
? ? R(d+1)�(d+1)

x?,query

0

and

u = Vec(U ) ? R(d+1)2

, U =

?

?

U11

u12
(u21)? u?1

?
? ? R(d+1)�(d+1),

where U11 = W KQ
11 ? Rd�d, u12 = wP V
particular components of W P V and W KQ, defined in (3.5).

21 ? Rd�1, u21 = wKQ

21 ? Rd�1, u?1 = wP V

22 ? R correspond to

37

C.1 Dynamical system

The next lemma gives the dynamical system when the covariance matrices in the prompts are i.i.d. sampled
from some distribution. Notice that in the lemma below, we do not assume ?? are almost surely diagonal.
The case when the covariance matrices are diagonal can be viewed as a special case of the following lemma.

Lemma C.1. Consider gradient flow on (4.10) with respect to u starting from an initial value that satisfies
Assumption 3.3. We assume the covariance matrices ?? are sampled from some distribution with finite third

moment and ?? are positive definite almost surely. We denote u = Vec (U ) := Vec

define

(cid:18)

?? =

1 +

(cid:19)

1
N

?? +

1
N

tr(?? )Id ? Rd�d.

Then the dynamics of U follows

?

?

U11

u12
(u21)? u?1

?

? and

d
dt
d
dt

U11(t) = ?u2
?1

u?1(t) = ?u?1 tr E

E [?? ?? U11?? ] + u?1E (cid:2)?2
?? ?? U11?? (U11)?(cid:105)
(cid:104)

?

(cid:3)

+ tr

(cid:16)

E (cid:2)?2
?

(cid:3) (U11)?(cid:17)

,

(C.1)

and u12(t) = 0d, u21(t) = 0d for all t ? 0.

Proof. This lemma is a natural corollary of Lemma 5.2. Notice that Lemma 5.2 holds for any fixed positive
definite ?? . So when ?? is random, if we condition on ?? , the dynamical system will be

d
dt
d
dt

U11(t) = ?u2

u?1(t) = ?u?1 tr

?1 [?? ?? U11?? ] + u?1

(cid:2)?2
?
?? ?? U11?? (U11)?(cid:105)
(cid:104)

(cid:3)

(cid:16)(cid:2)?2

?

(cid:3) (U11)?(cid:17)

,

(C.2)

+ tr

and u12(t) = 0d, u21(t) = 0d for all t ? 0. Then, we conclude by simply taking expectation over ?? .

The lemma above gives the dynamical system with general random covariance matrix. When ?? are
diagonal almost surely, we can actually simplify the dynamical system above. In this case, we have the
following corollary.

Corollary C.2. Under the assumptions of Lemma C.1, we further assume the covariance matrix ?? to be
diagonal almost surely. We denote uij(t) ? R as the (i, j)-th entry of U11(t), and further denote

?

?

?i = E

N + 1
N

?3
?,i +

1
N

?2
?,i �

?

??,j

? ,

d
(cid:88)

j=1

?i = E (cid:2)?2

?,i

(cid:3) ,

?ij = E

(cid:34)

N + 1
N

?2
?,i??,j +

1
N

??,i??,j �

(cid:35)

??,k

d
(cid:88)

k=1

38

(C.3)

for i, j ? [d], where the expectation is over the distribution of ?? . Then, the dynamical system (C.1) is
equivalent to

d
dt
d
dt

uii(t) = ??iu2

?1uii + ?iu?1 ?i ? [d],

uij(t) = ??iju2

?1uij ?i ?= j ? [d],

(C.4)

d
dt

u?1(t) = ?

d
(cid:88)

i=1

(cid:2)?iu?1u2

ii

(cid:3) ?

(cid:88)

i?=j

?iju?1u2

ij +

d
(cid:88)

i=1

[?iuii] .

Proof. This is directly obtained by rewriting the equation for each entry of U11 and recalling the assumption
that ?? (and hence ?? ) is diagonal almost surely.

C.2 Loss function and global minima

As in the proof of Theorem 4.1, we can actually recover the loss function in the random covariance case, up
to a constant.

Lemma C.3. The differential equations in (C.4) are equivalent to gradient flow on the loss function

?rdm(U11, u?1) = E tr

(cid:20) 1
2

?1?? ?? U11?? (U11)? ? u?1?2
u2

? (U11)?

(cid:21)

=

1
2

d
(cid:88)

i=1

(cid:2)?iu2

?1u2
ii

(cid:3) +

1
2

(cid:88)

i?=j

?iju2

?1u2

ij ?

d
(cid:88)

i=1

[?iuiiu?1]

(C.5)

with respect to uij?i, j ? [d] and u?1, from an initial value that satisfies Assumption 3.3.

Proof. This can be verified by simply taking gradient of ?rdm to show that

d
dt

uii = ?

??rdm
?uii

?i ? [d],

d
dt

uij = ?

??rdm
?uij

?i ?= j ? [d],

d
dt

u?1 = ?

??rdm
?u?1

.

Next, we solve for the minimum of ?rdm and give the expression for all global minima.

Lemma C.4. Let ?rdm be the loss function in (C.5). We denote

min ?rdm :=

min
U11?Rd�d,u?1?R

?rdm (U11, u?1) .

Then, we have

and

min ?rdm = ?

1
2

d
(cid:88)

i=1

?2
i
?i

(C.6)

?rdm(U11, u?1) ? min ?rdm =

(cid:18)

?i

uiiu?1 ?

1
2

d
(cid:88)

i=1

(cid:19)2

?i
?i

+

1
2

(cid:88)

i?=j

?iju2

?1u2
ij.

(C.7)

Moreover, denoting uij as the (i, j)-entry of U11, all global minima of ?rdm satisfy

u?1 � uij = I(i = j) �

?i
?i

.

39

(C.8)

Proof. From the definition of ?rdm, we have

?rdm =

(cid:18)

?i

uiiu?1 ?

1
2

d
(cid:88)

i=1

(cid:19)2

?i
?i

+

1
2

(cid:88)

i?=j

?iju2

?1u2

ij ?

1
2

d
(cid:88)

i=1

?2
i
?i

? ?

1
2

d
(cid:88)

i=1

?2
i
?i

.

The equation holds when uij = 0 for i ?= j ? [d] and u?1uii = ?i
?i
simply letting u?1 = 1 and uii = ?i
?i
for any constant c ?= 0, we can also achieve this global minimum.

for each i ? [d]. This can be achieved by
for i ? [d]. Of course, when we replace (u?1, uii) with (cu?1, c?1uii)

C.3 PL Inequality and global convergence

Finally, to end the proof, we prove a Polyak-?ojasiewicz Inequality on the loss function ?rdm, and then prove
global convergence. Before that, let�s first prove the balanced condition of parameters will hold during the
whole trajectory.

Lemma C.5 (Balanced condition). Under the assumptions of Lemma C.1, for any t ? 0, it holds that

u2
?1 = tr

U11(U11)?(cid:105)
(cid:104)

.

(C.9)

Proof. The proof is similar to the proof of Lemma A.3. From Lemma 5.2, we multiply the first equation in
(C.1) by (U11)? from the right to get

(cid:21)
U11(t)

(cid:20) d
dt

(U11)? = ?u2
?1

E

?? ?? U11?? (U11)?(cid:105)
(cid:104)

+ u?1E

(cid:104)

? (U11)?(cid:105)
?2

.

Also we multiply the second equation in Lemma C.1 by u?1 to obtain

(cid:18) d
dt

(cid:19)

u?1(t)

u?1(t) = ?u2

?1 tr E

(cid:104)

?? ?? U11?? (U11)?(cid:105)

+ u?1 tr

(cid:16)

E (cid:2)?2
?

(cid:3) (U11)?(cid:17)

,

Therefore, we have

(cid:20)(cid:18) d
dt

tr

(cid:19)

U11(t)

(U11(t))?

(cid:21)

=

(cid:18) d
dt

(cid:19)

u?1(t)

u?1(t).

Taking the transpose of the equation above and adding to itself gives

(cid:104)

U11(t)(U11(t))?(cid:105)

tr

=

d
dt

d
dt

(cid:0)u?1(t)2(cid:1) .

Notice that from Assumption 3.3, we know that

u?1(0)2 = ?2 = ?2 tr

(cid:104)

??????(cid:105)

= tr

(cid:104)
U11(0)(U11(0))?(cid:105)

.

So for any time t ? 0, the equation holds.

Next, similar to the proof of Theorem 4.1, we prove that, as long as the initial scale is small enough, u?1
will be positive along the whole trajectory and can be lower bounded by a positive constant, which implies
that the trajectories will be away from the saddle point at the origin.

40

Lemma C.6. We do gradient flow on ?rdm with respect to ui,j (?i, j ? [d]) and u?1. Suppose the initializa-
tion satisfies Assumption 3.3 with initial scale

0 < ? <

(cid:118)
(cid:117)
(cid:117)
(cid:116)

2 ?E?? ??2
F
(cid:104)
E ??? ?op ??? ?2

F

(cid:105) ,

?

d

then for any t ? 0, it holds that

u?1(t) > 0.

Proof. From the dynamics of gradient flow, we know the loss function ?rdm is non-increasing:

d?rdm
dt

=

d
(cid:88)

i,j=1

??rdm
?uij

�

duij
dt

+

??rdm
?u?1

�

du?1
dt

= ?

d
(cid:88)

i,j=1

(cid:21)2

(cid:20) ??rdm
?uij

?

(cid:20) ??rdm
?u?1

(cid:21)2

? 0.

Since we assume U11(0) = ???, we know the loss function at t = 0 is

?rdm(U11(0), u?1(0)) = E tr

(cid:20) ?4
2

?? ?? ????? ??? ? ?2?2

? ???

(cid:21)

.

(C.10)

(C.11)

From the property of trace, we know

E tr

(cid:104)
?2?2

? ???(cid:105)

From Von-Neumann�s trace inequality and the assumption that (cid:13)

(cid:13)F = 1, we know

E tr

(cid:20) ?4
2

?? ?? ????? ???

?

?4
2
?
?4

(cid:21)

?

?

d

E ??? ?op
(cid:13)???(cid:13)
d (cid:13)
2
(cid:13)
F
2

(cid:104)
E ??? ?op ??? ?2

F

(cid:105)

=

?

?4
2

d

(cid:104)

E ??? ?op ??? ?2

F

(cid:105)

.

= ?2 ?E?? ??2

F .
(cid:13)???(cid:13)
(cid:13)?? ????? ???(cid:13)
(cid:13)
(cid:13)F

(cid:13)
(cid:13)

From the assumptions on ? and ?? we know E?? ? ?= 0d�d and E ??? ?op ??? ?2
F > 0. Therefore, com-
paring the two displays above, we know when (C.10) holds, we must have ?rdm(0) < 0. So from the non-
increasing property of the loss function, we know ?rdm(t) < 0 for any time t ? 0. Notice that when u?1 = 0,
the loss function is also zero, which suggests that u?1(t) ?= 0 for any time t ? 0. Since u?1(0) > 0 and the
trajectory of u?1 must be continuous, we know that it stays positive at all times.

Lemma C.7. We do gradient flow on ?rdm with respect to ui,j (?i, j ? [d]) and u?1. Suppose the initializa-
tion satisfies Assumption 3.3 and the initial scale satisfies (C.10). Then, for any t ? 0, it holds that

u?1(t) ?

(cid:115)

2

?

?2
d ?E?2

? ?op

(cid:104)

2 ?E?? ??2

F ?

?

d?2 (cid:104)

E ??? ?op ??? ?2

F

(cid:105)(cid:105)

> 0.

(C.12)

Proof. From the dynamics of gradient flow, we know ?rdm is non-increasing (see the proof of Lemma C.6).
Recall the definition of the loss function:

?rdm(U11, u?1) = E tr

(cid:20) 1
2

?1?? ?? U11?? (U11)? ? u?1?2
u2

? (U11)?

(cid:21)

.

41

Since ?? commutes with ?? and they are both positive definite almost surely, we know that ?? ?? ? 0d�d
almost surely from Lemma D.1. Again, since U11?? (U11)? ? 0d�d almost surely, from Lemma D.1 we
have tr (cid:2) 1

2 u2

?1?? ?? U11?? (U11)?(cid:3) ? 0 almost surely. Therefore, we have
(cid:104)
u?1?2

?rdm(U11, u?1) ? ?E tr

? (U11)?(cid:105)

(cid:104)
u?1

= ? tr

(cid:0)E?2
?

(cid:1) (U11)?(cid:105)

.

From Von Neumann�s trace inequality (Lemma D.3) and the fact that u?1(t) > 0 for any t ? 0 (Lemma
(cid:13)
(cid:13)E?2
?1 =
C.6), we know ?rdm(U11(t), u?1(t)) ? ?
?
tr(U11(U11)?) = ?U11?2
F . Since u?1(t) > 0 for any time, we know actually u?1(t) = ?U11(t)?F . So we
have

(cid:13)
(cid:13)op ?U11?F . From Lemma C.5, we know u2

du?1

?

?rdm(U11(t), u?1(t)) ? ?

?

du?1(t)2 (cid:13)

(cid:13)E?2
?

(cid:13)
(cid:13)op .

From the proof of Lemma C.6, we know

?rdm(U11(t), u?1(t)) ? ?rdm(U11(0), u?1(0)) ?

Combine the two preceding displays above, we have

?

?4
2

d

(cid:104)
E ??? ?op ??? ?2

F

(cid:105)

? ?2 ?E?? ??2

F .

u?1(t) ?

(cid:115)

2

?

?2
d ?E?2

? ?op

(cid:104)

2 ?E?? ??2

F ?

?

d?2 (cid:104)

E ??? ?op ??? ?2

F

(cid:105)(cid:105)

> 0.

The last inequality comes from Lemma C.6.

Finally, we prove the PL Inequality, which naturally leads to the global convergence.

Lemma C.8. We do gradient flow on ?rdm with respect to ui,j (?i, j ? [d]) and u?1. Suppose the initializa-
tion satisfies Assumption 3.3 and the initial scale satisfies (C.10). If we denote

? = min {?i, i ? [d]; ?ij, i ?= j ? [d]}

and

? :=

?

2

? � ?2
d ?E?2

? ?op

(cid:104)
2 ?E?? ??2

F ?

?

then for any t ? 0, it holds that

d?2 (cid:104)

E ??? ?op ??? ?2

F

(cid:105)(cid:105)

> 0,

(C.13)

???rdm(U11, u?1)?2

2 :=

d
(cid:88)

i,j=1

(cid:12)
(cid:12)
(cid:12)
(cid:12)

??rdm
?uij

(cid:12)
2
(cid:12)
(cid:12)
(cid:12)

+

(cid:12)
(cid:12)
(cid:12)
(cid:12)

??rdm
?u?1

(cid:12)
2
(cid:12)
(cid:12)
(cid:12)

? ? (?rdm ? min ?rdm) .

(C.14)

Additionally, ?rdm converges to the global minimal value, uij and u?1 converge to the following limits,

lim
t??

uij(t) = I(i = j) �

(cid:35)? 1

4

(cid:34) d

(cid:88)

i=1

?2
i
?2
i

�

?i
?i

?i ? [d],

lim
t??

u?1(t) =

(cid:35) 1

4

.

(cid:34) d

(cid:88)

i=1

?i
?i

(C.15)

42

Translating back to the original parameterization, we have this is equivalent to

W KQ(t) =

lim
t??

W P V (t) =

lim
t??

?

?
?

?

?
?

(cid:13)
(cid:2)E?? ?2
(cid:13)
(cid:13)

?

(cid:3)?1 E (cid:2)?2
?

? 1
2

(cid:3)(cid:13)
(cid:13)
(cid:13)
F
0?
d

� (cid:2)E?? ?2

?

(cid:3)?1 E (cid:2)?2
?

(cid:3) 0d

0

?

?
? ,

0d�d

0?
d

(cid:13)
(cid:2)E?? ?2
(cid:13)
(cid:13)

?

0d
(cid:3)?1 E (cid:2)?2
?

1
2

(cid:3)(cid:13)
(cid:13)
(cid:13)
F

?

?
? ,

where ?? = N +1

N ?? + 1

N tr(?? )Id ? Rd�d and E is over ?? .

Proof. First, we prove the PL Inequality. From Lemma C.4, we know

?rdm(U11, u?1) ? min ?rdm =

(cid:18)

?i

uiiu?1 ?

1
2

d
(cid:88)

i=1

(cid:19)2

?i
?i

+

1
2

(cid:88)

i?=j

?iju2

?1u2
ij,

where ?i, ?ij, ?i are defined in (C.3). Meanwhile, we calculate the square norm of the gradient of ?rdm:

???rdm(U11, u?1)?2

2 :=

d
(cid:88)

i,j=1

(cid:12)
(cid:12)
(cid:12)
(cid:12)

??rdm
?uij

(cid:12)
2
(cid:12)
(cid:12)
(cid:12)

+

(cid:12)
(cid:12)
(cid:12)
(cid:12)

??rdm
?u?1

(cid:12)
2
(cid:12)
(cid:12)
(cid:12)

?

d
(cid:88)

i,j=1

(cid:12)
(cid:12)
(cid:12)
(cid:12)

??rdm
?uij

(cid:12)
2
(cid:12)
(cid:12)
(cid:12)

i u2
?2
?1

(cid:18)

uiiu?1 ?

=

d
(cid:88)

i=1

(cid:19)2

?i
?i

(cid:88)

+

i?=j

iju4
? 2

?1u2
ij.

Comparing the two displays above, we know in order to achieve ???rdm?2
suffices to make

2 ? ? (?rdm ? min ?rdm) , it

?iu?1(t)2 ?

?iju?1(t)2 ?

?
2
?
2

?i ? [d],

?i ?= j ? [d].

We define ? := min {?i, ?ij, i ?= j ? [d]} , then it is sufficient to make

?u?1(t)2 ?

?
2

.

From Lemma C.7, we know that we can actually lower bound u?1 from below by a positive constant. Then,
the inequality holds if we take

? :=

?

2

? � ?2
d ?E?2

? ?op

(cid:104)
2 ?E?? ??2

F ?

?

d?2 (cid:104)

E ??? ?op ??? ?2

F

(cid:105)(cid:105)

> 0.

Therefore, as long as we take ? as above, a PL inequality holds for ?rdm.

With an abuse of notation, let us write ?rdm(t) = ?rdm(U11(t), u?1(t)). Then, from the dynamics of

gradient flow and the PL Inequality ((C.14)), we know

d
dt

[?rdm(t) ? min ?rdm] = ? ???rdm(t)?2

2 ? ?? (?rdm(t) ? min ?rdm) ,

43

which by Gr�onwall�s inequality implies

0 ? ?rdm(t) ? min ?rdm ? exp(??t) [?rdm(0) ? min ?rdm] ? 0

when t ? ?. From Lemma C.4, we know

(cid:18)

?i

uiiu?1 ?

d
(cid:88)

i=1

(cid:19)2

?i
?i

(cid:88)

+

i?=j

?iju2

?1u2

ij ? 0 when t ? ?.

This implies

uiiu?1 ?

?i
?i

?i ? [d],

uiju?1 ? 0 ?i ?= j ? [d].

(C.16)

(cid:80)d

We take square of uii(t)u?1(t) and uij(t)u?1(t),
?2
u2
i
?1
?2
i
(cid:80)d
i,j=1 u2

ij ? (cid:80)d
ij. So we have

i,j=1 u2

i=1

then sum over all i, j ? [d]. Then, we get
. From Lemma C.5, we know for any t ? 0, u?1(t)2 = tr (cid:0)U11(U11)?(cid:1) =

u?1(t)4 = u2
?1

d
(cid:88)

i,j=1

u2
ij ?

d
(cid:88)

i=1

?2
i
?2
i

,

which implies

i=1
when t ? ?. Combining (C.16) and (C.17), we conclude

(cid:34) d

(cid:88)

u?1(t) ?

(cid:35) 1

4

?2
i
?2
i

uij(t) ? 0 ?i ?= j ? [d],

uii(t) ?

(cid:35)? 1

4

(cid:34) d

(cid:88)

i=1

?2
i
?2
i

�

?i
?i

?i ? [d].

(C.17)

D Technical lemmas

Lemma D.1 (Matrix Derivatives, Kronecker Product and Vectorization, [PP+08]). We denote A, B, X as
matrices and x as vectors. Then, we have
?x = (cid:0)B + B?(cid:1) x.

� ?x?Bx
� Vec(AXB) = (cid:0)B? ? A(cid:1) Vec(X).
� tr (cid:0)A?B(cid:1) = Vec(A)? Vec(B).
?X tr (cid:0)XBX?(cid:1) = XB? + XB.
?X tr (cid:0)AX?(cid:1) = A.
?X tr (cid:0)AXBX?C(cid:1) = A?C?XB? + CAXB.

�

�

�

?

?

?

44

Lemma D.2. If X is Gaussian random vector of d dimension, mean zero and covariance matrix ?, and
A ? Rd�d is a fixed matrix. Then

(cid:104)

XX ?AXX ?(cid:105)

E

(cid:16)

A + A?(cid:17)

= ?

? + tr(A?)?.

Proof. We denote X = (X1, ..., Xd)?. Then,

XX ?AXX ? = X(X ?AX)X ? =

?

?

d
(cid:88)

i,j=1

?
? XX ?.

AijXiXj

So we know (XX ?AXX ?)k,l =
XkXl. From Isserlis� Theorem in probability theory
(Theorem 1.1 in Michalowicz et al. [Mic+09], originally proposed in Wick [Wic50]), we know for any
i, j, k, l ? [d], it holds that

i,j=1 AijXiXj

(cid:16)(cid:80)d

(cid:17)

E(cid:2)XiXjXkXl

(cid:3) = ?ij?kl + ?ik?jl + ?il?jk.

Then, we have for any fixed k, l ? [d],

E(XX ?AXX ?)k,l =

d
(cid:88)

i,j=1

Aij?ij?kl + Aij?ik?jl + Aij?il?jk

= tr(A?)?kl + ??

k (A + A?)?l.

Therefore, we know

E(XX ?AXX ?) = ?

(cid:16)

A + A?(cid:17)

? + tr(A?)?.

Lemma D.3 (Von-Neumann�s Trace Inequality). Let U, V ? Rd�n with d ? n. We have

(cid:16)

tr

U ?V

(cid:17)

?

d
(cid:88)

i=1

?i(U )?i(V ) ? ?U ?op �

?

?i(V ) ?

d � ?U ?op?V ?F

d
(cid:88)

i=1

where ?1(X) ? ?2(X) ? � � � ? ?d(X) are the ordered singular values of X ? Rd�n.

Lemma D.4 ([MR99]). For any two positive semi-definitive matrices A, B ? Rd�d, we have

� tr[AB] ? 0.

� AB ? 0 if and only if A and B commute.

45

E Experiment details

In this section, we provide more details for the experiment in Figure 1. Our experimental setup is based on
the codebase provided by Garg et al. [Gar+22], with a modification that allows for the possibility that the
covariate distribution changes across prompts. We use the standard GPT2 architecture with 256 embedding
size, 12 layers and 8 heads [Rad+18] as implemented by HuggingFace [Wol+20]. For the GPT2 models, we
use the embedding method proposed by Garg et al. [Gar+22], where instead of concatenating x and y into a
single token, they are treated as separate tokens. It is also worth noting that the training objective function
for the GPT2 model is different than those we consider for the linear self-attention network: for the GPT2
model, the objective function is the average over the full length of the context sequence (predictions for each
xi using (xk, yk)k<i), while in our setting the objective function is only for the final query point. However,
in the figure, for both GPT2 and the linear self-attention model the error plotted corresponds to the error for
predicting the final query point.

In all experiments, covariates are sampled from a mean-zero Gaussian in d = 20 dimensions with either
fixed or random covariance matrix. For the fixed covariance case, we fix the covariance matrix to be identity;
for the random case, the covariance matrices are restricted to be diagonal and all diagonal entries are i.i.d.
sampled from the standard exponential distribution. The linear weights in all tasks are i.i.d. sampled from
standard Gaussian distribution and also independently from all covariates. We trained the model for 500000
steps using Adam [KB14] with a batch size of 64 and learning rate of 0.0001. We use the same curriculum
strategy of Garg et al. [Gar+22] for acceleration.

For testing the trained model, we used ordinary least squares as a baseline which is optimal for noiseless
linear regression tasks. For prompts at test time, covariates are sampled i.i.d. from a mean-zero Gaussian
In the random-
distribution. For the fixed-covariance evaluation, the covariance is the identity matrix.
covariance evaluation, the covariance is a random diagonal matrix with diagonal entries sampled from the
standard exponential distribution, multiplied by a scaling coefficient c ? {1, 4, 9}, i.e. for each task ?, the
covariance matrix in the random case is

?? = c � diag (??,1, ..., ??,d)

i.i.d.? Exponential(1) for any ? and i ? [d]. The plots in Figure 1 show the error averaged over
where ??,i
642 prompts, where we sample 64 covariance matrices for each curve and 64 prompts for each covariance
matrix. We compute 90% confidence interval over 1000 bootstrap trials for each teat.

References

[Abe+23]

Jacob Abernethy, Alekh Agarwal, Teodor V. Marinov, and Manfred K. Warmuth. �A Mech-
anism for Sample-Efficient In-Context Learning for Sparse Retrieval Tasks�. In: Preprint,
arXiv:2305.17040 (2023) (Cited on page 3).

[Ahn+23] Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. �Transformers learn
In: Preprint,

in-context

learning�.

for

to implement preconditioned gradient descent
arXiv:2306.00297 (2023) (Cited on pages 2, 4, 6).

[APG23]

[AL23]

Kabir Ahuja, Madhur Panwar, and Navin Goyal. �In-Context Learning through the Bayesian
Prism�. In: Preprint, arXiv:2306.04891 (2023) (Cited on pages 3, 16).

Kartik Ahuja and David Lopez-Paz. �A Closer Look at In-Context Learning under Distribution
Shifts�. In: Preprint, arXiv:2305.16704 (2023) (Cited on page 3).

46

[Aky+22]

[Ani+22]

[ACH18]

[Aro+19]

[Azu+21]

[Bai+23]

[Bel20]

[BPG20]

[CLC19]

[Dai+22]

[Dai+19]

Ekin Aky�urek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. �What learn-
ing algorithm is in-context learning? Investigations with linear models�. In: arXiv preprint
arXiv:2211.15661 (2022) (Cited on pages 2, 3, 6).

Cem Anil, Yuhuai Wu, Anders Johan Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay
Venkatesh Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur. �Ex-
ploring Length Generalization in Large Language Models�. In: Advances in Neural Information
Processing Systems (NeurIPS). 2022 (Cited on page 16).

Sanjeev Arora, Nadav Cohen, and Elad Hazan. �On the optimization of deep networks: Im-
plicit acceleration by overparameterization�. In: International Conference on Machine Learn-
ing. 2018, pp. 244�253 (Cited on page 8).

Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. �Implicit regularization in deep matrix
factorization�. In: Advances in Neural Information Processing Systems 32 (2019) (Cited on
pages 8, 17).

Shahar Azulay, Edward Moroshko, Mor Shpigel Nacson, Blake E Woodworth, Nathan Srebro,
Amir Globerson, and Daniel Soudry. �On the implicit bias of initialization shape: Beyond in-
finitesimal mirror descent�. In: International Conference on Machine Learning. 2021, pp. 468�
477 (Cited on page 8).

Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. �Transformers as Statis-
ticians: Provable In-Context Learning with In-Context Algorithm Selection�. In: Preprint,
arXiv:2306.04637 (2023) (Cited on page 3).

Mohamed Ali Belabbas. �On implicit regularization: Morse functions and applications to ma-
trix factorization�. In: arXiv preprint arXiv:2001.04264 (2020) (Cited on page 17).

Satwik Bhattamishra, Arkil Patel, and Navin Goyal. �On the computational power of transform-
ers and its implications in sequence modeling�. In: arXiv preprint arXiv:2006.09286 (2020)
(Cited on page 3).

Yuejie Chi, Yue M Lu, and Yuxin Chen. �Nonconvex optimization meets low-rank matrix fac-
torization: An overview�. In: IEEE Transactions on Signal Processing 67.20 (2019), pp. 5239�
5269 (Cited on page 17).

Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Zhifang Sui, and Furu Wei. �Why Can GPT Learn
In-Context? Language Models Secretly Perform Gradient Descent as Meta Optimizers�. In:
arXiv preprint arXiv:2212.10559 (2022) (Cited on page 3).

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhut-
dinov. �Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context�. In:
Association for Computational Linguistics (ACL). 2019 (Cited on page 16).

[Deh+19] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and ?ukasz Kaiser. Uni-
versal Transformers. 2019. arXiv: 1807.03819 [cs.CL] (Cited on page 3).

[Dos+21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,
Jakob Uszkoreit, and Neil Houlsby. �An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale�. In: International Conference on Learning Representations (ICLR). 2021
(Cited on page 1).

47

[DHL18]

[Ede+22]

[Gar+22]

[Gun+17]

Simon S Du, Wei Hu, and Jason D Lee. �Algorithmic regularization in learning deep homo-
geneous models: Layers are automatically balanced�. In: Advances in neural information pro-
cessing systems 31 (2018) (Cited on page 8).

Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. �Inductive biases and vari-
able creation in self-attention mechanisms�. In: International Conference on Machine Learn-
ing. 2022 (Cited on page 3).

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. �What can transformers
learn in-context? a case study of simple function classes�. In: arXiv preprint arXiv:2208.01066
(2022) (Cited on pages 2�4, 6, 11�13, 15, 20, 46).

Suriya Gunasekar, Blake E Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nati
Srebro. �Implicit regularization in matrix factorization�. In: Advances in Neural Information
Processing Systems 30 (2017) (Cited on page 17).

[Han+23] Chi Han, Ziqi Wang, Han Zhao, and Heng Ji. In-Context Learning of Large Language Models

Explained as Kernel Regression. 2023. arXiv: 2305.12766 [cs.CL] (Cited on page 3).

[JSL22]

[Jin+23]

[KB14]

[Li+23a]

[Li+23b]

Samy Jelassi, Michael Sander, and Yuanzhi Li. �Vision transformers provably learn spatial
structure�. In: Advances in Neural Information Processing Systems 35 (2022), pp. 37822�
37836 (Cited on page 4).

Jikai Jin, Zhiyuan Li, Kaifeng Lyu, Simon S Du, and Jason D Lee. �Understanding incremental
learning of gradient descent: A fine-grained analysis of matrix sensing�. In: arXiv preprint
arXiv:2301.11500 (2023) (Cited on page 17).

Diederik P Kingma and Jimmy Ba. �Adam: A method for stochastic optimization�. In: arXiv
preprint arXiv:1412.6980 (2014) (Cited on page 46).

Shuai Li, Zhao Song, Yu Xia, Tong Yu, and Tianyi Zhou. �The Closeness of In-Context Learn-
ing and Weight Shifting for Softmax Regression�. In: arXiv preprint arXiv:2304.13276 (2023)
(Cited on page 3).

Yingcong Li, M Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. �Transform-
ers as Algorithms: Generalization and Stability in In-context Learning�. In: arXiv preprint
arXiv:2301.07067 (2023) (Cited on page 6).

[LMZ18] Yuanzhi Li, Tengyu Ma, and Hongyang Zhang. �Algorithmic regularization in over-
parameterized matrix sensing and neural networks with quadratic activations�. In: Conference
On Learning Theory. 2018, pp. 2�47 (Cited on page 17).

[LLR23]

[LLL20]

Yuchen Li, Yuanzhi Li, and Andrej Risteski. �How do transformers learn topic structure: To-
wards a mechanistic understanding�. In: arXiv preprint arXiv:2303.04245 (2023) (Cited on
page 4).

Zhiyuan Li, Yuping Luo, and Kaifeng Lyu. �Towards resolving the implicit bias of gradient de-
scent for matrix factorization: Greedy low-rank learning�. In: arXiv preprint arXiv:2012.09839
(2020) (Cited on page 17).

[LCW21] Valerii Likhosherstov, Krzysztof Choromanski, and Adrian Weller. �On the expressive power

of self-attention matrices�. In: arXiv preprint arXiv:2106.03764 (2021) (Cited on page 3).

[Liu+23]

Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. �Trans-
formers Learn Shortcuts to Automata�. In: International Conference on Learning Representa-
tions (ICLR). 2023 (Cited on page 3).

48

[MR99]

[Mic+09]

[Min+22]

AR Meenakshi and C Rajian. �On a product of positive semidefinite matrices�. In: Linear
algebra and its applications 295.1-3 (1999), pp. 3�6 (Cited on page 45).

JV Michalowicz, JM Nichols, F Bucholtz, and CC Olson. �An Isserlis� theorem for mixed
Gaussian variables: Application to the auto-bispectral density�. In: Journal of Statistical
Physics 136 (2009), pp. 89�102 (Cited on page 45).

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and
Luke Zettlemoyer. �Rethinking the Role of Demonstrations: What Makes In-Context Learning
Work?� In: arXiv preprint arXiv:2202.12837 (2022) (Cited on page 3).

[Ope23]

OpenAI. GPT-4 Technical Report. 2023. arXiv: 2303.08774 [cs.CL] (Cited on page 1).

[Osw+22]

[PMB19]

[PP+08]

Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, Jo�ao Sacramento, Alexander Mord-
vintsev, Andrey Zhmoginov, and Max Vladymyrov. �Transformers learn in-context by gradient
descent�. In: arXiv preprint arXiv:2212.07677 (2022) (Cited on pages 2, 3, 6, 9).

Jorge P�erez, Javier Marinkovi�c, and Pablo Barcel�o. �On the turing completeness of modern
neural network architectures�. In: arXiv preprint arXiv:1901.03429 (2019) (Cited on page 3).

Kaare Brandt Petersen, Michael Syskind Pedersen, et al. �The matrix cookbook�. In: Technical
University of Denmark 7.15 (2008), p. 510 (Cited on pages 23, 44).

[Rad+18] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. �Improving language

understanding by generative pre-training�. In: (2018) (Cited on page 46).

[Rad+19] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.
�Language models are unsupervised multitask learners�. In: OpenAI blog 1.8 (2019), p. 9 (Cited
on page 15).

[SSX23] Mahdi Soltanolkotabi, Dominik St�oger, and Changzhi Xie. �Implicit Balancing and Regulariza-
tion: Generalization and Convergence Guarantees for Overparameterized Asymmetric Matrix
Sensing�. In: arXiv preprint arXiv:2303.14244 (2023) (Cited on page 17).

[TK23]

[Vas+17]

Asher Trockman and J Zico Kolter. �Mimetic Initialization of Self-Attention Layers�. In: arXiv
preprint arXiv:2305.09828 (2023) (Cited on page 6).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
?ukasz Kaiser, and Illia Polosukhin. �Attention is all you need�. In: Advances in Neural Infor-
mation Processing Systems 30 (2017) (Cited on page 5).

[WZW23] Xinyi Wang, Wanrong Zhu, and William Yang Wang. �Large Language Models Are Implicitly
Topic Models: Explaining and Finding Good Demonstrations for In-Context Learning�. In:
arXiv preprint arXiv:2301.11916 (2023) (Cited on page 3).

[Wic50]

[Wol+20]

Gian-Carlo Wick. �The evaluation of the collision matrix�. In: Physical review 80.2 (1950),
p. 268 (Cited on page 45).

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony
Moi, Pierric Cistac, Tim Rault, R�emi Louf, Morgan Funtowicz, et al. �Transformers: State-
of-the-art natural language processing�. In: Proceedings of the 2020 conference on empirical
methods in natural language processing: system demonstrations. 2020, pp. 38�45 (Cited on
page 46).

49

[Xie+21]

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. �An explanation of in-
context learning as implicit bayesian inference�. In: arXiv preprint arXiv:2111.02080 (2021)
(Cited on page 3).

[Yun+19] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J Reddi, and Sanjiv Ku-
mar. �Are transformers universal approximators of sequence-to-sequence functions?� In: arXiv
preprint arXiv:1912.10077 (2019) (Cited on page 3).

[Yun+20] Chulhee Yun, Yin-Wen Chang, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and
Sanjiv Kumar. �O (n) connections are expressive enough: Universal approximability of sparse
transformers�. In: Advances in Neural Information Processing Systems 33 (2020), pp. 13783�
13794 (Cited on page 3).

[Zha+23] Yufeng Zhang, Fengzhuo Zhang, Zhuoran Yang, and Zhaoran Wang. �What and How does In-
Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization�.
In: Preprint, arXiv:2305.19420 (2023) (Cited on page 3).

50


