Rethinking Link Prediction for Directed Graphs

Mingguo He 1 Yuhe Guo 1 Yanping Zheng 1 Zhewei Wei 1 Stephan G �unnemann 2 Xiaokui Xiao 3

5
2
0
2

y
a
M
1
2

]

G
L
.
s
c
[

2
v
4
2
7
5
0
.
2
0
5
2
:
v
i
X
r
a

Abstract

Link prediction for directed graphs is a crucial
task with diverse real-world applications. Recent
advances in embedding methods and Graph Neu-
ral Networks (GNNs) have shown promising im-
provements. However, these methods often lack
a thorough analysis of their expressiveness and
suffer from effective benchmarks for a fair evalu-
ation. In this paper, we propose a unified frame-
work to assess the expressiveness of existing meth-
ods, highlighting the impact of dual embeddings
and decoder design on directed link prediction
performance. To address limitations in current
benchmark setups, we introduce DirLinkBench, a
robust new benchmark with comprehensive cov-
erage, standardized evaluation, and modular ex-
tensibility. The results on DirLinkBench show
that current methods struggle to achieve strong
performance, while DiGAE outperforms other
baselines overall. We further revisit DiGAE the-
oretically, showing its graph convolution aligns
with GCN on an undirected bipartite graph. In-
spired by these insights, we propose a novel Spec-
tral Directed Graph Auto-Encoder SDGAE that
achieves state-of-the-art average performance on
DirLinkBench. Finally, we analyze key factors
influencing directed link prediction and highlight
open challenges in this field. The code is available
at here.

1. Introduction

A directed graph (or digraph) is a type of graph in which
the edges between nodes have a specific direction. These
graphs are commonly used to model real-world asymmet-
ric relationships, such as �following� and �followed� in
social networks (Leskovec & Sosi?c, 2016) or �link� and
�linked� in web pages (Page et al., 1999). Directed graphs
capture the inherent directionality of these relationships

1Gaoling School of Artifical Intelligence, Renmin University
of China 2Technical University of Munich 3National University of
Singapore. Contact to: Mingguo He <mingguo@ruc.edu.cn>.

A Preprint Version.

1

Table 1. Overview of directed graph learning methods.

Method

Name

Embedding

Embedding
Methods

Graph Neural
Networks

su, tu
HOPE (Ou et al., 2016)
su, tu
APP (Zhou et al., 2017)
su, tu
AROPE (Zhang et al., 2018)
su, tu
STRAP (Yin & Wei, 2019)
su, tu
NERD (Khosla et al., 2020)
su, tu
DGGAN (Zhu et al., 2021b)
su, tu
ELTRA (Hamedani et al., 2023)
su, tu
ODIN (Yoo et al., 2023)
su, tu
DiGAE (Kollias et al., 2022)
su, tu
CoBA (Liu et al., 2023)
su, tu
BLADE (Virinchi & Saladi, 2023)
hu, mu
Gravity GAE (Salha et al., 2019)
hu, mu
DHYPR (Zhou et al., 2022)
hu
DGCN (Tong et al., 2020b)
DiGCN & DiGCNIB (Tong et al., 2020a) hu
hu
DirGNN (Rossi et al., 2024)
hu
HoloNets (Koke & Cremers, 2024)
hu
NDDGNN (Huang et al., 2024)
zu
MagNet (Zhang et al., 2021)
zu
LightDiC (Li et al., 2024)
zu
DUPLEX (Ke et al., 2024)

and provide a more accurate representation of complex sys-
tems. Link prediction is a fundamental and widely studied
task in directed graphs, with numerous real-world applica-
tions. Examples include predicting follower relationships
in social networks (Liben-Nowell & Kleinberg, 2003), rec-
ommending products in e-commerce (Rendle et al., 2009),
and detecting intrusions in network security (Bhuyan et al.,
2013).

Machine learning techniques have been extensively devel-
oped to enhance link prediction performance on directed
graphs. Existing methods can be broadly categorized into
embedding methods and graph neural networks (GNNs).
Embedding methods aim to preserve the asymmetry of di-
rected graphs by generating two separate embeddings for
each node u: a source embedding su and a target embedding
tu (Hamedani et al., 2023; Yoo et al., 2023), which are also
known as content/context representations (Tang et al., 2015;
Ou et al., 2016; Yin & Wei, 2019). GNNs, on the other hand,
can be further divided into four classes based on the types of
embeddings they generate. (1) Source-target methods, simi-
lar to embedding methods, employ specialized propagation
mechanisms to learn distinct source and target embeddings
for each node (Kollias et al., 2022; Liu et al., 2023). (2)
Gravity-inspired methods, inspired by Newton�s law of uni-
Rd
versal gravitation, learn a real-valued embedding hu

?

Rethinking Link Prediction for Directed Graphs

R+ for each node u (Salha
and a mass parameter mu
?
et al., 2019; Zhou et al., 2022). (3) Single real-valued meth-
ods follow a conventional approach by learning a single
Rd (Rossi et al., 2024; Huang
real-valued embedding hu
et al., 2024). (4) Complex-valued methods use Hermitian
adjacency matrices, learning complex-valued embeddings
Cd (Zhang et al., 2021; Ke et al., 2024). We summa-
zu
rize these methods in Table 1.

?

?

Although the methods described above have achieved
promising results in directed link prediction, several chal-
lenges remain. First, it is unclear what types of embedding
are effective in predicting directed links, as there is a lack
of comprehensive research assessing these methods� expres-
siveness. Second, existing methods have not been fairly
compared and evaluated, highlighting the need for a robust
benchmark for directed link prediction. Current experi-
mental setups face multiple issues, such as the omission of
basic baselines (e.g., MLP shown in Figures 1(a) and 1(b)),
label leakage (illustrated in Tables 5 and 6), class imbalance
(shown in Figures 2(a) and 2(b)), single evaluation metrics,
and inconsistent dataset splits. We discuss these issues in
detail in Section 3.2.

In this paper, we first propose a unified learning framework
for directed link prediction methods to assess the expres-
siveness of different embedding types. We demonstrate that
dual methods (including source-target, gravity-inspired, and
complex-valued methods) are more expressive for directed
link prediction than single methods (i.e., single real-valued
methods). Meanwhile, we highlight the often-overlooked
importance of decoder design in achieving better perfor-
mance, as most research has primarily focused on encoders.
To address the limitations of existing experimental setups,
we introduce DirLinkBench, a robust new benchmark
for directed link prediction that offers comprehensive cov-
erage (seven real-world datasets, 16 baseline methods, and
seven evaluation metrics), standardized evaluation (unified
splits, feature inputs, and task setups), and modular exten-
sibility (support for adding new datasets, decoders, and
sampling strategies).

The results in DirLinkBench reveal that current methods
struggle to achieve strong and firm performance across di-
verse datasets. Interestingly, a simple directed graph auto-
encoder, DiGAE (Kollias et al., 2022), outperforms other
baselines in general. We revisit DiGAE from a theoretical
perspective and observe that its graph convolution is equiv-
alent to the GCN (Kipf & Welling, 2017) convolution on
an undirected bipartite graph. Building on this insight, we
propose SDGAE, a novel spectral directed graph auto-
encoder that learns arbitrary spectral filters via polynomial
approximation. SDGAE achieves state-of-the-art (SOTA)
results on four of the seven datasets and ranks highest on
average. Finally, we investigate key factors influencing di-

rected link prediction (e.g., input features, decoder design,
and degree distribution) and conclude with open challenges
to advance the field. We summarize our contributions as
follows.

� We propose a unified framework to assess the expres-
siveness of directed link prediction methods, showing
that dual methods are more expressive and highlighting
the importance of decoder design.

� We introduce DirLinkBench, a robust new benchmark
with comprehensive coverage, standardized evaluation,
and modular extensibility for directed link prediction.

� We propose a novel directed graph auto-encoder
SDGAE, inspired by the theoretical insights of Di-
GAE and achieving SOTA average performance on
DirLinkBench.

� We empirically analyze the factors affecting the perfor-
mance of directed link prediction and highlight open
challenges for future research.

2. Background and related work

In this section, we first introduce the background and related
work of directed graph learning methods, which can be
broadly categorized into embedding methods and graph
neural networks (GNNs). Additionally, we will introduce
some background on spectral-based GNNs for undirected
graphs.

|

|

|

E
|

represent the number of nodes and edges in

=
Notation. We consider a directed, unweighted graph
G
(V, E), with node set V and edge set E. Let n =
V
and
m =
,
G
respectively. We use A to denote the adjacency matrix of
, where Auv = 1 if there exists a directed edge from
G
node u to node v, and Auv = 0 otherwise. The Hermitian
is denoted by H and defined as H =
adjacency matrix of
2 ?(cid:1). Here, As = A
exp (cid:0)i ?
A? is the adjacency
As
?
, and ? =
matrix of the undirected graph derived from
A? is a skew-symmetric matrix. i is the imaginary unit.
A
We denote the out-degree and in-degree matrices of A by
Dout = diag(A1) and Dint = diag(A?1), respectively,
where 1 is the all-one vector. Let X
denote the
node feature matrix, where each node has a d?-dimensional
feature vector.

Rn�d?

?

?

?

G

G

2.1. Embedding Methods

Embedding methods for directed graphs aim to capture
asymmetric relationships. Most approaches assign each
node u two embedding vectors: a source embedding su
and a target embedding tu. These embeddings are typi-
cally learned using either factorization-based or random

2

Rethinking Link Prediction for Directed Graphs

walk-based techniques. Factorization-based methods in-
clude HOPE (Ou et al., 2016), which computes Katz similar-
ity (Katz, 1953) followed by singular value decomposition
(SVD)(Golub & Van Loan, 2013), and AROPE (Zhang et al.,
2018), which generalizes this approach to preserve arbitrary-
order proximities. STRAP (Yin & Wei, 2019) extends this
idea by combining Personalized PageRank (PPR) (Page
et al., 1999) scores from both the original and transposed
graphs before applying SVD. Random walk-based meth-
ods include APP (Zhou et al., 2017), which trains embed-
dings using PPR-guided random walks, and NERD (Khosla
et al., 2020), which samples nodes based on degree distribu-
tions. Other recent notable methods include DGGAN (Zhu
et al., 2021b) (which employs adversarial training), EL-
TRA (Hamedani et al., 2023) (based on ranking-oriented
learning ), and ODIN (Yoo et al., 2023) (which incorporates
degree bias separation). All these methods focus on gen-
erating source�target embeddings from graph structures to
support link prediction tasks.

2.2. Graph Neural Networks for Directed Graphs

Graph Neural Networks for directed graphs can be broadly
classified into four categories based on their embedding
strategies: (1) Source�target methods learn separate source
and target embeddings for each node. DiGAE (Kollias et al.,
2022) applies GCN�s convolutions (Kipf & Welling, 2017)
to both the adjacency matrix and its transpose. CoBA (Liu
et al., 2023) jointly aggregates source and target neigh-
bors, while BLADE (Virinchi & Saladi, 2023) introduces
an asymmetric loss to learn dual embeddings from local
neighborhoods.
(2) Gravity-inspired methods are moti-
vated by Newton�s law of universal gravitation, learning
real-valued node embeddings along with a scalar mass
parameter. Gravity GAE (Salha et al., 2019) combines
a gravity-based decoder with a GCN-like encoder, and
DHYPR (Zhou et al., 2022) extends this approach through
hyperbolic collaborative learning. (3) Single real-valued
methods generate a single real-valued embedding per node.
DGCN (Tong et al., 2020b) constructs a directed Laplacian
using first- and second-order proximities. MotifNet (Monti
et al., 2018) employs motif-based Laplacians. DiGCN and
DiGCNIB (Tong et al., 2020a) generalize directed Lapla-
cians using Personalized PageRank (PPR). DirGNN (Rossi
et al., 2024) introduces a flexible convolution framework
for directed message-passing neural networks (MPNNs),
and HoloNets (Koke & Cremers, 2024) leverage holomor-
phic functional calculus with an architecture similar to
DirGNN. (4) Complex-valued methods utilize Hermitian
adjacency matrices to learn complex-valued embeddings.
MagNet (Zhang et al., 2021) defines a magnetic Laplacian
to construct directed graph convolutions. LightDiC (Li et al.,
2024) scales this approach to large graphs via a decoupled
design, while DUPLEX (Ke et al., 2024) employs dual

3

GAT encoders with Hermitian adjacency matrices. Addi-
tional methods include adapting Transformers to directed
graphs (Geisler et al., 2023) and extending over-smoothing
analyses to the directed setting (Maskey et al., 2023). While
these methods propose various convolutional and propa-
gation mechanisms for directed graphs, fair evaluation for
directed link prediction tasks is often lacking. Many exist-
ing experimental setups omit key baselines or suffer from
significant issues, such as label leakage. These challenges
motivate the development of this work.

2.3. Spectral-based Graph Neural Networks

In recent years, spectral-based Graph Neural Networks
(GNNs) have garnered significant attention and demon-
strated strong performance across various tasks (Bo et al.,
2023). Many popular methods in this category approxi-
mate spectral graph filters using polynomials of the adja-
cency or Laplacian matrix (Defferrard et al., 2016; Chien
et al., 2021; He et al., 2021; Wang & Zhang, 2022; He
et al., 2022). Specifically, let As denote the adjacency ma-
, and let Ds be
trix of the derived undirected graph for
its diagonal degree matrix, where Ds[i, i] = (cid:80)
j As[i, j].
We define the symmetric normalized adjacency matrix as
s AsD?1/2
P = D?1/2
. The propagation process in spectral-
based GNNs is then given by:

G

s

Z = h(P)X

K
(cid:88)

k=0

?

wkPkX,

(1)

where h(P) represents the spectral graph filter and wk are
the polynomial coefficients. From a spectral perspective,
we denote the eigendecomposition of the symmetric nor-
malized adjacency matrix as P = U?U?, where U is
the matrix of eigenvectors and ? is the diagonal matrix of
eigenvalues. Accordingly, the spectral graph filter can be
expressed as h(P) = Uh(?)U?, meaning it operates as a
function of the eigenvalues ?. Thus, spectral-based GNNs
approximate the graph filter function h(?) in the spectral
domain using a polynomial expansion. Notably, spectral
filters can also be equivalently defined using the Lapla-
cian matrix. In this case, the propagation process becomes
Z = (cid:80)K
P is the normalized
Laplacian matrix. Within this framework, many powerful
spectral-based GNNs have been proposed. For example,
ChebNet (Defferrard et al., 2016) uses Chebyshev polyno-
mials to approximate spectral filters, while GCN (Kipf &
Welling, 2017) simplifies ChebNet to improve efficiency.
More recently, GPR-GNN (Chien et al., 2021) learns the
coefficients wk directly, and both BernNet (He et al., 2021)
and JacobiConv (Wang & Zhang, 2022) use Bernstein and
Jacobi polynomial bases, respectively. Despite their effec-
tiveness, these methods are primarily designed for undi-
rected graphs. The development of analogous approaches

k=0 wkLkX, where L = I

?

Rethinking Link Prediction for Directed Graphs

Table 2. A unified framework for directed link prediction methods.

)
�

Encoder Enc(
Embeddings (?u, ?u)
Single real-valued hu = ?u, ? = ?u
tu = ?u
Source-target
exp(cid:0)i ?u
(cid:1)

Complex-valued

su = ?u,
zu = ?u
hu = ?u, mu = g(?u)

Gravity-inspired

?

Possible Decoder Dec(
(cid:1);
?(cid:0)h?
u hv
?(cid:0)s?
(cid:1);
u tv
Direc(cid:0)zu, zv
?(cid:0)mv
? log

)
�
(cid:1);
MLP(cid:0)hu
hv
MLP(cid:0)su
(cid:1);
tv
?
(cid:1); MLP(cid:0)?u
?v
?u
?v
?
?
?
?(cid:0)mv
(cid:1);

hu

hv

?

?

?

?

2
2
?

?

(cid:1)

MLP(cid:0)hu
hv
?
LR(cid:0)su
(cid:1)
tv
?
(cid:1)

? log(cid:0)dist(hu, hv)(cid:1)(cid:1)

that leverage polynomial approximations of spectral filters
for directed graphs remains a relatively unexplored area.

3. Rethinking Directed Link Prediction

In this section, we will revisit the link prediction task for
directed graphs and introduce a unified framework to assess
the expressiveness of existing methods. Meanwhile, we
examine the current experimental setups for directed link
prediction tasks and highlight four significant issues.

3.1. Unified Framework for Directed Link Prediction

The link prediction task on directed graphs is to predict
?,
potential directed links (edges) in an observed graph
? and node feature X. Formally,
with the given structure of
Definition 3.1. Directed link prediction problem. Given
? = (V, E?) and node feature X, the
an observed graph
goal of directed link prediction is to predict the likelihood
of a directed edge (u, v)
V )

�
E?. The probability of edge (u, v) existing is given by

E? existing, where E?

(V

?

?

G

G

G

\

p(u, v) = f (u

v

?

| G

?, X).

(2)

) denotes a prediction model, such as embedding
The f (
�
methods or graph neural networks. Unlike link prediction
on undirected graphs (Zhang & Chen, 2018), for directed
graphs, it is necessary to account for directionality. Specifi-
cally, p(u, v) and p(v, u) are not equal; they represent the
probability of a directed edge existing from node u to node
v, and from node v to node u, respectively. To evaluate
the expressiveness of existing methods for directed link
prediction, we propose a unified framework:

(?u, ?u) = Enc(

u
G
p(u, v) = Dec(?u, ?u, ?v, ?v),

?, X, u),

?

?

V,

(u, v)

?

?

E?.

(3)

(4)

?

) represents an encoder function, which includes
Here, Enc(
�
Rd? ,
various methods described in Section 1. And ?u
Rd? are real-valued dual embeddings of dimensions
?u
d? and d?, respectively. Dec(
) is a decoder function tai-
�
lored to the specific encoder method. This framework uni-
fies existing methods for directed link prediction, as sum-
marized in Table 2. Specifically, here ? represents the acti-
vation function (e.g., Sigmoid), while MLP and LR denote

?

the multilayer perceptron and the logistic regression pre-
dictor, respectively. The symbols
represent the
Hadamard product and the vector concatenation process,
respectively. Then, we provide details on the embeddings
and possible decoders used by the different methods listed
in Table 2.

and

?

?

� For single real-valued encoder, we define the real-
valued embedding as hu = ?u and set ?u = ?, where
? denotes nonexistence. Possible decoders include:
?(cid:0)h?

(cid:1), MLP(cid:0)hu

(cid:1), MLP(cid:0)hu

hv

hv

(cid:1).

u hv

?

?

� For source-target encoder, we define the source em-
bedding as su = ?u and the target embedding as
V . Possible decoders
tu = ?u for each node u
?
include: ?(cid:0)s?
(cid:1), etc.
(cid:1), MLP(cid:0)su
tv
?
Here, s?
u tv denotes the inner product of the source and
target embeddings.

(cid:1), LR(cid:0)su

u tv

tv

?

� For complex-valued encoder, we define the complex-
exp(i?u), where
Possible decoders in-
(cid:1), where
?v
?
) is a direction-aware function defined in DU-

valued embedding as zu = ?u
i is the imaginary unit.
clude: Direc(cid:0)zu, zv
Direc(
�
PLEX (Ke et al., 2024).

(cid:1), MLP(cid:0)?u

?u
?

?v
?

?

� For gravity-inspired encoder, we define the real-valued
embedding as hu = ?u and set the mass parame-
ter mu = g(?u), where g(
) is a function or neu-
�
ral network that converts ?u into a scalar (Zhou
et al., 2022). Possible decoders include: ?(cid:0)mv
?
(cid:1) (Salha et al., 2019), ?(cid:0)mv
? log
?
? log(cid:0)dist(hu, hv)(cid:1)(cid:1) (Zhou et al., 2022). Here, ? is a
hyperparameter and dist(
) represents the hyperbolic
�
distance.

2
2
?

hu

hv

?

?

?

Based on this framework, if 0 < d?, d?
n, these encoders
involve two real embeddings ?u and ?u, which we refer to
as dual methods. Conversely, if d? = 0, these encoders
have only a single real embedding ?u, and are referred to
as single methods. Next, we analyse the expressiveness of
dual and single methods in terms of asymmetry preservation
and graph reconstruction.

Asymmetry preservation. Previous source-target encoders
have demonstrated that using the source and target embed-

4

Rethinking Link Prediction for Directed Graphs

Table 3. The results of MLP and baselines under PyGSD (He et al., 2023) setup on Direction Prediction (DP) task.

Method

Cora-ML

CiteSeer

Telegram

Cornell

Texas

Wisconsin

MLP

86.13

85.49
DGCN
DiGCN
85.37
DiGCNIB 86.12
86.33
MagNet

0.45

85.51

0.75
0.54
0.42
0.54

84.85
83.88
85.58
85.80

0.63

0.56
0.82
0.56
0.63

�

�
�
�
�

95.61

96.03
94.95
95.99
96.97

0.15

0.35
0.54
0.44
0.21

�

�
�
�
�

�

�
�
�
�

84.55
85.41
86.28
83.29

�

�
�
�
�

3.71
2.78
3.37
4.28

4.33

5.09
3.98
3.51
4.78

�

�
�
�
�

79.59
76.57
82.27
80.25

84.57
82.41
87.07
86.60

�

�
�
�
�

3.59
2.56
2.16
2.72

86.40

2.60

83.08

87.95

2.65

Table 4. The results of MLP and baselines under PyGSD (He et al., 2023) setup on Existence Prediction (EP) task.

Method

Cora-ML

CiteSeer

Telegram

Cornell

Texas

Wisconsin

MLP

78.85

0.83

71.03

76.45
DGCN
DiGCN
76.17
DiGCNIB 78.80
77.37
MagNet

�

�
�
�
�

0.49
0.49
0.51
0.45

0.89

82.72

0.79
0.88
0.91
0.71

83.18
83.12
84.49
85.82

0.57

1.55
0.43
0.51
0.39

�

�
�
�
�

�

�
�
�
�

70.72
72.00
74.55
71.47

2.29

72.53

68.41

67.95
67.16
69.77
68.98

2.05

2.27
1.82
2.05
2.27

�

�
�
�
�

69.58

63.65
63.54
67.60
65.94

�

�
�
�
�

2.40
3.85
3.44
1.88

2.27

2.86
2.47
2.79
2.53

�

�
�
�
�

68.12
67.01
70.78
71.23

dings effectively preserves asymmetric information in di-
rected graphs, a capability that single methods lack (Zhou
et al., 2017; Yin & Wei, 2019; Ou et al., 2016; Hamedani
et al., 2023). We extend this claim by arguing that all
dual methods can preserve asymmetric information in
directed graphs, including complex-valued and gravity-
inspired encoders. Specifically, complex-valued meth-
ods (Zhang et al., 2021; Ke et al., 2024) encode direction-
ality as a geometric difference in complex space, meaning
that edges from node u to node v and edges from node v to
node u can be distinguished by the difference between z?
u zv
and z?
v zu. The gravity-inspired methods (Salha et al., 2019;
Zhou et al., 2022), on the other hand, use direction-sensitive
2 and
gravity to preserve asymmetry, i.e., Gmu/
||
2 distinguish between edges from node u
Gmv/
||
to node v and edges from node v to node u, where G is the
gravitational constant. Thus, in terms of preserving asym-
metry in directed graphs, dual methods have a significant
advantage over single methods, as the former can naturally
preserve the asymmetry of directed edges, facilitating link
prediction task.

hu

hu

hv

hv

?

?

||

||

?u, ?u
{

) and decoder Dec(

Graph reconstruction. Graph reconstruction involves us-
and the decoder function
ing node embeddings
}
Dec(
) to compute the probability p(u, v) of each directed
�
edge. The top-m edges are then selected to reconstruct the
original graph. For accurate reconstruction, the encoder
1 if an
Enc(
?
�
edge exists from u to v, and p(u, v)
0 otherwise. This
task requires the model to capture structural properties of the
graph, including edge existence and directionality, reflecting
the expressiveness for the directed link prediction (Yin &
Wei, 2019; Hamedani et al., 2023). Dual methods bene-
fit from the theoretical advantage of asymmetry preserva-

) need ensure p(u, v)
�

?

(a) Direction Prediction

(b) Existence Prediction

Figure 1. The results of MagNet (Zhang et al., 2021) as reported
in the original paper, alongside the reproduced MagNet and MLP
results.
tion. When equipped with a suitable Dec(
) function, these
�
methods can achieve effective graph reconstruction. The
underlying intuition is that: source-target methods can rep-
resent the neighbor matrix as Auv = s?
u tv (Yin & Wei,
2019; Ou et al., 2016; Kollias et al., 2022), complex-valued
methods can represent the Hermitian adjacency matrix as
Huv = z?
u zv (Zhang et al., 2021; Ke et al., 2024), and
gravity-inspired methods can represent the neighbor matrix
2 (Salha et al., 2019; Zhou
as Auv = Gmu/
||
et al., 2022). In contrast, single methods lack the intrinsic
preservation of asymmetry, but they can partially capture
the graph structure by using asymmetric decoders, such as
MLP(hu
hv). While this allows for limited structural re-
construction, single methods are theoretically insufficient
to represent arbitrary directed graphs. As formalized in
Proposition 3.2 (see the Appendix for the proof), they fail to
reconstruct certain structures, such as directed ring graphs.
Therefore, although asymmetric decoders enhance the per-
formance of single methods, their overall expressiveness
remains limited compared to dual methods.

hu

hv

?

||

?

Proposition 3.2. Single methods (single real-valued
embedding hu) with an asymmetric decoder function

5

CornellTexasWisconsinCora-MLCiteSeer30405060708090100Accuracy(%)MagNet(reported)MagNet(reproduced)MLPCornellTexasWisconsinCora-MLCiteSeer30405060708090100Accuracy(%)MagNet(reported)MagNet(reproduced)MLPRethinking Link Prediction for Directed Graphs

hv) can capture graph structure and enable re-
MLP(hu
construction for some specific directed graphs, but not arbi-
trary directed graphs, such as directed ring graphs.

?

Overall, dual methods have a clear theoretical advantage
in both asymmetry preservation and graph reconstruc-
tion, making them more expressive for link prediction com-
pared to single methods. However, we also want to highlight
the critical role that decoder function design plays in link
prediction tasks. While single embeddings are inherently
limited, they can still benefit from asymmetric decoders in
practical applications. These theoretical insights motivate
a deeper empirical analysis of how different encoder and
decoder functions impact performance in link prediction
tasks.

3.2. Issues with Existing Experimental Setups

The existing experimental setups for directed link predic-
tion are generally divided into two categories. The first is
the multiple subtask setup, which includes tasks such as
existence prediction (EP), direction prediction (DP), three-
class prediction (3C), and four-class prediction (4C). This
approach treats directed link prediction as a multi-class clas-
sification problem, where the model must predict edges as
positive (original direction), inverse (reverse direction), bidi-
rectional (both directions), or nonexistent (no connection).
Specifically:

� EP: The model predicts whether a directed edge (u, v)
exists in the graph, treating both reverse and nonexis-
tent edges as nonexistent.

� DP: The model predicts the direction of edges for node

pairs (u, v), where either (u, v)

E or (v, u)

E.

?

?

� 3C: The model classifies an edge as positive, reverse,

or nonexistent.

� 4C: The model classifies edges into four classes: posi-

tive, reverse, bidirectional, or nonexistent.

The multiple subtask setup is widely adopted by existing
GNN methods (Zhang et al., 2021; He et al., 2023; Fiorini
et al., 2023; Lin & Gao, 2023; Ke et al., 2024; Li et al.,
2024). The second category is the non-standardized setup
defined in various papers (Yin & Wei, 2019; Yoo et al.,
2023; Zhou et al., 2022; Kollias et al., 2022; Liu et al.,
2023). These setups involve different datasets, inconsistent
splitting strategies, and varying evaluation metrics. Below,
we discuss the four significant issues with existing setups.

Issue 1: The Multi-layer Perceptron (MLP) is a neglected
but powerful baseline. Most existing setups fail to report
the performance of MLP. We evaluate MLP across three pop-
ular multiple subtask setups: MagNet (Zhang et al., 2021),

6

(a) Cora

(b) CiteSeer

Figure 2. The number of samples and the accuracy for each class
of DUPLEX (Ke et al., 2024) on the Cora and CiteSeer dataset in
the 4C task.

PyGSD (He et al., 2023), and DUPLEX (Ke et al., 2024),
which cover a variety of datasets and baselines. Figures 1(a)
and 1(b) present the results of our reproduced MagNet ex-
periments alongside the MLP performance, showing that
the MLP performs comparably to MagNet on the DP and EP
tasks. Tables 3 and 4 display MLP results alongside several
other baselines on the DP and EF tasks within the PyGSD
setup, with the highest values highlighted in bold. Across
six datasets, MLP demonstrates competitive performance,
achieving state-of-the-art (SOTA) results for both tasks on
the Texas and Wisconsin datasets. Tables 5 and 6 present
the replicated DUPLEX experiments and MLP results on
the Cora and CiteSeer datasets, showing that MLP achieves
competitive performance. Interestingly, despite DUPLEX
being a recent advancement in directed graph learning, MLP
outperforms it on certain tasks. These findings highlight
the lack of fundamental baselines in previous studies and
suggest that current benchmarks do not provide a sufficient
challenge. More importantly, the results indicate that MLP
has achieved SOTA performance across various setups and
datasets, challenging the conclusions of prior studies and
contradicting the theoretical assumption that dual methods
are more expressive.

Issue 2: Many benchmarks suffer from label leakage.
As defined in Definition 3.1, directed link prediction aims to
predict potential edges from observed graphs, with the key
principle that test edges must remain hidden during training
to avoid label leakage. However, some current setups violate
this principle. For example, (1) MagNet (Zhang et al., 2021),
PyGSD (He et al., 2023), and DUPLEX (Ke et al., 2024)
expose test edges during negative edge sampling in the
training process, indirectly revealing the test edges� presence
to the model. (2) LighDiC (Li et al., 2024) uses eigenvectors
of the Laplacian matrix of the entire graph as input features,
embedding test edge information in the training input. (3)
DUPLEX propagates information across the entire graph
during training, making the test edges directly visible to the
model. To investigate, we experiment with DUPLEX using
its original code. As shown in Tables 5 and 6, DUPLEX�
(original settings with label leakage) clearly outperforms
DUPLEX� (propagation restricted to training edges) due to
label leakage. The similar results are observed with MLP:

ReversePositiveBidirectionalNonexistent02000400060008000NumberofSamples650365023686503020406080100Accuracy(%)93.4892.5815.2287.4389.76NumberofSamplesOverallAccuracy(%)Accuracy(%)ReversePositiveBidirectionalNonexistent0100200300400NumberofSamples320319603205060708090100Accuracy(%)90.7894.892.8267.2284.77NumberofSamplesOverallAccuracy(%)Accuracy(%)Rethinking Link Prediction for Directed Graphs

Table 5. Link prediction results on Cora dataset under the DUPLEX (Ke et al., 2024) setup: results without superscripts are from the
DUPLEX paper, � indicates reproduction with test set edges in training, and � indicates reproduction without test set edges in training.
DP(ACC)

DP(AUC)

EP(AUC)

EP(ACC)

3C(ACC)

4C(ACC)

Method

MagNet
DUPLEX
DUPLEX�
MLP�
DUPLEX�
MLP�

81.4
93.2

�
�
93.49
88.53

87.43
84.00

0.3
0.1

0.21
0.22

0.20
0.29

�
�

�
�

89.4
95.9

�
�
95.61
95.46

91.16
91.52

0.1
0.1

0.20
0.18

0.24
0.25

�
�

�
�

88.9
95.9

�
�
95.25
95.76

88.43
90.83

0.4
0.1

0.16
0.21

0.16
0.16

�
�

�
�

95.4
97.9

�
�
96.34
99.25

91.74
96.48

0.2
0.2

0.23
0.06

0.38
0.28

�
�

�
�

66.8
92.2

�
�
92.41
79.97

84.53
72.93

0.3
0.1

0.21
0.48

0.34
0.21

�
�

�
�

63.0
88.4

�
�
89.76
78.49

81.36
71.51

0.3
0.4

0.25
0.26

0.46
0.20

�
�

�
�

Table 6. Link prediction results for CiteSeer dataset under the DUPLEX (Ke et al., 2024) setup: results without superscripts are from the
DUPLEX paper, � indicates reproduction with test set edges in training, and � indicates reproduction without test set edges in training.
DP(ACC)

DP(AUC)

EP(AUC)

EP(ACC)

3C(ACC)

4C(ACC)

Method

MagNet
DUPLEX
DUPLEX�
MLP�
DUPLEX�
MLP�

80.7
95.7

�
�
92.11
85.74

83.59
77.34

0.8
0.5

0.78
1.80

1.47
1.86

�
�

�
�

88.3
98.6

�
�
95.85
93.33

89.34
87.36

0.4
0.4

0.87
1.27

1.03
1.26

�
�

�
�

91.7
98.7

�
�
97.54
97.55

85.56
89.19

0.9
0.4

0.54
0.97

1.36
0.95

�
�

�
�

96.4
99.7

�
�
98.93
99.58

91.82
95.97

0.6
0.2

0.59
0.52

0.98
0.61

�
�

�
�

72.0
94.8

�
�
88.22
81.20

76.37
67.82

0.9
0.2

1.06
0.82

2.07
1.21

�
�

�
�

69.3
91.1

�
�
84.77
76.30

73.80
64.25

0.4
1.0

1.01
0.62

2.01
1.35

�
�

�
�

MLP� (using in/out degrees from test edges) significantly
outperforms MLP� (using only training-edge in/out degrees).
These findings underscore that even the leakage of degree
information can significantly impact performance.

Issue 3: Multiple subtask setups result in class imbal-
ances and limited evaluation metrics. The multiple sub-
task setups treat the directed link prediction task as a multi-
class classification problem, causing significant class im-
balances that hinder model training. For example, the 4C
task in DUPLEX classifies edges into reverse, positive, bidi-
rectional, and nonexistent. However, bidirectional edges
are rare in real-world directed graphs, and reverse edges
are often assigned arbitrarily, lacking meaningful semantic
interpretation. Figures 2(a) and 2(b) illustrate this imbal-
ance, highlighting the difficulty of accurately predicting bidi-
rectional and nonexistent edges on the Cora and CiteSeer
datasets. Additionally, these setups rely heavily on accuracy
as the evaluation metric, which provides a limited and poten-
tially misleading assessment of model performance. Given
the nature of link prediction, ranking-based metrics such
as Hits@K and Mean Reciprocal Rank (MRR) are more
appropriate, a perspective well-established in evaluating
undirected link prediction methods (Li et al., 2023).

Issue 4: Lack of standardization in dataset splits and
feature inputs. Current settings face inconsistent dataset
splits. In multiple subtask setups, edges are typically split
into 80% for training, 5% for validation, and 15% for testing.
However, class proportions are further manually adjusted

for balance, which leads to varying training and testing
ratios across different datasets. Non-standardized setups
are more confusing, for example, ELTRA (Hamedani et al.,
2023) uses 90% of edges for training, STRAP (Yin & Wei,
2019) uses 50%, and DiGAE (Kollias et al., 2022) uses 85%,
making cross-study results difficult to evaluate. Feature
input standards are also lacking. Embedding methods often
omit node features, while GNNs require them. MagNet and
PyGSD use in/out degrees, DUPLEX uses random normal
distributions, LightDiC uses original features or Laplacian
eigenvectors, and DHYPR (Zhou et al., 2022) uses identity
matrices. This inconsistency undermines reproducibility.
As shown in Tables 3, 4, 5, 6, and Figure 1, reproduced
results often deviate significantly, with some better and
others worse.

In the above experiments, we strictly follow the configu-
rations of each setup and reproduce the results using the
provided codes and datasets. For the MLP model, we im-
plement a simple two-layer network with 64 hidden units,
tuning the learning rate and weight decay to match each
setup for a fair comparison. These issues highlight the need
for a new benchmark in directed link prediction, enabling
fair evaluation and supporting future research.

4. New Benchmark: DirLinkBench

In this section, we introduce a new robust benchmark for
the directed link prediction tasks, DirLinkBench, which
offers three key advantages:

7

Rethinking Link Prediction for Directed Graphs

Table 7. Statistics of DirLinkBench datasets.

Datasets

#Nodes

#Edges Avg. Degree

#Features %Directed Edges

Description

Cora-ML
CiteSeer
Photo
Computers
WikiCS
Slashdot
Epinions

2,810
2,110
7,487
13,381
11,311
74,444
100,751

8,229
3,705
143,590
287,076
290,447
424,557
708,715

5.9
3.5
38.4
42.9
51.3
11.4
14.1

2,879
3,703
745
767
300
-
-

93.97
98.00
65.81
71.23
48.43
80.17
65.04

citation network
citation network
co-purchasing network
co-purchasing network
weblink network
social network
social network

� Comprehensive coverage. DirLinkBench includes
seven real-world datasets, 16 baselines, and seven eval-
uation metrics. These datasets span diverse domains,
scales, and structural properties, and are uniformly pre-
processed to support consistent evaluation. The base-
lines cover both embedding methods and GNNs under
fair settings. Notably, to the best of our knowledge,
DirLinkBench is the first to introduce ranking-based
metrics for evaluating directed link prediction.

� Standardized evaluation. DirLinkBench addresses
the issues of existing benchmarks discussed in Sec-
tion 3.2 by redesigning the task setup. It establishes
a unified framework for dataset splitting, feature ini-
tialization, and evaluation metrics, ensuring fairness,
consistency, and reproducibility across models.

� Modular extensibility. Built on PyTorch Geometric
(PyG) (Fey & Lenssen, 2019), DirLinkBench is highly
modular and extensible, facilitating the integration of
new datasets, model architectures, and configurable
components such as feature initialization strategies and
negative sampling schemes.

Next, we provide a detailed overview of DirLinkBench, cov-
ering its datasets, task setup, baseline methods, and results.

4.1. Dataset

DirLinkBench comprises seven real-world directed graphs
from diverse domains. Specifically, the datasets include:
(1) Two citation networks, Cora-ML (McCallum et al.,
2000; Bojchevski & G�unnemann, 2018) and CiteSeer (Sen
et al., 2008), where nodes represent academic papers and
directed edges denote citation relationships. (2) Two Ama-
zon co-purchasing networks, Photo and Computers (Shchur
et al., 2018), in which nodes denote products and directed
edges represent sequential purchase behavior.
(3) Wi-
kiCS (Mernyei & Cangea, 2020), a weblink network where
nodes correspond to computer science articles on Wikipedia
and directed edges indicate hyperlinks between articles. (4)
Two social networks, Slashdot (Ordozgoiti et al., 2020) and
In Slashdot, nodes
Epinions (Massa & Avesani, 2005).

represent users and directed edges capture explicit social
interactions such as friendships or replies, while in Epinions,
directed edges encode trust relationships, offering a view
into user-to-user reliability assessments.

These datasets are publicly available and have been widely
used in tasks such as node classification (Gasteiger et al.,
2019) and link prediction (He et al., 2023). However, many
of them contain noise, such as duplicate edges, self-loops,
and isolated nodes, that can negatively impact the evaluation
of link prediction methods. To address this, we preprocess
the datasets by removing duplicate edges and self-loops.
Following the protocols in (Gasteiger et al., 2019; Shchur
et al., 2018), we also eliminate isolated nodes and retain
only the largest connected component to ensure standard-
ized evaluation conditions for link prediction. We summa-
rize the statistical characteristics of the datasets in Table 7,
where Avg. Degree indicates average node connectivity, and
%Directed Edges reflects inherent directionality.

The datasets not only originate from different domains but
also vary significantly in size. For example, Epinions con-
tains over 700,000 edges and more than 100,000 nodes.
They also differ in average node degree; CiteSeer has the
lowest average degree (3.5), while WikiCS has the high-
est (51.3). Additionally, all datasets include original node
features except Slashdot and Epinions. Compared to previ-
ous benchmarks, our collection spans multiple domains,
dataset sizes, structural characteristics, and standard-
ized pre-processing, enabling a more comprehensive evalu-
ation of link prediction methods across diverse real-world
scenarios.

4.2. Task setup

In Section 3.2, we discussed the issues of existing
benchmarks, which mainly stem from their task setup.
DirLinkBench addresses these issues through the redesigned
task setup. First, we argue that the commonly used multi-
ple subtask setup is flawed and inadequate for evaluating
directed link prediction methods. In Issue 1, we demon-
strate that under this setup, a simple MLP achieves unex-
pectedly high accuracy (80�90%) across several datasets,

8

Rethinking Link Prediction for Directed Graphs

Table 8. Performance under the Hits@100 metric (mean � standard error, %). Results ranked first, second, and third are highlighted.
�TO� indicates methods that did not complete within 24 hours, and �OOM� indicates methods that exceeded memory limits.

Cora-ML

CiteSeer

Photo

Computers

WikiCS

Slashdot

Epinions

Method

STRAP
ODIN
ELTRA

79.09
54.85
87.45

60.61
MLP
70.15
GCN
79.72
GAT
APPNP
86.02
GPRGNN 86.03

63.32
DGCN
63.21
DiGCN
DiGCNIB 80.57
76.13
DirGNN

MagNet
56.54
DUPLEX 69.00

DHYPR
DiGAE

86.81
82.06

1.57
2.53
1.48

6.64
3.01
3.07
2.88
2.73

2.59
5.72
3.21
2.85

2.95
2.52

1.60
2.51

1.33

�
�
�

�
�
�
�
�

�
�
�
�

�
�

�
�

�

69.32
63.95
84.97

70.27
80.36
85.88
83.57
88.70

68.97
70.95
85.32
76.83

65.32
73.39

92.32
83.64

1.29
2.98
1.90

3.40
3.07
4.98
4.90
2.96

3.39
4.67
3.70
4.24

3.26
3.42

3.72
3.21

�
�
�

�
�
�
�
�

�
�
�
�

�
�

�
�

�

69.16
14.13
20.63

20.91
58.77
58.06
47.51
47.60

51.61
40.17
48.26
49.15

13.89
17.94

20.93
55.05

1.44
1.92
1.93

4.18
2.96
4.03
2.51
5.09

6.33
2.38
3.98
3.62

0.32
0.66

2.41
2.36

2.35

�
�
�

�
�
�
�
�

�
�
�
�

�
�

�
�

�

51.87
12.98
14.74

17.57
43.77
40.74
32.24
38.39

39.92
27.51
32.44
35.65

12.85
17.90

�
�
�

�
�
�
�
�

�
�
�
�

2.07
1.47
1.55

0.85
1.75
3.22
1.40
2.64

1.94
1.67
1.85
1.30

0.59
0.71

�
�
TO

76.27
9.83
9.88

0.92
0.47
0.70

�
�
�

12.99
38.37
40.47
20.23
20.87

25.91
25.31
28.28
50.48

0.68
1.51
4.10
1.72
3.15

4.10
1.84
2.44
0.85

�
�
�
�
�

�
�
�
�

10.81
8.52

0.46
0.60

�
�
TO

41.55

53.79

1.62

29.21

1.56

54.67

1.36

2.50

�

�

�

�

31.43
34.17
33.44

32.97
33.16
30.16
33.76
32.61

�
�
�

�
�
�
�
�
TO
TO
TO

�

1.21
1.19
1.00

58.99
36.91
41.63

0.51
1.22
3.11
1.05
1.05

44.59
46.10
43.65
41.99
41.14

0.82
0.47
2.53

1.62
1.37
4.88
1.23
2.10

�
�
�

�
�
�
�
�
TO
TO
TO

�

41.74

1.15

50.10

2.06

1.06
2.59

31.98
18.42

�
�
OOM/TO
0.93
41.95

1.72
4.34

28.01
16.50

�
�
OOM/TO
1.96
55.14

42.42

1.15

55.91

1.77

�

�

�

�

SDGAE

90.37

93.69

3.68

68.84

Avg. Rank

?
5.57
13.57
9.29

11.43
6.14
6.29
8.14
7.43

10.86
12.14
9.14
6.43

14.29
13.14

11.29
4.71

1.43

surpassing SOTA methods in some cases. This counterintu-
itive result highlights a fundamental weakness of the multi-
ple subtask setup�it fails to distinguish between simplistic
baselines and specialized approaches. Furthermore, in Is-
sue 3, we show that this setup inherently introduces class
imbalance and limits the applicability of ranking-based met-
rics. DirLinkBench adopts a binary classification task for
directed link prediction to address these limitations: given
two nodes u and v, the task is to predict whether a directed
edge exists from u to v. Notably, this setup aligns with the
directed link prediction problem definition in Definition 3.1.
Moreover, this setting has been widely adopted in prior stud-
ies on embedding methods (Yin & Wei, 2019; Hamedani
et al., 2023), and its extension to GNNs is straightforward
and intuitive.

G

G

Second, to address label leakage (Issue 2) and unstandard-
ized data splits (Issue 4), DirLinkBench adopts a standard-
ized task setup. Specifically, given a directed graph
, we
randomly split 15% of the edges for testing, 5% for valida-
tion, and use the remaining 80% for training, while ensuring
? remains weakly connected (He
that the training graph
et al., 2023). For testing and validation, we sample an equal
number of negative edges under the full graph
visible,
? is accessible.
while for training, only the training graph
Feature inputs are provided in three forms: (1) original
node features, (2) in/out degrees computed from the training
? (Zhang et al., 2021), or (3) random feature vec-
graph
tors sampled from a standard normal distribution (Ke et al.,
2024). For fair comparisons, we generate 10 fixed splits
using random seeds, and all models are evaluated on shared

G

G

G

splits to report the mean performance. Each model learns
from the training graph and feature inputs to compute the
probability p(u, v) of test edges.

4.3. Baseline

We carefully select 15 state-of-the-art baselines, includ-
ing three embedding methods: STRAP (Yin & Wei,
2019), ODIN (Yoo et al., 2023), ELTRA (Hamedani et al.,
2023); a basic method MLP; four classic undirected GNNs:
GCN (Kipf & Welling, 2017), GAT (Veli?ckovi�c et al., 2018),
APPNP (Gasteiger et al., 2019), GPRGNN (Chien et al.,
2021); four single real-valued methods: DGCN (Tong et al.,
2020b), DiGCN (Tong et al., 2020a), DiGCNIB (Tong et al.,
2020a), DirGNN (Rossi et al., 2024); two complex-valued
methods: MagNet (Zhang et al., 2021), DUPLEX (Ke et al.,
2024); a gravity-inspired method: DHYPR (Zhou et al.,
2022); and a source-target GNN: DiGAE (Kollias et al.,
2022). Note that some recent methods, such as CoBA (Liu
et al., 2023), BLADE (Virinchi & Saladi, 2023), and ND-
DGNN (Huang et al., 2024), are not included due to unavail-
able code.

For baseline implementation, we use the original code re-
leased by the authors or widely adopted libraries such as
PyTorch Geometric (PyG) (Fey & Lenssen, 2019) and Py-
Torch Geometric Signed Directed (PyGSD) (He et al., 2023).
For methods without publicly available link prediction code
(e.g., GCN, DGCN), we implement a variety of decoders
and loss functions. For methods with official link predic-
tion code (e.g., MagNet, DiGAE), we strictly follow the
authors� reported settings. Hyperparameters are tuned via

9

Rethinking Link Prediction for Directed Graphs

Table 9. Performance across seven different metrics (mean � standard error, %) on cora-ml, photo, and slashdot datasets. The results
ranked first and second are highlighted.

Hits@20

Hits@50

Hits@100

MRR

AUC

AP

ACC

Dataset

Cora-ML

Photo

Slashdot

2.55
5.47

4.83
8.93
4.63

4.97
7.34

3.39
3.19

3.80
3.35

�
�

�
�
�

�
�

�
�

�
�

Method

STRAP
ELTRA

MLP
GAT
APPNP

67.10
70.74

29.84
33.42
55.47

DiGCNIB 45.90
42.48
DirGNN

MagNet
29.38
DUPLEX 21.73

DiGAE
SDGAE

STRAP
ELTRA

MLP
GAT
APPNP

56.13
70.89

38.54
7.09

5.20
1.23

8.83
25.97
22.13

2.06
4.22
2.60

DiGCNIB 21.42
22.59
DirGNN

2.77
2.77

MagNet
DUPLEX

5.35
7.84

0.50
1.17

�
�

�
�
�

�
�

�
�

DiGAE
SDGAE

STRAP
ELTRA

MLP
GAT
APPNP

DirGNN

MagNet
DUPLEX

DiGAE
SDGAE

27.79
40.89

19.10
18.02

14.16
14.82
15.00

20.55

3.85
3.86

1.06
2.11

5.22
2.70
5.47

2.85

�
�

�
�

�
�
�

�

12.55
5.67

0.75
1.85

23.68
23.57

0.94
2.11

�
�

�
�

30.91
19.77

11.84
11.07
22.49

17.08
13.34

7.48
5.22

3.40
3.68
6.33

3.90
5.56

�
�

�
�
�

�
�

11.19
7.74

3.04
1.95

20.53
28.45

4.21
5.82

12.08
2.22

3.15
0.59

�
�

�
�

�
�

3.18
8.62
6.56

6.82
8.72

1.62
2.53

1.06
3.07
1.14

1.53
2.08

0.39
0.66

�
�
�

�
�

�
�

9.38
14.82

2.47
4.22

�
�

9.82
5.53

4.14
5.11
5.86

7.52

2.83
1.81

5.54
8.41

�
�

�
�
�

�

�
�

�
�

1.82
1.77

1.71
1.53
2.78

3.24

0.51
0.83

1.51
3.80

87.38
94.83

89.93
94.57
95.94

94.67
93.05

85.23
88.02

92.56
97.24

98.54
96.89

95.29
99.13
98.54

98.67
98.76

88.11
94.22

97.98
99.25

94.74
94.65

95.84
96.26
96.21

96.95

96.57
94.36

95.26
96.70

0.83
0.58

2.09
0.45
0.57

0.56
0.87

0.84
0.95

0.66
0.34

0.04
0.06

0.37
0.09
0.09

0.14
0.09

0.21
0.76

0.08
0.05

0.05
0.03

0.07
0.18
0.06

0.05

0.09
3.25

0.29
0.10

90.46
95.37

88.55
92.77
95.82

94.21
92.52

86.06
86.62

93.70
97.21

98.65
95.84

93.60
98.93
98.26

98.39
98.47

87.48
93.37

97.99
99.16

95.13
95.23

96.21
96.45
96.43

97.14

96.69
94.48

96.13
97.06

0.70
0.52

2.51
0.42
0.59

0.65
0.79

0.94
1.43

0.54
0.17

0.05
0.13

1.06
0.11
0.12

0.16
0.13

0.21
0.91

0.10
0.06

0.05
0.04

0.05
0.20
0.05

0.06

0.10
2.76

0.19
0.08

�
�

�
�
�

�
�

�
�

�
�

�
�

�
�
�

�
�

�
�

�
�

�
�

�
�
�

�

�
�

�
�

�
�

�
�
�

�
�

�
�

�
�

�
�

�
�
�

�
�

�
�

�
�

�
�

�
�
�

�

�
�

�
�

78.47
85.40

81.32
88.95
89.90

87.25
85.83

77.51
82.28

86.25
91.36

94.61
91.84

88.31
96.17
94.71

95.05
95.34

80.31
87.68

91.77
96.16

88.19
88.11

89.62
88.01
90.07

0.77
0.45

2.41
0.71
0.73

0.58
0.93

0.92
0.93

0.80
0.70

0.10
0.12

0.50
0.20
0.29

0.27
0.17

0.14
0.52

0.18
0.14

0.08
0.11

0.11
0.49
0.10

�
�

�
�
�

�
�

�
�

�
�

�
�

�
�
�

�
�

�
�

�
�

�
�

�
�
�

90.65

0.13

90.45
85.42

85.67
91.05

�

�
�

�
�

0.09
3.42

0.28
0.20

75.05
81.93

44.28
58.40
75.16

66.62
59.41

43.28
34.98

72.23
83.63

55.21
12.86

14.07
42.85
35.30

34.97
34.65

1.66
2.38

5.72
6.55
4.09

3.67
4.00

3.76
2.37

2.51
2.15

2.19
1.58

2.87
4.90
2.04

2.67
3.31

�
�

�
�
�

�
�

�
�

�
�

�
�

�
�
�

�
�

9.04
12.64

0.52
1.22

�
�

43.32
55.76

25.39
26.31

24.01
22.19
24.34

31.20

22.34
11.49

33.97
33.75

3.36
4.08

1.43
0.95

0.79
2.78
1.47

1.18

0.42
3.36

1.06
1.48

�
�

�
�

�
�
�

�

�
�

�
�

79.09
87.45

60.61
79.72
86.02

80.57
76.13

56.54
69.00

82.06
90.37

69.16
20.63

20.91
58.06
47.51

48.26
49.15

13.89
17.94

55.05
68.84

31.43
33.44

32.97
30.16
33.76

41.74

31.98
18.42

41.95
42.42

�
�

�
�
�

�
�

�
�

�
�

�
�

�
�
�

�
�

�
�

�
�

�
�

�
�
�

�

�
�

�
�

1.57
1.48

6.64
3.07
2.88

3.21
2.85

2.95
2.52

2.51
1.33

1.44
1.93

4.18
4.03
2.51

3.98
3.62

0.32
0.66

2.36
2.35

1.21
1.00

0.51
3.11
1.05

1.15

1.06
2.59

0.93
1.15

10

Rethinking Link Prediction for Directed Graphs

?

?

tv

tv

u tv

tv
?

sv
?

(cid:1), and LR(cid:0)su

grid search, adhering to the configurations specified in each
paper. Specifically, for STRAP, ODIN, and ELTRA, we
consider four decoder options: ?(cid:0)s?
(cid:1), LR(cid:0)su
(cid:1),
LR(cid:0)su
(cid:1) (Yoo et al., 2023). Em-
tu
?
bedding generation for these methods also follows the pa-
rameter settings provided in their respective papers. For
MLP, GCN, GAT, APPNP, GPRGNN, DGCN, DiGCN,
DiGCN-IB, and DirGNN, we consider two commonly used
loss functions: cross-entropy (CE) loss (Zhang et al., 2021;
Ke et al., 2024) and binary cross-entropy (BCE) loss (Kol-
lias et al., 2022; Zhou et al., 2022), and three decoders:
MLP(hu
u hv). For Mag-
Net, DUPLEX, DHYPR, and DiGAE, we adopt the loss
functions and decoders as reported in their original imple-
mentations. For all methods, we train for up to 2000 epochs
with an early stopping criterion of 200 epochs. We repeat
each experiment 10 times and report the mean and standard
deviation. More detailed hyperparameters for each baseline
are provided in the Appendix.

hv), and ?(h?

hv), MLP(hu

?

?

4.4. Result

For evaluation, we introduce ranking-based metrics to di-
rected link prediction benchmarks for the first time, includ-
ing Hits@20, Hits@50, Hits@100, and MRR, which are
widely used in undirected link prediction benchmarks (Hu
et al., 2020; Li et al., 2023). In addition, we report AUC, AP,
and ACC for comparative purposes. A detailed description
of all metrics is provided in the Appendix. For each method,
we report the best mean result across different combinations
of feature inputs, loss functions, and decoders. Table 8
shows the results under the Hits@100 metric, while Table 9
shows the results across all metrics for some datasets. Com-
plete results for all metrics are included in the supplemental
material due to space constraints.

The results reveal that ACC, AUC, and AP offer limited dis-
criminatory ability across different baselines. For example,
some simple undirected GNNs (e.g., GAT, APPNP) achieve
competitive performance, with only minor performance
gaps across methods. Conversely, methods that perform
well under ranking-based metrics (e.g., STRAP, DiGAE)
tend to perform relatively poorly. These findings suggest
that ACC, AUC, and AP are inadequate for reliably eval-
uating directed link prediction performance, aligning with
recent discussions about their limitations even in undirected
settings (Yang et al., 2015; Li et al., 2023). Therefore, we
argue that ranking-based metrics are better suited for the
link prediction task. Since Hits@100 is widely used in pop-
ular benchmarks (e.g., Open Graph Benchmark (OGB) (Hu
et al., 2020)) and reveals significant performance differences
among methods, we adopt it as our primary evaluation met-
ric.

From the results in Table 8, we observe that embedding

Figure 3. The bipartite graph representation of two toy directed
graphs.

methods maintain a strong advantage, even without feature
inputs. Early single real-valued undirected and directed
GNNs also demonstrate competitive performance. In con-
trast, several newer directed GNNs (e.g., MagNet (Zhang
et al., 2021), DUPLEX (Ke et al., 2024), DHYPR (Zhou
et al., 2022)) exhibit weaker performance or face scala-
bility challenges. We provide a deeper analysis of these
observations in Section 6. Interestingly, the simple directed
graph autoencoder DiGAE (Kollias et al., 2022) is the top-
performing method overall, but it underperforms on specific
datasets (e.g., Cora-ML, CiteSeer). This finding motivates
us to revisit DiGAE�s design and propose improved methods
to enhance directed link prediction.

5. New Method: SDGAE

In this section, we revisit DiGAE�s model architecture to
better understand its encoder�s graph convolution mecha-
nism. We then introduce a novel method: Spectral Directed
Graph Auto-Encoder (SDGAE).

5.1. Understand the Graph Convolution of DiGAE

DiGAE (Kollias et al., 2022) is a graph auto-encoder de-
signed for directed graphs. Its encoder�s graph convolutional
layer is denoted as

S(?+1) = ?

T(?+1) = ?

out

(cid:16) �D??
(cid:16) �D??

in

�A �D??

(cid:17)

in T(?)W(?)
outS(?)W(?)

T

S

,
(cid:17)

�A? �D??

(5)

(6)

.

Here, �A = A + I denotes the adjacency matrix with added
self-loops, and �Dout and �Din represent the corresponding
out-degree and in-degree matrices, respectively. S(?) and
T(?) denote the source and target embeddings at the ?-th
layer, initialized as S(0) = T(0) = X. The hyperparame-
ters ? and ? are degree-based normalization factors, ? is
the activation function (e.g., ReLU), and W(?)
S rep-
resents the learnable weight matrices. The design of this
graph convolution is inspired by the connection between

T , W(?)

11

132123456123456132123456123456(a)(b)(c)(d)(e)(f)Rethinking Link Prediction for Directed Graphs

GCN (Kipf & Welling, 2017) and the 1-WL (Weisfeiler &
Leman, 1968) algorithm for directed graphs. However, the
underlying principles of this convolution remain unexplored,
leaving its meaning unclear. Additionally, the heuristic hy-
perparameters ? and ? introduce significant challenges for
effective parameter optimization.

We revisit the graph convolution of DiGAE�s encoder and
observe that it corresponds to the GCN convolution (Kipf
& Welling, 2017) applied to an undirected bipartite graph.
Specifically, given a directed graph
with adjacency matrix
A, the adjacency matrix of its bipartite representation (Bang-
Jensen & Gutin, 2008) is defined as:
(cid:21)

G

( �A) :=

S

(cid:20) 0

�A
�A? 0

R2n�2n.

?

(7)

S

( �A) is a block matrix consisting of �A and �A?. It
Here,
S
( �A) represents the adjacency matrix of an
is evident that
undirected bipartite graph with 2n nodes. Figure 3 provides
two toy examples of directed graphs and their corresponding
undirected bipartite graph representations. Notably, the self-
loops in �A serve a fundamentally different purpose than in
GCN (Kipf & Welling, 2017). In this context, the self-loops
ensure the connectivity of the undirected bipartite graph.
Without the self-loops, as illustrated in graphs (b) and (e)
of Figure 3, the graph structure suffers from significant
connectivity issues. Based on these insights, we present the
following Lemma 5.1.
Lemma 5.1. If omitting degree-based normalization in Eqs.
(5) and (6), the graph convolution of DiGAE�s encoder is

(cid:2)S(?+1), T(?+1)(cid:3)?

= ?

(cid:18)

S

(cid:104)

( �A)

S(?)W(?)

S , T(?)W(?)

T

(cid:105)?(cid:19)

.

(8)

The proof and its extension with degree-based normalization
are provided in the Appendix. Lemma 5.1 and the extension
in its proof reveal that the graph convolution of DiGAE�s
encoder essentially corresponds to the GCN convolution
applied to an undirected bipartite graph.

5.2. Spectral Directed Graph Auto-Encoder (SDGAE)

Building on the understanding of DiGAE, we identify its
three key limitations: (1) DiGAE struggles to use deep
networks for capturing rich structural information due to
excessive learnable weight matrices, a problem also existing
in deep GCN (Peng et al., 2024). We provide experimen-
tal evidence for this in Section 6.1. (2) From the spectral
perspective, DiGAE uses a fixed low-pass filter and can-
not learn arbitrary spectral graph filters (Zhu et al., 2021a),
similar to GCN, since both rely on the same convolutional
formulation. (3) DiGAE relies on heuristic hyperparameters
? and ?, which are difficult to tune effectively.

To overcome these limitations, we propose SDGAE, which
uses a polynomial approximation to learn arbitrary graph

Figure 4. Performance comparison of SDGAE, DiGAE, and Di-
GAE with residual connections (i.e., DiGAE?) on the Cora-ML
and CiteSeer datasets, with varying numbers of convolutional lay-
ers or polynomial orders.
filters for directed graphs. SDGAE is inspired by spectral-
based undirected GNNs (Chien et al., 2021; He et al., 2021)
and incorporates symmetric normalization of the directed
adjacency matrix. The propagation process of SDGAE�s
encoder is defined as follows:

(cid:20) S(K)
T(K)

(cid:21)

=

K
(cid:88)

k=0

W(k)

(cid:20) 0

�A
�A? 0

(cid:21)k (cid:20) S(0)
T(0)

(cid:21)

.

(9)

?

T ?

T In])

?
�A �D?1/2
in

S In, w(k)

Here, W(k) = diag([w(k)
R2n�2n repre-
sents the diagonal coefficients matrix and w(k)
S , w(k)
R
are the corresponding polynomial coefficients. The pa-
N+ is the polynomial order and �A =
rameter K
�D?1/2
denotes the normalization of the directed
out
adjacency matrix �A. Below Lemma 5.2 (see the Appendix
for the proof) shows that this normalization is equivalent
( �A). In this Equation,
to the symmetric normalization of
S(0) and T(0) represent the initial source and target em-
beddings, while S(K) and T(K) denote the corresponding
embeddings after K propagation steps.

S

Lemma 5.2. The symmetrically normalized block adja-
cency matrix D?1/2
( �A)D?1/2
( �A), where DS =
S
diag( �Dout, �Din) is the diagonal degree matrix of

( �A).

=

S

S

S

S

Spectral Analysis.
Intuitively, SDGAE�s encoder is a
polynomial spectral-based GNN defined on an undirected
bipartite graph with 2n nodes, where the normalized ad-
( �A). Equation (9) aligns
jacency matrix is given by
S
with the propagation process of polynomial spectral-based
GNNs presented in Equation (1): Z = (cid:80)K
k=0 wkPkX. Un-
like the undirected methods, SDGAE maintains separate
source and target embeddings, with corresponding poly-
nomial coefficients w(k)
S and w(k)
T . SDGAE is capable of
approximating arbitrary graph filters in the spectral domain.
( �A)
Specifically, we denote the eigendecomposition of

S

12

12345678LorK102030405060708090100Hit@100(%)DiGAE?(Cora-ML)DiGAE?(CiteSeer)DiGAE(Cora-ML)DiGAE(CiteSeer)SDGAE(Cora-ML)SDGAE(CiteSeer)Rethinking Link Prediction for Directed Graphs

S

( �A) = US ?S U?

R2n�2n are the
S , where US , ?S
as
eigenvector and diagonal eigenvalue matrices, respectively.
The spectral graph filter of SDGAE�s encoder is then ex-
pressed as:

?

( �A)) = US

h(

S

(cid:32) K
(cid:88)

k=0

(cid:33)

W(k)?k
S

U?
S .

(10)

By learning the coefficients w(k)
S
approximate arbitrary graph filters h(
graphs.

S

and w(k)

T , SDGAE can
( �A)) on directed

Implementation. In the implementation of SDGAE, we use
two MLPs to initialize the source and target embeddings,
i.e., S(0) = MLPS(X) and T(0) = MLPT (X), following
the implementation of many spectral-based GNNs (Chien
et al., 2021; He et al., 2021). For polynomial coefficients,
and w(k)
we empirically find that directly learning w(k)
T
S
proves challenging, as different initialization strategies sig-
nificantly impact the results. One possible explanation is
that the coefficients must satisfy convergence constraints
in polynomial approximations of filter functions (He et al.,
2022). To address this, we instead adopt an iterative for-
mulation that implicitly learns the coefficients through the
propagation process:

S(k+1) = ?(k)
S
T(k+1) = ?(k)

T

�AT(k) + S(k),
�A?S(k) + T(k).

(11)

(12)

T

S and ?(k)

where ?(k)
are learnable scalar weights, initialized
to one. These weights express the polynomial coefficients
in Equation (9) through the iterative propagation process.
For example, when the polynomial order is set to K = 2,
the output can be expanded as:

S(2) = S(0) +

T(2) = T(0) +

(cid:16)

S

S + ?(0)
?(1)
(cid:16)
T + ?(0)
?(1)

T

(cid:17) �AT(0) + ?(1)
(cid:17) �A?S(0) + ?(1)

S ?(0)
T ?(0)

T

S

�A �A?S(0), (13)

�A? �AT(0). (14)

T

T = 1, w(1)

S + ?(0)
T , w(2)

S = ?(1)
T ?(0)

S , w(2)
T = ?(1)

S = 1, w(1)
T = ?(1)

This expansion means the equivalent polynomial coeffi-
S ?(0)
S = ?(1)
cients are: w(0)
and w(0)
T + ?(0)
S . No-
tably, the coefficients w(k) tend to decrease at higher or-
ders. This is because the learned values of ?(k) generally
lie within the range (0, 1), and the product of multiple ?(k)
terms at higher orders causes w(k) to diminish. The co-
efficients w(k) learned by SDGAE on real-world datasets,
shown in Figure 5, exhibit this trend. This convergence
property contributes to the efficiency of learning polynomial
filters.

For the experimental setting of SDGAE, we aligned its con-
figuration with that of other baselines in DirLinkBench to
ensure a fair comparison. Regarding the loss function and

13

Figure 5. Polynomial coefficients learned by SDGAE on the Cora-
ML and CiteSeer datasets with K = 5.
decoders, SDGAE uses BCE loss with three decoder op-
tv). For the
tions: ?(su
?
3, 4, 5
polynomial order K, we search over the set
. The
}
{
MLP architecture, including the number of layers and hid-
den units, is consistent with the baseline methods. Detailed
hyperparameter settings are provided in the Appendix.

tv), MLP(su
?

tv), and MLP(su

?

Time Complexity. The computational time complexity of
SDGAE primarily stems from its propagation process, as
defined in Equation (9). Based on the iterative implementa-
tion shown in Equations (11) and�(12), the time complexity
of SDGAE�s propagation is O(2Kmd), where m is the
number of edges, K is the polynomial order, and d is the
embedding dimension. This complexity scales linearly with
the number of edges m. In contrast, DiGAE has a higher
time complexity of O(2Lmd + Lnd2), where L is the num-
ber of layers and n is the number of nodes. This is due to
the use of multiple learnable weight matrices, i.e., W(?)
S
and W(?)
T as defined in Equations (5) and (6). So, SDGAE
achieves lower time complexity than DiGAE and shows
faster training performance.

Results. We report the directed link prediction results of
SDGAE on DirLinkBench under the Hits@100 metric in
Table 8, and across seven different evaluation metrics in
Table 9. Comprehensive results are also provided in the sup-
plemental material. As shown in Table 8, SDGAE achieves
the best performance on 4 out of the 7 datasets under the
Hits@100 metric and delivers competitive results on the
remaining three. Moreover, as shown in Table 9, SDGAE
ranks among the top two methods across all seven evalua-
tion metrics. Notably, SDGAE significantly outperforms
both DiGAE (Kollias et al., 2022) and the spectral-based
method GPRGNN (Chien et al., 2021), which is designed
for undirected graphs. These results collectively demon-
strate the substantial effectiveness of SDGAE for the task
of directed link prediction.

6. Analysis

In this section, we first analyze SDGAE with respect to the
choice of polynomial order K and the learned polynomial
coefficients, aiming to understand the reasons behind its im-
proved performance compared to DiGAE. Next, we conduct

012345k0246810w(k)Cora-MLw(k)Sw(k)T012345k0246810w(k)CiteSeerw(k)Sw(k)TRethinking Link Prediction for Directed Graphs

(a) Cora-ML

(b) Photo

(c) WikiCS

(d) Slashdot

Figure 6. Performance comparison of various GNN methods using original features, in/out degrees, or random features as input on four
datasets.

Table 10. Performance of DiGAE and SDGAE under different
adjacency matrix normalizations. Best results are highlighted in
bold.

Methods

Cora-ML

CiteSeer

Photo

Computers

DiGAE(?,?)
DiGAE(?1/2,?1/2)
SDGAE(?1/2,?1/2)

82.06�2.51
70.89�3.59
90.37�1.33

83.64�3.21
71.60�6.21
93.69�3.68

55.05�2.36
38.75�5.20
68.84�2.35

41.55�1.62
32.73�5.28
53.79�1.56

a series of experimental investigations on various aspects of
directed link prediction methods, including feature inputs,
loss functions and decoders, degree distribution, and nega-
tive sampling strategies, to offer new insights and highlight
open challenges in this field.

6.1. Comparative Analysis of SDGAE and DiGAE

To investigate the performance differences between SDGAE
and DiGAE in utilizing higher neighborhood information,
Figure 4 shows how SDGAE�s performance changes with
increasing polynomial order K, and how the performance
of DiGAE and DiGAE? (DiGAE with residual connections)
changes with increasing numbers of convolutional layers L.
The results on both Cora-ML and CiteSeer datasets show
that SDGAE�s performance consistently improves with in-
creasing K, achieving optimal results at K = 5. This trend
aligns with the theoretical advantages of using polynomial
filter approximations (Chien et al., 2021). In contrast, the
performance of DiGAE and DiGAE? declines as the number
of layers increases. This is attributed to DiGAE�s essentially
GCN-like architecture, which suffers from well-known is-

sues such as over-smoothing and difficulty training deeper
networks (Peng et al., 2024). Notably, although the itera-
tive formulation of SDGAE in Equations (11) and (12) may
superficially resemble the addition of residual connections,
it fundamentally differs by learning a polynomial filter as
shown in Equation (9). This fact is also supported by the
experiments with DiGAE?, where simply adding residual
connections fails to improve DiGAE�s performance. In sum-
mary, SDGAE effectively uses higher neighborhood infor-
mation by learning polynomial filter coefficients, enabling
it to achieve superior performance as K increases.

We show in Figure 5 the polynomial filter coefficients w(k)
S
and w(k)
learned by SDGAE on Cora-ML and CiteSeer.
T
These coefficients are computed from the learned weights
?(k)
S and ?(k)
T . We observe that the coefficients are largest
at k = 2 and gradually decrease at higher orders (e.g.,
k = 3, 4, 5), which aligns with the expected behavior of
polynomial expansions. SDGAE learns distinct sets of coef-
ficients for different datasets, effectively adapting its polyno-
mial filters to the underlying graph structure. This flexibility
contrasts with DiGAE, which cannot learn specific filter
functions.

Next, we analyze the impact of adjacency matrix normal-
ization on DiGAE�s performance. Table 10 presents the
results of DiGAE and SDGAE under different normaliza-
tion strategies. Here, DiGAE(?,?) refers to the original
settings used in the its paper (Kollias et al., 2022), where the
�A �D??
adjacency matrix is normalized as �D??
in , with (?, ?)
out
2. Notably,
0.0, 0.2, 0.4, 0.6, 0.8
searched over the grid
}
{

14

MLPGCNDiGCNIBDirGNNMagNetDHYPRDiGAESDGAE30405060708090100Hits@100(%)OriginalfeatureIn/outdegreeMLPGCNDiGCNIBDirGNNMagNetDHYPRDiGAESDGAE01020304050607080Hits@100(%)OriginalfeatureIn/outdegreeMLPGCNDiGCNIBDirGNNMagNetDiGAESDGAE0102030405060Hits@100(%)OriginalfeatureIn/outdegreeMLPGCNDirGNNMagNetDiGAESDGAE01020304050Hits@100(%)RandomfeatureIn/outdegreeRethinking Link Prediction for Directed Graphs

(a) Cora-ML

(b) Slashdot

(a) MagNet

(b) DiGAE

Figure 7. Performance comparison of three embedding methods
with four different decoders on the Cora-ML and Slashdot datasets.

(a) Cora-ML

(b) Photo

�A �D?1/2
in

Figure 8. Performance comparison of GNNs with various loss func-
tions and decoders on Cora-ML and Photo datasets.
this search space excludes the symmetric normalization
�D?1/2
, which is used in both DiGAE(?1/2,?1/2)
out
and SDGAE(?1/2,?1/2). The results show that applying
symmetric normalization to DiGAE does not improve per-
formance, suggesting that SDGAE�s performance gains are
not solely attributable to its normalization choice. Further-
more, as shown in Figure 9(b), modifying DiGAE�s decoder
can improve its performance on certain datasets. However,
DiGAE fails to achieve results comparable to SDGAE even
with these modifications.

These comparative experiments demonstrate that SDGAE
significantly outperforms DiGAE, primarily due to its theo-
retically grounded design based on polynomial graph filters.
By learning adaptive polynomial filters, SDGAE can better
capture structural patterns across different datasets.

6.2. Feature Input

We examine the impact of different feature inputs on GNN
performance in directed link prediction. Figure 6 shows
the results of various GNN methods using original features,
in/out degrees, or random features as inputs across four
datasets. Specifically, Figures 6(a), 6(b), and 6(c) compare
the performance of GNNs using original features versus
in/out degrees on Cora-ML, Photo, and WikiCS, respec-
tively. Figure 6(d) presents a similar comparison using
in/out degrees and random features on Slashdot, which lacks
original node features.

The results indicate that original features enhance GNN
performance on certain datasets (e.g., Cora-ML and Photo).
However, for datasets like WikiCS, in/out degrees are more

Figure 9. Performance comparison of MagNet and DiGAE with
different loss functions and decoders.
effective. For datasets without original features (e.g., Slash-
dot and Epinions), in/out degrees significantly outperform
random features. These findings highlight the critical role
of appropriate feature inputs in improving GNN per-
formance on directed link prediction tasks. Enhancing
feature quality remains an important direction for future
research, particularly for datasets with weak or missing
original features.

6.3. Loss Function and Decoder

?

?

sv
?

tu
?
tv), and inner product ?(s?

We analyze the impact of different decoders and loss func-
tions on the performance of directed link prediction meth-
ods. For the embedding methods STRAP, ELTRA, and
ODIN, we compare their performance with four decoders:
tv), extended
logistic regression with concatenation LR(su
?
tv), element-wise product
concatenation LR(su
LR(su
(Hamedani
et al., 2023; Yin & Wei, 2019; Yoo et al., 2023). Results
on the Cora-ML and Slashdot datasets are shown in Fig-
ure 7. For single real-valued GNN methods, MLP, GCN,
DiGCN-IB, and DirGNN, we evaluate different combina-
tions of loss functions and decoders on the Cora-ML and
Photo datasets, as shown in Figure 8. The settings include
hv), and BCE loss with three de-
CE loss with MLP(hu
?
coders: MLP(hu
hv), and inner product
?(h?

hv), MLP(hu

u tv)

?

?

u hv).

The results highlight the significant impact of both the de-
coder and loss function on model performance. For em-
bedding methods, even with fixed embeddings, different
decoders result in substantial performance variations. No-
tably, ELTRA and ODIN exhibit high sensitivity to decoder
choices on the Slashdot dataset. For GNNs, the results show
that BCE loss offers a consistent advantage over CE loss.

Additionally, we compare the performance of MagNet and
DiGAE under different loss functions and decoders. In Fig-
ure 9(a), we show that BCE loss consistently outperforms
CE loss for MagNet, underscoring the limitations of prior
approaches that rely on CE loss (Zhang et al., 2021; He
et al., 2023; Ke et al., 2024; Li et al., 2024). Since link pre-
diction is fundamentally a binary classification task, BCE
loss is more appropriate, consistent with findings from undi-

15

STRAPELTRAODIN2030405060708090100Hits@100(%)LR(suktv)LR(suksvktuktv)LR(su(cid:12)tv)?(s>utv)STRAPELTRAODIN01020304050Hits@100(%)LR(suktv)LR(suksvktuktv)LR(su(cid:12)tv)?(s>utv)MLPGCNDiGCNIBDirGNN30405060708090100Hits@100(%)CE+MLP(hukhv)BCE+MLP(hukhv)BCE+MLP(hu(cid:12)hv)BCE+?(h>uhv)MLPGCNDiGCNIBDirGNN01020304050607080Hits@100(%)CE+MLP(hukhv)BCE+MLP(hukhv)BCE+MLP(hu(cid:12)hv)BCE+?(h>uhv)Cora-MLCiteSeerPhotoComputers01020304040506070Hits@100(%)CEBCECora-MLCiteSeerPhotoComputers3040405060708090100Hits@100(%)InnerProductMLPScoreRethinking Link Prediction for Directed Graphs

Figure 10. Degree distribution of WikiCS graph and its reconstruction graph generated by four GNNs using the original node feature as
feature inputs.

Figure 11. Degree distribution of WikiCS�s reconstruction graph generated by STRAP and four GNNs using the in/out degrees as feature
inputs.

rected settings (Li et al., 2023). In Figure 9(b), we compare
two decoders for DiGAE, inner product and MLP score,
and observe that the MLP-based decoder achieves superior
performance across three datasets.

These findings lead to two key insights: (1) Decoder design
significantly affects model performance, and (2) BCE
loss is better suited for link prediction tasks than CE
loss. Furthermore, the poor performance of complex-valued
methods (e.g., MagNet and DUPLEX) may be partly at-
tributed to their reliance on CE loss and suboptimal decoder
choices.

6.4. Degree Distribution

We evaluate how well different models preserve the asym-
metry of directed graphs by analyzing their degree distribu-
tions. Following STRAP (Yin & Wei, 2019), we compute
the predicted probability for every edge and select the top
m? edges, where m? is the number of edges in the training
graph, to reconstruct the graph. Figure 10 compares the true
in-/out-degree distributions of the WikiCS training graph
with those of reconstructed graphs generated by four GNNs
(DirGNN (Rossi et al., 2024), MagNet (Zhang et al., 2021),
DiGAE (Kollias et al., 2022), and SDGAE), using original
node features as input. Figure 11 presents a similar compar-
ison, but the reconstructed graphs are generated by STRAP
and the same four GNNs using in/out degrees as feature
inputs.

The results show that STRAP and SDGAE most accurately
preserve the degree distributions, with STRAP performing

hv

exceptionally well in capturing in-degree components, ex-
plaining its strong performance on WikiCS. And DirGNN,
(cid:1), produces identical in-
using the decoder MLP(cid:0)hu
?
/out-degree distributions but still captures in-degree com-
ponents correctly. In contrast, MagNet fails to learn mean-
ingful degree distributions, resulting in poor performance.
Moreover, when GNNs are provided with in/out degrees
as input features, they better preserve the degree distribu-
tion, aligning with their improved WikiCS performance (see
supplemental material for detailed results). Finally, a di-
rect comparison between DiGAE and SDGAE reveals that
SDGAE more effectively maintains the degree distribution,
further demonstrating its advantage.

These findings reinforce the importance of preserving
asymmetry in directed link prediction, as discussed in
Section 3.1. They also highlight an underexplored chal-
lenge: the need for GNNs to better preserve in-/out-degree
distributions, a task that embedding methods like STRAP
currently handle more effectively.

6.5. Negative Sampling Strategy

We present a performance comparison of various GNNs
trained using different negative sampling strategies on the
Cora-ML and CiteSeer datasets in Table 11. Results are
reported using the Hits@100 metric, with the best results
highlighted in bold. In this context, �each run� strategy
refers to the default setting in DirLinkBench, where a ran-
dom negative sample is generated for each run and shared
across all models. In contrast, �each epoch� represents a

16

100101102103Degree100101102103FrequencyWikiCSindegreeoutdegree100101102103104Degree100101102103FrequencyWikiCS-DirGNN(F)indegreeoutdegree100101102103104Degree100101102103FrequencyWikiCS-MagNet(F)indegreeoutdegree100101102103Degree100101102103FrequencyWikiCS-DiGAE(F)indegreeoutdegree100101102103104Degree100101102103FrequencyWikiCS-SDGAE(F)indegreeoutdegree100101102103Degree100101102103FrequencyWikiCS-STRAPindegreeoutdegree100101102103Degree100101102103FrequencyWikiCS-DirGNN(D)indegreeoutdegree100101102103104Degree100101102103FrequencyWikiCS-MagNet(D)indegreeoutdegree100101102103Degree100101102103FrequencyWikiCS-DiGAE(D)indegreeoutdegree100101102103Degree100101102103FrequencyWikiCS-SDGAE(D)indegreeoutdegreeRethinking Link Prediction for Directed Graphs

Table 11. Performance comparison of various GNNs using different negative sampling strategies on the Cora-ML and CiteSeer datasets.
Results are reported under the Hits@100 metric, with the best results highlighted in bold.

Dataset

Sample

MLP

GCN

DiGCNIB

DirGNN

MagNet

DiGAE

SDGAE

Cora-ML

CiteSeer

each run
each epoch

each run
each epoch

60.61
34.15

70.27
66.92

6.64
3.54

3.40
6.50

�
�

�
�

70.15
59.86

80.36
69.48

�
�

�
�

3.01
9.89

3.07
6.60

80.57
56.74

85.32
61.69

3.21
4.08

3.70
6.87

�
�

�
�

76.13
49.89

2.85
3.59

�
�

76.83
53.8

4.24
�
12.41

�

56.54
54.79

65.32
70.56

2.95
2.98

3.26
2.06

�
�

�
�

82.06
79.76

83.64
87.32

2.51
3.28

3.21
3.79

�
�

�
�

90.37
89.71

93.69
92.12

1.33
2.36

3.68
3.96

�
�

�
�

strategy where different models randomly sample negative
edges in each training epoch. In both settings, the posi-
tive sample splits and test sets remain consistent for fair
comparison.

The results demonstrate that the choice of negative sampling
strategy during training can significantly affect model per-
formance, particularly for single real-valued GNNs, where
performance declines noticeably under the �each epoch�
strategy. In contrast, the �each run� strategy tends to im-
prove performance across most GNN models. These find-
ings underscore the importance of further research into neg-
ative sampling techniques. For example, heuristic-based
approaches have been proposed for undirected graphs as
alternatives to random sampling (Li et al., 2023), and simi-
lar methods could be explored for directed graphs in future
work.

7. Conclusion

This paper presents a unified framework for evaluating the
expressiveness of directed link prediction methods, em-
phasizing the theoretical importance of dual embeddings
and decoder design. To address the lack of standardized
benchmarks in this area, we introduce DirLinkBench, a
new robust benchmark featuring diverse real-world directed
graphs, standardized data splits, varied feature initializa-
tion strategies, and comprehensive evaluation metrics, in-
cluding ranking-based metrics introduced to this task. Us-
ing DirLinkBench, we find that current methods often per-
form inconsistently across datasets, and that simple design
choices, such as feature inputs, loss functions, decoders, and
negative sampling, can significantly impact performance.
Then we revisit the DiGAE model, showing its graph convo-
lution is theoretically equivalent to GCN on an undirected bi-
partite graph. Building on this insight, we propose SDGAE,
a novel Spectral Directed Graph Auto-Encoder that uses
polynomial approximation to learn graph filters for directed
graphs. SDGAE achieves state-of-the-art average perfor-
mance and better preserves directed structural properties.

Our findings highlight two key challenges for future re-
search: (1) How can more expressive and efficient de-
coders be developed, especially for complex-valued meth-
ods? (2) How can GNN architectures better capture and

preserve asymmetry, such as in- and out-degree distri-
butions? We believe that DirLinkBench, along with our
proposed SDGAE and the insights presented in this work,
will serve as a foundation for advancing the field of directed
link prediction. We hope this benchmark encourages the
development of more robust, expressive, and theoretically
grounded methods. The complete benchmark and imple-
mentation are provided in the Appendix and source code to
facilitate future research.

References

Bang-Jensen, J. and Gutin, G. Z. Digraphs: theory, algo-
rithms and applications. Springer Science & Business
Media, 2008.

Bhuyan, M. H., Bhattacharyya, D. K., and Kalita, J. K.
Network anomaly detection: methods, systems and tools.
Ieee communications surveys & tutorials, 16(1):303�336,
2013.

Bo, D., Wang, X., Liu, Y., Fang, Y., Li, Y., and Shi, C. A
survey on spectral graph neural networks. arXiv preprint
arXiv:2302.05631, 2023.

Bojchevski, A. and G�unnemann, S. Deep gaussian em-
bedding of graphs: Unsupervised inductive learning via
ranking. In ICLR, 2018.

Chien, E., Peng, J., Li, P., and Milenkovic, O. Adaptive
universal generalized pagerank graph neural network. In
ICLR, 2021.

Defferrard, M., Bresson, X., and Vandergheynst, P. Con-
volutional neural networks on graphs with fast localized
spectral filtering. NeurIPS, 29, 2016.

Fey, M. and Lenssen, J. E. Fast graph representation learning
with pytorch geometric. arXiv preprint arXiv:1903.02428,
2019.

Fiorini, S., Coniglio, S., Ciavotta, M., and Messina, E. Sig-
In AAAI, pp.

manet: One laplacian to rule them all.
7568�7576, 2023.

Gasteiger, J., Bojchevski, A., and G�unnemann, S. Predict
then propagate: Graph neural networks meet personalized
pagerank. In ICLR, 2019.

17

Rethinking Link Prediction for Directed Graphs

Geisler, S., Li, Y., Mankowitz, D. J., Cemgil, A. T.,
G�unnemann, S., and Paduraru, C. Transformers meet
In ICML, pp. 11144�11172. PMLR,
directed graphs.
2023.

Leskovec, J. and Sosi?c, R. Snap: A general-purpose network
analysis and graph-mining library. ACM Transactions
on Intelligent Systems and Technology (TIST), 8(1):1�20,
2016.

Golub, G. H. and Van Loan, C. F. Matrix computations.

JHU press, 2013.

Hamedani, M. R., Ryu, J.-S., and Kim, S.-W. Eltra: An
embedding method based on learning-to-rank to preserve
asymmetric information in directed graphs. In CIKM, pp.
2116�2125, 2023.

He, M., Wei, Z., Xu, H., et al. Bernnet: Learning arbi-
trary graph spectral filters via bernstein approximation.
NeurIPS, 34:14239�14251, 2021.

He, M., Wei, Z., and Wen, J.-R. Convolutional neural net-
works on graphs with chebyshev approximation, revisited.
NeurIPS, 35:7264�7276, 2022.

He, Y., Zhang, X., Huang, J., Rozemberczki, B., Cucuringu,
M., and Reinert, G. Pytorch geometric signed directed:
A software package on graph neural networks for signed
and directed graphs. In LoG, 2023.

Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B.,
Catasta, M., and Leskovec, J. Open graph benchmark:
Datasets for machine learning on graphs. NeurIPS, 33:
22118�22133, 2020.

Huang, J., Mo, Y., Hu, P., Shi, X., Yuan, S., Zhang, Z., and
Zhu, X. Exploring the role of node diversity in directed
graph representation learning. In IJCAI, pp. 2072�2080,
2024.

Katz, L. A new status index derived from sociometric anal-

ysis. Psychometrika, 18(1):39�43, 1953.

Ke, Z., Yu, H., Li, J., and Zhang, H. DUPLEX: Dual GAT
for complex embedding of directed graphs. In ICML, pp.
23430�23448, 2024.

Khosla, M., Leonhardt, J., Nejdl, W., and Anand, A. Node
representation learning for directed graphs. In ECML
PKDD, pp. 395�411. Springer, 2020.

Kipf, T. N. and Welling, M. Semi-supervised classification

with graph convolutional networks. In ICLR, 2017.

Koke, C. and Cremers, D. Holonets: Spectral convolutions

do extend to directed graphs. In ICLR, 2024.

Li, J., Shomer, H., Mao, H., Zeng, S., Ma, Y., Shah, N., Tang,
J., and Yin, D. Evaluating graph neural networks for link
prediction: Current pitfalls and new benchmarking. In
NeurIPS, 2023.

Li, X., Liao, M., Wu, Z., Su, D., Zhang, W., Li, R.-H., and
Wang, G. Lightdic: A simple yet effective approach for
large-scale digraph representation learning. Proceedings
of the VLDB Endowment, 17(7):1542�1551, 2024.

Liben-Nowell, D. and Kleinberg, J. The link prediction
In CIKM, pp. 556�559,

problem for social networks.
2003.

Lin, L. and Gao, J. A magnetic framelet-based convolutional
neural network for directed graphs. In ICASSP, pp. 1�5.
IEEE, 2023.

Liu, L., Chen, K.-J., and Liu, Z. Collaborative bi-
aggregation for directed graph embedding. Neural Net-
works, 164:707�718, 2023.

Maskey, S., Paolino, R., Bacho, A., and Kutyniok, G. A
fractional graph laplacian approach to oversmoothing. In
NeurIPS, 2023.

Massa, P. and Avesani, P. Controversial users demand local
trust metrics: An experimental study on epinions. com
community. In AAAI, volume 1, pp. 121�126, 2005.

McCallum, A. K., Nigam, K., Rennie, J., and Seymore, K.
Automating the construction of internet portals with ma-
chine learning. Information Retrieval, 3:127�163, 2000.

Mernyei, P. and Cangea, C. Wiki-cs: A wikipedia-based
benchmark for graph neural networks. arXiv preprint
arXiv:2007.02901, 2020.

Monti, F., Otness, K., and Bronstein, M. M. Motifnet: a
motif-based graph convolutional network for directed
graphs. In 2018 IEEE Data Science Workshop (DSW), pp.
225�228. IEEE, 2018.

Ordozgoiti, B., Matakos, A., and Gionis, A. Finding large
balanced subgraphs in signed networks. In Proceedings
of The Web Conference 2020, pp. 1378�1388, 2020.

Ou, M., Cui, P., Pei, J., Zhang, Z., and Zhu, W. Asymmetric
transitivity preserving graph embedding. In KDD, pp.
1105�1114, 2016.

Kollias, G., Kalantzis, V., Id�e, T., Lozano, A., and Abe, N.
Directed graph auto-encoders. In AAAI, pp. 7211�7219,
2022.

Page, L., Brin, S., Motwani, R., and Winograd, T. The
pagerank citation ranking : Bringing order to the web. In
The Web Conference, 1999.

18

Rethinking Link Prediction for Directed Graphs

Peng, J., Lei, R., and Wei, Z. Beyond over-smoothing: Un-
covering the trainability challenges in deep graph neural
networks. In CIKM, pp. 1878�1887, 2024.

Rendle, S., Freudenthaler, C., Gantner, Z., and Schmidt-
Thieme, L. Bpr: Bayesian personalized ranking from
implicit feedback. In UAI, pp. 452�461, 2009.

Yoo, H., Lee, Y.-C., Shin, K., and Kim, S.-W. Disentangling
degree-related biases and interest for out-of-distribution
generalized directed network embedding. In Proceedings
of the ACM Web Conference 2023, pp. 231�239, 2023.

Zhang, M. and Chen, Y. Link prediction based on graph

neural networks. NeurIPS, 31, 2018.

Zhang, X., He, Y., Brugnone, N., Perlmutter, M., and
Hirn, M. Magnet: A neural network for directed graphs.
NeurIPS, 34:27003�27015, 2021.

Zhang, Z., Cui, P., Wang, X., Pei, J., Yao, X., and Zhu, W.
Arbitrary-order proximity preserved network embedding.
In KDD, pp. 2778�2786, 2018.

Zhou, C., Liu, Y., Liu, X., Liu, Z., and Gao, J. Scalable
graph embedding for asymmetric proximity. In AAAI,
2017.

Zhou, H., Chegu, A., Sohn, S. S., Fu, Z., De Melo, G., and
Kapadia, M. D-hypr: Harnessing neighborhood modeling
and asymmetry preservation for digraph representation
learning. In CIKM, pp. 2732�2742, 2022.

Zhu, M., Wang, X., Shi, C., Ji, H., and Cui, P. Interpreting
and unifying graph neural networks with an optimization
framework. In Proceedings of the Web Conference 2021,
pp. 1215�1226, 2021a.

Zhu, S., Li, J., Peng, H., Wang, S., and He, L. Adversarial
In AAAI, pp. 4741�4748,

directed graph embedding.
2021b.

Rossi, E., Charpentier, B., Di Giovanni, F., Frasca, F.,
G�unnemann, S., and Bronstein, M. M. Edge directional-
ity improves learning on heterophilic graphs. In Learning
on Graphs Conference, pp. 25�1. PMLR, 2024.

Salha, G., Limnios, S., Hennequin, R., Tran, V.-A., and
Vazirgiannis, M. Gravity-inspired graph autoencoders for
directed link prediction. In CIKM, pp. 589�598, 2019.

Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B.,
and Eliassi-Rad, T. Collective classification in network
data. AI magazine, 29(3):93�93, 2008.

Shchur, O., Mumme, M., Bojchevski, A., and G�unnemann,
S. Pitfalls of graph neural network evaluation. Relational
Representation Learning Workshop, NeurIPS 2018, 2018.

Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., and Mei,
Q. Line: Large-scale information network embedding. In
WWW, pp. 1067�1077, 2015.

Tong, Z., Liang, Y., Sun, C., Li, X., Rosenblum, D., and Lim,
A. Digraph inception convolutional networks. NeurIPS,
33:17907�17918, 2020a.

Tong, Z., Liang, Y., Sun, C., Rosenblum, D. S., and Lim,
A. Directed graph convolutional network. arXiv preprint
arXiv:2004.13970, 2020b.

Veli?ckovi�c, P., Cucurull, G., Casanova, A., Romero, A., Li`o,
P., and Bengio, Y. Graph attention networks. In ICLR,
2018.

Virinchi, S. and Saladi, A. Blade: Biased neighborhood
sampling based graph neural network for directed graphs.
In WSDM, pp. 42�50, 2023.

Wang, X. and Zhang, M. How powerful are spectral graph
neural networks. In ICML, pp. 23341�23362. PMLR,
2022.

Weisfeiler, B. and Leman, A. The reduction of a graph to
canonical form and the algebra which appears therein. nti,
Series, 2(9):12�16, 1968.

Yang, Y., Lichtenwalter, R. N., and Chawla, N. V. Evaluat-
ing link prediction methods. Knowledge and Information
Systems, 45:751�782, 2015.

Yin, Y. and Wei, Z. Scalable graph embeddings via sparse
transpose proximities. In KDD, pp. 1429�1437, 2019.

19

Rethinking Link Prediction for Directed Graphs

A. Proof

A.1. The proof of Proposition 3.2

Proof. In the case of single methods, each node u in a directed graph is represented by a real-valued embedding hu. We
will use the directed graphs (a) and (d) from Figure 3 as examples: (a) is a directed ring graph, and (d) is a regular directed
hv) can enable reconstruction for graph (d)
graph. We will show that single methods with the decoder function MLP(hu
?
but not for graph (a).

For graph (d), which consists of three nodes and three edges (i.e., 1
embedding, h1, h2, h3
For the edge 1

2, the probability of the directed edge is given by:

Rd�1. Let the decoder be a simple MLP with the sigmoid activation function ? = Sigmoid(
�

1), each node is assigned a real-valued
).

2, 3

2, 3

?

?

?

?

?

p(1, 2) = ?(h1w1 + h2w2) > 0.5,
p(2, 1) = ?(h2w1 + h1w2) < 0.5,

(15)

(16)

where w1, w2

?

Rd�1 are the learnable weights. From these inequalities, we obtain the following system of constraints:

Similarly, for edges 2

3 and 3

?

?

1, we obtain the following inequalities:

h1w1 + h2w2 > 0, h2w1 + h1w2 < 0.

h3w1 + h2w2 > 0, h2w1 + h3w2 < 0,
h3w1 + h1w2 > 0, h1w1 + h3w2 < 0.

(17)

(18)

(19)

By solving the three sets of inequalities (17), (18), and (19), we find that they have a solution. For instance, if d = 2, one
possible solution is w1 = (1, 0), w2 = (0, 1), h1 = (1,
2). Therefore, single methods
with MLP(hu

hv) can successfully capture the graph structure and enable reconstruction for graph (d).
?

1, 1), and h3 = (2,

1), h2 = (

?

?

?

For the directed ring graph (a), which consists of three nodes and three edges (1
set of inequalities. For the edge 1
from the first, we get:

1), we similarly derive a
?
2, we obtain the same inequalities as in (17). By subtracting the second inequality

2, 2

3, 3

?

?

?

Similarly, for edges 2

3 and 3

?

?

(h1

?
1, we derive:

h2)w1 + (h2

(h2
(h3

?

?

h3)w1 + (h3
h1)w1 + (h1

h1)w2 > 0.

h2)w2 > 0,
h3)w2 > 0.

?

?

?

(20)

(21)

(22)

Adding these three inequalities (20), (21), and (22), results in 0 > 0, which is a contradiction and indicates that no
embeddings h1, h2, h3 and weights w1, w2 exist that can satisfy these conditions. The same result holds even when
nonlinearities are added to the MLP. Therefore, single methods with the decoder MLP(hu
hv) fail to compute the
?
probabilities for the directed ring graph.

This example demonstrates that while single methods with MLP(hu
reconstruction for certain directed graphs, they fail for directed ring graphs.

?

hv) can capture the graph structure and enable

A.2. The proof of Lemma 5.1

Proof. Substituting the block matrix

(8), we obtain:

( �A) =

S

(cid:21)

(cid:20) 0

�A
�A? 0

into the graph convolution of DiGAE�s encoder (i.e., Equation

(cid:21)

(cid:20) S(?+1)
T(?+1)

= ?

?

?

=

(cid:21) (cid:34)

(cid:32)(cid:20) 0

S(?)W(?)
�A
S
�A? 0
T(?)W(?)
T
(cid:16) �AT(?)W(?)
(cid:17)
?
(cid:16) �A?S(?)W(?)

? .

?

(cid:17)

T

?

S

(cid:35)(cid:33)

(23)

(24)

20

Rethinking Link Prediction for Directed Graphs

This result corresponds exactly to Equations (5) and (6) when the degree-based normalization of �A is not considered. If we
( �A) is given
include the degree-based normalization, it actually applies to
by diag

( �A). Notably, the diagonal degree matrix of

. Therefore, we have

(cid:16) �Dout, �Din

S

S

(cid:17)

(cid:21)

(cid:20) S(?+1)
T(?+1)

= ?

?

?

=

(cid:32)(cid:20) �D??
out
0
(cid:16) �D??
?
(cid:16) �D??

?

out

in

0
�D??
in
�A �D??
�A? �D??

(cid:21) (cid:20) 0

�A
�A? 0
(cid:17)
?

in T(?)W(?)
outS(?)W(?)

T

S

(cid:17)

? .

(cid:21) (cid:20) �D??
out
0

0
�D??
in

(cid:21) (cid:34)

S(?)W(?)
S
T(?)W(?)
T

(cid:35)(cid:33)

(25)

(26)

These results exactly equal the graph convolution used in DiGAE�s encoder, as defined in Equations (5) and (6). Therefore,
the graph convolution in DiGAE�s encoder matches the form of Equation (8), suggesting that it effectively corresponds to a
GCN convolution on an undirected bipartite graph.

A.3. The proof of Lemma 5.2

Proof. Given the block adjacency matrix

( �A) =

S

(cid:21)

(cid:20) 0

�A
�A? 0

and its degree matrix DS =

(cid:20) �Dout
0

0
�Din

(cid:21)

, we have

D?1/2
S

S

( �A)D?1/2
S

=

=

=

=

=

(cid:20) �Dout
0
(cid:34) �D?1/2
out
0

0
�Din

(cid:21)?1/2 (cid:20) 0

�A
�A? 0

0
�D?1/2
in

(cid:35) (cid:20) 0

�A
�A? 0

(cid:21) (cid:20) �Dout
0
(cid:21) (cid:34) �D?1/2

out
0

(cid:21)?1/2

0
�Din

(cid:35)

0
�D?1/2
in

�D?1/2
out

�A �D?1/2
in

0

?

?

0
�A �D?1/2
in

(cid:17)?

(cid:21)

?

?

(cid:20) 0

out

(cid:16) �D?1/2
�A
�A? 0
( �A).

S

(27)

(28)

(29)

(30)

(31)

S

Therefore, D?1/2
is equivalent to the symmetric
out
normalization of
performs symmetric normalization on the
adjacency matrix of an undirected bipartite graph, aligning with the common normalization scheme used in graph neural
networks (Kipf & Welling, 2017; Defferrard et al., 2016).

( �A)D?1/2
S
( �A). These findings suggest that �A = �D?1/2

( �A), showing that the normalization �A = �D?1/2

�A �D?1/2
in

�A �D?1/2
in

S
S

out

=

S

B. More Details of Experimental Settings

B.1. Metric description

Mean Reciprocal Rank (MRR) evaluates the capability of models to rank the first correct entity in link prediction tasks.
It assigns higher weights to top-ranked predictions by computing the average reciprocal rank of the first correct answer
across queries: MRR = 1
is the total number of queries and ranki denotes the position of the first
|Q|
correct answer for the i-th query. MRR emphasizes early-ranking performance, making it sensitive to improvements in top
predictions.

(cid:80)|Q|
i=1

, where

1
ranki

Q
|

|

(cid:80)N

Hits@K measures the proportion of relevant items that appear in the top-K positions of the ranked list of items. For N
queries, Hits@K= 1
K), where ranki is the rank of the i-th sample and the indicator function 1 is 1 if
N
ranki
K, and 0 otherwise. Following the OGB benchmark (Hu et al., 2020), link prediction implementations compare
each positive sample�s score against a set of negative sample scores. A �hit� occurs if the positive sample�s score surpasses
at least K-1 negative scores, with final results averaged across all queries.

i=1 1(ranki

?

?

Area Under the Curve (AUC) measures the likelihood that a positive sample is ranked higher than a random negative

21

Rethinking Link Prediction for Directed Graphs

(cid:80)M

(cid:80)N

j=1 1(spos
M �N

i >sneg
j )

i=1

, where M and N are positive/negative sample counts, spos

sample. AUC =
scores. Values approaching 1 indicate perfect separation of positive and negative edges.
Average Precision (AP) is defined as the area under the Precision-Recall (PR) curve. Formally, AP = (cid:80)N
�
Pi, where Pi is the precision at the i-th threshold, Ri is the recall at the i-th threshold, and N is the number of thresholds
considered.

their prediction

and sneg

i=1(Ri

Ri?1)

?

j

i

Accuracy (ACC) measures the proportion of correctly predicted samples among all predictions. Formally, ACC =
T P +T N +F P +F N , where T P , T N , F P , and F N represent true positives, true negatives, false positives, and false negatives,
respectively.

T P +T N

B.2. DirLinkBench Setting

Baseline Implementations. For MLP, GCN, GAT, and APPNP, we use the PyTorch Geometric (PyG) library (Fey &
Lenssen, 2019) implementations. For DCN, DiGCN, and DiGCNIB, we use the PyTorch Geometric Signed Directed
(PyGSD) library (He et al., 2023) implementations. For other baselines, we use the original code released by the authors.
Here are the links to each repository.

� PyG: https://github.com/pyg-team/pytorch geometric

� PyGSD: https://github.com/SherylHYX/PyGSD

� STRAP: https://github.com/yinyuan1227/STRAP-git

� ODIN: https://github.com/hsyoo32/odin

� ELTRA: https://github.com/mrhhyu/ELTRA

� GPRGNN: https://github.com/jianhao2016/GPRGNN

� DiGAE: https://github.com/gidiko/DiGAE

� DHYPR: https://github.com/hongluzhou/dhypr

� DirGNN: https://github.com/emalgorithm/DirGNN

� MagNet: https://github.com/matthew-hirn/magnet

� DUPLEX: https://github.com/alipay/DUPLEX

Hyperparameter settings. The hyperparameter settings for the baselines in DirLinkBench are detailed below, where
�hidden� represents the number of hidden units, �embedding� refers to the embedding dimension, �undirected� indicates
whether an undirected training graph is used, �lr� stands for the learning rate, and �wd� denotes weight decay.

� MLP: hidden: 64, embedding: 64, layer: 2, lr:

� GCN: hidden: 64, embedding: 64, layer: 2, undirected:

0.01, 0.005
{

, wd:
}

0.0, 5e-4
}
{
True, False
}
{

.

� GAT: hidden: 8, heads: 8, embedding: 64, layer: 2, undirected:

, lr:

, wd:

0.01, 0.005
{
True, False
{
0.1,0.2
{

0.0, 5e-4
{
0.01, 0.005
{
True, False
{

, undirected:
}

.
}
0.0, 5e-4
}
, wd:
0.01, 0.005
}

}
, lr:
}

, lr:
}

, wd:

}

{

{

� GPRGNN: hidden: 64, embedding: 64, layer: 2, Init: PPR, K: 10, ?:

0.1,0.2

, undirected:
}

True, False
{

, lr:
}

{

0.01,

{

� APPNP: hidden: 64, embedding: 64, layer: 2, K: 10, ?:

.
0.0, 5e-4
}
{

0.005

, wd:
}

.
0.0, 5e-4
}

{

� DGCN: hidden: 64, embedding: 64, lr:

0.01, 0.005
}

{

� DiGCN and DiGCNIB: hidden: 64, embedding: 64, ?:

, wd:

.
0.0, 5e-4
}

{
0.1,0.2
{

22

, layer: 2, lr:
}

0.01, 0.005
{

, wd:
}

0.0, 5e-4
}
{

.

Rethinking Link Prediction for Directed Graphs

Table 12. The parameters of SDGAE on different datasets.

Datasets

hidden

embedding MLP layer K

lr

Cora-ML
CiteSeer
Photo
Computers
WikiCS
Slashdot
Epinions

64
64
64
64
64
64
64

64
64
64
64
64
64
64

1
1
2
2
2
2
2

5
5
5
3
5
5
5

0.01
0.01
0.005
0.005
0.005
0.01
0.005

wd

0.0
0.0
0.0
0.0
5e-4
5e-4
0.0

� DirGNN: hidden: 64, embedding: 64, layer: 2, ?:
0.0, 5e-4
}
{

0.01, 0.005
{

, wd:
}

.

� MagNet: hidden: 64, embedding: 64, layer: 2, K:

.
5e-4
}

� DUPLEX: hidden: 64, embedding: 64, layer:

0.01, 0.005
{

, wd:
}

0.0, 5e-4
}
{

.

� DHYPR: hidden: 64, embedding: 32, proximity:

� DiGAE: hidden: 64, embedding: 64, layer:

.
5e-4
}

0.0, 0.5, 1.0
{

}

, jk:

�cat�, �max��
{

, normalize:
}

, lr:
True, False
}

{

1, 2

, q:
}

, lr:
0.05, 0.1, 0.15, 0.2, 0.25
}

{

0.01, 0.005
{

}

{

, wd:

0.0,
{

2, 3
}
{

0.1, 0.3
, head: 1, loss weight:
{

, loss decay:
}

0.0, 1e-2, 1e-4
{

, lr:
}

, ?:
}

1, 2
{
1, 2
{

0.01, 0.05, 1, 5
{
, (?, ?) :
}

2, lr:
0.0, 0.2, 0.4, 0.6, 0.8
}
{

, lr:
}

0.01, 0.001
{

, wd:
}
0.01, 0.005
{

0.0, 0.001
{

, wd:
}

{

.
}
0.0,

Regarding the selection of these hyperparameters, we begin by setting each model to use two layers, with both hidden and
embedding dimensions set to 64. This ensures that all methods have approximately the same number of learnable parameters,
a common practice in many GNN benchmarks (Hu et al., 2020; Chien et al., 2021). Some models have exceptions. For
example, DUPLEX uses three layers across all datasets in its official implementation; given its status as a recent advanced
method, we search over layers in
. For DYHPR, due to its high computational complexity and the fact that it generates
2, 3
}
{
multiple embeddings for each node, we follow its original implementation and set the embedding dimension to 32. We
retain the layer setting for DiGAE, which uses one or two layers in its released code.

For general hyperparameters such as learning rate and weight decay, we approximately keep the search space consistent
across all models. For model-specific parameters (e.g., q in MagNet, and ?, ? in DiGAE), we adhere to the search ranges
reported in their respective papers. All hyperparameters are tuned via grid search to identify the optimal settings under our
DirLinkBench benchmark.

B.3. Hyperparameter setting of SDGAE

For the SDGAE hyperparameter settings, we aligned our settings with those of other baselines in DirLinkBench to ensure
fairness. For the MLP used in X initialization, we set the number of layers to one or two, matching DiGAE�s convolutional
layer configurations. The number of hidden units and the embedding dimension were both set to 64. The learning rate (lr)
was chosen as either 0.01 or 0.005, and weight decay (wd) was set to 0.0 or 5e-4, following the configurations of most
GNN baselines in DirLinkBench. The polynomial order K was searched in the set
. We performed a grid search to
optimize parameters on the validation set, and Table 12 presents the corresponding SDGAE parameters for different datasets.

3, 4, 5
}

{

23


