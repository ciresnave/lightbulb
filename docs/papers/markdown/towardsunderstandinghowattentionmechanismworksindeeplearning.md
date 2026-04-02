4
2
0
2

c
e
D
4
2

]

G
L
.
s
c
[

1
v
8
8
2
8
1
.
2
1
4
2
:
v
i
X
r
a

Submitted month/day; Revised x/x; Published x/x

Towards understanding how attention mechanism works in
deep learning

ruantianyu17@mails.ucas.ac.cn

zsh@amss.ac.cn

Tianyu Ruan? ?

Shihua Zhang? ?
? Academy of Mathematics and Systems Science
Chinese Academy of Sciences
? School of Mathematical Sciences
University of Chinese Academy of Sciences

Editor: My editor

Abstract

Attention mechanism has been extensively integrated within mainstream neural network
architectures, such as Transformers and graph attention networks. Yet, its underlying
working principles remain somewhat elusive. What is its essence? Are there any connec-
tions between it and traditional machine learning algorithms? In this study, we inspect
the process of computing similarity using classic metrics and vector space properties in
manifold learning, clustering, and supervised learning. We identify the key characteristics
of similarity computation and information propagation in these methods and demonstrate
that the self-attention mechanism in deep learning adheres to the same principles but op-
erates more flexibly and adaptively. We decompose the self-attention mechanism into a
learnable pseudo-metric function and an information propagation process based on similar-
ity computation. We prove that the self-attention mechanism converges to a drift-diffusion
process through continuous modeling provided the pseudo-metric is a transformation of a
metric and certain reasonable assumptions hold. This equation could be transformed into
a heat equation under a new metric. In addition, we give a first-order analysis of attention
mechanism with a general pseudo-metric function. This study aids in understanding the
effects and principle of attention mechanism through physical intuition. Finally, we pro-
pose a modified attention mechanism called metric-attention by leveraging the concept of
metric learning to facilitate the ability to learn desired metrics more effectively. Experi-
mental results demonstrate that it outperforms self-attention regarding training efficiency,
accuracy, and robustness.

Keywords

Attention mechanism, Transformer, graph attention network, similarity computation, heat
diffusion

1 Introduction

The attention or self-attention mechanism is extensively applied in popular deep learning
architectures like Transformers (Vaswani et al., 2017; Dong et al., 2018) and graph attention
networks (Veli?ckovi�c et al., 2017). This mechanism enables the model to assign diverse

�2024 Tianyu Ruan and Shihua Zhang.

License: CC-BY 4.0, see https://creativecommons.org/licenses/by/4.0/. Attribution requirements are provided
at http://jmlr.org/papers/v23/21-0000.html.

Tianyu Ruan and Shihua Zhang

weights to various parts (data points, nodes, etc.) of the input sequence (data, graph, etc.)
based on their relevance when producing an output. This capability is particularly critical
for handling inputs where the lengths and relevance strengths of different parts can vary
significantly. As a result, this mechanism contributes to the broad applications of deep
learning in various fields, including natural language processing (Devlin et al., 2018; Brown
et al., 2020; Radford et al., 2019), computer vision (Dosovitskiy et al., 2020; Touvron et al.,
2021; Carion et al., 2020), graph mining (Liu et al., 2023) and bioinformatics (Dong and
Zhang, 2022; Zhang et al., 2023; Ji et al., 2021).

However, understanding the mathematical principle of attention mechanism is still chal-
lenging due to its interaction with normalization layers and feed-forward networks in neural
network architectures. Difficulties in understanding attention mechanism also stem from
the numerous parameters in neural networks and various engineering techniques. To our
knowledge, only a few studies have explored it in depth (Vuckovic et al., 2020; Dong et al.,
2021; Sander et al., 2022; Geshkovski et al., 2023). Sander et al. (2022) formalized the
self-attention mechanism with residual connections as a flow map and analyzed it from the
perspective of the Wasserstein gradient flow. In addition, they characterized the L2 self-
attention (Kim et al., 2021) using continuous dynamical systems. Geshkovski et al. (2023)
investigated attention mechanism in Transformers by assuming that data is distributed on
the unit sphere and making simplified assumptions about parameters and proved that the
distribution would converge to a single point under certain conditions, suggesting that at-
tention mechanism induces an aggregation tendency. However, they did not fully explain
how attention mechanism works or its connection to classical algorithms.

Moreover, several architectures, such as CRATE (Yu et al., 2024) and Probabilistic
Transformer (Wu and Tu, 2023), have been proposed. They often originate from inter-
pretable models and have information propagation mechanisms similar to attention mecha-
nism. While not strictly equivalent, they offer valuable insights into understanding attention
mechanism. For example, CRATE suggests that attention mechanism functions as a com-
pression process, whereas the Probabilistic Transformer explains it as an explicit iteration
of the Frank-Wolfe optimization algorithm.

We illustrate three architecture components, i.e., the residual block, the attention block,
and their recombination of a Transformer block (Figure 1). The attention block consists
of an information propagation process followed by a linear transformation. Some studies
modeled the residual block using an ordinary differential equation (E, 2017; Chen et al.,
2018). Some researchers modeled the recombination of the residual and attention blocks
using the flow map mentioned above. In this paper, we focus on the attention-based in-
formation propagation mechanism. This mechanism can be viewed as a message-passing
process akin to that in graph neural networks (GNNs) operating on fully connected graphs.
However, GNNs typically employ fixed edge weights and topologies, which limit theoretical
analysis to diffusion processes on graphs (Li et al., 2024). In contrast, attention mechanism,
as a learnable method of information propagation on fully connected graphs, has yet to be
thoroughly analyzed in terms of its limit behavior.

Many machine learning methods such as manifold learning (e.g., diffusion map (Coif-
man and Lafon, 2006), UMAP (McInnes et al., 2018)), clustering methods (e.g., k-means
clustering (Lloyd, 1982), fuzzy c-means clustering (Bezdek et al., 1984), Markov cluster-
ing (Van Dongen, 2008)) and supervised learning (e.g., k-nearest neighbors algorithm (Fix,

2

Towards Understanding how attention mechanism works in deep learning

1985), support vector machine (Cortes and Vapnik, 1995)), involve computing similarity
or dissimilarity for pairwise data points (or nodes in a graph). These approaches usually
consist of some of the following components successively: (1) Initializing similarity based on
the input data and the adopted pseudo-metric; (2) Strengthening similarity through some
transformation to make similar data points more similar or dissimilar ones more distinct;
(3) Normalizing similarity to transform the similarity matrix into a type of probability
distribution, which allows the use of probabilistic tools for comparison or further manipu-
lation. Traditional algorithms aim to compute similarity by combining classic metrics and
handcrafted designs to capture data features or extract information.

We can intuitively observe that attention mech-
anism also follows this principle of similarity com-
putation to capture the inherent patterns in data.
In this study, by exploring the common approach
of similarity computation in classic machine learn-
ing algorithms, we explain how this technique is uti-
lized in attention mechanism, thereby revealing its
connections to classical machine learning algorithms.
In addition, we illustrate that under certain assump-
tions like (1) the formulation of similarity can be de-
composed into a transformation of a learnable met-
ric function, a parameter on time scale and a soft-
max similarity computation; (2) the time-scale pa-
rameter is sufficiently small and (3) the manifold hy-
pothesis holds, and there are sufficient data points
from a continuous distribution, at-
sampled i.i.d.
tention mechanism for information propagation can
be approximated by a drift-diffusion process on the
manifold. Furthermore, we prove that, under certain
assumptions, this process can be reformalized as a
heat diffusion process under a new metric. This pro-
vides theoretical support for intuitively understand-
ing the working principle of the self-attention mech-
anism through physical intuition (Figure 2).

Figure 1: Illustration of three ar-
chitecture components in-
cluding the residual block
(A), the attention block
(C), and their recombina-
tion (B) of a Transformer
block.

In assumption (1), if the learnable function is a
general function rather than a metric function, we
can still give a first-order analysis to describe the
information propagation process of attention mecha-
nism. Although this process does not generally cor-
respond to a continuous dynamical system, it retains
a similar interpretation to the metric version. That
is to say, the zeroth-order effect is determined by the nearest data points, while the first-
order effect relates to a drift-diffusion process at these points. The main difference between
the two lies in how the nearest data points are defined.

From the perspective of similarity computation, we decompose the self-attention mecha-
nism into an information propagation process based on handcrafted similarity computation

3

Tianyu Ruan and Shihua Zhang

and a learnable pseudo-metric. Inspired by this, we propose a modified attention mechanism
by leveraging the concept of metric learning to enhance the ability to learn desired metrics
more effectively. We refer to this modified attention mechanism as �metric-attention�.

Figure 2: Illustration of the main idea. (a) Attention mechanism consists of two main steps:
(1) computing the similarity between nodes (data points or tokens), followed by
propagating node features to neighboring nodes, weighted by the similarities,
and (2) updating the features of the nodes. (b) Illustration of a drift-diffusion
process on the manifold where data reside. This process is driven by two main
forces: density guidance, which encourages local concentration, and diffusion,
which promotes global consistency of features. This study demonstrates that
attention mechanism can be considered a first-order approximation of the drift-
diffusion process on manifold, i.e., the short-time diffusion.

In Section 2, we introduce the typical techniques of similarity computation used in many
classic machine learning algorithms and clarify that attention mechanism also follows this
principle. Additionally, we introduce the heat kernel approximation as it relates to atten-
tion mechanism. In Section 3, we explore the limit properties of attention mechanism for
information propagation and prove that it can be regarded as a first-order approximation of
a drift-diffusion equation under certain conditions, which can be further transformed into
a heat equation. We also introduce a first-order analysis of information propagation in a
generalized pseudo-metric setting. In Section 4, by interpreting attention mechanism as a
combination of information propagation based on similarity computation and a learnable
pseudo-metric, we propose a metric-attention mechanism to improve information propa-

4

Towards Understanding how attention mechanism works in deep learning

gation. Numerical experiments demonstrate its superior performance to the self-attention
mechanism on various examples. Finally, we conclude this work and discuss its implications.

2 From similarity computation to attention mechanism

2.1 Similarity computation

Similarity computation is a set of engineering practices to generate similarity measures
between data points (or graph nodes). Different methods for similarity computation have
been developed. Most of them involve one, two, or three of the following three components:
initializing similarity, strengthening similarity, and normalizing similarity.

2.1.1 Initializing similarity

Initializing similarity is the first step for further computations and information extraction.
The point-to-point similarity is typically generated using a binary function D(�, �), where D
is often a simple transformation based on metrics, inner products, or topological structures.
A natural idea is that the closer two data points are, the higher their similarity. There-
fore, the distance between two data points is often transformed into a similarity measure
through monotonically decreasing mappings. Building on previous work, we have derived
the following metric-based function Dt,c:

Definition 1 (Metric-based similarity generation). Given a metric space (M, d), we define
the similarity function on M as:

Dc,t(x, y) = c(x) ? sign(t)d(x, y)t

where c is a specified function and t is a hyperparameter.

Bilinear functions are also commonly used to initialize similarity.

Definition 2 (QK-dot product). Given m � n matrices Q, K and vectors x, y ? Rn, where
x and y are column vectors, we define QK-dot product of x and y as follows:

D(x, y) = x �QK y = Qx � Ky = xT (QT K)y

Before the inner product mapping, people may apply a non-linear transformation to the
original features, which is equivalent to defining similarity using a certain kernel function
K(�, �).

Additionally, similarity may be defined using a local combination, which reflects how a

data point or node is represented by a combination of its neighbors.

Definition 3 (Local combination similarity). Given a set of data points {vi} ? Rn and
their adjacency relationships, we calculate ?ij to satisfy:

vi =

(cid:88)

?ijvj

vj ?N (i)

where N (i) is the set of neighbors of vi. The local combination similarity between vi and vj
is ?ij, denoted by Lc(vi, vj).

5

Tianyu Ruan and Shihua Zhang

In addition to feature-based similarity initialization methods, topology-based methods

have also been defined:

Definition 4 (k-th order adjacency similarity). Given a graph G(V = {vi}, E = {(vaj , vbj )}),
with adjacency matrix A, we define the k-th order adjacency similarity of vi and vj as (Ak)ij.

2.1.2 Strengthening similarity

The purpose of strengthening similarity is to make two data points that are relatively similar
become even more similar. It can work in conjunction with the normalization process, by
strengthening similarities and decreasing weaker ones to aid the aggregating process.

Definition 5 (r-Inflation). Given the hyperparameter r, we define the inflation operator
?r:

?r : Mm�n ? Mm�n
(?r(M ))ij = sign(r)M r
ij

Definition 6 (Exponential Inflation). Given a vector ? = (?1, � � � , ?n) of size n, we define
the exponential inflation operator ??
? :

(??

? (M ))ij = exp

2.1.3 Normalizing similarity

??

? : Mm�n ? Mm�n

(cid:19)

(cid:18) Mij
?i

The purpose of normalizing similarity is to transform the similarity matrix into a desired
form, which is often related to the modeling of the data. Given a similarity matrix S, where
each element is positive, four common normalization operations are typically used.

Definition 7 (Row normalization).

Nr : Mm�n ? Mm�n

(Nr(S)ij =

Sij
k Sik

(cid:80)

Definition 8 (Column normalization).

Nc : Mm�n ? Mm�n

(Nc(S)ij =

Sij
k Skj

(cid:80)

Definition 9 (Two-side normalization).

N2 : Mm�n ? Mm�n

Mij
(cid:80)

k Mik

k Mjk

(N2(M )ij =

(cid:80)

6

Towards Understanding how attention mechanism works in deep learning

Definition 10 (Global normalization).

Ng : Mm�n ? Mm�n
Mij
k,l Mkl

(Ng(M )ij =

(cid:80)

2.2 Similarity computation in machine learning

Here, we introduce the applications of similarity computation in manifold learning, clus-
tering, supervised learning, and attention mechanism in neural networks, highlighting the
distinct common characteristics shared by attention mechanism and traditional algorithms.

2.2.1 Similarity computation in manifold learning

Manifold learning refers to dimensionality reduction. Due to the curse of dimensional-
ity, high-dimensional data are often difficult to handle and cannot be visualized. There-
fore, dimensionality reduction methods are employed to preprocess the data. The prob-
lem statement of manifold learning is as follows: given the high-dimensional data points
{xi, i = 1, � � � , N } ? Rn, assuming that they are distributed on a low-dimensional mani-
fold, how to find their corresponding data points {yi, i = 1, � � � , N } in a low-dimensional
space Rl, where l ? n, such that the topology, distances or densities of data points on the
underlying manifold is preserved as much as possible?

How can we maintain the main structure of data during the process of dimensionality
reduction? Prior work often adopts the principle of similarity (dissimilarity or distance)
preservation. First, the coordinates of data points in the low-dimensional space are deter-
mined using some initialization method. Then, the coordinates are optimized so that the
pairwise similarities in the low-dimensional space closely approximate those in the original
data. The key difference among most approaches lies in the different similarity computation
techniques they employ. To show this, we summarize those used in several classical algo-
rithms including MDS (Kruskal and Wish, 1978), Isomap (Balasubramanian and Schwartz,
2002), LLE (Roweis and Saul, 2000), Laplacian eigenmaps (Belkin and Niyogi, 2003), Dif-
fusion map (Coifman and Lafon, 2006), SNE (Hinton and Roweis, 2002), t-SNE (Van der
Maaten and Hinton, 2008), UMAP (McInnes et al., 2018) (Table 1). These methods employ
hand-crafted similarity computation techniques.

2.2.2 Similarity computation in clustering

Fuzzy c-means clustering and k-means clustering Fuzzy c-means clustering cate-
gorizes data points into several classes such that data points within each class have high
similarity. This is achieved by alternately calculating similarity and updating class represen-
tatives based on similarity. To be specific, suppose we have data points {x1, � � � , xN } ? Rn
and class center {c1, � � � , cm} ? Rn:

� Calculate the similarity between data points and class center:

(D ?2

m?1 ,0)ij = ?ci ? xj?

?2
m?1

S = Nr ? Nc(D ?2

m?1 ,0)

7

Tianyu Ruan and Shihua Zhang

Table 1: Summary of typical techniques for computing similarity in manifold learning meth-

ods.

Method

Similarity
initialization

Similarity
strengthen

Normalization

None
None
None
??
?
??
?
??
?

MDS
Isomap
LLE
Laplacian eigenmaps
Diffusion map
SNE
t-SNE
UMAP

D1,0
D1,0 (Graph)
Lc
D2,0
D2,0
D2,0
D2,0 and D2,?1
D1,c and Dc,?1

where m is a hyper-parameter.

� Renew class center:

None
None
None
None
Nr ? N2
Nr

? and ??1 Ng and N (W ) = 1/2W + 1/2W T
??
N (W ) = W + W T ? W ? W T
??
? and ??1

ci =

(cid:88)

j

Sijxj

Repeat the above process until convergence. The ci represents the representative of class i
m?1 ,0)ij represent the probability that data point j belongs to class i. By taking
and (Nc ? D ?2
the limit ?2

m?1 ? ?? in the fuzzy c-means algorithm, we obtain the k-means algorithm.

Markov clustering algorithm (MCL) MCL is an unsupervised graph clustering algo-
rithm. The input is a (weighted) graph, and the output is a clustering of nodes. Here, we
describe the MCL process using the concepts of information passing and feature represen-
tation. MCL iteratively calculates:

H k+1 = S(k) � I

where H k+1 represents the clustering result of the algorithm at the k-th iteration, I is the
unit matrix which represents the feature matrix of one-hot vector, and the similarity matrix
is computed as:

S(k) = Nr ? ?2 ? M2(H k)

where M2(A) := A2, indicating the topological similarity of second order, ?2 is element-
wise squaring, and Nr represents the normalization performed on each row. H 1 equals the
input weight matrix. This process can be seen as computing node similarity iteratively and
updating node features based on this similarity. Once the process converges, the features
of each node are used to classify the nodes.

8

Towards Understanding how attention mechanism works in deep learning

2.2.3 Similarity computation in supervised learning

K-nearest neighbors algorithm (KNN) KNN is a supervised classification algorithm
that utilizes adjacency information (Fix, 1985). Given a set of data points {(xi, yi), i =
1, � � � , N }, where yi ? {0, 1} represents the label, the predicted label �y for a test data
point �x is computed as �y = sT y, where y = (y1, � � � , yN )T , and the similarity vector
s = (s1, � � � , sN )T is defined by:

s = Nc(A)

where A = (Ai, � � � , AN )T . Ai = 1 if xi is one of the k-nearest neighbors of �x; otherwise,
Ai = 0.

Support vector machine (SVM) SVM is a supervised binary classification algorithm
(Cortes and Vapnik, 1995). Given a dataset {(xi, yi), i = 1, � � � , N }, where yi ? {1, ?1},
and a kernel function K(�, �). SVM first computes the coefficients ?i and the bias term b
corresponding to the supporting plane. For a new sample �x, the predicted label is computed
as �y = sT y + b, where y = (y1, � � � , yN )T and s = (s1, � � � , sN )T is the similarity vector
defined as:

si = ?iK(�x, xi)

If �y > 0, then �x is classified into the class of +1; otherwise, it is classified into the class of
?1.

2.2.4 Similarity computation in attention mechanism for information

propagation

Neural networks
Information propagation modules are highly prevalent in neural net-
work architectures, particularly in graph neural networks and Transformers. These modules
are typically followed by linear transformations and nonlinear activations, which together
form the core of these architectures. An information propagation module can be expressed
as:

H k+1 = S(k)H k

where H k contains the features of each node in the k?th layer, S(k) is the similarity matrix
in which S(k)
represents the similarity between the i-th and j-th data points in the k-th
ij
layer. Different neural networks adopt various methods for similarity computation. For
example,

� Graph convolutional networks: S = Normalized(A + I), where A is the adjacency

matrix.

� Diffusionnet (Sharp et al., 2022): S = exp(tL), where t is a parameter and L is the

discrete Laplacian matrix.

� Transformer and graph attention network: Similarity between data points is initially
computed pairwise through a learnable pseudo-metric function f?(xi, xj). The result-
ing similarity matrix is then subjected to exponential scaling and row normalization

9

Tianyu Ruan and Shihua Zhang

to yield S:

Sij =

exp (?f?(xi, xj))
k exp (?f?(xi, xk))

(cid:80)

We observe that attention mechanism utilizes similarity computation and information
propagation, that are core components of classical algorithms. The key difference is that
classical algorithms often rely on manually designed similarity computation methods, which
limits their applicability. In contrast, attention mechanism incorporates learnable param-
eters, which makes them adaptive and suitable for more general network architectures.
Moreover, this formulation closely resembles the heat kernel approximation, enabling an
analysis of the properties of the attention mechanism from this perspective.

2.3 Approximate Laplacian-Beltrami operator by heat kernel

The heat kernel is deeply connected to the Laplacian-Beltrami operator, which is a funda-
mental tool in differential geometry and mathematical physics for studying the geometric
properties of surfaces and manifolds. This operator generalizes the Laplacian from Euclidean
spaces to general manifolds. Informally, it contains all the information of the Riemannian
manifold (Bronstein and Kokkinos, 2010). Here, we focus on the 0-form Laplacian opera-
tor, as it is the most commonly studied and has the most direct connection to attention
mechanism.

In the following, we introduce the method to approximate the Laplacian-Beltrami op-
i=1 and the heat kernel h?(x, y) =
, we define the weight matrix W of N � N by heat kernel h? and the diagonal

erator using the heat kernel. Given the data points {xi}N
? ?x?y?2
exp
2?
matrix D:

(cid:16)

(cid:17)

Wij = h?(xi, xj) = exp

?

(cid:18)

?xi ? xj?2
2?

(cid:19)

, Dii =

n
(cid:88)

j=1

Wij =

(cid:88)

j

(cid:18)

exp

?

(cid:19)

?xi ? xj?2
2?

The negative defined graph Laplacian for data points {xi}N
where

i=1 is defined as L = D?1W ? I,

Lij =

(cid:16)

exp

(cid:80)

k exp

(cid:17)

? ?xi?xj ?2
2?
(cid:16)
? ?xi?xk?2
2?

(cid:17) ? ?ij

where ?ij = 1 if i = j and ?ij = 0 otherwise. Given that the data points {xi}N
i=1 are uni-
formly distributed on a manifold, it has been shown that the graph Laplacian will converge
to the Laplacian of the manifold as ? ? 0 and N ? ?. The following theorem implies that
a heat kernel can approximate the Laplacian of a Riemannian manifold and provides the
convergence rate of this approximation.

Theorem 1 (Naive heat kernel approximator (Singer, 2006)). Suppose {xi}N
sampled from the uniform measure on a compact Riemannian manifold, then

i=1 are i.i.d.

1
?

N
(cid:88)

j=1

Lijf (xj) =

1
2

?f (xi) + O

(cid:18)

1
N 1/2?1/2+d/4

(cid:19)

, ?

10

Towards Understanding how attention mechanism works in deep learning

where f (�) is a smooth function, ? is the (negative defined) Laplacian-Beltrami operator of
the Riemannian manifold, and O represents the big O notation.

When the data are not sampled from a uniform distribution, Coifman and Lafon (2006)
offered a general Laplacian approximator.
If one still opts to use the Naive heat kernel
approximator, it will result in an additional component. The following lemma describes
the influence of distribution, and it is the key to depict the limit property of attention
mechanism:

Theorem 2 (Deviation of naive heat kernel approximator (Coifman and Lafon, 2006)).
Suppose the data points {xi}N
i=1 are i.i.d. sampled from a distribution on a compact Rie-
mannian manifold with density function p(x), then

1
?

N
(cid:88)

j=1

Lijf (xj) ?

(cid:18)

1
2

?f (xi) + 2

(cid:28) ?p
p

(cid:29)(cid:19)

(xi), ?f (xi)

3 Limit properties of attention mechanism for information propagation

Due to the analogous formulations of heat kernel approximation and information prop-
agation of attention mechanism, we can analyze the asymptotic properties of attention
mechanism by analogy. To start with, we will show a direct analysis for the metric setting
of the learnable pseudo-metric, which corresponds to an elegant continuous dynamical sys-
tem. This special setting is helpful for intuitively understanding attention mechanism by
physical intuition. We further generalize this analysis to the situation where the learnable
pseudo-metric function does not satisfy the metric setting. The primary difference between
these two settings lies in how neighbors are defined.

3.1 Assumptions

Network structure formulation We examine the attention coefficients computation
and information propagation steps in Transformer and graph attention network. Specifi-
cally, we pay attention to the following steps:

H new = SH old

where Sij =

exp (cid:0)?f?(H old
k exp (cid:0)?f?(H old
(cid:80)
the graph attention network, f?(x, y) = ??(?T
vectors, P is a learnable matrix and ? is an activation function.

)(cid:1)
,H old
k )(cid:1) . In the Transformer block, f?(x, y) = ?xT (QT K)y. In
j
,H old
2 P y), where ?i (i = 1, 2) are learnable

1 P x??T

i

i

We reformulate the similarity matrix of attention mechanism S as S?:

exp (cid:0) ?
k exp (cid:0) ? f?(H old
where ?/2 represents the time scale. In attention mechanism, the updated representation is
given by H new = S?H old, where ? = 1
2 .

S?,ij =

,H old
k )

(cid:80)

2?

(cid:1)

(cid:1)

i

i

f?(H old
2?

,H old
j

)

By this formulation, we suppose that:

11

Tianyu Ruan and Shihua Zhang

� Assumption 1: ? is sufficiently small.
� Assumption 2: (cid:112)c + f?(�, �) is a distance function d? induced by a Riemannian

metric g?, where c is a constant.

Example 1. In Transformer, if Q = K = I, then f?(x, y) = ?x � y. Given that the data is
distributed on the unit sphere, we have:

f?(x, y) = ?2 + ?x ? y?2
2

Generally, if Q = K, then f?(x, y) = ?xT QT Qy. Given that the data is distributed on the
ellipsoid {x ? Rd : xT QT Qx = 1}, we have:

f?(x, y) = ?2 + ?x ? y?2
Q

where ?x ? y?2

Q = (x ? y)T QT Q(x ? y).

Example 2. In graph attention network, if ?1 = ??2 and the activation is symmetric
about the origin, then:

f?(x, y) = ??(cid:0)?1P (x ? y)(cid:1)

also satisfies symmetry. If the data is distributed on a 1-d manifold and ?(x) = ?x2, then
f?(x, y) may satisfy the properties described above.

Our first assumption primarily facilitates the application of Theorem 2, setting the stage
for the convergence of the information propagation in the attention mechanism to a partial
differential operator. Although idealized, our second assumption is relatively reasonable in
the context of neural network configurations and similarity computations, offering critical
insights. As demonstrated in the previous examples, attention mechanism settings may
meet some assumptions under certain conditions.

Assumption 3 (Data distribution assumption) The final assumption relates to the
manifold hypothesis, which posits that meaningful high-dimensional data in real life often
lies on an intrinsic low-dimensional manifold, which is the theoretical basis for many machine
learning algorithms (Roweis and Saul, 2000; Coifman and Lafon, 2006; Wold et al., 1987).
This hypothesis has been validated in numerous cases, such as the MNIST handwritten
digit database (Deng, 2012), and has been widely adopted. Based on this hypothesis, many
algorithms have been developed for manifold fitting (Yao et al., 2024; Yao and Xia, 2019)
and manifold learning (Roweis and Saul, 2000; Coifman and Lafon, 2006), among others.

Following the manifold hypothesis, we assume that the data are distributed on a (com-
sampled from a random
pact, connected) Riemannian manifold M, and they are i.i.d.
variable X on M. This random variable has a density p(x) with respect to the volume
element ? of (M, g?):

P(X ? A) =

(cid:90)

A

p(x)d?, A ? B(M)

12

Towards Understanding how attention mechanism works in deep learning

where B(M) refers to the Borel set, g? is the Riemannian metric in Assumption 2. Through
an initial embedding H old, we observe the data in Euclidean space:

H old : M (cid:44)? Rn

We also assume that the dataset is sufficiently large (N ? ?). Thus, it is natural to adopt
a continuous perspective.

3.2 Attention limit operator.

Theorem 3 (Limit of attention mechanism for information propagation). Suppose the
assumptions are satisfied, then the dynamics of the feature of each data point in attention
mechanism for information propagation is a first-order approximator (with respect to ?) of
the PDE:

= ?g? H + 2

dH
dt
H(x, 0) = H old(x)

(cid:28) ?p
p

(cid:29)

, ?H

where ?g? is the Laplacian-Beltrami operator of manifold M with the Riemannian metric
given by Neural network (Assumption 2) and p is the density function (Assumption 3).

Based on this theorem, we define the attention limit operator Atg,p by ?g + 2

,
or equivalently ?g + 2 ?? log p, ?g�?, which characterizes the limit increment of features
influenced by the information propagation of attention mechanism.

(cid:68) ?p

(cid:69)
p , ?�

(cid:68) ?p

The attention limit operator reflects the combined information propagation effect of dif-
fusion and density-guided flow (Figure 2). The Atg,p limit operator contains two parts: the
(cid:69)
Laplacian term ? and the particle drift
term. The Laplacian term represents heat
p , ?�
diffusion, indicating heat diffuses uniformly in all directions within a homogeneous medium,
serving to make features tend towards consistency. The particle drift term represents par-
ticles always moving in the direction of the steepest change in probability density rather
than along contour lines, suggesting that density guides the flow of information. Addition-
ally, this PDE can be translated into its stochastic differential equation (SDE) counterpart,
which offers a similar interpretation.

Using this theorem, we can analyze the impact of metric scaling on the information
m2 , m is a constant bigger than 1, then:

propagation rate. If g2 = g1

?g2 = m2?g1

This implies that the rate of information propagation through diffusion can be increased by
compressing distances. The density-guided flow also varies with different metrics. Therefore,
neural networks can modulate the rate of information propagation by learning different
metrics.

Theorem 4 (Heat diffusion formulation). If the dimension of the manifold M is n ?= 2,
then there exists a metric �g such that the Atg,p operator is equivalent to a Laplacian-like

13

Tianyu Ruan and Shihua Zhang

operator of �g:

?g + 2

(cid:28) ?p
p

(cid:29)

, ?�

= f ?�g

where f = p4/n?2. Therefore, attention mechanism for information propagation can be
described by the heat diffusion equation:

dH
dt

= f ?�gH

This theorem allows us to interpret the dynamics of the attention operator by heat
diffusion, where f can be regarded as the specific heat capacity. To be specific, the heat
diffusion on the manifold (M, �g) with specific heat capacity c, thermal conductivity k and
material density ? follows the dynamics (Appendix B):

du
dt

=

1
c?

? � (k?u)

where u(x, t) is the temperature of x at time t. When k = 1, ? = 1, and c = f ?1, this
dynamic is the same as the dynamic of attention mechanism.

In this case, attention mechanism essentially learns a new metric on the manifold and
the data features undergo the transformation process like heat conduction under this new
metric. In this view, neural networks adjust the metric automatically so that features evolve
favorably for tasks. For example, in classification tasks, we desire faster heat conduction
within class regions, leading to quicker feature averaging; and between-class separation
regions undergo slower heat conduction, allowing for distinct features between classes.

Stationary function (Clustering tendency). By comparing this PDE with the stan-
dard heat equation, we explore the equilibrium states of the equation.

Attention limit dynamics:

Standard heat diffusion:

dH
dt
dH
dt

= f ?H

= ?H

Since the information propagation of vanilla self-attention mechanism under the metric
setting represents a first-order approximation of the PDE, we assume that f is a fixed
function. Due to the positivity of f , the two equations above share the same stable states:
any function that is stable under the attention limit dynamics is also stable under the
standard heat equation, and vice versa. Hodge theory provides insight into the relationship
between cohomology classes and harmonic equations: the dimension of harmonic n-forms
equals the dimension of the n-th cohomology group. We have:

dim{f : ?f = 0} = dim(H 0)

Thus, if a manifold is connected, the harmonic equations on it admit only constant
solutions. (Such inferences are not only valid on manifolds. Similar conclusions exist in
graph-based Hodge theory). Therefore, the attention limit dynamic can only stabilize at

14

Towards Understanding how attention mechanism works in deep learning

constant functions, which implies a clustering of features. In practice, this clustering phe-
nomenon does occur when there are too many attention blocks without skip connections
(Dong et al., 2021). Moreover, according to spectral theory, the number of near-zero eigen-
values reflects the number of clusters that may form before merging into a single cluster.
This property provides insight into the finite-time clustering behavior of the dynamical
system.

3.3 Multi-head attention

Multi-head attention combines multiple attention blocks. A k-head attention mechanism is
defined as:

Hi = SiH old, i = 1, � � � , k
H new = [H1V1, � � � , HkVk][W1, � � � , Wk]T

where Si is the similarity matrix for the i-th head, Vi is the value matrix of the i-th head,
W = [W1, � � � , Wk] is a learnable matrix. Reformalize the formula we have:

H new =

k
(cid:88)

i

SiH old �Wi

where �Wi = ViW T
attention block can be viewed as a first-order approximator of a combination of k PDEs:

i . Consequently, when assumptions 3.1 are satisfied, a multi-head self-

(cid:29)

, ?H

(cid:28) ?pi
pi

= ?giH + 2

dHi
dt
Hi(x, 0) = H old(x)
(cid:88)
H new =

HiWi

i

where gi is the Riemannian metric learned by the i-th attention block and pi is the density
function of the random variable, from which data are sampled, with respect to gi. Com-
pared to the single-head attention mechanism, multi-head attention simultaneously learns
multiple pseudo-metrics for information aggregation and combines the aggregated informa-
tion. This architecture may reduce the difficulty of learning meaningful pseudo-metrics,
thereby demonstrating a stronger capability to extract information.

3.4 An analysis for general pseudo-metric setting

Before presenting our analysis for the information propagation process of general pseudo-
metrics in attention mechanism, we provide an explanation of Theorem 3 to strengthen our
understanding of the role of the metric:

H new(x) = H old(x)
(cid:124) (cid:123)(cid:122) (cid:125)
Zeroth?order

+

?
2
(cid:124)

(cid:18)

?g? H old(x) + 2

(cid:28) ?p
p

(x), ?H old(x)

(cid:29)(cid:19)

(cid:123)(cid:122)
First?order

(cid:125)

+Higher order

15

Tianyu Ruan and Shihua Zhang

The zeroth-order term represents the information at the data point x itself, which can be
considered the �nearest� point to x as measured by the metric: since a distance function is
positive definite by definition, the only data point that is nearest to x is itself. Similarly,
the first-order term comes from the the drift-diffusion process at the nearest data point of x
measured by the metric. Inspired by this, if we use a general pseudo-metric function which
may not satisfy the conditions of a metric, the zeroth-order term should correspond to the
information at the nearest data points of x and the first-order term should be related to
a drift-diffusion process at the nearest data points of x, where the nearest data points is
define by pseudo-metric f?(x, y). The main difference is that a metric function d should
satisfy the following three conditions:

� d(x, y) ? 0 and d(x, x) = 0 ?? x = y

� d(x, y) = d(y, x)

� d(x, y) ? d(x, z) + d(z, y)

Therefore, for a metric function, the only data point nearest to x should be itself, while
for general pseudo-metric functions, the nearest data points may not be x itself and not be
unique.
Example 3. In Transformer, given the three conditions hold: (1) ?QT K = P T diag(a1, � � � , an)P ,
where P ? SO(n); (2) f?(x, y) = (cid:80) aix?
the unit sphere and there exists aix?

i, where x? = P x and y? = P y; (3) x and y lie on

i ?= 0, we obtain the following result (Appendix C):

iy?

argminyf?(x, y) = P T

?

?

a1x?
1
(cid:113)(cid:80) a2
i x?2
i

, � � � ,

?
T

?

anx?
n
(cid:113)(cid:80) a2
i x?2
i

Generally, if QT K is non-degenerate and x, y lie on an ellipsoid, f?(x, y) has a unique
minimizer y.

Denote argminyf?(x, y) as Ax. According to our previous analysis, the zeorth-order
effect should be an average of information in Ax and the first-order effect should be related
to a drift-diffusion process.
Theorem 5 (Informal). If Ax = {y?} and certain regularity conditions hold (Appendix C),
we have:

H new(x) = H old(y?) + ?Atf?,pH old(y?) + Higher order
where Atf?,p is a second-order partial differential operator related to the pseudo-metric f?
and the sampling density p.

We prove this theorem, along with a generalized version in which Ax is a manifold
in Appendix C. This first-order analysis demonstrates that using a general function as
the pseudo-metric provides a novel approach for defining nearest points, thereby effectively
establishing a new topology. Since the zeroth-order term may differ from the original feature,
this approach could offer a more flexible and powerful tool for information propagation.
However, it also implies that the training process might be more challenging since this
information propagation process does not necessarily correspond to a continuous dynamical
system.

16

Towards Understanding how attention mechanism works in deep learning

4 From self-attention to metric-attention

4.1 Metric-attention

As demonstrated above, attention mechanism is essentially a combination of an information
propagation process based on a handcrafted similarity computation and a learnable pseudo-
metric f?(�, �). We expect that neural networks can learn beneficial pseudo-metrics from
data. However, the self-attention mechanism formulates the pseudo-metric as:

f?(x, y) = ?xT QT Ky

which can be interpreted as performing a linear transformation on features followed by dot
product.These overly simplistic pseudo-metrics may struggle to capture complex similarity
relationships, and intuitively, we would prefer them to correspond to continuous dynamical
systems for easier training and better generalization. Therefore, we draw on the concept
of metric learning (Yu et al., 2016; Hu et al., 2014) and propose a modified attention
mechanism called metric-attention mechanism, in which f?(x, y) = ? �f?(x) ? �f?(y)?2
2, where
�f? is a learnable function. The metric-attention information propagation mechanism is
detailed in Algorithm 1:

Algorithm 1 Metric-attention information propagation mechanism
Require: Data matrix H old = [h1, h2, � � � , hN ]T , a neural network �f?.
1: Calculate similarity matrix S,

(cid:16)

exp

(cid:80)

k exp

(cid:17)

?? �f?(hi) ? �f?(hj)?2
2
(cid:16)

?? �f?(hi) ? �f?(hk)?2
2

(cid:17)

Sij =

2: Calculate H new = SH old
3: return H new

Note that when �f?(x) adopts a linear transformation, the model reduces to the L2
self-attention mechanism (Kim et al., 2021). Generally, the network �f? requires sufficient
parameters to capture complex relationships while remaining easy to train. To achieve this,
we suggest a single-hidden-layer MLP with a residual connection, which strikes a balance
between these two factors, and this setting is adopted in our experiments for demonstration.
In addition, this information propagation mechanism can be easily extended to graph data
by setting the similarity between non-adjacent nodes to zero.

4.2 Experiments

To evaluate the effectiveness of the metric-attention mechanism while minimizing the influ-
ence of other modules, we construct an information propagation network (IPN) by se-
quentially passing the input features through a linear transformation layer, a series of
attention-based information propagation modules, and another linear transformation layer.
For classification tasks, we employ a softmax classifier to classify features. We choose the
self-attention mechanism, L2 self-attention mechanism, and metric-attention mechanism

17

Tianyu Ruan and Shihua Zhang

as candidates for the information propagation block. The differences among them lie in
how they compute the pseudo-metric. Additionally, we replace the self-attention module
in Transformer with metric-attention to test the compatibility of it with other commonly
used modules. Experimental details can be found in Table 2 and Appendix A.

Table 2: Experimental details. lr: Learning rate. ?: Weight decay. dQ: Size of matrices Q

and K. dmlp: Width of one-hidden-layer MLP in metric-attention.

Dataset

Model

Moon
Mnist
Human segmentation
Multi30K

IPN
IPN
IPN
Transformer

lr

10?3
10?4
10?3
10?5

?

dQ

10?4
10?4
10?4
5 � 10?4

10 � 10
100 � 100
64 � 64
512 � 512

dmlp

10
100
64
512

Following previous studies, we measure the accuracy, robustness, and training efficiency
of different attention mechanisms. For the classification and segmentation tasks, the accu-
racy is measured as the proportion of predicted labels that match the ground truth labels
(or manually annotated labels). For the translation tasks, we use the Bleu metric to re-
flect their performance. Robustness is evaluated by the variance of accuracy at the end of
training. Finally, we demonstrate the training efficiency of these mechanisms by plotting
the training loss and test accuracy curves.

Figure 3: Experimental results on the MNIST dataset. The left subplot shows the loss
curve during the training process. The middle subplot shows the testing accuracy
during training. The right subplot is a violin plot of the test accuracy of three
structures at the end of training.

We conduct experiments on two vector-type data, i.e., the MNIST (Deng, 2012) and
Moon datasets to evaluate the performance of the self-attention, L2 self-attention, and

18

Towards Understanding how attention mechanism works in deep learning

Figure 4: Experimental results on the Moon dataset. (top-left) Visualization of this dataset.
(top-right) the testing accuracy of the three methods at the end of training.
(bottom-left and bottom-right) illustration of the testing accuracy and loss of the
three methods during the training process, respectively.

metric-attention information propagation mechanisms (Figures 3 and 4). Additionally, we
tested the metric-attention mechanism on a graph data (manifold data), i.e., the human
semantic segmentation dataset (Maron et al., 2017; Anguelov et al., 2005; Bogo et al., 2014;
Giorgi et al., 2007; Vlasic et al., 2008) to assess its performance (Figure 5). In these exper-
iments, the metric-attention mechanism resulted in lower classification loss during training
and higher test accuracy at the end of training compared to the baselines, demonstrating
its superior information processing capability and training efficiency. Moreover, violin plots
of 10 repeated experiments illustrate the robustness of this mechanism.

19

Tianyu Ruan and Shihua Zhang

Figure 5: Evaluation of self-attention, L2 self-attention and metric-attention methods on
the Human semantic segmentation dataset. (top-left) Visualization of four in-
stances in the dataset. (top-right) the testing accuracy of the three methods at
(bottom-left and bottom-right) illustration of the testing
the end of training.
accuracy and loss of the three methods during the training process, respectively.

We compared the performance of Transformers equipped with different information
propagation modules on the Multi30k translation task. By examining training loss, testing
loss (perplexity), and Bleu score curves, metric-attention demonstrates superior training ef-
ficiency, model performance and robustness (Figure 6). This suggests that it is compatible
with traditional architectures and demonstrates significant potential.

Compared to L2 self-attention and metric-attention, self-attention shows large variance
in accuracy both during the training process and at the end of training (Figures 3 and 5).
Additionally, in Figure 5, the loss curve also exhibits significant variance and jitter. These

20

Towards Understanding how attention mechanism works in deep learning

Figure 6: Experimental results on the Multi30k dataset. The top-left, bottom-left and
bottom-right figure respectively illustrate the training loss, testing loss and Bleu
scores during the training process. The top-right figure shows the Bleu scores of
three information propagation methods at the end of the training.

observations suggest that the training process of self-attention may not be robust. Kim
et al. (2021) indicate that the Lipschitz property of self-attention is poor, while the L2 form
has a better Lipschitz property, which is beneficial for robust training. Furthermore, our
analysis (Section 3.4) shows that the information propagation mechanism of self-attention
generally does not correspond to a continuous dynamical system; it only corresponds to a
continuous dynamical system under rather ideal assumptions. In contrast, both L2 self-
attention and metric-attention correspond to a continuous dynamical system under more
relaxed conditions. These two reasons can explain the phenomenon of instability in the
training process of the attention architecture observed in the experiments.

21

Tianyu Ruan and Shihua Zhang

From a theoretical perspective, both L2 self-attention and metric-attention correspond
to a continuous dynamical system determined by a learnable Riemannian metric. L2
self-attention obtains this Riemannian metric through linear transformation, while metric-
attention uses a mini-MLP with residual connections to learn the Riemannian metric. This
enables metric-attention to learn more complex metric relationships, thereby enhancing the
performance of the algorithm. This explains the superior results achieved by the metric-
attention architecture in various tasks (Figures 3, 4, 5 and 6).

5 Conclusion

In this study, we first examine the techniques for similarity computation in classical machine
learning algorithms such as manifold learning, clustering and supervised learning. We point
out that attention mechanism is essentially a composition of an information propagation
process based on a handcrafted similarity computation and a learnable pseudo-metric. This
highlights the strong connection between attention mechanism and traditional algorithms.
More importantly, we demonstrate the evolution of techniques over time: traditional algo-
rithms rely on manual design for similarity computation, while neural networks represented
by attention mechanism introduce learnable and adaptive techniques making them more
flexible. Under the assumption that the pseudo-metric is a transformation of a metric
function and some other assumptions, we utilize PDEs to explain the limit properties of
attention mechanisms and translate them into heat equations for intuitive comprehension.
For a general pseudo-metric, we fully account for its differences from a metric and provide a
first-order analysis. This helps us intuitively understand the working principle of attention
mechanism. That is, the features can be interpreted as updates using the nearest neighbors
in terms of the pseudo-metric.

We conclude that the parameters of attention mechanism are designed to learn a pseudo-
metric, which is used to compute pairwise similarities of data points. That implies we can
consider training attention blocks as searching for a helpful pseudo-metric. It shares the
same objective as metric learning. Thus, we integrate metric learning into attention mech-
anism. The difference between attention mechanism and metric learning lies in that metric
learning often directly learns the metric through supervised approaches, while attention
mechanism achieves this implicitly through propagation, making it more flexible. A signif-
icant advantage of attention mechanism is that it allows direct learning of a pseudo-metric
through the labels provided by the task without requiring the metric to serve as the super-
vised signal. Moreover, different layers can learn different metrics, making pseudo-metric
learning more powerful and flexible.

Previous researchers have attempted to understand deep neural networks (DNNs) using
continuous dynamical systems. For example, some studies (E, 2017; Chen et al., 2018)
have modeled residual neural networks (He et al., 2016) as ordinary differential equations
(ODEs). Gai and Zhang (2021) utilized optimal transport to understand residual neural
networks. Song et al. (2020) formalized denoising diffusion probabilistic models (Ho et al.,
2020) into stochastic differential equations (SDEs). Certain studies have linked graph neural
networks with diffusion processes (Li et al., 2024). Some studies utilized measure theory to
understand attention mechanism with residual connections (Geshkovski et al., 2023; Sander
et al., 2022; Vuckovic et al., 2020). These studies share the commonality of interpreting the

22

Towards Understanding how attention mechanism works in deep learning

characteristic layer-by-layer updates of features in DNNs as the temporal dimension of infor-
mation processing. Among these, some studies model the changes in features by dynamics
of measures, while others model these as dynamics of functions. Here we adopt the latter
perspective, distinguishing it from previous studies on attention mechanism (Geshkovski
et al., 2023; Sander et al., 2022; Vuckovic et al., 2020). In addition, We reveal the connec-
tions between attention mechanism and classical algorithms that have not been addressed
before.

However, this study has certain limitations. First, the limit properties require the
satisfaction of three assumptions, which may not hold in engineering practice. Second,
although attention mechanism is a crucial component of neural network architectures like
Transformers, these networks are complex engineering products with many modules (e.g.,
multilayer perceptron, skip connection) that are not yet fully understood as a collaborative
whole. Third, although we have tested the performance of the metric-attention mechanism,
further experiments are needed to figure out how this mechanism performs across different
domains and tasks.

Since attention mechanism can be viewed as a discretization of a heat equation with a
learnable metric, a natural question arises. Can new network blocks be designed based on
other PDEs that might be more universal or effective in specific domains? We anticipate
that this could become a prominent area of research in the future.

Acknowledgments

This work has been supported by the CAS Project for Young Scientists in Basic Re-
search [No. YSBR-034], the National Key Research and Development Program of China
[2019YFA0709501], and the National Natural Science Foundation of China [No. 12126605].

References

Dragomir Anguelov, Praveen Srinivasan, Daphne Koller, Sebastian Thrun, Jim Rodgers,
In ACM SIG-

and James Davis. Scape: Shape completion and animation of people.
GRAPH, pages 408�416. 2005.

Mukund Balasubramanian and Eric L Schwartz. The isomap algorithm and topological

stability. Science, 295(5552):7�7, 2002.

Mikhail Belkin and Partha Niyogi. Laplacian eigenmaps for dimensionality reduction and

data representation. Neural Computation, 15(6):1373�1396, 2003.

James C Bezdek, Robert Ehrlich, and William Full. FCM: The fuzzy c-means clustering

algorithm. Computers & Geosciences, 10(2-3):191�203, 1984.

Federica Bogo, Javier Romero, Matthew Loper, and Michael J Black. Faust: Dataset and
evaluation for 3d mesh registration. In Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, pages 3794�3801, 2014.

Michael M Bronstein and Iasonas Kokkinos. Scale-invariant heat kernel signatures for non-
rigid shape recognition. In 2010 IEEE Computer Society Conference on Computer Vision
and Pattern Recognition, pages 1704�1711. IEEE, 2010.

23

Tianyu Ruan and Shihua Zhang

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Lan-
guage models are few-shot learners. Advances in Neural Information Processing Systems,
33:1877�1901, 2020.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov,
In European

and Sergey Zagoruyko. End-to-end object detection with transformers.
Conference on Computer Vision, pages 213�229. Springer, 2020.

Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural
ordinary differential equations. Advances in Neural Information Processing Systems, 31,
2018.

Ronald R Coifman and St�ephane Lafon. Diffusion maps. Applied and Computational Har-

monic Analysis, 21(1):5�30, 2006.

Corinna Cortes and Vladimir Vapnik. Support-vector networks. Machine learning, 20:

273�297, 1995.

Li Deng. The mnist database of handwritten digit images for machine learning research

[best of the web]. IEEE Signal Processing Magazine, 29(6):141�142, 2012.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-
training of deep bidirectional transformers for language understanding. arXiv preprint
arXiv:1810.04805, 2018.

Kangning Dong and Shihua Zhang. Deciphering spatial domains from spatially resolved
transcriptomics with an adaptive graph attention auto-encoder. Nature Communications,
13(1):1739, 2022.

Linhao Dong, Shuang Xu, and Bo Xu. Speech-transformer: A no-recurrence sequence-
In 2018 IEEE International Conference on

to-sequence model for speech recognition.
Acoustics, Speech and Signal Processing (ICASSP), pages 5884�5888. IEEE, 2018.

Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need:
Pure attention loses rank doubly exponentially with depth. In International Conference
on Machine Learning, pages 2793�2803. PMLR, 2021.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain
Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers
for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Weinan E. A proposal on machine learning via dynamical systems. Communications in

Mathematics and Statistics, 5:1�11, 02 2017. doi: 10.1007/s40304-017-0103-z.

Desmond Elliott, Stella Frank, Khalil Sima�an, and Lucia Specia. Multi30k: Multilingual

english-german image descriptions. arXiv preprint arXiv:1605.00459, 2016.

24

Towards Understanding how attention mechanism works in deep learning

Evelyn Fix. Discriminatory analysis: Nonparametric discrimination, consistency properties,

volume 1. USAF school of Aviation Medicine, 1985.

Kuo Gai and Shihua Zhang. A mathematical principle of deep learning: Learn the geodesic

curve in the wasserstein space. arXiv preprint arXiv:2102.09235, 2021.

Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. A mathematical

perspective on transformers. arXiv preprint arXiv:2312.10794, 2023.

Daniela Giorgi, Silvia Biasotti, and Laura Paraboschi. Shape retrieval contest 2007: Wa-

tertight models track. SHREC Competition, 8(7):7, 2007.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for
In Proceedings of the IEEE Conference on Computer Vision and

image recognition.
Pattern Recognition, pages 770�778, 2016.

Geoffrey E Hinton and Sam Roweis. Stochastic neighbor embedding. Advances in Neural

Information Processing Systems, 15, 2002.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Ad-

vances in Neural Information Processing Systems, 33:6840�6851, 2020.

Junlin Hu, Jiwen Lu, and Yap-Peng Tan. Discriminative deep metric learning for face
verification in the wild. In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pages 1875�1882, 2014.

Yanrong Ji, Zhihan Zhou, Han Liu, and Ramana V Davuluri. Dnabert: Pre-trained bidi-
rectional encoder representations from transformers model for dna-language in genome.
Bioinformatics, 37(15):2112�2120, 2021.

Hyunjik Kim, George Papamakarios, and Andriy Mnih. The lipschitz constant of self-
attention. In International Conference on Machine Learning, pages 5562�5571. PMLR,
2021.

Joseph B Kruskal and Myron Wish. Multidimensional scaling. Number 11. Sage, 1978.

Yibo Li, Xiao Wang, Hongrui Liu, and Chuan Shi. A generalized neural diffusion framework
on graphs. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38,
pages 8707�8715, 2024.

Xingyan Liu, Qunlun Shen, and Shihua Zhang. Cross-species cell-type assignment from
single-cell rna-seq data by a heterogeneous graph neural network. Genome Research, 33
(1):96�111, 2023.

Stuart Lloyd. Least squares quantization in pcm.

IEEE Transactions on Information

Theory, 28(2):129�137, 1982.

Haggai Maron, Meirav Galun, Noam Aigerman, Miri Trope, Nadav Dym, Ersin Yumer,
Vladimir G Kim, and Yaron Lipman. Convolutional neural networks on surfaces via
seamless toric covers. ACM Transactions on Graphics, 36(4):71�1, 2017.

25

Tianyu Ruan and Shihua Zhang

Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation

and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An im-
perative style, high-performance deep learning library. Advances in Neural Information
Processing Systems, 32, 2019.

Fabian Pedregosa, Ga�el Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand
Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent
Dubourg, et al. Scikit-learn: Machine learning in python. Journal of Machine Learning
Research, 12(Oct):2825�2830, 2011.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.

Language models are unsupervised multitask learners. OpenAI Blog, 1(8):9, 2019.

Sam T Roweis and Lawrence K Saul. Nonlinear dimensionality reduction by locally linear

embedding. Science, 290(5500):2323�2326, 2000.

Michael E Sander, Pierre Ablin, Mathieu Blondel, and Gabriel Peyr�e. Sinkformers: Trans-
formers with doubly stochastic attention. In International Conference on Artificial Intel-
ligence and Statistics, pages 3515�3530. PMLR, 2022.

Nicholas Sharp, Souhaib Attaiki, Keenan Crane, and Maks Ovsjanikov. Diffusionnet: Dis-
cretization agnostic learning on surfaces. ACM Transactions on Graphics, 41(3):1�16,
2022.

Amit Singer. From graph to manifold laplacian: The convergence rate. Applied and Com-

putational Harmonic Analysis, 21(1):128�134, 2006.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon,
and Ben Poole. Score-based generative modeling through stochastic differential equations.
arXiv preprint arXiv:2011.13456, 2020.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles,
and Herv�e J�egou. Training data-efficient image transformers & distillation through at-
tention. In International Conference on Machine Learning, pages 10347�10357. PMLR,
2021.

Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of

Machine Learning Research, 9(11), 2008.

Stijn Van Dongen. Graph clustering via a discrete uncoupling process. SIAM Journal on

Matrix Analysis and Applications, 30(1):121�141, 2008.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
Gomez, (cid:32)Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in
Neural Information Processing Systems, 30, 2017.

26

Towards Understanding how attention mechanism works in deep learning

Petar Veli?ckovi�c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and

Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.

Daniel Vlasic, Ilya Baran, Wojciech Matusik, and Jovan Popovi�c. Articulated mesh anima-

tion from multi-view silhouettes. In ACM Siggraph, pages 1�9. 2008.

James Vuckovic, Aristide Baratin, and Remi Tachet des Combes. A mathematical theory

of attention. arXiv preprint arXiv:2007.02876, 2020.

Svante Wold, Kim Esbensen, and Paul Geladi. Principal component analysis. Chemometrics
and Intelligent Laboratory Systems, 2(1):37�52, 1987. ISSN 0169-7439. doi: https://doi.
org/10.1016/0169-7439(87)80084-9. URL https://www.sciencedirect.com/science/
article/pii/0169743987800849. Proceedings of the Multivariate Statistical Workshop
for Geologists and Geochemists.

Haoyi Wu and Kewei Tu. Probabilistic transformer: A probabilistic dependency model for
contextual word representation. In Findings of the Association for Computational Lin-
guistics: ACL 2023. Association for Computational Linguistics, 2023. doi: 10.18653/v1/
2023.findings-acl.482. URL http://dx.doi.org/10.18653/v1/2023.findings-acl.
482.

Zhigang Yao and Yuqing Xia. Manifold fitting under unbounded noise. arXiv preprint

arXiv:1909.10228, 2019.

Zhigang Yao, Jiaji Su, and Shing-Tung Yau. Manifold fitting with cyclegan. Proceedings of

the National Academy of Sciences, 121(5):e2311436121, 2024.

Jun Yu, Xiaokang Yang, Fei Gao, and Dacheng Tao. Deep multimodal distance metric
learning using click constraints for image ranking. IEEE Transactions on Cybernetics, 47
(12):4014�4024, 2016.

Yaodong Yu, Sam Buchanan, Druv Pai, Tianzhe Chu, Ziyang Wu, Shengbang Tong, Ben-
jamin Haeffele, and Yi Ma. White-box transformers via sparse rate reduction. Advances
in Neural Information Processing Systems, 36, 2024.

Shuang Zhang, Rui Fan, Yuti Liu, Shuang Chen, Qiao Liu, and Wanwen Zeng. Applica-
tions of transformer-based language models in bioinformatics: A survey. Bioinformatics
Advances, 3(1):vbad001, 2023.

27

Tianyu Ruan and Shihua Zhang

Appendices

Appendix A. Experiments and results

Toy dataset We evaluated the performance of metric-attention by comparing it with
self-attention and L2 self-attention using the Moon dataset. The Moon dataset is linearly
inseparable. The training and testing data are generated using sklearn with a noise level
of 0.2 (Pedregosa et al., 2011). The training set size is 400, and the test set size is 100.
We used the Adam optimizer and the cross-entropy loss function. For metric-attention, we
employed the Tanh activation function.

MNIST handwritten digit database For the MNIST dataset, we applied the same set-
tings as those used for the toy dataset, including the Adam optimizer and the cross-entropy
loss function. For metric-attention, we employed a one-hidden-layer MLP with a width
of 100, the Tanh activation function, and a residual connection to learn a representation
function as defined in Section 4.1.

Human semantic segmentation The human semantic segmentation dataset (Maron
et al., 2017) consists of numerous human 3D meshes along with their semantic segmenta-
tion. In each mesh, there are thousands of nodes and triangle faces and every face has a
label according to their semantic meaning. We adopted a fixed linear aggregation layer to
transform features of vertices into features of faces. We use the Adam optimizer and the
cross-entropy loss function.

Multi30k dataset We used the Multi30k dataset (Elliott et al., 2016) to train Trans-
formers with different information propagation blocks and evaluate their performance. We
adopted the Adam optimizer, the ReduceLROnPlateau learning schedule, and the cross-
entropy loss function. The number of heads is 8 and the number of layers is 6.

Implementation We implemented the deep neural networks using PyTorch (Paszke et al.,
2019). We conducted all experiments on a desktop computer with NVIDIA 2080Ti and 3.8
GHz AMD Ryzen 7 5800X 8-Core Process and 16 GB of memory and a computer with
NVIDIA 3090Ti. We partly used the Diffusionnet (Sharp et al., 2022) to preprocess the
human semantic segmentation dataset. We adapted the implementation for the Multi30k
dataset from https://github.com/hyunwoongko/transformer.

Appendix B. Proof of limit properties of metric setting

Proof of Theorem 2 Suppose the data points {xi}N
bution on a Riemannian manifold with the density function p(x), then

i=1 are i.i.d. sampled from a distri-

1
?

n
(cid:88)

j=1

Lijf (xj) ?

(cid:18)

1
2

?f (xi) + 2

(cid:28) ?p
p

(cid:29)(cid:19)

(xi), ?f (xi)

28

Towards Understanding how attention mechanism works in deep learning

Proof: (see Coifman and Lafon (2006)). If Theorem 1 is acknowledged, Theorem 2 can be
easily proven. Theorem 1 says:

(cid:16)

(cid:82) exp

(cid:82) exp

(cid:17)

? ?x?y?2
2?
(cid:16)
? ?x?y?2
2?

g(y)d?(y)
(cid:17)

d?(y)

= g(x) +

?
2

?g(x) + Higher order

As a result, letting g(y) = f (y)p(y) and g(y) = p(y), we get the following estimate by taking
a ratio:

(cid:16)

(cid:82) exp

(cid:82) exp

(cid:17)

? ?x?y?2
2?
(cid:16)
? ?x?y?2
2?

f (y)p(y)d?(y)
(cid:17)

p(y)d?(y)

=

f (x)p(x) + ?
p(x) + ?

2 ? (f (x)p(x)) + Higher order
2 ?p(x) + Higher order

(1)

We know that

? (f p) = p?f + f ?p + 2??f, ?p?

Then Eq. (1) can be written as:

(cid:16)

(cid:82) exp

(cid:82) exp

(cid:17)

? ?x?y?2
2?
(cid:16)
? ?x?y?2
2?

f (y)p(y)d?(y)
(cid:17)

p(y)d?(y)

= f (x) +

?
2 (p(x)?f + 2??f, ?p?) + Higher order

p(x) + ?

2 ?p(x) + Higher order
(cid:29)(cid:19)

(cid:18)

= f (x) +

?
2

(cid:28) ?p
p

?f (x) + 2

(x), ?f (x)

+ Higher order

We complete the proof by a simple rearrangement and the law of large numbers.

?

Proof of Theorem 3 If the aforementioned assumptions are satisfied, the dynamics of
the feature of each data point in attention mechanism for information propagation is a
first-order approximator (with respect to ?) of a PDE:

= ?g? H + 2

dH
dt
H(x, 0) = H old(x)

(cid:28) ?p
p

(cid:29)

, ?H

where ?g? is the Laplacian-Beltrami operator of manifold M with the Riemannian metric
given by neural network (Assumption 2) and p is the density function (Assumption 3).
Proof: Denote the matrix of exp{f?(xi, xj)} by W . Attention mechanism for information
propagation can be described by:

H new = D?1W H old

where D is the degree matrix. Subtract H old from both sides of the equation simultaneously,
we get

H new ? H old = (D?1W ? I)H old

29

Tianyu Ruan and Shihua Zhang

By Theorem 2 and the analogous argument in (Coifman and Lafon, 2006), we know that
(cid:0)(D?1W ? I)H(cid:1)(xi) = ?

p � ?H)(xi) + Higher order. Thus, we have

2 � (?H + 2 ?p

(H new ? H old)(xi) =

?
2

�

(cid:18)

?g? H old +

(cid:19)

� ?H old

?p
p

(xi) + Higher order

which is the formula of the Euler method of PDE (n ? ?):

dH
dt

= ?g? H +

?p
p

� ?H

?

Proof of Theorem 4 If the dimension n ?= 2, then there exists a metric �g such that the
At operator equals a Laplacian-like operator, i.e.,

?g + 2

(cid:28) ?p
p

(cid:29)

, ?�

= f ?�g

where f = p4/n?2.
Proof: Given a Riemannian metric g = gij, the Laplacian-Beltrami operator can be calcu-
lated by the following formula:

?g =

1
(cid:112)|g|

?i((cid:112)|g|gij?j)

Given another Riemannian metric �g = e2?g = e2?gij, which is a conformal metric of g, we
directly calculate its Laplacian:

?�g =

?i((cid:112)|�g|�gij?j)

=

1
(cid:112)|�g|
1
en?(cid:112)|g|
= e?2? 1
(cid:112)|g|

?i(e(n?2)?(cid:112)|g|gij?j)

?i((cid:112)|g|gij?j) + (n ? 2)e?2?(?i?)gij?j

= e?2? (?g + (n ? 2)???, ?�?)

We have

e2??�g = ?g + ??(n ? 2)?, ?�?

As n ? 3, n ? 2 ?= 0. Let ? = 2 log p

n?2 , we have:

p4/n?2?�g = ?g + 2

(cid:28) ?p
p

(cid:29)

, ?�

30

?

Towards Understanding how attention mechanism works in deep learning

Dynamic of heat diffusion Heat diffusion on the manifold (M, �g) with the specific heat
capacity c, thermal conductivity k and material density ? has dynamic:

du
dt

=

1
c?

? � (k?u)

When k = 1, ? = 1 and c = f ?1, this dynamic is the same as the dynamic of attention.
Proof: By the Fourier�s law, the flow of heat can be described by a vector field:

q = ?k?u

Denote the heat energy at point x and time t by Q(x, t), we know that the change of
temperature is proportional to the change of heat energy. To be specific:

Additionally, we know that the change in heat energy is equal to the net heat flux:

?Q
?t

= c?

?u
?t

Therefore, we have:

?Q
?t

= ?divq

?u
?t

=

1
c?

? � (k?u)

Let k = 1, ? = 1 and c = f ?1, we have

?u
?t

= f ?u

Appendix C. Proof of limit properties in general pseudo-metric setting

Example 3 We shall solve the following problem:

?

minimize:

subject to:

(cid:88)

(cid:88)

aix?

iy?
i

y?2
i = 1

The derivatives of Laplacian L(y?, ?) = (cid:80) aix?

iy?

i ? ?((cid:80) y?2

i ? 1) are:

?L
?y?
i
?L
??

= aix?

i ? 2?y?
i

(cid:88)

=

y?2
i ? 1

31

Tianyu Ruan and Shihua Zhang

By letting these derivatives equal 0, we have:

? =

y?
i =

i x?2
i

(cid:113)(cid:80) a2
2
aix?
i
(cid:113)(cid:80) a2
i x?2
i

Since for an ellipsoid, there exists a linear transform to transform it into a unit sphere, we
complete the proof.

?

The first-order analysis for limit properties in general pseudo-metric setting:
We firstly introduce our assumptions:

Formalization We focus on the information propagation process of attention mechanism:

H new = SH old

exp (cid:0)?f?(xi,xj )(cid:1)
k exp (cid:0)?f?(xi,xk)(cid:1) . We reformulate the similarity matrix of attention mecha-
(cid:80)

where Sij =

nism S as S?:

S?,ij =

,H old
j

)

(cid:1)

i

f?(H old
2?

exp (cid:0) ?
k exp (cid:0) ? f?(H old

i

2?

(cid:80)

,H old
k )

(cid:1)

where ?/2 represents the time scale. In attention mechanism, the updated representation is
given by H new = S?H old where ? = 1

2 . Based on this, we suppose:

Assumption 1:

? is sufficiently small.

Assumption 2: We adhere to the manifold hypothesis, which posits that: data lie on a
compact, connected Riemannian manifold M and they are i.i.d. sampled from a random
variable X whose density function p(x) with respect to the volume element dx. Suppose we
observe data in the Euclidean space by an embedding H old(x). Besides, we suppose that
we have sufficiently many data.

Assumption 3: Regularity conditions We assume that f? is a smooth function and
Ax = argminyf?(x, y) is a compact geodesic (flat) submanifold of M for all x ? M for
simplicity. In addition, we assume:

dim (cid:0)ker ??2

yf?(x, y)(cid:1) = dim Ax, ?y ? Ax

i.e., the Hessian matrix of f? is non-degenerate in the normal direction of the manifold Ax.
Denote argminf?(x, �) by Ax and denote its ? neighbors {y +z ? M, y ? Ax, z ? B(0, ?)}
by Ax,?. Denote min f?(x, �) as ?(x). Since Ax is a geodesic submanifold, we can use exp to
get the chart of tubular neighborhood {?, U � V }:

? : U � V ? Rn+m

32

Towards Understanding how attention mechanism works in deep learning

where n and m + n are the dimensionalities of Ax and M, respectively. ?(u, 0) ? U �
V, ?(u, 0) ? Ax, ?(U, V ) ? Ax,? and the distance from ?(u, v) to Ax equals ?v?.

We use y to represent a point on M and y? to represent a point on Ax. We denote f ? ?

by �f , H ? ? by �H, and p ? ? by �p.

Since the Hessian of f with respect to to v is non-degenerate, there exists constants

C1, C2 and ?1 sufficiently small, s.t.,

?(x) + C1?v?2 ? �f (u, v) ? ?(x) + C2?v?2, ??v?2 ? ?1

Besides, since exp

(cid:17)

(cid:16) ??v?2
2?

decreases exponentially, we have:

(cid:90)

exp

(cid:19)

(cid:18) ??v?2
2?

(cid:90)

dv ?

?

BC

?

exp

(cid:19)

(cid:18) ??v?2
2?

dv

where BC

?

? is a ball with radius C

?

?. Therefore, let U = B?, ? = C?1/2, we have:

exp

(cid:19) (cid:90)

(cid:18) ??(x)
2?

(cid:33)

(cid:32)

? �f?(u, v)
2?

exp

dv ? exp

(cid:18) ??(x)
2?

(cid:19) (cid:90)

BC

?

?

(cid:32)

? �f?(u, v)
2?

(cid:33)

dv

exp

Lemma 1. By ?, we parameterize Ax,? and have:

(cid:18)

dy =

1 ?

(cid:88)

1
6

Rkl(y?)vkvl

(cid:19)

dy?dv + O

(cid:16)

?3/2(cid:17)

where Rij is the Ricci curvature of M at y, dy is the volume element of M, dy? is the
volume element of Ax, y? = ?(u, 0).

Proof: In normal coordinates, the Taylor expansion of gij is:

gij = ?ij ?

1
3

(cid:88)

k,l

Rikjlvkvl + O(?v?3)

where Rikjl is the sectional curvature. As a result,

(cid:112)det g = 1 ?

(cid:88)

k,l

1
6

Rklvkvl + O(?v?3)

By this calculation, we complete the proof.

Expand �f up to the fourth order: by the Taylor series, for any y ? ?(U ),

�f?(u, v) ? �f?(u, 0) =

ckl
2

(cid:88)

k,l

vkvl +

(cid:88)

k,l,m

dklm
3!

vkvlvm +

(cid:88)

k,l,m,n

eklmn
4!

vkvlvmvn + O

(cid:16)

?5/2(cid:17)

?

where ckl = ?2 �f?
?vk?vl
(or functions of y ? Ax).

, dklm =

?3 �f?
?vk?vl?vm

, eklmn =

?4 �f?
?vk?vl?vm?vn

, where c, d, e are functions of u

33

Tianyu Ruan and Shihua Zhang

Note 1: For smooth functions, we can find C such that O(?5/2) is bounded by C?5/2.

This is because of the compactness of Ax, there exists a B?1 � B?2 such that ?v ? B?1:

(cid:12)
(cid:12)
(cid:12)

�f?(u, v) ? �f?((u, 0) ?

ckl
2

(cid:88)

k,l

vkvl ?

(cid:88)

k,l,m

dklm
3!

vkvlvm?

(cid:88)

k,l,m,n

eklmn
4!

vkvlvmvn

(cid:12)
(cid:12) ? C?v?5
(cid:12)

By the compactness, there exists a C independent of u (or y?).

Lemma 2. Denote (cid:0)1 ? 1
6

(cid:80) Rkl(y?)vkvl

(cid:1) dv by dy?. For y ? Ax,?, y = ?(u, 0),

(cid:33)

�g(u, v)dy?

(cid:32)

exp

(cid:90)

u�V

? �f?(u, v)
2?
(cid:19) (cid:16)

(cid:18) ?(x)
2?

=?m/2 exp

where

m0g(y) + ??Vf? , ?vg? + ?E(y)g(y) + ??c?1,vg + O

(cid:16)

?3/2(cid:17)(cid:17)

m0 =

(cid:90)

exp

(cid:18) ? (cid:80) ckl
2 vkvl
2

(cid:19)

dv, ?c?1,vg =

(cid:88)

c?1
kl

k

?2g
?vkvl

(cid:90)

(cid:88)

Vf? =

exp

(cid:18) ? (cid:80) ckl
2 vkvl
2

(cid:19) ? (cid:80)

k,l,m

dklm
3! vkvlvmvi
2

dv

?
?vi

i
(cid:90)

E(y) =

(cid:90)

(cid:90)

+

+

exp

exp

(cid:19) (cid:80)

k,l,m,n,o,p

dklmdnop

3!�3! vkvlvmvnvovp

dv

exp

(cid:18) ? (cid:80) ckl
2 vkvl
2
(cid:18) ? (cid:80) ckl
2 vkvl
2
(cid:18) ? (cid:80) ckl
2 vkvl
2

(cid:19) ? (cid:80) Rklvkvl
3

(cid:19) ? (cid:80)

k,l,m,n

8

dv

eklmn

4! vkvlvmvn
2

dv

Proof: By the definition, we have:

(cid:32)

exp

(cid:90)

u�V

�g(u, v)dy?

(cid:33)

? �f?(u, v)
2?
2! vkvl ? (cid:80)
ckl

k,l

(cid:32) ? (cid:80)

= exp

(cid:19) (cid:90)

(cid:18) ?(x)
2?

exp

(cid:18)

�

g(y) +

k,l,m

3! vkvlvm ? (cid:80)
dklm

k,l,m,n

eklmn

4! vkvlvmvn

(cid:33)

2?

(cid:88)

vk

?�g
?vk

+

1
2

(cid:88)

vkvl

?2�g
?vk?vl

(cid:19)

(cid:18)

�

1 ?

(cid:88)

1
6

(cid:19)

Rklvkvl

dv + O

(cid:16)

?3/2(cid:17)

34

Towards Understanding how attention mechanism works in deep learning

By the Taylor series, we have:

exp

= exp

(cid:32) ? (cid:80)

k,l

2! vkvl ? (cid:80)
ckl

k,l,m

3! vkvlvm ? (cid:80)
dklm

k,l,m,n

eklmn

4! vkvlvmvn

(cid:33)

2?

(cid:18) ? (cid:80)

k,l

ckl
2 vkvl

(cid:19)

2?

+ exp

(cid:18) ? (cid:80)

k,l

ckl
2 vkvl

(cid:19)

2?

(cid:32) ? (cid:80)

k,l,m

dklm
3! vkvlvm

2?

? (cid:80)

k,l,m,n

+

(cid:33)

eklmn

4! vkvlvmvn
2?

�

+ exp

(cid:18) ? (cid:80)

k,l

ckl
2! vkvl

(cid:19) (cid:80)

k,l,m,n,o,p

2?

dklmdnop

3!�3! vkvlvmvnvovp
2 � 4?2

(cid:16)

?3/2(cid:17)

+ O

Since the integral of an odd function equals zero, we have:

(cid:90)

u�V

(cid:32)

? �f?(u, v)
2?

exp

(cid:33)

�g(u, v)dy?

= exp

(cid:18) ?(x)
2?
(cid:90)

(cid:88)

(cid:19) (cid:90)

exp

(cid:19)

(cid:18) ? (cid:80) ckl
2 vkvl
2?
(cid:19) ? (cid:80)

exp

(cid:18) ? (cid:80) ckl
2 vkvl
2?

g(y)dv

k,l,m

dklm
3! vkvlvmvi
2?

?�g
?vi

dv

+

+

+

+

+

i

(cid:90)

(cid:90)

(cid:90)

(cid:90)

g(y)dv

(cid:19) ? (cid:80)

k,l,m,n

(cid:88)

vkvl

eklmn

4! vkvlvmvn
2?
?2�g
?vkvl

dv

(cid:88)

Rklvkvlg(y)dv

exp

exp

exp

exp

(cid:18) ? (cid:80) ckl
2 vkvl
2?
(cid:18) ? (cid:80) ckl
2 vkvl
2?
(cid:18) ? (cid:80) ckl
2 vkvl
2?
(cid:18) ? (cid:80) ckl
2 vkvl
2?

(cid:19) 1
2
(cid:19) ?1
6
(cid:19) (cid:80)

k,l,m,n,o,p

dklmdnop

3!�3! vkvlvmvnvovp

8?2

g(y)dv + O

(cid:16)

?3/2(cid:17)

Finally, we have:

(cid:32)

exp

(cid:90)

u�V

(cid:33)

�g(u, v)dy?

? �f?(u, v)
2?
(cid:19) (cid:16)

(cid:18) ?(x)
2?

=?m/2 exp

m0g(y) + ??Vf? , ?vg? + ?E(y)g(y) + ??c?1,vg + O

(cid:16)

?3/2(cid:17)(cid:17)

?
The following proposition gives the first-order expansion of information propagation of
attention mechanism when Ax = {y?}, which could be regarded as a generalization of the
heat kernel approximation.

Proposition 1.

(cid:82)

u�V exp
(cid:82)
u�V exp

(cid:17)

(cid:16) ? �f?(u,v)
2?
(cid:16) ? �f?(u,v)
2?

�g �p(u, v)dy?
(cid:17)

�p(u, v)dy?

= g(y?) +

?
m0

(cid:0)(cid:10)Vf? + 2?v log p � c?1, ?vg(cid:11) + ?c?1,vg(cid:1) (y?) + O

(cid:16)

?3/2(cid:17)

35

Tianyu Ruan and Shihua Zhang

Proof: By Lemma 2, we have:

(cid:33)

(cid:32)

? �f?(u, v)
2?

exp

�g �p(u, v)dy?

(cid:90)

??m/2

u�V
(cid:18) ?(x)
2?
(cid:18) ?(x)
2?

(cid:19) (cid:16)

(cid:19) (cid:16)

= exp

= exp

m0gp(y?) + ??Vf? , ?vgp?(y?) + ?Epg(y?) + ??c?1,vgp(y?) + O

(cid:16)

?3/2(cid:17)(cid:17)

m0gp(y?) + ??Vf? , ?vg?p(y?) + ?Vf? , ?vp?g(y?) + ?Epg(y?)

+?p?c?1,vg(y?) + ?g?c?1,vp(y?) + 2??vpc?1, ?vg?(y?) + O

(cid:16)

?3/2(cid:17) (cid:17)

Besides, we have:

(cid:90)

??m/2

exp

u�V
(cid:18) ?(x)
2?

(cid:19) (cid:16)

= exp

(cid:32)

? �f?(u, v)
2?

(cid:33)

�p(u, v)dy?

m0p(y?) + ??Vf? , ?vp?(y?) + ?Ep(y?) + ??c?1,vp(y?) + O

(cid:16)

?3/2(cid:17)(cid:17)

We complete the proof by taking a ratio.

Denote 1
m0

?
(cid:0)(cid:10)Vf? + 2? log p � c?1, ?g(cid:11) + ?c?1g(cid:1) by Atf?,pg. By Proposition 1 and the

law of large numbers, if Ax = {y?}, we have:

H new(x) = H old(y?) + ?Atf?,pH old(y?) + Higher order

which complete the proof of Theorem 5. Generally, we can depict the information propa-
gation of attention mechanism by the following theorem:

Theorem 6 (The first-order expansion for general pseudo-metric). As ? ? 0, the informa-
tion propagation of attention mechanism has the first-order expansion:

H new(x) = EY ?pm0(H old(Y )|Y ? Ax)
(cid:18) h0
pm0

(Y )|Y ? Ax

+EY ?pm0

+ ?

(cid:19)

(cid:18)

? EY ?pm0

(cid:18) h1
pm0

(cid:19)

(Y )|Y ? Ax

EY ?m0p(H old(Y )|Y ? Ax)

(cid:19)

+ Higher order

where

h0(y?) = EpH old(y?) + ?Vf? , ?vp? H old(y?) +

+ p?c?1,vH old(y?) + 2

?vH oldc?1, ?p
h1(y?) = ?Vf? , ?p?(y?) + ?c?1,vp(y?) + Ep(y?)

(cid:68)

(cid:68)

Vf? , ?vH old(cid:69)
(cid:69)
(y?) + H old?c?1,vp(y?)

p(y?)

Here, ?v means to take derivative in the direction of normal space of Ax. ?c?1,v means
to take weighted sum of the second-order derivatives by c?1 in the direction of normal space
of Ax (Lemma 2).

36

Towards Understanding how attention mechanism works in deep learning

Proof:

H old(y)p(y)dy

H old(y)p(y)dy + Higher order

(cid:33)

�H old(u, v)�p(u, v) dy?dy? + Higher order

(cid:19)

(cid:19)

(cid:18) ?f?(x, y)
2?
(cid:18) ?f?(x, y)
2?
? �f?(u, v)
2?

(cid:32)

exp

exp

(cid:90)

M

(cid:90)

exp

=

=

=

Ax,?
(cid:90)

(cid:90)

Ax

V

(cid:90)

(cid:90)

Ax

V

(cid:32)

? �f?(u, v)
2?

exp

(cid:33)

�H old(u, v)

(cid:18)

�p +

(cid:88)

vk

? �p
?vk

(cid:88)

+

vkvl

(cid:19)

?2 �p
?vk?vl

(u, 0) dy?dy? + Higher order

Let H old = 1, we have:

(cid:90)

exp

M

(cid:90)

(cid:90)

=

Ax

V

(cid:19)

(cid:18) ?f?(x, y)
2?
? �f?(u, v)
2?

(cid:32)

exp

p(y)dy

(cid:33) (cid:18)

�p +

(cid:88)

vk

? �p
?vk

(cid:88)

+

vkvl

(cid:19)

?2 �p
?vk?vl

(u, 0) dy?dy? + Higher order

By Lemma 2, we have:

(cid:82)

(cid:82)

Ax

V exp
(cid:82)
(cid:82)
V exp

(cid:16) ? �f?(u,v)
2?
(cid:16) ? �f?(u,v)
2?

Ax

(cid:17) �H old(u, v)�p(u, v) dy?dy?

(cid:17)

�p(u, v) dy?dy?

=

(cid:82)
Ax
(cid:82)
Ax

pm0H old(y?)dy? + ? (cid:82)
pm0(y?)dy? + ? (cid:82)

Ax

h0(y?)dy? + Higher order

Ax
h1(y?)dy? + Higher order

where

h0(y?) = EpH old(y?) + ?Vf? , ?vp? H old(y?) +

+ p?c?1,vH old(y?) + 2

(cid:68)
?vH oldc?1, ?p
h1(y?) = ?Vf? , ?p?(y?) + ?c?1,vp(y?) + Ep(y?)

As a result,

(cid:68)

Vf? , ?vH old(cid:69)
(cid:69)
(y?) + H old?c?1,vp(y?)

p(y?)

(cid:82)

Ax

(cid:82)
V exp
(cid:82)
(cid:82)
V exp

(cid:16) ? �f?(u,v)
2?
(cid:16) ? �f?(u,v)
2?

Ax

(cid:17) �H old(u, v)�p(u, v) dy?dy?

(cid:17)

�p(u, v) dy?dy?

(cid:82) pm0H old(y?) dy?

(cid:82) pm0(y?) dy? +

=

? (cid:82) h0(y?)dy? ? ? (cid:82) h1(y?)dy?
(cid:82) pm0(y?)dy?

(cid:82) pm0H old(y?)dy?
(cid:82) pm0(y?)dy?

+ Higher order

37

Tianyu Ruan and Shihua Zhang

where

(cid:82)

Ax
(cid:82)

pm0H old(y?) dy?

Ax

pm0(y?) dy? = EY ?pm0(H old(Y )|Y ? Ax)
(cid:18) h0
h0(y?) dy?
pm0(y?) dy? = EY ?pm0
pm0
(cid:18) h1
pm0

pm0(y?) dy? = EY ?pm0

(Y )|Y ? Ax

(Y )|Y ? Ax

h1 dy?

(cid:19)

(cid:19)

Ax

(cid:82)

Ax

(cid:82)

(cid:82)

Ax
(cid:82)

Ax

We complete the proof by the law of large numbers.

?

38


