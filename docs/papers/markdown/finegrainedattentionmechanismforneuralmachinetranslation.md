8
1
0
2

r
p
A
3

]
L
C
.
s
c
[

2
v
7
0
4
1
1
.
3
0
8
1
:
v
i
X
r
a

Fine-Grained Attention Mechanism for Neural
Machine Translation

Heeyoul Choi
Handong Global University
heeyoul@gmail.com

Kyunghyun Cho
New York University
kyunghyun.cho@nyu.edu

Yoshua Bengio
University of Montreal
CIFAR Senior Fellow
yoshua.bengio@umontreal.ca

Abstract

Neural machine translation (NMT) has been a new paradigm in machine trans-
lation, and the attention mechanism has become the dominant approach with the
state-of-the-art records in many language pairs. While there are variants of the at-
tention mechanism, all of them use only temporal attention where one scalar value
is assigned to one context vector corresponding to a source word. In this paper,
we propose a ?ne-grained (or 2D) attention mechanism where each dimension of
a context vector will receive a separate attention score. In experiments with the
task of En-De and En-Fi translation, the ?ne-grained attention method improves
the translation quality in terms of BLEU score. In addition, our alignment analysis
reveals how the ?ne-grained attention mechanism exploits the internal structure of
context vectors.

1

Introduction

Neural machine translation (NMT), which is an end-to-end approach to machine translation Kalch-
brenner and Blunsom (2013); Sutskever et al. (2014); Bahdanau et al. (2015), has widely become
adopted in machine translation research, as evidenced by its success in a recent WMT�16 transla-
tion task Sennrich et al. (2016); Chung et al. (2016b). The attention-based approach, proposed by
Bahdanau et al. (2015), has become the dominant approach among others, which has resulted in
state-of-the-art translation qualities on, for instance, En-Fr Jean et al. (2015a), En-De Jean et al.
(2015b); Sennrich et al. (2016), En-Zh Shen et al. (2016), En-Ru Chung et al. (2016a) and En-
Cz Chung et al. (2016a); Luong and Manning (2016). These recent successes are largely due to
better handling a large target vocabulary Jean et al. (2015a); Sennrich et al. (2015b); Chung et al.
(2016a); Luong and Manning (2016), incorporating a target-side monolingual corpus Sennrich et al.
(2015a); Gulcehre et al. (2015) and advancing the attention mechanism Luong et al. (2016); Cohn
et al. (2016); Tu et al. (2016).

We notice that all the variants of the attention mechanism, including the original one by Bahdanau
et al. (2015), are temporal in that it assigns a scalar attention score for each context vector, which
corresponds to a source symbol. In other words, all the dimensions of a context vector are treated
equally. This is true not only for machine translation, but also for other tasks on which the attention-
based task was evaluated. For instance, the attention-based neural caption generation by Xu et al.
(2015) assigns a scalar attention score for each context vector, which corresponds to a spatial lo-
cation in an input image, treating all the dimensions of the context vector equally. See Cho et al.
(2015) for more of such examples.

1

On the other hand, in Choi et al. (2017), it was shown that word embedding vectors have more than
one notions of similarities by analyzing the local chart of the manifold that word embedding vectors
reside. Also, by contextualization of word embedding, each dimension of the word embedding
vectors could play different role according to the context, which, in turn, led to better translation
qualities in terms of the BLEU scores.

Inspired by the contextualization of word embedding, in this paper, we propose to extend the atten-
tion mechanism so that each dimension of a context vector will receive a separate attention score.
This enables ?ner-grained attention, meaning that the attention mechanism may choose to focus
on one of many possible interpretations of a single word encoded in the high-dimensional context
vector Choi et al. (2017); Van der Maaten and Hinton (2012). This is done by letting the atten-
tion mechanism output as many scores as there are dimensions in a context vectors, contrary to the
existing variants of attention mechanism which returns a single scalar per context vector.

We evaluate and compare the proposed ?ne-grained attention mechanism on the tasks of En-De and
En-Fi translation. The experiments reveal that the ?ne-grained attention mechanism improves the
translation quality up to +1.4 BLEU. Our qualitative analysis found that the ?ne-grained attention
mechanism indeed exploits the internal structure of each context vector.

2 Background: Attention-based Neural Machine Translation

The attention-based neural machine translation (NMT) from Bahdanau et al. (2015) computes a
conditional distribution over translations given a source sentence X = (wx

1 , wx

2 , . . . , wx

T ):

p(Y = (wy

1 , wy

2 , . . . , wy

T (cid:48))|X).

(1)

This is done by a neural network that consists of an encoder, a decoder and the attention mechanism.

The encoder is often implemented as a bidirectional recurrent neural network (RNN) that reads the
source sentence word-by-word. Before being read by the encoder, each source word wx
t is projected
onto a continuous vector space:

xt = Ex[�, wx
t ],

(2)

where Ex[�, wx
E and |V | are the word embedding dimension and the vocabulary size, respectively.

t -th column vector of Ex ? RE�|V |, a source word embedding matrix, where

t ] is wx

The resulting sequence of word embedding vectors is then read by the bidirectional encoder recurrent
network which consists of forward and reverse recurrent networks. The forward recurrent network
reads the sequence in the left-to-right order while the reverse network reads it right-to-left:

??
h t =
??
h t =
??
h T +1 are initialized as all-zero vectors or trained as param-
where the initial hidden states
eters. The hidden states from the forward and reverse recurrent networks are concatenated at each
time step t to form an annotation vector h:

??
h t?1, xt),
??
h t+1, xt),

??
h 0 and

??
? (
??
? (

ht =

(cid:104)??

h t;

(cid:105)

??
h t

.

This concatenation results in a context C that is a tuple of annotation vectors:

C = {h1, h2, . . . , hT } .

The recurrent activation functions
(LSTM, Hochreiter and Schmidhuber (1997)) or gated recurrent units (GRU, Cho et al. (2014)).

??
? are in most cases either long short-term memory units

??
? and

The decoder consists of a recurrent network and the attention mechanism. The recurrent network is
a unidirectional language model to compute the conditional distribution over the next target word
given all the previous target words and the source sentence:

p(wy

t(cid:48)|wy

<t(cid:48), X).

2

By multiplying this conditional probability for all the words in the target, we recover the distribution
over the full target translation in Eq. (1).
The decoder recurrent network maintains an internal hidden state zt(cid:48). At each time step t(cid:48), it ?rst
uses the attention mechanism to select, or weight, the annotation vectors in the context tuple C.
The attention mechanism, which is a feedforward neural network, takes as input both the previous
decoder hidden state, and one of the annotation vectors, and returns a relevant score et(cid:48),t:

et(cid:48),t = fAtt(zt(cid:48)?1, ht),

(3)

which is referred to as score function Luong et al. (2016); Chung et al. (2016a). The function fAtt
can be implemented by fully connected neural networks with a single hidden layer where tanh()
can be applied as activation function. These relevance scores are normalized to be positive and sum
to 1.

?t(cid:48),t =

exp(et(cid:48),t)
k=1 exp(et(cid:48),k)

(cid:80)T

.

We use the normalized scores to compute the weighted sum of the annotation vectors

ct(cid:48) =

T
(cid:88)

t=1

?t(cid:48),tht,

(4)

(5)

which will be used by the decoder recurrent network to update its own hidden state by

zt(cid:48) = ?z(zt(cid:48)?1, yt(cid:48)?1, ct(cid:48)).

Similarly to the encoder, ?z is implemented as either an LSTM or GRU. yt(cid:48)?1 is a target-side word
embedding vector obtained by

yt(cid:48)?1 = Ey[�, wy

t(cid:48)?1],

similarly to Eq. (2).
The probability of each word i in the target vocabulary V (cid:48) is computed by
t(cid:48) = i|wy

<t(cid:48), X) = ? (W y

i zt(cid:48) + ci) ,

p(wy

where W y

i is the i-th row vector of W y ? R|V |�dim(zt(cid:48) ) and ci is the bias.

The NMT model is usually trained to maximize the log-probability of the correct translation given
a source sentence using a large training parallel corpus. This is done by stochastic gradient descent,
where the gradient of the log-likelihood is ef?ciently computed by the backpropagation algorithm.

2.1 Variants of Attention Mechanism

Since the original attention mechanism was proposed as in Eq. (3) Bahdanau et al. (2015), there
have been several variants Luong et al. (2016).

Luong et al. (2016) presented a few variants of the attention mechanism on the sequence-to-sequence
model Sutskever et al. (2014). Although their work cannot be directly compared to the attention
model in Bahdanau et al. (2015), they introduced a few variants for score function of attention
model�content based and location based score functions. Their score functions still assign a single
value for the context vector ht as in Eq. (3).

Another variant is to add the target word embedding as input for the score function Jean et al.
(2015a); Chung et al. (2016a) as follows:

et(cid:48),t = fAttY(zt(cid:48)?1, ht, yt(cid:48)?1),

(6)

and the score is normalized as before, which leads to ?t(cid:48),t, and fAttY can be a fully connected neural
network as Eq. (3) with different input size. This method provides the score function additional
information from the previous word. In training, teacher forced true target words can be used, while
in test the previously generated word is used. In this variant, still a single score value is given to the
context vector ht.

3

(a)

(b)

Figure 1: (a) The conventional attention mechanism and (b) The proposed ?ne-grained attention
mechanism. Note that (cid:80)
t ?t(cid:48),t = 1 in the conventional method, and (cid:80)
t(cid:48),t = 1 for all dimension
d in the proposed method.

t ?d

3 Fine-Grained Attention Mechanism

All the existing variants of attention mechanism assign a single scalar score for each context vector
ht. We however notice that it is not necessary to assign a single score to the context at a time, and
that it may be bene?cial to assign a score for each dimension of the context vector, as each dimension
represents a different perspective into the captured internal structure. In Choi et al. (2017), it was
shown that each dimension in word embedding could have different meaning and the context could
enrich the meaning of each dimension in different ways. The insight in this paper is similar to
Choi et al. (2017), except two points: (1) focusing on the encoded representation rather than word
embedding, and (2) using 2 dimensional attention rather than the context of the given sentence.

We therefore propose to extend the score function fAtt in Eq. (3) to return a set of scores correspond-
ing to the dimensions of the context vector ht. That is,

t(cid:48),t = f d
ed

AttY2D(zt(cid:48)?1, ht, yt(cid:48)?1),

(7)

t(cid:48),t is the score assigned to the d-th dimension of the t-th context vector ht at time t(cid:48). Here,
where ed
fAttY2D is a fully connected neural network where the number of output node is d. These dimension-
speci?c scores are further normalized dimension-wise such that

?d

t(cid:48),t =

The context vectors are then combined by

exp(ed
t(cid:48),t)
k=1 exp(ed

(cid:80)T

t(cid:48),k)

ct(cid:48) =

T
(cid:88)

t=1

?t(cid:48),t (cid:12) ht,

.

(8)

(9)

where ?t(cid:48),t is

(cid:104)
t(cid:48),t, . . . , ?dim(ht)
?1

t(cid:48),t

(cid:105)(cid:62)

, and (cid:12) an element-wise multiplication.

We contrast the conventional attention mechanism against the proposed ?ne-grained attention mech-
anism in Fig. 1.

4

4 Experimental Settings

4.1 Tasks and Corpora

We evaluate the proposed ?ne-grained attention mechanism on two translation tasks; (1) En-De and
(2) En-Fi. For each language pair, we use all the parallel corpora available from WMT�151 for
training, which results in 4.5M and 2M sentence pairs for En-De and En-Fi, respectively. In the
case of En-De, we preprocessed the parallel corpora following Jean et al. (2015a) and ended up with
100M words on the English side. For En-Fi, we did not use any preprocessing routine other than
simple tokenization.

Instead of space-separated tokens, we use 30k subwords extracted by byte pair encoding (BPE),
as suggested in Sennrich et al. (2015b). When computing the translation quality using BLEU, we
un-BPE the resulting translations, but leave them tokenized.

4.2 Decoding and Evaluation

Once a model is trained, we use a simple forward beam search with width set to 12 to ?nd a trans-
lation that approximately maximizes log p(Y |X) from Eq. (1). The decoded translation is then
un-BPE�d and evaluated against a reference sentence by BLEU (in practice, BLEU is computed
over a set of sentences.) We use newstest2013 and newstest2015 as the validation and test sets for
En-De, and newsdev2015 and newstest2015 for En-Fi.

4.3 Models

We use the attention-based neural translation model from Bahdanau et al. (2015) as a baseline,
except for replacing the gated recurrent unit (GRU) with the long short-term memory unit (LSTM).
The vocabulary size is 30K for both source and target languages, the dimension of word embedding
is 620 for both languages, the number of hidden nodes for both encoder and decoder is 1K, and the
dimension of hidden nodes for the alignment model is 2K.

Based on the above model con?guration, we test a variant of this baseline model, in which we feed
the previously decoded symbol yt?1 directly to the attention score function fAtt from Eq. (3) (AttY).
These models are compared against the model with the proposed ?ne-grained model (AttY2D).

We further test adding a recently proposed technique, which treats each dimension of word embed-
ding differently based on the context. This looks similar to our ?ne-grained attention in a sense that
each dimension of the representation is treated in different ways. We evaluate the contextualiza-
tion (Context) proposed by Choi et al. (2017). The contextualization enriches the word embedding
vector by incorporating the context information:

cx =

1
T

T
(cid:88)

t=1

NN?(xt),

where NN? is a feedforward neural network parametrized by ?. We closely follow Choi et al. (2017).

All the models were trained using Adam Kingma and Ba (2014) until the BLEU score on the val-
idation set stopped improving. For computing the validation score during training, we use greedy
search instead of beam search in order to minimize the computational overhead. That is 1 for the
beam search. As in Bahdanau et al. (2015), we trained our model with the sentences of length up to
50 words.

5 Experiments

5.1 Quantitative Analysis

We present the translation qualities of all the models on both En-De and En-Fi in Table 1. We
observe up to +1.4 BLEU when the proposed ?ne-grained attention mechanism is used instead of

1 http://www.statmt.org/wmt15/

5

Beam Width
Baseline
+AttY
+AttY2D
+Context(C)
+C+AttY
+C+AttY2D

En-De

En-Fi

1
17.57 (17.62)
19.15 (18.82)
20.49 (19.42)
19.13 (18.81)
20.96 (20.06)
22.37 (20.56)

12
20.78 (19.72)
21.41 (20.60)
22.50 (20.83)
22.13 (21.01)
23.25 (21.35)
23.74 (22.13)

1
6.07 (7.18)
7.38 (8.02)
8.33 (8.75)
7.47 (7.93)
8.67 (9.18)
9.02 (9.63)

12
7.83 (8.35)
8.91 (9.20)
9.32 (9.41)
8.84 (9.18)
10.01 (9.95)
10.20 (10.90)

Table 1: BLEU scores on the test sets for En-De and En-Fi with two different beam widths. The
scores on the development sets are in the parentheses. The baseline is the vanilla NMT model from
Bahdanau et al. (2015) with LSTM and BPE.

the conventional attention mechanism (Baseline vs Baseline+AttY vs Baseline+AttY2D) on the
both language pairs. These results clearly con?rm the importance of treating each dimension of the
context vector separately.

With the contextualization (+Context or +C in the table), we observe the same pattern of improve-
ments by the proposed method. Although the contextualization alone improves BLEU by up to +1.8
compared to the baseline, the ?ne-grained attention boost up the BLEU score by additional +1.4.

The improvements in accuracy require additional time as well as larger model size. The model size
increases 3.5% relatively from +AttY to +AttY2D, and 3.4% from +C+AttY to +C+AttY2D. The
translation times are summarized in Table. 2, which shows the proposed model needs extra time
(from 4.5% to 14% relatively).

Models
Baseline+AttY
Baseline+AttY2D
Baseline+C+AttY
Baseline+C+AttY2D

En-De
2,546
2,902 (+14.0%)
2,758
2,894 (+4.5%)

En-Fi
1,631
1,786 (+9.5%)
1,626
1,718 (+5.7%)

Table 2: Elapsed time (in seconds) for translation of test ?les. The test ?le �newstest2015� for
En-De has 2,169 sentences and �newstest2015� for En-Fi has 1,370 sentences. The numbers in the
parenthesis indicate the additional times for AttY2D compared to the corresponding AttY models.

5.2 Alignment Analysis

Unlike the conventional attention mechanism, the proposed ?ne-grained one returns a 3�D tensor
?d
t(cid:48),t representing the relationship between the triplet of a source symbol xt, a target symbol yt(cid:48) and
a dimension of the corresponding context vector cd
t . This makes it challenging to visualize the result
of the ?ne-grained attention mechanism, especially because the dimensionality of the context vector
is often larger (in our case, 2000.)

Instead, we ?rst visualize the alignment averaged over the dimensions of a context vector:

At,t(cid:48) =

1
dim(ct)

dim(ct)
(cid:88)

d=1

?d

t(cid:48),t.

This computes the strength of alignment between source and target symbols, and should be compa-
rable to the alignment matrix from the conventional attention mechanism.

In Fig. 2, we visualize the alignment found by (left) the original model from Bahdanau et al. (2015),
(middle) the modi?cation in which the previously decoded target symbol is fed directly to the con-
ventional attention mechanism (AttY), and (right) the averaged alignment At,t(cid:48) from the proposed
?ne-grained attention mechanism. There is a clear similarity among these three alternatives, but we
observe a more clear, focused alignment in the case of the proposed ?ne-grained attention model.

6

(a)

(b)

(c)

Figure 2: Attention assignments with different attention models in the En-De translation: (a) the
vanilla attention model (Att), (b) with target words yt(cid:48)?1 (AttY), and (c) the proposed attention
model (AttY2D).

Second, we visualize the alignment averaged over the target:

At,d =

1
|Y |

|Y |
(cid:88)

t(cid:48)=1

?d

t(cid:48),t.

This matrix is expected to reveal the dimensions of a context vector per source symbol that are
relevant for translating it without necessarily specifying the aligned target symbol(s).

Figure 3: Attention assignments with the ?ne-grained attention model. Due to the limit of the space,
only the ?rst 50 dimensions are presented. The vertical and the horizontal axes indicate the source
sub-words and the 50 dimensions of the context vector ht, respectively.

In Fig. 3, we can see very sparse representation where each source word receives different pattern
of attentions on different dimensions.
We can further inspect the alignment tensor ?d
t(cid:48),t by visualizing the d(cid:48)-th slice of the tensor. Fig. 4
shows 6 example dimensions, where different dimensions focus on different perspective of transla-
tion. Some dimensions represent syntactic information, while others do semantic one. Also, syntac-
tic information is handled in different dimensions, according to the word type, like article (�a� and
�the�), preposition (�to� and �of�), noun (�strategy�, �election� and �Obama�), and adjective (�Repub-
lican� and �re-@@�). As semantic information, Fig. 4(f) shows a strong pattern of attention on the
words �Republican�, �strategy�, �election� and �Obama�, which seem to mean �politics�. Although
we present one example of attention matrix, we observed the same patterns with other examples.

6 Conclusions

In this paper, we proposed a ?ne-grained (or 2D) attention mechanism for neural machine transla-
tion. The experiments on En-De and En-Fi show that the proposed attention method improves the
translation quality signi?cantly. When the method was applied with the previous technique, con-
textualization, which was based on the similar idea, the performance was further improved. With
alignment analysis, the ?ne-grained attention method revealed that the different dimensions of con-
text play different roles in neural machine translation.

7

(a)

(b)

(c)

(d)

(e)

(f)

Figure 4: Attention assignments in examplary dimensions with the ?ne-grained attention model:
attentions are focused on (a) article (�a� and �the�), (b) preposition (�to� and �of�), (c) noun (�strat-
egy�, �election� and �Obama�), (d) the alignments, (e) adjective (�Republican� and �re-@@�), and (f)
semantics words representing politics (�Republican�, �strategy�, �election� and �Obama�).

We ?nd it an interesting future work to test the ?ne-grained attention with other NMT models
like character-level models or multi-layered encode/decode models Ling et al. (2015); Chung et al.
(2016a). Also, the ?ne-grained attention mechanism can be applied to different tasks like speech
recognition.

Acknowledgments

The authors would like to thank the developers of Theano Bastien et al. (2012). This research was
supported by Basic Science Research Program through the National Research Foundation of Ko-
rea(NRF) funded by the Ministry of Education (2017R1D1A1B03033341). Also, we acknowledge
the support of the following agencies for research funding and computing support: NSERC, Cal-
cul Qu�ebec, Compute Canada, the Canada Research Chairs, CIFAR and Samsung. KC thanks the
support by Facebook and Google (Google Faculty Award 2016).

References

Bahdanau, D., K. Cho, and Y. Bengio, 2015: Neural Machine Translation by Jointly Learning to

Align and Translate. In Proc. Int�l Conf. on Learning Representations (ICLR).

Bastien, F., P. Lamblin, R. Pascanu, J. Bergstra, I. Goodfellow, A. Bergeron, N. Bouchard, D. Warde-
farley, and Y. Bengio, 2012: Theano : new features and speed improvements. In NIPS 2012 deep
learning workshop.

Cho, K., A. Courville, and Y. Bengio, 2015: Describing multimedia content using attention-based

encoder�decoder networks. IEEE Transactions on Multimedia, 17(11), 1875�1886.

8

Cho, K., B. van Merrienboer, D. Bahdanau, and Y. Bengio, 2014: On the properties of neural ma-
chine translation: Encoder-decoder approaches. In SSST-8, Eighth Workshop on Syntax, Semantics
and Structure in Statistical Translation, pp. 103�111.

Choi, H., K. Cho, and Y. Bengio, 2017: Context-dependent word representation for neural machine

translation. Computer Speech and Language, 45, 149�160.

Chung, J., K. Cho, and Y. Bengio, 2016a: A character-level decoder without explicit segmentation
for neural machine translation. In 54th Annual Meeting of the Association for Computational
Linguistics, pp. 1693�1703.

�, 2016b: Nyu-mila neural machine translation systems for wmt�16. In The First Conference on

Statistical Machine Translation (WMT).

Cohn, T., C. D. V. Hoang, E. Vymolova, K. Yao, C. Dyer, and G. Haffari, 2016: Incorporating
structural alignment biases into an attentional neural translation model. In NAACL-HLT, pp. 876�
885.

Gulcehre, C., O. Firat, K. Xu, K. Cho, L. Barrault, H.-C. Lin, F. Bougares, H. Schwenk, and
Y. Bengio, 2015: On using monolingual corpora in neural machine translation. arXiv preprint
arXiv:1503.03535.

Hochreiter, S. and J. Schmidhuber, 1997: Long short-term memory. Neural computation, 9(8),

1735�80.

Jean, S., K. Cho, R. Memisevic, and Y. Bengio, 2015a: On Using Very Large Target Vocabulary
for Neural Machine Translation. In 53rd Annual Meeting of the Association for Computational
Linguistics.

Jean, S., O. Firat, K. Cho, R. Memisevic, and Y. Bengio, 2015b: Montreal neural machine translation
systems for wmt15. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pp.
134�140.

Kalchbrenner, N. and P. Blunsom, 2013: Recurrent continuous translation models. EMNLP, 3(39),

413.

Kingma, D. P. and J. L. Ba, 2014: Adam: A method for stochastic optimization. arXiv preprint

arXiv:1412.6980.

Ling, W., I. Trancoso, C. Dyer, and A. W. Black, 2015: Character-based neural machine translation.

arXiv preprint arXiv:1511.04586.

Luong, M.-T. and C. D. Manning, 2016: Achieving open vocabulary neural machine translation
with hybrid word-character models. In 54th Annual Meeting of the Association for Computational
Linguistics, p. 1054?1063.

Luong, M.-T., H. Pham, and C. D. Manning, 2016: Effective approaches to attention-based neural
machine translation. In 2015 Conference on Empirical Methods in Natural Language Processing,
pp. 1412�1421.

Sennrich, R., B. Haddow, and A. Birch, 2015a: Improving neural machine translation models with

monolingual data. arXiv preprint arXiv:1511.06709.

�, 2015b: Neural machine translation of rare words with subword units. arXiv preprint

arXiv:1508.07909.

�, 2016: Edinburgh neural machine translation systems for wmt 16. In The First Conference on

Statistical Machine Translation (WMT).

Shen, S., Y. Cheng, Z. He, W. He, H. Wu, M. Sun, and Y. Liu, 2016: Minimum Risk Training
for Neural Machine Translation. In 54th Annual Meeting of the Association for Computational
Linguistics, pp. 1683�1692.

Sutskever, I., O. Vinyals, and Q. V. Le, 2014: Sequence to Sequence Learning with Neural Networks.

In Advances in Neural Information Processing Systems (NIPS).

Tu, Z., Z. Lu, Y. Liu, X. Liu, and H. Li, 2016: Modeling coverage for neural machine translation. In

54th Annual Meeting of the Association for Computational Linguistics, pp. 76�85.

Van der Maaten, L. and G. Hinton, 2012: Visualizing non-metric similarities in multiple maps.

Machine learning, 87(1), 33�55.

9

Xu, K., A. Courville, R. Zemel, and Y. Bengio, 2015: Show, Attend and Tell : Neural Image Caption
Generation with Visual Attention. In 32nd International Conference on Machine Learning, pp.
2048�2057.

10


