From Bytes to Ideas:
Language Modeling with Autoregressive U-Nets
Mathurin Videau1,?, Badr Youbi Idrissi1,?, Alessandro Leite3, Marc Schoenauer2, Olivier Teytaud1, David
Lopez-Paz1
1FAIR at Meta, 2TAU, INRIA and LISN, CNRS & Universit� Paris-Saclay, 3INSA Rouen Normandy,
LITIS, Rouen, France
?Equal contribution

Tokenization imposes a fixed granularity on the input text, freezing how a language model operates
on data and how far in the future it predicts. Byte Pair Encoding (BPE) and similar schemes split
text once, build a static vocabulary, and leave the model stuck with that choice. We relax this
rigidity by introducing an autoregressive U-Net that learns to embed its own tokens as it trains. The
network reads raw bytes, pools them into words, then pairs of words, then up to 4 words, yielding a
multi-scale representation of the sequence. At deeper stages, the model must predict further into the
future�anticipating the next few words rather than the next byte�so deeper stages focus on broader
semantic patterns while earlier stages handle fine details. When carefully tuning and controlling
pretraining compute, shallow hierarchies are on par with strong BPE baselines, and deeper hierarchies
exhibit a promising trend. Because tokenization now lives inside the model, the same system can
handle character-level tasks and carry knowledge across low-resource languages.

Date: June 18, 2025
Correspondence: Mathurin Videau at mathurin.videau@gmail.com
Code: https://github.com/facebookresearch/lingua/tree/main/apps/aunet

5
2
0
2

n
u
J

7
1

]
L
C
.
s
c
[

1
v
1
6
7
4
1
.
6
0
5
2
:
v
i
X
r
a

Figure 1 Three-stage Autoregressive U-Net (AU-Net). The model executes from left to right. The contracting path
compresses the sequence in two steps: Stage 1 processes raw bytes, Stage 2 keeps only the vector at each word
boundary, and Stage 3 keeps one vector per two words. Each contraction and expansion step supports arbitrary pooling
and upsampling patterns. After the deepest stage, the expanding path reverses the contracting path by duplicating
each coarse vector and applying position-specific linear layers. These are combined with skip connections from the
contracting path, gradually restoring sequence length and blending in high-level information. Deeper stages predict
further ahead and capture broad semantics, while shallower stages refine local detail.

1

A
?C?A?T
SA?T? O?N? T?H?E
 M?AC?A?T

SA?T??O?N??T?H?E??M
A?TTied linearStage 1
Dim 512
3 Layers
4% FLOPStage 1
Dim 512
3 Layers
4% FLOPStage 2
Dim 2048
3 Layers
11% FLOP
Stage 2
Dim 2048
3 Layers
11% FLOP
Stage 3
Dim 3072
18 Layers
70% FLOPContractionExpansionDecoder only Transfomer layersDifferent linearsAddition with residual connection

1 Introduction

Language models are about uncovering patterns in a sequence so they can guess what comes next. Before any
of that happens, we must decide what the pieces of that sequence�the tokens�actually are. That choice is
usually frozen in advance by a tokeniser that chops raw text into discrete units long before training begins.
Consider the sentence �The quick brown fox. � A character -level tokeniser feeds the model the stream {T,
h, e, ?, q, u} and asks it to predict the next letter i. A word -level tokeniser, in contrast, hands over {The,
quick} and expects the model to guess brown in one shot. Finer cuts lead to larger sequences and shorten
the look-ahead window, whereas coarser cuts lead to shorter sequences but make each token rarer and harder
to compare and predict. Regardless of granularity, some form of tokenisation is unavoidable: a sequence must
exist before any Transformer can run.

Byte-Pair Encoding (BPE) followed by a simple embedding table is by far the most popular approach. It works
by repeatedly merging the most frequent byte sequences in the training text until a preset vocabulary limit is
reached. This procedure leaves practitioners with just two intuitive dials. The first dial is the training corpus:
whichever text one feeds the algorithm�English prose, source code, or a multilingual mix�determines which
patterns are merged and therefore what the final tokens look like. The second dial is the vocabulary size:
raising this limit lets the merge process run for more steps, producing longer tokens and shorter sequences at
the cost of a larger embedding table and output softmax.

Most issues with tokenisation stem from the embedding operation rather than the splitting act itself. Each
token is typically mapped to an independent vector, meaning the network sees only opaque identifiers and
must rediscover, for instance, that strawberry and strawberries share nine letters. This reliance on isolated
embeddings hampers symbol-level reasoning and complicates transfer to dialects or rare languages. Finally,
this splitting is most often a preprocessing step, locking in a single level of granularity for all subsequent
model layers (see Section 2.2).

To address these limits, our Autoregressive U-Net (Section 2.1), or AU-Net (�oh-net�, /�oU nEt/), learns to
embed information directly from raw bytes, and allows for multiple stages of splitting. The purpose of an
embedding is to map tokens to vectors. Instead of using a lookup table, we use attention directly to embed the
tokens. Self-attention allows vectors at any position to summarize the entire preceding context. This enables
a simple pooling mechanism: we select these contextualized vectors at word boundaries (AU-Net-2), then
word pairs (AU-Net-3), and up to four-word chunks (AU-Net-4), forming a multi-stage embedding hierarchy.
This U-Net like architecture contracts sequences, preserving detail with skip connections, before expanding
them. During expansion, vectors representing coarser information are injected back into more fine grained
representations. Deeper stages, by operating on compressed views, inherently need to anticipate multiple
words ahead, similar to multi-token prediction (Gloeckle et al., 2024) but without auxiliary losses. This effect
allows deeper stages to guide shallower stages at the semantic level, while letting them handle finer details
like spelling.

Contributions (quantified in Section 3).
C1. Adaptive multi-level hierarchy. We train up to four end-to-end embedding stages with arbitrary, user-
specified split functions, extending prior work that relies either on fixed pooling or shallow hierarchies.

C2. Infinite vocab size. By operating directly on bytes, our model avoids predefined vocabularies and

memory-heavy embedding tables, allowing an unlimited number of unique tokens.

C3. Strong performance and scaling. Under identical pre-training budgets, a single level matches strong BPE
baselines, and a two or three-level hierarchy shows promising scaling trends. A selection of the results
are presented in Table 2.

C4. Practical Efficiency . We maintain comparable GPU throughput in wall-clock time instead of purely

theoretical compute gains. Our code is available in Meta Lingua (Videau et al. (2024))1.

C5. Stable scaling laws. We show that moving from token to byte-level training demands new batch size and

learning rate formulas to get smooth optimization.

1https://github.com/facebookresearch/lingua/tree/main/apps/aunet

2

2 Method

2.1 Autoregressive U-Net

Inspired by U-Net-like architectures (Ronneberger et al.,
2015; Nawrot et al., 2022), we propose an autoregres-
sive hierarchical model for language modeling, illustrated
in figure 1. This architecture features a contracting path,
which compresses the input sequence, and an expanding
path, which reconstructs it. Both paths are fully adaptive:
they do not require fixed pooling or upsampling sizes.
Pooling and upsampling operations can be designed inde-
pendently, even if we choose to make them symmetrical
in this paper. The only requirement is a splitting function, which specifies the positions in the sequence where
pooling should occur. This function is detailed in section 2.2.

Table 1 1B equivalent on 370B tokens

BPE
AU-Net 2
AU-Net 3
AU-Net 4

Hellaswag MMLU GSM8k

4e21
3e21
4e21
5e21

27.0
28.8
28.0
31.7

70.2
69.9
72.9
73.7

4.4
3.0
3.7
5.3

Model

FLOP

Our architecture is monolithic: unlike recent approaches (Pagnoni et al., 2024; Neitemeier et al., 2025) that
use local models, we apply attention globally at each stage (or within a sliding window), allowing every input
to attend to previous inputs. This ensures that words or word groups are not processed in isolation. To
preserve fine-grained information that might be lost during contraction, we introduce skip connections between
stages, following the approach in Ronneberger et al. (2015) and Nawrot et al. (2022). We also increase the
hidden dimension at each stage in proportion to its contraction factor, enabling richer representations as the
sequence is contracted. To keep computation tractable at the byte-level stage (Stage 1), where sequences are
longest, we restrict attention to a window.

2.1.1 Pooling and Upsampling

Since our pooling and upsampling are adaptive, we cannot rely on fixed window sizes. To address this, we
explored several pooling and upsampling strategies. In this section, we describe the method used in all
experiments reported in the main text. A complete description of the alternatives and ablation results can be
found in the appendix C.

Figure 2 Pooling simply selects the vectors at the positions specified by the splitting function. Upsampling then
expands each pooled vector to fill the next segment, applying a separate linear layer for each position. For instance,
the pooled vector representing the word �SAT?� is used to help predict �ON?�. This offset lets deeper stages predict
further ahead in the sequence. When using 4 stages, for example, this results in the deepest stage helping for the
prediction of the next four words.

Pooling. We adopt the simplest pooling strategy: selecting the indices identified by the splitting function and
projecting them to the next stage�s dimensionality using a linear layer. Since the preceding layers already
include attention mechanisms, we rely on these to do the pooling implicitly instead of relying on explicit cross
attention as used in Nawrot et al. (2022); Pagnoni et al. (2024).

Upsampling. The upsampling step maps coarse representations to finer ones for the next stage. As illustrated
in Figure 2, we duplicate each coarse vector to match the length of the following segment, applying distinct,
position-specific linear transformations to these duplicates. Since these transformations are shared across
segments but vary by position within a segment, we term this Multi-Linear Upsampling. In our experiments,

3

Residual
ConnectionSAT
 ONSAT

ONmodels with multiple stages are more sensitive to the specific choice of upsampling strategy, whereas for
pooling, many strategies work equally well.

2.1.2 Generation

During training, we process the entire input sequence in parallel, activating all stages simultaneously. At
inference, generation is autoregressive: the byte-level stage is active at every step, while deeper stages activate
less frequently according to the pooling pattern. Skip connections transmit information upward at each
stage, so deeper stages can integrate fine-grained details. This cascading, conditional activation enables
efficient inference: computationally intensive high-level stages activate rarely, but still effectively guide detailed
lower-level predictions. In practice, this means that we need to cache the latest vector at the output of each
stage to correctly propagate deeper stages� outputs.

2.2 Splitting Function

The AU-Net architecture supports flexible splitting strategies to define pooling points at each hierarchical stage.
The primary constraint is that any chosen splitting function must be stable to rightward insertion: appending
bytes should not alter prior pooling decisions, ensuring consistent autoregressive generation. Various methods
(e.g., fixed windows (Nawrot et al., 2022), entropy (Pagnoni et al., 2024), learned rules) are possible. Our
current work splits on spaces using different regular expressions at each stage (details in Appendix B).

This strategy defines a hierarchy: Stage 1 processes raw bytes; Stage 2 pools at word boundaries (identified
by the regex); Stage 3 pools after every two words(or sentence end); and Stage 4 after every four words (or
sentence end). This rule-based approach, inspired by pre-tokenization in systems like GPT-4o�s (Dagan et al.,
2024), is effective for Latin scripts. Extending robustly to languages without clear delimiters remains future
work. Unlike prior approaches Pagnoni et al. (2024); Neitemeier et al. (2025); Slagle (2024) that used similar
splits mainly to replace BPE in a single-stage context, AU-Net uses these user-defined splits for its multi-stage
hierarchical processing.

2.3 Evaluating on different scales

Large language models scale very predictably Kaplan et al. (2020); Hoffmann et al. (2022); Bi et al. (2024).
This allows us to estimate the performance of a model for a large compute budget. But more surprisingly, it
allows us to predict the optimal hyperparameters for models way beyond our ablation budget. Bi et al. (2024)
described a method for sweeping learning rates and batch sizes across a range of small models, and they
demonstrated that these results can be used to predict optimal hyperparameters for larger models. Following
their methodology, we show a different evolution of hyperparameters, both due to the data in our setup and to
the hierarchical model. These hyperparameters are then used to do scaling laws for a bigger range of compute
budgets to compare the baseline architecture and AU-Net. Throughout this paper, the scale of a run is its
total pre-training compute C measured in Floating Point Operation (FLOP):

C =

Fmodel / input-unit
(cid:125)
(cid:123)(cid:122)
(cid:124)
FLOPs per (forward+backward) pass per input unit

�

Ninput-unit
(cid:124)
(cid:125)
(cid:123)(cid:122)
number of units of training input

.

Following Bi et al. (2024), we define model size as the number of FLOPs per input unit instead of relying on
the number of parameters. This allows us to compare models with different architectures fairly. The formula
for the number of FLOP per input-unit for a decoder-only transformer is given by:

Fmodel / input-unit = 6N no-embed
(cid:125)

params
(cid:123)(cid:122)
linear term

(cid:124)

+ 6d L S
(cid:124) (cid:123)(cid:122) (cid:125)
attention term

.

params

where, N no-embed
is the number of parameters, excluding the embeddings. d is the dimension, S the sequence
length and L the number of layers. To scale up, one can either make the model bigger (Fmodel / input-unit ?),
give it more data (Ninput-unit ?), or do both. Gadre et al. (2024) showed that keeping the data-to-model ratio
?input-unit constant is key to getting smooth scaling laws and predictable performance, where:

?input-unit =

Ninput-unit
Fmodel / input-unit

.

4

We adopt this convention in all experiments and report the data-to-model ratio ?input-unit used in the
experiments.

Bytes versus tokens. On DCLM, a token sequence is on average k ? 4.56 times shorter than its byte sequence
when using the LLaMa 3 tokenizer.

Given some compression factor k between bytes and tokens, we want to express the equivalent ?bytes. To do
this, we note that Nbyte = k � Ntoken and Fmodel/byte = Fmodel/token/k. Therefore,

?byte = k2 Ntoken

Fmodel/token

= k2?token.

This factor allows us to compare the performance of our model with the baseline on the same scale, as they
will have seen the same amount of data and spent the same amount of FLOPs per token. Throughout the
paper, we always express the data-to-model ratio in LLaMa 3 tokens (?token).
FLOPS per byte for AU-Net. In the case of AU-Net, we cannot use the same formula as the baseline because of
the contraction and expansion happening in the model. However, we can still use the same formulas as long
as we account for the contraction at each stage. So the total FLOPs per byte for AU-Net is simply the sum of
each stage divided by the contraction factor.

Fmodel/byte =

L
(cid:88)

i=1

F i

model/byte
ki

,

where ki is the contraction factor at stage i.
This property allows us to have models with a higher number of parameters for the same compute budget and
data-to-model ratio.

Hyperparameter scaling laws Bi et al. (2024) showed that the regularity of scaling laws can be exploited to tune
very large models from a sweep over much smaller ones. We replicate their protocol on six miniature versions
of each architecture (baseline Transformer and AU-Net): we perform a quasi-random search over batch size
and learning rate, keep the configurations within 1% of the best validation loss, and fit BSZ(C) = A C ? and
LR(C) = B C ? to those points, with parameters A, ?, B and ?. We find the following formulas at the byte
level for AU-Net:

BSZAU-Net(C) = 0.66C 0.321

LRAU-Net(C) = 6.6 � C ?0.176.

And we run the same tuning for the BPE baseline, for which we find:

BSZBPE(C) = 29.9C 0.231

LRBPE(C) = 19.3 � C ?0.177.

3 Experimental Results

3.1 Experimental Setup
Data. For all experiments, we used DCLM (Li et al., 2024) as our pretraining dataset, excluding a very small
fraction for validation. This is around 4T training tokens (of GPTNeoXTokenizer). The corpus is mostly
English and targets mainly natural language understanding, i.e., it contains a marginal amount of code or
maths.
Baselines. We compare our approach to three different baselines: Transformers equipped with the BPE
tokenizer of LLaMa 3, Transformers trained directly on bytes, and Mamba (Gu and Dao, 2024) trained directly
on bytes. To keep the comparison fair, we trained each baseline with the same amount of data or compute.
For example, if a data budget of 273B training bytes is used to train the bytes level or AU-Net model, this
budget is converted to 60B training tokens for a transformer with LLaMa 3 tokenizer (Grattafiori et al., 2024)
because of the 4.56 compression rate measured on the DCLM corpus.
Hyperparameters. For a detailed overview of the hyperparameters, see appendix D. As explained in section 2.3,
we sweep batch size and learning rate values across model scales ranging from 25M to 500M. Then, we
extrapolate the best learning rate and batch size for any given compute budget.

5

Evaluation Metrics. All models are evaluated on a broad set of downstream tasks in a zero-shot setting,
occasionally including a few in-context examples directly in the prompt. These tasks fall into two categories:
(i) multiple-choice (MCQ) tasks, where the correct answer is selected as the option with the lowest normalized
negative log-likelihood (divided by the number of characters) Brown et al. (2020); and (ii) open-ended
generation tasks, where the model is allowed to freely generate its answer.
To highlight the strengths of AU-Net, we include specialized benchmarks targeting character-level manipulation
(CUTE Edman et al. (2024) appendix E) and low-resource language translation (FLORES-200, Costa-jussa
et al. (2024) section 3.4).

For clarity, we report a selection of key benchmark results in the main tables, including Hellaswag, ARC-Easy,
ARC-Challenge, MMLU, NQ, TQA, and GSM8K. Also, we report 95% confidence intervals for all tables using
bootstrap. A full breakdown of all evaluation results is provided in the appendix F.

In addition to task performance, the total training FLOPs and training throughput are provided for each
model, measured in bytes per second per GPU (bps) on H100 80GB GPUs (internal cluster) during the actual
training.

Implementation Details. As scaling is key to the success of large language models, our implementation balances
efficiency and simplicity. We use sequence packing along with full attention, a strategy shown to have little
to no impact on downstream performance (Li et al. (2024)). To reduce GPU memory pressure, all our
experiments rely on Fully Sharded Data Parallelism (FSDP).

For additional speed-ups, the entire model is compiled with torch.compile. Compilation, however, requires a
static computation graph, which clashes with the variable-length outputs produced by our adaptive pooling:
the number of bytes per word (and thus per stage) naturally varies across sentences. We resolve this by fixing
a maximum sequence length at every stage: sequences that exceed the limit are truncated abruptly, and
shorter ones are padded. This compromise yields a graph that is static for compilation while still supporting
adaptive hierarchical pooling in practice.

3.2 Equal Data Budget Results

We evaluate the effectiveness of hierarchical pooling by fixing the model�s primary hidden dimension to
2048 and maintaining a constant total training-data budget. The hidden dimension at each stage is scaled
proportionally to its contraction ratio as described in section 2.1. For instance, the byte-level stage uses a
dimension of 2048/4 = 512, the word-level stage uses 2048, and the 2-word level uses 1.5 � 2048 = 3072,
continuing in this manner for deeper stages. We assess the downstream performance of language models with
2, 3, and 4 stages at the 1B parameter scale. For the 8B model, we evaluate only the 1-stage configuration for
now. All variants are compared against a Transformer baseline using the LLaMA 3 tokenizer of the same
main hidden dimension. More ablations regarding pooling and the number of layers per stage can be found in
the appendix C.

As shown in table 2, hierarchical models consistently match or outperform their BPE-based counterparts.
This trend holds across various configurations and becomes especially pronounced as we introduce more
hierarchical stages. Notably, multi-stage AU-Net models (e.g., AU-Net 3 and AU-Net 4) outperform BPE
baselines on several benchmarks.

An interesting exception to this pattern is the TQA benchmark, which is a knowledge-intensive task evaluating
the generation of the model. AU-Net models along with byte-level baselines consistently underperform on
TQA compared to BPE-based models. This suggests that the performance gap may not stem solely from the
hierarchical structure. However, as model size and training data scale (e.g., at the 8B or 1B, 370B tokens
scale), this discrepancy seems to vanish.

We observe early signs of diminishing returns beyond a certain hierarchical depth. While AU-Net 4 improves
on reasoning-heavy tasks such as ARC-C and GSM8k, gains on benchmarks like Hellaswag and TQA are
less consistent. However, this effect may stem not from hierarchy itself, but from data efficiency: deeper
hierarchies might require more training data to fully realize their potential. Supporting this interpretation, we
find that AU-Net 2 and AU-Net 4 benefit significantly from additional training data, and that MMLU and
GSM8k performances continue to improve with increased stage, even at fixed scale.

6

Table 2 Downstream results comparing AU-Net to BPE and byte-level baselines. We report accuracy on key benchmarks
with 95% confidence intervals where applicable. Literature models are shown in italics; all models are trained on the
same corpus, unless specified. AU-Net variants differ in the number of stages. We also report compute budget and
empirical training speeds in bytes/sec.

Model

ParamsEmb. Flops bps

Hellaswag ARC E

ARC C MMLU

NQ

TQA

GSM8k

Dim=2048 (1B model), 60B tokens (data-to-model ratio of 10)

Transf. Byte
Mamba Byte
Transf. BPE
AU-Net 2
AU-Net 3
AU-Net 4

1.3B 1M 4e21
1.3B 1M 3e21
1.8B 525M 7e20
1.3B 1M 5e20
2.5B 1M 7e20
4.2B 1M 8e20

47k
32k
210k
225k
180k
155k

63.0 �1.0 61.2 �1.9 34.7 �2.7 24.7 �0.7 8.8 �0.9
63.0 �0.9 60.3 �2.0 33.6 �2.8 25.1 �0.7 8.2 �0.9
63.6 �1.0 62.8 �1.9 36.5 �2.7 26.2 �0.7 8.8 �0.9
64.2 �0.9 64.4 �1.9 35.2 �2.8 24.8 �0.7 8.8 �0.9
67.4 �0.9 65.9 �1.9 36.7 �2.7 26.3 �0.7 9.6 �1.0
66.4 �0.9 67.4 �1.9 37.0 �2.8 26.3 �0.7 5.1 �0.7

21.4 �0.8 2.5 �0.9
21.2 �0.7 2.1 �0.8
26.3 �0.8 2.3 �0.8
20.4 �0.7 2.7 �0.9
22.6 �0.8 2.3 �0.8
15.5 �0.7 3.5 �1.0

Dim=2048 (1B model), 370B tokens (data-to-model ratio of 40)

Transf. BPE
AU-Net 2
AU-Net 3
AU-Net 4

1.8B 525M 4e21
1.3B 1M 3e21
2.5B 1M 4e21
4.2B 1M 5e21

DCLM-1B-5�(145B)1
MegaByte (263B)2
Hierarchical (263B)3

1B
1.1B
1.1B

207M 1e21

-
-

-
1e21

210k
225k
180k
155k

-
73k
-

Dim=4096 (8B model), 200B tokens (data-to-model ratio of 5)

Transf. BPE
AU-Net 2

7.5B
9e21
1B
7.9B 1M 1e22

DCLM-7B-2�(276B)1
Hierarchical (263B)3
BLT (220B)� 4

7B
9.2B
8B

413M 1e22
1e22
1e22

-
-

BLT (1T)? 4
DCLM-7B (2.5T)? 5
LLaMa 3.1 (15T)� 5

8B
7B
8B

-

5e22
413M 1e23
6e23
1B

43k
41k

-
15k
-

-
-
-

? Trained on mix of DCLM and other datasets

� Trained on different corpus than DCLM

1 DCLM Li et al. (2024)
2 MegaByte Yu et al. (2023)
3 Hierarchical Neitemeier et al. (2025)
4 BLT Pagnoni et al. (2024)
5 LLaMa 3.1 Grattafiori et al. (2024)

70.2 �0.9 68.6 �1.9 38.5 �2.8 27.0 �0.7 13.5 �1.1 37.2 �0.9 4.4 �1.1
69.9 �0.9 68.6 �1.9 38.9 �2.7 28.8 �0.7 13.0 �1.1 32.5 �0.9 3.0 �0.9
3.7 �1.0
72.9 �0.9 72.3 �1.8 43.3 �2.8 28.0 �0.7 15.3 �1.2
39.1 �0.9
14.0 �1.1 35.5 �0.9 5.3 �1.2
73.7 �0.9

43.2 �2.9 31.7 �0.7

72.6 �1.8

66.1
38.9
46.5

70.2
54.9
65.0

40.6
23.4
30.5

26.4
25.1
26.0

-
-
-

29.3
9.6
9.6

1.1

-

77.2 �0.8 74.5 �1.8 49.2 �2.8 49.6 �0.8 21.1 �1.4
22.1 �1.3
79.1 �0.8

80.0 �1.6 51.2 �2.9

51.1 �0.8

77.8
56.3
72.2

78.1
76.6
66.8

52.6
44.2
38.8

50.8
32.0
25.2

-
-
-

51.1 �0.9
50.9 �0.9

10.7 �1.7
10.0 �1.6

50.9
33.1
-

4.3
-
-

80.6
80.4
83.3 �0.8 80.7 �1.5 54.8 �2.9 66.4 �0.8 29.1 �1.5 64.4 �0.9 54.7 �2.7

79.6
82.2

52.1
59.9

57.4
63.7

-
52.7

-
2.5

-
-

Finally, when comparing our models to similarly sized baselines from the literature (italicized in the table),
we find that AU-Net remains competitive, even while using significantly less training data. For instance,
BLT (1T) uses approximately 5� more compute than our 8B model, while only being better on MMLU.
Importantly, comparisons with literature models are fair, as all were trained on the same corpus: DCLM
(except for BLT (220B) and LLaMa 3.1 (15T)).

To further evaluate our approach, we now turn to scaling laws to better quantify how our architecture compares
to a standard Transformer with BPE. We focus on AU-Net 2 and AU-Net 3, using a data-to-model ratio of 2.
This choice is motivated by the diminishing returns observed when moving from AU-Net 3 to AU-Net 4 under
the same data-to-model ratio.

3.3 Scaling laws

Using the learning rate and batch size formulas (Section 2.3), we run pretrainings for a range of compute
budgets ranging from 1e19 to 1e22 flops (corresponding to models from 150M to 5.3B non embedding
parameters) for the baseline, with a data-to-model ratio of 10. This is roughly 2� the optimal data-to-model
ratio found by Kaplan et al. (2020).

The list of models chosen for each budget is detailed in the appendix G. Figure 3 shows the evolution of
performance on 6 downstream tasks for AU-Net and the BPE baseline. Here we mainly notice that 2 and 3
stage AU-Net models can match the performance of the BPE baseline when carefully controlling for compute

7

Figure 3 Downstream task performance scaling with compute (1e19-1e22 FLOPs). AU-Net (2/3 stages) generally tracks
a strong BPE Transformer baseline, which itself performs competitively against much larger models (e.g., LLaMa 3.1
8B on 15T tokens
100x compute). While AU-Net matches the baseline on tasks like Hellaswag and ARC Easy, and
catches up on TQA at higher compute, its performance improvement phase on MMLU and GSM8K appears to start
later. The general underperformance on GSM8K is also linked to limited math data in the DCLM pretraining corpus.

budget. This is the case for Hellaswag, Arc Easy, and NQ. For TQA, AU-Net both for 2 and 3 stages starts
with a performance gap, but the 3 stage model catches up at 1e22 flops. However, both 2-stage and 3-stage
AU-Net models are still behind the BPE baseline at 1e22 flops for GSM8K and MMLU. Most downstream
tasks follow a sigmoid pattern: performance is near chance at low compute, then rapidly improves before
plateauing. For AU-Net models, this transition appears to occur slightly later on tasks like GSM8K and
MMLU, suggesting that the benefits of a deep hierarchy may become more pronounced at larger scales.
Nevertheless, on many benchmarks, both our AU-Net variants and our BPE baseline achieve results remarkably
close to those of considerably larger models like LLaMa 3.1 8B (pretrained on 15T tokens, representing
100 times more compute than our largest run shown here). This proximity underscores the strength of our
BPE baseline, making AU-Net�s ability to match or trend towards it particularly noteworthy. The primary
exception where this close tracking is less apparent is GSM8K; however, this underperformance across all our
models is likely due to the pretraining corpus, as DCLM contains very little math data.

3.4 Extended Evaluations

We present results highlighting two specific advantages of byte-level training with AU-Net over BPE-based
Transformers: improved performance on multilingual benchmarks (Table 3) and character-level manipulation
tasks (Table 7 in the appendix E).

Table 3 show that both models perform surprisingly well on non-English languages, despite the fact that the
training corpus (DCLM) is heavily filtered to contain mostly English.

Cross-lingual generalization within language families. On the multilingual MMLU benchmark (Table 3 right),
languages using Latin scripts consistently benefit from byte-level modeling. We observe strong positive transfer
between related languages. For example, Germanic languages such as German, Swedish, and Dutch show an
average gain of around +3.0 points, while Romance languages like Italian, Spanish, Portuguese, and French
improve by approximately +4.0 points. These results suggest that operating at the byte level allows the

8

0.40.60.8AccuracyHellaswag0.20.40.6TQA0.10.20.3NQ102010221024Compute (FLOPs)0.40.60.8AccuracyArcEasy102010221024Compute (FLOPs)0.40.6MMLU102010221024Compute (FLOPs)0.00.20.4GSM8KBaseline2 stages3 stagesLlama 3.1 8B 15T tokensTable 3 Multilingual evaluation. Left: BLEU scores on the FLORES-200 benchmark across multiple languages. Higher
scores indicate better translation quality. Right: MMLU Exact Match (%) across 26 non-English languages. Results
are averaged per language across all tasks.

FLORES-200
(BLEU)

German
Dutch
Afrikaans
Faroese
Icelandic
Limburgish
Luxembourgish

Italian
Friulian
Ligurian
Lombard

Sardinian
Sicilian
Venetian

Spanish
Asturian
Catalan
Occitan

Portuguese
Galician
Papiamento
Kabuverdianu

Esperanto

Average

Lang. ? Eng.

Eng. ? Lang.

BPE

AU-Net 2

BPE

AU-Net 2

34.4 �1.2
24.7 �1.0
32.0 �1.3
8.7 �0.7
7.8 �0.6
15.3 �0.9
11.4 �0.8

29.1 �1.0
14.6 �0.8
16.5 �0.9
12.9 �0.9

14.3 �0.8
11.7 �0.8
19.8 �1.0

28.2 �1.0
24.0 �1.1
28.1 �1.1
28.0 �1.2

42.0 �1.3
29.6 �1.1
17.3 �0.9
13.7 �0.9

15.9 �1.0

20.9 �0.2

33.9 �1.2
25.0 �1.0
35.7 �1.3
9.9 �0.8
9.0 �0.7
19.9 �1.0
14.7 �0.9

30.1 �1.0
19.1 �1.0
21.8 �1.0
19.2 �1.0

18.2 �1.0
16.8 �0.9
25.4 �1.1

29.3 �1.0
28.6 �1.1
33.0 �1.2
35.5 �1.2

43.6 �1.3
34.0 �1.2
22.1 �1.1
20.8 �1.1

19.3 �1.0

24.6 �0.2

16.7 �0.8
12.3 �0.6
14.8 �0.8
1.8 �0.3
1.7 �0.3
5.7 �0.4
2.6 �0.3

15.1 �0.7
3.2 �0.3
3.4 �0.3
5.2 �0.4

4.3 �0.4
3.9 �0.4
5.8 �0.4

20.2 �0.7
10.3 �0.6
9.6 �0.5
4.8 �0.4

25.3 �1.0
9.9 �0.5
2.5 �0.3
2.4 �0.3

3.6 �0.4

8.0 �0.1

15.6 �0.7
11.7 �0.6
16.1 �0.8
2.9 �0.4
2.5 �0.3
6.7 �0.5
4.0 �0.3

15.3 �0.6
4.0 �0.3
3.9 �0.3
4.2 �0.3

4.5 �0.4
4.7 �0.4
5.6 �0.4

19.8 �0.7
8.2 �0.5
10.7 �0.6
6.2 �0.4

25.4 �1.0
10.2 �0.6
6.3 �0.4
5.1 �0.4

5.9 �0.4

8.7 �0.1

MMLU

English

Arabic
Bengali
Chinese
Czech
Dutch
Finnish
French
German
Greek
Hindi
Hungarian
Indonesian
Italian
Japanese
Korean
Persian
Polish
Portuguese
Romanian
Russian
Spanish
Swahili
Swedish
Telugu
Thai
Turkish
Vietnamese

Average

BPE

AU-Net 2

49.6 �0.8

29.1 �0.8
27.5 �0.7
33.0 �0.8
30.7 �0.8
34.5 �0.8
29.0 �0.7
37.3 �0.8
36.0 �0.8
29.2 �0.8
27.9 �0.7
29.0 �0.8
34.9 �0.8
36.2 �0.8
29.5 �0.7
28.4 �0.7
28.7 �0.7
30.3 �0.8
37.2 �0.8
34.0 �0.8
30.9 �0.8
37.6 �0.8
28.8 �0.7
33.5 �0.8
26.8 �0.7
28.0 �0.7
29.1 �0.7
31.4 �0.8

31.4 �0.1

51.1 �0.8

29.5 �0.8
27.6 �0.8
28.0 �0.7
32.2 �0.8
37.1 �0.8
29.3 �0.7
40.7 �0.8
37.6 �0.8
30.5 �0.8
27.5 �0.7
30.1 �0.8
37.3 �0.8
39.0 �0.8
28.2 �0.7
28.2 �0.8
28.6 �0.7
32.0 �0.8
40.9 �0.8
36.9 �0.8
31.2 �0.8
41.4 �0.8
29.9 �0.8
36.0 �0.8
27.4 �0.7
27.5 �0.7
30.0 �0.7
30.7 �0.7

32.4 �0.1

model to capture shared orthographic and morphological patterns across related languages.

Transfer to low-resource languages. The FLORES-200 benchmark (Table 3 left) includes many regional and
low-resource languages that are underrepresented or absent in the training data. This setting allows us to
test the model�s ability to generalize based on subword morphology and shared linguistic roots. Byte-level
modeling provides the flexibility to construct meaningful representations without requiring the presence of
these languages in the tokenizer or training corpus. We observe consistent gains in translation tasks into
English, where the model must primarily understand the source language. The advantage is particularly
clear for languages that share syntactic or morphological traits with more dominant relatives in the same
family. This also highlights the robustness of our model: it can produce meaningful translations even with
out-of-vocabulary words or forms unseen during training. In the reverse direction (English to low-resource),
generation remains more challenging.

4 Related Work

Traditional tokenization methods are important for computational efficiency (Ali et al., 2024; Rajaraman
et al., 2024; Gu et al., 2024; Lester et al., 2024), but impose fixed granularities. Early attempts to overcome
this rigidity explored adaptive vocabularies (Zheng et al., 2024), n-gram combinations (Deiseroth et al., 2024),
or alternative splitting criteria like entropy (Pagnoni et al., 2024). Our work, AU-Net, advances this by
integrating tokenization and representation learning into a multi-level, autoregressive U-Net architecture that
operates directly on bytes.

This hierarchical, adaptive-pooling design distinguishes AU-Net from prior works. For instance, Megabytes
(Yu et al., 2023) introduce a two stage LLM using local models but with fixed-size token blocks, unlike
AU-Net�s input-adaptive pooling. Neitemeier et al. (2025), Byte Latent Transformers (BLT) (Pagnoni et al.,
2024), and SpaceByte (Slagle, 2024) also process bytes or use specialized splitting functions. However, they
typically aim to replace BPE for a single effective processing stage or use local attention mechanisms. In
contrast, AU-Net leverages user-defined splits within a multi-stage architecture featuring distinct pooling

9

strategies that differ from the cross-attention methods in Nawrot et al. (2022); Pagnoni et al. (2024). Nawrot
et al. (2022) defined a similar U-Net architecture but with fixed pooling, much smaller models, and their
evaluations mainly focus on perplexity.

5 Conclusion

This paper introduces AU-Net, an autoregressive U-Net that processes raw bytes and learns hierarchical token
representations. By dynamically pooling bytes into words and multi-word chunks, AU-Net eliminates the
need for predefined vocabularies and large embedding tables. Experiments show that AU-Net matches strong
BPE baselines under controlled compute budgets, with deeper hierarchies demonstrating promising scaling
trends. Furthermore, its byte-level operation leads to improved performance on character-level tasks and
better generalization to low-resource languages. This approach offers a flexible and efficient alternative to
traditional tokenization methods, paving the way for more adaptable and versatile language models.

Limitations and further work

Our work uses DCLM, which is an English-only corpus. A direct limitation of our work is that it does not
support non-space-based languages, and it needs a predefined splitting function. This shows, for example, for
Chinese MMLU scores that are lower than the BPE baseline. One extension could be to learn directly the
splitting function. On the software side, as the number of parameters increases with the number of stages,
FSDP already struggles to overlap computation and communication even at 3/4 stages, it needs a minimum
amount of inputs to be fully overlapped.

References

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large
language models via multi-token prediction. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller,
Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, 41st International Conference on Machine Learning,
volume 235, pages 15706�15734, 2024.

Mathurin Videau, Badr Youbi Idrissi, Daniel Haziza, Luca Wehrstedt, Jade Copet, Olivier Teytaud, and David Lopez-
Paz. Meta Lingua: A minimal PyTorch LLM training library, 2024. URL github.com/facebookresearch/lingua.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmen-
tation. In Nassir Navab, Joachim Hornegger, William M. Wells, and Alejandro F. Frangi, editors, Medical Image
Computing and Computer-Assisted Intervention, pages 234�241, 2015.

Piotr Nawrot, Szymon Tworkowski, Micha? Tyrolski, Lukasz Kaiser, Yuhuai Wu, Christian Szegedy, and Henryk
Michalewski. Hierarchical transformers are more efficient language models. In Marine Carpuat, Marie-Catherine
de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Findings of the Association for Computational Linguistics,
pages 1559�1571, 2022.

Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, et al. Byte latent transformer: Patches scale better than tokens.

arXiv:2412.09871, 2024.

Pit Neitemeier, Bj�rn Deiseroth, Constantin Eichenberg, and Lukas Balles. Hierarchical autoregressive transformers:
Combining byte- and word-level processing for robust, adaptable language models. In 13th International Conference
on Learning Representations, 2025.

Gautier Dagan, Gabriel Synnaeve, and Baptiste Roziere. Getting the most out of your tokenizer for pre-training and
domain adaptation. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan
Scarlett, and Felix Berkenkamp, editors, 41st International Conference on Machine Learning, volume 235, pages
9784�9805, 2024.

Kevin Slagle. SpaceByte: Towards deleting tokenization from large language modeling. In A. Globerson, L. Mackey,
D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing
Systems, volume 37, pages 124925�124950, 2024.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec

Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv:2001.08361, 2020.

10

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego
de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican,
George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol
Vinyals, Jack W. Rae, and Laurent Sifre. Training compute-optimal large language models. In 36th International
Conference on Neural Information Processing Systems, 2022.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi
Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv:2401.02954, 2024.

Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao,
Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, et al. Language models scale reliably with over-training and on
downstream tasks. arXiv:2403.08540, 2024.

Jeffrey Li, Alex Fang, Georgios Smyrnis, et al. DataComp-LM: In search of the next generation of training sets for
language models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors,
Advances in Neural Information Processing Systems, volume 37, pages 14200�14282, 2024.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on

Language Modeling, 2024.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The llama 3 herd of models. arXiv:2407.21783, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan,
Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural
information processing systems, 33:1877�1901, 2020.

Lukas Edman, Helmut Schmid, and Alexander Fraser. CUTE: Measuring LLMs� understanding of their tokens. In Yaser
Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Conference on Empirical Methods in Natural Language
Processing, pages 3017�3026, 2024.

Marta Costa-jussa, James Cross, Onur �elebi, Maha Elbayad, et al. Scaling neural machine translation to 200

languages. Nature, 630, 06 2024.

Lili Yu, Daniel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, and Mike Lewis. MEGABYTE: Predicting
million-byte sequences with multiscale transformers. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt,
and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 78808�78823, 2023.

Mehdi Ali, Michael Fromm, Klaudia Thellmann, et al. Tokenizer choice for LLM training: Negligible or crucial? In
Kevin Duh, Helena Gomez, and Steven Bethard, editors, Findings of the Association for Computational Linguistics,
pages 3907�3924, 2024.

Nived Rajaraman, Jiantao Jiao, and Kannan Ramchandran. An analysis of tokenization: Transformers under markov
data. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in
Neural Information Processing Systems, volume 37, pages 62503�62556, 2024.

Shuhao Gu, Mengdi Zhao, Bowen Zhang, Liangdong Wang, Jijie Li, and Guang Liu. Retok: Replacing tokenizer to

enhance representation efficiency in large language model. arXiv:2410.04335, 2024.

Brian Lester, Jaehoon Lee, Alexander A Alemi, Jeffrey Pennington, Adam Roberts, Jascha Sohl-Dickstein, and Noah

Constant. Training LLMs over neurally compressed text. Transactions on Machine Learning Research, 2024.

Mengyu Zheng, Hanting Chen, Tianyu Guo, Chong Zhu, Binfan Zheng, Chang Xu, and Yunhe Wang. Enhancing
large language models through adaptive tokenizers. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet,
J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 113545�
113568, 2024.

Bj�rn Deiseroth, Manuel Brack, Patrick Schramowski, Kristian Kersting, and Samuel Weinbach. T-FREE: Subword
tokenizer-free generative LLMs via sparse representations for memory-efficient embeddings. In Yaser Al-Onaizan,
Mohit Bansal, and Yun-Nung Chen, editors, Conference on Empirical Methods in Natural Language Processing,
pages 21829�21851, 2024.

Vincent-Pierre Berges, Barlas O?uz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, and Gargi Ghosh. Memory layers

at scale. arXiv preprint arXiv:2412.09764, 2024.

11

Appendix

A Scaling Laws

Every parameter goes through a multiplication and addition per input unit in the forward pass, and twice
that in the backward pass resulting in 6 flops per parameter per input. For the attention mechanism, the
QK T operation dominates the computational cost, requiring 2 FLOPs (multiply and add) per dimension for
each query-key pair. With d dimensions, L layers, and a sequence length of S, this creates S dot products per
layer per input unit. Accounting for both forward and backward passes (3� multiplier), we get 6dLS FLOPs
total. This term becomes particularly significant at smaller scales where attention costs outweight the linear
parameter costs as Bi et al. (2024) already point out.

Notice how the batch size is not simply a difference in constant factor but also in the exponent. In our
experiments, we find that many values of batch size and learning rates are possible and that optimal models
for a given compute budget lie roughly on a line in BSZ/LR space such that both grow linearly with respect to
each other. This of course is only valid for a certain range of values above which the model becomes unstable
and loses performance. Our hypothesis is that simply scaling the batch size such that it equals k � BSZBPE
results in a model that is beyond that limit.

B Regular expression

To be concrete, the regular expression used to define Stage 1 pooling is shown below:

( \p{L}{1,16}) | \p{N}{1,3} | ?([^\s\p{L}\p{N}]){1,3}+[\r\n]* | \s*[\r\n] | \s+(?!\S) | \s+

Each component of the regex serves a distinct role:

� Letters (1�16 characters): captures typical alphabetic words.

� Numbers (1�3 digits): groups numerical tokens.

� Punctuation (1�3 non-alphanumeric chars): handles symbol groups and optional line breaks.

� Line breaks: captures �\r\n� combinations and surrounding whitespace.

� Trailing whitespace (non-followed by a non-space): captures text boundaries.

� General whitespace: handles space separation.

C Ablation

C.1 Pooling and Upsampling

We describe here the different pooling and upsampling strategies explored during our experiments. While all
pooling methods yielded comparable results, they offer different trade-offs in complexity and expressiveness.

Simple Pooling. This is the method used in our main experiments. We directly select the positions indicated
by the splitting function and retain only those tokens.
Cross-Attention Pooling. A cross-attention layer is applied between the original sequence and the pooled
tokens. This allows the downsampled representation to aggregate information flexibly from the full input.
Average Pooling. Tokens within each segment defined by the splitting function are averaged to produce a
single pooled representation.
Memory Layers Berges et al. (2024). Motivated by the concern that pooling might limit output diversity
compared to embedding-table, we experimented with appending a memory layer after pooling. This layer
retrieves learned embeddings based on the pooled inputs, potentially reintroducing back the diversity.

Simple Upsampling. Pooled tokens are inserted back into their original positions in the sequence, and additional
context is recovered via skip connections. Earlier-layer features complement the compressed representations,

12

Table 4 Comparison between the different upsampling tools. Notice that AU-Net 3 stages is much more sensitive to
upsampling.

Model Hswg Arc_-

E

Arc_-
C

PIQA SIQA Race_-

Dim=2048 (1B model), 60B tokens (data-to-model ratio of 10)
45.7
AU-Net 2 Simple
� 2.2

62.9
� 0.9

64.9
� 1.9

73.4
� 2.0

35.5
� 2.7

AU-Net 2
Average Pool

AU-Net 2
Memory Layer

AU-Net 2 Repeat
Up

AU-Net 2
Multi-Linear

AU-Net 3 Simple

AU-Net 3
Multi-Linear

62.5
� 1.0

62.8
� 0.9

64.2
� 0.9

63.5
� 0.9

60.6
� 1.0

66.0
� 0.9

61.5
� 2.0

66.5
� 1.9

64.4
� 1.9

64.4
� 1.9

60.8
� 2.0

64.1
� 1.9

35.4
� 2.7

34.4
� 2.7

35.2
� 2.7

35.3
� 2.7

32.3
� 2.7

35.7
� 2.7

72.9
� 2.0

72.2
� 2.0

74.4
� 2.0

74.0
� 2.0

72.1
� 2.1

75.1
� 2.0

44.7
� 2.2

45.3
� 2.2

46.1 �
2.2

45.3
� 2.2

46.3
� 2.2

45.9
� 2.2

M

54.1
� 2.6

52.2
� 2.6

55.2
� 2.6

53.9
� 2.6

55.1
� 2.5

53.1
� 2.6

55.4
� 2.5

Race_-
H

Winog NQ

TQA

39.3
� 1.6

36.9
� 1.6

38.7
� 1.6

39.0
� 1.6

39.6
� 1.6

38.6
� 1.6

39.3
� 1.6

60.5
� 2.7

60.4
� 2.7

61.3
� 2.7

61.7
� 2.7

62.6
� 2.6

62.0
� 2.6

64.0
� 2.6

7.7 �
0.9

7.2 �
0.8

8.0 �
0.9

8.8 �
0.9

8.3 �
0.9

6.0 �
0.8

7.3 �
0.9

16.6
� 0.7

15.5
� 0.7

16.6
� 0.7

20.4
� 0.7

18.4
� 0.7

13.3
� 0.6

18.7
� 0.7

and attention layers help propagate information across the sequence.
Cross-Attention Upsampling. A cross-attention layer is applied where each upsampled token attends to
the pooled representation. This mechanism allows the model to flexibly decompress higher-level abstract
representations, effectively extracting contextual information to reconstruct the outputs.
Repeat Upsampling. Inspired by nearest-neighbor upscaling in computer vision, each token in the compressed
sequence is repeated a variable number of times, as determined by the splitting function. For this strategy to
remain competitive during training, it is important to include local positional biases within each repeated
segment.
Multi-Linear Upsampling. Each pooled token is transformed using a different linear projection for each position
in the target segment. This allows upsampled tokens to vary based on their relative position while remaining
conditioned on the same source. This method is used in our main experiments due to its favorable balance
between simplicity and expressiveness.

C.2 Layer Allocations

To evaluate the impact of distributing different numbers of layers across stages, we conducted ablations
varying the layer allocation strategy. The first stage (byte level) is fixed to three layers for all models. As
shown in table 5, we allocate a certain percentage of the total layers to the final stage (stage 3), while ensuring
that each intermediate stage retains at least three layers.

We report results for several allocation schemes, and retain the 75% variant�where 75% of the layers are
allocated to the final stage�as the default configuration in the main paper.

D Hyperparameters

As explained in section 2.3, we use a specific batch size and learning rate for each compite budget and
architecture. Aside from this all other hyperparameters remains fixed. A summary table of all hyperparameters
can be found in table 6. We use sequence packing for dataloading during training along with FSDP.

13

Table 5 Comparison between the different percentage of layer in the last stage (the third one).

Model Hswg Arc_-

E

Arc_-
C

PIQA SIQA Race_-

Dim=2048 (1B model), 60B tokens (data-to-model ratio of 10)
46.8
� 2.2

AUNet 3 (25%)

65.3
� 0.9

63.3
� 1.9

36.0
� 2.8

74.2
� 2.0

AUNet 3 (50%)

AUNet 3 (75%)

66.0
� 0.9

67.4
� 0.9

64.1
� 1.9

65.9
� 1.9

35.7
� 2.7

36.7
� 2.7

75.1
� 2.0

75.5
� 2.0

45.9
� 2.2

46.9
� 2.2

M

54.8
� 2.6

55.4
� 2.6

55.4
� 2.5

Race_-
H

Winog NQ

TQA

38.7
� 1.6

39.3
� 1.6

40.5
� 1.6

63.4
� 2.6

64.0
� 2.7

64.2
� 2.6

8.9 �
0.9

7.3 �
0.9

9.6 �
1.0

21.0
� 0.8

18.7
� 0.7

22.6
� 0.8

Model

BPE
AU-Net

LR

BSZ

w.d.

lr min

grad clip

seqlen

total tokens

19.3C?0.177
6.6C?0.176

29.9C0.231
0.66C0.321

0.1
0.1

0.01 � lr_max
0.01 � lr_max

0.2
0.2

2048
8192

(Fmodel / token)2?token
(Fmodel / byte)2(20.7936?token)

Table 6 Summary of all hyperparameters. w.d. stands for weight decay. ?tokens corresponds to the data-to-model ratio
and is reported in bold in each result table, alongside the budget C. Flops per token/byte are detailed in table of
appendix G. Warmup spans 10% of the total training steps, and we employ a cosine learning rate scheduler. The total
number of steps is computed as total_tokens

.

BSZ

E CUTE Benchmark Detailed results

We evaluate both the 7.5B BPE baseline and AU-
Net 2 on the CUTE benchmark Edman et al. (2024),
which tests a model�s ability to manipulate both
words and characters. As shown in Table 7, our
byte-level model performs better on character-level
tasks, while the BPE baseline takes the lead on
word-level ones. This reflects a natural trade-off:
tokenizer-based models operate on word-like units,
making them less sensitive to character structure,
whereas byte-level models handle characters explic-
itly.

This contrast highlights a key design trade-off.
Byte-level models are more flexible with unseen or
morphologically rich inputs, while tokenized mod-
els benefit from stronger word-level priors. Sur-
prisingly, despite lacking explicit character access,
BPE models still perform well on spelling and re-
verse spelling tasks, suggesting that such skills can
emerge from token-level patterns with enough capacity and data.

Average

Table 7 Accuracy of BPE and AU-Net on word-level and
letter-level tasks in CUTE.

CUTE (EM)

Rand

BPE

AU-Net 2

Word Char Word Char

Spell
Inverse spell
Contains
Delete
Insert
Substitute
Swap
Sem/Ortho

0.0
0.0
50.0
0.0
0.0
0.0
0.0
50.0

12.5

-
-

69.9
29.6
15.9
37.5
5.5
66.0

36.9

91.5
80.6

66.7
16.4

9.6
7.6
1.6
40.6

39.3

-
-
61.3
20.6
6.5
21.2
3.3

75.1
33.1

97.3
91.7
59.8

22.3
7.8

12.3
1.9
48.1

42.7

14

k
8
M
S
G

A
Q
T

Q
N

g
o
n
i
W

-

_
e
c
a
R

-

_
e
c
a
R

A
Q
I
S

A
Q
I
P

A
Q
B
O

U
L
M
M

A
Q
S
C

q
l
o
o
B

-

_
c
r
A

-

_
c
r
gA
a
w
s
a
l
l
e
H

l
e
d
o
M

H

M

C

E

�
5
.
2

�
4
.
1
2

�
8
.
8

�
2
.
8
5

�
7
.
7
3

�
1
.
2
5

�
1
.
7
4

�
5
.
4
7

8
.
0

8
.
0

9
.
0

8
.
2

6
.
1

5
.
2

2
.
2

0
.
2

�
1
.
2

�
2
.
1
2

�
2
.
8

�
1
.
9
5

�
2
.
5
3

�
7
.
6
4

�
8
.
5
4

�
1
.
5
7

8
.
0

7
.
0

9
.
0

7
.
2

6
.
1

6
.
2

2
.
2

0
.
2

.

4
8
3

3
.
4
�

.

4
8
3

2
.
4
�

7
.
0

3
.
2

7
.
1

7
.
2

0
.
2

9
.
0

7
.
0

3
.
2

7
.
1

7
.
2

0
.
2

9
.
0

s
e
t
y
b

�
7
.
4
2

�
0
.
0
2

�
7
.
0
6

�
7
.
4
3

�
2
.
1
6

�
0
.
3
6

r
e
m
r
o
f
s
n
a
r
T

�
3
.
5
2

�
6
.
9
1

�
2
.
1
6

�
6
.
3
3

�
3
.
0
6

�
0
.
3
6

s
e
t
y
b

a
b
m
a
M

)
0
1
f
o
o
i
t
a
r

l

e
d
o
m
-
o
t
-
a
t
a
d
(
s
n
e
k
o
t
B
0
6

,
)
l
e
d
o
m
B
1
(
8
4
0
2
=
m
D

i

�
3
.
2

�
3
.
6
2

�
8
.
8

�
6
.
1
6

�
3
.
9
3

�
9
.
3
5

�
2
.
5
4

�
1
.
5
7

�
4
.
7
3

�
5
.
5
2

�
8
.
8
1

�
6
.
2
6

�
5
.
6
3

�
8
.
2
6

�
6
.
3
6

r
e
m
r
o
f
s
n
a
r
T

�
7
.
2

�
4
.
0
2

�
8
.
8

�
7
.
1
6

�
0
.
9
3

�
9
.
3
5

�
1
.
6
4

�
4
.
4
7

�
8
.
6
3

�
5
.
4
2

8
.
0

8
.
0

9
.
0

7
.
2

6
.
1

6
.
2

2
.
2

0
.
2

3
.
4

7
.
0

�
3
.
2

9
.
0

8
.
0

�
5
3

.

0
.
1

.

6
2
2

8
.
0
�

�
6
9

.

0
.
1

.

2
4
6

6
.
2
�

.

5
0
4

6
.
1
�

�
4
.
5
5

�
9
.
6
4

5
.
2

2
.
2

.

5
5
7

0
.
2
�

3
.
4

7
.
0

7
.
0

9
.
0

7
.
2

6
.
1

5
.
2

3
.
2

0
.
2

2
.
4

7
.
0

�
5
.
5
1

�
1
.
5

�
0
.
2
6

�
3
.
9
3

7
.
0

7
.
0

6
.
2

6
.
1

.

6
5
5

6
.
2
�

.

6
7
4

3
.
2
�

�
5
.
4
7

�
2
.
8
3

0
.
2

3
.
4

.

9
5
2

7
.
0
�

2
.
2
�

1
.
0
2

2
.
2

7
.
1

7
.
2

9
.
1

9
.
0

6
.
1

7
.
2

0
.
2

9
.
0

E
P
B

�
0
.
2
6

�
2
.
5
3

�
4
.
4
6

�
2
.
4
6

2

t
e
N
U
A

�
2
.
8
1

2
.
2

1
.
2

.

3
3
6

7
.
1
�

.

0
7
3

8
.
2
�

.

4
7
6

9
.
1
�

7
.
1

8
.
2

9
.
1

.

4
7
6

9
.
0
�

3

t
e
N
U
A

9
.
0

�
4
.
6
6

4

t
e
N
U
A

�
2
.
8
3

�
6
.
5
2

�
7
.
9
1

�
7
.
1
6

�
7
.
6
3

�
9
.
5
6

�
7
.
3

9
.
0

0
.
1

�
1
.
9
3

9
.
0

9
.
0

�
3
5
1

.

2
.
1

.

7
8
6

6
.
2
�

�
3
.
3
4

�
8
.
8
5

�
1
.
7
4

7
.
1

6
.
2

3
.
2

1
.
1

6
.
2

7
.
1

5
.
2

2
.
2

�
3
5

.

2
.
1

�
5
.
5
3

�
0
.
4
1

�
2
.
7
6

9
.
0

1
.
1

6
.
2

.

9
3
4

7
.
1
�

.

0
9
5

6
.
2
�

.

6
7
4

2
.
2
�

�
1
.
8
7

9
.
1

9
.
1

3
.
4
�

6
.
1
4

4
.
4

�
0
.
8
7

�
8
.
0
4

9
.
1

2
.
4

7
.
0

3
.
2

7
.
1

8
.
2

9
.
1

9
.
0

�
5
.
7
2

�
7
.
9
1

�
8
.
1
6

�
1
.
1
3

8
.
0

7
.
0

�
1
.
3
2

4
.
2

3
.
2

�
0
.
2
6

�
2
.
3
4

7
.
1

8
.
2

.

6
2
7

8
.
1
�

9
.
0
�

.

7
3
7

4

t
e
N
U
A

7
.
1

.

3
3
4

9
.
2
�

8
.
1

9
.
0

�
3
.
2
7

�
9
.
2
7

3

t
e
N
U
A

�
0
.
3

�
5
.
2
3

�
0
.
3
1

�
6
.
4
6

�
8
.
2
4

�
7
.
7
5

�
6
.
6
4

�
8
.
6
7

�
6
.
9
3

�
9
.
7
2

�
8
.
0
2

�
3
.
4
6

�
9
.
8
3

�
6
.
8
6

�
9
.
9
6

2

t
e
N
U
A

�
4
.
4

�
2
.
7
3

�
6
.
3
1

�
4
.
5
6

�
8
.
1
4

�
7
.
6
5

�
2
.
6
4

�
9
.
6
7

�
6
.
0
4

�
3
.
6
2

�
8
.
1
2

1
.
1

9
.
0

1
.
1

6
.
2

6
.
1

6
.
2

2
.
2

9
.
1

3
.
4

7
.
0

3
.
2

.

9
2
6

7
.
1
�

8
.
2

9
.
1

9
.
0

E
P
B

�
5
.
8
3

�
6
.
8
6

�
2
.
0
7

r
e
m
r
o
f
s
n
a
r
T

15

)
0
4
f
o
o
i
t
a
r

l

e
d
o
m
-
o
t
-
a
t
a
d
(
s
n
e
k
o
t
B
0
7
3

,
)
l
e
d
o
m
B
1
(
8
4
0
2
=
m
D

i

�
7
0
1

.

7
.
1

�
1
.
1
5

9
.
0

�
0
.
0
1

�
9
.
0
5

6
.
1

9
.
0

�
1
.
2
2

3
.
1

3
.
1

4
.
2
�

.

2
2
7

.

4
4
4

7
.
1
�

5
.
2

6
.
1

9
.
1
6

5
.
2
�

5
.
2

.

0
0
5

3
.
2
�

2
.
2

8
.
1
�

1
.
0
8

8
.
1

.

0
5
4

4
.
4
�

.

0
0
5

8
.
0
�

.

6
3
6

7
.
2
�

.

3
8
6

6
.
1
�

3
.
4

8
.
0

7
.
2

7
.
1

9
.
2
�

2
.
1
5

9
.
2

.

0
0
8

6
.
1
�

8
.
1

�
1
.
9
7

8
.
0

8
.
0

B
7

B
7

2

t
e
N
U
A

�
1
.
1
2

�
5
.
0
7

�
6
.
3
4

�
3
.
0
6

�
4
.
8
4

�
0
.
0
8

�
6
.
3
4

�
4
.
8
4

�
2
.
3
6

�
8
.
3
6

�
5
.
9
4

�
3
.
4
7

�
3
.
7
7

r
e
m
r
o
f
s
n
a
r
T

)
5
f
o
o
i
t
a
r

l

e
d
o
m
-
o
t
-
a
t
a
d
(
s
n
e
k
o
t
B
0
0
2

,
)
l
e
d
o
m
B
8
(
6
9
0
4
=
m
D

i

7
.
2

9
.
0

5
.
1

4
.
2

7
.
1

5
.
2

3
.
2

8
.
1

4
.
4

8
.
0

5
.
2

5
.
1

8
.
2

5
.
1

8
.
0

)
T
5
1
(

�
7
.
4
5

�
4
.
4
6

�
1
.
9
2

�
5
.
4
7

�
4
.
9
4

�
3
.
5
6

�
6
.
9
4

�
8
.
0
8

�
4
.
5
4

�
4
.
5
6

�
6
.
4
7

�
0
.
5
7

�
8
.
4
5

�
3
.
3
8

�
7
.
0
8

1
.
3

a
M
a
L
L

.
s
k
r
a
m
h
c
n
e
b

y
n
a
m
n
o

t
e
N
U
A

-

f
o

e
c
n
a
m
r
o
f
r
e
P

l

8
e
b
a
T

F Evaluation Benchmarks Details

G List of Models

H Model Configuration Tables

This appendix provides detailed configuration parameters for all models used in the experiments, organized
into three categories for clarity.

Table 9 Model architecture parameters including dimensions, layers, and FFN sizes. Semicolons separated values for
different stages in hierarchical models.

Name

Dim

Layers

Head Dim

FFN Dim

Transformer bytes 1B
Mamba bytes 1B
Transformer 1B BPE
AUNet 2 1B
AUNet 3 1B
AUNet 4 1B
Transformer 1B dm8 BPE
AUNet 2 1B dm8
AUNet 3 1B dm8
AUNet 4 1B dm8
Transformer 7B dm1
AUNet 2 7B dm1
Scaling baseline 1e19
Scaling baseline 2e19
Scaling baseline 4e19
Scaling baseline 8e19
Scaling baseline 1e20
Scaling baseline 3e20
Scaling baseline 5e20
Scaling baseline 1e21
Scaling baseline 2e21
Scaling baseline 3e21
Scaling baseline 6e21
Scaling baseline 1e22
Scaling AUNet 2 1e19
Scaling AUNet 2 2e19
Scaling AUNet 2 4e19
Scaling AUNet 2 8e19
Scaling AUNet 2 1e20
Scaling AUNet 2 3e20
Scaling AUNet 2 5e20
Scaling AUNet 2 9e20
Scaling AUNet 2 2e21
Scaling AUNet 2 3e21
Scaling AUNet 2 6e21
Scaling AUNet 2 1e22
Scaling AUNet 3 1e19
Scaling AUNet 3 2e19
Scaling AUNet 3 5e19
Scaling AUNet 3 7e19
Scaling AUNet 3 2e20
Scaling AUNet 3 3e20
Scaling AUNet 3 5e20
Scaling AUNet 3 9e20
Scaling AUNet 3 2e21
Scaling AUNet 3 3e21
Scaling AUNet 3 6e21
Scaling AUNet 3 1e22

2048
2048
2048
512; 2048
512; 2048; 3072
512; 2048; 3072; 4608
2048
512; 2048
512; 2048; 3072
512; 2048; 3072; 4608
4096
1024; 4096
1024
1152
1280
1536
1664
1792
2048
2304
2560
2816
3072
3456
256; 1024
256; 1152
256; 1280
384; 1536
384; 1536
512; 1920
512; 2048
512; 2304
640; 2560
640; 2688
768; 3200
896; 3584
256; 1024; 1536
256; 1152; 1792
256; 1280; 1920
256; 1280; 1920
384; 1536; 2304
384; 1536; 2304
512; 1920; 2816
512; 2048; 3072
512; 2304; 3456
640; 2560; 3840
640; 2688; 4096
768; 3200; 4864

128
64
128
64; 128
64; 128; 128
64; 128; 128; 128
128
64; 128
64; 128; 128
64; 128; 128; 128
128
64; 128
128
128
128
128
128
128
128
128
128
128
128
128
64; 128
64; 128
64; 128
64; 128
64; 128
64; 128
64; 128
64; 128
64; 128
64; 128
64; 128
64; 128
64; 128; 128
64; 128; 128
64; 128; 128
64; 128; 128
64; 128; 128
64; 128; 128
64; 128; 128
64; 128; 128
64; 128; 128
64; 128; 128
64; 128; 128
64; 128; 128

5632
5632
5632
1536; 5632
1536; 5632; 8192
1536; 5632; 8192; 12288
5632
1536; 5632
1536; 5632; 8192
1536; 5632; 8192; 12288
11008
4096; 14336
2816
3072
3584
4096
4608
4864
5632
6144
6912
7680
8192
9216
768; 2816
768; 3072
768; 3584
1024; 4096
1024; 4096
1536; 5120
1536; 5632
1536; 6144
1792; 6912
1792; 7168
2048; 8704
2560; 9728
768; 2816; 4096
768; 3072; 4864
768; 3584; 5120
768; 3584; 5120
1024; 4096; 6144
1024; 4096; 6144
1536; 5120; 7680
1536; 5632; 8192
1536; 6144; 9216
1792; 6912; 10240
1792; 7168; 11008
2048; 8704; 13056

25
50
25
3; 25
3; 3; 18
3; 3; 4; 10
25
3; 25
3; 3; 18
3; 3; 3; 12
32
3; 32
12
13
14
15
17
20
21
24
26
29
34
37
3; 11
3; 13
3; 14
3; 14
3; 19
3; 17
3; 21
3; 24
3; 26
3; 33
3; 32
3; 35
3; 3; 4
3; 3; 5
3; 3; 7
3; 3; 10
3; 3; 10
3; 3; 15
3; 3; 13
3; 3; 16
3; 3; 18
3; 3; 21
3; 3; 26
3; 3; 26

16

Table 10 Training configuration including computational costs, steps, batch sizes, and tokenization

Name

Total FLOPs Tokens/Step FLOPs/Token

Steps G/Acc Batch Size

Seq Len NGpus Tokenizer

Transformer bytes 1B
Mamba bytes 1B
Transformer 1B BPE
AUNet 2 1B
AUNet 3 1B
AUNet 4 1B
Transformer 1B dm8 BPE
AUNet 2 1B dm8
AUNet 3 1B dm8
AUNet 4 1B dm8
Transformer 7B dm1
AUNet 2 7B dm1
Scaling baseline 1e19
Scaling baseline 2e19
Scaling baseline 4e19
Scaling baseline 8e19
Scaling baseline 1e20
Scaling baseline 3e20
Scaling baseline 5e20
Scaling baseline 1e21
Scaling baseline 2e21
Scaling baseline 3e21
Scaling baseline 6e21
Scaling baseline 1e22
Scaling AUNet 2 1e19
Scaling AUNet 2 2e19
Scaling AUNet 2 4e19
Scaling AUNet 2 8e19
Scaling AUNet 2 1e20
Scaling AUNet 2 3e20
Scaling AUNet 2 5e20
Scaling AUNet 2 9e20
Scaling AUNet 2 2e21
Scaling AUNet 2 3e21
Scaling AUNet 2 6e21
Scaling AUNet 2 1e22
Scaling AUNet 3 1e19
Scaling AUNet 3 2e19
Scaling AUNet 3 5e19
Scaling AUNet 3 7e19
Scaling AUNet 3 2e20
Scaling AUNet 3 3e20
Scaling AUNet 3 5e20
Scaling AUNet 3 9e20
Scaling AUNet 3 2e21
Scaling AUNet 3 3e21
Scaling AUNet 3 6e21
Scaling AUNet 3 1e22

nan
nan
6.6e20
5.1e20
6.7e20
8.0e20
3.6e21
3.2e21
4.0e21
5.0e21
9.5e21
1.2e22
2.0e19
3.3e19
5.6e19
1.1e20
2.0e20
3.3e20
6.0e20
1.1e21
2.0e21
3.6e21
6.5e21
1.2e22
1.1e19
2.2e19
3.7e19
7.6e19
1.3e20
2.6e20
5.0e20
9.3e20
1.7e21
3.1e21
5.9e21
1.1e22
1.3e19
2.4e19
4.7e19
7.3e19
1.5e20
2.7e20
5.2e20
9.2e20
1.7e21
3.3e21
6.0e21
1.2e22

nan
nan
1.0e06
1.6e06
2.8e06
2.8e06
1.2e06
1.8e06
5.8e06
5.8e06
2.1e06
4.2e06
7.7e05
8.8e05
9.9e05
1.2e06
1.4e06
1.6e06
1.8e06
2.1e06
2.4e06
2.8e06
3.1e06
3.9e06
8.7e05
1.1e06
1.3e06
1.6e06
1.8e06
2.4e06
2.9e06
3.7e06
4.2e06
5.2e06
6.3e06
7.9e06
9.0e05
1.1e06
1.4e06
1.6e06
2.0e06
2.4e06
2.9e06
3.7e06
4.2e06
5.2e06
6.3e06
8.4e06

60000
60000
60000
180000
105000
105000
310000
950000
300000
300000
100000
277834
14008
16120
19438
24252
28160
32466
39573
46980
55900
64711
77520
84915
56240
63457
68794
79862
89768
99036
108807
119909
141463
153594
176692
193096
56992
64347
71962
79222
90845
100198
113583
119374
141594
158941
177921
187518

1
1
1
1
1
1
1
1
1
1
1
1
15
18
22
15
13
11
12
4
4
3
4
3
2
3
4
6
4
4
4
1
2
2
1
1
2
3
4
5
4
4
2
1
1
1
1
1

4
4
4
12
14
14
9
7
11
11
2
1
25
24
22
19
17
14
12
8
9
7
6
5
53
43
39
32
28
24
18
14
8
10
8
6
55
45
42
38
30
24
22
14
16
10
8
4

4096
4096
4096
8192
8192
8192
2048
8192
8192
8192
4096
8192
2048
2048
2048
2048
2048
2048
2048
2048
2048
2048
2048
2048
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192
8192

NaN
NaN
64
16
24
24
64
32
64
64
256
128
1
1
1
2
3
5
6
32
32
64
64
128
1
1
1
1
2
3
5
32
32
32
96
160
1
1
1
1
2
3
8
32
32
64
96
256

bytes
bytes
tiktoken
bytes
bytes
bytes
tiktoken
bytes
bytes
bytes
tiktoken
bytes
tiktoken
tiktoken
tiktoken
tiktoken
tiktoken
tiktoken
tiktoken
tiktoken
tiktoken
tiktoken
tiktoken
tiktoken
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes
bytes

nan
nan
1.1e10
1.8e09
2.3e09
2.8e09
9.9e09
1.8e09
2.3e09
2.9e09
4.5e10
1.0e10
1.9e09
2.3e09
2.9e09
4.0e09
5.1e09
6.5e09
8.6e09
1.2e10
1.5e10
2.0e10
2.7e10
3.6e10
2.4e08
3.2e08
4.2e08
6.0e08
7.9e08
1.1e09
1.5e09
2.1e09
2.9e09
3.9e09
5.3e09
7.3e09
2.5e08
3.4e08
4.8e08
5.9e08
8.6e08
1.1e09
1.6e09
2.1e09
2.9e09
4.0e09
5.4e09
7.6e09

17

Table 11 Optimization hyperparameters including learning rates, weight decay, and scheduler settings.

Name

Transformer bytes 1B
Mamba bytes 1B
Transformer 1B BPE
AUNet 2 1B
AUNet 3 1B
AUNet 4 1B
Transformer 1B dm8 BPE
AUNet 2 1B dm8
AUNet 3 1B dm8
AUNet 4 1B dm8
Transformer 7B dm1
AUNet 2 7B dm1
Scaling baseline 1e19
Scaling baseline 2e19
Scaling baseline 4e19
Scaling baseline 8e19
Scaling baseline 1e20
Scaling baseline 3e20
Scaling baseline 5e20
Scaling baseline 1e21
Scaling baseline 2e21
Scaling baseline 3e21
Scaling baseline 6e21
Scaling baseline 1e22
Scaling AUNet 2 1e19
Scaling AUNet 2 2e19
Scaling AUNet 2 4e19
Scaling AUNet 2 8e19
Scaling AUNet 2 1e20
Scaling AUNet 2 3e20
Scaling AUNet 2 5e20
Scaling AUNet 2 9e20
Scaling AUNet 2 2e21
Scaling AUNet 2 3e21
Scaling AUNet 2 6e21
Scaling AUNet 2 1e22
Scaling AUNet 3 1e19
Scaling AUNet 3 2e19
Scaling AUNet 3 5e19
Scaling AUNet 3 7e19
Scaling AUNet 3 2e20
Scaling AUNet 3 3e20
Scaling AUNet 3 5e20
Scaling AUNet 3 9e20
Scaling AUNet 3 2e21
Scaling AUNet 3 3e21
Scaling AUNet 3 6e21
Scaling AUNet 3 1e22

LR

0.003
0.003
0.003
0.00165
0.0015
0.0015
0.001
0.00094
0.0011
0.0011
0.001
0.000818
0.008152
0.007378
0.006633
0.005788
0.005204
0.004693
0.0042
0.003722
0.003357
0.003018
0.002701
0.002416
0.002923
0.002615
0.002377
0.002096
0.001906
0.001685
0.001507
0.001348
0.001214
0.00109
0.0009731
0.0008719
0.002872
0.002561
0.002279
0.00211
0.001852
0.001678
0.001496
0.001351
0.001213
0.001077
0.0009707
0.0008612

?1
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9
0.9

?2
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95
0.95

WD

0.033
0.033
0.033
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.05
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1
0.1

18

Scheduler Warmup

cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine
cosine

5000
5000
5000
10000
10000
20000
2000
10000
10000
10000
10000
5000
2000
2000
2000
2000
2000
2000
2000
2000
2000
2000
2000
2000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000
10000


