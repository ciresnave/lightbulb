3
2
0
2

g
u
A
1
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
7
0
2
6
0
.
8
0
3
2
:
v
i
X
r
a

THINKING LIKE AN EXPERT:MULTIMODAL
HYPERGRAPH-OF-THOUGHT (HOT) REASONING TO BOOST
FOUNDATION MODALS

Fanglong Yao1,3,�, Changyuan Tian1,2,3,�, Jintao Liu1,2,3,�, Zequn Zhang1,3,?,
Qing Liu1,3, Li Jin1,3, Shuchao Li1,3, Xiaoyu Li1,3,Xian Sun1,2,3?
1Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing 100190, China
2School of Electronic, Electrical and Communication Engineering, University of Chinese Academy of Sciences, China
3Key Laboratory of Network Information System Technology (NIST), Aerospace Information Research Institute,
Chinese Academy of Sciences, Beijing, China
{yaofanglong17,tianchangyuan21,liujintao201}@mails.ucas.ac.cn,
zqzhang1@mail.ie.ac.cn,{liuqing1,jinli,lisc,lixy01,sunxian}@aircas.ac.cn

August 14, 2023

ABSTRACT

Reasoning ability is one of the most crucial capabilities of a foundation model, signifying its capacity
to address complex reasoning tasks. Chain-of-Thought (CoT) technique is widely regarded as one
of the effective methods for enhancing the reasoning ability of foundation models and has garnered
significant attention. However, the reasoning process of CoT is linear, step-by-step, similar to personal
logical reasoning, suitable for solving general and slightly complicated problems. On the contrary, the
thinking pattern of an expert owns two prominent characteristics that cannot be handled appropriately
in CoT, i.e., high-order multi-hop reasoning and multimodal comparative judgement. Therefore, the
core motivation of this paper is transcending CoT to construct a reasoning paradigm that can think like
an expert. The hyperedge of a hypergraph could connect various vertices, making it naturally suitable
for modelling high-order relationships. Inspired by this, this paper innovatively proposes a multimodal
Hypergraph-of-Thought (HoT) reasoning paradigm, which enables the foundation models to possess
the expert-level ability of high-order multi-hop reasoning and multimodal comparative judgement.
Specifically, a textual hypergraph-of-thought is constructed utilizing triple as the primary thought to
model higher-order relationships, and a hyperedge-of-thought is generated through multi-hop walking
paths to achieve multi-hop inference. Furthermore, we devise a visual hypergraph-of-thought to
interact with the textual hypergraph-of-thought via Cross-modal Co-Attention Graph Learning for
multimodal comparative verification. Experimentations on the ScienceQA benchmark demonstrate
the proposed HoT-based T5 outperforms CoT-based GPT3.5 and chatGPT, which is on par with
CoT-based GPT4 with a lower model size.

1

Introduction

The immense success of ChatGPT [1] has triggered an explosive growth in foundation models, resulting in the
emergence of hundreds of alternative models competing in the field [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]. Reasoning ability
is one of the most crucial capabilities of a foundation model, signifying its capacity to address complex reasoning tasks,
such as science question answering [13]. Currently, one of the most effective techniques for enhancing the reasoning
ability of foundation models is the Chain-of-Thought (CoT) [14]. The key idea behind this approach is to guide
the foundation model to generate additional intermediate reasoning steps, rather than just providing the final answer.
Building upon this key idea, researchers have successively introduced methods like CoT-SC [15], Skeleton-of-Thought
(SoT) [16],Tree-of-Thought (ToT) [17], Graph-of-Thought (GoT) [18].

?Corresponding author. � Equal contribution.

A PREPRINT - AUGUST 14, 2023

Figure 1: The paradigms of Hypergraph-of-Thought (HoT), Chain-of-Thought (CoT), Tree-of-Thought (ToT), and
Graph-of-Thought (GoT). Using HoT as the reference, removing the hyperedge-of-thought and retaining only one
reasoning path becomes the CoT; Preserving multiple non-intersecting parallel reasoning paths constitutes the ToT;
Simplifying the hyperedge-of-thought to a regular edge that can only connect two thoughts forms the GoT.

Although the CoT endows foundation models with the ability to think like a person, the following limitations remain.
Firstly, the CoT is linear, making it challenging to achieve leaping logical reasoning. Secondly, CoT decomposes a
problem into step-by-step sequential procedures, with inadequate consideration for multi-concurrent steps and conflicts
between them. The limitations restrict foundation models to solving non-professional problems.

Compared to an ordinary person, the logical thinking mode of experts has two prominent characteristics, i.e., high-
order multi-hop reasoning and multimodal comparative judgement. Therefore, the core motivation of this paper is to
concentrate on simulating the logical thinking of experts and constructing a reasoning paradigm that goes beyond the
CoT to enhance the ability of foundation models to solve complex professional problems. Hypergraph is a powerful
modelling approach for complex networks that has been widely studied recently. Compared with regular graphs that can
only connect two vertices with con an edge, the hyperedge of hypergraph can connect an arbitrary number of vertices,
so it is naturally suitable for modelling high-order relationships [19]. Compared to previously modelling the logical
thinking process of ordinary people as a chain structure of sequences, we further simulate the logical thinking of experts
directly and formulate it as a hypergraph structure, i.e., Hypergraph-of-Thought (HoT). Meanwhile, as illustrated in
Figure 1, we believe HoT can be appropriately adjusted and degenerated into CoT, ToT, and GoT2.

This paper proposes an innovative multimodal hypergraph-of-thought reasoning paradigm, which empowers the
foundation model with expert-like high-order reasoning capability based on the representation ability of textual and
visual foundation models:

� Constructing a textual hypergraph-of-thought by utilizing triplets as basic thoughts to model higher-order
relationships. Exploiting multi-hop random walks to form long-range reasoning paths to constitute hyperedges-
of-thoughts and form multi-hop reasoning abilities.

� Simultaneously, k-means clusters image patches to construct hyperedges to form a visual hypergraph-of-

thought.

� Allset Transformer is selected to encode the thoughts of the textual and visual HoTs, respectively, to achieve

bidirectional updates of thought-to-hyperedge and hyperedge-to-thought.

In addition, to simulate the expert�s multimodal contrastive judgement capability, we pick out Cross-modal Co-Attention
Graph Learning to fully interact between different HoTs to avoid information conflicts from various modalities. We

2It is worth noting that CoT-SC and SoT can be catgorized within the ToT reasoning paradigm.

2

InputOutputInputOutputInputOutputInputOutput(a) Hypergraph-of-Thought (HoT)(b) Chain-of-Thought (CoT)(c) Tree-of-Thought (ToT)(d) Graph-of-Thought (GoT)ThoughtReasoning PathMulti-hop Reasoning PathHyperedge-of-ThoughtReasoning Path ThroughHyperedge-of-ThoughtsA PREPRINT - AUGUST 14, 2023

choose ScienceQA as the experimental dataset and follow the two-stage framework of Multimodal CoT [20], namely
the rational and answer generation stages. The experimental results are superior to CoT-based GPT-3.5 and ChatGPT,
demonstrating the potential of HoT to enhance the ability of foundation models.

The contributions can be summarized as follows:

(1) Inspired by the logical reasoning process of an expert to solve complex problems, the multimodal hypergraph-of-
thought reasoning paradigm for text and image is innovatively constructed, boosting the foundation models possessing
the abilities of high-order multi-hop reasoning and multimodal comparative judgement.

(2)We take triplet as the basic thought to construct a textual hypergraph-of-thought to model higher-order relationships,
and devise hyperedge-of-thought through multi-hop walking paths to achieve multi-hop inference. Further, utilizing
AllSet transformer to achieve alternating updates of thought-to-hyperedge and hyperedge-to-thought.

(3)This paper also constructs an visual hypergraph-of-thought and utilizes Cross-modal Co-Attention Graph Learning
to interact with the textual hypergraph-of-thought, achieving multimodal interactive verification.

2 Related work

2.1 CoT

The core idea of the COT [14] is to guide LLMs in the process of reasoning by generating a series of intermediate
inference steps, rather than directly providing the final answer. This reasoning mechanism effectively enhances the
complex reasoning ability of LLMs. Researchers have further developed CoT from various perspectives, building upon
this fundamental idea. In terms of paradigms, these efforts can be categorized as prompt-based CoT and fine-tuning-
based CoT. As for the targeted modalities, these endeavours can be classified as language-modal CoT and multi-modal
CoT.

In the current research landscape, prompt-based CoT represents the mainstream approach. Based on whether a few
demonstrations are provided to the LLMs, prompt-based CoT techniques can be categorized into few-shot CoT [14]
and zero-shot CoT[21]. Typically, standard CoT prompting falls under the category of few-shot CoT prompting, where
specific demonstrations must be provided as examples to guide LLMs� reasoning process [14]. On the other hand,
zero-shot CoT requires no additional demonstrations; instead, adding the prompt "Let�s think step by step" between
each answer can enhance the complex reasoning ability of LLMs.

Although prompt-based CoT techniques require no parameter adjustments, one major drawback is their reliance on
extensive models with billions of parameters, making them challenging to deploy on a large scale in practical applications.
Researchers have been investigating methods to imbue smaller models with similar reasoning capabilities to address this
issue and reduce the model size requirements for CoT. Fine-tuning-based CoT is considered a promising solution in this
regard. Fine-tuning-based CoT involves fine-tuning a smaller model (with fewer than 1 billion parameters) to generate
informative rationales, which are then used for reasoning and producing the final answers, rather than providing direct
responses. Namgyu Ho et al. [22] proposed fine-tune-CoT, which leverages prompt-based CoT with a large teacher
model to generate reasoning samples. These samples are then used for fine-tuning a smaller model, endowing it with
significant reasoning abilities. Similarly, Lucie Charlotte Magister et al. [23] explored knowledge extraction from large
language models like PaLM 540B and GPT-3 175B to smaller models with different sizes, such as T5 XXL, XL, and
base, which have parameters of 11 billion, 3 billion, and 220 million, respectively. With similar motivation, Liunian
Harold Li et al. [24] introduced Symbolic CoT Distillation, which samples rationales from a large teacher model to
train a smaller student model. These efforts demonstrate promising advances in enabling smaller models to exhibit
effective reasoning capabilities in CoT, allowing for more practical and scalable deployments in real-world scenarios.

The capabilities of CoT should not be limited to language models alone. As a result, researchers have been striving
to expand CoT beyond the language modality into the realm of multi-modal tasks, giving rise to Multi-modal CoT.
Building upon the core idea of CoT, Multi-modal CoT aims to facilitate multi-step reasoning for multi-modal inference
tasks, rather than providing direct answers. Zhang et al. [25] introduced the MM-CoT framework, representing the first
attempt to extend the CoT reasoning mechanism to multi-modal scientific question-answering tasks. This extension
significantly improves reasoning performance. Specifically, MM-CoT is a two-stage framework. In the first stage, it
generates rationales, and in the second stage, based on the rationales generated in the previous stage, it carries out
the final answer inference. Expanding on the MM-CoT framework, Yao et al. [26] proposed an enhancement by
incorporating additional graphs to model human thinking processes. Their approach, called GoT, leverages the graph
modality to improve reasoning performance.

3

A PREPRINT - AUGUST 14, 2023

Figure 2: Similar to [20] [18], we employ a two-stage framework for implementing CoT reasoning, comprising the
Rationale Generation and Answer Generation stages. The input in the Rationale Generation stage consists of textual
(question) and image modalities. Initially, we employ separate text and image encoders to encode textual and image
information, respectively. The parameters of textual and visual foundation models are frozen during the training process.
Additionally, we construct textual and visual HoTs for the respective modalities , and utilize the AllSetTransformer
hypergraph encoder to encode these hypergraphs. Subsequently, we establish a cross-modal cross-attention graph,
obtaining a cross-modal representation through the graph encoder. Following the gate-based fusion layer, the fused
representation of textual, visual, and cross-modal information is fed into the decoder to generate intermediate rationales.
In the Answer Generation stage, the distinction from the previous stage lies in both input and output. This stage�s input
encompasses the previous stage�s output, i.e., the rationale, and includes additional contextual information. The output
of this stage culminates in the final question answer.

2.2 Vision-Language LLMs

Recently, there has been significant interest in fine-tuning LLMs for vision-language instruction following. Researchers
have primarily explored methods for fine-tuning LLMs using multi-modal instruction-following data. Liu et al. [27]
propose groundbreaking research on generating multi-modal language-image instruction data using GPT-4, which only
contains the language modality. Through tuning on the generated data, they introduce LLaVA, a large multi-modal
model that connects visual encoders and LLMs for general visual and language understanding. Similarly, Luo et al. [28]
present an efficient and economical approach for fine-tuning LLMs in the context of vision-language instructions. They
introduce the MMA (Mixture-of-Modality Adaptation) framework, utilizing lightweight adapters to connect LLMs with
vision-language tasks. Wang et al. [29] address the challenge of gathering high-quality CoT rationales for answering
scientific questions in multi-modal scenarios. They propose the T-SciQ method, which generates high-quality CoT
rationales as teaching signals and trains smaller models for CoT reasoning. Horawalavithana et al. [30] introduce a
tuning framework called SciTune to align LLMs with scientific multi-modal instructions. They emphasize the limited
research on improving LLMs to align with scientific subjects, concepts, and objectives. By fine-tuning LLMs using
human-generated scientific instruction data, they train the LLaMA-SciTune, a large multi-modal model connecting
visual encoders and LLMs.

3 Method

3.1 Hypergraphof-Thought Construction

Hypergraphs offer a natural way to model higher-order relationships, where each hyperedge can connect any number of
objects. Previous research has demonstrated the application of hypergraphs in modelling higher-order semantics and
improving capabilities in multi-hop reasoning and effective multi-modal learning[31],[32]. Based on this observation,
utilizing hypergraphs to model essential higher-order relationships and aligning semantic information from different
modalities can enhance the model�s reasoning abilities. Therefore, our first step involves constructing hypergraphs
for the text and image modalities separately, aiming to capture the higher-order semantic relationships within each

4

Question:Which of these states is farthest north?Choices:(A) West Virginia, (B) Louisiana, (C) Arizona, (D) OklahomaRationale:<Completed by the first stage>Hyperedge EncoderHyperedge EncoderTextual Foundation ModelVisual Foundation Model Cross-Modal Co-Attention Graph GeneratorGraph Encoder Gated Fusion LayerT5 DecoderAnswer: (A) West Virgini Rationale Generation StageRationaleLecture:Maps have four cardinal directions, or main directions. ...Solution:To find the answer, look at the compass rose. ...        Answer Generation StageText EncodingTextual HoT ConstructionVisual HoT ConstructionImage EncodingA PREPRINT - AUGUST 14, 2023

Algorithm 1 Textual Hypergraph-of-Thought Construction
Output: Gtext(Vtext, Etext)
Input: T , G(V, E), k, N

Etext is empty
for n=1 to N do

vstart ?? RandomChoice(V )
etext ?? RandomWalk(vstart, G, k)

Append etext to Etext

end for
Return Gtext(Vtext, Etext)

Algorithm 2 Visual Hypergraph-of-Thought Construction
Output: Gimg(Vimg, Eimg)
Input: I, m

P ?? Swin(I)
Gimg ??k-means(P, m)
Return Gimg

modality. Additionally, inspired by the work of [32], we further construct a multi-modal hypergraph to align the
semantic information between the image and text modalities.

3.1.1 Textual Hypergraph-of-Thought Construction

Previous research has empirically demonstrated that using thought graphs to model human�s ability for leaps of
reasoning can improve the reasoning capabilities of vision-language LLMs. Taking this a step further, and inspired
by [31], we believe that using hypergraphs to model the human ability for leaps of reasoning is more appropriate.
Hypergraphs can capture higher-order reasoning paths and enhance multi-hop reasoning capabilities.

As shown in figure 3, following [26], we first construct a graph-of-thoughts, denoted as G = (V, E), where V represents
the set of thoughts and E represents the connections between thoughts. In the text hypergraph Gtext = (Vtext, Etext),
the text hypergraph shares the same node set as the graph of thoughts, i.e., V = Vtext. Hyperedge etext ? Etext
is defined as a multi-hop reasoning path, such as (Lionel Messi, place of birth, Rosario, is located in, Republic of
Argentina, is located in, South America). Multi-hop reasoning paths are obtained through random walks on the graph of
thoughts. It�s worth noting that we define a triplet as the basic unit of random walks. We have set two types of random
walks: one-hop random walks and k-hop random walks. One-hop random walks correspond to triple hyperedges, such
as (Lionel Messi, place of birth, Rosario). On the other hand, k-hop random walks refer to k consecutive walks, meaning
that the starting point of the current walk is the endpoint of the last walk. An example of a 2-hop walk is (Lionel Messi,
place of birth, Rosario, is located in, Republic of Argentina).

Figure 3: Example of Textual Hypergraph-of-Thought Construction

5

MapCardinal directionsnorthincludeCompass rosehaveNorth arrowMaphaveCardinal directionsnorthsoutheastwestCompass rosehaveNorth arrowMaps have four cardinal directions, or main directions. Those directions are north, south, east, and west. A compass rose is a set of arrows that point to the cardinal directions. A compass rose usually shows only the first letter of each cardinal direction. The north arrow points to the North Pole. On most maps, north is at the top of the map.A PREPRINT - AUGUST 14, 2023

3.1.2 Visual Hypergraph-of-Thought Construction

We believe that hypergraphs can capture high-order interactions among various local objects in images, enhancing the
modelling capability of global image information[33]. Therefore, we additionally construct an image hypergraph.

Firstly, we utilize Swin[34] to perform representation learning on the initial input images, obtaining representations for
a set of patches denoted as P ? Rp�d, where p represents the number of patches obtained by dividing the input image.
Let I denote the image, then we have:

Inspired by the hypergraph construction method introduced in [35], we use the K-means clustering algorithm to generate
the image hypergraph Gimg = (Vimg, Eimg). Specifically, each cluster is considered a hyperedge, so |Eimg| = m,
where m is a hyperparameter representing the number of clusters. We denote this process as:

P = Swin(I)

Gimg = k-means(m, P)

3.2 Textual and Visual Hypergraph-of-Thought Encoding

3.2.1 Hypergraph Encoder

Firstly, we define the hypergraph encoder. Many hypergraph neural networks have been proposed to learn hypergraph
representations effectively. In our work, we use the AllSet Transformer as the hypergraph encoder. The AllSet
Transformer consists of two multisets functions, namely the node-to-hyperedge function: fV?E , and the hyperedge-
to-node function: fE?V . Transformers parameterize both functions. Given a matrix S ? R|S|�d, representing a
d-dimensional vector for each element in the multiset S, the AllSet Transformer is defined as follows:

fV?E (S) = fE?V (S) = LN(Y + MLP(Y))

Y = LN (? + MHh,?(?, S, S)) , MHh,?(?, S, S) = (cid:13)
h
i=1O(i)
(cid:13)

O(i) = ?

(cid:18)

?(i) (cid:16)

K(i)(cid:17)T (cid:19)

V(i)

? ? (cid:13)
h
i=1?(i)
(cid:13)
K(i) = MLPK,i(S)
V(i) = MLPV,i(S)

where LN denotes layer normalization, h represents the number of heads in the multi-head attention mechanism,
? ? R1�hdh are learnable weights. || indicates the concatenation operation, and the output dimensions of M LP K,i and
M LP V,i are R|V|�dh.

3.2.2 Textual Hypergraph-of-Thought Encoding

Since the initial nodes on the text hypergraph are expressed as thoughts represented by words, we first need to obtain
embeddings for these thoughts. Following the setup of [26], we format Vtext as a sequence, i.e.,

p = [< s >, n0, < /s >, ..., < s >, nj, < /s >]

where < s > and < /s > are special tokens used to emphasize each graph of thought node, and ni represents a node
in the graph of thought. p represents the formatted token sequence. Then, we input the token sequence into the text
encoder, and the representations of each node corresponding to < s > output by the encoder are considered as the
initial representations of the nodes, i.e.,

(cid:104)

0, xn
xs

0 , xe

0, . . . , xs

|V|, xn

|V|, xe
|V|

(cid:105)

= Encodertext(p)

6

A PREPRINT - AUGUST 14, 2023

where xs
represents the dimension of the embeddings.

i is considered as the initial representation of node ni. Thus, we have X = [xs

0; ...; xs

|V|] ? R|V|�d, where d

Next, with the text hypergraph Gtext and the initial node embeddings X as inputs, we utilize the AllSet Transformer
encoder to encode the hypergraph nodes.

3.2.3 Visual Hypergraph-of-Thought Encoding

Unlike the text hypergraph, the nodes on the image hypergraph do not require an additional embedding layer, as their
initial representations P ? Rp�d are obtained from the Swin encoder. Similar to the encoding process for the text
hypergraph, we use the AllSet Transformer to encode the image hypergraph.

3.3 Cross-HoT Interaction

The textual and visual HoTs model high-order semantic information from the text and image modalities. To enhance
cross-modal interaction, we introduce the Cross-Modal Co-Attention Graph Learning module. The core idea of the
co-attention graph is to enable interaction between the hyperedges Etext of the text hypergraph and Eimg of the image
hypergraph. First, we use the multisets function fV?E introduced in 3.2.2 to learn the representations of the hyperedges,
i.e.,

ei
text = fVtext?Etext(Si

text)

img = fVimg?Eimg (Si
ei

img)

text ? Etext represents the representation of the i-th hyperedge in the text hypergraph, and Si

where ei
text|�d is
the set of vectors representing the i-th hyperedge in the text hypergraph, with each row representing the representation
of a node. Similarly, ei
img ? Eimg represents the representation of the i-th hyperedge in the image hypergraph,
img ? R|Si
img|�d is the set of vectors representing the i-th hyperedge in the image hypergraph, with each row
and Si
representing the representation of a node (patch). Thus, we have the text hyperedge representation Etext ? R|Etext|�d
and the image hyperedge representation Eimg ? R|Eimg|�d.
Next, based on the text and image hyperedges representations, we use vector inner product to construct the Cross-Modal
Co-Attention Graph. This process is represented as follows:

text ? R|Si

A = softmax

(cid:16)

W ? (EtextW c

text) (cid:0)EimgW c

img

(cid:1)?(cid:17)

where W c

text ? Rd�dc, W c

img ? Rd�dc , and W ? R|Etext|�|Eimg| are learnable weights.

zm = (EtextW m

text)? A (cid:0)EimgW m

img

(cid:1)

where zm represents the final fused representation, and Wm

text ? Rd�dm and Wm

img ? Rd�dm are learnable weights.

4 Experiments

4.1 Dataset

Our method has been assessed using the ScienceQA benchmark [13] is a pioneering multimodal science question
dataset. Notably, ScienceQA provides detailed lectures and explanations annotated alongside the answers. The dataset
comprises 21000 multiple-choice questions, covering various domains across three subjects, 26 topics, 127 categories,
and 379 skills. To facilitate evaluation, the benchmark dataset is divided into training, validation, and test splits,
consisting of 12726, 4241, and 4241 examples, respectively.

4.2

Implementation Details

Our experiments are conducted on 6 NVIDIA Tesla V100 32G GPUs. In our experimental setup, we employed the
T5 architecture [36] as our baseline model, utilizing both T5-base and T5-large model sizes. We initialized our model

7

A PREPRINT - AUGUST 14, 2023

Methods

Human

MCAN[39]
Top-Down[40]
BAN[41]
DFAF[42]
ViLT[45]
Patch-TRM[43]
VisualBERT[44]

UnifiedQABase [36]
GPT-3.5[46]

UnifiedQABase (CoT)[13]
GPT-3.5 (CoT)[13]
ChatGPT (CoT)[1]
GPT-4 (CoT)[1]

HoT-T5Base
HoT-T5Large

Size

NAT

SOC

LAN

TXT

IMG

NO

G1-6

G7-12

Avg

-

90.23

84.97

87.48

89.60

87.50

88.10

91.59

82.42

88.40

56.08
95M
70M
59.50
112M 60.88
74M
64.03
113M 60.48
90M
65.19
111M 59.33

223M 68.16
74.64
175B

223M 71.00
75.44
175B
78.82
-
85.48
-

223M 82.46
738M 84.46

46.23
54.33
46.57
48.82
63.89
46.79
69.18

69.18
69.74

76.04
70.87
70.98
72.44

78.07
79.08

58.09
61.82
66.64
63.55
60.27
65.55
61.18

74.91
76.00

78.91
78.09
83.18
90.27

82.00
84.64

59.43
62.90
62.61
65.88
63.20
66.96
62.71

63.78
74.44

66.42
74.68
77.37
82.65

81.18
82.89

51.17
54.88
52.60
54.49
61.38
55.28
62.17

61.38
67.28

66.53
67.43
67.92
71.49

75.20
75.81

55.40
59.79
65.51
64.11
57.00
64.95
58.54

77.84
77.42

81.81
79.93
86.13
92.89

85.09
88.15

51.65
57.27
56.83
57.12
60.72
58.04
62.96

72.98
76.80

77.06
78.23
80.72
86.66

81.86
83.88

59.72
62.16
63.94
67.17
61.90
67.50
59.92

65.00
68.89

68.82
69.68
74.03
79.04

80.62
82.47

54.54
59.02
59.37
60.72
61.14
61.42
61.87

70.12
73.97

74.11
75.17
78.31
83.99

81.42
83.38

Table 1: Overall performance compared to the baselines on the ScienceQA dataset.

using the pre-trained T5 checkpoint, specifically the UnifiedQA [37] variant to ensure a fair comparison. We adopt
DETR [38] for the visual foundation modal to obtain visual features. Our models underwent fine-tuning for 20 epochs,
employing a learning rate 5e-5. The maximum input sequence length is set to 512. We set the batch sizes for the base
and large models to 8 and 4, respectively.

4.3 Baselines

Following previous methods [13, 20], we select the following baselines for comparison:

� Visual question answering (VQA) models [39, 40, 41, 42, 43, 44, 45]. These methods treat the question,
context, and choices as the textual input and the image as the vision input. Then they predict the score
distribution over choice candidates via a linear classifier.

� Text-to-text LM models [36, 46]. UnifiedQA [36] and GPT-3.5 [46] regard this task as a text generation
problem and the model is trained to generate the target text. Then they obtain the prediction results from the
generated text.

� Text-to-text LLMs with CoT prompting [13]. It is worth noting that UnifiedQA and GPT-3.5 [13] adopt the

generated image captions as vision semantics.

4.4 Main Results

The main results are shown in Table 1. According to the evaluation results, we can observe that:

(1) HoT-T5 has demonstrated remarkable performance, outperforming previous methods by an impressive margin.
Moreover, it has even surpassed human performance levels. The performance improvement over baselines demonstrates
that the proposed Textual HoT, Visual HoT, and Cross-HoT Interaction can improve the expert-level ability of high-order
multi-hop reasoning and multimodal comparative judgement.

(2) Specifically, among the 8 question classes, HoT-T5Large significantly improved for questions accompanied by paired
images. This highlights the effectiveness of incorporating image features to enhance question-answering capabilities
compared to methods that rely solely on image captions for vision semantics, such as UnifiedQA and GPT-3.5.

(3) The two-stage framework used in the HoT-T5Large approach has also proven to contribute to its superior results,
which validates the potential and effectiveness of leveraging multimodal inputs and image features for this task.

8

A PREPRINT - AUGUST 14, 2023

Methods

NAT

SOC

LAN

TXT

IMG

NO

G1-6

G7-12

Avg

w/o Textual HoT
w/o Visual HoT
w/o Cross-HoT Interaction
HoT-T5Base

79.97
80.95
81.22
82.46

76.04
77.39
77.84
78.07

80.64
81.82
82.82
82.00

79.13
80.01
79.96
81.18

72.58
74.22
74.42
75.20

83.14
84.18
85.37
85.09

80.03
80.98
81.28
81.86

78.05
79.43
80.29
80.62

79.32
80.43
80.92
81.42

Table 2: Experimental results on ablation studies.

Figure 4: Case study on the ScienceQA dataset.

4.5 Ablation Study

In this section, we conduct ablation studies to evaluate the contributions of each component. The experimental results
are reported in Table 2. We can see that:

(1) After removing Textual HoT, the model performance drops significantly, demonstrating that the Textual HoT can
provide rich semantic information about the text, thus improving the high-order multi-hop reasoning ability.

(2) After removing the Visual HoT, the model performance becomes worse. This is because the Visual HoT can help
the model gain a deeper understanding of visual modalities and boost model performance.

(3) The model suffers from a performance decay after removing the Cross-HoT Interaction. The reason may be that the
Cross-HoT Interaction can combine semantic information of text and images for multimodal comparative judgement.

4.6 Case Study

To provide a more illustrative comparison between HoT and CoT, we have selected two representative examples
from the ScienceQA dataset for analysis. In Figure 4, we find that CoT produces wrong rations and answers, while
HoT achieves the right results. This indicates that HoT can address high-order multi-hop reasoning and multimodal
comparative judgement better. In Figure 5, we can observe that CoT produces right rationals but wrong answers. HoT
can leverage the generated rationales more effectively, thus achieving better performance than CoT.

5 Conclusion

This paper breaks through the previous chain-of-thought (CoT) pattern of being able to think like an ordinary person.
Inspired by expert logical thinking and combined with the advantages of hypergraphs, we construct a multimodal
hypergraph-of-thought (HoT) reasoning paradigm, enabling the foundation models to think like an expert and possess

9

GoldQuestion: Which continent is highlighted?Choices: (A) Asia                 (B) Europe                 (C) Australia        (D)North AmericaRational: A continent is one of the major land masses on the earth. Most people say there are seven continents. This continent is Europe.Answer: The answer is (B).HoT PredictionRational: A continent is one of the major land masses on the earth. Most people say there are seven continents. This continent is Europe.Answer: The answer is (B).CoT PredictionRational: A continent is one of the major land masses on the earth. Most people say there are seven continents. This continent is Asia.Answer: The answer is (A).Right rationales right answerWrong rationales wrong answerA PREPRINT - AUGUST 14, 2023

Figure 5: Case study on the ScienceQA dataset.

the abilities of high-order multi-hop reasoning and multimodal comparative judgement. Specifically, based on the
representation ability of the language foundation model, a textual HoT is constructed using triplets as the basic thought,
and multi-hop logical reasoning is achieved by combining multi-hop walking paths. Furthermore, with the image
features from a visual foundation model, a visual HoT is established, and cross-modal co-attention graph learning is
chosen to interact with the textual HoT, realizing multimodal comparative judgement. The experimental results in the
scienceQA dataset indicate that the proposed HoT-enhanced foundation models outperform CoT-based GPT3.5 and
chatGPT.

Although the current multimodal HoT has not achieved as excellent performance as imagined, it can open up a new
paradigm for instruction learning in foundation models. We believe multimodal HoT will tap its potential in the
increasingly popular era of multimodal foundation models in the future. The HoT will be hot!

References

[1] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng
Gao. Chameleon: Plug-and-play compositional reasoning with large language models. CoRR, abs/2304.09842,
2023.

[2] Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, and Xia Hu.

Harnessing the power of llms in practice: A survey on chatgpt and beyond, 2023.

[3] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts,
Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha
Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran,
Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari,
Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier
Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim,

10

GoldQuestion: Which animal's skin is better adapted as a warning sign to ward off predators?Choices: (A) lichen katydid (B) flamboyant cuttlefishRational:  The flamboyant cuttlefish has a poisonous body with brightly colored skin. Its skin is adapted to ward off predators. The lichen katydid has green and white patches on its body. Its skin is not adapted to be a warning sign that wards off predators.Answer: The answer is (B).HoT PredictionRational:  The flamboyant cuttlefish has a poisonous body with brightly colored skin. Its skin is adapted to ward off predators. The lichen katydid has green and white patches on its body. Its skin is not adapted to be a warning sign that wards off predators.Answer: The answer is (B).CoT PredictionRational:  The flamboyant cuttlefish has a poisonous body with brightly colored skin. Its skin is adapted to ward off predators. The lichen katydid has green and white patches on its body. Its skin is not adapted to be a warning sign that wards off predators.Answer: The answer is (A).Right rationales right answerRight rationales wrong answerA PREPRINT - AUGUST 14, 2023

Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M.
Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Olek-
sandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele
Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling
language modeling with pathways, 2022.

[4] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan,
Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig,
Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer
language models, 2022.

[5] BigScience Workshop, :, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili�c, Daniel
Hesslow, Roman Castagn�, Alexandra Sasha Luccioni, Fran�ois Yvon, Matthias Gall�, Jonathan Tow, Alexander M.
Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Beno�t Sagot, Niklas
Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-
Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Lauren�on,
Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa,
Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue,
Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, Dragomir Radev, Eduardo Gonz�lez
Ponferrada, Efrat Levkovizh, Ethan Kim, Eyal Bar Natan, Francesco De Toni, G�rard Dupont, Germ�n Kruszewski,
Giada Pistilli, Hady Elsahar, Hamza Benyamina, Hieu Tran, Ian Yu, Idris Abdulmumin, Isaac Johnson, Itziar
Gonzalez-Dios, Javier de la Rosa, Jenny Chim, Jesse Dodge, Jian Zhu, Jonathan Chang, J�rg Frohberg, Joseph
Tobing, Joydeep Bhattacharjee, Khalid Almubarak, Kimbo Chen, Kyle Lo, Leandro Von Werra, Leon Weber,
Long Phan, Loubna Ben allal, Ludovic Tanguy, Manan Dey, Manuel Romero Mu�oz, Maraim Masoud, Mar�a
Grandury, Mario �a�ko, Max Huang, Maximin Coavoux, Mayank Singh, Mike Tian-Jian Jiang, Minh Chien Vu,
Mohammad A. Jauhar, Mustafa Ghaleb, Nishant Subramani, Nora Kassner, Nurulaqilla Khamis, Olivier Nguyen,
Omar Espejel, Ona de Gibert, Paulo Villegas, Peter Henderson, Pierre Colombo, Priscilla Amuok, Quentin
Lhoest, Rheza Harliman, Rishi Bommasani, Roberto Luis L�pez, Rui Ribeiro, Salomey Osei, Sampo Pyysalo,
Sebastian Nagel, Shamik Bose, Shamsuddeen Hassan Muhammad, Shanya Sharma, Shayne Longpre, Somaieh
Nikpoor, Stanislav Silberberg, Suhas Pai, Sydney Zink, Tiago Timponi Torrent, Timo Schick, Tristan Thrush,
Valentin Danchev, Vassilina Nikoulina, Veronika Laippala, Violette Lepercq, Vrinda Prabhu, Zaid Alyafeai,
Zeerak Talat, Arun Raja, Benjamin Heinzerling, Chenglei Si, Davut Emre Ta�sar, Elizabeth Salesky, Sabrina J.
Mielke, Wilson Y. Lee, Abheesht Sharma, Andrea Santilli, Antoine Chaffin, Arnaud Stiegler, Debajyoti Datta,
Eliza Szczechla, Gunjan Chhablani, Han Wang, Harshit Pandey, Hendrik Strobelt, Jason Alan Fries, Jos Rozen,
Leo Gao, Lintang Sutawika, M Saiful Bari, Maged S. Al-shaibani, Matteo Manica, Nihal Nayak, Ryan Teehan,
Samuel Albanie, Sheng Shen, Srulik Ben-David, Stephen H. Bach, Taewoon Kim, Tali Bers, Thibault Fevry,
Trishala Neeraj, Urmish Thakker, Vikas Raunak, Xiangru Tang, Zheng-Xin Yong, Zhiqing Sun, Shaked Brody,
Yallow Uri, Hadar Tojarieh, Adam Roberts, Hyung Won Chung, Jaesung Tae, Jason Phang, Ofir Press, Conglong
Li, Deepak Narayanan, Hatim Bourfoune, Jared Casper, Jeff Rasley, Max Ryabinin, Mayank Mishra, Minjia
Zhang, Mohammad Shoeybi, Myriam Peyrounette, Nicolas Patry, Nouamane Tazi, Omar Sanseviero, Patrick von
Platen, Pierre Cornette, Pierre Fran�ois Lavall�e, R�mi Lacroix, Samyam Rajbhandari, Sanchit Gandhi, Shaden
Smith, St�phane Requena, Suraj Patil, Tim Dettmers, Ahmed Baruwa, Amanpreet Singh, Anastasia Cheveleva,
Anne-Laure Ligozat, Arjun Subramonian, Aur�lie N�v�ol, Charles Lovering, Dan Garrette, Deepak Tunuguntla,
Ehud Reiter, Ekaterina Taktasheva, Ekaterina Voloshina, Eli Bogdanov, Genta Indra Winata, Hailey Schoelkopf,
Jan-Christoph Kalo, Jekaterina Novikova, Jessica Zosa Forde, Jordan Clive, Jungo Kasai, Ken Kawamura,
Liam Hazan, Marine Carpuat, Miruna Clinciu, Najoung Kim, Newton Cheng, Oleg Serikov, Omer Antverg,
Oskar van der Wal, Rui Zhang, Ruochen Zhang, Sebastian Gehrmann, Shachar Mirkin, Shani Pais, Tatiana
Shavrina, Thomas Scialom, Tian Yun, Tomasz Limisiewicz, Verena Rieser, Vitaly Protasov, Vladislav Mikhailov,
Yada Pruksachatkun, Yonatan Belinkov, Zachary Bamberger, Zden?ek Kasner, Alice Rueda, Amanda Pestana,
Amir Feizpour, Ammar Khan, Amy Faranak, Ana Santos, Anthony Hevia, Antigona Unldreaj, Arash Aghagol,
Arezoo Abdollahi, Aycha Tammour, Azadeh HajiHosseini, Bahareh Behroozi, Benjamin Ajibade, Bharat Saxena,
Carlos Mu�oz Ferrandis, Daniel McDuff, Danish Contractor, David Lansky, Davis David, Douwe Kiela, Duong A.
Nguyen, Edward Tan, Emi Baylor, Ezinwanne Ozoani, Fatima Mirza, Frankline Ononiwu, Habib Rezanejad,
Hessie Jones, Indrani Bhattacharya, Irene Solaiman, Irina Sedenko, Isar Nejadgholi, Jesse Passmore, Josh Seltzer,
Julio Bonis Sanz, Livia Dutra, Mairon Samagaio, Maraim Elbadri, Margot Mieskes, Marissa Gerchick, Martha
Akinlolu, Michael McKenna, Mike Qiu, Muhammed Ghauri, Mykola Burynok, Nafis Abrar, Nazneen Rajani,
Nour Elkott, Nour Fahmy, Olanrewaju Samuel, Ran An, Rasmus Kromann, Ryan Hao, Samira Alizadeh, Sarmad
Shubber, Silas Wang, Sourav Roy, Sylvain Viguier, Thanh Le, Tobi Oyebade, Trieu Le, Yoyo Yang, Zach Nguyen,
Abhinav Ramesh Kashyap, Alfredo Palasciano, Alison Callahan, Anima Shukla, Antonio Miranda-Escalada,
Ayush Singh, Benjamin Beilharz, Bo Wang, Caio Brito, Chenxi Zhou, Chirag Jain, Chuxin Xu, Cl�mentine

11

A PREPRINT - AUGUST 14, 2023

Fourrier, Daniel Le�n Peri��n, Daniel Molano, Dian Yu, Enrique Manjavacas, Fabio Barth, Florian Fuhrimann,
Gabriel Altay, Giyaseddin Bayrak, Gully Burns, Helena U. Vrabec, Imane Bello, Ishani Dash, Jihyun Kang,
John Giorgi, Jonas Golde, Jose David Posada, Karthik Rangasai Sivaraman, Lokesh Bulchandani, Lu Liu, Luisa
Shinzato, Madeleine Hahn de Bykhovetz, Maiko Takeuchi, Marc P�mies, Maria A Castillo, Marianna Nezhurina,
Mario S�nger, Matthias Samwald, Michael Cullan, Michael Weinberg, Michiel De Wolf, Mina Mihaljcic, Minna
Liu, Moritz Freidank, Myungsun Kang, Natasha Seelam, Nathan Dahlberg, Nicholas Michio Broad, Nikolaus
Muellner, Pascale Fung, Patrick Haller, Ramya Chandrasekhar, Renata Eisenberg, Robert Martin, Rodrigo Canalli,
Rosaline Su, Ruisi Su, Samuel Cahyawijaya, Samuele Garda, Shlok S Deshmukh, Shubhanshu Mishra, Sid
Kiblawi, Simon Ott, Sinee Sang-aroonsiri, Srishti Kumar, Stefan Schweter, Sushil Bharati, Tanmay Laud, Th�o
Gigant, Tomoya Kainuma, Wojciech Kusa, Yanis Labrak, Yash Shailesh Bajaj, Yash Venkatraman, Yifan Xu,
Yingxin Xu, Yu Xu, Zhe Tan, Zhongli Xie, Zifan Ye, Mathilde Bras, Younes Belkada, and Thomas Wolf. Bloom:
A 176b-parameter open-access multilingual language model, 2023.

[6] Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi
Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang,
Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas
Dixon, Kun Zhang, Quoc V Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. Glam: Efficient scaling of language
models with mixture-of-experts, 2022.

[7] Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides,
Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer,
Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese,
Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia
Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme
Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna
Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-
Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby
Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d�Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik,
Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson,
Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell,
Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu,
and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher, 2022.

[8] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego
de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican,
George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W.
Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022.

[9] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia
Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo
Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng
Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will
Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris,
Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran,
Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi
Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein,
Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. Lamda: Language models
for dialog applications, 2022.

[10] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth�e Lacroix,
Baptiste Rozi�re, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave,
and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.

[11] OpenAI. Gpt-4 technical report, 2023.

[12] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur,

David Rosenberg, and Gideon Mann. Bloomberggpt: A large language model for finance, 2023.

[13] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark,
and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering.
Advances in Neural Information Processing Systems, 35:2507�2521, 2022.

[14] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny

Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.

12

A PREPRINT - AUGUST 14, 2023

[15] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny

Zhou. Self-consistency improves chain of thought reasoning in language models, 2023.

[16] Xuefei Ning, Zinan Lin, Zixuan Zhou, Huazhong Yang, and Yu Wang. Skeleton-of-thought: Large language

models can do parallel decoding, 2023.

[17] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan.

Tree of thoughts: Deliberate problem solving with large language models, 2023.

[18] Yao Yao, Zuchao Li, and Hai Zhao. Beyond chain-of-thought, effective graph-of-thought reasoning in large

language models, 2023.

[19] Xian Sun, Fanglong Yao, and Chibiao Ding. Modeling high-order relationships: Brain-inspired hypergraph-
induced multimodal-multitask framework for semantic comprehension. IEEE Transactions on Neural Networks
and Learning Systems, pages 1�15, 2023.

[20] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought

reasoning in language models. CoRR, abs/2302.00923, 2023.

[21] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models

are zero-shot reasoners, 2023.

[22] Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. arXiv preprint

arXiv:2212.10071, 2022.

[23] Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. Teaching small

language models to reason. arXiv preprint arXiv:2212.08410, 2022.

[24] Liunian Harold Li, Jack Hessel, Youngjae Yu, Xiang Ren, Kai-Wei Chang, and Yejin Choi. Symbolic chain-of-
thought distillation: Small models can also" think" step-by-step. arXiv preprint arXiv:2306.14050, 2023.

[25] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought

reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.

[26] Yao Yao, Zuchao Li, and Hai Zhao. Beyond chain-of-thought, effective graph-of-thought reasoning in large

language models. arXiv preprint arXiv:2305.16582, 2023.

[27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint

arXiv:2304.08485, 2023.

[28] Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, and Rongrong Ji. Cheap and quick: Efficient

vision-language instruction tuning for large language models. arXiv preprint arXiv:2305.15023, 2023.

[29] Lei Wang, Yi Hu, Jiabang He, Xing Xu, Ning Liu, Hui Liu, and Heng Tao Shen. T-sciq: Teaching multimodal
chain-of-thought reasoning via large language model signals for science question answering. arXiv preprint
arXiv:2305.03453, 2023.

[30] Sameera Horawalavithana, Sai Munikoti, Ian Stewart, and Henry Kvinge. Scitune: Aligning large language

models with scientific multimodal instructions. arXiv preprint arXiv:2307.01139, 2023.

[31] Yu-Jung Heo, Eun-Sol Kim, Woo Suk Choi, and Byoung-Tak Zhang. Hypergraph transformer: Weakly-supervised

multi-hop reasoning for knowledge-based visual question answering. arXiv preprint arXiv:2204.10448, 2022.

[32] Eun-Sol Kim, Woo Young Kang, Kyoung-Woon On, Yu-Jung Heo, and Byoung-Tak Zhang. Hypergraph attention
networks for multimodal learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pages 14581�14590, 2020.

[33] Yu Tian, Xian Sun, Ruigang Niu, Hongfeng Yu, Zicong Zhu, Peijin Wang, and Kun Fu. Fully-weighted hgnn:
Learning efficient non-local relations with hypergraph in aerial imagery. ISPRS Journal of Photogrammetry and
Remote Sensing, 191:263�276, 2022.

[34] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin
transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international
conference on computer vision, pages 10012�10022, 2021.

[35] Yue Gao, Zizhao Zhang, Haojie Lin, Xibin Zhao, Shaoyi Du, and Changqing Zou. Hypergraph learning: Methods
and practices. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2548�2566, 2022.

[36] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei
Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of
Machine Learning Research, 21(1):5485�5551, 2020.

13

A PREPRINT - AUGUST 14, 2023

[37] Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh
Hajishirzi. Unifiedqa: Crossing format boundaries with a single QA system. In Trevor Cohn, Yulan He, and
Yang Liu, editors, Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20
November 2020, volume EMNLP 2020 of Findings of ACL, pages 1896�1907. Association for Computational
Linguistics, 2020.

[38] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.
End-to-end object detection with transformers. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael
Frahm, editors, Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020,
Proceedings, Part I, volume 12346 of Lecture Notes in Computer Science, pages 213�229. Springer, 2020.
[39] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. Deep modular co-attention networks for visual question
answering. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA,
June 16-20, 2019, pages 6281�6290. Computer Vision Foundation / IEEE, 2019.

[40] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang.
Bottom-up and top-down attention for image captioning and visual question answering. In 2018 IEEE Conference
on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages
6077�6086. Computer Vision Foundation / IEEE Computer Society, 2018.

[41] Jin-Hwa Kim, Jaehyun Jun, and Byoung-Tak Zhang. Bilinear attention networks. In Samy Bengio, Hanna M.
Wallach, Hugo Larochelle, Kristen Grauman, Nicol� Cesa-Bianchi, and Roman Garnett, editors, Advances in
Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018,
NeurIPS 2018, December 3-8, 2018, Montr�al, Canada, pages 1571�1581, 2018.

[42] Peng Gao, Zhengkai Jiang, Haoxuan You, Pan Lu, Steven C. H. Hoi, Xiaogang Wang, and Hongsheng Li.
Dynamic fusion with intra- and inter-modality attention flow for visual question answering. In IEEE Conference
on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages
6639�6648. Computer Vision Foundation / IEEE, 2019.

[43] Pan Lu, Liang Qiu, Jiaqi Chen, Tanglin Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun
Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. In Joaquin
Vanschoren and Sai-Kit Yeung, editors, Proceedings of the Neural Information Processing Systems Track on
Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021.

[44] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and

performant baseline for vision and language. CoRR, abs/1908.03557, 2019.

[45] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region
supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on
Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning
Research, pages 5583�5594. PMLR, 2021.

[46] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E. Hinton. Big self-supervised
models are strong semi-supervised learners. In Hugo Larochelle, Marc�Aurelio Ranzato, Raia Hadsell, Maria-
Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual
Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

14


