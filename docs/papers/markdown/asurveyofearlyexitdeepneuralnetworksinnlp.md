A Survey of Early Exit Deep Neural Networks in NLP

Divya Jyoti Bajpai and Manjesh Kumar Hanawal
Department of IEOR, IIT Bombay
{divyajyoti.bajpai, mhanawal}@iitb.ac.in

5
2
0
2

n
a
J

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
0
7
6
7
0
.
1
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

However,

Deep Neural Networks (DNNs) have grown
increasingly large in size to achieve state-
of-the-art performance across a wide range
of tasks.
their high computa-
tional requirements make them less suitable
for resource-constrained applications. Also,
real-world datasets often consist of a mixture
of easy and complex samples, necessitating
adaptive inference mechanisms that account
for sample difficulty. Early exit strategies offer
a promising solution by enabling adaptive in-
ference, where simpler samples are classified
using the initial layers of the DNN, thereby ac-
celerating the overall inference process. By at-
taching classifiers at different layers, early exit
methods not only reduce inference latency but
also improve the model�s robustness against
adversarial attacks. This paper presents a com-
prehensive survey of early exit methods and
their applications in NLP.

1

Introduction

Deep Neural Networks (DNNs) such as BERT
(Devlin et al., 2018), GPT (Radford et al., 2019),
XLNet (Yang et al., 2019), ALBERT (Lan et al.,
2019), ViT (Alexey, 2020), BLIP-2 (Li et al.,
2023), Llama (Touvron et al., 2023) etc., have
expanded significantly in size, achieving signifi-
cant improvements in various Image and Natural
Language Processing (NLP) tasks. These mod-
els leverage large-scale pre-training on unlabeled
data, followed by fine-tuning on labeled datasets
to deliver state-of-the-art performance. The large
size of these DNNs introduces several challenges
in deployment. The first major issue is deploying
them on resource-constrained devices such as mo-
bile phones, edge devices, and IoT platforms to
maintain their high performance. The second is-
sue is �overthinking�, where DNNs continue pro-
cessing even when shallow layers could produce
correct inferences for easier samples as shown in

Figure 1: Difference between the DNN and EEDNN.

(Kaya et al., 2019; Michel et al., 2019; Zhou et al.,
2020). This unnecessary deep processing can
overfit irrelevant features, resulting in poor gen-
eralization and wasted computation. Additionally,
overthinking contributes to the models� suscepti-
bility to adversarial attacks (Zhou et al., 2020).

To address these issues, recent research has fo-
cused on accelerating DNN inference and making
their implementation feasible for limited-resource
environments. Techniques like pruning (Fan et al.,
2019; Michel et al., 2019), quantization (Zhang
et al., 2020; Bai et al., 2020; Kim et al., 2021)
and knowledge distillation (Sanh et al., 2019; Jiao
et al., 2019) have been employed to reduce the size
of DNNs. These compression methods decrease
the model size but often sacrifice the optimal per-
formance and versatility of the original networks.
These methods use the same processing on each
sample without any adpation, which makes them
static, leading to suboptimal performance and in-
efficient usage of resources. Real-world tasks con-
sist of samples with varying levels of complexity,
hence they do not need the same computational ef-
fort. This variability calls for input-adaptive infer-
ence methods that tailor the computational effort
to the complexity of each input.

Pos0.91Pos0.72Neg0.51Pos0.87Prediction scoreSuperb product!Superb product!Conventional DNNEarly Exit DNNLayer 1Layer 2Layer 3Layer LLayer 1Layer 2Layer 3Layer LCLCLC1C2C3

Early Exit (EE) (Teerapittayanon et al., 2016)
methods have emerged as a state-of-the-art input-
adaptive approach to address the challenges of
overthinking and latency in DNN inference. These
methods incorporate intermediate classifiers at
several layers within the DNN, allowing inference
to occur at multiple stages. The inference pro-
cess halts once the model reaches a sufficient level
of confidence in its prediction, enabling dynamic,
�anywhere� predictions.
Samples that achieve
high prediction confidence at the at shallower lay-
ers exit early, while only more complex samples
are processed deeper into the network (Xu and
McAuley, 2022). In Figure 1, we show the con-
ventional DNN and EEDNN where conventional
DNN exits the sample only at the final layer,
while the EEDNN infers the sample at the 3rd
layer as it gains sufficient confidence there. Any-
where classification allows these models to be par-
titioned and utilized for edge-cloud co-inference
setup where part of the DNN is deployed on the
edge and full-fledged DNN on the cloud.

The EE methods have been widely popular
in NLP tasks, where they are applied to Large-
Language Models (LLMs) and Vision-Language
Models (VLMs). Also, there are very few sys-
tematic surveys on early exit DNNs. Matsubara
et al. (2022) touches upon the EE framework as
the application for edge-cloud co-inference setup.
Han et al. (2021) reviewed the complete area of
dynamic neural networks. Being, one subset of
the dynamic neural networks, it only touches upon
the EE networks. Rahmath P et al. (2024) re-
views EEDNNs mostly on image tasks and briefly
touches upon the NLP methods.

Since EE methods have been widely adopted for
NLP tasks, a comprehensive survey of EEDNNs
for NLP is lacking. This gap motivates us to un-
dertake this survey. The aim of this survey is
to (1) provide a thorough overview and new in-
sights for researchers interested in early exit meth-
ods for NLP; (2) highlight the interconnections be-
tween different subareas, thereby minimizing re-
dundancy and the risk of reinventing the wheel;
and (3) summarize key challenges and outline po-
tential directions for future research in this evolv-
ing field.

2 Advantages of EEDNNs

Figure 2: The figure shows the average of the confi-
dence values over the true class across all the layers for
the SST-2 dataset.

based on the complexity of incoming samples.
The key benefits of EE models are outlined below:
1) Faster Inference: EE models come with
additional side branches (exits) attached to the
DNN. A significant advantage of EE models is
their ability to allocate computational resources
selectively at inference time, activating only rel-
evant sub-networks based on the input sample due
to attached exits. This results in faster inference,
as computational effort is minimized for simpler,
easier-to-recognize samples.

2) Input-Adaptiveness: EE models adapt com-
putational effort based on the complexity of in-
coming samples, using less power for easier sam-
ples without compromising accuracy. Figure 2 il-
lustrates this by plotting average confidence val-
ues on the true class across intermediate exits and
the final layer of the BERT models with EE at
every layer. Approximately 80% of samples, la-
beled as �confident,� exhibit high confidence and
are predicted in the initial layers. �Confused� sam-
ples show fluctuating confidence across classes,
indicating model uncertainty. Finally, �fake con-
fidence� samples fall outside the model�s scope,
where the model incorrectly becomes confident
about the wrong class, leading to mispredictions.
3) Generality: EE methods are versatile and
can be applied to a wide range of tasks, includ-
ing image classification, object detection, natu-
ral language processing, text generation, and im-
age captioning, often with minimal modifications
to the model design. This generalizability allows
EE methods developed for one task to be easily
adapted to others.

EE methods offer several advantages over static
models by dynamically adjusting computation

4) Interpretability: EE models enhance the in-
terpretability of DNNs by providing insights into

the decision-making process at each stage of the
network. By allowing users to observe which sam-
ples exit early and which proceed to the deeper
layers. These models offer a clearer understand-
ing of how the network differentiates between sim-
pler and more complex samples, facilitating a bet-
ter understanding of the data being processed. For
instance, Figure 2 provides a deeper insight into
the hardness of incoming samples and can help de-
tect OOD samples from a dataset that is out of the
model�s scope.

5) Robustness: EE models demonstrate in-
creased robustness against adversarial attacks
compared to traditional DNNs. The use of multi-
ple intermediate classifiers creates an ensemble ef-
fect, where the impact of noise or adversarial per-
turbations is mitigated by leveraging predictions
from different layers, resulting in more reliable
and confident final outputs (Zhou et al., 2020).

6) Distributed Inference: EE models offer
anytime prediction by attaching intermediate clas-
sifiers, making them well-suited for varying com-
putational budgets and hardware constraints. This
adaptability allows EE models to operate effec-
tively across different hardware platforms and dy-
namic environments, making them particularly
valuable in distributed computing setups (Teer-
apittayanon et al., 2017).
It could be easily
adapted to various mobile-edge, edge-cloud or
mobile-edge-cloud co-inference setups.

7) Mitigates Overthinking: EEs also solve the
overthinking issue in DNNs by not forcing a sam-
ple to pass through deeper layers even when the
sample has gained enough confidence in the ini-
tial layers. Sometimes excessive processing of
easy samples deeper into the backbone may lead to
wrong prediction due to irrelevant feature extrac-
tion. Mitigating this not only improves accuracy
but also reduces wasteful computation.

Other than these EEs, they also help reduce
overfitting, where the interaction between differ-
ent side branches acts as a regularizer for the
model. This solves the vanishing gradient problem
by giving the gradient signal from the initial layer
that is less prone to vanishing gradient issues.

These properties make Early Exit methods a
powerful tool for deploying DNNs in resource-
constrained environments and diverse application
areas, where efficiency, adaptability, and robust-
ness are critical. They have been widely adopted
in various fields such as image classification (Teer-
apittayanon et al., 2016; Huang et al., 2017;

Laskaridis et al., 2020; Dai et al., 2020; Wang
et al., 2020; Fang et al., 2020; Li et al., 2019a;
Phuong and Lampert, 2019; Li et al., 2019b;
Wo?czyk et al., 2021; KhademSohi et al.), NLP
tasks (Bapna et al., 2020; Elbayad et al., 2019; Liu
et al., 2021; Balagansky and Gavrilov, 2022; Xin
et al., 2021; Sun et al., 2022; Gao et al., 2023; Ba-
jpai and Hanawal, 2024b; Miao et al., 2024), im-
age captioning (Fei et al., 2022; Tang et al., 2023c;
Miao et al., 2024; Bajpai and Hanawal, 2024a) etc.

2.1 Areas of research

While Early Exit (EE) methods effectively ad-
dress the above-mentioned issues in DNN infer-
ence, they require careful design choices regard-
ing the confidence metric, training strategies and
exit criteria. Training EE-based DNNs (EEDNNs)
is inherently a multi-objective problem since each
intermediate classifier aims to optimize its perfor-
mance. The decision to exit at a particular layer
is based on the intermediate classifier being con-
fident and is governed by a confidence metric that
must exceed a predefined threshold. This thresh-
old setting is critical to the inference process, as
a higher threshold allows for more accurate pre-
dictions at deeper layers but may also increase la-
tency, while a lower threshold does the opposite.

Research on EEDNNs has primarily focused on
improving specific aspects, as summarized below:
1) Exiting Criteria: A key area of research
involves the choice of confidence metrics and
threshold settings tailored to specific tasks. This
includes strategies for leveraging the outputs of
multiple intermediate classifiers to achieve a bet-
ter estimate of the true label and setting thresh-
olds that balance the trade-off between accuracy
and efficiency (Zhou et al., 2020; Balagansky and
Gavrilov, 2022; Zhang et al., 2022; Xin et al.,
2020; Bajpai and Hanawal, 2024b).

2) Training Strategies: The training of exit
classifiers at multiple layers poses a multi-
objective optimization problem. The task of each
intermediate layer in the EEDNN has two objec-
tives: 1) Provide hidden representations such that
the exit classifier loss is minimized. 2) Hidden
representations should be such that the final layer
accuracy is also not compromised.

Various training approaches have been investi-
gated, such as joint optimization of all exits or
separate optimization of each exit and the back-
bone. Some works also distil the knowledge from
deeper layers to initial layers for better learning of

3.1 Setup

To construct an EEDNN, classifiers are integrated
at intermediate layers to map the hidden represen-
tations of the backbone network to output proba-
bilities. These additional classifiers not only pro-
vide regularization to the main network but also
offer more direct gradient signals for backpropa-
gation, particularly from shallower layers.

In designing an EEDNN, several key factors
must be considered: (1) the training strategy for
classifiers at all intermediate layers; (2) the ar-
chitecture of the classifiers, including their size,
depth, and complexity (e.g., a single linear layer
(Xin et al., 2020), multiple fully connected layers
(Fei et al., 2022) and combination of self-attention
and fully connected layers; (3) the exit criteria for
each classifier and the associated computational
cost; and (4) the optimal placement of exit points.

3.2 Training methods

Separate Training: Methods such as Xin et al.
(2020); Bajpai and Hanawal (2024a) perform sep-
arate training as detailed below and in Figure 3.
Let us consider that there are N layers in the back-
bone. We also consider that D represents the dis-
tribution of the dataset with a label class C used for
the backbone training. For fine-tuning the back-
bone, the loss function for ith exit is written as:

Li(?) = LCE(fi(x, ?), y)

(1)

Here, fi(x, ?) is the output of the classifier at-
tached at the ith exit, ? denotes the collection of
all the parameters, LCE is the cross-entropy loss
and (x, y) ? D.

In separate training,
fine-tuning in two stages:

the network undergoes

1) The first stage involves updating the embed-
ding layer, all transformer layers, and the final
classifier, with the loss function being solely LN .
This is just standard backbone fine-tuning.

2) In the second stage, the parameters fine-tuned
in the first stage are frozen, and only the remain-
ing components, excluding the final classifier, are
updated. Here, the loss function is (cid:80)N ?1
i=1 wiLi.
This approach ensures that the backbone parame-
ters remain fixed to preserve their optimal quality;
otherwise, the transformer layers might no longer
be optimized exclusively for the final layer, which
generally leads to a decline in its performance.

Joint Training: Methods such as Zhou et al.
(2020); Bajpai and Hanawal (2024b) perform Joint

Figure 3: Separate training vs Joint Training

the intermediate classifiers. Additionally, attach-
ing classifiers at multiple layers introduces more
parameters to the model, which raises the ques-
tion of how to strategically place these exits across
the network to avoid excessive model size, particu-
larly for very large models (Zhu, 2021; Zhou et al.,
2020; Wang et al., 2019; Xin et al., 2021).

3) Generalization of EEDNNs: While large
DNNs generally exhibit strong generalization ca-
pabilities, EEDNNs can inherit these properties,
but task-specific confidence metrics and thresh-
olds often constrain their generalization. As the
domain of the input data changes, the distribution
of confidence scores at the exits can also shift,
which requires addressing such concerns (Bajpai
and Hanawal, 2024b,c).

4) Handling Complex Tasks: For more com-
plex tasks, such as text generation, EEDNNs tend
to suffer a greater performance drop. This is
because the earlier layers typically capture only
syntactic information, while deeper layers are re-
quired to extract semantic meaning. A chal-
lenge remains in how to equip the initial layers of
EEDNNs with the higher-level information typi-
cally found in deeper layers of the network (Fei
et al., 2022; Bajpai and Hanawal, 2024a).

3 Foundation of Early Exit DNNs

EEDNNs belong to a class of dynamic neural net-
works that adaptively adjust the inference process,
by selectively using a subpart of the model based
on input sample complexity.
In this section, we
outline the general framework of EEDNN models:
their typical training and inference procedures.

Separate TrainingTraining the backboneTraining the ClassifiersyLy3y2y1y3y2y1yLyLTraining backbone with classifiers.Joint TrainingLayer 1Layer 2Layer 3Layer LLayer 1Layer 2Layer 3Layer LLayer 1Layer 2Layer 3Layer LCLCLC1C2C3C1C2C3CLEmbedEmbedEmbedFigure 4: Inference methods: 1) Max Probability: confidence is the maximum output of an individual classifier.
2) Patience-based: relies on prediction consistency between classifiers. 3) Ensemble: aggregates weighted results
from multiple classifiers.

Training where instead of first finetuning the back-
bone and freezing its weights, the complete back-
bone is simultaneously optimized (see Figure 3).
Hence the loss function is: L = (cid:80)N
i=1 wiLi.
This method simultaneously finetunes the back-
bone and learns the classifier weights.

The weights wi in both the separate and joint
training are the weights provided based on the cost
associated with each exit classifier. Most of the
methods replace wi = i with a justification that
more emphasis should be given to deeper layers.
However, DynExit (Wang et al., 2019) proposes
wi to be trainable parameters and use ?(wi) in-
stead of wi where ? is the sigmoid function. After
this step, the backbone is ready for inference.

Other methods: Some methods use a combina-
tion of the existing methods such as BERxiT (Xin
et al., 2021) uses the alternate training where in
one iteration the backbone weights are optimized
and in the next step the exit weights are optimized.
As the exits have two objectives, the motivation
for using this method is to have a good balance
between the two objectives.

Other than these methods some works addition-
ally use knowledge distillation between the layers
(Zhu, 2021; Geng et al., 2021) or distillation from
the final layer to the other intermediate classifiers
(Bajpai and Hanawal, 2024a).

3.3 Defining confidence

After training the backbone, it is necessary to de-
fine the confidence of the exit classifiers. This sub-
section details different measures of confidence

for deciding to exit.

Individual confidence-based: Let �Pi(c) repre-
sent the estimated probability that input x belongs
to class c ? C, and let Ci denote the confidence
in this estimate for the ith exit. CeeBERT (Baj-
pai and Hanawal, 2024b) defines confidence as the
maximum estimated probability across all classes,
i.e., Ci := maxc?C �Pi(c). In contrast, DeeBERT
(Xin et al., 2020) and ElasticBERT (Liu et al.,
2021) use the entropy of the �Pi(c) as the confi-
dence score. Note that these methods only use the
output from a single classifier.

Patience-based: PABEE (Zhou et al., 2020)
takes a different approach by defining confidence
based on prediction consistency across multiple
If predictions from several con-
exit classifiers.
secutive classifiers remain consistent,
the sam-
ple is inferred. LeeBERT (Zhu, 2021) also uti-
lizes patience-based exiting similar to PABEE.
The advantage of this method is that it reduces the
chances of adversarial attacks as its predictions are
based on multiple classifier�s output.

Distribution-based:

In this category, works
like PALBERT (Balagansky and Gavrilov, 2022)
introduce the Q-exit strategy, where a distribu-
tion p(i|x) is learned over exit classifiers, repre-
senting the probability that a sample exits at the
ith layer. A sample exits the backbone once the
cumulative distribution function (CDF) exceeds
JEI-DNN (Chataoui
a predetermined threshold.
et al., 2023) learns the distribution over the exit
layers using joint optimization without requiring
additional training. The major advantage of this

Pos0.88Pos0.65Prediction scoreSuperb product!Pos1Pos0Patience CounterSuperb product!0.120.880.350.65Prediction scoreSuperb product!Pos2S1(a) Max of Probability(b) Patience-based(c) Ensemble MethodsPos0.91S2  w1w2        : Sample exits        : Sample does not exits               : Aggregating score         : Weighing classifierLayer 1Layer 2Layer 3Layer LEmbedC1C2C3CLLayer 1C1Layer 2EmbedLayer 3Layer LC2C3CLEmbedLayer 1Layer 2Layer 3Layer LC1C2C3CLmethod is it does not require to verify the con-
fidence at every exit instead for every incoming
sample, an intermediate exit is assigned and it is
directly inferred at that exit.

Similarity-based: MuE (Tang et al., 2023c)
model decides upon exiting based on the similarity
score of the consecutive layers. At every layer, the
similarity of hidden representations with the pre-
vious layer is calculated and if it is less than the
given threshold, the sample exits the backbone.
The motivation for this method comes from the
fact that the hidden representations saturate once
sufficient features are extracted. The advantage of
this method is that it reduces the need for checking
the confidence values after processing through the
exit instead it can decide to exit based on similarity
reducing computational demands.

Ensemble methods: Methods such as ZTW
(Wo?czyk et al., 2021) use ensemble-based exit-
ing criteria where weights are provided to differ-
ent classifiers depending on the confidence in the
classifier�s prediction, a sample is exited from the
backbone once the ensemble score exceeds a pre-
defined threshold. Similarly, Sun et al. (2021) uses
a majority vote to decide early inference of a sam-
ple, if a certain number of classifiers agree on one
class, the sample exits the backbone. The advan-
tage of this method is the ensemble of multiple
classifiers making predictions more trustworthy.

Other methods: BERxiT (Xin et al., 2021) in-
troduces learning-to-exit modules that use a sepa-
rate network to estimate sample uncertainty rather
than traditional confidence measures. HASHEE
(Sun et al., 2022) employs a hash-based strategy,
assigning exit layers based on sample clustering
based on frequency or embedding space, without
relying on confidence. Gao et al. (2023) combine
patience and similarity-based methods, exits when
consecutive layer similarities fall below a thresh-
old repeatedly. He et al. (2024) uses signal-based
exiting, allowing exits to prioritize samples likely
to exit under different acceleration scenarios.

3.4 Choice of thresholds

The threshold used to decide whether to exit is a
crucial part of the EEDNNs. The threshold models
the accuracy-efficiency trade-off. The ways to set
the thresholds are as follows:

thresholds:

Static
as
BranchyNet
al., 2016),
(Teerapittayanon et
PABEE (Zhou et al., 2020), LeeBERT (Zhu,
2021), DeeBERT (Xin et al., 2020), DeeDiff

Methods

such

(Tang et al., 2023b), FastBERT (Liu et al., 2020),
FlexDNN (Fang et al., 2020), DynExit (Wang
et al., 2019), etc. set the threshold based on the
best-performing threshold on the validation split
of the dataset. Most of the methods focus on
maximizing the accuracy of the validation set.
These methods apply a static threshold either by
greedily choosing the threshold based on accuracy
or some combination of accuracy and latency
which is not the goal always.

Dynamic thresholds: Methods such as Cee-
BERT (Bajpai and Hanawal, 2024b) and UCBEE
(Pacheco et al., 2024) model the problem of choos-
ing the optimal threshold using a Multi-Armed
Bandit (MAB) setup.
In their mobile-cloud co-
inference setup, the threshold is used to decide
if a sample can be inferred locally or should be
offloaded to the cloud. CeeBERT (Bajpai and
Hanawal, 2024b) on the other hand learns the op-
timal threshold using Multi-Armed Bandits setup
under the case that the test data distribution is dif-
ferent from the training dataset.
It defines a re-
ward function for the threshold consisting of both
the confidence in prediction and the cost of pro-
cessing a sample into the backbone. MuE (Tang
et al., 2023c) also uses a dynamic threshold for im-
age captioning tasks where the threshold value de-
creases with the increasing length of the sentence.
MuE claims that the decoder tends to make fewer
mistakes as the sentence length gets longer.

3.5

Inference

During inference, as an input instance x sequen-
tially passes through layers 1, . . . , L, each exit
classifier positioned after the intermediate layers
produces a class label distribution. The inference
process halts at the ith exit classifier when the con-
fidence score Ci satisfies Ci ? ?, where the defi-
nition of Ci is as described in the previous section.
If the model does not reach a sufficient confidence
level by the final layer, the sample is inferred at
the final layer regardless of its confidence score.
This mechanism enables early exiting of a sam-
ple from the backbone when the confidence con-
dition is met, thus avoiding unnecessary traversal
through all layers.

4 Applications

In this section, we provide details of the applica-
tions of the early exit methods to different NLP do-
mains, such as text classification, natural language

inference (NLI), Language Translation, Sequence
Labeling and Image captioning tasks.

4.1 Text classification and NLI tasks

In most of the NLP tasks, the EE methods only at-
tach a linear classifier in the exit instead of a com-
plex structure as done on the image tasks. Dee-
BERT (Xin et al., 2020) first applied EEs to the
BERT backbone, it performed a separate training
and uses entropy as the confidence metric. Elas-
ticBERT (Liu et al., 2021) on the other hand per-
forms the training of the BERT backbone from
scratch i.e., during pre-training of the BERT back-
bone, the MLM and SOP heads are attached to ev-
ery layer instead of just the final layer. Hence af-
ter pre-training the backbone has learned weights
such the objective is not only to improve the fi-
nal layer�s performance. By pertaining the back-
bone from scratch with exits, it optimizes the per-
formance of the backbone for EE and final layer.

Some works such as PABEE (Zhou et al., 2020)
highlight the overthinking issues in the NLP tasks
and also show that these models not only perform
faster inference but also make the original model
robust to adversarial attacks. Since PABEE pro-
poses patience-based exiting criteria i.e., based on
prediction consistency, it does not rely on a single
classifier to decide exiting which makes it more
robust to the noise in the incoming samples.

BERxiT (Xin et al., 2021) performs an alter-
nating training strategy where in one iteration the
full backbone is optimized and in the next itera-
tion the exits are optimized. The exiting criteria
are learned where the decision to exit is taken by a
learned single linear layer that outputs uncertainty
in prediction. It empirically proves better perfor-
mance by alternate training and novel learning to
exit modules instead of only depending on the con-
fidence of the model.

Knowledge Distillation (KD) methods, initially
used to distil the knowledge of larger models into
smaller models have also been explored in early
exit models. FastBERT (Liu et al., 2020) uti-
lizes this strategy where it first finetunes the BERT
backbone and then attaches exits to the backbone.
Then the model weights are frozen and only exit
weights are trained where additional knowledge
distillation loss is applied from the final layer
to the student classifiers. LeeBERT (Zhu, 2021)
on the other hand, instead of learning from only
the final classifier allows knowledge to be dis-
tilled within multiple exits. It also uses cross-level

optimization by partitioning the training dataset,
where the training dataset is optimally split for
the backbone and the exit weights training i.e.,
the dataset used for backbone training is different
from the dataset used for exits training. KD loss
improves early exit accuracy by providing soft la-
bels with hard labels which improves accuracy as
well as efficiency.

Methods such as PALBERT (Balagansky and
Gavrilov, 2022) and ETFEE (Ji et al., 2023) have
proposed to alter the exit classifier�s configuration
where PALBERT extends transformer layers with
a Lambda layer that induces a generalized geomet-
ric distribution on the of exiting from the ith layer
(cid:81)i?1
j=1(1 ? ?j) where ?i is a
equal to p(i|x) = ?i
function of hidden representation at ith layer. ET-
FEE additionally has an adapter whose function is
to disentangle the task-specific and universal rep-
resentations. Also, instead of the classic classi-
fier, an equiangular tight frame (ETF) classifier is
added to enhance the classification ability of in-
ternal classifiers. Similarly Gao et al. (2023) uti-
lize the adapter module and perform parameter ef-
ficient fine-tuning for the exit classifiers and per-
form exiting based on the similarity between con-
secutive hidden layers. In these methods, the exits
are computationally expensive but are more accu-
rate as compared to other methods.

Liao et al. (2021) proposed a method that does
not use only a single classifier for inference but
all the past classifiers using ensemble strategies.
It also utilizes the future classifiers that have not
been explored by the sample by using an imitation
classifier which is a lightweight model with the
task of imitating the remaining transformer lay-
ers. It has improved the previous state-of-the-art
early exiting methods by using all the classifiers
and producing an ensemble effect. However, the
computational complexity of this method is higher
due to additional imitation classifiers that are used
to get the information from the deeper layers that
might not have been used due to the early exiting.
JEI-DNN (Chataoui et al., 2023) on the other
hand jointly learns a probability distribution along
with the classifier weights where it learns a dis-
tribution over the set of layers and during infer-
ence this distribution is utilized to decide the ex-
iting of the sample from a particular intermediate
exit without checking at other exits. This creates a
multi-objective problem and all tasks are simulta-
neously optimized. However, the balance between
different tasks needs to be maintained.

4.2 Text Summarization

4.5 Vision-language tasks

HASHEE (Sun et al., 2022) has applied early ex-
its for text summarization. Note that text summa-
rization is a more complex task as it involves the
generation of text, and hence requires better mod-
elling. The major contribution of HASHEE is it
does not require checking the confidence at ev-
ery layer instead it divides the vocabulary into n
buckets where n is the number of exits attached to
the backbone. The bucketing could be done based
on clustering, frequency and mutual information.
Each bucket is assigned one of the exits for infer-
ence. For instance, the tokens whose frequency
is higher are considered easier and are assigned
initial layers and the tokens that rarely appear are
assigned deeper layers. In this way, the computa-
tional cost is further reduced.

4.3 Sequence labeling tasks

Wang et al. (2020) proposed two early exiting
strategies for the sequence labeling tasks: 1) Sen-
tence level Early Exit (SENTEE) where complete
sentence exits together at one layer. To decide
which layer is suitable the uncertainty is defined as
the max of uncertainties over each token in the se-
quence. 2) TOKEE: The main issue of SENTEE is
that a sample cannot exit the backbone until each
token gets sufficient confidence. To circumvent
this TOKEE uses token level exiting i.e., as a token
in the sequence gets sufficient confidence, it is not
further processed saving the unnecessary compu-
tation of taking each token deep into the backbone.

4.4 Language Translation

HCN (Tsai et al., 2022) applies early exits to the
decoder of transformer models for language trans-
lation tasks. It performs separate training and dis-
tils final layer knowledge to the exits using knowl-
edge distillation loss The main issue faced was the
size of the exits has increasingly grown for the
translation tasks. To reduce the size of exits, HCN
reduces the vocab size for the shallower layers and
makes them learn about the specific token by not
adding up the loss of those tokens that are planned
to be removed from the vocab size. The choice of
the token used for different exits is made in a hier-
archal way where top-ki samples were kept for ith
exit based on their frequency in vocab, where k is
some constant. This significantly reduces the exit
classifier size further reducing the computational
complexity of the model.

Extending early exit methods to vision-language
tasks presents unique challenges: 1) Shallow lay-
ers primarily capture syntactic information, while
deeper layers encode semantic relations, making
initial exits lack semantic fusion capabilities. 2)
Image captioning models involve a large num-
ber of output classes equal to the vocabulary size,
resulting in significant parameter overhead when
adding classifiers to multiple exits.

DeeCap (Fei et al., 2022) addresses perfor-
mance degradation due to missing high-level
features by employing lightweight
imitation-
learning-based networks. An MLP mimics deeper
transformer layers using intermediate hidden rep-
resentations by outputting similar hidden repre-
sentations as the original transformer backbone,
mitigating the lack of high-level features. How-
ever, the computational complexity of this method
is quite high as the imitation network architecture
adds to the latency of the model.

introduces

al., 2023c)

MuE (Tang et

a
similarity-based exit criterion, assuming minimal
changes in hidden representations between layers
for confident samples. Exits occur when the sim-
ilarity score between consecutive layers falls be-
low a predefined threshold. Unlike other meth-
ods limited to decoders, MuE extends early exiting
to the encoder by halting feature extraction when
the threshold is met, passing the representations
directly to the decoder. The extension to the en-
coder also reduces the inference time in encoder-
decoder models. As the halting process does not
depend on the classifier�s confidence, it further re-
duces the inference time for performing inference
at every exit.

DEED (Tang et al., 2023a) uses adapter mod-
ules between exit classifiers and decoder layers to
minimize information loss in shallow layers.
It
standardizes intermediate classifiers across exits
and combines final layer loss with the average loss
from all exits to preserve backbone optimality.

CapEEN (Bajpai and Hanawal, 2024a)

in-
training
troduces a two-step training process:
the backbone without exits,
then freezing its
weights while training exits using cross-entropy
and knowledge distillation losses. Its variant, A-
CapEEN, leverages Multi-Armed Bandits to dy-
namically adjust exit thresholds during inference,
adapting to image noise.

5 Domain Generalization in EE Models

principles to identify the best action (threshold).

Large-scale DNNs have strong generalization ca-
pabilities across domains with similar tasks (Wang
et al., 2023) i.e., if a DNN model is trained on
one domain (source domain) say movie reviews,
then it performs well when it is tested on other do-
mains (target domain) such as electronic product
reviews. However, even when the underlying task
is the same, there is a performance drop due to the
change in the semantic structure of the reviews of
the different domains.

This property of better generalization to vari-
ous domains is also inherited by EEDNNs as they
are extensions of the DNNs. However, note that
EEDNNs highly depend on the exit confidence
values and the threshold is set based on that using
the validation split of the source dataset. However,
the confidence distribution at the exits changes
due to the change in the domain of the dataset.
This change in confidence distribution impacts the
trade-off between accuracy and efficiency. It ne-
cessitates the requirement of either adapting the
threshold value according to the target domain or
forcing the backbone to provide domain-invariant
features to the classifiers such that the confidence
distribution at the exits is not changed. The exist-
ing two types of methods are detailed below.

Threshold-based adaptation: CeeBERT (Ba-
jpai and Hanawal, 2024b) is the first work that
tries to solve the issue of domain adaptation in
EEDNNs by adapting the threshold based on the
unknown domain.
Since during the inference
phase data arrives in an online and unsupervised
manner, hence the problem is to find the optimal
threshold when the data arrives in an online and
unsupervised manner.

CeeBERT models this problem as a multi-
armed bandit setup, where the action set is the set
of thresholds. It defines the reward function as the
combination of the confidence of the classifier and
the latency incurred to get the prediction from the
classifier. The reward function is defined such that
it increases with an increase in confidence and de-
creases with an increase in latency. The objective
is to maximize the reward function which in turn
maximizes confidence over a sample while mini-
mizing the latency incurred. Since the confidence
distribution is unknown and depends on the tar-
get domain, CeeBERT uses the UCB algorithm to
solve the problem of finding the optimal thresh-
old. UCB algorithm uses exploration-exploitation

Feature-based adaptation: Threshold-based
domain adaptation only tunes the threshold based
on the new domain. DAdEE (Bajpai and Hanawal,
2024c) proposes a GAN-based framework to learn
domain-invariant features across all the layers. It
has a three-step procedure: 1) Supervised train-
ing: First a backbone with attached exits is trained
on the source domain with labels that perform
well on the source dataset. 2) Unsupervised do-
main adaptation: In this step, the domain adap-
tation takes place in a GAN-based setup. At ev-
ery layer, DAdEE attaches a discriminator with
a task to discriminate if a feature representation
is from the source domain or target domain. All
the layers have a task to generate representations
such that the discriminator can be fooled and can-
not distinguish between the source and target do-
main. Knowledge distillation is used to reduce
the impact of mode collapse, which is common in
GANs. 3) Inference: Finally, the third step in-
volves performing inference using the same clas-
sifiers as trained on the source domain. Since
the new model now generates representations that
cannot be distinguished between source and target
domain, it justifies the use of similar classifiers.

6 Further Applications

OOD Detection: Early Exit methods have also
been used for OOD detection by Zhou et al.
(2023) where the task is to determine the out-of-
distribution sample where the original backbone
was trained on the in-domain samples. The train-
ing loss is modified and added with a relative loss
that assesses the interdependency between exits.

During inference, the OOD sample is identified
as a sample that has not gained a sufficient num-
ber of votes from the classifiers. A sample is first
passed through the backbone and if the majority
vote of the classifiers reaches a certain threshold
then the sample is early inferred else, it is labeled
as an OOD sample.
Reinforcement

learning:
ZTW (Wo?czyk
et al., 2021) applies the early exit framework to
the Reinforcement Learning algorithm to acceler-
ate their inference time. It implements the idea of
cascaded connections by adding skip connections
that combine the output of mth layer of the model
with (m ? 1)th layer classifier output and passes it
to the mth classifier. This makes the model aware
of the previous classifier�s output and helps the

model to provide more confident results. ZTW ex-
periments with Q*-BERT and Pong, two popular
Atari 2600 environments.

Self-speculative decoding: Speculative decod-
ing is a method used to reduce the latency issues
in autoregressive decoding tasks. In this method,
two models are used, where a smaller draft model
is used to generate the tokens in an autoregressive
manner and then a larger model verifies the output
of the draft model in a non-autoregressive manner
saving lot of computation without losing accuracy.
Recently LayerSkip (Elhoushi et al., 2024) and
Draft & Verify (Zhang et al., 2023) combine early
exits with speculative decoding and name it as
self-speculative decoding. In this setup, the draft
model is replaced by some initial layers of the
large model. The early exit point is attached at a
chosen layer and then the tokens are generated in
an autoregressive manner and the tokens are veri-
fied using the final layer of the model.

Distributed Inference: Early exit (EE) meth-
ods optimize distributed inference across mobile,
edge, and cloud devices by enabling samples to
exit on different devices based on confidence, re-
ducing offloading costs. DDNN (Teerapittayanon
et al., 2017) pioneered this approach, but three key
challenges arise: 1) Optimal partitioning layer:
SplitEE (Bajpai et al., 2023, 2024) address this us-
ing a Multi-Armed Bandit (MAB) framework. 2)
Optimal threshold: UCBEE (Pacheco et al., 2024)
tackles threshold selection as an MAB problem,
optimizing over a predefined set. 3) DNN during
outages: UEEUCB (Hanawal et al., 2022) opti-
mize exit points with MABs, targeting image and
NLP tasks, respectively. DEE (Ju et al., 2021)
enhances robustness in dynamic conditions using
contextual bandits to handle distributional shifts.

7 Future Directions

In this section, we list some of the possible future
research directions.

7.1 Exit placement and size

For smaller models like BERT and ALBERT, exit
classifiers can be placed after every layer due to
the limited number of layers. However, for larger
models such as LLAMA and OPT, this approach
significantly increases parameters. For instance,
adding a classifier to each layer of OPT2.7B, with
a hidden size of 2560 and a vocabulary size V,
results in 130M parameters per classifier. With

32 layers, this totals 4B parameters, exceeding the
model size itself.

Additionally, placing more exits in initial lay-
ers improves efficiency but can lead to higher per-
formance degradation, while exits in deeper lay-
ers reduce performance drops but compromise ef-
ficiency. To balance these trade-offs, exits should
be strategically placed at intervals, as consecutive
layers often yield minimal additional information,
necessitating careful selection of layers for exit at-
tachment.

7.2 Risk in EEDNNs

Similar to DNNs, EEDNNS are also prone to the
risk of getting the wrong prediction. Note that the
EEDNNs are even at more risk as there are multi-
ple classifiers that can get wrong predictions. This
issue is brought up in Fast yet Safe (Jazbec et al.,
2024) paper where they show that the threshold
used for early exiting could also be used to mini-
mize the risk. However, it has very less insights on
if the model gains fake confidence over the wrong
class and gets predicted early. A thoughtful con-
sideration of this issue is necessary.

7.3 Overconfidence

In Figure 2, we plot the average confidence val-
ues of the exit classifiers across the backbone on
the true label of the incoming sample. The dataset
used is the SST-2 dataset with a task of sentiment
classification. We can observe that there are sam-
ples marked as �fake confidence�. These are the
samples where the samples have high confidence
towards the wrong class, this can lead to wrong
prediction at the initial layers. This can affect the
EEDNN accuracy and needs to be addressed.

8 Conclusion

EEDNNs address latency by enabling easier sam-
ples to exit at shallower layers, improving both ef-
ficiency and accuracy by mitigating overthinking.
They also tackle overfitting, vanishing gradients,
and distributed inference challenges. While sig-
nificant progress has been made, ongoing research
focuses on optimizing exit criteria, training meth-
ods, and addressing issues like overconfidence and
prediction errors. This survey highlights key de-
sign challenges to inspire further advancements,
positioning early-exit techniques as essential tools
for future computational systems.

References

Dosovitskiy Alexey. 2020. An image is worth 16x16
words: Transformers for image recognition at scale.
arXiv preprint arXiv: 2010.11929.

Haoli Bai, Wei Zhang, Lu Hou, Lifeng Shang, Jing Jin,
Xin Jiang, Qun Liu, Michael Lyu, and Irwin King.
2020. Binarybert: Pushing the limit of bert quanti-
zation. arXiv preprint arXiv:2012.15701.

Divya J Bajpai, Vivek K Trivedi, Sohan L Yadav, and
Manjesh K Hanawal. 2023. Splitee: Early exit in
deep neural networks with split computing. arXiv
preprint arXiv:2309.09195.

Divya Jyoti Bajpai and Manjesh Kumar Hanawal.
Image captioning with early
arXiv preprint

2024a.
exits and knowledge distillation.
arXiv:2410.04433.

Capeen:

Divya Jyoti Bajpai and Manjesh Kumar Hanawal.
2024b. Ceebert: Cross-domain inference in early
exit bert. In To appear in proceedings of the 62nd
conference of the Association for computational lin-
guistics: Findings Volume.

Divya Jyoti Bajpai and Manjesh Kumar Hanawal.
2024c. Dadee: Unsupervised domain adaptation in
early exit plms. arXiv preprint arXiv:2410.04424.

Divya Jyoti Bajpai, Aastha Jaiswal, and Manjesh Ku-
Image classifica-
mar Hanawal. 2024.
tion in split computing dnns with early exits. arXiv
preprint arXiv:2401.10541.

I-splitee:

Nikita Balagansky and Daniil Gavrilov. 2022. Palbert:
Teaching albert to ponder. Advances in Neural In-
formation Processing Systems, 35:14002�14012.

Ankur Bapna, Naveen Arivazhagan, and Orhan Fi-
rat. 2020. Controlling computation versus qual-
arXiv preprint
ity for neural sequence models.
arXiv:2002.07106.

Joud Chataoui, Mark Coates, et al. 2023.

Jointly-
learned exit and inference for a dynamic neural net-
In The Twelfth International Conference on
work.
Learning Representations.

Xin Dai, Xiangnan Kong, and Tian Guo. 2020. Epnet:
Learning to exit with flexible multi-branch network.
In Proceedings of the 29th ACM International Con-
ference on Information & Knowledge Management,
pages 235�244.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2018. Bert: Pre-training of deep
bidirectional transformers for language understand-
ing. arXiv preprint arXiv:1810.04805.

Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael
arXiv

Auli. 2019. Depth-adaptive transformer.
preprint arXiv:1910.10073.

Mostafa Elhoushi, Akshat Shrivastava, Diana
Liskovich, Basil Hosmer, Bram Wasti, Liangzhen
Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal,
Ahmed Roman, et al. 2024. Layer skip: Enabling
early exit inference and self-speculative decoding.
arXiv preprint arXiv:2404.16710.

Angela Fan, Edouard Grave, and Armand Joulin. 2019.
Reducing transformer depth on demand with struc-
tured dropout. arXiv preprint arXiv:1909.11556.

Biyi Fang, Xiao Zeng, Faen Zhang, Hui Xu, and
Mi Zhang. 2020. Flexdnn: Input-adaptive on-device
In 2020
deep learning for efficient mobile vision.
IEEE/ACM Symposium on Edge Computing (SEC),
pages 84�95. IEEE.

Zhengcong Fei, Xu Yan, Shuhui Wang, and Qi Tian.
2022. Deecap: Dynamic early exiting for efficient
image captioning. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recog-
nition, pages 12216�12226.

Xiangxiang Gao, Yue Liu, Tao Huang, and Zhongyu
Hou. 2023. Pf-berxit: Early exiting for bert with
parameter-efficient fine-tuning and flexible early ex-
iting strategy. Neurocomputing, 558:126690.

Shijie Geng, Peng Gao, Zuohui Fu, and Yongfeng
Zhang. 2021. Romebert: Robust training of multi-
exit bert. arXiv preprint arXiv:2101.09755.

Yizeng Han, Gao Huang, Shiji Song, Le Yang,
Dy-
Honghui Wang, and Yulin Wang. 2021.
IEEE Transac-
namic neural networks: A survey.
tions on Pattern Analysis and Machine Intelligence,
44(11):7436�7456.

Manjesh K Hanawal, Avinash Bhardwaj, et al. 2022.
Unsupervised early exit in dnns with multiple exits.
arXiv preprint arXiv:2209.09480.

Jianing He, Qi Zhang, Hongyun Zhang, Xuanjing
Huang, Usman Naseem, and Duoqian Miao. 2024.
Cosee: Consistency-oriented signal-based early ex-
iting via calibrated sample weighting mechanism.
arXiv preprint arXiv:2412.13236.

Gao Huang, Danlu Chen, Tianhong Li, Felix Wu,
Laurens Van Der Maaten, and Kilian Q Wein-
berger. 2017. Multi-scale dense networks for re-
source efficient image classification. arXiv preprint
arXiv:1703.09844.

Metod Jazbec, Alexander Timans, Tin Had?zi Veljkovi�c,
Kaspar Sakmann, Dan Zhang, Christian A Naes-
Fast yet safe:
seth, and Eric Nalisnick. 2024.
arXiv preprint
Early-exiting with risk control.
arXiv:2405.20915.

Yixin Ji, Jikai Wang, Juntao Li, Qiang Chen, Wenliang
Chen, and Min Zhang. 2023. Early exit with disen-
tangled representation and equiangular tight frame.
In Findings of the Association for Computational
Linguistics: ACL 2023, pages 14128�14142.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang,
Xiao Chen, Linlin Li, Fang Wang, and Qun Liu.
2019. Tinybert: Distilling bert for natural language
understanding. arXiv preprint arXiv:1909.10351.

Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang,
Haotang Deng, and Qi Ju. 2020. Fastbert: a self-
distilling bert with adaptive inference time. arXiv
preprint arXiv:2004.02178.

Weiyu Ju, Wei Bao, Liming Ge, and Dong Yuan. 2021.
Dynamic early exit scheduling for deep neural net-
work inference through contextual bandits. In Pro-
ceedings of the 30th ACM International Conference
on Information & Knowledge Management, pages
823�832.

Yigitcan Kaya, Sanghyun Hong, and Tudor Dumitras.
2019. Shallow-deep networks: Understanding and
In International
mitigating network overthinking.
conference on machine learning, pages 3301�3310.
PMLR.

Hossein KhademSohi, Mohammadamin Abedi, Yani
Ioannou, Steve Drew, Pooyan Jamshidi, and Hadi
Hemmati. Selfxit: An unsupervised early exit mech-
anism for deep neural networks. Transactions on
Machine Learning Research.

Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W
Mahoney, and Kurt Keutzer. 2021. I-bert: Integer-
only bert quantization. In International conference
on machine learning, pages 5506�5518. PMLR.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman,
Kevin Gimpel, Piyush Sharma, and Radu Soricut.
2019. Albert: A lite bert for self-supervised learn-
arXiv preprint
ing of language representations.
arXiv:1909.11942.

Stefanos Laskaridis, Stylianos I Venieris, Mario
Almeida, Ilias Leontiadis, and Nicholas D Lane.
2020. Spinn: synergistic progressive inference of
neural networks over device and cloud. In Proceed-
ings of the 26th annual international conference on
mobile computing and networking, pages 1�15.

En Li, Liekang Zeng, Zhi Zhou, and Xu Chen. 2019a.
Edge ai: On-demand accelerating deep neural net-
work inference via edge computing. IEEE Transac-
tions on Wireless Communications, 19(1):447�457.

Hao Li, Hong Zhang, Xiaojuan Qi, Ruigang Yang, and
Gao Huang. 2019b. Improved techniques for train-
ing adaptive deep networks. In Proceedings of the
IEEE/CVF international conference on computer vi-
sion, pages 1891�1900.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven
Hoi. 2023. Blip-2: Bootstrapping language-image
pre-training with frozen image encoders and large
In International conference on
language models.
machine learning, pages 19730�19742. PMLR.

Kaiyuan Liao, Yi Zhang, Xuancheng Ren, Qi Su,
Xu Sun, and Bin He. 2021. A global past-future
early exit method for accelerating inference of pre-
In Proceedings of the
trained language models.
2021 conference of the north american chapter of
the association for computational linguistics: Hu-
man language technologies, pages 2013�2023.

Xiangyang Liu, Tianxiang Sun, Junliang He, Lingling
Wu, Xinyu Zhang, Hao Jiang, Zhao Cao, Xuanjing
Huang, and Xipeng Qiu. 2021. Towards efficient
NLP: A standard evaluation and A strong baseline.

Yoshitomo Matsubara, Marco Levorato, and Francesco
Restuccia. 2022. Split computing and early exiting
for deep learning applications: Survey and research
challenges. ACM Computing Surveys, 55(5):1�30.

Ruijie Miao, Yihan Yan, Xinshuo Yao, and Tong
Yang. 2024. An efficient inference framework for
arXiv preprint
early-exit large language models.
arXiv:2407.20272.

Paul Michel, Omer Levy, and Graham Neubig. 2019.
Are sixteen heads really better than one? Advances
in neural information processing systems, 32.

Roberto G Pacheco, Divya J Bajpai, Mark Shifrin,
Rodrigo S Couto, Daniel S Menasch�e, Manjesh K
Hanawal, and Miguel Elias M Campista. 2024.
Ucbee: A multi armed bandit approach for early-exit
in neural networks. IEEE Transactions on Network
and Service Management.

Mary Phuong and Christoph H Lampert. 2019.
Distillation-based training for multi-exit architec-
In Proceedings of the IEEE/CVF interna-
tures.
tional conference on computer vision, pages 1355�
1364.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
Dario Amodei, Ilya Sutskever, et al. 2019. Lan-
guage models are unsupervised multitask learners.
OpenAI blog, 1(8):9.

Haseena Rahmath P, Vishal Srivastava, Kuldeep
Chaurasia, Roberto G Pacheco, and Rodrigo S
Couto. 2024.
Early-exit deep neural network-a
comprehensive survey. ACM Computing Surveys,
57(3):1�37.

Victor Sanh, Lysandre Debut, Julien Chaumond, and
Thomas Wolf. 2019. Distilbert, a distilled version
of bert: smaller, faster, cheaper and lighter. arXiv
preprint arXiv:1910.01108.

Tianxiang Sun, Xiangyang Liu, Wei Zhu, Zhichao
Geng, Lingling Wu, Yilong He, Yuan Ni, Guotong
Xie, Xuanjing Huang, and Xipeng Qiu. 2022. A
simple hash-based early exiting approach for lan-
guage understanding and generation. arXiv preprint
arXiv:2203.01670.

Tianxiang Sun, Yunhua Zhou, Xiangyang Liu, Xinyu
Zhang, Hao Jiang, Zhao Cao, Xuanjing Huang, and
Xipeng Qiu. 2021. Early exiting with ensemble in-
ternal classifiers. arXiv preprint arXiv:2105.13792.

Peng Tang, Pengkai Zhu, Tian Li, Srikar Appalaraju,
Vijay Mahadevan, and R Manmatha. 2023a. Deed:
for accelerat-
Dynamic early exit on decoder
arXiv
ing encoder-decoder transformer models.
preprint arXiv:2311.08623.

Shengkun Tang, Yaqing Wang, Caiwen Ding, Yi Liang,
Yao Li, and Dongkuan Xu. 2023b. Deediff: Dy-
namic uncertainty-aware early exiting for acceler-
ating diffusion model generation. arXiv preprint
arXiv:2309.17074.

Shengkun Tang, Yaqing Wang, Zhenglun Kong,
Tianchi Zhang, Yao Li, Caiwen Ding, Yanzhi Wang,
Yi Liang, and Dongkuan Xu. 2023c. You need mul-
tiple exiting: Dynamic early exiting for accelerating
In Proceedings of
unified vision language model.
the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 10781�10791.

Surat Teerapittayanon, Bradley McDanel, and Hsiang-
Tsung Kung. 2016. Branchynet: Fast inference via
In 2016
early exiting from deep neural networks.
23rd International Conference on Pattern Recogni-
tion (ICPR), pages 2464�2469. IEEE.

Surat Teerapittayanon, Bradley McDanel, and Hsiang-
Tsung Kung. 2017. Distributed deep neural net-
works over the cloud, the edge and end devices.
In 2017 IEEE 37th international conference on dis-
tributed computing systems (ICDCS), pages 328�
339. IEEE.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timoth�ee Lacroix,
Baptiste Rozi`ere, Naman Goyal, Eric Hambro,
Faisal Azhar, et al. 2023. Llama: Open and effi-
cient foundation language models. arXiv preprint
arXiv:2302.13971.

Chih-Shuo Tsai, Ying-Hong Chan, and Yao-Chung
Fan. 2022. Hierarchical cache transformer: Dy-
In 2022
namic early exit for language translation.
International Joint Conference on Neural Networks
(IJCNN), pages 1�9. IEEE.

Meiqi Wang, Jianqiao Mo, Jun Lin, Zhongfeng Wang,
and Li Du. 2019. Dynexit: A dynamic early-exit
strategy for deep residual networks. In 2019 IEEE
International Workshop on Signal Processing Sys-
tems (SiPS), pages 178�183. IEEE.

Yue Wang, Lijun Wu, Juntao Li, Xiaobo Liang, and
Min Zhang. 2023. Are the bert family zero-shot
learners? a study on their potential and limitations.
Artificial Intelligence, page 103953.

Zizhao Wang, Wei Bao, Dong Yuan, Liming Ge,
Nguyen H Tran, and Albert Zomaya. 2020. Acceler-
ating on-device dnn inference during service outage
through scheduling early exit. Computer Communi-
cations, 162:69�82.

Maciej Wo?czyk, Bartosz W�ojcik, Klaudia Ba?azy,
Igor T Podolak, Jacek Tabor, Marek �Smieja, and

Tomasz Trzcinski. 2021. Zero time waste: Recy-
cling predictions in early exit neural networks. Ad-
vances in Neural Information Processing Systems,
34:2516�2528.

Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and
Jimmy Lin. 2020. Deebert: Dynamic early exit-
ing for accelerating bert inference. arXiv preprint
arXiv:2004.12993.

Ji Xin, Raphael Tang, Yaoliang Yu, and Jimmy Lin.
2021. Berxit: Early exiting for bert with better fine-
tuning and extension to regression. In Proceedings
of the 16th conference of the European chapter of
the association for computational linguistics: Main
Volume, pages 91�104.

Canwen Xu and Julian McAuley. 2022. A survey on
dynamic neural networks for natural language pro-
cessing. arXiv preprint arXiv:2202.07101.

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Car-
bonell, Russ R Salakhutdinov, and Quoc V Le. 2019.
Xlnet: Generalized autoregressive pretraining for
language understanding. Advances in neural infor-
mation processing systems, 32.

Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen,
Gang Chen, and Sharad Mehrotra. 2023. Draft
& verify: Lossless large language model accelera-
tion via self-speculative decoding. arXiv preprint
arXiv:2309.08168.

Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao
Chen, Xin Jiang, and Qun Liu. 2020. Ternarybert:
Distillation-aware ultra-low bit bert. arXiv preprint
arXiv:2009.12812.

Zhen Zhang, Wei Zhu, Jinfan Zhang, Peng Wang, Rize
Jin, and Tae-Sun Chung. 2022. Pcee-bert: Acceler-
ating bert inference via patient and confident early
exiting. In Findings of the Association for Compu-
tational Linguistics: NAACL 2022, pages 327�338.

Wangchunshu Zhou, Canwen Xu, Tao Ge, Julian
McAuley, Ke Xu, and Furu Wei. 2020. Bert loses
patience: Fast and robust inference with early exit.
Advances in Neural Information Processing Sys-
tems, 33:18330�18341.

Yunhua Zhou, Jianqiang Yang, Pengyu Wang, and
Xipeng Qiu. 2023. Two birds one stone: Dynamic
ensemble for ood intent classification. In Proceed-
ings of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 10659�10673.

Wei Zhu. 2021. Leebert: Learned early exit for bert
with cross-level optimization. In Proceedings of the
59th Annual Meeting of the Association for Compu-
tational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Vol-
ume 1: Long Papers), pages 2968�2980.


