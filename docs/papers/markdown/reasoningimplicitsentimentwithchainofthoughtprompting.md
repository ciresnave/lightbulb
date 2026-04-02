Reasoning Implicit Sentiment with Chain-of-Thought Prompting?

Hao Fei1, Bobo Li2, Qian Liu3, Lidong Bing4, Fei Li2�, Tat-Seng Chua1
1Sea-NExT Joint Lab, School of Computing, National University of Singapore
2Key Laboratory of Aerospace Information Security and Trusted Computing,
Ministry of Education, School of Cyber Science and Engineering, Wuhan University

3Sea AI Lab,

4DAMO Academy, Alibaba Group

haofei37@nus.edu.sg, boboli@whu.edu.cn, liuqian@sea.com,
l.bing@alibaba-inc.com, lifei_csnlp@whu.edu.cn, dcscts@nus.edu.sg

3
2
0
2

n
u
J

9

]
L
C
.
s
c
[

4
v
5
5
2
1
1
.
5
0
3
2
:
v
i
X
r
a

Abstract

While sentiment analysis systems try to deter-
mine the sentiment polarities of given targets
based on the key opinion expressions in input
texts, in implicit sentiment analysis (ISA) the
opinion cues come in an implicit and obscure
manner. Thus detecting implicit sentiment re-
quires the common-sense and multi-hop rea-
soning ability to infer the latent intent of opin-
ion. Inspired by the recent chain-of-thought
(CoT) idea, in this work we introduce a Three-
hop Reasoning (THOR) CoT framework to
mimic the human-like reasoning process for
ISA. We design a three-step prompting princi-
ple for THOR to step-by-step induce the im-
plicit aspect, opinion, and finally the sentiment
polarity. Our THOR+Flan-T5 (11B) pushes the
state-of-the-art (SoTA) by over 6% F1 on su-
pervised setup. More strikingly, THOR+GPT3
(175B) boosts the SoTA by over 50% F1 on
zero-shot setting. Our code is open at https:
//github.com/scofield7419/THOR-ISA.

Introduction

1
Sentiment analysis (SA) aims to detect the sen-
timent polarity towards a given target based on
the input text. SA can be classified into explicit
SA (ESA) and implicit SA (ISA), where the for-
mer type is the current mainstream task, in which
the emotional expressions explicitly occur in texts
(Pontiki et al., 2014). Different from ESA, ISA
is much more challenging, because in ISA the
inputs contain only factual descriptions with no
explicit opinion expression directly given (Russo
et al., 2015). For example, given a text �Try the tan-
doori salmon!�, having no salient cue word, almost
all existing sentiment classifier1 predicts a neutral
polarity towards �the tandoori salmon�. Human can
easily determine the sentiment states accurately, be-
cause we always grasp the real intent or opinion

?The work is substantially supported by Alibaba Group

through the Alibaba Innovative Research (AIR) Program.

� Corresponding author: Fei Li.
1We pre-experiment with total 20 existing SA models.

Figure 1: Detecting the explicit and implicit sentiment
polarities towards targets. Explicit opinion expression
helps direct inference, while detecting implicit senti-
ment requires common-sense and multi-hop reasoning.

behind the texts. Thus, without truly understand-
ing how the sentiment is aroused, traditional SA
methods are ineffective to ISA.

In fact, it is critical to first discover the hidden
opinion contexts to achieve accurate ISA. For the
explicit case#1 in Fig. 1, it is effortless to capture
the overall sentiment picture (e.g., �environment�
is the aspect, �great� is the opinion), and thus can
precisely infer the positive polarity towards the
given target hotel. Inspired by such fine-grained
sentiment spirit (Xue and Li, 2018; Zhang et al.,
2021; Xu et al., 2020), we consider mining the
implicit aspect and opinion states. For the implicit
case#2 in Fig. 1, if a model can first infer the
key sentiment components, e.g., the latent aspect
�taste�, latent opinion �good and worth trying�, the
inference of final polarity can be greatly eased. To
reach the goal, the capabilities of common-sense
reasoning (i.e., infer what is �tandoori salmon�)
and multi-hop reasoning (i.e., infer the aspect and
then the opinion) are indispensable.

Fortunately, the recent great triumph of pre-
trained large-scale language models (LLMs) of-
fers a promising solution. On the one hand, LLMs
have been found to carry very rich world knowl-
edge, showing extraordinary ability on common-
sense understanding (Paranjape et al., 2021; Liu
et al., 2022). On the other hand, the latest chain-of-

The environment of the hotel is so great ! �Explicit SentimentpositiveTry the tandoori salmon !Case#1: Case#2: �Implicit SentimentReasoning the underlying intent/contextTandoori salmon is a dish made with salmon. By saying this, the speaker is recommending the tandoori salmon, mostly because he or she believes the taste of tandoori salmon is good and worth trying. Thus the polarity of tandoori salmon is positive.positiveMulti-hop reasoningCommon-sense reasoning

Figure 2: An illustration of our THOR framework for three-hop reasoning of implicit sentiment.

thought (CoT) idea has revealed the great potential
of LMs� multi-hop reasoning (Wei et al., 2022;
Zhou et al., 2022; Zhang et al., 2023), where an
LLM with some prompts can do chain-style reason-
ing impressively. Built on top of all these successes,
in this work we implement a Three-hop Reasoning
CoT framework (namely THOR) for ISA. Based on
an LLM, we design three prompts for three steps of
reasoning, each of which respectively infers 1) the
fine-grained aspect of the given target, 2) the under-
lying opinion towards the aspect, and 3) the final
polarity. With such easy-to-hard incremental rea-
soning, the hidden contexts of the overall sentiment
picture can be elicited step by step to achieve an
easier prediction of final polarity, which effectively
alleviates the difficulties of the task prediction.

To ensure the correctness of each reasoning step,
we consider a self-consistency mechanism for CoT
inspired by Wang et al. (2022b), which is to select
the candidate answers (at each step) with high vot-
ing consistency of inferred aspect and opinion. For
supervised fine-tuning setup, we further propose a
reasoning revising method. We use the intermedi-
ate reasoning answers as model inputs to predict the
final labels, where the supervision from gold labels
will teach LLM to generate more correct reason-
ing. On supervised fine-tuning setup, our Flan-T5
based THOR improves the current best-performing
baseline by more than 6% in F1 score, and such
margins are further magnified on zero-shot setup.
Most strikingly, our GPT3-based THOR with 175B
parameters boosts the baseline to a high-to 51.10%
increase of F1 score.

To sum up, this work contributes a multi-hop

reasoning solution for implicit sentiment detection,
which helps to achieve impressive improvement
over the traditional non-reasoning methods. To our
knowledge, this is the first attempt to successfully
extend the CoT idea to the sentiment analysis com-
munity. Our method is simple yet effective, and can
be broadly applied to other similar NLP problems
without much effort.

2 Three-hop Reasoning Framework

The task of SA (either ESA or ISA) is defined
as: given a sentence X with a target term t ?
X, a model determines the sentiment polarity y
towards t, i.e., positive, neutral or negative. We
solve the task using an off-the-shelf LLM with
prompt. For the standard prompt-based method,
we can construct the following prompt template as
LLM�s input:

Given the sentence X, what is the
sentiment polarity towards t?

The LLM should return the
�y=argmaxp(y|X, t).

answer via:

2.1 Chain-of-Thought Prompting

Now we consider the CoT-style prompt (Wei et al.,
2022; Fu et al., 2022) method for multi-step rea-
soning. Instead of directly asking LLM the final
result of y, in our THOR (cf. Fig. 2) we hope the
LLM infer the latent aspect and opinion informa-
tion before answering the finale y. We here define
the intermediate aspect term a and latent opinion
expression o. We construct the three-hop prompts
as follows.

Given the sentence "The new mobile phone can be just put in my pocket.", what is the sentiment polarity towards the new mobile phone?? Traditional Prompting:TargetThe new mobile phone can be just put in my pocket.The sentiment polarity of the given sentence is neutral. The sentence simply states that the new mobile phone can be put in the speaker's pocket and does not express any positive or negative sentiment towards the phone.? Three-hop Reasoning with CoT PromptingGiven the sentence "The new mobile phone can be just put in my pocket.", which specific aspect of the new mobile phone is possibly mentioned?Given the sentence, "The new mobile phone can be just put in my pocket.", the specific aspect of the new mobile phone mentioned in the sentence is the size or portability. The specific aspect of the new mobile phone mentioned in the sentence is the size or portability.PolaritypositiveAspectOpinionBased on the mentioned aspect of size and portability, the sentence implies that the phone is small enough to fit in the speaker's pocket. According to common sense, the implicit opinion of speaker towards the portability is good, because the speaker is able to easily carry the phone with them by placing it in their pocket, and find the phone to be convenient and easy to use.The sentiment polarity towards the new mobile phone based on the given sentence is positive. The speaker finds the phone to be convenient and easy to use, implying having a favorable view of the phone.1st-hop:Appending to the next contextInferring aspect2nd-hop:Inferring opinion3rd-hop:Inferring polarityAppending to the next contextInputOutputGiven the sentence, "The new mobile phone can be just put in my pocket.",Based on the common sense, what is the implicit opinion towards the mentioned aspect of the new mobile phone, and why?     pocket. According to common sense, the implicit opinion of speaker towards the portability is good, because the speaker can easily carry the phone with them by placing it in their pocket, and find the phone to be convenient and easy to use.Based on such opinion, what is the sentiment polarity Given the sentence, "The new mobile phone can be just put in my pocket.", the specific aspect of the new mobile phone mentioned in the sentence is the size or portability. Given the sentence, "The new mobile phone can be just put in my pocket.",Based on the mentioned aspect of size and portability, the sentence implies that the phone is small enough to fit in thespeaker's polarity towards the new mobile phone? Reasoning question in prompt:Aspect answer (A)Opinion answer (O)Step 1. We first ask LLM what aspect a is men-
tioned with the following template:

C1[Given sentence X], which specific
aspect of t is possibly mentioned?
C1 is the first-hop prompt context. This step can be
formulated as A=argmaxp(a|X, t), where A is the
output text which explicitly mentions the aspect a.
Step 2. Now based on X, t and a, we ask LLM
to answer in detail what would be the underlying
opinion o towards the mentioned aspect a:

C2[C1,A]. Based on the common sense,
what is the implicit opinion towards
the mentioned aspect of t, and why?
C2 is the second-hop prompt context which con-
catenates C1 and A. This step can be written as
O=argmaxp(o|X, t, a), where O is the answer text
containing the possible opinion expression o.

Step 3. With the complete sentiment skeleton (X,
t, a and o) as context, we finally ask LLM to infer
the final answer of polarity t:

C3[C2,O]. Based on the opinion, what
is the sentiment polarity towards t?
C3 is the third-hop prompt context. We note this
step as �y=argmaxp(y|X, t, a, o).

2.2 Enhancing Reasoning via Self-consistency
We further leverage the self-consistency mecha-
nism (Wang et al., 2022b; Li et al., 2022b) to con-
solidate the reasoning correctness. Specifically, for
each of three reasoning steps, we set the LLM de-
coder to generate multiple answers, each of which
will likely to give varied predictions of aspect a,
opinion o as well as the polarity y. At each step,
those answers with high voting consistency of in-
ferred a, o or y are kept. We select the one with
highest confidence as the context in next step.

2.3 Reasoning Revising with Supervision
We can also fine-tune our THOR when the on-
demand training set is available, i.e., supervised
fine-tuning setup. We devise a reasoning revising
method. Technically, at each step we construct a
prompt by concatenating 1) initial context, 2) this
step�s reasoning answer text and 3) final question,
and feed it into LLM to predict the sentiment label
instead of going to the next step reasoning. For ex-
ample, at end of step-1, we can assemble a prompt:
[C1,A, �what is the sentiment polarity towards t?�].
In the supervision of gold labels, the LLM will
be taught to generate more correct intermediate
reasoning that is helpful to the final prediction.

Restaurant

Laptop

All

ISA

All

ISA

� State-of-the-art baselines
BERT+SPC� (110M)
77.16 65.54 73.45 69.54
BERT+ADA� (110M)
80.05 65.92 74.18 70.11
BERT+RGAT� (110M)
81.35 67.79 74.07 72.99
BERTAsp+CEPT� (110M)
82.07 67.79 78.38 75.86
BERT+ISAIV� (110M)
81.40 69.66 77.25 78.29
BERTAsp+SCAPT� (110M) 83.79 72.28 79.15 77.59
� Prompt-based methods
BERT+Prompt (110M)
Flan-T5+Prompt (250M)
Flan-T5+Prompt (11B)
� CoT-based methods
Flan-T5+THOR (250M)
Flan-T5+THOR (11B)
w/o SelfConsistency
w/o Reason-Revising

82.98 71.70 79.75 67.63
87.45 79.73 85.16 82.43
86.03 77.68 84.39 80.27
86.88 78.42 84.83 81.69

81.34 70.12 78.58 75.24
81.50 70.91 79.02 76.40
84.72 75.10 82.44 78.91

Table 1: F1 results on supervised fine-tuning setup. Best
results are marked in bold. Scores by model with � are
copied from Li et al. (2021).

3 Experiments

Setups We experiment on the benchmark Se-
mEval14 Laptop and Restaurant datasets (Pontiki
et al., 2014), where all the instances are split into
explicit and implicit sentiment by Li et al. (2021).
Since the encoder-style BERT cannot generate texts
to support CoT, we use encoder-decoder style Flan-
T52 as our backbone LLM. We also test with GPT3
(Brown et al., 2020) and ChatGPT (Ouyang et al.,
2022). We used four versions of Flan-T5: 250M
(base), 780M (large), 3B (xl) and 11B (xxl), and
four versions of GPT3: 350M, 1.3B, 6.7B and
175B. Note that GPT3 does not release the model
parameters, and we use it in the prompting man-
ner via the API3. This also means that we cannot
perform supervised fine-tuning with GPT3. We
compare with the current best-performing base-
lines, including: BERT+SPC (Devlin et al., 2019),
BERT+ADA (Rietzler et al., 2020), BERT+RGAT
(Wang et al., 2020), BERTAsp+CEPT (Li et al.,
2021), BERT+ISAIV (Wang et al., 2022a) and
BERTAsp+SCAPT (Li et al., 2021). We consider
both the supervised fine-tuning and zero-shot se-
tups. We adopt the F1 as the evaluation metric.
On the few-shot setup, we re-implement the base-
lines via their source codes. Our experiments are
conducted with 4 NVIDIA A100 GPUs.

2https://huggingface.co/docs/transformers/

model_doc/flan-t5

3https://beta.openai.com/docs/models/gpt-3

Restaurant

Laptop

All

ISA All

ISA

� State-of-the-art baselines
BERT+SPC (110M)
21.76 19.48 25.34 17.71
27.48 22.04 25.68 18.26
BERT+RGAT (110M)
BERTAsp+SCAPT (110M) 30.02 25.49 25.77 13.70
� Prompt-based methods
BERT+Prompt (110M)
Flan-T5+Prompt (250M)
Flan-T5+Prompt (11B)
� CoT-based methods
Flan-T5+THOR (250M)
Flan-T5+THOR (3B)
Flan-T5+THOR (11B)
Flan-T5+ZeroCoT (11B)
GPT3+THOR (175B)

55.86 41.84 52.52 32.40
57.33 42.61 56.36 38.16
61.87 52.76 58.27 40.75
56.58 47.41 55.53 35.67
81.96 76.55 76.04 73.12

33.62 31.46 35.17 22.86
54.38 41.57 52.06 31.43
57.12 45.31 54.14 33.71

Figure 3: Influences of LLM scales.

Table 2: Model results on Zero-shot setting. We re-
implement the state-of-the-art baselines for the zero-
shot performance. �ZeroCoT� means prompting LLM
with the zero-shot CoT, �let�s think step by step� (Brown
et al., 2020).

Results on Supervised Fine-tuning The com-
parisons are shown in Table 1.
It is interesting
to see that the BERT with prompt learning under-
performs the SoTA baseline BERTAsp+SCAPT.
Even the Flan-T5-base (250M) with double-size pa-
rameters fails to beat the SoTA. BERTAsp+SCAPT
is pre-trained on the large-scale sentiment aspect-
aware annotation data, thus showing strong capa-
bility on SA. But with our THOR CoT prompting,
Flan-T5-base clearly outperforms SoTA. Further,
when using the larger LLM, i.e., with 11B param-
eters, we can find the vanilla prompt-based Flan-
T5 surpasses the best baseline. More prominently,
Flan-T5-11B with THOR shows significant boosts
for ISA, i.e., 7.45%(=79.73-72.28) on Restaurant
and 5.84%(=82.43-77.59) on Laptop, with average
improvement of 6.65%(7.45+5.84)/2 F1. Also the
ablations of the self-consistency and reasoning re-
vising mechanisms indicate their importances in
our THOR method.

Results on Zero-shot Reasoning In Table 2 we
compare the zero-shot performances. We can find
that the improvement of both prompt-based and
CoT-based methods over the current SoTA base-
line increases dramatically. But overall, the CoT-
based methods with our THOR show much more
significant improvement on ISA. For example, our
Flan-T5-11B THOR system gives over 30% F1 av-
erage improvement over the best-performing base-

Figure 4: Comparisons between GPT3&ChatGPT on
randomly-selected 50 ESA and 50 ISA instances.

line (BERTAsp+SCAPT) on two datasets. Most
strikingly, when THOR is equipped into super-
large LLM, i.e., GPT3-175B, we can observe
the impressive improvement, near to the level
by Flan-T5-11B THOR in supervised setting as
in Table 1. Specifically, it boosts the SoTA re-
sults by 51.94%(=81.96-30.02) on Restaurant and
50.27%(=76.04-25.77) on Laptop, with an average
51.10%(51.94+50.27)/2 F1 leap.

Influence of Different Model Sizes of LLMs
In
Table 1 and 2 we have witnessed the power by using
(very) large LLMs. In Fig. 3 we study the influ-
ence of different LLM scales. We see that with the
increasing model scale, the efficacy of our multi-
hop reasoning prompting is exponentially ampli-
fied. This coincides much with the existing findings
of CoT prompting methods (Wei et al., 2022; Zhou
et al., 2022; Fu et al., 2022), i.e., the larger the LMs,
the more significant improvement by CoT. Because
when the LLM is sufficiently large, the capabili-
ties on common-sense and multi-hop reasoning are
greatly developed and strengthened.

Improving ChatGPT with THOR The latest
birth of ChatGPT has brought revolutionary ad-
vancement in NLP and AI community. Here we
compare the improvement of our THOR on GPT3
(175B) and ChatGPT, respectively. In Fig. 4 we
show the testing results on 100 testing instances.
We can see that both LMs shows very high perfor-
mances on ESA, and the enhancements by THOR
are very limited. But prompting-based GPT3 and
ChatGPT still fail much on ISA, where our THOR

250M780M3B11B304356350M1.3B6.7B175B35506580T5F1onISAGPT3+Prompt(Rest)+THOR(Rest)+Prompt(Lap)+THOR(Lap)ESAISA758595ESAISA758595RestaurantF1LaptopGPT3+PromptGPT3+THORChatGPT+PromptChatGPT+THORcess, inferring the sentiment elements step by step
and finally understand the sentiment polarity in an
easy-to-hard manner.

Language model pre-training has received in-
creasing research attention for enhancing the utility
of downstream applications (Raffel et al., 2020)
Most recently, the large-scale language models
(LLMs) have shown great potential to the human-
level intelligence, e.g., ChatGPT (Ouyang et al.,
2022). LLMs have extensively demonstrated to
exhibit extraordinary abilities on common-sense
understanding (Paranjape et al., 2021; Liu et al.,
2022) and multi-hop reasoning (Wei et al., 2022;
Zhou et al., 2022). This work implements the im-
plicit sentiment reasoning built upon LMs, based
on the latest proposed chain-of-thought (CoT) idea.
CoT prompting is a gradient-free technique that
induces large LMs to produce intermediate reason-
ing steps leading to the final answer. Wei et al.
(2022) formally study the CoT prompting in lan-
guage models, in which they elicit LMs to generate
coherent series of intermediate reasoning steps that
direct to the final answer to the original question.

5 Conclusion

In this paper, we present a Three-hop Reason-
ing prompting framework to achieve the chain-of-
thought reasoning process for implicit sentiment
analysis. Based on the existing LLM, we design
three prompts for three steps of reasoning, each of
which respectively infers the fine-grained aspect,
the underlying opinion and the final polarity. On
the ISA datasets, different LLMs equipped with
our THOR show impressive performances over the
existing best-performing baselines on both the su-
pervised and zero-shot setups. We show that the
larger the LLMs, the more significant improvement
by our THOR method.

Acknowledgments

The work is also partially supported by the National
Key Research and Development Program of China
(No. 2022YFB3103602) and the Sea-NExT Joint
Lab at National University of Singapore.

Limitations

THOR helps unleash the full power of LLMs only
when being integrated into the large enough mod-
els, while on the middle or lower size LLMs, the
improvement by THOR will be limited to certain
extent, due to the emergence nature of LLMs.

Figure 5: Error analysis.

has improved them on ISA very considerably.

Failure Analysis
In Fig. 5 we show the error
rates of failure cases when using THOR, where
we summarize three error types. The Flan-T5-11B
LLM gives 48.27% error rate on zero-shot setup,
while it goes down to 12.79% when fine-tuned with
supervision. Unsupervised-GPT3 (175B) gives
similarity low error rate as with Supervised-T5,
while the latter fails much frequently on incapa-
bility of reasoning. In contrast to Supervised-T5,
the majority of failures in Unsupervised-GPT3
comes from the problematic data annotation. Since
Supervised-T5 is fine-tuned with supervision of
�false� labels, it may actually learn the spurious
correlations but with higher testing accuracy.

4 Related Work

Sentiment analysis has long been a hot research
topic in NLP community (Pang and Lee, 2007;
Dong et al., 2014; Shi et al., 2022). While the ex-
plicit SA models can make predictions based on
the opinion expressions effortlessly, the implicit
SA can be much more tricky due to the hidden
opinion characteristics (Li et al., 2021; Wang et al.,
2022a). And ISA is often more ubiquitous in real-
istic scenarios. Although efforts have been made to
ISA (Li et al., 2021; Wang et al., 2022a), existing
work can still be limited to the traditional paradigm
of inference. As aforementioned, ISA should be
addressed via reasoning, i.e., common-sense and
multi-hop reasoning. Thus, this work follows such
intuition, targeting solving ISA with a multi-hop
reasoning mechanism.

As a key branch of SA, the fine-grained SA has
been well explored (Wang et al., 2017; Li et al.,
2018, 2022a). The idea of fine-grained SA is to
break down the SA into several key sentiment el-
ements, including target, aspect, opinion and sen-
timent polarity, all of which together form a com-
plete sentiment picture in detail (Peng et al., 2020;
Fei et al., 2022). This work draws the same spirit
of fine-grained SA. We believe the reasoning of
implicit sentiment should be an incremental pro-

Errorrate(%)0102030405048.2714.6812.79Annotation errorReasoning failureUnsolvable instanceSup. T5 (11B)Unsup. GPT3 (175B)Unsup. T5 (11B)References

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, Christopher Hesse, Mark Chen, Eric
Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess,
Jack Clark, Christopher Berner, Sam McCandlish,
Alec Radford, Ilya Sutskever, and Dario Amodei.
2020. Language models are few-shot learners. In
Proceedings of the Annual Conference on Neural
Information Processing Systems.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training of
deep bidirectional transformers for language under-
standing. In Proceedings of the 2019 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, Volume 1 (Long and Short Papers), pages
4171�4186.

Li Dong, Furu Wei, Chuanqi Tan, Duyu Tang, Ming
Zhou, and Ke Xu. 2014. Adaptive recursive neu-
ral network for target-dependent Twitter sentiment
classification. In Proceedings of ACL, pages 49�54.

Hao Fei, Fei Li, Chenliang Li, Shengqiong Wu, Jingye
Li, and Donghong Ji. 2022. Inheriting the wisdom
of predecessors: A multiplex cascade framework for
unified aspect-based sentiment analysis. In Proceed-
ings of the Thirty-First International Joint Confer-
ence on Artificial Intelligence, IJCAI, pages 4096�
4103.

Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and
Tushar Khot. 2022. Complexity-based prompting for
multi-step reasoning. CoRR, abs/2210.00720.

Bobo Li, Hao Fei, Fei Li, Yuhan Wu, Jinsong Zhang,
Shengqiong Wu, Jingye Li, Yijiang Liu, Lizi Liao,
Tat-Seng Chua, and Donghong Ji. 2022a. Diaasq : A
benchmark of conversational aspect-based sentiment
quadruple analysis. CoRR, abs/2211.05705.

Xin Li, Lidong Bing, Wai Lam, and Bei Shi. 2018.
Transformation networks for target-oriented senti-
ment classification. In Proceedings of ACL, pages
946�956.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen,
Jian-Guang Lou, and Weizhu Chen. 2022b. On the
advance of making language models better reasoners.
CoRR, abs/2206.02336.

Jiacheng Liu, Alisa Liu, Ximing Lu, Sean Welleck, Pe-
ter West, Ronan Le Bras, Yejin Choi, and Hannaneh
Hajishirzi. 2022. Generated knowledge prompting
for commonsense reasoning. In Proceedings of the
60th Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
3154�3169.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Car-
roll L Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
2022.
Training language models to follow in-
structions with human feedback. arXiv preprint
arXiv:2203.02155.

Bo Pang and Lillian Lee. 2007. Opinion mining and
sentiment analysis. Foundations and Trends in Infor-
mation Retrieval, 2(1-2):1�135.

Bhargavi Paranjape,

Julian Michael, Marjan
Ghazvininejad, Hannaneh Hajishirzi, and Luke
Zettlemoyer. 2021. Prompting contrastive explana-
tions for commonsense reasoning tasks. In Findings
of the Association for Computational Linguistics:
ACL-IJCNLP 2021, pages 4179�4192.

Haiyun Peng, Lu Xu, Lidong Bing, Fei Huang, Wei Lu,
and Luo Si. 2020. Knowing what, how and why: A
near complete solution for aspect-based sentiment
analysis. In Proceedings of the AAAI Conference on
Artificial Intelligence, pages 8600�8607.

Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Har-
ris Papageorgiou, Ion Androutsopoulos, and Suresh
Manandhar. 2014. SemEval-2014 task 4: Aspect
based sentiment analysis. In Proceedings of the 8th
International Workshop on Semantic Evaluation (Se-
mEval 2014), pages 27�35.

Colin Raffel, Noam Shazeer, Adam Roberts, Kather-
ine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the
limits of transfer learning with a unified text-to-text
transformer. Journal of Machine Learning Research,
21:140:1�140:67.

Alexander Rietzler, Sebastian Stabinger, Paul Opitz,
and Stefan Engl. 2020. Adapt or get left behind:
Domain adaptation through BERT language model
finetuning for aspect-target sentiment classification.
In Proceedings of the Twelfth Language Resources
and Evaluation Conference, pages 4933�4941.

Irene Russo, Tommaso Caselli, and Carlo Strapparava.
2015. SemEval-2015 task 9: CLIPEval implicit po-
larity of events. In Proceedings of the 9th Interna-
tional Workshop on Semantic Evaluation (SemEval
2015), pages 443�450.

Zhengyan Li, Yicheng Zou, Chong Zhang, Qi Zhang,
and Zhongyu Wei. 2021. Learning implicit sentiment
in aspect-based sentiment analysis with supervised
contrastive pre-training. In Proceedings of the 2021
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 246�256.

Wenxuan Shi, Fei Li, Jingye Li, Hao Fei, and Donghong
Ji. 2022. Effective token graph modeling using a
novel labeling strategy for structured sentiment anal-
ysis. In Proceedings of the 60th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 4232�4241.

A Appendix

Here we present several pieces of real testing exam-
ples. We compare THOR with the vanilla prompt-
ing method, and the zero-shot CoT method (Prompt
+ �Lets think step by step�). We perform the com-
parisons based on the ChatGPT.4

Kai Wang, Weizhou Shen, Yunyi Yang, Xiaojun Quan,
and Rui Wang. 2020. Relational graph attention net-
work for aspect-based sentiment analysis. In Pro-
ceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 3229�
3238.

Siyin Wang, Jie Zhou, Changzhi Sun, Junjie Ye, Tao
Gui, Qi Zhang, and Xuanjing Huang. 2022a. Causal
intervention improves implicit sentiment analysis. In
Proceedings of the 29th International Conference on
Computational Linguistics, pages 6966�6977.

Wenya Wang, Sinno Jialin Pan, Daniel Dahlmeier, and
Xiaokui Xiao. 2017. Coupled multi-layer attentions
for co-extraction of aspect and opinion terms.
In
Proceedings of the AAAI Conference on Artificial
Intelligence, pages 3316�3322.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V.
Le, Ed H. Chi, and Denny Zhou. 2022b. Self-
consistency improves chain of thought reasoning in
language models. CoRR, abs/2203.11171.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. 2022.
Chain of thought prompting elicits reasoning in large
language models. CoRR, abs/2201.11903.

Lu Xu, Hao Li, Wei Lu, and Lidong Bing. 2020.
Position-aware tagging for aspect sentiment triplet
extraction. In Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing
(EMNLP), pages 2339�2349.

Wei Xue and Tao Li. 2018. Aspect based sentiment
analysis with gated convolutional networks. In Pro-
ceedings of the 56th Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 2514�2523.

Wenxuan Zhang, Yang Deng, Xin Li, Yifei Yuan, Li-
dong Bing, and Wai Lam. 2021. Aspect sentiment
quad prediction as paraphrase generation. In Pro-
ceedings of the 2021 Conference on Empirical Meth-
ods in Natural Language Processing, pages 9209�
9219.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex
Smola. 2023. Automatic chain of thought prompting
in large language models. In The Eleventh Interna-
tional Conference on Learning Representations.

Denny Zhou, Nathanael Sch�rli, Le Hou, Jason Wei,
Nathan Scales, Xuezhi Wang, Dale Schuurmans,
Olivier Bousquet, Quoc Le, and Ed H. Chi. 2022.
Least-to-most prompting enables complex reasoning
in large language models. CoRR, abs/2205.10625.

4https://chat.openai.com/, Dec. 15, 2022

Figure 6: Vanilla prompt-based result for testing case-I.

Figure 7: Result by zero-shot CoT method for testing case-I.

� Case-I

Input text:

� Case-II

Input text:

I just need to walk downstairs to get to
the metro station as it is below the hotel
I�m living in.

The gold sentiment label is positive towards the
metro station.

In Fig. 6, 7 and 8, we show that our THOR
successfully induces the ChatGPT to finally give a
correct decision on sentiment polarity, where the
other two methods fail.

Lunch came with pickels and slaw, no
extra charge.

The gold sentiment label is positive towards Lunch.
Fig. 9, 10 and 11 shows the results and the
LLM�s response, respectively. Our THOR induces
the ChatGPT to draw a correct decision on senti-
ment polarity, but the other two methods still fail.

Figure 8: Result by our THOR method for testing case-I.

Figure 9: Vanilla prompt-based result for testing case-II.

Figure 10: Result by zero-shot CoT method for testing case-II.

Figure 11: Result by our THOR method for testing case-II.


