Reasoning Over Paragraph Effects in Situations

Kevin Lin, Oyvind Tafjord, Peter Clark, and Matt Gardner
Allen Institute for Arti?cial Intelligence
{kevinl,oyvindt,peterc,mattg}@allenai.org

9
1
0
2
c
e
D
5
1

]
L
C
.
s
c
[

2
v
2
5
8
5
0
.
8
0
9
1
:
v
i
X
r
a

Abstract

A key component of successfully reading a
passage of text is the ability to apply knowl-
edge gained from the passage to a new situa-
tion. In order to facilitate progress on this kind
of reading, we present ROPES, a challeng-
ing benchmark for reading comprehension tar-
geting Reasoning Over Paragraph Effects in
Situations. We target expository language de-
scribing causes and effects (e.g., �animal pol-
linators increase ef?ciency of fertilization in
?owers�), as they have clear implications for
new situations. A system is presented a back-
ground passage containing at least one of these
relations, a novel situation that uses this back-
ground, and questions that require reasoning
about effects of the relationships in the back-
ground passage in the context of the situation.
We collect background passages from science
textbooks and Wikipedia that contain such phe-
nomena, and ask crowd workers to author sit-
uations, questions, and answers, resulting in
a 14,322 question dataset. We analyze the
challenges of this task and evaluate the perfor-
mance of state-of-the-art reading comprehen-
sion models. The best model performs only
slightly better than randomly guessing an an-
swer of the correct type, at 61.6% F1, well be-
low the human performance of 89.0%.

Background: Scientists think that the earliest ?owers
attracted insects and other animals, which spread
pollen from ?ower to ?ower. This greatly increased
the ef?ciency of
fertilization over wind-spread
pollen, which might or might not actually land on
another ?ower. To take better advantage of this
animal labor, plants evolved traits such as brightly
colored petals to attract pollinators. In exchange for
pollination, ?owers gave the pollinators nectar.

Situation: Last week, John visited the national park
near his city. He saw many ?owers. His guide explained
him that there are two categories of ?owers, category
A and category B. Category A ?owers spread pollen
via wind, and category B ?owers spread pollen via
animals.

Question: Would category B ?ower have more or less
ef?cient fertilization than category A ?ower?
Answer: more

Question: Would category A ?ower have more or less
ef?cient fertilization than category B ?ower?
Answer: less

Question: Which category of ?owers would be more
likely to have brightly colored petals?
Answer: Category B

Question: Which category of ?owers would be less
likely to have brightly colored petals?
Answer: Category A

Figure 1: Example questions in ROPES.

1 Introduction

Comprehending a passage of text requires being
able to understand the implications of the passage
on other text that is read. For example, after read-
ing a background passage about how animal pol-
linators increase the ef?ciency of fertilization in
?owers, a human can easily deduce that given two
types of ?owers, one that attracts animal pollina-
tors and one that does not, the former is likely
to have a higher ef?ciency in fertilization (Figure
1). This kind of reasoning however, is still chal-
lenging for state-of-the-art reading comprehension

models. Recent work in reading comprehension
has seen impressive results, with models reaching
human performance on well-established datasets
(Devlin et al., 2019; Wang et al., 2017; Chen et al.,
2016), but so far has mostly focused on extract-
ing local predicate-argument structure, without the
need to apply what was read to outside context.

We introduce ROPES1, a reading compre-
hension challenge that focuses on understanding
causes and effects in an expository paragraph, re-
quiring systems to apply this understanding to

1https://allennlp.org/ropes

novel situations.
If a new situation describes an
occurrence of the cause, then the system should
be able to reason over the effects if it has properly
understood the background passage.

We constructed ROPES by ?rst collecting
background passages from science textbooks and
Wikipedia articles that describe causal relation-
ships. We showed these paragraphs to crowd work-
ers and asked them to write situations that involve
the relationships found in the background passage,
and questions that connect the situation and the
background using the causal relationships. The
answers are spans from either the situation or the
question. The dataset consists of 14,322 questions
from various domains, mostly in science and eco-
nomics.

In analyzing the data, we ?nd (1) that there are
a variety of cause / effect relationship types de-
scribed; (2) that there is a wide range of dif?culties
in matching the descriptions of these phenomena
between the background, situation, and question;
and (3) that there are several distinct kinds of rea-
soning over causes and effects that appear.

To establish baseline performance on this
dataset, we use a reading comprehension model
based on RoBERTa (Liu et al., 2019), reaching an
accuracy of 61.6% F1. Most questions are de-
signed to have two sensible answer choices (eg.
�more� vs. �less�), so this performance is little
better than randomly picking one of the choices.
Expert humans achieved an average of 89.0% F1
on a random sample.

2 Related Work

Reading comprehension There are many read-
ing comprehension datasets (Richardson et al.,
2013; Rajpurkar et al., 2016; Kwiatkowski et al.,
the majority of which
2019; Dua et al., 2019),
principally require understanding local predicate-
argument structure in a passage of text. The
success of recent models suggests that machines
are becoming capable of this level of understand-
ing. ROPES challenges reading comprehension
models to handle more dif?cult phenomena: un-
derstanding the implications of a passage of text.
ROPES is also particularly related to datasets
focusing on �multi-hop reasoning� (Yang et al.,
2018; Khashabi et al., 2018), as by construction
answering questions in ROPES requires connect-
ing information from multiple parts of a given pas-
sage.

2018),

ShARC (Saeidi et al.,

The most closely related datasets to ROPES
Open-
are
BookQA (Mihaylov et al., 2018), and QuaRel
(Tafjord et al., 2019). ShARC shares the same
goal of understanding causes and effects (in terms
of speci?ed rules), but frames it as a dialogue
where the system has to also generate questions to
gain complete information. OpenBookQA, sim-
ilar to ROPES, requires reading scienti?c facts,
but it is focused on a retrieval problem where a
system must ?nd the right fact for a question (and
some additional common sense fact), whereas
ROPES targets reading a given, complex passage
of text, with no retrieval involved. QuaRel is also
focused on reasoning about situational effects in
the �causes�
a question-answering setting, but
are all pre-speci?ed, not read from a background
passage, so the setting is limited.

Recognizing textual entailment The applica-
tion of causes and effects to new situations has
a strong connection to notions of entailment�
ROPES tries to get systems to understand what
is entailed by an expository paragraph. The setup
is fundamentally different, however:
instead of
giving systems pairs of sentences to classify as
entailed or not, as in the traditional formulation
inter
(Dagan et al., 2006; Bowman et al., 2015,
alia), we give systems questions whose answers
require understanding the entailment.

3 Data Collection

Background passages: We automatically scraped
passages from science textbooks2 and Wikipedia
�causes,�
that contained causal connectives eg.
�leads to,� and keywords that signal qualitative re-
lations, e.g. �increases,� �decreases.�3. We then
manually ?ltered out the passages that do not have
at least one relation. The passages can be cate-
gorized into physical science (49%), life science
(45%), economics (5%) and other (1%). In total,
we collected over 1,000 background passages.

Crowdsourcing questions We used Amazon
Mechanical Turk (AMT) to generate the situations,
questions, and answers. The AMT workers were
given background passages and asked to write sit-
uations that involved the relation(s) in the back-
ground passage. The AMT workers then authored

2We used life science and physical science concepts from
www.ck12.org, and biology, chemistry, physics, earth sci-
ence, anatomy and physiology textbooks from openstax.org
3We scraped Wikipedia online in March and April 2019

questions about the situation that required both the
background and the situation to answer. In each
human intelligence task (HIT), AMT workers are
given 5 background passages to select from and
are asked to create a total of 10 questions. To mit-
igate the potential for easy lexical shortcuts in the
dataset, the workers were encouraged via instruc-
tions to write questions in minimal pairs, where a
very small change in the question results in a dif-
ferent answer. Two examples of these pairs are
given in Figure 1: switching �more� to �less� re-
sults in the opposite ?ower being the correct an-
swer to the question.

Statistic

# of annotators
# of situations
# of questions

avg. background length
avg. situation length
avg. question length
avg. answer length

background vocabulary size
situation vocabulary size
question vocabulary size

Train

Dev

7
1411
10924

121.6
49.1
10.9
1.3

8616
6949
1457

2
203
1688

90.7
63.4
12.4
1.4

2008
1077
1411

Test

2
300
1710

123.1
55.6
10.6
1.4

3988
2736
1885

Table 1: Key statistics of ROPES. In total there were
588 background passages selected by the workers.

Type

Background

C
(70%)

Q (4%)

C&Q
(26%)

Scientists think that the earliest ?owers at-
tracted insects and other animals, which
spread pollen from ?ower to ?ower. This
greatly increased the ef?ciency of fertil-
ization over wind-spread pollen ...
... As decibel levels get higher, sound
waves have greater intensity and sounds
are louder. ...
...
Predators can be keystone species .
These are species that can have a large
effect on the balance of organisms in an
if all of the
ecosystem.
For example,
wolves are removed from a population,
then the population of deer or rabbits
may increase...

Table 2: Types of relations in the background passages.
C refers to causal relations and Q refers to qualitative
relations.

4 Dataset Analysis

We qualitatively and quantitatively analyze the
phenomena that occur in ROPES. Table 1 shows
the key statistics of the dataset. We randomly sam-
ple 100 questions and analyze the type of relation
in the background, grounding in the situation, and

Type

Background

Situation

Explicit
(67%)

Common
sense
(13%)

Lexical
gap
(20%)

As decibel levels get
higher, sound waves
have greater intensity
and sounds are louder.

if we want to con-
...
vert a substance from
a gas to a liquid or
from a liquid to a
solid, we remove en-
ergy from the system
... Continued exercise
is necessary to main-
tain bigger,
stronger
muscles...

...First, he went to
stage one, where
the music was
playing in high
decibel.
She remem-
...
bered they would
be needing ice so
she grabbed and
empty
tray
ice
and ?lled it...
... Mathew goes to
the gym ...
does
very
intensive
workouts.

Table 3: Types of grounding found in ROPES.

reasoning required to answer the question.

Background passages We manually annotate
whether the relation in the background passage be-
ing asked about is causal (a clear cause and ef-
fect in the background), qualitative (e.g., as X in-
creases, Y decreases), or both. Table 2 shows the
breakdown of the kinds of relations in the dataset.

Grounding To successfully apply the relation in
the background to a situation, the system needs to
be able to ground the relation to parts of the situ-
ation. To do this, the model has to either ?nd an
explicit mention of the cause/effect from the back-
ground and associate it with some property, use
a common sense fact, or overcome a large lexical
gap to connect them. Table 3 shows examples and
breakdown of these three phenomena.

Question reasoning Table 4 shows the break-
down and examples of the main types of questions
by the types of reasoning required to answer them.
In an effect comparison, two entities are each asso-
ciated with an occurrence or absence of the cause
described in the background and the question asks
to compare the effects on the two entities. Simi-
larly, in a cause comparison, two entities are each
associated with an occurrence or absence of the
effect described in the background and the ques-
tion compares the causes of the occurrence or ab-
In an effect prediction, the question asks
sence.
to directly predict the effect on an occurrence of
the cause on an entity in the situation. Finally, in
cause prediction, the question asks to predict the
cause of an occurrence of the effect on an entity in
the situation. The majority of the examples are ef-
fect or cause comparison questions; these are chal-

Reasoning

Background

Situation

Question

Effect
comparison
(71%)

... gas atoms change to ions
that can carry an electric cur-
rent. The current causes the
Geiger counter to click. The
faster the clicks occur, the
higher the level of radiation.

Effect
prediction
(5%)

Continued exercise is
...
necessary to maintain bigger,
stronger muscles. ...

Cause
comparison
(15%)

... This carbon dioxide is
then absorbed by the oceans,
which lowers the pH of the
water...

... Location A had very high
radiation; location B had low
radiation

location A
Would
have faster or slower
clicks
than location
B?

... Mathew goes to the gym
5 times a week and does very
intensive workouts. Damen
on the other hand does not go
to the gym at all and lives a
mostly sedentary lifestyle.

The biologists found out that
the Indian Ocean had a lower
water pH than it did a decade
ago, and it became acidic. The
water in the Arctic ocean still
had a neutral to basic pH.

Given Mathew suffers
an injury while work-
ing out and cannot
go to the gym for
3 months, will Math-
ews strength increase
or decrease?

Which ocean has a
lower content of car-
bon dioxide in its wa-
ters?

Answer

faster

decrease

Arctic

Cause
prediction
(1%)

Other (8%)

... Conversely, if we want to
convert a substance from a gas
to a liquid or from a liquid
to a solid, we remove energy
from the system and decrease
the temperature. ...

...
she grabbed and empty
ice tray and ?lled it. As
she walked over to the freezer
... When she checked the
tray later that day the ice was
ready.

Charging an object
...
by touching it with another
charged object is called charg-
ing by conduction. ... induc-
tion allows a change in charge
without actually touching
the charged and uncharged
objects to each other.

... In case A he used conduc-
tion, and in case B he used
induction.
In both cases he
used same two objects. Fi-
nally, John tried to charge his
phone remotely. He called
this test as case C.

Did the freezer add or
remove energy from
the water?

remove

Which
experiment
would be less appro-
priate for case C, case
A or case B?

case A

Table 4: Example questions and answers from ROPES, showing the relevant parts of the associated passage and
the reasoning required to answer the question. In the last example, the situation grounds the desired outcome and
asks which of two cases would achieve the desired outcome.

lenging, as they require the model to ground two
occurrences of causes or effects.

Dataset split
In initial experiments, we found
splitting the dataset based on the situations re-
sulted in high scores due to annotator bias
from proli?c workers generating many examples
(Geva et al., 2019). We follow their proposal and
separate training set annotators from test set anno-
tators, and ?nd that models have dif?culty gener-
alizing to new workers.

5 Baseline performance

We use the RoBERTa question answering model
proposed by Liu et al. (2019) as our baseline and
concatenate the background and situation to form
the passage, following their setup for SQuAD. To
estimate the presence of annotation artifacts in our
dataset (and as a potentially interesting future task

Development
EM

F1

RoBERTa BASE
- background

RoBERTa LARGE
- background
+ RACE

Human

38.0
40.7

59.7
48.7
60.1

-

53.5
59.3

70.2
55.2
73.5

-

Test

F1

45.5
46.1

61.1
60.4
61.6

89.0

EM

35.8
33.7

55.4
53.6
55.5

82.7

Table 5: Performance of baselines and human perfor-
mance on the dev and test set.

where background reading is done up front), we
also run the baseline without the background pas-
sage. Table 5 presents the results for the baselines,
which are signi?cantly lower than human perfor-
mance. We also experiment with ?rst ?ne-tuning
on RACE (Lai et al., 2017) before ?ne-tuning on
ROPES.

Human performance is estimated by expert hu-
man annotation on 400 random questions with the
same metrics as the baselines. None of the ques-
tions share the sample background or situation to
ensure that the humans do not have an unfair ad-
vantage over the model by using knowledge of
how the dataset is constructed, e.g.
the fact that
pairs of questions like in Table 1 will have oppo-
site answers.

6 Conclusion

We present ROPES, a new reading compre-
hension benchmark containing 14,322 questions,
which aims to test the ability of systems to apply
knowledge from reading text in a new setting. We
hope that ROPES will aide efforts in tying lan-
guage and reasoning together for more comprehen-
sive understanding of text.

7 Acknowledgements

We thank the anonymous reviewers for their feed-
back. We also thank Dheeru Dua and Nelson Liu
for their assistance with the crowdsourcing setup,
and Kaj Bostrom for the human evaluation. We
are grateful for discussion with other AllenNLP
and Aristo team members and the infrastructure
provided by the Beaker team. Computations on
beaker.org were supported in part by credits from
Google Cloud.

References

Samuel R. Bowman, Gabor Angeli, Christopher Potts,
and Christopher D. Manning. 2015. A large anno-
tated corpus for learning natural language inference.
In EMNLP.

Danqi Chen, Jason Bolton, and Christopher D Man-
ning. 2016. A thorough examination of the cnn/daily
mail reading comprehension task. arXiv preprint
arXiv:1606.02858.

Ido Dagan, Oren Glickman, and Bernardo Magnini.
2006. The pascal recognising textual entailment
Lecture Notes in Computer Science,
challenge.
pages 177�190.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. Bert: Pre-training of deep
bidirectional transformers for language understand-
ing. In NAACL.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel
Stanovsky, Sameer Singh, and Matt Gardner. 2019.
DROP: A reading comprehension benchmark requir-
ing discrete reasoning over paragraphs. In NAACL.

Mor Geva, Yoav Goldberg, and Jonathan Berant. 2019.
Are we modeling the task or the annotator? an inves-
tigation of annotator bias in natural language under-
standing datasets. EMNLP.

Daniel Khashabi, Snigdha Chaturvedi, Michael A.
Roth, Shyam Upadhyay, and Dan Roth. 2018. Look-
ing beyond the surface: A challenge set for reading
comprehension over multiple sentences. In NAACL-
HLT.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
?eld, Michael Collins, Ankur Parikh, Chris Al-
berti, Danielle Epstein, Illia Polosukhin, Jacob De-
vlin, Kenton Lee, Kristina Toutanova, Llion Jones,
Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai,
Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019.
Natural questions: a benchmark for question answer-
ing research. In TACL.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang,
and Eduard Hovy. 2017. Race: Large-scale reading
comprehension dataset from examinations. arXiv
preprint arXiv:1704.04683.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized bert pretraining ap-
proach. arXiv preprint arXiv:1907.11692.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish
Sabharwal. 2018. Can a suit of armor conduct elec-
tricity? a new dataset for open book question an-
swering. In Proceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing,
pages 2381�2391.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100,000+ questions for
machine comprehension of text. In EMNLP.

Matthew Richardson, Christopher J. C. Burges, and
Erin Renshaw. 2013. Mctest: A challenge dataset
for the open-domain machine comprehension of text.
In EMNLP.

Marzieh Saeidi, Max Bartolo, Patrick Lewis, Sameer
Singh, Tim Rockt�aschel, Mike Sheldon, Guillaume
Bouchard, and Sebastian Riedel. 2018.
Interpreta-
tion of natural language rules in conversational ma-
chine reading. arXiv preprint arXiv:1809.01494.

Oyvind Tafjord, Peter Clark, Matt Gardner, Wen-tau
Yih, and Ashish Sabharwal. 2019. Quarel: A dataset
and models for answering questions about qualita-
tive relationships. In AAAI.

Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang,
and Ming Zhou. 2017. Gated self-matching net-
works for reading comprehension and question an-
swering. In Proceedings of the 55th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 189�198.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan R. Salakhutdinov,
and Christopher D. Manning. 2018. Hotpotqa: A
dataset for diverse, explainable multi-hop question
answering. In EMNLP.


