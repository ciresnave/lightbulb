OPEN ACCESS

Trends in
Cognitive Sciences

Review
Cognitive maps and schizophrenia

Matthew M. Nour 1,2,*, Yunzhe Liu 3,4, Mohamady El-Gaby 5, Robert A. McCutcheon 1, and
Raymond J. Dolan 2,3,6

Structured internal representations (�cognitive maps�) shape cognition, from imag-
ining the future and counterfactual past, to transferring knowledge to new settings.
Our understanding of how such representations are formed and maintained in bio-
logical and arti?cial neural networks has grown enormously. The cognitive map-
ping hypothesis of schizophrenia extends this enquiry to psychiatry, proposing
that diverse symptoms � from delusions to conceptual disorganization � stem
from abnormalities in how the brain forms structured representations. These
abnormalities may arise from a con?uence of neurophysiological perturbations
(excitation-inhibition imbalance, resulting in attractor instability and impaired
representational capacity) and/or environmental factors such as early life psycho-
social stressors (which impinge on representation learning). This proposal thus
links knowledge of neural circuit abnormalities, environmental risk factors, and
symptoms.

If the organism carries a �small-scale model� of external reality and of its own possible actions
within its head, it is able to try out various alternatives, conclude which is the best of them, re-
act to future situations before they arise, utilise the knowledge of past events in dealing with
the present and future, and in every way to react in a much fuller, safer, and more competent
manner to the emergencies which face it

[Kenneth Craik, The Nature of Explanation, 1943 [1]]

Representation: biological psychiatry�s missing mediating layer
Psychiatry lacks a mechanistic understanding of how neurobiological abnormalities cause symp-
toms. This explanatory gap impedes development of theory-guided treatments and has contrib-
uted to a decades-long stagnation in clinical outcomes [2,3]. One reason stems from the nature of
psychiatric symptoms and signs (�phenomena�), which � to a ?rst approximation � re?ect aberra-
tions in cognition and goal-directed behavior. In schizophrenia, these phenomena include delu-
sions, hallucinations, conceptual disorganization, and impairments in abstract/inferential
reasoning, planning, and social functioning [4�7]. How are we to understand these clinical and
cognitive manifestations at the level of the brain?

Psychiatric phenomena reside at a different level of explanation to the neurophysiological pro-
cesses from which they emerge. Symptoms are about things in the world in a way that synapses
are not [3,8]. Accordingly, to bridge a divide between neurophysiology and symptoms, brain-
based explanations must address how neural activity comes to be about things in the world.
This necessitates considering the mediating layer of neural representation (see Glossary) [8].

In this review, we outline advances in cognitive neuroscience that concern how the brain forms
structured internal representations of the world � cognitive maps � that organize knowledge
to guide learning, inference, and behavior [9�12]. While much of this knowledge concerns
hippocampal�entorhinal cortex representations of physical space, growing evidence indicates

Highlights
The brain�s ability to build structured
models (cognitive maps) of the environ-
ment is central for adaptive cognition
and behavior.

Recent work across theoretical and
cognitive neuroscience reveals neural
and algorithmic mechanisms of map
construction.

This work highlights roles for neural at-
tractor dynamics in hippocampal com-
plex and prefrontal cortex and a key
contribution of early life environment.

A cognitive mapping hypothesis of
schizophrenia states that both neuro-
physiological abnormalities and envi-
ronmental risk factors (psychosocial
stressors) cause symptoms by im-
pinging on cognitive mapping pro-
cesses (e.g., through neural attractor
instability and biased structure learn-
ing, respectively).

Preclinical and clinical studies in schizo-
to aberrant cognitive
phrenia point
mapping, with potential involvement of
hippocampal complex.

1Department of Psychiatry, University of
Oxford, Oxford, OX3 7JX, UK
2Max Planck University College London
Centre for Computational Psychiatry and
Ageing Research, London, WC1B 5EH,
UK
3State Key Laboratory of Cognitive
Neuroscience and Learning,
IDG/McGovern Institute for Brain
Research, Beijing Normal University,
Beijing, 100875, China
4Chinese Institute for Brain Research,
Beijing, 102206, China
5Nuffield Department of Clinical
Neurosciences. University of Oxford,
Oxford, OX3 9DU, UK
6Wellcome Centre for Human
Neuroimaging, University College
London, London, WC1N 3AR, UK

184

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2
� 2024 The Author(s). Published by Elsevier Ltd. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).

https://doi.org/10.1016/j.tics.2024.09.011

Trends in Cognitive Sciences

OPEN ACCESS

that conserved algorithmic principles are at play across diverse task domains. We discuss how this
program can inform an understanding of psychiatric phenomena, with a focus on schizophrenia.

*Correspondence:
matthew.nour@psych.ox.ac.uk
(M.M. Nour).

Cognitive maps: the core of cognition and adaptive behavior
The study of cognitive maps began with a careful observation of behavior. In the ?rst half of the
20th century, psychologists such as Tolman, Harlow, and Craik speculated on the existence of
structured internal representations on the basis of observing animal behavior that was dif?cult
to reconcile with contemporary behaviorist theories of decision-making [1,10,13,14]. Tolman de-
scribed experiments in which rats appeared to learn detailed �?eld maps� of spatial environments
in the absence of reward (�latent learning�), update these maps following surprises, and use them
when engaging in deliberative behavior (�vicarious trial-and-error�) [13]. Harlow showed that � in
some cases � primates exhibited task knowledge that could generalize beyond the context in
which it was acquired, to accelerate learning in new tasks that shared a common underlying
structure (�learning to learn�) [14].

Here, a deep insight is that that behavior can strongly imply, and sometimes guarantee, the exis-
tence of certain kinds of internal representations. This fact constituted a fatal blow to the central
tenet of behaviorism � that a discussion of cognitive states is beyond the reach of experimental
psychology � and laid the groundwork for the cognitive revolution of the later 20th century [15].

In the modern era, our understanding of how internal representations guide behavior has
progressed within diverse mathematical frameworks such as reinforcement learning (RL),
hierarchical Bayesian inference, and deep neural networks [16�18]. In RL, a particularly in?uential
intellectual strand in cognitive neuroscience, agents are endowed with task representations
comprised of states (�where am I?�) and actions (�what can I do here?�). An agent�s task represen-
tation, in combination with its learning and decision-making algorithms, profoundly shapes be-
havior. This is exempli?ed by a classical distinction between �model-free� and �model-based�
RL agents.

In RL, model-free agents are endowed with a simple task representation in which state�action
values are updated through trial-and-error, akin to the stimulus-response mechanisms of behavior-
ism (although, even here, representation may not be trivial, see Box 1). Model-based agents, by
contrast, learn more complex, structured internal models that account for how states are related,
akin to Tolman�s cognitive maps (Figure 1A), enabling them to engage in planning, non-local credit
assignment, and counterfactual reasoning [16]. The model-free versus model-based distinction
speaks to differences in both internal representation and algorithm. In practice, however, these
facets are deeply connected: the structure of an agent�s task representation has profound implica-
tions for the computational ef?ciency and inferential reach of model-based algorithms [12].

What makes a good task representation? The best task representations capture environmental
features that maximally facilitate prediction, inference, and decision-making and promote reuse
of knowledge in new environments [9,10,12,19] (�solving problems in representation, not by ex-
haustive computation� [12]). This often necessitates tracking latent states, which are inferred as
opposed to directly observed (Box 1). We note that these representational desiderata are echoed
in recent discussions of internal world models in arti?cial intelligence (AI) and active inference [20].

For psychiatry, the key point is that internal representation and behavior are inextricably linked.
Representation can be inferred from behavior precisely because representation constrains be-
havior. Clinically-relevant biases in inference and planning arise from task representations that
emphasize behaviorally irrelevant environmental features and/or postulate incorrect latent causal

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

185

OPEN ACCESS

Trends in Cognitive Sciences

Box 1. Challenges of map building in the real world

Curses of dimensionality and partial observability

Real world experiences are not annotated with �state� and �action� labels. When agents attempt to model the world based
solely on sensory data, they face two core challenges. First, each experience contains an overwhelming number of sensory
features, creating an expansive state space that is dif?cult to navigate or learn from (the �curse of dimensionality�). Second,
many factors critical for decision-making are not directly tied to sensory inputs (�partial observability�) [19].

To overcome these challenges, agents must infer the latent (abstracted) states that constitute an environment. This might
involve collapsing across super?cially distinct sensory states that are behaviorally equivalent, or conversely, mapping su-
per?cially similar sensory episodes to different latent states based on experience history (state aliasing) [19]. The resulting
state space is a substrate for learning and decision-making algorithms (the states and actions that both model-free and
model-based RL algorithms operate on belong to the internal representation, not the world per se) (see Figure IA).

Finding good abstractions

Good task representations rely on latent state spaces that are low-dimensional, sparse, and disentangled (or factorized).
This re?ects a heuristic that the causal structure of an environment is often well approximated by a small set of latent causal
factors that act (somewhat) independently and can thus be understood in isolation. Such representations signi?cantly sim-
plify learning and planning. They also afford generalization by allowing ?exible reuse of knowledge across different contexts
[9,10,12,48].

The brain is endowed with inductive biases that afford useful abstractions. These biases are shaped through learning pro-
cesses acting over both evolutionary and developmental time (e.g., meta-learning and hierarchical concept learning)
[10,12,18,117,119].

The construction of reward

Real world experience also does not come labeled with exogenous �rewards�. These, too, must be constructed. Hedonic
responses might be a function of an agent�s perceived progress towards internally-de?ned goal states, reminiscent of path
integration on an internal representation that spans multiple goal-relevant dimensions (social, ?nancial, etc.) [12,123].

(A)

Sensory state sequence

(B)

Suboptimal task representation

(C)

Better task representation

START

You are here

END

-

You are here

Figure I. Navigating and planning in task space. (A) Internal representations of task states and their (action-
dependent) relationships allow agents to track progress to goals, interpret new sensory data, and plan. The same
sequence of observations (sensory states) can give rise to multiple internal task representations (e.g., B and C, the latter
accounting for state aliasing in a manner that affords improved inference and planning). Inspired by [12,123].

Trends inin Cognitive
Trends

Sciences
Cognitive Sciences

structures. To understand how representation-level pathology might arise from neurophysiolog-
ical dysfunction, we ?rst need to understand the neural coding motifs that support cognitive
mapping itself.

Neural mechanisms of map construction
In order to facilitate inference, planning, and knowledge generalization, the brain must construct
internal representations that are structured in a particular way. Starting in the 1970s, electrophys-
iological studies in behaving rodents have uncovered neural activity patterns in hippocampal

186

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Glossary
Abstraction: distilling information from
one domain that is shared with related
domains (typically, common structure,
�abstracting away� sensory features,
referred to as a �structural code�).
Attractor: recurrent points or paths
(stable states) in neural activity space
that emerge from the biophysical and
connectivity properties of a recurrent
network.
Cognitive map: structured internal
representations that capture information
about the states in an environment and
the manner in which they are related.
Applies to spatial and non-spatial
domains alike.
Credit assignment: updating the value
of states and actions following rewards
or punishments, according to an internal
world model (cognitive map).
Generalization (transfer): the transfer
of (suitably abstracted) information from
one domain to another.
Grid cells: neurons in medial entorhinal
cortex and mPFC that respond when an
animal is in one of a set of spatial
locations structured in a regular grid tiling
a state space. The population grid code
is conserved across environments,
potentially serving as a global coordinate
system (structural code) for
generalization.
Latent (hidden) state: an inferred
environmental state that is not directly
observable. A grouping of sensory
states based on behaviorally-relevant
commonalities.
Object and boundary vector cells:
found in medial entorhinal cortex and
subiculum, respectively, these neurons
encode vectors to salient objects or
boundaries, preserving this relational
coding across environments (like grid
cells). Serve as a local coordinate system
(structural code) for generalization.
Place cells: hippocampal neurons that
respond when an animal is in a particular
location (state). The population place
code is not conserved (i.e., remaps)
across environments.
Preplay: hippocampal networks are
precon?gured into sequential activation
motifs which, in subsequently
encountered environments, may come
to correspond to behaviorally-
meaningful sequences.
Reinforcement learning (RL):
mathematical formalism describing how
agents select actions in environments to
maximize the sum of (temporally
discounted) expected future reward.

Trends in Cognitive Sciences

(A)

Model-free and model-based RL agents

Learn values from experience

Behaviour after transition change

Behaviour after reward change

Model-free

Model-based

Model-free agent task representation

Model-based agent task representation

State

UP

DOWN

LEFT

RIGHT

3,0

3,1

2,1

1,1

V=0

V=0

V=0

V=1

V=1

V=0

V=0

V=0

V=1

V=0

V=0

V=0

V=0

V=0

V=0

V=1

Look-up table of state-action values

(B)

Hippocampal-entorhinal
state representations

Loc. A

Loc. B

T

R

Update on removal of barrier

T: State-State transition matrix
R: Reward function
O: Observation function (in partially-observable environments)

(C)

Place cell sequences:
behaviour

Place cell sequences:
rest

OPEN ACCESS

Replay sequences: during rest (awake
immobility and non-REM sleep), place
cells ?re in time-compressed sequences
in conjunction with SWR events.
Representation: a neural activity
pattern that is �about� a state of the
world, can be reinstated in the absence
of the eliciting state, and plays a causal
role in generating behavior. Cognition is
computation (transformations) over
representations [8]. A representation is a
functioning isomorphism between a set
of processes in the brain and a
behaviorally-important aspect of the
world [15].
Representational geometry:
similarity structure relating neural
representations of task states (Box 3).
Sharp wave ripple (SWR): large
amplitude deviations coupled with 140�
200 Hz �ripple� oscillations seen during
rest in hippocampal local ?eld potential
(LFP). Accompany replay.
State aliasing: near-identical sensory
states that map to distinct latent states.
Theta sequences: during behavior,
place cells ?re in time-compressed
sequence nested within LFP theta
cycles, tracing potential behavioral
paths.

s

l
l

e
c
d
i
r

G

r
o
t
c
e
v

j

t
c
e
b
O

s

l
l

e
c

l
l

e
c
e
c
a
P

l

Behavioural trajectory
Decoded neural
trajectory

Theta sequence

Reverse replay Inferential replay

Place cells

LFP
Theta

Time

LFP
SWR

Trends inin Cognitive
Trends

Sciences
Cognitive Sciences

�

�

�
� vs. V A2, S1

�), reward functions (P RjS1, A1

Figure 1. Cognitive maps: neural and behavioral correlates. (A) In reinforcement learning (RL), model-free agents
learn the value of each state�action pair through direct experience and store this value in a form of look-up table. The
decision to perform action A1 or A2 at state S1 is informed by how likely each state�action pairing was to lead to
�). (Temporal discounting of value estimates has been omitted.) By contrast,
reward in the past (i.e., V A1, S1
model-based agents learn an internal model of the task structure, comprising action-dependent Markovian state transi-
tion functions (P S2jS1, A1
�). Model-based agents
can update policies when the environment changes (barriers removed, new rewards added). (B) Selected spatial repre-
sentations in hippocampal�entorhinal cortex (adapted from [12]), spanning medial entorhinal structural codes (grid cells
and object vector cells [25�28]) and hippocampal place cells [22�24]. (C) Place cell sequences betray spatial map com-
putation. T-maze, where rodents are trained to turn left or right at a junction (never running the full top segment). Left:
�look-ahead� theta sequences during behavior (supporting spatial memory and planning), during local ?eld potential
(LFP) theta oscillations [21,29�32]. Right: replay sequences during rest (reverse replay, supporting credit assignment
[34], and sequences that �stitch together� experiences, supporting relational inference [40,41]) during LFP sharp wave
ripple (SWR) oscillations.

�), and observation functions (P O1jS1

�

�

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

187

OPEN ACCESS

Trends in Cognitive Sciences

formation and prefrontal cortex (PFC) that bear striking correspondence to representational mo-
tifs initially conceived through observation of behavior. Here, we provide a selected overview of
key results (for more detailed reviews, see [10,12,21]).

Maps of space in hippocampal�entorhinal cortex
The hippocampal formation encodes environmental states and the relationships between them. Dur-
ing behavior, hippocampal place cells encode the animal�s current location in task space (i.e., state),
which may be synonymous with spatial location in simple tasks (i.e., place ?elds) [22�24], or corre-
spond to location in a latent task representation in more complex tasks (e.g., those involving state
aliasing, see Box 1) [12]. Neurons in neighboring medial entorhinal cortex and subiculum encode re-
lationships between states, abstracted from task-speci?c sensory information (�structural codes�,
such as grid cells, as well as object and boundary vector cells) [25�28] (Figure 1B).

Hippocampus also encodes structured information about task states in cell assembly se-
quences. During behavior, place cell theta sequences chart future behavioral trajectories in a
manner that supports sequential memory and planning [21,29�32], while, during resting periods,
hippocampal replay sequences recapitulate past experiences in both forward and backward
directions in a manner thought to support memory consolidation and credit assignment
[33�36] (Figure 1C).

Hippocampal assembly sequences also encode information that goes beyond direct experience.
Thus, replay can play out novel behavioral trajectories [37�39], �stitch together� fragments of ex-
perience [40,41] (Figure 1C), and display non-behavioral sampling statistics [42]. Preplay se-
quences have also been observed, so-called because the episodes they encode do not
appear to correspond to any prior environment but instead acquire behavioral relevance with re-
spect to place maps learned in future environments (potentially re?ecting a role in scaffolding new
experiences) [43,44].

Together, these observations constitute strong evidence that hippocampus is key to organizing
knowledge of how environmental states are structured. They also point to a role in knowledge
generalization, whereby as-yet unexperienced associations (state�state relationships or conjunc-
tions) are inferred through an application of prior knowledge about the relational structure of the
environment. Indeed, it has long been known that lesions to hippocampal formation in rodents
impair relational inference [45,46].

Generalization through abstraction
The computational principles underlying knowledge generalization are incompletely understood.
Leading accounts implicate structural codes, as carried in grid and object vector cells. These
codes contain information about the way states are related, where such information is abstracted
from any speci?c sensory experience, conserved across contexts, and factorized (Box 1). This
enables structural codes to act as a generalizable basis set � a repertoire of representational
�building blocks� � for new map construction. These building blocks (or primitives) can be ?exibly
combined to form conjunctive state representations in new environments [9,10,12,17,44,47,48],
potentially implemented in hippocampal replay [9,12,44,48]. Compositional coding provides a
powerful mechanism for knowledge generalization: primitives come pre-packaged with structural
information that permits zero-shot relational inferences (�a barrier separates location A and B�)
and behavioral competencies (�avoid barriers�) [9,12].

Thus, knowledge generalization and state-space construction can be cast as inference on the
compositional structure of the environment [12], in which the repertoire of representational

188

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

building blocks � potentially carried by hippocampal�entorhinal structural codes � constitutes
prior information about the latent structure and behavioral affordances of new environments.
Under this account, pathologies of generalization � such as those seen in situational anxiety
and paranoia � might stem from abnormal
inference itself (weighting of prior and likelihood
information) or from biases in the composition and distribution of one�s prior structural repertoire
(as discussed later). This constitutes an underexplored point of contact between cognitive
mapping and predictive coding accounts of schizophrenia and psychiatric syndromes more
broadly (Box 2).

Domain-general cognitive mapping
For cognitive mapping to serve as a viable explanatory framework in psychiatry, the aforemen-
tioned neural coding schemes must extend beyond a representation of physical space. Indeed,

Box 2. Cognitive mapping and predictive coding

Predictive coding

Predictive Coding conceptualizes the brain as a hierarchical Bayesian inference machine that seeks to maximize the evi-
dence for its internal model of the world. This objective can be cast as minimizing the discrepancy (prediction error) be-
tween the brain�s predictions of observable sensory data and sensory data itself. Predictions arise from the brain�s
generative model and project from higher to lower hierarchical layers (top-down). Sensory data is carried from lower to
higher hierarchical layers (bottom-up). The integration between these sources of information can be formalized as Bayes-
ian inference: prior beliefs (predictions) combine with likelihood information (sensory data) to update a posterior belief (new
representation of the environment). The magnitude of the belief update is determined by the relative precision (reliability) of
prior and likelihood information [125].

Active inference extends this framework to action. In order to minimize the discrepancy between internal predictions and sensory
data, agents can modify which sensory information is sampled (action), in addition to updating internal beliefs (perception) [20].

Predictive coding accounts of psychosis

Predictive coding accounts of psychosis posit that positive psychotic symptoms are a product of aberrant hierarchical in-
ference, resulting in abnormal posterior beliefs about the latent causes generating observable data. This computational ab-
normality might relate to abnormalities in cortical glutamatergic/GABAergic signaling (that carry prior and likelihood
information) and dopamine/acetylcholine signaling (thought to modulate precision-weighting). The canonical predictive
coding account posits a reduction in precision of top-down priors (consequent on glutamatergic and GABAergic impair-
ments in frontal cortex) and a strengthening of precision in bottom-up likelihoods (consequent on mesostriatal
hyperdopaminergia), resulting in delusions and hallucinations (both construed as �false inferences�) [125]. Extensions to
this canonical account exist. The incorrigible nature of delusions, for example, has been proposed to re?ect increased prior
precision at higher hierarchical levels, alongside (and potentially compensating for) weakened prior precision at lower levels
[116]. Predictive coding�s focus on prediction is shared by earlier computational theories of psychosis, such as the
�aberrant salience� hypothesis and �comparator (corollary discharge) model� [125].

Relationship to cognitive mapping

Predictive coding and cognitive mapping share a core premise: the brain constructs internal models of the latent causal struc-
ture of the environment, a process driven by Bayesian inference. Predictive coding offers a formal process theory for how the
brain might approximate Bayesian inference through message passing in cortical circuits. This framework is thus theoretically
capable of advancing models linking neurophysiology to brain-wide computational principles [125]. We have suggested that
many facets of cognitive mapping might also be fruitfully understood within a similar Bayesian framework, in which structural
codes constitute priors for state-space construction and relational inference in new environments.

Despite this convergence, cognitive mapping and predictive coding diverge in their focus. Cognitive mapping emphasizes
speci?c coding motifs in brain regions that support relational inference, goal-directed planning, and task representation.
These include abstract structural codes in medial entorhinal cortex, conjunctive state representations in hippocampus,
goal-centered map representations in prefrontal cortex, and hippocampal ensemble sequences. Attention is given to
these coding motifs facilitates ef?cient knowledge generalization and map construction
how the nature of
(e.g., abstraction, factorization) [10,12]. This difference in focus links cognitive mapping theories to a large body of empir-
ical research on spatial map representation in rodents and domain-general mapping in humans (Box 3). It may also cast
known neurophysiological disturbances in schizophrenia in a new light. Here, one intriguing question relates to how
mesolimbic dopamine signaling and midbrain�hippocampal
interaction impacts hippocampal cognitive mapping
[68,126,127].

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

189

OPEN ACCESS

Trends in Cognitive Sciences

as early as the 1970s, O�Keefe and Nadel speculated that hippocampus might support domain-
general cognitive mapping [23] � a proposal now supported by abundant evidence.

Animal studies report place- and grid-like coding of olfactory [49], auditory [50], and social [51]
variables, as well as in tasks involving non-spatial relational inference [41]. Human functional
neuroimaging studies, in combination with innovative analytic techniques (Box 3), also provide
evidence for domain-general cognitive mapping in hippocampal formation. Thus, using
functional magnetic resonance imaging (fMRI), the representational geometry of hippocampal�
entorhinal cortex task-evoked activation patterns has been shown to mirror latent task structure
in an abstracted manner that facilitates knowledge generalization [52�54]. fMRI has also been
used to uncover grid-like coding in entorhinal cortex during spatial navigation (virtual [55] and
imagined [56]) and when participants �navigate� structured conceptual and social spaces
[57,58].

fMRI and magnetoencephalography (MEG) have also been used to index spontaneous sequen-
tial neural state reactivations, indicative of replay, in tasks involving sequential inference and
decision-making [48,59,60]. Such activity bears several correspondences to hippocampal replay
in rodents. fMRI replay localizes to hippocampus [59], while in MEG, replay re?ects inferred (as
opposed to merely experienced) task structure and is coincident with high-frequency oscillations
that source-localize to hippocampus [akin to hippocampal sharp wave ripples (SWRs),
Figure 1C] and reverses in direction following reward [48]. MEG-measured replay is also com-
posed of factorized representations of abstracted structural codes and sensory codes (Box 1),
where the former can be detected prior to task exposure and may facilitate structural knowledge
transfer (generalization) [48] (cf. rodent preplay [43,44]).

Beyond medial temporal lobe, human medial PFC (mPFC) also exhibits hallmarks of domain-
general cognitive mapping, including grid-like coding [55,57,58], abstracted task schemas [61],
and conjunctive state representations that re?ect knowledge of task structure [47,62]. In rodents,
PFC tracks progress toward internally-generated goals, re?ecting use of cognitive map represen-
tations in ?exible planning [63]. Likewise, orbitofrontal cortex is causally implicated in the repre-
sentation of latent states, as revealed by tasks involving state aliasing [59,64�66]. The latter
?ndings raise an intriguing hypothesis that PFC might encode the �structure of an agent�s goal-
directed behavior�, which can be contrasted to hippocampal formation�s specialization in
encoding the �structure of the world� [63].

An exciting research area concerns how different brain regions interact to support cognitive map-
ping functions. Goal-centric PFC representations also encode spatial information [63] (presumed
to originate in hippocampal�entorhinal cortex), while hippocampal place cells are biased by goal
information [67,68] (presumed to originate in PFC). In rodents, non-human primates, and humans
hippocampal replay and SWRs are temporally correlated to neocortical activity ?uctuations
[69�73]. Here, there appears to be some speci?city for default mode network (DMN) [71�73], a
collection of predominantly-midline cortical regions considered to occupy the deepest layers of
a cortical processing hierarchy and proposed to support world models at the highest level of ab-
straction (e.g., narrative schemas) [74�77].

Cognitive mapping and schizophrenia: behavioral and neural evidence
Before moving to a discussion of how a cognitive mapping framework might inform psychiatry,
we offer a brief distillation of our discussion thus far. First, an understanding of how internal rep-
resentations are acquired and structured is key to understanding the diversity of behavior and
cognition, including the extremes that are the focus of psychiatrists and clinical psychologists.

190

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

Box 3. Indexing representations in task-based neuroimaging

Decoding and encoding

These are supervised machine learning models that link neural response vectors to task state labels. Decoding predicts la-
bels from neural activity, while encoding predicts neural activity from labels. Their generalization to unseen data (e.g., �leave-
one-trial-out� cross-validation) is taken as evidence that neural activity represents task states. In tasks where multiple sensory
states map to a single latent state, cross-condition generalization performance (CCGP) can indicate abstract representations
[cross-validation leaves out all trials from one sensory state (condition) and labels correspond to abstract states] [48,128].

Representational geometry

Representational geometry examines how neural responses relate across task states � a second order statistic [129,130].
Repetition suppression uses neural adaptation to quantify neural response relatedness: if A and B share a representation,
the neural response to B is reduced after A [47,52,53,62,129]. Representational similarity analysis (RSA) quanti?es
(correlation or Euclidean) distance between each state�s multi-voxel/multi-sensor activity vector [41,53,54,83,130]
(Figure I). These methods are insensitive to the neural axes of representation � geometry is conserved under rotations.

Grid coding and replay

Sequential neural replay can be indexed using temporally delayed linear modeling (TDLM) in MEG, which combines neural
decoding and time-lagged regression [47,48,83,131] (see Figure 2C in the main text). Analogous approaches can be used
in fMRI, exploiting within- and between-TR variation in decoder performance [59,132]. Grid-like codes can be measured
by exploiting the fact that grid cells are organized in modules, where cells within a module share grid spacing (resolution)
and orientation (angle). During virtual navigation tasks, module-aligned movement results in increased grid cell ?ring (a
combination of increased grid ?eld sampling and alignment to conjunctive grid cell preferred ?ring direction) and this is
re?ected in neural activity proxies [55�58] (see Figure 2D in the main text).

Dimensionality reduction

These tools project high-dimensional neural activity to a lower-dimensional space, maximizing some objective
(e.g., principal component analysis (PCA): orthogonal basis set, iteratively maximizing explained residual variance). Dimen-
sionality reduction might uncover modes of variation in population activity that mirror the structure of internal representa-
tions (cf. neural manifold theories [3,8,128,133]). Several techniques exist � from linear to non-linear; from unsupervised to
supervised � mandating careful consideration of the assumptions and biases inherent in any one method [133].

(A)

(B)

Inferior frontal gyrus and insula

Anterior temporal lobe

Superior temporal gyrus

Trends inin Cognitive
Trends

Sciences
Cognitive Sciences

Figure I. Representational similarity of non-spatial task. (A) Task structure used to generate stimulus sequence
(each node a picture, each edge a random walk sequential transition). (B) Representational geometry of functional
magnetic resonance imaging (fMRI) responses to each stimulus (node) (searchlight representational similarity analysis
(RSA) combined with multidimensional scaling). Adapted from [53].

Second, a great deal about the structure of internal representations, including candidate neural
coding motifs, can be inferred from a careful study of behavior, particularly in conjunction with
computational modeling. Third, the recent focus on neural map representations in cognitive

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

191

OPEN ACCESS

Trends in Cognitive Sciences

neuroscience has con?rmed the existence of many such coding motifs. This research program is
now suf?ciently mature to begin to inform a new era of mechanistic models in psychiatry.

In the remainder of this review, we turn to the speci?c instance of schizophrenia. We begin by
considering how a cognitive mapping account might accommodate clinical features in people di-
agnosed with schizophrenia (PScz) and highlight behavioral and neural evidence for cognitive
map dysfunction. We then speculate on how map dysfunction might arise from potential
upstream causal factors, focusing ?rst on neurophysiological abnormalities and second on
(somewhat independent) early life environmental risk factors.

Starting with behavior
Many symptoms and signs in schizophrenia can be understood as manifestations of abnormali-
ties at the level of internal representation, broadly dichotomized as those affecting inference and
generalization (e.g., delusions, paranoia, ideas of reference) and those affecting sequential sam-
pling from underlying representations (e.g., formal thought disorder) (Table 1).

In behavioral experiments using curated tasks, PScz demonstrate impairments in model-based
decision-making [78,79], relational inference [80�83], and sequential planning [5�7,84]. A cogni-
tive mapping framework additionally invites us to consider behavioral consequences in more ab-
stract task domains, where the state space comprises concepts and relational structure
corresponds to pairwise semantic similarity. Approaches to operationalizing this more abstract
relational structure include inferring it from participant behavior (e.g., word association data [85]
or similarity judgements [86]) or using learned embedding spaces from pretrained AI language
models. Intriguingly, there is tentative evidence that some aspects of these �semantic spaces�
are represented by coding motifs originally evolved for spatial navigation, in line with domain-
general cognitive mapping theories. Thus, behavior in category ?uency tasks (�name as many an-
imals as you can�) is well described by patch foraging models of animal search behavior [87] and
is accompanied by hippocampal activity patterns that track semantic category switches [86] and
distances [88].

In PScz, language-based studies report abnormalities in semantic distances separating consec-
utive words and utterances [89]. In one study employing a category ?uency task, word lists gen-
erated by PScz were signi?cantly less predictable using semantic relatedness information derived
from an AI language model (Figure 2B) [90], reminiscent of early accounts of schizophrenia as a
disorder of �loosened associations� (Table 1).

Neural evidence of cognitive map dysfunction
Neural signatures of replay and grid coding have recently been investigated in PScz using
functional neuroimaging during tasks that require participants to use knowledge of how
states are related.

In one MEG study, participants were tasked with inferring how pictures were sequentially related
by combining information gathered from direct experience and a prelearned abstracted task
schema (i.e., a structural prior) [83]. PScz displayed impaired neural replay for inferred task struc-
ture and abnormal replay-related hippocampal ripple oscillations in the rest period immediately
following learning, despite no behavioral evidence for impaired ceiling-level knowledge or
accelerated forgetting (Figure 2C), reminiscent of genetic mouse models of schizophrenia
[67,91,92]. There were also differences between PScz and control participants in the representa-
tional geometry of task-evoked activations, with control participants alone displaying an
abstracted code for ordinal position that emerged as a function of learning.

192

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

Table 1. Symptoms and signs of schizophrenia through a cognitive mapping lens; accompanied by item code
from the Positive and Negative Symptoms of Schizophrenia (PANSS) scale [122]

Symptom or sign

Description

Cognitive mapping perspective

Delusions (P1, P5, P6, G9)

Fixed, abnormal beliefs, often
persecutory or bizarre

Conceptual disorganization
(P2)

In formal thought disorder, disrupted
semantic relationships between
successive speech utterances
(e.g., �loosening of associations�)

Perceptual abnormalities
(P3)

Hallucinations (false perceptions) or
illusions (distorted perceptions)

Negative symptoms
(N1�N4)

Including apathy, amotivation,
emotional blunting

Cognitive impairment (N5,
N7)

De?cits in relational inference,
sequential planning, and higher-order
(e.g., analogical) reasoning [5�7,78�84]

Phenomenology (of
delusions)

Some patients experience delusions as a
radical restructuring of the experienced
world, involving internal struggles for
understanding and control, and reduced
feelings of uncertainty [124]

Delusions re?ect maladaptive internal
representations of environmental states
and relational structures, arising from
abnormalities in structural generalization
and inference. Neurally, these de?cits may
stem from E/I imbalances, leading to
shallow attractor dynamics and disrupted
generative replay (see main text)

This may re?ect disorganization in
internal representations (disrupted
cognitive maps of semantic space) or in
stochastic sampling processes acting
on these maps (e.g., consequent upon
shallow attractor dynamics, Figure 2A)

Predictive coding accounts propose
that perceptual abnormalities result
from disruption at the level of prior
beliefs (internal representations) about
the causal structure of the environment
(e.g., increased or decreased precision)
(Box 2)

Some negative symptoms might stem
from impairments in representing latent
task states and goals, and tracking
proximity and progress to goals [123]
(Box 1). mPFC cognitive maps are
sensitive to goal information
(e.g., progress-to-goal representations
[63])

Planning requires an internal
representation of how of states in the
environment are related
(e.g., action-dependent state�state
transition matrix, Figure 1A). Relational
inference rests on transfer of prior
structural knowledge to new setting,
enabling inference of �missing link�
associations that have not been directly
observed. Analogical reasoning rests on
an ability to infer a shared relational
structure (isomorphism) between two
super?cially-different domains

These descriptions resemble insight or
�ah-ha!� moments: the sudden
discovery of a new way of
understanding a problem in terms of a
familiar and parsimonious latent causal
structure � a form of structural
knowledge generalization [117].
Anecdotally, insight usually arises
following a period of of?ine cognition (cf.
generative/constructive of?ine replay
[9,12,44])

In subsequent work, the authors identi?ed impaired SWR-DMN coupling in PScz [93] and � using
positron emission tomography in the same participants � an association between replay-
associated ripple power and hippocampal NMDA-receptor availability [94] (thought to play a
crucial role in cortical E/I balance and replay [95�97]).

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

193

OPEN ACCESS

Trends in Cognitive Sciences

(A)

(B)

(C)

(D)

Trends inin Cognitive
Trends

Sciences
Cognitive Sciences

Figure 2. Cognitive mapping and schizophrenia. (A) In hippocampus, shallow attractors [97,109] predispose to
abnormal sequential replay, potentially leading to further entrenchment of abnormal attractor basins (see main text).
Schematic inspired by [109]. (B) In a category ?uency task, words can be represented as high-dimensional vectors in seman-
tic space using an arti?cial intelligence (AI) word embedding model. Each participant�s word list can then be construed as a
trajectory through semantic space (here a two-dimensional projection of a 300-dimensional embedding). The �actual� trajec-
tory can be compared with one that minimizes semantic distance �traveled� (�optimal�). At the group level, word lists gene-
rated by people diagnosed with schizophrenia (PScz) deviated from 'optimality' more than controls [90]. Figures from [90].
(C) Top: magnetoencephalography (MEG) sequence learning task. Participants need to infer the correct transitions between
eight task pictures (�structural sequences�) [83]. Bottom: during a post-learning rest session, PScz exhibited reduced evi-
dence for spontaneous neural replay of task structure compared with controls (using temporally delayed linear modeling,
see Box 3 in the main text). In controls, replay was evident at ~50 ms state�state transition lag. Figures from [83]. *** denotes
PFWE < 0.001. (D) Top left: MEG spatial memory task. Participants navigate to remembered locations in a virtual arena. Top
middle: exemplar grid cell spatial auto-correlogram. In functional neuroimaging studies, grid-like coding can be indexed by
modeling neural activity as a function of movement direction (e.g., grid-aligned vs. misaligned, see Box 3 in the main text)
(schematic inspired by [55]). Top right: grid-like theta modulation in right entorhinal cortex across all participants. Bottom:
grid-like coding was reduced in PScz (data ?gures from [100]).

In another MEG study using a spatial memory task, PScz displayed reduced theta phase coupling
(1�8 Hz) between mPFC and medial temporal lobe during cued spatial memory recall [81], in line
with ?ndings in two animal models [98,99]. PScz also exhibited reduced entorhinal grid-like
coding (4�10 Hz power modulation) during virtual spatial navigation [100] (Figure 2D).

These clinical MEG studies thus provide the ?rst direct evidence for impaired hippocampal�
entorhinal cognitive mapping signatures in PScz.

194

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

Paths to symptoms: cognitive maps as a mediating layer
How might cognitive map dysfunction arise from upstream neuropathological and environmental
risk factors?

Neurobiology: from shallow attractors to symptoms
Schizophrenia is associated with diverse neurobiological abnormalities in brain regions that sup-
port cognitive mapping. In hippocampus, this includes reduced grey matter volume [101], re-
duced synaptic density [102], increased resting state perfusion and metabolism, and abnormal
task-related activation [103]. Similar abnormalities are reported in PFC [101,102,104,105]. How-
ever, such ?ndings are rarely anatomically-speci?c and do not shed light on how neurophysiolog-
ical disturbances result in representation-level abnormalities that drive speci?c symptoms.

More promising are convergent ?ndings that a ?nal common pathway in the neurobiology of
schizophrenia is a disruption in the balance between excitatory glutamatergic and inhibitory
GABAergic signaling in cortical circuits, termed E/I imbalance, as evidenced by electroencepha-
lographic abnormalities in gamma oscillations and sensory gating de?cits (among many other
?ndings) [96,97,106].

From a dynamical systems perspective, E/I balance in recurrent cortical circuits underpins neural
attractor dynamics � a network�s ability to maintain stable population activity patterns
[97,107�109]. By contrast, E/I imbalance disrupts a neural attractor landscape, rendering
attractors unstable or �shallow� � that is, increasingly sensitive to internal or external noise
[109]. Consistent with this account, genetic rodent models of schizophrenia exhibit a reduction
in the number of stable cortical activity states [107,108].

The ability of a network to maintain stable attractors underpins both its representational and
computational capacities. This property makes explanations couched in terms of abnormal at-
tractor dynamics a promising candidate for explanatory models in psychiatry. Accordingly,
such explanations have been advanced to explain cognitive and behavioral symptoms not
only in schizophrenia but also in other conditions involving working memory impairment, inat-
tention, and behavioral variability, such as attention de?cit hyperactivity disorder (ADHD)
[97,108,110,111].

A cognitive mapping perspective casts neural attractor instability in a subtly new light, considering
not only its immediate consequences (e.g., sensitivity to noise), but also downstream effects on
the structure of an agent�s internal representations [109]. This is because many of the discussed
hippocampal�entorhinal neural coding motifs � from spatially-tuned place and grid ?elds, to se-
quential replay � can be construed as neural attractors [109].

In PScz, the evidence for attractor instability in hippocampal�entorhinal codes is indirect and
includes the aforementioned MEG studies. However, there is abundant evidence for hippo-
campal place coding disruption in rodent models of schizophrenia, which are engineered to re-
?ect known genetic and neurodevelopmental risk factors (e.g., 22q11.2 deletion/Setd1a
mutation [67,108] and maternal immune activation [112], respectively; see [113] for an over-
view). Here, notable ?ndings include abnormal elevations in population-level SWR power and
rate [67,91,92,112], disrupted place cell coactivation patterns during rest [91] (a prerequisite
for replay), and disrupted place cell theta sequences, theta phase coupling, theta phase pre-
cession, and gamma-theta coherence [112,114]. Animal studies also ?nd population place
map instability over time, impaired goal-related remapping [67], and overly generalized place
?eld coding [115].

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

195

OPEN ACCESS

Trends in Cognitive Sciences

Thus, E/I imbalance, as seen in (rodent models of) schizophrenia, has a potential to disrupt hip-
pocampal (and presumably, entorhinal) coding motifs that have been directly tied to neural
state representation and cognitive map construction. Importantly, such representations need
to be maintained in the absence of driving sensory information, either because they manifest dur-
ing rest periods (i.e., replay), or because the mapping between on-task sensory information and
corresponding latent state representation is indirect and weak (Box 1). It is in precisely such sit-
uations that attractor instability is most vulnerable to being unmasked [109].

One speculative path from attractor instability to symptoms concerns the role of replay in cogni-
tive map construction. During rest, population activity governed by shallow attractor dynamics
may transition between states that are �far apart� in representational space, or ?it between neigh-
boring attractor basins. This results in novel ensemble sequences and compositions [44] that are
implausible under an appropriately conditioned world model. Such information might neverthe-
less be used to update and extend map representations, entrenching abnormal attractor basins,
which in turn constrain further cycles of inference, generative/constructive replay, and map exten-
sion (Figure 2A). Speculatively, in clinical settings these abnormal dynamics might manifest as
conceptual disorganization and belief instability (?itting between weakly associated conceptual
states � a direct consequence of shallow attractors), or delusions (entrenched, yet inappropriate,
attractor basis � arising from the interaction between shallow attractors and constructive replay)
(Table 1) [97,109,111]. This dynamical account dovetails with clinical observations that delusions
may initially be somewhat malleable, before crystalizing into more incorrigible, ?xed beliefs (see
also Box 2 and [116]).

Another path to symptoms concerns task domains that necessitate maintenance of latent state
representations, which, by de?nition, are not resolvable from sensory information alone and are
thus vulnerable to neural attractor instability. Latent state representations are indispensable in
tasks that require abstract reasoning (i.e., cognition on structural features), planning (requiring
hierarchical task decomposition into subgoals), social cognition (where latent variables span
social kinship graphs, role-relationship schemas, and the mental states of others), and context-
dependent behavioral routines (necessitating formation and maintenance of overarching goals
and task schemas) (Box 1). An impaired ability to maintain stable latent state representations
might account for some aspects of cognitive impairments in schizophrenia, particularly in social
cognition and executive functions spanning abstract reasoning, goal maintenance, and planning
[5�7,84]. More speculatively, this impairment may also relate to delusions, many of which com-
prise beliefs about the latent structure of the world (e.g., webs of interactions between people
or entities in paranoia) and one�s location within it (e.g., the center of the web).

The early environment: from biased representation learning to symptoms
Any viable model of schizophrenia � or indeed, any psychiatric condition � must be capable of ac-
commodating the considerable in?uence of vulnerability factors such as carer abuse, parental
loss, bullying, and immigrant status [4]. How are we to understand the role of these environmental
factors within a brain-based model of schizophrenia?

Previous attempts at an integrated model have adopted one of two avenues. The ?rst is to note
that early life risk factors � including psychosocial stress � exert lasting effects on neurobiology
(e.g., synaptic spine density and dopaminergic signaling [4,104]), likely to impact neural pro-
cesses important for neural computation (e.g., E/I balance). The second is to posit that psycho-
social stressors foment maladaptive cognitive schemata that bias appraisals, potentially leading
to symptoms such as paranoia [4]. It has been dif?cult to integrate both accounts within a single
mechanistic model of neurocognitive function.

196

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

As discussed, under a cognitive mapping framework, state-space construction is cast as infer-
ence over compositional structure [12]; agents do not construct representations of new environ-
ments de novo but instead reuse (compose) representational building blocks (primitives) from a
pre-existing repertoire [9,10,12]. Compositional primitives come prepackaged with information
about how states are related (i.e., abstracted structure) and state�action values that bias behavior
(i.e., pre-credit assigned) [9,12] and may themselves be composed of yet simpler primitives
[12,44,117,118] (perhaps discovered through replay [9,44]). Thus, an agent�s task representation
(posterior distribution over structure) will be exquisitely sensitive to both the content and probabil-
ity weighting of their repertoire of primitives (priors) (see Box 2 for connections to hierarchical
Bayesian accounts of psychopathology).

Recent work on representation learning in arti?cial neural networks con?rms that this representa-
tional repertoire is in?uenced by the model�s training data. Within meta-RL, the distribution of
tasks an agent encounters shapes how it �learns to learn� in new domains � including heuristics
on environmental reward rates, controllability, and volatility [119]. Relatedly, in disentangled rep-
resentation learning, if an agent encounters a training environment in which two (actually distinct)
latent variables are spuriously correlated, it will be impossible to learn appropriately disentangled
neural representations of the environment�s true latent causal structure (Box 1), leading to nega-
tive transfer effects (�entangled tasks lead to entangled representations� [12]). Finally, recent work
in hierarchical concept learning demonstrates that the order in which information is presented
during learning has profound consequences for how learning unfolds and that it is easy to get
stuck in maladaptive learning traps when faced with chaotic curricula [117].

The connection to psychiatry is that model training data might serve as a useful abstraction of an
early developmental environment [119]. Indeed, multiple known early environmental risk factors
for adult psychiatric conditions � such as inconsistent caregiver behavior, poverty, pervasive bul-
lying, and the reliable co-occurrence of authority ?gures and abuse � can be cast as disruptions in
the content or statistics of this critical early environment. We speculate that such biases might
promote learning of structural heuristics and primitives (schemata) that predispose to maladap-
tive inferences in adult life (helplessness, paranoia), which, in turn, manifest as psychiatric syn-
dromes. It is plausible that this process might occur independent of any �abnormality� at the
level of neurophysiology.

The claim that early life experience biases cognitive schemata is not new [4]. Yet, the cognitive
mapping framework � by placing the question of neural representation center stage � presents
a unique avenue for theorizing about, and testing, the effects of speci?c stressors on representa-
tional structure. One starting point might be to use �toy model� arti?cial neural networks that per-
mit in silico interventions pertaining to both environmental (training data) and biological risk factors
(learning rules and network architecture) [117,120,121].

Concluding remarks
We began by re?ecting that biological psychiatry needs to advance models that are capable of
explaining how neurophysiological perturbations and environmental risk factors cause symp-
toms. We have argued that this is only possible by engaging with a mediating layer of neuro-
cognitive representation, which considers how population neural activity and environmental
statistics give rise to structured internal representations of the world (cognitive maps). Cogni-
tive maps in?uence all aspects of cognition and their study is now a major focus across neuro-
science and AI. We believe that biological psychiatry has much to gain from incorporating these
advances, both in understanding disorders such as schizophrenia as well as a myriad other
psychiatric conditions.

OPEN ACCESS

Outstanding questions
Do aberrant cognitive maps cause
psychotic symptoms? As yet, there is
scant direct evidence that neural
processes involved in cognitive map
representations are causally implicated
in the emergence of delusions, thought
disorder, or other psychotic phenomena.

Can discrete symptoms and signs
(e.g., ?xed delusions vs. thought disor-
der) be mapped to speci?c aberrant cog-
nitive mapping processes (e.g., aberrant
map representation vs. stochastic map
sampling)?

implicated

in
(presumably,

If abnormal representational capacity is
psychotic
causally
involving
symptoms
representations
with
personal meaning and the social
world), to what extent do the relevant
neural mechanisms overlap with those
identi?ed in
hippocampal
rodent
complex for spatial mapping?

concerned

Can cognitive map organization be
indexed using naturalistic behavior
alone, for example, analyzing natural
language as a trajectory through
semantic space?

How does a cognitive mapping theory of
schizophrenia differ from applications to
other psychiatric conditions (depression,
anxiety, personality disorder), which also
involve
and
relational transfer and have associations
with similar early life psychosocial risk
factors?

inferences

abnormal

can

understanding

How
of
pharmacotherapy be integrated into a
cognitive mapping framework? Do
effective drugs � despite somewhat di-
verse receptor actions � converge on
a common mechanism to stabilize neu-
ral attractor dynamics, facilitating adap-
tive updating of map representations?

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

197

OPEN ACCESS

Trends in Cognitive Sciences

We have reviewed growing behavioral and neural evidence implicating disorganized cognitive
maps in schizophrenia and have outlined a roadmap for future work that aims at an integrated ex-
planatory model of the condition, spanning an understanding of circuit-level algorithmic disruption
and the role of the early life environment alike.

The ultimate goal is that improved mechanistic understanding will drive improvements in clinical
outcomes. This might come from the development of biological interventions that restore represen-
tational capacity to disrupted neural networks, or psychological interventions that promote learning
better structural primitives. We need to acknowledge that this is likely to be a long road and there
are many open questions (see Outstanding questions). Yet � as our discussion illustrates � if we
seek to reliably predict and intervene on a system, then we must ?rst understand it.

Acknowledgments
M.M.N.�s work is funded by an NIHR Clinical Lectureship in Psychiatry (University of Oxford). Y.L.�s work is funded by the
Chinese National Science and Technology Innovation 2030 Major Program (2022ZD0205500). M.E.-G.�s work is funded
by a Wellcome Collaborator award (214314/Z/18/Z). R.A.M.�s work is funded by a Wellcome Trust Clinical Research Career
Development (224625/Z/21/Z). R.J.D. is supported by the Max Planck Society (MPS). M.M.N. thanks Rick Adams for helpful
discussions and comments on an early draft of this manuscript.

Declaration of interests
R.A.M. has received speaker or consultancy fees from Karuna, Janssen, Boehringer Ingelheim, and Otsuka, and codirects a com-

pany that designs digital resources to support treatment of mental ill health. All other authors declare no competing interests.

References

1. Craik, K. (1943) The Nature of Explanation, Cambridge University

Press

2. Miller, A. and Raison, C. (2023) Burning down the house:
reinventing drug discovery in psychiatry for the development
of targeted therapies. Mol. Psychiatry 28, 68�75

17. Whittington, J.C. et al. (2020) The Tolman-Eichenbaum ma-
chine: unifying space and relational memory through generali-
sation in the hippocampal formation. Cell 183, 1249�1263
18. Tenenbaum, J.B. et al. (2011) How to grow a mind: statistics,

structure, and abstraction. Science 331, 1279�1285

3. Nour, M.M. et al. (2022) Functional neuroimaging in psychiatry

19. Radulescu, A. et al. (2021) Human representation learning.

and the case for failing better. Neuron 110, 2524�2544

Annu. Rev. Neurosci. 44, 253�273

4. Howes, O.D. and Murray, R.M. (2014) Schizophrenia : an inte-
grated sociodevelopmental-cognitive model. Lancet 383,
1677�1687

5. Lee, M. et al. (2024) Cognitive function and variability in antipsy-
chotic drug�naive patients with ?rst-episode psychosis: a sys-
tematic review and meta-analysis. JAMA Psychiatry 81,
468�476

6. Knapp, F. et al. (2017) Planning performance in schizophrenia
patients: a meta-analysis of the in?uence of task dif?culty and
clinical and sociodemographic variables. Psychol. Med. 47,
2002�2016

7. Kerns, J.G. et al. (2008) Executive functioning component
mechanisms and schizophrenia. Biol. Psychiatry 64, 26�33
8. Barack, D.L. and Krakauer, J.W. (2021) Two views on the cog-

nitive brain. Nat. Rev. Neurosci. 22, 359�371

9. Bakermans, J.J.W. et al. (2024) Constructing future behaviour
in the hippocampal formation through composition and replay.
arXiv, Published online April 7, 2023. https://doi.org/10.1101/
2023.04.07.536053

10. Behrens, T.E.J. et al. (2018) What is a cognitive map? Organiz-
ing knowledge for ?exible behavior. Neuron 100, 490�509
11. Bellmund, J.L.S. et al. (2018) Navigating cognition: spatial

codes for human thinking. Science 362, eaat6766

12. Whittington, J.C.R. et al. (2022) How to build a cognitive map.

Nat. Neurosci. 25, 1257�1272

13. Tolman, E.C. (1948) Cognitive maps in rats and men. Psychol.

Rev. 55, 189�208

14. Harlow, H.F. (1949) The formation of learning sets. Psychol.

Rev. 56, 51�65

15. Piantadosi, S.T. and Gallistel, C.R. (2024) Formalising the role of
behaviour in neuroscience. Eur. J. Neurosci. 60, 4756�4770
16. Dolan, R.J. and Dayan, P. (2013) Goals and habits in the brain.

Neuron 80, 312�325

20. Pezzulo, G. et al. (2024) Generating meaning: active inference and
the scope and limits of passive AI. Trends Cogn. Sci. 28, 97�112
21. Moser, E.I. et al. (2008) Place cells, grid cells, and the brain�s
spatial representation system. Annu. Rev. Neurosci. 31, 69�89
22. O�Keefe, J. and Dostrovsky, J. (1971) Short communications
the hippocampus as a spatial map. Preliminary evidence from
unit activity in the freely-moving rat. Brain Res. 34, 171�175
23. O�Keefe, J. and Nadel, L. (1978) The Hippocampus as a Cogni-

tive Map, Clarendon Press

24. Wilson, M.A. and McNaughton, B.L. (1993) Dynamics of the
hippocampal ensemble code for space. Science 261,
1055�1058

25. Fyhn, M. et al. (2004) Spatial representation in the entorhinal

cortex. Science 305, 1258�1264

26. Hafting, T. et al. (2005) Microstructure of a spatial map in the

entorhinal cortex. Nature 436, 801�806

27. H�ydal, �.A. et al. (2019) Object-vector coding in the medial

entorhinal cortex. Nature 568, 400�404

28. Lever, C. et al. (2009) Boundary vector cells in the subiculum of
the hippocampal formation. J. Neurosci. 29, 9771�9777
29. Gupta, A.S. et al. (2012) Segmentation of spatial experience by
hippocampal theta sequences. Nat. Neurosci. 15, 1032�1039
30. Johnson, A. and Redish, A.D. (2007) Neural ensembles in CA3
transiently encode paths forward of the animal at a decision
point. J. Neurosci. 27, 12176�12189

31. Skaggs, W.E. et al. (1996) Theta phase precession in hippo-
campal neuronal populations and the compression of temporal
sequences. Hippocampus 6, 149�172

32. Wikenheiser, A.M. and Redish, A.D. (2015) Hippocampal theta
sequences re?ect current goals. Nat. Neurosci. 18, 289�294
33. Buzs�ki, G. (2015) Hippocampal sharp wave-ripple: a cognitive
biomarker for episodic memory and planning. Hippocampus
25, 1073�1188

198

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

34. Foster, D.J. and Wilson, M.A. (2006) Reverse replay of behav-
ioural sequences in hippocampal place cells during the awake
state. Nature 440, 680�683

35. Lee, A.K. and Wilson, M.A. (2002) Memory of sequential expe-
rience in the hippocampus during slow wave sleep. Neuron 36,
1183�1194

36. Wilson, M.A. and McNaughton, B.L. (1994) Reactivation of
hippocampal ensemble memories during sleep. Science 265,
676�679

37. Pfeiffer, B.E. and Foster, D.J. (2013) Hippocampal place-cell
sequences depict future paths to remembered goals. Nature
497, 74�79

38. Widloski, J. and Foster, D.J. (2022) Flexible rerouting of hippocam-
pal replay sequences around changing barriers in the absence of
global place ?eld remapping. Neuron 110, 1547�1558.e8

39. �lafsd�ttir, H.F. et al. (2015) Hippocampal place cells construct
reward related sequences through unexplored space. eLife 4,
1�17

40. Gupta, A.S. et al. (2010) Hippocampal replay is not a simple

function of experience. Neuron 65, 695�705

41. Barron, H.C. et al. (2020) Neuronal computation underlying in-
ferential reasoning in humans and mice. Cell 183, 228�243.e21
42. Stella, F. et al. (2019) Hippocampal reactivation of random tra-
jectories resembling Brownian diffusion. Neuron 102, 450�461
43. Dragoi, G. and Tonegawa, S. (2011) Preplay of future place cell
sequences by hippocampal cellular assemblies. Nature 469,
397�401

44. Kurth-Nelson, Z. et al. (2023) Replay and compositional com-

putation. Neuron 111, 454�469

45. Dusek, J.A. and Eichenbaum, H. (1997) The hippocampus
and memory for orderly stimulus relations. Proc. Natl. Acad.
Sci. U. S. A. 94, 7109�7114

46. Bunsey, M. and Eichenbaum, H. (1996) Conservation of hippo-
campal memory function in rats and humans. Nature 379,
255�257

47. Schwartenbeck, P. et al. (2023) Generative replay underlies
compositional inference in the hippocampal-prefrontal circuit.
Cell 186, 4885�4897.e14

64. Wilson, R.C. et al. (2014) Orbitofrontal cortex as a cognitive

map of task space. Neuron 81, 267�279

65. Schuck, N.W. et al. (2016) Human orbitofrontal cortex repre-
sents a cognitive map of state space. Neuron 91, 1402�1412
66. Wikenheiser, A.M. et al. (2017) Suppression of ventral hippo-
campal output impairs integrated orbitofrontal encoding of
task structure. Neuron 95, 1197�1207.e3

67. Zaremba, J.D. et al. (2017) Impaired hippocampal place cell
dynamics in a mouse model of the 22q11.2 deletion. Nat.
Neurosci. 20, 1612�1623

68. Retailleau, A. and Morris, G. (2018) Spatial rule learning and
corresponding CA1 place cell reorientation depend on local
dopamine release. Curr. Biol. 28, 836�846.e4

69. Logothetis, N.K. et al. (2012) Hippocampal-cortical interaction
during periods of subcortical silence. Nature 491, 547�553
70. Liu, X. et al. (2021) Multimodal neural recordings with Neuro-
FITM uncover diverse patterns of cortical�hippocampal
interactions. Nat. Neurosci. 24, 886�896

71. Higgins, C. et al. (2021) Replay bursts in humans coincide with
activation of the default mode and parietal alpha networks.
Neuron 109, 882�893

72. Kaplan, R. et al. (2016) Hippocampal sharp-wave ripples in?u-
ence selective activation of the default mode network. Curr.
Biol. 26, 686�691

73. Huang, Q. et al. (2024) Replay-triggered brain-wide activation in

humans. Nat. Commun. 15, 7185

74. Yeshurun, Y. et al. (2021) The default mode network: where the
idiosyncratic self meets the shared social world. Nat. Rev.
Neurosci. 22, 181�192

75. Margulies, D.S. et al. (2016) Situating the default-mode network
along a principal gradient of macroscale cortical organization.
Proc. Natl. Acad. Sci. U. S. A. 113, 12574�12579

76. Hahamy, A. et al. (2023) The human brain reactivates context-
speci?c past information at event boundaries of naturalistic
experiences. Nat. Neurosci. 26, 1080�1089

77. Baldassano, C. et al. (2017) Discovering event structure in contin-
uous narrative perception and memory. Neuron 95, 709�721.e5
78. Morris, R.W. et al. (2018) Impairments in action-outcome learn-

48. Liu, Y. et al. (2019) Human replay spontaneously reorganizes

ing in schizophrenia. Transl. Psychiatry 8, 54

experience. Cell 178, 640�652

49. Bao, X. et al. (2019) Grid-like neural representations support
olfactory navigation of a two-dimensional odor space. Neuron
102, 1066�1075.e5

50. Aronov, D. et al. (2017) Mapping of a non-spatial dimension by
the hippocampal-entorhinal circuit. Nature 543, 719�722
51. Omer, D.B. et al. (2018) Social place-cells in the bat hippocampus.

Science 359, 218�224

52. Garvert, M.M. et al. (2017) A map of abstract relational knowledge
in the human hippocampal�entorhinal cortex. eLife 6, 1�20
53. Schapiro, A.C. et al. (2013) Neural representations of events arise
from temporal community structure. Nat. Neurosci. 16, 486�492
54. Baram, A.B. et al. (2021) Entorhinal and ventromedial prefrontal
cortices abstract and generalize the structure of reinforcement
learning problems. Neuron 109, 713�723

55. Doeller, C.F. et al. (2010) Evidence for grid cells in a human

memory network. Nature 463, 657�661

56. Horner, A.J. et al. (2016) Grid-like processing of imagined

navigation. Curr. Biol. 26, 842�847

57. Constantinescu, A.O. et al. (2016) Organizing conceptual knowl-
edge in humans with a gridlike code. Science 352, 1464�1468
58. Park, S.A. et al. (2021) Inferences on a multidimensional social

hierarchy use a grid-like code. Nat. Neurosci. 24, 1�13

59. Schuck, N.W. and Niv, Y. (2019) Sequential replay of non-spatial
task states in the human hippocampus. Science 364, eaaw5181
60. Kurth-Nelson, Z. et al. (2016) Fast sequences of non-spatial

state representations in humans. Neuron 91, 194�204

61. Samborska, V. et al. (2022) Complementary task representa-
tions in hippocampus and prefrontal cortex for generalizing
the structure of problems. Nat. Neurosci. 25, 1314�1326
62. Barron, H.C. et al. (2013) Online evaluation of novel choices
by simultaneous representation of multiple memories. Nat.
Neurosci. 16, 1492�1498

63. El-Gaby, M. et al.

(2024) A cellular basis for mapping
behavioural structure. Nature, https://www.nature.com/articles/
s41586-024-08145-x

79. Culbreth, A.J. et al. (2016) Reduced model-based decision-
making in schizophrenia. J. Abnorm. Psychol. 125, 777�787
80. Titone, D. et al. (2004) Transitive inference in schizophrenia: impair-
ments in relational memory organization. Schizophr. Res. 68,
235�247

81. Adams, R.A. et al. (2020) Impaired theta phase coupling under-
lies frontotemporal dysconnectivity in schizophrenia. Brain 143,
1261�1277

82. Armstrong, K. et al. (2012) Impaired associative inference in
patients with schizophrenia. Schizophr. Bull. 38, 622�629
83. Nour, M.M. et al. (2021) Impaired neural replay of inferred rela-

tionships in schizophrenia. Cell 184, 4315�4328.e17

84. Thai, M.L. et al. (2019) A meta-analysis of executive dysfunction
in patients with schizophrenia: different degree of impairment in
the ecological subdomains of the Behavioural Assessment of
the Dysexecutive Syndrome. Psychiatry Res. 272, 230�236
85. Fradkin, I. and Eldar, E. (2023) Accumulating evidence for myr-
iad alternatives: modeling the generation of free association.
Psychol. Rev. 130, 1492�1520

86. Lundin, N.B. et al. (2023) Neural evidence of switch processes
during semantic and phonetic foraging in human memory.
Proc. Natl. Acad. Sci. U. S. A. 120, e2312462120

87. Hills, T.T. et al. (2012) Optimal foraging in semantic memory.

Psychol. Rev. 119, 431�440

88. Solomon, E.A. et al. (2019) Hippocampal theta codes for dis-
tances in semantic and temporal spaces. Proc. Natl. Acad.
Sci. U. S. A. 116, 24343�24352

89. Corcoran, C.M. and Cecchi, G.A. (2020) Using language process-
ing and speech analysis for the identi?cation of psychosis and other
disorders. Biol. Psychiatry Cogn. Neurosci. Neuroimaging 5,
770�779

90. Nour, M. et al. (2023) Trajectories through semantic spaces in
schizophrenia and the relationship to ripple bursts. Proc. Natl.
Acad. Sci. U. S. A. 120, e2305290120

91. Suh, J. et al. (2013) Impaired hippocampal ripple-associated re-
play in a mouse model of schizophrenia. Neuron 80, 484�493

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

199

OPEN ACCESS

Trends in Cognitive Sciences

92. Altimus, C. et al. (2015) Disordered ripples are a common
to
feature of genetically distinct mouse models relevant
schizophrenia. Mol. Neuropsychiatry 1, 52�59

93. Nour, M.M. et al. (2023) Reduced coupling between of?ine
neural replay events and default mode network activation in
schizophrenia. Brain Commun. 5, fcad056

94. Nour, M. et al. (2022) Relationship between replay-associated
ripple power and hippocampal NMDA receptor binding in
schizophrenia. Schizophr. Bull. Open 3, sgac044

95. Dupret, D. et al. (2010) The reorganization and reactivation of
hippocampal maps predict spatial memory performance. Nat.
Neurosci. 13, 995�1002

96. Krystal, J.H. et al. (2017) Impaired tuning of neural ensembles
and the pathophysiology of schizophrenia: a translational and
computational neuroscience perspective. Biol. Psychiatry 81,
874�885

97. Rolls, E.T. et al. (2008) Computational models of schizophrenia
and dopamine modulation in the prefrontal cortex. Nat. Rev.
Neurosci. 9, 696�709

98. Sigurdsson, T. et al. (2010) Impaired hippocampal�prefrontal
synchrony in a genetic mouse model of schizophrenia. Nature
464, 763�767

99. Dickerson, D.D. et al. (2010) Abnormal

long-range neural
synchrony in a maternal immune activation animal model of
schizophrenia. J. Neurosci. 30, 12424�12431

100. Convertino, L. et al. (2022) Reduced grid-like theta modulation

in schizophrenia. Brain 146, 2191�2198

112. Munn, R.G.K. et al. (2023) Disrupted hippocampal synchrony fol-
lowing maternal immune activation in a rat model. Hippocampus
33, 995�1008

113. Winship, I.R. et al. (2019) An overview of animal models related

to schizophrenia. Can. J. Psychiatr. 64, 5�17

114. Speers, L.J. et al. (2021) Hippocampal sequencing mecha-
nisms are disrupted in a maternal immune activation model of
schizophrenia risk. J. Neurosci. 41, 6954�6965

115. Mesbah-Oskui, L. et al. (2015) Hippocampal place cell and in-
hibitory neuron activity in disrupted-in-schizophrenia-1 mutant
mice: implications for working memory de?cits. NPJ Schizophr.
1, 1�7

116. Petrovic, P. and Sterzer, P. (2023) Resolving the delusion para-

dox. Schizophr. Bull. 49, 1425�1436

117. Zhao, B. et al. (2023) A model of conceptual bootstrapping in

human cognition. Nat. Hum. Behav. 8, 125�136

118. Hofstadter, D. (2001) Analogy as the core of cognition. In The
Analogical Mind: Perspectives from Cognitive Science
(Gentner, D. et al., eds), pp. 499�538, MIT Press

119. Nussenbaum, K. and Hartley, C.A. (2024) Understanding the
development of reward learning through the lens of meta-
learning. Nat. Rev. Psychol. 3, 424�438

120. Whittington, J.C.R. et al. (2023) Disentangling with biological con-
straints: a theory of functional cell types. arXiv, Published online
March 31, 2023. https://doi.org/10.48550/arXiv.2210.01768
121. Doerig, A. et al. (2023) The neuroconnectionist research

programme. Nat. Rev. Neurosci. 24, 431�450

101. Brugger, S.P. and Howes, O.D. (2017) Heterogeneity and
homogeneity of regional brain structure in schizophrenia a
meta-analysis. JAMA Psychiatry 74, 1104�1111

122. Kay, S. et al. (1987) The positive and negative syndrome scale
(PANSS) for schizophrenia. Schizophr. Bull. 13, 261�276
123. Hall, A.F. et al. (2024) The computational structure of consum-

102. Radhakrishnan, R. et al. (2021) In vivo evidence of lower synaptic
vesicle density in schizophrenia. Mol. Psychiatry 26, 7690�7698
103. McHugo, M. et al. (2019) Hyperactivity and reduced activation
of anterior hippocampus in early psychosis. Am. J. Psychiatry
176, 1030�1038

104. Howes, O.D. and Onwordi, E.C. (2023) The synaptic hypothesis
of schizophrenia version III: a master mechanism. Mol. Psychiatry
28, 1843�1856

matory anhedonia. Trends Cogn. Sci. 28, 541�553

124. Ritunnano, R. et al. (2022) Subjective experience and meaning
of delusions in psychosis: a systematic review and qualitative
evidence synthesis. Lancet Psychiatry 9, 458�476

125. Sterzer, P. et al. (2018) The predictive coding account of

psychosis. Biol. Psychiatry 84, 634�643

126. Gomperts, S.N. et al. (2015) VTA neurons coordinate with the

hippocampal reactivation of spatial experience. eLife 4, 1�22

105. Onwordi, E.C. et al. (2020) Synaptic density marker SV2A is
reduced in schizophrenia patients and unaffected by antipsy-
chotics in rats. Nat. Commun. 11, 246

127. McNamara, C.G. et al. (2014) Dopaminergic neurons promote
hippocampal reactivation and spatial memory persistence.
Nat. Neurosci. 17, 1658�1660

106. Howes, O.D. and Shatalina, E.

Integrating the
neurodevelopmental and dopamine hypotheses of schizophre-
nia and the role of cortical excitation-inhibition balance. Biol.
Psychiatry 92, 501�513

(2022)

107. Hamm, J.P. et al. (2017) Altered cortical ensembles in mouse

models of schizophrenia. Neuron 94, 153�167.e8

108. Hamm, J.P. et al. (2020) Aberrant cortical ensembles and
schizophrenia-like sensory phenotypes in Setd1a+/? mice.
Biol. Psychiatry 88, 215�223

128. Bernardi, S. et al. (2020) The geometry of abstraction in the hip-

pocampus and prefrontal cortex. Cell 183, 954�967

129. Barron, H.C. et al. (2016) Repetition suppression: a means to
index neural representations using BOLD? Philos. Trans. R.
Soc. B Biol. Sci. 371, 20150355

130. Diedrichsen, J. and Kriegeskorte, N. (2017) Representational
models: a common framework for understanding encoding,
pattern-component, and representational-similarity analysis.
PLoS Comput. Biol. 13, 1�33

109. Musa, A. et al. (2022) The shallow cognitive map hypothesis : a
hippocampal framework for thought disorder in schizophrenia.
Schizophrenia 8, 34

131. Liu, Y. et al. (2021) Temporally delayed linear modelling
(TDLM) measures replay in both animals and humans. eLife
10, e66917

110. Hauser, T.U. et al. (2016) Computational psychiatry of ADHD:
neural gain impairments across Marrian levels of analysis.
Trends Neurosci. 39, 63�73

132. Wittkuhn, L. and Schuck, N.W. (2021) Dynamics of fMRI pat-
terns re?ect sub-second activation sequences and reveal replay
in human visual cortex. Nat. Commun. 12, 1795

111. Adams, R.A. et al. (2018) Attractor-like dynamics in belief
updating in schizophrenia. J. Neurosci. 38, 9471�9485

133. Roads, B.D. and Love, B.C. (2024) The dimensions of dimen-

sionality. Trends Cogn. Sci. 28, 1118�1131

200

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2


