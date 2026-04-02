Trends in
Cognitive Sciences

OPEN ACCESS

Opinion
Large-scale interactions in predictive
processing: oscillatory versus transient
dynamics

Martin Vinck 1,2,*, Cem Uran 1,2,*, Jarrod R. Dowdall 3, Brian Rummell 1, and Andres Canales-Johnson 4,5,*

How do the two main types of neural dynamics, aperiodic transients and oscil-
lations, contribute to the interactions between feedforward (FF) and feedback
(FB) pathways in sensory inference and predictive processing? We discuss
three theoretical perspectives. First, we critically evaluate the theory that
gamma and alpha/beta rhythms play a role in classic hierarchical predictive
coding (HPC) by mediating FF and FB communication, respectively. Second,
we outline an alternative functional model in which rapid sensory inference is
mediated by aperiodic transients, whereas oscillations contribute to the stabi-
lization of neural representations over time and plasticity processes. Third, we
propose that the strong dependence of oscillations on predictability can be ex-
plained based on a biologically plausible alternative to classic HPC, namely
dendritic HPC.

Theories of neural dynamics and their role in predictive processing
To construct an internal model of the environment, the brain performs inference on the statistical
nature of its inputs by integrating sensory evidence with prior knowledge [1,2]. Various theories ex-
plain how sensory inference is implemented by local recurrent networks and interactions between
FF and FB (see Glossary) pathways [1�5]. For example, classic HPC theory posits that FF path-
ways carry sensory prediction errors, whereas FB pathways convey prediction signals [1,2]. In
this Opinion article we compare three theoretical proposals for how transient dynamics and oscil-
lations facilitate these interactions.

First, we consider the theory that prediction errors are transmitted through gamma rhythm
(30�80 Hz) oscillations [6,7]), whereas prediction signals are transmitted through alpha
rhythm and beta rhythm (10�20 Hz) oscillations [8�15]. We critically evaluate this theory by
examining the relationship between (i) functional, structural, and effective connectivity, and
(ii) neural rhythms and stimulus predictability.

Second, we propose an alternative functional model which implies dual roles for transient dynamics
and oscillations. In this model, rapid sensory inference relies on communication during aperiodic
transients, whereas rhythms are involved in stabilizing neural representations and facilitating
plasticity during the FB-dominated late phase of stimulus processing.

Third, we suggest that the emergence of rhythmic synchronization for stimuli with high spatio-
temporal predictability can be explained by a biologically plausible alternative to classic HPC,
namely dendritic HPC [3]. This mechanistic model is compatible with the proposed alternative
functional model.

Highlights
We contrast the roles of two main types
of neural dynamics, namely transients
and oscillations, in predictive processing
and sensory inference.

We propose that oscillations stabilize
neural representations over time and fa-
cilitate plasticity processes during the
late, feedback-dominated phase of sen-
sory processing.

Oscillations emerge for sensory inputs
with high spatiotemporal predictability,
which ?ts better with dendritic rather
than classic hierarchical predictive cod-
ing principles.

Based on recent evidence, we critically
evaluate the theory that gamma and
alpha/beta rhythms carry prediction
error and prediction signals, respectively.

For instance, we argue that unpredicted
stimuli enhance broadband ?uctuations
and aperiodic transients, whereas
predicted stimuli boost narrow-band
gamma oscillations.

Finally, based on the speed of cortical
processing, we argue that transient,
non-oscillatory dynamics are the main
conduit for inter-areal communication
during sensory inference.

1Ernst Str�ngmann Institute (ESI) for
Neuroscience, in Cooperation with the
Max Planck Society, 60528 Frankfurt am
Main, Germany
2Donders Centre for Neuroscience, De-
partment of Neurophysics, Radboud
University, 6525 Nijmegen, The Nether-
lands
3Robarts Research Institute, Western
University, London, ON, Canada

https://doi.org/10.1016/j.tics.2024.09.013
� 2024 The Authors. Published by Elsevier Ltd. This is an open access article under the CC BY-NC license (http://creativecommons.org/licenses/by-nc/4.0/).

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

133

OPEN ACCESS

Trends in Cognitive Sciences

Rhythms for FF and FB communication?
The theory that gamma and alpha/beta rhythms underlie FF and FB communication ('gamma-FF/
alpha-beta-FB') relies on three fundamental assertions: (i) gamma and alpha/beta oscillations
are widespread and are observed in all brain areas [16,17]; (ii) there is a consistent relationship
between gamma and alpha/beta in?uences and anatomical connectivity [10,18]; and (iii) the
functional interactions between brain areas, namely predictions and errors, are reliably and
accurately captured by methods such as Granger causality in?uences between local field
potential (LFP) signals.

4Facultad de Ciencias de la Salud,
Universidad Catolica del Maule,
3480122 Talca, Chile
5Department of Psychology, University
of Cambridge, Cambridge CB2 3EB, UK

*Correspondence:
martin.vinck@esi-frankfurt.de (M. Vinck),
cem.uran@esi-frankfurt.de (C. Uran), and
afc37@cam.ac.uk (A. Canales-Johnson).

In light of contrasting evidence for the three assertions above, we propose an alternative model,
referred to as 'frequency-speci?c networks', which is consistent with empirical studies showing
that (i) gamma and alpha/beta rhythms are not widespread, but occur in distinct cortical
networks; (ii) the relationship between FF/FB anatomical connectivity and gamma-FF/alpha-
beta-FB Granger-causal in?uences is not consistent across brain areas; and (iii) Granger-causal
in?uences between LFP signals do not re?ect the frequency-speci?c transmission of predictions
and errors, but instead represent the unique power spectral signature of each cortical network.

How widespread are gamma and alpha/beta oscillations?
Understanding the ubiquity and breadth of these oscillations requires invasive electrophysiologi-
cal recordings from multiple areas, ideally simultaneously and in a manner that minimizes volume
conduction. Few datasets satisfy these criteria, but these data suggest highly localized oscillatory
networks. For instance, in one set of electrocorticography (ECoG) recordings from 15 areas
(occipital, parietal, and frontal) in macaque, only four of the 15 areas showed strong narrow
band [19] gamma peaks in the LFP power spectra. These peaks were most prominent in
areas V1 and V2 [20], and gamma-band Granger causality in?uences were most prevalent be-
tween pairs of areas which included V1 or V2 [14]. These observations are supported by another
dataset of 55 macaque areas, which reported that narrow-band gamma oscillations in LFP were
con?ned to areas V1 and V2 [21]. In addition, several studies have reported particularly strong
gamma synchronization in areas V1 and V2 [21�24].

Similarly, several multi-area datasets of invasive recordings in macaques have shown a localized
beta-band network (a 'beta-core') [14,20,21,25]. Beta-band Granger-causal in?uences were
mainly associated with parietal regions such as 7A, S1, area 5, and motor cortex [14], and LFP
power spectra showed clear alpha and beta peaks in parietal and frontal but not occipital areas
[20,21]. These ?ndings are consistent with previous reports of strong beta oscillations in the so-
matosensory network, which were especially prominent during the delay periods of sensorimotor
tasks [15,21,25�27].

It has been argued that the LFP power spectral analyses might fail to reveal the presence of
narrow-band rhythmic synchronization that would otherwise be evident in the spike�LFP cou-
pling [28]. On the contrary, we ?nd that the results of spike�LFP coupling are often consistent
and agree with the LFP power spectra. For example, spike�LFP coupling in parietal cortex
showed beta but not gamma synchronization, which matched the LFP power spectra [29]. More-
over, simultaneous V1 and V4 recordings have revealed locally generated gamma synchroniza-
tion in V1 but not in V4, which was consistent with the LFP spectra [22]. In fact, LFP power
spectra may be prone to overestimate the prevalence of gamma/beta oscillatory dynamics
across brain areas as a result of volume conduction [29�31].

Taken together, the evidence is more in line with the frequency-speci?c network hypothesis rather
than with discrete gamma-FF/alpha-beta-FB communication channels. We note that the

134

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

reported differences in narrow-band oscillations between areas suggest at least a major differ-
ence in oscillation strength between regions, although they may also suggest a genuine absence
of some oscillations in particular areas.

The consistency of the gamma-FF/alpha-beta-FB pattern with anatomical connectivity
The gamma-FF/alpha-beta-FB hypothesis was originally motivated by the proposal that gamma
and alpha/beta synchronization are generated in FF projecting supragranular and FB projecting
infragranular layers, respectively [10,32]. However, later studies challenged the gamma-supra/
alpha-beta-infra hypothesis by showing that gamma and alpha/beta can be equally prominent
in both infra- and supragranular layers (Box 1).

Box 1. Gamma supragranular, alpha/beta infragranular?

It has been proposed that gamma and alpha/beta oscillations are respectively generated in supragranular and
infragranular layers, the primary sources of feedforward (FF) and feedback (FB) projections [33]. However, we argue that
the evidence does not suggest a clear laminar separation between rhythms (cf [111]). Thus, we argue that the main source
of variability in rhythms is the cortical area (e.g., V1 vs S1) rather than the cortical layer.

Recordings with laminar probes and/or post-mortem histological veri?cation indicate that both supra- and infragranular
layers of macaque V1 contain a source of gamma-band synchronization in spikes and local ?eld potentials (LFPs)
[112�114] (Figure IA). Although a recent study ?nds stronger LFP gamma power in supragranular as compared to
infragranular layers of many macaque areas [16], the LFP spectra indicate broadband rather than narrow-band gamma
oscillations (Figure IC). Laminar differences in broadband LFP power can be explained by, for example, differences in
the slope of power spectra as a result of dendritic ?ltering [43,115]. An earlier study suggested strong gamma synchroni-
zation in the supragranular layers of V1, V2, and V4, but not in infragranular layers [32]. However, this study was based on
inserting individual electrodes at different depths, which lacks the accuracy of laminar and histological techniques, and
could have included regions such as L4 which may exhibit weak gamma [112�114].

Studies have drawn different conclusions regarding the laminar distribution of alpha/beta rhythms. Earlier work found
a dominant alpha source in the infragranular layers of early visual areas [116]. Even stronger alpha power in
infragranular layers may re?ect the alignment of signals to infragranular alpha before computing the current source
density (CSD) [117,118]. A recent study reports a cortically widespread pattern of stronger unipolar LFP alpha/beta
power [16,45]. However, opposite conclusions based on the same data are reached when using bipolar derivations
or CSD signals (Figure S1 in [119] andFigure I C; cf [16]). Differences between unipolar LFP pro?les may not re?ect
the strength of rhythms but could re?ect passive dendritic ?ltering and local volume conduction [43]. An analysis of
CSDs and spiking signals in laminar recordings from human cortex suggests that the alpha rhythm can be equally or
more prominent in supragranular than in infragranular layers [117,118] (Figure IB). Finally, studies suggest that LFP
signals show beta rhythmicity across all layers [120�122], and one study reported relatively strong beta transients
in supragranular CSDs [122].

(A)

L2/3

L4

L5/6

0

40

80

Frequency (Hz)

(B)
L1

L2

L3

L4

CSD

L5
L6

0
�250
Time relative to alpha sink (ms)

250

l

r
a
o
p
n
u

i

,
r
e
w
o
p

g
o
L

(C)

Alpha/beta

Broadband/
�gamma�

Unipolar

Bipolar
/CSD

4

3

2

1

1

L2/3
L5/6

Power
L2/3

Power
L5/6

10

Frequency

100

Trends inin Cognitive
Trends

Sciences
Cognitive Sciences

Figure I. Relationship of layers to cortical rhythms. (A) Laminar recordings from macaque V1 during an attention
task. A gamma source is present in both supra- and infragranular layers (adapted, with permission, from [112]). (B) Unit
recordings of human alpha during eye closure show strongest spike phase locking in L3 (adapted, with permission,
from [117]). (C) LFP power spectra in macaque prefrontal cortex (adapted, with permission, from [45]). Note that the
reported gamma effect comprises broadband ?uctuations. The reversed pattern of alpha/beta and high-frequency
power is speci?c to unipolar recordings but is not found in CSD and bipolar recordings, as illustrated by the bar plot on
the right that visualizes the difference between unipolar and bipolar/CSDs [16,111,119].

OPEN ACCESS

Glossary
Alpha rhythm: 8�14 Hz rhythms that
are found in visual, somatosensory (mu
rhythm), and auditory (tau rhythm)
cortices during wakeful rest.
Beta rhythm: a 20 Hz rhythm
commonly observed in parietal and (pre)
motor areas, mainly during the delay
period of sensorimotor tasks. A distinct
higher-frequency beta rhythm is found in
(pre)frontal areas and increases during
the delay periods and post-trial epochs
of working memory and attention tasks.
Broadband: refers to changes in power
or functional connectivity occurring in a
wide frequency range, for example as a
result of the in?uence of ?ring rates.
Directed asymmetry index (DAI):
quanti?es the asymmetry of LFP
Granger-causal in?uences (as [GCout ?
GCin] / [GCin + GCout]). Positive DAI �
SLN correlations indicate an association
of structural FF connectivity with
Granger-causal in?uences in that band.
Electrocorticography (ECoG): also
referred to as iEEG, a procedure that
involves intracranial measurements of
cortical surface ?eld potentials.
Feedforward (FF) and feedback
(FB): anatomically speaking, a
projection from a lower to a higher brain
area (FF), or from a higher to a lower area
(FB). Hierarchy can be de?ned in multiple
ways, such as through cortical
gradients, anatomical connectivity, or
functional response properties.
Gamma rhythm: synchronized spiking
in the 30�80 Hz range that is usually
associated with active processing during
awake states. These rhythms are
prominent in hippocampus, visual
cortex, and olfactory cortex.
Granger causality: a measure of
causal in?uences between stochastic
signals. Granger causality in the
frequency domain is commonly de?ned
as the fraction of explained power in a
signal Y by observing X, after
discounting the information that Y has
about itself from the past.
Local field potential (LFP): electrical
potential differences measured
extracellularly in the lower-frequency
range (0�200 Hz). At lower frequencies,
the LFP primarily measures volume
return currents caused by spatially and
temporally coherent transmembrane
currents. At higher frequencies, the LFP
also contains a direct contribution of
spikes.
Narrow band: refers to changes in
power or functional connectivity

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

135

OPEN ACCESS

Trends in Cognitive Sciences

occurring in a narrow frequency range,
which may re?ect a change in the
amplitude of a rhythm. Narrow-band
oscillations are characterized by spectral
peaks and synchronization of spiking
activity.
Supragranular labeled neurons
(SLNs): at the anatomical level the % of
SLNs in area X projecting to area Y gives
a good indication of hierarchical
distance. These allow each cortical area
to be placed on a hierarchy, which
agrees very well with descriptions of FF
connectivity based on projections to
layer IV.
Variational autoencoder: a machine
learning architecture in which an input
(e.g., image) x is encoded via some
latent variable space z, with a density
q(z|x). For static inputs, the mapping of x
onto q(z|x) (i.e., inference) can be
performed with a FF neural network.
Starting from the latent space q(z|x), the
posterior p(x|z) can be computed
(i.e., the negative free energy), which
requires a generative neural network.

A direct test of the gamma-FF/alpha-beta-FB hypothesis requires large-scale, invasive record-
ings of neural activity, thereby precluding volume conduction. Only a few studies have analyzed
such data. For instance, one study [10] analyzed a unique dataset of ECoG recordings from
eight macaque visual areas and computed the correlation between the inter-areal anatomical
connectivity as re?ected by supragranular labeled neurons (SLNs) [33] and the LFP
Granger-causal in?uences quanti?ed by the directed asymmetry index (DAI) (a similar ap-
proach utilizing human magnetoencephalography data is described in [18]). The correlation co-
ef?cients the authors reported were in the range of ?0.2 for beta (~20 Hz) and +0.4 for the
gamma range [10]. Although this study can be viewed as evidence for a consistent relationship
between gamma-FF/alpha-beta-FB and anatomical connectivity, we note that correlation co-
ef?cients of ?0.2 and +0.4 indicate that SLNs explain 4% and 16% of the beta and gamma
Granger-causal in?uences, respectively. Moreover, further examination of these results re-
vealed that the correlations in each band are largely driven by a few pairs of areas that have par-
ticularly strong gamma or beta Granger-casual in?uences (i.e., V1 and V2 for gamma, and
posterior parietal areas for beta; see Figure I in Box 2). Furthermore, several studies have
found FF beta, which challenges the assertion that beta rhythms uniquely re?ect FB communi-
cation [25,26]. We argue that these results are not exceptions to the rule but point to a different
rule which is consistent with frequency-speci?c networks.

Is effective communication mediated by gamma and alpha/beta synchronization?
One may counter that the gamma-FF/alpha-beta-FB pattern may still be functionally relevant
for a small subset of area pairs. We contend that this argument hinges on the ability of
Granger causality to accurately capture inter-areal
interactions. A recent computational
framework (coherence through communication, CTCOM) challenges this assumption and
suggests that the Granger-causal in?uence from a sender to a receiver is simply a function
of the oscillatory power in the sender. That is, the Granger-causal interaction can simply be
the result of the correlation of the sender with its own projected inputs to the receiver, rather
resonance between the sender and receiver
than resulting from entrainment or
[29,30,34�36]. According to CTCOM, Granger-causal
in?uences between LFPs are thus
not necessarily functionally relevant but can emerge simply as a result of connectivity and dif-
ferences in oscillatory power. The question that Granger-causal analysis cannot answer is the
extent to which gamma and alpha/beta oscillatory inputs are effective at driving the spiking
activity of neurons in postsynaptic target areas.

The dominant view of spike synchronization is that it enhances the impact on postsynaptic tar-
gets [7,37,38]. Importantly, however, when synchronization is con?ned to a narrow-frequency
band, its effects necessarily depend on the resonance and ?ltering properties unique to each
neuron type [34,39]. For example, excitatory neurons exhibit strong low-pass ?ltering that
may render afferent, high-frequency inputs ineffective at driving these neurons [40]. Indeed, re-
cent work suggests that gamma rhythms preferentially activate fast-spiking inhibitory interneu-
rons rather than excitatory neurons in downstream areas [22,40]. For instance, although robust
LFP�LFP gamma coherence is observed between V1 and V4 [22,41], gamma rhythms in V1
are correlated with the spikes of fast-spiking interneurons in the input layer (L4) of V4 but not
of excitatory neurons [22]. These observations appear to be consistent across various cortical
systems, including the hippocampus [42], and suggest that gamma-band FF communication is
probably inhibitory (i.e., FF inhibition) rather than excitatory. A similar argument can be made
concerning beta-frequency FB. FB predominantly arrives at the apical dendrites of pyramidal
neurons, which show substantial low-pass ?ltering that can severely dampen synaptic poten-
tials at beta frequencies [43,44]. It is possible that top-down beta FB may therefore predomi-
nantly drive GABAergic interneurons in lower areas.

136

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

Box 2. Gamma-feedforward, alpha/beta-feedback?

A study by Bastos et al. [10] is cited as key evidence for the gamma-FF/alpha-beta-FB hypothesis. Bastos et al. se-
lected eight of 15 areas for analysis and computed correlations across 56 datapoints (i.e., 7 � 8 area pairs; each area
pair contributed two highly correlated datapoints). We argue that the gamma-FF/alpha-beta-FB pattern is not consis-
tent within the set of analyzed areas because (i) Bastos et al.[10] do not provide direct statistical evidence for a consis-
tent pattern across area pairs, because the signi?cance of correlations is tested across subsets of trials (via
bootstrapping), not area pairs. (ii) Their reported correlations are possibly in?ated because each area contributes to
a total of 14 (i.e., 25% of 56) area pairs, violating the assumption of statistically independent observations. Conse-
quently, a single area can easily drive correlations. (iii) The reported supragranular labeled neuron (SLN) � directed
asymmetry index (DAI) correlations are weak to moderate, suggesting only a loose relationship between structural
and functional connectivity. This is unlikely to be due to measurement noise because the correlations and Granger-
causality spectra were highly consistent across subsets of trials. (iv) Crucially, examination of the individual area pairs
shown by Bastos et al.[10] (their Figure S3) suggests the FF-gamma/alpha-beta-FB pattern is driven by a small number
of areas. In particular, the correlations between SLN and gamma DAI seem to be entirely driven by V1 and V2, which
contribute to 50% (i.e., 28/56) of all area pairs. Likewise, the relationship between beta DAI and anatomical connectivity
seems to be driven by areas around the parietal cortex [dorsal prelunate (DP) and 7A] in which beta is strong (Figure IA)
[119]. Because gamma and beta are prevalent in distinct networks, only very few anatomically connected area pairs
show both the gamma-FF and beta-FB patterns. (v) The selection of areas included for analysis can strongly bias
the results. Bastos et al.[10] did not include parietal and frontal areas in which beta is most prominent; however, other
work suggests that this would have yielded many FF beta in?uences [25,26,33].

In sum, there appears to be no general and consistent gamma-FF/alpha-beta-FB pattern within the visual system, even
though gamma oscillations may be characteristic of lower visual areas and beta oscillations of visual areas occupying in-
termediate hierarchical positions (e.g., DP and 7A). Furthermore, in the reviewed large-scale recording studies, gamma ap-
pears to be absent among somatosensory and motor areas [20,21,33], while beta appears to be associated with areas
lower in the hierarchy [25,26]. This characterization is likely premature because much work will be necessary to precisely
characterize the beta network. Moreover, surface ECoG recordings, as used in Bastos et al.[10], do not necessarily re?ect
local spiking synchronization [29], and it remains to be tested whether beta in DP and 7A LFPs re?ects intrinsic synchro-
nization or activity from nearby areas such as 5, 7B, and S1 (e.g., via volume conduction or synaptic inputs). Furthermore,
we note that some magnetoencephalography studies did report narrow-band gamma in somatosensory/motor cortex
[23,123,124], while other studies using invasive recordings in monkey/human somatosensory and motor cortex appear
to show broadband ?uctuations rather than narrow-band gamma [125�129].

(A)

x
e
d
n

i

y
r
t
e
m
m
y
s
a
d
e
t
c
e
r
i

D

Gamma

+0.5

+0.5

Beta

7A

8M

DP

�0.5

100

0
% Supragranular
projection neurons
With V1 and V2
Without V1 and V2

TEO

�0.5

V4

8L

0
% Supragranular
projection neurons

With DP and 7A
Without DP and 7A

8M

TEO

V4

8L

V2

V1

(B)

Feedforward
beta

Beta core

S1 7A

DP

Feedback
beta

Trends inin Cognitive
Trends

Sciences
Cognitive Sciences

Figure I. Consistency and interpretation of gamma-FF, alpha/beta-FB patterns. (A) We argue that the
positive correlation between gamma DAI (y axis) and FF anatomical connectivity (x axis) (as observed by Bastos et
al.[10]) is driven by V1 and V2, and is absent when V1 and V2 are not included. In the case of beta, we argue that
the correlation is driven by DP and 7A. Signi?cant Granger asymmetries are shown based on Figure S3 in Bastos et
al.[10]. The areas are ordered hierarchically, with arrows from lower to higher nodes corresponding to FF
in?uences at beta frequencies originate from
connections (cf [119]). (B) We argue that strong Granger-causal
regions with strong beta power, both in the FF [25,26] and FB direction. Abbreviation: TEO, inferotemporal area.
Image adapted, with permission, from [10]).

In sum, the idea that FF/FB Granger-causal in?uences between LFPs re?ect entrainment and
effective information propagation needs to be revised in light of the CTCOM model, as well as
the frequency- and cell type-speci?c effects of oscillations on spiking activity in postsynaptic
targets.

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

137

OPEN ACCESS

Trends in Cognitive Sciences

Oscillations and broadband dynamics in predictive processing
Narrow-band gamma versus broadband ?uctuations in predictive processing
We next discuss studies that directly tested the relationship between prediction error and gamma
oscillations. Studies have reported both positive [8,9,11,45,46] and negative [47�49] relation-
ships between gamma-frequency power and prediction error. We suggest that these discrepan-
cies re?ect opposite correlations between broadband ?uctuations and narrow-band gamma
with stimulus predictability (Figure 1). We argue that the empirical evidence does not suggest a
role for gamma oscillations in conveying sensory prediction errors, and that ?ndings on gamma
are better explained by dendritic HPC rather than classic HPC models (Figure 2).

We suggest that the positive correlations between gamma-frequency power and prediction error ob-
served in several studies re?ect increased spiking activity (Figure 1). Broadband 'gamma-frequency'
power can be driven by aperiodic activity and spiking, and is commonly used as a proxy for spiking
activity in ECoG studies [19,50]. The reported increases in gamma-frequency power for
unpredicted stimuli are typically transient, extend beyond 100Hz, and lack a narrow-band
gamma peak, suggesting broadband ?uctuations [11,51�53]. Therefore, increases in gamma-
frequency power for unpredicted stimuli might re?ect concurrent increases in spiking activity
[8,45,47,54] (Figure 1). To test this hypothesis, a recent study distinguished rhythmic components
from broadband ?uctuations using spectral decomposition techniques and multiscale/multifractal
analyses. These analyses suggest that increased LFP gamma-frequency power for unpredicted
stimuli re?ects aperiodic processes instead of narrow-band gamma [53].

We suggest that negative correlations between gamma power and prediction errors speci?cally
involve narrow-band oscillations (Figure 1). Indeed, recent studies suggest that gamma synchro-
nization in area V1 increases systemically with spatial predictability [23,47�49,55] � the match
between receptive ?eld inputs and contextual predictions [2]. Narrow-band V1 gamma also

Broadband fluctuations

Auditory cortex

0

100

0

100

Visual cortex

Narrowband
gamma

r
e
w
o
p
P
F
L

r
e
w
o
p
P
F
L

Unpredicted stimulus
Predicted stimulus

Transient dynamics

Sustained

Time

s
e
t
a
r
g
n
i
r
i
F

s
e
t
a
r

g
n
i
r
i
F

0

Frequency

100

0

Frequency

100

Stimulus onset

Cognitive Sciences
Sciences
Figure 1. Broadband ?uctuations versus narrow-band gamma. In the main text we argue that narrow-band gamma
oscillations are typically increased for predicted stimuli, whereas broadband 'gamma-frequency' activity increases for
unpredicted stimuli. We argue that the broadband increase is explained by concurrent increases in spiking activity. In
some systems such as visual cortex, narrow-band gamma oscillations are frequently observed. Differences between local
?eld potential (LFP) spectra will re?ect both the broadband and narrow-band gamma effect. By contrast, in other systems
such as auditory cortex, differences in LFP power predominantly re?ect broadband ?uctuations [53].

Trends inin Cognitive
Trends

138

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

Classic hierarchical predictive coding

Dendritic hierarchical predictive coding

(1) TD predictions from level l+1 to l

(1) TD predictions from level l+1 to l

(2) PEs at level l

(3) PEs at level l-1

(2) Apical PEs

(3) Basal PEs

(4) Update of representations at level l

(4) Update of representations at level l

(5) Feedforward from level l to l+1

Hypothesized dynamics in classic HPC

Hypothesized dynamics in dendritic HPC

Level l+1

Predictions (1)

TD prediction (1)

(2) Feedforward PEs

Apical PE

(2)

Level l

Predictions

(4) Update

(3) Feedforward PEs

Level l-1

Gamma

Alpha/Beta

(4)

E

Spiking output

(5)

E

Basal PE (3)

Lateral
inhibition

I

I

E/I imbalance

Mismatch

Feedforward input

Tight E/I balance

Match

Unpredictable stimulus

Predictable stimulus

Unpredictable stimulus

Predictable stimulus

Trends inin Cognitive
Trends

Sciences
Cognitive Sciences

Figure 2. Emergence of rhythms in classic versus dendritic hierarchical predictive coding (HPC). In classic HPC
models, sensory inference results from interactions at each l-th hierarchical level between feedforward (FF) and feedback (FB)
pathways that carry sensory prediction error (PE) and prediction signals, respectively [1,2]. It was hypothesized that these FF
error and FB prediction signals are transmitted via 30�80 Hz gamma oscillations in super?cial layers, and alpha/beta oscilla-
tions in infragranular layers, respectively [8�10]. This hypothesis predicts strong gamma amplitude for unpredicted stimuli and
weak amplitude for predicted stimuli, but does not entail a dependence of alpha/beta (main text for details). We argue that the
emergence of gamma oscillations is better accounted for by the dendritic HPC model, a biologically plausible predictive cod-
ing model in which local excitation/inhibition (E/I) interactions play an important role [3]. The dendritic HPC model builds on the
anatomical observation that FF and FB projections preferentially target basal and apical dendrites, respectively. Dendritic
HPC does not contain specialized neurons ((cid:1)l) for conveying FF error signals, and attributes error representation to voltage
?uctuations in basal and apical dendrites. Error terms at the basal dendrites result from lateral inhibition that predicts and can-
cels out the FF inputs. Based on the dendritic HPC model, we reason that a stimulus with high spatiotemporal predictability
gives rise to tight E/I balance and sparse spiking activity, thereby promoting the emergence of fast network oscillations (as
observed in [47]). We postulate that similar principles may also account for oscillations in other bands such as beta and per-
haps alpha. Abbreviation: TD, top-down.

increases with temporal predictability [23]. V1 gamma synchronization is present during regular
stimulus movement and is disrupted by unpredictable motion [56,57]. Likewise, gamma synchro-
nization increases with stimulus repetition across trials and decreases for novel stimuli [58,59].

The lack of a systematic relationship between gamma oscillations and top-down attention across
studies [12,22,38,60�62] further contradicts the hypothesis that gamma oscillations convey pre-
diction errors that are weighted by precision [46]. Another ?nding that contradicts this hypothesis
is that narrow-band V1 gamma power decreases with the bottom-up salience of natural receptive
?eld stimuli [47].

Alpha/beta and predictive processing
The evidence linking alpha/beta rhythms to the communication of predictions is highly indirect and,
in our view, is subject to multiple interpretations. Studies have reported transient suppression of

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

139

OPEN ACCESS

Trends in Cognitive Sciences

alpha/beta-band power for unpredicted as compared to predicted stimuli ([11,45,51,63]; cf
[9,52,53]). Such a suppression seems to be expected given that alpha/beta rhythms are
suppressed by transient increases in neural activity during sensory stimulation, attention, eye
movements, and movement initiation ([18,27,28,64]; cf [65]).

Nevertheless, it has been suggested that suppression of alpha/beta for unpredicted stimuli
provides indirect evidence for the role of alpha/beta in transmitting predictions. For instance,
some authors [11] have argued that the suppression of alpha/beta rhythms re?ects the
transient updating of current sensory predictions by prediction errors, whereas others [45]
have argued that alpha/beta FB exerts inhibitory effects on lower hierarchical
levels, such
that the suppression of top-down alpha/beta enhances sensory prediction errors in lower
hierarchical levels.

We think these arguments do not make a clear case, and we offer several counterarguments. (i) It
is assumed that suppression of alpha/beta rhythms causes neural activation for unpredicted
stimuli, and that, at the same time, increased neural activation for these stimuli leads to the sup-
pression of alpha/beta, but this leads to a chicken-or-egg problem. (ii) In classic HPC, prediction
errors do not result from suppression of top-down FB. Instead, prediction errors are the differ-
ence between local representations and top-down FB [2] (Figure 2). That is, in classic HPC,
top-down FB is necessary to compute stimulus prediction errors, which seems incompatible
with the suppression of alpha/beta. Generally, the assumption that top-down FB is suppressive
is questionable because FB exerts excitatory rather than inhibitory effects on the local represen-
tational units in PC models [2,3].

Predictability: a common principle underlying rhythmic activity?
In contrast to the gamma-FF/alpha-beta-FB hypothesis, which suggests opposing functional
roles for the two bands, we propose that the emergence of gamma and alpha/beta rhythms re-
?ects a common mechanism. For instance, both gamma and alpha-beta rhythms tend to occur
during relatively stationary periods, are transiently disrupted by sensory cues or movements, and
occur during periods in which ?ring rates are relatively low [27,29,34,41]. Hence, the emergence
of oscillations may generally follow a common pattern related to the later phases of sensory pro-
cessing and may re?ect the stability of sensory representations. We further argue that the emer-
gence of oscillations is crucially dependent on the spatiotemporal predictability of sensory inputs.
In this framework, oscillatory states may be relatively widespread across the cortex but allow for
the possibility that speci?c frequencies may be more or less prominent across cortical networks.
In this view, differences in frequencies do not differentiate distinct functional roles and instead re-
?ect differences in area-speci?c circuits, intrinsic dynamics, integration, stimulus representation,
biophysical time constants [21,24,66], or input drive [67,68].

Transients versus oscillations in predictive processing
In this section we outline a functional model contrasting the roles of transients and oscillations in
sensory inference. Contrary to classic HPC, we propose that aperiodic transients, rather than os-
cillations, mediate rapid sensory inference. Instead, our model posits that oscillations play a role in
stabilizing neural dynamics and plasticity during the late phase of stimulus processing (Figure 3).
Accordingly, we suggest that cortical circuits continuously alternate between transients and os-
cillatory states (i.e., states of stability and plasticity), thereby creating two distinct sensory pro-
cessing phases that are FF- and FB-dominated, respectively. Notably, our model is consistent
with the observation that gamma oscillations are disrupted by transients [e.g., stimulus onsets
and (micro)saccades] [34,41,69,70] and only stabilize 100 ms or later following stimulus or
(micro)saccade onset [34,41,71].

140

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

Recognition
q(z | x)

100 ms
IT

e
c
n
e
r

Infe

V4

V4

G
e
n

B
a
c
k
pro
erativ
p
a
e m
g
atin
o
g errors
d
el

V2

V1

V4

PIT

CIT

AIT

Retina

LGN

40 ms
V1

Transient

160 ms
V1

Plasticity
Posterior p(x|z)

State space

Stability

Stabilize representations
Leave �eligibility traces�

Contributions of rhythms to stabilization and plasticity

Feedback learning signal

Plasticity

RNN

Hebbian mechanisms (STDP)

Synchronization

Spike phase shift

Pref. stim
Non-pref. stim

Input

Predictability of inputs

Activation strength

Trends inin Cognitive
Trends

Sciences
Cognitive Sciences

Figure 3. Transients versus oscillations in sensory inference. We posit that sensory inference predominantly relies on
aperiodic transients, whereas rhythms play a role in stabilizing neural representations and plasticity processes during the late,
feedback (FB)-dominated phase of stimulus processing. In the visual system, stimulus onset typically leads to rapid cascade
of transients across the ventral stream. Stimulus onset latencies in IT are ~100 ms and the inference of object properties has
been largely completed by ~120 ms [72]. This is noted by the inference model q(z|x) � the probability distribution over the
latent variable z. Learning not only in supervised neural networks but also in self-supervised neural networks such as
variational autoencoders, entails that feedback (FB) from higher levels reaches early sensory areas. Here the FB may carry
back-propagating errors that interact with local eligibility traces to instruct plasticity of the local recurrent connections [95].
Furthermore, for self-supervised learning, top-down generative networks can compute the posterior probability of the
inputs given the inferred latents, p(x|z), which in turn instructs plasticity. Gamma oscillations in early visual areas emerge
relatively late after stimulus onset, and will be prominent when this FB arrives. As discussed in the text, gamma oscillations
can facilitate plasticity processes, for example by synchronizing neurons that receive spatiotemporally correlated inputs
[47], or via activation-dependent gamma phase-shifting [94]. We also argue that gamma may stabilize the neural activity
during the later stimulus phases, thus narrowing the region of state space that is occupied. Abbreviations: AIT, anterior
inferotemporal cortex; CIT, central inferotemporal cortex; IT, inferotemporal cortex; LGN, lateral geniculate nucleus; PIT,
posterior inferotemporal cortex; V1,2,4, visual cortical areas 1, 2, and 4.

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

141

OPEN ACCESS

Trends in Cognitive Sciences

Transient dynamics
V1 neurons respond to visual input with latencies of 30�40 ms, followed by a cascade of neural
activations across the ventral stream, culminating in high-level object representations in the in-
ferior temporal cortex (IT) by ~120 ms [72]. This swift processing enables non-human primates
to perform object recognition for brief stimulus presentations of ~100 ms [72,73]. The forma-
tion of object-selective responses in IT occurs within a 60�80 ms window, suggesting that a
signi?cant portion of this time is used for FF transmission through the visual areas V1, V2,
and V4 to IT [72].

Therefore, stimulus inference may largely depend on the FF sweep [72�74], comprising a cas-
cade of transient responses. Transients need to be distinguished from oscillations because
they contain broadband energy and could disrupt ongoing oscillations by perturbing the balance
between excitation and inhibition [75]. In state space, transients correspond to a smooth and
short path from one neural representation to another (i.e., from 'A' to 'B'), whereas oscillations
comprise stochastic ?uctuations in some circumscribed region of state space (e.g., around a
?xed point 'B') [76] (Figure 3).

Transient inputs can be ampli?ed via several mechanisms that do not involve rhythmic synchro-
nization: (i) speci?c input patterns can be selectively ampli?ed dependent on the synaptic weight
matrix stored in local recurrent and FF connections; (ii) unstable recurrent excitation that is
balanced by strong inhibitory FB enables fast, strong responses to sensory inputs [75,77]; and
(ii) transients contain broadband synchronized activity (resulting in large event-related potentials)
which may enhance the impact on downstream receivers [78] and encode information via rapid
spike sequences [74,79].

The idea that inference relies on rapid transients is at odds with the view of inference within classic
HPC. In classic HPC, inference relies on iterative optimization of neural representations across all
hierarchical levels (Figure 2) [2]. This iterative process can be slow and result in low-frequency
oscillations [65,80]. However, such iterative optimization is not mandatory: in generative inference
models such as variational autoencoders, inference can be rapidly performed via a FF net-
work. In contrast to HPC, stimulus predictions are then generated only once inference is com-
pleted. These stimulus predictions are required for self-supervised learning but not for inference
itself [81].

Oscillations and stabilization
In sum, we argue that stimulus inference depends crucially on neural transients. These transients
represent transitions between different neural states and are therefore ill suited to stabilizing and
maintaining neural representations. The stabilization and maintenance of neural representations
may instead depend on oscillatory dynamics, which re?ect the movement of the system around
a ?xed point or along a limit cycle in state space [82,83], and can be conceived as a dynamical
state that keeps the system in a speci?c con?guration. It thus makes sense to look for functional
roles of oscillations in the stabilization of neural activity, which may be played not only by gamma
oscillations, for example in visual cortex, but also by beta oscillations, for example in somatosen-
sory and motor cortex [84,85]. The stabilization of neural activity may be important for many behav-
iors and cognitive functions (e.g., working memory, expectation) but also for learning processes.

Gamma oscillations may stabilize neural responses by reducing their trial-by-trial variability. Ex-
citation/inhibition (E/I) models suggest that when neural populations engage in gamma syn-
chronization, neural representations exhibit less variability and the information per spike is
maximized [86]. Consistent with this model, neural variability tends to be relatively low for stimuli

142

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

that induce V1 gamma oscillations [23], and gamma-synchronized spikes carry more informa-
tion than non-synchronized spikes [87]. The balanced E/I interactions producing gamma oscil-
lations may also reduce the covariability of neural responses (i.e., noise correlations) occurring
at lower frequencies [87,88]. At the same time, the spiking correlations induced at gamma fre-
quencies may not signi?cantly impact on downstream excitatory neurons, and instead target
fast-spiking interneurons [22], and can therefore increase the signal-to-noise ratio of sensory
transmission [89].

It has been argued that the highly stochastic nature of gamma oscillations, that exhibit ample var-
iability in instantaneous cycle amplitude and duration [76,90], may prevent a functional role for
gamma in neural coding and communication [71,90]. Nevertheless, this stochasticity may be re-
quired for a role in stabilization processes because the variability of neural representations is min-
imized for stochastic rather than regular (harmonic) oscillations [86].

Oscillations, feedback, and plasticity
Both self-supervised and supervised neural network architectures typically assume separate
phases for inference and plasticity/learning. For example, in deep (FF) neural networks, the for-
ward pass is followed by a top-down ?ow of error signals, which are crucial to instruct gradient
descent (Figure 3). Likewise, in generative inference models such as variational autoencoders,
the inference phase [q(z|x)] is followed by generative top-down activity that is necessary to ap-
proximate the surprise of the sensory input data [(?ln p(x|z)], which in turn instructs gradient de-
scent. In a biological neural network, this implies that the presentation of a stimulus ?rst elicits a
phase in which neural circuits compute and perform inference, which is then followed by a
phase in which synaptic connections are ?ne-tuned.

It remains an open problem how biological neural networks can transition between these two
phases and what aspects of neural activity facilitate this transition [91]. We propose that tran-
sients and oscillations serve as alternating 'phases' for inference and plasticity, where oscillations
shift the circuit into a FB-dominated mode and open the window for plasticity. Although we focus
our arguments on gamma oscillations during vision, we consider that beta oscillations may play a
similar role in somatosensory processing [69].

During each stimulus presentation or eye ?xation, stimulus inference is completed in IT at ~120�
160 ms [92]. Assuming a processing delay from IT to V1 of ~40 ms, the backward pass should
arrive in V1 after ~160�200 ms (i.e., towards the end of each eye ?xation). During this late stimulus
phase, inter-areal interactions tend to be FB- rather than FF-dominated [93], and V1 gamma os-
cillations are most prominent [70]. As discussed above, gamma oscillations may dampen FF in-
formation ?ow (and thereby terminate the inference phase) by recruiting inhibitory activity in the FF
direction. At the same time, gamma oscillations may play a role in FB-related computations, me-
diated by two mechanisms.

(i) The FB arriving during the backward pass needs to be integrated with activity patterns that oc-
curred during the forward pass. Because the backward pass arrives with a large delay compared
to the onset of the forward pass (Figure 3), the forward pass needs to leave a 'trace'. Gamma
could contribute to maintaining 'traces' of the preceding transient activations, where the initial
transient activation determines subsequent oscillatory wave patterns and the gamma phase of
spiking [94]. These traces can then interact with FB signals arriving in the late stimulus phase to
coordinate learning [95]. Theoretical studies suggest an advantage of oscillatory network activity
for learning processes because they mitigate both the vanishing and exploding gradient problem
in recurrent neural networks [96].

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

143

OPEN ACCESS

Trends in Cognitive Sciences

(ii) The synchronization of spiking activity can in?uence synaptic plasticity processes via spike
time-dependent plasticity (STDP) mechanisms [97�99] (Figure 3). Experimental manipulations
of rhythms suggest that gamma synchronization facilitates the formation of synaptic plasticity
[99�101]. The precise pattern of synaptic weight changes is likely determined by the structure
of the visual stimulus. The structure of the visual stimulus determines the strength of synchroniza-
tion because gamma-rhythm synchronization occurs between neural populations where each
predicts the sensory inputs of the other [47]. The stimulus structure also determines the phase
of synchronization because the stimulus input drive is converted into a spike gamma-phase
code [94]. The resulting temporal spiking patterns may recruit Hebbian mechanisms, such as
STDP, which then interact with top-down learning signals to drive synaptic plasticity [91,95].

The stochastic nature of gamma oscillations [76,90] is compatible with a functional role in plastic-
ity processes. First, because learning takes place on long timescales that encompass many stim-
ulus presentations, the instantaneous ?uctuations in gamma-cycle amplitude are averaged out.
Second, learning algorithms require a degree of stochasticity that avoids over?tting and promotes
generalization in training neural networks [102,103]. Random sampling is also crucial for training
self-supervised algorithms such as variational autoencoders that have an inherent probabilistic
nature [81]. A recent study has shown that stochastic gamma-band oscillations are an emergent
feature of E/I networks performing random sampling from a latent distribution [104].

Explaining gamma from dendritic HPC models
We have presented several arguments against the theory that prediction errors are transmitted
through gamma (30�80 Hz) oscillations and that prediction signals are transmitted through
alpha/beta (10�20 Hz) oscillations. We propose that the emergence of narrow-band oscillations
is well explained by dendritic HPC, a biologically realistic alternative to classic HPC.

A key component of dendritic HPC is the interaction between local E/I neurons [3]. The model
builds on the observation that FF and FB projections preferentially target basal and apical den-
drites, respectively. Unlike classic HPC, dendritic HPC does not require specialized projection
neurons for conveying FF error signals ((cid:1)); instead, neurons only transmit representations (r). Den-
dritic HPC attributes prediction error representation to voltage ?uctuations in basal and apical
dendrites, which are integrated at the soma and drive spiking (r) (Figure 2). At basal dendrites,
error terms result from local inhibition which can predict and cancel out the FF inputs from the pre-
ceding hierarchical level (Figure 2).

The interaction between E/I neurons in the dendritic HPC model is consistent with the various
conditions that are associated with prominent gamma oscillations. For instance, stimuli with
high spatiotemporal predictability (e.g., surfaces of homogeneous objects) would give rise to a
tight E/I balance and sparse spiking activity, which are factors associated with the emergence
of gamma oscillations [23,47,48,60,67,71,105,106].

In the gamma cycle, there is a substantial phase delay between balanced excitatory and inhibitory
activity [60]. Such a delay between excitation and inhibition ensures that there is always some re-
sidual spiking activity [60,106]. That is, the 'ground state' of a neural circuit in the case of highly
predictable stimuli may be sparse, oscillatory ?ring [47,48] rather than vanishing FF prediction er-
rors. Such residual ?ring may be important to sustain FF information transmission and support
stimulus representations even in the case of fully predictable stimuli.

Importantly, in dendritic HPC, the E/I balance in a local circuit depends on the predictability of the
FF inputs that it receives (driving the basal prediction error), but not on the predictability of ?ring

144

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

rate representations by top-down FB (driving the apical prediction error; Figure 2). We therefore
predict that gamma oscillations should be correlated with the predictability of FF inputs into a cir-
cuit. Indeed, it was recently shown that gamma oscillations in V1 are speci?cally (positively) cor-
related with the low-level predictability of sensory inputs [47], which was de?ned as the extent to
which low-level image features can be predicted from the spatial context.

V1 ?ring rates, by contrast, show the strongest (negative) correlation with the predictability of
high-level image features derived from neural networks for object recognition [47]. Consistently,
various manipulations of object segmentation have been shown to affect V1 ?ring rates but not
V1 gamma synchronization [47,107,108]. In dendritic HPC, the modulation of V1 activity by
high-level predictability is mediated by top-down FB arriving at the apical dendrites (Figure 2).
This top-down FB predicts local spiking activity, resulting in 'apical' prediction errors that drive
spiking and increased ?ring rates.

Concluding remarks
The dominant paradigm regarding oscillations is sender-focused and portrays oscillations as
causal entities that have a functional impact on receiving areas [17]. We have instead adopted
a receiver-focused perspective on oscillations which describes oscillations in the receiver as
the result of FF lower-level predictability. We have outlined a functional model in which rapid sen-
sory inference is mediated by aperiodic transients, whereas oscillations contribute to local func-
tions, in particular the stabilization of neural representations over time and space, as well as
modulating plasticity processes.

Our discussion underscores the point that linear connectivity measures, such as Granger cau-
sality, have limited use in studying inter-areal communication because these measures are
blind to non-linear interactions [34,36,72]. It is crucial to consider the non-linear and recurrent
nature of stimulus inference processes [34]. Indeed, a recent study suggested that transients in
lower and higher areas exhibit synergistic encoding of sensory prediction errors, re?ecting such
nonlinearities [109].

Causal perturbations are warranted to test the hypotheses presented here (see Outstanding
questions). For example, we hypothesize that perturbing activity during transients should have a
strong impact on sensory inference, as shown in [110], whereas perturbation of gamma oscillations
should speci?cally affect the stabilization of neural representations over time and learning.

Acknowledgments
We thank Craig Richter, Conrado Bosman, and Wolf Singer for helpful comments on different versions of this manuscript. M.

V., C.U., and B.R. were supported by an ERC starting grant (SPATEMP, EU), a BMBF (Germany) grant (Computational Life
Sciences, project BINDA, 031L0167), DFG VI grants (908/5-1 and 908/7-1), the NWO VIDI, and the Dutch Brain Interface

Initiative. A.C.J. is supported by an ANID/FONDECYT regular (1240899) research grant.

Declaration of interests
The authors declare no competing interests.

References

1. Friston, K. (2010) The free-energy principle: a uni?ed brain theory?

Nat. Rev. Neurosci. 11, 127�138

2. Rao, R.P. and Ballard, D.H. (1999) Predictive coding in the vi-
sual cortex: a functional interpretation of some extra-classical
receptive-?eld effects. Nat. Neurosci. 2, 79�87

3. Mikulasch, F.A. et al. (2023) Where is the error? hierarchical pre-
dictive coding through dendritic error computation. Trends
Neurosci. 46, 45�59

4. Singer, W. (2021) Recurrent dynamics in the cerebral cortex: in-
tegration of sensory evidence with stored knowledge. Proc.
Natl. Acad. Sci. U. S. A. 118, e2101043118

5. Heeger, D.J. (2017) Theory of cortical function. Proc. Natl.

Acad. Sci. U. S. A. 114, 1773�1782

6. Bosman, C.A. et al. (2014) Functions of gamma-band synchroniza-
tion in cognition: from single circuits to functional diversity across cor-
tical and subcortical systems. Eur. J. Neurosci. 39, 1982�1999

OPEN ACCESS

Outstanding questions
How does
the dependence of
oscillatory dynamics on predictability
differ between different modalities
(e.g., olfactory, auditory, visual) and fre-
quency bands?

What determines the frequency of
rhythms and their emergence in
different networks, and to what extent
are there shared mechanisms and
functions between alpha, beta, and
gamma?

Do neural populations in the visual cortex
generate an intrinsic beta rhythm that is
independent of sensorimotor and frontal
beta?

How does the phase-locking strength
of single cells vary across cortical
layers, and differ between different cell
types and projection neurons?

What is the functional impact of alpha
and beta rhythms on the spiking
activity of distinct cell types and laminar
compartments in receiving areas?

the

are

precise

What
circuit
mechanisms, such as those involving
layers and cell types, that account for
rhythms on
the dependence of
temporal and spatial predictability?

Can the dendritic predictive coding
model account for the emergence of
rhythmic dynamics, and to what
extent do aspects such as recurrent
excitation or resonance, which are
lacking in this model, matter?

What are the respective contributions
of horizontal and top-down FB con-
nections to the modulation of neural
activity by predictability?

Do oscillatory dynamics causally
contribute to the stabilization of neural
representations?

Does gamma synchronization contribute
causally to perception, such as to the
stability of perception over time?

Can oscillations mediate a switch from
FF- to FB-dominated communication
between areas?

Do plasticity processes in the later
phases of sensory processing causally
depend on oscillations?

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

145

OPEN ACCESS

Trends in Cognitive Sciences

7. Vinck.Panzeri, SGamma-band synchronization and information
transmission. In Principles of Neural Coding (Quiroga-Quian, R.
and Panzeri, S., eds), pp. 449�000, CRC Press., eds Principles of
Neural CodingCRC Press, pp. 449�469Gamma-band synchroniza-
tion and information transmission. In Principles of Neural Coding
(Quiroga-Quian, R. and Panzeri, S., eds), pp. 449�000, CRC Press
8. Bastos, A.M. et al. (2012) Canonical microcircuits for predictive

coding. Neuron 76, 695�711

9. Arnal, L.H. and Giraud, A.L. (2012) Cortical oscillations and sen-

sory predictions. Trends Cogn. Sci. 16, 390�398

10. Bastos, A.M. et al. (2015) Visual areas exert feedforward and
feedback in?uences through distinct frequency channels.
Neuron 85, 390�401

11. Chao, Z.C. et al. (2018) Large-scale cortical networks for hierar-
chical prediction and prediction error in the primate brain. Neu-
ron 100, 1252�1266

12. van Kerkoerle, T. et al. (2014) Alpha and gamma oscillations
characterize feedback and feedforward processing in monkey
visual cortex. Proc. Natl. Acad. Sci. U. S. A., 201402773
13. Richter, C.G. et al. (2017) Top-down beta enhances bottom-up

gamma. J. Neurosci. 37, 6698�6711

14. Vezoli, J. et al. (2021) Brain rhythms de?ne distinct interaction
networks with differential dependence on anatomy. Neuron
109, 3862�3878

15. Nougaret, S. et al. (2024) Low and high beta rhythms have dif-
ferent motor cortical sources and distinct roles in movement
control and spatiotemporal attention. PLoS Biol. 22, e3002670
16. Mendoza-Halliday, D. et al. (2024) A ubiquitous spectrolaminar
motif of local ?eld potential power across the primate cortex.
Nat. Neurosci. 27, 547�560

17. Fries, P. (2015) Rhythms for cognition: communication through

coherence. Neuron 88, 220�235

18. Michalareas, G. et al. (2016) Alpha-beta and gamma rhythms
subserve feedback and feedforward in?uences among human
visual cortical areas. Neuron 89, 384�397

19. Ray, S. and Maunsell, J.H. (2011) Different origins of gamma
rhythm and high-gamma activity in macaque visual cortex.
PLoS Biol. 9, e1000610

20. Parto-Dezfouli, M. et al. (2023) Enhanced behavioral perfor-
mance through interareal gamma and beta synchronization.
Cell Rep. 42, 113249

21. Hoffman, S.J. et al. (2024) The primate cortical LFP exhibits
multiple spectral and temporal gradients and widespread
task-dependence
short-term memory.
J. Neurophysiol. 132, 206�225

during

visual

How do top-down learning signals
precisely interact with local gamma
synchronization, and what is the role
of burst spikes?

What are the distinct contributions of
transients and oscillations to stimulus
inference?

How do non-linear interaction mea-
sures depend on stimulus predictabil-
ity, and how do these interactions
differ between oscillations
and
transients?

To what extent do transients and
oscillations provide synergistic and
redundant information across cortical
space?

32. Buffalo, E.A. et al. (2011) Laminar differences in gamma and
alpha coherence in the ventral stream. Proc. Natl. Acad. Sci.
U. S. A. 108, 11262�11267

33. Vezoli, J. et al. (2021) Cortical hierarchy, dual counterstream ar-
chitecture and the importance of top-down generative net-
works. Neuroimage 225, 117479

34. Vinck, M. et al. (2023) Principles of large-scale neural interac-

tions. Neuron 111, 987�1002

35. Dowdall, J.R. et al. (2023) Attentional modulation of inter-areal
coherence explained by frequency shifts. NeuroImage 277,
120256

36. Dowdall, J.R. and Vinck, M. (2023) Coherence fails to reliably
capture inter-areal interactions in bidirectional neural systems
with transmission delays. NeuroImage 271, 119998

37. Salinas, E. and Sejnowski, T.J. (2001) Correlated neuronal ac-
tivity and the ?ow of neural information. Nat. Rev. Neurosci. 2,
539�550

38. Fries, P. et al. (2001) Modulation of oscillatory neuronal syn-
chronization by selective visual attention. Science 291,
1560�1563

39. Izhikevich, E.M. et al. (2003) Bursts as a unit of neural informa-
tion: selective communication via resonance. Trends Neurosci.
26, 161�167

40. Schneider, M. et al. (2023) Cell-type-speci?c propagation of vi-

sual ?icker. Cell Rep. 42, 112492

41. Bosman, C. et al. (2012) Attentional stimulus selection through
selective synchronization between monkey visual areas. Neuron
75, 875�888

42. Schomburg, E.W. et al. (2014) Theta phase segregation of
input-speci?c gamma patterns in entorhinal-hippocampal net-
works. Neuron 84, 470�485

43. Linden, H. et al. (2010) Intrinsic dendritic ?ltering gives low-pass
power spectra of local ?eld potentials. J. Comput. Neurosci. 29,
423�444

44. Vaidya, S.P. and Johnston, D. (2013) Temporal synchrony and
gamma-to-theta power conversion in the dendrites of CA1 py-
ramidal neurons. Nat. Neurosci. 16, 1812�1820

45. Bastos, A.M. et al. (2020) Layer and rhythm speci?city for pre-
dictive routing. Proc. Natl. Acad. Sci. 117, 31459�31469
46. Bauer, M. et al. (2014) Attentional modulation of alpha/beta and
gamma oscillations re?ect functionally distinct processes.
J. Neurosci. 34, 16117�16125

47. Uran, C. et al. (2022) Predictive coding of natural images by V1
?ring rates and rhythmic synchronization. Neuron 110,
1240�1257

22. Spyropoulos, G. et al. (2024) Distinct feedforward and feedback
pathways for cell-type speci?c attention effects. Neuron 112,
2423�2434

48. Peter, A. et al. (2019) Surface color and predictability determine
contextual modulation of V1 ?ring and gamma oscillations. eLife
8, e42101

23. Vinck, M. and Bosman, C.A. (2016) More gamma more predic-
tions: gamma-synchronization as a key mechanism for ef?cient
integration of classical receptive ?eld inputs with surround pre-
dictions. Front. Syst. Neurosci. 10, 35

24. Onorato, I. et al. (2020) A distinct class of bursting neurons with
strong gamma synchronization and stimulus selectivity in mon-
key V1. Neuron 105, 180�197

25. Brovelli, A. et al. (2004) Beta oscillations in a large-scale sensori-
motor cortical network: directional in?uences revealed by granger
causality. Proc. Natl. Acad. Sci. U. S. A. 101, 9849�9854
26. Salazar, R. et al. (2012) Content-speci?c fronto-parietal syn-
chronization during visual working memory. Science 338,
1097�1100

27. Kilavik, B.E. et al. (2013) The ups and downs of beta oscillations

in sensorimotor cortex. Exp. Neurol. 245, 15�26

28. Brunet, N. et al. (2014) Gamma or no gamma, that is the ques-

tion. Trends Cogn. Sci. 18, 507�509

29. Schneider, M. et al. (2021) A mechanism for inter-areal coher-
ence through communication based on connectivity and oscil-
latory power. Neuron 109, 4050�4067

30. Pesaran, B. et al. (2018) Investigating large-scale brain dynam-
ics using ?eld potential recordings: analysis and interpretation.
Nat. Neurosci. 21, 903�919

31. Vinck, M. et al. (2016) Cell-type and state-dependent synchro-
nization among rodent somatosensory, visual, perirhinal cortex,
and hippocampus CA1. Front. Syst. Neurosci. 9, 187

49. Shirhatti, V. et al. (2022) Gamma oscillations in primate primary
visual cortex are severely attenuated by small stimulus disconti-
nuities. PLoS Biol. 20, e3001666

50. Miller, K.J. et al. (2009) Power-law scaling in the brain surface

electric potential. PLoS Comput. Biol. 5, e1000609

51. Jiang, Y. et al. (2022) Constructing the hierarchy of predictive
auditory sequences in the marmoset brain. Elife 11, e74653
52. Todorovic, A. et al. (2011) Prior expectation mediates neural ad-
aptation to repeated sounds in the auditory cortex: an MEG
study. J. Neurosci. 31, 9118�9123

53. Canales-Johnson, A. et al. (2021) Broadband dynamics rather
than frequency-speci?c rhythms underlie prediction error in
the primate auditory cortex. J. Neurosci. 41, 9374�9391
54. Parras, G.G. et al. (2017) Neurons along the auditory pathway
exhibit a hierarchical organization of prediction error. Nat.
Commun. 8, 2148

55. Hermes, D. et al. (2019) An image-computable model for the
stimulus selectivity of gamma oscillations. Elife 8, e47035

56. Kruse, W. and Eckhorn, R. (1996)

Inhibition of sustained
gamma oscillations (35�80 Hz) by fast transient responses in
cat visual cortex. Proc. Natl. Acad. Sci. U. S. A. 93, 6112�6117
57. Kayser, C. et al. (2003) Responses to natural scenes in cat V1.

J. Neurophysiol. 90, 1910�1920

58. Brunet, N.M. et al.

(2014) Stimulus repetition modulates
gamma-band synchronization in primate visual cortex. Proc.
Natl. Acad. Sci. U. S. A. 111, 3626�3631

146

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

Trends in Cognitive Sciences

OPEN ACCESS

59. Peter, A. et al. (2021) Stimulus-speci?c plasticity of macaque v1

spike rates and gamma. Cell Rep. 37, 110086

60. Vinck, M. et al. (2013) Attentional modulation of cell-class-
speci?c gamma-band synchronization in awake monkey area
V4. Neuron 80, 1077�1089

61. Chalk, M. et al. (2010) Attention reduces stimulus-driven
gamma frequency oscillations and spike ?eld coherence in V1.
Neuron 66, 114�125

62. Das, A. and Ray, S. (2018) Effect of stimulus contrast and visual
attention on spike-gamma phase relationship in macaque pri-
mary visual cortex. Front. Comput. Neurosci. 12, 66

63. von Stein, A. et al. (2000) Top-down processing mediated by
interareal synchronization. Proc. Natl. Acad. Sci. 97,
14748�14753

64. Popov, T. et al. (2021) Alpha oscillations link action to cognition:
an oculomotor account of
the brain's dominant rhythm.
BioRxiv, Published online September 24, 2021. https://doi.
org/10.1101/2021.09.24.461634

85. Engel, A.K. and Fries, P. (2010) Beta-band oscillations � signal-
ling the status quo? Curr. Opin. Neurobiol. 20, 156�165
86. Chalk, M. et al. (2016) Neural oscillations as a signature of ef?-
cient coding in the presence of synaptic delays. Elife 5, e13824
87. Womelsdorf, T. et al. (2012) Orientation selectivity and noise
correlation in awake monkey area V1 are modulated by the
gamma cycle. Proc. Natl. Acad. Sci. U. S. A. 109, 4302�4307
88. Renart, A. et al. (2010) The asynchronous state in cortical cir-

cuits. Science 327, 587�590

89. Hamilton, L.S. et al. (2013) Optogenetic activation of an inhibi-
tory network enhances feedforward functional connectivity in
auditory cortex. Neuron 80, 1066�1076

90. Burns, S.P. et al. (2011) Is gamma-band activity in the local ?eld
potential of V1 cortex a �clock� or ?ltered noise? J. Neurosci. 31,
9658�9664

91. Payeur, A. et al. (2021) Burst-dependent synaptic plasticity can
coordinate learning in hierarchical circuits. Nat. Neurosci. 24,
1010�1019

65. Alamia, A. and VanRullen, R. (2019) Alpha oscillations and trav-
eling waves: signatures of predictive coding? PLoS Biol. 17,
e3000487

92. Kar, K. et al. (2019) Evidence that recurrent circuits are critical to
the ventral stream's execution of core object recognition behav-
ior. Nat. Neurosci. 22, 974�983

66. Murray, J.D. et al. (2014) A hierarchy of intrinsic timescales

across primate cortex. Nat. Neurosci. 17, 1661�1663

67. Wang, X.J. (2010) Neurophysiological and computational prin-
ciples of cortical rhythms in cognition. Physiol. Rev. 90,
1195�1268

68. Roberts, M.J. et al. (2013) Robust gamma coherence between
macaque V1 and V2 by dynamic frequency matching. Neuron
78, 523�536

69. Confais, J. et al. (2020) Is there an intrinsic relationship between
lfp beta oscillation amplitude and ?ring rate of individual neurons
in macaque motor cortex? Cereb. Cortex Commun. 1, tgaa017
70. Gieselmann, M.A. and Thiele, A. (2008) Comparison of spatial
integration and surround suppression characteristics in spiking
activity and the local ?eld potential
in macaque V1. Eur.
J. Neurosci. 28, 447�459

71. Ray, S. and Maunsell, J.H. (2015) Do gamma oscillations play a

role in cerebral cortex? Trends Cogn. Sci. 19, 78�85

72. DiCarlo, J.J. et al. (2012) How does the brain solve visual object

recognition? Neuron 73, 415�434

73. Yamins, D.L. et al. (2014) Performance-optimized hierarchical
models predict neural responses in higher visual cortex. Proc.
Natl. Acad. Sci. 111, 8619�8624

74. Thorpe, S. et al. (2001) Spike-based strategies for rapid pro-

cessing. Neural Netw. 14, 715�725

75. Murphy, B.K. and Miller, K.D. (2009) Balanced ampli?cation: a
new mechanism of selective ampli?cation of neural activity pat-
terns. Neuron 61, 635�648

76. Spyropoulos, G. et al. (2022) Spontaneous variability in gamma
dynamics described by a damped harmonic oscillator driven by
noise. Nat. Commun. 13, 2019

77. van Vreeswijk, C. and Sompolinsky, H. (1996) Chaos in neuro-
nal networks with balanced excitatory and inhibitory activity.
Science 274, 1724�1726

78. Bruno, R.M. and Sakmann, B. (2006) Cortex is driven by weak
but synchronously active thalamocortical synapses. Science
312, 1622�1627

79. Yiling, Y. et al. (2023) Robust encoding of natural stimuli by neu-
ronal response sequences in monkey visual cortex. Nat.
Commun. 14, 3021

80. Bogacz, R. (2017) A tutorial on the free-energy framework for
modelling perception and learning. J. Math. Psychol. 76,
198�211

81. Kingma, D.P. et al. (2014) Semi-supervised learning with deep
generative models. In Advances in Neural Information Process-
ing Systems (27) (Gharamani, G. et al., eds), pp. 3581�3589,
NeurIPS

82. Park, I.M. et al. (2023) Persistent learning signals and working
memory without continuous attractors. ArXiv, Published online
August 24, 2023. http://dx.doi.org/10.48550/arXiv.2308.12585
83. Wallace, E. et al. (2011) Emergent oscillations in networks of

93. Semedo, J.D. et al. (2022) Feedforward and feedback interac-
tions between visual cortical areas use different population ac-
tivity patterns. Nat. Commun. 13, 1099

94. Vinck, M. et al. (2010) Gamma-phase shifting in awake monkey

visual cortex. J. Neurosci. 30, 1250�1257

95. Bellec, G. et al. (2020) A solution to the learning dilemma for re-
current networks of spiking neurons. Nat. Commun. 11, 3625
96. Rusch, T.K. and Mishra, S. (2020) Coupled oscillatory recurrent
neural network (CORNN): an accurate and (gradient) stable
architecture for learning long time dependencies. ArXiv, Pub-
lished online October 2, 2021. http://dx.doi.org/10.48550/
arXiv.2010.00951

97. Chrobak, J.J. and Buzs�ki, G. (1998) Operational dynamics in
the hippocampal-entorhinal axis. Neurosci. Biobehav. Rev. 22,
303�310

98. Traub, R.D. et al. (1998) Gamma-frequency oscillations: a neu-
ronal population phenomenon, regulated by synaptic and intrin-
sic cellular processes, and inducing synaptic plasticity. Prog.
Neurobiol. 55, 563�575

99. Anisimova, M. et al. (2023) Spike-timing-dependent plasticity
rewards synchrony rather than causality. Cereb. Cortex 33,
23�34

100. Galuske, R.A. et al. (2019) Relation between gamma oscillations
and neuronal plasticity in the visual cortex. Proc. Natl. Acad.
Sci. 116, 23317�23325

101. Wespatat, V. et al. (2004) Phase sensitivity of synaptic modi?ca-
tions in oscillating cells of rat visual cortex. J. Neurosci. 24,
9067�9075

102. Kingma, D.P. and Ba, J. (2014) A method for stochastic optimi-
zation. ArXiv, Published online December 22, 2014. http://dx.
doi.org/10.48550/arXiv.1412.6980

103. Schug, S. et al. (2021) Presynaptic stochasticity improves en-
ergy ef?ciency and helps alleviate the stability-plasticity di-
lemma. Elife 10, e69884

104. Echeveste, R. et al. (2020) Cortical-like dynamics in recurrent
circuits optimized for sampling-based probabilistic inference.
Nat. Neurosci. 23, 1138�1149

105. B�rgers, C. and Kopell, N.J. (2008) Gamma oscillations and

stimulus selection. Neural Comput. 20, 383�414

106. Den�ve, S. and Machens, C.K. (2016) Ef?cient codes and bal-

anced networks. Nat. Neurosci. 19, 375�382

107. Chen, G. et al. (2017) Distinct inhibitory circuits orchestrate
cortical beta and gamma band oscillations. Neuron 96,
1403�1418

108. Roelfsema, P.R. et al. (2004) Synchrony and covariation of ?ring
rates in the primary visual cortex during contour grouping. Nat.
Neurosci. 7, 982�991

109. Gelens, F. et al. (2024) Distributed representations of prediction
error signals across the cortical hierarchy are synergistic. Nat.
Commun. 15, 3941

stochastic spiking neurons. PLoS One 6, e14804

110. Resulaj, A. et al. (2018) First spikes in visual cortex enable per-

84. Gelastopoulos, A. et al. (2019) Parietal low beta rhythm pro-
vides a dynamical substrate for a working memory buffer.
Proc. Natl. Acad. Sci. 116, 16613�16620

ceptual discrimination. Elife 7, e34044

111. Mackey, C. et al. (2024) A ubiquitous spectrolaminar motif of
local ?eld potential power across the primate cortex? OSF

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2

147

OPEN ACCESS

Trends in Cognitive Sciences

Prepr., Published online April 11, 2024. http://dx.doi.org/10.
31219/osf.io/6wfkx

112. Gieselmann, M.A. and Thiele, A. (2022) Stimulus dependence
of directed information exchange between cortical layers in ma-
caque v1. Elife 11, e62949

113. Xing, D. et al. (2012) Laminar analysis of visually evoked activity
in the primary visual cortex. Proc. Natl. Acad. Sci. U. S. A. 109,
13871�13876

114. Livingstone, M.S. (1996) Oscillatory ?ring and interneuronal cor-
relations in squirrel monkey striate cortex. J. Neurophysiol. 75,
2467�2485

115. Kajikawa, Y. and Schroeder, C.E. (2015) Generation of ?eld po-
tentials and modulation of their dynamics through volume inte-
gration of cortical activity. J. Neurophysiol. 113, 339�351
116. Bollimunta, A. et al. (2008) Neuronal mechanisms of cortical
alpha oscillations in awake-behaving macaques. J. Neurosci.
28, 9976�9988

117. Halgren, M. et al. (2019) The generation and propagation of the
human alpha rhythm. Proc. Natl. Acad. Sci. 116, 23772�23782
118. Haegens, S. et al. (2015) Laminar pro?le and physiology of the ?
rhythm in primary visual, auditory, and somatosensory regions
of neocortex. J. Neurosci. 35, 14341�14352

119. Vinck, M. et al. (2022) The neural dynamics of feedforward and feed-
back interactions in predictive processing. PsyArXiv, Published on-
line October 31, 2021. http://dx.doi.org/10.31234/osf.io/n3afb
120. Witham, C.L. and Baker, S.N. (2012) Coding of digit displace-
ment by cell spiking and network oscillations in the monkey
sensorimotor cortex. J. Neurophysiol. 108, 3342�3352

121. Watanabe, H. et al. (2012) Reconstruction of movement-related
intracortical activity from micro-electrocorticogram array signals
in monkey primary motor cortex. J. Neural Eng. 9, 036006
122. Sherman, M.A. et al. (2016) Neural mechanisms of transient
neocortical beta rhythms: Converging evidence from humans,
computational modeling, monkeys, and mice. Proc. Natl.
Acad. Sci. 113, E4885�E4894

123. Bauer, M. et al. (2006) Tactile spatial attention enhances
gamma-band activity in somatosensory cortex and reduces
low-frequency activity in parieto-occipital areas. J. Neurosci.
26, 490�501

124. Schoffelen, J.M. et al. (2005) Neuronal coherence as a mecha-
interaction. Science 308,

nism of effective corticospinal
111�113

125. Ray, S. et al. (2008) Effect of stimulus intensity on the spike�
local ?eld potential relationship in the secondary somatosensory
cortex. J. Neurosci. 28, 7334�7343

126. Haegens, S. et al. (2011) Beta oscillations in the monkey senso-
rimotor network re?ect somatosensory decision making. Proc.
Natl. Acad. Sci. U. S. A. 108, 10708�10713

127. Miller, K. et al. (2009) Decoupling the cortical power spectrum
reveals real-time representation of individual ?nger movements
in humans. J. Neurosci. 29, 3132�3137

128. Jensen, M.A. et al. (2023) A motor association area in the

depths of the central sulcus. Nat. Neurosci. 26, 1165�1169

129. Ryun, S. et al. (2017) Tactile frequency-speci?c high-gamma
activities in human primary and secondary somatosensory cor-
tices. Sci. Rep. 7, 15442

148

Trends in Cognitive Sciences, February 2025, Vol. 29, No. 2


