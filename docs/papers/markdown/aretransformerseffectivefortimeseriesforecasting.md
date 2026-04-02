Are Transformers Effective for Time Series Forecasting?

Ailing Zeng1*, Muxi Chen1*, Lei Zhang2, Qiang Xu1
1The Chinese University of Hong Kong
2International Digital Economy Academy (IDEA)
{alzeng, mxchen21, qxu}@cse.cuhk.edu.hk

{leizhang}@idea.edu.cn

2
2
0
2

g
u
A
7
1

]
I

A
.
s
c
[

3
v
4
0
5
3
1
.
5
0
2
2
:
v
i
X
r
a

Abstract

Recently, there has been a surge of Transformer-based
solutions for the long-term time series forecasting (LTSF)
task. Despite the growing performance over the past few
years, we question the validity of this line of research in this
work. Speci?cally, Transformers is arguably the most suc-
cessful solution to extract the semantic correlations among
the elements in a long sequence. However, in time series
modeling, we are to extract the temporal relations in an
ordered set of continuous points. While employing posi-
tional encoding and using tokens to embed sub-series in
Transformers facilitate preserving some ordering informa-
tion, the nature of the permutation-invariant self-attention
mechanism inevitably results in temporal information loss.
To validate our claim, we introduce a set of embarrass-
ingly simple one-layer linear models named LTSF-Linear
for comparison. Experimental results on nine real-life
datasets show that LTSF-Linear surprisingly outperforms
existing sophisticated Transformer-based LTSF models in
all cases, and often by a large margin. Moreover, we con-
duct comprehensive empirical studies to explore the im-
pacts of various design elements of LTSF models on their
temporal relation extraction capability. We hope this sur-
prising ?nding opens up new research directions for the
LTSF task. We also advocate revisiting the validity of
Transformer-based solutions for other time series analysis
tasks (e.g., anomaly detection) in the future. Code is avail-
able at: https://github.com/cure-lab/LTSF-
Linear.

1. Introduction

Time series are ubiquitous in today�s data-driven world.
Given historical data, time series forecasting (TSF) is a
long-standing task that has a wide range of applications,
including but not limited to traf?c ?ow estimation, en-

*Equal contribution

ergy management, and ?nancial investment. Over the past
several decades, TSF solutions have undergone a progres-
sion from traditional statistical methods (e.g., ARIMA [1])
and machine learning techniques (e.g., GBRT [11]) to
deep learning-based solutions, e.g., Recurrent Neural Net-
works [15] and Temporal Convolutional Networks [3, 17].

Transformer [26] is arguably the most successful se-
quence modeling architecture, demonstrating unparalleled
performances in various applications, such as natural lan-
guage processing (NLP) [7], speech recognition [8], and
computer vision [19, 29]. Recently, there has also been a
surge of Transformer-based solutions for time series anal-
ysis, as surveyed in [27]. Most notable models, which
focus on the less explored and challenging long-term time
series forecasting (LTSF) problem, include LogTrans [16]
(NeurIPS 2019), Informer [30] (AAAI 2021 Best paper),
Autoformer [28] (NeurIPS 2021), Pyraformer [18] (ICLR
2022 Oral), Triformer [5] (IJCAI 2022) and the recent FED-
former [31] (ICML 2022).

The main working power of Transformers is from its
multi-head self-attention mechanism, which has a remark-
able capability of extracting semantic correlations among
elements in a long sequence (e.g., words in texts or 2D
patches in images). However, self-attention is permutation-
invariant and �anti-order� to some extent. While using var-
ious types of positional encoding techniques can preserve
some ordering information, it is still inevitable to have tem-
poral information loss after applying self-attention on top
of them. This is usually not a serious concern for semantic-
rich applications such as NLP, e.g., the semantic meaning
of a sentence is largely preserved even if we reorder some
words in it. However, when analyzing time series data,
there is usually a lack of semantics in the numerical data
itself, and we are mainly interested in modeling the tempo-
ral changes among a continuous set of points. That is, the
order itself plays the most crucial role. Consequently, we
pose the following intriguing question: Are Transformers
really effective for long-term time series forecasting?

Moreover, while existing Transformer-based LTSF so-

1

lutions have demonstrated considerable prediction accu-
racy improvements over traditional methods, in their exper-
iments, all the compared (non-Transformer) baselines per-
form autoregressive or iterated multi-step (IMS) forecast-
ing [1, 2, 22, 24], which are known to suffer from signi?cant
error accumulation effects for the LTSF problem. There-
fore, in this work, we challenge Transformer-based LTSF
solutions with direct multi-step (DMS) forecasting strate-
gies to validate their real performance.

Not all time series are predictable, let alone long-term
forecasting (e.g., for chaotic systems). We hypothesize that
long-term forecasting is only feasible for those time series
with a relatively clear trend and periodicity. As linear mod-
els can already extract such information, we introduce a set
of embarrassingly simple models named LTSF-Linear as a
new baseline for comparison. LTSF-Linear regresses histor-
ical time series with a one-layer linear model to forecast fu-
ture time series directly. We conduct extensive experiments
on nine widely-used benchmark datasets that cover various
real-life applications: traf?c, energy, economics, weather,
and disease predictions. Surprisingly, our results show that
LTSF-Linear outperforms existing complex Transformer-
based models in all cases, and often by a large margin (20%
? 50%). Moreover, we ?nd that, in contrast to the claims in
existing Transformers, most of them fail to extract temporal
relations from long sequences, i.e., the forecasting errors are
not reduced (sometimes even increased) with the increase of
look-back window sizes. Finally, we conduct various abla-
tion studies on existing Transformer-based TSF solutions to
study the impact of various design elements in them.
To sum up, the contributions of this work include:

� To the best of our knowledge, this is the ?rst work to
challenge the effectiveness of the booming Transform-
ers for the long-term time series forecasting task.

� To validate our claims, we introduce a set of em-
barrassingly simple one-layer linear models, named
LTSF-Linear,
and compare them with existing
Transformer-based LTSF solutions on nine bench-
marks. LTSF-Linear can be a new baseline for the
LTSF problem.

� We conduct comprehensive empirical studies on var-
ious aspects of existing Transformer-based solutions,
including the capability of modeling long inputs, the
sensitivity to time series order, the impact of posi-
tional encoding and sub-series embedding, and ef?-
ciency comparisons. Our ?ndings would bene?t future
research in this area.

With the above, we conclude that the temporal model-
ing capabilities of Transformers for time series are exag-
gerated, at least for the existing LTSF benchmarks. At the
same time, while LTSF-Linear achieves a better prediction

accuracy compared to existing works, it merely serves as a
simple baseline for future research on the challenging long-
term TSF problem. With our ?ndings, we also advocate
revisiting the validity of Transformer-based solutions for
other time series analysis tasks in the future.

2. Preliminaries: TSF Problem Formulation

C}L

1, ..., X t

For time series containing C variates, given historical
data X = {X t
t=1, wherein L is the look-back
window size and X t
i is the value of the ith variate at the tth
time step. The time series forecasting task is to predict the
values �X = { �X t
t=L+1 at the T future time steps.
When T > 1, iterated multi-step (IMS) forecasting [23]
learns a single-step forecaster and iteratively applies it to
obtain multi-step predictions. Alternatively, direct multi-
step (DMS) forecasting [4] directly optimizes the multi-step
forecasting objective at once.

1, ..., �X t

C}L+T

Compared to DMS forecasting results, IMS predictions
have smaller variance thanks to the autoregressive estima-
tion procedure, but they inevitably suffer from error accu-
mulation effects. Consequently, IMS forecasting is prefer-
able when there is a highly-accurate single-step forecaster,
and T is relatively small. In contrast, DMS forecasting gen-
erates more accurate predictions when it is hard to obtain an
unbiased single-step forecasting model, or T is large.

3. Transformer-Based LTSF Solutions

Transformer-based models [26] have achieved unparal-
leled performances in many long-standing AI tasks in natu-
ral language processing and computer vision ?elds, thanks
to the effectiveness of the multi-head self-attention mech-
anism. This has also triggered lots of research interest
in Transformer-based time series modeling techniques [20,
27]. In particular, a large amount of research works are ded-
icated to the LTSF task (e.g., [16, 18, 28, 30, 31]). Con-
sidering the ability to capture long-range dependencies
with Transformer models, most of them focus on the less-
explored long-term forecasting problem (T (cid:29) 1)1.

When applying the vanilla Transformer model to the
LTSF problem,
including the
it has some limitations,
quadratic time/memory complexity with the original self-
attention scheme and error accumulation caused by the au-
toregressive decoder design. Informer [30] addresses these
issues and proposes a novel Transformer architecture with
reduced complexity and a DMS forecasting strategy. Later,
more Transformer variants introduce various time series
features into their models for performance or ef?ciency im-
provements [18,28,31]. We summarize the design elements
of existing Transformer-based LTSF solutions as follows
(see Figure 1).

1Due to page limit, we leave the discussion of non-Transformer fore-

casting solutions in the Appendix.

2

Figure 1. The pipeline of existing Transformer-based TSF solutions. In (a) and (b), the solid boxes are essential operations, and the dotted
boxes are applied optionally. (c) and (d) are distinct for different methods [16, 18, 28, 30, 31].

Time series decomposition: For data preprocessing, nor-
malization with zero-mean is common in TSF. Besides,
Autoformer [28] ?rst applies seasonal-trend decomposition
behind each neural block, which is a standard method in
time series analysis to make raw data more predictable [6,
13]. Speci?cally, they use a moving average kernel on the
input sequence to extract the trend-cyclical component of
the time series. The difference between the original se-
quence and the trend component is regarded as the seasonal
component. On top of the decomposition scheme of Aut-
oformer, FEDformer [31] further proposes the mixture of
experts� strategies to mix the trend components extracted
by moving average kernels with various kernel sizes.
Input embedding strategies: The self-attention layer in
the Transformer architecture cannot preserve the positional
information of the time series. However, local positional
information, i.e.
the ordering of time series, is important.
Besides, global temporal information, such as hierarchical
timestamps (week, month, year) and agnostic timestamps
(holidays and events), is also informative [30]. To enhance
the temporal context of time-series inputs, a practical design
in the SOTA Transformer-based methods is injecting sev-
eral embeddings, like a ?xed positional encoding, a channel
projection embedding, and learnable temporal embeddings
into the input sequence. Moreover, temporal embeddings
with a temporal convolution layer [16] or learnable times-
tamps [28] are introduced.
Self-attention schemes: Transformers rely on the self-
attention mechanism to extract the semantic dependen-
cies between paired elements. Motivated by reducing
the O (cid:0)L2(cid:1) time and memory complexity of the vanilla
Transformer, recent works propose two strategies for ef-
?ciency. On the one hand, LogTrans and Pyraformer
explicitly introduce a sparsity bias into the self-attention
scheme. Speci?cally, LogTrans uses a Logsparse mask to
reduce the computational complexity to O (LlogL) while
Pyraformer adopts pyramidal attention that captures hierar-
chically multi-scale temporal dependencies with an O (L)
time and memory complexity. On the other hand, In-
former and FEDformer use the low-rank property in the
self-attention matrix. Informer proposes a ProbSparse self-

attention mechanism and a self-attention distilling operation
to decrease the complexity to O (LlogL), and FEDformer
designs a Fourier enhanced block and a wavelet enhanced
block with random selection to obtain O (L) complexity.
Lastly, Autoformer designs a series-wise auto-correlation
mechanism to replace the original self-attention layer.
Decoders: The vanilla Transformer decoder outputs se-
quences in an autoregressive manner, resulting in a slow in-
ference speed and error accumulation effects, especially for
long-term predictions. Informer designs a generative-style
decoder for DMS forecasting. Other Transformer variants
employ similar DMS strategies. For instance, Pyraformer
uses a fully-connected layer concatenating Spatio-temporal
axes as the decoder. Autoformer sums up two re?ned de-
composed features from trend-cyclical components and the
stacked auto-correlation mechanism for seasonal compo-
nents to get the ?nal prediction. FEDformer also uses a
decomposition scheme with the proposed frequency atten-
tion block to decode the ?nal results.

The premise of Transformer models is the semantic cor-
relations between paired elements, while the self-attention
mechanism itself is permutation-invariant, and its capabil-
ity of modeling temporal relations largely depends on posi-
tional encodings associated with input tokens. Considering
the raw numerical data in time series (e.g., stock prices or
electricity values), there are hardly any point-wise semantic
correlations between them. In time series modeling, we are
mainly interested in the temporal relations among a contin-
uous set of points, and the order of these elements instead
of the paired relationship plays the most crucial role. While
employing positional encoding and using tokens to embed
sub-series facilitate preserving some ordering information,
the nature of the permutation-invariant self-attention mech-
anism inevitably results in temporal information loss. Due
to the above observations, we are interested in revisiting the
effectiveness of Transformer-based LTSF solutions.

4. An Embarrassingly Simple Baseline

In the experiments of existing Transformer-based LTSF
solutions (T (cid:29) 1), all the compared (non-Transformer)

3

(d) Decoder(c) Encoder(b) Embedding(a) PreprocessingForecasting Output(d) Decoder(c) Encoder(b) Embedding(a) PreprocessingOutputInputSeasonal-trend decompositionNormalizationTimestamppreparationChannel projectionFixed positionLocal timestampGlobal timestampProbSparseand distilling self-attention @InformerSeries auto-correlation with decomposition @AutoformerMulti-resolution pyramidal     attention @PyraformerFrequency enhanced block with decomposition@FEDformerLogSparseand convolutional self-attention @LogTransDirect Multi-Step (DMS) @InformerDMS with auto-correlation and decomposition @AutoformerDMS along spatio-temporal dimension @PyraformerDMS with frequency attention anddecomposition@FEDformerIterated Multi-Step (IMS)@LogTransbaselines are IMS forecasting techniques, which are known
to suffer from signi?cant error accumulation effects. We
hypothesize that the performance improvements in these
works are largely due to the DMS strategy used in them.

5. Experiments

5.1. Experimental Settings

Dataset. We conduct extensive experiments on nine
widely-used real-world datasets, including ETT (Electricity
Transformer Temperature) [30] (ETTh1, ETTh2, ETTm1,
ETTm2), Traf?c, Electricity, Weather,
ILI, Exchange-
Rate [15]. All of them are multivariate time series. We
leave data descriptions in the Appendix.

Figure 2. Illustration of the basic linear model.

To validate this hypothesis, we present the simplest DMS
model via a temporal linear layer, named LTSF-Linear, as a
baseline for comparison. The basic formulation of LTSF-
Linear directly regresses historical time series for future
prediction via a weighted sum operation (as illustrated in
Figure 2). The mathematical expression is �Xi = W Xi,
where W ? RT �L is a linear layer along the temporal axis.
�Xi and Xi are the prediction and input for each ith vari-
ate. Note that LTSF-Linear shares weights across different
variates and does not model any spatial correlations.

LTSF-Linear is a set of linear models. Vanilla Linear is
a one-layer linear model. To handle time series across dif-
ferent domains (e.g., ?nance, traf?c, and energy domains),
we further introduce two variants with two preprocessing
methods, named DLinear and NLinear.

� Speci?cally, DLinear is a combination of a Decom-
position scheme used in Autoformer and FEDformer
with linear layers. It ?rst decomposes a raw data in-
put into a trend component by a moving average ker-
nel and a remainder (seasonal) component. Then, two
one-layer linear layers are applied to each component,
and we sum up the two features to get the ?nal predic-
tion. By explicitly handling trend, DLinear enhances
the performance of a vanilla linear when there is a clear
trend in the data.

� Meanwhile, to boost the performance of LTSF-Linear
when there is a distribution shift in the dataset, NLin-
ear ?rst subtracts the input by the last value of the se-
quence. Then, the input goes through a linear layer,
and the subtracted part is added back before making
the ?nal prediction. The subtraction and addition in
NLinear are a simple normalization for the input se-
quence.

We

Evaluation metric. Following previous works [28, 30,
31], we use Mean Squared Error (MSE) and Mean Absolute
Error (MAE) as the core metrics to compare performance.
Compared methods.
recent
[31], Aut-
Transformer-based methods:
oformer
and
[18],
LogTrans [16]. Besides, we include a naive DMS method:
Closest Repeat (Repeat), which repeats the last value in the
look-back window, as another simple baseline. Since there
are two variants of FEDformer, we compare the one with
better accuracy (FEDformer-f via Fourier transform).

FEDformer
[30], Pyraformer

include ?ve

Informer

[28],

5.2. Comparison with Transformers

Quantitative results. In Table 2, we extensively eval-
uate all mentioned Transformers on nine benchmarks, fol-
lowing the experimental setting of previous work [28, 30,
31]. Surprisingly, the performance of LTSF-Linear sur-
passes the SOTA FEDformer in most cases by 20% ? 50%
improvements on the multivariate forecasting, where LTSF-
Linear even does not model correlations among variates.
For different time series benchmarks, NLinear and DLin-
ear show the superiority to handle the distribution shift and
trend-seasonality features. We also provide results for uni-
variate forecasting of ETT datasets in the Appendix, where
LTSF-Linear still consistently outperforms Transformer-
based LTSF solutions by a large margin.

FEDformer achieves competitive forecasting accuracy
on ETTh1. This because FEDformer employs classical time
series analysis techniques such as frequency processing,
which brings in time series inductive bias and bene?ts the
ability of temporal feature extraction. In summary, these re-
sults reveal that existing complex Transformer-based LTSF
solutions are not seemingly effective on the existing nine
benchmarks while LTSF-Linear can be a powerful baseline.
Another interesting observation is that even though the
naive Repeat method shows worse results when predict-
ing long-term seasonal data (e.g., Electricity and Traf?c), it
surprisingly outperforms all Transformer-based methods on
Exchange-Rate (around 45%). This is mainly caused by the
wrong prediction of trends in Transformer-based solutions,
which may over?t toward sudden change noises in the train-
ing data, resulting in signi?cant accuracy degradation (see
Figure 3(b)). Instead, Repeat does not have the bias.

Qualitative results. As shown in Figure 3, we plot

4

Linear ??Linear ??Look-back WindowRemainderTrendForecasting Output?????�??????�??????�??????�??????�?????�?History ?timestepsFuture ?timesteps(a) The whole structure of DLinear(b) One Linear LayerDatasets
Variates
Timesteps
Granularity

ETTh1&ETTh2 ETTm1 &ETTm2

7
17,420
1hour

7
69,680
5min

Traf?c
862
17,544
1hour

Electricity Exchange-Rate Weather
8
7,588
1day

321
26,304
1hour

21
52,696
10min

ILI
7
966
1week

Table 1. The statistics of the nine popular datasets for the LTSF problem.

Linear*

DLinear*

IMP.
MSE

NLinear*
MSE MAE MSE MAE MSE MAE
0.237
0.249
0.267
0.301
0.203
0.293
0.414
0.601
0.282
0.287
0.296
0.315
0.237
0.282
0.319
0.362
1.081
0.963
1.024
1.096
0.399
0.416
0.443
0.490
0.353
0.418
0.465
0.551
0.343
0.365
0.386
0.421
0.260
0.303
0.342
0.421

0.140
0.153
0.169
0.203
0.081
0.157
0.305
0.643
0.410
0.423
0.436
0.466
0.176
0.220
0.265
0.323
2.215
1.963
2.130
2.368
0.375
0.405
0.439
0.472
0.289
0.383
0.448
0.605
0.299
0.335
0.369
0.425
0.167
0.224
0.281
0.397

0.237
0.250
0.268
0.301
0.207
0.304
0.432
0.750
0.282
0.287
0.295
0.315
0.236
0.276
0.312
0.365
0.985
1.036
1.060
1.104
0.397
0.429
0.476
0.592
0.352
0.413
0.461
0.595
0.352
0.369
0.393
0.435
0.262
0.308
0.373
0.435

0.141
0.154
0.171
0.210
0.089
0.180
0.331
1.033
0.410
0.423
0.435
0.464
0.182
0.225
0.271
0.338
1.683
1.703
1.719
1.819
0.374
0.408
0.429
0.440
0.277
0.344
0.357
0.394
0.306
0.349
0.375
0.433
0.167
0.221
0.274
0.368

0.237
0.248
0.265
0.297
0.208
0.300
0.415
0.780
0.279
0.284
0.290
0.307
0.232
0.269
0.301
0.348
0.858
0.859
0.884
0.917
0.394
0.415
0.427
0.453
0.338
0.381
0.400
0.436
0.348
0.375
0.388
0.422
0.255
0.293
0.327
0.384

27.40% 0.140
23.88% 0.153
21.02% 0.169
17.47% 0.203
45.27% 0.082
42.06% 0.167
33.69% 0.328
46.19% 0.964
30.15% 0.410
29.96% 0.423
29.95% 0.436
25.87% 0.466
18.89% 0.176
21.01% 0.218
22.71% 0.262
19.85% 0.326
47.86% 1.947
36.43% 2.182
34.43% 2.256
34.33% 2.390
0.80% 0.375
3.57% 0.418
6.54% 0.479
13.04% 0.624
19.94% 0.288
19.81% 0.377
25.93% 0.452
14.25% 0.698
21.10% 0.308
21.36% 0.340
17.07% 0.376
21.73% 0.440
17.73% 0.168
17.84% 0.232
15.69% 0.320
12.58% 0.413

FEDformer

Autoformer

Informer

Pyraformer*

LogTrans

Repeat*

MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE
0.946
0.193
0.950
0.201
0.961
0.214
0.975
0.246
0.196
0.148
0.289
0.271
0.396
0.460
0.681
1.195
1.079
0.587
1.087
0.604
1.095
0.621
1.097
0.626
0.254
0.217
0.292
0.276
0.338
0.339
0.394
0.403
1.701
3.228
1.884
2.679
1.798
2.622
1.677
2.857
0.713
0.376
0.733
0.420
0.744
0.459
0.756
0.506
0.422
0.346
0.473
0.429
0.508
0.496
0.517
0.463
0.665
0.379
0.690
0.426
0.707
0.445
0.729
0.543
0.328
0.203
0.371
0.269
0.410
0.325
0.465
0.421

0.201
0.222
0.231
0.254
0.197
0.300
0.509
1.447
0.613
0.616
0.622
0.660
0.266
0.307
0.359
0.419
3.483
3.103
2.669
2.770
0.449
0.500
0.521
0.514
0.358
0.456
0.482
0.515
0.505
0.553
0.621
0.671
0.255
0.281
0.339
0.433

0.308
0.315
0.329
0.355
0.278
0.380
0.500
0.841
0.366
0.373
0.383
0.382
0.296
0.336
0.380
0.428
1.260
1.080
1.078
1.157
0.419
0.448
0.465
0.507
0.388
0.439
0.487
0.474
0.419
0.441
0.459
0.490
0.287
0.328
0.366
0.415

0.317
0.334
0.338
0.361
0.323
0.369
0.524
0.941
0.388
0.382
0.337
0.408
0.336
0.367
0.395
0.428
1.287
1.148
1.085
1.125
0.459
0.482
0.496
0.512
0.397
0.452
0.486
0.511
0.475
0.496
0.537
0.561
0.339
0.340
0.372
0.432

0.258
0.266
0.280
0.283
0.968
1.040
1.659
1.941
0.684
0.685
0.734
0.717
0.458
0.658
0.797
0.869
4.480
4.799
4.800
5.278
0.878
1.037
1.238
1.135
2.116
4.315
1.124
3.188
0.600
0.837
1.124
1.153
0.768
0.989
1.334
3.048

0.274
0.296
0.300
0.373
0.847
1.204
1.672
2.478
0.719
0.696
0.777
0.864
0.300
0.598
0.578
1.059
5.764
4.755
4.763
5.264
0.865
1.008
1.107
1.181
3.755
5.602
4.721
3.647
0.672
0.795
1.212
1.166
0.365
0.533
1.363
3.379

0.386
0.386
0.378
0.376
0.376
1.748
1.874
1.943
2.085
0.867
0.869
0.881
0.896
0.622
0.739
1.004
1.420
7.394
7.551
7.662
0.664
0.790
0.891
0.963
0.645
0.788
0.907
0.963
0.543
0.557
0.754
0.908
0.435
0.730
1.201
3.625

0.449
0.443
0.443
0.445
1.105
1.151
1.172
1.206
0.468
0.467
0.469
0.473
0.556
0.624
0.753
0.934
2.012
2.031
2.057
2.100
0.612
0.681
0.738
0.782
0.597
0.683
0.747
0.783
0.510
0.537
0.655
0.724
0.507
0.673
0.845
1.451

0.368
0.386
0.394
0.439
0.752
0.895
1.036
1.310
0.391
0.379
0.420
0.472
0.384
0.544
0.523
0.741
1.677
1.467
1.469
1.564
0.713
0.792
0.809
0.865
1.525
1.931
1.835
1.625
0.571
0.669
0.871
0.823
0.453
0.563
0.887
1.338

0.357
0.368
0.380
0.376
0.812
0.851
1.081
1.127
0.384
0.390
0.408
0.396
0.490
0.589
0.652
0.675
1.444
1.467
1.468
1.560
0.740
0.824
0.932
0.852
1.197
1.635
1.604
1.540
0.546
0.700
0.832
0.820
0.642
0.757
0.872
1.328

1.588
1.595
1.617
1.647
0.081
0.167
0.305
0.823
2.723
2.756
2.791
2.811
0.259
0.309
0.377
0.465
6.587
7.130
6.575
5.893
1.295
1.325
1.323
1.339
0.432
0.534
0.591
0.588
1.214
1.261
1.283
1.319
0.266
0.340
0.412
0.521

W

e
h
t
a
e

c
?
f
a
r
T

g
n
a
h
c
x
E

t
i
c
i
r
t
c
e
l
E

Methods
Metric
y 96
192
336
720
e 96
192
336
720
96
192
336
720
r 96
192
336
720
24
36
48
60
1 96
192
336
720
2 96
192
336
720
1 96
m
192
T
336
T
E
720
2 96
m
192
T
336
T
E
720

h
T
T
E

h
T
T
E

I
L
I

- Methods* are implemented by us; Other results are from FEDformer [31].

Table 2. Multivariate long-term forecasting errors in terms of MSE and MAE, the lower the better. Among them, ILI dataset is with
forecasting horizon T ? {24, 36, 48, 60}. For the others, T ? {96, 192, 336, 720}. Repeat repeats the last value in the look-back window.
The best results are highlighted in bold and the best results of Transformers are highlighted with a underline. Accordingly, IMP. is the
best result of linear models compared to the results of Transformer-based solutions.

the prediction results on three selected time series datasets
with Transformer-based solutions and LTSF-Linear: Elec-
tricity (Sequence 1951, Variate 36), Exchange-Rate (Se-
quence 676, Variate 3), and ETTh2 ( Sequence 1241, Vari-
ate 2), where these datasets have different temporal patterns.
When the input length is 96 steps, and the output horizon
is 336 steps, Transformers [28, 30, 31] fail to capture the
scale and bias of the future data on Electricity and ETTh2.
Moreover, they can hardly predict a proper trend on aperi-
odic data such as Exchange-Rate. These phenomena further
indicate the inadequacy of existing Transformer-based solu-
tions for the LTSF task.

5.3. More Analyses on LTSF-Transformers

Can existing LTSF-Transformers extract temporal re-
lations well from longer input sequences? The size of the
look-back window greatly impacts forecasting accuracy as

5

it determines how much we can learn from historical data.
Generally speaking, a powerful TSF model with a strong
temporal relation extraction capability should be able to
achieve better results with larger look-back window sizes.

input

impact of
conduct

look-back win-
?

To study the
experiments with L
dow sizes, we
{24, 48, 72, 96, 120, 144, 168, 192, 336, 504, 672, 720}
for long-term forecasting (T=720). Figure 4 demonstrates
the MSE results on two datasets. Similar to the observations
from previous studies [27, 30], existing Transformer-based
models� performance deteriorates or stays stable when
the look-back window size increases.
the
performances of all LTSF-Linear are signi?cantly boosted
with the increase of look-back window size. Thus, existing
solutions tend to over?t temporal noises instead of extract-
ing temporal information if given a longer sequence, and
the input size 96 is exactly suitable for most Transformers.

In contrast,

(a) Electricity

(b) Exchange-Rate

(c) ETTh2

Figure 3. Illustration of the long-term forecasting output (Y-axis) of ?ve models with an input length L=96 and output length T =192
(X-axis) on Electricity, Exchange-Rate, and ETTh2, respectively.

Additionally, we provide more quantitative results in the
Appendix, and our conclusion holds in almost all cases.

(a) 720 steps-Traf?c

(b) 720 steps-Electricity

Figure 4. The MSE results (Y-axis) of models with different look-
back window sizes (X-axis) of long-term forecasting (T=720) on
the Traf?c and Electricity datasets.

What can be learned for long-term forecasting? While
the temporal dynamics in the look-back window signi?-
cantly impact the forecasting accuracy of short-term time
series forecasting, we hypothesize that long-term forecast-
ing depends on whether models can capture the trend and
periodicity well only. That is, the farther the forecasting
horizon, the less impact the look-back window itself has.

Methods
Input
Electricity
Traf?c

FEDformer
Far
0.265
0.645

Close
0.251
0.631

Autoformer
Far
0.287
0.675

Close
0.255
0.677

Table 3. Comparison of different input sequences under the MSE
metric to explore what LTSF-Transformers depend on. If the in-
put is Close, we use the 96th, ..., 191th time steps as the input
sequence. If the input is Far, we use the 0th, ..., 95th time steps.
Both of them forecast the 192th, ..., (192 + 720)th time steps.

To validate the above hypothesis, in Table 3, we compare
the forecasting accuracy for the same future 720 time steps
the
with data from two different look-back windows: (i).
original input L=96 setting (called Close) and (ii). the far
input L=96 setting (called Far) that is before the original

6

96 time steps. From the experimental results, the perfor-
mance of the SOTA Transformers drops slightly, indicat-
ing these models only capture similar temporal information
from the adjacent time series sequence. Since capturing the
intrinsic characteristics of the dataset generally does not re-
quire a large number of parameters, i,e. one parameter can
represent the periodicity. Using too many parameters will
even cause over?tting, which partially explains why LTSF-
Linear performs better than Transformer-based methods.

Are the self-attention scheme effective for LTSF? We
verify whether these complex designs in the existing Trans-
former (e.g., Informer) are essential. In Table 4, we gradu-
ally transform Informer to Linear. First, we replace each
self-attention layer by a linear layer, called Att.-Linear,
since a self-attention layer can be regarded as a fully-
connected layer where weights are dynamically changed.
Furthermore, we discard other auxiliary designs (e.g., FFN)
in Informer to leave embedding layers and linear layers,
named Embed + Linear. Finally, we simplify the model to
one linear layer. Surprisingly, the performance of Informer
grows with the gradual simpli?cation, indicating the unnec-
essary of the self-attention scheme and other complex mod-
ules at least for existing LTSF benchmarks.

Methods
e 96
g
n
192
a
h
336
c
x
E
720
1 96
192
336
720

h
T
T
E

Informer Att.-Linear

0.847
1.204
1.672
2.478
0.865
1.008
1.107
1.181

1.003
0.979
1.498
2.102
0.613
0.759
0.921
0.902

Embed + Linear
0.173
0.443
1.288
2.026
0.454
0.686
0.821
1.051

Linear
0.084
0.155
0.301
0.763
0.400
0.438
0.479
0.515

Table 4. The MSE comparisons of gradually transforming In-
former to a Linear from the left to right columns. Att.-Linear is
a structure that replaces each attention layer with a linear layer.
Embed + Linear is to drop other designs and only keeps embed-
ding layers and a linear layer. The look-back window size is 96.

Can existing LTSF-Transformers preserve temporal
Self-attention is inherently permutation-

order well?

0501001502002503001.51.00.50.00.51.0GrouthTruthAutoformerInformerFEDformerDLinear050100150200250300101234GrouthTruthAutoformerInformerFEDformerDLinear0501001502002503001.51.00.50.00.51.0GrouthTruthAutoformerInformerFEDformerDLinear244872961201441681923365046727200.40.60.81.01.21.4TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear244872961201441681923365046727200.200.250.300.350.40TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinearMethods
Predict Length

e
g
n
a
h
c
x
E

1
h
T
T
E

96
192
336
720
Average Drop
96
192
336
720
Average Drop

Linear
Half-Ex.
Shuf.
0.169
0.133
0.243
0.208
0.345
0.320
0.836
0.819
27.26% 46.81%
0.431
0.824
0.471
0.824
0.505
0.825
0.528
0.846
81.06% 4.78%

FEDformer

Half-Ex.
Shuf.
0.162
0.160
0.275
0.275
0.439
0.439
1.122
1.122
0.20%
-0.09%
0.405
0.753
0.436
0.730
0.453
0.736
0.470
0.720
73.28% 3.44%

Ori.
0.161
0.274
0.439
1.122
N/A
0.376
0.419
0.447
0.468
N/A

Autoformer

Half-Ex.
Shuf.
0.160
0.158
0.277
0.271
0.435
0.430
1.113
1.113
1.12%
0.09%
0.458
0.838
0.491
0.774
0.497
0.752
0.524
0.696
56.91% 0.46%

Ori.
0.152
0.278
0.435
1.113
N/A
0.455
0.486
0.496
0.525
N/A

Informer
Half-Ex.
Shuf.
0.959
1.004
1.014
1.023
1.177
1.181
1.196
1.210
-0.12% -0.18%
0.971
0.971
1.231
1.232
1.691
1.693
2.715
2.716
0.18%
1.98%

Ori.
0.952
1.012
1.177
1.198
N/A
0.974
1.233
1.693
2.720
N/A

Ori.
0.080
0.162
0.286
0.806
N/A
0.395
0.447
0.490
0.520
N/A

Table 5. The MSE comparisons of models when shuf?ing the raw input sequence. Shuf. randomly shuf?es the input sequence. Half-EX.
randomly exchanges the ?rst half of the input sequences with the second half. Average Drop is the average performance drop under all
forecasting lengths after shuf?ing. All results are the average test MSE of ?ve runs.

invariant, i.e., regardless of the order. However, in time-
series forecasting, the sequence order often plays a crucial
role. We argue that even with positional and temporal em-
beddings, existing Transformer-based methods still suffer
from temporal information loss. In Table 5, we shuf?e the
raw input before the embedding strategies. Two shuf?ing
strategies are presented: Shuf. randomly shuf?es the whole
input sequences and Half-Ex. exchanges the ?rst half of
the input sequence with the second half. Interestingly, com-
pared with the original setting (Ori.) on the Exchange Rate,
the performance of all Transformer-based methods does not
?uctuate even when the input sequence is randomly shuf-
?ed. By contrary, the performance of LTSF-Linear is dam-
aged signi?cantly. These indicate that LTSF-Transformers
with different positional and temporal embeddings preserve
quite limited temporal relations and are prone to over?t on
noisy ?nancial data, while the LTSF-Linear can model the
order naturally and avoid over?tting with fewer parameters.
For the ETTh1 dataset, FEDformer and Autoformer in-
troduce time series inductive bias into their models, mak-
ing them can extract certain temporal information when the
dataset has more clear temporal patterns (e.g., periodicity)
than the Exchange Rate. Therefore, the average drops of
the two Transformers are 73.28% and 56.91% under the
Shuf. setting, where it loses the whole order information.
Moreover, Informer still suffers less from both Shuf. and
Half-Ex. settings due to its no such temporal inductive bias.
Overall, the average drops of LTSF-Linear are larger than
Transformer-based methods for all cases, indicating the ex-
isting Transformers do not preserve temporal order well.

How effective are different embedding strategies? We
study the bene?ts of position and timestamp embeddings
used in Transformer-based methods. In Table 6, the fore-
casting errors of Informer largely increase without posi-
tional embeddings (wo/Pos.). Without timestamp embed-
dings (wo/Temp.) will gradually damage the performance
of Informer as the forecasting lengths increase. Since In-
former uses a single time step for each token, it is necessary
to introduce temporal information in tokens.

Methods

Embedding

FEDformer

Autoformer

Informer

All
wo/Pos.
wo/Temp.
wo/Pos.-Temp.
All
wo/Pos.
wo/Temp.
wo/Pos.-Temp.
All
wo/Pos.
wo/Temp.
wo/Pos.-Temp.

Traf?c

96
0.597
0.587
0.613
0.613
0.629
0.613
0.681
0.672
0.719
1.035
0.754
1.038

192
0.606
0.604
0.623
0.622
0.647
0.616
0.665
0.811
0.696
1.186
0.780
1.351

336
0.627
0.621
0.650
0.648
0.676
0.622
0.908
1.133
0.777
1.307
0.903
1.491

720
0.649
0.626
0.677
0.663
0.638
0.660
0.769
1.300
0.864
1.472
1.259
1.512

Table 6. The MSE comparisons of different embedding strate-
gies on Transformer-based methods with look-back window size
96 and forecasting lengths {96, 192, 336, 720}.

Rather than using a single time step in each token, FED-
former and Autoformer input a sequence of timestamps to
embed the temporal information. Hence, they can achieve
comparable or even better performance without ?xed po-
sitional embeddings. However, without timestamp embed-
dings, the performance of Autoformer declines rapidly be-
cause of the loss of global temporal information. Instead,
thanks to the frequency-enhanced module proposed in FED-
former to introduce temporal inductive bias, it suffers less
from removing any position/timestamp embeddings.

Is training data size a limiting factor for existing LTSF-
Transformers? Some may argue that the poor performance
of Transformer-based solutions is due to the small sizes of
the benchmark datasets. Unlike computer vision or nat-
ural language processing tasks, TSF is performed on col-
lected time series, and it is dif?cult to scale up the training
In fact, the size of the training data would in-
data size.
deed have a signi?cant impact on the model performance.
Accordingly, we conduct experiments on Traf?c, compar-
ing the performance of the model trained on a full dataset
(17,544*0.7 hours), named Ori., with that trained on a
shortened dataset (8,760 hours, i.e., 1 year), called Short.
Unexpectedly, Table 7 presents that the prediction errors

7

with reduced training data are lower in most cases. This
might because the whole-year data maintains more clear
temporal features than a longer but incomplete data size.
While we cannot conclude that we should use less data for
training, it demonstrates that the training data scale is not
the limiting reason for the performances of Autoformer and
FEDformer.

contributions do not come from proposing a linear model
but rather from throwing out an important question, show-
ing surprising comparisons, and demonstrating why LTSF-
Transformers are not as effective as claimed in these works
through various perspectives. We sincerely hope our com-
prehensive studies can bene?t future work in this area.

Methods
Dataset
96
192
336
720

FEDformer
Short
Ori.
0.568
0.587
0.584
0.604
0.601
0.621
0.608
0.626

Autoformer
Short
Ori.
0.594
0.613
0.616
0.621
0.621
0.622
0.650
0.660

Table 7. The MSE comparison of two training data sizes.

Is ef?ciency really a top-level priority? Existing LTSF-
Transformers claim that the O (cid:0)L2(cid:1) complexity of the
vanilla Transformer is unaffordable for the LTSF problem.
Although they prove to be able to improve the theoretical
time and memory complexity from O (cid:0)L2(cid:1) to O (L), it is
unclear whether 1) the actual inference time and memory
cost on devices are improved, and 2) the memory issue is
unacceptable and urgent for today�s GPU (e.g., an NVIDIA
Titan XP here). In Table 8, we compare the average prac-
tical ef?ciencies with 5 runs. Interestingly, compared with
the vanilla Transformer (with the same DMS decoder), most
Transformer variants incur similar or even worse inference
time and parameters in practice. These follow-ups introduce
more additional design elements to make practical costs
high. Moreover, the memory cost of the vanilla Transformer
is practically acceptable, even for output length L = 720,
which weakens the importance of developing a memory-
ef?cient Transformers, at least for existing benchmarks.

Method
DLinear

MACs
0.04G
Transformer� 4.03G
3.93G
4.41G
0.80G
4.41G

Memory
Time
Parameter
687MiB
0.4ms
139.7K
6091MiB
26.8ms
13.61M
3869MiB
14.39M
49.3ms
Informer
7607MiB
14.91M 164.1ms
Autoformer
241.4M?
7017MiB
3.4ms
Pyraformer
4143MiB
40.5ms
20.68M
FEDformer
- � is modi?ed into the same one-step decoder, which is implemented in the source code from Autoformer.
- ? 236.7M parameters of Pyraformer come from its linear decoder.

Table 8. Comparison of practical ef?ciency of LTSF-Transformers
under L=96 and T=720 on the Electricity. MACs are the number of
multiply-accumulate operations. We use Dlinear for comparison
since it has the double cost in LTSF-Linear. The inference time
averages 5 runs.

6. Conclusion and Future Work

Conclusion. This work questions the effectiveness of
emerging favored Transformer-based solutions for the long-
term time series forecasting problem. We use an em-
barrassingly simple linear model LTSF-Linear as a DMS
forecasting baseline to verify our claims. Note that our

Future work. LTSF-Linear has a limited model ca-
pacity, and it merely serves a simple yet competitive base-
line with strong interpretability for future research. For ex-
ample, the one-layer linear network is hard to capture the
temporal dynamics caused by change points [25]. Conse-
quently, we believe there is a great potential for new model
designs, data processing, and benchmarks to tackle the chal-
lenging LTSF problem.

8

Appendix:
Are Transformers Effective for Time Series Forecasting?

In this Appendix, we provide descriptions of non-
Transformer-based TSF solutions, detailed experimental
settings, more comparisons under different look-back win-
dow sizes, and the visualization of LTSF-Linear on all
datasets. We also append our code to reproduce the results
shown in the paper.

A. Related Work: Non-Transformer-Based

TSF Solutions

As a long-standing problem with a wide range of ap-
plications, statistical approaches (e.g., autoregressive inte-
grated moving average (ARIMA) [1], exponential smooth-
ing [12], and structural models [14]) for time series fore-
casting have been used from the 1970s onward. Generally
speaking, the parametric models used in statistical methods
require signi?cant domain expertise to build.

To relieve this burden, many machine learning
techniques such as gradient boosting regression tree
(GBRT) [10, 11] gain popularity, which learns the tempo-
ral dynamics of time series in a data-driven manner. How-
ever, these methods still require manual feature engineer-
ing and model designs. With the powerful representation
learning capability of deep neural networks (DNNs) from
abundant data, various deep learning-based TSF solutions
are proposed in the literature, achieving better forecasting
accuracy than traditional techniques in many cases.

Besides Transformers, the other two popular DNN archi-

tectures are also applied for time series forecasting:

� Recurrent neural networks (RNNs) based methods
(e.g., [21]) summarize the past information compactly
in internal memory states and recursively update them-
selves for forecasting.

� Convolutional neural networks (CNNs) based meth-
ods (e.g., [3]), wherein convolutional ?lters are used
to capture local temporal features.

RNN-based TSF methods belong to IMS forecasting
techniques. Depending on whether the decoder is imple-
mented in an autoregressive manner, there are either IMS
or DMS forecasting techniques for CNN-based TSF meth-
ods [3, 17].

B. Experimental Details

B.1. Data Descriptions

We use nine wildly-used datasets in the main paper. The

� ETT (Electricity Transformer Temperature) [30]2 con-
sists of two hourly-level datasets (ETTh) and two 15-
minute-level datasets (ETTm). Each of them contains
seven oil and load features of electricity transformers
from July 2016 to July 2018.

� Traf?c3 describes the road occupancy rates.

It con-
tains the hourly data recorded by the sensors of San
Francisco freeways from 2015 to 2016.

� Electricity4 collects the hourly electricity consumption

of 321 clients from 2012 to 2014.

� Exchange-Rate [15]5 collects the daily exchange rates

of 8 countries from 1990 to 2016.

� Weather6 includes 21 indicators of weather, such as air
temperature, and humidity. Its data is recorded every
10 min for 2020 in Germany.

� ILI7 describes the ratio of patients seen with in?uenza-
like illness and the number of patients.
It includes
weekly data from the Centers for Disease Control and
Prevention of the United States from 2002 to 2021.

B.2. Implementation Details

For existing Transformer-based TSF solutions: the im-
plementation of Autoformer [28], Informer [30], and the
vanilla Transformer [26] are all taken from the Autoformer
work [28];
the implementation of FEDformer [31] and
Pyraformer [18] are from their respective code repository.
We also adopt their default hyper-parameters to train the
models. For DLinear, the moving average kernel size for
decomposition is 25, which is the same as Autoformer. The
total parameters of a vanilla linear model and a NLinear
are TL. The total parameters of the DLinear are 2TL. Since
LTSF-Linear will be under?tting when the input length is
short, and LTSF-Transformers tend to over?t on a long
lookback window size. To compare the best performance
of existing LTSF-Transformers with LTSF-Linear, we re-
port L=336 for LTSF-Linear and L=96 for Transformers by
default. For more hyper-parameters of LTSF-Linear, please
refer to our code.

2https://github.com/zhouhaoyi/ETDataset
3http://pems.dot.ca.gov
4https : / / archive . ics . uci . edu / ml / datasets /

ElectricityLoadDiagrams20112014

5https : / / github . com / laiguokun / multivariate -

time-series-data

6https://www.bgc-jena.mpg.de/wetter/
7https : / / gis . cdc . gov / grasp / fluview /

details are listed in the following.

fluportaldashboard.html

9

C. Additional Comparison with Transformers

We

further

compare LTSF-Linear with LTSF-
Transformer
for Univariate Forecasting on four ETT
datasets. Moreover, in Figure 4 of the main paper, we
demonstrate that existing Transformers fail to exploit large
look-back window sizes with two examples. Here, we give
comprehensive comparisons between LTSF-Linear and
Transformer-based TSF solutions under various look-back
window sizes on all benchmarks.

C.1. Comparison of Univariate Forecasting

We present the univariate forecasting results on the four
ETT datasets in table 9. Similarly, LTSF-Linear, especially
for NLinear can consistently outperform all transformer-
based methods by a large margin in most time. We ?nd
that there are serious distribution shifts between training
and test sets (as shown in Fig. 5 (a), (b)) on ETTh1 and
ETTh2 datasets. Simply normalization via the last value
from the lookback window can greatly relieve the distribu-
tion shift problem.

C.2. Comparison under Different Look-back Win-

dows

In Figure 6, we provide the MSE comparisons of ?ve
LTSF-Transformers with LTSF-Linear under different look-
back window sizes to explore whether existing Transform-
ers can extract temporal well from longer input sequences.
For hourly granularity datasets (ETTh1, ETTh2, Traf?c,
and Electricity), the increasing look-back window sizes are
{24, 48, 72, 96, 120, 144, 168, 192, 336, 504, 672, 720},
which represent {1, 2, 3, 4, 5, 6, 7, 8, 14, 21, 28, 30}
days. The forecasting steps are {24, 720}, which mean {1,
30} days. For 5-minute granularity datasets (ETTm1 and
ETTm2), we set the look-back window size as {24, 36, 48,
60, 72, 144, 288}, which represent {2, 3, 4, 5, 6, 12, 24}
hours. For 10-minute granularity datasets (Weather), we set
the look-back window size as {24, 48, 72, 96, 120, 144,
168, 192, 336, 504, 672, 720}, which mean {4, 8, 12, 16,
20, 24, 28, 32, 56, 84, 112, 120} hours. The forecasting
steps are {24, 720} that are {4, 120} hours. For weekly
granularity dataset (ILI), we set the look-back window size
as {26, 52, 78, 104, 130, 156, 208}, which represent {0.5, 1,
1.5, 2, 2.5, 3, 3.5, 4} years. The corresponding forecasting
steps are {26, 208}, meaning {0.5, 4} years.

As shown in Figure 6, with increased look-back win-
dow sizes, the performance of LTSF-Linear is signi?cantly
boosted for most datasets (e.g., ETTm1 and Traf?c), while
this is not the case for Transformer-based TSF solutions.
Most of their performance ?uctuates or gets worse as
the input lengths increase. To be speci?c, the results of
Exchange-Rate do not show improved results with a long
look-back window (from Figure 6(m) and (n)), and we at-

10

tribute it to the low information-to-noise ratio in such ?nan-
cial data.

D. Ablation study on the LTSF-Linear

D.1. Motivation of NLinear

If we normalize the test data by the mean and variance of
train data, there could be a distribution shift in testing data,
i.e, the mean value of testing data is not 0. If the model
made a prediction that is out of the distribution of true value,
a large error would occur. For example, there is a large er-
ror between the true value and the true value minus/add one.
Therefore, in NLinear, we use the subtraction and addition
to shift the model prediction toward the distribution of true
value. Then, large errors are avoided, and the model perfor-
mances can be improved. Figure 5 illustrates histograms of
the trainset-test set distributions, where each bar represents
the number of data points. Clear distribution shifts between
training and testing data can be observed in ETTh1, ETTh2,
and ILI. Accordingly, from Table 9 and Table 2 in the main
paper, we can observe that there are great improvements
in the three datasets comparing the NLinear to the Linear,
showing the effectiveness of the NLinear in relieving dis-
tribution shifts. Moreover, for the datasets without obvi-
ous distribution shifts, like Electricity in Figure 5(c), using
the vanilla Linear can be enough, demonstrating the similar
performance with NLinear and DLinear.

D.2. The Features of LTSF-Linear

Although LTSF-Linear is simple, it has some compelling

characteristics:

� An O(1) maximum signal traversing path length:
The shorter the path, the better the dependencies are
captured [18], making LTSF-Linear capable of cap-
turing both short-range and long-range temporal rela-
tions.

� High-ef?ciency: As LTSF-Linear is a linear model
with two linear layers at most, it costs much lower
memory and fewer parameters and has a faster infer-
ence speed than existing Transformers (see Table 8 in
main paper).

� Interpretability: After training, we can visualize
weights from the seasonality and trend branches to
have some insights on the predicted values [9].

� Easy-to-use: LTSF-Linear can be obtained easily

without tuning model hyper-parameters.

D.3. Interpretability of LTSF-Linear

Because LTSF-Linear is a set of linear models,
the
weights of linear layers can directly reveal how LTSF-
Linear works. The weight visualization of LTSF-Linear can

Linear

DLinear

NLinear

Informer

LogTrans

Autoformer

FEDformer-f

FEDformer-w

Methods
Metric MSE MAE MSE MSE MAE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE
0.468
0.215
0.409
0.245
0.546
0.270
0.629
0.299
0.379
0.271
0.429
0.330
0.437
0.378
0.387
0.420
0.171
0.140
0.317
0.186
0.459
0.231
0.579
0.250
0.208
0.198
0.275
0.245
0.302
0.279
0.321
0.325

1 96
h
192
T
336
T
E
720
2 96
h
192
T
336
T
E
720
1 96
m
192
T
336
T
E
720
2 96
m
192
T
336
T
E
720

0.377
0.395
0.381
0.355
0.373
0.387
0.401
0.439
0.277
0.310
0.591
0.586
0.225
0.283
0.336
0.435

0.193
0.217
0.202
0.183
0.213
0.227
0.242
0.291
0.109
0.151
0.427
0.438
0.088
0.132
0.180
0.300

0.283
0.234
0.386
0.475
0.217
0.281
0.293
0.218
0.049
0.157
0.289
0.430
0.075
0.129
0.154
0.160

0.071
0.114
0.107
0.126
0.153
0.204
0.246
0.268
0.056
0.081
0.076
0.110
0.065
0.118
0.154
0.182

0.206
0.262
0.258
0.283
0.306
0.351
0.389
0.409
0.183
0.216
0.218
0.267
0.189
0.256
0.305
0.335

0.214
0.256
0.269
0.280
0.306
0.380
0.412
0.438
0.149
0.206
0.209
0.248
0.189
0.252
0.301
0.368

0.080
0.105
0.120
0.127
0.156
0.238
0.271
0.288
0.036
0.069
0.071
0.105
0.063
0.110
0.147
0.219

0.180
0.204
0.244
0.359
0.279
0.329
0.367
0.426
0.123
0.156
0.182
0.210
0.183
0.227
0.261
0.320

0.079
0.104
0.119
0.142
0.128
0.185
0.231
0.278
0.033
0.058
0.084
0.102
0.067
0.102
0.130
0.178

0.177
0.204
0.226
0.226
0.278
0.324
0.355
0.381
0.122
0.149
0.172
0.207
0.182
0.223
0.259
0.318

0.189
0.078
0.091
0.172
0.133
0.176
0.213
0.292
0.028
0.043
0.059
0.080
0.066
0.094
0.120
0.175

0.359
0.212
0.237
0.340
0.283
0.330
0.371
0.440
0.125
0.154
0.180
0.211
0.189
0.230
0.263
0.320

0.056
0.071
0.098
0.189
0.131
0.176
0.209
0.276
0.028
0.045
0.061
0.080
0.063
0.092
0.119
0.175

0.053
0.069
0.081
0.080
0.129
0.169
0.194
0.225
0.026
0.039
0.052
0.073
0.063
0.090
0.117
0.170

Table 9. Univariate long sequence time-series forecasting results on ETT full benchmark. The best results are highlighted in bold and the
best results of Transformers are highlighted with a underline.

(a) ETTh1 channel6

(b) ETTh2 channel3

(c) Electricity channel3

(d) ILI channel6

Figure 5. Distribution of ETTh1, ETTh2, Electricity, and ILI dataset. A clear distribution shift between training and testing data can be
observed in ETTh1, ETTh2, and ILI.

also reveal certain characteristics in the data used for fore-
casting.

Here we take DLinear as an example. Accordingly, we
visualize the trend and remainder weights of all datasets
with a ?xed input length of 96 and four different forecasting
horizons. To obtain a smooth weight with a clear pattern in
visualization, we initialize the weights of the linear layers
in DLinear as 1/L rather than random initialization. That
is, we use the same weight for every forecasting time step
in the look-back window at the start of training.

How the model works: Figure 7(c) visualize the
weights of the trend and the remaining layers on the
Exchange-Rate dataset. Due to the lack of periodicity and
seasonality in ?nancial data, it is hard to observe clear pat-
terns, but the trend layer reveals greater weights of informa-
tion closer to the outputs, representing their larger contribu-
tions to the predicted values.

Periodicity of data: For Traf?c data, as shown in Fig-
ure 7(d), the model gives high weights to the latest time
step of the look-back window for the 0,23,47...719 forecast-

ing steps. Among these forecasting time steps, the 0, 167,
335, 503, 671 time steps have higher weights. Note that 24
time steps are a day, and 168 time steps are a week. This
indicates that Traf?c has a daily periodicity and a weekly
periodicity.

References

[1] Adebiyi A Ariyo, Adewumi O Adewumi, and
Charles K Ayo. Stock price prediction using the arima
model. In 2014 UKSim-AMSS 16th International Con-
ference on Computer Modelling and Simulation, pages
106�112. IEEE, 2014. 1, 2, 9

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua
Bengio. Neural machine translation by jointly learn-
ing to align and translate. arXiv: Computation and
Language, 2014. 2

[3] Shaojie Bai, J Zico Kolter, and Vladlen Koltun.
An empirical evaluation of generic convolutional and

11

(a) 24 steps-ETTh1

(b) 720 steps-ETTh1

(c) 24 steps-ETTh2

(d) 720 steps-ETTh2

(e) 24 steps-ETTm1

(f) 576 steps-ETTm1

(g) 24 steps-ETTm2

(h) 576 steps-ETTm2

(i) 24 steps-Weather

(j) 720 steps-Weather

(k) 24 steps-Traf?c

(l) 720 steps-Traf?c

(m) 24 steps-Exchange

(n) 720 steps-Exchange

(o) 24 steps-ILI

(p) 60 steps-ILI

Figure 6. The MSE results (Y-axis) of models with different look-back window sizes (X-axis) of the long-term forecasting (e.g., 720-time
steps) and the short-term forecasting (e.g., 24 time steps) on different benchmarks.

recurrent networks for sequence modeling.
preprint arXiv:1803.01271, 2018. 1, 9

arXiv

[4] Guillaume Chevillon. Direct multi-step estimation
Journal of Economic Surveys,

and forecasting.
21(4):746�785, 2007. 2

[5] Razvan-Gabriel Cirstea, Chenjuan Guo, Bin Yang,
Tung Kieu, Xuanyi Dong, and Shirui Pan. Triformer:

Triangular, variable-speci?c attentions for long se-
quence multivariate time series forecasting�full ver-
sion. arXiv preprint arXiv:2204.13767, 2022. 1

[6] R. B. Cleveland. Stl : A seasonal-trend decomposition
procedure based on loess. Journal of Of?ce Statistics,
1990. 3

[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and

12

244872961201441681923365046727200.30.40.50.60.70.80.91.0TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear244872961201441681923365046727200.40.60.81.01.21.4TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear244872961201441681923365046727200.20.40.60.81.01.2TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear2448729612014416819233650467272012345TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear24364860721442880.20.30.40.50.6TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear24364860721442880.40.50.60.70.80.91.01.11.2TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear24364860721442880.100.150.200.250.30TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear24364860721442881234TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear244872961201441681923365046727200.100.150.200.250.300.350.400.450.50TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear244872961201441681923365046727200.40.60.81.01.21.41.6TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear244872961201441681923365046727200.40.50.60.70.80.9TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear244872961201441681923365046727200.40.60.81.01.21.4TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear244872961201441681923365046727200.00.20.40.60.81.01.21.4TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear244872961201441681923365046727200.51.01.52.02.53.0TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear265278104130156208234567TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinear2652781041301562082345678TransformerInformerAutoformerFEDformerPyraformerLinearNLinearDLinearFigure 7. Visualization of the weights(T*L) of LTSF-Linear on several benchmarks. Models are trained with a look-back window L
(X-axis) and different forecasting time steps T (Y-axis). We show weights in the remainder and trend layer.

Kristina Toutanova. Bert: Pre-training of deep bidirec-

tional transformers for language understanding. arXiv

13

In-96, Out-96(b1) Remainder(b2) TrendIn-96, Out-192(b3) Remainder(b4) TrendIn-96, Out-336(b5) Remainder(b6) TrendIn-96, Out-720(b7) Remainder(b8) TrendElectricityIn-96, Out-96(c1) Remainder(c2) TrendIn-96, Out-192(c3) Remainder(c4) TrendIn-96, Out-336(c5) Remainder(c6) Trend(c7) Remainder(c8) TrendExchange-RateIn-36, Out-24(f1) Remainder(f2) TrendIn-36, Out-36(f4) TrendIn-36, Out-48(f5) Remainder(f6) Trend(f7) Remainder(f8) TrendILI(f3) RemainderIn-96, Out-96(d1) Remainder(d2) TrendIn-96, Out-192(d3) Remainder(d4) TrendIn-96, Out-336(d5) Remainder(d6) Trend(d7) Remainder(d8) TrendTrafficIn-96, Out-96(e1) Remainder(e2) TrendIn-96, Out-192(e3) Remainder(e4) TrendIn-96, Out-336(e5) Remainder(e6) Trend(e7) Remainder(e8) TrendWeatherIn-96, Out-720In-36, Out-60In-96, Out-720In-96, Out-720In-96, Out-96(a1) Remainder(a2) TrendIn-96, Out-168(a3) Remainder(a4) TrendIn-96, Out-336(a5) Remainder(a6) TrendIn-96, Out-720(a7) Remainder(a8) TrendETTh1preprint arXiv:1810.04805, 2018. 1

[8] Linhao Dong, Shuang Xu, and Bo Xu.

Speech-
transformer: a no-recurrence sequence-to-sequence
In 2018 IEEE Inter-
model for speech recognition.
national Conference on Acoustics, Speech and Signal
Processing (ICASSP), pages 5884�5888. IEEE, 2018.
1

[9] Ruijun Dong and Witold Pedrycz. A granular time se-
ries approach to long-term forecasting and trend fore-
casting. Physica A: Statistical Mechanics and its Ap-
plications, 387(13):3253�3270, 2008. 10

[10] Shereen Elsayed, Daniela Thyssens, Ahmed Rashed,
Hadi Samer Jomaa, and Lars Schmidt-Thieme. Do
we really need deep learning models for time series
forecasting? arXiv preprint arXiv:2101.02118, 2021.
9

[11] Jerome H Friedman. Greedy function approximation:
a gradient boosting machine. Annals of statistics,
pages 1189�1232, 2001. 1, 9

[12] Everette S Gardner Jr. Exponential smoothing: The
Journal of forecasting, 4(1):1�28,

state of the art.
1985. 9

[13] James Douglas Hamilton.

Time series analysis.

Princeton university press, 2020. 3

[14] Andrew C Harvey. Forecasting, structural time series

models and the kalman ?lter. 1990. 9

[15] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and
Hanxiao Liu. Modeling long- and short-term tempo-
ral patterns with deep neural networks. international
acm sigir conference on research and development in
information retrieval, 2017. 1, 4, 9

[16] Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou,
Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. En-
hancing the locality and breaking the memory bottle-
neck of transformer on time series forecasting. Ad-
vances in Neural Information Processing Systems, 32,
2019. 1, 2, 3, 4

[17] Minhao Liu, Ailing Zeng, Zhijian Xu, Qiuxia Lai, and
Qiang Xu. Time series is a special sequence: Fore-
casting with sample convolution and interaction. arXiv
preprint arXiv:2106.09305, 2021. 1, 9

[18] Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao
Lin, Alex X Liu, and Schahram Dustdar. Pyraformer:
Low-complexity pyramidal attention for long-range
time series modeling and forecasting. In International
Conference on Learning Representations, 2021. 1, 2,
3, 4, 9, 10

In Proceedings of the IEEE/CVF
shifted windows.
International Conference on Computer Vision, pages
10012�10022, 2021. 1

[20] LIU Minhao, Ailing Zeng, LAI Qiuxia, Ruiyuan Gao,
Min Li, Jing Qin, and Qiang Xu. T-wavenet: A tree-
structured wavelet neural network for time series sig-
nal analysis. In International Conference on Learning
Representations, 2021. 2

[21] G�bor Petneh�zi. Recurrent neural networks for time
series forecasting. arXiv preprint arXiv:1901.00069,
2019. 9

[22] David Salinas, Valentin Flunkert, and Jan Gasthaus.
Deepar: Probabilistic forecasting with autoregressive
recurrent networks. International Journal of Forecast-
ing, 2017. 2

[23] Souhaib Ben Taieb, Rob J Hyndman, et al. Recur-
sive and direct multi-step forecasting: the best of both
worlds, volume 19. Citeseer, 2012. 2

[24] Sean J. Taylor and Benjamin Letham. Forecasting at

scale. PeerJ Prepr., 2017. 2

[25] Gerrit JJ van den Burg and Christopher KI Williams.
An evaluation of change point detection algorithms.
arXiv preprint arXiv:2003.06222, 2020. 8

[26] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, ?ukasz
Kaiser, and Illia Polosukhin. Attention is all you need.
Advances in neural information processing systems,
30, 2017. 1, 2, 9

[27] Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi
Chen, Ziqing Ma, Junchi Yan, and Liang Sun. Trans-
arXiv preprint
formers in time series: A survey.
arXiv:2202.07125, 2022. 1, 2, 5

[28] Jiehui Xu, Jianmin Wang, Mingsheng Long, et al.
Autoformer: Decomposition transformers with auto-
correlation for long-term series forecasting. Advances
in Neural Information Processing Systems, 34, 2021.
1, 2, 3, 4, 5, 9

[29] Ailing Zeng, Xuan Ju, Lei Yang, Ruiyuan Gao,
Xizhou Zhu, Bo Dai, and Qiang Xu. Deciwatch: A
simple baseline for 10x ef?cient 2d and 3d pose esti-
mation. arXiv preprint arXiv:2203.08713, 2022. 1
[30] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai
Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.
Informer: Beyond ef?cient transformer for long se-
In The Thirty-Fifth
quence time-series forecasting.
AAAI Conference on Arti?cial Intelligence, AAAI
2021, Virtual Conference, volume 35, pages 11106�
11115. AAAI Press, 2021. 1, 2, 3, 4, 5, 9

[19] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei,
Zheng Zhang, Stephen Lin, and Baining Guo. Swin
transformer: Hierarchical vision transformer using

[31] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang,
Liang Sun, and Rong Jin. Fedformer: Frequency en-
hanced decomposed transformer for long-term series

14

forecasting. In International Conference on Machine
Learning, 2022. 1, 2, 3, 4, 5, 9

15


