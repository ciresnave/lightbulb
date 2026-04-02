Concise Machine Learning

Jonathan Richard Shewchuk
May 5, 2025

Department of Electrical Engineering and Computer Sciences
University of California at Berkeley
Berkeley, California 94720

Abstract

This report contains lecture notes for UC Berkeley�s introductory class on Machine Learning. It covers
many methods for classification and regression, including five and a half lectures on neural networks, and
a few methods for clustering and dimensionality reduction. It is concise because nothing is included that
cannot be written or spoken in a single semester�s lectures (with whiteboard lectures and almost no slides!)
and because the choice of topics is limited to a small selection of particularly useful, popular algorithms.

Supported in part by the National Science Foundation under Awards CCF-1423560 and CCF-1909204, in part by the University of
California Lab Fees Research Program, and in part by an Alfred P. Sloan Research Fellowship. The claims in this document are
those of the author. They are not endorsed by the sponsors or the U.S. Government.

Keywords: machine learning, classification, regression, density estimation, dimensionality reduction, clus-
tering, perceptrons, support vector machines (SVMs), Gaussian discriminant analysis, linear discriminant
analysis (LDA), quadratic discriminant analysis (QDA), logistic regression, decision trees, random forests,
ensemble learning, bagging, boosting, AdaBoost, neural networks, convolutional neural networks (CNNs,
ConvNets), residual neural networks (ResNets), batch normalization, AdamW, nearest neighbor search,
least-squares linear regression, logistic regression, polynomial regression, ridge regression, Lasso, bias-
variance decomposition, maximum likelihood estimation (MLE), principal components analysis (PCA),
singular value decomposition (SVD), random projection, latent factor analysis, latent semantic indexing,
k-means clustering, hierarchical clustering, spectral graph clustering, the kernel trick, learning theory

Contents

1

Introduction; Classification; Train, Validate, Test

2 Linear Classifiers, the Centroid Method, and Perceptrons

3 Perceptron Learning; Maximum Margin Classifiers

4 Soft-Margin Support Vector Machines; Features

5 Machine Learning Abstractions and Numerical Optimization

6 Decision Theory; Generative and Discriminative Models

7 Gaussian Discriminant Analysis; Maximum Likelihood Estimation

8 Eigenvectors and the (Anisotropic) Multivariate Normal Distribution

9 Anisotropic Gaussians: MLE, QDA, and LDA Revisited

10 Regression, including Least-Squares Linear and Logistic Regression

11 Polynomial and Weighted Regression; Newton�s Method; ROC Curves

12 Statistical Justifications; the Bias-Variance Decomposition

13 Shrinkage: Ridge Regression, Subset Selection, and Lasso

14 Decision Trees

15 More Decision Trees, Ensemble Learning, and Random Forests

16 Neural Networks

17 Vanishing Gradients; ReLUs; Output Units and Losses; Neurobiology

18 Neurobiology; Faster Neural Network Training

19 Convolutional Neural Networks

20 Unsupervised Learning: Principal Components Analysis

21 The Singular Value Decomposition; Clustering

i

1

7

13

18

25

31

36

41

47

54

59

65

71

76

81

89

96

102

109

117

126

22 The Pseudoinverse; Better Generalization for Neural Nets

23 Residual Networks; Batch Normalization; AdamW

24 Boosting; Nearest Neighbor Classification

25 Nearest Neighbor Algorithms: Voronoi Diagrams and k-d Trees

A Bonus Lecture: Learning Theory

B Bonus Lecture: The Kernel Trick

C Bonus Lecture: Spectral Graph Clustering

D Bonus Lecture: Multiple Eigenvectors; Latent Factor Analysis

E Bonus Lecture: High Dimensions; Random Projection

134

140

146

151

157

163

168

176

183

ii

About this Report

This report compiles my lectures notes for UC Berkeley�s class CS 189/289A, Machine Learning, which
is both an undergraduate and introductory graduate course. I hope it will serve as a fast introduction to
the subject for readers who are already comfortable with vector calculus, linear algebra, probability, and
statistics. Please consult my CS 189/289A web page1 as an addendum to this report; it includes an extended
description of each lecture and additional web links and reading assignments related to the lectures. Consider
this report and the web page to be living documents; both will be refined a bit every time I teach the class.

The term �lecture notes� has shifted to include long textbook-style treatments written by professors as
supplements to their classes. Not so here. This report compiles the actual notes that I lecture from. I call
it Concise Machine Learning because I include almost nothing that I do not have time to write or speak
during one fourteen-week semester of twice-weekly 80-minute lectures. (After holidays and the midterm
exam, that amounts to 25 lectures.) Words that appear [in brackets] are spoken; everything else is written on
the �whiteboard��in my class, a tablet computer. My whiteboard software permits me to incorporate (and
write on) figures, included here. However, I am largely anti-Powerpoint and I resort to prepared slides for
just one brief segment during the semester (to discuss the V1 visual cortex).

These notes might be ideal for mathematically sophisticated readers who want to learn the basics of machine
learning as quickly as possible. But they�re not ideal for everybody. The time limitation necessitates that
many details are omitted. I think that the most mathematically well-prepared readers will be able to fill in
those details themselves. But many readers, including most students who take the class, will need additional
readings or discussion sections for greater detail. My class web page lists additional readings for most of
the lectures, many of them from three textbooks that have been kindly made available for free on the web:
An Introduction to Statistical Learning with Applications in R,2 second edition, by Gareth James, Daniela
Witten, Trevor Hastie, and Robert Tibshirani, Springer, New York, 2021, ISBN # 978-1-0716-1417-4; The
Elements of Statistical Learning: Data Mining, Inference, and Prediction,3 second edition, by Trevor Hastie,
Robert Tibshirani, and Jerome Friedman, Springer, New York, 2008; and Deep Learning4 by Christopher
M. Bishop with Hugh Bishop, Springer, 2024. Readers wanting the verbose kind of �lecture notes� should
consider the fine ones written by Stanford University�s Andrew Ng.5 I have no interest in duplicating these
efforts; instead, I�m aiming for the neglected niche of �shortest introduction.� (And perhaps also �best stolen
illustrations.�)
The other thing that makes this report concise is the choice of topics. CS 189/289A was introduced at UC
Berkeley in the spring of 2013 by Prof. Jitendra Malik, and most of his topic choices remain intact here.
Jitendra told me that he only taught a machine learning algorithm if he or his collaborators had used it
successfully for some application. He said, �the machine learning course is too important to leave to the
machine learning experts��that is, users of machine learning algorithms often have a more clear-eyed view
of their usefulness than inventors of machine learning algorithms.

I thank Peter Bartlett, Alyosha Efros, Isabelle Guyon, and Jitendra Malik�the previous teachers of CS
189/289A�for their lectures and lecture notes, from which I learned the topic myself. While I�ve given the
lectures my own twist and rearranged the material a lot, I am ultimately making incremental improvements
(and perhaps incremental worsenings) to a structure they handed down to me.

1https://people.eecs.berkeley.edu/?jrs/189/
2https://www.statlearning.com
3https://hastie.su.domains/ElemStatLearn/
4https://www.bishopbook.com
5http://cs229.stanford.edu/notes2020spring/

iii

iv

1 Introduction; Classification; Train, Validate, Test

[Spring 2025]

CS 189 / 289A
Machine Learning
Jonathan Shewchuk
https://people.eecs.berkeley.edu/?jrs/189/

Homework 1 due next Wednesday.

Questions: Please use Ed Discussion, not email.
please use public for most questions so other people can benefit.]

[Ed Discussion has an option for private questions, but

For personal matters only, jrs@berkeley.edu

Discussion sections (Tue & Wed):

Attend any section. [We�ll put up a list on Ed Discussion.]
[We might have a few advanced sections, including research discussion or exam problem preparation.]
Sections start Tuesday. [Next week.]

[Enrollment: 736 students max. 349 waitlisted. Expecting many drops. EECS grads have highest priority;
CD/DS undergrads second; non-EECS grads third; a few concurrent enrollment students will be admitted.]

[Textbooks: Available free online. Linked from class web page.]

Prerequisites

[or another vector calculus course]

Vector calculus: Math 53
Linear algebra: Math 54, Math 110, or EE 16A+16B
Probability: CS 70, EECS 126, or Stat 134
Plentiful programming experience
NOT CS 188

[or another probability course]
[TAs have no obligation to look at your code.]

[or another linear algebra course]

Springer Texts in StatisticsGareth JamesDaniela WittenTrevor HastieRobert TibshiraniAn Introduction to Statistical Learningwith Applications in RSecond�EditionSpringer Series in StatisticsTrevor HastieRobert TibshiraniJerome FriedmanSpringer Series in StatisticsThe Elements ofStatistical LearningData Mining,Inference,and PredictionThe Elements of Statistical LearningDuring the past decade there has been an explosion in computation and information tech-nology.With it have come vast amounts ofdata in a variety offields such as medicine,biolo-gy,finance,and marketing.The challenge ofunderstanding these data has led to the devel-opment ofnew tools in the field ofstatistics,and spawned new areas such as data mining,machine learning,and bioinformatics.Many ofthese tools have common underpinnings butare often expressed with different terminology.This book describes the important ideas inthese areas in a common conceptual framework.While the approach is statistical,theemphasis is on concepts rather than mathematics.Many examples are given,with a liberaluse ofcolor graphics.It should be a valuable resource for statisticians and anyone interestedin data mining in science or industry.The book�s coverage is broad,from supervised learning(prediction) to unsupervised learning.The many topics include neural networks,supportvector machines,classification trees and boosting�the first comprehensive treatment ofthistopic in any book.This major new edition features many topics not covered in the original,including graphicalmodels,random forests,ensemble methods,least angle regression & path algorithms for thelasso,non-negative matrix factorization,and spectral clustering.There is also a chapter onmethods for �wide�data (p bigger than n),including multiple testing and false discovery rates.Trevor Hastie,Robert Tibshirani,and Jerome Friedmanare professors ofstatistics atStanford University.They are prominent researchers in this area:Hastie and Tibshiranideveloped generalized additive models and wrote a popular book ofthat title.Hastie co-developed much ofthe statistical modeling software and environment in R/S-PLUS andinvented principal curves and surfaces.Tibshirani proposed the lasso and is co-author ofthevery successful An Introduction to the Bootstrap.Friedman is the co-inventor ofmany data-mining tools including CART,MARS,projection pursuit and gradient boosting.�springer.comSTATISTICSISBN978-0-387-84857-0Trevor Hastie � Robert Tibshirani � Jerome FriedmanThe Elements of Statictical LearningHastie � Tibshirani � FriedmanSecond Edition2

Jonathan Richard Shewchuk

Grading: 189

40% 7 Homeworks. Late policy: 5 slip days total
20% Midterm: Monday, March 17, 7:00�9:00 PM
40% Final Exam: Friday, May 16, 3�6 PM (not on Berkeley time)

Grading: 289A

40% HW
20% Midterm
20% Final
20% Project

Cheating

� Discussion of HW problems is encouraged. Showing other students small amounts of code is okay.
� All homeworks, including programming, must be written individually.
� All code must be typed by you. Do not use LLMs, Autopilot, or chatbots to autocomplete or write

code, nor to answer math problems.

� You may use LLMs or chatbots to help debugging or understanding, but you MUST include a com-

plete transcript of the conversation in an appendix at the end of your homework.

� We will actively check for plagiarism.
� Typical penalty is a large NEGATIVE score, but I reserve right to give an instant F for even one

violation, and will always give an F for two.

[Last year, we had to punish 12 people for cheating. It was not fun. Please don�t make me do it again.]

CORE MATERIAL

� Finding patterns in data; using them to make predictions.
� Models and statistics help us understand patterns.
� Optimization algorithms �learn� the patterns.

[The most important part of this is the data. Data drives everything else.
You cannot learn much if you don�t have enough data.
You cannot learn much if your data has bad quality.
But it�s amazing what you can do if you have lots of good data.
Machine learning has changed a lot in the last two decades because the internet has made truly vast quantities
of data available. For instance, with a little patience you can download tens of millions of photographs. Then
you can build a 3D model of Paris.
Neural networks had fallen out of favor early this millennium, but they came back big around 2012 because
researchers found that they work so much better when you have vast quantities of data.]

Introduction; Classification; Train, Validate, Test

3

CLASSIFICATION

� Collect training points with class labels: reliable debtors & defaulted debtors
� Evaluate new applicants�predict their class

creditcardscrop.pdf (ISL, Figure 4.1) [The problem of classification. We are given data
points, each belonging to one of two classes: orange crosses represent people who de-
faulted on their credit cards, and blue circles represent those who didn�t. Then we are given
additional points whose class is unknown, and we are asked to predict what class each new
point is in. Given the credit card balance and annual income of new applicants, predict
whether they will default on their debt.]

[Draw this figure by hand. classify.pdf ]
[Draw 2 colors of dots, almost but not quite linearly separable.]
[�How do we classify a new point?� Draw a point in a third color.]
[One possibility: look at its nearest neighbor.]
[Another possibility: draw a linear decision boundary; label it.]
[Those are two different models for the nature of this data.]

4.2WhyNotLinearRegression?129BalanceIncomeDefaultDefault050010001500200025000200004000060000NoYes05001000150020002500BalanceNoYes0200004000060000IncomeFIGURE4.1.TheDefaultdataset.Left:Theannualincomesandmonthlycreditcardbalancesofanumberofindividuals.Theindividualswhodefaultedontheircreditcardpaymentsareshowninorange,andthosewhodidnotareshowninblue.Center:Boxplotsofbalanceasafunctionofdefaultstatus.Right:Boxplotsofincomeasafunctionofdefaultstatus.4.2WhyNotLinearRegression?Wehavestatedthatlinearregressionisnotappropriateinthecaseofaqualitativeresponse.Whynot?Supposethatwearetryingtopredictthemedicalconditionofapatientintheemergencyroomonthebasisofhersymptoms.Inthissimpli?edexample,therearethreepossiblediagnoses:stroke,drugoverdose,andepilepticseizure.Wecouldconsiderencodingthesevaluesasaquantita-tiveresponsevariable,Y,asfollows:Y=?????1ifstroke;2ifdrugoverdose;3ifepilepticseizure.Usingthiscoding,leastsquarescouldbeusedto?talinearregressionmodeltopredictYonthebasisofasetofpredictorsX1,...,Xp.Unfortunately,thiscodingimpliesanorderingontheoutcomes,puttingdrugoverdoseinbetweenstrokeandepilepticseizure,andinsistingthatthedi?erencebetweenstrokeanddrugoverdoseisthesameasthedi?erencebetweendrugoverdoseandepilepticseizure.Inpracticethereisnoparticularreasonthatthisneedstobethecase.Forinstance,onecouldchooseanequallyreasonablecoding,Y=?????1ifepilepticseizure;2ifstroke;3ifdrugoverdose.decision boundary4

Jonathan Richard Shewchuk

[We�ll learn some ways to compute linear decision boundaries in the next several lectures. But for now, let�s
compare these two methods.]

[Two examples of classifiers for
classnear.pdf, classlinear.pdf (ESL, Figures 2.3 & 2.1)
the same data: a nearest neighbor classifier (left) and a linear classifier (right). The decision
boundaries are in black.]

[At left we have a nearest neighbor classifier, which classifies a new point by finding the nearest point
in the training data, and assigning it the same class. At right we have a linear classifier, which guesses
that everything above the line is brown, and everything below the line is blue. At right, the linear decision
boundary�the black line�is explicitly computed by the classifier. At left, the decision boundary is not
computed; the classifier just takes a new point and computes the distances to all the training points.]

[The neighbor classifier at left has a big advantage: it classifies all the training data correctly, whereas the
linear classifier does not. But the linear classifier has an advantage too. Somebody please tell me what.]

classnear.pdf, classnear15.pdf (ESL, Figures 2.3 & 2.2)
and a 15-nearest neighbor classifier.

[A 1-nearest neighbor classifier

[The 15-nearest neighbor classifier classifies a new point by looking at its 15 nearest neighbors and letting
them vote for the correct class.]

[The left figure is an example of what�s called overfitting.
In the left figure, observe how intricate the
decision boundary is that separates the positive examples from the negative examples. It�s a bit too intricate
Intuitively, that smoothness is
to reflect reality.
probably more likely to correspond to reality.]

In the right figure, the decision boundary is smoother.

162.OverviewofSupervisedLearning1?Nearest Neighbor ClassifierooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooFIGURE2.3.Thesameclassi?cationexampleintwodimensionsasinFig-ure2.1.Theclassesarecodedasabinaryvariable(BLUE=0,ORANGE=1),andthenpredictedby1-nearest-neighborclassi?cation.2.3.3FromLeastSquarestoNearestNeighborsThelineardecisionboundaryfromleastsquaresisverysmooth,andap-parentlystableto?t.Itdoesappeartorelyheavilyontheassumptionthatalineardecisionboundaryisappropriate.Inlanguagewewilldevelopshortly,ithaslowvarianceandpotentiallyhighbias.Ontheotherhand,thek-nearest-neighborproceduresdonotappeartorelyonanystringentassumptionsabouttheunderlyingdata,andcanadapttoanysituation.However,anyparticularsubregionofthedecisionbound-arydependsonahandfulofinputpointsandtheirparticularpositions,andisthuswigglyandunstable�highvarianceandlowbias.Eachmethodhasitsownsituationsforwhichitworksbest;inparticularlinearregressionismoreappropriateforScenario1above,whilenearestneighborsaremoresuitableforScenario2.Thetimehascometoexposetheoracle!Thedatainfactweresimulatedfromamodelsomewherebe-tweenthetwo,butclosertoScenario2.Firstwegenerated10meansmkfromabivariateGaussiandistributionN((1,0)T,I)andlabeledthisclassBLUE.Similarly,10moreweredrawnfromN((0,1)T,I)andlabeledclassORANGE.Thenforeachclasswegenerated100observationsasfollows:foreachobservation,wepickedanmkatrandomwithprobability1/10,and2.3LeastSquaresandNearestNeighbors13Linear Regression of 0/1 Response...............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooFIGURE2.1.Aclassi?cationexampleintwodimensions.Theclassesarecodedasabinaryvariable(BLUE=0,ORANGE=1),andthen?tbylinearregression.Thelineisthedecisionboundaryde?nedbyxT�?=0.5.Theorangeshadedregiondenotesthatpartofinputspaceclassi?edasORANGE,whiletheblueregionisclassi?edasBLUE.ThesetofpointsinIR2classi?edasORANGEcorrespondsto{x:xT�?>0.5},indicatedinFigure2.1,andthetwopredictedclassesareseparatedbythedecisionboundary{x:xT�?=0.5},whichislinearinthiscase.Weseethatforthesedatathereareseveralmisclassi?cationsonbothsidesofthedecisionboundary.Perhapsourlinearmodelistoorigid�oraresucherrorsunavoidable?Rememberthattheseareerrorsonthetrainingdataitself,andwehavenotsaidwheretheconstructeddatacamefrom.Considerthetwopossiblescenarios:Scenario1:ThetrainingdataineachclassweregeneratedfrombivariateGaussiandistributionswithuncorrelatedcomponentsanddi?erentmeans.Scenario2:Thetrainingdataineachclasscamefromamixtureof10low-varianceGaussiandistributions,withindividualmeansthemselvesdistributedasGaussian.AmixtureofGaussiansisbestdescribedintermsofthegenerativemodel.One?rstgeneratesadiscretevariablethatdetermineswhichof162.OverviewofSupervisedLearning1?Nearest Neighbor ClassifierooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooFIGURE2.3.Thesameclassi?cationexampleintwodimensionsasinFig-ure2.1.Theclassesarecodedasabinaryvariable(BLUE=0,ORANGE=1),andthenpredictedby1-nearest-neighborclassi?cation.2.3.3FromLeastSquarestoNearestNeighborsThelineardecisionboundaryfromleastsquaresisverysmooth,andap-parentlystableto?t.Itdoesappeartorelyheavilyontheassumptionthatalineardecisionboundaryisappropriate.Inlanguagewewilldevelopshortly,ithaslowvarianceandpotentiallyhighbias.Ontheotherhand,thek-nearest-neighborproceduresdonotappeartorelyonanystringentassumptionsabouttheunderlyingdata,andcanadapttoanysituation.However,anyparticularsubregionofthedecisionbound-arydependsonahandfulofinputpointsandtheirparticularpositions,andisthuswigglyandunstable�highvarianceandlowbias.Eachmethodhasitsownsituationsforwhichitworksbest;inparticularlinearregressionismoreappropriateforScenario1above,whilenearestneighborsaremoresuitableforScenario2.Thetimehascometoexposetheoracle!Thedatainfactweresimulatedfromamodelsomewherebe-tweenthetwo,butclosertoScenario2.Firstwegenerated10meansmkfromabivariateGaussiandistributionN((1,0)T,I)andlabeledthisclassBLUE.Similarly,10moreweredrawnfromN((0,1)T,I)andlabeledclassORANGE.Thenforeachclasswegenerated100observationsasfollows:foreachobservation,wepickedanmkatrandomwithprobability1/10,and2.3LeastSquaresandNearestNeighbors1515-Nearest Neighbor Classifier...............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooFIGURE2.2.Thesameclassi?cationexampleintwodimensionsasinFig-ure2.1.Theclassesarecodedasabinaryvariable(BLUE=0,ORANGE=1)andthen?tby15-nearest-neighboraveragingasin(2.8).Thepredictedclassishencechosenbymajorityvoteamongstthe15-nearestneighbors.InFigure2.2weseethatfarfewertrainingobservationsaremisclassi?edthaninFigure2.1.Thisshouldnotgiveustoomuchcomfort,though,sinceinFigure2.3noneofthetrainingdataaremisclassi?ed.Alittlethoughtsuggeststhatfork-nearest-neighbor?ts,theerroronthetrainingdatashouldbeapproximatelyanincreasingfunctionofk,andwillalwaysbe0fork=1.Anindependenttestsetwouldgiveusamoresatisfactorymeansforcomparingthedi?erentmethods.Itappearsthatk-nearest-neighbor?tshaveasingleparameter,thenum-berofneighborsk,comparedtothepparametersinleast-squares?ts.Al-thoughthisisthecase,wewillseethatthee?ectivenumberofparametersofk-nearestneighborsisN/kandisgenerallybiggerthanp,anddecreaseswithincreasingk.Togetanideaofwhy,notethatiftheneighborhoodswerenonoverlapping,therewouldbeN/kneighborhoodsandwewould?toneparameter(amean)ineachneighborhood.Itisalsoclearthatwecannotusesum-of-squarederrorsonthetrainingsetasacriterionforpickingk,sincewewouldalwayspickk=1!Itwouldseemthatk-nearest-neighbormethodswouldbemoreappropriateforthemixtureScenario2describedabove,whileforGaussiandatathedecisionboundariesofk-nearestneighborswouldbeunnecessarilynoisy.Introduction; Classification; Train, Validate, Test

5

Classifying Digits

sevensones.pdf [In the MNIST digit recognition problem, we are given handwritten digits,
and we are asked to learn to distinguish them. See Homework 1.]

Express these images as vectors

3
0
0
3

3
0
0
3

3
2
1
3

3
3
3
3

?

?

???????????????????????????????????????????????????????????????????????????????

3
3
3
3
0
0
2
3
0
0
1
3
3
3
3
3

?

???????????????????????????????????????????????????????????????????????????????

Images are points in 16-dimensional space. Linear decision boundary is a hyperplane.

TRAIN, VALIDATE, TEST

How we classify:

� We are given labeled data�sample points with class labels.
� Hold back a subset of the labeled points, called the validation set. Maybe 20%. The other 80% is the
training set. [Warning: the term training data is not used consistently. Often �training data� refers to
all the labeled data. You have to judge from context.]

� Train one or more classifiers: they learn to distinguish 7 from not 7. Use training set to learn model

weights. Do NOT use validation set to train!!!

� Usually, train multiple learning algorithms, or one algorithm with multiple hyperparameter settings,

or both [using the same training set for each].

� Validate the trained classifiers on the validation set. Choose classifier/hyperparameters with lowest
validation error. Called validation. [When we do validation, we are not learning any more. We are
checking what classes our trained classifiers assign to our validation set, and counting how often
they�re right. We use this to judge our models�not how well they remember the training set labels.]
� Optional: Test the best classifier on a test set of NEW data. Final evaluation. Typically you do NOT

have the labels. [But somebody else might have them, and assign you a score!]

Classi?ca9on(Pipeline(�?Collect(Training(Images(�?Posi9ve:((�?Nega9ve:((�?Training(Time(�?Compute(feature(vectors(for(posi9ve(and(nega9ve(example(images(�?Train(a(classi?er(�?Test(Time(�?Compute(feature(vector(on(new(test(image:((�?Evaluate(classi?er((6

Jonathan Richard Shewchuk

[When I underline a word or phrase, that usually means it�s a definition. My advice to you is to memorize
the definitions I cover in class.]

3 kinds of error:

� Training error: fraction of training set not classified correctly. [This is zero with the 1-nearest neighbor
classifier, but nonzero with the 15-nearest neighbor and linear classifiers. But that doesn�t mean the
1-nearest neighbor classifier is always better. Remember that you cannot include the validation data
in this calculation, even if somebody calls it �training data.�]

� Validation error: fraction of validation set misclassified. Use this to choose classifier/hyperparameters.
[You didn�t use the validation set to train, so even the 1-nearest neighbor classifier can classify these
points wrong. Validation error is almost always higher than training error.]

� Test error: fraction of test set misclassified. Used to evaluate you.

Most ML algorithms have a few hyperparameters that control over/underfitting, e.g. k in k-nearest neighbors.

overfitlabeled.pdf (modified from ESL, Figure 2.4)

� overfitting: when the validation/test error deteriorates because the classifier becomes too sensitive to

outliers or other spurious patterns.

� underfitting: when the validation/test error deteriorates because the classifier is not flexible enough to

fit patterns.

� outliers: points with atypical labels (e.g., rich borrower who defaulted anyway).

Increase risk of

overfitting.

[In machine learning, the goal is to create a classifier that generalizes to new examples we haven�t seen yet.
Overfitting and underfitting are both counterproductive to that goal. So we�re always seeking a compromise:
we want decision boundaries that make fine distinctions without being downright superstitious.]

Kaggle.com:

� Runs ML competitions, including our HWs
� We may use 2 test sets:

public set: test scores available during competition
private set: test scores available after competition

[The private test set prevents you from �cheating� by throwing lots of models at the public test set
until you find a lucky one.]

error ratek: # of nearest neighbors0.100.150.200.250.30151101 69 45 31 21 11  7  5  3  1TrainTestBayesLinearover?t!best (7)under?ttest errortraining errorLinear Classifiers, the Centroid Method, and Perceptrons

7

2 Linear Classifiers, the Centroid Method, and Perceptrons

CLASSIFIERS

You are given sample of n observations [aka examples], each with d features [aka predictors].
Some observations belong to class C; some do not.

Example: Observations are ice cream lovers
Features are height & age (d = 2)
Some are in class �chocolate,� some prefer vanilla
Goal: Predict preferred flavor based on their height & age.

Represent each observation as a point in d-dimensional space,
called a sample point / a feature vector / independent variables.

overfitting

height

height

height

age

age

age

[Draw this by hand; decision boundaries last. classify3.pdf ]
[We draw these lines/curves separating C�s from V�s. Then we use these curves to predict which future
borrowers will default. In the last example, though, we�re probably overfitting, which could hurt our predic-
tions.]

decision boundary: the boundary chosen by our classifier to separate items in the class from those not.

overfitting: When decision boundary fits spurious detail so well that it doesn�t classify future points well.

[A reminder that underlined phrases are definitions, worth memorizing.]

Some (not all) classifiers work by computing a

decision function: A function f (x) that maps a point x to a scalar such that

f (x) > 0
f (x) ? 0

if x ? class C;
if x (cid:60) class C.

Aka predictor function or discriminant function.

For these classifiers, the decision boundary is {x ? Rd : f (x) = 0}
[That is, the set of all points where the decision function is zero.]
Usually, this set is a (d ? 1)-dimensional surface in Rd.
{x : f (x) = 0} is also called an isosurface of f for the isovalue 0.
f has other isosurfaces for other isovalues, e.g., {x : f (x) = 1}.

CCVVVVVVVVVCCCCCCCVVVVCCCVVVCCCCCVCVVVVVVVCCCCCC8

Jonathan Richard Shewchuk

radiusplot.pdf, radiusiso.pdf [3D plot and isocontour plot of the cone] f (x, y) = (cid:112)

x2 + y2 ? 3.

[Imagine a decision function in Rd, and imagine its (d ? 1)-dimensional isosurfaces.]

radiusiso3d.pdf

[One of these spheres could be the decision boundary.]
linear classifier: The decision boundary is a line/plane.
Usually uses a linear decision function.

-2-1012344445555-6-4-20246-6-4-20246Linear Classifiers, the Centroid Method, and Perceptrons

9

Linear Classifier Math

[I will write vectors in matrix notation.]
?

?

Vectors: x =

= [x1 x2 x3 x4 x5]?

????????????????????

x1
x2
x3
x4
x5

????????????????????

Think of x as a point in 5-dimensional space.

Conventions (often, but not always):

uppercase roman = matrix, random variable, set X
lowercase roman = vector
x
Greek = real scalar
?
n = # of sample points
Some integers:
d = # of features (per point)

= dimension of sample points

i j k = indices
f ( ), s( ), . . .

function (often scalar)

Euclidean inner product (aka dot product): x � y = x1y1 + x2y2 + ... + xdyd

also written x?y
Clearly,

f (x) = w � x + ? is a linear function in x.
(cid:113)

?

Euclidean norm: ?x? =

x � x =
?x? is the length (aka Euclidean length) of a vector x.
Given a vector x (cid:44) 0,
�Normalize a vector x�: replace x with x

x
?x? is a unit vector (length 1).

+ ... + x2
d

+ x2
2

x2
1

?x? .

Use dot products to compute angles:

y

?

x

acute

cos ? = x � y
?x? ?y?

=

�

x
?x?
(cid:124)(cid:123)(cid:122)(cid:125)
length 1

y
?y?
(cid:124)(cid:123)(cid:122)(cid:125)
length 1

right
(orthogonal)

obtuse

x � y > 0

x � y = 0

x � y < 0

Given a linear decision function f (x) = w � x + ?, the decision boundary is

H = {x : w � x = ??}.

The set H is called a hyperplane. (A line in 2D, a plane in 3D.)

[A hyperplane is what you get when you generalize the idea of a plane to higher dimensions. The three most
important things to understand about a hyperplane is (1) it has dimension d ? 1 and it cuts the d-dimensional
space into two halves; (2) it�s flat; and (3) it�s infinite.]

10

Jonathan Richard Shewchuk

Theorem: Let x, y be 2 points that lie on H. Then w � (y ? x) = 0.
Proof: w � (y ? x) = ?? ? (??) = 0.

[Therefore, w is orthogonal to any line segment that lies on H.]

w is called the normal vector of H,
because (as the theorem shows) w is normal (perpendicular) to H.
[I.e., w is perpendicular to every line on H.]

w � x = ?2

w

w � x = 1

w � x = 0

[Draw black part first, then red parts. hyperplane.pdf ]

If w is a unit vector, then f (x) = w � x + ? is the signed distance from x to H. [See Discussion 1.]
I.e., positive on w�s side of H; negative on other side.

Moreover, the distance from H to the origin is ?. [How do we know that?]
Hence ? = 0 if and only if H passes through origin.

[w does not have to be a unit vector for the classifier to work.
If w is not a unit vector, w � x + ? is the signed distance times some real.
If you want to fix that, you can rescale the equation by computing ?w? and dividing both w and ? by ?w?.]
The coefficients in w, plus ?, are called weights (or parameters or regression coefficients).

[That�s why we call the vector w; �w� stands for �weights.�]

The training points are linearly separable if there exists a hyperplane that correctly classifies all the training
points.

[At the beginning of this lecture, I showed you one plot that�s linearly separable and two that are not.]

[We will investigate some linear classifiers that only work for linearly separable data and some that do a
decent job with non-separable data. Obviously, if your data are not linearly separable, a linear classifier
cannot do a perfect job. But we�re still happy if we can find a classifier that usually predicts correctly.]

A Simple Classifier

Centroid method: compute mean �C of all training points in class C and mean �X of all points NOT in C.

We use the decision function

f (x) = (�C ? �X)
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
normal vector

�x ? (�C ? �X) �

�C + �X
2
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
midpoint between �C, �X

so the decision boundary is the hyperplane that bisects line segment w/endpoints �C, �X.

Linear Classifiers, the Centroid Method, and Perceptrons

11

[Draw data, then �C, �X, then line & normal. centroid.pdf ]

[In this example, there�s clearly a linear classifier that classifies every training point correctly, and the cen-
troid method isn�t it.
Note that this is hardly the worst example I could have given.
If you�re in the mood for an easy puzzle, pull out a sheet of paper and think of an example, with lots of
training points, where the centroid method misclassifies every training point but two.]

[Nevertheless, there are circumstances where this method works well, like when all your positive examples
come from one Gaussian distribution, and all your negative examples come from another.]

[We can sometimes improve this classifier by adjusting the scalar term ? to minimize the number of mis-
classified points. Then the hyperplane has the same normal vector, but a different position.]

Perceptron Algorithm (Frank Rosenblatt, 1957)

Slow, but correct for linearly separable points.

Uses a numerical optimization algorithm, namely, gradient descent.

[Poll:
How many of you know what gradient descent is?
How many of you know what the backpropagation algorithm is?
How many of you know what a linear program is?
How many of you know what a quadratic program is?

We�re going to learn what these things are. As machine learning people, we will be heavy users of optimiza-
tion methods. Unfortunately, I won�t have time to teach you algorithms for many optimization problems,
but we�ll learn a few. To learn more, take EECS 127.]

Consider n sample points X1, X2, ..., Xn.

[The reason I�m using capital X here is because we typically store these vectors in a matrix X.]

For each sample point, the label yi =

(cid:40)

1 if Xi ? class C, and
?1 if Xi (cid:60) C.

For simplicity, consider only decision boundaries that pass through the origin. (We�ll fix this later.)

XXXCCCCCXCXX12

Jonathan Richard Shewchuk

Goal: find weights w such that

Xi � w ? 0
Xi � w ? 0

if yi = 1, and
if yi = ?1.

[remember, Xi � w is the signed distance]

Equivalently: yiXi � w ? 0.

? inequality called a constraint.

Idea: We define a risk function R that is positive if some constraints are violated. Then we use optimization
to choose w that minimizes R.

[That�s how we train a perceptron classifier.]

Define the loss function
(cid:40)

L(�y, yi) =

0 if yi �y ? 0, and

?yi �y otherwise.

[Here, �y is the classifier�s prediction, and yi is the correct answer, called the label.]

If �y has the same sign as yi, the loss function is zero (happiness).
But if �y has the wrong sign, the loss function is positive.

[For each training point, you want to get the loss function down to zero, or as close to zero as possible. It�s
called the �loss function� because the bigger it is, the bigger a loser your classifier is.]

Define risk function (aka objective function or cost function)

R(w) = 1
n
= 1
n

n(cid:88)

i=1
(cid:88)

i?V

L(Xi � w, yi)

?yiXi � w

where V is the set of indices i for which yiXi � w < 0.

If w classifies all X1, . . . , Xn correctly, then R(w) = 0.
Otherwise, R(w) is positive, and we want to find a better w.

Goal: Solve this optimization problem:

Find w that minimizes R(w).

riskplot.pdf [Plot of risk R(w). Every point in the dark green flat spot is a minimum. We�ll
look at this more next lecture.]

Perceptron Learning; Maximum Margin Classifiers

13

3 Perceptron Learning; Maximum Margin Classifiers

Perceptron Algorithm (cont�d)

Recall:

� linear decision fn f (x) = w � x
� decision boundary {x : f (x) = 0}
� sample points X1, X2, . . . , Xn ? Rd; class labels y1, . . . , yn = �1
� goal: find weights w such that yiXi � w ? 0
� goal, revised: find w that minimizes R(w) =

?yiXi � w

(cid:88)

(for simplicity, no ?)
(a hyperplane through the origin)

[risk function]

where V = {i : yiXi � w < 0}.

i?V

[Our original problem was to find a separating hyperplane in one space, which I�ll call x-space. But we�ve
transformed this into a problem of finding an optimal point in a different space, which I�ll call w-space. It�s
important to understand transformations like this, where a geometric structure in one space becomes a point
in another space.]

Objects in x-space transform to objects in w-space:

x-space

hyperplane:
point:

{z : w � z = 0}
x

w-space

point:
hyperplane:

w
{z : x � z = 0}

Point x lies on hyperplane {z : w � z = 0} ? w � x = 0 ? point w lies on hyperplane {z : x � z = 0} in w-space.

[So a hyperplane transforms to a point that represents its normal vector. And a sample point transforms to
the hyperplane whose normal vector is the sample point.]

[In this algorithm, the transformations happen to be symmetric: a hyperplane in x-space transforms to a
point in w-space the same way that a hyperplane in w-space transforms to a point in x-space. That won�t
always be true for the decision boundaries we use this semester.]

If we want to enforce inequality x � w ? 0, that means

� in x-space, x should be on the same side of {z : w � z = 0} as w
� {z : x � z = 0} as x
� in w-space, w �

� � �

�

�

x-space

w-space

[Draw this by hand. xwspace.pdf ]
the x-space sample
[Observe that
points are the normal vectors for the
w-space lines. We can choose w to be
anywhere in the shaded region.]

w

w

[For a sample point x in class C, w and x must be on the same side of the hyperplane that x transforms into.
For a point x not in class C (marked by an X), w and x must be on opposite sides of the hyperplane that x
transforms into. These rules determine the shaded region above, in which w must lie.]

[Again, what have we accomplished? We have switched from the problem of finding a hyperplane in x-
space to the problem of finding a point in w-space. That�s a better fit to how we think about optimization
algorithms.]

CXX14

Jonathan Richard Shewchuk

[Let�s take a look at the risk function these three sample points create.]

riskplot.pdf, riskiso.pdf [Plot & isocontours of risk R(w). Note how R�s creases match the
lines in the w-space drawn above.]

[In this plot, we can choose w to be any point in the bottom pizza slice; all those points minimize R.]

[We have an optimization problem; we need an optimization algorithm to solve it.]

An optimization algorithm: gradient descent on R.

[Draw the typical steps of gradient descent on the plot of R.]

Given a starting point w, find gradient of R with respect to w; this is the direction of steepest ascent.
Take a step in the opposite direction. Recall [from your vector calculus class]

?R(w) =

?

??????????????????

?R(w) = ?

?R
?w1
?R

?w2...

?R
?wd
(cid:88)

i?V

?

??????????????????

and

?w(z � w) =

?

????????????????

z1
z2
...
zd

?

????????????????

= z

?yiXi � w = ?

(cid:88)

i?V

yiXi

At any point w, we walk downhill in direction of steepest descent, ??R(w).

w ? arbitrary nonzero starting point (good choice is any yiXi)
while R(w) > 0

V ? set of indices i for which yiXi � w < 0
w ? w + ?

(cid:88)

yiXi

return w

i?V

? > 0 is the step size aka learning rate, chosen empirically. [Best choice depends on input problem!]

Problem: Slow! Each step takes O(nd) time. [Can we improve this?]

-4-2024-4-2024Perceptron Learning; Maximum Margin Classifiers

15

Optimization algorithm 2: stochastic gradient descent

Idea:

each step, pick one misclassified Xi;
do gradient descent on loss fn L(Xi � w, yi).

Called the perceptron algorithm. Each step takes O(d) time.
[Not counting the time to search for a misclassified Xi.]

while some yiXi � w < 0
w ? w + ? yiXi

return w

[Stochastic gradient descent is quite popular and we�ll see it several times more this semester, especially
for neural networks. However, stochastic gradient descent does not work for every problem that gradient
descent works for. The perceptron risk function happens to have special properties that guarantee that
stochastic gradient descent will always succeed.]

What if separating hyperplane doesn�t pass through origin?
Add a fictitious dimension. Decision fn is

f (x) = w � x + ? = [w1 w2 ?] �

?

?????????

?

?????????

x1
x2
1

Now we have sample points in Rd+1, all lying on hyperplane xd+1 = 1.
Run perceptron algorithm in (d + 1)-dimensional space.
d dimensions by using a hyperplane through the origin in d + 1 dimensions.]

[We are simulating a general hyperplane in

[The perceptron algorithm was invented in 1957 by Frank Rosenblatt at the Cornell Aeronautical Laboratory.
It was originally designed not to be a program, but to be implemented in hardware for image recognition on
a 20 � 20 pixel image. Rosenblatt built a Mark I Perceptron Machine that ran the algorithm, complete with
electric motors to do weight updates.]

frankrosenblatt.jpg, perceptron.jpg [Frank Rosenblatt (from Cornell Chronicle) and his
Mark I Perceptron Machine. This is what it took to process a 20 � 20 image in 1957.]

16

Jonathan Richard Shewchuk

[Then he held a press conference where he predicted that perceptrons would be �the embryo of an electronic
computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of
its existence.� We�re still waiting on that.]

[Perceptron Convergence Theorem: If data is linearly separable, perceptron algorithm will find a linear
classifier that classifies all data correctly in at most O(r2/?2) iterations, where r = max ?Xi? is �radius of
data� and ? is the �maximum margin.�]
[I�ll define �maximum margin� shortly.]

[We�re not going to prove this, because perceptrons are obsolete.]
[Although the step size/learning rate ? doesn�t appear in that big-O expression, it does have an effect on the
running time, but the effect is hard to characterize. The algorithm gets slower if ? is too small because it has
to take lots of steps to get down the hill. But it also gets slower if ? is too big for a different reason: it jumps
right over the region with zero risk and oscillates back and forth for a long time.]

[Although stochastic gradient descent is faster for this problem than gradient descent, the perceptron algo-
rithm is still slow. There�s no reliable way to choose a good step size ?. Fortunately, optimization algorithms
have improved a lot since 1957. You can get rid of the step size by using a decent modern �line search� al-
gorithm. Better yet, you can find a better decision boundary much more quickly by quadratic programming,
which is what we�ll talk about next.]

MAXIMUM MARGIN CLASSIFIERS

The margin of a linear classifier is the distance from the decision boundary to the nearest training point.
What if we make the margin as wide as possible?

w � x + ? = ?1

w � x + ? = 1
w � x + ? = 0

[Draw this by hand. maxmargin.pdf ]

We enforce the constraints

yi (w � Xi + ?) ? 1

for i ? [1, n]

[Notice that the right-hand side is a 1, rather than a 0 as it was for the perceptron algorithm. It�s not obvious,
but this a better way to formulate the problem, partly because it makes it impossible for the weight vector w
to get set to zero.]

CXXXXXXCCCCCPerceptron Learning; Maximum Margin Classifiers

17

Recall: if ?w? = 1, signed distance from hyperplane to Xi is w � Xi + ?.
Otherwise, it�s w

[We�ve normalized the expression to get a unit weight vector.]

Hence the margin is mini

? 1

?w? .

[We get the inequality by substituting the constraints.]

?w? � Xi + ?
?w? .
?w? |w � Xi + ?|
1
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
?1

To maximize the margin, minimize ?w?. Optimization problem:

Find w and ? that minimize ?w?2
subject to yi(Xi � w + ?) ? 1

for all i ? [1, n]

Called a quadratic program in d + 1 dimensions and n constraints.
It has one unique solution!

[If the points are linearly separable; otherwise, it has no solution.]

[A reason we use ?w?2 as an objective function, instead of ?w?, is that the length function ?w? is not smooth
at w = 0, whereas ?w?2 is smooth everywhere. This makes optimization easier.]

The solution gives us a maximum margin classifier, aka a hard-margin support vector machine (SVM).

[Technically, this isn�t really a support vector machine yet; it doesn�t fully deserve that name until we add
features and kernels.]
At the optimal solution, the margin is exactly 1
There is a slab of width 2

?w? . [Because at least one constraint holds with equality.]
?w? containing no sample points [with the hyperplane running along its middle].

[Let�s see what these constraints look like in weight space.]

weight3d.pdf, weightcross.pdf [This is an example of what the linear constraints look like
in the 3D weight space (w1, w2, ?) for the SVM we�ve been studying with three training
points. The SVM is looking for the point nearest the ?-axis that lies above the blue plane
(representing an in-class training point) but below the red and pink planes (representing
out-of-class training points). In this example, that optimal point lies where the three planes
intersect. At right we see a 2D cross section w1 = 1/17 of the 3D space, because the
optimal solution lies in this cross section. The constraints say that the solution must lie
in the leftmost pizza slice, while being as close to the origin as possible, so the optimal
solution is where the three lines meet.]

[Like the perceptron algorithm, a hard-margin SVM works only with linearly separable point sets. We�ll fix
that in the next lecture.]

-1.0-0.8-0.6-0.4-0.2w2-1.0-0.50.51.0alpha18

Jonathan Richard Shewchuk

4 Soft-Margin Support Vector Machines; Features

SOFT-MARGIN SUPPORT VECTOR MACHINES (SVMs)

Solves 2 problems:

� Hard-margin SVMs fail if data not linearly separable.
�

sensitive to outliers.

�

�

�

sensitive.pdf (ISL, Figure 9.5) [Example where one outlier moves the hard-margin SVM
decision boundary a lot.]

Idea: Allow some points to violate the margin, with slack variables.

Modified constraint for point i:
yi(Xi � w + ?) ? 1 ? ?i

[Observe that the only difference between these constraints and the hard-margin constraints we saw last
lecture is the extra slack term ?i.]
[We also impose new constraints, that the slack variables are never negative.]

?i ? 0

[This inequality ensures that all sample points that don�t violate the margin are treated the same; they all
have ?i = 0. Point i has nonzero ?i if and only if it violates the margin.]

w � x + ? = 0

?5/?w?

?1/?w?

?4/?w?

?3/?w?

?2/?w?

1/?w?

1/?w? (margin)
slacker+.pdf [A margin where some points have slack.]

Re-define �margin� to be 1/?w?. [For soft-margin SVMs, the margin is no longer the distance from the
decision boundary to the nearest training point; instead, it�s 1/?w?.]

9.2SupportVectorClassi?ers345?10123?10123?10123?10123X1X1X2X2FIGURE9.5.Left:Twoclassesofobservationsareshowninblueandinpurple,alongwiththemaximalmarginhyperplane.Right:Anadditionalblueobservationhasbeenadded,leadingtoadramaticshiftinthemaximalmarginhyperplaneshownasasolidline.Thedashedlineindicatesthemaximalmarginhyperplanethatwasobtainedintheabsenceofthisadditionalpoint.�Greaterrobustnesstoindividualobservations,and�Betterclassi?cationofmostofthetrainingobservations.Thatis,itcouldbeworthwhiletomisclassifyafewtrainingobservationsinordertodoabetterjobinclassifyingtheremainingobservations.Thesupportvectorclassi?er,sometimescalledasoftmarginclassi?er,supportvectorclassi?ersoftmarginclassi?erdoesexactlythis.Ratherthanseekingthelargestpossiblemarginsothateveryobservationisnotonlyonthecorrectsideofthehyperplanebutalsoonthecorrectsideofthemargin,weinsteadallowsomeobservationstobeontheincorrectsideofthemargin,oreventheincorrectsideofthehyperplane.(Themarginissoftbecauseitcanbeviolatedbysomeofthetrainingobservations.)Anexampleisshownintheleft-handpanelofFigure9.6.Mostoftheobservationsareonthecorrectsideofthemargin.However,asmallsubsetoftheobservationsareonthewrongsideofthemargin.Anobservationcanbenotonlyonthewrongsideofthemargin,butalsoonthewrongsideofthehyperplane.Infact,whenthereisnoseparatinghyperplane,suchasituationisinevitable.Observationsonthewrongsideofthehyperplanecorrespondtotrainingobservationsthataremisclassi?edbythesupportvectorclassi?er.Theright-handpanelofFigure9.6illustratessuchascenario.9.2.2DetailsoftheSupportVectorClassi?erThesupportvectorclassi?erclassi?esatestobservationdependingonwhichsideofahyperplaneitlies.ThehyperplaneischosentocorrectlyCCCCXCCXXXCCCCCCXXXXXXXXXXXXCXSoft-Margin Support Vector Machines; Features

19

To prevent abuse of slack, we add a loss term to objective fn.

Optimization problem:

Find w, ?, and ?i that minimize ?w?2 + C
subject to

yi(Xi � w + ?) ? 1 ? ?i
?i ? 0

i=1 ?i
for all i ? [1, n]
for all i ? [1, n]

(cid:80)n

. . . a quadratic program in d + n + 1 dimensions and 2n constraints.
[It�s a quadratic program because its objective function is quadratic and its constraints are linear inequalities.]
C > 0 is a scalar regularization hyperparameter that trades off:
small C
maximize margin 1/?w?
underfitting
(misclassifies much
training data)
less sensitive

big C
keep most slack variables zero or small
overfitting
(awesome training, awful test)

desire
danger

outliers
boundary more �flat�

very sensitive
more sinuous

[The last row only applies to nonlinear decision boundaries, which we�ll discuss next. Obviously, a linear
decision boundary can�t be �sinuous��though it can overfit.]

Use validation to choose C.

svmC.pdf (ISL, Figure 9.7) [Examples of how the slab varies with C. Smallest C at upper
left; largest C at lower right.]

[One way to think about slack is to pretend that slack is money we can spend to buy permission for a sample
point to violate the margin. The further a point penetrates the margin, the bigger the fine you have to pay.
We want to make the margin as wide as possible, but we also want to spend as little money as possible. If
the regularization parameter C is small, it means we�re willing to spend lots of money on violations so we
can get a wider margin. If C is big, it means we�re cheap and we won�t pay much for violations, even though
we�ll suffer a narrower margin. If C is infinite, we�re back to a hard-margin SVM.]

3489.SupportVectorMachines?1012?3?2?10123?1012?3?2?10123?1012?3?2?10123?1012?3?2?10123X1X1X1X1X2X2X2X2FIGURE9.7.Asupportvectorclassi?erwas?tusingfourdi?erentvaluesofthetuningparameterCin(9.12)�(9.15).ThelargestvalueofCwasusedinthetopleftpanel,andsmallervalueswereusedinthetopright,bottomleft,andbottomrightpanels.WhenCislarge,thenthereisahightoleranceforobservationsbeingonthewrongsideofthemargin,andsothemarginwillbelarge.AsCdecreases,thetoleranceforobservationsbeingonthewrongsideofthemargindecreases,andthemarginnarrows.butpotentiallyhighbias.Incontrast,ifCissmall,thentherewillbefewersupportvectorsandhencetheresultingclassi?erwillhavelowbiasbuthighvariance.ThebottomrightpanelinFigure9.7illustratesthissetting,withonlyeightsupportvectors.Thefactthatthesupportvectorclassi?er�sdecisionruleisbasedonlyonapotentiallysmallsubsetofthetrainingobservations(thesupportvec-tors)meansthatitisquiterobusttothebehaviorofobservationsthatarefarawayfromthehyperplane.Thispropertyisdistinctfromsomeoftheotherclassi?cationmethodsthatwehaveseeninprecedingchapters,suchaslineardiscriminantanalysis.RecallthattheLDAclassi?cationrule20

Jonathan Richard Shewchuk

FEATURES

Q: How to do nonlinear decision boundaries?

A: Make nonlinear features (aka basis functions) that lift points into a higher-dimensional space.
High-d linear classifier ? low-d nonlinear classifier.

[Added features work with all classifiers�not only linear classifiers like perceptrons and SVMs, but also
classifiers that are not linear.]

Example 1: The parabolic lifting map

? : Rd ? Rd+1
(cid:34)

?(x) =

x
?x?2

(cid:35)

? lifts x onto paraboloid xd+1 = ?x?2

[We�ve added one new feature, ?x?2. Even though the new feature is just a function of other input features,
it gives our linear classifier more power. Now an SVM can have spheres as decision boundaries.]
Find a linear classifier in ?-space.
It induces a sphere classifier in x-space.

x2

?x?2

x1

x1

Theorem: ?(X1), . . ., ?(Xn) are linearly separable iff X1, . . ., Xn are separable by a hypersphere.
(Possibly an ?-radius hypersphere = hyperplane.)
Proof: Consider hypersphere in Rd w/center c & radius ?. x is inside iff

[Draw this by hand. circledec.pdf ]

?x ? c?2 < ?2
?x?2 ? 2c � x + ?c?2 < ?2

[?2c? 1]
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
normal vector in Rd+1

(cid:35)

(cid:34)

x
?x?2
(cid:124)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:125)
?(x)

< ?2 ? ?c?2

Hence points inside sphere ? lifted points underneath hyperplane in ?-space.
[The implication works in both directions.]

[Hyperspheres include hyperplanes as a special, degenerate case. A hyperplane is essentially a hypersphere
with infinite radius. So hypersphere decision boundaries can do everything hyperplane decision boundaries
can do, plus a lot more. With the parabolic lifting map, if you pick a hyperplane in ?-space that is vertical,
you get a hyperplane in x-space.]

XXXXXXXCCCCCCXXXXXXXXXCCCCCCXXXXXSoft-Margin Support Vector Machines; Features

21

Example 2: Ellipsoid/hyperboloid/paraboloid decision boundaries

[Draw 2D examples of ellipse & hyperbola.]

In 3D, these have the formula

Ax2
1

+ Bx2
2

+ Cx2
3

+ Dx1x2 + Ex2x3 + F x3x1 + Gx1 + Hx2 + I x3 + ? = 0

[Here, the capital letters are scalars, not matrices.]

quadrics.png (courtesy Rahul Narain) [Quadrics in 3D.]

[If we add all the quadratic monomials as features, our decision boundaries can be arbitrary ellipsoids,
hyperboloids, paraboloids, and more.]

?(x) = [x2
1

x2
2

x2
3

x1x2

x2x3

x3x1

x1

x2

x3]?

[For perceptron or regression, add a
1 at end. For SVM, the 1 is built-in.]

Decision function is [A B C D E F G H I]
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
w?

�?(x) + ?

[Now, our decision function can be any degree-2 polynomial. Each component of ? is also called a
basis function, as it is a function of x and the decision function is a linear combination of basis functions.]

Isosurface defined by this equation is called a quadric.
A linear decision boundary in ?-space imposes a quadric decision boundary in x-space.
[The word quadric just means an isosurface of a degree-2 polynomial. In the special case of two dimen-
sions, it�s also known as a conic section. Our decision boundary can be an arbitrary ellipsoid, hyperboloid,
paraboloid, cylinder, etc.]
[When d is large, there are order-d2 cross-terms in ?-space! So we are adding a lot of new features. This
will impose a serious computational cost on a classifier like a support vector machine. But it might be worth
it to find good classifiers for data that aren�t linearly separable.]

?(x) : Rd ? R(d2+3d)/2

[For perceptron or regression, add 1 for the fictitious dimension.]

[If all these extra features make the classifier overfit or make it too slow, you can leave out the cross-terms
and include only quadratic terms like x2
2, etc. Then the number of added features is linear in d, not
quadratic in d. If you do that, your decision boundaries can be axis-aligned ellipsoids and axis-aligned
hyperboloids, but they can�t be rotated in arbitrary ways.]

1, x2

22

Jonathan Richard Shewchuk

Example 3: Decision fn is degree-p polynomial

E.g., a cubic in R2:

?(x) = [x3
x2
1x2
1
?(x) : Rd ? RO(d p)

x1x2
2

x3
2

x2
1

x1x2

x2
2

x1

x2]?

degree5.pdf [Hard-margin SVMs with degree 1/2/5 decision functions. Observe that the
margin tends to get wider as the degree increases.]

[Increasing the degree like this accomplishes two things.

� First, the data might become linearly separable when you lift them to a high enough degree, even if

the original data are not linearly separable.

� Second, raising the degree can widen the margin, so you might get a more robust decision boundary

that generalizes better to test data.]

degree 1 ?

? degree 10

d1.pdf, d2.pdf, . . . , d10.pdf [Decision boundaries found by linear regression with polyno-
mial features of maximum degrees from 1 through 10. Circles are training points; X�s are
validation points. (Implementation courtesy Josh Levine.)]

Figure6:Thee?ectofthedegreeofapolynomialkernel.Thepolynomialkernelofdegree1leadstoalinearseparation(A).Higherdegreepolynomialkernelsallowamore?exibledecisionboundary(B-C).ThestylefollowsthatofFigure5.features.Thedimensionalityofthefeature-spaceassociatedwiththeaboveexampleisquadraticinthenumberofdimensionsoftheinputspace.Ifweweretousemonomialsofdegreedratherthandegree2monomialsasabove,thedimensionalitywouldbeexponentialind,resultinginasubstantialincreaseinmemoryusageandthetimerequiredtocomputethediscriminantfunction.Ifourdataarehigh-dimensionaltobeginwith,suchasinthecaseofgeneexpressiondata,thisisnotacceptable.Kernelmethodsavoidthiscomplexitybyavoidingthestepofexplicitlymappingthedatatoahighdimensionalfeature-space.Wehaveseenabove(Equation(5))thattheweightvectorofalargemarginseparatinghyperplanecanbeexpressedasalinearcombinationofthetrainingpoints,i.e.w=Pni=1yi?ixi.Thesameholdstrueforalargeclassoflinearalgorithms,asshownbytherepresentertheorem(see[2]).Ourdiscriminantfunctionthenbecomesf(x)=nXi=1yi?ih (xi), (x)i+b.(7)Therepresentationintermsofthevariables?iisknownasthedualrepre-sentation(cf.Section�Classi?cationwithLargeMargin�).Weobservethatthedualrepresentationofthediscriminantfunctiondependsonthedataonlythroughdotproductsinfeature-space.Thesameobservationholdsforthedualoptimizationproblem(Equation(4))whenreplacexiwith (xi)(analogouslyforxj).Ifthekernelfunctionk(x,x0)de?nedask(x,x0)=? (x), (x0)?(8)10Soft-Margin Support Vector Machines; Features

23

12

9

d
e
fi
i
s
s
a
l
c
s
i

m

6

5

3

3

2 2

1 1

1

2 3 4 5 6 7 8 9

10

degree

misclass.pdf
with polynomial features of degrees 1 through 10.]

[Misclassified validation points for the linear regressions depicted above,

[Here we see a U-shaped curve, as we do with the regularization parameter C in soft-margin SVMs. This
example obtains best results with degree 2 or 3 polynomials. A linear classifier underfits, whereas classifiers
of degree 4 or greater overfit; generalization gets worse as the decision boundary becomes too flexible. The
degree is an example of a hyperparameter that can be optimized by validation.]

[If you use polynomial features with a soft-margin SVM, now you have two hyperparameters: the degree
and the regularization hyperparameter C. Generally, the optimal C will be different for every polynomial
degree, so when you change the degree, you should run validation again to find the best C for that degree.]

[With polynomials, we�re really blowing up the number of features!
If you have, say, 100 features per
sample point and you want to use degree-4 decision functions, then each lifted feature vector has a length of
roughly 4 million, and your learning algorithm will take approximately forever to run.]

[However, there is an extremely clever trick that allows us to work with these huge feature vectors very
quickly, without ever computing them. It�s called �kernelization� or �the kernel trick.� So even though it
appears now that working with degree-4 polynomials is computationally infeasible, it can actually be done
quickly.]

24

Jonathan Richard Shewchuk

[So far I�ve talked only about polynomial features. But features can get much more complicated than
polynomials, and they can be tailored to fit a specific problem. Let�s consider a type of feature you might
use if you wanted to implement a handwriting recognition algorithm.]

Example 5: Edge detection

Edge detector: algorithm for approximating grayscale/color gradients in image, e.g.,

� tap filter
� Sobel filter
� oriented Gaussian derivative filter

[images are discrete, not continuous fields, so approximation of gradients is necessary.]

[See �Image Derivative� on Wikipedia.]

Collect line orientations in local histograms (each having 12 orientation bins per region); use histograms as
features (instead of raw pixels).

Paper: Maji & Malik, 2009.

orientgrad.png [Image histograms.]

[If you want to, optionally, use these features in future homeworks and try to win the Kaggle competition,
this paper is a good online resource.]

[When they use a linear SVM on the raw pixels, Maji & Malik get an error rate of 15.38% on the test set.
When they use a linear SVM on the histogram features, the error rate goes down to 2.64%.]

[Many applications can be improved by designing application-specific features. There�s no limit but your
own creativity and ability to discern the structure hidden in your application.]

Machine Learning Abstractions and Numerical Optimization

25

5 Machine Learning Abstractions and Numerical Optimization

ML ABSTRACTIONS [some meta comments on machine learning]

[When you write a large computer program, you break it down into subroutines and modules. Many of you
know from experience that you need to have the discipline to impose strong abstraction barriers between
different modules, or your program will become so complex you can no longer manage nor maintain it.]

[When you learn a new subject, it helps to have mental abstraction barriers, too, so you know when you can
replace one approach with a different approach. I want to give you four levels of abstraction that can help
you think about machine learning. It�s important to make mental distinctions between these four things, and
the code you write should have modules that reflect these distinctions as well.]
APPLICATION/DATA

data labeled or not?
yes: labels categorical (classification) or quantitative (regression)?
no: similarity (clustering) or positioning (dimensionality reduction)?

MODEL

[what kinds of hypotheses are permitted?]

e.g.:
� decision fns: linear, polynomial, logistic, neural net, . . .
� nearest neighbors, decision trees
� features
� low vs. high capacity (affects overfitting, underfitting, inference)

OPTIMIZATION PROBLEM

� variables, objective fn, constraints
e.g., unconstrained, convex program, least squares, PCA

OPTIMIZATION ALGORITHM

e.g., gradient descent, simplex, SVD

[In this course, we focus primarily on the middle two levels. As a data scientist, you might be given an
application, and your challenge is to turn it into an optimization problem that we know how to solve. We
will talk about optimization algorithms, but usually data analysts use optimization codes that are faster and
more robust than what they would write themselves.]
[The second level, the model, has a huge effect on the success of your learning algorithm. Sometimes you
get a big improvement by tailoring the model or its features to fit the structure of your specific data. The
model also has a big effect on whether you overfit or underfit. And if you want a model that you can interpret
so you can do inference, the model has to have a simple structure. Lastly, you have to pick a model that
leads to an optimization problem that can be solved. Some optimization problems are just too hard.]

[It�s important to understand that when you change something in one level of this diagram, you probably
have to change all the levels underneath it. If you switch your model from a linear classifier to a neural net,
your optimization problem changes, and your optimization algorithm changes too.]

26

Jonathan Richard Shewchuk

[Not all machine learning methods fit this four-level decomposition. Nevertheless, for everything you learn
in this class, think about where it fits in this hierarchy. If you don�t distinguish which math is part of the
model and which math is part of the optimization algorithm, this course will be very confusing for you.]

OPTIMIZATION PROBLEMS

[I want to familiarize you with some types of optimization problems that can be solved reliably and effi-
ciently, and the names of some of the optimization algorithms used to solve them. An important skill for
you to develop is to be able to go from an application to a well-defined optimization problem. That skill
depends on your ability to recognize well-studied types of optimization problems.]

Unconstrained

Goal: Find w that minimizes (or maximizes) a continuous objective fn f (w).

f is smooth if its gradient is continuous too.

A global minimum of f is a value w such that f (w) ? f (v) for every v.
�
A local minimum � � � �

�
for every v in a tiny ball centered at w.
[In other words, you cannot walk downhill from w.]

�

�

�

�

global minimum

local minima

[Draw this by hand. minima.pdf ]

Usually, finding a local minimum is easy;

finding the global minimum is hard. [or impossible]

Exception: A function is convex if for every x, y ? Rd,
the line segment connecting (x, f (x)) to (y, f (y)) does not go below f (�).

x

y

[Draw this by hand. convex.pdf ]

Formally: for every x, y ? Rd and ? ? [0, 1], f (x + ?(y ? x)) ? f (x) + ?( f (y) ? f (x)).
E.g., perceptron risk fn is convex and nonsmooth.

Machine Learning Abstractions and Numerical Optimization

27

[When you sum together convex functions, you always get a convex function. The perceptron risk function
is a sum of convex loss functions, so it is convex.]

A [continuous] convex function [on a closed, convex domain] has either

� no minimum (goes to ??), or
� just one local minimum, or
� a connected set of local minima that are all global minima with equal f .

[The perceptron risk function is in the last category.]
[In the last two cases, if you walk downhill, you eventually reach a global minimum.]

Gradient descent: repeat w ? w ? ? ? f (w)

[Gradient descent with different learning
learningrates20.gif (Gajanan Bhat, gbhat.com)
rates ?. Top left: painfully small. Top right: reasonable, but still smaller than ideal. Bottom
left: reasonable, but larger than ideal. Bottom right: too large; diverges. This is an animated
GIF; see https://gbhat.com/machine learning/gradient descent learning rates.html .]

� Fails/diverges if ? too large.
� Slow if ? too small.
� ? often optimized by trial & error [for slow learners like neural networks].

[The best value of ? is hard to guess. One common technique for dealing with divergence is to check whether
a step of gradient descent increases the function value rather than decreasing it; if so, reduce the step size.]

[That�s a simple example of what�s called an adaptive learning rate or a learning rate schedule. These
adaptations become even more important when you do stochastic gradient descent or when you optimize
non-convex, very twisty objective functions. We�ll revisit the idea when we learn neural networks.]

28

Jonathan Richard Shewchuk

[One interesting aspect of gradient descent that these figures illustrate is that it usually never reaches the
exact local minimum. Instead, it gets closer and closer forever, but never exactly reaches the true minimum.
We call this behavior �convergence.� The last question of Homework 2 will give you some understanding
of why convergence happens under the right conditions.]

[When we have a feature space with more than one dimension, another problem arises, which is that the
learning rate that�s good for one direction might be terrible in another direction. Consider the three examples
of gradient descent below.]

goodcondition.pdf, illcondition105.pdf, illcondition055.pdf [Left: 20 iterations of gradi-
ent descent on a well-conditioned quadratic function, f (w) = 2w2
2, with a modest step
1
size ? = 0.105. Center: 20 iterations on an ill-conditioned function, f (w) = 10w2
2; the
1
same step size is now too large. Right: after reducing the step size to ? = 0.055, we have
convergence again but we aren�t approaching the minimum nearly as quickly.]

+ w2

+ w2

[The step size that works for the left example is too large for the center example; it diverges in the w1-
direction. At right, we reduce the step size and obtain convergence. But now convergence is slow in the
w2-direction.]

High ellipticity of the contours, a.k.a. ill-conditioning of the Hessian, means no learning rate is good in all
directions.

[The Hessian matrix is said to be ill-conditioned if its largest eigenvalue is much larger than its small-
est eigenvalue. Ill-conditioning can be a problem even for simple methods like linear regression, making
In response to these observations, there are adaptive learning rate algo-
it harder to solve the problem.
rithms that explicitly choose different learning rates for different weights. Famous examples are Adam and
RMSprop.]

[There are many applications where you don�t have a convex objective function. Then gradient descent
usually can find a local minimum, but not necessarily a global minimum. And often there is no guarantee
that the local minimum you find will be nearly as good as the global minimum. Nevertheless, gradient
descent is used for a lot of nonconvex machine learning problems too. For example, neural networks try
to optimize an objective function that has lots of local minima. But stochastic gradient descent is still the
algorithm of choice for training neural nets. We�ll talk more later in the semester about why.]

-4-224w1-2246w2-4-224w1-2246w2-4-224w1-2246w2Machine Learning Abstractions and Numerical Optimization

29

Linear Program

Linear objective fn + linear inequality constraints.
Goal: Find w that maximizes (or minimizes) c � w

subject to Aw ? b

where A is n � d matrix, b ? Rn, expressing n linear constraints:
Ai � w ? bi,

i ? [1, n]

c

in w-space:

optimum

feasible
region

active constraint

active constraint

[Draw this by hand.

linprog.pdf ]

The set of points w that satisfy all constraints is a convex polytope called the feasible region F [shaded].
The optimum is the point in F that is furthest in the direction c.
A point set P is convex if for every p, q ? P, the line segment with endpoints p, q lies entirely in P.

[What does convex mean?]

[What is a polytope? Just a polyhedron, generalized to higher dimensions.]

The optimum achieves equality for some constraints (but not most), called the active constraints of the
optimum. [In the figure above, there are two active constraints. In an SVM, active constraints correspond to
the training points that touch or violate the slab, and these points are also known as support vectors.]

[Sometimes, there is more than one optimal point. For example, in the figure above, if c pointed straight up,
every point on the top horizontal edge would be optimal. The set of optimal points is always convex.]

Example: EVERY feasible point (w, ?) gives a linear classifier:

Find w, ? that satisfies yi(w � Xi + ?) ? 1

for all i ? [1, n]

[This is the problem of finding a feasible point. This problem can be cast as a slightly different linear
program that uses an objective function to make all the inequalities be satisfied strictly if that�s possible.]
IMPORTANT: The data are linearly separable iff the feasible region is not the empty set.
? Also true for maximum margin classifier (quadratic program)

[The most famous algorithm for linear programming is the simplex algorithm, invented by George Dantzig
in 1947. The simplex algorithm is indisputably one of the most important and useful algorithms of the
20th century. It walks along edges of the feasible region, traveling from vertex to vertex until it finds an
optimum.]
[Linear programming is very different from unconstrained optimization; it has a much more combinatorial
flavor. If you knew which constraints would be the active constraints once you found the solution, it would
be easy; the hard part is figuring out which constraints should be the active ones. There are exponentially
many possibilities, so you can�t afford to try them all. So linear programming algorithms tend to have a
very discrete, computer science feeling to them, like graph algorithms, whereas unconstrained optimization
algorithms tend to have a continuous, numerical mathematics feeling.]

30

Jonathan Richard Shewchuk

[Linear programs crop up everywhere in engineering and science, but they�re usually in disguise. An ex-
tremely useful talent you should develop is to recognize when a problem is a linear program.]

[A linear program solver can find a linear classifier, but it can�t find the maximum margin classifier. We
need something more powerful.]

Quadratic Program

Quadratic, convex objective fn + linear inequality constraints.
Goal: Find w that minimizes f (w) = w?Qw + c?w

subject to Aw ? b

where Q is a symmetric, positive semidefinite matrix.

[A matrix is positive semidefinite if w?Qw ? 0 for all w.]
Example: Find maximum margin classifier.

+ w2

quadratic.pdf, quadratic3D.pdf [Left: A hard-margin SVM minimizes the objective func-
tion w2
2. Right: There is also an ?-axis, so the isosurfaces of the objective function
1
are really cylinders. On the left isocontours, draw two polygons�one with one active
constraint, and one with two�and show the constrained minimum for each polygon. �In a
hard-margin SVM, we are looking for the point in this polygon that�s closest to the ?-axis.�]
[If Q is positive definite, a quadratic program has just one unique local minimum, which is therefore the
global minimum. But in a support vector machine, Q is not definite; it is only positive semidefinite, because
the bias term ? is a weight but it does not influence the objective function. Sometimes positive semidefinite
quadratic programs have multiple solutions, but SVMs are a special case where there is only one unique
minimum. By the way, if Q is indefinite, then f is not convex, the minimum is not always unique, and
quadratic programming is NP-hard. But we won�t need that kind of quadratic program in this class.]

Algs for quadratic programming:

� Simplex-like [commonly used for general-purpose quadratic programs, but not as good for SVMs as

the following two algorithms that specifically exploit properties of SVMs]
� Sequential minimal optimization (SMO, used in LIBSVM, �SVC� in scikit)
� Coordinate descent (used in LIBLINEAR, �LinearSVC� in scikit)

Numerical optimization @ Berkeley: EECS 127/227AT/227BT/227C.

1234567891010101011111111121212121313131314141414151515151616161617171717-3-2-10123-3-2-10123Decision Theory; Generative and Discriminative Models

31

6 Decision Theory; Generative and Discriminative Models

DECISION THEORY aka Risk Minimization

[Today I�m going to talk about a style of classifier very different from SVMs. The classifiers we�ll cover in
the next few weeks are based on probability.]

[One aspect of probabilistic data is that sometimes a point in feature space doesn�t have just one class.
Suppose your data is adult men and women with just one feature: their height. You want to train a classifier
that takes in an adult�s height and returns a classification, man or woman. Suppose you are asked to predict
the sex of a 5�5� adult. Well, your training set includes some 5�5� women and some 5�5� men. What should
you do?]
[In your feature space, you have two training points at the same location with different classes. More
generally, the height distributions of men and women overlap. Obviously, in that case, you can�t draw a
decision boundary that classifies all points with 100% accuracy.]
Multiple sample points with different classes could lie at same point: we want a probabilistic classifier.

Suppose 10% of population has cancer, 90% doesn�t.
Probability distributions for occupation conditioned on cancer, P(X|Y):

job
cancer
no cancer

(X)
(Y = 1)
(Y = ?1)

miner
20%
1%

farmer

other
50% 30%
10% 89%

[caps here mean random variables, not matrices.]

[I made these numbers up. Please don�t take them as medical advice.]
Recall: P(X) = P(X|Y = 1) P(Y = 1) + P(X|Y = ?1) P(Y = ?1)
P(X = farmer) = 0.5 � 0.1 + 0.1 � 0.9 = 0.14

[. . . so 14% of random people are farmers]

You meet a farmer. Guess whether he has cancer?

[If you�re in a hurry, you might see that 50% of people with cancer are farmers, but only 10% of people with
no cancer are farmers, and conclude that a typical farmer probably has cancer. But that would be wrong,
because that reasoning fails to take the prior probabilities into account.]

Bayes� Theorem:

? prior prob.

? posterior probability
P(Y = 1|X) = P(X|Y = 1)P(Y = 1)
P(Y = ?1|X) = P(X|Y = ?1)P(Y = ?1)

P(X)

P(X)

? if X = farmer

= 0.05
0.14
= 0.09
0.14

[These two probs always sum to 1.]

P(cancer | farmer) = 5/14 ? 36%.

[So we probably shouldn�t diagnose cancer.]

[BUT . . . we�re assuming that we want to maximize the chance of a correct prediction. But that�s not always
the right assumption. If you�re developing a cheap screening test for cancer, you�d rather have more false
positives and fewer false negatives. A false negative might mean somebody misses an early diagnosis and
dies of a cancer that could have been treated if caught early. A false positive just means that you spend more
money on more accurate tests. When there�s an asymmetry between the awfulness of false positives and
false negatives, we can quantify that with a loss function.]

32

Jonathan Richard Shewchuk

A loss function L(�y, y) specifies badness if classifier predicts �y, true class is y.

E.g., L(�y, y) =

?
????
????

1 if �y = 1, y = ?1,
5 if �y = ?1, y = 1,
0 if �y = y.

false positive is bad
false negative is BAAAAAD
[loss should always be zero for a perfectly correct prediction!]

A 36% probability of loss 5 is worse than a 64% prob. of loss 1,
so we recommend further cancer screening.

The loss fn above is asymmetrical.
[A loss is symmetrical if it is the same for false positives and false negatives. For example . . . ]

The 0-1 loss function is L(�y, y) =

(cid:40)

1 if �y (cid:44) y,
0 if �y = y.

[always 1 for a wrong prediction]
[always 0 for a correct prediction]

[Another application where you want a very asymmetrical loss function, besides medical diagnosis, is spam
detection. Putting a good email in the spam folder is much worse than putting spam in your inbox.]
Let r : Rd ? �1 be a decision rule, aka classifier:
a fn that maps a feature vector x to 1 (�in class�) or ?1 (�not in class�).
The risk for r is the expected loss over all values of x, y.

[Memorize this definition!]

R(r) = E[L(r(X), Y)]
(cid:88)

=

x
= P(Y = 1)

?
?????L(r(x), 1) P(Y = 1|X = x) + L(r(x), ?1) P(Y = ?1|X = x)

?
????? P(X = x)

(cid:88)

x

L(r(x), 1) P(X = x|Y = 1) + P(Y = ?1)

(cid:88)

x

L(r(x), ?1) P(X = x|Y = ?1).

The Bayes decision rule aka Bayes classifier is the fn r? that minimizes functional R(r).
Assuming L(1, 1) = L(?1, ?1) = 0,

(cid:40)

r?(x) =

1 if L(?1, 1) P(Y = 1|X = x) > L(1, ?1) P(Y = ?1|X = x),
?1 otherwise.

When L is symmetrical, [the big, key principle you should memorize is]
pick the class with the biggest posterior probability.
[But if the loss function is asymmetrical, then you must weight the posteriors with the losses.]
In cancer example, r?(miner) = 1, r?(farmer) = 1, and r?(other) = ?1.

The Bayes risk, aka optimal risk, is the risk of the Bayes classifier.
[In our cancer example, the last expression for risk R gives:]

R(r?) = 0.1(5 � 0.3) + 0.9(1 � 0.01 + 1 � 0.1) = 0.249.

No decision rule gives a lower risk.

[It is interesting that, if we really know all these probabilities, we really can construct an ideal probabilistic
classifier. But in real applications, we rarely know these probabilities; the best we can do is use statistical
methods to estimate them.]
Deriving/using r? is called risk minimization.

[Did you memorize the two boldfaced lines above yet?]

Decision Theory; Generative and Discriminative Models

33

Continuous Distributions

Suppose X has a continuous probability density fn (PDF).

Review: [Go back to your CS 70 or stats notes if you don�t remember this.]

f (x)

[Draw this by hand.

integrate.pdf ]

x1

x2

area under whole curve = 1 =

x

(cid:90) ?

??

prob. that random variable X ? [x1, x2] =

(cid:90) x2

x1

f (x) dx

[shaded area]

f (x) dx

expected value of g(X) : E[g(X)] =

(cid:90) ?

??

g(x) f (x) dx

mean � = E[X] =

(cid:90) ?

??

x f (x) dx

[Perhaps our cancer statistics look like this.]

variance ?2 = E[(X ? �)2] = E[X2] ? �2

fX|Y=1(x)

fX|Y=?1(x)

x

Draw this figure by hand (cancerconditional.png) [The area under each curve is 1.]

[Let�s use the 0-1 loss function. In other words, suppose you want a classifier that maximizes the chance of
a correct prediction. The wrong answer would be to look where these two curves cross and make that be the
decision boundary. As before, it�s wrong because it doesn�t take into account the prior probabilities.]
Suppose P(Y = 1) = 1/3, P(Y = ?1) = 2/3, 0-1 loss.
fX|Y=1(x)P(Y = 1)

fX|Y=?1(x)P(Y = ?1)

Bayes optimal decision boundary

x

Draw this figure by hand (cancerposterior.png)

[To maximize the chance you�ll predict correctly whether somebody has cancer, the Bayes decision rule
looks up x on this chart and picks the curve with the highest probability. In this example, that means you
pick cancer when x is left of the optimal decision boundary, and no cancer when x is to the right.]

34

Jonathan Richard Shewchuk

Define risk as before, replacing summations with integrals.

R(r) = E[L(r(X), Y)]
= P(Y = 1)

(cid:90)

L(r(x), 1) fX|Y=1(x) dx +

(cid:90)

P(Y = ?1)

L(r(x), ?1) fX|Y=?1(x) dx.

For Bayes decision rule, Bayes risk is the area under minimum of functions above. [Shade it.]
Assuming L(1, 1) = L(?1, ?1) = 0,

(cid:90)

R(r?) =

min
y=�1

L(?y, y) fX|Y=y(x) P(Y = y) dx.

[If you want to use an asymmetrical loss function, just scale the curves vertically in the figure above.]

If L is 0-1 loss,

R(r) = P(r(x) is wrong)
and the Bayes optimal decision boundary is {x : P(Y = 1|X = x)
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
decision fn

=

}

0.5
(cid:124)(cid:123)(cid:122)(cid:125)
isovalue

[then the risk has a particularly nice interpretation:]
[which makes sense, because R is the expected loss.]

fX|Y=1(x)P(Y = 1)

fX|Y=?1(x)P(Y = ?1)

qda3d.pdf, qdacontour.pdf [Two different views of the same 2D Gaussians.]

Bayes optimal decision boundary

[Notice that the accuracy of the probabilities is most important near the decision boundary. Far away from
the decision boundary, a bit of error in the probabilities probably wouldn�t change the classification.]

[You can also have multi-class classifiers, choosing among three or more classes. The Bayesian approach is
a particularly convenient way to generate multi-class classifiers, because you can simply choose whichever
class has the greatest posterior probability. Then the decision boundary lies wherever two or more classes
are tied for the highest probability.]

Decision Theory; Generative and Discriminative Models

35

3 WAYS TO BUILD CLASSIFIERS

(1) Generative models (e.g., LDA)

[We�ll learn about LDA next lecture.]

� Assume sample points come from probability distributions, different for each class.
� Guess form of distributions
� For each class C, fit distribution parameters to class C points, giving fX|Y=C(x)
� For each C, estimate P(Y = C)
� Bayes� Theorem gives P(Y|X)
� If 0-1 loss, pick class C that maximizes P(Y = C|X = x)

equivalently, maximizes fX|Y=C(x) P(Y = C)

[posterior probability]

(2) Discriminative models (e.g., logistic regression)

[We�ll learn about logistic regression in a few weeks.]

� Model P(Y|X) directly

(3) Find decision boundary (e.g., SVM)

� Model r(x) directly (no posterior)

Advantage of (1 & 2): P(Y|X) tells you probability your guess is wrong

[This is something SVMs don�t do.]

Advantages of (1): you can diagnose outliers: f (x) is very small;

stabler for outliers or few training points.

Disadvantages of (1): often hard to estimate distributions accurately;

real distributions rarely match standard ones.

[What I�ve written here doesn�t actually define the phrases �generative model� or �discriminative model.�
The proper definitions accord with the way statisticians think about models. A generative model is a full
probabilistic model of all variables, whereas a discriminative model provides a model only for the target
variables that we want to predict.]

[It�s important to remember that we rarely know precisely the value of any of these probabilities. There is
usually error in all of these probabilities. In practice, generative models are most popular when you have
phenomena that are well approximated by the normal distribution or another �nice� distribution. Generative
methods also tend to be more stable than other methods when the number of training points is small or when
there are a lot of outliers.]

36

Jonathan Richard Shewchuk

7 Gaussian Discriminant Analysis; Maximum Likelihood Estimation

GAUSSIAN DISCRIMINANT ANALYSIS

Fundamental assumption: each class has a normal distribution [a Gaussian].

X ? N(�, ?2) : f (x) =

[� & x = vectors; ? = scalar; d = dimension]

?

1
2??)d

(cid:32)
?

exp

(cid:33)

.

?x ? �?2
2?2

(
For each class C, suppose we know mean �C and variance ?2
and prior ?C = P(Y = C).

C, yielding PDF fX|Y=C(x),

fX|Y=C(x) ?C

fX|Y=D(x) ?D

QC(x)

QD(x)

Bayes optimal decision boundary

qda3d.pdf, qdacontour.pdf, Q.pdf
Bayes optimal decision boundary is an ellipse.]

[Probability density functions for two classes. The

[This PDF is a simplified version of the multivariate normal distribution. It is multivariate: x and � can be
vectors, and this plot shows a 2D feature space. But the variance ?2 is just a scalar; for simplicity, we will
avoid the covariance matrix until next lecture. That�s why the isocontours are circles, not ellipses. I call this
the isotropic normal distribution, because the variance is the same in every direction. Next lecture, we�ll
use the usual multivariate normal distribution, where the isosurfaces are ellipsoids.]
Given x, Bayes decision rule r?(x) predicts class C that maximizes fX|Y=C(x) ?C.
[Remember our last lecture�s main principle: pick the class with the biggest posterior probability!]
ln ? is monotonically increasing for ? > 0, so it is equivalent to maximize

QC(x) = ln

(cid:16)

(

?

2?)d fX|Y=C(x) ?C

(cid:17) = ?

?x ? �C?2
2?2
C

? d ln ?C + ln ?C.

[QC is quadratic in x]

[In a 2-class problem, you can also incorporate an asymmetrical loss function by adding ln L(not C, C)
In a multi-class problem, asymmetric loss may be more difficult to account for, because the
to QC(x).
penalty for guessing wrong might depend on both the wrong prediction and the true class.]

Quadratic Discriminant Analysis (QDA)

Suppose only 2 classes C, D. Then the Bayes classifier is
C if QC(x) ? QD(x) > 0,
D otherwise.

r?(x) =

(cid:40)

[Picks the class with the biggest posterior probability]

Decision fn is QC(x) ? QD(x) (quadratic); Bayes decision boundary is {x : QC(x) ? QD(x) = 0}.

Gaussian Discriminant Analysis; Maximum Likelihood Estimation

37

� In 1D, B.d.b. may have 1 or 2 points.
� In d-D, B.d.b. is a quadric.

[Solutions to a quadratic equation]
[In 2D, that�s a conic section; see figure above]

[You might not be satisfied with just predicting how each point is classified. One of the great things about
QDA is that you can also estimate the probability that your prediction is correct. Let�s work that out.]

To recover posterior probabilities in 2-class case, use Bayes.

P(Y = C|X) =

fX|Y=C ?C
fX|Y=C ?C + fX|Y=D ?D

recall eQC(x) = (

?

2?)d fX|Y=C(x) ?C [by definition of QC]

P(Y = C|X = x) =

eQC(x)
eQC(x) + eQD(x)
= s(QC(x) ? QD(x)),

=

1
1 + eQD(x)?QC(x)
where

s(?) =

1
1 + e??

? logistic fn aka sigmoid fn

[recall QC ? QD is the decision fn]

logistic.pdf
beside it:] s(0) = 1
monotonically increasing.

[The logistic function. Write
2 , s(?) ? 1, s(??) ? 0,

[We interpret s(0) = 1
2 as saying that on
the decision boundary, there�s a 50% chance
of class C and a 50% chance of class D.]

Multi-class QDA: [QDA works very naturally with more than 2 classes.]

multiplicative.pdf [Multi-class QDA partitions the feature space into regions. In two or
more dimensions, you typically wind up with multiple decision boundaries that adjoin each
other at joints. It looks like a sort of Voronoi diagram. In fact, it�s a special kind of Voronoi
diagram called a multiplicatively, additively weighted Voronoi diagram.]

-4-2024x0.20.40.60.81.0s(x)38

Jonathan Richard Shewchuk

Linear Discriminant Analysis (LDA)

[LDA is a variant of QDA with linear decision boundaries. It�s less likely to overfit than QDA.]
Fundamental assumption: all the Gaussians have same variance ?2.
[The equations simplify nicely in this case.]
QC(x) ? QD(x) = (�C ? �D) � x
?2
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
w�x

+ ln ?C ? ln ?D.
?
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
+?

?�C?2 ? ?�D?2
2?2

[The quadratic terms in QC and QD canceled each other out!]
Now it�s a linear classifier!

� decision boundary is w � x + ? = 0
� posterior is P(Y = C|X = x) = s(w � x + ?)

[The effect of �s(w � x + ?)� is to scale and translate the logistic fn in x-space.]

lda1d.pdf, lda2d.pdf [Two Gaussians (red) and the logistic function (black). The logistic
function is the right Gaussian divided by the sum of the Gaussians. Observe that even when
the Gaussians are 2D, the logistic function still looks 1D.]

Special case: if ?C = ?D = 1
2

This is the centroid method!

? (�C ? �D) � x ? (�C ? �D) �

(cid:19)

(cid:18) �C + �D
2

= 0.

Multi-class LDA: choose C that maximizes linear discriminant fn

�C � x
?2

?

?�C?2
2?2

+ ln ?C.

voronoi.pdf [When you have many classes, their LDA decision boundaries form a classical
Voronoi diagram if the priors ?C are equal. All the Gaussians have the same width.]

-3-2-1123x0.20.40.60.81.0P(x)Gaussian Discriminant Analysis; Maximum Likelihood Estimation

39

MAXIMUM LIKELIHOOD ESTIMATION OF PARAMETERS (Ronald Fisher, circa 1912)

[To use Gaussian discriminant analysis, we must first fit Gaussians to the sample points and estimate the
class prior probabilities. We�ll do priors first�they�re easier, because they involve a discrete distribution.
Then we�ll fit the Gaussians�they�re less intuitive, because they�re continuous distributions.]
Let�s flip biased coins! Heads with probability p; tails w/prob. 1 ? p. [But we don�t know p.]

10 flips, 8 heads, 2 tails. [Let me ask you a weird question.] What is the most likely value of p?

# of heads is X ? B(n, p), binomial distribution:

P(X = x) =

(cid:33)

(cid:32)
n
x

px (1 ? p)n?x

[this is the probability of getting exactly x heads in n coin flips]

Prob. of x = 8 heads in n = 10 flips is

P(X = 8) = 45p8 (1 ? p)2

def= L(p)

Written as a fn of distribution parameter p, this prob. is the likelihood fn L(p).

Maximum likelihood estimation (MLE): A method of estimating the parameters of a statistical model by
picking the params that maximize [the likelihood function] L.
. . . is one method of density estimation: estimating a PDF [probability density function] from data.

[Let�s phrase it as an optimization problem.]

Find p that maximizes L(p).

binomlikelihood.pdf [Graph of L(p) for this example.]

Solve by finding critical point of L:

dL
dp

= 360p7(1 ? p)2 ? 90p8(1 ? p) = 0

? 4(1 ? p) ? p = 0 ? p = 0.8

[It shouldn�t seem surprising that a coin that is biased so it comes up heads 80% of the time is the coin most
likely to produce 8 heads in 10 flips.]
[Note: d2L
dp2

(cid:17) ?18.9 < 0 at p = 0.8, confirming it�s a maximum.]

[Here�s how this applies to prior probabilities.]
Suppose our training set is n points, with x in class C. Then our estimated prior for class C is �?C = x/n.

0.20.40.60.81.00.050.100.150.200.250.3040

Jonathan Richard Shewchuk

Likelihood of a Gaussian

Given sample points X1, X2, . . . , Xn, find best-fit Gaussian.

[Now we want to fit a normal distribution to data, instead of a binomial distribution. If you draw a random
point from a normal distribution, what is the probability that it will be exactly at X1?]

[Zero. So it might seem like we have a problem here. With a continuous distribution, the probability of
generating any particular point is zero. But we�re just going to ignore that and do �likelihood� anyway.]

Likelihood of drawing these points [in the specified order] is

L(�, ?; X1, . . . , Xn) = f (X1) f (X2) � � � f (Xn).

[How do we maximize this?]

The log likelihood ?(�) is the ln of the likelihood L(�).
Maximizing likelihood ? maximizing log likelihood.

?(�, ?; X1, ..., Xn) = ln f (X1) + ln f (X2) + ... + ln f (Xn)
(cid:32)

?

?Xi ? �?2
2?2

(cid:33)
2? ? d ln ?
? d ln
?
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
ln of normal PDF

n(cid:88)

=

i=1

= 0

??
??
Xi ? �
?2

Set ?�? = 0,

?�? =

??
??

=

n(cid:88)

i=1
n(cid:88)

i=1

= 0 ? �� = 1
n

?Xi ? �?2 ? d?2
?3

i=1
= 0 ? �?2 = 1
dn

n(cid:88)

i=1

?Xi ? �?2

n(cid:88)

Xi

[Find the critical point of ?]

[The hats � mean �estimated�]

We don�t know � exactly, so substitute �� for � to compute �?.

Takeaway: use sample mean & variance of pts in class C to estimate mean & variance of Gaussian for
class C.

For QDA:

estimate conditional mean ��C & conditional variance �?2
& estimate the priors:

C of each class C separately [as above]

�?C = nC
(cid:80)
D nD

? total sample points in all classes

[�?C is the coin flip parameter]

For LDA: same means & priors; one variance for all classes:

�?2 = 1
dn

(cid:88)

(cid:88)

C

{i:yi=C}

?Xi ? ��C?2

? pooled within-class variance

[Notice that although LDA is computing one variance for all the data, each sample point contributes with
respect to its own class�s mean. This gives a very different result than if you simply use the global mean!
It�s usually smaller than the global variance. We say �within-class� because we use each point�s distance
from its class�s mean, but �pooled� because we then pool all the classes together.]

Eigenvectors and the (Anisotropic) Multivariate Normal Distribution

41

8 Eigenvectors and the (Anisotropic) Multivariate Normal Distribution

EIGENVECTORS

[I don�t know if you were properly taught about eigenvectors here at Berkeley, but I sure don�t like the way
they�re taught in most linear algebra books. So I�ll start with a review. You all know the definition of an
eigenvector:]
Given square matrix A, if Av = ?v for some vector v (cid:44) 0, scalar ?, then
v is an eigenvector of A and ? is the eigenvalue of A associated w/v.

[But what does that mean? It means that v is a magical vector that, after being multiplied by A, still points
in the same direction, or in exactly the opposite direction.]

Eigenvalue 2:

v

Av

A2v

A3v

Eigenvalue ? 1
2:

Aw

A3w

A2w

w

Draw this figure by hand (eigenvectors.pdf)

[For most matrices, most vectors don�t have this property. So the ones that do are special, and we call them
eigenvectors.]
[Clearly, when you scale an eigenvector, it�s still an eigenvector. Only the direction matters, not the length.
Let�s look at a few consequences.]

Theorem:

if v is eigenvector of A w/eigenvalue ?,
then v is eigenvector of Ak w/eigenvalue ?k

Proof: A2v = A(?v) = ?Av = ?2v, etc.

Theorem: moreover, if A is invertible,

then v is eigenvector of A?1 w/eigenvalue 1/?

[k is a +ve integer; we will use Theorem later]

Proof: A?1v = A?1( 1

? Av) = 1
? v

[look at the figures above, but go from right to left.]

[Stated simply: When you invert a matrix, the eigenvectors don�t change, but the eigenvalues get inverted.
When you square a matrix, the eigenvectors don�t change, but the eigenvalues get squared.]

[Those theorems are pretty obvious. The next theorem is not obvious at all.]

42

Jonathan Richard Shewchuk

Spectral Theorem:

every real, symmetric n � n matrix has real eigenvalues and
i v j = 0
n eigenvectors that are mutually orthogonal, i.e., v?

for all i (cid:44) j

[This takes about a page of math to prove. One detail is that a matrix can have more than n eigenvector
directions. If two eigenvectors happen to have the same eigenvalue, then every linear combination of those
eigenvectors is also an eigenvector. Then you have infinitely many eigenvector directions, but they all span
the same plane. So you just arbitrarily pick two vectors in that plane that are orthogonal to each other. By
contrast, the set of eigenvalues is always uniquely determined by a matrix, including the multiplicity of the
eigenvalues.]
We can use them as a basis for Rn.

Building a Matrix with Specified Eigenvectors

[There are a lot of applications where you�re given a matrix, and you want to extract the eigenvectors and
eigenvalues. But when you�re learning the math, I think it�s more intuitive to go in the opposite direction.
Suppose you know what eigenvectors and eigenvalues you want, and you want to create the matrix that has
those eigenvectors and eigenvalues.]

Choose n mutually orthogonal unit n-vectors v1, . . . , vn [so they specify an orthonormal coordinate system]
Let V = [v1
Observe: V ?V = I

? n � n matrix

vn]

. . .

v2

[off-diagonal 0�s because the vectors are orthogonal]
[diagonal 1�s because they�re unit vectors]

? V ? = V ?1 ? VV ? = I

V is orthonormal matrix: acts like rotation (or reflection)

Choose some eigenvalues ?i:
?

Let ? =

????????????????

?1
0
...
0

0
?2

0

. . .

0
0
...
. . .
. . . ?n

?

????????????????

[diagonal matrix of eigenvalues]

Defn. of eigenvector: AV = V?
[This is the same definition of eigenvector I gave you at the start of the lecture�Av = ?v�but this version
covers all n eigenvectors in one statement. How do we find the A that satisfies this equation?]

? AVV ? = V?V ?
Theorem: A = V?V ? = (cid:80)n

[which proves . . . ]

i=1 ?i viv?
i(cid:124)(cid:123)(cid:122)(cid:125)
outer product: n � n matrix, rank 1

has chosen eigenvectors/values

This is a matrix factorization called the eigendecomposition.
Example: [Using the eigenvectors and eigenvalues from the start of the lecture]
(cid:35)

(cid:35) (cid:34)

(cid:35) (cid:34)

(cid:35)

(cid:34)

(cid:34)

A =

?
1/
2
?
2 ?1/

?
2
?
2

1/
1/

0

2
0 ?1/2

?
1/
2
?
2 ?1/

?
2
?
2

1/
1/

=

3/4 5/4
5/4 3/4

.

[every real, symmetric matrix has one]

[This completes our task of finding a symmetric matrix with specified orthonormal eigenvectors and eigen-
values. Again, it is more common in practice that you are given a symmetric matrix, such as a sample
covariance matrix, and you need to compute its eigenvectors and eigenvalues. That�s harder. But I think that
going from eigenvectors to the matrix helps to build intuition.]

Eigenvectors and the (Anisotropic) Multivariate Normal Distribution

43

Observe: A2 = V?V ?V?V ? = V?2V ?
[This is another way to see that squaring a matrix squares its eigenvalues without changing its eigenvectors.
It also suggests a way to define a matrix square root.]
Given a symmetric PSD matrix ?, we can find a symmetric square root A = ?1/2:

A?1 = (V?V ?)?1 = (V ?)?1??1V ?1 = V??1V ?

compute eigenvectors/values of ?
take square roots of ?�s eigenvalues
reassemble matrix A [with the same eigenvectors as ? but changed eigenvalues]

[Again, the first step of this algorithm�computing the eigenvectors and eigenvalues of a matrix�is much
harder than the remaining two steps.]

Visualizing Quadratic Forms

[My favorite way to visualize a symmetric matrix is to graph something called the quadratic form, which
shows how applying the matrix affects the length of a vector.]
The quadratic form of M is x?Mx.

Suppose you want a matrix whose quadratic form has the isocontours at right below, which are circles
transformed by A. [The same matrix A I�ve been using, which stretches along the direction with eigenvalue 2
and shrinks along the direction with eigenvalue ?1/2.]

isocontours
transformed by A
??

z-space
qs(z) = ?z?2

x-space
qe(x) = ???

?z?2

???

circles.pdf, ellipses.pdf, circlebowl.pdf, ellipsebowl.pdf
[Both figures at left are plots of ?z?2, and both figures at right are plots of x?A?2x.
(Draw the stretch direction (1, 1) with eigenvalue 2 and the shrink direction (1, ?1) with
eigenvalue ? 1

2 on the ellipses at right.)]

1234555566667777-2-1012-2-10121.2.2.3.3.4.4.5.5.6.6.7.7.8.8.9.9.10.10.11.11.12.12.13.13.14.14.15.15.16.16.17.17.18.18.19.19.-2-1012-2-101244

Jonathan Richard Shewchuk

That is, we want qe(Az) = qs(z).
Answer: set x = Az.
Then qe(x) = qs(z) = qs(A?1x) = ?A?1x?2 = x?A?2x.
The isocontours of the quadratic form x?A?2x are ellipsoids determined by the eigenvectors/values of A.
{x : x?A?2x = 1} is an ellipsoid with

axes v1, v2, . . . , vn and
radii ?1, ?2, . . . , ?n
because if vi has length 1 (vi lies on unit circle), x = Avi has length ?i (Avi lies on the ellipsoid).
Therefore, isocontours of x?Mx are ellipsoids determined by eigenvectors/values of M?1/2.
[The eigenvalues of M?1/2 are the inverse square roots of the eigenvalues of M.]
Special case: A (or M) is diagonal ? eigenvectors are coordinate axes

? ellipsoids are axis-aligned

[Draw axis-aligned isocontours for a diagonal metric.]

A symmetric matrix M is

positive definite
positive semidefinite
indefinite
invertible

if w?Mw > 0 for all w (cid:44) 0 ? all eigenvalues positive
if w?Mw ? 0 for all w ? all eigenvalues nonnegative
if +ve eigenvalue & ?ve eigenvalue
if no zero eigenvalue

pos definite

pos semidefinite

indefinite

posdef.pdf, possemi.pdf, indef.pdf
[Examples of quadratic forms for positive definite, positive semidefinite, and indefinite ma-
trices. Positive eigenvalues correspond to axes where the curvature goes up; negative eigen-
values correspond to axes where the curvature goes down. (Draw the eigenvector directions,
and draw the flat trough in the positive semidefinite bowl.)]

Every squared matrix is pos semidef, including A?2. [Eigenvalues of A?2 are squared, cannot be negative.]
If A?2 exists, it is pos def. [An invertible matrix has no zero eigenvalues.]
What about the isosurfaces of x?Mx for a +ve semidef, singular M?

[If M is only positive semidefinite, but not positive definite, the isosurfaces are cylinders instead of ellipsoids.
These cylinders have ellipsoidal cross sections spanning the directions with nonzero eigenvalues, but they
run in straight lines along the directions with zero eigenvalues.]

Eigenvectors and the (Anisotropic) Multivariate Normal Distribution

45

ANISOTROPIC GAUSSIANS

[Let�s revisit the multivariate Gaussian distribution, with different variances along different directions.]

X ? N(�, ?)

[X and � are d-vectors. X is a random variable with mean �.]

f (x) =

(cid:112)

1
(2?)d|?|

(cid:32)
?

exp

1
2
? determinant of ?

(x ? �)? ??1 (x ? �)

(cid:33)

? is the d � d SPD covariance matrix.
??1 is the d � d SPD precision matrix.

Write f (x) = n(q(x)), where q(x) = (x ? �)? ??1 (x ? �)

?

?

R ? R, exponential Rd ? R, quadratic

[Now q(x) is a function we understand�it�s just a quadratic bowl centered at �, the quadratic form of the
precision matrix ??1. The other function n(�) is a simple, monotonic, convex function, an exponential of the
negation of half its argument. This mapping n(�) does not change the isosurfaces.]
Principle: given monotonic n : R ? R, isosurfaces of n(q(x)) are same as q(x) (different isovalues).

?

?

n(�)

q(x)

f (x) = n(q(x))

ellipsebowl.pdf, ellipses.pdf, exp.pdf, gauss3d.pdf, gausscontour.pdf
[(Show this figure on a separate �whiteboard� for easy reuse next lecture.) A paraboloid
(left) becomes a bivariate Gaussian (right) after you compose it with a suitable scalar func-
tion (center).]

1.2.2.3.3.4.4.5.5.6.6.7.7.8.8.9.9.10.10.11.11.12.12.13.13.14.14.15.15.16.16.17.17.18.18.19.19.-2-1012-2-101201234x0.050.100.15n(x)0.0360.0360.0720.0720.1080.1080.1440.1440.180.2160.2520.2880.3240.36-2-1012-2-101246

Jonathan Richard Shewchuk

[One of the main ideas is that if you understand the isosurfaces of a quadratic function, then you understand
the isosurfaces of a Gaussian, because they�re the same. The differences are in the isovalues�in particular,
the Gaussian achieves its maximum at the mean, and decreases to zero as you move infinitely far away from
the mean.]
The isocontours of (x ? �)???1(x ? �) are determined by eigenvectors/values of ?1/2.

isocontours transformed
by ?1/2
??

d(x, �) = (cid:13)(cid:13)(cid:13)??1/2x ? ??1/2�

Aside: q(x) is the squared distance from ??1/2x to ??1/2�. Consider the metric
(cid:113)
(x ? �)???1(x ? �) = (cid:112)
[So we think of the precision matrix as a �metric tensor� which defines a metric, a sort of warped distance
from x to the mean �.]

(cid:13)(cid:13)(cid:13) =

q(x).

Covariance

Let R, S be random variables�column vectors or scalars
Cov(R, S ) = E[(R ? E[R]) (S ? E[S ])?] = E[RS ?] ? �R �?
S
Var(R) = Cov(R, R)
If R is a vector, covariance matrix for R is

Var(R) =

?

????????????????

Var(R1)
Cov(R2, R1)
...

Cov(R1, R2)
Var(R2)

Cov(Rd, R1) Cov(Rd, R2)

. . . Cov(R1, Rd)
Cov(R2, Rd)
...
Var(Rd)

. . .
. . .

?

????????????????

[symmetric; each Ri is scalar]

[. . . as you did in Homework 2.]

For a Gaussian R ? N(�, ?), one can show Var(R) = ?.
[An important point is that statisticians didn�t just arbitrarily decide to call ? a covariance matrix. Rather,
statisticians discovered that if you find the covariance of the normal distribution by integration, it turns out
that the covariance is ?. This is a happy fact; it�s rather elegant.]
Ri, R j independent ? Cov(Ri, R j) = 0
Cov(Ri, R j) = 0 AND multivariate normal dist. ? Ri, R j independent
all features pairwise independent ? Var(R) is diagonal
Var(R) is diagonal AND multi normal

[the reverse implication is not generally true, but . . . ]

[the reverse is not generally true, but . . . ]

?

f (x)
(cid:124)(cid:123)(cid:122)(cid:125)
multivariate

= f (x1) f (x2) � � � f (xd)
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
univariate Gaussians

? ellipsoids are axis-aligned, with squared radii on diagonal of ? = Var(R)

[So when the features are independent, you can write the multivariate Gaussian PDF as a product of uni-
variate Gaussian PDFs. When they aren�t, you can do a change of coordinates to the eigenvector coordinate
system, and write it as a product of univariate Gaussian PDFs in eigenvector coordinates. You did something
very similar in Q7.2 of Homework 2.]

1234555566667777-2-1012-2-10121.2.2.3.3.4.4.5.5.6.6.7.7.8.8.9.9.10.10.11.11.12.12.13.13.14.14.15.15.16.16.17.17.18.18.19.19.-2-1012-2-1012Anisotropic Gaussians: MLE, QDA, and LDA Revisited

47

9 Anisotropic Gaussians: MLE, QDA, and LDA Revisited

GDA WITH ANISOTROPIC GAUSSIANS

[Recall from our last lecture the probability density function of the multivariate normal distribution in its
full generality. x and � are d-vectors.]

Normal PDF: f (x) = n(q(x)),

n(q) =

?
R ? R, exponential

(cid:112)

e?q/2,

1
(2?)d|?|
?
determinant of ?

q(x) = (x ? �)? ??1 (x ? �).

?

Rd ? R, quadratic

[The covariance matrix ? and its symmetric square root and its inverse all play roles in our intuition about
the multivariate normal distribution. Consider their eigendecompositions.]
? = V?V ?

covariance matrix

? eigenvalues of ? are variances along the eigenvectors, ?ii = ?2
i

?1/2 = V?1/2V ? maps spheres to ellipsoids

[recall end of last lecture]

? eigenvalues of ?1/2 are Gaussian widths / ellipsoid radii / standard deviations,

?

?ii = ?i

?1/2
??

?? quadratic form

q(x) = (x ? �)? ??1 (x ? �)

??1 = V??1V ?

precision matrix (metric tensor)

[? quadratic form of ??1 defines contours]

[Recall from last lecture
that the isocontours of the
multivariate normal
distribution are the same as
the isocontours of the
quadratic form of the
precision matrix ??1.]

?

?

n(�)

q(x)

f (x) = n(q(x))

1234555566667777-2-1012-2-10121.2.2.3.3.4.4.5.5.6.6.7.7.8.8.9.9.10.10.11.11.12.12.13.13.14.14.15.15.16.16.17.17.18.18.19.19.-2-1012-2-10121.2.2.3.3.4.4.5.5.6.6.7.7.8.8.9.9.10.10.11.11.12.12.13.13.14.14.15.15.16.16.17.17.18.18.19.19.-2-1012-2-101201234x0.050.100.15n(x)0.0360.0360.0720.0720.1080.1080.1440.1440.180.2160.2520.2880.3240.36-2-1012-2-101248

Jonathan Richard Shewchuk

Maximum Likelihood Estimation for Anisotropic Gaussians

MLE40pts.pdf, MLE40pts3D.pdf [Maximum likelihood estimation takes these 40 points
as input and outputs this Gaussian. Note that the points do not actually come from a normal
distribution; they come from a uniform distribution over a tilted rectangle. Nevertheless,
the Gaussian is a decent approximation of that.]

Given training points X1, . . . , Xn and classes y1, . . . , yn, find best-fit Gaussians.
Let nC = # of training pts in class C.

[Once again, we want to choose the Gaussian parameters that maximize the likelihood of generating the
training points in a specified class. This time I won�t derive the maximum-likelihood Gaussian; I�ll just tell
you the answer.]

For QDA:

�?C = 1
nC

(cid:88)

i:yi=C

(Xi ? ��C) (Xi ? ��C)?
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
outer product matrix, d � d

? conditional covariance for pts in class C

Prior �?C, mean ��C: same as Lecture 7.
[�?C is number of training points in class C � total training points; ��C is mean of training points in class C.]

�?C is positive semidefinite, but not always definite!
[If there are some zero eigenvalues, the standard version of QDA just doesn�t work. We can try to fix it by
eliminating the zero-variance dimensions (eigenvectors). Homework 3 suggests a way to do that.]

For LDA:

�? = 1
n

(cid:88)

(cid:88)

(Xi ? ��C) (Xi ? ��C)?

C

i:yi=C

? pooled within-class covariance matrix

-4-20246810-4-20246810Anisotropic Gaussians: MLE, QDA, and LDA Revisited

49

[Let�s revisit QDA and LDA and see what has changed now that we use the multivariate normal distribution
in its full, anisotropic generality. The short answer is �not much has changed, but the graphs look cooler.�
Conflicting notation warning: capital X represents a random variable, but later it will represent a matrix.]

QDA

Choosing C that maximizes P(Y = C|X = x) is equivalent to maximizing the quadratic discriminant fn

QC(x) = ln

(cid:16)

(

?

2?)d fX|Y=C(x) ?C

(cid:17) = ?

1
2

(x ? �C)? ??1

C (x ? �C) ?

1
2

ln |?C| + ln ?C.

[This works for any number of classes. In a multi-class problem, you just pick the class with the greatest
quadratic discriminant for x.]
2 classes: Decision fn QC(x) ? QD(x) is quadratic, but may be indefinite.

? Decision boundary is a quadric.
Posterior is P(Y = C|X = x) = s(QC(x) ? QD(x)) where s(�) is logistic fn.

fX|Y=C(x) & fX|Y=D(x)

QC ? QD

?

?

s(QC ? QD)

s(�)

qdaaniso3d.pdf, qdaanisocontour.pdf, qdaanisodiff3d.pdf, qdaanisodiffcontour.pdf,
logistic.pdf, qdaanisoposterior3d.pdf, qdaanisoposteriorcontour.pdf
[(Show this figure on a separate �whiteboard.�) An example where the decision boundary
is a hyperbola�which is not possible with isotropic Gaussians. At left, two anisotropic
Gaussians. Center left, the difference QC ? QD. After applying the logistic function to this
difference we obtain the posterior probabilities at right, which tells us the probability that x
is in class C. Observe that we can see the decision boundary in all three contour plots: it is
QC ? QD = 0 and s(QC ? QD) = 0.5. We don�t need to apply the logistic function to find
the decision boundary, but we do need to compute it if we want the posterior probabilities.]

-3-2-10123-3-2-10123-3-2-10123-3-2-10123-4-2024x0.20.40.60.81.0s(x)0.10.10.20.20.30.30.40.40.50.50.60.60.70.70.80.80.90.9-3-2-10123-3-2-1012350

Jonathan Richard Shewchuk

[This procedure has two interpretations. If we actually know the exact, true parameters ?C, �C, and ?C, this
procedure gives us the Bayes classifier and the Bayes optimal decision boundary. By contrast, when we
estimate �?C, ��C, and �?C from data, this procedure is the QDA algorithm. We hope the QDA classifier will
approximate the Bayes classifier. Sometimes in our textbooks, you will see examples where they plot both
the Bayes optimal decision boundary and the decision boundary computed by a learning algorithm. (See
the figure two pages forward.) When you see that, the authors know the exact, true probability distributions
because they have chosen them and written a program that produces synthetic data from those distributions.
With data from the real world, you cannot know the Bayes optimal decision boundary.]

Multi-class QDA:

aniso.pdf
their QDA decision boundaries form an
anisotropic Voronoi diagram. Interestingly, a cell of this diagram might not be connected.]

[When you have many classes,

LDA

One ? for all classes. Decision fn is
[Once again, the quadratic terms cancel each other out so the decision function is linear and the decision
boundary is a hyperplane.]

QC(x) ? QD(x) = (�C ? �D)? ??1 x
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
w?x

�?
C

??1 �C ? �?
D

??1 �D
+ ln ?C ? ln ?D
?
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
+?

2

.

Decision boundary is w?x + ? = 0.
Posterior is P(Y = C|X = x) = s(w?x + ?).

Multi-class LDA: choose class C that maximizes the linear discriminant fn

�?
C

??1 x ?

�?
C

??1 �C
2

+ ln ?C.

[works for any # of classes]

[Note that we use a linear solver to efficiently compute �?
C
points quickly.]

??1 just once, so the classifier can evaluate test

Anisotropic Gaussians: MLE, QDA, and LDA Revisited

51

fX|Y=C(x) & fX|Y=D(x)

QC ? QD

?

?

s(QC ? QD)

s(�)

ldaaniso3d.pdf, ldaanisocontour.pdf, ldaanisodiff3d.pdf, ldaanisodiffcontour.pdf,
logistic.pdf, ldaanisoposterior3d.pdf, ldaanisoposteriorcontour.pdf
[(Show this figure on a separate �whiteboard.�) In LDA, the decision boundary is always a
hyperplane. Note that Mathematica messed up the top left plot a bit; there should be no red
in the left corner, nor blue in the right corner.]

LDAdata.pdf (ESL, Figure 4.11) [An example of LDA with messy data. The real-world
distributions almost surely aren�t Gaussians, but LDA still works reasonably well.]

-3-2-10123-3-2-10123-3-2-10123-3-2-10123-4-2024x0.20.40.60.81.0s(x)0.10.20.30.40.50.60.70.80.9-3-2-10123-3-2-10123oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo����������������������52

Jonathan Richard Shewchuk

Notes on QDA/LDA

� For 2 classes,

� LDA has d + 1 parameters (w, ?);

[. . . in the decision function. We estimated more statistical parameters than that, but only the
degrees of freedom of the decision function matter for diagnosing underfitting or overfitting.]

d(d + 3)
� QDA has
2
� LDA more likely to underfit;
� QDA more likely to overfit.

+ 1 params;

[The danger is much bigger when the dimension d is large.]

ldaqda.pdf (ISL, Figure 4.9) [In these examples, the Bayes optimal decision boundary is
purple (and dashed), the QDA decision boundary is green, the LDA decision boundary is
black (and dotted). When the Bayes optimal boundary is linear, as at left, LDA gives a
more stable fit whereas QDA may overfit. When the Bayes optimal boundary is curved, as
at right, QDA often gives you a better fit.]

� QDA on data doesn�t find true optimum Bayes classifier.

� estimate distributions from finite data.
� real-world data not perfectly Gaussian.

� Changing priors or loss = adding constants to discriminant fns.

[So it�s very easy. In the 2-class case, it�s equivalent to changing the isovalue . . . ]

� Posterior gives decision boundaries for 10% probability, 50%, 90%, etc.

choosing isovalue = probability p is equivalent to

� choosing ?C = 1 ? p, ?D = p; OR
� choosing asymmetrical loss p for false positive, 1 ? p for false negative.
� With added features, LDA can give nonlinear boundaries; QDA nonquadratic.

[LDA & QDA are the best method in practice for many applications. In the STATLOG project, either LDA
or QDA were among the top three classifiers for 10 out of 22 datasets. But it�s not because all those datasets
are Gaussian. LDA & QDA work well when the data can only support simple decision boundaries such as
linear or quadratic, because Gaussian models provide stable estimates. See ESL, Section 4.3.]

?4?2024?4?3?2?1012?4?2024?4?3?2?1012X2Anisotropic Gaussians: MLE, QDA, and LDA Revisited

53

Some Terms

Let X be n � d design matrix of sample pts
Each row i of X is a sample pt X?
i .

[Now I�m using capital X as a matrix instead of a random variable vector. I�m treating Xi as a column vector
to match the standard convention for multivariate PDFs like the Gaussian, but X?
i
centering X: subtracting ��? from each row of X. X ? ?X
[ ��? is the mean of all the rows of X. Now the mean of all the rows of ?X is zero.]

is a row of X.]

Let R be drawn from uniform distribution on sample pts. Sample covariance matrix is

Var(R) = 1
n

?X? ?X.

[This is the simplest way to remember how to compute a covariance matrix for QDA. Imagine you have a
design matrix XC that contains only the sample points of class C; then you have �?C = 1
nC
[When we have points from an anisotropic Gaussian distribution, sometimes it�s useful to perform a linear
transformation that maps them to an axis-aligned distribution, or maybe even to an isotropic distribution.]
decorrelating ?X: applying rotation Z = ?XV, where Var(R) = V?V ?
[rotates the sample points to the eigenvector coordinate system]
Then Var(Z) = ?.
[Proof: Var(Z) = 1

[Z has diagonal covariance. If Xi ? N(�, ?), then approximately, Zi ? N(0, ?).]

n V ? ?X? ?XV = V ?Var(R)V = V ?V?V ?V = ?.]

n Z?Z = 1

?XC.]

?X?
C

original.jpg, centered.jpg, decorrelated.jpg, whitened.jpg

sphering ?X: applying transform W = ?X Var(R)?1/2
whitening X: centering + sphering, X ? W

[Recall that ??1/2 maps ellipsoids to spheres.]

Then W has covariance matrix I.

[If Xi ? N(�, ?), then approximately, Wi ? N(0, I).]

[Whitening input data is often used with other machine learning algorithms, like SVMs and neural networks.
The idea is that some features may be much bigger than others�for instance, because they�re measured in
different units. SVMs penalize violations by large features more heavily than they penalize small features.
Whitening the data before you run an SVM puts the features on an equal basis.]

[One nice thing about discriminant analysis is that whitening is built in.]
[Incidentally, what we�ve done here�computing a sample covariance matrix and its eigenvectors/values�
is about 75% of an important unsupervised learning method called principal components analysis, or PCA,
which we�ll learn later in the semester.]

54

Jonathan Richard Shewchuk

10 Regression, including Least-Squares Linear and Logistic Regression

REGRESSION aka Fitting Curves to Data

Classification:
Regression:

given point x, predict class (often binary)
given point x, predict a numerical value

[Classification gives a discrete prediction, whereas regression gives us a quantitative prediction, usually on a
continuous scale. We�ve already seen an example of regression in Gaussian discriminant analysis. QDA and
LDA don�t just estimate a classifier; they also give us the probability that a particular prediction is correct.
So QDA and LDA do regression on probability values.]

� Choose form of regression fn h(x; w) with parameters w

(h = hypothesis)

� like decision fn in classification [e.g., linear, quadratic, logistic in x]

� Choose a cost fn (objective fn) to optimize

� usually based on a loss fn; e.g., empirical risk = expected loss on data

Some regression fns:

(1)
(2)
(3)

linear: h(x; w, ?) = w � x + ?
polynomial [equivalent to linear regression with added polynomial features]
logistic: h(x; w, ?) = s(w � x + ?)

recall: logistic fn s(?) = 1

1+e??

[The last choice is interesting. You�ll recall that LDA produces a posterior probability function with this
expression. So the logistic function seems to be a natural form for modeling certain probabilities. If we want
to model posterior probabilities, sometimes we use LDA; but alternatively, we could skip fitting Gaussians
to points, and instead just try to directly fit a logistic function to a set of probabilities.]

Some loss fns: let �y be prediction h(x); y be true label

L(�y, y) = (�y ? y)2
L(�y, y) = |�y ? y|
L(�y, y) = ?y ln �y ? (1 ? y) ln(1 ? �y)

squared error
absolute error
logistic loss, aka cross-entropy: y ? [0, 1], �y ? (0, 1)

Some cost fns to minimize:

i=1 L(h(Xi), yi)
i=1 L(h(Xi), yi)
i=1 ?i L(h(Xi), yi)

(cid:80)n
J(h) = 1
n
J(h) = maxn
J(h) = (cid:80)n
J(h) = (a), (b), or (c) +??w?2
J(h) = (a), (b), or (c) +??w??1

n �]

[you can leave out the � 1

mean loss
maximum loss
weighted sum [some points are more important than others]
?2 penalized/regularized
?1 penalized/regularized

(A)
(B)
(C)

(a)
(b)
(c)
(d)
(e)

Some famous regression methods:

Least-squares linear regr.:
Weighted least-squ. linear:
Ridge regression:
Lasso:
Logistic regr.:
Least absolute deviations:
Chebyshev criterion:

(1) + (A) + (a)
(1) + (A) + (c)
(1) + (A) + (a) + (d)
(1) + (A) + (a) + (e)
(3) + (C) + (a)
(1) + (B) + (a)
(1) + (B) + (b)

?
???
???

quadratic cost; minimize w/calculus

quadratic program
convex cost; minimize w/gradient descent
(cid:27)

linear program

[I have given you several choices of regression function, several choices of loss function, and several choices
of objective function. You can snap one part out and replace it with a different one. But the optimization
algorithm and its speed depend crucially on which parts you pick. Let�s consider some examples.]

Regression, including Least-Squares Linear and Logistic Regression

55

LEAST-SQUARES LINEAR REGRESSION (Gauss, 1801)

Linear regression fn (1) + squared loss fn (A) + cost fn (a).

Find w, ? that minimizes

n(cid:88)

i=1

(Xi � w + ? ? yi)2.

linregress.pdf (ISL, Figure 3.4) [An example of linear regression.]

Convention: X is n � d design matrix of sample pts

y is n-vector of scalar labels

?

???????????????????????????????????

. . . X1 j
X2 j

. . . X1d
X2d

Xi j

Xid

X11 X12
X21 X22
...
Xi1 Xi2
...

Xn1 Xn2

Xn j

Xnd

?

???????????????????????????????????

? point X?
i

?
feature column X? j

?

?????????????????????????????????

?

?????????????????????????????????

y1
y2

...

yn

?
y

Usually n > d.
Recall fictitious dimension trick [from Lecture 3]: rewrite h(x) = x � w + ? as

[But not always.]

[x1

x2

1]

?

?????????

w1
w2
?

?

?????????

.

Now X is an n � (d + 1) matrix; w is a (d + 1)-vector.
[We rewrite the optimization problem above:]

[We�ve added a column of all-1�s to the end of X.]

Find w that minimizes ?Xw ? y?2 = RSS(w), for residual sum of squares

��������������������������������������������������������������������������X1X256

Jonathan Richard Shewchuk

Optimize by calculus:

minimize RSS(w) = w?X?Xw ? 2y?Xw + y?y

? RSS = 2X?Xw ? 2X?y = 0

? X?X
(cid:124)(cid:123)(cid:122)(cid:125)
(d+1)�(d+1)

w = X?y
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
(d+1)?vectors

? the normal equations [w unknown; X & y known]

If X?X is singular, problem is underconstrained.
[. . . because the sample points all lie on a common subspace (through the origin).]
[Notice that X?X is always positive semidefinite, but not always positive definite.]
We use a linear solver to find w = (X?X)?1X?
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)

y.

X+, the pseudoinverse of X, (d+1)�n

[never actually invert the matrix!]

where H
(cid:124)(cid:123)(cid:122)(cid:125)
n�n

directly, but we are interested in the fact that w is a linear transformation of y.]

[We never compute X+
[X is usually not square, so X can�t have an inverse. However, every X has a pseudoinverse X+
is invertible, then X+
Observe: X+X = (X?X)?1X?X = I ? (d + 1) � (d + 1)
Observe: the predicted values of yi are �yi = w � Xi ? �y = Xw = XX+y = Hy
is called the hat matrix because it puts the hat on y.

is a �left inverse.�]

[which explains the name �left inverse�]

= XX+

, and if X?X

[Ideally, H would be the identity matrix and we�d have a perfect fit, but if n > d + 1, then H is singular.]

Advantages:

� Easy to compute; just solve a linear system.
� Unique, stable solution. [. . . except when the problem is underconstrained.]

Disadvantages:

� Very sensitive to outliers, because errors are squared!
� Fails if X?X is singular. [Which means the problem is underconstrained, has multiple solutions.]
[In discussion section 6, we�ll address how to handle the underconstrained case where X?X is singular.]

[Apparently, least-squares linear regression was first posed and solved in 1801 by the great mathematician
Carl Friedrich Gauss, who used least-squares regression to predict the trajectory of the planetoid Ceres.
A paper he wrote on the topic is regarded as the birth of modern linear algebra.]

LOGISTIC REGRESSION (David Cox, 1958)

Logistic regression fn (3) + logistic loss fn (C) + cost fn (a).
Fits �probabilities� in range [0, 1].

Usually used for classification. The input yi�s can be probabilities,
but in most applications they�re all 0 or 1.

QDA, LDA: generative models
logistic regression: discriminative model
[We�ve learned from LDA that in classification, the posterior probabilities are often modeled well by a
logistic function. So why not just try to fit a logistic function directly to the data, skipping the Gaussians?]

Regression, including Least-Squares Linear and Logistic Regression

57

With X and w including the fictitious dimension; ? is w�s last component . . .

Find w that minimizes

J =

n(cid:88)

i=1

L(s(Xi � w), yi) = ?

n(cid:88)

i=1

?
?
?????yi ln s(Xi � w) + (1 ? yi) ln (1 ? s(Xi � w))
????? .

L(�y, 0)

L(�y, 0.7)

logloss0.pdf, loglosspt7.pdf [Plots of the loss L(�y, y) for y = 0 (left) and y = 0.7 (right). As
you might guess, the left function is minimized at �y = 0, and the right function is minimized
at �y = 0.7. These loss functions are always convex.]

J(w) is convex! Solve by gradient descent.
[To do gradient descent, we�ll need to compute some derivatives.]

s?(?) =

d
d?

1
1 + e??

=

e??
(1 + e??)2

= s(?) (1 ? s(?))

logistic.pdf, dlogistic.pdf [Plots of s(?) (left) and s?(?) (right).]

Let si = s(Xi � w)

?w J = ?

= ?

(cid:88) (cid:32)

(cid:88) (cid:32)

(cid:33)

?si

1 ? yi
1 ? si
(cid:33)

yi
si
yi
si

?si ?

?

1 ? yi
1 ? si

si(1 ? si) Xi

(cid:88)

= ?

(yi ? si) Xi

= ?X?(y ? s(Xw))

where s(Xw) =

?

????????????????

?

????????????????

s1
s2
...
sn

[applies s component-wise to Xw]

0.00.20.40.60.81.0z1234L(z)0.00.20.40.60.81.0z1234L(z)-4-2024x0.20.40.60.81.0s(x)-4-2024x0.050.100.150.200.25s(cid:1)(x)58

Jonathan Richard Shewchuk

Gradient descent rule: w ? w + ? X?(y ? s(Xw))
Stochastic gradient descent: w ? w + ? (yi ? s(Xi � w)) Xi
Works best if we shuffle points in random order, process one by one.
For very large n, sometimes converges before we visit all points!
[This looks a lot like the perceptron learning rule. The only difference is that the �?si� part is new.]
Starting from w = 0 works well in practice.

problogistic.png, by �mwascom� of Stack Overflow
http://stackoverflow.com/questions/28256058/plotting-decision-boundary-of-logistic-regression
[An example of logistic regression.]

If sample pts are linearly separable and w � x = 0 separates them (with decision boundary touching no pt),
scaling w to have infinite length causes s(Xi � w) ? 1 for a pt i in class C, s(Xi � w) ? 0 for a pt not in class C,
and J(w) ? 0 [in the limit as ?w? ? ?].

[Moreover, making w grow extremely large is the only way to get the cost function J to approach zero.]

Therefore, logistic regression always separates linearly separable pts!

[In this case, the cost function J(w) has no finite local minimum, but gradient descent will �converge� to a
solution, in the sense that the cost J will get arbitrarily close to zero, though of course the weight vector w
will never become infinitely long. Mathematically speaking, w doesn�t converge at all�it diverges�though
J(w) does converge to zero.]
[A 2018 paper by Soudry, Hoffer, Nacson, Gunasekar, and Srebro shows that gradient descent applied to
logistic regression eventually converges to the maximum margin classifier, but the convergence is very, very
slow. A practical logistic regression solver should use a different optimization algorithm.]

Polynomial and Weighted Regression; Newton�s Method; ROC Curves

59

11 Polynomial and Weighted Regression; Newton�s Method; ROC Curves

LEAST-SQUARES POLYNOMIAL REGRESSION

Replace each Xi with feature vector ?(Xi) with all terms of degree 0 . . . p
e.g., ?(Xi) = [X2

i1 Xi1Xi2 X2

i2 Xi1 Xi2

1]?

[Notice that we�ve added the fictitious dimension �1� here, so we don�t need to add it again to do linear or
logistic regression. This basis covers all polynomials quadratic in Xi1 and Xi2.]

Otherwise just like linear or logistic regression.
Log. reg. + quadratic features = same form of posteriors as QDA.

Very easy to overfit!

overunder.png, degree20.png, UScensusquartic.png

[Here are some examples of polynomial overfitting, to show the importance of choosing the polynomial
degree very carefully. At left, we have sampled points from a degree-3 curve (black) with added noise. We
show best-fit polynomials of degrees 2, 4, 6, and 8 found by regression of the black points. The degree-4
curve (green) fits the true curve (black) well, whereas the degree-2 curve (red) underfits and the degree-6
and 8 curves (blue, yellow) overfit the noise and oscillate. The oscillations in the yellow degree-8 curve are
a characteristic problem of polynomial interpolation.]

[At upper right, a degree-20 curve shows just how insane high-degree polynomial oscillations can get. It
takes a great deal of densely spaced data to tame the oscillations in a high degree curve, and there isn�t
nearly enough data here.]

[At lower right, somebody has regressed a degree-4 curve to U.S. census population numbers. The curve
doesn�t oscillate, but can you nevertheless see a flaw? This shows the difficulty of extrapolation outside the
range of the data. As a general rule, extrapolation is much harder than interpolation. The k-nearest neighbor
classifier is one of the few that does extrapolation decently without occasionally returning crazy values.]

60

Jonathan Richard Shewchuk

order10extrap.pdf [From Mehta, Wang, Day, Richardson, Bukov, Fisher, and Schwab, �A
High-Bias, Low-Variance Introduction to Machine Learning for Physicists.�]

[This example shows that a fitted degree-10 polynomial (green) can be tamed by using a very large amount
of training data (left), even if the training data is noisy. The training data was generated from a different
degree-10 polynomial, with noise added. On the right, the same curves are plotted, but the blue diamonds
are test points, some of which go outside the range of the training data. We see that the degree-10 regression
does decent extrapolation for a short distance, albeit only because the original data was also from a degree-10
polynomial.]

WEIGHTED LEAST-SQUARES REGRESSION

Linear regression fn (1) + squared loss fn (A) + cost fn (c).

[The idea of weighted least-squares is that some sample points might be more trusted than others, or there
might be certain points you want to fit particularly well. So you assign those more trusted points a higher
weight. If you suspect some points of being outliers, you can assign them a lower weight.]
Assign each sample pt a weight ?i; collect ?i�s in n � n diagonal matrix ?.
Greater ?i ? work harder to minimize (�yi ? yi)2

[�yi is predicted label for Xi]

recall: �y = Xw

Find w that minimizes (Xw ? y)??(Xw ? y)

=

n(cid:88)

i=1

?i (Xi � w ? yi)2.

[As with ordinary least-squares regression, we find the minimum by setting the gradient to zero, which leads
us to the normal equations.]
Solve for w in normal equations: X??Xw = X??y

NEWTON�S METHOD

Iterative optimization method for smooth fn J(w).
Often much faster than gradient descent. [We�ll use Newton�s method for logistic regression.]

Idea: You�re at point v. Approximate J(w) near v by quadratic fn.

Jump to its unique critical pt. Repeat until bored.

Polynomial and Weighted Regression; Newton�s Method; ROC Curves

61

[Three iterations of Newton�s method in one-
newton1.pdf, newton2.pdf, newton3.pdf
dimensional space. We seek the minimum of the blue curve, J. Each brown curve is a
local quadratic approximation to J. Each iteration, we jump to the bottom of the brown
parabola.]

newton2D.png [Steps taken by Newton�s method in two-dimensional space.]

Taylor series about v:

?J(w) = ?J(v) + (?2J(v)) (w ? v) + O(?w ? v?2)

where ?2J(v) is the Hessian matrix of J at v.

Approximate critical pt w by setting ?J(w) = 0:

w ? v ? (?2J(v))?1 ?J(v)

[This is an iterative update rule you can repeat until it converges to a solution. As usual, we probably don�t
want to compute a matrix inverse directly. It is faster to solve a linear system of equations, typically by
Cholesky factorization or the conjugate gradient method.]

Newton�s method:

pick starting point w
repeat until convergence

e ? solution to linear system (?2J(w)) e = ??J(w)
w ? w + e

Warning: Doesn�t know difference between minima, maxima, saddle pts.

Starting pt must be �close enough� to desired critical pt.

-224-20-101020304050-224-20-101020304050-224-20-10102030405062

Jonathan Richard Shewchuk

[If the objective function J is actually quadratic, Newton�s method needs only one step to find the exact
solution. The closer J is to quadratic, the faster Newton�s method tends to converge.]

[Newton�s method is superior to gradient descent with a fixed step size for some optimization problems for
at least two reasons. First, it tries to find the right step length to reach the minimum, rather than just walking
an arbitrary distance downhill. Second, rather than follow the direction of steepest descent, it tries to choose
a better descent direction.]

[Nevertheless, it has some major disadvantages. The biggest one is that computing the Hessian can be quite
expensive, and it has to be recomputed every iteration. It can work well for low-dimensional weight spaces,
but you would never use it for a neural network, because there are too many weights. Newton�s method
also doesn�t work for most nonsmooth functions. It particularly fails for the perceptron risk function, whose
Hessian is zero, except where the Hessian is not even defined.]

LOGISTIC REGRESSION (continued)

[Let�s use Newton�s method to solve logistic regression faster.]

Recall: s?(?) = s(?) (1 ? s(?)),

si = s(Xi � w),

s =

?w J(w) = ?

n(cid:88)

i=1

(yi ? si) Xi = ?X?(y ? s)

?

????????????????

s1
s2
...
sn

?

????????????????

,

[Now let�s derive the Hessian too, so we can use Newton�s method.]

w J(w) =
?2

n(cid:88)

i=1

si(1 ? si) XiX?
i

= X??X

where ? =

?

????????????????

s1(1 ? s1)
0
...
0

0
s2(1 ? s2)

0

. . .

. . .
. . .

0
0
...
sn(1 ? sn)

?

????????????????

? is +ve definite ?w ? X??X is +ve semidefinite ?w ? J is convex.
[The logistic regression cost function is convex, so Newton�s method finds a globally optimal point if it
converges at all.]

Newton�s method:

w ? 0
repeat until convergence

e ? solution to normal equations (X??X) e = X?(y ? s)
w ? w + e

Recall: ?, s are fns of w

[Notice that this looks a lot like weighted least squares, but the weight matrix ? and the right-hand-side
vector y ? s change every iteration. So we call it . . . ]
An example of iteratively reweighted least squares.

[We need to be very careful with the analogy, though. The weights don�t have the same meaning they
had when we learned weighted least-squares regression, because there is no ? on the right-hand side of
(X??X) e = X?(y ? s). Contrary to what you�d expect, a small weight in ? causes the Newton iteration to
put more emphasis on a point when it computes e.]

Polynomial and Weighted Regression; Newton�s Method; ROC Curves

63

[Misclassified points far from the decision boundary have the most influence on the step e, and correctly
classified points far from the decision boundary have the least (because yi ? si is small for such a point).
Points near the decision boundary have medium influence. But if there are no misclassified points far from
the decision boundary, then points near the decision boundary have most of the influence.]

[Here�s one more idea for speeding up logistic regression.]
Idea: If n very large, save time by using a random subsample of the pts per iteration. Increase sample size
as you go.
[The principle is that the first iteration isn�t going to take you all the way to the optimal point, so why waste
time looking at all the sample points? Whereas the last iteration should be the most accurate one.]

LDA vs. Logistic Regression

Advantages of LDA:

� For well-separated classes, LDA stable; log. reg. surprisingly unstable
� > 2 classes easy & elegant; log. reg. needs modifying (softmax regression) [see Discussion 6]
� LDA slightly more accurate when classes nearly normal, especially if n is small

Advantages of log. reg.:

� More emphasis on decision boundary; always separates linearly separable pts

[Correctly classified points far from the decision boundary have a small effect on logistic regression�
albeit a bigger effect than they have on SVMs�whereas misclassified points far from the decision
boundary have the biggest effect. By contrast, LDA gives all the sample points equal weight when
fitting Gaussians to them. Weighting points according to how badly they�re misclassified is good for
reducing training error, but it can also be bad if you want stability or insensitivity to bad data.]

logregvsLDAuni.pdf [Logistic regression vs. LDA for a linearly separable data set with
a very narrow margin. Logistic regression (center) always succeeds in separating linearly
separable classes, because the cost function approaches zero for a maximum margin classi-
fier. In this example, LDA (right) misclassifies some of the training points.]

� More robust on some non-Gaussian distributions (e.g., dists. w/large skew)
� Naturally fits labels between 0 and 1 [usually probabilities]

[When you use logistic regression with added quadratic features, you get a quadric decision boundary, just
as you do with QDA. Based on what I�ve said here, do you think logistic regression with quadratic features
gives you exactly the same classifier as QDA?]

0.00.51.01.52.00.00.51.01.52.0x10.00.51.01.52.00.00.51.01.52.0x1x20.00.51.01.52.00.00.51.01.52.0x1x264

Jonathan Richard Shewchuk

ROC CURVES (for test sets)

false negative rate

false positive rate

specificity = true negative rate

true positive
rate = % of
+ve classified
as +ve, aka
sensitivity

sensitivity

always
classify
positive

random classifiers

always classify negative

false positive rate = % of ?ve classified as +ve

ROC.pdf

[This is a ROC curve. That stands for receiver operating characteristics, which is an awful name but we�re
stuck with it for historical reasons.
A ROC curve is a way to evaluate your classifier after it is trained.
It is made by running a classifier on the test set or validation set.
It shows the rate of false positives vs. true positives for a range of settings.
We assume there is a knob we can turn to trade off false positives against false negatives. For our purposes,
that knob is the posterior probability threshold for Gaussian discriminant analysis or logistic regression.
However, neither axis of this plot is that knob.]
x-axis: �false positive rate = % of ?ve classified as +ve�
y-axis: �true positive rate = % of +ve classified as +ve aka sensitivity�
�false negative rate�: vertical distance from curve to top
�specificity�: horizontal distance from curve to right
[You generate this curve by trying every probability threshold; for each threshold, measure the false positive
& true positive rates and plot a point.]
upper right corner: �always classify +ve (Pr ? 0)�
lower left corner: �always classify ?ve (Pr > 1)�
diagonal: �random classifiers�
[A rough measure of a classifier�s effectiveness is the area under
the curve. For a classifier that is always correct, the area under
the curve is one. For the random classifier, the area under the
curve is 1/2, so you�d better do better than that.]
IMPORTANT: In practice, the trade-off between false negatives
and false positives is usually negotiated by choosing a point on
this plot, based on real test data, and NOT by taking the choice
of threshold that�s best in theory.

[1? false positive rate; �true negative rate�]

[1? sensitivity]

[Close up, ROC curves are made of horizontal and vertical line segments (see the figure at right), as the test
data is finite and there are only finitely many thresholds where some test point�s classification changes.]

ROC Curve0.00.20.40.60.81.00.00.20.40.60.81.0ROC Curve0.00.20.40.60.81.00.00.20.40.60.81.0Statistical Justifications; the Bias-Variance Decomposition

65

12 Statistical Justifications; the Bias-Variance Decomposition

STATISTICAL JUSTIFICATIONS FOR REGRESSION

[So far, I�ve talked about regression as a way to fit curves to points. Recall that early in the semester I divided
machine learning into 4 levels: the application, the model, the optimization problem, and the optimization
algorithm. My last two lectures about regression were at the bottom two levels: optimization. But why did
we pick these cost functions? Today, let�s take a step up to the second level, the model. I will describe
some models, how they lead to those optimization problems, and how they contribute to underfitting or
overfitting.]

Typical model of reality:

� sample points come from unknown prob. distribution: Xi ? D.
� y-values are sum of unknown, non-random fn + random noise:
D? has mean zero.

yi = g(Xi) + ?i,

?i ? D?,

?Xi,

[g = �ground truth.�]

[We are positing that reality is described by a �ground truth� function g. We don�t know g, but g is not
random; it represents a consistent relationship between X and y that we can estimate. We add to g a ran-
dom variable ?, which represents measurement errors and all the other sources of statistical error when we
measure real-world phenomena. Notice that the noise is independent of X. That�s a pretty questionable
assumption, and often it does not apply in practice, but that�s all we�ll have time to deal with this semester.
Also notice that this model leaves out systematic errors, like when your measuring device adds one to every
measurement, because we usually can�t diagnose systematic errors from data alone.]

Goal of regression: find h that estimates g.
Ideal approach: choose h(x) = EY [Y|X = x]
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)

= g(x) + E[?] = g(x).

[If this expectation exists at all, it partly justifies our model of reality. We can retroactively define g to be
this expectation.]

Least-Squares Cost Function from Maximum Likelihood

Suppose ?i ? N(0, ?2); then yi ? N(g(Xi), ?2).
Recall that log of normal PDF is

ln f (yi) = ?

(yi ? �i)2
2?2

? constant

& log likelihood is

? �i = g(Xi)

?(g; X, y) = ln ( f (y1) f (y2) � � � f (yn)) = ln f (y1) + . . . + ln f (yn) = ?

(cid:88)

(yi ? g(Xi))2 ? constant.

Takeaway: Max likelihood on �parameter� g ? choose a g that minimizes

(yi ? g(Xi))2.

[We treat g as a �distribution parameter.� If the noise is normally distributed, maximum likelihood tells us
to estimate g by least-squares regression.]
[However, I�ve told you in previous lectures that least-squares is very sensitive to outliers. If the error is
truly normally distributed, that�s not a big deal, especially when you have a lot of sample points. But in
the real world, the distribution of outliers often isn�t normal. Outliers might come from wrongly measured
measurements, data entry errors, anomalous events, or just not having a normal distribution. When you have
a heavy-tailed distribution of noise, for example, least-squares isn�t a great choice.]

1
2?2
(cid:80)

66

Jonathan Richard Shewchuk

Empirical Risk

The risk for hypothesis h is expected loss R(h) = E[L] over all (X, Y) in some joint distribution.
Discriminative model: we don�t know X�s dist. D. How can we minimize risk?
[If we have a generative model, we can estimate the joint probability distribution for X and Y and derive the
expected loss. That�s what we did for Gaussian discriminant analysis. But today I�m assuming we don�t
have a generative model, so we don�t know those probabilities. Instead, we approximate the distribution in
a very crude way: we pretend that the sample points are the distribution.]

Empirical distribution: the discrete uniform distribution over the sample pts

Empirical risk: expected loss under empirical distribution

�R(h) = 1
n

n(cid:88)

i=1

L(h(Xi), yi)

[The hat on the R indicates that �R is only a cheap approximation of the true, unknown statistical risk R
we really want to minimize. Often, this is the best we can do. For many but not all distributions, the
empirical risk converges to the true risk in the limit as n ? ?. Choosing h that minimizes �R is called
empirical risk minimization.]

Takeaway: this is why we [usually] minimize the average of the loss fns.

Logistic Loss from Maximum Likelihood

What cost fn should we use for probabilities?

Actual probability pt Xi is in class C is yi; predicted prob. is h(Xi).
Imagine ? duplicate copies of Xi, with yi ? in class C, and (1 ? yi) ? not.
[The size of ? isn�t very important, but imagine that yi ? and (1 ? yi) ? are both integers for all i. If yi is
irrational, approximate it with a very close rational number.]

[Let�s use maximum likelihood estimation to choose the hypothesis most likely to generate these labels for
these sample points. The following likelihood is the probability of generating these labels in a particular
fixed order.]

Likelihood is L(h; X, y) =

n(cid:89)

i=1

h(Xi)yi ?(1 ? h(Xi))(1?yi) ?

Log likelihood is ?(h) = ln L(h)
?
?
?????
?????yi ln h(Xi) + (1 ? yi) ln(1 ? h(Xi))
i
(cid:88)

= ?

(cid:88)

= ??

logistic loss fn L(h(Xi), yi).

Takeaway: Max likelihood ? minimize
logistic losses.
[So the principle of maximum likelihood explains where the weird logistic loss function comes from.]

(cid:80)

Statistical Justifications; the Bias-Variance Decomposition

67

THE BIAS-VARIANCE DECOMPOSITION

There are 2 sources of error in a hypothesis h:
bias:

error due to inability of hypothesis h to fit g perfectly
e.g., fitting quadratic g with a linear h
error due to fitting random noise in data
e.g., we fit linear g with a linear h, yet h (cid:44) g.

variance:

g

h

x

g

h

x

biasvar.pdf [Draw this figure by hand. At left, the error due to bias: a linear hypothesis h
just can�t fit a degree-2 ground truth g well. At right, the error due to variance: although
h could fit g perfectly, the noise in the data misleads it.]

Model: Xi ? D, ?i ? D?, yi = g(Xi) + ?i
fit hypothesis h to X, y

[remember that D? has mean zero]

Now h is a random variable; i.e., its weights are random
Consider arbitrary pt z ? Rd (not necessarily a sample pt!) & ? = g(z) + ?,
[So z is arbitrary, whereas ? is random.]
Note: E[?] = g(z); Var(?) = Var(?)
Risk fn when loss = squared error:

? ? D?

[the mean comes from g, and the variance comes from ?]

R(h) = E[L(h(z), ?)]

? take expectation over possible training sets X, y & values of ?

[Stop and take a close look at this expectation. Remember that the hypothesis h is a random variable. We are
taking a mean over the probability distribution of hypotheses. That seems pretty weird if you�ve never seen
it before. But remember, the training data X and y come from a joint probability distribution. We use the
training data to choose weights, so the weights that define h also come from some probability distribution.
It might be hard to work out what that distribution is, but it exists. This �E[�]� is integrating the loss over all
possible values of the weights.]

= E[(h(z) ? ?)2]
= E[h(z)2] + E[?2] ? 2 E[? h(z)]
= Var(h(z)) + E[h(z)]2 + Var(?) + E[?]2 ? 2E[?] E[h(z)]
= (E[h(z)] ? E[?])2 + Var(h(z)) + Var(?)
+
= (E[h(z)] ? g(z))2
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
bias2 of method

Var(h(z))
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
variance of method

Var(?)
(cid:124)(cid:123)(cid:122)(cid:125)
irreducible error

+

[Observe that ? and h(z) are independent]

[This is called the bias-variance decomposition of the risk function. Let�s look at an intuitive interpretation
of these three parts.]

68

Jonathan Richard Shewchuk

bvn.pdf [In this example, we�re trying to fit a sine wave with lines, which obviously aren�t
going to be accurate. At left, we have generated 50 different hypotheses (lines). Each line
was generated from 20 random training points by least-squares linear regression. At up-
per right, the black curve is the ground truth function g, and the red line is the expected
hypothesis�an average over infinitely many hypotheses. We see that at most points in fea-
ture space, the bias (difference between the black and red curves) is large, because lines
don�t fit sine waves well. However, the bias is small at some points�where the sine wave
crosses the red line. At center right, the variance is the expected squared difference between
a random hypothesis (black line) and the expected hypothesis (red line) at an arbitrary point
in feature space. At lower right, the irreducible error is the expected squared difference be-
tween a random test point�s noisy label and the sine wave. The irreducible error is optional;
it only makes sense to talk about it if we can make real-world measurements at test points.]

This is pointwise version [of the bias-variance decomposition.]
Mean version: let z ? D be random variable; take mean over D of bias2, variance.

[So you can decompose one test point�s error into these three numbers, or you can decompose the error of
the hypothesis over its entire range into three numbers, which tells you roughly how big they�ll be on a large
test set.]

!"Bias, Variance, Noise!"#$%#&"#�()*+"$),-./0"1$/23./)4#567)$/)#(89Statistical Justifications; the Bias-Variance Decomposition

69

[Now I will write down a list of consequences of what we�ve just learned.]

� Underfitting = too much bias
� Most overfitting caused by too much variance
� Training error reflects bias but not much variance; test error reflects both
[which is why low training error can fool you when you�ve overfitted]

� For many distributions, variance ? 0 as n ? ?
� If h can fit g exactly, for many distributions bias ? 0 as n ? ?
� If h cannot fit g well, bias is large at �most� points
� Adding a good feature reduces bias; adding a bad feature rarely increases it
� Adding a feature usually increases variance
� Can�t reduce irreducible error [hence its name]
� Noise in test set affects only Var(?);

noise in training set affects only bias & Var(h)

[don�t add a feature unless it reduces bias more]

� We can�t precisely measure bias or variance of real-world data

[because we cannot know g exactly and our noise model might be wrong]

� But we can test learning algs by choosing g & making synthetic data

[At left, a data set is fit with
splinefit.pdf, biasvarspline.pdf (ISL, Figures 2.9 and 2.12)
splines having various degrees of freedom. The synthetic data is taken from the black curve
with added noise. At center, we plot training error (gray) and test error (red) as a function
of the number of degrees of freedom. At right, we plot the squared test error as a sum of
squared bias (blue) and variance (orange). As the number of degrees of freedom increases,
the training and test errors both decrease up to 6 degrees because the bias decreases, but for
more degrees of freedom the test error increases because the variance increases.]

[The bias-variance decomposition is sometimes called the �bias-variance trade-off,� because sometimes you
see a U-shaped curve like this, for some hyperparameter like polynomial degree or C in SVMs. But that�s
misleading; it�s not really a trade-off. Sometimes both bias and variance are very high; sometimes both are
very low. If you try to fit 30 periods of a sine wave with a degree-10 polynomial fit to 15 points sampled
from the sine wave, both your bias and variance will be very high. Then you�re underfitting and overfitting at
the same time! They�re not opposites. There are neural networks for image classification with 99% accuracy
on test sets, so bias and variance are both admirably low. Always we seek models that can fit the ground
truth well, but aren�t easily perturbed by noise in the data.]

020406080100246810122510200.00.51.01.52.02.5Mean Squared Error2510200.00.51.01.52.02.570

Jonathan Richard Shewchuk

Example: Least-Squares Linear Reg.

For simplicity, no fictitious dimension.
[This implies that our linear regression function has to be zero at the origin.]

Model:

(ground truth is linear)

g(z) = v?z
[So we could fit g perfectly with a linear h if not for the noise in the training set.]
Let e be noise n-vector, ei ? N(0, ?2)
Training labels: y = Xv + e
[X & y are the inputs to linear regression. We don�t know v or e.]

Lin. reg. computes weights

w = X

+

+

y = X

(Xv + e) = v + X

+
e.
(cid:124)(cid:123)(cid:122)(cid:125)
noise in weights

.

[We want w = v, but the noise in y becomes noise in w.]

BIAS is |E[h(z)] ? g(z)| = |E[w?z] ? v?z| = |z?E[w ? v]| = |z?E[X+e]| = 0.
[E[X+e] is zero because X+
Warning: This does not mean h(z) ? g(z) is always 0!

Sometimes +ve, sometimes ?ve, mean over training sets is 0.
[Those deviations from the mean are captured in the variance.]

and e are independent, and e�s Gaussian PDF is symmetric around zero.]

[When the bias is zero, a perfect fit is possible. But when a perfect fit is possible, not all learning methods
give you a bias of zero; here it�s a benefit of the squared error loss function. With a different noise or a
different loss function, we might have a nonzero bias even fitting a linear h to a linear g.]
VARIANCE is Var(h(z)) = Var(w?z) = Var(v?z + (X+e)?z) = Var(z?X+e)
[This is the dot product of a vector z?X+
reduces it to a one-dimensional Gaussian along the direction z?X+
the 1D Gaussian times the squared length of the vector z?X+

with an isotropic, normally distributed vector e. The dot product
, so this variance is just the variance of

.]

= ?2 (cid:13)(cid:13)(cid:13)z?X
= ?2z?(X?X)?1z.

+(cid:13)(cid:13)(cid:13)

2 = ?2z?(X?X)?1X?X(X?X)?1z

If we choose coordinate system so D has mean zero, then X?X ? n Var(D) as n ? ?, so for z ? D,

Var(h(z)) ? ?2 d
n

.

[Where d is the dimension�the number of features per sample point.]
[Why? With the eigendecomposition Var(D) = V?V ?, we have E[z?Var(D)?1z] = E[???1/2V ?z?2] =
(cid:80)d
i=1 E[(vi � z)2]/?i. But as z ? D, E[(vi � z)2] = Var[vi � z] = ?i, so E[z?Var(D)?1z] = d. Approximating the
covariance Var(D) with the sample covariance matrix gives E[z?(X?X)?1z] ? d/n.]

Takeaways: Bias can be zero when hypothesis function can fit the real one!
[This is a nice property of the squared error loss function.]

Variance portion of RSS (overfitting) decreases as 1/n (sample points),

increases as d (features)
or O(d p) if you use degree-p polynomials.

Shrinkage: Ridge Regression, Subset Selection, and Lasso

71

13 Shrinkage: Ridge Regression, Subset Selection, and Lasso

RIDGE REGRESSION aka Tikhonov Regularization

Least-squares linear regression + ?2 penalized mean loss. (1) + (A) + (a) + (d).

Find w that minimizes ?Xw ? y?2 + ? ?w??2 = J(w)

where w? is w with component ? replaced by 0.
X has fictitious dimension but we DON�T penalize ?.
Adds a regularization term, aka a penalty term, for shrinkage: to encourage small ?w??. Why?

(1) Guarantees positive definite normal eq�ns; always unique solution.
[Standard least-squares linear regression yields singular normal equations when the sample points lie on a
common hyperplane in feature space�for example, when d > n.]

lslrcontour.pdf, lslr.pdf, ridge.pdf, ridgecontour.pdf [The cost function J(w) without and
with regularization. This plot ignores the dimension of the bias term ?.]

[At left, we see a cost function for least-squares regression, a positive semidefinite quadratic form. This cost
function has many minima, and the regression problem is said to be ill-posed. By adding a small penalty
term, we obtain a positive definite quadratic form (right), with one unique minimum. �Regularization�
implies that we are turning an ill-posed problem into a well-posed problem.]

[That was the original motivation, but the next has become more important in machine learning . . . ]

(2) Reduces overfitting by reducing variance. Why?

Example:

Input X1 = (0, 0) with label 0; X2 = (1, 1) with label 0; X3 = (0.51, 0.49) with label 1.
Linear regr. gives 50x1 ? 50x2. [This linear function fits all three points exactly.]
Big weights!

[Weights this big would be justified if there were big differences between
labels, or if there were small distances between points, but neither is true.
Large weights imply that tiny changes in x can cause huge changes in y.
Consider that the labels don�t differ by more than 1 and the points are
separated by distances greater than 0.7. So these disproportionately large
weights are a sure sign of overfitting.]
So we penalize large weights.
[This use of regularization is closely related to the first one. When you have
large variance and a lot of overfitting, it implies that your problem is close
to being ill-posed, even though technically it might be well-posed.]

-4-2024-4-2024-4-2024-4-20240.20.40.60.81.0x10.20.40.60.81.0x272

Jonathan Richard Shewchuk

least-squares solution

w2

�w

ridge solution for
several values of ?

isocontours of ?Xw ? y?2

w1

isocontours of ?w?2

ridgeterms2.pdf (redrawing of ISL, Figure 6.7) [In this plot of weight space, for simplic-
ity, we�re not using a bias term ? (we set it to zero). �w is the least-squares solution. The red
ellipses are isocontours of ?Xw ? y?2. The blue circles are isocontours of ?w?2, centered at
the origin. The ridge regression solution lies where a red isocontour just touches a blue iso-
contour tangentially. As ? increases, the solution will occur at a more outer red isocontour
and a more inner blue isocontour. This shrinks w and helps to reduce overfitting.]

Setting ?J = 0 gives normal eq�ns

(X?X + ?I?) w = X?y

[Don�t penalize the bias term ?.]

where I? is identity matrix w/bottom right set to zero.
[Don�t worry; X?X + ?I? is always positive definite for ? > 0, assuming X ends with a column of 1�s.]
Algorithm: Solve for w. Return h(z) = w?z.
Increasing ? ? more regularization; smaller ?w??
Recall [from the previous lecture] our data model y = Xv + e, where e is noise.
Variance of ridge regr. at test pt z is Var(z?(X?X + ?I?)?1X?e).
As ? ? ?, variance ? 0, but bias increases.

test error

variance

bias2 = (E[h] ? g)2

h ? 0 as ? ? ?

ridgebiasvar.pdf (ISL, Figure 6.5) [Plot of bias2 & variance as ? increases.]

[The test error as a function of ? is a U-shaped curve. We find the bottom by validation. Regularization is
intended to reduce the variance, but this method of regularization also increases the bias.]

? is a hyperparameter; tune by (cross-)validation.

Ideally, features should be �normalized� to have same variance.
Alternative: use asymmetric penalty by replacing I? w/other diagonal matrix. [For example, if you use
polynomial features, you could use different penalties for monomials of different degrees.]

Mean Squared Error1e?011e+011e+030102030405060?Shrinkage: Ridge Regression, Subset Selection, and Lasso

73

Bayesian Justification for Ridge Reg.

Assign a prior probability on w?: w? ? N(0, ?2), with PDF f (w?) ? e??w??2/(2?2)
[This prior probability says that we think weights close to zero are more likely to be correct.]
Apply MLE to maximize the posterior prob.

Bayes� Theorem: posterior fW|X,Y (w) =

fY|X,W(y) f (w?)
fY|X(y)

Maximize log posterior

= ln fY|X,W(y) + ln f (w?) ? const
= ?const ?Xw ? y?2 ? const ?w??2 ? const
? Minimize ?Xw ? y?2 + ? ?w??2

[We are treating w and y as random variables, but X as a fixed constant�it�s not random.]
This method (using MLE, but maximizing posterior) is called maximum a posteriori (MAP).
[A prior probability on the weights is another way to understand regularizing ill-posed problems.]

FEATURE SUBSET SELECTION

[Some of you may have noticed as early as Homework 1 that you can sometimes get better performance on
a spam classifier simply by dropping some useless features.]

All features increase variance, but not all features reduce bias.
Idea:

Identify poorly predictive features, ignore them (weight zero).
Less overfitting, smaller test errors.
2nd motivation: Inference. Simpler models convey interpretable wisdom.

Useful in all classification & regression methods.
Sometimes it�s hard: Different features can partly encode same information.

Combinatorially hard to choose best feature subset.

Alg: Best subset selection. Try all 2d ? 1 nonempty subsets of features. [Train one classifier per subset.]

Choose best classifier by (cross-)validation. Slow.

[Obviously, best subset selection isn�t feasible if we have a lot of features. But it gives us an �ideal�
algorithm to compare practical algorithms with. If d is large, there is no algorithm that�s guaranteed to find
the best subset and that runs in acceptable time. But heuristics often work well.]

Heuristic 1: Forward stepwise selection.
Start with null model (0 features); repeatedly add best feature until validation errors start increasing (due to
overfitting) instead of decreasing. At each outer iteration, inner loop tries every feature & chooses the best
by validation. Requires training O(d2) models instead of O(2d).
Not perfect:

e.g., won�t find the best 2-feature model if neither of those
features yields the best 1-feature model.

[That�s why it�s a heuristic.]

Heuristic 2: Backward stepwise selection.
Start with all d features; repeatedly remove feature whose removal gives best reduction in validation error.
Also trains O(d2) models.

[Forward stepwise is a better choice when you suspect only a few features will be good predictors; e.g.,
spam. Backward stepwise is better when most features are important. If you�re lucky, you�ll stop early.]

74

Jonathan Richard Shewchuk

LASSO (Robert Tibshirani, 1996)

Least-squares linear regression + ?1 penalized mean loss. (1) + (A) + (a) + (e).
�Least absolute shrinkage and selection operator.�
[This is a regularized regression method similar to ridge regression, but it has the advantage that it often
naturally sets some of the weights to zero.]

Find w that minimizes ?Xw ? y?2 + ? ?w??1 where ?w??1 = (cid:80)d
i=1

|wi|. (Don�t penalize ?.)

Recall ridge regr.: isosurfaces of ?w??2 are hyperspheres.
The isosurfaces of ?w??1 are cross-polytopes.
The unit cross-polytope is the convex hull of all the positive & negative unit coordinate vectors.

?w?1

= 1

[Draw this figure by hand crosspolys.pdf ]

[You get larger and smaller cross-polytope isosurfaces by scaling these.]

w2

�w

w2

�w

isocontours of ?Xw ? y?2

isocontours of ?Xw ? y?2

w1

isocontours of ?w?1

w1

isocontours of ?w?2

lassoridge2.pdf [Isocontours of the terms of the objective function for the Lasso appear at
left. Compare with the ridge regression isocontours at right.]

[The red ellipses are the isocontours of ?Xw ? y?2, and the least-squares solution lies at their center. The
isocontours of ?w??1 are diamonds centered at the origin (blue). The solution lies where a red isocontour
just touches a blue diamond. What�s interesting is that for large values of ?, the red isocontour touches just
the tip of a diamond. Then the weight w1 gets set to zero. That�s what we want to happen to features that
don�t have enough predictive power. For small values of ?, the red isosurface just barely touches a side of a
diamond instead of a tip of the diamond, and no weight gets set to zero.]

[When you go to higher dimensions, you might have several weights set to zero. For example, in 3D, if the
red isosurface just touches a sharp vertex of a cross-polytope, two of the three weights get set to zero. If it
just touches a sharp edge of a cross-polytope, one weight gets set to zero. If it just touches a flat side of a
cross-polytope, no weight is zero.]

Shrinkage: Ridge Regression, Subset Selection, and Lasso

75

lassoweights.pdf (ISL, Figure 6.6) [Weights as a function of ?.]

[This shows the weights for a typical linear regression problem with about 10 variables. You can see that as
lambda increases, more and more of the weights become zero. Only four of the weights are really useful for
prediction; they�re in color. Statisticians used to choose ? by looking at a chart like this and trying to eyeball
a spot where there aren�t too many predictors and the weights aren�t changing too fast. But nowadays they
prefer validation.]

Sometimes sets some weights to zero, especially for large ?.
Algs: subgradient descent, least-angle regression (LARS), forward stagewise

[Lasso can be reformulated as a quadratic program, but it�s a quadratic program with 2d constraints, because
a d-dimensional cross-polytope has 2d facets. In practice, special-purpose optimization methods have been
developed for Lasso. I�m not going to teach you one, but if you need one, look up the last two of these
algorithms. LARS is built into the R Programming Language for statistics.]

[As with ridge regression, you should probably normalize the features first before applying Lasso.]

Standardized Coefficients205010020050020005000?2000100200300400?76

Jonathan Richard Shewchuk

14 Decision Trees

DECISION TREES

Nonlinear method for classification and regression.

Uses tree with 2 node types:

� internal nodes test feature values (usually just one) & branch accordingly
� leaf nodes specify class h(x)

Outlook (x1)

sunny

overcast

rain

Humidity (x2)

yes

Wind (x3)

> 75%

? 75%

> 20

? 20

no

yes

no

yes

100
x2

75

50

25

0

no

yes

yes

check
x3

sunny

rain

overcast
x1

[Draw this by hand. dectree.pdf Deciding whether to go out for a picnic.]

� Cuts x-space into rectangular cells
� Works well with both categorical and quantitative features
� Interpretable result (inference)
� Decision boundary can be arbitrarily complicated

linear classifer

decision tree

treelinearcompare2.pdf (redrawing of ISL, Figure 8.7)
vs. decision trees on 2 examples.]

[Comparison of linear classifiers

Decision Trees

77

Consider classification first. Greedy, top-down learning heuristic:
[This algorithm is more or less obvious, and has been rediscovered many times. It�s naturally recursive. I�ll
show how it works for classification first; later I�ll talk about how it works for regression.]
Let X be n � d design matrix; y ? Rn be labels.
Let S ? {1, 2, . . . , n} be set of sample point indices.
Top-level call: S = {1, 2, . . . , n}.

GrowTree(S )

if (yi = C for all i ? S and some class C) then {

return new leaf(C)

[We say the leaves are pure]

} else {

choose best splitting feature j and splitting value ?
S l = {i ? S : Xi j < ?}
S r = {i ? S : Xi j ? ?}
return new node( j, ?, GrowTree(S l), GrowTree(S r))

[Or you could use ? and >]

(*)

}

(*) How to choose best split?

[All features, and all splits within a feature.]

� Try all splits.
� For a set S , let J(S ) be the cost of S .
� Choose the split that minimizes J(S l) + J(S r); or,
the split that minimizes weighted average

|S l|J(S l) + |S r|J(S r)
|S l| + |S r|

.

[Here, I�m using the vertical bars | � | to denote set cardinality.]

How to choose cost J(S )?
[I�m going to start by suggesting a mediocre cost function, so you can see why it�s mediocre.]

Idea 1 (bad): Label S with the class C that labels the most points in S .
J(S ) ? # of points in S not in class C.

20 C

10 D

J(S ) = 10

x1

20 C

10 D

x2

10 C

10 D

10 C

0 D

10 C

5 D

10 C

5 D

J(S l) = 10

J(S r) = 0

J(S l) = 5

J(S r) = 5

[Draw this by hand. badcost.pdf ]
Problem: J(S l) + J(S r) = 10 for both splits, but left split is much better. Weighted avg prefers right split!
[There are many different splits that all have the same total cost. We want a cost function that better distin-
guishes between them.]

78

Jonathan Richard Shewchuk

Idea 2 (good): Measure the entropy.
Let Y be a random class variable, and suppose P(Y = C) = pC.
The surprise of Y being class C is ? log2 pC.
� event w/prob. 1 gives us zero surprise.
� event w/prob. 0 gives us infinite surprise!

[An idea from information theory.]

[Always nonnegative.]

[In information theory, the surprise is equal to the expected number of bits of information we need to
transmit which events happened to a recipient who knows the probabilities of the events. Often this means
using fractional bits, which may sound crazy, but it makes sense when you�re compiling lots of events into
a single message; e.g., a sequence of biased coin flips.]

The entropy of an index set S is the average surprise [when you draw a point at random from S ],

H(S ) = ?

(cid:88)

C

pC log2 pC,

where pC =

|{i ? S : yi = C}|
|S |

.

[The proportion of points in S
that are in class C.]

If all points in S belong to same class? H(S ) = ?1 log2 1 = 0.
Half class C, half class D? H(S ) = ?0.5 log2 0.5 ? 0.5 log2 0.5 = 1.
n points, all different classes? H(S ) = ? log2
[The entropy is the expected number of bits of information we need to transmit to identify the class of a
sample point in S chosen uniformly at random. It makes sense that it takes 1 bit to specify C or D when
each class is equally likely. And it makes sense that it takes log2 n bits to specify one of n classes when each
class is equally likely.]

= log2 n.

1
n

entropy.pdf [Left: plot of the entropy H(pC) when there are only two classes. The proba-
bility of the second class is pD = 1 ? pC, so we can plot the entropy with just one dependent
variable. Right: plot of the entropy H(pC, pD) when there are three classes. The probability
of the third class is pE = 1 ? pC ? pD. Observe that the entropy is strictly concave.]

0.00.20.40.60.81.0pC0.20.40.60.81.0HDecision Trees

79

Weighted avg entropy after split is Hafter =

|S l| H(S l) + |S r| H(S r)
|S l| + |S r|

.

Choose split that maximizes information gain H(S ) ? Hafter.

[Which is just the same as minimizing Hafter.]

? 10

30 lg 10

30

(cid:17) 0.918

20 C

10 D

x3

H(S ) = ? 20

30 lg 20

30

10 C

9 D

10 C

1 D

H(S l) = ? 10

19 lg 10

19

? 9

19 lg 9

11 lg 10

11

? 1

11 lg 1

11

(cid:17) 0.439

19

(cid:17) 0.998
Hafter = 0.793
[Draw this by hand.

H(S r) = ? 10
info gain = 0.125
infogain.pdf ]

Info gain always positive except it is zero when one child is empty or
for all C, P(yi = C|i ? S l) = P(yi = C|i ? S r).

[Which is the case for the second split we considered.]

[Recall the graph of the entropy.]

H(pC)

entropy: strictly concave

J(pC) = % misclassified: concave, not strict

1

0.5

0

0

0.2

H(S l)

Hafter

H(S )

info gain

50%

J(S l)

J(S ) = Jafter

H(S r)

pC

0.4

0.6

0.6
[Draw this by hand on entropy.pdf. concave.pdf ]

0.4

0.2

0.8

1

0

0%

J(S r)

pC

0.8

1

[Suppose we pick two points on the entropy curve, then draw a line segment connecting them. Because the
entropy curve is strictly concave, the interior of the line segment is strictly below the curve. Any point on
that segment represents a weighted average of the two entropies for suitable weights. If you unite the two
sets into one parent set, the parent set�s value pC is the weighted average of the children�s pC�s. Therefore,
the point directly above that point on the curve represents the parent�s entropy. The information gain is
the vertical distance between them. So the information gain is positive unless the two child sets both have
exactly the same pC and lie at the same point on the curve.]

[On the other hand, for the graph on the right, plotting the % misclassified, if we draw a line segment
connecting two points on the curve, the segment might lie entirely on the curve. In that case, uniting the two
child sets into one, or splitting the parent set into two, changes neither the total misclassified sample points
nor the weighted average of the % misclassified. The bigger problem, though, is that many different splits
will get the same weighted average cost; this test doesn�t distinguish the quality of different splits well.]

}80

Jonathan Richard Shewchuk

[By the way, the entropy is not the only function that works well. Many concave functions work fine,
including the simple polynomial p(1 ? p).]

More on choosing a split:

� For binary feature xi: children are xi = 0 & xi = 1.
� If xi has 3+ discrete values: split depends on application.

[Sometimes it makes sense to use multiway splits; sometimes binary splits.]

� If xi is quantitative: sort xi values in S ; try splitting between each pair of unequal consecutive values.

[We can radix sort the points in linear time, and if n is huge we should.]
Clever bit: As you scan sorted list from left to right, you can update entropy in O(1) time per point!6
[This is important for obtaining a fast tree-building time.]
[Draw a row of C�s and X�s; show how we update the # of C�s and # of X�s in each of S l and S r as we
scan from left to right.]

xi

scan.pdf

Algs & running times:

� Classify test point: Walk down tree until leaf. Return its label.

Worst-case time is O(tree depth).
For binary features, that�s ? d.
Usually (not always) ? O(log n).

[Quantitative features may go deeper.]

� Training: For binary features, try O(d) splits at each node.

try O(n?d) splits; n? = points in node
? O(n?d) time at this node

For quantitative features,
Either way
[Training on quantitative features is asymptotically just as fast as training on binary features because
of our clever way of computing the entropy for each split.]
Each point participates in O(depth) nodes, costs O(d) time in each node.
[This is an amortized analysis: we are charging O(d depth) time to each sample point.]
Running time ? O(nd depth).
[As nd is the size of the design matrix X, and the depth is often logarithmic, this is a surprisingly
reasonable running time.]

6Let C be the number of class C sample points to the left of a potential split and c be the number to the right of the split. Let
D be the number of class not-C points to the left of the split and d be the number to the right of the split. Update C, c, D, and d
at each split (in O(1) time per split) as you move from left to right. At each potential split, calculate the entropy of the left set as
? C
d
c+d . Note: log 0 is undefined, but this
formula works if we use the convention 0 log 0 = 0.

(cid:17)
, where n? =
It follows that the weighted average of the two entropies is ? 1
n?
C + D + c + d is the total number of sample points stored in this treenode. Choose the split that minimizes this weighted average.

C+D and the entropy of the right set as ? c

(cid:16)
C log2

+ D log2

+ d log2

+ c log2

C+D log2

C+D log2

c+d log2

c+d log2

? D

D
C+D

C
C+D

C
C+D

? d

c
c+d

c
c+d

d
c+d

D

CXXCX1X2X1X0C2C2X1C1C2X1X3X0X1C1C1C1CMore Decision Trees, Ensemble Learning, and Random Forests

81

15 More Decision Trees, Ensemble Learning, and Random Forests

DECISION TREE VARIATIONS

[Last lecture, I taught you the vanilla algorithm for building decision trees and using them to classify test
points. There are many variations on this basic algorithm; I�ll discuss a few now.]

Decision Tree Regression

Creates a piecewise constant regression fn. [This seems too rudimentary to be true, but it�s true.]

x1 < 3

x1 ? 3

x2 < 3 x2 ? 3

x2 < 2 x2 ? 2

x1 < 1

x1 ? 1

x2 < 2

x2 ? 2

x2

h

x1

x2

x1

treeregresstree.pdf [Decision tree regression.]

(cid:88)

yi, the mean label for training pts i ? S .

Leaf stores label �S = 1
|S |

i?S

Cost J(S ) = Var({yi : i ? S }) = 1
|S |

(cid:88)

i?S

(yi ? �S )2.

[So if all the points in a leaf have the same y-value, then the cost is zero.]
[We choose the split that minimizes the weighted average of the variances of the two children after the split.]

Stopping Early

[The basic version of the decision tree algorithm keeps subdividing treenodes until every leaf is pure. We
don�t have to do that; sometimes we prefer to stop subdividing treenodes earlier.]

Why?

� Limit tree depth (for speed)
� Limit tree size (big data sets)
� Pure tree may overfit
� Given noise or overlapping distributions, pure leaves tend to overfit; better to stop early and estimate

posterior probs

82

Jonathan Richard Shewchuk

[When you have strongly overlapping class distributions, refining the tree down to one training point per leaf
is absolutely guaranteed to overfit, giving you a classifier akin to the 1-nearest neighbor classifier. It�s better
to stop early, then classify each leaf node by taking a vote of its training points; this gives you a classifier
akin to a k-nearest neighbor classifier.]

treeoverfit.pdf [Overlapping distributions cause pure decision trees to overfit. Compare
this decision tree with the Bayes decision rule; the Bayes optimal decision boundary would
be just one point.]

[Alternatively, you can use the points to estimate a posterior probability for each leaf, and return that. If
there are many points in each leaf, the posterior probabilities might be reasonably accurate.]

x2

13
20

|

)
X
Y
(
p

4
20

2
20

1
20

6
10

|

)
X
Y
(
p

0
10

2
10

2
10

x1

leaf2.pdf [In the decision tree at left, each leaf has multiple classes. Instead of returning
the majority class, each leaf could return a posterior probability histogram.]

Leaves with multiple points return

� a majority vote or class posterior probs (classification) or
� an average (regression).

How to stop? Select stopping condition(s):

� Next split doesn�t reduce entropy/error enough (dangerous; pruning is better)
� Most of node�s points (e.g., > 90%) have same class
� Node contains few training points (e.g., < 10)
� Box�s edges are all tiny
� Depth too great
� Use validation to compare

[super-fine resolution may be pointless]
[risky if there are still many training points in the box]

[especially for big data]

[to deal with outliers & overlapping distribs]

[The last is the slowest but most effective way to know when to stop: use validation to decide whether
splitting the node lowers your validation error. But if your goal is to avoid overfitting, it�s generally even
more effective to grow the tree a little too large and then use validation to prune it back . . . ]

More Decision Trees, Ensemble Learning, and Random Forests

83

Pruning

Grow tree too large; greedily remove each split whose removal improves validation performance.
[We have to do validation once for each split that we�re considering reversing.]
More reliable than stopping early.

[The reason why pruning often works better than stopping early is because often a split that doesn�t seem to
make much progress is followed by a split that makes a lot of progress. If you stop early, you�ll never find
out. Pruning is simple and highly recommended when you have enough time to build and prune the tree.]

[At left, a decision tree pre-
prunedhitters.pdf, prunehitters.pdf (ISL, Figures 8.5 & 8.2)
dicting the salaries of baseball players from years in Major League Baseball and hitting
average: R1 = $165,174, R2 = $402,834, R3 = $845,346. At right, a plot of decision tree
leaf nodes vs. mean squared error. The graph shows that the decision tree with the best
validation accuracy has three leaves, so that tree appears at left.]

[In this example, a 10-leaf decision tree was constructed to predict the salaries of baseball players, based
on their years in the league and average hits per season. Then the tree was pruned by validation. The best
decision tree on the validation data turned out to have just three leaves.]
[Observe that this tree is very interpretable. You could easily explain it to grandpa.]

[It might seem expensive to do validation once for each split we consider reversing. But you can do it pretty
cheaply. What you don�t do is reclassify every validation point from scratch. Instead, you first compute
which leaf each validation point winds up in, then for each leaf you make a list of its validation points.
When you are deciding whether to remove a split, you just look at the validation points in the two leaves
you�re thinking of removing, and see how they will be reclassified and how that will change the error rate.
You can compute this very quickly.]

pruning

validation points

prunecheck.pdf
[After we determine
which leaf boxes each validation point
ends up in, we find that pruning these two
leaves improves the validation accuracy.
Box colors indicate the majority classes
of the training points (not shown). Local
validation accuracy improves from 7/16
to 9/16.]

YearsHits1117.523814.524R1R3R22468100.00.20.40.60.81.0Tree SizeMean Squared ErrorTrainingCross?ValidationTestXXXXCXXXXCCCCCCCXXXXXXXXXCCCCCCX84

Jonathan Richard Shewchuk

Multivariate Splits

Find non-axis-aligned splits with other classification algs or by generating them randomly.

x2 < 6 x2 ? 6

x1 < 8 x1 ? 8

x1 < 13

x1 ? 13

x2 < 3

x2 ? 3

x2 < 7

x2 ? 7

x2 < 8

x2 ? 8

x1 < 5

x1 ? 5

x1 < 10

x1 ? 10

x2 < 5

x2 ? 5

multivar.pdf [An example where an ordinary decision tree needs many splits to approxi-
mate a diagonal linear decision boundary, but a single multivariate split takes care of it.]

[Here you can use other classification algorithms such as SVMs, logistic regression, and Gaussian discrimi-
nant analysis. Decision trees permit these algorithms to find nonlinear decision boundaries by making them
hierarchical.]

May gain better classifier at cost of worse interpretability or speed.
[Standard decision trees are very fast because they check only one feature at each treenode. But if there are
hundreds of features, and you have to check all of them at every level of the tree to classify a point, it slows
down classification a lot.]
[A good compromise is to set a limit on the number of features you check at each treenode�say, three. You
can use forward stepwise selection at each treenode to choose the three features.]
[On exams, assume we check only one feature per treenode unless we say otherwise!]

ENSEMBLE LEARNING

Decision trees are fast, simple, interpretable, easy to explain,
invariant under scaling/translation, robust to irrelevant features.

But not the best at prediction. [Compared to previous methods we�ve seen.]
High variance. [Though we can achieve very low bias.]

[For example, suppose we take a training data set, split it into two halves, and train two decision trees, one
on each half of the data. It�s not uncommon for the two trees to turn out very different. In particular, if the
two trees pick different features for the very first split at the root of the tree, then it�s quite common for the
trees to be completely different. So decision trees tend to have high variance.]

[So let�s think about how to fix this. As an analogy, imagine that you are generating random numbers
from some distribution. If you have just one random number, its variance might be high. But if you have
n random numbers and take their average, then the variance of that average is n times smaller. So you might
ask yourself, can we reduce the variance of decision trees by taking an average answer of a bunch of decision
trees? Yes we can.]

More Decision Trees, Ensemble Learning, and Random Forests

85

wisdom.jpg, penelope.jpg [James Surowiecki�s book �The Wisdom of Crowds� and Pene-
lope the cow. Surowiecki tells us this story . . . ]

[A 1906 county fair in Plymouth, England had a contest to guess the weight of an ox. A scientist named
Francis Galton was there, and he did an experiment. He calculated the median of everyone�s guesses. The
median guess was 1,207 pounds, and the true weight was 1,198 pounds, so the error was less than 1%. Even
the cattle experts present didn�t estimate it that accurately.]

[National Public Radio repeated the experiment in 2015 with a cow named Penelope whose photo they
published online. They got 17,000 guesses, and the average guess was 1,287 pounds. Penelope�s actual
weight was 1,355 pounds, so the crowd got it to within 5 percent.]

[The main idea is that sometimes the average opinion of a bunch of idiots is better than the opinion of one
expert. And so it is with learning algorithms. We call a learning algorithm a weak learner if it does better
than guessing randomly. And we combine a bunch of weak learners to get a strong one.]

We can take average of output of
� different learning algs
� same learning alg on many training sets
� bagging: same learning alg on many random subsamples of one training set
� random forests: randomized decision trees on random subsamples

[if we have tons of data]

[These last two are the most common ways to use averaging, because usually we don�t have enough training
data to use fresh data for every learner.]
Metalearner takes test point, feeds it into all T learners, returns majority class or average output.
[Averaging is not specific to decision trees; it can work with many different learning algorithms. But it
works particularly well with decision trees.]

Regression algs: take median or mean output [of all the weak learners]
Classification algs: take majority vote OR average posterior probs

[Apology to readers: I show some videos in this lecture, which cannot be included in this report.]

[Show averageaxis.mov] [Here�s a simple classifier that takes an average of �stumps,� trees of depth 1.
Observe how good the posterior probabilities look.]
[Show averageaxistree.mov] [Here�s a 4-class classifier with depth-2 trees.]

86

Jonathan Richard Shewchuk

[The Netflix Prize was an open competition for the best collaborative filtering algorithm to predict user
ratings for films, based on previous ratings. It ran for three years and ended in 2009 with a $1,000,000 prize.
The winning team, BellKor�s Pragmatic Chaos, used an extreme ensemble method that took an average of
many different learning algorithms. A couple of top teams combined into one team so they could combine
their methods. They said, �Let�s average our models and split the money,� and that�s what happened.]

Use learners with low bias (e.g., deep decision trees).
High variance & some overfitting are okay. Averaging reduces the variance!
[Each learner may overfit, but each overfits in its own unique way.]
Averaging sometimes reduces bias & increases flexibility a bit, but not reliably.

e.g., creating nonlinear decision boundary from linear classifiers.

[Averaging rarely reduces bias as much as it reduces variance, so get the bias small before averaging.]
Hyperparameter settings usually different than 1 learner. [Averaging reduces variance more than bias.]

[Sometimes the number of learners is said to be a hyperparameter, but extra learners improve the variance
without worsening the bias. The main limit on the number of learners is computation time. So you trade off
time for improved variance.]

Bagging = Bootstrap AGGregatING (Leo Breiman, 1994)

[Leo Breiman was a statistics professor right here at Berkeley. He did his best work after he retired in 1993.
The bagging algorithm was published the following year, and then he went on to co-invent random forests
as well. Unfortunately, he died in 2005.]

leobreiman3.png [Leo Breiman]
[Bagging is a randomized method for creating many different learners from the same data set. It works well
with many different learning algorithms. One exception seems to be k-nearest neighbors; bagging mildly
degrades it.]
Given n-point training sample, generate random subsample of size n? by sampling with replacement. Some
points chosen multiple times; some not chosen.

More Decision Trees, Ensemble Learning, and Random Forests

87

1 3 4 6 8 9

?
6 3 6 1 1 9

?
8 8 4 9 1 8
If n? = n, ? 63.2% are chosen. [On average; this fraction varies randomly.]
Build learner. Points chosen j times have greater weight:
[If a point is chosen j times, we want to treat it the same way we would treat j different points all bunched
up infinitesimally close together.]

� Decision trees: j-time point has j � weight in entropy.
� SVMs: j-time point incurs j � penalty to violate margin.
� Regression: j-time point incurs j � loss.

Build T learners from T subsamples.

Random Forests

Bagging + trees isn�t random enough!
[With bagging, often the decision trees look very similar. Why is that?]
One really strong predictor ? same feature split at top of every tree.
[For example, if you�re building decision trees to identify spam, the first split might always be �viagra.�
Random sampling might not change that.
If the trees are too similar, then taking their average doesn�t
reduce the variance much.]
Let�s reduce the correlation between different trees. [That makes averaging work better.]

Idea: At each treenode, take random sample of m features (out of d).

?

d works well for classification; m ? d/3 for regression.

Choose best split from m features.
[We�re not allowed to split on the other d ? m features!]
Different random sample for each treenode.
m ?
[So if you have a 100-dimensional feature space, you randomly choose 10 features and pick the one
of those 10 that gives the best split. But m is a hyperparameter, and you might get better results by
tuning it for your particular application. These values of m are good starting guesses.]
Smaller m ? more randomness, less tree correlation, more bias

[One reason this works is if there�s a really strong predictor, only a fraction of the trees can choose that pre-
dictor as the first split. That fraction is m/d. So the split tends to �decorrelate� the trees. And that means
that when you take the average of the trees, your average will have less variance than a single tree.]
[You have to be careful, though, because you don�t want to dumb down the trees too much in your quest for
decorrelation. Averaging works best when you have very strong learners that are also diverse. But it�s hard
to create a lot of learners that are very different yet all very smart. The Netflix Prize winners did it, but it
was a huge amount of work.]

Sometimes test error drops even at 100s or 1,000s of decision trees!
Disadvantages: slow; loses interpretability/inference.
[But the compensation is it�s a more accurate predictor than a single decision tree.]

[I will end by showing you examples of a very non-standard method for random forests that works magic in
certain difficult circumstances.]
Idea: generate s random multivariate splits (oblique lines, quadrics); choose best split.
[You have to be clever about how you generate random decision boundaries; I�m not going to discuss that.
I�ll just show lots of examples.]

88

Jonathan Richard Shewchuk

[Show treesidesdeep.mov] [Lots of good-enough conic random decision trees.]
[Show averageline.mov]
[Show averageconic.mov]
[Show square.mov] [Depth 2; look how good the posterior probabilities look.]
[Show squaresmall.mov] [Depth 2; see the uncertainty away from the center.]
[Show spiral2.mov] [Doesn�t look like a decision tree at all, does it?]
[Show overlapdepth14.mov] [Overlapping classes. This example overfits!]
[Show overlapdepth5.mov] [Better fit.]

500.pdf [Random forest classifiers for 4-class spiral data. Each forest takes the average of
400 trees. The top row uses trees of depth 4. The bottom row uses trees of depth 12. From
left to right, we have axis-aligned splits, splits with lines with arbitrary rotations, and splits
with conic sections. Each split is chosen to be the best of 500 random choices.]

randomness.pdf [Random forest classifiers for the same data. Each forest takes the average
of 400 trees. In these examples, all the splits are axis-aligned. The top row uses trees of
depth 4. The bottom row uses trees of depth 12. From left to right, we choose each split from
1, 5, or 50 random choices. The more choices, the less bias and the better the classifier.]

Neural Networks

89

16 Neural Networks

NEURAL NETWORKS

Can do both classification & regression.

[They tie together many ideas from the course: perceptrons, linear regression, logistic regression, ensembles
of learners, and stochastic gradient descent. They also tie in the idea of lifting sample points to a higher-
dimensional feature space, but with a new twist: neural nets can learn features themselves.]

[I want to begin by reminding you of the story I told you at the beginning of the semester, about Frank
Rosenblatt�s invention of perceptrons in 1957. Remember that he held a press conference where he predicted
that perceptrons would be �the embryo of an electronic computer that [the Navy] expects will be able to
walk, talk, see, write, reproduce itself and be conscious of its existence.�]

[Perceptron research continued until something unfortunate happened in 1969. Marvin Minsky, one of the
founding fathers of AI, and Seymour Papert published a book called �Perceptrons.� Sounds promising?
Well, part of the book was devoted to things perceptrons can�t do. One of those things is XOR.]

XOR

0

1

x2

0

0

1

x1

1

1

0

[Think of the four outputs here as training points in two-dimensional space. Two of them are in class 1, and
two of them are in class 0. We want to find a linear classifier that separates the 1�s from the 0�s. Can we do
it? No.]

[The XOR problem is also called parity, especially when you have more features: the input is a bunch of
bits and you answer whether the number of 1�s is even or odd. It was known even then that you could solve
parity problems by adding extra layers of perceptrons, but Minsky and Papert gave technical proofs about
some circumstances where this can�t be done, and those limitations were misinterpreted. The book had a
devastating effect on the field. After its publication, almost no research was done on neural net-like ideas for
a decade, a time we now call the first �AI Winter.� Shortly after the book was published, Frank Rosenblatt
died in a boating accident.]

[There are several almost obvious ways to get around the XOR problem. Here�s the easiest.]

If you add one new quadratic feature, x1x2, XOR is linearly separable in 3D.

1

0

1

0

[Draw this by hand. xorcube.pdf ]

[Now we can find a plane that cuts through the cube obliquely and separates the 0�s from the 1�s.]

90

Jonathan Richard Shewchuk

[However, there�s an even more powerful way to do XOR. The idea is to design linear classifiers whose
output is the input to other linear classifiers. That way, you should be able to emulate arbitrary logic circuits.
Suppose I put together some linear decision functions like this.]

x1

x2

linear combo

linear combo

linear combo

z

[Draw this by hand.

lincombo.pdf ]

[Interpret the output as true if z is greater than one-half or false if z is less than one-half. Can I do XOR with
this?]

A linear combo of linear combos is a linear combo . . . only works for linearly separable points.

[We need one more idea to make neural nets. We need to add some sort of nonlinearity between the linear
combinations. Let�s call these boxes that compute linear combinations �neurons.� If a neuron sends the
linear combination it computes through some nonlinear function before sending it on to other neurons, then
the neurons can act somewhat like logic gates. The nonlinearity could be as simple as clamping the output
so it can�t go below zero. That�s what people usually use in practice these days.]

[However, the traditional choice was to use the logistic function. The logistic function can�t go below zero
or above one, which is nice because it can�t ever get huge and oversaturate the other neurons it�s sending
information to. The logistic function is also smooth, which means it has well-defined gradients and Hessians
we can use for optimization. And we know that the logistic is often a good model for posterior probabilities.]

[With logistic functions between the linear combinations, here�s a two-layer perceptron that computes the
XOR function.]

x1

x2

s(30 ? 20x1 ? 20x2)

NAND

s(20x1 + 20x2 ? 10)

a

b

OR

s(20a + 20b ? 30)

x1 ? x2

AND

[Draw this by hand. xorgates.pdf ]

[The big question is: can an algorithm learn a function like this?]

Neural Networks

91

Network with 1 Hidden Layer

x1, . . . , xd ; xd+1 = 1
Input layer:
h1, . . . , hm ; hm+1 = 1
Hidden units:
�y1, . . . , �yk
Output layer:
Layer 1 weights: m � (d + 1) matrix V
k � (m + 1) matrix W
Layer 2 weights:

[Index d + 1 is the fictitious dimension.]

V ?
i
W?
i

is row i: weights into hi
is row i: weights into �yi

V11

V21

V33

x1

x2

1

h1

h2

h3

1

�y1

�y2

W12

W24

[Draw this by hand. neuralnetwork.pdf ]

Recall [logistic function] s(?) =

1

1 + e?? . Other nonlinear fns can be used, called the activation fns.

For vector u, s(u) =

?

???????????

?

???????????

s(u1)
s(u2)
...

, s1(u) =

?

????????????????

?

????????????????

s(u1)
s(u2)
...
1

[We apply s to a vector component-wise.]

h = s1(V x)
�y = s(Wh) = s(W s1(V x))

. . . that is, hi = s(Vi � x)

[Neural networks often have more than one output. This allows us to build multiple classifiers that share
hidden units. One of the interesting advantages of neural nets is that if you train multiple classifiers simul-
taneously, sometimes some of them come out better because they can take advantage of particularly useful
hidden units that first emerged to support one of the other classifiers.]

[We can add more hidden layers, and for image recognition tasks it�s common to have 6 to 200 hidden
layers. There are many variations you can experiment with�for instance, you can have connections that go
forward more than one layer.]

92

Training

Jonathan Richard Shewchuk

Usually stochastic or batch gradient descent.

Pick loss fn L(�y, y)
? ?

e.g., L(�y, y) = ?�y ? y?2.

predictions true labels (could be vectors)

Find V and W that minimize the cost fn J(V, W) = 1
n

n(cid:88)

i=1

L(�y(Xi), Yi).

[I�m using a capital Y here because Y is a matrix with one row for each training point and one column for
each output unit of the neural net. Each training point has a whole vector of labels Yi, stored as a row of Y.]

Usually there are many local minima!
[The cost function for a neural net is, generally, not even close to convex. Sometimes, it�s possible to wind
up in a bad minimum. Usually, you can avoid bad minima by having lots of units in each layer.]

[Now let me ask you this. Suppose we start by setting all the weights to zero, and then we do gradient
descent on the weights. What will go wrong?]
[This neural network has a symmetry: there�s really no difference between one hidden unit and any other
hidden unit. The gradient descent algorithm has no way to break the symmetry between hidden units. You
can get stuck in a situation where all the weights out of an input unit have the same value, and all the weights
into an output unit have the same value, and they have no way to become different from each other. To avoid
this problem, and in the hopes of finding a better local minimum, we start with random weights.]

Let w be a vector containing all the weights in V & W. Batch gradient descent:

w ? vector of random weights
repeat

w ? w ? ? ?J(w)

[We�ve just rewritten all the weights as a vector for notational convenience. When you actually write the
code, for the sake of speed, you should probably operate directly on the weight matrices V and W.]

[It�s important to make sure the random weights aren�t too big, because if a unit�s output gets too close to
zero or one, it can get �stuck,� meaning that a modest change in the input values causes barely any change
in the output value. Stuck units tend to stay stuck. I�ll say more about that next lecture.]

[Instead of batch gradient descent, we can use stochastic gradient descent, which means we use the gradient
of one training point�s loss function at each step. Typically, we shuffle the points in a random order, or just
pick one randomly at each step. I�ll say more about that next week.]

[The hard part of this algorithm is computing the gradient. If you simply derive one derivative for each
weight, you�ll find that for a network with many layers of hidden units, it takes time linear in the number of
edges in the neural network to compute a derivative for one weight. Multiply that by the number of weights.
We�ll spend the rest of this lecture learning to improve the running time to linear in the number of edges.]

Naive gradient computation: O(edges2) time
Backpropagation: O(edges) time

Neural Networks

93

Computing Gradients for Arithmetic Expressions

[Let�s see what it takes to compute the gradient of an arithmetic expression. It turns into repeated applica-
tions of the chain rule from calculus.]

+

d

? f
?d

= ? f
?e
?d
?e
= c ? f
?e

c

Goal: compute ? f =

?

?????????????

?

?????????????

? f
?a
? f
?b
? f
?c

�

e

? f
?e

= ? f
? f
? f
?e
= 2e ? f
? f

2e

e2

f

? f
? f

= 1

a

? f
?a

b

? f
?b

c

? f
?c

?d
?a

1

= ? f
?d
= ? f
?d

?d
?b

1

= ? f
?d
= ? f
?d

= ? f
?e
?c
?e
= d ? f
?e

d

d = a + b
= 1

?d
?b

?d
?a

= 1

e = cd

?e
?c

= d

?e
?d

= c

f = e2
? f
= 2e
?e

Each value z gives partial derivative of the form

where z is an input to n.

? f
?z

= ? f
?n

?n
?z

computed during forward pass

computed during backward pass after forward pass
�backpropagation�

[Draw this by hand. gradientsarith.pdf Draw the black diagram first. Then the goal (upper
right). Then the green and red expressions, from left to right, leaving out the green arrows.
Then the green arrows, starting at the right side of the page and moving left. Lastly, write
the text at the bottom. (Use the same procedure for the next two figures.)]

94

Jonathan Richard Shewchuk

[What if a unit�s output goes to more than one unit? Then we need to understand a more complicated version
of the chain rule. This is a standard rule of multivariate calculus:]

?
??

L(y1(?), y2(?)) = ?L
?y1

?y1
??

+ ?L
?y2

?y2
??

= ?yL �

?
??

y

[With this rule, let�s compute gradients for an expression from least-squares linear regression.]

w1
?L
?w1

w2
?L
?w2

?

?L
??

= ?L
?�y1
= X11

?�y1
?w1
?L
?�y1

+ ?L
?�y2
+X21

?�y2
?w1
?L
?�y2

= ?L
?�y1
= X12

?�y1
?w2
?L
?�y1

+ ?L
?�y2
+X22

?�y2
?w2
?L
?�y2

= ?L
?�y1
= ?L
?�y1

+ ?L
?�y2

?�y1
??
+ ?L
?�y2

?�y2
??

X11w1 + X12w2 + ?

X21w1 + X22w2 + ?

�y1
?L
?�y1

�y2
?L
?�y2

= 2(�y1 ? y1)

Loss

?�y ? y?2

= 2(�y2 ? y2)

[Draw this by hand. gradientsmulti.pdf ]

[Observe that we�re doing dynamic programming here. We�re computing the solutions of subproblems, then
using each solution to compute the solutions of several bigger problems.]

[In one sense, all we�ve done here is to rederive the fact that the gradient of the least-squares regression
cost function is ?wL = 2X?(�y ? y), where �y = Xw. But the way we�ve divided it into a forward pass and a
backward pass gives us a way to generalize it by adding more layers of computations.]

Neural Networks

95

The Backpropagation Alg.

is row i of weight matrix V [and likewise for rows of W]

[Backpropagation is a dynamic programming algorithm for computing the gradients we need to do neural
net stochastic gradient descent in time linear in the number of weights.]
Recall s?(?) = s(?) (1 ? s(?));
V ?
i
hi = s(Vi � x), so
�y j = s(W j � h), so

?Vi hi = s?(Vi � x) x = hi (1 ? hi) x
?W j �y j = s?(W j � h) h = �y j (1 ? �y j) h
= �y j (1 ? �y j) W j
?h �y j
[Here is the arithmetic expression for the same neural network I drew for you three illustrations ago. It looks
very different when you depict it like this, but don�t be fooled; it�s exactly the same network I started with.
But now we treat the weights V and W as the inputs, rather than the point x.]

W
?W j L

V
?Vi L

= ?L
?�y j
= ?L
?�y j

?W j �y j
�y j (1 ? �y j) h

s(V x)

?Vi hi
hi (1 ? hi) x

= ?L
?hi
= ?L
?hi

h

?h L

s(Wh)

�y
?�y L = 2(�y ? y)

L

?�y ? y?2

j=1

= (cid:80)k
= (cid:80)

j

?L
?�y j

?h �y j

?L
?�y j
�y j (1 ? �y j) W j

Compute ?V L, ?W L one row at a time.

[Draw this by hand. backalg.pdf ]

[Note that h and �y are computed during the forward pass, and ?�yL, ?hL, ?W L, and ?V L are computed during
the backward pass. In particular, we can�t compute ?V L until after we compute ?hL, and we can�t compute
that until after we compute ?�yL. The loss L doesn�t need to be explicitly computed at all! We can compute
all the gradients without it.]

96

Jonathan Richard Shewchuk

17 Vanishing Gradients; ReLUs; Output Units and Losses; Neurobiology

THE VANISHING GRADIENT PROBLEM; ReLUs

[Last lecture, we put a logistic function at the output of every unit except the input units. These units are
called sigmoid units. But in practice, sigmoid units are usually a poor choice for hidden layers.]
Problem: When unit output s is close to 0 or 1 for most training points, s? = s(1 ? s) ? 0, so gradient descent
changes s very slowly. Unit is �stuck.� Slow training.

maximum curvature

{
flat spot

{linear region

flat spot
}

logistic.pdf [Draw flat spots, �linear� region, & maximum curvature points (at s(?) (cid:17) 0.21
and s(?) (cid:17) 0.79) of the sigmoid function. Ideally, we would stay away from the flat spots.]

[This is called the vanishing gradient problem. The more layers your network has, the more problematic
this problem becomes. Most of the early attempts to train deep, many-layered neural nets failed.]

Solution: Replace sigmoids with ReLUs: rectified linear units.
ramp fn: r(?) = max{0, ?}.

(cid:40)

r?(?) =

1, ? ? 0,
0, ? < 0.

r(?)

?

[The derivative is not defined at zero, but we just pretend it is for the sake of gradient descent.]

Most neural networks today use ReLUs for the hidden units.
[However, it is still common to use sigmoids for the output units in classification problems.]

[ReLUs are preferred over sigmoids as hidden units because in practice, they�re much less likely to get stuck.
But the derivative r? is sometimes zero, so you might wonder if ReLUs can get stuck too. Fortunately, it�s
rare for a ReLU�s gradient to be zero for all the training data; it�s usually zero for just some training points.
But yes, ReLUs sometimes get stuck too; just not as often as sigmoids.]

[The output of a ReLU can be arbitrarily large; the fact that ReLUs don�t saturate like sigmoids do leaves
them vulnerable to a related problem called the �exploding gradient problem.� It is not a big problem in
shallow networks, but it becomes a big problem in deep or recurrent networks.]

[Even though ReLUs are linear in each half of their range, they�re still nonlinear enough to easily compute
functions like XOR. Of course, if you replace sigmoids with ReLUs, you have to change the derivation of
backprop to reflect the changes in the gradients. We�ll do that later in this lecture.]

!!!"#"!!#$"#$!#$%#$&�$#""!#Vanishing Gradients; ReLUs; Output Units and Losses; Neurobiology

97

OUTPUT UNITS

[Many neural networks use ReLUs for all or most of the hidden units, but ReLUs are rarely used as output
units. The output units are chosen to fit the application, and there are three common choices.]
Most output units are linear units (regression) or sigmoid/logistic or softmax units (classification).

[When you train a neural network with these output units by gradient descent, the last layer of edges of
the network is solving a problem in linear regression, logistic regression, or softmax regression by gradient
descent. Or maybe all three!]

(1) Linear units for regression.
Given vector h of unit values in last hidden layer, output layer computes �y = Wh.
Activation fn is the identity fn. [You could say there is no activation function.]
Then the final layer of edges is doing linear regression (on values of h & y)!
Usually trained with squared-error loss. If so, it�s least-squares linear regression.

[When we train a neural network by gradient descent, each linear output unit finds the solution of a linear
regression problem by gradient descent. In principle we could find the weights entering that unit by solving
the normal equations. But we don�t, because the hidden unit values keep changing during training.]

(2) Sigmoid units [aka logistic units] for [two-class] classification.
Let y ? Rk be vector of labels; yi ? [0, 1].
Given hidden layer h, output layer computes pre-activation a = Wh ? Rk and applies sigmoid activations to
obtain prediction �y = s(a).

[Here, s is the logistic function applied component-wise to the vector a. The labels yi are usually 0�s and 1�s,
but �yi can never be exactly 0 or 1. So it might be better to choose target labels like yi = 0.05 or yi = 0.95,
because then a neural network with enough weights and sufficiently wide layers can achieve �y = y exactly
for every training point! Unless there are co-located training points with different labels.]

Loss fn: Use logistic loss instead of squared error. Fixes vanishing gradients at output!
[The logistic loss function prevents output units from suffering the vanishing gradient problem, but it can�t
solve the vanishing gradient problem for hidden units. So we don�t use sigmoid hidden units.]

[When we train a neural network by gradient descent, each sigmoid output unit finds the solution of a logistic
regression problem by gradient descent.]

98

Jonathan Richard Shewchuk

(3) Softmax units for k-class classification.
[E.g., in the MNIST digit recognition problem, we would have k = 10 softmax output units, one for each
digit class.]
Let y ? Rk be vector of labels for training pt x [indicating x�s membership in the k classes].
[It is easy to design a neural network to solve more than one multi-class classification problem simultane-
ously, but for ease of notation let�s suppose we�re solving just one, so there are only k output units.]

Strongly recommended: choose training labels so

k(cid:88)

yi = 1.

i=1

We commonly use a one-hot encoding: one label is 1, the others are 0.
[But one-hot encoding has a disadvantage we�ve already discussed for sigmoids: each softmax prediction �yi
can never be exactly 1 or 0. It might be better to choose target labels such as 0.9, 0.05, and 0.05. Think of
the labels as posterior probabilities, so they should sum to 1.]
Given hidden layer h, output layer computes pre-activation a = Wh ? Rk and applies softmax activation to
obtain prediction �y ? Rk.

Softmax output is �yi(a) =

eai
j=1 ea j

(cid:80)k

.

Each �yi ? (0, 1);

k(cid:88)

i=1

�yi = 1.

[Interpret �yi as an estimate of the posterior probability that the input belongs to class i.]

Loss fn: Use cross-entropy. Fixes vanishing gradients at output.

For k-class softmax output, cross-entropy is L(�y, y) = ?

k(cid:88)

i=1

yi ln �yi.

? true labels

? prediction

(cid:27)

k-vectors

[When we train a neural network by gradient descent, if there are softmax output units, those units find the
solution of a softmax regression problem by gradient descent.]

[Cross-entropy losses are only for softmax and sigmoid outputs. For linear or ReLU outputs, cross-entropy
makes no sense, but squared error loss makes sense.]

Vanishing Gradients; ReLUs; Output Units and Losses; Neurobiology

99

Backpropagation for Outputs

For backprop, we need ?W L and ?hL, where h is last hidden unit layer, W is weights of last edge layer.

output + loss
�y =
L(�y, y) =
?W L =
?hL =

linear + squared error
Wh
?�y ? y?2
2(�y ? y) h?
2W?(�y ? y)

? (cid:80)

sigmoid + logistic loss
s(Wh)

i(yi ln �yi + (1 ? yi) ln(1 ? �yi))

(�y ? y) h?
W?(�y ? y)

softmax + cross-entropy
softmax(Wh)
? (cid:80)
i yi ln �yi
(�y ? y) h?
W?(�y ? y)
(cid:80)k

assuming

i=1 yi = 1

[It�s interesting that all three types of outputs produce essentially the same form of gradients for the final
layer of the network, except that the predictions �y are different in each case. This is true even though each
linear or sigmoid output unit is independent, but the softmax outputs units are coupled with each other.]
[Observe that even for sigmoid and softmax units, both ?W L and ?hL are linear in the error �y ? y. This is a
nice outcome, when you consider the exponentials and logarithms we started with. It implies that sigmoid
units with logistic loss do not get stuck when the sigmoid derivatives are small. This is related to the fact
that the logistic loss goes to infinity as the predicted value �yi approaches zero or one. The vanishing gradient
of the sigmoid unit is compensated for by the exploding gradient of the logistic loss.]
Note: we don�t need to compute ?�yL.
Instead, we eliminate �y by substituting �y(W, h) into L(�y, y).
[. . . before taking derivatives. This makes it easier both to derive the math and to write the code.]

[Now I will show you how to perform backpropagation for two hidden layers of ReLU units, a k-class
softmax output, the cross-entropy loss function, and ?2 regularization�which may improve test accuracy.]

W
?W L = (�y ? y)h?+2?W

V
?Vi L

= ?L
?hi
= ?L
?hi

?Vi hi
r?(Vi � g) g

+2?Vi

U
?Ui L

= ?L
?gi
= ?L
?gi

r(U x)

?Ui gi
r?(Ui � x) x

+2?Ui

�yi =

eWi�h
j=1 eW j�h

(cid:80)k

�y

r(Vg)

h
?h L = W ?(�y ? y)

k(cid:88)

?

yi ln �yi

i=1
+? (?U?2
+
F
+ ?W?2
?V?2
F)
F

L

?2 regularization (optional)

To add more hidden layers,
copy this.

g
?g L = (cid:80)m
j=1
= (cid:80)
?L
?h j

j

?L
?g h j
?h j
r?(V j � g) V j

[Draw this by hand. backpropsoft2.pdf ]
[Note that r(U x) is the ramp function applied component-wise to the vector U x. The derivative r?(Ui � x) is
always zero or one. Observe that we don�t need to compute the loss L at all. We also don�t compute ?�yL, as
we said above, but we do need to compute the value of �y to compute gradients.]

100

Jonathan Richard Shewchuk

Derivations

[I won�t go over this page of derivations in lecture, but I include them here for completeness. Students who
want to understand neural networks deeply should spend some time going through these.]
Linear output units (�y = Wh) with squared error loss:

L(�y, y) = ?�y ? y?2 = ?Wh ? y?2 =

k(cid:88)

(Wi � h ? yi)2,

i=1

?Wi L = 2(Wi � h ? yi) h = 2(�yi ? yi) h,
?W L = 2(�y ? y) h?,
?hL = 2W?(Wh ? y) = 2W?(�y ? y).

Sigmoid [logistic] output units (�y = s(a) = s(Wh)) with logistic loss:

L(�y, y) = ?

k(cid:88)

(yi ln �yi + (1 ? yi) ln(1 ? �yi)) = ?

(cid:32)

k(cid:88)

yi ln

1
1 + e?ai

+ (1 ? yi) ln

(cid:32)

1 ?

(cid:33)(cid:33)

1
1 + e?ai

i=1
(cid:32)
yi ln(1 + e?ai) ? (1 ? yi) ln

k(cid:88)

i=1
(cid:33)

e?ai
1 + e?ai

=

=

i=1
k(cid:88)

(cid:0)

i=1

yi ln(1 + e?ai) ? (1 ? yi)(?ai ? ln(1 + e?ai)

k(cid:88)

(cid:0)

(cid:1) =

i=1

(1 ? yi)ai + ln(1 + e?ai)

(cid:1) ,

?L
= 1 ? yi ?
?ai
ai = Wi � h,

=

e?ai
1 + e?ai
?Wiai = h,

1
1 + e?ai

?hai = Wi,

? yi = �yi ? yi,

?Wi L = ?L
?ai
?W L = (�y ? y) h?,

?Wiai = (�yi ? yi) h,

?hL =

k(cid:88)

i=1

?L
?ai

?hai =

k(cid:88)

i=1

(�yi ? yi)Wi = W?(�y ? y).

Softmax output units (�y = softmax(a) = softmax(Wh)) with cross-entropy loss:

[This derivation uses the assumption that

k(cid:88)

i=1

yi = 1 for each training point�s labels.]

L(�y, y) = ?

k(cid:88)

i=1

?L
?ai

= ?yi +

k(cid:88)

yi ln �yi = ?

?

????????

eai

(cid:44) k(cid:88)

j=1

i=1
?

????????

ea j

= �yi ? yi.

?

????????

yi

ai ? ln

?

????????

ea j

k(cid:88)

j=1

= ?

k(cid:88)

i=1

yiai + ln

k(cid:88)

j=1

ea j,

From here, we repeat the last four lines of the sigmoid derivation.

Vanishing Gradients; ReLUs; Output Units and Losses; Neurobiology

101

NEUROBIOLOGY

[The field of artificial intelligence started with some wrong premises. The early AI researchers attacked
problems like chess and theorem proving, because they thought those exemplified the essence of intelligence.
They didn�t pay much attention at first to problems like vision and speech understanding. Any four-year-old
can do those things, and so researchers underestimated their difficulty.]

[Today, we know better. Computers can beat world chess champions, but they still can�t play with toys well.
We�ve come to realize that rule-based symbol manipulation is not the primary defining mark of intelligence.
Even rats do computations that we�re hard pressed to match with our computers. We�ve also come to realize
that these are different classes of problems that require very different styles of computation. Brains and
computers have very different strengths and weaknesses, which reflect their different computing styles.]

[Neural networks are partly inspired by the workings of actual brains. Let�s take a look at a few things we
know about biological neurons, and contrast them with both neural nets and traditional computation.]

� CPUs: largely sequential, nanosecond gates, fragile if gate fails
superior for arithmetic, logical rules, perfect key-based memory

� Brains: very parallel, millisecond neurons, fault-tolerant

[Neurons are continually dying. You�ve probably lost a few since this lecture started. But you probably
didn�t notice. And that�s interesting, because it points out that our memories are stored in our brains
in a diffuse representation. There is no one neuron whose death will make you forget that 2 + 2 = 4.
Artificial neural nets often share that resilience. Brains and neural nets seem to superpose memories
on top of each other, all stored together in the same weights, sort of like a hologram.]

[In the 1920�s, the psychologist Karl Lashley conducted experiments to identify where in the brain
memories are stored. He trained rats to run a maze, and then made lesions in different parts of the
cerebral cortex, trying to erase the memory trace. Lashley failed; his rats could still find their way
through the maze, no matter where he put lesions. He concluded that memories are not stored in
any one area of the brain, but are distributed throughout it. Neural networks, properly trained, can
duplicate this property.]

superior for vision, speech, associative memory

[By �associative memory,� I mean noticing connections between things. One thing our brains are very
good at is retrieving a pattern if we specify only a portion of the pattern.]

[It�s impressive that even though a neuron needs a few milliseconds to transmit information to the next
neurons downstream, we can perform very complex tasks like interpreting a visual scene in a tenth of a
second. This is possible because neurons run in parallel, but also because of their computation style.]

[Neural nets try to emulate the parallel, associative thinking style of brains, and they are the best techniques
we have for many fuzzy problems, including most problems in vision and speech. Not coincidentally, neural
nets are also inferior at many traditional computer tasks such as multiplying 10-digit numbers or compiling
source code.]

102

Jonathan Richard Shewchuk

18 Neurobiology; Faster Neural Network Training

NEUROBIOLOGY (cont�d)

neurons.pdf

� Neuron: A cell in brain/nervous system for thinking/communication
� Action potential or spike: An electrochemical impulse fired by a neuron to communicate w/other

neurons

� Axon: The limb(s) along which the action potential propagates; �output�
[Most axons branch out eventually, sometimes profusely near their ends.]
[It turns out that squids have a very large axon they use for fast propulsion by expelling jets of water.
The mathematics of action potentials was first characterized in these squid axons, and that work won
a Nobel Prize in Physiology in 1963.]

� Dendrite: Smaller limb by which neuron receives info; �input�
� Synapse: Connection from one neuron�s axon to another�s dendrite

[Some synapses connect axons to muscles or glands.]

� Neurotransmitter: Chemical released by axon terminal to stimulate dendrite

[When an action potential reaches an axon terminal, it causes tiny containers of neurotransmitter, called
vesicles, to empty their contents into the space where the axon terminal meets another neuron�s dendrite.
That space is called the synaptic cleft. The neurotransmitters bind to receptors on the dendrite and influence
the next neuron�s body voltage. This sounds incredibly slow, but it all happens in 1 to 5 milliseconds.]
You have about 1011 neurons, each with about 104 synapses.

Neurobiology; Faster Neural Network Training

103

Analogies: [between artificial neural networks and brains]

� Output of unit ? firing rate of neuron

[An action potential is �all or nothing��all action potentials have the same shape and size. The output
of a neuron is not signified by voltage like the output of a transistor. The output of a neuron is the
frequency at which it fires. Some neurons can fire at nearly 1,000 times a second, which you might
think of as a strong �1� output. Conversely, some types of neurons can go for minutes without firing.
But some types of neurons never stop firing, and for those you might interpret a firing rate of 10 times
per second as a �0�.]

� Weight of connection ? synapse strength
� Positive weight ? excitatory neurotransmitter (e.g., glutamine)
� Negative weight ? inhibitory neurotransmitter (e.g., GABA, glycine) [Gamma aminobutyric acid.]

[A typical neuron is either excitatory at all its axon terminals, or inhibitory at all its terminals. It can�t
switch from one to the other. Artificial neural nets have an advantage here.]

� Linear combo of inputs ? summation

[A neuron fires when the sum of its inputs, integrated over time, reaches a high enough voltage.
However, the neuron body voltage also decays slowly with time, so if the action potentials are coming
in slowly enough, the neuron might not fire at all.]

� Logistic/sigmoid fn ? firing rate saturation

[A neuron can�t fire more than 1,000 times a second, nor less than zero times a second. This limits its
ability to overpower downstream neurons. We accomplish the same thing with the sigmoid function.]

� Weight change/learning ? synaptic plasticity

[Donald] Hebb�s rule (1949): �Cells that fire together, wire together.�
[This doesn�t mean that the cells have to fire at exactly the same time. But if one cell�s firing tends to
make another cell fire more often, their excitatory synaptic connection tends to grow stronger. There�s
a reverse rule for inhibitory connections. And there are ways for neurons that aren�t even connected
to grow connections.]
[There are simple computer learning algorithms based on Hebb�s rule. They can work, but they�re
generally not nearly as fast or effective as backpropagation.]

[Backpropagation is one part of artificial neural networks for which any analogy is doubtful. There have
been some proposals that the brain might do something vaguely like backpropagation,7 but it seems tenuous.
Learning in brains is still not well understood.]

[As computer scientists, our primary motivation for studying neurology is to try to get clues about how we
can get computers to do tasks that humans are good at. But neurologists and psychologists have also been
part of the study of neural nets from the very beginning. Their motivations are scientific: they�re curious
how humans think, and how we can do the things we do.]

7See Lillicrap et al., �Backpropagation and the Brain,� Nature Reviews Neuroscience 21, pages 335-�346, April 2020.

104

Jonathan Richard Shewchuk

HEURISTICS FOR FASTER TRAINING

[A big disadvantage of neural nets is that they take a long, long time to train compared to other classifi-
cation methods we�ve studied. Here are some ways to speed them up. Unfortunately, you usually have to
experiment with techniques and hyperparameters to find which ones will help with your particular applica-
tion. I suggest you implement vanilla backpropagation first, usually in combination with stochastic gradient
descent and intelligent weight initialization, and experiment with fancy heuristics only after you get that
working.]

(1) ReLUs. [To fix the vanishing gradient problem, as described in the previous lecture.]

(2) Stochastic gradient descent (SGD): faster than batch on large, redundant data sets.
[Whereas batch gradient descent walks downhill on one cost function, stochastic descent takes a very short
step downhill on one point�s loss function and then another short step on another point�s loss function.
The cost function is the sum of the loss functions over all the sample points, so one batch step is akin
to n stochastic steps and does roughly the same amount of computation. But if you have many different
examples of the digit �9�, they contain much redundant information, and stochastic gradient descent learns
the redundant information more quickly�often much more quickly. Conversely, if the data set is so small
that it encodes little redundant information, batch gradient descent is typically faster.]

z

w1
x1

w2
x2

batchvsstochmod.pdf (LeCun et al., �Efficient BackProp�) [Left: a perceptron with only
two weights trained to minimize the mean squared error cost function, and its 2D training
data. Center: batch gradient descent makes only a little progress each epoch. Epochs
alternate between red and blue. Right: stochastic descent decreases the error much faster
than batch descent. Again, epochs alternate between red and blue.]

One epoch presents every training point once. Training usually takes many epochs, but if sample is huge
[and carries lots of redundant information], SGD can take less than one epoch.

(cid:2)1(cid:2)0(cid:2)2(cid:1)0y(cid:1)1(cid:1)1.4(cid:1)1.2(cid:1)1(cid:1)0.8(cid:1)0.6(cid:1)0.4(cid:1)0.200.20.40.60.811.21.4(cid:1)1.4(cid:1)1.2(cid:1)1(cid:1)0.8(cid:1)0.6(cid:1)0.4(cid:1)0.200.20.40.60.811.21.4(cid:20)(cid:24)(cid:27)(cid:25)(cid:26)(cid:31)(cid:1)(cid:30)(cid:29)(cid:21)(cid:22)(cid:24)(cid:1)(cid:6)(cid:1)(cid:5)(cid:4)(cid:13)(cid:1)(cid:5)(cid:4)(cid:11)(cid:1)(cid:5)(cid:4)(cid:9)(cid:1)(cid:5)(cid:4)(cid:7)(cid:5)(cid:5)(cid:4)(cid:7)(cid:5)(cid:4)(cid:9)(cid:5)(cid:4)(cid:11)(cid:5)(cid:4)(cid:13)(cid:6)(cid:5)(cid:5)(cid:4)(cid:7)(cid:5)(cid:4)(cid:9)(cid:5)(cid:4)(cid:11)(cid:5)(cid:4)(cid:13)(cid:6)(cid:6)(cid:4)(cid:7)(cid:6)(cid:4)(cid:9)(cid:6)(cid:4)(cid:11)(cid:6)(cid:4)(cid:13)(cid:7)(cid:17)(cid:28)(cid:25)(cid:1)(cid:18)(cid:19)(cid:16)(cid:1)(cid:2)(cid:23)(cid:15)(cid:3)(cid:5)(cid:6)(cid:7)(cid:8)(cid:9)(cid:10)(cid:11)(cid:12)(cid:13)(cid:14)(cid:6)(cid:5)(cid:1)(cid:7)(cid:5)(cid:1)(cid:6)(cid:10)(cid:1)(cid:6)(cid:5)(cid:1)(cid:10)(cid:5)(cid:2)(cid:5)(cid:4)(cid:1)(cid:3)(cid:6)(cid:20)(cid:24)(cid:27)(cid:25)(cid:26)(cid:31)(cid:1)(cid:30)(cid:29)(cid:21)(cid:22)(cid:24)(cid:1)(cid:6)(cid:1)(cid:5)(cid:4)(cid:13)(cid:1)(cid:5)(cid:4)(cid:11)(cid:1)(cid:5)(cid:4)(cid:9)(cid:1)(cid:5)(cid:4)(cid:7)(cid:5)(cid:5)(cid:4)(cid:7)(cid:5)(cid:4)(cid:9)(cid:5)(cid:4)(cid:11)(cid:5)(cid:4)(cid:13)(cid:6)(cid:5)(cid:5)(cid:4)(cid:7)(cid:5)(cid:4)(cid:9)(cid:5)(cid:4)(cid:11)(cid:5)(cid:4)(cid:13)(cid:6)(cid:6)(cid:4)(cid:7)(cid:6)(cid:4)(cid:9)(cid:6)(cid:4)(cid:11)(cid:6)(cid:4)(cid:13)(cid:7)(cid:17)(cid:28)(cid:25)(cid:1)(cid:18)(cid:19)(cid:16)(cid:1)(cid:2)(cid:23)(cid:15)(cid:3)(cid:5)(cid:6)(cid:7)(cid:8)(cid:9)(cid:10)(cid:11)(cid:12)(cid:13)(cid:14)(cid:6)(cid:5)(cid:1)(cid:7)(cid:5)(cid:1)(cid:6)(cid:10)(cid:1)(cid:6)(cid:5)(cid:1)(cid:10)(cid:5)(cid:2)(cid:5)(cid:4)(cid:1)(cid:3)(cid:6)(cid:2)(cid:1)(cid:5)(cid:3)(cid:4)Neurobiology; Faster Neural Network Training

105

(3) SGD with minibatches.
Choose a minibatch size b; e.g. 256.
Repeatedly perform gradient descent on the sum of the loss functions of b randomly chosen points.

[Although we perform gradient descent on a minibatch of training points all at once, we don�t call it batch
gradient descent. We still call it stochastic gradient descent.]

Advantages [compared to SGD done just one point at a time]:

� Less �bouncy�; usually converges more quickly.

[SGD bounces around wildly. Minibatches reduce the variance of the steps by a factor of
maintaining the advantages of SGD.]

� Can use parallelism, vectorization, GPUs efficiently.

?

b while

[The backpropagation computations are fully independent from one training point to another, so it�s
very easy to compute gradients for multiple points in parallel or through vectorization.]

� Better speed because of memory hierarchy.

[You should lay out the activations for the training points in the minibatch next to each other in
memory. With the right memory layout and minibatch size, your use of the caches and memory
hierarchy can be very efficient. Performing SGD on 64 training points might be almost as fast as
performing SGD on one. The bottleneck in neural network training is memory latency, not arithmetic.]

[Minibatches nearly always work faster than processing just one training point at a time. They are standard
in implementations of neural network training.]
Typically, we shuffle training pts, partition into ?n/b? minibatches.
An epoch presents each minibatch once. [Reshuffling for each epoch is optional.]
[It is important to randomize well so your minibatch is a representative subsample of the training points.
Sometimes practitioners store each class in a separate list, shuffle each class separately, and build mini-
batches with a proportional number of training points from each class.]
[Be forewarned that the best learning rate ? will be different for different values of the minibatch size b, and
there isn�t always a predictable relationship between b and the best ?.]

(4) To choose learning rate ?, use a small random subsample of training data.
[Practitioners have found that the size of the training set has only a weak effect on the best choice of ?. So
use a subsample to quickly estimate a good learning rate, then apply it to your whole training set. This is
very easy to do, and it can save you a lot of time!]

(5) Emphasizing schemes. [Neural networks learn the most redundant examples quickly, and the most rare
examples slowly. This motivates emphasizing schemes, which repeat the rare examples more often.]

� Stochastic: present examples from rare classes more often, or w/bigger ?.
� Batch: examples from rare classes have bigger losses.

[Emphasizing schemes are a natural way to incorporate an asymmetric loss function. Suppose you�re devel-
oping a test for a rare disease, and you have a small number of examples of people with the disease, but many
examples of people without it. The classifier might decide to always return negative, �no disease,� because
it can achieve 99% accuracy that way. So we�d like to impose larger losses on examples with positive labels.
For batch gradient descent, we make their losses bigger in the cost function. For SGD there are two options:
use a bigger loss for rare examples, equivalent to presenting them with a larger learning rate ?; or present
them more often. It�s not clear which of these two options will train faster, as those extra presentations take
more computation but shorter steps have better convergence properties.]

[Emphasizing schemes can also be used to emphasize misclassified training points, like the Perceptron
Algorithm does, but that can backfire if those points are bad outliers.]

106

Jonathan Richard Shewchuk

(6) Normalizing the training pts.

� Center each feature so mean is zero.
� Then scale each feature so variance ? 1.

normalize.jpg [A 2D example of normalizing points.]

[Remember that the power of neural networks comes from the nonlinearity of the activation function, and
the nonlinearity of a sigmoid or ReLU unit falls where the linear combination of values coming in is close
to zero. Centering makes it easier for the first layer of hidden units to be in the nonlinear operating region.]

[Neural networks are an example of an optimization algorithm whose cost function tends to have better-
conditioned Hessians if the input features are normalized, so it may converge to a local minimum faster.]

illcondition105.pdf, illcondition055.pdf, goodcondition.pdf

[Recall these illustrations from Lecture 5. Gradient descent on a function with an ill-conditioned Hessian
matrix can be slow because a large step size diverges in one direction (left) while a small step size converges
slowly in another direction (center). Normalizing the data might improve the conditioning of the Hessian
(right), thus speeding up gradient descent. Moreover, if you use ?2-regularization, normalization makes it
penalize the features more equally.]
[You could go even further and whiten the data, as we discussed in Lecture 9, but whitening takes ?(nd2+d3)
time for n training points with d features, so it takes too much time if d is very large; whereas normalizing
takes ?(nd) time.]
[Remember that whatever linear transformation you apply to the training points, you must later apply the
same linear transformation to the test points you want to classify!]

-4-224w1-2246w2-4-224w1-2246w2-4-224w1-2246w2Neurobiology; Faster Neural Network Training

107

(7) Initializing weights. [Proper initialization of weights is very important, especially for deep networks.
Consider this carefully for Homework 6!]

[Recall that we initialize a neural network with random weight values to break the symmetry between hidden
units. If we make those random values too small, they might never grow enough, especially if the network is
deep. If we make them too large, we may cause the exploding gradient problem in ReLUs, or the vanishing
gradient problem in sigmoid units. Here are some rules of thumb for initializing random weights.]

Consider the variance of each unit�s output, given random weights.
Principle: output of unit should have same variance as each of its inputs.8

(cid:16)

?

(cid:112)

(cid:112)

For a unit with fan-in ? (not counting bias term), initialize each incoming edge to . . .
[The fan-in of a unit is the number of connections entering the unit.]
For a ReLU unit, a weight in N(0, 2/?) or U
[This is called He initialization, after Kaiming He.]
(cid:112)
?
For a sigmoid unit, make it N(0, 12.8/?) or U
(cid:16)
For a linear or tanh unit, make it N(0, 1/?) or U
[This initialization is sometimes called Xavier initialization, but it isn�t quite what Xavier Glorot originally
proposed. A tanh unit is very similar to a sigmoid unit, but its output is centered at zero, whereas sigmoid
outputs are centered at 0.5. I don�t recommend you use either sigmoid or tanh units as hidden units, but if
you do, the tanh is preferable for that reason.]

38.4/?
(cid:17)
3/?

38.4/?,
(cid:112)
3/?,

6/?,

6/?

(cid:112)

(cid:112)

?

(cid:17)

(cid:16)

(cid:17)

.

.

.

[The reason we divide by the fan-in is because the more inputs a unit has, the greater its incoming signal is.
So we must make the weights smaller to match the unit�s output variance to the variance of each input.]

Set bias terms to zero. [Bias terms can too easily overpower signals coming from earlier layers. For ReLU
units, some people suggest setting the bias terms to a small positive constant so they�re more likely to be
turned on at first, but other people say it gives worse performance in practice.]
Linear output unit: set bias term to the mean label.
Sigmoid output unit: set bias term so default output is the mean label.
[E.g., if 90% of your training points are in class C, set the bias so the sigmoid output defaults to 0.9. Andrej
Karpathy says that if you don�t initialize the output unit biases, the first few minibatches are largely wasted
learning the mean labels.]

(8) Momentum. Gradient descent changes �momentum� m slowly. [The intuition is that if you�ve taken
many steps in roughly the same direction, you should go faster in that direction.]

m ? ??? ?J(w)
repeat

w ? w + m
m ? ? m ? ? ?J(w)

Good for both batch & stochastic. Choose hyperparameter ? < 1.
[Here, J is the cost for the minibatch, which could be anything from a single training point to the whole
training set. The hyperparameter ? specifies how much momentum persists from iteration to iteration.]
[I�ve seen conflicting advice on ?. Some researchers set it to 0.9; some set it close to zero. Geoff Hinton
suggests starting at 0.5 and slowly increasing it to 0.9 or higher as the gradients get small.]
[If ? is large, you should usually choose ? small to compensate, but you might still use a large ?? in the first
line so the initial velocity is reasonable.]

8For an explanation of these suggestions, see Siddharth Krishna Kumar, �On Weight Initialization in Deep Neural Networks.�

108

Jonathan Richard Shewchuk

sgdmomentumgodoy.png (Daniel Godoy)
[Left: 50 steps of SGD (with 16-point mini-
batches) don�t get very close to the minimum (red). Right: 50 steps with momentum do get
close to the minimum, but overshoot it several times.]

[A problem with momentum is that once it gets close to a good minimum, it overshoots the minimum, then
oscillates around it. But it often gets us close to a good minimum sooner. We see both phenomena above.]

pretzelwaterpark.jpg [How I imagine a neural network�s cost function. It does not resem-
ble a parabolic bowl. The downhill paths from the start to the local minima are sinuous.
The swimmers here employ gradient descent with momentum with great success.]

Convolutional Neural Networks

109

19 Convolutional Neural Networks

CONVOLUTIONAL NEURAL NETWORKS (ConvNets; CNNs)

[Convolutional neural nets drove a big resurgence of interest in neural nets starting in 2012. Often you�ll
hear the buzzword deep learning, which refers to neural nets with many layers. Most image recognition
networks are deep and convolutional. In 2018, the Association for Computing Machinery gave the Alan M.
Turing Award to Geoff Hinton, Yann LeCun, and Yoshua Bengio for their work on deep neural networks.]
Vision: inputs are images. 400 � 400 image = 160,000 pixels.
If we connect them all to 160,000 hidden units ? 25.6 billion connections.
[With so many weights, the network is very slow to train or even to use once trained.]

[Remember that early in the semester, I told you that you can get better performance on the handwriting
recognition task by using edge detectors. Edge detectors have two interesting properties. First, each edge
detector looks at just one small part of the image. Second, the edge detection computation is the same no
matter which part of the image you apply it to. Let�s apply these two properties to neural net design, plus
one new idea: we�ll learn the edge detectors instead of hard-coding them.]

ConvNet ideas:

� Local connectivity: A hidden unit connects only to a small patch of units in previous layer.

[A unit in the first hidden layer doesn�t look at the whole image. It looks only at a small number of
pixels�typically 9, 25, or 49 pixels. This speeds up both training and classification considerably.]
� Shared weights: Groups of hidden units share same set of weights, called a filter aka mask aka kernel.

Each filter operates on every patch of image.

patches

4

3

9

1

0

5

7

8

2

4

3

9

1

0

5

7

8

2

7

8

2

3

9

5

4

1

0

not a matrix!

input units

image

hidden units

activation map

shared weights

filter

convlayer.pdf [Applying a filter to an image. Every hidden unit uses the same nine shared
weights. In this example, a 6 � 6 image is covered by 16 overlapping 3 � 3 patches, yielding
a 4 � 4 activation map of hidden units. We learn the filter by backpropagation.]

110

Jonathan Richard Shewchuk

If image size is J � K and filter size is M � M, the activation map is (J ? M + 1) � (K ? M + 1) hidden
units�one for each patch.

A convolutional layer learns multiple filters. There is one activation map per filter. A channel is an activation
map, OR another dimension such as the red/green/blue channels of an input image.

caredges.pdf (Cezanne Camacho) [Two Sobel filters (one horizontal, one vertical) and a Laplacian filter
applied to an image, yielding three activation maps (three channels). Note that these filters were not learned
by a CNN (but they could be).]

5
-3
-2

8
-9
-6
-3 4
3
7
4
0
1
98
-9
2 5 0

filter = 3D array

input units
3 channels: red, green, blue

hidden units
4 channels: activation maps

shared weights
4 filters (4D array)

convchannels.pdf [A convolutional layer (of edges). If a layer�s input has more than one
channel, then each filter is represented by a three-dimensional matrix. The set of all filters
is represented by a four-dimensional matrix.]

00000-100-1-1-14-1-2-1121-110-220-110Sobel horizontalLaplacianSobel verticalConvolutional Neural Networks

111

[The output of a convolutional layer has multiple channels, and usually so does the input. The layer�s output
has one channel per filter, and these channels becomes inputs to downstream convolutional layers. The input
to the neural network often has multiple channels too; most commonly, the color channels of a color image.]
If edge layer l has C(l?1) channels in and C(l) channels out (C(l) filters),

� # of weights/filter = C(l?1) � M � M;
� # of weights in layer = C(l?1) � C(l) � M � M;
� # of units out = C(l)� # of patches = C(l) � (J ? M + 1) � (K ? M + 1).

Typically, each convolutional hidden unit ends with a ReLU activation.
[But there are exceptions in many modern CNNs.]

Options for bias terms:

� Untied bias: C(l) � (J ? M + 1) � (K ? M + 1) bias terms�one per unit out.
� Tied bias: C(l) bias terms�one per filter/channel out.
� No bias terms. [This option is usually not immediately followed by a ReLU.]

[Untied bias terms are a lot of extra weights, sometimes more weights than the filters! Sometimes they give
better test accuracy, but not always, so try validating both ways.]

Benefits of shared weights:

� Much less memory needed. [Better cache behavior too.]
� Regularization. [It�s unlikely that a weight will become spuriously large if it�s used in many places.]
� If one filter learns to detect edges, every patch has an edge detector.
[Because the filter that detects edges is applied to every patch.]
ConvNets exploit repeated structure in images, audio.

� A filter destined to become an edge detector learns on edges in every part of every image.

[So it can learn the idea faster.]

[In a neural net, you can think of hidden units as added features that we learn, as opposed to added features
that you code up yourself. Convolutional neural nets take them to the next level by learning features from
multiple patches simultaneously and then applying those features everywhere, not just in the patches where
they were originally learned.]

LeNet5.png Architecture of LeNet5.

[ConvNets were first popularized by the success of Yann LeCun�s �LeNet 5� handwritten digit recognition
software. LeNet 5 has six hidden layers! Hidden layers 1 and 3 are convolutional layers with shared weights.
Layers 2 and 4 are pooling layers that make the image smaller, with no weights at all. Layers 5 and 6 are
fully-connected layers of hidden units with no shared weights. A great deal of experimentation went into

112

Jonathan Richard Shewchuk

figuring out the number of layers and their sizes. At its peak, LeNet 5 was responsible for reading the zip
codes on 10% of US Mail. Another Yann LeCun system was deployed in ATMs and check reading machines
and was reading 10 to 20% of all the checks in the US by the late 90�s. LeCun is one of the Turing Award
winners I told you about earlier.]

[Show Yann LeCun�s video LeNet5.mov, illustrating LeNet 5.]

Downsampling

[At the output of LeNet 5, we have to compress the information down to a single 10-unit output. Experience
shows that this is best done by slowly compressing the information in the image through a sequence of
layers, rather than connecting a very large layer of hidden units directly to the output. This observation
echoes classic image processing techniques that were developed before neural networks. The two popular
methods of downsampling are called pooling and strided convolution.]

Max pooling: Reduce a J � K image to ?J/2? � ?K/2? or ?J/3? � ?K/3? [as illustrated].

maxavgpool.pdf (Ekpenyong et al.) [Max pooling and average pooling.]

Average pooling: likewise, but each unit out is the average of four units in.
[Average pooling was used in LeNet 5, but max pooling seems more popular now.]

Pooling layers have no weights. Nothing to train!
[But you still have to think carefully about how to do backpropagation through them.]

Strided convolution: Patches overlap less (or not at all).

stride.pdf (Vadlamani & Patel) [Strided convolution.]

SpringerNature2021LATEXtemplateArticleTitle7Fig.5Convolutionoperationwithpoolingsizeof2x23.6FullyConnectedLayerTheoutputfromthe?nal(andany)poolingandconvolutionallayeris?attenedandthenfedintothefullyconnectedlayerasdescribedinFig6.Afterpassingthroughthefullyconnectedlayers,the?nallayerusesthesoftmaxactivationfunctionwhichisusedtogetprobabilitiesoftheinputbeinginaparticularclass(classi?cation).Fig.6FullyConnectedLayerThe?nalCNNarchitectureissummarizedinTable1:Itisalsonote-worthythatweaddedthebatchnormalizationanddropoutlayersintheCNNarchitecture.Thebatchnormalizationisaregularizationtechniquethathelpstopreventover?tting[16].Thisprocessisimportantbecauseitimprovesthespeed,performance,andstabilityoftraining.Itdoesthisbynormalizingeachlayer�sinputsbysquashingthevaluestoazeromeanandunitvarianceinthecurrentbatch.Dropoutlayerwasalsoimple-mentedper-layerinthenetwork.Duringtraining50%ofthelayeroutputswererandomlyignored(droppedout),thishelpspreventover?ttingaswell.K(i,j)isthekernelor?lterofsizei?j,S(i,j)istheoutputfeaturedmap.PropertiesofConvolutionoperation:Theconvolutionoperationsatis?escommutativeproperty,theequationisgivenbelow.[12]S(i,j)=(K?I)(i,j)=XmXnI(i m,j n)K(m,n))(5)TheCross-Correlationcanbeobtainedby?ipingthenegativesignorreplacethenegativesignwithpositivesign.TheCross-Correlationcanbegivenas.[12]S(i,j)=(K?I)(i,j)=XmXnI(i+m,j+n)K(m,n))(6)Theconvolutionoperationforantwo-dimensionalimageisexplainedinthebelowFig.2.Theimageofsize5?5istakenasinputfortheconvolutionoperation.Thekernelorthe?lterofsize3?3istaken.TheoutputoftheconvolutionoperationiscalledConvolvedFeaturemap.Theconvolutionoperationisperformedbyslidingthekernelor?lterontheimagepixel.TheconvolvedoutputfeaturemapisextractedasshowninthebelowFig.2.Theconvolutionallayerhasseveral?ltersFig.2.ConvolutionOperationorkernels.Foreachinputimagetheconvolutionoperationisrepeatedwithseveral?ltersofdifferentsizestoobtaintheoutputfeaturedmap.FiltersorKernels:Theconvolutionoperationisperformedwithseveralnumberof?lterstoobtaindifferentfeaturesoftheimage.Someofthe?ltersweuseingeneralareedgedetection,sharpen,blur,Gaussianblurandmore.Someofthe?ltersandtheoutputfeaturemapsaregiveninthebelowFig.3.[13]Fig.3.Commonlyused?ltersinConvolutionOperation[13]Strides:De?nedasnumberofpixelsshiftedovertheinputimagetoapply?lterduringconvolutionoperation.Theconvolutionoperationisperformedbyshifting?lterwithonepixelifthevalueofstrideisselectedasone.Theconvolutionoperationisperformedbyshifting?lterwithtwopixelifthevalueofstrideisselectedastwo.theexampleforthestridevaluewithtwoisexplainedinthebelowFig.4.InthebelowFig.4theinputofsize7?7,?lterofsize3?3andstridevalueof2aretakentoperformtheconvolutionoperation[13].Fig.4.ConvolutionOperationwithStridevalue2Padding:Duringconvolutionoperationsometimesthe?lterisnotperfectly?ttotheinputimage.ToavoidthissituationPaddingoperationisheldontheinputimage.Therearetwotypesofpaddingoperationareappliedtotheinputimageduringtheconvolutionoperation.[13]Convolutional Neural Networks

113

AlexNet

[I told you three lectures ago that neural net research was popular in the 60�s, but the 1969 book Perceptrons
killed interest in them throughout the 70�s. They came back in the 80�s, but interest was partly killed off a
second time in the 00�s by . . . guess what? By support vector machines. SVMs work well for a lot of tasks,
they�re much faster to train, and they more or less have only one hyperparameter, whereas neural nets take a
lot of work to tune.]

[Neural nets are now in their third wave of popularity. The single biggest factor in bringing them back is
probably big data. Thanks to the internet, we now have absolutely huge collections of images to train neural
nets with, and researchers have discovered that neural nets often give better performance than competing
algorithms when you have huge amounts of data to train them with. In particular, convolutional neural nets
are now learning better features than hand-tuned features. That�s a recent change.]

[The event that brought attention back to neural nets was the ImageNet Image Classification Challenge in
2012. The winner of that competition was a neural net, and it won by a huge margin, about 10%. It�s called
AlexNet, and it�s surprisingly similarly to LeNet 5, in terms of how its layers are structured. However,
there are some recent innovations that led to their prize-winning performance, in addition to the fact that the
training set had 1.4 million images: they used ReLUs, dropout, data augmentation, and GPUs for training.]

alexnet.pdf (Krizhevsky et al., 2012, 2017) [Architecture of AlexNet.]

[When ConvNets were first applied to image analysis, researchers found that some of the learned filters are
edge detectors! Here are the first layers of filters learned by AlexNet.]

filtersalex.png (Krizhevsky et al., 2012, 2017) [Filters learned by the first layer of AlexNet.]

114

Jonathan Richard Shewchuk

[Not all of the features are edge detectors; many of them are more concerned with color. But more than half
of them resemble mathematical functions called Gabor filters, which detect edges and also textures.]

gabor.pdf (Bishop, Figure 10.11) [Gabor filters. Not learned; these are math functions.]

[AlexNet learned some simple color-specific edge detectors, but I find it noteworthy that the higher-frequency
texture detectors are not sensitive to color at all. Apparently, the CNN decided it can separate fine texture
from color.]

[Unfortunately, we can�t just draw the filters learned by subsequent convolutional layers, because they�re
3D arrays that don�t carry much visual information. Instead, Zeiler and Ferguson (2013) have a technique
where they determine which patches from the training set most trigger a particular filter, and they draw nine
of those. They also have a technique for determining which pixels of those patches are most relevant to
triggering the filter, and they plot the relevance of each patch pixel. This reveals that in the third example
for convolutional layer 4, the filter is primarily responding to the grass in these images.]

Conv layer 2 (four of the filters):

10.3.VisualizingTrainedCNNs303Figure10.11ExamplesofGabor?ltersde?nedby(10.6).Theorientationangle?variesfrom0inthetoprowto?/2inthebottomrow,whereasthefrequencyvariesfrom?=1intheleftcolumnto?=10intherightcolumn.convolutionalneuralnetworks.Theneocognitronhadmultiplelayersofprocessingcomprisinglocalreceptive?eldswithsharedweightsfollowedbylocalaveragingormax-poolingtoconferpositionalinvariance.However,itlackedanend-to-endtrain-ingproceduresinceitpredatedthedevelopmentofbackpropagation,relyinginsteadongreedylayer-wiselearningthroughanunsupervisedclusteringalgorithm.10.3.2Visualizingtrained?ltersSupposewehaveatraineddeepCNNandwewishtoexplorewhatthehiddenunitshavelearnedtodetect.Forthe?ltersinthe?rstconvolutionallayerthisisrelativelystraightforward,astheycorrespondtosmallpatchesintheoriginalinputimagespace,andsowecanvisualizethenetworkweightsassociatedwiththese?ltersdirectlyassmallimages.The?rstconvolutionallayercomputesinnerproductsbetweenthe?ltersandthecorrespondingimagepatches,andsotheunitwillhavealargeactivationwhentheinnerproducthasalargemagnitude.Figure10.12showssomeexample?ltersfromthe?rstlayerofaCNNtrainedontheImageNetdataset.Weseearemarkablesimilaritybetweenthese?ltersandtheGabor?ltersofFigure10.11.However,thisdoesnotimplythataconvolutionalneuralnetworkisagoodmodelofhowthebrainworks,becauseverysimilarresultscanbeobtainedfromawidevarietyofstatisticalmethods(Hyv�arinen,Hurri,andHoyer,2009).Thisisbecausethesecharacteristic?ltersareageneralpropertyofthestatisticsofnaturalimagesandthereforeproveusefulforimageunderstandinginbothnaturalandarti?cialsystems.Althoughwecanvisualizethe?ltersinthe?rstlayerdirectly,thesubsequentlayersinthenetworkarehardertointerpretbecausetheirinputsarenotpatchesofimagesbutgroupsof?lterresponses.Oneapproach,analogoustothatusedbyHubelandWiesel,istopresentalargenumberofimagepatchestothenetworkandVisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.Convolutional Neural Networks

115

Conv layer 3:

Conv layer 4:

Conv layer 5:

VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.VisualizingandUnderstandingConvolutionalNetworksLayer 2Layer 1Layer 3Layer 4Layer 5Figure2.Visualizationoffeaturesinafullytrainedmodel.Forlayers2-5weshowthetop9activationsinarandomsubsetoffeaturemapsacrossthevalidationdata,projecteddowntopixelspaceusingourdeconvolutionalnetworkapproach.Ourreconstructionsarenotsamplesfromthemodel:theyarereconstructedpatternsfromthevalidationsetthatcausehighactivationsinagivenfeaturemap.Foreachfeaturemapwealsoshowthecorrespondingimagepatches.Note:(i)thethestronggroupingwithineachfeaturemap,(ii)greaterinvarianceathigherlayersand(iii)exaggerationofdiscriminativepartsoftheimage,e.g.eyesandnosesofdogs(layer4,row1,cols1).Bestviewedinelectronicform.116

Jonathan Richard Shewchuk

The V1 Visual Cortex

[The idea to exploit local connectivity in CNNs was inspired by the human visual system, as well as by
techniques used in image processing.]

[Show slides on the visual cortex, available from the CS 189 web page. Sorry, readers, there are too many
images to include here. The narration is below.]

[Neurologists can stick needles into individual neurons in animal brains. After a few hours the neuron dies,
but until then they can record its action potentials. In this way, biologists quickly learned how some of the
neurons in the retina, called retinal ganglion cells, respond to light. They have interesting receptive fields,
illustrated in the slides, which show that each ganglion cell receives excitatory stimulation from receptors in
a small patch of the retina but inhibitory stimulation from other receptors around it.]

[The signals from these cells propagate to the V1 visual cortex in the occipital lobe at the back of your
skull. The V1 cells proved harder to understand. David Hubel and Torsten Wiesel of the Johns Hopkins
University put probes into the V1 visual cortex of cats, but they had a very hard time getting any neurons to
fire there. However, a lucky accident unlocked the secret and ultimately won them the 1981 Nobel Prize in
Physiology.]
[Show video HubelWiesel.mp4, taken from YouTube: https://www.youtube.com/watch?v=IOHayh06LJ4 ]

[The glass slide happened to be at the particular orientation the neuron was sensitive to. The neuron doesn�t
respond to other orientations; just that one. So they were pretty lucky to catch that.]
[The simple cells act as line detectors and/or edge detectors by taking a linear combination of inputs from
retinal ganglion cells. It�s fascinating, and surely not a coincidence, that humans and CNNs for vision both
have edge detectors in their early layers.]

[The complex cells act as location-independent line detectors by taking inputs from many simple cells,
which are location dependent. It�s reminiscent of max pooling.]

[Later researchers showed that local connectivity runs through the V1 cortex by projecting certain images
onto the retina and using radioactive tracers in the cortex to mark which neurons had been firing. Those
images show that the neural mapping from the retina to V1 is retinatopic, i.e., locality preserving. This is a
big part of the inspiration for convolutional neural networks!]

[Unfortunately, as we go deeper into the visual system, layers V2 and V3 and so on, we know less and less
about what processing the visual cortex does.]

Unsupervised Learning: Principal Components Analysis

117

20 Unsupervised Learning: Principal Components Analysis

UNSUPERVISED LEARNING

We have sample points, but no labels!
No classes, no y-values, nothing to predict.
Goal: Discover structure in the data.

Examples:

� Clustering: partition data into groups of similar/nearby points.
� Dimensionality reduction: data often lies near a low-dimensional subspace (or manifold) in feature

space; matrices have low-rank approximations.
[Whereas clustering is about grouping similar sample points, dimensionality reduction is about iden-
tifying a continuous variation from sample point to sample point.]

� Density estimation: fit a continuous distribution to discrete data.

[When we use maximum likelihood estimation to fit Gaussians to sample points, that�s density esti-
mation, but we can also fit functions more complicated than Gaussians.]

PRINCIPAL COMPONENTS ANALYSIS (PCA) (Karl Pearson, 1901)

Goal: Given sample points in Rd, find k directions that capture most of the variation.
reduction.)

(Dimensionality

3dpca.pdf (ISL, Figure 12.2)
3D space on the left
principal component space.]

[Example of 3D points projected to 2D by PCA. The
is the

is the feature space, and the 2D space on the right

First principal componentSecond principal component?1.0?0.50.00.51.0?1.0?0.50.00.51.0118

Jonathan Richard Shewchuk

pcadigits.pdf [The (high-dimensional) MNIST digits projected to a 2D subspace (from
784D). Two dimensions aren�t enough to fully separate the digits, but observe that the digits
0 (red) and 1 (orange) are well on their way to being separated.]

Why?

� Reducing # of dimensions makes some computations cheaper, e.g., regression.
� Remove irrelevant dimensions to reduce overfitting in learning algs.

Like subset selection, but the �features� aren�t axis-aligned;
they�re linear combos of input features.

� Find a small basis for representing variations in complex things, e.g., faces, genes.

[Sometimes PCA is used as a preprocess before regression or classification for the first two reasons.]
Let X be n � d design matrix. [No fictitious dimension.]
i Xi = 0. (Replace X with ?X.)
From now on, assume X is centered:
[We center the data in the usual way: by computing the sample mean, then subtracting the sample mean
from each sample point.]

(cid:80)

[Let�s start by seeing what happens if we pick just one principal direction.]
Let w be a unit vector.
The orthogonal projection of point x onto vector w is �x = (x � w) w.

x

w

�x
If w not unit, �x = x � w

?w?2 w.

[The idea is that we�re going to pick the best direction w, then project all the data down onto w so we can
analyze it in a one-dimensional space. Of course, we lose a lot of information when we project down from
d dimensions to just one. So, suppose we pick several directions. Those directions span a subspace, and we
want to project points orthogonally onto the subspace. This is easy if the directions are orthogonal to each
other.]

IntroductionPrincipalComponentsAnalysisLaurensvanderMaatenandGeo?reyHinton,JMLR2008(MCLab)t-SNEOctober30,20144/33Unsupervised Learning: Principal Components Analysis

119

Given orthonormal directions v1, . . . , vk, �x = (cid:80)k
[The word �orthonormal� means they�re all mutually orthogonal and all have length 1.]

i=1(x � vi) vi.

v2

x

�x

v1

Often we want just the k principal coordinates x � vi in principal component space.
[Often we don�t actually want the projected point in Rd. Sometimes we do, but often we just want the
principal coordinates.]
X?X is square, symmetric, positive semidefinite, d � d matrix. [As it�s symmetric, its eigenvalues are real.]
Let 0 ? ?1 ? ?2 ? . . . ? ?d be its eigenvalues.
[sorted]
Let v1, v2, . . . , vd be corresponding orthogonal unit eigenvectors. These are the principal components.
[. . . and the most important principal components will be the ones with the greatest eigenvalues. I will show
you this in three different ways.]

PCA derivation 1: Fit a Gaussian to data with maximum likelihood estimation.
Choose k Gaussian axes of greatest variance.

MLEPCA.pdf
greatest eigenvalue is drawn.]

[A Gaussian fitted to sample points. The principle component with the

Recall that MLE estimates a covariance matrix �? = 1

n X?X.

[Presuming X is centered.]

PCA Alg:

� Center X.
� Optional: Normalize X. Units of measurement different?

� Yes: Normalize.

[Bad for principal components to depend on arbitrary choice of scaling.]

� No: Usually don�t.

[If several features have the same unit of measurement, but some of them have smaller variance
than others, that difference is usually meaningful. In particular, you should never normalize
image pixels individually.]
� Compute unit eigenvectors/values of X?X.

first choice120

Jonathan Richard Shewchuk

� Choose k. (Optional: based on the eigenvalue sizes.)
� For the best k-dimensional subspace, pick eigenvectors vd?k+1, . . . , vd.
� Compute the k principal coordinates x � vi of each training/test point.

[When we do this projection, we have two choices: we can project the original, un-centered training
data OR we can project the centered training data. But if we do the latter, we have to translate the test
data by the same vector we used to translate the training data when we centered it.]

[End of algorithm.]

% of variability =

d(cid:88)

?i

i=d?k+1
d(cid:88)

?i

i=1

variance.pdf [Plot of # of eigenvectors vs. percentage of sample variance captured for a
17D data set. In this example, just 3 eigenvectors capture 70% of the variance.]

normalize.pdf (ISL, Figure 12.4) [Projection of 4D data onto a 2D subspace. Each point
represents one metropolitan area. Normalized data at left; unnormalized data at right. The
arrows show the four original axes projected on the two principal components. When the
data are not normalized, rare occurrences like murder have little influence on the principal
directions. Which is better? It depends on whether you think that low-frequency events like
murder and rape should have a larger influence.]

[If you are using PCA as a preprocess for a supervised learning algorithm, there�s a more effective way to
choose k: validation.]

First Principal ComponentSecond Principal Component**************************************************?0.50.00.5MurderAssaultUrbanPopRapeScaled?3?2?10123?100?50050100150First Principal ComponentSecond Principal Component**************************************************?3?2?10123?0.50.00.5?100?50050100150?0.50.00.51.0?0.50.00.51.0MurderAssauUrbanPopRapeUnscaledUnsupervised Learning: Principal Components Analysis

121

PCA derivation 2: Find direction w that maximizes sample variance of projected data.
[In other words, when we project the data down, we don�t want it all to bunch up; we want to keep it as
spread out as possible.]

project.jpg [Points projected on a line. We wish to choose the orientation of the green line
to maximize the sample variance of the blue points.]

Find w that maximizes Var({ �X1, �X2, . . . , �Xn}) = 1
n

(cid:33)2

(cid:32)

n(cid:88)

i=1

Xi �

w
?w?

= 1
n

?Xw?2
?w?2

= 1
n

w?X?Xw
w?w
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
Rayleigh quotient of X?X and w

[This fraction is a well-known construction called the Rayleigh quotient. When you see it, you should smell
eigenvectors nearby. How do we maximize this?]
If w is an eigenvector vi of X?X, Ray. quo. = ?i
? of all eigenvectors, vd achieves maximum variance ?d/n.
One can show vd beats every other vector too.
[Because every vector w is a linear combination of eigenvectors, and so its Rayleigh quotient will be a
convex combination of eigenvalues. It�s easy to prove this, but I don�t have the time. For the proof, look up
�Rayleigh quotient� in Wikipedia.]
[So the top eigenvector gives us the best direction. But we typically want k directions. After we�ve picked
one direction, then we have to pick a second direction that�s orthogonal to the best direction. But subject to
that constraint, we again pick the direction that maximizes the sample variance.]
What if we constrain w to be orthogonal to vd? Then vd?1 is optimal.
[And if we need a third direction orthogonal to vd and vd?1, the optimal choice is vd?2. And so on.]

122

Jonathan Richard Shewchuk

PCA derivation 3: Find direction w that minimizes mean squared projection distance.

PCAanimation.gif [This is an animated GIF; unfortunately, the animation doesn�t work in
the PDF lecture notes. Find the direction of the black line for which the sum of squares of
the lengths of the red lines is smallest.]

[You can think of this as a sort of least-squares linear regression, with one subtle but important change. In-
stead of measuring the error in a fixed vertical direction, we�re measuring the error in a direction orthogonal
to the principal component direction we choose.]

projlsq.png, projpca.png [Least-squares linear regression vs. PCA. In linear regression,
the projection direction is always �vertical� (measured in the label coordinate); whereas in
PCA, the projection direction is orthogonal to the projection hyperplane. In both methods,
however, we minimize the sum of the squares of the projection distances.]

Find w that minimizes

n(cid:88)

i=1

(cid:13)(cid:13)(cid:13)Xi ? �Xi

(cid:13)(cid:13)(cid:13)

2 =

n(cid:88)

(cid:13)(cid:13)(cid:13)(cid:13)(cid:13)

Xi ?

2

(cid:13)(cid:13)(cid:13)(cid:13)(cid:13)
Xi � w
?w?2 w

n(cid:88)

=

i=1

i=1

(cid:32)

?
??????
?Xi?2 ?

Xi �

(cid:33)2?
??????

w
?w?

= constant ? n (variance from derivation 2).

Minimizing mean squared projection distance = maximizing variance.
[From this point, carry on with the same reasoning as derivation 2.]

Unsupervised Learning: Principal Components Analysis

123

europegenetics.pdf (Lao et al., Current Biology, 2008.) [Illustration of the first two prin-
cipal components of the single nucleotide polymorphism (SNP) matrix for the genes of
various Europeans. The design matrix has 2,547 people from these locations in Europe
(right), and 309,790 SNPs per person. Each SNP is binary, so think of it as 309,790 dimen-
sions of zero or one. The output (left) shows spots on the first two principal components
where there was a high density of projected people from a particular national type. What�s
amazing about this is how closely the projected genotypes resemble the geography of Eu-
rope.]

Eigenfaces

X contains n images of faces, d pixels each.
[If we have a 200 � 200 image of a face, we represent it as a vector of length 40,000, the same way we
represent the MNIST digit data.]
Face recognition: Given a query face, compare it to all training faces; find nearest neighbor in Rd.
[This works best if you have several training photos of each person you want to recognize, with different
lighting and different facial expressions.]
Problem: Each query takes ?(nd) time.
Solution: Run PCA on faces. Reduce to much smaller dimension d?.

Now nearest neighbors takes O(nd?) time.
[Possibly even less. We�ll talk about speeding up nearest-neighbor search at the end of the
semester. If the dimension is small enough, you can sometimes do better than linear time.]

[If you have 500 stored faces with 40,000 pixels each, and you reduce them to 40 principal components,
then each query face requires you to read 20,000 stored principal coordinates instead of 20 million pixels.]

124

Jonathan Richard Shewchuk

facerecaverage.jpg, facereceigen0.jpg, facereceigen119.jpg, facereceigen.jpg [Images of
the the eigenfaces with the 32 largest eigenvalues. The �average face� is the mean used
to center the data.]

Unsupervised Learning: Principal Components Analysis

125

eigenfaceproject.pdf [Images of a face (left) projected onto the first 4 and 50 eigenvectors, with
the average face added back. The 50-eigenvector image is blurry but good enough for face recog-
nition. (These projections are in feature space, not in principle components space. The principle
coordinates are what you would use in the nearest neighbor classifier.)]

For best results, equalize the intensity distributions first.

facerecequalize.jpg [Image equalization.]

[Eigenfaces encode both face shape and lighting. Some people say that the first 3 eigenfaces are usually all
about lighting, and you sometimes get better facial recognition by dropping the first 3 eigenfaces.]

[Eigenfaces are not a state-of-the-art face recognition algorithm, not even close. But inspecting these images
can give you some intuition about PCA.]

[Optional: Show Blanz�Vetter face morphing video (morphmod.mpg).]

[Blanz and Vetter use PCA in a more sophisticated way for 3D face modeling. They take 3D scans of
people�s faces and find correspondences between peoples� faces and an idealized model. For instance, they
identify the tip of your nose, the corners of your mouth, and other facial features, which is something the
original eigenface work did not do. Instead of feeding an array of pixels into PCA, they feed the 3D locations
of various points on your face into PCA. This works more reliably.]

126

Jonathan Richard Shewchuk

21 The Singular Value Decomposition; Clustering

THE SINGULAR VALUE DECOMPOSITION (SVD) [and its Application to PCA]

Problems with PCA: Computing X?X takes ?(nd2) time.

X?X is poorly conditioned ? numerically inaccurate eigenvectors.
[The SVD improves both these problems.]

[Earlier this semester, we talked about the eigendecomposition of a square, symmetric matrix. Unfortu-
nately, nonsymmetric matrices don�t have nice eigendecompositions, and non-square matrices don�t have
eigenvectors at all. Happily, there is a similar decomposition that works for all matrices, even if they�re not
symmetric and not square.]
Fact: Every X ? Rn�d has a singular value decomposition X = UDV ? of the form

X

U

=

=

u1

D
diagonal
?1
0

?2 . . .

un

0

0

?d

0

n � d

n � n

n � d

U?U = I = UU?

min{n,d}(cid:88)

V ?

=

?iuiv?
i
(cid:124)(cid:123)(cid:122)(cid:125)

i=1

rank 1
outer product
matrix

v1

vd

d � d

V ?V = I = VV ?
orthonormal vi�s are
right singular vectors of X

orthonormal ui�s are left singular vectors of X

[Draw this by hand; write summation at the right last. fullsvd.pdf ]

Diagonal entries ?1, . . . , ?min{n,d} of D are nonnegative singular values of X.

[By convention, singular values are never negative, but some of them might be zero.]

vi is an eigenvector of X?X w/eigenvalue ?2

Fact:
Proof: X?X = V D?U?UDV ? = V D?DV ?

i , so the SVD solves PCA.

which is an eigendecomposition of X?X, with each ?2

i on the diagonal of D?D.

[The columns of V are the eigenvectors of X?X, which are the principal components we need for PCA. The
SVD also tells us their eigenvalues, which are the squares of the singular values of X. That�s related to why
the SVD is more numerically stable: the ratios between singular values are smaller than the ratios between
eigenvalues, so it�s easier to stably compute the singular values of X than the eigenvalues of X?X.]
Important: Row i of UD gives the principal coordinates of sample point Xi (i.e., ?i, ? j, Xi � v j = ? jUi j).
[So we don�t need to explicitly compute the inner products Xi � v j; the SVD has already done it for us.]
Proof: XV = UDV ?V = UD, so (XV)i j = (UD)i j.

The Singular Value Decomposition; Clustering

127

The Compact SVD

Singular vectors with singular value zero are useless for PCA [and many other applications], motivating the
compact SVD. Let r be the rank of X.

X

=

=

U

u1

D
diagonal
?1
0

?2 . . .

?r

r � r

0

ur

n � d

n � r

U?U = I

=

r(cid:88)

i=1

?iuiv?
i

V ?

v1

vr

r � d

V ?V = I

[Draw this by hand. compsvd.pdf ]

?1, . . . , ?r are nonzero singular values of the full SVD, & have the same left/right singular vectors.

[There are r nonzero singular values, and we can express X with their singular vectors alone. A nice conse-
quence is that D is invertible. If X is a centered design matrix for sample points that all lie on a line, then
X has rank 1 and there is only one nonzero singular value. If the centered sample points span a subspace of
dimension r, X has rank r and there are r nonzero singular values.]

[We might save a fair amount of time by not computing the left and right singular vectors with singular value
zero. Observe that the columns of U are still orthogonal, but it is no longer true that UU? = I. The same
goes for V.]

Fact: We can find the k greatest singular values & corresponding vectors in O(ndk) time.

[So we can save time by computing some of the singular vectors without computing all of them.]
[There are approximate, randomized algorithms that are even faster, producing an approximate
SVD in O(nd log k) time. These are starting to become popular in algorithms for very big data.]
[ https://code.google.com/archive/p/redsvd/ ]

[In the next lecture, we will use the compact SVD to help us understand the Moore-Penrose pseudoinverse
and its application to least-squares linear regression.]

CLUSTERING

Partition data into clusters so points in a cluster are more similar than across clusters.
Why?

� Discovery: Find songs similar to songs you like; determine market segments
� Hierarchy: Find good taxonomy of species from genes
� Quantization: Compress a data set by reducing choices
� Graph partitioning: Image segmentation; find groups in social networks

128

Jonathan Richard Shewchuk

zito.pdf (from a talk by Michael Pane) [k-means clusters that classify Barry Zito�s base-
ball pitches. Here we discover that there really are distinct classes of baseball pitches.]

k-Means Clustering aka Lloyd�s Algorithm (Stuart Lloyd, 1957)

Goal: Partition n points into k disjoint clusters.

Assign each sample point Xi a cluster label yi ? [1, k].
Cluster i�s mean is �i = 1
ni

X j, given ni points in cluster i.

(cid:88)

y j=i

Find y that minimizes

k(cid:88)

(cid:88)

i=1

y j=i

(cid:13)(cid:13)(cid:13)X j ? �i

2

(cid:13)(cid:13)(cid:13)

.

[Sum of squared distances from points to their cluster means.]

NP-hard. Solvable in O(nkn) time. [Try every partition.]

k-means heuristic: Alternate between

(1) y j�s are fixed; update �i�s
(2) �i�s are fixed; update y j�s

Halt when step (2) changes no assignments.

[So, we have an assignment of points to clusters. We compute the cluster means. Then we reconsider the
assignment. A point might change clusters if some other�s cluster�s mean is closer than its own cluster�s
mean. Then repeat.]

Barry Zito60657075808590!150!100 !50   0  50 100 150!150!100 !50   0  50 100 150Start SpeedSide SpinBack Spin!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!4-SeamFastball2-SeamFastballChangeupSliderCurveballBlackRedGreenBlueLightBlueThe Singular Value Decomposition; Clustering

129

Step (1): One can show (calculus) the optimal �i is the mean of the points in cluster i.

[This is easy calculus, so I leave it as a short exercise.]

Step (2): The optimal y assigns each point X j to the closest mean �i.

[If there�s a tie, and one of the choices is for X j to stay in the
same cluster as the previous iteration, always take that choice.]

[. . . so both steps minimize the cost function, but they don�t optimize all the variables at once.]

2means.png [An example of 2-means. Odd-numbered steps reassign the data points.
Even-numbered steps compute new means.]

4meansanimation.gif [This is an animated GIF of 4-means with many points. Unfortu-
nately, the animation doesn�t work in the PDF lecture notes.]

130

Jonathan Richard Shewchuk

Both steps decrease objective fn unless they change nothing.
[Therefore, the algorithm never returns to a previous assignment.]
Hence alg. must terminate. [As there are only finitely many assignments.]
[This argument says that Lloyd�s algorithm never loops forever. But it doesn�t say anything optimistic about
the running time, because we might see O(kn) different assignments before we halt. In theory, one can
actually construct point sets in the plane that take an exponential number of iterations, but those don�t come
up in practice.]
Usually very fast in practice. Finds a local minimum, often not global.
[. . . which is not surprising, as this problem is NP-hard.]

4meansbad.png [An example where 4-means clustering fails.]

Getting started:

� Forgy method: choose k random sample points to be initial �i�s; go to (2).
� Random partition: randomly assign each sample point to a cluster; go to (1).
� k-means++: like Forgy, but biased distribution.

[Each �i is chosen with a preference for points far from previous �i�s.]

[k-means++ is a little more work, but it works well in practice and theory. Forgy seems to be better than
random partition, but Wikipedia mentions some variants of k-means for which random partition is better.]

For best results, run k-means multiple times with random starts.

kmeans6times.pdf (ISL, Figure 10.7) [Clusters found by running 3-means 6 times on the
same sample points, each time starting with a different random partition. The algorithm
finds three different local minima.]

320.9235.8235.8235.8235.8310.9The Singular Value Decomposition; Clustering

131

[Why did we choose that particular objective function to minimize? Partly because it is equivalent to mini-
mizing the following function.]

Equivalent objective fn: the within-cluster variation

Find y that minimizes

k(cid:88)

i=1

1
ni

(cid:88)

(cid:88)

y j=i

ym=i

(cid:13)(cid:13)(cid:13)X j ? Xm

(cid:13)(cid:13)(cid:13)

2

[At the minimizer, this objective function is equal to twice the previous one. It�s a worthwhile exercise to
show that�it�s harder than it looks. The nice thing about this expression is that it doesn�t include the means;
it�s a function purely of the sample points and the clusters we assign them to. So it�s more compelling.]

[before applying k-means]

Normalize the data?
Same advice as for PCA. Sometimes yes, sometimes no.
[If some features are much larger than others, they will tend to dominate the Euclidean distance. So if you
have features in different units of measurement, you probably should normalize them. If you have features
in the same unit of measurement, you usually shouldn�t, but it depends on context.]
[One difficulty with k-means is that you have to choose the number k of clusters before you start, and there
isn�t any reliable way to guess how many clusters will best fit the data. The next method, hierarchical
clustering, has the advantage in that respect. By the way, there is a whole Wikipedia article on �Determining
the number of clusters in a data set.�]

Hierarchical Clustering

Creates a tree; every subtree is a cluster.
[So some clusters contain smaller clusters.]

Bottom-up, aka agglomerative clustering:
start with each point a cluster; repeatedly fuse pairs.

Top-down, aka divisive clustering:
start with all pts in one cluster; repeatedly split it.

[When the input is a point set, agglomerative clustering is used much more in practice than divisive cluster-
ing. But when the input is a graph, it�s the other way around: divisive clustering is more common.]

132

Jonathan Richard Shewchuk

We need a distance fn for clusters A, B:

d(A, B) = max{d(w, x) : w ? A, x ? B}
d(A, B) = min{d(w, x) : w ? A, x ? B}
d(A, B) = 1
w?A
|A| |B|
d(A, B) = d(�A, �B)

complete linkage:
single linkage:
average linkage:
centroid linkage:
[The first three of these linkages work for any distance function, even if the input is just a matrix of distances
between all pairs of sample points. The centroid linkage only really makes sense if we�re using the Euclidean
distance.]

where �S is mean of S

x?B d(w, x)

(cid:80)

(cid:80)

Greedy agglomerative alg.:
Repeatedly fuse the two clusters that minimize d(A, B)
Naively takes O(n3) time.
[But for complete and single linkage, there are more sophisticated algorithms called CLINK and SLINK,
which run in O(n2) time. A package called ELKI has publicly available implementations.]

Dendrogram: Illustration of the cluster hierarchy (tree) in which the vertical axis encodes all the linkage
distances.

dendrogram.pdf (ISL, Figure 10.9) [Example of a dendrogram cut into 1, 2, or 3 clusters.]

Cut dendrogram into clusters by horizontal line according to your choice of # of clusters OR intercluster
distance.

[It�s important to be aware that the horizontal axis of a dendrogram has no meaning. You could swap some
treenode�s left subtree and right subtree and it would still be the same dendrogram. It doesn�t mean anything
that two leaves happen to be next to each other.]

024681002468100246810The Singular Value Decomposition; Clustering

133

[Comparison of average, complete (max), and single
linkages.pdf (ISL, Figure 10.12)
(min) linkages. Observe that the complete linkage gives the best-balanced dendrogram,
whereas the single linkage gives a very unbalanced dendrogram that is sensitive to outliers
(especially near the top of the dendrogram).]

[Probably the worst of these is the single linkage, because it�s very sensitive to outliers. Notice that if you
cut this example into three clusters, two of them have only one sample point. It also tends to give you a very
unbalanced tree.]

[The complete linkage tends to be the best balanced, because when a cluster gets large, the farthest point in
the cluster is always far away. So large clusters are more resistant to growth than small ones. If balanced
clusters are your goal, this is your best choice.]

[In most applications you probably want the average or complete linkage.]

Warning: centroid linkage can cause inversions where a parent cluster is fused at a lower height than its
children.

[So statisticians don�t like it, but nevertheless, centroid linkage is popular in genomics.]

[As a final note, all the clustering algorithms we�ve studied so far are unstable, in the sense that deleting
a few sample points can sometimes give you very different results. But these unstable heuristics are still
the most commonly used clustering algorithms. And it�s not clear to me whether a truly stable clustering
algorithm is even possible.]

Average LinkageComplete LinkageSingle Linkage134

Jonathan Richard Shewchuk

22 The Pseudoinverse; Better Generalization for Neural Nets

THE PSEUDOINVERSE AND THE SVD

[We�re done with unsupervised learning. For the rest of the semester, we go back to supervised learning.]

[The singular value decomposition can give us insight into the pseudoinverse and its use in least-squares
linear regression. If you attended Discussion Section 6, you worked through an explanation of this, but now
that I�ve introduced the compact SVD in Lecture 21, I�d like to summarize it.]
Let X be any n � d matrix. Let X = UDV ? be its compact SVD. Let r = rank X.
Recall that U ? Rn�r, D ? Rr�r is diagonal & invertible, V ? Rd�r, U?U = I, V ?V = I.
The Moore�Penrose pseudoinverse of X is X+ = V D?1U?.
[This is a better pseudoinverse than the one I defined in Lecture 10, not least because it�s always defined.]

It�s d � n.

Observe:
(1) XX+ = UU?, which is symmetric. Proof: XX+ = UDV ?V D?1U? = UDD?1U? = UU?.
(2) X+X = VV ?. [The proof is analogous to (1).]
(3) If r = n, then XX+ = In�n and X+
(4) If r = d, then X+X = Id�d and X+
(5) By (3), if X is invertible (r = n = d), X+ = X?1. [The pseudoinverse is the inverse when one exists.]
(6) These are compact SVDs: X+ = V D?1U?, X? = V DU?, (X+

is a right inverse. Proof: U is square, U?U = I ? UU? = I; use (1).
is a left inverse. [The proof is analogous to (3) and uses (2).]

)? = UD?1V ?.

[If a factorization has the form of a compact SVD, it is a compact SVD.]
X+

is like X? with the nonzero singular values inverted.

(7) Given a compact SVD X = UDV ?, null X = null V ?.

Proof: V ?w = 0 ? Xw = UDV ?w = 0 ? D?1U?UDV ?w = 0 ? V ?w = 0.
)? = null V ? = null X.

(8) By (6) & (7), null X+ = null U? = null X? and null (X+

So row X+ = col X and col X+ = row X.

(9) (1) & (2) give eigendecompositions: XX+ = [ U Unull ]

has the same four fundamental subspaces as X?.
, X+X = [ V Vnull ]
[Unull and Vnull have orthonormal column vectors spanning the null spaces of XX+

(cid:104) Ir�r 0
0 0
and X+X.]

(cid:105) (cid:104) U?
U?
null

(cid:105) (cid:104) V ?
V ?
null

(cid:104) Ir�r 0
0 0

X+

(cid:105)

(cid:105)

.

, XX+
(10) By (8) & (9), all have rank r: X, U, V, X+
(11) By (9), every w ? col U is an eigenvector of XX+
is identity map on col X.

As col U = col X, XX+

, X+X.

with eigenvalue 1; all other eigenvalues are 0.

[Symmetrically,] X+X is identity map on row X.

[In summary, the psuedoinverse is as close to an inverse of X as anything can be. Let�s visualize what the
pseudoinverse does. When you apply X to a vector in row X, you get a vector in col X; then when you
apply X+

to the result, you get the original vector back.]

col X

X
X+
Xv1 = ?1u1

X+(?1u1) = v1

u2

u1

row X

v2

v1

rowcol.pdf [The singular vectors are perpendicular, but we are viewing the planes from oblique angles.]

The Pseudoinverse; Better Generalization for Neural Nets

135

[If we think of X as a linear function that maps row X to col X, and we ignore the other dimensions of Rd
and Rn, then that linear function is a bijection. The inverse of that bijection is the pseudoinverse X+
Linear function f : row X ? col X, p (cid:55)? X p is a bijection.
Its inverse is f ?1 : col X ? row X, q (cid:55)? X+q.

.]

+

The r right singular vectors vi are an orthonormal basis for row X.
The r left singular vectors ui are an orthonormal basis for col X.
Xvi = ?iui.
[X maps each right singular vector to some scalar multiple of the corresponding left singular vector. The
corresponding singular value tells us how much longer the vector gets when we map it. X+
maps each left
singular vector to some scalar multiple of a right singular vector.]

ui = 1
?i

vi.

X

[Usually we don�t think of X as a function from row space to column space. Usually we think of X as a
function from some bigger space Rd to a bigger space Rn. In our figure above, X might be a 4 � 3 matrix, but
its rank is only two. Then X isn�t a bijection any more, and neither is its pseudoinverse X+
. So X maps every
point in R3 to a point on the plane col X. When you map a three-dimensional space down to two dimensions,
it can�t be a bijection, so X doesn�t have an inverse. Just a pseudoinverse.]

[You can think of X as a function that orthogonally projects a three-dimensional point down onto the row
space of X, then uses the bijection above to finish the mapping. Symmetrically, you can think of X+
as a
function that orthogonally projects a four-dimensional point down onto the column space of X, then uses the
inverse bijection. Here�s an illustration of mapping p to X p and q to X+q.]

q

col X

X p

u2

u1

X+q

rowcolpr.pdf

p

row X

v2

v1

[With the compact SVD, we can show that the pseudoinverse always gives a solution to least-squares linear
regression, even when X?X is singular.]
Theorem: A solution to the normal equations X?Xw = X?y is w = X+y.
Proof: X?Xw = X?XX+y = V DU?UDV ?V D?1U?y = V D2D?1U?y = V DU?y = X?y.
If the normal eq�ns have multiple solutions, w = X+y is the least-norm solution; i.e., it minimizes ?w? among
all solutions. [If you attended Discussion Section 6, you might have proven this yourself.]
[This way of solving the normal equations is very helpful when X?X is singular because n < d or the
sample points lie on a subspace of the feature space. But observe that if X has a very small singular value,
the reciprocal of that singular value will be very large and have a very large effect on w; but when that
singular value is exactly zero, it has no effect on w! So when we have a really tiny singular value, should we
pretend it is zero? Ridge regression implements this policy to some degree; review Discussion Worksheet 12
for details.]

136

Jonathan Richard Shewchuk

BETTER GENERALIZATION FOR NEURAL NETWORKS

[Classic methods for preventing overfitting, such as subset selection, ?2 regularization, and ensembles of
learners, sometimes help neural networks to generalize better to points they haven�t been trained on.]

(1) Get more data. [This is the best method. Andrej Karpathy writes that �It is a very common mistake to
spend a lot of engineering cycles trying to squeeze juice out of a small dataset when you could instead be
collecting more data.�]

(2) Data augmentation. Augment data set with modified versions of training points.

augmentation.pdf, (Bishop, Figure 9.1)
[Examples of data augmentation applied to an
original image (a). (b) Reflection. (c) Scaling. (d) Translation. (e) Rotation. (f) Changing
brightness and contrast. (g) Added noise. (h) Color shift.]

[You can see that these augmentations do not change the fact that the image should be classified as a cat.]

pixmix.pdf, (Hendrycks et al., �PixMix�, 2022) [More varieties of data augmentation.]

[Hendrycks et al. note that �For state-of-the-art models, data augmentation can improve clean accuracy [on
the test set] comparably to a 10� increase in model size. Further, data augmentation can improve out-of-
distribution robustness [on images from a distribution different than the training set] comparably to a 1,000�
increase in labeled data.�]

[One point they make is that while adding Gaussian noise is one augmentation that helps improve gener-
alization to new images, it�s even more effective to add artifacts that stimulate hidden units, such as the
hidden units that detect edges in an image. So their augmentation methods mix images with other images
that introduce spurious structure, not just Gaussian noise.]

2589.REGULARIZATION(a)(b)(c)(d)(e)(f)(g)(h)Figure9.1Illustrationofdatasetaugmentation,showing(a)theoriginalimage,(b)horizontalinversion,(c)scaling,(d)translation,(e)rotation,(f)brightnessandcontrastchange,(g)additivenoise,and(h)colourshift.Anexampleofapproach2isthetechniqueoftangentpropagation(Simardetal.,1992)inwhicharegularisationtermisaddedtotheerrorfunctionduringtraining.Thistermdirectlypenalizeschangesintheoutputresultingfromchangesintheinputvariablesthatcorrespondtooneoftheinvarianttransformations.Alimitationofthistechnique,inadditiontotheextracomplexityoftraining,iscanonlycopewithsmalltransformations(e.g.,translationsbylessthanapixel).Approach3isknownasdatasetaugmentation.Itisoftenrelativelyeasytoimplementandcanprovetobeveryeffectiveinpractice.Itisoftenappliedinthecontextofimageanalysisasitstraightforwardtocreatethetransformedtrainingdata.Figure9.1showsexamplesofsuchtransformationsappliedtoanimageofacat.Formedicalimagesofsofttissue,dataaugmentationcouldalsoincludecontinuous�rubbersheet�deformations(Ronneberger,Fischer,andBrox,2015).Forsequentialtrainingalgorithms,suchasstochasticgradientdescent,thedatasetcanbeaugmentedbytransformingeachinputdatapointbeforeitispresentedtothemodelsothat,ifthedatapointsarebeingrecycled,adifferenttransformation(drawnfromanappropriatedistribution)isappliedeachtime.Forbatchmethods,asimilareffectcanbeachievedbyreplicatingeachdatapointanumberoftimesandtransformingeachcopyindependently.Wecananalysetheeffectofusingaugmenteddatabyconsideringtransforma-tionsthatrepresentsmallchangestotheoriginalexamplesandthenmakingaTaylorexpansionoftheerrorfunctioninpowersofthemagnitudeofthetransformation(Bishop,1995c;Leen,1995;Bishop,2006).Thisleadstoaregularizederrorfunc-tioninwhichtheregularizerpenalizesthegradientofthenetworkoutputwithrespectMethodBaselineCutoutMixupCutMixPIXMIXCorruptionsmCE(#)50.0+0.051.5+1.548.0 2.051.5+1.530.5 19.5AdversariesError(#)96.5+0.098.5+1.097.4+0.997.0+0.592.9 3.9ConsistencymFR(#)10.7+0.011.9+1.29.5 1.212.0+1.35.7 5.0CalibrationRMSError(#)31.2+0.031.1 0.113.0 18.129.3 1.88.1 23.0AnomalyDetectionAUROC(")77.7+0.074.3 3.471.7 6.074.4 3.389.3+11.6Table1.PIXMIXcomprehensivelyimprovessafetymeasures,providingsigni?cantimprovementsoverstate-of-the-artbaselines.Weobservethatpreviousaugmentationmethodsintroducefewadditionalsourcesofstructuralcomplexity.Bycontrast,PIXMIXincorporatesfractalsandfeaturevisualizationsintothetrainingprocess,activelyexposingmodelstonewsourcesofstructuralcomplexity.We?ndthatPIXMIXisabletoimprovebothrobustnessanduncertaintyestimationandisthe?rstmethodtosubstantiallyimproveallexistingsafetymeasuresoverthebaseline.thatexistinghelpwithsomesafetymetricsbutharmoth-ers.Thisraisesthequestionofwhetherimprovingallsafetymeasuresispossiblewithasinglemodel.Whilepreviousaugmentationmethodscreateimagesthataredifferent(e.g.,translations)ormoreentropic(e.g.,additiveGaussiannoise),wearguethatanimportantunder-exploredaxisiscreatingimagesthataremorecomplex.Asopposedtoentropyordescriptivedif?culty,whichismax-imizedbypurenoisedistributions,structuralcomplexityisoftendescribedintermsofthedegreeoforganization[28].Aclassicexampleofstructurallycomplexobjectsisfrac-tals,whichhaverecentlyprovenusefulforpretrainingim-ageclassi?ers[22,35].Thus,aninterestingquestioniswhethersourcesofstructuralcomplexitycanbeleveragedtoimprovesafetythroughdataaugmentationtechniques.WeshowthatParetoimprovementsarepossiblewithPIXMIX,asimpleandeffectivedataprocessingmethodthatleveragespictureswithcomplexstructuresandsub-stantiallyimprovesallexistingsafetymeasures.PIXMIXconsistsofanewdataprocessingpipelinethatincorpo-ratesstructurallycomplex�dreamlike�images.Thesedreamlikeimagesincludefractalsandfeaturevisualiza-tions.We?ndthatfeaturevisualizationsareasuit-ablesourceofcomplexity,therebydemonstratingthattheyhaveusesbeyondinterpretability.Inextensiveexperi-ments,we?ndthatPIXMIXprovidessubstantialgainsonabroadrangeofexistingsafetymeasures,outperform-ingnumerouspreviousmethods.Codeisavailableatgithub.com/andyzoujm/pixmix.2.RelatedWorkRobustness.Out-of-distributionrobustnessconsidershowtomakeMLmodelsresistanttovariousformsofdatashiftattesttime.Geirhosetal.,2019[11]uncoveratexturebiasinconvolutionalnetworksandshowthattrainingondiversestylizedimagescanimproverobustnessattest-time.TheImageNet-C(orruptions)benchmark[15]consistsofdiverseimagecorruptionsknowntotrackrobustnessonsomerealworlddatashifts[13].ImageNet-CisusedtotestmodelsthataretrainedonImageNet[7]andisusedasaheld-out,moredif?culttestset.TheyalsointroduceImageNet-P(erturbations)formeasuringpredictionconsistencyundervariousnon-adversarialinputperturbations.OthershaveintroducedadditionalcorruptionsforevaluationcalledImageNet-C[33].TheImageNet-R(enditions)benchmarkmeasuresperformancedegradationundervariousrenditionsofobjectsincludingpaintings,cartoons,graf?ti,embroidery,origami,sculp-The Pseudoinverse; Better Generalization for Neural Nets

137

(3) Subset selection. [Recall Lecture 13.]

(4) ?2 regularization, aka weight decay.
Add ? ?w?2 to the cost/loss fn, where w is vector of all weights in network.
[w includes all the weights in all the weight matrices, rewritten as a vector.]

[We regularize for the same reason we do it in ridge regression: we suspect that overly large weights are
spurious.]

[With a neural network, it�s not clear whether penalizing the bias terms is bad or good. Penalizing the
bias terms has the effect, potentially positive, of drawing each ReLU or sigmoid unit closer to the center of
its nonlinear operating region. I would suggest to try both ways and use validation to decide whether you
should penalize the bias terms or not. Also, you could try using a different hyperparameter for the bias terms
than the ? you use for the other weights.]

Effect: step ?wi = ??

has extra term ?2?? wi

?J
?wi

Weight wi decays by factor 1 ? 2?? if not reinforced by training.

weightdecayoff.pdf, weightdecayon.pdf (ESL, Figure 11.4) Write �10 hidden units + soft-
max + cross-entropy loss.� [Examples of 2D classification without (left) and with (right)
weight decay. Observe that in the second example, the decision boundary (black) better
approximates the Bayes optimal boundary (dashed purple curve).]

[AlexNet is a famous example of a network that used both ?2 regularization and momentum. Just add the
?2 penalty to the cost function J and plug that cost function into the momentum algorithm from Lecture 18.
AlexNet set ? = 0.0005 and the momentum decay term to ? = 0.9. They adjusted ? manually throughout
training.]

(5) Train for a very long time.
[Andrej Karpathy: �I�ve often seen people tempted to stop the model
training when the validation loss seems to be leveling off. In my experience networks keep training for an
unintuitively long time. One time I accidentally left a model training during the winter break and when I got
back in January it was SOTA (�state of the art�).9]

9http://karpathy.github.io/2019/04/25/recipe/

Neural Network - 10 Units, No Weight Decay...............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooTraining Error: 0.100Test Error:       0.259Bayes Error:    0.210Neural Network - 10 Units, Weight Decay=0.02 ...............Neural Network - 10 Units, Weight Decay=0.02 ...............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooTraining Error: 0.160Test Error:       0.223Bayes Error:    0.210138

Jonathan Richard Shewchuk

(6) Ensemble of neural nets. Random initial weights + SGD + (optionally) bagging.
[Ensembles work well for neural nets, reportedly improving test accuracy by 2�3%. Random initial weights
and random minibatches ensure that each neural net finds a different local minimum. If you have finite
training data, validate to see if bagging helps or not. Obviously, ensembles of neural nets are very slow.]

For speed, sometimes the ensembles share the same early layers.
[Then only the last layers of each neural network are trained separately.]

(7) Dropout emulates an ensemble in one network.

dropout1.pdf, dropout2.pdf

During training, temporarily disable a random subset of the units, along with all edges in and out.

� No forward signal, no weight updates for edges in or out of disabled unit.
� Disable each hidden unit with probability (typically) 0.5.
� Disable each input unit with probability (typically) 0.2.
� Disable a different random subset for each SGD minibatch.

After training, before testing: enable all units. If units in a layer were disabled with probability p, multiply
all edge weights out of that layer by p.
[When we disabled units, the edge weights had to grow large to make up for their disabled neighbors. At
test time, all units and edges are enabled, so we have to reduce the weights to compensate.]
[Dropout gives an effect similar to averaging over multiple neural networks, but it�s faster to train. Dropout
usually gives better generalization than ?2 regularization. Geoff Hinton and his co-authors give an example
where they trained a network to classify MNIST digits. Their network without dropout had a 1.6% test error;
it improved to 1.3% with dropout on the hidden units only; and further improved to 1.1% with dropout on
the input units too.]

[Recall Karl Lashley�s rat experiments, where he tried to make rats forget how to run a maze by introduc-
ing lesions in their cerebral cortexes, and it didn�t work. He concluded that the knowledge is distributed
throughout their brains, not localized in one place. Dropout is a way to force neural networks to distribute
knowledge throughout the weights.]

Double Descent

[Early neural network researchers sometimes struggled with their networks falling into bad local minima
and failing to achieve low training errors. But with experience and greater computational power, we�ve
discovered that these problems can usually be solved simply by adding more units to every hidden layer. We
call this �making the network wider.� Sometimes you also have to add more layers. But if your layers are
wide enough, and there are enough of them, a well-designed neural network can typically output exactly the
correct label for every training point, which implies that you�re at a global minimum of the cost function.]

The Pseudoinverse; Better Generalization for Neural Nets

139

Hidden layers are wide enough + numerous enough ? network output can interpolate the label (�y = y) for
every training pt ? find global minimum of cost fn.

[I have pointed out that if you use sigmoid or softmax output units, you can�t set the labels to exactly 1 or 0
and achieve interpolation, as sigmoid and softmax outputs are strictly between 0 and 1; but with labels like
0.1 and 0.9, interpolating the labels is a realistic goal! And the linear output units used for regression can
interpolate arbitrary numbers. Bottom line: if you fall into a bad local minimum, your network is too small.]

[One reason it took so long to make this discovery is that researchers believed that having too many weights
in a neural network would cause overfitting. It turns out that�s only half true. Empirically, we sometimes
observe a phenomenon called �double descent,� illustrated below.]

doubledescent.pdf (Nakkiran et al., �Deep Double Descent�)
[A classic double descent
curve (solid blue) for test error. The horizontal axis indicates the number of units in each
hidden layer of a residual neural network used for image recognition, and the vertical axis
measures the the test error (solid curve) and training error (dashed curve).]

[Consider the solid blue curve, showing the test error as the width of a network increases. The horizontal
axis is the number of units per hidden layer. As that number increases, at left the test error exhibits the
classic U-shaped bias-variance �tradeoff.� But when we pass the point where the network is interpolating
the labels and continue to add more weights, we sometimes see a second �descent,� where the test error
starts to decrease again and ultimately gets even lower than before! The peak in the middle of the curve
tends to be larger when there is more noise in the labels. Observe that the test error continues to fall even
after the training error is zero. The takeaway is, �bigger models are often better.�]

[The currently accepted explanation for double descent, per Nakkiran et al., is that �at the interpolation
threshold . . . the model is just barely able to fit the training data; forcing it to fit even slightly-noisy or mis-
specified labels will destroy its global structure, and result in high test error. However for over-parameterized
models, there are many interpolating models that fit the training set, and SGD is able to find one that
�absorbs� the noise while still performing well on the distribution.�]

[Double descent has also been observed in decision trees and even in linear regression where we add random
features to the training points (thereby adding more weights to the linear regression model).]

DEEPDOUBLEDESCENT:WHEREBIGGERMODELSANDMOREDATAHURTPreetumNakkiran?HarvardUniversityGalKaplun�HarvardUniversityYaminiBansal�HarvardUniversityTristanYangHarvardUniversityBoazBarakHarvardUniversityIlyaSutskeverOpenAIABSTRACTWeshowthatavarietyofmoderndeeplearningtasksexhibita�double-descent�phenomenonwhere,asweincreasemodelsize,performance?rstgetsworseandthengetsbetter.Moreover,weshowthatdoubledescentoccursnotjustasafunctionofmodelsize,butalsoasafunctionofthenumberoftrainingepochs.Weunifytheabovephenomenabyde?ninganewcomplexitymeasurewecalltheeffectivemodelcomplexityandconjectureageneralizeddoubledescentwithrespecttothismeasure.Furthermore,ournotionofmodelcomplexityallowsustoidentifycertainregimeswhereincreasing(evenquadrupling)thenumberoftrainsamplesactuallyhurtstestperformance.1INTRODUCTIONFigure1:Left:Trainandtesterrorasafunctionofmodelsize,forResNet18sofvaryingwidthonCIFAR-10with15%labelnoise.Right:Testerror,shownforvaryingtrainepochs.AllmodelstrainedusingAdamfor4Kepochs.Thelargestmodel(width64)correspondstostandardResNet18.Thebias-variancetrade-offisafundamentalconceptinclassicalstatisticallearningtheory(e.g.,Hastieetal.(2005)).Theideaisthatmodelsofhighercomplexityhavelowerbiasbuthighervari-ance.Accordingtothistheory,oncemodelcomplexitypassesacertainthreshold,models�over?t�withthevariancetermdominatingthetesterror,andhencefromthispointonward,increasingmodelcomplexitywillonlydecreaseperformance(i.e.,increasetesterror).Henceconventionalwisdominclassicalstatisticsisthat,oncewepassacertainthreshold,�largermodelsareworse.�However,modernneuralnetworksexhibitnosuchphenomenon.Suchnetworkshavemillionsofparameters,morethanenoughto?tevenrandomlabels(Zhangetal.(2016)),andyettheyperformmuchbetteronmanytasksthansmallermodels.Indeed,conventionalwisdomamongpractitionersisthat�largermodelsarebetter��(Krizhevskyetal.(2012),Huangetal.(2018),Szegedyetal.?WorkperformedinpartwhilePreetumNakkiranwasinterningatOpenAI,withIlyaSutskever.Weespe-ciallythankMikhailBelkinandChristopherOlahforhelpfuldiscussionsthroughoutthiswork.CorrespondenceEmail:preetum@cs.harvard.edu�Equalcontribution1140

Jonathan Richard Shewchuk

23 Residual Networks; Batch Normalization; AdamW

TRAINING DEEP NETWORKS

Most influential ideas: ResNets, batch normalization, layer normalization.
[These ideas enable deep networks to train. They reduce the likelihood of encountering the vanishing gradi-
ent or exploding gradient problems, but don�t quite eliminate them.]

Batch Normalization

[Batch normalization has played a huge role in making it easier to train very deep neural networks since its
introduction in 2015, and it�s still a mainstay today. It seems to make the cost function uglier, though; when
you can train a network without it, it might generalize better.]

Recall batch normalization from Homework 6: For a vector a of activations,
� a batch-norm layer learns parameters ?i and ?i for each activation ai;
� calculate the sample mean � of a and the sample variances ?2
� for some small ?, the layer outputs vector z with zi = ?i + ?i

i of ai over a minibatch;
ai ? �i
(cid:113)
+ ?

.

?2
i

[We didn�t say much in the homework about where batch normalization layers are used. There is some
disagreement over whether it�s best to place them before or after a nonlinear activation function. Both ways
are commonly used. The only clear rules are to never use batch normalization for outputs, and never use
ReLUs for inputs.]

hidden units

batch normalization

ReLUs

ReLUs

batch normalization

hidden units

batchrelu.pdf [The original authors proposed the left version.]

[Batch normalization is often applied to image data in convolutional neural networks, but not quite like this.
We do not normalize each pixel separately. Remember that in a CNN, we rely heavily on relationships
between adjacent pixels. Normalizing each pixel separately could introduce spurious edges and ruin the
network�s ability to recognize images. But we can normalize an entire image as a whole, and we can
normalize each channel separately.]

Batch-norm for images: compute mean & variance over all images in minibatch AND all pixels in each
image. One ? and one ? per channel.

Layer normalization: compute mean & variance over all hidden units (all pixels and all channels), but not
(necessarily) over images. One ? and one ? per image.

[Layer normalization ensures that for any one image and any one layer of hidden units, the hidden unit
vector will lie on a sphere with center ? and radius ?. Layer normalization is easier to parallelize than batch
normalization, and it is particularly useful in recurrent neural networks.]

Residual Networks; Batch Normalization; AdamW

141

Residual Neural Networks (ResNets)

[Look at this famous figure depicting two-dimensional cross sections through the cost function of deep
convolutional neural networks. At right is the cost for a residual neural network. At left is the cost if we
remove the �residual connections� that characterize residual networks. You can guess which one is easier to
optimize.]

lossskip.pdf, (Hao Li et al., 2018)

Idea: design a network with layers that can easily represent the identity fn, e.g., when all weights are zero.
[There are two observations that help to motivate this idea.]
Motivation 1: Networks with nonlinear activations have great difficulty representing the identity. Let�s fix
that.
[If a hidden layer has negative unit values, a subsequent hidden layer with ReLUs cannot replicate those
values. You might be able to replicate them at a linear output layer if you�re very clever. Instead, let�s
redesign our networks to make it easy.]
Motivation 2: Consider a linear neural network �y = WLWL?1 � � � W1x with square matrices Wi.
Given an n � d design matrix X and an n � d label matrix Y, solve this linear regression problem by gradient
descent [batch or stochastic].

Find matrices that minimize J(WL, WL?1, . . . , W1) = ?WLWL?1 � � � W1X? ? Y ??2
F.

The cost fn is very smooth around (I, I, . . . , I), but much more complicated in regions close to (0, 0, . . . , 0).
[There�s a paper showing that near the identity matrices, every critical point of the cost function is a global
minimum�much like in the figure at top right. So even a simple linear neural network suffices to show
some of the behavior in the figure.]
Any network satisfying WLWL?1 � � � W1 = Y ?(X?)
If L is sufficiently large, there is a solution or approximate solution where each Wi is �close� to I.
[If you multiply together enough matrices that are close to the identity matrix, you can obtain any square,
invertible matrix, and you can get very close to any square matrix. So for sufficiently many layers, there are
solutions in the �nice� region of the cost function. SGD starting from the identity network will find them.]

is a solution.

+

VisualizingtheLossLandscapeofNeuralNetsHaoLi1,ZhengXu1,GavinTaylor2,ChristophStuder3,TomGoldstein11UniversityofMaryland,CollegePark2UnitedStatesNavalAcademy3CornellUniversity{haoli,xuzh,tomg}@cs.umd.edu,taylor@usna.edu,studer@cornell.eduAbstractNeuralnetworktrainingreliesonourabilityto?nd�good�minimizersofhighlynon-convexlossfunctions.Itiswell-knownthatcertainnetworkarchitecturedesigns(e.g.,skipconnections)producelossfunctionsthattraineasier,andwell-chosentrainingparameters(batchsize,learningrate,optimizer)produceminimiz-ersthatgeneralizebetter.However,thereasonsforthesedifferences,andtheireffectsontheunderlyinglosslandscape,arenotwellunderstood.Inthispaper,weexplorethestructureofneurallossfunctions,andtheeffectoflosslandscapesongeneralization,usingarangeofvisualizationmethods.First,weintroduceasimple�?lternormalization�methodthathelpsusvisualizelossfunctioncurvatureandmakemeaningfulside-by-sidecomparisonsbetweenlossfunctions.Then,usingavarietyofvisualizations,weexplorehownetworkarchitectureaffectsthelosslandscape,andhowtrainingparametersaffecttheshapeofminimizers.1IntroductionTrainingneuralnetworksrequiresminimizingahigh-dimensionalnon-convexlossfunction�ataskthatishardintheory,butsometimeseasyinpractice.DespitetheNP-hardnessoftraininggeneralneurallossfunctions[2],simplegradientmethodsoften?ndglobalminimizers(parametercon?gurationswithzeroornear-zerotrainingloss),evenwhendataandlabelsarerandomizedbeforetraining[42].However,thisgoodbehaviorisnotuniversal;thetrainabilityofneuralnetsishighlydependentonnetworkarchitecturedesignchoices,thechoiceofoptimizer,variableinitialization,andavarietyofotherconsiderations.Unfortunately,theeffectofeachofthesechoicesonthestructureoftheunderlyinglosssurfaceisunclear.Becauseoftheprohibitivecostoflossfunctionevaluations(whichrequiresloopingoverallthedatapointsinthetrainingset),studiesinthis?eldhaveremainedpredominantlytheoretical.(a)withoutskipconnections(b)withskipconnectionsFigure1:ThelosssurfacesofResNet-56with/withoutskipconnections.Theproposed?lternormalizationschemeisusedtoenablecomparisonsofsharpness/?atnessbetweenthetwo?gures.32ndConferenceonNeuralInformationProcessingSystems(NIPS2018),Montr�al,Canada.arXiv:1712.09913v3  [cs.LG]  7 Nov 2018142

Jonathan Richard Shewchuk

[Of course, you would never do linear regression this way. But when we add nonlinear activation functions
such as ReLUs between the matrices, the phenomenon persists that the cost function is ugly near the origin
and can be relatively nice where the network is computing an function near the identity function at each
layer.]

Takeaways:

� Initializing near I (plus small random weights) is better than initializing near zero.
� More layers makes it more likely there�s a solution in a �nice� part of the cost fn.

ResNets use residual connections aka skip connections to add hidden layer values to subsequent layers.
Networks are constructed by repeating one of these motifs.

most common motif today

original ResNet motif

h[i]

hidden units

h[i]

n
o
i
t
c
e
n
n
o
c

l
a
u
d
i
s
e
r

z[ j]

mix of linear layers
(fully-connected,
convolutional,
batch normalization,
etc.),
nonlinear activations,
and hidden units

n
o
i
t
c
e
n
n
o
c

l
a
u
d
i
s
e
r

h[ j] = h[i] + z[ j]

hidden units

motifs.pdf

z[ j]

h[ j]

activations

[The motif on the right was used by the original ResNet paper, with ReLU activations after the residual
connection. But the motif on the left seems to be more popular now.]
If all weights are zero, left motif sets h[ j] = h[i] by default.
Right motif with ReLUs copies all positive units but zeros out negative ones.

[A big advantage of the motif on the left is that if it is advantageous to send some hidden unit values from
an early layer to a later layer unchanged, the network has the opportunity to learn to do that. The motif on
the right can do that with positive hidden unit values if it uses ReLU activations.]

If the �ideal� mapping from h[i] to h[ j] is expressed by a function f , the left motif is trying to learn f (h) ? h
(which we hope is small).

Residual Networks; Batch Normalization; AdamW

143

[The first ResNets were CNNs for image classification. The authors won first place in the 2015 ImageNet
Large Scale Visual Recognition Challenge with an ensemble of six ResNets, two of which had 152 layers.
This was the biggest advance in neural network vision performance since the AlexNet paper that changed
modern computer vision in 2012. Here is the building block for one of their smaller ResNets, a 34-layer
model. At right is the authors� schematic of the whole ResNet-34 network with normalizations omitted.]

h[i]

a[i+1]

h[i+1]

a[i+2]

n
o
i
t
c
e
n
n
o
c

l
a
u
d
i
s
e
r

64 channels �56 � 56

3 � 3 convolutions

64 � 64 � 3 � 3 weights

64 channels �56 � 56

batch normalization

ReLUs

ResNet-34 building block
(two convolutional layers)

64 channels �56 � 56

3 � 3 convolutions

64 � 64 � 3 � 3 weights

64 channels �56 � 56

batch normalization

ReLUs

h[i+2]

64 channels �56 � 56

resnet34.pdf, resnet34he.pdf

[Observe that there is a layer of ReLU activations in the middle of the motif, with linear convolutions
before and after it. Each convolution is followed by a batch normalization layer. The biggest ResNet in
the competition-winning ensemble, with 152 layers, uses a motif with three convolution layers, three batch
normalization layers, and three ReLU layers�two within the residual connection and one after.]

[Below is an example of the building block for a more modern convolutional ResNet, called ConvNeXt,
produced by a collaboration between Facebook and Berkeley. Unlike the original ResNet, it does not place
an activation function after the residual connection.]
[The authors found that one layer normalization step per three convolutional layers suffices, whereas the
original ResNets used a batch normalization step after every convolutional layer. Instead of a ReLU, they
use a GELU, which stands for Gaussian Error Linear Unit. It�s similar in shape to a ReLU but it�s smooth,
with no discontinuity. It appears to give better test accuracy than ReLUs in some circumstances, though not
all. It is popular in transformers for speech generation.]

7x7 conv, 64, /2pool, /23x3 conv, 643x3 conv, 643x3 conv, 643x3 conv, 643x3 conv, 643x3 conv, 643x3 conv, 128, /23x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 256, /23x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 512, /23x3 conv, 5123x3 conv, 5123x3 conv, 5123x3 conv, 5123x3 conv, 512avg poolfc 1000image3x3 conv, 5123x3 conv, 643x3 conv, 64pool, /23x3 conv, 1283x3 conv, 128pool, /23x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 256pool, /23x3 conv, 5123x3 conv, 5123x3 conv, 512pool, /23x3 conv, 5123x3 conv, 5123x3 conv, 5123x3 conv, 512pool, /2fc 4096fc 4096fc 1000imageoutput size: 112output size: 224output size: 56output size: 28output size: 14output size: 7output size: 1VGG-1934-layer plain7x7 conv, 64, /2pool, /23x3 conv, 643x3 conv, 643x3 conv, 643x3 conv, 643x3 conv, 643x3 conv, 643x3 conv, 128, /23x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 1283x3 conv, 256, /23x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 2563x3 conv, 512, /23x3 conv, 5123x3 conv, 5123x3 conv, 5123x3 conv, 5123x3 conv, 512avg poolfc 1000image34-layer residualFigure3.ExamplenetworkarchitecturesforImageNet.Left:theVGG-19model[41](19.6billionFLOPs)asareference.Mid-dle:aplainnetworkwith34parameterlayers(3.6billionFLOPs).Right:aresidualnetworkwith34parameterlayers(3.6billionFLOPs).Thedottedshortcutsincreasedimensions.Table1showsmoredetailsandothervariants.ResidualNetwork.Basedontheaboveplainnetwork,weinsertshortcutconnections(Fig.3,right)whichturnthenetworkintoitscounterpartresidualversion.Theidentityshortcuts(Eqn.(1))canbedirectlyusedwhentheinputandoutputareofthesamedimensions(solidlineshortcutsinFig.3).Whenthedimensionsincrease(dottedlineshortcutsinFig.3),weconsidertwooptions:(A)Theshortcutstillperformsidentitymapping,withextrazeroentriespaddedforincreasingdimensions.Thisoptionintroducesnoextraparameter;(B)TheprojectionshortcutinEqn.(2)isusedtomatchdimensions(doneby1?1convolutions).Forbothoptions,whentheshortcutsgoacrossfeaturemapsoftwosizes,theyareperformedwithastrideof2.3.4.ImplementationOurimplementationforImageNetfollowsthepracticein[21,41].Theimageisresizedwithitsshortersideran-domlysampledin[256,480]forscaleaugmentation[41].A224?224cropisrandomlysampledfromanimageoritshorizontal?ip,withtheper-pixelmeansubtracted[21].Thestandardcoloraugmentationin[21]isused.Weadoptbatchnormalization(BN)[16]rightaftereachconvolutionandbeforeactivation,following[16].Weinitializetheweightsasin[13]andtrainallplain/residualnetsfromscratch.WeuseSGDwithamini-batchsizeof256.Thelearningratestartsfrom0.1andisdividedby10whentheerrorplateaus,andthemodelsaretrainedforupto60?104iterations.Weuseaweightdecayof0.0001andamomentumof0.9.Wedonotusedropout[14],followingthepracticein[16].Intesting,forcomparisonstudiesweadoptthestandard10-croptesting[21].Forbestresults,weadoptthefully-convolutionalformasin[41,13],andaveragethescoresatmultiplescales(imagesareresizedsuchthattheshortersideisin{224,256,384,480,640}).4.Experiments4.1.ImageNetClassi?cationWeevaluateourmethodontheImageNet2012classi?-cationdataset[36]thatconsistsof1000classes.Themodelsaretrainedonthe1.28milliontrainingimages,andevalu-atedonthe50kvalidationimages.Wealsoobtaina?nalresultonthe100ktestimages,reportedbythetestserver.Weevaluatebothtop-1andtop-5errorrates.PlainNetworks.We?rstevaluate18-layerand34-layerplainnets.The34-layerplainnetisinFig.3(middle).The18-layerplainnetisofasimilarform.SeeTable1forde-tailedarchitectures.TheresultsinTable2showthatthedeeper34-layerplainnethashighervalidationerrorthantheshallower18-layerplainnet.Torevealthereasons,inFig.4(left)wecom-paretheirtraining/validationerrorsduringthetrainingpro-cedure.Wehaveobservedthedegradationproblem-the4144

Jonathan Richard Shewchuk

n
o
i
t
c
e
n
n
o
c

l
a
u
d
i
s
e
r

h[i]

a[i+1]

h[i+1]

a[i+2]

h[i+2]

a[i+3]

h[i+3]

96 channels �J � K

depthwise convolution
(channels connect one-to-one)
96 � 7 � 7 weights

96 channels �J � K

7 � 7 convolutions

layer normalization

96 channels �J � K

1 � 1 convolutions

96 � 384 weights

384 channels �J � K

GELUs

384 channels �J � K

1 � 1 convolutions

384 � 96 weights

96 channels �J � K

ConvNeXt building block
(three convolutional layers)

96 channels �J � K

convnext.pdf

[The authors found that they get better test accuracy if they use 7 � 7 convolution filters, not just smaller
ones. But these filters use depthwise convolution, which means that each output channel receives input from
only one output channel. By contrast, traditional convolutional layers connect every output channel to every
input channel. The advantage of depthwise convolution is a big savings in weights and time.]

[Another interesting design choice is the use of an inverted bottleneck, where they temporarily increase the
number of channels from 96 to 384 and then decrease it again to 96. This creates very wide layers that lead
to better generalization to test data. To prevent the number of weights from exploding, they use the odd idea
of a 1 � 1 convolution going into and out of the 384 channels. This simply means that there is an all-to-all
connection from 96 channels to 384 channels, but there are no connections between adjacent �pixels� in
the activation maps. Note that the 1 � 1 convolutions are not depthwise convolutions! A 1 � 1 depthwise
convolution would not permit any mixing of information at all.]

[The combination of depthwise convolutions and 1 � 1 convolutions causes channel mixing to be separated
from spatial mixing. Both kinds of mixing take place, but they take place in different convolutional layers.
This permits us to greatly reduce the number of weights while still allowing information to eventually flow
everywhere.]

Residual Networks; Batch Normalization; AdamW

145

AdamW

�Adaptive moment estimation with weight decay.�
An optimization method faster than SGD. Warning: may reduce test accuracy.
Let J be losses summed over a minibatch. Let wi be a weight. Intuition:

gives more useful information than its relative magnitude.

� Sign of

?J
?wi
� We change wi slowly if

?J
?wi

changes sign frequently; otherwise stay fast.

[Roughly speaking, each weight has its own learning rate.]

� Keep exponential moving averages mi of

?J
?wi
ri. Typically mi/

[first moment] and ri of
?

?J
?wi
ri ? �1, but smaller if sign (?J/?wi) changes often.

[second raw moment].

?

(cid:32)

(cid:33)2

Each step has ?wi ? ?mi/

m ? 0; r ? 0; t = 0
repeat

t ? t + 1
g ? ?J(w)
m ? ? m + (1 ? ?) g
r ? ? r + (1 ? ?)g ? g
�m ? m/(1 ? ?t)
�r ? r/(1 ? ?t)

?i, wi ? wi ? ?

(cid:32)

[correcting bias]
? �mi?
�ri + ?

+ ?wi

(cid:33)

[elementwise multiplication; square each component of g]

[correcting bias, as m was initialized to zero]

Typical parameters: ? = 0.001, ? = 0.9, ? = 0.9999, ? = 10?8.
Weight decay term ? regularizes; set by validation. If ? = 0, it�s just called Adam.

sgdadamgodoy.png (Daniel Godoy) [Left: 50 steps of SGD (with 16-point minibatches)
don�t get very close to the minimum (red). Right: 50 steps with Adam.]

146

Jonathan Richard Shewchuk

24 Boosting; Nearest Neighbor Classification

ADABOOST (Yoav Freund and Robert Schapire, 1997)

AdaBoost (�adaptive boosting�) is an ensemble method for classification (or regression) that

� reduces bias [compare with random forests and other ensembles, which reduce variance];
� trains learners on weighted sample points [like bagging];
� uses different weights for each learner;
� increases weights of misclassified training points;
� gives bigger votes to more accurate learners.

Input: n � d design matrix X, vector of labels y ? Rn with yi = �1.
Ideas:

� Train T classifiers G1, . . . , GT .
[�T � stands for �trees�]
� Weight for training point Xi in Gt grows according to how many of G1, . . . , Gt?1 misclassified it.

[Moreover, if Xi is misclassified by very accurate learners, its weight grows even more.]
[And, the weight shrinks every time Xi is correctly classified by a learner.]
� Train Gt to try harder to correctly classify training pts with larger weights.
� Metalearner is a linear combination of learners. For test point z, M(z) = (cid:80)T

t=1 ?tGt(z).

Each Gt is �1, but M is continuous. Return sign of M(z).

[Remember that in the previous lecture on ensemble methods, I talked briefly about how to assign different
weights to training points. It varies for different learning algorithms. For example, in regression we usually
modify the risk function by multiplying each point�s loss function by its weight. In a soft-margin support
vector machine, we modify the objective function by multiplying each point�s slack by its weight.]
[Boosting works with many learning algorithms, but it was originally developed for decision trees, and
boosted decision trees are very popular and successful. To weight points in decision trees, we use a weighted
entropy where instead of computing the proportion of points in each class, we compute the proportion of
weight in each class.]
In iteration T , what classifier GT and coefficient ?T should we choose? Pick a loss fn L(prediction, label).

Find GT & ?T that minimize

Risk = 1
n

n(cid:88)

i=1

L(M(Xi), yi),

M(Xi) =

T(cid:88)

t=1

?tGt(Xi).

AdaBoost metalearner uses exponential loss function

L(�?, ?) = e?�?? =

?
???
???

e?�? ? = +1
e�?
? = ?1

Important: label ? is binary, Gt is binary, but �? = M(Xi) is continuous!
[This loss function is for the metalearner only. We will discover later that the ideal cost function for each
individual learner Gt is just the total weight of the misclassified points. In practice, our individual learners
are often classification algorithms like decision trees that don�t explicitly try to minimize any loss function
at all. Even when the practice doesn�t match the theory, boosting still usually works quite well.]

[The exponential loss function has the advantage that it pushes hard against badly misclassified points. So
it�s usually better than the squared error loss function for classification in a metalearner. It�s similar to why
in neural networks we prefer the cross-entropy loss function over the squared error for sigmoid outputs.]

n � Risk =

=

=

n(cid:88)

i=1
n(cid:88)

i=1
n(cid:88)

Boosting; Nearest Neighbor Classification

147

L(M(Xi), yi) =

e?yi M(Xi)

n(cid:88)

i=1

exp

?

???????

?yi

T(cid:88)

t=1

?tGt(Xi)

?

???????

=

n(cid:88)

T(cid:89)

i=1

t=1

e??tyiGt(Xi) ?

yiGt(Xi) = �1
if ?1, Gt misclassifies Xi

=

T ?1(cid:89)

t=1

e??tyiGt(Xi)

i e??T yiGT (Xi), where w(T )
w(T )
(cid:88)

(cid:88)

i

w(T )
i

+ e?T

w(T )
i

i=1
= e??T

yi(cid:44)GT (Xi)

= e??T

yi=GT (Xi)
n(cid:88)

w(T )
i

i=1

+ (e?T ? e??T )

(cid:88)

yi(cid:44)GT (Xi)

w(T )
i

.

[correctly classified and misclassified]

What GT minimizes the risk? The learner that minimizes the sum of w(T )
[This is interesting. By manipulating the formula for the risk, we�ve discovered what weight we should
assign to each training point. To minimize the risk, we should find the classifier that minimizes the sum of
the weights w(T )
, as specified above, over the misclassified points. It�s a complicated function, but we can
compute it. A useful observation is that each learner�s weights are related to the previous learner�s weights:]

over all misclassified pts Xi!

i

i

Recursive definition of weights:

w(T +1)

i

= w(T )

i e??T yiGT (Xi) =

?
??
??

w(T )
i e??T
w(T )
i e?T

yi = GT (Xi),
yi (cid:44) GT (Xi).

[This recursive formulation is a nice benefit of choosing the exponential loss function. Notice that a weight
shrinks if the point was classified correctly by learner T , and grows if the point was misclassified.]
[Now, you might wonder if we should just pick a learner that classifies all the training points correctly. But
that�s not always possible. If we�re using a linear classifier on data that�s not linearly separable, some points
must be classified wrongly. Moreover, it�s NP-hard to find the optimal linear classifier, so in practice GT
will be an approximate best learner, not the true minimizer of the weighted training error. But that�s okay.]
[You might ask, if we use decision trees, can�t we get zero training error? Usually we can. But interestingly,
boosting is usually used with short, imperfect decision trees instead of tall, pure decision trees, for reasons
I�ll explain later.]
[Now, let�s derive the optimal value of ?T .]
To choose ?T , set

Risk = 0.

d
d?T

0 = ?e??T

n(cid:88)

i=1

w(T )
i

+ (e?T + e??T )

(cid:88)

yi(cid:44)GT (Xi)

w(T )
i

;

[now divide both sides by the first term]

0 = ?1 + (e2?T + 1) errT , where errT =

?T = 1
2

ln

(cid:32)

1 ? errT
errT

(cid:33)

.

[So now we have derived the optimal metalearner!]

(cid:80)

yi(cid:44)GT (Xi) w(T )
(cid:80)n
i=1 w(T )

i

i

; ? GT �s weighted error rate

148

Jonathan Richard Shewchuk

� If errT = 0, ?T = ?.
� If errT = 1/2, ?T = 0.

[So a perfect learner gets an infinite vote.]

[So a learner with 50% weighted training error gets no vote at all.]

[More accurate learners get bigger votes in the metalearner. Interestingly, a learner with training error worse
than 50% gets a negative vote. A learner with 60% error is just as useful as a learner with 40% error; the
metalearner just reverses the signs of its votes. It�s so bad, it�s good.]

[Now we can state the AdaBoost algorithm.]

AdaBoost alg:

1. Initialize weights wi ? 1
2. for t ? 1 to T

n , ?i ? [1, n].

a. Train Gt with weights wi.

b. Compute weighted error rate err ?

(cid:80)

misclassified wi
all wi

(cid:80)

c. Reweight pts: wi ? wi �

(cid:40)

3. return metalearner h(z) = sign

?

???????

e?t , Gt misclassifies Xi
e??t , otherwise
???????

?tGt(z)

?

.

T(cid:88)

t=1

(cid:32)

(cid:33)
.

1 ? err
err

ln

; coefficient ?t ?
?
?????
?????

= wi �

1
2
1?err
err
err
1?err .

(cid:113)

(cid:113)

,

boost.pdf [At left, all the training points have equal weight. After choosing a first linear
classifier, we increase the weights of the misclassified points and decrease the weights of the
correctly classified points (center). We train a second classifier with these weighted points,
then again adjust the weights of the points according to whether they are misclassified by
the second classifier.]

Why boost decision trees? [As opposed to other learning algorithms?] Why short trees?

� Boosting reduces bias reliably, but not always variance. AdaBoost trees are impure to reduce
overfitting. [Recall again that random forests use ensembles to reduce variance, but AdaBoost uses
them to reduce bias. The AdaBoost variance is more complicated: it often decreases at first, because
successive trees focus on different features, but often it later increases. Sometimes boosting overfits
after many iterations, and sometimes it doesn�t; it�s hard to predict when it will and when it won�t.]

� Fast.

[We�re training many learners, and running many learners at classification time too. Short

decision trees that only look at a few features are very fast at both training and testing.]

� No hyperparameter search needed. [Unlike SVMs, neural nets, etc.] [UC Berkeley�s Leo Breiman

called AdaBoost with decision trees �the best off-the-shelf classifier in the world.�]
� Easy to make a tree beat 45% training error [or some other threshold] consistently.

Boosting; Nearest Neighbor Classification

149

� AdaBoost + short trees is a form of subset selection.

[Features that don�t improve the metalearner�s predictive power enough aren�t used at all. This helps
reduce overfitting and running time, especially if there are a lot of irrelevant features.]

� Linear decision boundaries don�t boost well.

[It takes a lot of boosting to make linear classifiers model really nonlinear decision boundaries well, so
SVMs aren�t a great choice. Recall from Discussion Section 9 that ensembles of depth-one stumps are
an even worse choice, because they can�t even do XOR. Methods with nonlinear decision boundaries
benefit more from boosting, because they allow boosting to reduce the bias faster much. Even depth-
two decision trees boost substantially better than depth-one decision trees.]

More about AdaBoost:

� Posterior prob. can be approximated: P(Y = 1|x) ?
� Exponential loss is vulnerable to outliers; for corrupted data, use other loss.

1
1 + e?2M(x) .

[Loss functions have been derived for dealing with outliers. Unfortunately, they have more compli-
cated weight computations.]

� If every learner beats error � for � < 50%, metalearner training error will eventually be zero. [You

will prove this in Homework 7.]

� [The AdaBoost paper and its authors, Freund and Schapire, won the 2003 G�odel Prize, a prize for

outstanding papers in theoretical computer science.]

trainboost.pdf, testboost.pdf (ESL, Figures 10.2, 10.3)
[Training and testing errors for
AdaBoost with stumps, depth-one decision trees that make only one decision each. At
left, observe that the training error eventually drops to zero, and even after that the average
loss (which is continuous, not binary) continues to decay exponentially. At right, the test
error drops to 5.8% after 400 iterations, even though each learner has an error rate of about
46%. AdaBoost with more than 25 stumps outperforms a single 244-node decision tree. In
this example no overfitting is observed, but there are other datasets for which overfitting is
a problem.]

01002003004000.00.20.40.60.81.0Boosting IterationsTraining ErrorMisclassification RateExponential Loss01002003004000.00.10.20.30.40.5Boosting IterationsTest ErrorSingle Stump244 Node Tree150

Jonathan Richard Shewchuk

NEAREST NEIGHBOR CLASSIFICATION

[I saved the simplest classifier for the end of the semester.]

Idea: Given query point q, find the k training pts nearest q.

Distance metric of your choice.
Regression: Return average label of the k pts.
Classification: Return class with the most votes from the k pts OR

return histogram of class probabilities.

[The histogram of class probabilities tries to estimate the posterior probabilities of the classes. Obviously,
the histogram has limited precision. If k = 3, then the only probabilities you�ll ever return are 0, 1/3, 2/3,
or 1. You can improve the precision by making k larger, but you might underfit. The histogram works best
when you have a huge amount of data.]

allnn.pdf (ISL, Figures 2.15, 2.16) [Examples of 1-NN, 10-NN, and 100-NN. A larger k
smooths out the boundary. In this example, the 1-NN classifier is overfitting the data, and
the 100-NN classifier is badly underfitting. The 10-NN classifier does well: it�s reasonably
close to the Bayes decision boundary (purple). Generally, the ideal k depends on how dense
your data is. As your data gets denser, the best k increases.]

[There are theorems showing that if you have a lot of data, nearest neighbors can work quite well.]

Theorem (Cover & Hart, 1967):
As n ? ?, the 1-NN error rate is < 2B ? B2
if only 2 classes, ? 2B ? 2B2

where B = Bayes risk.

[There are a few technical requirements of this theorem. The most important is that the training points and
the test points all have to be drawn independently from the same probability distribution. Here, we are using
the 0-1 loss to define the Bayes risk; so the Bayes risk is the smallest possible error rate over that distribution.
The theorem applies to any separable metric space, so it�s not just for the Euclidean metric.]

Theorem (Fix & Hodges, 1951):
As n ? ?, k ? ?, k/n ? 0,

k-NN error rate converges to B.

[Which means Bayes optimal.]

ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooKNN:K=10ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooKNN:K=1ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooKNN:K=100Nearest Neighbor Algorithms: Voronoi Diagrams and k-d Trees

151

25 Nearest Neighbor Algorithms: Voronoi Diagrams and k-d Trees

NEAREST NEIGHBOR ALGORITHMS

Exhaustive k-NN Alg.

Given query point q:

� Scan through all n training pts, computing distances to q.
� Maintain a max-heap with the k shortest distances seen so far.

[Whenever you encounter a training point closer to q than the point at the top of the heap, you remove
the heap-top point and insert the better point. Obviously you don�t need a heap if k = 1 or even 5, but
if k = 99 a heap will substantially speed up keeping track of the 99th-shortest distance.]

Time to train classifier: 0
Query time: O(nd + n log k)

[This is the only O(0)-time algorithm we�ll learn this semester.]

expected O(nd + k log n log k) if random pt order

[It�s a cute theoretical observation that you can slightly improve the expected running time by randomizing
the point order so that only expected O(k log n) heap operations occur. But in practice I can�t recommend it;
you�ll probably lose more from cache misses than you�ll gain from fewer heap operations.]

Can we preprocess training pts to obtain sublinear query time?

2�5 dimensions: Voronoi diagrams
Medium dim (up to ? 30): k-d trees
Large dim: exhaustive k-NN, but can use PCA or random projection

locality sensitive hashing [still researchy, not widely adopted]

Voronoi Diagrams

Let X be a point set. The Voronoi cell of w ? X is
Vor w = {p ? Rd : ?p ? w? ? ?p ? v? ?v ? X}
[A Voronoi cell is always a convex polyhedron or polytope.]
The Voronoi diagram of X is the set of X�s Voronoi cells.

152

Jonathan Richard Shewchuk

voro.pdf, vormcdonalds.jpg, voronoiGregorEichinger.jpg, saltflat3.jpg
[Voronoi diagrams sometimes arise in nature (salt flats, giraffe, crystallography).]

giraffe-1.jpg, srsi2.png (Vladislav Blatov), vortex.pdf (Ren�e Descartes)

[Perhaps the first frequent users of Voronoi cells were crystallographers, who call them Voronoi�Dirichlet
polyhedra. Above we see how the polyhedra can clarify the crystal structure of the low-temperature phase
of strontium silicide, ?-SrSi2.]

[Believe it or not, the first published Voronoi diagram dates back to 1644, in the book �Principia Philosophiae�
by the mathematician and philosopher Ren�e Descartes. He claimed that the solar system consists of vortices.
In each region, matter is revolving around one of the fixed stars (vortex.pdf). His physics was wrong, but
his idea of dividing space into polyhedral regions has survived.]
Size (e.g., # of vertices) ? O(n?d/2?)
[This upper bound is tight when d is a small constant. As d grows, the tightest asymptotic upper bound is
somewhat smaller than this, but the complexity still grows exponentially with d.]
. . . but often in practice it is O(n).
[Here I�m leaving out a �constant� that may grow exponentially with d.]

Nearest Neighbor Algorithms: Voronoi Diagrams and k-d Trees

153

Point location: Given query point q ? Rd, find the point w ? X for which q ? Vor w.
[We need a second data structure that can perform this search on a Voronoi diagram efficiently.]
2D: O(n log n) time to compute V.d. and a trapezoidal map for pt location

O(log n) query time

[because of the trapezoidal map]
[That�s a pretty great running time compared to the linear query time of exhaustive search.]
dD: Use binary space partition tree (BSP tree) for pt location. [Unfortunately, it�s difficult to characterize
the running time of this strategy, although it is often logarithmic in 3�5 dimensions.]

1-NN only! [A standard Voronoi diagram supports only 1-nearest neighbor queries. If you want the k nearest
neighbors, there is something called an order-k Voronoi diagram that has a cell for each possible k nearest
neighbors. But nobody uses those, for two reasons. First, the size of an order-k Voronoi diagram is ?(k2n)
in 2D, and worse in higher dimensions. Second, there�s no reliable software available to compute one.]

[There are also Voronoi diagrams for other distance metrics, like the ?1 and ?? norms.]

[Voronoi diagrams are good for 1-nearest neighbor queries in two dimensions, and maybe up to 5 dimen-
sions, and they�re a great concept for understanding the problem of nearest neighbor search. But k-d trees
are much simpler, and probably faster in 6 or more dimensions.]

k-d Trees

[Just like in a decision tree, each treenode in a k-d tree represents a
�Decision trees� for NN search.
rectangular box in feature space, and we split a box by choosing a splitting feature and a splitting value. But
we use different criteria for choosing splits.] Differences:

� Choose splitting feature w/greatest width: feature i in maxi, j,k(X ji ? Xki).

[With nearest neighbor search, we don�t care about the entropy. Instead, what we want is that if we
draw a sphere around the query point, it won�t intersect very many boxes of the decision tree. So it
helps if the boxes are nearly cubical, rather than long and thin.]
Cheap alternative: rotate through the features. [At depth 1 we split on the first feature, at depth 2 we
split on the second feature, and so on. This builds the tree faster, by a factor of O(d).]

X ji+Xki
2

.

� Choose splitting value: median point for feature i; OR midpoint

Median guarantees ?log2 n? tree depth; O(nd log n) tree-building time.
[. . . or just O(n log n) time if you rotate through the features. An alternative to the median is splitting
at the box center, which improves the aspect ratios of the boxes, but it could unbalance your tree.
A compromise strategy is to alternate between medians at odd depths and centers at even depths,
which also guarantees an O(log n) depth.]
� Each internal node stores a training point.

[. . . that lies in the node�s box. Usually the splitting point.]
[Some k-d tree implementations have points only at the leaves, but it�s better to have points in internal
nodes too, so when we search the tree, we often stop searching earlier.]

7

9

1

5

2

3

6

4

8

10

11

1

3

5

2

root represents R2

6

10

4

7

8

right halfplane

lower right
quarter plane

9

11

every subtree
represents a box

[Draw this by hand. kdtreestructure.pdf ]

154

Jonathan Richard Shewchuk

[Just like in a decision tree, each subtree represents an axis-aligned box in feature space. All the training
points stored in a subtree are in that box.]
[Once the tree is built, the classification algorithm is very different from decision trees. Most importantly,
you usually have to visit multiple leaves of the tree to find the nearest neighbor. To save time, we sometimes
use an approximate nearest neighbor algorithm, instead of demanding the exact nearest neighbor.]

After tree is built (training), classify test pts:
Goal:

given query pt q, find a training pt w such that ?q ? w? ? (1 + ?) ?q ? u?,
where ? � ? is ?p norm for some p ? [1, ?]
& u is the nearest training pt in that norm.
? = 0 ? exact NN;

? > 0 ? approximate NN.
Each subtree represents a box B = [s1, t1] � [s2, t2] � � � � � [sd, td].
[Think of s as the lower left corner of the box, and t as the opposite corner.]
Think of B as an infinite point set.
The distance from q to B is dist(q, B) = min
z?B

?q ? z?.

[An si can be ??, and a ti can be ?.]

[This norm is the same norm we seek neighbors in.]

[k-d trees are not limited to the Euclidean (?2) norm.]

q1

dist(q1, B)

dist(q2, B)

q2

q3
dist(q3, B) = 0

B

[Draw this by hand. dist.pdf ] [Point-to-box distances.]

The minimizer�s components are zi =

Query alg. maintains:

?
????
????

si, qi < si,
qi, qi ? [si, ti],
qi > ti.
ti,

� Nearest neighbor found so far (or k nearest).
� Binary min-heap of unexplored boxes/subtrees, keyed by distance from q.

goes down ?
?
goes up

q

[Draw this by hand. query.pdf ] [A query in progress.]

nearest so far

[We search the boxes nearest q first, hoping that we will never need to search most of the boxes or their
associated subtrees. The binary heap makes it fast to find the box nearest q, because each box in the heap
has a numerical key, the distance from q to the box. The search stops when the distance from q to the
kth-nearest neighbor found so far ? the distance from q to the nearest unexplored box (times 1 + ?). For
example, in the figure above, the query will never visit the box at far lower right, because it doesn�t intersect
the circle. That�s how we avoid searching most of the tree�when we�re lucky.]

Nearest Neighbor Algorithms: Voronoi Diagrams and k-d Trees

155

Alg. for 1-NN query. Interpret each B as both a box and a treenode.
Q ? min-heap containing root node with key zero
r ? ?
while Q not empty and (1 + ?) � minkey(Q) < r

B ? removemin(Q)
v ? B�s training point
if ?q ? v? < r then { w ? v;
B?, B?? ? child boxes of B
if (1 + ?) � dist(q, B?) < r then insert(Q, B?, dist(q, B?))
if (1 + ?) � dist(q, B??) < r then insert(Q, B??, dist(q, B??))

r ? ?q ? v? }

return w

[The key for B? is dist(q, B?)]

For k-NN, replace �r� and �w� with a max-heap holding the k nearest neighbors.

Why ?-approximate NN?

q

[Draw this by hand. kdtreeproblem.pdf ] [A worst-case exact NN query.]

[In the worst case, we may have to visit every node in the k-d tree to find the exact nearest neighbor. In that
case, the k-d tree is slower than simple exhaustive search. This is an example where an approximate nearest
neighbor search can be much faster. In practice, settling for an approximate nearest neighbor sometimes
improves the speed by a factor of 10 or even 100, because you don�t need to look at most of the tree to do a
query. This is especially true in high dimensions�in a high-dimensional space, the nearest point often isn�t
much closer than a lot of other points.]

[I want to emphasize the fact that exhaustive nearest neighbor search really is one of the first classifiers you
should try in practice, even if it seems too simple. So here�s an example of a research paper that uses a
120-nearest neighbor classifier to solve a problem.]

im2gpspress.pdf

156

Jonathan Richard Shewchuk

[In 2008, James Hays and our own Prof. Alexei Efros wrote a paper on geolocalization, where the goal is to
take a query photograph and determine where on earth the photo was taken. Their training data was 6 million
GPS-tagged photos downloaded from Flickr. The bottom line is that by using 120-nearest neighbors, they
came within 64 km of the correct location about 50% of the time. That was good for the time, but I think
that in 2025, you could do much better with the data available to us today.]

RELATED CLASSES [if you like machine learning, consider these courses in 2024�25]

CS 180/280A (fall): Computer Vision/Photography
CS 182/282A (fall): Deep Neural Networks
EECS 183 (fall?): Natural Language Processing
CS 185/285 (fall?): Deep Reinforcement Learning
CS 194-196/294-196 (fall): Agentic AI (D. Song)
CS C281A (fall): Statistical Learning Theory [C281A is the most direct continuation of CS 189/289A.]
EECS 127 (both), 227AT (both): Numerical Optimization [a core part of ML]
[It�s hard to overemphasize the importance of numerical optimization to machine learning, as well as other
CS fields like graphics, theory, and scientific computing.]
EECS 126 (both): Random Processes [Markov chains, expectation maximization, PageRank]
EE C106A/B (fall/spring): Intro to Robotics [dynamics, control, sensing]
Math 110 (both): Linear Algebra [but the real gold is in Math 221]
Math 221 (fall): Matrix Computations [how to solve linear systems, compute SVDs, eigenvectors, etc.]
CS C281B (spring): Learning & Decision Making
CS C267 (spring): Scientific Computing [parallelization, practical matrix algebra, some graph partitioning]
CS C280 (spring): Computer Vision (Efros, Kanazawa)
CS 294-162 (fall): ML Systems (Gonzalez/Stoica/Zaharia)
CS 294-286 (fall): Machine Learning in Social Settings (Chang)
NEU 100A (fall): Cellular and Molecular Neurobiology
VS 265 (?): Neural Computation

Bonus Lecture: Learning Theory

157

A Bonus Lecture: Learning Theory

LEARNING THEORY: WHAT IS GENERALIZATION?

[One thing humans do well is generalize. When you were a young child, you only had to see a few examples
of cows before you learned to recognize cows, including cows you had never seen before. You didn�t have
to see every cow. You didn�t even have to see log n of the cows.]

[Learning theory tries to explain how machine learning algorithms generalize, so they can classify data
they�ve never seen before. It also tries to derive mathematically how much training data we need to general-
ize well. Learning theory starts with the observation that if we want to generalize, it helps to constrain what
hypotheses we allow our learner to consider.]

A range space (aka set system) is a pair (P, H), where
P is set of all possible test/training points (can be infinite)
H is hypothesis class, a set of hypotheses (aka ranges, aka classifiers):

each hypothesis is a subset h ? P that specifies which points h predicts are in class C.
[So each hypothesis h is a 2-class classifier, and H is a set of sets of points.]

Examples:

1. Power set classifier: P is a set of k numbers; H is the power set of P, containing all 2k subsets of P.

e.g., P = {1, 2}, H = {?, {1}, {2}, {1, 2}}

2. Linear classifier: P = Rd; H is the set of all halfspaces; each halfspace has the form {x : w � x ? ??}.
[In this example, both P and H are infinite. In particular, H contains every possible halfspace�that
is, every possible linear classifier in d dimensions.]

[The power set classifier sounds very powerful, because it can learn every possible hypothesis. But the
reality is that it can�t generalize at all. Imagine we have three training points and three test points in a row.]

[The power set classifier can classify these three test points any way you like. Unfortunately, that means it
has learned nothing about the test points from the training points. By contrast, the linear classifier can learn
only two hypotheses that fit this training data. The leftmost test point must be classified class C, and the
rightmost test point must be classified class Not-C. Only the test point in the middle can swing either way.
So the linear classifier has a big advantage: it can generalize from a few training points. That�s also a big
disadvantage if the data isn�t close to linearly separable, but that�s another story.]
[Now we will investigate how well the training error predicts the test error, and how that differs for these
two classifiers.]

Suppose all training pts & test pts are drawn independently from same prob. distribution D defined on
domain P. [D also determines each point�s label. Classes C and Not-C may have overlapping distributions.]

Let h ? H be a hypothesis [a classifier]. h predicts a pt x is in class C if x ? h.
The risk aka generalization error R(h) of h is the probability that h misclassifies a random pt x drawn
from D�i.e., the prob. that x ? C but x (cid:60) h or vice versa.
[Risk is almost the same as the test error. To be precise, the risk is the mean test error for test points drawn
randomly from D. For a particular test set, sometimes the test error is higher, sometimes lower, but on
average it is R(h). If you had an infinite amount of test data, the risk and the test error would be the same.]

CCN???158

Jonathan Richard Shewchuk

Let X ? P be a set of n training pts drawn from D
The empirical risk aka training error �R(h) is % of X misclassified by h.
[This matches the definition of empirical risk I gave you in Lecture 12, if you use the 0-1 loss function.]
h misclassifies each training pt w/prob. R(h), so total misclassified has a binomial distribution.
As n ? ?, �R(h) better approximates R(h).

binom20.pdf, binom500.pdf Consider a hypothesis whose risk of misclassification is 25%.
[Plotted are probability mass functions of the number of misclassified training points for 20
points and 500 points, respectively. For 20 points, the training error is not a reliable estimate
of the risk: the hypothesis might get �lucky� with misleadingly low training error.]

[If we had infinite training data, this distribution would become infinitely narrow and the training error
would always be equal to the risk. But we can�t have infinite training data. So, how well does the training
error approximate the risk?]
Hoeffding�s inequality tells us prob. of bad estimate:

Pr(| �R(h) ? R(h)| > ?) ? 2e?2?2n.

[Hoeffding�s inequality is a standard result about how likely it is that a number drawn from a binomial
distribution will be far from its mean. If n is big enough, then it�s very unlikely.]

hoeffding.pdf [Hoeffding�s bound for the unambitious ? = 0.1. It takes at least 200 training
points to have high confidence of attaining that error bound.]

[One reason this matters is because we will try to choose the best hypothesis. If the training error is a bad
estimate of the test error, we might choose a hypothesis we think is good but really isn�t. So we are happy
to see that the likelihood of that decays exponentially in the amount of training data.]

51015200.050.100.150.201002003004005000.010.020.030.04050100150200250300points0.20.40.60.81.0badestimateprobabilityBonus Lecture: Learning Theory

159

Idea for learning alg: choose �h ? H that minimizes �R(�h)! Empirical risk minimization.
[None of the classification algorithms we�ve studied actually do this, but only because it�s computationally
infeasible to pick the best hypothesis. Support vector machines can find a linear classifier with zero training
error when the training data is linearly separable. But when it isn�t, SVMs try to find a linear classifier with
low training error, but they don�t generally find the one with minimum training error. That�s NP-hard.]
[Nevertheless, for the sake of understanding learning theory, we�re going to pretend that we have the com-
putational power to try every hypothesis and pick the one with the lowest training error.]
Problem: if too many hypotheses, some h with high R(h) will get lucky and have very low �R(h)!

[This brings us to a central idea of learning theory. You might think that the ideal learning algorithm would
have the largest class of hypotheses, so it could find the perfect one to fit the data. But the reality is that you
can have so many hypotheses that some of them just get lucky and score far lower training error than their
actual risk. That�s another way to understand what �overfitting� is.]
[More precisely, the problem isn�t too many hypotheses. Usually we have infinitely many hypotheses, and
that�s okay. The problem is too many dichotomies.]

Dichotomies

A dichotomy of X is X ? h, where h ? H.
[A dichotomy picks out the training points that h predicts are in class C. Think of each dichotomy as a
function assigning each training point to class C or class Not-C.]

[Draw this by hand. dichotomies.pdf ] [Three examples of dichotomies for three points in
a hypothesis class of linear classifiers, and one example (right) that is not a dichotomy.]

[For n training points, there could be up to 2n dichotomies. The more dichotomies there are, the more likely
it is that one of them will get lucky and have misleadingly low empirical risk.]
Extreme case: if H allows all 2n possible dichotomies, �R(�h) = 0 even if every h ? H has high risk.
[If our hypothesis class permits all 2n possible assignments of the n training points to classes, then one of
them will have zero training error. But that�s true even if all of the hypotheses are terrible and have a large
risk. Because the hypothesis class imposes no structure, we overfit the training points.]
If H induces ? dichotomies, Pr(at least one dichotomy has | �R ? R| > ?) ? ?, where ? = 2? e?2?2n.
[Let�s fix a value of ? and solve for ?.] Hence with prob. ? 1 ? ?, for every h ? H,

| �R(h) ? R(h)| ? ? =

(cid:114)

1
2n

ln

2?
?

.

[This tells us that the smaller we make ?, the number of possible dichotomies, and the larger we make n,
the number of training points, the more accurately the training error will approximate how well the classifier
performs on test data.]
smaller ? or larger n ? training error probably closer to true risk (& test error).

CCNCCNCCNCCN160

Jonathan Richard Shewchuk

[Smaller ? means we�re less likely to overfit. We have less variance, but more bias. This doesn�t necessarily
mean the risk will be small. If our hypothesis class H doesn�t fit the data well, both the training error and
the test error will be large. In an ideal world, we want a hypothesis class that fits the data well, yet doesn�t
produce many dichotomies.]
Let h? ? H minimize R(h?); �best� classifier.
[Remember we picked the classifier �h that minimizes the empirical risk. We really want the classifier h? that
minimizes the actual risk, but we can�t know what h? is. But if ? is small and n is large, the hypothesis �h
we have chosen is probably nearly as good as h?.]
With prob. ? 1 ? ?, our chosen �h has nearly optimal risk:

R(�h) ? �R(�h) + ? ? �R(h?) + ? ? R(h?) + 2?,

(cid:114)

? =

1
2n

ln

2?
?

.

[This is excellent news! It means that with enough training data and a limit on the number of dichotomies,
empirical risk minimization usually chooses a classifier close to the best one in the hypothesis class.]

Choose a ? and an ?.
The sample complexity is the # of training pts needed to achieve this ? with prob. ? 1 ? ?:

n ?

1
2?2 ln

2?
?

.

[If ? is small, we won�t need too many training points to choose a good classifier. Unfortunately, if ? = 2n
we lose, because this inequality says that n has to be bigger than n. So the power set classifier can�t learn
much or generalize at all. We need to severely reduce ?, the number of possible dichotomies. One way to
do that is to use a linear classifier.]

The Shatter Function & Linear Classifiers

[How many ways can you divide n points into two classes with a hyperplane?]

# of dichotomies: ?H(X) = |{X ? h : h ? H}|
?H(X)
shatter function: ?H(n) =

max
|X|=n,X?P

? [1, 2n] where n = |X|
[The most dichotomies of any point set of size n]

Example: Linear classifiers in plane. H = set of all halfplanes. ?H(3) = 8:

[Draw this by hand.
these three points. The other four dichotomies are the complements of these four.]

shatter.pdf ] [Linear classifiers can induce all eight dichotomies of

NCCNNCCCCCCCBonus Lecture: Learning Theory

161

?H(4) = 14:
[Instead of showing you all 14 dichotomies, let me show you dichotomies that halfplanes cannot learn,
which illustrate why no four points have 16 dichotomies.]

[Draw this by hand. unshatter.pdf ] [Examples of dichotomies of four points in the plane
that no linear classifier can induce.]

[This isn�t a proof that 14 is the maximum, because we have to show that 15 is not possible for any four
points in the plane. The standard proof uses a famous result called Radon�s Theorem.]
Fact: for all range spaces, either ?H(n) is polynomial in n, or ?H(n) = 2n ?n ? 0.

[This is a surprising fact with deep implications. Imagine that you have n points, some of them training
points and some of them test points. Either a range space permits every possible dichotomy of the points,
and the training points don�t help you classify the test points at all; or the range space permits only a
polynomial subset of the 2n possible dichotomies, so once you have labeled the training points, you have
usually cut down the number of ways you can classify the test points dramatically. No shatter function ever
occupies the no-man�s-land between polynomial and 2n.]

[For linear classifiers, we know exactly how many dichotomies there can be.]
(cid:32)
n ? 1
i

Cover�s Theorem [1965]: linear classifiers in Rd allow up to ?H(n) = 2

d(cid:88)

i=0

For n ? d + 1, ?H(n) = 2n.
For n ? d + 1, ?H(n) ? 2
and the sample complexity needed to achieve R(�h) ? �R(�h) + ? ? R(h?) + 2? with prob. ? 1 ? ? is

[Observe that this is polynomial in n! With exponent d.]

(cid:16) e(n?1)
d

(cid:17)d

(cid:33)

dichotomies of n pts.

n ?

1
2?2

(cid:32)
d ln

n ? 1
d

+ d + ln

(cid:33)

.

4
?

[Observe that the logarithm turned the exponent d into a factor!]

Corollary:

linear classifiers need only n ? ?(d) training pts
for training error to accurately predict risk or test error.

[In a d-dimensional feature space, we need more than d training points to train an accurate linear classifier.
But it�s reassuring to know that the number we need is linear in d. By contrast, if we have a classifier that
permits all 2n possible dichotomies however large n is, then no amount of training data will guarantee that
the training error of the hypothesis we choose approximates the true risk.]
[The constant hidden in that big-? notation can be quite large. For example, if you choose ? = 0.1 and
? = 0.1, then setting n = 550 d will always suffice. (For very large d, n = 342 d will do.) If you want a lot of
confidence that you�ve chosen one of the best hypotheses, you have to pay for it with a large sample size.]

[This sample complexity applies even if you add polynomial features or other features, but you have to count
the extra features in d. So the number of training points you need increases with the number of polynomial
terms.]

CCCNNCNCCNCC162

Jonathan Richard Shewchuk

VC Dimension

The Vapnik�Chervonenkis dimension of (P, H) is

VC(H) = max{n : ?H(n) = 2n}.

? Can be ?.

Say that H shatters a set X of n pts if ?H(X) = 2n.
VC(H) is size of largest X that H can shatter.
[This means that X is a point set for which all 2n dichotomies are possible.]

[I told you earlier that if the shatter function isn�t 2n for all n, then it�s a polynomial in n. The VC dimension
is motivated by an observation that sometimes makes it easy to bound that polynomial.]

Theorem: ?H(n) ?

VC(H)(cid:88)

i=0

(cid:33)
.

(cid:32)
n
i

Hence for n ? VC(H), ?H(n) ?

(cid:32)

en
VC(H)

(cid:33)VC(H)

.

[So the VC dimension is an upper bound on the exponent of the polynomial. This theorem is useful because
often we can find an easy upper bound on the VC dimension. You just need to show that for some number n,
no set of n points can have all 2n dichotomies.]
Corollary: O(VC(H)) training pts suffice for accuracy.

[Again, the hidden constant is big.]

[If the VC dimension is finite, it tells us how the sample complexity grows with the number of features.
If the VC dimension is infinite, no amount of training data will make the classifier generalize well.]

Example: Linear classifiers in plane.
Recall ?H(3) = 8: there exist 3 pts shattered by halfplanes.
But ?H(4) = 14: no 4 pts are shattered.
Hence:

� VC(H) = 3 [The VC dimension of halfplanes is 3.]
e3
� ?H(n) ?
27
� O(1) sample complexity.

[The shatter function is polynomial.]

n3

[The VC dimension doesn�t always give us the tightest bound. In this example, the VC dimension promises
that the number of ways halfplanes can classify the points is at worst cubic in n; but Cover�s Theorem says
it�s quadratic in n. In general, linear classifiers in d dimensions have VC dimension d + 1, which is one
dimension looser than the exponent Thomas Cover proved. That�s not a big deal, though, as the sample
complexity and the accuracy bound are both based on the logarithm of the shatter function. So if we get the
exponent wrong, it only changes a constant in the sample complexity.]

[The important thing is simply to show that there is some polynomial bound on the shatter function at all.
VC dimension is not the only way to do that, but often it�s the easiest.]

[The main point you should take from this lecture is that if you want to have generalization, you need to
limit the expressiveness of your hypothesis class so that you limit the number of possible dichotomies of a
point set. This may or may not increase the bias, but if you don�t limit the number of dichotomies at all, the
overfitting could be very bad. If you limit the hypothesis class, your artificial child will only need to look at
O(d) cows to learn the concept of cows. If you don�t, your artificial child will need to look at every cow in
the world, and every non-cow too.]

Bonus Lecture: The Kernel Trick

163

B Bonus Lecture: The Kernel Trick

KERNELS

Recall featurizing map ? : Rd ? RD. d input features; D features after featurization (?).
Degree-p polynomials blow up to D ? ?(d p) features.
[When d and p are not small, this gets computationally intractable really fast. As I said in Lecture 4, if you
have 100 features per feature vector and you want to use degree-4 polynomial decision functions, then each
featurized feature vector has a length of roughly 4 million.]
Today, magically, we use those features without computing them!

Observation: In many learning algs,

� the weights can be written as a linear combo of training points, &
� we can use inner products of ?(x)�s only ? don�t need to compute ?(x)!

Suppose w = X?a =

n(cid:88)

aiXi for some a ? Rn.

i=1

Substitute this identity into alg. and optimize n dual weights a (aka dual parameters)
instead of D primal weights w.

Kernel Ridge Regression

Center X and y so their means are zero: Xi ? Xi ? �X,
This lets us replace I? with I in normal equations:

yi ? yi ? �y,

Xi,d+1 = 1 [don�t center the 1�s!]

(X?X + ?I)w = X?y.

[To kernelize ridge regression, we need the weights to be a linear combination of the training points. Unfor-
tunately, that only happens if we penalize the bias term wd+1 = ?, as these normal equations do. Fortunately,
when we center X and y, the �expected� value of the bias term is zero. The actual bias won�t usually be
exactly zero, but it will often be close enough that we won�t do much harm by penalizing the bias term.]

Suppose a is a solution to

(XX? + ?I)a = y.

[Always has a solution if ? > 0.]

Then X?y = X?XX?a + ?X?a = (X?X + ?I)X?a.
Therefore, w = X?a is a solution to the normal equations, and w is a linear combo of training points!

a is a dual solution; solves the dual form of ridge regression:

Find a that minimizes ?XX?a ? y?2 + ??X?a?2.

[We obtain this dual form by substituting w = X?a into the original ridge regression cost function.]
Training: Solve (XX? + ?I)a = y for a.
Testing: Regression fn is

h(z) = w?z = a?Xz =

n(cid:88)

i=1

ai (X?

i z)

? weighted sum of inner products

164

Jonathan Richard Shewchuk

Let k(x, z) = x?z be kernel fn.
[Later, we�ll replace x and z with ?(x) and ?(z), and that�s where the magic will happen.]
Let K = XX? be n � n kernel matrix. Note Ki j = k(Xi, X j).
K may be singular. If so, probably no solution if ? = 0. [Then we must choose a positive ?. But that�s okay.]
Always singular if n > d + 1. [But don�t worry about the case n > d + 1, because you would only want to
use the dual form when d > n, i.e., for polynomial features. But K could still be singular when d > n.]
Dual/kernel ridge regr. alg:

Ki j ? k(Xi, X j)

?i, j,
Solve (K + ?I) a = y
for each test pt z
h(z) ? (cid:80)n

i=1 ai k(Xi, z)

for a

? O(n2d) time
? O(n3) time

? O(nd) time/test pt

Does not use Xi directly! Only k.

[This will become important soon.]

[Important: dual ridge regression produces the same predictions as primal ridge regression (with a penal-
ized bias term)! The difference is the running time; the dual algorithm is faster if d > n, because the primal
algorithm solves a d � d linear system, whereas the dual algorithm solves an n � n linear system.]

The Kernel Trick (aka Kernelization)

[Here�s the magic part. We can compute a polynomial kernel without actually computing the features.]
The polynomial kernel of degree p is k(x, z) = (x?z + 1)p.
Theorem: (x?z + 1)p = ?(x)??(z) for some ?(x) containing every monomial in x of degree 0 . . . p.
Example for d = 2, p = 2:

(x?z + 1)2 = x2
1z2
1
= [x2
1
= ?(x)??(z)

+ x2
x2
2

+ 2x1z1x2z2 + 2x1z1 + 2x2z2 + 1
2z2
2
?
1] [z2
2x1x2
1

2x1

2x2

?

?

?

2z1z2

?

2z1

?

2z2

1]?

z2
2

[This is how we�re defining ?.]

?

2. If you try a higher polynomial degree p, you�ll see a wider variety of these
[Notice the factors of
constants. We have no control of the constants that appear in ?(x), but they don�t matter much, because the
primal weights w will scale themselves to compensate. Even though we don�t directly compute the primal
weights, they implicitly exist in the form w = X?a.]
Key win: compute ?(x)??(z) in O(d) time instead of O(D) = O(d p), even though ?(x) has length D.
Kernel ridge regr. replaces Xi with ?(Xi): let k(x, z) = ?(x)??(z),
but doesn�t compute ?(x) or ?(z); it computes k(x, z) = (x?z + 1)p.

Running times for 3 ridge algs:

primal
O(D3 + D2n) O(n3 + n2D)

dual (no kernel trick)

train
test (per test pt) O(D)

O(nD)

kernel
O(n3 + n2d)
O(nd)

[I think what we�ve done here is pretty mind-blowing: we can now do polynomial regression with an expo-
nentially long, high-order polynomial in less time than it would take even to write out the final polynomial.
The running time can be asymptotically smaller than D, the number of terms in the polynomial.]

Bonus Lecture: The Kernel Trick

165

Kernel Logistic Regression

Let ?(X) be n � D matrix with rows ?(Xi)?.
Featurized logi. regr. with batch grad. descent:

w ? 0
repeat until convergence

w ? w + ? ?(X)? (y ? s(?(X) w))

for each test pt z

h(z) ? s(w??(z))

[?(X) is the design matrix of the featurized training points.]

[starting point is arbitrary]

apply s component-wise to vector ?(X) w

Dualize with w = ?(X)?a.
Then the code �a ? a + ? (y ? s(?(X) w))� has same effect as �w ? w + ? ?(X)?(y ? s(?(X) w))�.
Let K = ?(X) ?(X)?.
Note that Ka = ?(X) ?(X)?a = ?(X) w.
Dual/kernel logistic regression:

[The n � n kernel matrix; but we don�t compute ?(X)�we use the kernel trick.]
[And ?(X) w appears in the algorithm above.]

a ? 0
?i, j,
Ki j ? k(Xi, X j)
repeat until convergence

a ? a + ? (y ? s(Ka))

[starting point is arbitrary]
? O(n2d) time (kernel trick)

? O(n2) time/iteration [apply s component-wise]

for each test pt z
?

h(z) ? s

??????

n(cid:88)

i=1

ai k(Xi, z)

?

??????

? O(nd) time/test pt [kernel trick]

[For classification, you can skip the logistic function s(�) and just compute the sign of the summation.]

[Kernel logistic regression computes the same answer as the primal algorithm, but the running time changes.]
Important: running times depend on original dimension d, not on length D of ?(�)! Training for j iterations:
Primal: O(nD j) time

Dual (no kernel trick): O(n2D + n2 j) time

Kernel: O(n2d + n2 j) time

Alternative training: stochastic gradient descent (SGD). Primal logistic SGD step is
(cid:1) ?(Xi).

yi ? s(?(Xi)?w)

w ? w + ? (cid:0)

Dual logistic SGD maintains a vector q = Ka ? Rn. Note that qi = (?(X) w)i = ?(Xi)? w.
Let K?i denote column i of K.

[If you choose a different starting point, set q ? Ka.]

a ? 0; q ? 0; ?i, j, Ki j ? k(Xi, X j)
repeat until convergence

choose random i ? [1, n]
ai ? ai + ? (yi ? s(qi))
q ? q + ? (yi ? s(qi)) K?i

? O(1) time
? computes q = Ka in O(n) time, not O(n2) time

[SGD updates only one dual weight ai per iteration; that�s a nice benefit of the dual formulation. We cleverly
update q = Ka in linear time instead of performing a quadratic-time matrix-vector multiplication.]
Primal: O(D j) time

Dual (no kernel trick): O(n2D + n j) time

Kernel: O(n2d + n j) time

Alternative testing: If # of training points and test points both exceed D/d, classifying with primal weights w
may be faster. [This applies to ridge regression as well.]

w = ?(X)?a
for each test pt z
(cid:0)
h(z) ? s

w??(z)

(cid:1)

? O(nD) time (once only)

? O(D) time/test pt

166

Jonathan Richard Shewchuk

The Gaussian Kernel

[Mind-blowing as the polynomial kernel is, I think our next trick is even more mind-blowing. Since we can
now do fast computations in spaces with exponentially large dimensions, why don�t we go all the way and
generate feature vectors in an infinite-dimensional space?]
Gaussian kernel, aka radial basis fn kernel: there exists a ? : Rd ? R? such that

k(x, z) = exp

(cid:33)

(cid:32)
?

?x ? z?2
2?2

[This kernel takes O(d) time to compute.]

[In case you�re curious, here�s the feature vector that gives you this kernel, for the case where you have only
one input feature per sample point.]
e.g., for d = 1,

?(x) = exp

(cid:32)

?

x2
2?2

(cid:33) (cid:34)

1,

?

x
?

1!

,

x2
?

,

x3
?

?2

2!

?3

3!

(cid:35)?

, . . .

[This is an infinite vector, and ?(x) � ?(z) is a series that converges to k(x, z). Nobody actually uses this value
of ?(x) directly, or even cares about it; they just use the kernel function k(�, �).]
[At this point, it�s best not to think of points in a high-dimensional space. It�s no longer a useful intuition.
Instead, think of the kernel k as a measure of how similar or close together two points are to each other.]

Key observation:

hypothesis h(z) = (cid:80)n
[The dual weights are the coefficients of the linear combination.]
[The Gaussians are a basis for the hypothesis.]

j=1 a j k(X j, z) is a linear combo of Gaussians centered at training pts.

gausskernel.pdf [A hypothesis h that is a linear combination of Gaussians centered at four
training points, two with positive weights and two with negative weights. If you use ridge
regression with a Gaussian kernel, your �linear� regression will look something like this.]

Bonus Lecture: The Kernel Trick

167

Very popular in practice! Why?
� Gives very smooth h.
� Behaves somewhat like k-nearest neighbors, but smoother.
� Oscillates less than polynomials (depending on ?).
� k(x, z) interpreted as a similarity measure. Maximum when z = x; goes to 0 as distance increases.
� Training points �vote� for value at z, but closer points get weightier vote.

[In fact, h is infinitely differentiable; it�s C?-continuous.]

[The �standard� kernel k(x, z) = x � z assigns more weight to training point vectors that point in roughly the
same direction as z. By contrast, the Gaussian kernel assigns more weight to training points near z.]

Choose ? by validation.
? trades off bias vs. variance:

larger ? ? wider Gaussians & smoother h ? more bias & less variance

[The decision boundary (solid black) of a soft-
gausskernelsvm.pdf (ESL, Figure 12.3)
margin SVM with a Gaussian kernel. Observe that in this example, it comes reasonably
close to the Bayes optimal decision boundary (dashed purple). The dashed black curves are
the boundaries of the margin. The small black disks are the support vectors that lie on the
margin boundary.]

[By the way, there are many other kernels that, like the Gaussian kernel, are defined directly as kernel
functions without worrying about ?. But not every function can be a kernel function. A function is qualified
only if it always generates a positive semidefinite kernel matrix, for every sample. There is an elaborate
theory about how to construct valid kernel functions. However, you probably won�t need it. The polynomial
and Gaussian kernels are the two most popular by far.]
[As a final word, be aware that not every featurization ? leads to a kernel function that can be computed
faster than ?(D) time. In fact, the vast majority cannot. Featurizations that can are rare and special.]

...............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo�������������������������������Training Error: 0.160Test Error:       0.218Bayes Error:    0.210168

Jonathan Richard Shewchuk

C Bonus Lecture: Spectral Graph Clustering

SPECTRAL GRAPH CLUSTERING

Input: Weighted, undirected graph G = (V, E). No self-edges.

wi j = weight of edge (i, j) = ( j, i); zero if (i, j) (cid:60) E.

[Think of the edge weights as a similarity measure. A big weight means that the two vertices want to be
in the same cluster. So the circumstances are the opposite of the last lecture on clustering. Then, we had a
distance or dissimilarity function, so small numbers meant that points wanted to stay together. Today, big
numbers mean that vertices want to stay together.]

Goal: Cut G into 2 (or more) pieces Gi of similar sizes,

but don�t cut too much edge weight.
[That�s a vague goal. There are many ways to make this precise.
Here�s a typical goal, which we�ll solve approximately.]
e.g., Minimize the sparsity
Mass(G1) Mass(G2) , aka cut ratio
where Cut(G1, G2) = total weight of cut edges

Cut(G1,G2)

Mass(G1) = # of vertices in G1 OR assign masses to vertices

[The denominator �Mass(G1) Mass(G2)� penalizes imbalanced cuts.]

minimum
bisection

sparsest
cut

minimum
cut

maximum
cut

graph.pdf [Four cuts. All edges have weight 1.
Upper left: the minimum bisection; a bisection is perfectly balanced.
Upper right: the minimum cut. Usually very unbalanced; not what we want.
Lower left: the sparsest cut, which is good for many applications.
Lower right: the maximum cut; in this case also the maximum bisection.]

Sparsest cut, min bisection, max cut all NP-hard.
[Today we will look for an approximate solution to the sparsest cut problem.]

[We will turn this combinatorial graph cutting problem into algebra.]

Bonus Lecture: Spectral Graph Clustering

169

Let n = |V|. Let y ? Rn be an indicator vector:

(cid:40)

yi =

1 vertex i ? G1,
?1 vertex i ? G2.

Then wi j

(yi ? y j)2
4

(cid:40)

=

wi j
0

(i, j) is cut,
(i, j) is not cut.

Cut(G1, G2) =

(cid:88)

(i, j)?E

wi j

(yi ? y j)2
4

[This is quadratic, so let�s try to write it with a matrix.]

= 1
4

= 1
4

????????

(i, j)?E
?

(cid:88)

(cid:88)

(cid:16)

wi j y2

i ? 2wi j yi y j + wi j y2

j

(cid:17)

?2wi j yi y j
(i, j)?E
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
off-diagonal terms

?

????????

+

n(cid:88)

(cid:88)

y2
wik
i
i=1
k(cid:44)i
(cid:124)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:123)(cid:122)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:32)(cid:125)
diagonal terms

,

= y?Ly
4
(cid:40) ?wi j,
(cid:80)

k(cid:44)i wik,

i (cid:44) j,
i = j.

where Li j =

L is symmetric, n � n Laplacian matrix for G.

[Draw this by hand graphexample.png ]
[L is effectively a matrix representation of G. For the purpose of partitioning a graph, there is no need to
distinguish edges of weight zero from edges that are not in the graph.]
[We see that minimizing the weight of the cut is equivalent to minimizing the Laplacian quadratic form
y?Ly. This lets us turn graph partitioning into a problem in matrix algebra.]
[Usually we assume there are no negative weights, in which case Cut(G1, G2) can never be negative, so it
follows that L is positive semidefinite.]
Define 1 = [1 1
. . .
1 is an eigenvector of L with eigenvalue 0.

[It�s easy to check that each row of L sums to zero.]

1]?; then L1 = 0, so

[If G is a connected graph and all the edge weights are positive, then this is the only zero eigenvalue. But if
G is not connected, L has one zero eigenvalue for each connected component of G. It�s easy to prove, but
time prevents me.]

170

Jonathan Richard Shewchuk

Bisection: exactly n/2 vertices in G1, n/2 in G2. Write 1?y = 0.
[So we have reduced graph bisection to this constrained optimization problem.]
Minimum bisection:
Find y that minimizes y?Ly
subject to
and

? binary constraint
? balance constraint

?i, yi = 1 or yi = ?1
1?y = 0

Also NP-hard. We relax the binary constraint. ? fractional vertices!

[A very common approach in combinatorial optimization algorithms is to relax some of the constraints so
a discrete problem becomes a continuous problem. Intuitively, this means that you can put 1/3 of vertex 7
in graph G1 and the other 2/3 of vertex 7 in graph G2. You can even put ?1/2 of vertex 7 in graph G1 and
3/2 of vertex 7 in graph G2. This sounds crazy, but the continuous problem is much easier to solve than the
combinatorial problem. After we solve the continuous problem, we will round the vertex values to +1/?1,
and we�ll hope that our solution is still close to optimal.]
[But we can�t just drop the binary constraint. We still need some constraint to rule out the solution y = 0.]
?
New constraint: y must lie on hypersphere of radius

n.

[Draw this by hand. circle.pdf ] [Instead of constraining y to lie at a vertex of the hyper-
cube, we constrain y to lie on the hypersphere through those vertices.]

Relaxed problem:

Minimize y?Ly
subject to
and

y?y = n
1?y = 0

(cid:41)

= Minimize

y?Ly
y?y

= Rayleigh quotient of L & y

(subject to same two constraints)

2

5

3

1

1

3

y2

1

y?Ly = 0

y?Ly = 8
y?Ly = 16
y?Ly = 24

y1

1?y = 0

y3

cylinder.pdf [The isosurfaces of y?Ly are elliptical cylinders. The gray cross-section is
the hyperplane 1?y = 0. We seek the point that minimizes y?Ly, subject to the constraints
that it lies on the gray cross-section and that it lies on a sphere centered at the origin.]

Bonus Lecture: Spectral Graph Clustering

171

2

5

3

1

1

3

y2

y?y = 3

v2

v3

y?Ly = 16.6077
y?Ly = 12
y?Ly = 6

y3

y1

endview.pdf [The same isosurfaces restricted to the hyperplane 1?y = 0. The solution is
constrained to lie on the outer circle.]

[You should remember this Rayleigh quotient from the lecture on PCA. As I said then, when you see a
Rayleigh quotient, you should smell eigenvectors nearby. The y that minimizes this Rayleigh quotient is the
eigenvector with the smallest eigenvalue. We already know what that eigenvector is: it�s 1. But that violates
our balance constraint. As you should recall from PCA, when you�ve used the most extreme eigenvector
and you need an orthogonal one, the next-best optimizer of the Rayleigh quotient is the next eigenvector.]
Let ?2 = second-smallest eigenvalue of L.
Eigenvector v2 is the Fiedler vector. v2 solves the relaxed problem.
[It would be wonderful if every component of the Fiedler vector was 1 or ?1, but that happens more or less
never. So we round v2. The simplest way is to round all positive entries to 1 and all negative entries to ?1.
But in both theory and practice, it�s better to choose the threshold as follows.]

Spectral partitioning alg:

� Compute Fiedler vector v2 of L
� Round v2 with a sweep cut:
= Sort components of v2.
= Try the n ? 1 cuts between successive components. Choose min-sparsity cut.

[If we�re clever about updating the sparsity, we can try all these cuts in time linear in the number
of edges in G.]

specgraph.pdf, specvector.pdf
Right: what the un-rounded Fiedler vector looks like.]

[Left: example of a graph partitioned by the sweep cut.

5101520(cid:45)0.4(cid:45)0.20.20.40.6172

Jonathan Richard Shewchuk

[One consequence of relaxing the binary constraint is that the balance constraint no longer forces an exact
bisection. But that�s okay; we�re cool with a slightly unbalanced cut if it means we cut fewer edges. Even
though our discrete problem was the minimum bisection problem, our relaxed, continuous problem will be
an approximation of the sparsest cut problem. This is a bit counterintuitive.]

lopsided.pdf [A graph for which an unbalanced cut (left) is sparser than a balanced one
(right).]

Vertex Masses

[Sometimes you want the notion of balance to accord more prominence to some vertices than others. We
can assign masses to vertices.]

Let M be diagonal matrix with vertex masses on diagonal.
New balance constraint: 1?My = 0.
[This new balance constraint says that G1 and G2 should each have the same total mass. It turns out that this
new balance constraint is easier to satisfy if we also revise the sphere constraint a little bit.]
New ellipsoid constraint: y?My = Mass(G) = (cid:80)
[Instead of a sphere, now we constrain y to lie on an axis-aligned ellipsoid.]

Mii.

[Draw this by hand. ellipse.pdf ] [The constraint ellipsoid passes through the points of the
hypercube.]

Now solution is Fiedler vector of generalized eigensystem Lv = ?Mv.
[Most algorithms for computing eigenvectors and eigenvalues of symmetric matrices can easily be adapted
to compute eigenvectors and eigenvalues of symmetric generalized eigensystems too.]

[For the grad students, here�s the most important theorem in spectral graph partitioning.]
(cid:113)

Fact: Sweep cut finds a cut w/sparsity ?

2?2 maxi

Lii
Mii

: Cheeger�s inequality.

The optimal cut has sparsity ? ?2/2.

[So the spectral partitioning algorithm is an approximation algorithm, albeit not one with a constant factor
of approximation. Cheeger�s inequality is a very famous result in spectral graph theory, because it�s one of
the most important cases where you can relax a combinatorial optimization problem to a continuous opti-
mization problem, round the solution, and still have a provably decent solution to the original combinatorial
problem.]

Bonus Lecture: Spectral Graph Clustering

173

Vibration Analogy

vibrate.pdf

[For intuition about spectral partitioning, think of the eigenvectors as vibrational modes in a physical system
of springs and masses. Each vertex models a point mass that is constrained to move freely along a vertical
rod. Each edge models a vertical spring with rest length zero and stiffness proportional to its weight, pulling
two point masses together. The masses are free to oscillate sinusoidally on their rods. The eigenvectors of the
generalized eigensystem Lv = ?Mv are the vibrational modes of this physical system, and their eigenvalues
are proportional to their frequencies.]

grids.pdf [Vibrational modes in a path graph and a grid graph.]

[These illustrations show the first four eigenvectors for two simple graphs. On the left, we see that the first
eigenvector is the eigenvector of all 1�s, which represents a vertical translation of all the masses in unison.
That�s not really a vibration, which is why the eigenvalue is zero. The second eigenvector is the Fiedler
vector, which represents the vibrational mode with the lowest frequency. Each component indicates the
amplitude with which the corresponding point mass oscillates. At any point in time as the masses vibrate,
roughly half the mass is moving up while half is moving down. So it makes sense to cut between the positive
components and the negative components. The third eigenvector also gives us a nice bisection of the grid
graph, entirely different from the Fiedler vector. Some more sophisticated graph clustering algorithms use
multiple eigenvectors.]

[I want to emphasize that spectral partitioning takes a global view of a graph. It looks at the whole gestalt
of the graph and finds a good cut. By comparison, the clustering algorithms we saw last lecture were much
more local in nature, so they�re easier to fool.]

v3v2v1v4174

Jonathan Richard Shewchuk

Greedy Divisive Clustering

Partition G into 2 subgraphs; recursively partition them.
[The sparsity is a good criterion for graph clustering. Use G�s sparsest cut to divide it into two subgraphs,
then recursively cut them. You can stop when you have the right number of clusters. Alternatively, you can
make a finer tree and then prune it back.]

The Normalized Cut

Set vertex i�s mass Mii = Lii. [Sum of edge weights adjoining vertex i.]
[That is how we define a normalized cut, which turns out to be a good choice for many different applica-
tions.]
Popular for image segmentation.
[Image segmentation is the problem of looking at a photograph and separating it into different objects. To
do that, we define a graph on the pixels.]
For pixels with coordinate pi, brightness bi, use graph weights

wi j = exp

?
??????

?pi ? p j?2
?

?

|bi ? b j|2
?

?
?????

or zero if ?pi ? p j? large.

[We choose a distance threshold, typically less than 4 to 10 pixels apart. Pixels that are far from each other
aren�t connected. ? and ? are empirically chosen constants. It often makes sense to choose ? proportional
to the variance of the brightness values.]

baseballsegment.pdf (Shi and Malik, �Normalized Cut and Image Segmentation�)
[A segmentation of a photo of a scene from a baseball game (upper left). The other figures
show segments of the image extracted by recursive spectral partitioning.]

Bonus Lecture: Spectral Graph Clustering

175

baseballvectors.pdf (Shi and Malik) [Eigenvectors 2�9 from the baseball image.]

Invented by [our own] Prof. Jitendra Malik and his student Jianbo Shi.

176

Jonathan Richard Shewchuk

D Bonus Lecture: Multiple Eigenvectors; Latent Factor Analysis

Clustering w/Multiple Eigenvectors

[When we use the Fiedler vector for spectral graph clustering, it tells us how to divide a graph into two
graphs. If we want more than two clusters, we can use divisive clustering: we repeatedly cut the subgraphs
into smaller subgraphs by computing their Fiedler vectors. However, there are several other methods to
subdivide a graph into k clusters in one shot that use multiple eigenvectors rather than just the Fiedler
vector v2. These methods sometimes give better results. They use k eigenvectors in a natural way to cluster
a graph into k subgraphs.]
For k clusters, compute first k eigenvectors v1 = 1, v2, . . . , vk of generalized eigensystem Lv = ?Mv.
Scale them so that v?

1. Now V ?MV = I. [The eigenvectors are M-orthogonal.]

i Mvi = 1. E.g., v1 =

1?(cid:80)

Mii

V1

Vn

v1

V =

=

vk

n � k

[V�s columns are the eigenvectors with the k
smallest eigenvalues.]
[Yes, we do include the all-1�s vector v1 as one of
the columns of V.]

[Draw this by hand. eigenvectors.pdf ]

Row Vi is spectral vector [my name] for vertex i. [The rows are vectors in a k-dimensional space I�ll call the
�spectral space.� When we were using just one eigenvector, it made sense to cluster vertices together if their
components were close together. When we use more than one eigenvector, it turns out that it makes sense to
cluster vertices together if their spectral vectors point in similar directions.]

Normalize each row Vi to unit length.
[Now you can think of the spectral vectors as points on a unit sphere centered at the origin.]

[Draw this by hand vectorclusters.png ] [A 2D example showing two clusters on a circle.
If the graph has k components, the points in each cluster will have identical spectral vectors
that are exactly orthogonal to all the other components� spectral vectors (left). If we modify
the graph by connecting these components with small-weight edges, we get vectors more
like those at right�not exactly orthogonal, but still tending toward distinct clusters.]

k-means cluster these vectors.

[Because all the spectral vectors lie on the sphere, k-means clustering will cluster together vectors that are
separated by small angles.]

Bonus Lecture: Multiple Eigenvectors; Latent Factor Analysis

177

compkmeans.png, compspectral.png [Comparison of point sets clustered by k-means�
just k-means by itself, that is�vs. a spectral method. To create a graph for the spectral
method, we use an exponentially decaying function to assign weights to pairs of points, like
we used for image segmentation but without the brightnesses.]

Invented by [our own] Prof. Michael Jordan, Andrew Ng [when he was still a student at Berkeley], Yair
Weiss.

[This wasn�t the first algorithm to use multiple eigenvectors for spectral clustering, but it has become one of
the most popular.]

178

Jonathan Richard Shewchuk

LATENT FACTOR ANALYSIS [aka Latent Semantic Indexing]

[You can think of this as dimensionality reduction for matrices.]

Suppose X is a term-document matrix:
row i represents document i; column j represents term j.
[Term-document matrices are usually sparse, meaning most entries are zero.]
Xi j = occurrences of term j in doc i
better: log (1+ occurrences)

[aka bag-of-words model]

[Term = word.]

[So frequent words don�t dominate.]
[Better still is to weight the entries so rare words give big entries and common words like �the� give small
entries. To do that, you need to know how frequently each word occurs in general. I�ll omit the details, but
this is the common practice.]

Recall SVD X = UDV ? =

d(cid:88)

i=1

?iuiv?

i . Suppose ?i ? ? j for i ? j.

Unlike PCA, we usually don�t center X.
For large ?i, ui and vi represent a cluster of documents & terms.

�

� Large components in ui mark docus using similar/related terms, i.e., a genre.
�
� E.g., u1 might have large components for the romance novels,
�
�

� vi mark frequent terms in that genre.

for terms �passion,� �ravish,� �bodice� . . .

�

�

�

�

v1

[. . . and ?1 would give us an idea how much bigger the romance novel market is than the markets for every
other genre of books.]

[v1 and u1 tell us that there is a large subset of books that tend to use the same large subset of words. We
can read off the words by looking at the larger components of v1, and we can read off the books by looking
at the larger components of u1.]

[The property of being a romance novel is an example of a latent factor. So is the property of being the sort
of word used in romance novels. There�s nothing in X that tells you explicitly that romance novels exist,
but the similar vocabulary is a hidden connection between them that gives them a large singular value. The
vector u1 reveals which books have that genre, and v1 reveals which words are emphasized in that genre.]

Like clustering, but clusters overlap: if u1 picks out romances &
u2 picks out histories, they both pick out historical romances.

[So you can think of latent factor analysis as a sort of clustering that permits clusters to overlap. Another
way in which it differs from traditional clustering is that the u-vectors contain real numbers, and so some
points have stronger cluster membership than others. One book might be just a bit romance, another a lot.]

Bonus Lecture: Multiple Eigenvectors; Latent Factor Analysis

179

Application in market research:
identifying consumer types (hipster, suburban mom) & items bought together.
[For applications like this, the first few singular vectors are the most useful. Most of the singular vectors are
mostly noise, and they have small singular values to tell you so. This motivates approximating a matrix by
using only some of its singular vectors.]

Truncated SVD X? =

r(cid:88)

i=1

?iuiv?
i

is a low-rank approximation of X, of rank r.

[Assuming ?r > 0.]

[We choose the singular vectors with the largest singular values, because they carry the most information.]

X?

=

u1

ur

n � d

n � r

?1

. . .

0

?r
0
r � r

v1
vr

r � d

[Draw this by hand.

truncate.pdf ]

X? is the rank-r matrix that minimizes the [squared] Frobenius norm

?X ? X??2
F

=

(cid:88)

(cid:16)

(cid:17)2

Xi j ? X?
i j

i, j

Applications:

� Fuzzy search. [Suppose you want to find a document about gasoline prices, but the document you
want doesn�t have the word �gasoline�; it has the word �petrol.� One cool thing about the reduced-
rank matrix X? is that it will probably associate that document with �gasoline,� because the SVD tends
to group synonyms together.]

� Denoising.

[The idea is to assume that X is a noisy measurement of some unknown matrix that
probably has low rank. If that assumption is partly true, then the reduced-rank matrix X? might be
better than the input X.]

� Matrix compression. [As you can see above, if we use a low-rank approximation with a small rank
r, we can express the approximate matrix as an SVD that takes up much less space than the original
matrix. Often this low-rank approximation supports faster matrix computations.]

� Collaborative filtering: fills in unknown values, e.g., user ratings.

[Suppose the rows of X represents Netflix users and the columns represent movies. The entry Xi j is
the review score that user i gave to movie j. But most users haven�t reviewed most movies. We want
to fill in the missing values. Just as the rank reduction will associate �petrol� with �gasoline,� it will
tend to associate users with similar tastes in movies, so the reduced-rank matrix X? can predict ratings
for users who didn�t supply any.]

180

Jonathan Richard Shewchuk

PREDICTING PERSONALITY FROM FACES

hu.pdf

Hu et. al (2017).

Big Five (BF) model of personality:

� O: openness
� C: conscientiousness
� E: extraversion
� A: agreeableness
� N: neuroticism

[Researchers have found that these five personality factors are approximately orthogonal to each other. They
are highly heritable and highly stable during adulthood.]

Can we predict these traits from 3D faces?

[Studies have shown that people looking at photographs of static faces with neutral expressions can iden-
tify the traits better than chance, especially for conscientiousness, extraversion, and agreeableness. This
experiment asks whether machine learning can do the same with 3D reconstructions of faces. The subjects
were 834 Han Chinese volunteers in Shanghai, China. We don�t know whether any of these results might
generalize to people who are not Han Chinese.]

[The faces were scanned in high-resolution 3D and a non-rigid face registration system was used to fit a
grid of 32,251 vertices to each face in a manner that maps each vertex to an appropriate landmark on the
face. (They call this �anatomical homology.�) So the design matrix X was 834 � 100,053, representing 834
subjects with 32,251 3D features each.]

[Subject personalities were evaluated with a self-questionnaire, namely our own Berkeley Personality Lab�s
Big Five Inventory, translated into Chinese. The authors treated men and women separately.]

(cid:153)(cid:153)(cid:153)(cid:484)(cid:144)(cid:131)(cid:150)(cid:151)(cid:148)(cid:135)(cid:484)(cid:133)(cid:145)(cid:143)(cid:512)(cid:149)(cid:133)(cid:139)(cid:135)(cid:144)(cid:150)(cid:139)(cid:136)(cid:139)(cid:133)(cid:148)(cid:135)(cid:146)(cid:145)(cid:148)(cid:150)(cid:149)(cid:12)(cid:20)(cid:19)(cid:22)(cid:15)(cid:27)(cid:28)(cid:25)(cid:17)(cid:26)(cid:1)(cid:23)(cid:18)(cid:1)(cid:24)(cid:17)(cid:25)(cid:26)(cid:23)(cid:22)(cid:15)(cid:21)(cid:20)(cid:27)(cid:29)(cid:1)(cid:23)(cid:22)(cid:1)(cid:16)(cid:17)(cid:22)(cid:26)(cid:17)(cid:1)(cid:889)(cid:7)(cid:3)(cid:136)(cid:131)(cid:133)(cid:139)(cid:131)(cid:142)(cid:3)(cid:139)(cid:143)(cid:131)(cid:137)(cid:135)(cid:149)(cid:12)(cid:20)(cid:21)(cid:17)(cid:1)(cid:4)(cid:28)(cid:887)(cid:32)(cid:1)(cid:5)(cid:20)(cid:17)(cid:29)(cid:20)(cid:1)(cid:14)(cid:20)(cid:23)(cid:22)(cid:19)(cid:887)(cid:481)(cid:888)(cid:481)(cid:3)(cid:19)(cid:135)(cid:144)(cid:137)(cid:133)(cid:138)(cid:135)(cid:144)(cid:137)(cid:3)(cid:3)(cid:28)(cid:889)(cid:32)(cid:1)(cid:7)(cid:28)(cid:1)(cid:11)(cid:20)(cid:15)(cid:23)(cid:887)(cid:32)(cid:1)(cid:5)(cid:20)(cid:22)(cid:19)(cid:30)(cid:17)(cid:1)(cid:13)(cid:15)(cid:22)(cid:890)(cid:32)(cid:1)(cid:7)(cid:20)(cid:1)(cid:5)(cid:20)(cid:22)(cid:890)(cid:1)(cid:31)(cid:1)(cid:6)(cid:28)(cid:22)(cid:1)(cid:13)(cid:15)(cid:22)(cid:19)(cid:887)(cid:12)(cid:150)(cid:3)(cid:138)(cid:131)(cid:149)(cid:3)(cid:142)(cid:145)(cid:144)(cid:137)(cid:3)(cid:132)(cid:135)(cid:135)(cid:144)(cid:3)(cid:149)(cid:146)(cid:135)(cid:133)(cid:151)(cid:142)(cid:131)(cid:150)(cid:135)(cid:134)(cid:3)(cid:150)(cid:138)(cid:131)(cid:150)(cid:3)(cid:133)(cid:151)(cid:135)(cid:149)(cid:3)(cid:145)(cid:144)(cid:3)(cid:150)(cid:138)(cid:135)(cid:3)(cid:138)(cid:151)(cid:143)(cid:131)(cid:144)(cid:3)(cid:136)(cid:131)(cid:133)(cid:135)(cid:3)(cid:135)(cid:154)(cid:139)(cid:149)(cid:150)(cid:3)(cid:150)(cid:138)(cid:131)(cid:150)(cid:3)(cid:131)(cid:142)(cid:142)(cid:145)(cid:153)(cid:3)(cid:145)(cid:132)(cid:149)(cid:135)(cid:148)(cid:152)(cid:135)(cid:148)(cid:149)(cid:3)(cid:150)(cid:145)(cid:3)(cid:143)(cid:131)(cid:141)(cid:135)(cid:3)(cid:148)(cid:135)(cid:142)(cid:139)(cid:131)(cid:132)(cid:142)(cid:135)(cid:3)(cid:140)(cid:151)(cid:134)(cid:137)(cid:143)(cid:135)(cid:144)(cid:150)(cid:149)(cid:3)(cid:145)(cid:136)(cid:3)(cid:145)(cid:150)(cid:138)(cid:135)(cid:148)(cid:149)(cid:495)(cid:3)(cid:146)(cid:135)(cid:148)(cid:149)(cid:145)(cid:144)(cid:131)(cid:142)(cid:139)(cid:150)(cid:155)(cid:3)(cid:150)(cid:148)(cid:131)(cid:139)(cid:150)(cid:149)(cid:484)(cid:3)(cid:11)(cid:145)(cid:153)(cid:135)(cid:152)(cid:135)(cid:148)(cid:481)(cid:3)(cid:134)(cid:139)(cid:148)(cid:135)(cid:133)(cid:150)(cid:3)(cid:135)(cid:152)(cid:139)(cid:134)(cid:135)(cid:144)(cid:133)(cid:135)(cid:3)(cid:145)(cid:136)(cid:3)(cid:131)(cid:149)(cid:149)(cid:145)(cid:133)(cid:139)(cid:131)(cid:150)(cid:139)(cid:145)(cid:144)(cid:3)(cid:132)(cid:135)(cid:150)(cid:153)(cid:135)(cid:135)(cid:144)(cid:3)(cid:136)(cid:131)(cid:133)(cid:139)(cid:131)(cid:142)(cid:3)(cid:149)(cid:138)(cid:131)(cid:146)(cid:135)(cid:149)(cid:3)(cid:131)(cid:144)(cid:134)(cid:3)(cid:146)(cid:135)(cid:148)(cid:149)(cid:145)(cid:144)(cid:131)(cid:142)(cid:139)(cid:150)(cid:155)(cid:3)(cid:139)(cid:149)(cid:3)(cid:143)(cid:139)(cid:149)(cid:149)(cid:139)(cid:144)(cid:137)(cid:3)(cid:136)(cid:148)(cid:145)(cid:143)(cid:3)(cid:150)(cid:138)(cid:135)(cid:3)(cid:133)(cid:151)(cid:148)(cid:148)(cid:135)(cid:144)(cid:150)(cid:3)(cid:142)(cid:139)(cid:150)(cid:135)(cid:148)(cid:131)(cid:150)(cid:151)(cid:148)(cid:135)(cid:484)(cid:3)(cid:23)(cid:138)(cid:139)(cid:149)(cid:3)(cid:149)(cid:150)(cid:151)(cid:134)(cid:155)(cid:3)(cid:131)(cid:149)(cid:149)(cid:135)(cid:149)(cid:149)(cid:135)(cid:134)(cid:3)(cid:150)(cid:138)(cid:135)(cid:3)(cid:146)(cid:135)(cid:148)(cid:149)(cid:145)(cid:144)(cid:131)(cid:142)(cid:139)(cid:150)(cid:155)(cid:3)(cid:131)(cid:150)(cid:150)(cid:148)(cid:139)(cid:132)(cid:151)(cid:150)(cid:135)(cid:149)(cid:3)(cid:145)(cid:136)(cid:3)(cid:894)(cid:889)(cid:890)(cid:3)(cid:11)(cid:131)(cid:144)(cid:3)(cid:6)(cid:138)(cid:139)(cid:144)(cid:135)(cid:149)(cid:135)(cid:3)(cid:152)(cid:145)(cid:142)(cid:151)(cid:144)(cid:150)(cid:135)(cid:135)(cid:148)(cid:149)(cid:3)(cid:523)(cid:890)(cid:886)(cid:891)(cid:3)(cid:143)(cid:131)(cid:142)(cid:135)(cid:149)(cid:3)(cid:131)(cid:144)(cid:134)(cid:3)(cid:890)(cid:888)(cid:895)(cid:3)(cid:136)(cid:135)(cid:143)(cid:131)(cid:142)(cid:135)(cid:149)(cid:524)(cid:481)(cid:3)(cid:151)(cid:150)(cid:139)(cid:142)(cid:139)(cid:149)(cid:139)(cid:144)(cid:137)(cid:3)(cid:150)(cid:138)(cid:135)(cid:3)(cid:420)(cid:152)(cid:135)(cid:486)(cid:136)(cid:131)(cid:133)(cid:150)(cid:145)(cid:148)(cid:3)(cid:146)(cid:135)(cid:148)(cid:149)(cid:145)(cid:144)(cid:131)(cid:142)(cid:139)(cid:150)(cid:155)(cid:3)(cid:143)(cid:145)(cid:134)(cid:135)(cid:142)(cid:3)(cid:523)(cid:494)(cid:5)(cid:139)(cid:137)(cid:3)(cid:9)(cid:139)(cid:152)(cid:135)(cid:495)(cid:524)(cid:481)(cid:3)(cid:131)(cid:144)(cid:134)(cid:3)(cid:133)(cid:145)(cid:142)(cid:142)(cid:135)(cid:133)(cid:150)(cid:135)(cid:134)(cid:3)(cid:150)(cid:138)(cid:135)(cid:139)(cid:148)(cid:3)(cid:144)(cid:135)(cid:151)(cid:150)(cid:148)(cid:131)(cid:142)(cid:3)(cid:889)(cid:7)(cid:3)(cid:136)(cid:131)(cid:133)(cid:139)(cid:131)(cid:142)(cid:3)(cid:139)(cid:143)(cid:131)(cid:137)(cid:135)(cid:149)(cid:484)(cid:3)(cid:7)(cid:135)(cid:144)(cid:149)(cid:135)(cid:3)(cid:131)(cid:144)(cid:131)(cid:150)(cid:145)(cid:143)(cid:139)(cid:133)(cid:131)(cid:142)(cid:3)(cid:133)(cid:145)(cid:148)(cid:148)(cid:135)(cid:149)(cid:146)(cid:145)(cid:144)(cid:134)(cid:135)(cid:144)(cid:133)(cid:135)(cid:3)(cid:153)(cid:131)(cid:149)(cid:3)(cid:135)(cid:149)(cid:150)(cid:131)(cid:132)(cid:142)(cid:139)(cid:149)(cid:138)(cid:135)(cid:134)(cid:3)(cid:131)(cid:133)(cid:148)(cid:145)(cid:149)(cid:149)(cid:3)(cid:150)(cid:138)(cid:135)(cid:3)(cid:889)(cid:7)(cid:3)(cid:136)(cid:131)(cid:133)(cid:139)(cid:131)(cid:142)(cid:3)(cid:139)(cid:143)(cid:131)(cid:137)(cid:135)(cid:149)(cid:3)(cid:139)(cid:144)(cid:3)(cid:145)(cid:148)(cid:134)(cid:135)(cid:148)(cid:3)(cid:150)(cid:145)(cid:3)(cid:131)(cid:142)(cid:142)(cid:145)(cid:153)(cid:3)(cid:138)(cid:139)(cid:137)(cid:138)(cid:486)(cid:134)(cid:139)(cid:143)(cid:135)(cid:144)(cid:149)(cid:139)(cid:145)(cid:144)(cid:131)(cid:142)(cid:3)(cid:147)(cid:151)(cid:131)(cid:144)(cid:150)(cid:139)(cid:150)(cid:131)(cid:150)(cid:139)(cid:152)(cid:135)(cid:3)(cid:131)(cid:144)(cid:131)(cid:142)(cid:155)(cid:149)(cid:135)(cid:149)(cid:3)(cid:145)(cid:136)(cid:3)(cid:11)(cid:16)(cid:14)(cid:16)(cid:18)(cid:26)(cid:16)(cid:15)(cid:7)(cid:1)(cid:4)(cid:6)(cid:1)(cid:9)(cid:25)(cid:19)(cid:28)(cid:1)(cid:4)(cid:2)(cid:3)(cid:5)(cid:8)(cid:14)(cid:14)(cid:16)(cid:21)(cid:24)(cid:16)(cid:15)(cid:7)(cid:1)(cid:4)(cid:6)(cid:1)(cid:9)(cid:12)(cid:20)(cid:25)(cid:12)(cid:22)(cid:28)(cid:1)(cid:4)(cid:2)(cid:3)(cid:6)(cid:10)(cid:25)(cid:13)(cid:19)(cid:18)(cid:23)(cid:17)(cid:16)(cid:15)(cid:7)(cid:1)(cid:27)(cid:27)(cid:1)(cid:27)(cid:27)(cid:1)(cid:27)(cid:27)(cid:27)(cid:27)(cid:9)(cid:10)(cid:2)(cid:8)Bonus Lecture: Multiple Eigenvectors; Latent Factor Analysis

181

Uses partial least squares (PLS) to find associations between personality & faces.

[Everything from here to the end is spoken, not written.]

Partial least squares (PLS) is like a supervised version of PCA. It takes in two matrices X and Y with the
same number of rows. In our example, X is the face data and Y is the personality data for the 834 subjects.
Like PCA, PLS finds a set of vectors in face space that we think of as the most important components. But
whereas PCA looks for the directions of maximum variation in X, PLS looks for the directions in X that
maximize the correlation with the personality traits in matrix Y.

The researchers found the top 20 or so PLS components and used cross-validation to decide which compo-
nents have predictive power for each personality trait. They found that the top two components for extraver-
sion in women were predictive, but no components for the other four traits in women were predictive. Men
are easier to analyze: they found two or three components were predictive for each of extraversion, agree-
ableness, conscientiousness, and neuroticism in men. However, the correlations were statistically significant
only for agreeableness and conscientiousness.

male.pdf
[The relationship between male faces, agreeableness, and conscientiousness.
The large, colored faces are the mean faces. Colors indicate the values in the most pre-
dictive PLS component vector.]

More agreeable men correlate with much wider mouths that look a bit smiley even when neutral; stronger,
forward jaws; wider noses; and shorter faces, especially shorter in the forehead, compared to less agreeable
men. More conscientious men tend to have higher, wider eyebrows; wider, opened eyes; a withdrawn upper
lip with more mouth tension; and taller faces with more pronounced brow ridges (the bone protuberance
above the eyes). The authors note that men with low A and C scores look both more relaxed and more
indifferent.

182

Jonathan Richard Shewchuk

female.pdf [The relationship between female faces and extraversion. The large, colored
face is the mean face. Colors illustrate the most predictive PLS component vector.]

More extraverted women correlate with rounder faces, especially in profile, with a more protruding nose
and lips but a recessed chin, whereas the introverts have more flat, square-shaped faces. To my eyes, the
extraverts also have more expressive mouths.

It�s interesting is that physiognomy, the art of judging character from facial shape, used to be considered
a pseudoscience, but it�s been making a comeback in recent years with the help of machine learning. One
reason it fell into disrepute is because, historically, it was sometimes applied across races in fallacious and
insulting ways. But if you want to train classifiers that guess people�s personalities with some accuracy, you
probably need a different classifier for each race. This is a classifier trained exclusively for one race, Han
Chinese, which is probably part of why it works as well as it does. If you tried to train one classifier to work
on many different races, I suspect its performance would be much worse.

Another thing that�s notable is that the authors were able to find statistically significant correlations for some
personality traits, but the majority of traits defeated them. So while physiognomy has some predictive power,
it�s only weakly predictive. It�s an open question whether machine learning will ever be able to predict
personality from visual information substantially better than this or not. Adding a time dimension and
incorporating people�s movements and dynamic facial expressions seems like a promising way to improve
personality predictions.

Tools like this raise some ethical issues. The one that concerns me the most is that, if tools like this are
emerging now, many governments probably already had similar tools ten years ago, and have probably been
using them to profile us.

One student asked whether these methods might be used by employers to screen prospective employees.
I think that tools like this are inferior to simply giving an interviewee a personality test. Such tests are legal
in the USA, so long as their questions are not found to violate an employee�s right to privacy and the results
are not used to discriminate against legally protected groups. The most troubling part of using physiognomy
to screen employees would not be that personality testing is unlawful. (It isn�t, and quite a few companies
do it.) It would be that physiognomy isn�t nearly accurate enough. An employer who uses a poorly designed
or unvalidated personality test to make personnel decisions might run a higher risk that a court might rule
that the test could have a discriminatory effect, violating Title VII of the Civil Rights Act of 1964. Also,
they probably won�t make good decisions. But perhaps in the future, better measurements, better statistical
procedures, and better algorithms might overcome these problems.

Bonus Lecture: High Dimensions; Random Projection

183

E Bonus Lecture: High Dimensions; Random Projection

THE GEOMETRY OF HIGH-DIMENSIONAL SPACES

[High-dimensional geometry sometimes acts in ways that are completely counterintuitive, defying our intu-
itions from low-dimensional geometry.]

Consider a random point p ? N(0, I) ? Rd.
What is the distribution of its length?

[Looking at the one-dimensional normal distribu-
tion, you would expect it to be very common that
the length is close to zero, a bit less common that
the length is close to 1 or ?1, and not rare for the
length to be close to 2 or ?2. But in high dimen-
sions, that intuition is completely wrong.]

normal.pdf [A one-dimensional normal distribution.]

[If the dimension is very high, the vast majority of the random points are at approximately the same distance
from the mean. So they lie in a thin shell. Why? To answer that, let�s study the square of the distance. By
Pythagoras� Theorem, the squared distance from p to the mean is]

?p?2 = p2
1

+ p2
2

+ . . . + p2
d

[Each component pi is sampled independently from a univariate normal distribution with mean zero and
variance one. The square of a component, p2
i , is said to come from a chi-squared distribution. So is ?p?2.]

pi ? N(0, 1),

i ? ?2(1),
p2

E[p2

i ] = 1,

Var(p2

i ) = 2,

?p?2 ? ?2(d)

[Recall that when you add d independent, identically distributed random numbers, you scale their mean and
variance by d, and the standard deviation is the square root of the variance.]

E[?p?2] = d E[p2
Var(?p?2) = d Var(p2
SD(?p?2) =

2d

?

1] = d
1) = 2d

d with a thickness proportional to 4?
?

?

For large d, ?p? is concentrated in a thin shell around radius
[The mean value of ?p? isn�t exactly
deviation is much, much smaller. Likewise, the standard deviation of ?p? isn�t exactly 4?
[So if d is about a million, imagine a million-dimensional egg whose radius is about 1,000, and the thickness
of the shell is about 67, which is about 10 times the standard deviation. The vast majority of random points
are in the eggshell. Not inside the egg; actually in the shell itself. It is counterintuitive that random vectors
sampled from a high-dimensional normal distribution almost all have almost the same length.]

d, but it is close, because the mean of ?p?2 is d and the standard
2d, but it�s close.]

2d.

[There is a statistical principle hiding here. Suppose you want to estimate the mean of a distribution�in
this case, the distribution ?2(1). The standard way to do that is to sample very many numbers from the
distribution and take their mean. The more numbers you sample, the more accurate your estimate is�that
is, the smaller the standard deviation of your sample mean is. When we sample a vector from a million-
dimensional normal distribution and compute its length, that�s exactly what we�re doing!]

-3-2-1123x0.10.20.30.4f(x)184

Jonathan Richard Shewchuk

What about a uniform distribution? Consider concentric spheres of radii r & r ? ?.

r ? ?

r

[Draw this by hand concentric.pdf ] [Concentric balls. In high dimensions, almost every
point chosen uniformly at random in the outer ball lies outside the inner ball.]

Volume of outer ball ? rd
Volume of inner ball ? (r ? ?)d
Ratio of inner ball volume to outer =
(cid:18)

(cid:19)d

=

1 ?

? exp

(r ? ?)d
rd

(cid:32)
?

?d
r

?
r

(cid:33)

which is small for large d.

E.g., if

?
r

= 0.1 & d = 100, inner ball has 0.9100 = 0.0027% of volume.

Random points from uniform distribution in ball: nearly all are in thin outer shell.

�

�

� Gaussian

�

: nearly all are in some thin shell.

Consequences:

� In high dimensions, sometimes the nearest neighbor and 1,000th-nearest neighbor don�t differ much!
� k-means clustering and nearest neighbor classifiers are less effective for large d.

Angles between Random Vectors

What is the angle ? between a random p ? N(0, I) ? Rd and an arbitrary q ? Rd?
Without loss of generality, set q = [1 0
[The value of q doesn�t matter, because the direction that p points in is uniformly distributed over all possible
directions. By a formula we learned early this semester, the angle between p and q is ?, where . . . ]

0 . . . 0]?.

cos ? =

p � q
?p? ?q?

= p1
?p?

E[cos ?] = 0;

SD(cos ?) ?

SD(p1)
?
d

= 1
?
d

If d is large, cos ? is almost always very close to zero; ? is almost always very close to 90?!

[In high-dimensional spaces, two random vectors are almost always very close to orthogonal. To put it
another way, an arbitrary vector is almost orthogonal to the vast majority of all the other vectors!]
[A former CS 189/289A head TA, Marc Khoury, has a nice short essay entitled �Counterintuitive Properties
of High Dimensional Space�, which you can read at https://people.eecs.berkeley.edu/?jrs/highd ]

Bonus Lecture: High Dimensions; Random Projection

185

RANDOM PROJECTION

An alternative to PCA as preprocess for clustering, classification, regression.
Approximately preserves distances between points!

[We project onto a random subspace instead of the PCA subspace, but sometimes it preserves distances
better than PCA. Because it roughly preserves the distances, algorithms like k-means clustering and nearest
neighbor classifiers will give similar results to what they would give in high dimensions, but they run much
faster. It works best when you project a very high-dimensional space to a medium-dimensional space.]

Pick a small ?, a small ?, and a random subspace S ? Rd of dimension k, where k =

For any pt q, let �q be orthogonal projection of q onto S , multiplied by

?

[The multiplication by

d/k helps preserve the distances between points after you project.]

(cid:113)

d
k .

(cid:38)

2 ln(1/?)
?2/2 ? ?3/3

(cid:39)

.

Johnson�Lindenstrauss Lemma (modified):
For any two pts q, w ? Rd, (1 ? ?) ?q ? w?2 ? ? �q ? �w?2 ? (1 + ?) ?q ? w?2 with probability ? 1 ? 2?.
Typical values: ? ? [0.02, 0.5], ? ? [1/n3, 0.05].

[You choose ? and ? according to your needs.]

[The squared distance between two points after projecting changes by less than 2%, or less than 50%, as you
wish. In practice, experiment with k to find the best speed-accuracy tradeoff. If you want all inter-sample-
point distances to be accurate, you should set ? smaller than 1/n2, so you need a subspace of dimension
?(log n). Reducing ? doesn�t cost much (because of the logarithm), but reducing ? costs more. You can
bring 1,000,000 sample points down to a 10,000-dimensional space with at most a 6% error in the distances.]
[What is remarkable about this result is that the dimension d of the input points doesn�t matter!]

100000to1000.pdf [Comparison of inter-point distances before and after projecting points
in 100,000-dimensional space down to 1,000 dimensions.]

[Why does this work? A random projection of q ? w is like taking a random vector and selecting k compo-
nents. The mean of the squares of those k components approximates the mean for the whole population.]

[How do you get a uniformly distributed random projection direction? You can choose each component
from a univariate Gaussian distribution, then normalize the vector to unit length. How do you get a random
subspace? You can choose k random directions, then use Gram�Schmidt orthogonalization to make them
mutually orthonormal. Interestingly, Indyk and Motwani show that if you skip the expensive normalization
and Gram�Schmidt steps, random projection still works almost as well, because random vectors in a high-
dimensional space are nearly equal in length and nearly orthogonal to each other with high probability.]

JLExperimentsData:20-newsgroups,from100.000featuresto1.000(1%)MATLABimplementation:1/sqrt(k).*randn(k,N)%*%X.
