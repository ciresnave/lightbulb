Journal of Information Systems Engineering and Management
2025,  10(13s)
e-ISSN:  2468-4376
https://www.jisem-journal.com/

Research Article

A Review: Data Overfitting and Underfitting Techniques

Kiran Maharana1, Ravita Mishra2, Bhushankumar Nemade3*, Deven Shah4, Manish Rana5, Namdeo
Badhe6, Vikas Kaul7
1Researcher, ICICI Lombard GIC Ltd, Mumbai, India

2Vivekanand Education Society�s Institute of Technology, Mumbai, India.

5St. John College of Engineering and Management, Palghar, Mumbai, India.

6Thakur College of Engineering and Technology, Mumbai, India.

 3,4,7L. R Tiwari College of Engineering, Mumbai, India

3*bnemade@gmail.com

ARTICLE INFO

ABSTRACT

Received: 01 Dec 2024

Revised: 15 Jan 2025

Accepted: 30 Jan 2025

Modern  problem  statements  demand  advanced,  futuristic,  cutting-edge,  progressive  and
trend-setting  solutions.  In  today�s  landscape,  Artificial  Intelligence  (AI)  stands  out  as  a
revolutionary,  data-driven  technology  reshaping  our  perception  of  modern  applications.
However, when AI  interacts  with  Machine  Learning  (ML),  another  data-driven  domain,  a
significant  hurdle  emerges  concerning  the  quality  and  quantity  of  data.  This  problem
profoundly impacts the performance of ML Models, prompting engineers and architects to
address a myriad of questions before selecting a suitable mode. Consequently, altering the
architecture and models becomes arduous due to data-related issues, which can be numerous.
Hence, a solution must be sought that caters to both needs. This review paper delves into the
various challenges related to Data Overfitting and Data Underfitting, exploring methods for
identification,  key  questions  for  consideration,  underlying  reasons,  techniques,  real-life
examples, synthetic data illustrations, and a plethora of case studies from diverse research
papers showcasing how they tackled the problem and how overfitting and underfitting can be
solved.

Keywords: Overfitting, Underfitting, Machine Learning (ML), Model Performance, Cross-
Validation, Regularization, Hyperparameter Tuning,  Feature Selection, Model Complexity,
Feature Engineering.

INTRODUCTION

Data  overfitting  and  underfitting  are  critical  challenges  in  machine  learning  that  significantly  impact  the
performance and generalization ability of models. Overfitting occurs when a model learns to capture noise or
random fluctuations in the training data, leading to poor performance on unseen data [1]. On the other hand,
underfitting arises when a model is too simple to capture the underlying patterns in the data, resulting in poor
performance on both the training and validation sets [2].

These  phenomena  have  profound  implications  for  the  success  of  machine  learning  models  across  various
domains. For instance, in a study by Hastie et al. (2009), it was found that overfitting occurs in approximately
30%  of  machine  learning  models  trained  on  real-world  datasets.  Similarly,  in  Kaggle  competitions,  where
participants  submit  machine  learning  models  to  solve  specific  challenges,  overfitting  affects  nearly  40%  of
submissions. These statistics underscore the prevalence and impact of overfitting in practical machine learning
applications [3].

On  the  other  hand,  underfitting  is  also  a  significant  concern,  particularly  in  scenarios  involving  small-scale
datasets. A meta-analysis by Wolpert (1996) revealed that underfitting occurs in approximately 20% of machine
learning models across various domains. Small-scale datasets are particularly susceptible to underfitting due to
the limited amount of information available for the model to learn from.

Given  the  pervasive  nature  of  overfitting  and  underfitting,  it  is  crucial  for  researchers  and  practitioners  to
develop effective techniques to mitigate these challenges. This review paper aims to provide an overview of state-
of-the-art techniques for addressing data overfitting and underfitting in machine learning models. By examining
existing methodologies and their effectiveness across different scenarios, this paper seeks to offer insights into
best practices for improving model performance and generalization.

Furthermore, Underfitting occurs when a model fails to capture the underlying patterns in the training data.
This  results  in  poor  performance  on  both  the  training  and  unseen  (test)  data.  Studies  have  shown  that
underfitting can lead to an average accuracy drop of 10-30% compared to a well-generalized model [4]. For
example, imagine training a spam filter model on a dataset of emails. If the model is underfitted, it might not

Copyright � 2024 by Author/s and Licensed by JISEM. This is an open access article distributed under the Creative Commons Attribution License

which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.

388

J INFORM SYSTEMS ENG, 10(13s)

learn the nuances of spam emails and end up classifying legitimate emails as spam (high false positives) or miss
actual  spam  emails  altogether  (high  false  negatives)  [5].  Overfitting  happens  when  a  model  memorizes  the
training data too well, including noise and irrelevant details. This leads to excellent performance on the training
data but poor performance on unseen data. Statistics suggest that overfitting can inflate a model's accuracy on
the training data by 15-50% compared to its actual performance on unseen data [6]. For instance, consider a
stock price prediction model. If the model overfits, it might learn specific patterns in the historical data that
don't generalize well to future trends. This could lead to inaccurate predictions and significant financial losses
[7].

1.1 Key Questions to Address

What are the primary causes of data overfitting and underfitting?

How do regularization techniques help mitigate overfitting and underfitting?

What role does cross-validation play in preventing overfitting and underfitting?

How do ensemble methods improve model robustness against overfitting and underfitting?

What are the trade-offs associated with different techniques for addressing overfitting and underfitting?

How can practitioners select the most suitable technique for their specific dataset and modeling goals?

What  are  the  key  considerations  when  evaluating  model  performance  in  the  context  of  overfitting  and
underfitting? [2]

What emerging trends and future directions are shaping the field of combating overfitting and underfitting in
machine learning? [1]

Addressing  these  questions  will  provide  a  comprehensive  understanding  of  the  challenges  posed  by  data
overfitting  and  underfitting  and  offer  valuable  insights  into  effective  strategies  for  improving  model
performance and generalization [5].

Considering the  above key  questions  and concept of  overfitting  and underfitting  to understand  the  problem
using a real-life example from the field of healthcare: predicting patient readmission risk in hospitals.

Overfitting Example:

Imagine a predictive model trained to forecast the likelihood of patient readmission based on various medical
features  such  as  age,  diagnosis,  and  length  of  hospital  stay.  If  the  model  overfits,  it  might  capture  noise  or
anomalies  present  in  the  training  data,  such  as  temporary  fluctuations  in  patient  outcomes  or  specific
idiosyncrasies of certain hospital units.

For instance, the model might erroneously learn that patients with a particular diagnosis code are highly likely
to be readmitted, even though this correlation is spurious and not reflective of a genuine medical relationship.
As  a  result,  the  overfitted  model  might  make  overly  optimistic  predictions  of  readmission  risk,  leading
healthcare providers to allocate unnecessary resources or interventions to low-risk patients, while potentially
missing opportunities to intervene with high-risk patients who are not accurately identified.

Underfitting Example:

Conversely,  if  the  predictive  model underfits  the  data,  it  fails  to  capture  the  complex  relationships between
patient characteristics and readmission risk. For instance, it might oversimplify the prediction task by ignoring
relevant features or assuming linear relationships where nonlinear patterns exist.

In this scenario, the underfitted model might struggle to distinguish between patients with different readmission
risks, leading to suboptimal decision-making in clinical practice. Healthcare providers relying on this model
may fail to identify patients at high risk of readmission, leading to inadequate preventive measures or follow-up
care. As a result, patient outcomes may suffer, and healthcare resources may be inefficiently allocated.

In both cases, striking the right balance between model complexity and generalization is crucial for developing
accurate  and  reliable  predictive  models  in  healthcare  and  other  domains.  This  illustrates  the  importance  of
addressing overfitting and underfitting to ensure the effectiveness and reliability of machine learning models in
real-world applications.

LITERATURE SURVEY

Ying,  Xue.  (2019).  An  Overview  of  Overfitting  and  its  Solutions.  Journal  of  Physics:  Conference  Series  [8].
Overfitting is a significant issue in supervised machine learning, hindering model generalization. It occurs due
to noise, limited training sets, and classifier complexity. Strategies to reduce overfitting include early-stopping,
network  reduction,  data-expansion,  and  regularization.  These  strategies  aim  to  prevent  overfitting,  exclude
noises, fine-tune hyper-parameters, and ensure model performance in real-world scenarios.

L�pez, et. al, Overfitting, Model Tuning, and Evaluation of Prediction Performance [9]. This study examines

389

J INFORM SYSTEMS ENG, 10(13s)

overfitting  and  underfitting  in  machine  learning,  distinguishing  their  impacts.  It  highlights  the  trade-off
between  prediction  accuracy  and  model  interpretability,  contrasting  explanatory  and  predictive  modeling
approaches.  The  research  emphasizes  cross-validation  and  model  tuning  to  mitigate  these  issues,  offering
insights  for  optimal  performance.  Essential  evaluation  metrics  enable  robust  assessment  across  different
response variables, enhancing decision-making in model deployment.

Xu, et. al, Empirical Study of Overfitting in Deep Learning for Predicting Breast Cancer Metastasis. Cancers
[10].  This  empirical  study  investigates  overfitting  in  deep  learning  for  breast  cancer  metastasis  prediction.
Utilizing  feedforward  neural  network  (FNN)  models,  overfitting's  detrimental  impact  on  prediction
performance is observed, with hyperparameter settings significantly influencing outcomes. Through grid search
experiments on 11 hyperparameters, such as iteration-based decay, learning rate, batch size, L2, and L1, their
effects  on  overfitting  and  model  performance  are  elucidated.  Notably,  learning  rate,  decay,  and  batch  size
emerge  as  critical  influencers.  These  findings  illuminate  the  nuanced  interplay of  hyperparameters,  offering
insights for optimizing FNN models in clinical prediction tasks.

Abdul Salam, et. al, The Effect of Different Dimensionality Reduction Techniques on ML Overfitting [11]. This
paper investigates the impact of different dimensionality reduction techniques on machine learning overfitting.
It addresses the challenge of training models with datasets containing numerous attributes, which often leads
to  overfitting.  Nine  dimensionality  reduction  methods  are  compared,  including  missing-values  ratio,  low
variance filter, high correlation filter, random forest, PCA, LDA, backward feature elimination, forward feature
construction, and rough set theory. Results demonstrate that the random forest classifier effectively mitigates
overfitting  while  maintaining  or  improving  performance.  The  study  underscores  the  importance  of
dimensionality reduction in enhancing model efficiency and accuracy, offering valuable insights for optimizing
machine learning models.

J. Kolluri, et. al, Reducing Overfitting Problem in Machine Learning Using Novel L1/4 Regularization Method
[12]. The machine learning model faces overfitting and underfitting issues. Underfitting arises from inadequate
complexity,  while  overfitting  results  from  excessive  data.  Regularization  techniques  like  Lasso  and  L2
regularization  address  these  issues.  A  novel  method,  L1/4  regularization,  efficiently  tackles  overfitting,
especially in gene data analysis. It's concluded that L1/4 regularization outperforms other methods in mitigating
overfitting.

Gu  Y,  et.  al,  An  Optimal  Sample  Data  Usage  Strategy  to  Minimize  Overfitting  and  Underfitting  Effects  in
Regression  Tree  Models  Based  on  Remotely-Sensed  Data  [13].  Regression tree  models  play  a pivotal  role  in
ecosystem mapping, yet they often grapple with overfitting and underfitting issues. This study introduces a novel
approach to optimize model accuracy and robustness by devising a sampling strategy and rule selection method.
Leveraging Landsat 8 data, the study showcases that a six-rule model trained on 80% of the dataset achieves
the  lowest  prediction  errors.  This  methodology offers  valuable  insights  for  enhancing  remote  sensing-based
ecosystem modeling by mitigating overfitting and underfitting effects, thereby facilitating accurate and reliable
mapping of ecosystem parameters like biomass, cover, and carbon flux.

Liu  Z.,  et.  al,  Dropout  Reduces  Underfitting  [14].  The  paper  discusses  how  dropout,  known  for  addressing
overfitting,  can  also  mitigate  underfitting  by  reducing  gradient  variance  early  in  training.  It  proposes  early
dropout  for  underfitting  models  and  late  dropout  for  overfitting  ones.  Experiments  show  improved
generalization  on  ImageNet  and  vision  tasks.  This  suggests  dropout's  broader  utility  in  neural  network
regularization,  especially  with  large  datasets.  The  availability  of  code  encourages  replication  and  further
investigation.

Bu,  C.,  et.  al,  Research  on  Overfitting  Problem  and  Correction  in Machine  Learning  [15].  Machine  learning,
pivotal in artificial intelligence, relies on training data for problem-solving. Discrepancies between input and
training  data  often  lead  to  overfitting,  impeding  success.  Generalization,  reflecting  real  data  characteristics
accurately, is critical. To understand learning and generalization, we studied polynomial curve fitting. Matrix
theory  aided  in  deriving  polynomial  fits  from training  data.  They tackled  overfitting  through numerical  and
maximum  likelihood  analyses,  then outlined  regularization methods to  enhance  generalization  and  mitigate
overfitting.

Kiran  Maharana,  et.  al,  A  review:  Data  pre-processing  and  data  augmentation  techniques  [16].  The  paper
explores data preprocessing and augmentation techniques to enhance machine learning model performance. It
addresses  issues  such  as  data  quality  and  overfitting  in  deep  learning.  This  consumes  50%  to  80%  of  the
classification process, and augmentation techniques to address overfitting in deep learning models due to data
scarcity. Techniques like data transformation and augmentation methods like geometric transformations are
discussed.  It  emphasizes  the  importance  of  correct  data  handling  and  augmentation  for  improved  model
generalization and accuracy, advocating for further research in automated data preparation procedures.

C.  Khosla  et.  al,  Enhancing  Performance  of  Deep  Learning  Models  with  different  Data  Augmentation
Techniques:  A  Survey  [17].  The  survey  paper  explores  data  augmentation  techniques  in  deep  convolutional

390

J INFORM SYSTEMS ENG, 10(13s)

neural networks (CNNs) for computer vision tasks, emphasizing the critical role of large training datasets. It
discusses  methods  to  enhance  model  performance  through  label-preserving  transformations.  While
comparative  studies  are  limited,  one  study  compared  techniques  like  GANs  and  cropping,  finding  flipping,
cropping, rotation, and WGAN to be effective. Another study on CIFAR-10 highlighted augmentation's potential
to boost accuracy by up to 2.83%. Further analysis considers design decisions like test-time augmentation and
dataset size.

Cuthbert, L., et. al, Moving Vehicle Detection and Classification Using Gaussian Mixture Model and Ensemble
Deep Learning Technique [18]. This research introduces an ensemble DL technique for vehicle classification in
traffic surveillance, aiming to improve real-time performance and accuracy. By integrating AHE and feature
extraction  methods  such  as  SPT  and  WLD,  the  proposed  technique  achieved  significant  enhancements  in
classification accuracy. Specifically, on the MIOvision Traffic Camera Dataset and the BIT Vehicle Dataset, the
ensemble DL technique demonstrated remarkable classification accuracies of 99.13% and 99.28%, respectively.
These results represent substantial improvements over existing benchmark techniques, showcasing a maximum
increase of 11.17% in classification accuracy. The adoption of hybrid feature descriptors further reduced training
time, improved classification accuracy, and mitigated overfitting issues, enhancing the overall performance of
the ensemble DL approach.

Ravita et al. (2020) offer a paradigm that they name the Inductive Learning Approach-5 (ILA-5), with the goal
of  reducing  the  dependence  on  human  labor  and  improving  how  their  results  are  interpreted.  ILA-5  is  a
supervised  learning  model  that  generates  recommendations  based  on  a  predetermined  set  of  rules  for
categorizing  data.  The  model  is  "trained"  on  a  sizable  dataset  taken  from  the  real  world  and  consisting  of
resumes and job postings. The authors examine the usefulness of ILA-5 by contrasting it to a number of different
baseline models in order to evaluate its effectiveness and interpretability. The findings demonstrate that the
ILA-5 model may produce suggestions that are reliable and easy to understand [19].

 CONCEPT OF OVERFITTING

 The undesired tendency of the Machine Learning model to predict outcomes accurately for training data but
not for unseen data or testing data is known as Overfitting. In mathematical modelling, overfitting is termed as
creation of analysis that corresponds too closely or perfectly to a given set of data, but drastically fails to fit data
or correctly forecast future observations [19].

It occurs when model is trained with data with a large amount of data with high variance. The model tends to
learn noise, inaccurate data and as there are too many features available and, the model is not able to categorize
the data properly.

The main cause of overfitting is when the non-linear methods is used to build non-realistic data models using
linear data.

What happens when data is overfit? The data works well with model during training, but it may be wrong for
testing set.

What is the simple solution to avoid overfitting? A simple solution is to use linear algorithm with linear data or
having a maximal depth for decision tree.

A.

Reasons

?

?

?

?

?

?

?

Training dataset does not contain enough data

Training dataset contain noisy data or irrelevant information

Training is very high

The Model Complexity is high

Too many parameters

Parameters range of values

Low Bias and High Variance

B.

Identify

The simple methods to detect overfit issues is to test the data on models and its comprehensive behaviour: [24]

Accuracy of Training and Testing Set: When accuracy of training model is high whereas for testing model
the accuracy is very low. For   example, accuracy of training set is 99% and testing set is 60%.

Accuracy is Constant: Plotting Training Error and Validation Error

Overfitting can be identified at different stages of the machine learning life cycle where holdout method and
validation  set  testing  can  be  used.  Plotting  training  error  and  validation  error  can  be  used  to  estimate  the
threshold at which overfitting occurs [20]. As shown below, the model is expecting to decrease both. However,

391

J INFORM SYSTEMS ENG, 10(13s)

after the threshold point the validation error increases and training error simultaneously decreases. And in case
of further training the model may overfit. Figure 3.1 depicts the training error over validation error.

Figure 3.1  Training Error and Validation Error

    1.

If validation loss is smaller than training loss

2.

3.

4.

5.

6.

Training accuracy is greater than validation accuracy

After N epochs, training loss ceased to decrease.

The training loss rises rather than to fall.

Training loss is equivalent to zero.

Validation loss is either unstable or constant.

                                                          Figure 3.2 Plot of Epochs Vs Accuracy

In this the training set is split into K-folds or sets contained equal number of samples. This   includes several
iterations   in the training process. Steps to follow are �

1.

2.

3.

Use one subset as the validation data and the rest K-1 subsets to train the machine learning model.

Track the model's effectiveness on the validation sample.

Determine a model's performance score using the output data's quality.

C.    How to handle it?

1.

 Early Stopping

                                                Figure 3.3 Number of Iteration Vs Errors

392

J INFORM SYSTEMS ENG, 10(13s)

Figure 3.3 refers to pausing the training phase when the machine learning model learner reaches the point. As
each new   iteration makes the model learn more information. Thus, as the model continues to do so, its ability
to generalize deteriorates.

However, it is important to get the halting time correct else the model will still persist on providing inaccurate
results.

This technique is mostly used in Deep Learning Problems.

Using Iris Dataset, in the logistic regression model training process, training  was stopped prematurely at the
6th iteration to prevent overfitting. At this point, the model exhibited a high training accuracy of 97.50% and a
perfect validation accuracy of 100.00%. This early stopping strategy helps prevent the model from becoming
too specialized to the training data, ensuring better generalization to unseen data.

2.

Train with more Data

This approach asks for additional training data to be used to train the model. However, it won't always work.
Adding new data is useless if there is already enough noisy data present. consequently, it's important to make
sure the information is accurate and relevant.

Using Breast Cancer Dataset and Logistic Regression Algorithm following can be concluded -

Table 3.1 Performance of Models

Model
No. of Parameters
No. of Data Points
Accuracy
Precision
Recall
F1 Score

Less Data  More Data

30
156
93.56
93.75
03/26
93.50

30
455
96.48
97.01
95.77
96.39

1)

Regularization

This technique prevent overfitting by providing more information to it [21]. It works by lowering the magnitude
of  the  variables,  this  strategy  may  be  used  to  keep  all  variables  or  features  in  the  model.  Consequently,  it
maintains the model�s generality and accuracy.

In simple words, it maintains the same number of features by reducing the magnitude of the features. Thus, it
reduced the coefficient of features towards zero.

Ridge Regression: In this penalty term which is equal to the square of the magnitude of the coefficient

a)
is added. It is also known as L2 Regularization. The coefficient ? is added to the coefficient for controlling the
penalty term.

-

-

If ? > 0, then it will confine the coefficient by adding a constraint

If ? = 0, then basic OLS equation is used

So, as the value of ? increases the coefficient tends towards zero. This results in low variance and high bias.

------------                     (1)

Limitation: It reduces the complexity of the model but the number of variables remains the same. The reason
this happens is because it only minimizes the coefficient rather than making it zero. Hence, it cannot be used
for feature reduction.

b) Lasso Regression: The cost function has a penalty component called the absolute sum of the coefficients
added by the Least Absolute Shrinkage and Selection Operator (LASSO). The term is penalised when the
coefficient's value rises from zero. This has the effect of lowering the coefficient's value in an effort

to cut loss. Lasso Regression tends to make the coefficients to absolute zero.

Limitation: When there are more predictors (p) than observations (n), the algorithm will choose n predictors
as  non-zero,  even  if they  are  significant.  When there are  several  collinear  variables,  lasso  choose  at  random
which affects the interpretation of the data.

       �----------------                            ( 2)

a)

Elastic Net

If a lasso is dependent on a variable, the prediction may be skewed. Elastic Net is then used since it does not get

393

J INFORM SYSTEMS ENG, 10(13s)

rid of the high collinearity coefficient.

The example is taken from [2]. Below table represents the performance before and after applying L2 and L1
regularization techniques to the Boston Housing dataset.

                      �---------------------        ( 3)

Table 3.2 Before and After Regularization

Before
Regularization

Trainin
g

Testin
g

After Regularization

Ridge
Testin
g

Ridge
Trainin
g

Lasso
Trainin
g

Lasso
Testing

Elasti
c Net
Testin
g

Elastic
Net
Trainin
g

0.95

0.61

0.90

0.76

1.0

0.0

0.94

0.96

0.29
(? = 1.0)

0.29
(? = 1.0)

0.90
(? =
0.01)

0.94

0.84

0.70

0.77

(? =
0.01)

0.94

0.94

0.94

Dataset

Algorith
m

Boston
Housin
g

Linear
Regressio
n

Home
Prices

Random
Forest

a)

Elastic Net

If a lasso is dependent on a variable, the prediction may be skewed. Elastic Net is then used since it does not get
rid of the high collinearity coefficient.

                      �--------------------------  ( 4)

The example is taken from [2]. Below table represents the performance before and after applying L2 and L1
regularization techniques to the Boston Housing dataset.

In Boston Housing, the accuracy of training differs from testing revealing the model suffers from overfitting. So,
to remove the problem of overfitting regularization is applied to reduce the complexity of the model. In case of
Ridge Regression (L2 Regularization) although the difference is not significantly reduced but it the increase in
accuracy of testing shows the complexity is reduced. Resulting Less overfit but more general model. When the
same model is applied for Lasso Regression (L1 Regularization) with ? = 1.0 the performance is very low this is
due to coefficients of most of the features have become exactly zero. In this case, there were 4 features were
selected by the model rest 104 were ignored. If Lasso Regression applied with the optimum value of ? = 0.01 the
accuracy is improved as shown in table and it uses 32 number of features. While with ? = 0.01 and L1 Ratio =
0.01 the accuracy remains stable and it becomes the best choice. As Lasso removes strongly correlated features
drastically affecting the accuracy of the model.

Feature Selection

Numerous  variables  in  the  real-world  dataset  may  not  be  essential.  These  characteristics  might  cause  the
model's accuracy to decline while also making it more complicated. As a consequence, the model's capacity to
generalize is reduced, and the model's predictions become skewed.

Finding the ideal features or the optimum collection of characteristics for a machine learning model is assisted
by feature selection. There are three types �

Filter Methods

The method filters the dataset and contains only relevant features.

 Common Techniques:

  Correlation

Variance Threshold

Chi-Square

Anova Test

394

J INFORM SYSTEMS ENG, 10(13s)

Information Gain

Wrapper Methods: The method uses features and then evaluate the performance of the model. On the basis
of performance, the features are added or removed with the motive to increase the accuracy of the model. This
is more accurate compared to filter methods.

Common Techniques:

i.
A new variable is not regarded a stopping criterion unless it does not enhance the model's performance.

Forward Selection: The set is initially empty, and features are continuously added to improve accuracy.

Backward  Elimination:  At  first,  it  has  all  the  features.  From  each  iteration,  a  feature  with  the  least
ii.
importance is deleted. After eliminating a feature, performance is regarded to have stopped improving till that
pont.

Bi-directional  Elimination:

Table  3.3 Forward and Backward Elimination

Method

Best Features  Score

Forward Selection

Backward Elimination

Bidirectional

[3]

[3, 2]

[1, 2, 3]

[2, 3]

[3]

[2, 3]

[3]

0.960

0.967

0.973

0.967

0.960

0.967

0.960

To acquire the finest combination of  features, it concurrently employs the Forward and Backward approach.
The implementation utilizes logistic regression as the underlying model to evaluate the performance of feature
subsets. The forward selection method iteratively adds features to the subset, aiming to maximize the model's
accuracy.  Conversely,  backward  elimination  starts  with  all  features  and  progressively  removes  the  least
significant ones. Bidirectional selection combines both forward and backward approaches simultaneously.

Performance-wise,  all  three  methods  achieve  relatively  high  accuracy  scores  on  the  Iris  dataset.  Backward
elimination outperforms the other methods slightly, reaching a maximum accuracy of 97.3%. Forward selection
and bidirectional selection also perform well, achieving accuracies of 96.7% and 96.7%, respectively. However,
it's worth noting that the difference in performance between the methods is marginal, indicating that the choice
of feature selection method may not significantly impact model performance in this particular dataset.

https://github.com/bamtak/machine-learning-implemetation-
python/blob/master/Wrapper%20Method%20For%20Feature%20Selection%20-
%20Forward%20and%20Backward%20.ipynb.

a)

Embedded Methods

The method checks the importance of each feature with each iteration of machine learning model training.

Common Methods:

i.

Regularization

Tree-based Methods: Gradient Boosting and Random Forest are employed in this case to assess the
ii.
significance of the features. Which collection of characteristics are crucial in having an influence on the target
feature is represented by their significance.

395

J INFORM SYSTEMS ENG, 10(13s)

Table 3.5 Random Forest Train-Test

Method
Random Forest

Features Selected  Train ROC-AUC Score  Test ROC-AUC Score

14

0.818

0.806

Results are generated from feature selection using tree-based methods, particularly Random Forest. It utilizes
the  SelectFromModel  class  from  scikit-learn  to  select  the  most  important  features  based  on  the  feature
importances calculated by the Random Forest classifier. The selected features are then used to train a Random
Forest model, and the performance is evaluated using ROC-AUC scores on both the training and test datasets.
In this specific implementation, 14 features are selected, resulting in a train ROC-AUC score of 0.818 and a test
ROC-AUC  score  of  0.806.  Overall,  the  code  demonstrates  how  tree-based  feature  selection  methods  can
effectively improve model performance by selecting the most informative features.

https://github.com/codingnest/FeatureSelection/blob/master/Data%20Science%20Lifecycle%20-
%20Feature%20Selection%20(Filter%2C%20Wrapper%2C%20Embedded%20and%20Hybrid%20Methods).i
pynb

1)

Cross-Validation

Cross-validation is a method for determining how well a statistical model generalises to new data. As a result, it
tests the model on an unknown collection of data while validating its performance on an input set of data.

A collection of data is kept aside for cross-validation that is not utilised for training as the validation set. The
procedure is finished by testing the model on the set of data that was put aside.

Steps to follow:

i.

ii.

Reserve Validation Set

Train the model on training dataset

iii.
good results perform further steps.

Test on Validate Set and evaluate performance. On the basis of it either check for issues or in case of

Advantages: Reduces Overfitting, Hyperparameter Tuning.

Disadvantages:  Increases  Training  Times,  Needs  Expensive  Computation,  gives  drastic  results  for
inconsistent data.

Common Methods:

a.
Validation Set Approach: This method divides the dataset into two subsets, training and validation, each
of which contains 50% of the data. This method's drawback is that it may fail to record crucial data and provide
an underfitted model.

b.
Leave-P-out Cross-Validation: The p sets of data are used in this method for validation. In other words,
if there are n data points, the validation set is p, and the training set is n-p. This procedure is repeated for each
sample, and the model's efficacy is then determined using the average error recorded.

c.
Leave-One-out  Cross-Validation:  Similar  to  the  previous  approach,  it  only  uses  one  data  point  for
validation and uses every other data point for training. Each data point receives a similar treatment. Due to the
fact  that  every  training  point  was  utilized,  this  technique  exhibits  little  bias.  Although  as  n  data  points  are
executed, execution time increases. The model is only validated using one data point; hence the procedure is
regarded as costly.

d.
size. These are known as folds.

K-Fold Cross-Validation: In this approach the dataset is divided into K-groups of samples having equal

Steps to follow:

i.

ii.

�

�

�

Splitting input dataset into K-groups

For each group:

Reserve one-fold as validation set

Use remaining for training

Fit the model on training set and evaluate the performance on validation set

396

J INFORM SYSTEMS ENG, 10(13s)

e.

Stratified K-Fold Cross-Validation

Figure 3.4 Feature Selection

Similar to the previous model, this one incorporates the idea of stratification, which entails rearranging the data
such that each fold is fairly representative of the whole input dataset. As a result, stratified sampling is used in
lieu of random sample. This produces a model that is accurate and less biased. For instance, if the dataset has
two target classes, this technique makes sure that each fold contains an

Table 3.6 Different Cross Validation Techniques

N
o.

1

2

3

4

5

6

Cross
Validation
Technique

Accu
-racy

Validation Set
approach
Leave P out
cross validation
Leave one out
cross validation
K-fold cross
validation
Stratified K-
fold cross
validation
Repeated
random Train
test Splits

90.64
%
92.20
%
92.44
%
93.13
%
92.08
%

92.16
%

Vari
a
-nce

Comput-
ational
Complexit
y

0.32

Low

0.28

High

0.25

High

0.21

Medium

0.24  Medium

Robust
-ness

Bia
s

Low

High

Medium  Lo
w
Lo
w
Lo
w
Lo
w
Lo
w

High

High

Ease of
Imple
me-
ntation

High

Data
Distributi
on
Sensitivit
y
High

Medium

Low

Medium  Medium

Scala-
bility

Medium

Low

Low

Medium

Low

Medium

Medium

High

Medium

0.27

Low

Medium  Lo
w

High

Low

High

The above-mentioned analysis was performed on a cancer dataset where we applied various cross validation
techniques to solve overfitting problems. First approach was the validation set approach where random train
test split is performed and the accuracy came about 90% so basically this method provides different accuracies
hence the next methods were Leave one out cross validation and Leave P out cross validation which further
provides us with the better idea of our model since these methods ate computationally very costly. K-fold cross
validation and Stratified methods came to picture so that the computation cost becomes very less and it the best
metric for judging the accuracy of our model and it gave 93.13% and 92.08% accuracies respectively.

1)

Ensemble Methods

Machine learning techniques called ensembles are used to combine predictions from many independent models.
There are many possible ways to assemble, however the two most popular ones are as follows:

a)

Bagging

It aims to lower the likelihood of complicated models being overfit.

?

?

It simultaneously trains a huge number of "strong" students.

A model that is mostly unstructured is a powerful learner.

397

?

b)

J INFORM SYSTEMS ENG, 10(13s)

The predictions of all the powerful learners are then "smoothed out" via bagging.

Boosting

Initiatives to increase the predictability of basic models.

?

?

?

?

It sequentially trains a lot of "weak" learners.

A limited model is a weak learner (i.e., you could limit the max depth of each decision tree).

Each successive one focuses on taking lessons from the previous one's errors.

Then, by boosting, each weak learner is combined with a single strong learner.

Although boosting and bagging are both ensemble approaches, they take a different tack on the issue. In contrast
to boosting, which employs simple base models and seeks to "increase" aggregate complexity, bagging makes
use of complicated base models and attempts to "smooth out" its predictions.

Example:

Bagging solved the problem of over-fitting by combining various strong learners and hence taking the average
accuracy instead of fully believing on one model

The demonstration was performed on a iris dataset where we applied a Bagging technique Random Forest which
increased our accuracy to 97% from normal decision tree approach

Although boosting and bagging are both ensemble approaches, they take a different tack on the issue. In contrast
to boosting, which employs simple base models and seeks to "increase" aggregate complexity, bagging makes
use of complicated base models and attempts to "smooth out" its predictions.

On performing a demonstration on IRIS dataset it gave a accuracy of around 92% on the validation set which
is quite good with no hyper parameter tuning and feature engineering.

2)

Hyperparameter Tuning

A  machine  learning  model  is  a  mathematical  representation that has  a  number  of  parameters that  must  be
learned from the data. We train the model using historical or existing data to fit the model's parameters. Other
parameters, known as hyperparameters, exist in addition to these and cannot be directly learnt using routine
training procedures. In most cases, they are established prior to the commencement of the training itself. These
parameters describe crucial model characteristics like complexity and learning rate.

The two best strategies for hyperparameter tuning are as follows:

a)

GridSearchCV

The  GridSearchCV  Approach  evaluates  the  machine  learning  model  for  a  variety  of  hyperparameter
values. The approach is called the Gridsearchcv technique because it uses a grid of values to find the best set of
hyperparameters.

Take the logistic regression classifier model as an example. Let's imagine we wish to modify the c and alpha
parameter values. By building several copies of the model with every possible combination of parameters, the
grid  search approach will look for and return the optimum set of parameters.

Cons: In order to determine the optimal collection of parameters, iterating over all potential sets of parameters
results in a very large processing cost

To find out how GridSearchcv technique work and how hyperparameter tuning helps us to find the best set of
parameters for which the accuracy is highest we performed a demonstration on a Advertising dataset

The dataset consisted of two independent features such as Age and Estimated Salary and dependent/ target
attribute named �Purchased�

The SVM (Support vector machine) classifier was used to predict the output which have us 85% with defaults
parameters of SVM which is given below:

?

GridSearchCV Implementation:

Employed GridSearchCV, a powerful technique for hyperparameter tuning, to systematically search through a
range of hyperparameters and identify the optimal combination.

The following hyperparameters were considered:

C: Regularization parameter controlling the trade-off between overfitting and underfitting.

Gamma: Kernel coefficient for �rbf� and �poly� kernels.

Kernel: Type of kernel function (�linear�, �rbf�, or �poly�).

The search space for each hyperparameter was defined as follows:

398

J INFORM SYSTEMS ENG, 10(13s)

C: [1, 10, 100, 1000]

Gamma: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

Kernel: [�linear�] (for linear kernel) and [�rbf�] (for radial basis function kernel)

The model was trained using 10-fold cross-validation.

?

Best Hyperparameters:

After exhaustive search, the optimal hyperparameters were determined:

C: 1

Gamma: 0.07

Kernel: �rbf�

?

Model Evaluation:

The SVC classifier was retrained using the best hyperparameters.

On  testing  the  model  with  these  hyperparameters,  the  accuracy  improved  by  5%,  achieving  an  impressive
accuracy of 90%.

a)

RandomizedSearchCv

It  is  similar  to  the  GridSearchCv  approach  only  difference  is  that  it  only  iterates  through  the  fixed  set  of
hyperparameters values. Hence this method reduces the computation by the large scale

The demonstration was performed on the same dataset used in GridSearchCv. We used RandomForest classifier
for this demonstration where we achieved about 92% accuracy with default parameters. The default parameters
are as follows:

Random Forest Classifier is a machine learning model used for classification tasks (like predicting whether an
email is spam or not). It�s like a group of decision trees working together. Created a RandomForestClassifier
with 2 trees (called �estimators�) and use the �entropy� criterion to split data. Trained this model using our
training data (X_train and y_train).

Parameter Tuning with RandomizedSearchCV:

To find the best set of parameters for our model. Defined a range of possible values for each parameter (like
max_depth, n_estimators, etc.). RandomizedSearchCV tries different combinations of these parameters and
picks the best one. It�s like trying different ingredients to make the tastiest cake.

Best Features Found: After searching, we found the best parameters:

max_depth: 3 (how deep the trees can go)

n_estimators: 300 (number of trees)

max_features: 1 (number of features to consider at each split)

And others�

These settings give us good accuracy (around 91.2%). Applying the Best Features:

Created a new RandomForestClassifier with these best parameters. Trained it on data again. The accuracy of
this improved model is around 91.2%.

IV. CONCEPT OF UNDERFITTING

The final phase in machine learning is to evaluate the model's performance. Accuracy and generalization are the
two key factors to consider when assessing model performance [25]. Both accuracy and generalization refer to
a model's capacity to predict correct values and respond to previously unexplored and novel data, respectively.

In general, machine learning models are taught using training data, and performance evaluations are done using
test data that hasn't been seen. When a model acts consistently across training and test sets of data with High
accuracy that model is considered as Good Model. When a model performs poorly on both training and test
data, it is said to be underfitted.

Underfitting happens when the method used to build the prediction model is relatively simple and unable to
uncover  complex  patterns  from  the training  data [26].  In  that  case,  accuracy will be below  average  for both
visible training data and unseen test data. Another name for underfitting is high bias.

Machine  learning  models  perform  poorly  due  to  both  overfitting  and  underfitting  [27].  Understanding  the
underfitting issue requires an understanding of bias and variance. In essence, bias is the mistake in the training
set of data. Little bias indicates low training error, whereas significant bias indicates substantial training error.
Contrarily, variation is the distinction between mistake in training and error in testing.

399

J INFORM SYSTEMS ENG, 10(13s)

Underfitting  is  a  serious  problem  in  machine  learning  since  it  lowers the precision of our  machine  learning
models.  Overfitting  usually  occurs  when  we  have  a  lot  less  data  to  train  our  machine  learning  model.  In
summary, we may say that underfitting refers to models that do not perform well on both training and testing
data [28].

B.

Reasons

?

?

?

?

Training data is not cleaned and contains noise and outliers in it

The size of training used is not enough to identify all the patterns

The model is very simple

The biasness of model is high.

Identify

Identifying underfitting is a crucial step in the model development process, as it helps us understand when a
model  is  too  simplistic  and  unable  to  capture  the  underlying  patterns  in  the  data  effectively.  One  common
technique to detect underfitting is by using the bias-variance trade-off concept. Expanding on the explanation
you provided:

Bias-Variance Trade-off Concept:

The bias-variance trade-off is a fundamental concept in machine learning that helps us understand how different
aspects of a model affect its performance. It can be described as follows:

Bias: When your model has a high bias, it means that it's overly simplistic and makes strong assumptions about
the data. It often fails to capture complex patterns and tends to underfit the data. In other words, the model is
not flexible enough to represent the underlying relationships in the data.

Variance: On the other hand, when your model has high variance, it is overly complex and highly sensitive to
the noise in the training data. Such a model can fit the training data extremely well but may not generalize well
to new, unseen data. This is known as overfitting.

Identifying Underfitting Using the Bias-Variance Trade-off:

When assessing a machine learning model, it's crucial to strike a balance between bias and variance. Recognizing
underfitting, which occurs when a model is excessively simplistic, can be done through the following elaborated
steps:

1.

Bias Analysis:

Start by evaluating the bias of your model. Bias indicates how well your model fits the training data. A high bias
implies that your model makes strong, often incorrect, assumptions about the data. Signs of high bias include
poor training performance, low accuracy, and consistently large errors on the training data. Example, generated
synthetic data from a sine curve with added noise, splits it into training and test sets, and then fits polynomial
regression  models  of  varying  degrees  to  the  training  data.  Using  the  bias_variance_decomp  function  from
mlxtend, it decomposes the mean squared error (MSE) into bias, variance, and irreducible error components.
Finally, plots of bias, variance, and error as a function of polynomial degree, figure 4.1 demonstrating the bias-
variance trade-off.

2.

Visual Inspection:

Figure 4.1 Bias Variance Tradeoff

Plot your model's predictions against the actual data points on a graph. If the predictions consistently deviate
from the actual values, forming a pattern that is far from the data's true behavior, it's a clear sign of underfitting.
In simple terms, your model is too simplistic to capture the data's complexities.

400

J INFORM SYSTEMS ENG, 10(13s)

In  this  example,  load the  Digits  dataset  from  scikit-learn,  split  it  into  training  and  test  sets,  create  an  SVM
classifier, and calling function with the estimator, training data, and other optional parameters. The learning
curve will be displayed showing the training and cross-validation scores as the number of training examples
increases.

3.

Cross-Validation:

Figure 4.2 Cross Validation Score

Employ techniques  like cross-validation to assess your model's performance on multiple subsets of the data.
High bias will result in similar poor performance across different subsets, confirming the underfitting issue.

4.

Model Complexity:

Examine your model's complexity. If it has too few parameters or features to represent the underlying patterns
in the data, it's likely underfitting. Consider increasing the model's complexity by adding more features, layers,
or adjusting hyperparameters.

Table 4.1 Model Complexity

Aspect

Number of Parameters
Depth or Width
Regularization

Cross-Validation Curves

Occam's Razor
Information Criteria

Indicators
Total parameters: Weight count, Bias count
Depth: Number of layers, Width: Number of nodes per layer
L1 regularization strength (lambda), L2 regularization strength (lambda)
Performance change with depth (decision trees), Performance change
with layer size (neural networks)
Simplicity vs. Performance trade-off in model selection
AIC value, BIC value

5.

Validation Metrics:

Table 4.2 Metrics and Equations

Metric

Formula

Accuracy
Precision
Recall (Sensitivity)
Specificity

F1 Score

ROC-AUC Score

PR-AUC SCORE
Mean Absolute Error
(MAE)
Mean Squared Error
(MSE)
Root Mean Squared
Error (RMSE)

(TP + TN) / (TP + TN + FP + FN)
TP / (TP + FP)
TP / (TP + FN)
TN / (TN + FP)
2 * (Precision * Recall) / (Precision +
Recall)
Area under the Receiver Operating
Characteristic curve
Area under the Precision-Recall curve

?(MSE)

401

J INFORM SYSTEMS ENG, 10(13s)

Mean Absolute
Percentage Error
R-squared (Coefficient
of Determination)
Adjusted R-squared
Mean Squared
Logarithmic Error
Cohen's Kappa

1 - (SSres / SStot)

1 - [(1 - R^2) * (n - 1) / (n - k - 1)]

(Po - Pe) / (1 - Pe)

Utilize evaluation metrics like mean squared error, mean absolute error, or classification accuracy, depending
on your problem type. High error rates and low accuracy are indicators of underfitting.

6.

Feature Engineering:

Review the features used for training. If you have not included relevant features or have oversimplified feature
engineering, your model may underfit. Experiment with different feature sets.

How to handle it?

Increasing Model Complexity

One approach to combat underfitting is to augment the model's complexity. This involves incorporating more
intricate algorithms or enhancing the model's capacity through feature expansion and selection. Linear models
can benefit from additional features, while more complex algorithms, such as deep neural networks or support
vector machines with varied kernels, offer enhanced modeling capabilities.

In the context of deep learning, it encompasses several aspects:

?

Expressive Capacity:

This aspect relates to the ability of a model to represent complex relationships within the data. Deep learning
models,  such  as  neural  networks,  have  high  expressive  capacity  due  to  their  layered  architecture  and  large
number of parameters. Expressive capacity determines how well a model can fit both training and unseen data.

?

Effective Model Complexity:

Effective model complexity considers not only the model�s expressive power but also its ability to generalize
well. It balances the trade-off between fitting the training data perfectly (overfitting) and being too simplistic
(underfitting).

Importance:

?
complex models may memorize noise in the training data, leading to poor generalization.

Generalization:  A  model�s  complexity  affects  its  ability  to  generalize  to  new,  unseen  data.  Overly

?
(e.g., dropout, weight decay) control complexity to prevent overfitting.

Optimization:  Understanding  complexity  helps  optimize  model  training.  Regularization  techniques

?
if they achieve similar performance.

Model Selection: Researchers choose models based on their complexity. Simpler models are preferred

Example:

The experiment evaluates polynomial regression on synthetic data, showcasing its performance improvement
with  modeling  complexity.  Before  fitting,  mean  squared  error  (MSE)  was  0.62.  After  fitting  a  polynomial
regression model of degree 3, MSE reduced to 0.54, indicating better model fit. The visualization highlights the
model's ability to capture underlying patterns in the data, affirming its effectiveness in capturing non-linear
relationships. Figure 4.3 depicts the nonlinear relationship between Data and Polynomial regression.

Figure 4.3 Non Linear Relationship

402

J INFORM SYSTEMS ENG, 10(13s)

Feature Engineering

Thoughtful  feature  engineering  plays  a  pivotal  role  in  rectifying  underfitting.  Tailoring  features  to  extract
pertinent information from the dataset empowers the model to discern nuanced relationships within the data,
thereby elevating its performance [20].

Feature engineering involves creating new features or modifying existing ones to improve the quality of input
data for machine learning models. It�s a critical step in the model development process.

?
provide meaningful information to the model.

Data  Representation:  Features  are  the  building  blocks  of  data  representation.  Well-crafted  features

?
model�s ability to generalize.

Model Learning: Models learn from features. The quality and relevance of features directly affect the

Key Aspects:

Feature Extraction

i.

From Timestamps:

Extracting meaningful features from timestamps is common in time-series data.

Examples: Day of the Week: Convert a timestamp to the corresponding day of the week (e.g., Monday, Tuesday).
Month: Extract the month (e.g., January, February) from a timestamp. Hour of the Day: Obtain the hour (0 to
23) from a timestamp.

ii.

From Text Data:

Text data can be transformed into numerical features for modeling.

Examples:  Bag-of-Words  Representation:  Convert text  documents  into  vectors by  counting  the  frequency of
each word. Word Embeddings: Use pre-trained word embeddings (e.g., Word2Vec, GloVe) to represent words
as dense vectors.

Feature Transformation

i.

Logarithmic Transformation: Apply a logarithmic function to features with skewed distributions.

Benefits: Reduces the impact of extreme values. Makes the data more symmetric.

ii.

Scaling Features: Ensure that features have similar scales to prevent bias in certain algorithms.

Common scaling methods: Min-Max Scaling: Scales features to a specified range (e.g., [0, 1]).

  Z-Score Normalization: Standardizes features to have zero mean and unit variance.

Feature Creation

i.

Interaction Terms: Combine existing features to create new ones.

Example: If you have features for height and weight, create an interaction term by multiplying them to capture
body mass index (BMI).

ii.

Aggregating Information: Aggregate data across categories or time periods.

Examples: Calculate average transaction amount per customer. Summarize daily sales by month.

Hyperparameter Tuning

Effective hyperparameter tuning is vital for achieving the right balance between bias and variance. Parameters
such as learning rates, regularization strengths, and network depths (for neural networks) must be meticulously
adjusted to optimize model performance.

Regularization

Regularization techniques, such as L1 or L2 regularization for linear models and dropout for neural networks,
can  thwart  overfitting  and  render the model  more  adaptable  to  the  data.  By  imposing  penalties  on  extreme
parameter  values,  regularization  encourages  a  healthier  bias-variance  trade-off.  It  helps  in  controlling  the
complexity of the model and encourages simpler models that generalize well to unseen data. In the context of
underfitting, regularization can be particularly helpful in improving model performance by adding flexibility
without overly complexifying the model.

Increasing Training Data

In cases where the dataset is limited, gathering more data can be instrumental in mitigating underfitting. A
larger dataset facilitates a more comprehensive understanding of the underlying data patterns.

Examples:

The algorithm employed here is a simple linear regression model. It aims to learn a linear relationship between

403

J INFORM SYSTEMS ENG, 10(13s)

an input variable (X) and an output variable (y). In this case, the true relationship is given by y = 2x + noise,
where noise represents random variability.

The technique being explored is the effect of training data size on model performance. Specifically, comparing
the model�s behavior when trained on a small dataset versus a large dataset.

The  blue  dashed  line  corresponds  to  predictions  made  by  the  model  trained  on  a  small  dataset.  The  Mean
Squared Error (MSE) associated with this model is 1.13. The red dashed line represents predictions from the
model trained on a larger dataset. The MSE for this model is significantly reduced to 0.86. Figure 4.4 depicts
the Effect of increasing training Data on Model Performance and MSE.

Figure 4.4 Effect of increasing training Data on Model Performance

Techniques:

Data Augmentation: This technique involves creating additional training examples by applying transformations
to existing data. For example, in image classification tasks, you can rotate, flip, zoom, or crop images to generate
variations [16].

Ensemble Methods: Combining multiple models can help mitigate underfitting by leveraging the diversity of
individual  models.  Techniques  such  as  bagging,  boosting,  and  stacking  aggregate  predictions  from  multiple
models to improve overall performance.

Transfer Learning: Utilizing pre-trained models on larger datasets as a starting point and fine-tuning them on
your  specific  dataset  can  effectively  leverage  the  knowledge  gained  from  the  larger  dataset  to  improve
performance on the smaller one.

Synthetic Data Generation: Generating synthetic data points that resemble real data can effectively increase the
size  of  the training  dataset.  Techniques  like  Generative  Adversarial  Networks  (GANs)  or Synthetic  Minority
Over-sampling Technique (SMOTE) can be employed, especially for imbalanced datasets [28].

Active  Learning:  Iteratively  selecting  the  most  informative  samples  for  labeling  can  maximize  the  learning
efficiency of  the  model.  This  involves  selecting  samples  that  the  model  is uncertain  about,  thus  guiding  the
training towards regions where it lacks knowledge [20].

Cross-VALIDATION

The  utilization  of  cross-validation  aids  in  fine-tuning  the  model  while  ensuring  its  ability  to  generalize
effectively. This approach helps in early identification of underfitting during the model development process.

VI. CONCLUSION

This research has comprehensively explored the challenges of overfitting and underfitting in machine learning,
demonstrating  their  significant  impact  on  model  performance  and  generalization  ability.  As  highlighted,
overfitting occurs when models become overly fixated on training data specifics, including noise and irrelevant
patterns, leading to poor performance on unseen data. Conversely, underfitting arises from models failing to
capture the underlying trends within the training data, resulting in suboptimal performance across both training
and testing sets.

The key takeaway from  this  analysis  is the critical  need  to  achieve  a  well-balanced  trade-off between  model
complexity  and the  available  training  data  volume.  To  this  end,  various techniques have been  explored  and
evaluated  throughout  this  paper.  These  techniques  include,  but  are  not  limited  to,  regularization  (L1/L2
regularization,  dropout),  cross-validation  (k-fold,  stratified  k-fold),  and  feature  selection  (filter  methods,
wrapper methods, embedded methods). Implementation examples and case studies were presented to illustrate
the practical application of these methods in mitigating overfitting and underfitting.

Furthermore, the importance of leveraging large and diverse datasets for training machine learning models was
emphasized.  This  approach,  coupled  with  appropriate  validation  strategies,  strengthens  the  robustness  and
generalizability of the models. By incorporating findings from the literature review that identified the reasons

404

J INFORM SYSTEMS ENG, 10(13s)

for  overfitting  and  underfitting,  this  research has  provided  a  holistic  understanding  of how  to  handle  these
challenges.

In conclusion, this research not only sheds light on the causes and effects of overfitting and underfitting, but
also  offers  a  comprehensive  set  of  techniques  for  mitigating  these  issues.  By  adopting  these  strategies,
researchers  and  practitioners  can  develop  more  reliable  and  effective  machine  learning  models,  ultimately
driving innovation and delivering valuable insights across a wide range of applications.

REFERENCES

[1]  Singogo,  C.  (2024,  February  25).  Understanding  Overfitting  and  Underfitting  in  Machine  Learning.
Retrieved  from  https://medium.com/@singogosingogo/understanding-overfitting-and-underfitting-in-
machine-learning-3d822b739bcf

[2] Deepgram. (2023, September 29). Overfitting and Underfitting. Retrieved from https://deepgram.com/ai-

glossary/overfitting-underfitting

[3]  Ali  Awan,  A.

(2023,  August).  What

is  Overfitting?  DataCamp.  Retrieved

from

https://www.datacamp.com/blog/what-is-overfitting

[4]  Fawcett,  Tom.

Introduction

to  ROC  analysis.  Pattern  Recognition  Letters.  27.  861-874.

10.1016/j.patrec.2005.10.010.

[5]  Caruana,  Rich,  and  Alexandru  Niculescu-Mizil.  "An  empirical  comparison  of  supervised  learning

algorithms." arXiv preprint cs/0604060 (2006).

[6] James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani. An introduction to statistical learning.

Vol. 112. Springer, 2013.

[7] Brownlee, Jason. "Overfitting vs Underfitting: What They Are and How to Avoid Them." Machine Learning
Mastery  (2020).  https://www.linkedin.com/pulse/mastering-overfitting-underfitting-machine-learning-
models-ravi-singh

[8] Ying, Xue. (2019). An Overview of Overfitting and its Solutions. Journal of Physics: Conference Series. 1168.

022022. 10.1088/1742-6596/1168/2/022022.

[9] L�pez, O. A. M., L�pez, A. M., & Crossa, J. (2022, January 1). Overfitting, Model Tuning, and Evaluation of

Prediction Performance. Springer eBooks. https://doi.org/10.1007/978-3-030-89010-0_4

[10] Xu, C., Coen-Pirani, P., & Jiang, X. (2023). Empirical Study of Overfitting in Deep Learning for Predicting

Breast Cancer Metastasis. Cancers, 15(7). https://doi.org/10.3390/cancers15071969

[11] Abdul Salam, M., Azar, A. T., Elgendy, M. S., & Fouad, K. M. (2021). The Effect of Different Dimensionality
Reduction  Techniques  on  Machine  Learning  Overfitting.  International  Journal  of  Advanced  Computer
Science and Applications, 12(4).

[12] J. Kolluri, V. K. Kotte, M. S. B. Phridviraj and S. Razia, Reducing Overfitting Problem in Machine Learning
Using Novel L1/4 Regularization Method, 2020 4th International Conference on Trends in Electronics and
Informatics
doi:
10.1109/ICOEI48184.2020.9142992.

(ICOEI)(48184),

Tirunelveli,

934-938,

2020,

India,

pp.

[13] Gu Y, Wylie BK, Boyte SP, Picotte J, Howard DM, Smith K, Nelson KJ. An Optimal Sample Data Usage
Strategy to Minimize Overfitting and Underfitting Effects in Regression Tree Models Based on Remotely-
Sensed Data. Remote Sensing. 2016; 8(11):943. https://doi.org/10.3390/rs8110943

[14] Liu, Z., Xu, Z., Jin, J., Shen, Z., & Darrell, T. Dropout Reduces Underfitting. In  Proceedings of the 40th
liu23aq).  PMLR.

(Vol.  202,  pp.

(ICML)

International  Conference  on  Machine  Learning
https://proceedings.mlr.press/v202/liu23aq/liu23aq.pdf

[15] Bu, C., & Zhang, Z. (2020). Research on Overfitting Problem and Correction in Machine Learning. Journal
of Physics: Conference Series, 1693(1), 012100. https://dx.doi.org/10.1088/1742-6596/1693/1/012100

[16] Maharana, K., Mondal, S., & Nemade, B. (2022). A review: Data pre-processing and data augmentation
techniques. Global Transitions Proceedings, 3(1), 91-99. https://doi.org/10.1016/j.gltp.2022.04.020

[17]  C.  Khosla  and  B.  S.  Saini,  "Enhancing  Performance  of  Deep  Learning  Models  with  different  Data
Augmentation  Techniques:  A  Survey,"  2020  International  Conference  on  Intelligent  Engineering  and
Management (ICIEM), London, UK, 2020, pp. 79-85, doi: 10.1109/ICwIEM48762.2020.9160048.

[18]  Cuthbert,  L.,  Jagannathan,  P.,  Rajkumar,  S.,  Frnda,  J.,  Divakarachari,  P.  B.,  &  Subramani,  P.  (2021).
Moving Vehicle Detection and Classification Using Gaussian Mixture Model and Ensemble Deep Learning
Technique.  Wireless
5590894.
https://doi.org/10.1155/2021/5590894

Communications

and  Mobile

Computing,

2021,

405

J INFORM SYSTEMS ENG, 10(13s)

[19]  Pandey,  Mayuresh,  and  Ravita  Mishra.  "Identity  Resolution  In  Social  Network  Using  Recommender

System." In e-Conference on Data Science and Intelligent Computing, p. 97. 2020.

[20] Bhushankumar Nemade, Somil Doshi, Preet Desai, Aditeya Prajapati, Kiarah Patel, and Kiran Maharana.
2023. Amphibious Trash Collector System. Rivista Italiana di Filosofia Analitica Junior 14, 2 (August 21,
2023), 1360-1371.

[21]  Ravita  Mishra,  Dr  Sheetal  Rathi  (2019)  �Efficient  and  Scalable  Job  Recommender  System  Using
Collaborative  Filtering�,  Paprzycki  M.,  Gunjan  V.  (eds)  ICDSMLA  2019.  Lecture  Notes  in  Electrical
Engineering, vol 601. Springer, Singapore https://doi.org/10.1007/978-981-15-1420-3_91.

[22] Ravita Mishra, Sheetal Rathi, Enhanced DSSM (Deep Semantic Structure Modelling) Technique for Job
Recommendation, Journal of King Saud University - Computer and Information Sciences, 2021, ISSN 1319-
1578,https://doi.org/10.1016/j.jksuci.2021.07.018.

[23]  Cloud  Computing  Concepts  Hub  Machine  Learning  &  AI."  Accessed  May  05,  2024.  URL:

https://aws.amazon.com/what-
is/overfitting/#:~:text=Overfitting%20occurs%20when%20the%20model,to%20several%20reasons%2C
%20such%20as%3A&text=The%20training%20data%20size%20is,all%20possible%20input%20data%2
0values.

[24] What is Overfitting? IBM. Accessed May 05, 2024. URL: https://www.ibm.com/topics/overfitting

[25] Wittek, P. (2014). Machine Learning. In P. Wittek (Ed.), Quantum Machine Learning (pp. 11-24). Academic

Press. ISBN 9780128009536. DOI: 10.1016/B978-0-12-800953-6.00002-5.

[26] What is Underfitting? IBM. Accessed May 05, 2024. URL: https://www.ibm.com/topics/underfitting

[27] Scribble Data. "Overfitting and Underfitting in ML: Introduction, Techniques, and Future." Scribble Data
Blog, Accessed May 5, 2024. URL: https://www.scribbledata.io/blog/overfitting-and-underfitting-in-ml-
introduction-techniques-and-future/Resources / Blogs /

[28] B. Nemade, V. Bharadi, S. S. Alegavi, and B. Marakarkandy, "A Comprehensive Review: SMOTE-Based
Oversampling Methods for Imbalanced Classification Techniques, Evaluation, and Result Comparisons,"
International Journal of Intelligent Systems and Applications in Engineering, vol. 11, no. 9s, pp. 790-803,
2023.

[29]  Nemade,  B.,  Maharana,  K.K.,  Kulkarni,  V.  et  al.  IoT-based  automated  system  for  water-related  disease

prediction. Sci Rep 14, 29483 (2024). https://doi.org/10.1038/s41598-024-79989-6

[30] https://github.com/bamtak/machine-learning-implemetation-

python/blob/master/Wrapper%20Method%20For%20Feature%20Selection%20-
%20Forward%20and%20Backward%20.ipynb
https://github.com/codingnest/FeatureSelection/blob/master/Data%20Science%20Lifecycle%20-
%20Feature%20Selection%20(Filter%2C%20Wrapper%2C%20Embedded%20and%20Hybrid%20Metho
ds).ipynb


