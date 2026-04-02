CS 230 � Deep Learning

https://stanford.edu/~shervine

VIP Cheatsheet: Convolutional Neural Networks

Max pooling

Average pooling

Purpose

Each pooling operation selects the
maximum value of the current view

Each pooling operation averages
the values of the current view

Afshine Amidi and Shervine Amidi

November 26, 2018

Illustration

Overview

Comments

- Preserves detected features
- Most commonly used

- Downsamples feature map
- Used in LeNet

(cid:114) Architecture of a traditional CNN � Convolutional neural networks, also known as CNNs,
are a speci?c type of neural networks that are generally composed of the following layers:

(cid:114) Fully Connected (FC) � The fully connected layer (FC) operates on a ?attened input where
each input is connected to all neurons. If present, FC layers are usually found towards the end
of CNN architectures and can be used to optimize objectives such as class scores.

The convolution layer and the pooling layer can be ?ne-tuned with respect to hyperparameters
that are described in the next sections.

Types of layer

(cid:114) Convolutional layer (CONV) � The convolution layer (CONV) uses ?lters that perform
convolution operations as it is scanning the input I with respect to its dimensions. Its hyperpa-
rameters include the ?lter size F and stride S. The resulting output O is called feature map or
activation map.

Filter hyperparameters

The convolution layer contains ?lters for which it is important to know the meaning behind its
hyperparameters.

(cid:114) Dimensions of a ?lter � A ?lter of size F � F applied to an input containing C channels is
a F � F � C volume that performs convolutions on an input of size I � I � C and produces an
output feature map (also called activation map) of size O � O � 1.

Remark: the convolution step can be generalized to the 1D and 3D cases as well.

(cid:114) Pooling (POOL) � The pooling layer (POOL) is a downsampling operation, typically applied
after a convolution layer, which does some spatial invariance. In particular, max and average
pooling are special kinds of pooling where the maximum and average value is taken, respectively.

Remark: the application of K ?lters of size F � F results in an output feature map of size
O � O � K.

(cid:114) Stride � For a convolutional or a pooling operation, the stride S denotes the number of pixels
by which the window moves after each operation.

Stanford University

1

Winter 2019

CS 230 � Deep Learning

https://stanford.edu/~shervine

(cid:114) Zero-padding � Zero-padding denotes the process of adding P zeroes to each side of the
boundaries of the input. This value can either be manually speci?ed or automatically set through
one of the three modes detailed below:

CONV

POOL

FC

Valid

P = 0

Pstart =

Pend =

Same
j Sd I

S e?I+F ?S

l Sd I

S e?I+F ?S

k

m

2

2

Pstart ? [[0,F ? 1]]

Pend = F ? 1

Full

Illustration

Value

Illustration

Purpose

- No padding

- Drops last
convolution if
dimensions do not
match

- Padding such that feature

map size has size

l I

m

S

- Output size is
mathematically convenient
- Also called �half� padding

- Maximum padding
such that end
convolutions are
applied on the limits
of the input
- Filter �sees� the input
end-to-end

Input size

I � I � C

Output size

O � O � K

I � I � C

O � O � C

Nin

Nout

Number of
parameters

Remarks

(F � F � C + 1) � K

0

(Nin + 1) � Nout

- One bias parameter
per ?lter
- In most cases, S < F
- A common choice
for K is 2C

- Pooling operation
done channel-wise

- In most cases, S = F

- Input is ?attened
- One bias parameter
per neuron
- The number of FC
neurons is free of
structural constraints

(cid:114) Receptive ?eld � The receptive ?eld at layer k is the area denoted Rk � Rk of the input
that each pixel of the k-th activation map can �see�. By calling Fj the ?lter size of layer j and
Si the stride value of layer i and with the convention S0 = 1, the receptive ?eld at layer k can
be computed with the formula:

Tuning hyperparameters

(cid:114) Parameter compatibility in convolution layer � By noting I the length of the input
volume size, F the length of the ?lter, P the amount of zero padding, S the stride, then the
output size O of the feature map along that dimension is given by:

Rk = 1 +

k
X

j=1

(Fj ? 1)

j?1
Y

i=0

Si

O =

I ? F + Pstart + Pend
S

+ 1

In the example below, we have F1 = F2 = 3 and S1 = S2 = 1, which gives R2 = 1+2 � 1+2 � 1 =
5.

Remark: often times, Pstart = Pend (cid:44) P , in which case we can replace Pstart + Pend by 2P in
the formula above.

Commonly used activation functions

(cid:114) Understanding the complexity of the model � In order to assess the complexity of a
model, it is often useful to determine the number of parameters that its architecture will have.
In a given layer of a convolutional neural network, it is done as follows:

(cid:114) Recti?ed Linear Unit � The recti?ed linear unit layer (ReLU) is an activation function g
that is used on all elements of the volume. It aims at introducing non-linearities to the network.
Its variants are summarized in the table below:

Stanford University

2

Winter 2019

CS 230 � Deep Learning

https://stanford.edu/~shervine

ReLU

g(z) = max(0,z)

Leaky ReLU

g(z) = max((cid:15)z,z)
with (cid:15) (cid:28) 1

ELU

Bounding box detection

Landmark detection

g(z) = max(?(ez ? 1),z)
with ? (cid:28) 1

Detects the part of the image where
the object is located

- Detects a shape or characteristics of
an object (e.g. eyes)
- More granular

Non-linearity complexities
biologically interpretable

Addresses dying ReLU
issue for negative values

Di?erentiable everywhere

(cid:114) Softmax � The softmax step can be seen as a generalized logistic function that takes as input
a vector of scores x ? Rn and outputs a vector of output probability p ? Rn through a softmax
function at the end of the architecture. It is de?ned as follows:

Box of center (bx,by), height bh
and width bw

Reference points (l1x,l1y), ...,(lnx,lny)

p =

(cid:19)

(cid:18)p1
...
pn

where

pi =

exi
n
X

exj

j=1

(cid:114) Intersection over Union � Intersection over Union, also known as IoU, is a function that
quanti?es how correctly positioned a predicted bounding box Bp is over the actual bounding
box Ba. It is de?ned as:

IoU(Bp,Ba) =

Bp ? Ba
Bp ? Ba

Object detection

(cid:114) Types of models � There are 3 main types of object recognition algorithms, for which the
nature of what is predicted is di?erent. They are described in the table below:

Image classi?cation

Classi?cation
w. localization

Detection

- Classi?es a picture

- Predicts probability
of object

- Detects object in a picture
- Predicts probability of
object and where it is
located

- Detects up to several objects
in a picture
- Predicts probabilities of objects
and where they are located

Traditional CNN

Simpli?ed YOLO, R-CNN

YOLO, R-CNN

Remark: we always have IoU ? [0,1]. By convention, a predicted bounding box Bp is considered
as being reasonably good if IoU(Bp,Ba) (cid:62) 0.5.

(cid:114) Anchor boxes � Anchor boxing is a technique used to predict overlapping bounding boxes.
In practice, the network is allowed to predict more than one box simultaneously, where each box
prediction is constrained to have a given set of geometrical properties. For instance, the ?rst
prediction can potentially be a rectangular box of a given form, while the second will be another
rectangular box of a di?erent geometrical form.

(cid:114) Non-max suppression � The non-max suppression technique aims at removing duplicate
overlapping bounding boxes of a same object by selecting the most representative ones. After
having removed all boxes having a probability prediction lower than 0.6, the following steps are
repeated while there are boxes remaining:

(cid:114) Detection � In the context of object detection, di?erent methods are used depending on
whether we just want to locate the object or detect a more complex shape in the image. The
two main ones are summed up in the table below:

� Step 1: Pick the box with the largest prediction probability.

� Step 2: Discard any box having an IoU (cid:62) 0.5 with the previous box.

Stanford University

3

Winter 2019

CS 230 � Deep Learning

https://stanford.edu/~shervine

Face veri?cation and recognition

(cid:114) Types of models � Two main types of model are summed up in table below:

Face veri?cation

Face recognition

- Is this the correct person?
- One-to-one lookup

- Is this one of the K persons in the database?
- One-to-many lookup

(cid:114) YOLO � You Only Look Once (YOLO) is an object detection algorithm that performs the
following steps:

� Step 1: Divide the input image into a G � G grid.

� Step 2: For each grid cell, run a CNN that predicts y of the following form:

y = (cid:2) pc,bx,by,bh,bw,c1,c2,...,cp
}

{z
repeated k times

|

,...(cid:3)T

? RG�G�k�(5+p)

where pc is the probability of detecting an object, bx,by,bh,bw are the properties of the
detected bouding box, c1,...,cp is a one-hot representation of which of the p classes were
detected, and k is the number of anchor boxes.

� Step 3: Run the non-max suppression algorithm to remove any potential duplicate over-

lapping bounding boxes.

(cid:114) One Shot Learning � One Shot Learning is a face veri?cation algorithm that uses a limited
training set to learn a similarity function that quanti?es how di?erent two given images are. The
similarity function applied to two images is often noted d(image 1, image 2).

(cid:114) Siamese Network � Siamese Networks aim at learning how to encode images to then quantify
how di?erent two images are. For a given input image x(i), the encoded output is often noted
as f (x(i)).

(cid:114) Triplet loss � The triplet loss � is a loss function computed on the embedding representation
of a triplet of images A (anchor), P (positive) and N (negative). The anchor and the positive
example belong to a same class, while the negative example to another one. By calling ? ? R+
the margin parameter, this loss is de?ned as follows:

�(A,P,N ) = max (d(A,P ) ? d(A,N ) + ?,0)

Remark: when pc = 0, then the network does not detect any object. In that case, the corre-
sponding predictions bx, ..., cp have to be ignored.

(cid:114) R-CNN � Region with Convolutional Neural Networks (R-CNN) is an object detection algo-
rithm that ?rst segments the image to ?nd potential relevant bounding boxes and then run the
detection algorithm to ?nd most probable objects in those bounding boxes.

Remark: although the original algorithm is computationally expensive and slow, newer archi-
tectures enabled the algorithm to run faster, such as Fast R-CNN and Faster R-CNN.

(cid:114) Motivation � The goal of neural style transfer is to generate an image G based on a given
content C and a given style S.

Neural style transfer

Stanford University

4

Winter 2019

CS 230 � Deep Learning

https://stanford.edu/~shervine

(cid:114) Activation � In a given layer l, the activation is noted a[l] and is of dimensions nH � nw � nc

(cid:114) Content cost function � The content cost function Jcontent(C,G) is used to determine how
the generated image G di?ers from the original content image C. It is de?ned as follows:

Jcontent(C,G) =

1
2

||a[l](C) ? a[l](G)||2

(cid:114) Style matrix � The style matrix G[l] of a given layer l is a Gram matrix where each of its
elements G[l]
quanti?es how correlated the channels k and k0 are. It is de?ned with respect to
kk0
activations a[l] as follows:

Remark: use cases using variants of GANs include text to image, music generation and syn-
thesis.
(cid:114) ResNet � The Residual Network architecture (also called ResNet) uses residual blocks with a
high number of layers meant to decrease the training error. The residual block has the following
characterizing equation:

a[l+2] = g(a[l] + z[l+2])

(cid:114) Inception Network � This architecture uses inception modules and aims at giving a try
at di?erent convolutions in order to increase its performance. In particular, it uses the 1 � 1
convolution trick to lower the burden of computation.

[l]

n

HX

G[l]
kk0

=

[l]

n

wX

ijka[l]
a[l]
ijk0

?

?

?

i=1

j=1

Remark: the style matrix for the style image and the generated image are noted G[l](S) and
G[l](G) respectively.

(cid:114) Style cost function � The style cost function Jstyle(S,G) is used to determine how the
generated image G di?ers from the style S. It is de?ned as follows:

J [l]
style

(S,G) =

1
(2nH nwnc)2

||G[l](S) ? G[l](G)||2
F

=

1
(2nH nwnc)2

ncX

(cid:16)

k,k0=1

kk0 ? G[l](G)
G[l](S)

kk0

(cid:17)2

(cid:114) Overall cost function � The overall cost function is de?ned as being a combination of the
content and style cost functions, weighted by parameters ?,?, as follows:

J(G) = ?Jcontent(C,G) + ?Jstyle(S,G)

Remark: a higher value of ? will make the model care more about the content while a higher
value of ? will make it care more about the style.

Architectures using computational tricks

(cid:114) Generative Adversarial Network � Generative adversarial networks, also known as GANs,
are composed of a generative and a discriminative model, where the generative model aims at
generating the most truthful output that will be fed into the discriminative which aims at
di?erentiating the generated and true image.

Stanford University

5

Winter 2019


