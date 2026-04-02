This page
intentionally left
blank

Copyright � 2006, New Age International (P) Ltd., Publishers
Published by New Age International (P) Ltd., Publishers

All rights reserved.

No part of this ebook may be reproduced in any form, by photostat, microfilm,
xerography, or any other means, or incorporated into any information retrieval
system, electronic or mechanical, without the written permission of the publisher.
All inquiries should be emailed to rights@newagepublishers.com

ISBN (13) : 978-81-224-2700-4

PUBLISHING FOR ONE WORLD

NEW AGE INTERNATIONAL (P) LIMITED, PUBLISHERS
4835/24, Ansari Road, Daryaganj, New Delhi - 110002
Visit us at www.newagepublishers.com

To

My parents

This page
intentionally left
blank

FOREWORD

It gives me great pleasure to write the foreword to Dr. Nazrul Islam�s book entitled �Tensors and Their
Applications. I know the author as a research scholar who has worked with me for several years. This
book is a humble step of efforts made by him to prove him to be a dedicated and striving teacher who
has worked relentlessly in this field.

This book fills the gap as methodology has been explained in a simple manner to enable students

to understand easily. This book will prove to be a complete book for the students in this field.

Ram Nivas
Professor,
Department of Mathematics and Astronomy,
Lucknow University,
Lucknow

This page
intentionally left
blank

PREFACE

�Tensors� were introduced by Professor Gregorio Ricci of University of Padua (Italy) in 1887
primarily as extension of vectors. A quantity having magnitude only is called Scalar and a quantity with
magnitude  and  direction  both,  called  Vector.  But  certain  quantities  are  associated  with  two  or  more
directions, such a quantity is called Tensor. The stress at a point of an elastic solid is an example of a
Tensor which depends on two directions one normal to the area and other that of the force on it.

Tensors have their applications to Riemannian Geometry, Mechanics, Elasticity, Theory of Relativity,

Electromagnetic Theory and many other disciplines of Science and Engineering.

This book has been presented in such a clear and easy way that the students will have no difficulty

in understanding it. The definitions, proofs of theorems, notes have been given in details.
The subject is taught at graduate/postgraduate level in almost all universities.
In the end, I wish to thank the publisher and the printer for their full co-operation in bringing out

the book in the present nice form.

Suggestions for further improvement of the book will be gratefully acknowledged.

Dr. Nazrul Islam

This page
intentionally left
blank

CONTENTS

Foreword ............................................................................................................. vii
Preface ................................................................................................................ ix

Chapter�1 Preliminaries ........................................................................................... 1-5
1.1.
n-dimensional Space ............................................................................................... 1
1.2. Superscript and Subscript ....................................................................................... 1
1.3. The Einstein's Summation Convention ...................................................................... 1
1.4. Dummy Index ....................................................................................................... 1
1.5. Free Index ............................................................................................................. 2
1.6. Kr�necker Delta ..................................................................................................... 2
Exercises ............................................................................................................... 5

Chapter�2 Tensor Algebra ..................................................................................... 6-30
2.1.
Introduction .......................................................................................................... 6
2.2. Transformation of Coordinates ................................................................................ 6
2.3. Covariant and Contravariant Vectors ......................................................................... 7
2.4. Contravariant Tensor of Rank Two .......................................................................... 9
2.5. Covariant Tensor of Rank Two ................................................................................ 9
2.6. Mixed Tensor of Rank Two..................................................................................... 9
2.7. Tensor of Higher Order ......................................................................................... 14
2.8. Scalar or Invariant ................................................................................................ 15
2.9. Addition and Subtraction of Tensors ....................................................................... 15
2.10. Multiplication of Tensors (Outer Product of Tensors) ............................................... 16
2.11. Contraction of a Tensor ........................................................................................ 18
2.12.
Inner Product of Two Tensors .............................................................................. 18
2.13. Symmetric Tensors .............................................................................................. 20
2.14. Skew-symmetric Tensor ....................................................................................... 20
2.15. Quotient Law ....................................................................................................... 24

xii

Tensors and Their Applications

2.16. Conjugate (or Reciprocal) Symmetric Tensor .......................................................... 25
2.17. Relative Tensor .................................................................................................... 26
Examples ............................................................................................................ 26
Exercises ............................................................................................................. 29

Chapter�3 Metric Tensor and Riemannian Metric ............................................ 31-54

3.1. The Metric Tensor ............................................................................................... 31
3.2. Conjugate Metric Tensor (Contravariant Tensor) ...................................................... 34
3.3. Length of a Curve ................................................................................................ 42
3.4. Associated Tensor ................................................................................................ 43
3.5. Magnitude of Vector ............................................................................................. 43
3.6. Scalar Product of Two Vectors .............................................................................. 44
3.7. Angle Between Two Vectors .................................................................................. 45
3.8. Angle Between Two Coordinate Curves .................................................................. 47
3.9. Hypersurface ....................................................................................................... 48
3.10. Angle Between Two Coordinate Hyper surface ........................................................ 48
n-Ply Orthogonal System of Hypersurfaces ............................................................. 49
3.11.
3.12. Congruence of Curves .......................................................................................... 49
3.13. Orthogonal Ennuple .............................................................................................. 49
Examples ............................................................................................................ 52
Exercises ............................................................................................................. 54

Chapter�4 Christoffel�s Symbols and Covariant Differentiation ................................ 55-84

4.1. Christoffel�s Symbol............................................................................................. 55
4.2. Transformtion of Christoffel�s Symbols .................................................................. 64
4.3. Covariant Differentiation of a Covariant Vector ........................................................ 67
4.4. Covariant Differentiation of a Contravariant Vector ................................................... 68
4.5. Covariant Differentiation of Tensors ....................................................................... 69
4.6. Ricci�s Theorem .................................................................................................. 71
4.7. Gradient, Divergence and Curl ............................................................................... 75
4.8. The Laplacian Operator ......................................................................................... 80
Exercises ............................................................................................................. 83

Chapter�5 Riemann-Christoffel Tensor ............................................................ 85-110

5.1. Riemann-Christoffel Tensor ................................................................................... 85
5.2. Ricci Tensor ........................................................................................................ 88
5.3. Covariant Riemann-Christoffel Tensor .................................................................... 89

................................... 91
5.4. Properties of Riemann-Christoffel Tensors of First Kind
5.5. Bianchi Identity .................................................................................................... 94
5.6. Einstein Tensor .................................................................................................... 95
5.7. Riemannian Curvature of Vn .................................................................................. 96

lkjiR

Contents

xiii

5.8. Formula For Riemannian Curvature in Terms of Covariant

Curvature Tensor of Vn ......................................................................................... 98
5.9. Schur�s Theorem ............................................................................................... 100
5.10. Mean Curvature ................................................................................................. 101
5.11. Ricci Principal Directions .................................................................................... 102
5.12. Einstein Space ................................................................................................... 103
5.13. Weyl Tensor or Projective Curvature Tensor ......................................................... 104
Examples .......................................................................................................... 106
Exercises ........................................................................................................... 109

Chapter�6 The e-systems and the Generalized Kr�necker Deltas ................ 111-115

6.1. Completely Symmetric ......................................................................................... 111
6.2. Completely Skew-symmetric ................................................................................ 111
6.3.
e-system ........................................................................................................... 112
6.4. Generalized Kr�necker Delta ................................................................................ 112

6.5. Contraction of

jki
� � ��

............................................................................................ 114

Exercises ........................................................................................................... 115

Chapter�7 Geometry ........................................................................................ 116-141

7.1. Length of Arc .................................................................................................... 116

7.2. Curvilinear Coordinates in

3E .............................................................................. 120
7.3. Reciprocal Base System Covariant and Contravariant Vectors .................................. 122
7.4. On The Meaning of Covariant Derivatives ............................................................. 127
7.5.
Intrinsic Differentiation ....................................................................................... 131
7.6. Parallel Vector Fields ........................................................................................... 134
7.7. Geometry of Space Curves ................................................................................. 134
7.8. Serret-Frenet Formulae ....................................................................................... 138
7.9. Equations of A Straight Line ................................................................................ 140
Exercises ........................................................................................................... 141

Chapter�8 Analytical Mechanics ..................................................................... 142-169

8.1.
Introduction ...................................................................................................... 142
8.2. Newtonian Laws ................................................................................................ 142
8.3. Equations of Motion of Particle ............................................................................ 143
8.4. Conservative Force Field ..................................................................................... 144
8.5. Lagrangean Equation of Motion ........................................................................... 146
8.6. Applications of Lagrangean Equations ................................................................... 152
8.7. Hamilton�s Principle ............................................................................................ 153
8.8.
Integral Energy .................................................................................................. 155
8.9. Principle of Least Action ..................................................................................... 156

xiv

Tensors and Their Applications

8.10. Generalized Coordinates ...................................................................................... 157
8.11. Lagrangean Equation of Generalized Coordinates ................................................... 158
8.12. Divergence Theorem, Green�s Theorem, Laplacian Operator and Stoke�s

Theorem in Tensor Notation ................................................................................ 161
8.13. Gauss�s Theorem ............................................................................................... 164
8.14. Poisson�s Equation ............................................................................................. 166
8.15. Solution of Poisson�s Equation ............................................................................. 167
Exercises ........................................................................................................... 169

Chapter�9 Curvature of a Curve, Geodesic .................................................... 170-187

9.1. Curvature of Curve, Principal Normal................................................................... 170
9.2. Geodesics ......................................................................................................... 171
9.3. Euler�s Condition ............................................................................................... 171
9.4. Differential Equations of Geodesics ...................................................................... 173
9.5. Geodesic Coordinates ......................................................................................... 175
9.6. Riemannian Coordinates ...................................................................................... 177
9.7. Geodesic Form of a Line Element ........................................................................ 178
9.8. Geodesics in Euclidean Space .............................................................................. 181
Examples .......................................................................................................... 182
Exercises ........................................................................................................... 186

Chapter�10 Parallelism of  Vectors ................................................................. 188-204

10.1. Parallelism of a Vector of Constant Magnitude (Levi-Civita�s Concept) ..................... 188
10.2. Parallelism of a Vector of Variable Magnitude ......................................................... 191
10.3. Subspace of Riemannian Manifold ........................................................................ 193
10.4. Parallelism in a Subspace .................................................................................... 196
10.5. Fundamental Theorem of Riemannian Geometry Statement ..................................... 199
Examples .......................................................................................................... 200
Exercises ........................................................................................................... 203

Chapter�11 Ricci�s Coefficients of Rotation and Congruence ....................... 205-217
11.1. Ricci�s Coefficient of Rotation ............................................................................. 205
11.2. Reason for the Name �Coefficients of Rotation� .................................................... 206
11.3. Curvature of Congruence .................................................................................... 207
11.4. Geodesic Congruence ......................................................................................... 208
11.5. Normal Congruence ........................................................................................... 209
11.6. Curl of Congruence ............................................................................................ 211
11.7. Canonical Congruence ........................................................................................ 213
Examples .......................................................................................................... 215

Exercises ........................................................................................................... 217

Contents

xv

Chapter�12 Hypersurfaces .............................................................................. 218-242
Introduction ...................................................................................................... 218
12.1.
12.2. Generalized Covariant Differentiation .................................................................... 219
12.3. Laws of Tensor Differentiation ............................................................................ 220
12.4. Gauss�s Formula ................................................................................................ 222
12.5. Curvature of a Curve in a Hypersurface and Normal Curvature, Meunier�s Theorem,

Dupin�s Theorem ............................................................................................... 224
12.6. Definitions ......................................................................................................... 227
12.7. Euler�s Theorem ................................................................................................ 228
12.8. Conjugate Directions and Asymptotic Directions in a Hypersurface.......................... 229
12.9. Tensor Derivative of Unit Normal......................................................................... 230
12.10. The Equation of Gauss and Codazzi ..................................................................... 233
12.11. Hypersurfaces with Indeterminate Lines of Curvature ............................................ 234
12.12. Central Quadratic Hypersurfaces .......................................................................... 235
12.13. Polar Hyperplane ................................................................................................ 236
12.14. Evolute of a Hypersurface in an Euclidean Space ................................................... 237
12.15. Hypersphere ......................................................................................................238
Exercises ........................................................................................................... 241

Index .................................................................................................................... 243-245

This page
intentionally left
blank

CHAPTER � 1

PRELIMINARIES

1.1 n-DIMENSIONAL  SPACE

In three dimensional rectangular space, the coordinates of a point are (x, y, z). It is convenient to write
(x1, x2, x3) for (x, y, z). The coordinates of a point in four dimensional space are given by (x1, x2, x3, x4).
In general, the coordinates of a point in  n-dimensional space are given by (x1, x2,  x3,...., xn) such  n-
dimensional space  is denoted by Vn.

1.2 SUPERSCRIPT  AND  SUBSCRIPT

In the symbol
in the lower position are called subscripts.

ij

klA , the indices i, j written in the upper position are called superscripts and k, l written

1.3 THE  EINSTEIN'S  SUMMATION  CONVENTION

Consider the sum of the series

xa
2
drop the sigma sign and write convention as

1
xa
1

S

+

=

2

++
...

n xa

n

=

n

=

1

i

xa
i

i

.

 By using summation convention,

n

1=

i

i

xa
i

=

i

i xa

This convention is called Einstein�s Summation Convention and stated as

�If a suffix occurs twice in a term, once in the lower position and once in the upper position then

that suffix implies sum over defined range.�
If the range is not given, then assume that the range is from 1 to n.

1.4 DUMMY  INDEX

Any index which is repeated in a given term is called a dummy index or dummy suffix. This is also called
Umbral or Dextral Index.

e.g. Consider the expression ai xi where i is dummy index; then
ai xi =

1
xa
1

xa
2

n xa

+(cid:215)

(cid:215)+

+

n

2

S
S
(cid:215)
2

and

Tensors and Their Applications

ajxj =

1
xa
1

+

2
xa
2

(cid:215)+

+(cid:215)

n

nxa

These two equations prove that

aixi = aj  xj

So, any dummy index can be replaced by any other index ranging the same numbers.

1.5 FREE  INDEX

Any index occurring only once in a given term is called a  Free Index.

e.g. Consider the expression

i

j
i xa

 where j is free index.

1.6 KR�NECKER  DELTA

The symbol

i
j

is defined by

, called Kr�necker Delta (a German mathematician Leopold Kr�necker, 1823-91 A.D.)

i
d =
j

if1

=

i

if0

i

j

j

Similarly d

ij and d

ij are defined as

if1
if0

=

i
i

j
j

ij

=   (cid:238)

=

ij

if1

if0

=

i

i

j

j

and

Properties
1.

If x1, x2, ... xn are independent coordinates, then
i
x

= 0 if

i �

 j

j

i

j

x
x

x

i

x

j

x

= 1 if

i = j

=

i
j

This implies that

It is also written as

i

x
k

x

k

j

x

x

=

i
jd .

2.

3.

i
i

2
2

d+

d+d=d

1
1
(cid:215)+++=d
111

d+(cid:215)

n
n

(cid:215)+

3
3
+(cid:215)

1

(by summation convention)

i
i
d = n
i
i
=

ij

a

j
k

ik

a

Since

3 d  =
ja
j
2

a

31

+

32

a

1
2

+

33

a

2
2

3
2

(cid:215)+

+(cid:215)

3
nna
2

(as j is dummy index)

(cid:215)
d
(cid:238)
(cid:237)
(cid:236)
�
d
(cid:237)
(cid:236)
�
d
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
d
�
�
�
�
(cid:215)
(cid:215)
d

d
(cid:215)
d
d
d
Preliminaries

3

In general,

4.

EXAMPLE  1

Write

Solution

=  a32

� as(

1
2

=

�

3
2

(cid:215)=

=(cid:215)

=

�

n
2

0

and

�

=

)1

2
2

ija d  =
j
k

a

i

11
k

+

a

i

22
k

+

a

i

33
k

(cid:215)+

+(cid:215)

ik

a

(cid:215)+

k
k

+(cid:215)

in

a

n
k

ija d

i
j

i
j

j
k

j
k

j
k

= ika

= i
k

=

 =

i
1

i
k

 sa(

d=

1
k

2
k

(cid:215)=

d=(cid:215)

=

n
k

0

and

=

)1

k
k

d+

1
k

i
2

d+

2
k

i
3

i
�as(
1

d=

i
2

d=

(cid:215)+

d+(cid:215)

(cid:215)=

d=(cid:215)

3
k

i
3

i
k

=

i
i

i
n

(cid:215)+

d+(cid:215)

0

and

n
k

=

)1

i
n

i
i

df
dt

=

f
1

x

1
dx
dt

+

f
2

x

2

dx
dt

(cid:215)+

+(cid:215)

f
n

x

n

dx
dt

 using summation convention.

df
dt

df
dt

 =

f
1

x

1

dx
dt

+

f
2

x

2

dx
dt

(cid:215)+

+(cid:215)

f
n

x

n

dx
dt

i

dx
dt

 =

i

x

EXAMPLE  2

Expand: (i) aij xixj; (ii) glm gmp

Solution

(i)

i
xxa
ij

j

=

1

xxa
1
j

j

+

2

xxa
2
j

j

(cid:215)+

+(cid:215)

n

xxa
nj

j

=

11
xxa
11

+

2

xxa
22

2

(cid:215)+

+(cid:215)

nn
xxa
nn

i
xxa
ij

j

=

a

21
x
)

(

+

(

x

22
)

(cid:215)+

a

22

11

+(cid:215)

a

nn x
(

n

2

)
(as i and j are dummy indices)

(ii)

lm gg

=

mp

gg
11
l

p

+

gg
2
l

2

p

(cid:215)+

+(cid:215)

gg
ln

np

, as m is dummy index.

EXAMPLE  3

If aij are constant and aij = aji, calculate:

(i)

x�

k

(

xxa
i
ij

j

)

(ii)

x �
k

x
l

(

xxa
ij
i

)

j

Solution

(i)

(

xxa
i
ij

j

)

a

ij

=

x�

k

(

xx
i

j

)

x
k

(cid:215)
d
(cid:215)
d
(cid:215)
d
d
d
d
(cid:215)
d
d
d
d
d
d
d
(cid:215)
d
(cid:215)
d
d
d
d
d
d
(cid:215)
�
�
(cid:215)
�
�
�
�
�
�
(cid:215)
�
�
�
�
�
f
�
(cid:215)
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
�
�
4

Tensors and Their Applications

 =

xa
ij

i

x

x

j

k

+

xa
ij

j

x
x

i

k

=

xa
i
ij

d

jk

+

xa
ij

d

j

ik

,

as

x

j

x

k

=

d

jk

)

x
i

+

(

a

ij

=

(

a

ij

jk
 =  aik xi + akj xj
 = aik xi + aki xi

)

x

j
ik
as aij
jk = aik
as j is dummy index

(

xxa
i
ij

j

)

kx

 = 2aik xi

as given aik = aki

(

(ii)

xxa
i
ij
x
Differentiating it w.r.t. xl :

k

)

j

= 2aikxi

(2

)

j

xxa
i
ij
x
l

x
k

(2

)

j

xxa
i
ij
x
l

x
k

=

2

=

2

a

x
i
ik x
ika d

i
l

l

= 2alk

as

ika d

i
l

 = alk.

EXAMPLE  4
If
where aij are constant then show that

aij xi xj= 0

aij + aji = 0

Solution

Given

aijxixj= 0
almxlxm= 0
Differentiating it w.r.t. xi partially,

x�
i

a

lm

(

ml

xxa
lm

)

= 0

ml
xx

(

)

= 0

x
i

since i and j are dummy indices

a

lm

l

x
x
i

m

x

+

a

lm

Since

m

x
x

l

x

= 0

i

l

x
x
i

=

l
i

and

m

x
x
i

d=

m
i

�
�
�
�
�
�
d
d
d
�
�
�
�
�
�
�
�
�
�
�
�
(cid:222)
�
�
�
�
�
�
�
�
�
d
�
�
Preliminaries

5

a

d
lm

ml
x
i

+

a
lm

d

lm
x
i

= 0

as

a

lm

=d
l
i

a

im

 and

xa
im
a

lm

m

+

xa
li
=

a

l

= 0

.li

m
i

Differentiating it w.r.t. xj partially

a

im

m

x
x

j

+

a

li

x
x

l

j

= 0

a

im

+

m
j

a

li

l
j

= 0

aij + aji= 0

Proved.

EXERCISES

1. Write the following using the summation convention.

(i) (x 1)2 + (x2)2 + (x3)2 + . . . + (xn)2
(ii)
(iii)

ds2 = g11 (dx1)2 + g22(dx2)2 + . . . + gnn(dxn)2
a1x1x3 + a2x2x3 + . . . + anxnx3

2. Expand the following:

(i) aijxj

3. Evaluate:

(

ag

i

)

(ii)

i

x�

(iii)

i

k
i BA

(i)

jx d

i
j

(ii)

i
j

j
k

k
l

  (iii)

i
j

j
i

4. Express b i j y i y j in the terms of x variables where yi = cij xj and bijcik =

.i
k

ANSWERS

1.
2.

(ii) ds2 = gij dxidxj
(i) xixj
(i) ai1x1 + ai2x2 + ai3x3 + . . . + ainxn

(iii) ai x i x 3 .

(ii)

1
x

(

1
ag

)

+

2

x

(

ag

2

)

(cid:215)+

+(cid:215)

(

ag

n

)

n

x

(iii)

1

k
BA
1

+

2

k
BA
2

++
...

n

k
BA
n

3.

(i) xi

(ii)

i
l

(iii) n

4. Cij xi xj

d
�
�
�
�
d
d
�

d
d
d
d
d
d
�
�
(cid:215)
�
�
�
�
d
CHAPTER � 2

TENSOR    ALGEBRA

2.1 INTRODUCTION

A  scalar  (density,  pressure,  temperature,  etc.)    is  a  quantity  whose  specification  (in  any  coordinate
system) requires just one number. On the other hand, a vector (displacement, acceleration, force, etc.)
is a quantity whose specification requires three numbers, namely its components with respect to some
basis.  Scalers  and  vectors  are  both  special  cases  of  a  more  general  object  called  a  tensor  of  order
n whose specification in any coordinate system requires 3n numbers, called the components of tensor.
In fact, scalars are tensors of order zero with 3� = 1 component. Vectors are tensors of order one with
31 = 3 components.

2.2 TRANSFORMATION  OF  COORDINATES

In three dimensional rectangular space, the coordinates of a point are  (x,  y,   z) where x, y, z are real
numbers. It is convenient to write (x1, x2, x3) for
 or simply xi where i = 1, 2, 3. Similarly in
zyx
,
,(
n- dimensional space, the coordinate of a point are n-independent variables (x1, x2,..., x n) in X-coordinate
system. Let

 be coordinate of the same point in Y-coordinate system.
(
x
1 �  be independent single valued function of x1, x2,...., xn, so that,
, 2
x

,...,
nx

nx

x

)

)

,

,

,

1

2

Let

x

or

1x =

2x =

1
1
xxx
(

,

2

,

....,

nx

)

x

2

1
xx
,

(

2

,

....,

nx

)

3x =

x

3

1
xx
,

(

2

,

....,

nx

)

M
nx =

ix =

M

x

n

1
xx
,

(

2

,

...,

x

n

)

i

1
xxx
(

,

2

,

...,

x

n

)

;

i = 1, 2, �, n

�(1)

Tensor Algebra

7

Solving these equations and expressing xi as functions of

1
xx
,

2

,...,

nx

,

 so that

ix  =

1
i
xxx
(

,

2

,...,

x

n

);

i = 1, 2, ..., n

The equations (1) and (2) are said to be a transformation of the coordinates from one coordinate

system to another

2.3 COVARIANT  AND  CONTRAVARIANT  VECTORS  (TENSOR  OF  RANK  ONE)
Let (x1, x2, ..., xn) or xi be coordinates of a point in X-coordinate system and
coordinates of the same point in the Y-coordinate system.

,...,

nx

x

x

)

(

,

2

1

  or

ix  be

Let  Ai,  i   =   1,  2,  ...,  n  (or  A1,  A2,  ...,  An) be  n   functions  of  coordinates  x1,  x2,  ...,  xn
iA in Y-coordinate system then according

in X-coordinate system. If the quantities Ai are transformed to
to the law of transformation

iA  =

i

x

j

x

j

A

or

jA  =

j

i

x

x

iA

Then Ai are called components of contravariant vector.
Let

,iA   =i

 1, 2,..., n (or A1, A2,  �,  An)  be  n  functions  of  the  coordinates  x1,  x2,  ...,  xn
iA   in  Y-coordinate  system  then

iA   are  transformed  to

in  X-coordinate  system.  If  the  quantities
according to the law of transformation

iA  =

j

x

i

x

A

j

or

jA  =

i

x

j

x

A
i

Then Ai are called components of covariant vector.
The contravariant (or covariant) vector is also called a contravariant (or covariant) tensor of rank

one.

Note: A superscript is always used to indicate contravariant component and a subscript is always used to indicate

covariant component.

EXAMPLE  1

If  xi  be  the  coordinate  of  a  point  in  n-dimensional  space  show  that  dxi  are  component  of  a

contravariant vector.

Solution

Let    x1,  x2,  ...,  xn or  xi  are  coordinates  in  X-coordinate  system  and

1
,
xx

2

,...,

nx

  or

ix   are

coordinates in Y-coordinate system.
If

ix =

ixd

=

i

1
xxx
(

,

2

,...,

x

n

)

i

x

1

x

1

+

dx

i

x

2

x

2

(cid:215)+

dx

+(cid:215)

i

x

n

x

n

dx

�
�
�
�

�
�
�
�
�
�
(cid:215)
�
�
�
�
8

Tensors and Their Applications

ixd

 =

i

x

j

x

j

dx

It is law of transformation of contravariant vector. So,

idx  are components of a contravariant

vector.

EXAMPLE  2

Show that

ix�

 is a covariant vector where f

 is a scalar function.

Solution

Let  x1,  x2,  ...,  xn  or

ix   are  coordinates  in  X-coordinate  system  and

1

x

,

x

2

,

...,

nx

or

ix   are

coordinates in Y-coordinate system.

Consider

(

x

1

,

x

2

,...,

nx

)

 =

1
x

(

,

2

x

,...,

nx

)

1
x

+

 =

1

x

2

x

2

x

(cid:215)+

+(cid:215)

n

x

n

x

+

2

x

2

x

i

x

(cid:215)+

+(cid:215)

n

x

i

x

n

x

 =

ix�

1

x

 =

ix�

 =

ix�

j

j

i

x

x

x

1
x

i

x

j

x

i

x

j

x

or

It is law of transformation of component of covariant vector. So,

vector.

EXAMPLE  3

 is component of covariant

ix�

Show that the velocity of fluid at any point is a component of contravariant vector

or
Show that the component of tangent vector on the curve in n-dimensional space are component

of contravariant vector.

Solution

Let

1
dx
dt

,

2

dx
dt

�

,

,

n

dx
dt

 be the component of the tangent vector of the point

1
xx
,

(

2

,...,

nx

)

 i.e.,

dxi
dt

 be the component of the tangent vector in X-coordinate system. Let the component of tangent

�
�
f
�
f
f
f
�
�
�
f
�
(cid:215)
�
�
f
�
�
�
f
�
f
�
�
�
�
f
�
(cid:215)
�
�
�
f
�
�
�
�
f
�
f
�
�
�
�
f
�
f
�
�
f
�
�
�
f
�
Tensor Algebra

9

vector of the point

1
xx
,

(

2

,...,

nx

)

 in Y-coordinate system are

function of

1
x

,

x

2

,

...,

nx

 which is a function of t. So,

xd i
dt

.

 Then

x

1

,

2

x

,

...,

nx

 or

ix  being a

xd i
dt

xd i
dt

 =

 =

i

x
1

dt

1
dx
dt

+

i

x

2

dx

2

dx
dt

(cid:215)+

+(cid:215)

i

x

n

dx

n

dx
dt

i

x

j

dx

j

dx
dt

It  is  law  of  transformation  of  component  of  contravariant  vector.  So,

contravariant vector.

dxi
dt

  is  component  of

i.e.  the  component  of  tangent  vector  on  the  curve  in  n-dimensional  space  are  component  of

contravariant vector.

2.4 CONTRAVARIANT  TENSOR  OF  RANK  TWO
Let Aij  (i, j = 1, 2, ...,  n) be  n2 functions of coordinates  x1, x2, ...,  xn in  X-coordinate system. If the

quantities

ijA are transformed to

ijA  in  Y-coordinate system having coordinates

1

x

,

x

2

,

...,

nx

.  Then

according to  the law of transformation

ijA  =

i

x

k

x

j

x

l

x

kl

A

Then

ijA  are called components of Contravariant Tensor of rank two.

2.5 COVARIANT  TENSOR  OF  RANK  TWO

Let Aij (i, j = 1, 2, ..., n) be
quantities
according to the law of transformation,

ijA  are transformed to

2n  functions of coordinates x1,  x2, ..., xn in  X-coordinate system. If the

ijA  in  Y-coordinate system having coordinates

1
,
xx

2

,...,

nx

,  then

ijA =

k

x

i

x

l

x

j

x

A
kl

Then Aij called components of covariant tensor of rank two.

2.6 MIXED  TENSOR  OF  RANK  TWO

Let

i
jA  (i, j = 1, 2, ..., n) be  n2 functions of coordinates x1, x2, ...,  xn in  X-coordinate system. If the

i

quantities
according to the law of transformation

jA  are transformed to

i

jA  in  Y-coordinate system having coordinates

1
xx
,

2

,...,

nx

,

  then

i
jA  =

i

x
k

x

l

x
x

k
A
l

Then

i

jA  are called components of mixed tensor of rank two.

�
(cid:215)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
10

Note:

(i) The rank of the tensor is defined as the total number of indices per component.
(ii) Instead of saying that � Aij are the components of a tensor of rank two� we shall often say � Aij  is a tensor

of rank two.�

Tensors and Their Applications

THEOREM 2.1 To show that the Kr�necker delta is a mixed tensor of rank two.

Solution

Let X and Y be two coordinate systems. Let the component of Kronecker delta in X-coordinate

i
j

system
transformation

  and  component  of  Kr�necker  delta  in  Y-coordinate be

i
j

,  then  according  to  the  law  of

i
j

i
j

i
j

This shows that Kr�necker

EXAMPLE  4

 =

=

i

x

j

x

i

x

k

x

=

i

x
k

x

l

x

j

x

k

x

l

x

l

x

j

x

k
l

 is mixed tensor of rank two.

If

iA  is a covariant tensor, then prove that

A
i
j

x

 do not form a tensor..

Solution

Let X and Y be two coordinate systems. As given

iA  is a covariant tensor. Then

Differentiating it w.r.t.

jx

iA =

k

x

i

x

A
k

A
i
j

x

A
i
j

x

=

=

j

k

x

x

i

x

k

x

i

x

A
k

A
k
j

x

+

A
k

2

k

x

i
xx

j

It is not any law of transformation of tensor due to presence of second term. So,

�(1)

A
i
j

x

 is not a

tensor.

THEOREM  2.2 To  show  that
system.

i
j

  is  an  invariant  i.e.,  it  has  same  components  in  every  coordinate

Proof: Since

i
j

 is a mixed tensor of rank two, then

i
j

=

i

x
k

x

l

x

j

x

d

k
l

d
d
d
�
�
�
�
�
�
�
�
d
d
�
�
�
�
d
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
d
d
d
�
�
�
�
Tensor Algebra

11

=

=

i

x
k

x

i

x
k

x

l

x

j

x
k

x

j

x

d

k
l

, as

l

x

j

x

=

d

k
l

k

x

j

x

i
j

=

i

x

j

x

d=

i
j

 , as

i

x

j

x

d=

i
j

So,

i
j

 is an invariant.

THEOREM 2.3 Prove that the transformation of a contravariant vector is transitive.

or
Prove that the transformation of a contravariant vector form a group.

Proof: Let

iA  be a contravariant vector in a coordinate system

ixi =
(

2,1

n
,...,

)

. Let  the coordinates

xi be transformed to the coordinate system
When coordinate xi be transformed to
p

pA =

x

q

x

ix  be transformed to

ix  and
ix , the law of transformation of a contravariant vector is

ix .

q

A

... (1)

When coordinate

ix  be transformed to

ix , the law of transformation of contravariant vector is

iA =

iA =

iA =

i

x

p

i

i

x
x

x

i

x
q

x

p

A

p

x

q

x

q

A

Aq from (1)

This shows that if we make direct transformation from

ix  to

ix , we get same law of transformation.

This property is called that transformation of contravariant vectors is transitive or form a group.

THEOREM 2.4 Prove that the transformation of a covariant vector is transitive.

Prove that the transformation of a covariant vector form a group.

or

Proof: Let

iA  be a covariant vector in a coordinate system

transformed to the coordinate system

ix  and
ix  be transformed to

When coordinate

ixi =
(

,2,1
ix  be transformed to
ix , the law of transformation of a covariant vector is

. Let the coordinates

...,
ix .

n

)

ix be

pA =

q

x

p

x

A
q

...  (1)

When coordinate

ix  be transformed to

ix , the law of transformation of a covariant vector is

iA =

p

x

i

x

A

p

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
d
�
�
�
�
d
�
�
�
�
�
�
�
�
�
�
�
�
�
�
12

Tensors and Their Applications

iA =

iA =

q

x

p

x

A
q

A
q

p

x

i

x
q
x

i

x

This shows that if we make direct transformation from

ix  to

ix , we get same law of transformation.

This property is called that transformation of covariant vectors is transitive or form a group.

THEOREM 2.5 Prove that the transformations of tensors form a group

or

Prove that the equations of transformation a tensor (Mixed tensor) posses the group property.

jA  be a mixed tensor of rank two in a coordinate system

i

Proof: Let
ix  be transformed to the coordinate system
ix  be transformed to

When coordinate

2,1
ix  be transformed to

ixi =
(

,...,
n
ix .

ix  and
ix , the transformation of a mixed tensor of rank two is
s

)

. Let the coordinates

p
qA =

p

x

r

x

When coordinate

ix  be transformed to

two is

x

q

r
A
s

x
ix , the law of transformation of a mixed tensor of rank

... (1)

i
jA =

=

i
jA =

i

x

p

x

i

p

x

x

q

j

q

j

x

x

x

x

p
A
q

p

x

r

x

s

q

x

x

r
A
s

 from (1)

i

x
r

x

s

x

j

x

r
A
s

This shows that if we make direct transformation from

ix  to

ix , we get same law of transformation.

This property is called that transformation of tensors form a group.

THEOREM 2.6 There is no distinction between contravariant and covariant vectors when we restrict
ourselves to rectangular Cartesian transformation of coordinates.

Proof: Let P(x, y) be a point with respect to the rectangular Cartesian axes X and Y. Let
 be the
coordinate of the same point P in another rectangular cartesian axes  X  and Y , Let (l1, m1) and (l2, m2)
be the direction cosines of the axes  X ,  Y  respectively. Then the transformation relations are given by

yx
,(

)

x

=

+
ymxl
1
+
ymxl
2
and solving these equations, we have
xl
1

=

+

=

y

x

1

2

yl
2
+

=

ymxmy
1

2

put

x =

1x

,

y =

,2x

x =

,1x

y =

2x

...(1)

...(2)

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:254)
(cid:253)
(cid:252)

Tensor Algebra

13

Consider the contravariant transformation

iA =

iA =

1A =

2A =

i

j

i

x

x

x

1
x

1

x

1
x
x

2

1
x

A

;j

2,1=j

1
A

+

1
A

+

1
A

+

i

2

x

x

1

x

2

x
x

2

2

x

2

A

2

A

2

A

1l

, but

x =

1x

,

y =

2x

x =

1x

,

,

y =

2x

x
x

x
y

y
x

=

1

1

x

x

=

l
1

.

=

m
1

=

=

l

2

=

1

x

2

x
2

x

1
x

;

;

y
y

=

m
2

=

2

2

x

x

for

2,1=i

.

x =
x

From (1)

Then

Similarly,

So, we have

1

+
AmAl
1
+
AmAl
2
Consider the covariant transformation
j

A

A

=

=

2

1

2

2

1

1

2

iA =

iA =

1A =

2A =

for

2,1=i

.

From (3)

x

i

x
1
x

i

x

1
x
1

x
x

x

A

;j

2,1=j

+

A
1

+

A
1

A
2

A
2

2

x

1
x

2

x

1

x
x

x

1

2

+

A
1

2

2

A
2

=

+

AmAlA
11
21
1
+
AmAl
A
12
22
2

=

...(3)

..(4)

...(5)

�
�

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:239)
(cid:239)
(cid:254)
(cid:239)
(cid:239)
(cid:253)
(cid:252)
�
�
�
�
�
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
�
�

�
�
�
�
�
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
14

Tensors and Their Applications

So, from (4) and (5), we have

Hence the theorem is proved.

A =

1 A
1

 and

A =
2

A
2

2.7 TENSORS  OF  HIGHER  ORDER

(a)  Contravariant  tensor  of  rank  r

Let

iiA ...21
ri

be nr function of coordinates x1, x2, ..., xn in X-coordinates system. If the quantities

iiA ...2
i

i

r

are transformed to
to the law of transformation

iiA ...21
ri

 in Y-coordinate system having coordinates

1

x

,

x

2

,

...,

nx

. Then according

iiA ...21
ri

=

i
1

x

p
1

x

i
2

K2

p

x

x

i

r

x

p

r

x

ppA

21

...

rp

Then

iiA ...2
i

i

r

 are called components of contravariant tensor of rank r.

(b)  Covariant  tensor  of  rank  s

Let

jjA ...21

sj

 be

sn   functions  of  coordinates  x1,  x2, ...,  xn in  X-coordinate  system.  If  the  quantities

 are transformed to

jjA ...21
according to the law of transformation

jjA ...21

sj

sj

 in  Y- coordinate system having coordinates

1
,
xx

2

,...,

nx

.  Then

jjA ...21

sj

=

q
1

x

j
1

x

q

2

j

2

x

x

q

s

x

j

s

x

qqA
, 2

1

...,
sq

Then

jjA ...21

sj

 are called the components of covariant tensor of rank s.

(c)  Mixed  tensor  of  rank  r  +  s

 be  nr+s  functions  of  coordinates  x1,  x2, ...,  xn in  X-coordinate  system.  If  the  quantities

 in  Y-coordinate system having coordinates

1

x

,

x

2

,

� .  Then

nx

,

ii
21

jjA ...

i
r
...
j

s

21

ii
21

Let
jjA ...
according to the law of transformation

 are transformed to

jjA ...

i
r
...
j

i
r
...
j

ii
21

21

21

s

s

ii
21

jjA ...

i
r
...
j

21

=

s

i
1

x

p
1

x

i

2

x

p
2

x

i
r

x

p

r

x

q
1

x

j
1

x

q

2

j

2

x

x

q

s

x

j

s

x

pp
qqA
21

...
p
...
q

21

s

r

Then

ii
21

jjA ...

i
r
...
j

21

s

 are called component of mixed tensor of rank
)sr,

  is  known  as  tensor  of  type  (

j

iiA ...

j
j
21
...
i

r + .
s

A  tensor  of  type

,  In  (r,s),  the  first  component  r
indicates the rank of contravariant tensor and the second component s indicates the rank of covariant
tensor.

21

s

r

Thus the tensors

ijA  and

ijA  are type (0, 2) and (2, 0) respectively while tensor

i

jA  is type (1, 1).

�
�
�
�
�
�

�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�

�
�
�
�
�
�

�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
Tensor Algebra

EXAMPLE

15

ijk

lmA  is a mixed tensor of type (3, 2) in which contravariant tensor of rank three and covariant

tensor of rank two. Then according to the law of transformation

ijk
lmA

=

2.8 SCALAR  OR  INVARIANT

i

x
a

x

j

b

x

x

k

x

g
x

a

x

l

x

b

x
m

x

abA

A function

1
xf
(

,

2

x

,

...,

nx

)

 is called Scalar or an invariant if its original value does not change upon

transformation of coordinates from  x1, x2, ..., xn to

1

x

,

x

2

,

...,

nx

. i.e.

1
x

(

,

2

x

,...,

nx

)

=

(

x

1

,

x

2

,...,

nx

)

Scalar is also called tensor of rank zero.

For example,

i BA
i

 is scalar..

2.9 ADDITION  AND  SUBTRACTION  OF  TENSORS

THEOREM 2.7 The sum (or difference) of two tensors which have same number of covariant and the
same contravariant indices is again a tensor of the same rank and type as the given tensors.
ii
jjB ...
Proof: Consider two tensors
21
rank s and contravariant tensor of rank r.). Then according to the law of transformation
i
1

 of the same rank and type (i.e., covariant tensor of

jjA ...

 and

i
r
...
j

i
r
...
j

ii
21

21

21

q
1

i
r

q

q

2

2

i

s

s

s

x

x

x

x

x

x

ii
21

jjA ...

i
r
...
j

21

ii
21

jjB ...

i
r
...
j

21

=

=

s

s

A

ii
21
jj
21

...
i
r
...
j

s

B

ii
21
jj
21

...
i
r
...
j

s

=

A

ii
21
jj
21

...
i
r
...
j

s

B

ii
21
j
j
21

...
i
r
...
j

s

=

A

pp
21
qq
21

...
p
...
q

s

r

B

pp
21
qq
21

...
p
...
q
s

r

=

p
1

x

p
2

x

p

r

x

j
1

x

j

2

x

j

s

x

i
1

x

p
1

x

i

2

x

p
2

x

i
1

x

p
1

x

i

2

x

p
2

x

i
r

x

p

r

x

i

r

x

p

r

x

q
1

x

j
1

x

q
1

j
1

x

x

q

2

j
2

x

x

q

s

x

j

s

x

�

q

2

x

j

2

x

q

s

x

j

s

x

(

ii
21

jjC ...

i
r
...
j

21

s

pp
qqC
21

...
p
...
q

21

s

r

pp
qqA
21

...
p
...
q

21

s

r

pp
qqB
21

...
p
...
q

21

s

r

A

pp
21
qq
21

...
p
...
q

s

)r

r

B

pp
21
qq
21

...
p
...
q

s

and

Then

If

and

So,

ii
21

jjC ...

i
r
...
j

21

=

s

i
1

x

p
1

x

i

2

x

p
2

x

i
r

x

p

r

x

q
1

x

j
1

x

q

2

j

2

x

x

This is law of transformation of a mixed tensor of rank r+s. So,

rank r+s or of type  (r, s).

q

s

x

j

1

s

x
i
,
jC ,...,
i
21
,
j

i
r
,...,
j
s

1

2

pp
,
qqC
1
,

,...,
2
,...,
q

2

p
r

s

is a mixed tensor of

�
�
�
�
�
�
�
�
�
�
a
b
g
f
f
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
(cid:215)
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�

�
�
�
�
�
�

�
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
�
�
�
�

16

EXAMPLE  5

Tensors and Their Applications

If

ij

kA  and

lm

nB  are tensors then their sum and difference are tensors of the same rank and type.

Solution

As given

ij

kA  and

ij

kB  are tensors. Then according to the law of transformation

and

then

If

So,

The shows that

ij
kA =

ij
kB =

A �
ij
k

ij
B
k

=

i

x

p

x

i

x

p

x

i

x

p

x

j

x

q

x

j

x

q

x

j

x

q

x

r

k

x

x

r

k

x

x

r

k

x

x

pq
A
r

pq
B
r

(

pq
A
r

)pq

B
r

A �
ij
k

B

ij
k

=

ij

kC  and

A �
pq
r

B

pq
r

 =

pq
rC

ij
kC =

i

x

j

x

r

x

C

pq
r

x
kC  is a tensor of same rank and type as

x

x

ij

p

q

k

ij

kA  and

ij
kB .

2.10 MULTIPLICATION  OF  TENSORS  (OUTER  PRODUCT  OF  TENSOR)

THEOREM 2.8 The multiplication of two tensors is a tensor whose rank is the sum of the ranks of
two tensors.

Proof: Consider two tensors

ii
21

jjA ...

i
r
...
j

21

 (which is covariant tensor of rank s and contravariant tensor of

s

rank r) and
according to the law of transformation.

21

n

...
k
kk
llB
21
...
l

m

 (which is covariant tensor of rank m and contravariant tensor of rank n).  Then

ii
21

jjA ...

i
r
...
j

21

=

s

and

...
k
kk
llB
21
...
l
n

21

m

=

Then their product is

ii
21

jjA ...

i
r
...
j

21

s

llB ...

kk
21
...

l

21

n

k

m

=

i
1

x

p
1

x

k

1

1

x

x

i
1

x

p
1

x

i

2

x

p
2

x

k

2

2

x

x

i
r

x

p

r

x

q
1

x

j
1

x

q

2

j

2

x

x

k

m

m

x

x

1

x

l
1

x

2

x

l

2

x

q

r

j

s

x

x

n

n

x
lx

pp
qqA
21

...
p
...
q

21

s

r

B

21

m

...
...

n

21

i

r

x

p
r

x

q
1

j
1

x

x

q

s

x

j

s

x

k

x
a

x

1

1

k

x
a

x

m

m

b

1

x

l
1

x

pp
qqA
21

...
p
...
q

21

s

n

m

x
lx

r

B

21

m

...
...

n

21

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
�
�
�
�

(cid:215)
(cid:215)
(cid:215)
�
�
�
�
a
a
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
�
�
b
b
a
�
�
b

a
a
a
b
b
b

(cid:215)
(cid:215)
(cid:215)
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
(cid:215)
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
�
�
b
a
a
a
b
b
b
Tensor Algebra

17

If

and

So,

C

ii
21
jj
21

...
kki
21
r
...
llj
21
s

m

...
k
...
nl

=

A

ii
21
jj
21

...
i
r
...
j
s

...
k
kk
B
21
...
l
ll
21

n

n

pp
qqC
21

...
p
...
q

21

s

r

21
...

21

...

n

m

=

pp
qqA
21

...,
p
...,
q

21

s

r

B

21

m

...
...

n

21

ii
jjC
21

21

...
kki
21
r
...
llj
21
s

m

k
...
...
l

n

=

i
1

x

p
1

k

m

m

x

x

x

i
r

x

p

r

x

1

x

l
1

x

q

1

j
1

n

x

x

x

l
n

x

q

s

j

s

x

x

k

1

1

x

x

pp
qqC
21

...
...
q

21

21

aa
p
r
bb

21

s

a
...
b
...

n

h

This  is  law  of  transformation  of  a  mixed  tensor  of  rank

+

.nsmr

++

. So,

ii
jjC
21

21

i
...
...

kkr
21
llj
21
s

k
...
m
...
l

n

  is  a

mixed tensor of rank
open proudct of two tensors.

+

nsmr

++

. or of type  (

+ ,
smr

+

)n

. Such product is called outer product or

THEOREM 2.9 If Ai and Bj are the components of a contravariant and covariant tensors of rank one
then prove that AiBj are components of a mixed tensor of rank two.
iA   is  contravariant  tensor  of  rank  one  and
Proof:  As
according to the law of transformation

jB   is  covariant  tensor  of  rank  one.  Then

and

iA =

jB =

Multiply (1) and (2), we get

i BA

j

=

i

x
k

x

l

x

j

x

i

x

k

x

k

A

B

l

l

x

j

x

k
BA
l

...(1)

...(2)

This is law of transformation of tensor of rank two. So,

i BA

j

 are mixed tensor of rank two.

Such product is called outer product of two tensors.

EXAMPLE  6

Show that the product of two tensors

i

jA  and

kl

mB  is a tensor of rank five.

Solution

As

i

jA  and

kl

mB  are tensors. Then by law of transformation

i
jA =

i

x

p

x

q

j

x

x

p
A
q

and

kl
mB =

k

x

r

x

l

s

x

x

t

x
m

x

rs
B
t

a
a
a
b
b
b
a
a
a
b
b
b
(cid:215)
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
a
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
(cid:215)
�
�
(cid:215)
(cid:215)
(cid:215)
b
b
a

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
18

Tensors and Their Applications

Multiplying these, we get

kl
i
j BA
m

=

i

x
p
x

q

j

x
x

k

r

x
x

l

s

x
x

t

x
rs
p
BA
qm
t
x

This is law of transformation of tensor of rank five. So,

i
j BA

kl
m

 is a tensor of rank five.

2.11 CONTRACTION  OF  A  TENSOR
The process of getting a tensor of lower order (reduced by 2) by putting a covariant index equal to a
contravariant index and performing the summation indicated is known as Contraction.

In  other  words,  if  in  a  tensor  we  put  one  contravariant  and  one  covariant  indices  equal,  the

process is called contraction of a tensor.

For example, consider a mixed tensor

ijk

lmA  of order five. Then by law of transformation,

ijk
lmA

=

i

x

p

x

j

q

x

x

k

x

r

x

s

l

x

x

t

x
m

x

Put the covariant index l = contravariant index i, so that

pqr
A
st

pqr
A
st

k

x

r

x

s

p

x

x

s

i

x

x

t
x
m

x

t

x
m

x

pqr
A
st

s
p

pqr
A
st

Since

s

p

x

x

d=

s
p

A

pqr
pt

t

x
m

x

t

x
m

x

ijk
imA

=

=

=

ijk
imA

=

i

x

p

x

j

x

q

x

j

x

q

x

j

x

q

x

j

q

k

r

k

r

k

r

x

x

x

x

x

x

x

x

This is law of transformation of tensor of rank 3. So,
imA  is a tensor of rank 3 and type (1, 2)
lmA  is a tensor of rank 5 and type (2, 3). It means that contraction reduces rank of tensor by

ijk

ijk

while
two.

Consider the tensors

2.12 INNER  PRODUCT  OF  TWO  TENSORS
kA  and
l =  then the result is

k

ij

l

ij
k BA

k
mn

putting
tensors.

mnB  if we first form their outer product

ij
k BA

l
mn

 and contract this by

  which  is  also  a  tensor,  called  the  inner  product  of  the  given

Hence  the  inner  product  of  two  tensors  is  obtained  by  first  taking  outer  product  and  then

contracting it.

EXAMPLE  7

If Ai and Bi are the components of a contravariant and covariant tensors of rank are respectively

then prove that AiBi is scalar or invariant.

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
d
�
�
�
�
�
�
�
�
�
�
�
�
�
�
Tensor Algebra

Solution

19

As Ai and Bi are the components of a contravariant and covariant tensor of rank one respectively,

then according to the law of the transformation

Multiplying these, we get

iA =

i BA

i

=

=

=

=

i BA
i

i

x

p

x

i

x

p

q

x
x

p

A

 and

iB  =

q

x

i

x

B
q

q

x

i

x

p
BA
q

,q

  since

q

x

p

x

d=

q
p

p
BA

p

x
q
p
BAd
p
q

p BA

p

This shows that AiBi is scalar or Invariant.

EXAMPLE  8

If

i

jA  is mixed tensor of rank 2 and

kl

mB  is mixed tensor of rank 3. Prove that

i
j BA

jl
m

 is a mixed

tensor of rank 3.

Solution

As

i

jA  is mixed tensor of rank 2 and

kl

mB  is mixed tensor of rank 3. Then by law of transformation

i
jA =

i

x

p

x

q

j

x

x

p
A
q

 and

kl
mB  =

k

x

r

x

l

s

x

x

t

x
m

x

rs
B
t

...(1)

...(2)

Put k = j then

Multiplying (1) & (2) we get

jl
mB =

i
j BA

jl
m

=

=

jl
i
j BA
m

=

j

x

r

x

i

x

p

x

i

x

p

x

i

x

p

x

l

s

x

x

q

j

x

x

l

x

s

x

l

x

s

x

t

x
m

x

B

rs
t

j

r

x

x

l

s

x

x

t

x
m

x

rs
p
BA
t
q

t

x
q
rm

x

rs
p
BA
t
q

since

q

x

j

x

j

r

x

x

 =

q

x

r

x

 =

q
r

t

x
m

x

p
BA
q

qs
t

since

q

r Bd

rs
t

 =

qs
tB

This is the law of transformation of a mixed tensor of rank three. Hence

i
j BA

jl
m

 is a mixed tensor

of rank three.

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
d
�
�
�
�
�
�
�
�
�
�
�
�
d
�
�
�
�
�
�
20

Tensors and Their Applications

2.13 SYMMETRIC  TENSORS

A  tensor  is  said  to  be  symmetric  with  respect  to  two  contravariant  (or  two  covariant)  indices  if  its
components remain unchanged on an interchange of the two indices.

EXAMPLE

(1) The tensor

(2) The tensor

ijA  is symmetric if
lmA  is symmetric if

ijk

ji

A =
ij
A
A =
ijk
lm

A

jik
lm

THEOREM  2.10  A  symmetric  tensor  of  rank  two  has  only

+nn
(

)1

1
2

  different  components  in  n-

dimensional space.
Proof: Let

ijA  be a symmetric tensor of rank two. So that

A =
ij

ji

A

.

The component of

ijA  are

11

A

21

A

31

A

M
1
n
A

12

A

22

A

32

A

M
n
A

2

13

A

23

A

33

A

L

L

L

1
n
A
2

n

A

3
n

A

MLM
3
n
A

A

nn

L

i.e., Aij will have n2 components. Out of these n2 components, n components A11, A22, A33, ..., Ann are
different. Thus remaining components are (n2� n). In which A12 = A21,  A23 = A32 etc. due to symmetry.

So,  the  remaining  different  components  are

2
n -

(

n

)

1
2

.  Hence  the  total  number  of  different

components

+

n

=

1
2

2

(

n

=

n

)

nn
(

+

)1

1
2

2.14 SKEW-SYMMETRIC  TENSOR

A tensor is said to be skew-symmetric with respect to two contravariant (or two covariant) indices if
its components change sign on interchange of the two indices.

EXAMPLE

(i) The tensor

(ii) The tensor

ijA  is Skew-symmetric of
lmA  is Skew-symmetric if

ijk

ij

A

-=

ji

A

ijk
A
lm

-=

jik
A
lm

THEOREM 2.11 A  Skew  symmetric  tensor  of  second  order has only

-nn
(

)1

1
2

  different  non-zeroo

components.

Proof: Let

ijA  be a skew-symmetric tensor of order two. Then

ij
A

-=

ji

A

.

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
-
Tensor Algebra

21

The components of

ijA  are

0
21

A

31

A

M
n
1
A

12

A

0
32

A

M
n
A

2

13

A

23

A

0

L

L

L

1
n
A
2

n

A

3

n

A

MLM
n
3
A
0

L

[
Since

ii
A

-=

ii
A

ii

2

A

(cid:222)=
0

ii
A

(cid:222)=
0

11

A

=

22

A

(cid:215)=

nn

A

=

]0

i.e., Aij will have  n2  components. Out of these  n2 components, n  components A11, A22, A33, ..., Ann
are zero. Omitting there, then the remaining components are n2 �n. In which A12 = � A21, A13 = � A31

etc. Ignoring the sign. Their remaining the different components are

Hence the total number of different non-zero components =

Note: Skew-symmetric tensor is also called anti-symmetric tensor.

1
2

(

1
2
-nn
(

2
n -

n

)

.

)1

THEOREM 2.12 A covariant or contravariant tensor of rank two say Aij can always be written as the
sum of a symmetric and skew-symmetric tensor.
Proof: Consider a covariant tensor Aij. We can write  Aij as

ijA =

ijA =

ijS =

where

Now,

+

(

A
ij

1
2
S +
ij T
ij
1
2

(

A +
ij A

+

A

ji

)

1
2

(

A
ij

A

)

ji

)

ji

 and

T
ij

=

1
2

(

A
ij

A

ji

)

jiS =

1
2

(

A +

ji A
ij

)

jiS = ijS

ijS  is symmetric tensor..

So,
and

1
2

(

A +
ij A

ji

)

(

A -

ji A
ij

)

ijT =

jiT =

1
2

(

A -
ij A

ji

)

=

jiT =

1
2
ijT-

ijT =

jiT-

So,

ijT  is Skew-symmetric Tensor..

or

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�

(cid:215)
(cid:215)
(cid:222)
-
-
-
22

EXAMPLE  9

Tensors and Their Applications

 If

AAa=f
jk

j

.k

 Show that we can always write

k
AAb=f
jk

j

 where

jkb  is symmetric.

Solution

As given

Interchange the indices i and j

f =

j

AAa

jk

k

Adding (1) and (2),

f =

k

AAa
jk

j

f2 =

(

a

jk

+

j

)
AAa
kj

k

...(1)

...(2)

1
2

(

a

jk

+

j

)
AAa
kj

k

j

AAb

jk

k

f =

f =

where

b

jk

=

1
2

(

a

jk

+

a

kj

)

To show that

jkb  is symmetric.

Since

jkb

=

kjb =

=

1
2
1
2
1
2

(

a +
jk

a

kj

)

(

a +
kj

a

)

jk

(

a +
jk

a

kj

)

kjb = jkb

So,

jkb  is Symmetric.

EXAMPLE  10

If

iT  be the component of a covariant vector show that

symmetric covariant tensor of rank two.

Solution

T
i
j

x

T

x

j

i

 are component of a Skew-

As

iT  is covariant vector. Then by the law of transformation

iT =

k

x

i

x

T
k

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
Tensor Algebra

Differentiating it w.r.t. to

jx  partially,,

T
i
j

x

T

x

j

j

T

j

Similarly,

=

=

=

k

x

i

x

T
k

j

x

� 2

k

x

j

x

i

x

� 2

k

x

j

x

i

x

+

T
k

+

T
k

� 2

k

x

+

T
k

j

=

i
xx
Interchanging the dummy indices k & l

x

i

T

x

j

i

=

� 2

k

x

i
xx

j

+

T
k

23

...(1)

...(2)

k

x

i

x

k

x

i

x

k

x

j

x

k

x

i

x

T

x

k
j

l

x

j

x

l

i

x

x

l

x

j

x

T

x

k
l

T
k
l
x

T
l
k

x

Substituting (1) and (2), we get

T
i
j

x

T

x

j

i

=

k

i

x

x

l

x

j

x

T
k
l
x

T
l
k

x

This is law of transformation of covariant tensor of rank two. So,

T
i
j

x

T

x

j
i

 are component of

a covariant tensor of rank two.

To show that

Let

T
i
j

x

T

x

j
i

 is Skew-symmetric tensor..

T
i
j

x
T

j
i
x

ijA =

jiA =

=

T

j
i

x
T
i
j

x

T
i
j

x

T

x

j
i

jiA =

ijA =

ijA-
jiA-

T

x

j
i

 is Skew-symmetric.

 are component of a Skew-symmetric covariant tensor of rank two.

or

So,

A
ij

=

So,

T
i
j

x

T
i
j

x
T

x

j
i

�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
-
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
�
�
-
�
�
�
�
-
�
�
�
�
-
�
�
�
�
-
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
-
�
�
-
�
�
�
�
-
�
�
24

2.15 QUOTIENT  LAW

Tensors and Their Applications

By this law, we can test a given quantity is a tensor or not. Suppose given quantity be A and we do not
know that A is a tensor or not. To test A, we take inner product of A with an arbitrary tensor, if this
inner product is a tensor then A is also a tensor.

Statement

If the inner product of a set of functions with an atbitrary tensor is a tensor then these set of

functions are the components of a tensor.

The proof of this law is given by the following examples.

EXAMPLE  11

Show that the expression  A(i,j,k) is a covariant tensor of rank three if  A(i,j,k)Bk    is  covariant

tensor of rank two and Bk is contravariant vector

Solution

Let X and Y be two coordinate systems.
As given A (i, j, k)Bk is covariant tensor of rank two then

,(
iA

kBkj
,
)

 =

p

x

i

x

q

j

x

x

,
BrqpA

),

(

r

...(1)

Since

kB  is contravariant vector. Then
k
x

kB =

r

B

r

x

or

B

r

=

r

x

k

x

k

B

So, from (1)

,(
iA

kBkj
,
)

=

,(
iA

kBkj
,
)

=

kjiA
),
,(

=

p

x

i

x

p

x

i

x

p

x

i

x

q

j

q

j

x

x

x

x

q

x

i

x

,
rqpA
),

(

r

k

x

x

k

B

r

k

x

x

r

k

x

x

,
BrqpA

),

(

k

rqpA
),
,

(

As

kB  is arbitrary..

So,

iA
,(

kj
),

 is covariant tensor of rank three.

EXAMPLE  12

If A (i, j, k)A iB jCk is a scalar for arbitrary vectors Ai, B j, Ck. Show that A(i, j, k) is a tensor of

type (1, 2).

Solution

Let X and Y be two coordinate systems. As given

,(
iA

i CBAkj
,

)

j

=

k

,(
iA
p CBArqpA
q
(
),

,

i CBAkj
,

)

j

 is scalar. Then

k

r

...(1)

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
Tensor Algebra

25

Since

i BA ,

i

 and

kC  are vectors. Then

iA =

jB =

kC =

p

A

q

B

i

x

p

j

q

x
x

x
k

r

C

x

r

x

or

pA =

or

or

qB =

rC =

p

x

i

x
q
x

j

r

k

x
x

x

i

A

j

B

k

C

So, from (1)

,(
iA

i
,
CBAkj

)

j

=

k

,(
rqpA
),

p

x

i

x

q

j

x

x

k

x

r

x

As

i

CBA

,

,

j

 are arbitrary..

k

i

j

CBA
k

Then

kjiA
),
,(

=

p

i

x
x

q

j

x
x

k

r

x
x

rqpA
),
,(

So, A(i, j, k) is tensor of type (1, 2).

2.16   CONJUGATE  (OR  RECIPROCAL)  SYMMETRIC  TENSOR

Consider a covariant symmetric tensor
d =

0�d

elements

ijA  i.e.,

 and

.

Now, define

ijA
ijA  by

ijA  of rank two. Let  d denote the determinant

ijA   with  the

ijA =

Cofactor

 of

A
ij

is

 the

determinan

t

A
ij

d

ijA  is a contravariant symmetric tensor of rank two which is called conjugate (or Reciprocal) tensor
of

ijA .

THEOREM 2.13 If

ijB  is the cofactor of
Bij
d

ijA =

Then prove that

kj

d=

ij AA

k
i

.

ijA  in the determinant d = |Aij| �

 0 and Aij defined as

Proof: From the properties of the determinants, we have two results.

(i)

BA ij
ij

=

d

A
ij

B
ij
d

= 1

ij
ij AA

= 1,

given

ij =

A

B
ij
d

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�

(cid:222)
Tensors and Their Applications

26

(ii)

ij BA

kj

0=

A
ij

B

kj
d

= 0,

0�d

kj
ij AA

= 0

if

i �

k

kj
ij AA

=

if1

=

i

k

if0

i

k

kj
ij AA

= k
i

from (i) & (ii)

i.e.,

2.17 RELATIVE  TENSOR

If the components of a tensor

ii
21

jjA ...

i
r
...
j

21

 transform according to the equation

s

kk
...
k
A
21
ll
...
l
21

s

r

=

x
x

ii
21

jjA ...

i
r
...
j

21

k

1

x

i
1

x

k

2

x

i

2

x

k

r

x

i
x

r

j
1

l

r

x

x

j
2

l

2

x

x

s

j
s

l

s

x

x

Hence

,1=w

21

ii
21

i
r
...
j

 is called a relative tensor of weight w

jjA ...
 the relative tensor is called a tensor density. If w = 0 then tensor is said to be absolute.

 is the Jacobian of transformation. If

, where

r

x
x

MISCELLANEOUS  EXAMPLES

1. Show  that  there  is  no  distinction  between  contravariant  and  covariant  vectors  when  we

restrict ourselves to transformation of the type

where a's and b's are constants such that

ix =

mi
xa
m

+

b

;i

i
r aa

i
m

= r
m

Solution

Given that

or

ix =
m

=

i
m xa

mi
xa
m
x -
i

b

+

i

b

i

Multiplying both sides (2) by
mi
xaa
m
mr
m x

i
r

=

,i
ra  we get
i
xa
=
r

i

i
ab

or

rx =

sx =

Differentiating Partially it w.r.t. to

s

x

i

x

= i
sa

 as given

 as

x =
mr
m

i
m

i
r aa
r
x

d=

r
m

...(1)

...(2)

..(3)

i

i

i

i
xa
r

i
xa
r

i
xa
s

i
r

i
r

i
r

i
ab

i
ab

i
ab

i
s

ix

(cid:238)
(cid:237)
(cid:236)
�
d
w
�
�

(cid:215)
(cid:215)
(cid:215)
�
�
�
�
�
�
�
�
(cid:215)
�
�
�
�
(cid:215)
(cid:215)
(cid:215)
�
�
d
-
d
-
-
d
-
�
�
Tensor Algebra

Now, from (1)

s

i
xa
s

+

i

b

i
sa

ix =
i

=

x

s

x

The contravariant transformation is

i

x

s

iA =

x
The covariant transformation is

s

A

 =

s

i
s Aa

iA =

s

i

x

x

A
s

 =

i
s Aa
s

27

...(4)

...(5)

...(6)

Thus from (5) and (6), it shows that there is no distinction between contravariant and
covariant tensor law of transformation

2.

If the tensors
satisfying the equations

ija  and

ijg  are symmetric and

i

i vu ,

 are components of contravariant vectors

(

a

ij

kg
ij

)

u

= 0,

, =
j

i

2,1

,...,
n

i

i

)

v

= 0,

k

.

k
.0=j

i
ij vua

(

a

ij

Prove that

i
ij vug

k'g

ij
,0=j

Solution

The equations are

(

a

ij

kg
ij

)

u

(

a

ij

)
vgk
ij

i

i

= 0

= 0

...(1)

...(2)

Multiplying (1) and (2) by u j and v j respectively and subtracting, we get

i
vua
ij

j

i
uva
ij

j

i
vukg
ij

j

�+

j

vugk
ij

i

0=

Interchanging i and j  in the second and fourth terms,

i

vua
ij

j

j

uva
ji

i

i

vukg
ij

j

�+

i
vugk
ji

j

0=

As

ija  and

ijg  is symmetric i.e.,

a =
ij

a

ji

 &

g =
ij

g

ji

kg
ij

i

j
uv

�+

i
vugk
ij

(

k

i

)
vugk
ij

j

j

= 0

= 0

Multiplying (1) by

j

uva
ij

i

i

j

vug
ij
jv , we get
i
j

vukg
ij

= 0

= 0 since

k

k

k

k

0�

i
ij vua

j

= 0 as

i

vug
ij

j

 = 0.

Proved.

�
�
�
�
�
�
-

-
�
�
-
�
-
-
-
-
-
-
-
�
�
-
(cid:222)
�
�
-
28

Tensors and Their Applications

3.

If

ijA  is a Skew-Symmetric tensor prove that

(

i
j

k
l

d+

i
l

k
j

A)
ik

= 0

Solution

Given

ijA  is a Skew-symmetric tensor then

A
ij

-=

A

ji

.

Now,

(

i
j

k
l

d+

(

i
j

k
l

d+

i
l

i
l

k
j

A)
ik

=

=

=

i
j

k
l

A
ik

d+

i
l

k
j

A
ik

d+

i
l

A
ij

i
A
il
j
A +

jl A
lj

k
j

A)
ik

=  0

as

A

jl

-=

A
lj

4.

If

ija

ija  is symmetric tensor and
0=

 or

0=

.

kb

ib  is a vector and

ijba

k

+

jkba
i

+

kiba

j

0=

 then prove that

Solution

The equation is

ba
k
ij

+

ba
jk

i

+

ba
ki

j

= 0

ba
k
ij

+

ba
i
jk

+

ba
ki

j

= 0

By tensor law of transformation, we have

a

pq

p

x

i

x

q

j

x

x

b

r

r

x

k

x

+

a

pq

p

j

x

x

q

x

k

x

b
r

r

i

x

x

+

a

pq

p

k

x

x

q

i

x

x

b

r

r

x

j

x

= 0

pqba

r

p

x

i

x

q

j

x

x

+

r

x

k

x

p

j

x

x

q

x

k

x

+

r

i

x

x

p

k

x

x

q

x

i

x

r

j

x

x

= 0

pqba

r

0=

pqa

ija

0=
0=

 or

 or

0=
0=

rb
kb

a
5. If

mn
+

a

mn
If  mna

nm
xx

=

=

a

nm

nm
xxb
mn
+

b
nm

b
mn

 for arbitrary values of

rx , show that a(mn) = b(mn) i.e.,

 and  mnb

 are symmetric tensors then further show the

a

mn

=

b
mn

.

Solution

Given

a

mn

nm
xx

nm
xxb
mn

=

(

a

mn

b
mn

nm
xx

)

= 0

d
d
d
d
d
d
d
d
d
d
d
d
d
(cid:222)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:222)
(cid:222)
(cid:222)
-
Tensor Algebra

29

Differentiating w.r.t.

ix  partially

(

a

in

n

)

x

+

(

a

b
in

mi

b

mi

m

)

x

Differentiating again w.r.t.

= 0
jx  partially

b
(

=

mn

)

a

nm

b

,

mn

=

b
nm

.

(

a

ij

b
ij

)

+

(

)

ji

= 0

b
a
ji
a +
ij

a

ji

or

a

mn

+

a

nm

=

=

ji

b +
b
ij
+

b

mn

b

 or

a

nm

(

mn

)

=

Also, since  mna
So,

 and  mnb

 are symmetric then

a

mn

mna2
mna

mnb2
=
= mnb

EXERCISES

1. Write down the law of transformation for the tensors

(i)

ijA

(ii)

ij
kB

(iii)

ijk
lmC

2. If

pq

rA  and

s

tB  are tensors then prove that

s
pq
r BA
t

 is also a tensor..

3.

If Aij is a contravariant tensor and Bi is covariant vector then prove that
three and Aij Bj is a tensor of rank one.

ijBA

k

 is a tensor of rank

4. If Ai is an arbitrary contravariant vector and Cij Ai Aj is an invariant show that Cij + Cji is a covariant

tensor of the second order.

5. Show that every tensor can be expressed in the terms of symmetric and skew-symmetric tensor.

6. Prove that in n-dimensional space, symmetric and skew-symmetric tensor have

independent components respectively.

n
2

+n
(

)1

 and

-n
(

)1

n
2

7. If

ijU

0�

 are components of a tensor of the type (0, 2) and if the equation

fU

ij

+

gU

0=

ji

 holds

w.r.t to a basis then prove that either f = g and

ijU  is skew-symmetric or f = �g and

ijU  is symmetric.

8. If

ijA  is skew-symmetric then

(

k
i
BB
l
j

+

k
ABB
ik
j

i
l

)

=

0

.

9. Explain the process of contraction of tensors. Show that

ij
ijaa

d=

i
j

.

-
-
-
-
30

Tensors and Their Applications

10. If

pq

rA  is a tensor of rank three. Show that

pr

rA  is a contravariant tensor of rank one.

11. If

ij
ka

gml
i
j

k

 is a scalar or invariant,

gml
,

,

j

i

k

 are vectors then

ij
ka  is a mixed tensor of type (2, 1).

12. Show that if

hijka

hih
mlml

0=k

 where

i

 and

i

 are components of two arbitrary vectors then

a

hijk

+

a

hkji

+

a

jihk

+

a

jkhi

 = 0

13. Prove that AijBiC j is invariant if Bi and C j are vector and Aij is tensor of rank two.
14. If A(r, s, t) be a function of the coordinates in n-dimensional space such that for an arbitrary vector
Br of the type indicated by the index a A(r, s, t)Br is equal to the component Cst of a contravariant

tensor of order two. Prove that A(r, s, t) are the components of a tensor of the form

st
rA .

15. If Aij and Aij are components of symmetric relative tensors of weight w. show that

ij

A

=

ij
A

w

2-

x
x

  and

A
ij

=

A
ij

w

2+

x
x

16. Prove that the scalar product of a relative covariant vector of weight w  and a relative contravariant

vector of weight  w�  is a relative scalar of weight

�+
ww

.

l
m
�
�

�
�
CHAPTER � 3

METRIC  TENSOR  AND  RIEMANNIAN  METRIC

3.1 THE  METRIC  TENSOR

In  rectangular  cartesian  coordinates,  the  distance  between  two  neighbouring  point  are  (x,  y,  z)  and
2

2

2

2

dz

)

 is given by

ds

=

dx

+

dy

+

dz

.

+

(

x

ydx
,

+

zdy
,

+

In n-dimensional space, Riemann defined the distance  ds between two neighbouring points

ix

and

i

x

+

i

dx

(

i

=

n
,...2,1

)

 by quadratic differential form

2ds =

g

11

(

dx

21
)

+

g

12

dx

1
dx

2

(cid:215)+

+(cid:215)

g

1
n

1
dx

dx

n

2

+

1
dx

+

(

)

12

g

dx

g
+    .  .  .  .  .  .  .  .  ..  .  .  .  .  .  .  .  .    .    .  .  .  .  .  .  .   +

dx

22

(

22
)

+(cid:215)

(cid:215)+

g

2

n

dx

2

n

dx

+

g

n
1

dx

n

1
dx

+

g

n

2

dx

n

dx

2

(cid:215)+

+(cid:215)

g

nn

(

dx

n

2

)

2ds =

dxg
ij

j

i

dx

ji =
,(

n
)
,...2,1

...(1)

using summation convention.

Where

ijg are the functions of the coordinates

ix  such that

g =

ijg

0�

The quadratic differential form (1) is called the Riemannian Metric or Metric or line element for n-
nV   and

dimensional space and such n-dimensional space is called Riemannian space and denoted by
ijg  is called Metric Tensor or Fundamental tensor..

The geometry based on Riemannian Metric is called the Riemannian Geometry.

THEOREM 3.1 The Metric tensor

ijg  is a covariant symmetry tensor of rank two.

Proof: The metric is given by

2ds =

dxg
ij

j

i

dx

...(1)

(cid:215)
(cid:215)
(cid:215)
(cid:215)
32

Tensors and Their Applications

Let  xi  be  the  coordinates  in  X-coordinate  system  and

system. Then metric ds2 = gij dxidxj transforms to

ds =2

Since distance being scalar quantity.

xdxdg

ij

.

ix   be  the  coordinates  in  Y-coordinate
i

j

So,

2ds =

dxg
ij

j

i

dx

=

i
xdxdg
ij

j

...(2)

The theorem will be proved in three steps.
(i) To show that  dxi is a contravariant vector.

If

ix =

2
1
i
xxx
(
,

x
,...

n

)

ixd

=

ixd

=

i

i

x

x

1
dx

+

i

x
2

x

2

dx

(cid:215)+

+(cid:215)

i

n

x

x

n

dx

i

x
k

x

k

dx

It is law of transformation of contravariant vector. So,

idx  is contravariant vector..

(ii) To show that

ijg  is a covariant tensor of rank two. Since

from equation (2)

ixd

=

i

x
k

x

k

dx

 and

jxd

 =

j

x �
x

l

l

x

dxg
ij

dxg
ij

j

j

i

i

dx

dx

=

g

ij

=

g

ij

dx

l

k

dx

=

g

ij

g

kl

i

k

x

x

i

x

k

i

x
x
k

x

j

x

l

x

l

dx

k

x

l

dx

dx

k

l

dx

k
dx

j

l

j

x

x
x

l

x

Since

dxg
ij

j

i

dx

=

g

kl

dx

k

l

dx

 (i, j are dummy indices).

g

kl

g

ij

or

or

i

x
k

x

j

x

l

x

dx

k

l

dx

= 0

g

kl

g

ij

i

x
k

x

j

x

l

x

= 0 as

kdx  and

ldx  are arbitrary..

klg =

g

ij

ijg =

g

kl

j

k

x

x

k

x

i

x

j

x

l

x

l

x

j

x

So,

ijg  is covariant tensor of rank two.

�
�
(cid:215)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
-
�
�
�
�
-
�
�
�
�
�
�
�
�
Metric Tensor and Riemannian Metric

(iii) To show that

ijg  is symmetric. Then

ijg  can be written as

ijg =

ijg =

ijA =

ijB =

+

)

g

ji

1
2

(

g

ij

g

)

ji

(

ij

+

g

1
2
A +
ij B
ij

g +
ij

(

g

)

ji

 = symmetric

(

g -
ij

g

)

ji

 = Skew-symmetric

1
2
1
2

where

Now,

Interchanging the dummy indices in

dxB
ij

=

(

A
ij

+

B
ij

)

i
dx

dx

j

 from (3)

dxg
ij

i

dx

(

g

ij

A
ij

)

dx

i

dx

j

j

=

i
dxB
ij

dx

j

i

j

dx

, we have

j

i
dxB
ij

dx

i
dxB
ij

dx

j

j

=

=

i

dx

dxB
ji
dxB-
ij

j

i

dx

33

(4)

Since

ijB  is Skew-symmetric i.e.,

B
ij

-=

B

ji

dxB
ij

j

i

dx

+

dxB
ij

dxB2
ij

j

j

i

i

dx

dx

= 0

= 0

i
dxB
ij

dx

j

= 0

So, from (4),

(

g

ij

A
ij

)

dx

j

i

dx

= 0

ijg = ijA  as

dx ,

i dx

j

 are arbitrary..

So,

ijg  is symmetric since

ijA  is symmetric. Hence

ijg  is a covariant symmetric tensor of rank

two. This is called fundamental Covariant Tensor.

THEOREM 3.2 To show that

dxg
ij

j

i

dx

 is an invariant.

Proof: Let
in Y-coordinate system.

ix  be coordinates of a point in X-coordinate system and

ix  be coordinates of a same  point

Since

ijg  is a Covariant tensor of rank two.

Then,

ijg =

g

kl

k

x

i

x

1

x

j

x

-
-
(cid:222)
-
(cid:222)
�
�
�
�
34

Tensors and Their Applications

g

ij

g

kl

k

x

i

x

l

x

j

x

= 0

g

ij

g

kl

k

x

i

x

l

x

j

x

i

dx

dx

j

= 0

(

dxg
ij

i

dx

j

)

=

g

kl

=

g

kl

k

x

i

x

k

x

i

x

l

x

j

x

i

dx

dx

j

i

dx

l

x

j

x

j

dx

dxg
ij

j

i

dx

=

g

kl

dx

l

k

dx

So,

dxg
ij

j

i

dx

 is an ivariant.

3.2 CONJUGATE  METRIC  TENSOR:  (CONTRAVARIANT  TENSOR)

The conjugate Metric Tensor to

ijg , which is written as

ijg , is defined by

ijg =

Bij
g

 (by Art.2.16, Chapter 2)

where

ijB  is the cofactor of

ijg  in the determinant

=

g

ijg

0�

.

By theorem on page 26

So,

kj
ij AA

= k
i

kj

ij gg

= k
i

Note

(i) Tensors g ij  and g ij are Metric Tensor or Fundamental Tensors.
(ii)

g ij is called first fundamental Tensor and g ij

 second fundamental Tensors.

EXAMPLE  1

Find the Metric and component of first and second fundamental tensor is cylindrical coordinates.

Solution

Let (x1,  x2, x3) be the Cartesian coordinates and

(

2
1
xxx
,

,

3

)

 be the cylindrical coordinates of a

point. The cylindrical coordinates are given by

x =

r

cosq

,

y

= r

sin q

,

z =

z

So that

1
x

=

,
xx

2

=

3

,
xy

=

z

 and

1

x

=

,
xr

2

q=

3

=

,

x

z

...(1)

Let

ijg   and

ijg   be  the  metric  tensors  in  Cartesian  coordinates  and  cylindrical  coordinates

respectively.

(cid:222)
�
�
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
-
�
�
�
�
�
�
�
�
d
d

35

...(2)

...(3)

Metric Tensor and Riemannian Metric

The metric in Cartesian coordinate is given by

2ds =

dx

2

+

2

+

dy

2

dz

2ds =

dx
(

21
)

+

dx
(

22
)

+

dx
(

23
)

But

2ds =

dxg
ij

j

i

dx

=

g
dx
(
11

21
)

+

g
12

1
2
dx
dx

+

g

13

dx

1

dx

3

+

g

dx

2

dx

1

21

+

g
22

(dx

22
)

g+

23

dx

2

dx

3

g+

dx

31

1

3

dx

Comparing (2) and (3), we have
=

=

=

=

1

 and

g

g

g

g

g

12

13

11

22
On transformation

33

g+

32

dx

3

dx

2

g+

33

(dx

33
)

=

g

21

=

g

23

=

g

31

=

g

32

=

0

g

ij

=

g

ij

i

i

x

x

j

j

x

x

for i = j = 1.

,

 since

ijg  is Covariant Tensor of rank two. (i, j = 1, 2, 3)

11g

=

g

11

2

1
x

1

x

+

g

22

2

x

1

x

+

g

33

2

3

x

1
x

since

g

=

g

13

12

(cid:215)=

=(cid:215)

g

32

=

0

.

11g

=

g

11

2

x
r

+

g

22

2

y
r

+

g

33

2

z
r

Since

x

= cos
r
y
r

,

,

y

= r

=

sin q

,

sin q
z
r

,

z =

z

0=

cosq

and

g

11

=

g

22

=

g

33

=

=

x
r
1

.

Put i = j = 2.

2

+q

sin

2

+q

0

cos

11g =
11g = 1

22g

=

g

11

2

+(cid:247)

g

22

1

2

x

x

2

+(cid:247)

g

33

2

2

x

x

22g

=

g

11

2
+(cid:247)

g

22

x
q

2
+(cid:247)

g

33

y
q

z
q

2

3

2

x

x

2

�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:215)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
q

�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
36

Tensors and Their Applications

Since

g

11

=

g

22

=

g

33

=

1

x

-=

sin q

r

,

y

=

cosq

r

,

� z

0=

22g

= (

r

sin

)

2

(

+

r

cos

)

2

+

0

2

r

sin

2

+q

2

r

2

cos

=

22g

= 2r

33g

=

g

11

2

+(cid:247)

g

22

1
x

3

x

2

3

x

x

+(cid:247)

g

33

3

3

x

x

=

g

11

2
+(cid:247)

x
z

g

22

y
z

+(cid:247)

g

33

z
z

,0=

y
z
g =
22

z
z

1=

. So,

=g
33

1

.

2

,

r

=g
33

1

Put i = j = 3.

Since

So,

,0=

x
z
=g
11

,1

g

and

=

g

=

g

=

g

=

g

=

g

=

0

12

21
(i) The metric in cylindrical coordinates

32

23

13

31

2ds =

2ds =

i

ij

xdxdg
)
(
21
xd

g

11

since

g

12

j

i

+

=

, =j
.3,2,1
)
(
22
xd

g

22

(
xd

)23

+

g

33

g

13

(cid:215)=

=(cid:215)

g

32

=

0

(ii) The first fundamental tensor is

2ds =

2

dr

+

2
dr
(

q

)

2

+

2

d

f

ijg =

g

g

g

11

21

31

g

12
g

22

g

13
g

23

001

=

0

2

r

0

g

32

g

33

100

since

1

0

0

0
2r
0

0

0

1

g =

=

g ij

g = 2r

q
�
�

q
�
�

q
�
q
q
-
q
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�

�
�

�
�

(cid:215)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
Metric Tensor and Riemannian Metric

37

(iii) The cofactor of g are given by
B =
11
=
B

=B
22
=
B

,1
=

,2
B

B =
33
=
B

r
=

21

13

23

31

B
12

2

r
B

32

=

0

and

The second fundamental tensor or conjugate tensor is

ij =

g

B
ij
g

.

g in
11

g

 of
g

=

1

cofactor

11g

=

11g

=

22g

=

B

11
g

=

2

2

r

r

B =
12
g

1
2

r

33g

=

B

33
g

=

2

2

r

r

=

1

and

12

g

=

13

g

=

g

21

=

g

23

=

g

31

=

32

g

=

0

001

Hence the second fundamental tensor in matrix form is

0

0

1
2r
100

.

EXAMPLE  2

Find the matrix and component of first and second fundamental tensors in spherical coordinates.

Solution

Let

1
x

(

,

2

x

,

x

3

)

 be the cartesian coordinates and

(

x

1

,

2

x

,

x

3

)

 be the spherical coordinates of a

point. The spherical coordinates are given by
,

= r

cos

sin

x

y

= r

sin

sin

,

= cos
r

z

So that

1
x

=

,
xx

2

=

,
xy

3

=

z

 and

1

x

=

2

xr
,

=

3

q

,

x

=

f

Let

ijg  and

ijg  be the metric tensors in cartesian and spherical coordinates respectively..

The metric in cartesian coordinates is given by
+

2

2

dx

2ds =
2ds = (

dx

+
)
21

2

+

dz
)
22

+

dy
(
dx
  (
i

;j

dx

, =j

)23

(
dx
)3,2,1

But

2ds =

i

dxg
ij

�
�
�
�
�
�
�
�
�
�
�
�
f
q

f
q

q
38

Tensors and Their Applications

g

=

g

=

g

11

33
On transformation

22

=

1

 and

g

12

=

g

23

=

g

13

=

g

21

=

g

31

=

g

32

=

0

ijg =

g

ij

i

i

x

x

j

j

x

x

(since

ijg  is covariant tensor of rank two) (where i, j = 1,2,3).

ijg =

g

11

1
x

i

x

1

x

j

x

+

g

22

2

x

i

x

2

j

x

x

+

g

33

3

i

x

x

3

1

x

x

since i, j are dummy indices.

  Put i = j =  1

11g =

g

11

2

+(cid:247)

g

22

1

1

x

x

2

1

x

x

2

+

g

33

11g =

g

11

2
+(cid:247)

g

22

x
r

y
r

2

+

g

33

Since

x

= r

sin

cos

,

y =

r

sin

q

sin

f

,

= cos
r

z

x
r

=

sin

cos

,

=

sin

sin

,

y
r

g

11

=

g

22

=

g

33

=

1

.

and
So,

2

3

1

x

x

2

=

cos

z
r

z
r

11g = (

sin

cos

)

2

+

(
sin

sin

)

2

+

2

cos

put i = j = 2

11g = 1

since

g

=

g

22

=

g

33

=

1

11

22g

=

g

11

2

+(cid:247)

g

22

1

2

x

x

2

2

x

x

+(cid:247)

g

33

2

3

2

x

x

22g

=

g

11

x

2
+(cid:247)

g

22

y

2
+(cid:247)

g

33

2

z

x

=

r

cos

cos

,

y

=

r

cos

sin

,

z

-=

sinr

= (
r

22g

cos

cos

)

2

+

(
r

cos

sin

)

2

(
-+

r

sin

)2

22g

= 2r

(cid:222)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
f
q

q
�
�
f
q

f
q
�
�

q
�
�
q
f
q
f
q
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
q
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
q
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
q
�
�

f
q
q
�
�

f
q
q
�
�

q
q
�
�
q
f
q
f
q
Metric Tensor and Riemannian Metric

39

Put i = j = 3

since

g

11

=

g

22

=

g

33

=

1

and

So, we have

and

33g

=

g

11

2

+(cid:247)

g

22

1
x

3

x

2

+(cid:247)

g

33

2

3

x

x

2

3

3

x

x

33g

=

g

11

x

2

+(cid:247)

g

22

y

2

+(cid:247)

g

33

2

z

x

-=

r

sin

sin

,

y

=

r

sin

cos

,

z

=

0

= (

33g

r

sin

sin

)

2

+

(
r

sin

cos

)

2

+

0

33g

=

r

2 sin

q2

=g
11

,1

g =
22

r

,2

g

33

=

2

r

sin

2

g

12

=

g

13

=

g

21

=

g

23

=

g

31

=

g

32

=

0

(i) The Metric in spherical coordinates is

2ds =

2ds =

i

ij

xdxdg
)
(
21
xd

g

11

;j

i

+

g

, =j
(
xd

3,2,1
)
22

22

+

(
xd

)23

g

33

(ii) The Metric tensor or first fundamental tensor is

2ds =

2

+

dr

2
dr

2

+

2

r

2

sin

2

d

ijg =

g

g

g

11

21

31

g

12

g

22
g

32

g

g

13

23

g

33

0
2

r

01

=

0

2

r

0

0

00

2

r

sin

2

0

0

=

4

r

sin

2

q

and

g =

g ij

=

1

0

0

(iii) The  cofactor  of  g  are  given  by

=
B
31 B
13

=

B

23

= B

32

=

0

0
=B
11

2

r

,1

2

q
sin
B =
22

r

,2

B

33

=

2

r

sin

2

  and

=
B
12 B

21

=

The second fundamental tensor or conjugate tensor is

ij =

g

B
ij
g

.

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
f
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
f
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
f
�
�
f
�
�
f
q
f
�
�
f
q
f
�
�
f
q
f
q
-

q

f
q
q
�
�
�
�
�
�
�
�
�
�
q
�
�
�
�
�
�
�
�
�
�

q

40

Tensors and Their Applications

cofactor

11g

=

g

11 in

g

=

 of
g

B
11
g

=

4

4

r

r

2

2

sin

sin

11g

= 1

22g

=

B

22
g

=

2

4

r

r

sin

sin

2

2

22g

1
= 2
r

33g

=

33g

=

B

33
g

=

2

r

4

r

sin

2

1
2 sin

r

q2

=

13

g

12

g

and
Hence the fundamental tensor in matrix form is

31

g

=

g

32

=

0

21

=

=

g

ijg =

11

21

31

g

g
g

12

22

32

g

g
g

13

23

33

g

g
g

=

0

0

0

0
1
2

r

0

0

0

1

2

r

sin

q2

EXAMPLE  3

If the metric is given by

2ds = (

5

dx

)
21

(
dx

) +
22

(
4 dx

) -23

+

3

6

dx

1
dx

2

+

4

dx

3

2

dx

Evaluate (i) g  and (ii)

ijg .

Solution

The metric is

2ds  =

dxg
ij

i

dx

;j

2ds =
+

i
,(

g

11

=j
(
dx

)3,2,1
)
21

+

g

12

dx

1
dx

2

+

1
dx

dx

3

+

g

g

13

dx

21

2

1
dx

g

22

(

dx

22
)

+

g

2

dx

dx

3

+

g

dx

3

1
dx

+

g

dx

3

dx

2

+

g

(

dx

23
)

33

32

31

23

Since

ijg  is symmetric

g =
ij

g

ji

i.e.,

12g

,21g

g

23

=

=

g

32

,

g

13

=

g

31

q
q
q
q
q
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:222)

Metric Tensor and Riemannian Metric

So,

2ds =

g

11

dx

21
)

(

+

g

22

(

dx

22
)

+

g

33

(

dx

23
)

+

2

g

12

1
dx

dx

2

Now, the given metric is

+

2

g

23

2

dx

dx

3

+

2

g

13

1
dx

dx

3

41

...(1)

2ds =

(5

dx

21
)

+

(3

dx

22
)

+

(4

dx

23
)

6

1
dx

dx

2

+

4

dx

3

2

dx

...(2)

Comparing (1) and (2) we have

11g  =

,5

g

22

=

,3

g

33

=

2,4

g

12

-=

6

g

12

-=

=

3

g

21

2

g

23

(cid:222)=
4

g

23

==
2

g

,

g

13

32

==
0

g

31

g =

g ij

=

g

g

g

11

21

31

(ii) Let

ijB  be the cofactor of

ijg in g.

Then

g

12
g

22

g

13
g

23

g

32

g

33

=

5

3

0

03

23

=

4

42

11B =

Cofactor

 of

=g
11

22B =

Cofactor

 of

=g
22

33B =

Cofactor

 of

=g
33

12B =

Cofactor

 of

g

-=

12

13B =

Cofactor

 of

g

=

13

23

42

05

40

=

8

=

20

5

3

3

3

=

6

23

40

33

20

=

12

=

B

21

-=

=

6

B

31

23B =

Cofactor

 of

g

23

=

�

5

0

3

2

-=

=

10

B

32

Since

gij  =

Bij
g

We have
B
11
g

=

g

11

=

8
4

=

;2

22 =

g

,5

33 =g

,

3
2

12

g

= g

21

=

,3

13

g

= g

31

-=

3
2

,

23

g

= g

32

-=

5
2

-
(cid:222)
-
-
-
-
-
-
-

42

Hence,

Tensors and Their Applications

ijg =

2

3

3
2

3

5

5
2

3
2
5
2
3
2

3.3 LENGTH  OF  A  CURVE

Consider  a  continuous  curve  in  a  Riemannian
current point on it are expressible as functions of some parameter, say t.

nV   i.e.,  a  curve  such  that  the  coordinate

ix   of  any

The equation of such curve can be expressed as

ix =

)(txi

The length ds of the arc between the points whose coordinate s are

ix  and

x +
i

i

dx

 given by

2ds =

dxg
ij

j

i

dx

If s be arc length of the curve between the points

1P  and

2P  on the curve which correspond to

the two values  1t  and  2t

 of the parameter t.

s = (cid:242)

P
2

P
1

=

ds

t

2

t

1

g

ij

i

dx
dt

j

dx
dt

21

dt

NULL  CURVE

If

g

ij

i

dx
dt

j

dx
dt

0=

 along a curve. Then s = 0. Then the points

1P  and

2P  are at zero distance, despite

of the fact that they are not coincident. Such a curve is called minimal curve or null curve.

EXAMPLE  4

A curve is in spherical coordinate xi is given by

1x = ,t

2

x

=

sin 1

1
t

 and

3

x

=

2

2

t

1

Find length of arc 1 �

 t � 2.

Solution

In spherical coordinate, the metric is given by

2ds =

dx

21
)

(

+

21
x
()

(

dx

22
)

+

1
x

(

sin

x

22
()

dx

23
)

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
-
-
-
-
(cid:242)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
-
-
Metric Tensor and Riemannian Metric

43

given

1x = ,t

2

x

=

sin 1

3

x

=

2

2

t

1

1dx = dt ,

dx

2

=

1
2

t

dt

,

3

dx

(cid:215)=
2

(
t

2

1
2

)
1

21

2

t

dt

,

1
t

1

1

2

1
t

2dx =

dt
12 -

t

,

t

3

dx

=

2

t

2

t

1

dt

2ds = (

dt

)

2

+

2

t

2

dt

t

2

t

1

+

t

sin

sin

2

1

1
t

2

2

t

2

t

1

dt

2ds =

2

+

dt

2

dt
2

t

+

1

t

2

t

4
2

1

(

dt

)2

2ds =

2

t
5
2

t

1

2

dt

ds =

5

t
2 -

t

dt

1

Now, the length of arc,

1

t

,2

 is

t

2

t
1

ds =

5

2

1

t

2

t

=

dt

1

5
2

1

2

t
21

2

1

=  15  units

3.4 ASSOCIATED  TENSOR

A tensor obtained by the process of inner product of any tensor

tensor

ijg  or

ijg  is called associated tensor of given tensor..

ii
21

jjA ...

i
...

21

r
j

s

 with either of the fundamental

e.g. Consider a tensor

ijkA  and form the following inner product

i Ag
ijk

=

jkA
;

j
Ag

ijk

=

k

AgA

;

ik

ijk

=

A
ij

All these tensors are called Associated tensor of

ijkA .

Associated  Vector

Consider  a  covariant  vector

contravariant vector

jA . Then

iA .  Then
=

Ag
jk

j

ik
Ag
i

=

k

A

  is  called  associated  vector  of

iA .  Consider  a

A
k

 is called associated vector of

jA .

-

-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
-

-
-
-

-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
-
�
�
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
-
-
-
-
-
-
�
�
(cid:242)
(cid:242)
�
�
�
�
�
�
�
�
-
-
a
a
a
a
a
a
44

Tensors and Their Applications

3.5 MAGNITUDE  OF  VECTOR

The magnitude or length A of contravariant vector

.iA  Then A  is defined by

or

A =

i
ij AAg

j

2A =

i
ij AAg

j

Also,

A =2

j AA

j

 as

i
Ag
ij

=

A

j

i.e., square of the magnitude is equal to scalar product of the vector and its associate.

The magnitude or length A of covariant vector

iA . Then A  is defined by

or

A =

ij AAg
i

j

2A =

ij AAg
i

j

A vector of magnitude one is called Unit vector. A vector of magnitude zero is called zero vector

or Null vector.

3.6 SCALAR  PRODUCT  OF  TWO  VECTORS
r
Let  A

r  be two vectors. Their scalar product is written as
=

 and  B

rr (cid:215)
BA

i BA
i

rr (cid:215)
BA

 and defined by

Also,

Thus

i.e.,

rr (cid:215)
BA

=

i
BA

rr (cid:215)
BA

=

BA
i

i

i

=

i
BAg

ij

j

 since

B =
i

Bg
ij

=

ij

BAg
i

j

 since

i
B =

ij
Bg

j

j

rr (cid:215)
AA

=

A =

i
AA
i
A =r

=

i
AAg

ij

j

=

2A

i
ij AAg

j

r
 and  B

Angle  between  two  vectors
r
 be two vectors. Then
Let  A
rr (cid:215)
BA

rr
cosBA

=

cos =

rr
BA
rr
BA

=

i
BAg

ij

j

i
AAg

ij

j

i
BBg

ij

j

r
A

=

i
ij AAg

j

r
;  B

 =

i
ij BBg

j

since

This is required formula for

cos

.

Definition

r
The inner product of two contravariant vectors  A

r
iA  and  B

)

(

or

(
or

iB  associated with a symmetric

)

tensor

ijg  is defined as

i
ij BAg

j

. It is denoted by
rr
( BAg
,

i
ij BAg

=

)

j

q
(cid:222)
q
(cid:215)
q
Metric Tensor and Riemannian Metric

THEOREM 3.3 The necessary and sufficient condition that the two vectors  A
if

rr
=BAg
,
(

0

)

r
r  and  B

Proof: Let  q

r
 be angle between the vectors  A

r
 and  B

 then

or

rr (cid:215)
BA
rr (cid:215)
BA

i
ij BAg

j

cos

=

=

=

=

rr
cosBA

AB

cos

AB

cos

i
BAg

ij

j

AB

r
If  A

r
 and  B

 are orthogonal then

=q

cos

=q

0

 then from (1)

2

= 0

= 0 since

)BAg
(

rr
,

 =

i
ij BAg

j

j

i
ij BAg
)BAg
(
0=j

rr
,

Conversely if

i
ij BAg

 then from (1)

r
So, two vectors  A

r
 &  B

cos

=

0

=q

.

2

 are orthogonal.

Note:  (i) If  Ar  and  Br

 be unit vectors. Then

cos

=

r
r =
B
A
rr (cid:215)
BA

= 1. Then

i
ij BAg

j

=

45

 at 0 be orthogonal

...(1)

Proved.

(ii) Two vectors Ar  and  Br  will be orthogonal if angle between them is

=q

 i.e.,

2

 then

2

cos

=

cos

=q

= 0

2

3.7 ANGLE  BETWEEN  TWO  VECTORS

THEOREM 3.4 To show that the definition of the angle between two vectors is consistent with the
requirement cos 2q �

 1.

To justify the definition of the angle between two vectors.

OR

OR

To  show  that  the  angle  between  the  contravariant  vectors  is  real  when  the  Riemannian  Metric  is
positive definition.
Proof: Let q

r  then
r  and  B
 be the angle between unit vectors  A
=
=
BA
j

i
BAg

ij

j

j

j

ij
BBA
i

cos

=

=

ij

BAg
i

j

=

i
BA
i

q
q
q
(cid:222)
q
(cid:222)
p
(cid:222)
q
p
(cid:222)
q
p
p
q
p
q
46

Tensors and Their Applications

To show that q
Consider the vector

 is real i.e., |cosq
lA +
i

i mB

| �

 1.

 when  l  and  m  are scalars. The square of the magnitude of

lA +

i mB

i

=

g

ij

i
lA

(

+

i

mB

()

lA

j

+

mB

j

)

2
AAlg

i

ij

j

+

g

ij

lmA

j

i

B

+

m

lg

ij

i
AB

j

+

=

Since

and

=

l

2

+

2

lm

cos

+q

2

m

i
ij AAg

j

=

2 =A

;1

i
BBg

ij

j

= B

2 =

.1

2

BBgm
ij

i

j

i
ij BAg

j

=

cosq

;

 as  A

r  &  B

r  are unit vector i.e.,

r
A

(cid:222)=
1

2 =

A

1

.

.0�
Since square of magnitude of any vector
+
i mB
i

So, the square of the magnitude of
� 0
or
(
+
ml

lm
2
2
mm

cos
2

� 0

m
2

l
)
2

cos

cos

+q

lA

+

+

2

2

.0�

+
ml

(

cos

cos
This inequality holds for the real values of l & m.

m

1(

)

)

� 0

2

+

2

2

if

1

2

cos

cos

q2

cos

� 0

� 1
� 1

THEOREM 3.5 The magnitude of two associated vectors are equal.
Proof: Let A and B be magnitudes of associate vectors Ai and Ai respectively. Then
i
ij AAg

2A =

j

and

From equation (1)

2B =

ij AAg
i

j

2A =

i
AAg
ij

)

(

j

2A =

j

j AA

since

Ag
ij

i

=

j

A

 (Associate vector)

From equation (2)

2B =

(

ij

AAg
i

)

j

2B =

j AA

j

Proved.

...(1)

...(2)

...(3)

...(4)

q
-
q
q
-
q
q
-
(cid:222)
q
Metric Tensor and Riemannian Metric

47

since

ij
Ag
i

=

j

A

from (3) and (4)

So, magnitude of

iA  and

2B

2A =
A = B
iA  are equal.

3.8 ANGLE  BETWEEN  TWO  COORDINATE  CURVES
xi
Let a  nV  referred to coordinate
xl alone varies. Thus the coordinate curve of parameter

,2,1

...

(,

=

n

)

i

where

sC i,

 are constants.

Differentiating it, we get

ix =

,ic

i

 except

lx  is defined as
i =

l

. For a coordinate curve of parameter xl, the coordinate

...(1)

Let

iA  and

iB  be the tangent vectors to a coordinate curve of parameters

px  and

qx  respectively..

idx = 0,

, except

i

i =  and
l

ldx

0�

Then

iA =

dx =
i

,0,...0(

x

p

)0...0,

iB =

dx =
i

,0,...0(

x

q

)0...0,

If  q

 is required angle then

cos =

i
BAg

ij

j

i
AAg

ij

j

i
BBg

ij

j

=

=

p

BAg

pq

q

p

AAg

pp

p

q

BBg

qq

q

p

BAg

pq

q

BAgg
qq

pp

p

q

cos =

g

pq

gg
pp

qq

which is required formula for  q

.

The angle

ijw  between the coordinate curves of parameters

ix  and

jx  is given by

ijwcos

=

g

ij

gg
ii

jj

...(2)

...(3)

...(4)

(cid:222)
"
"
q
q
48

Tensors and Their Applications

If these curves are orthogonal then

  (cid:222)

ijwcos

=

cos =
2

0

ijg = 0

Hence the

ix  coordinate curve and

jx  coordinate curve are orthogonal if

ijg

0=

.

3.9 HYPERSURFACE
The n  equations    xi =    xi  (u1) represent a subspace of
(n �1) equations in xj,s which represent one dimensional curve.

nV . If we eliminate the parameter  u1,  we  get

Similarly the n equations xi =  xi (u1,u2) represent two dimensional subspace of Vn. If we eliminating
the parameters u1,  u2, we get n �2 equations in  xi,s which represent two dimensional curve  Vn.  This
two dimensional curve define a subspace, denoted by V2 of Vn.

Then n equations xi =  xi (u1, u2, ... un�1)  represent n � 1 dimensional subspace Vn�1 of Vn. If we
  which  represent  n  �1

eliminating  the  parameters  u1,  u2,  ...un�1,  we  get  only  one  equation  in
dimensional curve in Vn. This particular curve is called hypersurface of Vn.

sxi,

Let  f

 be a scalar function of coordinates

.ix  Then

( ix

)

  =  constant  determines  a  family  of

hypersurface of Vn.

3.10 ANGLE  BETWEEN  TWO  COORDINATE  HYPERSURFACE

Let

and

( ix

)

= constant

( ix

)

= constant

represents two families of hypersurfaces.
Differentiating equation (1), we get

� f
x�
i

i

dx

= 0

 is orthogonal to

.idx  Hence

This shows that

ix�

tangential to hypersurface (1).

...(1)

...(2)

...(3)

 is normal to

=f

constant,

since

idx   is

ix�

Similarly

and (2) then  w
is given by

 is normal to the hypersurface (2). If w

 is the angle between the hypersurface (1)

ix�
 is also defined as the angle between their respective normals. Hence required angle  w

cos

=

ij

g

yf
j
i
x
x

ij

g

f
i

x

f
j

x

ij

g

yy
j

i

x

x

...(4)

p
f
f
y
f
�
f
�
y
�
w
�
�
�
�
�
�
�
�
�
�
�
�
Metric Tensor and Riemannian Metric

If we take

and

f =

=px

constant

=

=qx

constant

The angle  w

 between (5) and (6) is given by

cos

=

ij

g

ij

g

p

x

i

x

x

x

p

i

x
x
p

j

q

j

x
x

ij

g

q

x

i

x

q

j

x

x

=

ij

g

p
i

q
j

ij

g

p
i

q
j

ij

g

q
i

q
j

cos

=

pq

g

g

pp

g

qq

49

...(5)

...(6)

...(7)

The angle

ij

 between the coordinate hypersurfaces of parameters

ix  and

jx  is given by

cos

=

ij

ij

g

ii
gg

jj

...(8)

If the coordinate hypersurfaces of parameters

ix  and

jx  are orthogonal then

=

ij

2

cos

= 0

ij

from (8), we have

0=ijg

.

Hence the coordinate hypersurfaces of parameters xi and xj are orthogonal if

0=ijg

.

3.11 n-PLY  ORTHOGONAL  SYSTEM  OF  HYPERSURFACES
If in a  nV  there are n families of hypersurfaces such that, at every point, each hypersurface is orthogonal
to the
 hypersurface of the other families which pass through that point, they are said to form as
n-ply orthogonal system of hypersurfaces.

1-n

3.12 CONGRUENCE  OF  CURVES

A family of curves one of which passes through each point of

nV  is called a congruence of curves.

3.13 ORTHOGONAL  ENNUPLE
An orthogonal ennuple in a Riemannian

nV  consists of n mutually orthogonal congruence of curves.

y
w
�
�
�
�
�
�
�
�
�
�
�
�
d
d
d
d
d
d
w
w
w
w
p
(cid:222)
w
50

Tensors and Their Applications

THEOREM 3.6 To find the fundamental tensors

ijg  and

ijg  in terms of the components of the unit

tangent

e h

=

(

h

,...2,1

n

)

 to an orthogonal ennuple.

Proof: Consider n unit tangents

 of an orthogonal ennuple
nV . The subscript h followed by an upright bar simply distinguishes one congruence

 to conguence

n
,...2,1

,...2,1

e h

h

h

n

)

(

(

)

ei
h

=

=

in a Riemannian
from other. It does not denote tensor suffix.

The contravariant and covariant components of

|he  are denoted by

|he  and

ihe |  respectively..

Suppose any two congruences of orthogonal ennuple are

|he  and

|ke  so that

j
i
eeg
|
|
k
h
ij

= h
k

i
h ee
|

|
ik

= h
k

j
i
eeg
|
k
h

ij

|

j
i
eeg
|
h
h

ij

|

= 0

= 1

from (1),

and

We define

i
he | =

cofactor

 of
e
|
ih

in

determinan

t

e
|
ih

e

|
ih

Also, from the determinant property, we get

n

=
1

h

i
h ee
|

|
jh

=

i
j

n

=

1

h

i
h ee
|

|
jh

g j k =

jk

i
j g

n

=

1

h

i
h ee
|

k
|
h

=

ikg

Multiplying by

jke

or

Again multiplying (2) by

.ikg

or

from (3) and (4)

n

=

1

h

i
gee
|
jh
h

|

ik

=

i
j g

ik

jkg

= (cid:229)

jhkh ee
|
|

n

ijg = (cid:229)

=

1

h

e
ih e
|

|
jh

...(1)

...(2)

...(3)

...(4)

...(5)

d
d

(cid:229)
d
(cid:229)
d
(cid:229)
(cid:229)
d
Metric Tensor and Riemannian Metric

n

ijg = (cid:229)

i
h ee
|

j
|
h

=

1

h

This is the required results.

51

...(6)

Corollary: To find the magnitude of any vector u is zero if the projections of u on

|he  are all zero.

Proof: Let

Then

or

i.e.,

iu = (cid:229)

i eu
| =
ik

n

=

1

h

n

=

1

h

i
hh eC

|

i
eeC
|
hh

|
ik

=

n

=

1

h

C
h

=

h
k

C

k

kC =

i eu
|
ik

kC =  projection of

iu  on

ike |

Using (8), equation (7) becomes

n

iu = (cid:229)

=

1

h

j
eu

i
e
|
kjh

|

Now,

2u =

i
uu

i

=

i
eC
hh

|

eC
k

|
ik

 from (7)

h

k

...(7)

...(8)

= (cid:229)

= (cid:229)

= (cid:229)

2u =

eeCC
|

i
hk

h

|
ik

,
kh

,
kh

h

n

=

1

h

h CC

k

h
k

hCC

h

(
hC

)

2

This implies that u = 0 iff
Hence the magnitude of a vector  u is zero iff all the projections of  u  (i.e. of  ui) on  n  mutually

 iff

0

.

2 =u

hC

0=

orthogonal directions

i
he |  are zero.

(cid:229)
(cid:229)
d
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:229)
(cid:229)
d
(cid:229)
52

Tensors and Their Applications

Miscellaneous  Examples

1.

If p and q  are orthogonal unit vectors, show that

(

gg
hj

ik

gg
hk

ij

)

Solution

ih

qpqp

j

k

= 1

Since p and q  are orthogonal unit vectors. Then

j

ij qpg
i

= 0,

2

p

= q

2

=

1

.

Now,

(

gg
hj

ik

gg
hk

ij

)

ih

qpqp

j

k

=

h
qqppgg

i

j

k

hj

ik

h
pqqpgg

k

i

j

hk

ij

j

()

i

qqg
ik

k

)

(

g

hk

k

j
qp

()

i
pqg
ij

j

)

h

(

=

ppg

hi
= p2.q2 � 0.0
= 1 .  1

2.

If  q

 is the inclination of two vectors A and B show that

= 1 (since

h

ppg

hi

j

=

&1

h

qpg

hk

k

=

0

)

q2

sin

=

(

gg
hi

ik

hk

h
BBAAgg
j

)
j

ij

k

j

i

h
BBAAgg

hj

ik

Solution
If  q

  be the angle between the vectors A and B then

cos =

But

sin

2

-=q
1

2

cos

j

BAg

ij

i

i
AAg
ij

j

i

BBg

ik

k

q2

sin

=

1

(

(

i
ABg
ij

j

h

AAg

hj

()
j

h

BAg

hk

()

i

BBg

ik

k

k

)

)

=

(

gg
hj

ik

hk

h
BBAAgg
i

)
j

ij

k

i

j

h
BBAAgg

hj

ik

k

k

3.

If

ijX  are components of a symmetric covariant tensor and u, v are unit orthogonal to w and

satisfying the relations
g+

i

)
ug
ij

(

X

ij

w

w

j

j

= 0

= 0

g

ij

)

v

i

d+

(

X
where a � b

ij

 prove that u and v are orthogonal and that

-
-
-
-
-
q
q
-
-
a
-
b
-
Metric Tensor and Riemannian Metric

i

vuX
ij

j

= 0

Solution

Suppose Xij is a symmetric tensor. Since

i vu ,

j

 are orthogonal to

iw  then

i wu
i

i wv
i

= 0

= 0

given

(

X

ij

)
ug
ij

i

g+

w

j

= 0

where a � b

(

X

ij

.

i
)
vg
ij

d+

w

j

= 0

Multiply (3) & (4) by

j uv ,

j

 respectively and using (1) and (2), we have

(

X

ij

(

X

ij

i
)
vug

ij

i
)
uvg

ij

j

j

= 0

= 0

53

...(1)

...(2)

...(3)

...(4)

...(5)

...(6)

Interchanging the suffixes i & j  in the equation (6) and since

ij Xg ,

ij

 are symmetric, we get

... (7)

...(8)

Proved.

ij
Subtract (6) & (7) we get

ij

(

X

i
)
vug

(

 and

i
ij vug)
.0�

Since
Hence,

j

j

= 0

= 0

i

vug
ij

j

= 0

So, u and v are orthogonal.
Using (8) in equation (5) & (6), we get

i
ij vuX

i

= 0

4. Prove the invariance of the expression

1
dxg

dx

...2

ndx

 for the element volume.

Solution

Since

ijg  is a symmetric tensor of rank two. Then

ijg =

k

x

i

x

l

x

j

x

g

kl

Taking determinant of both sides

Since

x =
x

J

 (Jacobian)

ijg

=

k

x

i

x

l

x

j

x

g

kl

klg

= g &

g ij

=

g

a
-
b
-
a
-
b
-
a
-
a
-
b
a
�
b
a
-
b
�
�
�
�
�
�
�
�
�
�
Tensors and Their Applications

54

So,

or

g =

2gJ

J =

g
g

Now, the transformation of coordinates from

lx  to

ix , we get

dx

1

dx

2

...

ndx

x
x

1
xdxd

...2

nxd

1
xdxJd

...2

nxd

=

=

dx

1
dx

...2

ndx

=

1
xdxd

...2

nxd

g
g

1
dxg

2
dx

...

ndx

=

1
xdxdg

...2

nxd

So, the volume element

dv

=

1
dxg

dx

2

...

ndx

 is invariant.

EXERCISES

1. For the Metric tensor gij defined gkl and prove that it is a contravariant tensor.
2. Calculate the quantities g i j  for a V3 whose fundamental form in coordinates u, v, w, is
2

gdwdu

hdudv

fdvdw

cdw

adu

bdv

+

+

+

+

+

2

2

2

2

2

3. Show that for an orthogonal coordinate system

g11 =

1
g
11

,  g22 =

1
22g

,

  g33 =

1
g

33

4. For a V2 in which

=

g

11

,
gE

12

=

,
gF

21

=

G

 prove that

g =

EG

2

F

,

11

=

g

gG

,

12

g

-=

,
ggF

22

=

gE

5. Prove that the number of independent components of the metric

ijg  cannot exceed

1
2

+nn
(

)1

.

6. If vectors ui, vi are defined by ui = g ij uj, v i = g ij v j show that ui = g ij u j, u ivi = uiv i  and ui gijuj = uigijuj
7. Define magnitude of a unit vector. prove that the relation of a vector and its associate vector is

reciprocal.

8. If  q

 is the angle between the two vectors

iA  and

iB  at a point, prove that

(

gg
hi

q2

sin

=

ih

j

k

BBAAgg
j

ik
hk
BBAAgg
hi

)
ij
ih

jk

k

9. Show that the angle between two contravariant vectors is real when the Riemannian metric is positive

definite.

�
�
-
-
CHAPTER � 4

CHRISTOFFEL'S  SYMBOLS  AND  COVARIANT

DIFFERENTIATION

4.1 CHRISTOFFEL'S  SYMBOLS

The German Mathematician Elwin Bruno Christoffel defined symbols

]kij,
[

=

1
2

+

g

ki
j
x

g

kj

i

x

g

ji

k

x

,  (
i

,

=

)n
,...2,1

,
kj

called Christoffel 3-index symbols of the first kind.

and

k

ji

=

g lk

[
ij

,

]l

...(1)

...(2)

called Christoffel 3-index symbols of second kind, where

jig  are the components of the metric Tensor

or fundamental Tensor.

There are  n  distinct Christoffel symbols of each kind for each independent

jig . Since

jig   is

symmetric  tensor  of  rank  two  and  has

+nn
(

)1

1
2

independent components of Christoffel�s symbols are

  independent  components.  So,  the  number  of
)1

)
=+
1

(
2
nn

(
nn

+

n

.

1
2

1
2

THEOREM 4.1 The Christoffel's symbols  [

]kij,

 and

k

ji

and j.
Proof: By Christoffel�s symbols of first kind

 are symmetric with respect to the indices i

]kij,
[

=

Interchanging i and j, we get

[

]kji,

=

1
2

1
2

+

g

ik
j
x

g

jk

i

x

g

x

ij

k

g

jk

i

x

+

g

ik
j
x

g

ji

k

x

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:215)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
56

Tensors and Their Applications

=

1
2

+

g

x

ki
j

g

kj

i

x

g

ji

k

x

 since

g =
ij

g

ji

Also, by Christoffel symbol of second kind

[

kji
,

]

=

[

kij
,

]

k

ji

k

ji

=

g lk

=

g lk

[
ij

,

[

ji

,

]l
]l

=

k

ij

 since  [
ij

]
, =
l

[

ji

,

]l

THEOREM 4.2  To prove that

(i)

(ii)

(iii)

]mij,
[

 =

g mk

k

ji

[
ik

,

]

+

[

j

jk

,

]i

=

g

x

ij
k

g

x

ij
k

=

jl

g

i

kl

im

g

j

km

Proof: (i) By Christoffel�s symbol of second kind

k

ji

=

g lk

[
ij

,

]l

Multiplying this equation by

,mkg

 we get

g mk

g km

k

ji

k

ji

g

mk

lk

g

[
ij

,

]l

[
ij

]l

,

l
m

 as

g

mk g

lk

d=

l
m

=

=

= [

]mij,

(ii) By Christoffel�s symbol of first kind

[
ik ,

]j

=

[

]i
jk , =

and

adding (1) and (2),

[
ik

,

]

+

[

j

jk

,

]i

=

1
2

1
2

1
2

+

g

jk

i

x

+

g

ki
j
x

g

x

ji

k

g

ji

k

x

+

g

ji

k

x

g

ji

k

x

g

x

ki
j

g

jk

i

x

 since

g =
ij

g

ji

Proved.

...(1)

...(2)

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
Christoffel's Symbols and Covariant Differentiation

57
57

[
ik

,

]

+

[

j

jk

,

]i

=

(cid:215) 2

1
2

g

x

ij
k

=

g

x

ij
k

(iii) Since

ij gg

lj

d=

i
l

.

Differentiating it w.r.t. to

kx , we get

ij

g

g

x

lj
k

+

g

lj

ij

k

g

x

= 0

Multiplying this equation by

lmg

, we get

ij
gg

lm

g

x

lj
k

+

lm
gg

lj

ij

g

k

x

= 0

lm
gg

lj

m
j

ij

k

g

x

ij

g

k

x

im

g

k

x

im

g

k

x

Interchanging m and j, we get

ij

k

g

x

ij

k

g

x

or

=

=

=

=

=

=

ij
gg

lm

g

x

lj
k

ij
gg

lm

{

[
lk

]
, +
j

[

]
}l

jk

,

 since

=

[
lk

,

]

+

[

j

].

jk

,

l

g

x

lj
k

{

[
lk

,

j

ij

g

}

]

lm

g

{

[

lm

g

ij

g

}l
]

jk

,

lm

g

i

kl

ij

g

m

kj

lj

g

ij

g

i

kl

i

kl

im

g

j

km

im

g

j

km

 as

g =
lj

g

jl

Proved.

THEOREM 4.3 To show that

i

i

j

=

)

g

log

(
jx

Proof: The matrix form of

ikg  is

ikg =

g

g

11

21

M
g

1
n

g

12
g

22

...

...

g

1
n
G
2

n

g

n

2

...

g

nn

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
-
�
�
d
-
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
-
-
58

Tensors and Their Applications

and

But

Take

g =

ikg

=

g

g

11

21

M
g

n

1

g

g

12

22

...

...

g

g

1
n

2

n

g

n

2

...

gnn

il

d=

l
k

ik gg
l =

.k

ik

ik gg

=

1=

k
k

ikg

= [

ikg

] 1-

 =

Gik  (Theorem 2.13 , Pg 25)
g

where

ikG  is cofactor of

ikg  in the determinant

ikg

g =

ikGg
ik

...(1)

Differentiating w.r.t.

ikg  partially

Now,

But

G =
ik

ik

gg

g
ikg

g
jx

g
jx

g
jx

g
jx

�1
g �

�1
g �

=

ikG  since

g
g

ik

ik

1=

=

g
g

ik

g

ik
j
x

=

G
ik

g

ik
j
x

ik

gg

=

ik

g

=

g

ik
j
x

g

ik
j
x

=

g ik

{

[

]
, +
i

[
]
}kij
,

jk

 as

[

=

]
, +
i

]kij
[
,

jk

g

ik
j
x

=

=

[

]
, +
i

jk

ik

g

]kij
[
,

ik

g

k

kj

+

i

ji

d
(cid:222)
(cid:222)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
Christoffel's Symbols and Covariant Differentiation

59
59

 as k  is dummy indices

�1
g �

�1
g �

g
jx

g
jx

=

i

ij

+

i

ij

=

2

i

j

i

1
g �
2

g
jx

g

)

log(
jx

=

=

i

ij

i

j

i

Proved.

EXAMPLE  1

If

ijg

0�

 show that

g

j

x

ki

=

x j

[
ik

,

]

ki

(

[

]
[
a+a

j

,

j

,

]

)

Solution

By Christoffel�s symbol of second kind
[
,ik

g

=

ki

]

Multiplying it by

bg

, we get

g

=

g

[
,ik

]

= [

a,ik

]

 as

g

b g

1=

g

g

ki

ki

Differentiating it w.r.t. to

jx  partially

g

x j

 =

x j

ki

[
,ik

]

g

j

x

ki

+

ki

since

]a,ik
[

=

x j

= [

]
[
b+b

j

,

]a

j

,

g

j

x

g
jx

g

j

x

ki

+

ki

([

b+b
[
]

j

,

j

,

])

 =

[
,ik

]

x j

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
�
�
a
b
b
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
-
a
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
b
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
b
a
b
a
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
b
b
a
a
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
�
�
a
b
a
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
�
�
b
a
a
b
�
�

�
�
a
b
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
�
�
a
b
a
�
�
60

Tensors and Their Applications

g

j

x

ki

=

x j

[
ik

,

]

ki

(

[

]
[
b+b

i

,

j

,

]

)

Solved.

EXAMPLE  2

Show  that  if

ijg

0=

  for

i �

j

  then  (i)

k

i

j

0=

  whenever  i,  j  and  k   are  distinct.

(ii)

i

ii

=

1
2

log

g

ii

i

x

  (iii)

i

i

j

=

1
2

ii

g
j

log

x

 (iv)

i

j

j

-=

1
g

2

ii

g

x

jj
i

Solution

The Christoffel�s symbols of first kind

]kij,
[

=

(a) If

i

==
j

k

The equation (1) becomes
]i

[
ii,

=

(b) If

i

�=
j

k

The equation (1) becomes

]kii,
[

=

1
2

1
2

1
2

Since

ikg

0=

 as

i �

 (given)

k

g

jk

i

x

+

g

ik
j
x

g

x

ij

k

...(1)

g

ii
i
x

+

g

ki
i
x

g

ki
i
x

�

g

x

ii
k

]kii,
[

=

1
2

g

x

ii
k

 or  [

]

-=

jj

,

i

�1

2

g

jj
i
x

[
]i
ij, =

=

1
2

1
2

+

g

ji

i

x

g

x

ii
j

g

ij

i

x

g

x

ii
j

,

 as

ijg

,0=

i �

j

(c)

�=
k

i

j

(d)

i

j

k

]kij,
[

= 0  as

ijg

0=

,

kjg

0=

,

i

j

k

(i) as i, j, k are distinct i.e.,

i

j

k

k

i

j

=

g kl

[
ij

,

]l

 since

,0=lkg

k �

l

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
�
�
a
b
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
-
a
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
-
�
�
�
�
�
�
�
�
�
-
�
�
�
�
�
�

�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)

Christoffel's Symbols and Covariant Differentiation

61
61

(ii)

(iii)

�=
k

i

j

(iv)

�=
k

j

i

k

i

j

i

ii

i

ii

i

ji

i

i

j

i

j

j

i

jj

= 0

=

g ii

[
ii

,

]i

=

ii

g

1
2

g

ii
i
x

 from (a)

1
g �
ii

g

ii
i
x

ii

g

 as

1=
g

ii

ii

g
i

log

x

=

=

2

1
2

[
ij

,

]i

g ii

=

=

=

=

=

=

1
g ii

1
2

[
ij

,

]i

ii

 as

g

1=
g

ii

ii

g
j

log

x

[

]i

jj

,

g ii

11
g
2

ii

g

x

jj
i

1
g

ii

2

g

x

jj
i

 from (b)

EXAMPLE  3

If

ds

2

=

2

+

dr

2
dr

2

+

2

r

sin

2

d

2

,

 find the values of

(i)

[22, 1] and [13, 3], (ii)

1

22

and

3

31

Solution

The given metric is metric in spherical coordinates,

1
x =
,
r

2
q=x

,

f=x
3

.

Clearly,

11g = 1,

g =
22

r

,2

g

33

=

2

r

sin

2

 and

jig

0=

 for

i �

j

Solved.

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:215)
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
-
f
q
q
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)

q
62

Also,

Tensors and Their Applications

11g

= 1,

22

=

g

1
2
r

,

33

=

g

1

2

r

sin

2

 (See Ex. 2, Pg. 39, and

,0=ijg

  for

i �

.j

)

(i) Christoffel Symbols of first kind are given by

]kij,
[

=

Taking

i

2== j

 and

1=k

 in (1)

]1,22
[

=

=

=

Taking

i

=

,1

j

==
k

3

 in (1)

]3,13
[

=

=

1
2

1
2

1
2

1
2

1
2

g

jk

i

x

+

g

x

ik
j

g

x

ij

k

,

i

,

=kj
,

3,2,1

...(1)

+

g

21
2
x

g

21
2
x

g

22
1
x

 Since g21 =  0.

+

0
2

x

0
2

x

2

r

1
x

1
2

2

r
r

-=

r

+

g

33
1
x

g
13
3
x

g

13
3
x

2

r

2

sin
r

 since

=g
13

0

]3,13
[
(ii) Christoffel symbols of the second kind are given by

sinr

=

q2

Taking k = 1, i = j = 2.

k

ji

1

22

1

22

1

22

=

kl

g

[
ij

,

]
l

=

k

1

g

[
ij

]
1,

+

[

]
2,

ji

+

k

2

g

[

]3,

ij

k

3

g

=

11

g

[
]
1,22

+

12

g

[
2,22

]

+

]3,22
[

13

g

= [

]
1,221

+

[
2,220

]

+

]3,220
[

 Since

12

g

= g

13

=

0

= r-

q
�
�
�
�
�
�
�
�
-
�
�
�
�

�
�
�
�
�
�
�
�
-
�
�
�
�
�
�
�
�
�
�
�
�
-
�
�
�
�
�
�
-
�
�
�
�
�
�
�
�
-
�
�
�
�
�
q
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
Christoffel's Symbols and Covariant Differentiation

63
63

and

3

31

3

31

=

=

=

[
]
1,13

31

g

+

g

32

[
2,13

]

+

]3,13
[

33

g

1

2

r

sin

2

1

2

r

sin

2

]3,13
[

 Since

31

g

= g

32

=

0

r

sin

2

=q

1
r

4.2 TRANSFORMATION  OF  CHRISTOFFEL'S  SYMBOLS

The fundamental tensors

coordinates

ix . Let

g

,
ij g

ijg  and
ij

 and

[

ijg  are functions of coordinates

ix  and  [

]kij,

 is also function of

kij
,

]

 in another coordinate system

ix .

(i )  Law  of  Transformation  of  Christoffel's  Symbol  for  First  Kind
Let  [

 is a function of coordinate

 in another coordinate system

]kji ,

ix  and

kij
,

]

[

[

kij
,

]

=

1
2

+

g

ik
i
x

g

x

ik
j

g

x

ij

k

Since

ijg  is a covariant tensor of rank two. Then

ijg =

p

x

i

x

2

x

j

x

g

pq

Differentiating it w.r.t. to

kx , we get

ix . Then

...(1)

...(2)

g

x

ij
k

g

ji
k
x

=

=

=

p

x

i

x

p

x

i

x

q

x

j

x

q

x

j

x

k

x

k

x

g

pq

+

g

pq

p

x

i

x

q

j

x

x

g

pq

k

x

p

2

x

k

x

i

x

+

q

x

j

x

p

x

i

x

2

q

x

k

x

j

x

g

pq

+

p

x

i

x

q

j

x

x

g

pq

r

x

r

x

k

x

...(3)

Interchanging i, k and also interchanging p, r in the last term in equation (3)

g

kj
i
x

=

p

2

x

i
xx

k

+

q

j

x

x

p

k

x

x

2

q

x

i
xx

j

+

g

pq

r

x

k

x

q

j

x

x

g

rq

p

x

p

x

i

x

...(4)

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
q
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:215)
q
-
�
�
�
�
�
�
�
�
-
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
64

Tensors and Their Applications

and interchanging j, k and also interchange q, r in the last term of equation (3)

g

x

ik
j

=

2

x
j

x

p

i

x

+

q

x

i

x

p

x

i

x

2

q

x

k

x

j

x

+

g

pq

p

x

i

x

r

x

k

x

q

j

x

x

g

pr

q

x

...(5)

Substituting the values of equations (3), (4) and (5) in equation (1), we get

kij
,[

]

=

1
2

2

p

x2
i
xx

j

q

k

x

x

g

pq

+

kij
,[

]

=

kij
,[

]

=

p

2

x

i
xx

j

p

2

x

i
xx

j

q

k

x

x

q

x

k

x

+

g

pq

+

g

pq

p

x

i

x

p

x

i

x

p

i

x

x

q

i

x

x

q

x

i

x

q

j

x

x

r

k

x

x

g

x

rp
q

g

+

qr
p

x

g

pq
r

x

r

x

k

x

1
2

g

rp

q

x

+

g

qr

p

x

g

pq

r

x

[

]rpq
,

r

x

k

x

...(6)

It  is  law  of  transformation  of  Christoffel's  symbol  of  the  first  kind.  But  it  is  not  the  law  of

transformation of any tensor due to presence of the first term of equation (6).

So, Christoffel's symbol of first kind is not a tensor.

(ii)  Law  of  Transformation  of  Christoffel's  Symbol  of  the  Second  Kind

g lk

[
ij

,

l

]

=

Let

k

i

j

ix . Then

 is function of coordinates

ix  and

g kl

],[
ij
l

=

k

i

j

 in another coordinate system

ij
],[
l

=

p

2

x

i
xx

j

q

l

x

x

+

g

pq

p

x

i

x

q

x

j

x

r

l

x

x

[

]rpq
,

 from (6)

As

klg  is contravariant tensor of rank two.

klg

=

k

x

s

x

l

x

t

x

st

g

Now

klg

ij
],[
l

=

=

=

=

k

x

s

x

k

x

s

x

k

x

s

x

k

x

s

x

l

t

x

x

st

g

p

2

x
i
xx

j

q

x

l

x

+

g

pq

k

x

s

x

l

x

t

x

st

g

p

x

i

x

l

t

x

x

q

l

x

x

st

g

p

2

x
i
xx

j

+

g

pq

k

x

s

x

l

x

t

x

r

x

l

x

q

j

x

x

p

x

i

x

r

x

l

x

q

j

x

x

[

]rpq
,

stg [

]rpq,

st

2
t

g

p

2

x
i
xx

g

pq

j

+

p

2

x

i
xx

j

sq
gg

pq

+

k

x

s

x

k

x

s

x

r
t

p

x

i

x

p

x

i

x

q

j

x

x

[

]rpq
,

st

g

 as

l

x

t
x

q

l

x

x

d=

q
t

q

j

x

x

sr

g

[

]rpq
,

 as

g =
st

g

sr

r
t

�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
d
�
�
�
�
d
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
d
Christoffel's Symbols and Covariant Differentiation

65
65

g kl

ij
],[
l

=

p

x

s

x

p

2

x

i
xx

j

+

s
p

k

x

s

x

p

x

i

x

2

x

j

x

s

qp

Since

sq gg

pq

=

s
p

 and

[

g sr

,
rpq

]

=

s

qp

k
j

i

=

k

x

s

x

2

s

x

i
xx

j

+

k

x

s

x

p

x

i

x

q

j

x

x

s

qp

....(7)

It is law of transformation of Christoffel�s symbol of the second kind. But it is not the law of

transformation of any tensor. So,  Christoffel�s symbol of the second kind is not a tensor.

Also, multiply (7) by

s

x

k

x

,

 we get

s

k

x

xd

k
ji

Since

s

x

k

x

k

x

s

x

s

k

x

x

k

i

j

� 2

s

x

i
xx

j

=

=

=

=

s

x

k

x

k

x

s

x

2

s

x

i
xx

j

+

s

x

k

x

k

x

s

x

p

x

i

x

q

x

j

x

s

qp

1=

s
s

s2
x

i
xx

j

+

p

x

i

x

q

j

x

x

s

qp

s

k

x
x

k

i

j

p

i

x
x

q

j

x
x

s
qp

...(8)

It is second derivative of

sx  with respect to  x �s in the terms of Christoffel�s symbol of second

kind and first derivatives.

THEOREM 4.4 Prove that the transformation of Christoffel�s Symbols form a group i.e., possess the
transitive property.
Proof: Let the coordinates

ix  be transformed to the coordinate system

jx  be transformed to

ix  and

ix .

When coordinate
second kind (equation (7)) is

ix  be transformed to

ix , the law of transformation of Christoffel�s symbols of

k
j

i

=

k

s

x

x

2

s

x

i

x

j

x

+

k

s

x

x

p

i

x

x

q

j

x

x

s

qp

...(1)

When coordinate

ix  be transformed to

ix . Then

r

vu

=

k
j

i

i

x
u

x

j

v

x

x

+

r

x

k

x

k

2

x

u

x

v

x

r

k

x

x

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
d
�
�
�
�
�
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
d
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)

�
�
�
�
�
�
�
�
�
�
�
66

Tensors and Their Applications

q

j

x

x

i

x

u

x

j

v

x

x

+

r

k

x

x

2

s

x

i
xx

j

k

x

s

x

i

x

u

x

j

v

x

x

r

x

k

x

s

qp

=

k

x

s

x

+

2

u

x

k

x

v

x

p

x

i

x

r

k

x

x

r

vu

=

s

qp

p

u

x

x

q

v

x

x

+

r

s

x

x

2

u

x

k

x

v

x

+

r

k

x

x

i

x
u

x

j

v

x

x

2

s

x

i

x

j

x

r

s

x

x

=

p

u

x

x

as

p

x

i

x

Since we know that

i

x
u

x

s

x

i

x

i

x

u

x

=

s

x

u

x

Differentiating (3) w.r.t. to

,vx

 we get

s

i

x

x

v

x

2

s

x

i
xx

j

Mutiply (5) by

2

s

x

i
xx

j

j

v

x

x

+

s

i

x

x

v

x

i

x

u

x

=

� 2

s

x

u

x

v

x

i

x

u

x

+

s

i

x

x

2

i

x

u

x

v

x

 =

� 2

s

x

u

x

v

x

i

x
u

x

j

v

x

x

r

x

s

x

.

i

x
u

x

+

r

s

x

x

2

i

x

u

x

v

x

s

s

x

x

r

s

x

x

 =

� 2

s

x

u

x

v

x

r

s

x

x

Replace dummy index i by k in second term on  L.H.S.

2

s

x

j

x

i

x
u

r

x

+

j

i
xx

x
x
Using (5) in equation (2), we get

x

x

v

s

2

u

k

x

v

x

r

k

x

x

 =

� 2

s

x

u

x

v

x

r

s

x

x

r

vu

=

s

qp

p

u

x

x

q

v

x

x

+

r

x

s

x

2

s

x

u

x

v

x

r

s

x

x

...(2)

...(3)

...(4)

...(5)

...(6)

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
Christoffel's Symbols and Covariant Differentiation

67
67

The equation (6) is same as the equation (1). This shows that if we make direct transformation
ix   we  get  same  law  of  transformation.  This  property  is  called  that  transformation  of
ix  to

from
Christoffel's symbols form a group.

4.3 COVARIANT  DIFFERENTIATION  OF  A  COVARIANT  VECTOR

iA  and

Let
Then

iA  be the components of a covariant vector in coordinate systems

ix  and

ix  respectively..

iA =

p

x

i

x

A

p

Differentiating (1) partially w.r.t. to

jx ,

A
i
j

x

A
i
j

x

=

=

� 2

p

x

j

x

i

x

� 2

p

x

j

x

i

x

+

A

p

+

A

p

p

x

i

x

p

x

i

x

A

x

p
j

A
p
q

x

q

j

x

x

It is not a tensor due to presence of the first term on the R.H.S. of equation (2).
Now, replace dum m y  index p by s in the first term on R.H.S. of (2), we have

A
i
j

x

=

� 2

s

x

j

x

i

x

+

A
s

p

x

i

x

A

x

p
q

q

j

x

x

Since we know that from equation (8), page 65,

� 2

s

x

i
xx

j

=

s

k

x

x

k

i

j

p

i

x

x

q

j

x

x

s
qp

...(1)

...(2)

...(3)

 Substituting the value of

� 2

s

x

i
xx

j

A
i
j

x

A
i
j

x

=

=

=

 in equation (3), we have

s

x

k

x

k

i

j

p

x

i

x

q

j

x

x

s

qp

+

A
s

p

x

i

x

A

x

p
q

q

j

x

x

k
j

i

k

i

j

s

k

x

x

A
s

+

A
k

p

i

x

x

p

x

i

x

q

x

j

x

q

x

j

x

s
qp

+

A
s

p

i

x

x

q

x

j

x

A
p
q

x

A
p
q

x

A
s

s
qp

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
68

Tensors and Their Applications

A
i
j

x

A
k

k

i

j

=

p

x

i

x

q

j

x

x

Now, we introduce the comma notation

p

A
q

x

iA , =
j

A
i
j

x

A
k

k

i

j

Using (5), the equation (4) can be expressed as

iA , =
j

p

x

i

x

q

j

x

x

A

,
qp

A
s

s

qp

...(4)

...(5)

..(6)

It is law of transformation of a covariant tensor of rank two. Thus,

iA ,

j

 is a covariant tensor of

rank two.

So,

iA ,

j

 is called covariant derivative of

iA  with respect to

jx .

4.4 COVARIANT  DIFFERENTIATION  OF  A  CONTRAVARIANT  VECT OR
iA  be the component of contravariant vector in coordinate systems

iA  and

ix  and

ix  respectively..

Let
Then

or

iA =

sA =

Differentiating it partially w.r.t. to

i

x

s

s

x
x

s

A

i

A

i

x
jx , we get

x
Since from equation (8) on page 65,

x

s

A

j

=

� 2

s

x

j

i

x

i

A

+

s

i

x

x

i

A

j

x

...(1)

� 2

s

x

j

x

i

x

=

s

k

x

x

k

i

j

i

A

p

x

i

x

q

j

x

x

s

qp

substituting the value of

� 2

s

x

j

x

i

x

 in the equation (1), we get

s

j

A

A

q

A

j

x

=

=

s

k

x

x

s

k

x

x

k

i

j

k
j

i

i

A

i

A

p

i

x

x

p

x

i

x

q

j

x

x

s
qp

i

A

+

i

A

q

x

j

x

s
qp

+

s

i

x

x

s

i

x

x

i

j

A

x

i

j

A

x

s

A
q

A

(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
Christoffel's Symbols and Covariant Differentiation

69
69

Interchanging the dummy indices i and k in the first term on R.H.S. and put

p

x

i

x

i

A

=

p

A

 we get

s

A
q

x

q

j

x

x

q

x

j

x

s

A
q

x

+

p

A

s

qp

i

j

A

x

+

k

A

i
jk

=

=

=

s

i

x

x

s

x

i

x

i

x

s

x

i
jk

k

A

q

x

j

x

p

A

s
qp

+

s

i

x

x

i

j

A

x

i

jk

k

A

+

i

A

j

x

q

j

x

x

s

A
q

x

+

p

A

s

qp

Now, we introduce the comma notation

i
jA, =

+

k

A

A
i
j

x

i

jk

Using (3), the equation (2) can be expressed as

i
jA, =

p

x

i

x

q

j

x

x

A

,
qp

It is law of transformation of a mixed tensor of rank two. Thus,
jAi ,

 is called covariant derivative of

iA  with respect to

jx .

two.

...(2)

...(3)

...(4)

iA , j is a mixed tensor of rank

4.5 COVARIANT  DIFFERENTIATION  OF  TENSORS

Covariant  derivative  of  a  covariant  tensor  of  rank  two.

jiA  and
Let
respectively then

jiA  be the components of a covariant tensor of rank two in coordinate system

ijA =

p

x

i

x

q

j

x

x

A

pq

Differentiating (1) partially w.r.t. to

kx

ix  and

ix

...(1)

A
ij
k

x

A
ij
k

x

=

=

p

x

i

x

p

x

i

x

q

x

j

x

q

j

x

x

A

pq

k

x

+

k

x

p

x

i

x

q

x

j

x

A

pq

A

pq
r
x

+

r

x

k

x

p

2

x

k

x

i

x

q

j

x

x

+

A

pq

p

x

i

x

2

k

x

q

x

j

x

A

pq

...(2)

as

A

pq
k
x

 =

A

pq
r
x

r

x

k

x

 (since

pqA  components in

ix  coordinate)

�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
70

Tensors and Their Applications

A

pq

A

pq

� 2

p

x

i
xx

k

� 2

p

x

i
xx

k

q

j

q

j

x

x

x

x

=

A
lq

� 2

l

x

i
xx

k

q

x

j

x

=

A
lq

q

j

x

x

h

ki

l

x

h

x

l

rp

p

x

i

x

r

x

k

x

Since we know that from equation (8) on page 65.

� 2

l

x

i
xx

k

A

pq

A

pq

� 2

p

x

i
xx

k

� 2

p

x

i
xx

k

q

j

q

j

x

x

x

x

=

=

=

h
ki

l

x

h

x

l
rp

p

i

x

x

r

k

x

x

h
ki

A
lq

l

x

h

x

2

j

x

x

l
rp

A
lq

p

i

x

x

q

j

x

x

r

k

x

x

h
ki

A
hj

l
rp

A
lq

p

i

x

x

q

j

x

x

r

k

x

x

...(3)

as

hjA =

A
lq

l

x

h

x

q

j

x

x

 by equation (1)

and

A

pq

p

x

i

x

2

q

x

j

x

k

x

=

A
pl

=

A
pl

p

x

i

x

p

x

i

x

=

=

h
kj

h
kj

A

pl

A
ih

A

pq

p

x

i

x

2

q

x

j

x

k

x

2

l

x

j

x

k

x

h

kj

p

i

x

x

l

x
h

x

l

x
h

x

l

rq

q

j

x

x

r

x

k

x

l
rq

q

j

x

x

r

k

x

x

p

x

i

x

A
pl

l
rq

p

i

x

x

q

j

x

x

r

k

x

x

A

pl

...(4)

Substituting the value of equations (3) and (4) in equation (2) we get,

A
ij
k

x

=

A

pq

r

x

A
lq

l
rp

A

pl

l
rq

p

x

i

x

q

x

j

x

A
ij
k

x

h
kj

A
ih

h
ki

A
hj

=

A

pq

r

x

A
lq

l

rp

A

pl

l

rq

r

k

x

x

p

x

i

x

+

h
ki

+

A
hj

h
kj

A
ih

q

j

x

x

r

k

x

x

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
�
�

(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
�
�
Christoffel's Symbols and Covariant Differentiation

71
71

kijA , =

A
ij
k

x

h

kj

A
ih

h

ki

A
hj

, then

kijA , =

A

,
rpq

p

x

i

x

q

j

x

x

r

x

k

x

It is law of transformation of a covariant tensor of rank three. Thus,  Aij,k  is a covariant tensor

of rank three.
kijA ,

So,

 is called covariant derivative of

kx .
Similarly we define the covariant derivation xk of a tensors

ijA  w.r.t. to

ijA  and

i

jA  by the formula

kAij , =

i

kjA , =

ij

A

k

x

i
A
j
k

x

+

lj

A

+

A

l
j

i

kl

i

kl

+

il

A

i
A
l

j

kl

l

kj

and

In general, we define the covariant deriavative xk of a mixed tensor

ij

abA ...

l
...

 by the formula

c

ij

abA ...

l
...

=

,
kc

A

...
l
ij
...
ab
c
k
x

A

l
...
ij
...
pb

c

+

A

pj
ab

...
l
...
c

i

kp

+

A

ip
ab

l
...
...
c

j

kp

(cid:215)+

+(cid:215)

A

p
...
ij
...
ab
c

l

kp

p

ka

A

l
...
ij
...
ap

c

p

kb

l
...
ij
A
...
ab

p

p

kc

Note:

kiA ,

 is also written as

(cid:209)=

A
ki
,

A
i

.

k

4.6 RICCI'S  THEOREM
The covariant derivative of Kronecker delta and the fundamental tensors gij and gij is zero.
Proof: The covariant derivative xp of Kronecker delta is

i
kj,

=

i
j
k

x

d+

l
j

i

kl

i
l

l

kj

=

0

+

i
kj,

= 0 as

i

kj

i
j
k

x

i

kj

=

;0

l
j

i

kl

=

i

kj

Also, consider first the tensor gij and the covariant derivative of gij is

kijg , =

g

x

ij
k

g

mj

m

ki

g

im

m

kj

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:215)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:215)
(cid:215)
(cid:215)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
d
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
�
d
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
72

Tensors and Their Applications

kijg , =

g

x

ij
k

[
ik

,

]

j

[

]i

jk

,

 as

g mj

=

[
ik

,

]j

m

ki

But

So,

g

x

ij
k

= [
ik

]
, +
j

[

]i

,
jk

g

x

ij
k

g

x

ij
k

kijg , =

kijg , = 0

We can perform  a similar calculation for the tensor  gij.

Since we know that

im gg

im
gg
,
k

mj

+

im
gg

,
kmj

mj
d=

i
,
kj

d=

i
j

. Similarly taking covariant derivative, we get

But

 gmj,k = 0 and

i
kj
,

=

.0

 So,

g

im
k
,

=

 as0

g

mj

0

EXAMPLE  4

Prove that if Aij is a symmetric tensor then
(
A
i

j
iA , =
j

1
xg

j

)

g

j

jk

A

1
2

g

jk
i
x

Solution

Given that

ijA  be a symmetric tensor. Then

ijA =

jiA

We know that

Put

k =  we get

,j

j
kiA , =

j
iA , =
j

=

=

j
iA , =
j

j

A
i
k

x

+

l
A
i

j

kl

j

A
l

l

ki

j

A
i
j

x

j

A
i
j

x

+

l
A
i

+

l
A
i

j

l

j

(
log

j

A
l

)

g

l

ji

hl

j
gA
l

[

]hij
,

l

x

g
j

j

j

A
i
x

1

g

+

j

A
i
g

j

(
A
i
x

j

x
)

g

]hij
[
,

jh

A

 since

ijA  is symmetric.

]kij
[
,

jk

A

...(1)

...(2)

...(3)

-
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
�
�
d
�
�
�
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
-
�
�
�
�
-
�
�
�
�
-
�
�
Christoffel's Symbols and Covariant Differentiation

73
73

But

]kij
[
,

A jk

=

]kij
[
,

A jk

=

1
2

1
2

jk

A

g

jk

i

x

+

g

ki
j
x

g

x

ij

k

jk

A

g

jk

i

x

+

jk

A

g

ki
j
x

jk

A

g

x

ij

k

jk

A

g

x

ki
j

kj

A

=

g

x

ji
k

On Interchanging the dummy indices j & k.

jk

A

jk

A

g

x

ki
j

g

x

ij
k

jk

A

g

x

ki
j

jk

A

=

g

x

ji
k

 since

A =
ij

ji

A

= 0 as

g =
ij

g

ji

Using (5), equation (4) becomes

]kij
[
,

A jk

=

jk

A

1
2

g

jk
i
x

Put the value of

]kij
[
,

A jk

 in equation (3), we get
)

j

g

(
A
i
x

j

1

g

j
iA , =
j

jk

A

1
2

g

jk

i

x

EXAMPLE  5

Prove that

k

k

i

j

i

j

b
are the Christoffel symbols formed from the symmetric tensors

a

ija  and

ijb .

 are components of a tensor of rank three where

Solution

Since we know that from equation (8), page 65.

� 2

s

x

i
xx

j

s

k

x

x

k
j

i

k

i

j

=

=

=

s

k

x

x

k
j

i

p

i

x

x

q

j

x

x

s
qp

s2
x

i
xx

j

+

p

x

i

x

q

j

x

x

s

qp

2

s

x

i
xx

j

+

p

x

i

x

q

j

x

x

s

qp

k

x

s

x

or

...(4)

...(5)

Proved.

k

i

j

a

and

k

i

j

b

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
�
�
�
�
�
�
(cid:222)
�
�
-
�
�
�
�
�
�
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)

�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
74

Tensors and Their Applications

Using this equation, we can write

k

i

j

k

i

j

k

i

j

=

=

=

a

b

b

2

s

x

i
xx

j

+

2

s

x

i
xx

j

+

p

i

x

x

p

i

x

x

q

j

x

x

q

j

x

x

s
qp

s
qp

a

b

k

s

x

x

k

s

x

x

s
qp

a

s
qp

b

p

i

x

x

q

j

x

x

k

s

x

x

and

Subtracting, we obtain

k

i

j

a

Put

s

s

qp

a

qp

b

=

s
pqA

Then above equation can written as

x
It is law of transformation of tensor of rank three.

x

x

i

j

s

k
ijA =

A

s
pq

p

x

q

x

k

x

So,

k

k

i

j

a

i

j

b

EXAMPLE  6

 are components of a tensor of rank three.

If a specified point, the derivatives of gij w.r.t. to xk  are all zero. Prove that the components of

covariant derivatives at that point are the same as ordinary derivatives.
Solution

Given that

Let

i

jA  be tensor..

g

x

ii
k

= 0,

i

,

,
kj

  at  P0

Now, we have to prove that

i
A
kj
,

=

i
A
j
k

x

at

P
0

.

i

kjA , =

i
A
j
k

x

+

A

j

i

k

i

A

kj

...(1)

...(2)

(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
"
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
a
Christoffel's Symbols and Covariant Differentiation

75
75

Since

k

i

j

 and  [

]kij,.

 both contain terms of the type

g

x

ij
k

 and using equation (1) we get

k

i

j

0 =

]kij,
[

=

 at

0P .

So, equation (2) becomes

i

kjA , =

i
j

A
k

x

 at

0P

4.7 GRADIENT,  DIVERGENCE  AND  CURL

(a) Gradient
If  f

 be a scalar function of the coordinates, then the gradient of   f

 is denoted by

grad f =

ix�

which is a covariant vector.

(b) Divergence
The divergence of the contravariant vector

iA  is defined by

iA div

=

i

A
i

x

+

k

A

i

ik

It is also written as

i
iA,

The divergence of the covariant vector

iA  is defined by

iA div

=

ik Ag
ik

EXAMPLE  7

Prove that div

Solution:

i
A

= 1
g

(

)

k

Ag
k

x

If

iA  be components of contravariant vector then

iA div

=

=

i
A
i,

i

A
i

x

+

k

A

i

ik

Since

So,

i

ik

=

k

x

(
log

)

=

g

1

g

g
k

x

iA div

=

i

A
i

x

+

1

g

g
k

x

k

A

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
f
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
76

Tensors and Their Applications

Since i is dummy index. Then put

i =

,k

 we get

div Ai =

k

A
k

x

iA div

=

�1
g

(c) Curl
Let

iA  be  a covariant vector then

g
k

k

A

+

(

1

g

k

Ag
k

x

x
)

...(1)

Proved

iA , =
j

ijA , =

A
i
j

x

A

x

j
i

A
k

A
k

k

ji

k

ij

 is covariant tensor of second order, which is called curl of

iA .

and

are covariant tensor.

So,

A
i

,

j

=

A

,
ij

A
i
j

x

A

x

j
i

Thus

Note:

Since

curl

iA

=

A
i

,

j

A

,
ij

curl Ai is a skew-symmetric tensor.

Aj, i � A i,j = � (Ai,j � A j,i)

EXAMPLE  8

If

ijA  be a skew-symmetric tensor of rank two. Show that

A
,
kij

+

A

jk

,
i

+

A
ki

,

j

=

+

A
ij
k

x

A

jk
i
x

+

A
ki
j
x

Solution

Since we know that

kijA , =

jkA , =
i

kiA , =
j

A
ij
k

x

A

jk
i
x

A
ki
j
x

A
lj

l

ki

A
il

l

kj

A
lk

A
li

h

ij

l

jk

A

jl

A
kl

l

kj

l

i

j

�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
-
�
�
-
-
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
Christoffel's Symbols and Covariant Differentiation

77
77

Adding these, we get

A
,
kij

+

A

jk

,
i

+

A
ki

,

j

=

+

A
ij
k

x

A

jk

i

x

+

A
ki
j

x

A
lj

l

ki

+

A

jl

l

ik

Since

l

ki

 is symmetric i.e.,

A
li

=

l

jk

l

ik

l

ki

+

A
il

l

kj

A
kl

l

i

j

+

A
lk

l

ij

 etc.

=

A
ij
k

x

+

l

k

Since

ijA  is skew-symmetric. Then

A
lj

j
-=

So,

A +
li A
il

= 0 and

A
kl

A

jk
i
x

+

A
ki
j
x

(

A
li

)

+

A
il

+

A

jl

A
+

jl

A
lk

A
lj
0=

l

ki

l

i

+

(

A
lj

A

jl

)

(

+

A
kl

A
lk

)

. Similarly,,

j
0=

A
,
kij

+

A

jk

,
i

+

A
ki

,

j

=

+

A
ij
k

x

A

jk
i
x

+

A
ki
j

x

THEOREM 4.5 A necessary and sufficient condition that the curl of a vector field vanishes is that
the vector field be gradient.

Proof: Suppose that the curl of a vector

iA  vanish so that
=
A
i

A

0

ij
,

j

,

iA
To prove that
Since from (1),

(cid:209)=

,

...(1)

iA

=

curl
  f

 is scalar..

A
i

,

j

A

,
ij

= 0

A
i
j

x

A
j
i
x

A
i
j

x

j

A
i dx
j

x

= 0

=

=

A

j
i

x
A
j dx
i
x

j

(
j
dxA

)j

iA =

(

dxA
j

)

j

i

x

Integrating it we get

idA =

i

x�

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
(cid:222)
�
�
�
�
�
�
-
f
-
(cid:222)
�
�
-
�
�
(cid:222)
�
�
�
�
(cid:222)
�
�
�
�
(cid:222)
�
(cid:242)
�
�
78

Tensors and Their Applications

=

i

x

j

dxA
j

iA =

or

iA =
Conversely suppose that a vector
iA =

To prove curl

0=

iA

Now,

,

 where

(cid:242)=f

j

j dxA

ix�
.

iA  is such that
 is scalar..

  f

,

iA =

=f

ix�

� 2
x �
j x
i
� 2
i xx �

j

A
i
j

x
A

x
A
j
i
x

j
i

=

=

= 0

curl

iA

=

A
i

,

j

=

A

ij
,

A
i
j

x

A
j
i

x

=

0

curl

iA

= 0

Proved.

A
i
j

x

and

So,

So,

So,

THEOREM 4.6 Let  f

 and y

 be scalar functions of coordinates xi. Let A be an arbitrary vector then

(i)

div

(

)
A

f=

(cid:215)+
AAdiv

(ii)

(iii)

(

(2

f=

)

y+y

f=

)

2

y+y

2

f2(cid:209)+y

(iv)

div

(

y=f
)

2

(cid:209)+f

Proof: (i) Since we know that

iA div

=

replace Ai by

iAf

, we get

div(

iAf

)

=

(

(

)

i

Ag
i

x

)

i
Ag
i

x

�1
g

�1
g

...(1)

(cid:242)
�
�
f
�
f
(cid:209)
f
(cid:209)
f
�
(cid:209)
�
�
�
f
�
�
�
f
�
�
-
�
�
�
�
-
�
�
-
f
(cid:209)
f
f
(cid:209)
(cid:209)
f
y
(cid:209)
y
(cid:209)
(cid:215)
(cid:209)
(cid:209)
f
y
(cid:209)
y
(cid:209)
(cid:215)
f
(cid:209)
(cid:209)
y
�
�
f
Christoffel's Symbols and Covariant Differentiation

79
79

1

g

i

x

f

=

=

=

i

)

+f

i

Ag

(

Ag
i

x

i

x

i

A

f+

1

g

(

)

i
Ag
i

x

i

A

+

divf

i

A

Thus

div( Af
)

=

f+

A

 div

A

(ii) By definition of gradient,

)

(
ix�

y+

i

x

y+y

i

x

(f

(f

)

=

=

)

=

Thus

(iii) Taking divergence of both sides in equation (3), we get

( div

(2 f

[
f div

( div

)

)

=

=

=

y+y

]f

+

)

( div

)

f+y

( div

(cid:209)+

)

y+f

( div

)

(2 f

)

=

( div

y+

)

( div

(cid:209)+f
2)

Thus,

(iv) Replace A by

 in equation (3), we get

(2 f

)

=

2

y+y

2

(cid:209)+f
2

( div

( div

)

)

=

=

f+y

( div

)

f+y

2

...(2)

...(3)

THEOREM 4.7 Let

(i)

curl

(

)
A

iA  be a covariant vector and  f
�=
A
curlA

f+f

 a scalar function. Then

(ii)

curl

(

(cid:209)=f
)

Proof: (i) Let

iA  be a covariant vector then
curl
A
i

curl

A

=

Replacing

iA  by

iAf

, we get
= (
(
)
iAf
f

)
,
jA
i

curl

=

A
i

,

j

A

,
ij

) iA
(
f
,

j

�
�
�
�
�
�
�
�
�
f
�
(cid:215)
�
�
�
�
(cid:215)
�
f
�
(cid:215)
(cid:209)
(cid:215)
f
(cid:209)
y
(cid:209)
f
y
�
�
f
�
�
y
�
f
y
(cid:209)
f
(cid:209)
(cid:209)
f
f
y
(cid:209)
(cid:209)
(cid:209)
y
(cid:209)
f
(cid:209)
y
y
(cid:209)
f
f
(cid:209)
(cid:209)
(cid:215)
y
y
(cid:209)
(cid:209)
(cid:215)
f
(cid:209)
y
(cid:209)
y
(cid:209)
(cid:215)
f
(cid:209)
y
(cid:209)
f
y
(cid:209)
y
(cid:209)
(cid:215)
f
(cid:209)
(cid:209)
f
y
(cid:209)
y
(cid:209)
f
y
(cid:209)
(cid:209)
(cid:215)
f
(cid:209)
y
(cid:209)
f
y
(cid:209)
(cid:209)
(cid:215)
f
(cid:209)
(cid:209)
f
y
(cid:209)
�
f
(cid:209)
y
-
-
80

Tensors and Their Applications

=

,

jA
i

f+

A

,
ji

,

i

A

j

A

,
ij

=

(

A
i

,

iAj
),
j

f+

(

A
i

,

j

A

ij
,

)

=

A
i

f+f

curl

A
i

So,

curl

( Af
)

=

A

f+f

curl
A

...(1)

(ii) Replacing A by

 in equation (1), we get

Interchange of  f

curl
(
 and  y

)

=

curl
(

(cid:209)+

)

, we get

curl
(

)

=

curl
(

(cid:209)+f
)

.

Since curl  (
So,

)
.0=f

curl
(

 )

=

.

Proved.

4.8 THE  LAPLACIAN  OPERATOR
The operator

2

 is called Laplacian operator  read as "del square".

THEOREM 4.8 If  f

 is a scalar function of coordinates

ix  then

Proof: Since

and

2 =

1
xg

k

kr

gg

r

x

2 =

div

grad

which is covariant vector.

grad

f

=

,

rx�

...(1)

But we know that any contravariant vector

kA  associated with

rA  (covariant vector) is

kA =

kr Ag
r

 (Sec Art. 3.4, Pg 43)

Now, the contravariant vector

kA  associated with

 (Covariant vector) is

rx�

Since

kA =

kr

g

iA div

=

1

g

r

x

(

)

k

Ag
k

x

,

 (Sec Ex. 7, Pg 75)

f
-
f
-
f
-
f
-
f

(cid:209)
�
(cid:209)
�
y
(cid:209)
y
(cid:209)
f
f
(cid:209)
�
y
y
(cid:209)
f
f
(cid:209)
y
y
(cid:209)
�
f
(cid:209)
y
(cid:209)
f
(cid:209)
y
y
(cid:209)
�
f
(cid:209)
(cid:209)
f
(cid:209)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
f
�
�
�
f
(cid:209)
f
f
�
f
�
�
f
�
�
�
Christoffel's Symbols and Covariant Differentiation

81
81

So, from (1)

2 =

 div

g

kr

=(cid:247)

r

x

1

g

kr

gg

k

x

r

x

Proved.

EXAMPLE  9

Show that, in the cylindrical coordinates,

V2

=

1
r

r

r

+(cid:247)

V
r

1
2

r

2
V
2

+

2
V
2

z

by Tensor method

Solution

The cylindrical coordinates are

r q
,(

,

z

).

 If V is a scalar function of

r q
,(

,

z

).

Now,

Since

Let

V2

=

V(cid:209)

V(cid:209)

=

i

+

j

V

+

k

V
r

V
z

Then

=

V

+

iA
i
1

jA
2

j

+

kA

k
,3

is

1A =

,

V
r
 since  V(cid:209)

= V

A
2

,

A
3

=

V
z

...(1)

 is covariant tensor. The metric in cylindrical coordinates

here,

Since

1
x =
r
,
ds =2

g

2
q=x
i

dx

ij

dx

2ds =

2

+

dr

2
dr

2

+

2

dz

,

x =3

z

.

j

11g = 1,

g =
22

r

,2

=g
33

1

and others are zero.

g =

g ij

=

g

g

g

11

21

31

g

g

12

22

g

13
g

23

001

=

0

2

r

0

=

2

r

g

32

g

33

100

Now,

g = r    (See Pg. 34, Example 1)

( div V(cid:209)

)

=

 div

i
A

(

)

k

Ag
k

x

= 1
g

f
(cid:209)
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
f
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
f
�
(cid:209)
�
�
q
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:209)
(cid:215)
(cid:209)
�
�
q
�
�
�
�
�
�

q
�
�

�
�
(cid:209)
q

�
�
82

Tensors and Their Applications

1

g

(

)

+

1
Ag
1
x

1

g

=

( div V(cid:209)

)

=

1
r

)

(

1
rA
r

+

(

rA

2

)

+

2

)

+

(

3

)

Ag
3

x

(

Ag
2

x

3

)

(

rA
z

...(2)

We can write

Put

1=k

Similarly,

and

So,

kA =
kA =

kq Ag
q
k
1
Ag
1

 (Associated tensor)
+
+

k
2
Ag
2

k
3
Ag
3

1A =
1A =

11
Ag
1
11 Ag
1

+

12
Ag
2

+

13
Ag
3

 as

g

12

= g

13

=

0

2A =
3A =

22 Ag
2

33 Ag
3

11g

=

Cofactor

g
 of

in

g

11

=

g

22g

=

 of
Cofactor
g

g

22

in

g

=

=

1

2

2

r

r

1
2

r

33g

=

Cofactor

 of

g

in

g

33

g

=

2

2

r

r

=

1

 (See. Pg. 34, Ex.1)

1A =

11
Ag
1

=

A
1

2A =

22
Ag
2

=

3A =

33
Ag
3

=

A
2

1
2
r
A
.3

or

A =
1 A
,1

A =
2

1
2
r

A
2

,

A =
3

A
.3

from (1), we get

from (2),

1A =

,

2

A

=

V
r

1
2
r

V

,

=3

A

V
z

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
q
�
�
�
�

�
�

q
�
�

�
�
Christoffel's Symbols and Covariant Differentiation

83
83

So,

V2

=

 div

Ai

=

1
r

r

r

+(cid:247)

V
r

r

1
2

r

V

+(cid:247)

r

z

V
z

( div

V(cid:209)

)

=

=

( div

V(cid:209)

)

=

V2

=

1
r

1
r

1
r

1
r

r

r

r

r

V
r

V
r

+(cid:247)

1
r

V

+(cid:247)

r

z

V
z

+(cid:247)

21
r

V
2

+

r

V
2

2

z

r

r

r

r

+(cid:247)

+(cid:247)

V
r

V
r

1
2

r

1
2

r

2
V
2

2
V
2

+

+

2
V
2

z

2
V
2

z

EXERCISES

1. Prove that the expressions are tensors

(a)

lA ,
ij

=

A
ij
l

x

A

j

li

A
i

jl

(b)

r
A
jki

,
l

=

A

r
jki
l
x

r

A

jk

li

A

r
ki

lj

r
A
ij

 +

lk

r

l

ijkA

2. Prove that

j
iA , =
j

1

g

g

)

(

j

A
i
x

j

j
A
k

k

i

j

3. If

ijkA  is a skew-symmetric tensor show that

1
g �

k

x

(

jki

Ag

)

 is a tensor..

4. Prove that the necessary and sufficient condition that all the Christoffel symbols vanish at a point is

that gij are constant.

5. Evaluate the Christoffel symbols in cylindrical coordinates.
6. Define covariant differentiation of a tensor w.r. to the fundamental tensor gij. Show that the covariant
differentiation of sums and products of tensors obey the same result as ordinary differentiation.

7. Let contravariant and covariant components of the same vector  A be Ai and Ai respectively then

prove that

A div

i

=

iA div

(cid:209)
�
�
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
q
�
�
q
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
q
�
�
q
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
q
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
q
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:209)
�
�
q
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
a
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
�
�
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
84

Tensors and Their Applications

8. If

ijA  is the curl of a covariant vector. Prove that

A

,
kij

+

A

jk

,

i

+

A

ki

,

j

= 0

9. To prove that

u (cid:209)

u

= � ucurl u if u a vector of constant magnitude.

10. A necessary and sufficient condition that the covariant derivative vector be symmetric is that the

vector must be gradient.

11. Show that, in spherical coordinates

2
V

 =

1
2
r

2

r

r

+(cid:247)

V
r

1
sin

2

r

sin

V

+(cid:247)

1
sin

2

2

r

2
V
2

by tensor method.

(cid:215)
(cid:209)
f
�
�
q
?
(cid:246)
(cid:231)
?
(cid:230)
q
�
�
q
q
�
�
q
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
CHAPTER � 5

RIEMANN-CHRISTOFFEL  TENSOR

5.1 RIEMANN-CHRISTOFFEL  TENSOR
If Ai is a covariant tensor then the covariant x j derivative of Ai is given by

Ai,j =

A
i
j

x

A

i

j

...(1)

Differentiating covariantly the equation (1) w.r. to

,kx

 we get

jkiA ,

=

A

x

ji
,
k

A

,

j

ki

A
i

,

kj

=

k

x

A
i
j

x

A

i

j

ki

A

j

x

A

j

A
i

x

A

ki

ki

jkiA ,

=

2

k

x

A
i
x

j

i

j

k

x

A

A

x

k

ki

A

x

j

ki

ki
Interchanging  j and k in equation (2), we get

+

A

j

kj

+

A
i

x

kj

i

A

...(2)

iA , =
kj

2

j

A
i
x

k

x

ki
j

x

A

A

j

x

A
k

x

i

j

ki

a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
�
�
a
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
b
-
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
�
�
�
�
b
a
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
g
-
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
g
a
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
�
�
�
a
a
a
g
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
g
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
-
�
�
�
a
a
a
86

`

Tensors and Their Applications

+

i

j

A

k

ik

+

A
i

x

jk

i

A

...(3)

Subtract equation (3) from (2), we get

A
i

,

jk

A
i

,

kj

=

ki

A

j

i

j

k

x

A

i

j

k

A

+

ki
j

x

A

Interchanging of  a

 and  b

 in the first and third term of above equation, we get

A
i

,

jk

A
i

,

kj

=

ki
j

x

+

i

j

k

x

ki

j

i

j

k

A

...(4)

where

A
i

,

jk

A
i

,

kj

=

jkiRA

jkiR

=

ki
j

x

j
k

i

x

+

ki

j

i

j

k

...(5)

...(6)

a covariant tensor of rank three. Hence it follows from quotient law that

Since Ai is an arbitrary covariant tensor of rank one and difference of two tensors Ai, jk � Ai, kj is
jkiR  is a mixed tensor of rank
j

jkiR is called Riemann Christoffel tensor or Curvature tensor for the metric

four. The tensor

dxg
ij

dx

.

i

The symbol

jkiR is called Riemann�s symbol of second kind.

Now, if the left hand side of equation (4) is to vanish i.e., if the order of covariant differentiation

is to be immaterial then

jkiR

= 0

Since  Aa  is arbitrary. In general

  0,  so  that  the  order  of  covariant  differentiation  is  not
immaterial, It is clear from the equation (4) that �a necessary and sufficient condition for the validity  of
inversion of the order of covariant differentiation is that the tensor

jkiR  vanishes identically..

jkiR �

Remark
The tensor

i
iklR

=

k

x

i

kj

+

l

x
i

j

l

i

k

i

l

kj

lj

...(7)

g
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
g
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
b
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
a
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
-
a
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
a
a
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
a
a
a
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
Riemann-Christoffel Tensor

87

THEOREM  5.1 Curvature tensor

jkiR  is anti symmetric w.r.t. indices j and k.

Proof: We know that from (7) curvature tensor

Interchanging  j and k, we get

j

x

k

x

+

j

k

jkiR

=

i

j

ki

i

j

ki

jkiR

=

a
iR
kj

=

ki
j

x

i

j

k

x

i

j

k

x

+

ki
j

x

+

j

ki

k

i

j

k

i

j

j

ki

=

ki
j

x

i

j

k

x

+

j

ki

k

i

j

ikR

j

= �

jkiR

So,

jkiR  antisymmetric w.r.t. indices j and k.

Theorem 5.2 To prove that

R

jki

+

R

jk

i

+

R

jki

= 0

Proof: Since we know that

ijkR =

j

x

k

x

+

j

k

i

j

ki

i

j

ki

kijR

=

ki
j

x

i

j

k

x

+

j

ki

i

j

k

...(1)

a
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
�
�
�
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
-
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
a
a
a
a
a
a
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
�
�
�
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
88

Similarly

and

Tensors and Their Applications

jR

ki

=

ij
k

x

kj
i

x

+

k

j

i

kj

i

jkiR

=

jk
i

x

ik
j

x

+

i

jk

ik

j

...(2)

...(3)

On adding (1), (2) and (3), we get

R

kji

+

R

ikj

+

R

jik

= 0

This is called cyclic property.

5.2 RICCI  TENSOR

The curvature tensor
its lower indices

jkiR  can be contracted in three ways with respect to the index a

 and any one of

Now, from equation (7), art. 5.1,

kjR

,

iR

a k

,

ajiR

jkR

=

j

x

k

x

+

j

k

j

j

k

k

=

=

k

j

x

j

+

k

x

g

2

log
j

x

k

x

�

2

log
k

x

x

g
j

j

k

k

j

 and  a

 and   b

 are free indices

kjR

0=

.

Since

=

k

g

log
kx

Also for

ajiR .

Write

jiR  for

ajiR

a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
-
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
a
a
a
a
a
a

a

a
a
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
a
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
a
�
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
a
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
a
a
a
a
Riemann-Christoffel Tensor

89

ijR =

R
ij

=

j

x

x

j

+

i

j

i

i

j

i

ijR =

i

j

x

i

j

x

+

j

i

i

j

ijR =

g

log2
j
x

i

x

j

+

i

x

j

i

i

j

...(1)

Interchanging the indices i and j we get

jiR =

j

i

x

ij

x

+

i

j

j

i

jiR =

g

log2
i
xx

j

j

+

i

x

i

j

i

j

...(2)

 and  b

(Since  a
Comparing (1) and (2), we get

 are dummy indices in third term).

Thus

ijR  is a symmetric Tensor and is called Ricci Tensor..

ijR =

jiR

For

a kiR

:

a kiR

=

R
ik

-=

R

ik

5.3 COVARIANT  RIEMANN-CHRISTOFFEL  TENSOR
The associated tensor

lkjiR

=

i Rg

jkl

...(1)

is known as the covariant Riemann-Christoffel tensor or the  Riemann-Christoffel  tensor  of  the  first
kind.

Expression for

ijklR

ijklR

=

a
i Rg
a jkl

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
�
�
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
-
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
-
�
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
-
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
-
�
�
�
a
a
a
-
a
a
a
a
90

Tensors and Their Applications

=

g

i

k

x

j

l

l

x

kj

+

lj

k

kj

l

�(2)

Now,

Similarly,

g

i

k

x

j

l

g

i

l

x

kj

=

=

=

x
[

g

i

k

]

i

,
jl
k

x

]

i

[

,
jk
l

x

lj

lj

g

i
k
x

g

i
k
x

lj

g

i
x

l

kj

...(3)

...(4)

Using (3) and (4) in equation (2), we get
]

[

]

[

g

i
x

j

l

k

x

By the

formula

udv
dx

=

duv
dx

vdu
dx

+

g

i
l
x

g

i
k
x

kj

j

l

+

g

i

j

l

k

g

i

kj

l

+

g

i

j

l

k

g

i

kj

l

jk

i

,
l

g

i
l
x

(

[
il

,

]
[
a+a

]

)

,
il

(

[
ik

,

]
[
a+a

]
)ik
,

lj

[

[

]

+

i

,
l

jk

x

]

i

+

,
jk
l

x

kj

kj

[

]

,
ik

[

]il
,

kj

]

i

+

[

jk
,
l

x

[
il

,

]

kj

[
ik

,

]

j

l

lkjiR

=

[

]i
jl, =

[

1
2

k

x
jk

] [

,

i

k

x
jl

]

,

i

+

kj

[
ik

,

j

,

[
il

l
]

]

+

g

x

li
j

g

ji

l

x

g

jl

i

x

...(5)

...(6)

...(7)

lkjiR

=

=

lkjiR

=

i

,
k

jl

x

[

[

]

i

]

i

,
jl
k

x

,
jl
k

x

+

lj

]

i

[

jl
,
k

x

lkjiR

=

It is also written as

But we know that

(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
a
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
�
�
a
�
�
�
�
�
�
-

(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
�
-
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
�
�
a
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
-
�
�
a
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
-
�
�
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
-
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
-
�
�
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
Riemann-Christoffel Tensor

and

[

]i
jk , =

1
2

+

g

ki
j
x

g

ji

k

x

g

ik
i
x

Using (7) and (8), equation (5) becomes

91

...(8)

ijklR

=

ijklR

=

1
2

+

1
2

+

g

x

li
j

g

ji

l

x

g

jl

i

x

1
2

l

x

+

g

ki
j
x

g

ji

k

x

g

jk

i

x

[
il

,

]

[
ik

,

]

j

l

k

x

kj

g

2

j

il

x

k

x

+

2

jk

g
i
xx

2

g
j

2

g

jl

k

i
xx

+

ik

l

x

l

x

[
il

,

]

kj

[
ik

,

]

lj

...(9)

Since

=

g

[

]b

,jk

 and

lj

kj

=

g

[

]b

,jl

lkjiR

=

1
2

2

g

j

x

+

il
x

k

2

g
jk
i
l
xx

2

g
j

x

2

g

jl
i
l
xx

ik
x

l

+

g

[

jk

,

][
il

,

]

g

[

jl

,

][
ik

,

]

...(10)

This is expression for

lkjiR

.

The equation (9) can also be written as

lkjiR

=

1
2

2

j

g

il
x

k

x

+

2

g
jk
i
l
xx

+

g

kj

i

l

2

g
j

x

g

2

g

jl
i
k
xx

ik
x

l

j

l

ki

5.4 PROPERTIES  OF  RIEMANN-CHRISTOFFEL  TENSORS  OF  FIRST  KIND

lkjiR

(i)

R

jikl

(ii)

R

ijlk

-=

-=

R

ijkl

R

ijkl

(iii)

(iv)

ijkl

R

R

=
klij R
+

R

ijkl

+

R

iljk

0=

iklj

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
b
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
a
b
-
a
b
a
b
a
b
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
a
b
a
b
92

Tensors and Their Applications

Proof: We know that from equation (9), Pg. 91.

ijklR

=

1
2

+

2

j

g

il
x

k

x

+

2

g
i
xx

jk
l

2

2

x

g
j

ik
x

l

2

g

jl
k

i
xx

[
il

,

]

kj

[
ik

,

]

j

l

...(1)

(i)

Interchanging of i and j in (1), we get

jiklR

=

1
2

2

g

jl
k
i
xx

=

2

x

g
j

ik
x

l

2

g

jk
l
i
xx

2

j

x

g

il
x

k

+

ki

[

jl

,

]

li

[

jk

,

]a

2

g

jl
i
k
xx

=

1
2

2

j

g

il
x

k

x

+

2

g
jk
i
l
xx

jiklR

=

ijklR-

ijklR

=

jiklR-

2

g
j

x

+

ik
x

l

ki

[

jl

,

]

[

jk

,

]a

i

l

Interchange l and k in equation (1) and Proceed as in (i)
Interchange i and k in (1), we get

or

(ii)
(iii)

2

g

k

x

jl
x

i

2

g

k

x

lj
x

i

+

+

[
kl
,

]

ij

[
ki
,

]

lj

[
kj
,

]

l

i

[
ki
,

]

l

j

kjilR

=

1
2

2

g
j

x

+

kl
x

i

2

g

k

x

ji
x

l

2

g
j

x

ki
x

l

Now interchange j and l, we get

klijR

=

1
2

For

2

g

kj
l
xx

+

i

2

k

x

g

li
x

j

2

g

ki
l
xx

j

[
,kj

]

=

il

g

il

k

j

=

=

g

jk

il

[
,li

]b

jk

[
,li

]b

=

[
,li

]

jk

il

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
Riemann-Christoffel Tensor

93

So,

klijR

=

1
2

2

g

kj
i
l
xx

+

2

k

x

g

li
x

j

2

g

ki
l
xx

2

g

k

x

lj
x

i

[
li

,

]

+

jk

[
ki
,

]

l

j

j

klijR

=

ijklR

,

 from (1)

from equation (1)

ijklR

=

ikljR

=

iljkR

=

1
2

1
2

1
2

2

g

j

x

+

il
x

k

2

g

k

x

+

ij
x

l

2

g

jk
l
i
xx

2

g

kl
i
xx

j

2

g

j

x

jk
x

l

2

g

k

x

il
x

j

2

g

j

x

2

g

kj
l
i
xx

+

jl
k
x

kj

[
il

,

]

[
ik

,

]

lj

[
ij

,

]

+

lk

[
il

,

jk

2

g

ik
l
xx

+

j

2

g

lj
k
i
xx

2

g

ij
k
l
xx

2

g

lk
i
xx

+

j

l

j

[
ik

,

]

[
ij

,

kl

]

]

On adding these equations, we get

R

ijkl

+

R

iklj

+

R

iljk

= 0

This property of

ijklR  is called cyclic property..

Theorem 5.3 Show that the number of not necessarily independent components of curvature tensor

does not exceed

1
12

2
2
-nn
(

)1

.

Or
Show  that  number  of  distinct  non-vanishing  components  of  curvature  tensor  does  not  exceed
1
12

2
2
-nn
(

)1

.

Proof: The distinct non-vanishing components of

ijklR  of three types.

(i) Symbols with two distinct indices i.e.,

ijijR . In this case total number of distinct non-vanishing

components of

ijklR  are

1
2

-nn
(

)1

.

(ii) Symbols  with  three  distinct  indices  i.e.,

ijikR .  In  this  case,  total  number  of  distinct  non-

vanishing components

ijklR  are

1
2

nn
(

()1

n

)2

.

(iii) Symbols

ijklR  with four distinct indices. In this case, total number of distinct non-vanishing

components of

ijklR  is

( 2
2
-nn
12

)1

.

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
-
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
-
-
94

Tensors and Their Applications

Hence the number of distinct non-vanishing components of the curvature tensor

ijklR  does not

2
2
-nn
(

)1

.

exceed

1
12
Remark

When

ijklR  is of the form

iiiiR  i.e., all indices are same

In this case,

iiiiR  has no components.

5.5 BIANCHI  IDENTITY

It states that

R

+

R

i
,
jlm

k

+

R

i
jmk

,
l

= 0

i
mjkl
,
+

and

R

hjkl

,

m

R

,
hjlm

k

+

R

hjmk

,
l

= 0

Proof: Introducing geodesic coordinate1 in which Christoffel symbols are constant with the pole at  0P .

Since we know that

i
jklR

=

i
jklR

=

k

x

i

jk

i

jl
k

x

l

x
i

jl

+

i

jk
l

x

i

mk

m

jk

i

ml

m

jl

+

i

mk

m

jl

m

jk

i

ml

i

mjklR , =

2

i

lj

m

x

k

x

i

2

kj
l
m

x

x

...(1)

Since

i

kj

,

m

lj

 etc. are constant at pole.

So, their derivatives are zero.

i

l

i

m

l

x

i

m

x

i

+

i
jlmR

=

lj

mj

j

l

mj

1 Details of geodesic coordinate given in chapter curvature in curve . Geodesic.

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
-
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
Riemann-Christoffel Tensor

i

mj
l

x

=

i

l
m

j

x

+

i

l

i

mj

m

j

l

and

i

jlmR , =
k

i

2

mj
l

x

k

x

i

lj
m

x

2

k

x

i
jmkR

=

m

x
i

mj

+

k

x
i

kj

i

m

i

k

mj

kj

i

kj
m

x

i

2

kj
l
m

x

x

i
jmkR

=

i
jmkR

=

i

mj
k

x

i

+

i

2

mj
l

x

k

x

i

kjm

k

mj

On adding (1), (2) and (3), we get

R

i
,
mjkl

+

R

i
,
jlm

k

+

R

i
jmk

,
l

= 0

Multiplying

i
mjklR ,

 by

hig  i.e.,
i
mjkl

hi Rg

, =

hjklR ,

m

Then equation (4) becomes

R

hjkl

,

m

+

R

,
hjlm

k

+

R

hjmk

,
l

= 0

95

...(2)

...(3)

...(4)

...(5)

Since every term of equation (4) and (5) is a tensor. So, equation (4) and (5) are tensor equations
0P  is an arbitrary point of  nV . Thus there hold

and therefore hold in every coordinate system. Further,

throughout

nV . Hence equation (4) or (5) is called Bianchi identity..

5.6 EINSTEIN  TENSOR

Theorem 5.4 To prove the tensor

R

i
j

1
2

i
j

R

 is divergence free.

Proof: We know that from equation (5)

(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
-
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
-
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
d
-
96

Tensors and Their Applications

R

hjkl

,

m

+

R

,
hjlm

k

+

R

hjmk

,
l

= 0

Multiply it by

hl gg

,jk

 we get

hl

Rgg

jk

+

,
mhjkl

hl

Rgg

jk

+

,
hjlm

k

hl

Rgg

jk

hjmk

jk
Rg

mjk
,

+

hl
gg

jk

(

R

hjml
,

k

+

)

hl
gg

jk

(

R

jhmk

,

l

Since

R

hjlm

-=

R

hjml

 &

R

hjmk

-=

R

jhmk

jk
Rg

,
mjk

jk
Rg

jm

,

k

hl
Rg

,
hm

l

= 0

RmR

,

k
,
km

R

l
,
lm

= 0 Since

jk
Rg

jk

=

R

=  0

= 0

,

l

)

RmR

,

k
,
km

R

k
,
km

= 0

,
mR

2

k
kmR
,

= 0

R k
,
km

1
2

,
mR

= 0

R

k
,
km

R

k
m

1
2

1
2

k
m

,
kR

= 0

since

,
mR

d=

k
m ,
kR

k
m

R

= 0

,

k

k
R
m

1
2

k
m

R

 is divergence free.

The tensor

k
R
m

1
2

k
m

=
k
GR
m

 or

R

i
j

1
2

i
j

R

 is known as Einstein Tensor..

5.7 RIEMANN  CURVATURE  OF  A  Vn
Consider two unit vectors  pi and  qi  at  a  point  P0 of  Vn.  These  vectors  at  P0  determine  a  pencil  of
directions deferred by ti = a pi + b qi. a
 being parameters. One and only one geodesic will pass
ip . Similarly one and only one geodesic will pass through in the direction
through P0 in the direction of
qi. These two geodesics through P0 determined by the orientation of the unit vectors pi and qi. Let this
surface is denoted by S.

 and b

The Gaussian curvature of  S at P0 is defined to be the Riemannian Curvature of  nV  at

0P  for the

orientation determined by pi and qi.

Let  the  coordinates

iy  of

nV   are  Riemannian  coordinates  with  origin  at  P0.  The  equation  of

surfaces S in given by

i.e.,

iy =

(

p

i

i
b+a
q

)

s

iy =

1

i
up

+

2

i
uq

...(1)

...(2)

-
-
-
-
-
-
-
-
-
-
d
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
d
-
d
-
d
-
d
-
Riemann-Christoffel Tensor

97

1u  and

where

s

=
2u . Here
ds 2
b=
b

Let

1

and
u
1u  and

=

2

,

u

 three parameters namely

s
2u  are coordinates of any current point on S.

b
,

s,

 can be reduced to two parameters

du

du

 be the metric for the surface  S. where

bb

=

g

ij

i

y

u

j

y

u

 (a

, b

 = 1, 2)

Let

Let

dR

k

and

 be the Christoffel symbols corresponding to the coordinates yi and ua  .

j

i
 and R hijk be curvature tensor corresponding to the metrices bab duadub and

g

dyg
ij

j

i

dy

.

Since  the  Greek  letters

b,

d,g,

  take  values  1,2  and  so  that  the  number  of  independent  non-

vanishing  components  of

R

transform the coordinate system

1
 is
12
au  to

2
(
nn
�u

2

)1

  for

,2=n

  i.e.,  they  are

(
2
22

2

)
1

=

1
12

 and suppose that the corresponding value of

.1

  Let  us

1212R

  are

'
1212R

.

Then

'
1212R

=

R

u

u

1

u

u

2

u

u

1

u

u

2

=

R
1

=

R
12

1

u

1

u

1

u

1

u

u

u

2

2

u

2

u

u

u

1

u

u

1

u

u

2

u

2

u

+

R
2

+

R

21

= 1212R

1

u

1

u

2

u

2

u

1

u

1

u

2

u

2

u

+

R

1221

2

u

1

u

2

1

u

u

1

u

1

u

u

u

2

1

u

2

u

2

u

2

u

u

u

u

u

1

1

2

u

1

u

2

u

u

u

2

u

1

u

2

u

+

R

2112

2

u
11

u

1

u

2

u

1

u

2

u

2

u

2

u

+

R

2121

2

u

1

u

1

u

2

u

2

u

1

u

1

u

2

u

R

2112

=

1

u

1

u

2

u

2

u

2

1

u

1

u

2

u

2

u

2

2

u

u

+

u

2

u

2

u

1

u

1

u

2

u

2

=

R
1212 J

2

 where

J

=

1

1

u
u

2

u

1

u

1

2

u
u

2

u

2

u

=

u
u

so,

1212R �

=

R
1212 J

2

...(3)

b
a
a
b
a
a
a
b
a
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
g
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)

a
b
g
a
-
a
b
g
d
-
(cid:215)
a
�
�
�
�
�
�
�
�
�
�
�
�
d
g
b
a
a
b
g
d
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
d
g
b
b
g
d
d
g
b
b
g
d
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
d
g
g
d
d
g
g
d
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
-
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
98

Again

or

from (3) and (4), we get

u

u

u

u

u

u

u

u

�b

�b

=

b

=

b

b� =

2bJ

R
1212 =
b

R

=
1212 K
b

(
say

)

Tensors and Their Applications

...(4)

...(5)

This shows that the quantity K is an invariant for transformation of coordinates. The invariant K

is defined to be the Guasian curvature of S. Hence K is the Riemannian Curvature of S at

0P .

Since Riemannian Coordinates  yi with the origin at  0P . We have as geodesic coordinates with the

pole at

0P .
Therefore

Then

and

k

i

j

,

g

= 0 at

0P

b

hijkR

=

dR

=

1212R

=

[
,
hij

]

g

k

y

+

]

g

[
,
hik
j

y

 at

0P .

[

]

a,

b

+

[

a,

]

b

u
[
]
1,21
2

u

b

+

u
[
]
1,22
1

b

u

at

0P

 at

0P .

from (5) we get

K =

1
b

]
[
1,21
2

u

b

+

]
[
1,22
1

b

u

...(6)

...(7)

This is required expression for Riemannian curvature at

0P .

5.8 FORMULA  FOR  RIEMANNIAN  CURVATURE  IN  THE  TERMS  OF  COVARIANT

CURVATURE  TENSOR  OF  Vn

Let

k

i

j

and

g

 be the Christoffel symbols of second kind relative to the metrices

b

du

du

a
b
b
d
a
g
a
b
�
�
�
�
�
�
(cid:222)
a
b
b
d
a
g
a
b
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
g
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
-
a
b
g
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
b
d
�
�
b
g
�
-
g
d
�
�
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
g
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
a
b
99

...(8)

...(9)

...(10)

Riemann-Christoffel Tensor

and

dyg
ij

j

i

dy

 respectively. We have

[

g,

]

b

=

Now,

[
]
b1,21

[
]
1,21
2

u

b

Interchanging h and k, we get
[
]
1,21
2

b

u

Similarly,

[
]
1,22
1

u

b

=

=

=

=

=

=

=

i

y

u

i

y

2

u

j

y

u

j

y

1

u

k

y

u

k

y

1

u

[
,
kij

]

g

[
,
kij

]

g

i

j

ppq

k

[
,
kij

]

g

 using (2)

]

g

[
,
kij
2

u

j
i
ppq

k

]

g

[
kij
,
h

y
[
,
kij
h

y

h

y

2

u

]

g

]

g

[
,
hij
k

y

]

g

[
,
hij
k

y

i

j

ppq

k

i

j
qppq

k

i

k
ppqq

j

i

k
ppqq

j

h

h

h

k

Using (9) and (10), equation (6) becomes

1212R

=

1212R

=

Since

h

i
qpqp

j

i

h

j
Rqpqp

k

hijk

[
hij
,

]

g

k

y

+

]

g

[
hik
,
j

y

at

P
0

at P
0

...(11)

bb

=

g

ij

11b =

g

ij

i

y

u

i

y

1

u

11b =

h

ppg

hj

j

y

u

j

1

y

u

j

=

i

ppg

ij

j

a
b
g
b
a
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
-
a
b
a
�
�
�
�
�
�
�
�
100

Similarly

22b

=

i
qqg
ij

j

=

i

qqg
ik

k

12b =

i

qpg
ij

j

=

j

qpg
ji

i

=

h

qpg

hk

Tensors and Their Applications

k

b
11

b
12

b
21

b

22

=

bb
11

22

bb
12

21

kj
ih
ggqpqp
hi

[

]hk

gg
ij

ik

b =

b =

Dividing (11) and (12), we get

K =

R

1212
b

=

k

ih

Rqqp
k

(

gg
ik

hj

hijk

ih

qpqp

j

...(12)

...(13)

gg
ij

hk

)

This is formula for Riemannian Curvature of  nV  at
and

iq  at

0P .

0P  determined by the orientation of Unit vectors

ip

5.9 SCHUR�S  THEOREM

If at each point, the Riemannian curvature of a space is independent of the orientation choosen then it
is constant throughout the space.

Proof: If K is the Riemannian curvature of

nV  at P for the orientation determined by unit vectors

ip

and

iq  then it is given by

gg
ik
Let K be independent of the orientation choosen. Then equation (1) becomes

qpqp

gg
ij

hk

hj

)

(

K =

h

j
Rqpqp

k

i

hijk
ih

j

k

K =

hijkR

=

R

hijk

gg
hj
ik
ggK
(

ik

hj

gg
ij

hk
gg
ij

)

hk

...(1)

...(2)

We have to prove that K is constant throughout the space

nV .

If

,2=N

 the orientation is the same at every point. So, consider the case of

nV  when

.2>n

Since

ijg  are constants with respect to covariant differentation, therefore covariant differentation

of (2) gives

hijkR , =
l

(

gg
ik

hj

Kgg
hk

)

ij

,
l

...(3)

where K,l is the partial derivative of K.
Taking the sum of (3) and two similar equations obtained by cyclic permutation of the suffices,

j, k and l.

-
-
-
-
-
-
-
Riemann-Christoffel Tensor

101

(

gg
hj

ik

Kgg
ij

hk

)

+

,
i

(

gg
hk

il

Kgg
ik

hl

)

+

,

j

(

gg
hl

ij

Kgg
il

hj

)

,

k

 =

R

hijk

,

l

+

R

hikl

,

j

+

R

hilj
,

k

...(4)

Here

2>n
Multiplying (4) by

 therefore three or more distinct values to indices j, k, m can be given .
,j
l

hjg  and using

 we get

hj gg

d=

hl

(

ng

ik

d

h
i

g

hk

,)
k

l

+

(

g

d

j
k

il

g

ik

d

l

j

)

K

,

+

d
(

h
i

j

lg
h

ng

il

)

K

,

k

 = 0

or,

(

n

)1

Kg
ik

,

l

-++
1(0

)
Kgn
il

,

k

 = 0 for

i
j

=

,0

i

j

Kg
ik

,
l

Kg
il

,

k

 = 0

Multiplying by

ikg  and using

we get

or

or

ikg d

=

il ggn
,

ik

ik

d=

k
l

,

,
nK

l

k
l

K

,

k

= 0 or

(

n

(

n

kK, = 0 as
K
lx

= 0.

)1

)1

lK
,
lK
,

=

=

0

0

integrating it, we get K = constant. This proves that the partial derivatives of K w.r.t. to x�s are all zero.

Consequently K is constant at P. But P is an arbitrary point of

nV . Hence K is constant throughout  nV .

5.10 MEAN  CURVATURE

The sum of mean curvatures of a
the ennuple choosen. Obtain the value of this sum.

nV  for a mutually orthogonal directions at a point, is independent of

Or

Prove that the mean curvature (or Riccian Curvature) in the direction

nV  is the
sum of  n � 1 Riemmanian curvatures along the direction pairs consisting of the direction and  n  �  1
other directions forming with this directions an orthogonal frame.

ie  at a point of a

Proof: Let

i
he

 be the components of unit vector in a given direction at a point P of a  nV . Let

i
ke

 be the

components of unit vector forming an orthogonal ennuple.

Let the Riemannian curvature at P of

nV  for the orientation determined by

i
hl

 and

e i
k

(
h

)k

 be

denoted by

hkK  and given by

hkK =

p
ee
h
q
p
eeee
k
h

s
k

r
h

q
k
(

r
ll
k

s
k

R

pqrs

qg
pr

qs

g

g

qr

)

ps

=

(

r
p
gee
h
h

()

pr

s
k
1

q
k
1

p
h
1
s
gee
k

r
Reeee
h
1
p
h

pqrs
s
gee
k

q
k

qs

(

)

()

r
gee
h

q
k

)

qr

ps

...(1)

-
-
-
-
-
-
-
�
d
-
d
-
-
-
�
�
�
-
-
102

Tensors and Their Applications

Since unit vectors

i
he

l,

i
k

 are orthogonal. Therefore

and

r
gee
h

p
h

pr

= 1

s
gee
h

p
h

ps

= 0 etc.

Using these in equation (1), then equation (1) becomes

hkK =

hkK =

r
Reeee
h

s
k

p
h

q
k

pqrs

11

0

0

r
Reeee
h

s
k

p
h

q
k

pqrs

...(2)

n

=

1

k

n

=

1

k

Put

hkK

n

= (cid:229)

k

=

1

p
r
Reeee
h
h

q
k

s
k

pqrs

K

=
hk M

.  Then

h

hM =

p
ee
h

r
h

n

k

=1

s
q
Ree
k
k

pqrs

=

=

r
Rgee
h

qs

p
h

pqrs

r
Rgee-
h

qs

p
h

qprs

This shows that

hM  is independent of (n � 1) orthogonal direction choosen to complete an orthogonal

hM =

p
Ree-
h

r
h

pr

...(3)

ennuple. Here

hM  is defined as mean curvature or Riccian curvature of
 to

Summing the equation (1) from

 we get

h =

,n

1=h

nV  for the direction

p
he 1 .

n

=

1

h

n

=

1

h

hM

=

p
r
h Ree
|
|
h

pr

=

pr Rg-

pr

= �R

hM

=

pr
Rg

pr

-==

R

or

This proves that the sum of mean curvatures for n mutual orthogonal directions is independent of the
directions chosen to complete an orthogonal ennuple and has the value

.R-

5.11 RICCI�S  PRINCIPAL  DIRECTIONS

Let

i
he 1  is not a unit vector and the mean curvature

hM  is given by

hM =

i
eeR
h

ij

j
h

i
eeg
h

ij

j
h

�
-
�
(cid:229)
(cid:229)
(cid:229)
(cid:229)
-
(cid:229)
-
Riemann-Christoffel Tensor

(

R

ij

+

gM
h

ij

)

i
ee
h

j
h

= 0

Differentiating it w.r. to

i
he
,1

 we get

h

M
i
e
h

eeg
jk

j
h

k
h

+

(2

R

ij

+

)
eMg
ij

j
h

 =  0

For maximum and minimum value of

hM .

h

M
i
e
h

= 0

Then equation (1) becomes

(

R
ij

+

i
)
eMg
ij
h

= 0

103

...(1)

These are called Ricci's Principal direction of the space as they are principal directions of Ricci

tensor

.ijR

5.12 EINSTEIN  SPACE

A space, which is homogeneous relative to the Ricci tensor

ijR  is called Einstein space.

If space is homogeneous then we have
ijgl

ijR =

Inner multiplication  by

,ijg

 we get

R = nl

 since

gR ij
ij

=  and

R

ij
gg

=

n

ij

...(1)

from (1)

l =

ijR =

1
n

R
n

R

ijg

Hence a space is an Einstein space if

R =
ij

R
n

g

 at every point of the space.

ij

Theorem 5.5 To show that a space of constant Curvature is an Einstein space.

Proof: Let the Riemannian curvature  K  at  P of

nV  for the orientation determined by

ip  and

,iq   is

given by

K =

i

k

h

j
Rqpqp
j

k

(

gg
ik

hj

ih

qpqp

hijk

gg
ij

hk

)

Since K is constant and independent of the orientaion.

(cid:222)
�
�
�
�
(cid:222)
-
104

Tensors and Their Applications

K =

R

hijk

(

gg
ik

hj

gg
ij

hk

)

hijkR

=

ggK
(

ik

hj

gg
ij

hk

)

Multiplying by

hkg

hk

Kg

(

gg
ik

hj

gg
ij

hk

)

=

hk Rg

hijk

K

(

h
i

g

hj

ng

)

ij

= ijR

Since

hk gg

d=

ik

;h
i

hk
gg

hk

=  and

n

hk
Rg
hi

jk

=

R
ij

gK
(

ij

ng

)

ij

= ijR

K

1( -

)
ijgn

= ijR

Multiplying by

,ijg

 we get

from (1) & (2)

Kn -
1(

n

)

= R as

ij
Rg
ij

=

R

...(1)

...(2)

ijR =

1(

)
gn

ij

R
(
1

n

=

)

n

1
n

Rg

ij

ijR =

R
n

ijg

This is necessary and sufficient condition for the space

nV  to be Einstein space.

5.13   WEYL  TENSOR  OR  PROJECTIVE  CURVATURE  TENSOR

Weyl Tensor denoted as

hijkW  and defined by

hijkW =

R

hijk

1

+

1

n

(

Rg
hj
ki

gg
kh

ij

)

Theorem 5.6 A necessary and sufficient condition for a Riemannian

)3>nVn
(

  to  be  of  constant

curvature to that the Weyl tensor vanishes identically throughout

nV .

Proof: Necessary Condition:

Let K be Riemannian Curvature of

nV . Let K = constant.

We have to prove that

hijkW

0=

Since we know that

h

j
Rqpqp

k

i

K =

(

gg
hj

ik

gg
ij

hk

)

hijk
ih

qpqp

j

 = constant

k

-
-
-
-
d

-
-
(cid:215)
-
(cid:222)
-
-
-
Riemann-Christoffel Tensor

Since K be independent of the orientation determined by the vector

ip  and

iq .

Then

K =

R

hijk

gg
hj

ik

gg
ij

hk

Multiplying by

,hkg

 we get

hk Rg

hijk

=

hk

ggKg
(

hj

ik

gg
ij

hk

)

ijR =

K

(

k
j

g

ik

ng

)

ij

ijR =

K

1( -

)
ijgn

Multiplying by

ijg  again, we get

ij Rg
ij

=

K

1( -

ij

)
ij ggn

R =

K

1( -

)
nn

Putting the value of K from (3) in (2), we get

ijR =

R
n

ijg

The equation (3) shows that R is constant since K is constant.
Now, the  W tensor is given by

hijkW =

R

hijk

1

+

1

n

[

Rg
ik

hj

gg
hk

ij

]

from (5), we get

105

...(1)

...(2)

...(3)

...(4)

hijkW =

R

hijk

+

R

hijk

=

+

1

1

n

R

g

ik

R
n

g

hj

g

hk

R
n

g

ij

[

gg
ik

hj

gg
hk

ij

]

n

1(

n

)

=

R

hijk

+

R

n

1(

n

)

R

hijk
K

,

 by. eqn. (1)

hijkW =

R

hijk

+

K

R

hijk
K

hijkW =

hijkR2

,

 by equation (3)

Since K is constant. The equation (5) shows that

hijkW

0=

.

This proves necessary condition.

-
-
-
d
-
-
�
�
�
�
�
�
-
-
-
-
(cid:215)
-
106

Sufficient  Condition

Let

hijkW

.0=

 Then we have to prove that K is constant.

Tensors and Their Applications

Now,

hijkW

.0=

R

hijk

+

Multiplying by

1

n
1
,hkg

 we get

[

Rg
ik

hj

Rg
hk
ij

]

= 0

+

R
ij

1

1

n

[

hk

Rgg
hj

ik

hk

Rgg
ij

hk

]

= 0

+

R
ij

1

1

n

[

h
i

R
hj

nR
ij

]

= 0

+

R
ij

R
ij
n

1

1(

n

)

= 0

ijR2

= 0

ijR = 0

=

.0

hijk
0=

hijkR

Since

(cid:222)=
0

hk
Rg

R
ij
hkg

 = 0 or
,0=
hijkR
0=hkg

 then

If

If

 then clearly K = 0. So, K is constant.

K =

R

hijk

(

gg
hj

ik

k

j

ih

qpqp
ih

0

g

ij

)

qpqp

j

k

R
h

hijk
j

gpp

j

k

h

i
qpqp
k
gqq

()

i

hj

)

ik

K =

(

R

hijk

=

k

j

h

i
qpqp
2
2

p

q

K =

R

hijk

ih

qpqp

j

k

 since

2

p

= q
,1 2

=

1

K = constant as

ijR

0=

This proves sufficient condition.

EXAMPLE  1

For a

2V  referred to an orthogonal system of parametric curves (g12 = 0) show that
=

=

12R = 0,

gR
11

22

gg
22

11

R
1221

Consequently

R =

ij
Rg
ij

=

ijR =

1
2

ijRg

.

1221

R
2
gg
11

22

(cid:222)
-
-
-
-
-
d
-
-
-
(cid:222)
(cid:222)
(cid:222)
-
(cid:215)
Riemann-Christoffel Tensor

Solution

Given that

=g
12

0

 so that

12 =g

.0

Also,

ijg =

1
g ij

11

=

g

1
g

11

22

=

g

 &

1
g

22

.

The metric of

2V  is given by

2ds =

dxg
ij

i

dx

;j

 (i, j = 1, 2)

2ds =

g

11

(

dx

21
)

+

g

22

(

dx

22
)

 Since

=g
12

.0

We know that

hk
Rg
1

hijk

=

R
ij

.

and

(i) To prove

=R
12

0

g =

g ij

=

g

g

11

21

g

12
g

22

=

g

11
0

0

g

22

=

gg
11

22

hk
Rg
h
12

=

1
h
Rg
h

k

21Rg

 as

R
1121

2121

 as

hR

122

=

0

0

2
=

12R =
=
12R = 0
=
R
1221

11
11R =

gR
22

hk
Rg
h

11
k

=

2
k
Rg

211
k

11R =

22
Rg

2112

=

R
2112
g

22

11R =

22R =

R
2112
g
22
hk
Rg
h

=

1
k
Rg

122
k

22
k

11
Rg

1221

=

=

R
1221
g

11

22R =

R
1221
g

11

(ii) To prove

gR
11

22

=

So,

and

So,

from (1) and (2)

(iii) To prove

R11g22 = R1221=R22g11

R =

1221

R
2
gg
11

22

R =

ij
Rg
ij

=

1
i
Rg
i

1

+

2
i
Rg
i

2

107

...(1)

...(2)

� (3)

(cid:222)
108

Tensors and Their Applications

=

=

=

R =

R =

R =

(iv) To prove

R
ij

1=
2

Rg

ij

11
Rg

11

+

22
Rg

22

 for

12 =g

0

R
g

22

22

+

11

R +
g

11

11

R
1221
gg
22
R
2
gg
11

1221

22

1221

[
R
Qgg

22

11

by

eqn.

]3)(

22

1221

R
2
gg
11
R
1221
g

2

 as

g

=

gg
11

22

1221R
The eqn (3) expressed as

=

11gR

22

=

it becomes

1
2

1
2

Rg

Rg =

gR
22

11

.

11R =

22R =

=

=

Rg
2
g

22

Rg
2
g

11

So,

gRg
11
g
2

22

22

=

g

22

=

Rg
11
g
2

11

Rg
11
2

Rg
22
2

11R =

12R =

1
2
1
2

Rg  &
11

R =
22

1
2

g

22

g  as
12

R
12

==
0

g

12

This prove that

R =
ij

1
2

R
ij

.

EXAMPLE  2

The metric of the V2 formed by the surface of a sphere of radius r is

=2ds

2
dr

2

+

2

r

sin

2

in spherical polar coordinates. Show that the surface of a sphere is a surface of constant curvature

2

.

d
1
2r

Solution
Given that

sin
Since r is radius of curvature then r is constant.

r

2ds =

2
dr

2

+

2

2

2

d

f
q
q
f
q
q
Riemann-Christoffel Tensor

109

g

11

=

2

r

,

g

22

=

2

r

sin

2

,

g

12

=

,0

=

g

gg
11

22

=

4

r

sin

2

.

We can prove

Now, the Riemannian curvature K of Vn is given by

R1221 = r2 sin2 q

K =

j

ih

qpqp
k

(

gg
hj

ik

ih

qpqp

j

k

.

gg
hk

ij

)

At any point of

2V  there exists only two independent vectors.
Consider two vectors whose components are (1, 0) and (0, 1) respectively in

2V . Then

K =

R
1212 =
gg
22
11

R

1212
g

K =

2

4

r

r

sin

sin

2

2

=

1
2

r

=

constant.

EXERCISES

ijklR

=

[

],
jl
i
k
x

[

],
i
l

jk
x

+

[
il

,

]

ik
[

,

]

kj

lj

2

g

il
j
k
xx

+

2

g
i
xx

jk
l

2

g
ik
j
xx

l

2

g

jl
k

i
xx

+ a
g

(

[

jk

il
,[],

]

[

jl

[],

ik

,

)]

.

1. Show that

2. Show that

ijklR

=

1
2

3. Using the formula of the problem 2. Show that

R

ijkl

-=

R

jikl

-=

R

ijlk

=

R

klij

  and

R

ijkl

+

R

iklj

+

R

iljk

0=

4. Show that the curvature tensor of a four dimensional Riemannian space has at the most 20 distinct

non-vanishing components.

5. (a) If prove that the process of contraction applied to the tensor

h

ijkR  generates only one new tensor

ijR  which is symmetric in i and j.

(b)

If

2
ds

=

g
11

(

21
)

dx

+

g

dx
(

22
)

+

g

(

dx

23
.)

33

22

 Prove that

R =
ij

1
g

hh

R

ihhj

,

 (h, i, j being unequal).

6. Show that when in a V3 the coordinates can be chosen so that the components of a tensor gij are zero

when i, j, k are unequal then

(i)

R

hj

1=
g

ii

R

hiij

q

q
-

q
q
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
-
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
�
-
�
�
(cid:247)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
-
�
�
�
-
�
�
�
�
�
�
a
b
-
a
b
b
110

Tensors and Their Applications

(ii)

R

hh

=

1
g

ii

+

R

hiih

1
g

jj

R

hjjh

7. Prove that if

iR =

j Rg
ij

 then

R
i

,

=

1
2

R
i

x

and hence deduce that when n > 2 then scalar curvature of an Einstein space is constant.

8. If the Riemannian curvature K of  Vn at every point of a neighbourhood U of Vn is independent of the
direction chosen, show that K is constant throughout the neighbourhood U. Provided n > 2.

9. Show that a space of constant curvature K0 is an Einstein space and that R = K 0n (1 �n ).

10. Show that the necessary and sufficient condition that  nV  be locally flat in the neighbourhood of 0 is

that Riemannian Christoffel tensor is zero.

11. Show that every V2 is an Einstein space.
12. For two dimensional manifold prove that

K =

R-
2

13. Show that if Riemann-Christoffel curvature tensor vanishes then order of covariant differentiation is

commutative.

a
a
�
�
a
a
CHAPTER � 6

THE  e-SYSTEMS  AND  THE  GENERALIZED

KR�NECKER  DELTAS

The  concept  of  symmetry  and  skew-symmetry  with  respect  to  pairs  of  indices  can  be  extended  to
cover to pairs of indices can be extended to cover the sets of quantities that are symmetric or skew-
symmetric  with  respect  to  more  than  two  indices.  Now,  consider  the  sets  of  quantities
  or
iA ...
represent tensor.

 depending on k indices written as subscripts or superscripts, although the quantities A may not

iA ...

ki

i

k

l

6.1 COMPLETELY  SYMMETRIC

iA ...1
The system of quantities
if the value of the symbol  A is unchanged by any permutation of the indices.

) depending on k indices, is said to be completely symmetric

(or

ki

iA ...1

ki

6.2 COMPLETELY  SKEW-SYMMETRIC

ki

iA ...1

The systems
) depending on k indices, is said to be completely skew-symmetric if the
value of the symbol A is unchanged by any even permutation of the indices and A merely changes the
sign after an odd permutation of the indices.

 or (

iA ...1

ki

Any permutation of n distinct objects say a permutation of n distinct integers, can be accomplished
by  a  finite  number  of  interchanges  of  pairs  of  these  objects  and  that  the  number  of  interchanges
required to bring about a given permutation form a perscribed order is always even or always odd.

In any skew-symmetric system, the term containing two like indices is necessarily zero. Thus if

one has a skew-symmetric system of quantities

ijkA  where i, j, k assume value 1, 2, 3. Then

122A

123A

=

=

0

=A
112
213A-

,

=
312 A
A

123

 etc.

In general, the components

ijkA  of a skew-symmetric system satisfy the relations.

ijkA =

-=

A
ikj

A

jik

ijkA =

A =
jki

A

kij

-
112

6.3 e-SYSTEM

Tensors and Their Applications

Consider a skew-symmetric system of quantities

ie ...1

ni

(

ie
or

1

,...,

ni

)

 in which the indices

i ...1

ni

 assume

values 1,2,...n. The system

ie ...1

ni

ie
or

...1

(

ni

)

 is said to be the e-system if

+=

;1

i
when
i
,
21

,...,

i

n

an

even

permutatio

 ofn

number
1,

2,

...,

n

e
i
1

...

i
n

i
e
or
(
1

...

i
n

)

-=

;1

i
when
i
,
21

,...,

i

n

an

 odd

permutatio

 ofn

number

2,1,

...,

n

=

0

in

 all

other

cases

EXAMPLE  1

Find the components of system eij when i, j takes the value 1,2.

Solution

The components of system eij are

e
11

,

e
12

,

e
21

,

e

.

22

By definition of e-system, we have
11e = 0,
12e = 1,
- e
21e =
12
22e

= 0,

indices are same
since i j has even permutation of 12

-=

1

since i j has odd permutation of 12

indices are same

EXAMPLE  2

Find the components of the system

jkie

.

Solution

By the definition of e-system,
123e
213e
ijke

=

=

e

231

e
132

= e
= e

321

321

=

1
-=

1

= 0 if any two indices are same.

6.4 GENERALISED  KR�NECKER  DELTA

A symbol
is called a generalised Kr�necker delta provided that

i
1
j
1

...
...

i
k
j
k

 depending on k superscripts and k subscripts each of which take values from 1 to n,

(a)

it is completely skew-symmetric in superscripts and subscripts

(cid:239)
(cid:239)
(cid:239)
(cid:239)
(cid:238)
(cid:239)
(cid:239)
(cid:239)
(cid:239)
(cid:237)
(cid:236)
d
The e-Systems and the Generalized Kr�necker Deltas

113

(b)

if  the  superscripts  are  distinct  from  each  other  and  the  subscripts  are  the  same  set  of
numbers as the superscripts.

The value of symbol

=

;1

i
...
...

i
1
j
1

k
j
k

-=

an

even

number

of

 transposi

tion

 is

required

 to

arrange

 the

superscrip

 ts

in the

same

order

 as

subscripts
.

1;

where

 odd

number

of

 transposi

tions

arrange

 the

superscrip

 ts

in the

same

order

 as

subscripts

 all

other

cases

 the

 value

of

 the

symbol

 is

zero

=

0,

in

EXAMPLE  3

Find the values of

.ij
kl

Solution

By definition of generalised Kronecker Delta,

ij
kl

(cid:215)=

0=

 if i = j or k = l or if the set. ij is not the set kl.

=(cid:215)

0

11
pq

= 22
pq

=

23
13

i.e.,

i.e.,

and

i.e.,

ij
kl

12
12

= 1 if kl is an even permutation of ij

=

d=

21
21

13
13

d=

31
31

d=

(cid:215)=

23
23

=(cid:215)

1

ij
kl

1-=

 if kl is an odd permutation of ij.

12
21

=

d=

31
13

13
31

d=

21
12

(cid:215)=

-=(cid:215)

1

Theorem 6.1 To prove that the direct product

ii
21

...

i

n

e

e

jj
21

...

j
n

 of two systems

ie ...1

ni

 and

jje

...21

nj

 is the

generalized Kr�necker delta.

Proof: By definition of generalized Kr�necker delta, the product

ii
21

...

i

n

e

e

jj
21

...

j
n

 has the following  values.

(i) Zero if two or more subscripts or superscripts are same.

(ii) +1,  if  the  difference  in  the  number  of  transpositions  of

i
1

, 2
i

,...,
ni

 and

j
1

, 2
j

,...,

nj

  from

1,2,...n is an even number.

(iii) �1, if the difference in the number of transpositions of i1, i2,...,in  and j1, j2, ...jn from 1, 2,..n

an odd number.

Thus we can write

ii
21

...

i

n

e

e

j
j
21

...

j

n

=

ii
21
jj
21

i
...
...

n
j

n

THEOREM 6.2 To prove that

(i)

iie

...21

ni

=

ii
21
jj
21

i
...
...

n
j

n

(ii)

iie

21

...

i

n

d=

ii
21
jj
21

...
i
n
...

j
n

�
�
�
�
�
�
�
�
d

d
d
d
d
(cid:215)
d
d
d
(cid:215)
d
d
d
(cid:215)
d
d
d
114

Tensors and Their Applications

Proof: By Definition of e-system,

...21

iie

ni

(

or

iie

...21

ni

)

 has the following values.

, 2
i
i
1
, 2
i

ni
,...,
,...,
ni

(i) +1; if

 is an even permutation of numbers 1,2,...n.

-1; if

(ii)
i
1
(iii) 0; in all other cases
Hence by Definition of generalized kr�necker delta, we can write

 is an odd permutation of numbers 1,2,...n

iie

21

...

i
n

d=

ii
...
i
21
n
...21
n

(1)

and

(2)

e
ii
21

...

i
n

d=

...21
n
ii
...
i
21
n

6.5  CONTRACTION  OF

jki
����

Let us contract

jki

 on k and  g

. For n = 3, the result is

ijk

=

1
ij

1

d+

ij

2

2

d+

ij

3

3

d=

ij

This expression vanishes if i and j are equal or if  a

 and  b

 are equal.

If i = 1, and j = 2, we get

123
.
3b

Hence

+

;1

 if

1;

 if

12

=

 is

an

even

permutatio

12 ofn

 is

an

odd

permutatio

12 ofn

0;

 if

 is

not

permutatio

12 ofn

Similarly results hold for all values of  a
Hence

 and  b

 selected from the set of numbers 1, 2, 3.

+

;1

1;

0;

ij

=

ji
 is  if

an

even

permutatio

 ofn

 if

ji

 is

an

 odd

permutatio

 ofn

if

 two

of

 the

subscripts

or

superscrip

 are ts

equal

or when th

 e

subscripts

and

superscrip

 are ts

not

formed

from

 the

same

numbers.

If we contract

ij

.

 To contract

a system depending on two indices

i

=

ij

1
2

 first contract it and the multiply the result by

.

1
2

 We obtain

=

ij

j

1
2

(

d+

i

1
1

i

2
2

d+

i

3
3

)

a
b
g
d
a
b
g
d
a
b
a
b
a
b
a
b
d
a
d
a
b
d
�
�
�
�
�
�
a
b

a
b
-

a
b

a
b
d
�
�
�
�
�
�
�
�
a
b
-
a
b

a
b
d
a
b
d
a
d
a
a
a
a
d
d
The e-Systems and the Generalized Kr�necker Deltas

115

It i = 1 in

i

 then we get

This vanishes unless

1=a

1

=

1
2
 and if a

(

d+

12
2

)13

3

 = 1 then

=
.11
1

Similar result can be obtained by setting i = 2 or i = 3. Thus

i

 has the values.

(i) 0 if

(,

=

,

i

),3,2,1

i
a=i

(ii) 1 if
By counting the number of terms appearing in the sums. In general we have

.

...(1)

...(2)

...(3)

...(4)

...(5)

We can also deduce that

i

=

1
- 1

n

ij

 and

j

=

ij
ij

nn
(

)1

...

ii
21
jj
21

i
...

r

j
r

=

(
(

kn
rn

)!
)!

ii
21
j
j
21

...
...

ii
rr
j
j
rr

1

i
k
...

...

1

j

k

and

or

...

ii
21
jj
21

i
...

=

nn
(

()1

n

)2

r

j
r

(

rn

=+
)1

n
!

n

r
!

ii
21

...

i

n

e

e

ii
21

...

i

n

= n!

and from (2) we deduce the relation

ii
21

...

ii
rr

+

...
1

i

n

e

e

jj
21

...

j

j
rr

+
1

...

i

n

= n!

EXERCISE

1. Expand for n = 3

i

(a)

j

(b)

j

xx12
i
ij

(c)

ij

j

i
yx

(d)

ij
ij

2. Expand for n = 2

(a)

ij aae
21
i
j

(b)

ij aae
12
i
j

(c)

ij
i
aae

j
=b

ij
.ae

3. Show that

ijk
ijk

!3=

 if i, j, k = 1, 2, 3.

4. If a set of quantities

iiA ...21

ki

 is skew-symmetric in the subscripts (K in number) then

i

i
...
1
j
...1

k
kj

A ...
i
1

i
k

=  k!

jA ...1

kj

5. Prove that

ijk

g

 is a covariant tensor of rank three where where

is the usual permutation

ijk

symbol.

a
d
a
a
a
d
d
d
a
d
a
a
�
a
d
a
d
-
d
d
-
-
d
-
-
d
-
-
(cid:215)
(cid:215)
(cid:215)
-
-
a
a
d
d
d
a
b
d
d
a
d
d
e
e
CHAPTER � 7

GEOMETRY

7.1 LENGTH  OF  ARC

Consider the n-dimensional space R be covered by a coordinate system X and a curve  C so that
i =

...(1)
which  is  one-dimensional  subspace  of  R.  Where  t  is  a  real  parameter  varying  continuously  in  the
interval

 The one dimensional manifold C is called arc of a curve.

ixC :

...,2,1

(txi

),

=

n

(

)

t

t

t

1

.2

dxi
dt
2

dx
dt

Let  F

1
,
xx

2

,...,

x

n

,

1
dx
dt

2

dx
dt

,...,

n

dx
dt

assume that

xF
,

dx
dt

>(cid:247)

,0

 unless every

1
xxF

,

2

,...,

n
,
kx

1
dx
dt

,

k

The integral

  be  a  continuous  function  in  the  interval

t

1

t

t

.2

  Wee

0=

 and that for every positive number k

dx
dt

,...,

k

n

dx
dt

kF

  =

2

1
,
xx

,...,

x

n

,

1
dx
dt

,

2

dx
dt

,...

n

dx
dt

.

is called the length of C and the space  R is said to be metrized by equation (2).

s = (cid:242)

t

2

t

1

xF
,

dx
dt

dt

...(2)

Different choices of functions

xF ,

 lead to different metric geometrices.

If one chooses to define the length of arc by the formula

s = (cid:242)

t

2

t

1

g

pq

x
)(

p

dx
dt

q

dx
dt

dt

,

 (p,q = 1, 2, ..., n)

...(3)

where

g

pq

)(
x

p

xd
dt

q

xd
dt

 is a positive definite quadratic form in the variable

dx p
dt

,

  then  the  resulting

geometry is the Riemannian geometry and space R metrized in this way is the Riemannian n-dimensional
space Rn.

�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
Geometry

117

Consider the coordinate transformation

xT
:

i

=

1
i
xx
(

,...,

x

n

)

 such that the square of the element

of arc ds,

can be reduced to the form

2ds =

g

pq

dx

p

q

dx

2ds =

i xdxd

i

...(4)

...(5)

Then the Riemannian manifold
nR  is said to reduce to an n-dimensional Euclidean manifold E n.
The Y-coordinate system in which the element of arc of C in En is given by the equation (5) is
called  an  orthogonal  cartesian  coordinate  system.  Obviously,  En  is  a  generalization  of  the  Euclidean
plane determined by the totality of pairs of real values
 are associated
with  the  points  of  the  plane  referred  to  a  pair  of  orthogonal  Cartesian  axes  then  the  square  of  the
element of arc ds assumes the familiar form

 If these values

1 xx
,

1 x
,

).

x

(

)

(

2

2

THEOREM 7.1 A function F

2ds =
dx
dt

x,

xd

21
)

(

+

(

xd

22
.)

 satisfying the condition F

,
kx

dx
dt

=(cid:247)

kF

x

,

dx
dt

  for  every

k >  0.  This  condition  is  both  necessary  and  sufficient  to  ensure  independence  of  the  value  of  the

integral

s

(cid:242)=

t

2

t
1

F

x,

dx
dt

is replaced by some function  t =

 dt of a particular mode of parametrization of C. Thus if t in
[

 and we denote

. so that

 by

)(s

]s
( )

)(txi

)(l

xi

i

xC
:

i =

i
tx
)(

)(si

 we have

equality

t

2

t
1

xF

,

dx
dt

dt

= (cid:242)

s

2

s
1

F

,(

)

ds

i =

ds i
ds

 and

where

s
( 2
Proof: Suppose that k is an arbitary positive number and put t = ks so that t1=ks1 and t2=ks2. Then
 becomes

)(txi

 and

s
( 1

t
1

).

=

)

t

2

ixC :

f=

f=

and

xC i
:

(

ks
)

=

)(si

=

)(ti
i

dx

ks

(
ds

)

=

k

dx

)
ks

i

(
dt

Substituting these values in

or

= (cid:242)

s

s =

s =

t

2

t
1

s

2

s
1

s

2

s
1

xF

,

dx
dt

)
(
,
ksxF

 dt we get

)

dx

(
ks
dt

kds

[

F

(

s

),

]ds

)(
s

We must have the relation

F

,(

=

)

F

kx
,

dx
dt

=(cid:247)

kF

x
,

dx
dt

.

 Conversely, if this relation is true

(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
?
(cid:246)
(cid:231)
?
(cid:230)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
f
f
x

x
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
x
�
x
x
�
x
x
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
(cid:242)
x
�
x
(cid:242)
x
�
x
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
?
(cid:246)
(cid:231)
?
(cid:230)
118

Tensors and Their Applications

for every line element of C and each k > 0 then the equality of integrals is assumed for every choice of

parameter

t

f=

(

s
),

1

s
)(

>

,0

s
1

s

s

2

 with  and

t

f=

2

s
( 2

).

Note:

(i)

Here take those curves for which

i
)(tx

 and

i
dx
dt

 are continuous functions in

t
1

t

t

.2

(ii) A function

xF ,

dx
dt

 satisfying the condition

kxF

,

dx
dt

=(cid:247)

kF

,
x

dx
dt

 for every k  >  0 is called

positively homogeneous of degree 1 in the

i

dx
dt

.

EXAMPLE  1

What  is  meant,  consider  a  sphere  S  of radius  a,  immersed  in  a  three-dimensional  Euclidean
.
3

,3E  with centre at the origin (0, 0, 0) of the set of orthogonal cartesian axes

XXXO -
1

2

manifold

Solution
Let T be a plane tangent to S at (0, 0,�a) and the points of this plane be referred to a set of orthogonal
cartesian axes O� �Y 1Y 2 as shown in figure. If we draw from
,OP  interesting

 a radial line

0) 0,

(0,

 O

the sphere S  at

1
xxP

(

,

2

,

x

3

)

 and plane  T at

1
xxQ

(

,

2

,

sphere S are in one-to-one correspondence with points

a

(

 then the points  P on the lower half of the
)
1 xx
,

 of the tangent plane T..

)

2

3
X

O

P1

O�

P2

C

Fig. 7.1

X2

Y2

Q2

K

Q1

1
X

Y1

2

,

x

3

)

 is any point on the radial line OP,, then symmetric equations of this line is

1
x
1

x

0

0

=

2

2

x

x

=

0

0

3

x

a

l=

0
0

1
xxP

(

,

If

or

x
Since the images Q of points P lying on S, the variables xi satisfy the equation of S,

a

x

x

1x =

,1xl

2

l=

-=3

,2

...(6)

(

2

or

21
)
x
(
x

+
)

21

(

22
)

x
(
x

+
)

22

+

(

x

23
)

= 2a

+

2

a

= 2a

�
�
f
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
?
(cid:246)
(cid:231)
?
(cid:230)
-
-
-
-
-
-
-
-

l
�
�
�
�
�
�
l
Geometry

Solving for  l

 and substituting in equation (6), we get

1x =

3x =

1
xa

(

x

21
)

+

(

x

22
)

+

2

a

,

2

x

=

2

xa

(

x

21
)

+

(

x

22
)

+

(

x

23
)

2

a

(

x

21
)

+

(

x

22
)

+

(

x

23
)

and

119

...(7)

These are the equations giving the analytical one-to-one correspondence of the points Q on T and

points P on the portion of S under consideration.
2

2

3

2

Let

1
xxP
(
1

,

,

x

)

 and

1
xP
(
2
21 PP

on S. The Euclidean distance

 along C, is given by the formula

+

1
dx

,

x

+

dx

3

x

+

dx

3

)

 be two close points on some curve C lying

,

2ds =

dx

i dx

,i

 (i = 1, 2, 3)

...(8)

Since

Thus equation (8) becomes

idx =

2ds =

i

p

x
x

i
x
p
x

xd

,p

  (P =1, 2)

i
x
q
x

p
xdxd

q

=

g

pq

xdxdx
)(

p

q

,

 (p,q = 1, 2)

where

i
x
x
If the image  K of C on T is given by the equations

 are functions of

ix  and

g pq

)(x

=

g

pq

p

.

i

q

x
x

K

:

1

x

=

1
tx
)(

2

x

=

2

x

t
(

),

t

1

t

t

2

then the length of C can be computed from the integral

s = (cid:242)

t

2

t
1

g

pq

p

xd
dt

q

xd
dt

dt

A straight forward calculation gives

ds2 =

+

1

{
(

1
2
a

and

xd
(

21
)

+

(

xd

22
)

+

1
xdx

(

2

1
2
a

x

21
)

+

(

x

22
)

21
)

2
xdx
} 2

...(9)

-
�
�
�
�
�
�
�
�
�
�
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
�
�
�
�
�
�
�
�
-
120

Tensors and Their Applications

s =

t

2

t
1

22

21

xd
dt

+

xd
dt

+

1

{

1
2
a

+

1
2
a

1
x

2

xd
dt

2

x

21

xd
dt

(

x

21
)

+

(

x

22
)

}

dt

So, the resulting formulas refer to a two-dimensional manifold determined by the variables

 in
the cartesian plane  T  and that the geometry of the surface of the sphere imbeded in a three-dimensional
2R  with metric given by equation (9).
Euclidean manifold can be visualized on a two-dimensional manifold

)

(

1 xx
,

2

If the radius of S is very large then in equation (9) the terms involving

1
2

a

 can be neglected. Then

equation (9) becomes

22
.)
Thus for large values of a, metric properties of the sphere  S are indistinguishable from those of

2ds =

...(10)

21
)

xd

xd

(

(

+

the Euclidean plane.

The chief point of this example is to indicate that the geometry of sphere imbedded in a Euclidean
3-space,  with  the  element  of  arc  in  the  form  equation  (8),  is  indistinguishable  from  the  Riemannian
geometry of a two-dimensional manifold
2R  with metric (9). the latter manifold, although referred to
a cartesian coordinate system Y, is not Euclidean since equation (9) cannot be reduced by an admissible
transformation to equation (10).

7.2 CURVILINEAR  COORDINATES IN

Let

)(xP

 be the point, in an Euclidean 3-space

3E
3E , referred to a set of orthogonal Cartesian coordinates Y..

Consider a coordinate transformation
xxi
(

ixT :

=

1

2

,

x

,

x

3

),

=i

(

)3,2,1

Such that

=

J

i
x
x

j

0�

 in some region R of

3E . The inverse coordinate transformation

=i
T :1-
(
will  be  single  values  and  the  transformations  T  and
2

xxx i
1
,
(

ix

),

=

x

,

3

2

3

3

2

1
xx
,

(

,

x

)

 and

1
xx
,

(

,

x

).

between the sets of values

)3,2,1
1-T   establish  one-to-one  correspondence

The triplets of numbers
(
If one of the coordinates

2

2

1
,
xx
1
xx
,

3

3

,
,

x
x

 is called curvilinear coordinates of the points P in R.
)
 is held fixed and the other two allowed to vary then the point

P traces out a surface, called coordinate surface.

If we set

1 =x

constant

 in T then

1
1
xxx
(

,

2

,

x

3

)

= constant

...(1)

defines a surface. If constant is allowed to assume different values, we get a one-parameter family of

2

x

1
xx
,

(

2

,

x

3

)

  =  constant  and

x

3

1
xx
,

(

2

,

x

3

)

  =  constant  define  two  families  of

surfaces. Similarly,
surfaces.

The surfaces

(cid:242)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
121

� (2)

Geometry

1x =

,1c

2

x

=

c
2

,

3

x

=

c
3

3

Y

3

X

P

2

X

1
X

2

Y

1

Y

Fig. 7.2

intersect in one and only one point. The surfaces defined by equation (2) the coordinate surfaces
and intersection of coordinate surface pair-by pair are the coordinate lines. Thus the line of intersection
3x  is the only

 line because along this the line the variable

coordinate

 is the

3 -x

x =
x =  and
2
1
of
c
c
2
1
one that is changing.

EXAMPLE  2

Consider a coordinate system defined by the transformation

Y3

X2

X1

O

X3

Y  2

Y 1

Fig. 7.3

1x =

1
x

sin

x

2x =

1
x

sin

x

2

2

cos

x

3

sin

3

x

3x =

1 cos x
x

2

...(3)

...(4)

...(5)

122

Tensors and Their Applications

1 =x

The surfaces
planes passing through the  Y 3-axis (Fig. 7.3).
The squaring and adding equations (3), (4) and (5) we get,

  are  spheres,

constant

constant

2 =x

  are  circluar  cones  and  x3  =  constant  are

(

x

21
)

+

(

x

22
)

+

(

x

23
)

=

1
x

(

sin

x

2

cos

x

23
)

+

1
x

(

sin

2

x

sin

x

23
)

+

1

(

x

cos

x

22
)

On solving

(

x

21
)

+

(

x

22
)

+

(

x

3

)

2

=

(x

21 )

Now, squaring and adding equations (3) and (4), we get

1x =

(

x

21
)

+

(

x

22
)

+

(

x

23
)

(

x

21
)

+

(

x

22
)

=

1
x

(

sin

x

2

sin

x

23
)

+

1
x

(

sin

2

x

sin

x

23
)

(

x

21
)

+

(

x

22
)

=

21
x
)

(

(sin

x

22
)

1 sin x
x

2

=

(

x

21
)

+

(

x

22
)

Divide (7) and (5), we get

2
tan x =

21
x
)

(

+

(

x

22
)

3
x

1

tan

2x =

21
x
)

(

+

(

x

22
)

3

x

or

Divide (3) and (4), we get

2

x

1

x

3

tan x

=

3x =

1

tan

2

1

x

x

...(6)

...(7)

...(8)

...(9)

So, the inverse transformation is given by the equations (6), (8) and (9).

If

1
x

>

0,0

<

2

x

p<

0,

3

x

p<

.2

 This is the familiar spherical coordinate system.

7.3 RECIPROCAL  BASE  SYSTEMS

Covariant  and  Contravariant  Vectors

Let a cartesian coordinate system be determined by a set of orthogonal base vectors
position vector  rr  of any point

 can be expressed as

xP
(

)

3

2

1

r
,1b

rr
2 ,bb
3

 then the

i

 (i = 1, 2, 3)

...(1)

x
,
rr =

x
,
r
i xb

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
-
(cid:222)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
-
�
Geometry

123

Since the base vectors

r  are independent of the position of the point
ib

xP
(

1

,

x

2

,

x

3

)

. Then from (1),

rdr =

r
i xdb

i

...(2)

If

xP
(

1

,

x

2

,

x

3

)

  and

xQ
(

1

+

1
xxd
,

2

+

2
xxd
,

3

+

xd

3

)

  be  two  closed  point.  The  square  of  the

element of arc ds between two points is
2ds =

r (cid:215)
r
rdrd

from equation (2),

2ds =

=

r
i
xdbxdb
i

r

j

r

r (cid:215)
xdxdbb
j
i

i

j

j

2ds =

i
xdxd

;j

 since

ij

r
(cid:215) r
i bb

j

d=

ij

3

X

a 2

2

X

a 1

1

X

3

Y

a 3

P

r

b 3

O

b2

b1

2

Y

1

Y

Fig. 7.4.

rrr
bbb
3
2
1

,

,

(

 are

orthogonal

base

 vector

i.e.,

=

r
r
bb
1
1
as

r
r
&1
bb
1
2
=
=
,1

i

�ij

=

)0

.

j

j

=

,0

i

2ds =

i xdxd

;i

a familiar expression for the square of element of arc in orthogonal cartesian coordinates.
Consider the coordinate transformation
xxi
1
(

 (i = 1, 2, 3)

),

x

x

,

,

3

2

ix =

define a curvilinear coordinate system X. The position vector  rr  is a function of coordinates
i.e.,

.ix

Then

rr =

( ixrr

),

 (i = 1, 2, 3)

rdr =

� r
r
i

x

i

dx

...(3)

(cid:215)
d

(cid:215)
(cid:215)

�
�
Tensors and Their Applications

124

and

where

� r
r
ix

The vector

Put

i
xdxd

j

2ds =

=

r(cid:215)
r
rdrd
r
r
x

i

r
r
x

j

2ds =

dxg
ij

j

i

dx

ijg =

r
r
i
x

r
r
x

j

is a base vector directed tangentially to X i - coordinate curve.

...(4)

...(5)

...(6)

� r
r
ix

= iar

Then from (3) and (4)

Now, from equations (2) and (6), we get

rdr =

i xdar

i

 and

g

ij

=

r (cid:215)
r
aa
i

j

j dxar

j dxar

j

j

=

r
i xdb

i

�r
b
i

=

jar =

�r
b
i

i

j

i

j

x
x

x
x

j

xd

,

 as

xd

j

 are

arbitrary

So, the base vectors

jar  transform according to the law for transformation of components of

covariant vectors.

The components of base vectors
ar

1

,iar  when referred to X-coordinate system, are
(:

,0,0(:

),0,0,

,0(:

),0,

ar

ar

).

a

a

2

2

3

a
1

3

and they are not necessarily unit vectors.
In general,

If the curvilinear coordinate system X is orthogonal. Then

11g =

r
r
(cid:215) aa
1
1

,1

g

22

=

r
a

2

r
a

2

,1

g

33

=

r
a

3

r
a

3

.1

ijg =

r
r
aa
i

j

=

r
r
aa
i

j

cos

=

ij

,0

 if

i �

.j

Any vector  A

r  are can be written in the form

r =A

rdk r

 where k is a scalar..

...(7)

...(8)

...(9)

�
�
(cid:215)
�
�
�
�
(cid:215)
�
�
�
�
�
(cid:222)
�

�
�
(cid:215)

�
(cid:215)
q
(cid:215)
Geometry

125

Since

r
rd

=

i

r
r
i xd
x

 we have

� r
r
i

i
kdx

(

)

r =
A

r =
A

i

x
i Aar
r
iA  are the contravariant components of the vector  A

A =
i

where
Consider three non-coplanar vectors

. The numbers

kdx

i

r �
r
a
a
2
1ar = [
rrr
aaa
321

3

] ,

2

r
a

=

] ,

r
r
a
a
1
3
[
rrr
aaa
321
2ar  and

3

=

r
a

r
r
a
a
1
2
[
rrr
aaa
321
3ar  and  [

rrr
321 aaa

]

...(10)

]

  is  the  triple  scalar

  etc.  denote  the  vector  product  of

prduct

r �
r
2 a
a
,3
r
a
.3

2

where
r
r
aa
1
Now,

r (cid:215)
r
1 a
a
1

r
a
= [

r
a
3
2
rrr
aaa
321

r
a
1
]

=

r (cid:215)
r
1 a
a

2

]
rrr
=aaa
322

Since  [
Similarly,

.0

r
a
= [

r
a
3
2
rrr
aaa
321

r
a
]

2

=

]
]

[
rrr
aaa
321
[
rrr
aaa
321
[
rrr
aaa
322
[
rrr
aaa
321

=

.1

]
]

=

.0

r (cid:215)
r
1 a
a
r (cid:215)
r
2 a
a

2

=

2

r
a

3

= 1,

(cid:215)=

r
(cid:215) a
1
r
r
(cid:215) a
a

3

=
13

=(cid:215)

0

Then we can write

r (cid:215)
r
i a
a

j

= i
j

EXAMPLE  3

To show that  [

rrr
aaa
321

]

=

g

Solution

 and [

]
=rrr
21
aaa

3

 where g =

ijg .

1

g

The components of base vectors

ar

1

(:

a
1

Then

ia  are
ar

2

),0,0,

,0(:

a

2

)0,

and

ar

3

,0,0(:

a

)

3

[
rrr
321 aaa

]

=

a
1

0

00

a

2

=

aaa
321

0

a

3

...(11)

00

�
�
�

�

�
�
(cid:215)
(cid:215)
�
(cid:215)
�
(cid:215)
d

126

and

g=

g ij

=

g

g

g

11

21

31

Tensors and Their Applications

g

13

g

23
g

33

g

12
g

22

g

32

from equations (8) and (9), we have
r
(cid:215) r
aa
1
1

11g =

=

2
a
1

g

11

Similarly

and

So,

2
2a =

,22g

a =
2
3

g

33

12g

=

r
r
(cid:215) aa
1

2

=

,0

g

13

=

r
r
aa
1

3

=

0

 etc.

g =

2
a
1

0

0

0
2
2

a

0

0

0

a

2
3

=

2
2
aaa
2
1

2
3

g =
from eqn. (11) and (12), we have

321 aaa

[
rrr
321 aaa

]

=

g

Since the triple products

[

=rrr
21
aaa

]

3

1

g

 Moreover,,

.

1ar =

r �
r
2
a
a
rrr
21
aaa

3

3

[

,

]

r
a

2

=

3

r
r
a
a
rrr
21
aaa

1

3

[

,

]

r
a

3

=

[

...(12)

]

1

2

r
r
a
a
rrr
32
1
aaa

The system of vectors

1

r
r
r
2
aaa

,

,

3

 is called the reciprocal base system.

Hence if the vectors

1

r
r
r
2
aaa

,

,

3

then the reciprocal system of vector defines the same system of coordinates.

 are unit vectors associated with an orthogonal cartesian coordinates
Solved.

The differential of a vector  rr  in the reciprocal base system is

r =

rd

r
i xda
.i

where

where

idx  are the components of
2ds =
=

.rdr  Then
r (cid:215)
r
rdrd
r
i
dxa
i

()

(

r
j
dxa

)

j

=

r (cid:215)
i
a

r
j
dxa
i

dx

j

2ds =

ij

g

dx
i

dx

j

ijg =

i

r
a

(cid:215) r
a

j

=

ji

g

...(13)

(cid:222)

(cid:215)

�
�
(cid:215)
Geometry

127

The system of base vectors determined by equation (10) can be used to represent an arbitrary

vector A in the form

r =
i AaA r
,i

 where

Taking scalar product of vector

r (cid:215)
r
i
aaA
i

j

=

i

iA  are the covariant components of
,jar
i aA r  with the base vector
 we get
.i
  as
A
j
i

(cid:215) r
i a

r
a

d=

A

=

i
j

j

j

r
.A

7.4 ON  THE  MEANING  OF  COVARIANT  DERIVATIVES

THEOREM 7.2 If  A

r  is a vector along the curve in

3E . Prove that

r
A
j

x

=

r
aaA
a
,
j

iar  as

...(1)

Also, prove that

=

i
A
,

j

.

 Where

iA  are component of

r
.A

Proof: A vector  A

r  can be expressed in the terms of base vectors
i aA r

r =
A

i

i
A
j

x

where

r
a

i

=

 and

iA  are components of

r
.A

r
r
i

x
The partial derivative of  A

r  with respect to

� r
A
jx

=

i
A
j

x

+

r
a

i

jx  is
r
a

i

A

i
j

Since

g

ij

=

r (cid:215)
r
aa
i

.j

Differentiating partially it w.r.t.

,kx

g

x

ij
k

Similarly,

g

jk
i
x
g

ik
j
x

and

r
Since  A

 can be written as

=

=

=

 we have
r
a

r
a

j

+

i
k

x

r
a

j
i
x
r
a
i
j

x

r
a
k

+

r
a

k

+

Taking scalar product with

 we have

i

r
Aa
i

=

r
i
Aa
i

r =
A
,jar

x

r
a

x

j
k

r
a

k
i
x
r
a

k
j

x

r
a
i

r
a

j

r
a
i

r (cid:215)
r
Aaa
i

j

i

i

ij Ag

=

=

r (cid:215)
i
a

j
i

r
Aa
j
i
A =
i

A

j

As

r
r
aa
i

j

=

r
,
ag
ij

i

r
a

j

d=

i
j

 and

i
j

A =
i

A

.j

We see that the vector obtained by lowering the index in

iA  is precisely the covariant vector

.iA

d
�
�
�
�
�
�
�
�
�
�
�
�
�
(cid:215)
�
�
(cid:215)
�
�
�
�
(cid:215)
�
�
(cid:215)
�
�
�
�
(cid:215)
�
�
(cid:215)
�
�
(cid:222)
d
(cid:215)
(cid:215)
d
128

Tensors and Their Applications

The two sets of quantities

iA  and

iA  are represent the same vector  A

r  referred to two different

base systems.

EXAMPLE  4

Show that

d=a
j

i gg

.j
i

Solution

Since we know that

Then

But

So,

Now,

Substituting the value of

or

Hence

aig

=

r
r
(cid:215) aai

 and

g

j

=

j

r
a

r
a

j

i gg

=

=

=

(

r
r
aa
i

r
a

()

j

(

r
r
aa
i

j

r
a

()

r
a

r
a

)

)

j
i

 as

r
(cid:215) r
i aa

j

d=

j
i

j

i gg

= j
i

 as

.1=

iar =

� r
a
i
j

x

� r
a
i
j

x

=

=

� r
r
ix

2

r
r

x

j

i

dx

� r
a

x

j
i

=

2

r
r

i
xx

j

=

r
a

j
i
x

kij
,[

]

=

1
2

+

g

ik
j
x

g

jk

i

x

g

x

ij

k

g

ik
j
x

,

g

jk
i
x

and

[ij,k] =

(cid:215) 2

1
2

r
a

x

g

x

ij
k

,

 we get

r
i a
j

k

[ij,k] =

r
a
r
i a
j

x

k

� r
a
i
j

x

r
kakij
]
,[

,

=

  as

=1
r
a

k

k

r
a

, Christoffel�s symbol

r
a

x

i
j

r
a

r]
,[
akij

k

r
(cid:215) a

=

a
a
a
a
(cid:215)
a
a
a
a
(cid:215)
(cid:215)
a
a
(cid:215)
(cid:215)
a
a
d
d
a
a
d
d
a
a
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�

�
�
�
�
�
�
(cid:215)
�
�
(cid:215)
�
�
�
a
(cid:215)
�
�
a
129

...(2)

Geometry

akgkij
]
,[

=

,

 Since

k

g

=

k

r
a

r
a

r
a
i
j

x

r
a

.

=

ji

,

  as

k]
,
gkji

[

=a

i

j

Substituting the values of

� r
a
i
j

x

� r
a
i
j

x

� r
A
jx

� r
A
jx
� r
A
jx

r
a

=

i

j

 in equation (1), we get

=

=

=

i
A
j

x

A

j

x

+

r
a
i

i

j

r
i
aA

r
a

+

r
i
aA

i

j

+

A

j

x

i

j

r
i
aA

=

r,
aA j

  since

=

A
,

j

+

A

j

x

i

A

i

j

Thus, the covariant derivative

jA,

  of  the  vector

aA   is  a  vector  whose  components  are  the

components of

� r
A
jx

 referred to the base system

.iar

If the Christoffel symbols vanish identically i.e.,

Substituting this value in equation (1), we get

0=

 the

ji

ar
x

i
j

,0=

 from (2).

 But

� r
A
jx

=

i
jA, =

i
jA, =

i

A r
a
x

j

i

i
A
j

x

i
A
j

x

+

i

j

A

 as

i

j

.0=

r  is a vector along the curve in

THEOREM 7.3 If  A
r .
of
.A
Proof: If  A

r  can be expressed in the form
i aA r

r =
A

i

3E . Prove that

kj aA r

,

j

 where e

jA  are components

Proved.

a
a
(cid:215)
a
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
�
a
a
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
�
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
a
a
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
�
�
�
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
130

Tensors and Their Applications

where

iA  are components of

r
.A

r
The partial derivative of  A

kx  is

 with respect to
� r
A
kx

A
i
k

r
a

=

+

x

i

A
i

i

r
a

k

x

...(1)

Since

r
a

(cid:215) r
i a

j

d=

,i
j

 we have,

Differentiating it partially w.r. to

,kx

 we get

i

r
a

k

x

r
a

j

+

i

r
a

r
a

x

j
k

= 0

But

r
a

(cid:215) r
i a

d=

.i

 Then

i

r
a

k

x

i

r
a

k

x

i

r
a

k

x

r
a

j

=

i

r
a

r
a

x

j
k

=

r
r
aa i

,

   Since

kj

r
a

x

j
k

=

r
a

kj

r
a

j

=

i

kj

r
a

j

=

i

� r
a

k

x

=

i

kj

i

kj

r
,ja

   as

=1
r
a

j

j

r
a

substituting the value of

 in equation (1), we get

i

� r
a

k

x
� r
A
kx

� r
A
kx
� r
A
kx

=

=

=

A
i
k

x

A
i
k

x

i

r
a

A
i

j

r
a

A
i

i

kj

j

r
a

i

kj

j

r
a

A
i
k

x

A
i

i

kj

r�
a

j

=

kj aA r

,

j

,

  Since

=

A

kj,

A
j
k

x

A
i

i

kj

Proved.

�
�
�
�
�
�
�
(cid:215)
(cid:215)
�
�
(cid:215)
�
�
�
�
(cid:215)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:215)
-
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
a
(cid:215)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
d
-
a
(cid:215)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
Geometry

131

7.5 INTRINSIC  DIFFERENTIATION

Let a vector field

r
)(xA

 and

ixC :

=

(txi

),

t

1

t

t

2

be a curve in some region of

.3E  The vector

r
)(xA

 depend on the parameter t and if

)(xA

 is a

differentiable vector then

r
Ad
dt
r
Ad
dt

� r
A
jx

Since we know

So,

� r
A
j

x

=

j

dx
dt

=

r
aA
,
j

j

dx
dt

r
,
aA
j

=

=

+

A
x

j

i

j

r
a

 (See Pg. 127, Theo. 7.2)

r
Ad
dt

r
Ad
dt

=

=

A

j

x

dA
dt

+

+

r
i
aA

i

j

j

dx
dt

i
A

j

dx
dt

r
a

i

j

The formula

dA
dt

+

i

A

j

dx
dt

i

j

 is called the absolute or Intrinsic derivative of

aA  with respect

to parameter t and denoted by

So,

A
t

=

dA
dt

+

i

A

i

j

Some  Results

A
t

.

j

dx
dt

 is contravariant vector. If A is a scalar then, obviously,

=

A
t

A
t

.

(i)

If

iA  be covariant vector
Ai
t

dAi
dt

=

A

dx
dt

i

(ii)

(iii)

ij

A
t

A

i
j

t

=

=

ij
A
dt

+

A

i
j

dt

+

i

i

j

A

dx
dt

+

j

i

A

dx
dt

A

j

dx
dt

i
A

dx
dt

i

�
�
(cid:215)
�
a
a
�
a
a
a
a
�
�
�
�
�
�
�
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
a
�
�
a
a
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
a
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
d
d
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
d
d
a
a
d
d
d
d
d
d
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
-
b
a
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
d
d
d
b
a
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
d
d
d
132

Tensors and Their Applications

(iv)

A

i
jk
t

=

A

i
jk
dt

+

i

A

jk

dx
dt

i
A

k

dx
dt

j

A

i
j

dx
dt

k

EXAMPLE  5

If

ijg  be components of metric tensor, show that

g ij
t

.0=

Solution

The intrinsic derivative of

ijg  is

g ij
t

g ij
t

=

=

=

=

dg

ij
dt

g

j

dx
dt

i

g

i

j

dx
dt

g

ij

x

dx
dt

g

j

dx
dt

i

g

i

j

dx
dt

g

ij

x

g ij
x

g

�

j

i

g

i

j

dx
dt

[
i

]

,

j

[

]

,
ij

dx
dt

jg

=

[

i

,

j

and ]

i

[
b=

]ij
,

.

g

i

j

g ij
x
g ij
t

=

[

i

j

]

b+
[
j

,

,

i

].

= 0.

as

But

So,

EXAMPLE  6

Prove that

i

(

AAgd
ij
dt

Solution

Since

i

aAg
ij

j

 is scalar..

Then

i

(

AAgd
ij
dt

j

)

=

i

2

Ag
ij

i

A
t

j

)

=

(

i
AAg

ij

j

)

t

b
a
b
a
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
d
d
d
d
d
d
d
b
a
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
-
b
a
b
a
b
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
-
�
�
b
a
a
b
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
-
�
�
d
d
b
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
-
b
-
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a

(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
b
a
b
a
b
�
�
b
d
d
d
d
d
d
Geometry

133

j

)

(

i
AA
t

,

  since

ijg  is independent of t.

=

g

ij

=

g

ij

i

A
t

j

A

+

i

A

j

A
t

j

A
t

 as

ijg  is symmetric.

Interchange i and j in first term, we get

i

(

AAgd
ij
dt

i

(

AAgd
ij
dt

+

i

A

j

)

j

)

=

i

Ag
ij

=

2

i
Ag
ij

j

j

A
t

A
t

EXAMPLE  7

Proved.

Prove that if A is the magnitude of

iA  then
i
i ,
AA
j
A

jA, =

Solution

Given that A is magnitude of
Since

iA . Then

i
ik AAg

k

i
ik AAg

k

=

=

i AA

i

2A
,jx

Taking covariant derivative w.r. to

 we get

i
AAg
,

ik

j

k

+

i

k
AAg
,
j

ik

=

2

jAA,

Interchange the dummy index in first term, we get

k
AAg
,
j

ki

i

+

i

k
AAg
,
j

ik

=

2

jAA,

2

k
i
ik AAg
,
j

=

2

jAA,

i
k
ik AAg
, =
j
)k
(
AgA
,

=

ik

j

i

i AA , =

i

j

jAA,

jAA,

jAA,

  since

k
Ag
,
ik
j

=

A
i

,

j

jA, =

i

i ,
AA
j
A

Proved.

d
d
�
�
�
�
�
�
d
d
d
d
�
�
�
�
�
�
d
d
d
d
d
d
134

Tensors and Their Applications

7.6 PARALLEL  VECTOR  FIELDS

Consider a curve

ixC :
=
r  localized at point P of C. If we construct at every point of C a
in some region of
3E  and a vector  A
vector equal to A in magnitude and parallel to it in direction, we obtain a parallel field of vector along the
curve C.

  (i = 1, 2, 3)

,2

),

t

t

t

1

(txi

3

Y

3

X

P

2

X

C

1
X

O

2

Y

1

Y

r  is a parallel field along C then the vector  A

if  A

r  do not change along the curve and we can write

r
Ad
dt
Ai
t

.0=

 It follows that the components

iA  of  A

r  satisfy a set of simultaneous differential equations

0=

 or

i

dAi
dt

+

A

dx
dt

 = 0

This is required condition for the vector field

iA  is parallel.

7.7 GEOMETRY  OF  SPACE  CURVES

Let the parametric equations of the curve  C in

3E  be

ixC :
The square of the length of an element of C is given by

(txi

),

=

t

t

t

2

1

   (i = 1, 2, 3).

and the length of arc s of C is defined by the integral

2ds =

dxg
ij

j

i

dx

s = (cid:242)

t

2

t

1

g

ij

i
dx
dt

j

dx
dt

dt

from (1), we have

g

ij

i

dx
ds

j

dx
ds

= 1

...(1)

...(2)

...(3)

�
�
�
�
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a

�
�
Geometry

135

Put

i

dx
ds

l=

.i

 Then equation (3) becomes

= 1
The vector  lr , with components

ijg

i

j

...(4)
  is  a  unit  vector.  Moreover, ,  lr   is  tangent  to  C,  since  its

,i

components

i

, when the curve C is referred to a rectangular Cartesian coordinate Y,, becomes

These are precisely the direction cosines of the tangent vector to the curve  C.

Consider a pair of unit vectors  lr  and  mr  (with components

i

 and

i

 respectively) at any point

P of C. Let  lr  is tangent to C at P Fig. (7.6).

i =l

xd i
ds

.

l

l  + dl

(Q(x + dx)

C

P x( )

r

r + dr

O

Fig. 7.6

The cosine of the angle  q

 between  lr  and  mr  is given by the formula

cos =
and if  lr  and  mr  are orthogonal, then equation (5) becomes

ijg

i

j

i

j

= 0
Any vector  mr  satisfying equation (6) is said to be normal to C at P..
Now, differentiating intrinsically, with respect to the are parameter s, equation (4), we get

ijg

g

ij

i

s

j

+

g

ij

j

s

i

= 0

as

ijg  is constant with respect to s.

Interchange indices i and j in second term of equation (7) we get

g

ij

i

s

j

+

g

ij

i

s

j

= 0

Since

ijg  is symmetric. Then

i

2

g

ij

j

s

= 0

...(5)

...(6)

�(7)

l
l
l
l
l
m
q
l
l
m
l
l
d
d
l
l
d
d
l
l
d
d
l
l
d
d
l
d
d
l
l
m
136

Tensors and Their Applications

i

g

ij

j

s

= 0

we see that the vector

j

s

 either vanishes or is normal to C  and if does not vanish

we denote the unit vector co-directional with

j

s

 by

j

 and write

j

=

j

1
sK

,

=K

j

s

...(8)

where K > 0 is so chosen as to make

j

 a unit vector..

The vector

j

 is called the  Principal normal vector  to the curve  C  at the point P and  K is  the

curvature of C.

The plane determined by the tangent vector  lr  and the principal normal vector  mr  is called the

osculating plane to the curve  C at P.
Since  mr  is unit vector

ijg

i

j

= 1

...(9)

Also, differentiating intrinsically with respect to s to equation (6), we get

g

ij

j

s

j

+

i

g

ij

or

i

g

ij

g

ij

i

j

s

i

g

ij

j

s

j

s

j

s

 = 0

=

=

g

ij

j

i

s

ijKg

i

j

  Since

i

s

m=
K

i

,K-

=

  since

ijg

i

j

.1=

+

K

= 0

i

g

ij

j

s

i

g

ij

+

g

ij

i

j

K

= 0  as

ijg

i

j

1=

j

s

l+
K

j

= 0

This shows that the vector

j

s

l+
K

j

 is orthogonal to

.i

(cid:222)
d
d
l
l
d
d
l
d
d
l
m
m
d
d
l

d
d
l
m
m
m
m
d
d
l
l
m
d
d
l
d
d
m
l
m
d
d
l
-
m
m
-
d
d
l
d
d
m
l
m
m
d
d
m
l
l
l
d
d
m
l
l
l
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
d
d
m
l
d
d
m
l
Geometry

Now, we define a unit vector

,vr  with components

,jv

 by the formula

1

iv =

j

s

l+
K

j

 the vector  vr  will be orthogonal to both  lr  and  mr .

i

=t

where
To choose the sign of  t

l+
K

s

i

 in such a way that
i

k

j

v

= 1

eg

ijk

137

...(10)

...(11)

so that the triad of unit vectors  lr ,  mr  and  nr  forms a right handed system of axes.

Since

ijke

  is  a  relative  tensor  of  weight  1-

 and

=

g

2

i

j

x
x

  it  follows  that

eg=

ijk

ijk

  is  an

obsolute tensor and hence left hand side of equation (11) is an invariant  kv  in equation (11) is determined
by the formula

kv =

ijk

i

j

where

 and

i

 are the associated vectors

j

ig

 and

ig

 and

ijk

...(12)

ijk

e

 is an absolute tensor..

1=
g

 appearing in equation (10) is called the  torsion of C at P and the vector  vr  is the

The number  t

binormal.

We have already proved that in Theorem 7.2, Pg. 127.

� r
A
ix

=

r
a aA i

,

if the vector field  A

r  is defined along C, we can write

� r
A
i

x

i

x
s

=

A
,

i

i

x
s

r
a

Using definition of intrinsic derivative,

A
s
Then equation  (13) becomes

r
Ad
ds

a,
A

i

=

i
dx
ds

=

A r
a
s

;(cid:215)

  as

r
A
i

x

i

dx
ds

=

r
Ad
ds

...(13)

...(14)

Let  rr  be the position vector of the point P on C then the tangent vector  lr  is determined by

rdr
ds

=

r

l=

r
i a

i

from equation (14), we get

rd r
2
2
ds

=

r

d
ds

=

r
a

=

r
c

s

...(15)

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
d
d
m
t
d
d
m
m
l
�
�
e
m
l
e
l
m
a
a
l
a
a
m
e
�
a
�
�
�
a
a
�
�
d
d
a
(cid:215)
(cid:215)
d
d
a
a
�
�
l
(cid:222)
d
d
l
l
a
a
138

Tensors and Their Applications

where  cr  is a vector perpendicular to
With each point P of C we can associate a constant K , such that

.

lr

r
=Kc

mr

 is a unit vector..

Since

from equation (8), we get

cr
K

= mr

mr =

1
sK

r
a

,

 from (15)

mr =

am ar

,a

  since

=

1
sK d

7.8 SERRET-FRENET  FORMULA

The serret-frenet formulas are given by

i

l1
sK

 or

i

s

 =

i
,
KK

>

0
 where

K

=

i

s

l+
K

i

 or

i

s

 =

i Kl
�

i

 where t =

i

s

l+
K

i

i

s

k

(i)

(ii)

(iii)

i

i

=

=

1

k

s

 = �

First two formulas have already been derived in article (7.7), equation (8) and (10).

Proof of (iii)

From equation (12), article (7.7), we have

ijk

= kn

i

j

where

i

,

i

,

k

 are mutually orthogonal.

Taking intrinsic derivative with respect to s, we get

ijk

i
s

e+

ijk

j

i

j

s

=

From formulas (i) and (ii), we get

ijk

K

i

j

ijk

e+

(

i

�

K

j

)

j

=

ijk

(

i

�

Kl

j

)

j

=

ijk

i

j

�

K

ijk

=

j

i

k

s

k

s

k

s

k

s

, Since

ijk

 = 0

j

i

a
a
d
d
l
d
l
m
a
a
m
d
d
d
d
l
d
d
l
m

n
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
d
d
m
t
d
d
m
t
n
d
d
m
d
d
n
t
m
m
l
e
n
m
l
d
d
m
l
m
d
d
l
e
d
d
n
l
t
n
l
m
m
e
d
d
n
t
n
l
e
d
d
n
m
m
e
l
l
e
t
n
l
e
d
d
n
Geometry

139

Since

Since

Then

So,

or

ijk

i

j

= 0

ijk

i

j

=

k

s

ijk

m =

k

i

j

, but

ijk

 are skew-symmetric.

ijk

km�

=

i

j

m k�

=

k

s

k

s

i

s

m�

=

m�

=

k

i

This is the proof of third Serret-Frenet Formula
Expanded form of Serret-Frenet Formula.

(i)

(ii)

(iii)

i

d
ds

i

d
ds

i

dv
ds

+

+

+

i
kj

j

k

dx
ds

i

jk

i

jk

j

k

dx
ds

j

k

dx
ds

=

iKm

or

2
i
xd
2
ds

+

i
kj

j

dx
ds

k

dx
ds

=

iKm

=

i Kl
�

i

=

m�

i

EXAMPLE 8
Consider a curve defined in cylindrical coordinates by equation

1
x

2

3

x

x

=

a
q=

=

0

s
)(

This curve is a circle of radius a.
The square of the element of arc in cylindrical coordinates is

2ds =

dx

21
)

(

+

21
x
()

(

dx

22
)

+

(

dx

23
)

gij = 0,
so that g11 = 1,
It is easy to verify that the non-vanishing Christoffel symbols are (see Example 3, Page 61)

g33 = 1,

g22 =

(x

,

21)

i �

j

1

22

= 1x ,

2

12

 =

2

21

1
 =  1
x

.

l
l
e
t
n
l
e
d
d
n
n
l
e
e
n
l
e
t
d
d
n
d
d
n
t
(cid:222)
d
d
n
t
l
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
l
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
m
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
m
t
n
n
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
t
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
140

The components of the tangent vector l

 to the circle C are

i

 =

Tensors and Their Applications

dxi
ds

 so that  1l

 = 0,  2l

 =

dq
ds

,

3l

 = 0.

Since l

 is a unit vector, gij

il

j= 1 at all points of C and this requires that

)
(
21
x

d
ds

2

2

=

a

2

 = 1

d
ds

So,

d
ds

2

=

1
2

a

 and by Serret-Frenet first formula (expanded form), we get

1mK

=

1

d
ds

+

1
kj

j

k

dx
ds

 =

1 l
22

2mK

=

3mK

=

2

d
ds

+

3

d
ds

+

2
kj

3
kj

j

j

k

dx
ds

k

dx
ds

 =

2 l
12

 = 0

2

2

dx2
ds

dx1
ds

 =

�

1
a

 = 0

Since � is unit vector,

ijg

i

j

 = 1 and it follows that K =

1
a

,  1m = �1,

2m

 = 0,

3m

 = 0

Similarly we can shows that t  = 0 and

1

n=
,0

2

n=
,0

3

=

.1

7.9 EQUATIONS  OF  A  STRAIGHT  LINE

Let

iA  be a vector field defined along a curve C in
s1 �

ixC :

)(sxi

=

.

3E  such that
 s �

 s2,

 (i = 1, 2, 3).

s being the arc parameter.

If the vector field

iA  is parallel then from article 7.6 we have

Ai
s

b

dx
ds

= 0

= 0

or

dAi
ds

+

i

A

(1)

We shall make use of equation (1) to obtain the equations of a straight line in general curvilinear
coordinates. The characteristic property of straight lines is the tangent vector  lr  to a straight line is
directed along the straight line. So that the totality of the tangent vectors lr  forms a parallel vector field.

l
l
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
q
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
q
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
q
l
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
l
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
l
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
l
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
l
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
l
m
m
n
d
d
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
Geometry

141

Thus the field of tangent vector

i =l

xd i
sd

 must satisfy equation (1), we have

i

s

=

2
i
xd
2
ds

+

i

dx
ds

dx
ds

= 0

The equation

2
i
xd
2
ds

+

i

xd
sd

xd
sd

 = 0 is the differential equation of the straight line.

EXERCISE

j

)

 = gij

i

A
t

j

B

+

i

Ag
ij

i

B
t

1. Show that

i

(

BAgd
ij
dt

2. Show that

A

, �
ji

A

,
ij

 =

A
i
j
x

�

A
j
i
x

3. If

A =
i

Ag
ij

j

 show that

A
,
ki

=

a
Ag
a
,
k
i

i

(

BAgd
ij
dx

k

4. Show that

5. Show that

j

)

 =

BA
,
ki

i

+

i
AB
,
ki

2

i

2

ds

2

i

2
s

2

i

2
s

=

=

=

dK
ds

d
ds

i

�

K

(

i

�

K

i

)

i

(�

K

2

t+

2

)

i

�

i

dK
ds

(

K

i

�

i

v

�)

i

d
ds

6. Find the curvature and tension at any point of the circular helix C whose equations in cylindrical

coordinates are

: 1
xC

=

a
,

2
q=x

,

q=3x

Show that the tangent vector l  at every point of C makes a constant angle with the direction of
axis.  Consider  C also in the form y1 =  a cosq
rectangular Cartesion.

3X -
. Where the coordinates yi  are

, y2 =  a sinq

, y3 = q

d
d
l
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
d
d
d
d
�
�
�
�
l
d
l
t
n
m

d
m
d
l
m
n
t
d
n
d
m
t
t
l
t

CHAPTER � 8

ANALYTICAL  MECHANICS

8.1 INTRODUCTION

Analytical mechanics is concerned with a mathematical description of motion of material bodies subjected
to the action of forces. A material body is assumed to consist of a large number of minute bits of matter
connected in some way with one another. The attention is first focused on a single particle, which is
assumed to be free of constraints and its behaviour is analyzed when it is subjected to the action of
external forces. The resulting body of knowledge constitutes the mechanics of a particle. To pass from
mechanics of a single particle to mechanics of aggregates of particles composing a material body, one
introduces  the  principle  of  superposition  of  effects  and  makes  specific  assumptions  concerning  the
nature  of  constraining  forces,  depending  on  whether  the  body  under  consideration  is  rigid,  elastic,
plastic, fluid and so on.

8.2 NEWTONIAN  LAWS

1. Every body continues in its state of rest or of uniform motion in a straight line, except in so

far as it is compelled by impressed forces to change that state.

2. The change of motion is proportional to the impressed motive force and takes place in the

direction of the straight line in which that force is impressed.

3. To every action there is always an equal and contrary reaction; or the mutual actions of two

bodies are always and oppositely directed along the same straight line.

The first law depends for its meaning upon the dynamical concept of force and on the kinematical

idea of uniform rectilinear motion.

The second law of motion intorduces the kinematical concept of motion and the dynamical idea
of force. To understand its meaning it should be noted that Newton uses the term motion in the sense
of momentum, i.e., the product of mass by velocity, this, "change of motion" means the time of change
of momentum.

In vector notation, the second law can be stated as

r =
F

)

( r
vmd
dt

� (1)

Analytical Mechanics

If we postulate the invariance of mass then equation (1) can be written as
am r

r =
F

from (1) if

So that

hence  vr  is constant vector..

r = 0 then
F

)

( r
vmd
dt

 = 0.

vm r = constant.

143

� (2)

Thus the first law is a consequence of the second.
The third law of motion states that accelerations always occur in pairs. In term of force we may
say that if a force acts on a given body, the body itself exerts an equal and oppositely directed force on
some other body. Newton called the two aspects of the force of action and reaction.

8.3 EQUATIONS  OF  MOTION  OF  A  PARTICLE

THEOREM 8.1  The work done in displacing a particle along its trajectory is equal to the change in
the kinetic energy of particle.
Proof: Let the equation of path C of the particle in E3 be

ixC : =

)(txi

and the curve  C the trajectory of the particle. Let at time  t, particle is at P

txi
({

)}.

If vi be the component of velocity of moving particle then

and if ai be the component of acceleration of moving particle then

vi =

dxi
dt

ai =

i

v
t

=

i

dv
dt

+

i
kj

j

v

k

xd
td

ai =

i

2
xd
2

dt

+

i

kj

j

dx
dt

k

dx
dt

� (1)

� (2)

� (3)

where

vi
t
metric tensor gij

 is the intrinsic derivative and the

i

kj

  are  the  Christoffel  symbols  calculated  from  the

If m be the mass of particle. Then by Newton�s second law of motion

Fi =

m

i

v
t

=

i

ma

� (4)

dW =

We define the element of work done by the force  F
rdF rr
.
Since the components of  F

.

r  and  rdr  are F i and dxi respectively..

r  in producing a displacement  rdr  by invariant

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
d
144

Then

Tensors and Their Applications

dW = gij Fi dxj

= Fj dxj where Fj = gij Fi

� (5)

The work done in displacing a particle along the trajectory C, joining a pair of points P1 and P2, is

line integral

� (6)

� (7)

P
1
using equation (4) then equation (6) becomes

W =  (cid:242)

P
2

i

i dxF

W =   (cid:242)

= (cid:242)

P
1

P
2

P
1

P
2

gm

ij

i

v
t

j

dx

gm

ij

i

v
t

j

dx
dt

dt

W =   (cid:242)
Since gij v i v j is an invariant then

P
2

P
1

gm

ij

i

v
t

v

j

dt

or

(

i
vvg
ij

j

)

t

=

d
dt

(

i
ij vvg

j

)

d
dt

(

i
ij vvg

j

)

=  2

g

ij

i
v
t

j

v

g

ij

i
v
t

j

v

=

1
2

d
dt

(

i
ij vvg

j

)

using this result in equation (7), we get

W=  (cid:242)

P
2

P

1

dm
2
dt

(

i
vvg
ij

j

)

dt

W=

m
2

[

i
ij vvg

j

]

P
2
P
1

Let T2 and T1 is kinetic energy at P2 and P1 respectively.

W = T2 � T1

where T =

m
2

i
vvg
ij

j

=

2vm
2

 is kinetic energy of particle.

We have the result that the work done by force  Fi in displacing the particle from the point P1 to

the point  P2  is  equal  to  the  difference  of  the  values  of  the  quantity  T =
beginning of the displacement.

1
2

  mv2  at  the  end  and  the

8.4 CONSERVATIVE  FORCE  FIELD
The force field Fi is such that the integral W =  (cid:242)

P
2

P
1

i

i dxF

 is independent of the path.

Therefore the integrand Fi dxi is an exact differential

dW = Fi dxi

� (8)

d
d
d
d
d
d
d
d
d
d
(cid:222)
d
d
Analytical Mechanics

145

of the work function  W.  The  negative  of  the  work  function  W  is  called  the  force  potential  or

potential energy.

We conclude from equation (8) that

Fi = �

V
ix

� (9)

where potential energy V is a function of coordinates xi. Hence, the fields of force are called conservative

if Fi = �

V
ix

.

THEOREM  8.2 A  necessary  and  sufficient  condition  that  a  force  field  Fi,  defined  in  a  simply
connected region, be conservative is that Fi,j = Fj,i.

Proof: Suppose that Fi conservative. Then Fi = �

V
ix

Now,

Fi,j =

F
i
j

x

k

ji

F
k

V
i
x
j

x

k

ji

F
k

Fi,j =

=

2
V
x

j

i

x

k
ij

F
k

Fj,i =

F

x

j
i

k

ij

F
k

Fj,i = �

� 2

V

i
xx

j

k

ij

F
k

and

Similarly,

From equation (1) and (2), we get

Fi,j = Fj,i

conversely, suppose that Fi,j = F j,i

Then

F
i
j

x

k

ji

F
k

=

F

x

j
i

k

ij

F
k

F
i
j

x

=

F

x

j as
i

k

ji

 due to symmetry..

� (1)

� (2)

�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:222)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
146

Tensors and Their Applications

V
ix

Take Fi =
Then

So, we can take

Hence, Fi is conservative.

F
i
j

x

F
i
j

x

=

=

=

=

Fi  =

V
j

x

x

2
V
j x
2
V
i xx

i

j

i

x

F

x

j
i

V
ix

.

8.5 LAGRANGEAN  EQUATIONS  OF  MOTION

Consider a particle moving on the curve

)(txi
At time t, let particle is at point P (xi).

ixC :

=

The kinetic energy T =

1
2

 mv2 can be written as

T =

1
2

i
xxgm
&&
ij

j

j
xx
&&

k

� (1)

Since

ix&  = vi.

or

T =

Differentiating it with respect to

T
&�
ix

=

=

=

=

=

1
2
1
2

jk

gm

1
2
ix& , we get
1
2

gm

jk

j

i

x
&
x
&

k

x
&

+

j

x
&

k

i

x
&
x
&

(

gm

jk

d+

k

j
i

x
&

k
i

x
&

)j

gm

jk

+

k

j
i

x
&

1
2

gm

jk

j

k
i

x
&

1
2

1
2

(
xgm
&
ik

)j

k

+

xg
&
ji

xgm
(
&

ij

j

+

xg
&
ij

j

)

as

g

ij

=

g

ji

�
�
-
�
�
�
�
�
-
�
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
�
�
-
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
d
d
d
Analytical Mechanics

or

T
&�
ix
T
&�
ix

=

ij xgm &

j

=

k

ik xgm
&

Differentiating equation (2) with respect to t, we get

d
dt

T
ix
&

m

=

(

d
dt

)k

ik xg
&

m

=

d
dt

k

+

xg
&
ik

k
xg
&&
ik

m

=

g
ik
j
x

j

dx
dt

k

x
&

+

k

xg
&&
ik

d
dt

T
&�
ix

=

m

g

ik
j
x

k

+
j
xgmxx
&&
&&

ik

k

1
2

Since T =

xxgm
&
jk
Differentiating it with respect to xi, we get

&

j

k

Now,

T
ix

T
i

x

d
dt

T
i

x
&

=

1
2

m

g

jk
i
x

k

j
xx
&
&

=

m

g

ik
j
x

kj
xx
&&

+

k
xgm
&&
ik

1
2

m

g

jk
i
x

kj
xx
&&

147

�(2)

� (3)

� (4)

=

k
xgm
&&
ik

+

=

k
xgm
&&
ik

+

k
xgm
&&
ik

=

+

1
2

1
2

1
2

m

m

m

g

ik
j
x

g

ik
j
x

k

j
xx
&&

+

k

j
xx
&&

+

1
2

1
2

j

xxgm
&
ik

&

k

1
2

m

g

jk
i
x

k

j
xx
&
&

m

g

x

ij
k

j

k
xx
&
&

1
2

m

g

jk
i
x

k

j
xx
&
&

+

g

ik
j
x

g

x

ij

k

g

jk

i

x

k

j
xx
&
&

=

=

=

k
xgm
&&
ik

[+

jkm

j
],
xxi
&
&

k

k
xgm
&&
ik

+

il
ggm

[

jk

il

j
],
xxi
&
&

k

+
ggmxgm

il

l
&&

il

[

jk

il

j
],
xxi
&
&

k

�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
-
�
�
�
�
-
�
�
�
-
�
�
�
�
�
�
�
�
�
�
�
-
�
�
�
�
148

Tensors and Their Applications

[
l
xgm
&&
il

=

+

il

g

[

jk

j
],
xxi
&&

]k

=

gm

il

+

l
x
&&

l

kj

k

j
xx
&
&

,�

as

g il

[

jk

],
i

=

l

kj

T
i

x

=

ilagm

,l

Since

l

a

=

+

l
x
&&

l

kj

k

j
xx
&&

d
dt

T
i

x
&

where al is component of acceleration

or

d
dt

d
dt

T
i

x
&

T
i

x
&

T
i

x

T
i

x

= m ai

= Fi

� (5)

where Fi = m ai, component of force field. The equation (5) is Lagrangean equation of Motion.

For a conservative system, Fi = �

V
ix

.

 Then equation (5) becomes

d
dt

T
i

x
&

T
i

x
&

T
i

x

=

V
ix

)

(
VT
i

x

= 0

or

d
dt

� (6)

Since the potential  V is a function of the coordinates  xi  alone.  If  we  introduce  the  Lagrangean

function

Then equation (6) becomes

L = T � V

d
dt

L
i

x
&

L
i

x

= 0

EXAMPLE  1

� (7)

Show that the covariant components of the acceleration vector in a spherical coordinate system

with

and

ds2 =

dx

21
)

(

+

1
dxx

22
)

(

+

21
x
)

(

sin

2

x

2

(

dx

23
)

  are

a1 =

1
x
&&

1
xx
(
&

22
)

3

1
xx
(
&

sin

x

22
)

[
(
[
(

d
dt

d
dt

a2 =

a3 =

]

(

21
x
)

2

x
&

1
x

sin

x

22
)

21
)
x
]3

x
&

sin

x

2

cos

x

2

23
)

(

x
&

�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
-
�
-
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
-
-
Analytical Mechanics

Solution

In spherical coordinate system, the metric is given by

ds2 =

dx

21
)

(

+

1
dxx

22
)

(

+

21
x
)

(

sin

2

x

2

(

dx

23
)

If v is velocity of the paiticle then

149

If T be kinetic energy then

v2 =

2
=(cid:247)

ds
dt

1
dx
dt

2

+

21
x
)

(

2

2

dx
dt

+

1
x

(

sin

x

22
)

2

3

dx
dt

v2 =

(

21
x
)
&

+

21
x
()

(

22
)

x
&

+

1
x

(

sin

x

22
()

23
)

x
&

T =

T =

1
2
1
2

2

mv
[
xm
(
&

21
)

+

21
x
()

(

x
&

22
)

+

1
x

(

sin

x

22
()

]23
)

x
&

� (1)

By Lagrangean equation of motion

= Fi and m ai = Fi
where Fi and ai are covariant component of force field and acceleration vector respectively.

x
&

x

d
dt

T
i

T
i

So,

Take i = 1,

d
dt

T
i

x
&

T
i

x

= m ai

m a1 =

d
dt

T
1
x
&

T
1

x

from (1), we get

� (2)

[
1
xx
(2
&

m
2

22
)

+

2

1
x

(sin

x

22
()

]23
)

x
&

m a1 =

a1 =

1
2

m

1
xd
&
dt

1
)2(
x
&

d
dt
[
1
xx
(
&

22
)

+

1
x

(sin

x

22
()

]23
)

x
&

Take i = 2,

a1 =

1
x
&&

1
xx
(
&

22
)

3

1
xx
(
&

sin

x

22
)

T
2

x

T
2

x
&
d
dt

m a2 =

m a2 =

a2 =

d
dt

1
2

m

[
(

d
dt

[
(

21
x
2)

2

x
&

]

1
2

xm
(2

21
)

2

sin

x

cos

x

2

23
)

(

x
&

]

21
x
)

2

x
&

21
x
)

(

sin

x

2

cos

x

2

23
)

(

x
&

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
-
-
-
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
-
150

Take i = 3

Tensors and Their Applications

m a3 =

m a3 =

d
dt

T
3
x
&

[
(2

x
&

1
2

m

d
dt

T
3
x

3

()

1
x

sin

x

22
)

] 0

[
x
&

d
dt

a3 =

3

21
x
)

(

(sin

x

]22
)

EXAMPLE  2

Use Lagrangean equations to show that, if a particle is not subjected to the action of forces then
its trajectory is given by yi = ait + bi where ai and bi are constants and the yi are orthogonal cartesian
coordinates.
Solution

If v is the velocity of particle. Then we know that,

i
yyg
&&
ij
where yi are orthogonal cartesian coordinates.

v2 =

j

Since

So,

But,

i �

gij = 0,
gij = 1, i = j

j

v2 =

(

iy&

2)

T =

T =

1
2
1
2

2mv

,

T is kinetic energy..

iym &
(

2)

The Lagrangean equation of motion is

d
dt

T
i
y
&

T
i
y

= Fi

Since particle is not subjected to the action of forces.
So, Fi = 0

Then

d
dt

1
2

iym
&2

� 0 = 0

m

i

yd
& = 0
dt

�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
Analytical Mechanics

151

or

yd i
& = 0
dt

iy& = ai
yi = ait + bi

where ai and bi are constant.

EXAMPLE  3

Prove that if a particle moves so that its velocity is constant in magnitude then its acceleration

vector is either orthogonal to the velocity or it is zero.
Solution

If vi be the component of velocity of moving particle then

vi =

dxi
dt

or

vi =

ix&

given |v| = constant.
Since

i
vvg
ij

j

=

| v

2|

= constant

Taking intrinsic derivative with respect to t, we get

(

i
ij vvg

j

)

= 0

td

g

ij

i

v
t

j

v

+

i

v

j

v
t

j

v
t

i

v
t

j

v

g

ij

g

ij

i

v
t

i

v
t

j

v

+

i

vg
ij

j

v

+

j

vg
ji

2

g

ij

g

ij

i

v
t

i
v
t

j

v

= 0

= 0

= 0

= 0, (Interchange dummy index i and j in second term)

= 0  as  gij = gji

This shows that acceleration vector

vi
t

 is either orthogonal to vi or zero i.e.,

vi
t

.0=

(cid:222)
(cid:222)
d
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
d
d
d
d
d
d
d
d
d
d
d
d
d
d
d
d
d
d
d
d
152

Tensors and Their Applications

8.6 APPLICATIONS  OF  LAGRANGEAN  EQUATIONS

(i) Free-Moving  Particle
If a particle is not subjected to the action of forces, the right hand side of equation (5), 148, vanishes.
Then we have

d
dt

T
i

x
&

T
i

x

= 0

� (1)

If xi be rectangular coordinate system, then T =

1
2

i

i yym &&

.

Hence, the equation (1) becomes m

iy&&

 = 0. Integrating it we get yi = ait + bi, which represents

a straight line.
(ii) Simple  Pendulum
Let  a  pendulum  bob  of  mass  m  be  supported  by  an  extensible  string.  In  spherical  coordinates,  the
metric is given by

If T be the kinetic energy, then

ds2 =

2

+

dr

2
dr

2

+

2

r

sin

2

2

d

T =

2

mv

=

1
2

1
2

2

rm
(
&

f+
r

22
&

+

2

r

sin

2

2

)

&

� (1)

O

r

Y 2

Y 1

R

mg cos f

P

mg

mg sin f

Y 3

Fig. 8.1.

from Lagrangean equation of motion

d
dt

T
i

x
&

T
i

x

= Fi

i = 1, 2, 3

x1 =

2

xr
,

f= x
,

3

q=

.

So, take

 x1 = r

�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
q
f
f
q
f
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
q
f
Analytical Mechanics

d
dt

T
r
&

T
r

= mg cos

R-

from (1), we have

r
&&

2

f
r
&

r

sin

2

2

qf
&

= g cos

R-
m

Take x2 =

,

 we have

+f
&&

r

2

r
&&
and take x3 =

r

sin

cos

2

&

= � g sin  f

,

 we have

d
dt

q&r
2

(

sin

2

)

= 0

153

� (2)

� (3)

� (4)

If the motion is in one plane, we obtain from equations (2), (3), and (4), by taking

.0=q&

r
&&

2f
&

r

= g cos

R-
m

r

f+f
2
r
&&
&&

= � g sin  f

If  r&  = 0,  we  get,

-=f
&&

g
r

  sinf

  which  is  equation  of  simple  pendulum  supported  by  an

inextensible string. For small angles of oscillation the vibration is simple harmonic. For large vibration
the solution is given in the term of elliptic functions.

8.7 HAMILTON�S  PRINCIPLE

If a particle is at the point P1 at the time  t1 and at the point  P2 at the time  t2, then the motion of the
particle takes place in such a away that

t

2

t
1

(

d+
i
xFT
i

)

dt

= 0

where xi = xi (t) are the coordinates of the particle along the trajectory and xi +
along a varied path beginning at P1 at  time t1 and ending at P2 at time t2.
Proof: Consider a particle moving on the curve

ixd

 are the coordinates

At time t, let particle is at P(xi). If T is kinetic energy. Then

ixC :

=

(txi

),

t

1

t

t

2

T =

1
2

i
xxgm
&&
ij

j

or T = T

(

i

i xx &
,

)

 i.e, T is a function of xi and

to be C is

ix& . Let C' be another curve, joining t1 and t2 close

xC'
:

i

t
),(

=

i
tx
)(

d+

i
tx
)(

�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
f
-
-
f
f
q
f
f
-
f
q
f
-
f
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
(cid:242)
d
�
�
e
Tensors and Their Applications

154

At t1 and t2

xi =

i

x

=

i

x

e+

i

x

( 1txi

)

= 0 and

txi
( 2

)

=

0

But T = T

(

i

i xx &
,

).

If  Td

 be small variation in T.. Then

+

i

x
&

T
i

x

i

x

Now,

{
(

t

2

t
1

+

xFT
i

)

Td

=

}
dt

i

=

=

T
i

x
&

t

2

t

1

T
i

x

i

x

+

�2
Tt
i

x

t
1

i
dtx

+

T
i

x
&

t

2

t
1

i

x
&

d+
xF
i

i

dt

T
i

x
&

i
dtx
&

+

t

2

t
1

i

xF
i

dt

Integrating second term by taking

T
&�
ix

 as 1st term

=

t

2

t

1

T
i

x

i
dtx

+

T
i

x
&

i

x

t

2

t
1

t
2

t
1

d
dt

T
i

x
&

i
dtx

+

t

2

t
1

i

dtxF
i

Since

i
tx

)(
1

d=
,0

i
tx
(

)

2

=

.0

i

x

T
i

x
&

t

2

t
1

=

.0

then

So,

(
d

t

2

t
1

+

d
xFT

i

)dt

i

= (cid:242)

�2

t

t

1

T
i

x

i

x

dt

t

2

t

1

d
dt

T
i

x
&

i
dtx

t
+ 2
t
1

i
dtxF
i

t

(
d2

t

1

d+
xFT
i

i

)

dt

= (cid:242)

t

2

t

1

T
i
x

d
dt

+(cid:247)

T
i
x
&

i

d
dtxF

i

since particle satisfies the Lagrangean equation of motion. Then

d
dt

T
i

x

T
i

x
&

d
dt

T
i

x

T
i

x
&

=  Fi

= � Fi

or

d
(cid:222)
d
d
d
�
�
d
�
�
(cid:242)
d
d
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
d
�
�
d
�
�
(cid:242)
(cid:242)
(cid:242)
d
d
�
�
d
�
�
(cid:242)
(cid:242)
(cid:242)
d
d
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
d
�
�
d
�
�
d
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
d
�
�
(cid:242)
(cid:242)
d
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
d
�
(cid:242)
d
(cid:242)
�
�
�
�
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
Analytical Mechanics

So,

(

d+
xFT
i

i

)

dt

=  (cid:242)

t

-2
[

1

t

+

F
i

(

d+
xFT
i

i

)

dt

=  0

t

2

t

1

t

2

t
1

dtxF

]

i

i

155

Proved.

8.8 INTEGRAL  OF  ENERGY

THEOREM 8.3 The motion of a particle in a conservative field of force is such that the sum of its
kinetic and potential energies is a constant.
Proof: Consider a particle moving on the curve

C : xi =
t
At time t, let particle is at P (xi). If T is kinetic energy. Then

(txi

),

t

t

2

1

or

T =

T =

1
2

1
2

i
xxgm
&&
ij

j

i
ij vvgm

j

As T is invariant. Then
Taking intrinsic derivative with respect to t, we get

dT
dt

=

=

=

=

=

=

T
t

1
t 2

i
ij vvgm

j

1
2

1
2

1
2

1
2

gm

ij

gm

ij

gm

ij

2
gm

ij

i

v
t

i

v
t

i
v
t

i

v
t

j

v

+

i
v

j

v
t

j

v

+

g

ij

j

v

+

g

ji

j

v
t

i

v

i
v
t

j

v

j

v

as

g =
ij

g

ji

dT
dt

dT
dt

=

gm

ij

=

gm

ij

i

v
t

j

v
t

j

v

i

v

or

, Since i and j are dummy indices.

(cid:242)
d
d
(cid:242)
d
�
�
d
d
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
d
d
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
d
d
d
d
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
d
d
d
d
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
d
d
d
d
d
d
d
d
d
d
156

Tensors and Their Applications

=

vagm
ij

j

i

as

j

v
t

=

j

a

= m ai vi,

since gij aj = ai

= Fi vi,

dT
dt

dT
dt

Since Fi = m ai is a covariant component of force field.

But given Fi is conservative, then

Fi = �

V
ix

,

 where V is potential energy..

So,

dT
dt

= �

=

=

dT
dt

V
i

x

V
i

x

i

v

i

dx
dt

dV
dt

dT +
dt

dV
dt

= 0

d
dt

+
VT
(

=
0)

T + V = h, where h is constant.

8.9 PRINCIPLE  OF  LEAST  ACTION

Let us consider the integral

A = (cid:242)

P
2

p
1

dsmv
.

evaluated over the path

Proved.

(1)

ixC :

=

(txi

),

t

1

t

t

2

where C is the trajectory of the particle of mass m moving in a conservative field of force.
In the three dimensional space with curvilinear coordinates, the integral (1) can be written as

A =

P
2

gm(cid:242)

p
1

i

dx
dt

ij

j

dx

=

Pt
(
2

)

Pt
(
1

)

gm

ij

i

dx
dt

j

dx
dt

dt

d
d
�
�
�
�
�
�
-
-
(cid:222)
(cid:222)
�
�
(cid:242)
Analytical Mechanics

157

Since T =

1
2

gm

ij

i

dx
dt

j

dx
dt

, we have

A = (cid:242)

Pt
(
2

)

Pt
(
1

)

2

dtT

This integral has a physical meaning only when evaluated over the trajectory C, but its value can

be computed along any varied path joining the points P1 and P2.

Let us consider a particular set of admissible paths C' along which the function T + V, for each

value of parameter t, has the same constant value h. The integral A is called the action integral.

The  principle  of  least  action  stated  as  �of  all  curves  C'  passing  through  P1  and  P2  in  the
neighbourhood of the trajectory C, which are traversed at a rate such that, for each C', for every value
of t, T + V = h, that one for which the action integral  A is stationary is the trajectory of the particle.�

8.10 GENERALIZED  COORDINATES

In  the  solution  of  most  of  the  mechanical  problems  it  is  more  convenient  to  use  some  other  set  of
coordinates  instead  of  cartesian  coordinates.  For  example,  in  the  case  of  a  particle  moving  on  the
 are only
surface of a sphere, the correct coordinates are spherical coordinates r,
two variable quantities.

 where  q

 and  f

,

Let there be a particle or system of n particles moving under possible constraints. For example, a
point mass of the simple pendulum or a rigid body moving along an inclined plane. Then there will be
a minimum number of independent coordinates required to specify the motion of particle or system of
particles. The set of independent coordinates sufficient in number to specify unambiguously the system
configuration  is  called  generalized  coordinates  and  are  denoted  by
 where  n  is  the  total
number of generalized coordinates or degree of freedom.

nq

...

q

q

,

,

2

1

Let there be N particles composing a system and let

 be the positional
coordinates of these particles referred to some convenient reference frame in E3. The system of N free
particles  is  described  by  3N  parameters.  If  the  particles  are  constrained  in  some  way,  there  will  be

(),3,2,1

,...2,1

(,)

N

)

i

xi
(

=

=a

certain relations among the coordinates

ix )
( a

 and suppose that there are r such independent relations,

f

i

(

1
x
)1(

,

2
x
)1(

,

x

3
)1(

;

1
x
)2(

,

2
x
)2(

,

3
x
)2(

1
x
;...
(

N

)

2
x
(
N

)

3
x
(
N

)

)

 =   0, (i = 1, 2, ..., r)

� (1)

By using these r equations of constraints (1), we can solve for some r coordinates in terms of the
remaining 3N � r coordinates and regard the latter as the independent generalized coordinates qi. It is
more convenient to assume that each of the 3N coordinates is expressed in terms of 3N  �  r  =  n
independent variables qi and write 3N equations.

ix )
( a

= ix )
( a

1
qq
,

(

2

,...,

n
tq
),

� (2)

where we introduced the time parameter t which may enter in the problem explicitly if one deals with
moving  constraints.  If  t  does  not  enter  explicitly  in  equation  (2),  the  dynamical  system  is  called  a
natural  system.

The velocity of the particles are given by differentiating equations (2) with respect to time. Thus

ix )
( a&

=

x

i
(
q

)

j

j

q
&

+

)

x

i
(
t

� (3)

f
q
a

�
�
�
�
a
a
158

Tensors and Their Applications

The time derivatives
For symmetry reasons, it is desirable to introduced a number of superfluous coordinates qi  and
describe the system with the aid of k > n coordinates q1, q2,..., qk. In this event there will exist certain
relations of the form

iq&  of generalized coordinates qi the generalized velocities.

f

j

( 1
q

....,

k
tq
),

= 0

Differentiating it we get

j

f

i

q

+

i

q
&

j

f

t

= 0

� (4)

� (5)

It is clear that they are integrable, so that one can deduce from them equations (4) and use them

to eliminate the superfluous coordinates .

In some problems, functional relations of the type

j

1

2

k
tq
),
&

1

k
qq
;
&

,

,...,

,...,

qqF
(

= 0, ( j  = 1, 2, 3, ..., m)

� (6)
arise which are non-integrable. If non-integrable relations (6) occurs in the problems we shall say that
the given system has k � m degrees of freedom, where m is the number or independent non-integrable
relations (6) and k is the number of independent coordinates. The dynamical systems involving non-
integrable relations (6) are called non-holonomic to distinguish them from holonomic systems in which
the number of degrees of freedom is equal to the number of independent generalized coordinates.

In other words, a holonomic system is one in which there are no non-integrable relations involving

the generalized velocities.

8.11 LAGRANGEAN  EQUATIONS  IN  GENERALIZED  COORDINATES

Let there be a system of particle which requires  n independent generalized coordinates or degree of
freedom to specify the states of its particle.

The position vectors xr are expressed as the function of generalized coordinates

q i

i
(,

=

2,1

n
,...,

)

and the time t i.e.,

xr = xr

1
qq
,

(

2

,...,

n
tq
,

);

( =r

),3,2,1

The velocity

rx&  of any point of the body is given by

rx& =

=

r

x

j

q

r

x

j

q

j

dq
dt

+

r

x
t

j

+

q
&

r

x
t

,

 ( j = 1, 2,  ..., n)

where

 are the generalized velocities.

jq&
Consider the relation, with n degree of freedom,
nq

1
qq
,

,...,

(

2

xr = xr

)

� (1)

involve n independent parameters qi. The velocities

rx&  in this case are given by

rx& =

r

x

j

q

,j

q
&

(r = 1, 2, 3;  j = 1, 2, ..., n)

�(2)

�
�
�
�
�
�
�
�
�
�
�
�
�
�
Analytical Mechanics

159

where

jq&

 transform under any admissible transformation,

kq =

q

k

( 1
q

q
,...,

n

),

(k = 1, 2, ..., n)

� (3)

in accordance with the contravariant law.

The kinetic energy of the system is given by the expression of the form

T =

1
2

gm

rs

r
xx
&
&

s

,

(r,s = 1,2,3,)

� (4)

where m is the mass of the particle located at the point xr. The grs are the components of the metric
tensor.

Substituting the value of

rx&  from equation (2), then equation (4) becomes
x

x

r

s

T =

gm

1
2

rs

i

q

j

q

j

i
qq
&&

where

T =

1
2

j

i
ij qqa
&&

ija =

gm

rs

r

x

i

q

s

r

j

q

,

� (5)

(r, s = 1, 2, 3), (i, j = 1, ..., n)

1
2

j

i
ij qqa
&&

Since T =

ija
are components of a covariant tensor of rank two with respect to the transformations (3) of generalized
coordinates.

ija  are symmetric, we conclude that the

 is an invariant and the quantities

Since  the  kinetic  energy  T  is  a  positive  definite  form  in  the  velocities

i a
|,

ij

q&

>
.0|

  Then  we

construct the reciprocal tensor

ija .

Now, from art. 8.5, Pg. 146, by using the expression for the kinetic energy in the form (5), we

obtain the formula,

d
dt

T
i

q
&

where the Christoffel symbol

Put

T
i

q

l

kj

a

il

=

+

l

q
&&

l

kj

k

j
qq
&&

(6)

 are constructed from the tensor akl.

l

+

l
q
&&

kj
so, the equation (6), becomes

k

j
qq
&
&

= Ql

d
dt

T
i

q
&

T
i

q

= ail Ql

=   Qi (i = 1, 2, ..., n)

� (7)

S
�
�
�
�
S
�
�
�
�
S
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
160

Tensors and Their Applications

Now, from the realtions

=

r

j

x
&
q
&

r

x

j

q

,

r

i

x
&
q

=

2

r

x

i
qx

j

j

q
&

 and

(2) and (4).

r

i

x&
q

=

d
dt

r

i

x
q

 and using equations

Then by straightforward calculation, left hand member of equation (7) becomes

x
r q
in which aj =  gij ai is acceleration of the point P.

d
dt

= (cid:229)

am

T
i

T
i

q
&

q

r

i

Also, Newton's second law gives

m ar = Fr

where

s

rF '

 are the components of force F acting on the particle located at the point P..

From the equation (9), we have

am

r

i

x
r q

= (cid:229)

x
F
r q

r

i

and equation (8) can be written as

d
dt

T
i

q
&

T
i

q

=

x
F
r q

r

i

comparing (7) with (8), we conclude that

iQ = (cid:229)

x
F
r q

r

i

where vector Qi is called generalized force.

The equations

d
dt

T
i

q
&

T
i

q

= Qi

� (8)

� (9)

� (10)

� (11)

are known as Lagrangean equations in generalized coordinates.

They give a system of n second order ordinary differential equations for the generalized coordinates qi.
The solutions of these equations in the form

Represent the dynamical trajectory of the system.

iqC :

= qi (t)

If there exists a functions

q"
)
such systems,  equation (11) assume the form

1
qqV
(

,...,

,

2

d
dt

L
i

q
&

L
i

q

= 0

where L = T � V is the kinetic potential.

 such that the system is said to be conservative and for

� (12)

�
�
�
�
�
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:229)
�
�
�
�
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:229)
�
�
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
Analytical Mechanics

Since L (q,  q& ) is a function of both the generalized coordinates and velocities.

dL
dt

=

+

i
q
&&

L
i

q
&

L
i

q

i

q
&

from (12), we have

L
iq

 =

d
dt

L
iq
&

.(cid:247)

Then equation (13), becomes

dL
dt

=

=

L
i

q
&

+

i
q
&&

d
dt

L
i

q
&

i

q
&

d
dt

L
i

q
&

i

q
&

since L = T � V but the potential energy V is not a function of the

iq&

since

L
&�
i
q

i

q
&

=

T i
q
&
i
q
&

2=
T

T=

1
2

i
qqa
&
&
ij

j

.

Thus, the equation (14) can be written in the form

161

� (13)

� (14)

(
Ld

)2
T

dt

=

+

(

VTd
dt

)

=

0

which implies that T + V = h (constant).

Thus, along the dynamical trajectory, the sum of the kinetic and potential energies is a constant.

8.12 DIVERGENCE  THEOREM,  GREEN'S  THEOREM,  LAPLACIAN  OPERATOR  AND

STOKE'S  THEOREM  IN  TENSOR  NOTATION

(i) Divergence  Theorem
Let  F

r  be a vector point function in a closed region V bounded by the regular surface  S. Then

r
F

div

= (cid:242)

S

V

r
dsnF �

where  n�  is outward unit normal to S.

� (1)

Briefly the theorem states that the integral with subscript V is evaluated over the volume V while
r  over the surface S.

the integral in the right hand side of (1) measures the flux of the vector quantity  F

In orthogonal cartesian coordinates, the divergence of  F

r  is given by the formula

div  F

r =

+

1

F

1

x

2

F

2

x

+

3

F

3

x

� (2)

�
�
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
-
(cid:242)
(cid:215)
�
�
�
�
�
�
162

Tensors and Their Applications

r
If the components of  F

F i  then the covariant derivative of F i  is

 relative to an arbitrary curvilinear coordinate system X are denoted by

i
jF , =

+

i

j

F
x

i
jk

k

F

The invariant

i
jF,

r .
 in cartesian coordinates represents the divergence of the vector field  F

Also,

nF �(cid:215)r

=

i

nFg

ij

j

=

i
nF

i

since

ng
ij

j

=

n

i

Hence we can rewrite equation (1) in the form

i
i dVF,

V

= (cid:242)

S

i

dSnF
i

(ii) Symmetrical  form  of  Green's  Theorem

� (3)

Let
and  y

1
xx
,

(

2

,

3
x

)

 and

1
xx
,

(

2

,

3
x

)

 be two scalar function in V.. Let

 and

i

i

 be the gradients of  f

 respectively, so that

=

=

i

x�
i

 and

 =

=

i

i

x�

Put Fi =

i

 and from the divergence of   we get

j
iF, =

ij
Fg
i

,

j

=

ij

(

g

y+

i

,

j

)

j

i

Substituting this in equation (3), we get
y+

ij

g

(

)

dV

i

j

i

,

j

dSni

i

=

S

V
y=y

Since

,i

 then

Also, the inner product

ijg

ijg

=

i

,

j

2

 can be written as

i

j

ijg

=

i

j

.

where  (cid:209)

 denote the gradient and

2

 denote the Laplacian operator..

Hence the formula (4) can be written in the form

� (4)

� (5)

ij

(

g

+

ij

g

,
ji

.

i

)

dV

J

V

(

V

2

(cid:209)+y

dV)

= (cid:242)
= (cid:242)

where

n
�

y=y

=

i

n

i

2

dV

=

V

n

.

n�

n�

n�

S

S

S

dS

dS

V

dV

� (6)

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:242)
f
y
f
y
f
(cid:209)
f
�
f
y
(cid:209)
y
�
y
f
y
f
f
y
(cid:242)
f
f
y
(cid:242)
f
y
(cid:209)
y
y
(cid:209)
f
y
f
y
y
(cid:209)
f
(cid:209)
(cid:209)
(cid:242)
f
y
f
y
y
(cid:209)
(cid:215)
f
(cid:242)
y
(cid:209)
(cid:215)
f
(cid:209)
f
y
(cid:209)
(cid:215)
f
(cid:242)
y
(cid:209)
f
(cid:242)
(cid:242)
y
(cid:209)
(cid:215)
f
(cid:209)
-
y
(cid:209)
(cid:215)
f
�
y
�
(cid:209)
(cid:215)
Analytical Mechanics

Interchanging  f

 and  y

 in equation (5), we get

dV2

n�

=

S

V

V

dV

Subtracting equation (5) from equation (6), we get

2

(

V

2

dV)

= (cid:242)

S

n

dS

n

This result is called a symmetric form of Green's theorem.

(iii ) Expansion  form  of  the  Laplacian  Operator
The Laplacian of  y

 is given by

2

=

ijg

i

,

j

from (5)

163

� (7)

� (8)

when written in the terms of the christoffel symbols associated with the curvilinear coordinates

xi covering E3,

2

ij

g

=

2

i
xx

j

k

ji

k

x

and the divergence of the vector F i  is

i
iF, =

+

i

F

i

x

i

ij

j

F

i

ij

log

g

=

x j

But we know that

The equation (10) becomes

i
iF, =

i
iF, =

+

(

i

F

i

x

1

g

log

j

Fg

j

x

i

)

Fg
i

x

or

If putting F i  =

ij

g

=

ij

g

j

x

 in equation (11), we get

j

ijg

=

,
ij

1

g

ij

gg

i

x

j

x

� (9)

�(10)

� (11)

� (12)

But from equation (5), we know that
ijg

=

2

,
ij

(cid:242)
f
(cid:209)
y
(cid:242)
(cid:242)
f
(cid:209)
(cid:215)
y
(cid:209)
-
f
(cid:209)
(cid:215)
y
(cid:242)
f
(cid:209)
y
-
y
(cid:209)
f
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
f
�
y
-
�
y
�
f
y
(cid:209)
y
y
(cid:209)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
y
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
y
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
y
�
y
�
y
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
y
�
�
y
(cid:209)
y
164

Tensors and Their Applications

Hence equation (12) becomes

2

=

ij

g

=

,
ij

1

g

ij

gg

i

x

j

x

It is expansion form of Laplacian operator.

(iv) Stoke's  Theorem

Let a portion of regular surface S be bounded by a closed regular curve C and let  F
function defined on S and on C. The theorem of Stokes states that

r  be any vector point

.�
n

curl

r
dsF

= (cid:242)

r
F l�.

ds

S

C

� (13)

where  l
cartesian coordinates are determined from

 is the unit tangent vector to  C and curl  F

r  is the vector whose components in orthogonal

e
1

e
2

e
3

curl  F

r =

1
x
1
F

2

x
2
F

 =

3

x
3
F

r
F

� (14)

where ei being the unit base vectors in a cartesian frame.

We consider the covariant derivative Fi,j of the vector Fi and form a contravariant vector

ijkF ,
r .
we define the vector G to be the curl of  F

Gi = �

kj

� (15)

Since  n� . curl  F

r  =

i

-=

Gn
i

ijk

nF
,
kj

i

 and the components of the unit tangent vector l

 and

dxi
ds

.

Then equation (13) may be written as

ijk

dsnF ,
i

kj

S

= (cid:242)

F
i

C

i

dx
ds

ds

The integral  (cid:242)

i

i dxF

c

 is called the circulation of  F

r  along the contour C.

� (16)

8.13 GAUSS'S  THEOREM

The  integral  of  the  normal  component  of  the  gravitational  flux  computed  over  a  regular  surface  S
containing gravitating masses within it is equal to  mp4
Proof: According to Newton's Law of gravitation, a particle P of mass m exerts on a particle Q of unit

 where m is the total mass enclosed by S.

mass located at a distancer r from P. Then a force of magnitude

F =

m
2r

.

Consider a closed regular surface S drawn around the point P and let  q

 be the angle between the
unit outward normal to  n�   to  S  and  the  axis  of  a  cone  with  its  vertex  at  P..  This  cone  subtends  an
element of surface dS.

y
(cid:209)
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
y
�
�
y
(cid:242)
�
�
�
�
�
�
�
(cid:209)
e
e
(cid:242)
e
-
Analytical Mechanics

165

The flux of the gravitational field produced by m is

r
dSnF �.

S

= (cid:242)

S

m

cos
2

r

2
dwr
cos

where dS =

 and dw is the solid angle subtended by dS.

2dwr
cos
Thus, we have,

r
dSnF �.

S

= (cid:242)

S

dwm

p=
4

m

� (1)

F

dS

n

r

dw

P
m

S

Fig. 8.2.

If there are n discrete particles of masses mi located within S, then
= (cid:229)

m
i

cos
2

n

i

r
nF �.

=

1

i

r
i

and total flux is

r
�.
dSnF

4

=

S

n

=

1

i

im

� (2)

The  result  (2)  can  be  easily  generalized  to  continuous  distributions  of  matter  whenever  such

distribution no where melt the surface S.

The contribution to the flux integration from the mass element  r dV contained within V, is
dV

cos

r
�.
dSnF

= (cid:242)

S

2

r

S

dS

and the contribution from all masses contained easily within S is

r
�.
dSnF

= (cid:242)

cos

dV

2

r

S

V

dS

S

� (3)

(cid:242)
q
q
q
(cid:242)
q
(cid:242)
(cid:229)
p
(cid:242)
q
r
(cid:242)
(cid:242)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
r
q
q
� (4)

� (5)

166

Tensors and Their Applications

 denotes the volume integral over all bodies interior to S. Since all masses are assumed to be

where  (cid:242)
interior to S,r never vanishes. So that the integrand in equation (3)  is continuous and one can interchange
to order of integration to obtain

V

r
�.
dSnF

S

= (cid:242)

V

cos

dS

s

2

r

dV

dS

p=

.4

 Since it represents the flux due to a unit mass contained within S.

cos

2

r

But  (cid:242)
S
Hence

r
�.
dSnF

4

=

V

p=
dV 4

m

S

where m denotes the total mass contained within S.

Proved.
Gauss's theorem may be extended to cases where the regular surface S cuts the masses, provided

that the density S is piecewise continuous.

Let S cut some masses. Let S' and S" be two nearby surfaces, the first of which lies wholly within
S and the other envelopes S. Now apply Gauss's theorem to calculate the total flux over S" produced by
the distribution of masses enclosed by S since S" does not intersect them.

We have

((cid:242)

r
)�.
nF i

dS

=

mp4

S"

 refers to the flux due to the masses located inside  S and m is  the  total
where the subscript i on
mass within S. On the other hand, the net flux over S' due to the masses outside S, by Gauss's theorem
is

nF �(cid:215)r

r
)�
nF

(

o ds

= 0

S
where the subscript o on

'

nF �(cid:215)r

 refers to the flux due to the masses located outside  S.

Now if we S' and S" approach S, we obtain the same formula (5) because the contribution to the

total flux from the integral  (cid:242)

r
)�
nF

(

o dS

 is zero.

S

'

8.14   POISSON'S  EQUATION

By divergence theorem, we have

and by Gauss's Theorem,

from these, we have

r
dsnF �.

= (cid:242)

r
dVF

 div

V

4

dV

V

r
dsnF �.

=

S

S

(cid:242)
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
q
r
q
(cid:242)
(cid:242)
r
p
(cid:242)
(cid:215)
(cid:215)
(cid:242)
(cid:242)
(cid:242)
r
p
Analytical Mechanics

167

div(

r
F

4

)

dV

= 0

Since this relation is true for an arbitrary V and the integrand is piecewise continuous, then

v

By the definition of potential function V, we have

r
div  F

r4
= p

and

So,

r =
F
V(cid:209)div

=

V

V2

div  F
V

r = p
r4
r4
) = p

div (

V2

= �

r4

which is equation of poisson.

If the point P is not occupied by the mass, then r
the potential function V satisfies Laplace's equation

V2

= 0

8.15 SOLUTION  OF  POISSON�S  EQUATION

 = 0. Hence at all points of space free of matter

We  find  the  solution  of  Poisson�s  Equation  by  using  Green�s  symmetrical  formula.  We  know  that
Green's symmetrical formula

2

(

2

dV)

=

where V is volume enclosed by S and  f

V

n

S

 and  y

dS

n

 are scalar point functions.

� (1)

Put

1=f
r

 where r is the distance between the points

1
xxP
(

,

2

,

3
x

)

 and

1
yyQ
(

,

2

,

3

y

)

 and V  is

the gravitational potential.

n

S

Q y( )

r

P x( )

n

Fig. 8.3.

(cid:242)
p
r
-
-
(cid:209)
(cid:209)
-
(cid:209)
(cid:209)
p
(cid:209)
(cid:242)
f
(cid:209)
y
-
y
(cid:209)
f
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
f
�
y
-
�
y
�
f
168

Since

  has  a  discontinuity  at

x =
i

y

,i

  delete  the  point  P(x)  from  region  of  integration  by

Tensors and Their Applications

surrounding it with a sphere of radius  e

 and volume  V'. Apply Green�s symmetrical formula to the

1
r

region V � V' within which

1
r

 and V possess the desired properties of continuity..

In region

1
r
Then equation (1) becomes

(cid:209)=f

V'

V

,

2

2

=

.0

1
r

V

V'

2

dV

= (cid:242)

S

1
nr

1
r
n

+

dS

S'

1
nr

1
r
n

ds

� (2)

where  n�  is the unit outward normal to the surface S + S' bounding V � V' . S' being the surface of the

sphere of radius  e

 and

-=

n

.

r

Now

1
nr

S

'

1
r
n

dS

=

=

=

=

1
nr

S

'

( )

1
r
n

1
r

1
r

r

r

r

r

S

'

S

'

S

'

1
r
r

dS

2
dwr

2

r

y+

dw

dw

4

r

e=

r

S'

 � (3)

where  y

 is the mean value of V over the sphere S' and w denote the solid angle.

                              Let

1
xx
,

(

2

,

x

3

)

y=

(

P

)

as

?r

0

 then as

0

 from (3), we have

1
nr

S'

1
r
n

=

4

(

P

)

Then equation (2) becomes

V

2

Since

 then  (cid:242)

0

V'

1
r

21
r

dV

=

S

1
nr

1
r
n

dS

4

(

P

)

dV

 = 0.

(cid:209)
-
(cid:242)
-
y
(cid:209)
(cid:242)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
y
-
�
y
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
y
-
�
y
�
�
�
�
�
(cid:242)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
y
-
�
y
�
(cid:242)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
y
-
�
y
�
-
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
y
-
�
y
�
-
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
y
�
-
(cid:242)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
y
-
�
y
�
(cid:242)
y
p
-
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
y
�
e
-
y
?
e
(cid:242)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
y
-
�
y
�
p
y
-
(cid:242)
y
(cid:209)
p
y
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
y
-
�
y
�
(cid:242)
?
e
y
(cid:209)
Analytical Mechanics

(Py

)

=

1
4

1
r

V

2

dV

+

1
4

1
nr

S

dS

1
4

S

1
r dS
n

This gives the solution of Poisson's equation at the origin.
If  y

 is regular at infinity, i.e., for sufficiently large value of r,  y

 is such that

(

)

m�
r

 and

r

m
2r

169

� (4)

� (5)

where m is constant.

If integration in equation (4) is extended over all space, so that

?r

. Then, using equation (5),

equation (4) becomes

(Py

)

=

1
4

2

r

dV

� (6)

But  y

 is a potential function satisfying the Poisson's equation i.e.

2

-=y

4

.

Hence, from (6), we get

This solution is Unique.

(Py

)

= (cid:242)

dV
r

EXERCISES

1. Find, with aid of Lagrangian equations, the trajectory of a particle moving in a uniform gravitational

field.

2. A particle is constrained to move under gravity along the line yi = cis (i = 1, 2, 3). Discuss the motion.
3. Deduce from Newtonian equations the equation of energy T + V = h, where h is constant.
4. Prove that

i
i dSn

S

=

V

2

dV

where

=

i

.

i

x�

5. Prove that the curl of a gradient vector vanishes identically.

(cid:242)
(cid:242)
(cid:242)
�
�
y
p
-
�
y
�
p
y
(cid:209)
p
-
y
�
�
y
�
�
(cid:242)
�
y
(cid:209)
p
-
p
r
(cid:209)
�
r
(cid:242)
y
y
(cid:209)
(cid:242)
y
�
y
CHAPTER � 9

CURVATURE  OF  CURVE,  GEODESIC

9.1 CURVATURE  OF  CURVE:  PRINCIPAL  NORMAL

Let C  be a curve in a given
as functions of the arc length s. Then the unit tangent t to the curve the contravariant components

ix  of the current point on the curve expressed

nV  and let the coordinates

it =

dxi
ds

...(1)

The intrinsic derivative (or desired vector) of

 along the curve  C  is called the  first  curvaturee
nV  and is denoted by  pr . The magnitude of curvature vector  pr is  called
nV  and is denoted by K:

it

vector of curve C  relative to
first curvature of C relative

So,

where

K =
iP  are contravariant components of  pr  so that

ppg

ij

i

j

iP =

i ,

t

j

j

dx
ds

i

t

j

x

+

t

i

j

i

dx
ds

i

t

j

x

j

dx
ds

+

t

j

dx
ds

i

j

i

dt
ds

+

dx
ds

j

dx
ds

i

2
xd
2

ds

+

j

dx
ds

k

dx
ds

i

j

i

jk

,

 Replacing dummy index  a

 by k

=

=

=

=

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
Curvature of Curve, Geodesic

171

ip =

i

2
xd
2

ds

+

If   n�   is a unit vector in the direction of

pr =

nk �

j

k

i

dx
ds

dx
ds
,pr  then we have

kj

  as

i

kj

=

i

jk

The vector  n�  is called the Unit principal normal.

9.2 GEODESICS

Geodesics on a surface in Euclidean three dimensional space may be defined as the curve along which
lies the shortest distance measured along the surface between any two points in its plane.
But  when  the  problem  of  find  the  shortest  distance  between  any  two  given  points  on  a  surface  is
3V  as follows:
treated properly, it becomes very complicated and therefore we define the geodesics in
(i) Geodesic in a surface  is defined as the curve of stationary length on a surface between any

(ii)

two points in its plane.
In
everywhere zero.

3V   geodesic  is  also  defined  as  the  curve  whose  curvature  relative  to  the  surface  is

By generalising these definitions we can define geodesic in Riemannian

nV as

(i) Geodesic  in  a  Riemannian
joining two points on it.

nV   is  defined  as  the  curve  of  minimum  (or  maximum)  length

(ii) Geodesic is the curve whose first curvature relative to

nV  is zero at all points.

9.3 EULER'S    CONDITION

THEOREM 9.1 The Euler condition for the integral

t
1

o
to be staionary are

t

i
i
xxf
(
&

,

)

dt

f

i

x

d
dt

f

i

x
&

= 0

where

ix& =

dxi
dt

 i = 1, 2, 3,...

Proof: Let C be a curve in a  nV  and A, B two fixed points on it. The coordinates
P on C are functions of a single parameter t. Let  0t
A and B respectively.

ix  of the current point
 and  1t  be the values of the parameter for the points

To find the condition for the integral
t
1

i
i
xxf
&

(

,

)

dt

to be stationary.

t

0

...(1)

Let the curve suffer an infinitesimal deformation to

 the points A and B remaining fixed while

the current points P(xi) is displaced to P' (xi + h

i = 0 at A and B both.

,C�
i) such that h

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
(cid:242)
172

Tensors and Their Applications

A

P'

P

C�

C

B

In this case the value of integral (1) becomes  I �
So,

Fig. 9.1

By Taylor's theorem

+
yhxF

(

,

I � =

(
xF

i

t
1

t

0

h+

i

,

i

x
&

h+

i

&

)
dt

+

k

)

=

,(
yxf

)

+

h

+

k

f
x

(cid:215)+(cid:247)

f
y

I � = (cid:242)

I � =

t
1

t

0

t
1

t

0

i
xxF
&

(

,

i

)

+

i
xxF
(
&

,

i

)

dt

+

F
i
x

t
1

t

0

i

+

F
i
x

i

(cid:215)+(cid:247)

&

dt

+h
i

F
i
x

F
i
x
&

i

&

dt

(Neglecting higher order terms in small quantities

i

)

I � =

+

I

t
1

t

0

Id

=

I

=

I

F
i
x

t
1

t

0

i

+

i

&

i

dt

f
x
&

+h
i

F
i
x

F
i
x
&

i

&

dt

=

i

&

i

j

z
x

j

x
&

�1
t

t

0

F
i
x
&

i

dt

&

=

F
i
x
&

i

t
1

t
0

�

t
1

t
0

d
dt

F
i
x
&

i

dt

=

t
1

t

0

d
dt

F
i
x
&

i

dt

...(2)

...(3)

Then

where

Now,

(cid:242)
(cid:215)
(cid:215)
(cid:215)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
(cid:215)
(cid:215)
?
(cid:246)
(cid:231)
?
(cid:230)
h
�
�
h
�
�
(cid:242)
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
h
�
�
�
�
h
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
h
�
�
h
�
�
(cid:242)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
h
�
�
�
�
-
�

�
�
h
(cid:242)
h
�
(cid:242)
h
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
h
�
�
(cid:242)
h
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
Curvature of Curve, Geodesic

since

F
i

x
&

i

t
)(

t

1

t

0

=

,0

i

(

t

)

1

h=

i

(

t

)

0

=

0

Then equation (2) becomes

The integral I is stationary if

d I

Id

F
i
x

d
dt

F
i
x
&

i

dt

it

t

0

= (cid:242)
.0=

i.e., if (cid:242)

t
1

t

0

F
i
x

d
dt

F
i
x
&

i

dt

= 0

173

...(4)

Since

i

 are arbitrary and hence the integrand of the last integral vanishes, so that

F
i
x

d
dt

F
i

x
&

= 0,  (i = 1, 2,..., n)

� (5)

Hence the necessary and sufficient condition for the integral (1) to be stationary are

F
i

x

�

d
dt

F
i

dx

= 0,

(i = 1, 2, �, n)

These are called Euler's conditions for the integral I to be stationary.

9.4 DIFFERENTIAL  EQUATIONS  OF  GEODESICS

To obtain the differential equations of a geodesic in a
(or maximum) length joining two points A and B on it.
Proof: Consider a curve C in  nV  joining two fixed points A and B on it and
point P on it.

,nV  using the property that it is a path of minimum

The length of curve  C is

s = (cid:242)

B

A

g

ij

i

xd
dt

j

xd
dt

dt

ds
dt

=

g

ij

j

i

xd
dt

xd
dt

Put

or

ds
dt

=

g

ij

s& =

xd
dt

i

j

xd
dt
=&&

j

i

xxg
ij

=

 F

(say)

F

Then equation (1) becomes

s = (cid:242)

B

A

dtF

)(txi

 be the coordinates of

...(1)

...(2)

...(3)

�
�
�
�
�
�
�
�
h
�
�
�
�
�
�
h
�
�

h
�
�
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
h
�
�
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
h
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
174

Tensors and Their Applications

Since curve  C  is  geodesic,  then  the  integral  (3)  should  be  stationary,  we  have  from  Euler's

condition

F
i

x

d
dt

F
i

x
&

= 0

Differentiating equation (2) with respect to

kx  and

kx&

 we get,

...(4)

and

F
kx

F
kx
&�

F
kx
&

d
dt

=

1
& �
s
2

g

x

ij
k

j

i
xx
&&

2

=

1
s
2
&

i

=(cid:247)

xg
&
ik

i
xg
&
ik

1
s
&

=

1
2

s
&

gs
&&

ik

+

i

x
&

Putting these values in equation (4), we get

1
s
2
&

g

x

ij

k

j

i
xx
&&

1
2

s

xgs
&&
&
ik

i

+

1
s
&

g

x

ik
j

i

j
xx
&
&

+

g

x

ik
j

i

j
xx
&
&

+

i

xg
&&
ik

1
s
&

i
xg
&&
ik

 = 0

1
s
&

1
s
&

i
xg
&&
ik

i

+

xg
&
ik

s
&&
s
&

g

x

ik
j

1
2

g

x

ij
k

j

i
xx
&&

= 0

i
xg
&&
ik

i

xg
&
ik

s
&&
s
&

multiplying it by

kmg

,

 we get

[
]
,+
xxijk
&

&

i

j

= 0

But

km

g

i
xg
&&
ik

s
&&
s
&

km

g

xg
&
ik

i

+

km

g

[
]
i
,
xxijk
&&

j

= 0

g

kmg

ik

d=

m
i

 and

[
g km ,
ijk

]

=

m

i

j

m

x
&&

m

+

x
&

s
&&
s
&

m

i

j

j

i
xx
&&

= 0

m

2
xd
2

dt

m

dx
dt

s
&&
s
&

+

m

kj

j

dx
dt

k

dx
dt

 = 0

Replacing

dummy

index

i by
k

...(5)

This is the differential equation for the geodesic  in parameter t.
Taking

 Then equation (5) becomes

.0

,1

=

=

=

s

st
,
&

s
&&

m

2
xd
2

ds

+

m

kj

j

dx
ds

k

dx
ds

= 0 ...(6)

(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
-
�
�
�
�
�
�
�
�
-
-
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
-
-
-

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
Curvature of Curve, Geodesic

which may also written as

k

dx
ds

m

dx
ds

= 0

,
k

175

Then the intrinsic derivative (or derived vector) of the unit tangent to a geodesic in the direction of the
nV  is a line whose first curvature relative to
curve is everywhere zero. In otherwords, a geodesic of
nV  is identically zero.

THEOREM 9.2 To prove that one and only one geodesic passes through two specified points lying in

a neighbourhood of a point O of a

.nV

OR

To  prove  that  one  and  only  one  geodesic  passes  through  a  specified  point  O  of
direction.
Proof: The differential equations of a geodesic curve in a

nV  are

nV   in  a  prescribed

m

2
xd
2

ds

+

m

kj

j

dx
ds

k

dx
ds

= 0

These equations are  n  differential equations of the second order. Their complete integral involves 2n
arbitrary constants. These may be determined by the n coordinates of a point P on the curve and the n
components of the unit vector in the direction of the curve at  P. Thus, in general, one and only one
geodesic passes through a given point in a given direction.

9.5 GEODESIC  COORDINATES

A cartesian coordinate system is one relative to which the coefficients of the fundamental form are
constants.  Coordinates  of  this  nature  do  not  exists  for  an  arbitrary  Riemannian  Vn..  It  is,  however,
possible to choose a coordinate system relative to which the quantities gij are locally constant in the
neighbourhood of an arbitrary point P0 of Vn. Such a cartesian coordinate system is known as geodesic
coordinate system with the pole at P0.

The quantities

ijg  are said to be locally constants in the neighbourhood of a point

0P  if

and

g

x

g

x

ij
k

ij
k

= 0  at

0P

� 0   elsewhere

This shows that  [
Since the covariant derivative of Aij with respect to xk is written as

= 0 at

0P .

,
kij

,0

=

]

j

i

k

kijA , =

A
ij
k

x

h

kj

A
ih

h

ki

A

jh

, see pg. 71

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
-
�
�
176

Tensors and Their Applications

The covariant derivative of

ijA  at

0P  with respect to

kx  reduces to the corresponding ordinary

derivatives. Hence

kijA , =

A
ij
k

x

  at

0P

THEOREM 9.3 The necessary and sufficient condition that a system of coordintes be geodesic with
pole at P0 are that their second covariant derivatives with respect to the metric of the space all vanish
at P0.
Proof: We know that (equation 8, Pg. 65)

� 2

s

x

j

x

j

x

s

i

x

x

j

x

=

=

s

x

k

x

s

x

k

x

k

i

j

k

ji

p

x

i

x

p

x

i

x

q

j

x

x

q

j

x

x

s

qp

s

qp

or

Interchanging the coordinate system

ix  and

ix  in equation (1), we get

s

i

x

x

s

qp

j

x

q

j

x

x

p

x

i

x

=

=

=

s

k

x

x

j

x

j

x

k

ji

s

x

i

x
)

(
x

s
i
,

x

s
k
,

p

x

i

x

q

j

x

x

s

qp

k

i

j

since

s

k

x

x

=

x

s
k
,

at

P
0

s

k

x

x

k

i

j
k

Thus,

=

s
,),(
ix

j

  since

=

0

at

P
0

i

j

s
ijx, =

p

x

i

x

q

j

x

x

s

qp

Necessary  Condition
Let

sx  be a geodesic coordinate system with the pole at

0P  so that

Hence from (2), we have

s

qp

= 0  at

0P

s

ijx, = 0  at

0P

Sufficient  Condition

Conversely suppose that

=s
ijx
,

0

 at

0P .

...(1)

...(2)

�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
Curvature of Curve, Geodesic

177

Then equation (2) becomes

s

qp

p

x

i

x

q

j

x

x

= 0

s

qp

=

0

at

0P  as
,

p

x

i

x

0�

 and

q

j

x

x

0

at

P
0

So,

sx is a geodesic coordinate system with the pole at P0.

9.6 RIEMANNIAN  COORDINATES

A particular type of geodesic coordinates introduced by Riemann and known as Riemannian coordinates.
Let C be any geodesic through a given point  0P , s the length of the curve measured from  0P  and
quantities defined by

 the

i

i

=

i

dx
ds

o

the  subscript  zero  indicating  as  usual  that  the  function  is  to  be  evaluated  at
represents  that  only  one  geodesic  will  pass  through  P0  in  the  direction  of  x
coordinates of a point P on the geodesic C such that

...(1)

0P .  The  quantities  x
i  in  Vn.  Let  yi  be  the

i

where  s  is  the  arc  length  of  the  curve  from
coordinates.

yi = sx

i

...(2)
0P  to  P  .  The  coordinates  yi  are  called  Riemannian

The differential equation of geodesic C in terms of coordinates

iy  relative to

nV  is given by

i

2
yd
2

ds

+

i

kj

i

dy
ds

j

dy
ds

= 0

where

i

kj

 is a christoffel symbol relative to the coordinates

.iy

The differential equation (3) will be satisfied by (2), we have,

+0

i

kj

i

kj

i

j

= 0  since

i

dy
ds

x=

i

i

j

= 0

or

using equation (2), equation (4) becomes

i

kj

i

y
s

j

y
s

= 0  as

x=

i

i

y
s

or

i

kj

i yy

j

= 0

The equation (5) hold throughout the Riemannian

nV .

...(3)

...(4)

...(5)

�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:222)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
x
x
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
x
x
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
x
x
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
178

Tensors and Their Applications

Since

i

y

j

,0

y

,0

  from (5) we get

i

kj

= 0  at

0P

Hence the Riemannian coordinates are geodesic coordinate with the pole at

0P .

THEOREM 9.4 The necessary and sufficient condition that the coordinates

iy  be Riemannian coordinates

is that

i

kj

j

i yy

0=

 hold throughout the Riemannian

nV .

Proof: If

iy  are Riemannian coordinates then the condition

i

kj

j

i yy

 = 0 (from equation 5) throughout

the Riemannian

nV .

Conversely if

i

kj

j

i yy

0=

  hold  then

2
yd
2
ds

+

i

kj

i

dy
ds

j

dy
ds

  =  0  are  saitsfied  by

i

y

x=
s

.i

Hence yi are Riemannian coordinates.

9.7 GEODESIC  FORM  OF  A  LINE  ELEMENT
Let  f
hypersurface
1 =x
lines of parameter
1 =x

 be a scalar invariant whose gradiant is not zero. Let the hypersurface f

 = 0 be taken as coordinates
  and  the  geodesics  which  cut  this  hypersurface  orthogonally  as  the  coordinate
0
,1x  this parameter measuring the length of arc along a geodesic from the hypersurface

.

0
Since

1dx  is the length of the vector

i

 is given by

i.e.,

i

uug

ij

j

1
dx

11

1
dx

2u =
)21dx
(
g
=
11g = 1

Now, if vi is the tangent vector to the hypersurface x1 = 0 then we have

ndx
since the vectors ui and vi are orthogonal vectors.
Then,

iv =

,...,

,0(

dx

dx

,

3

2

)

i
ij vug

1
vug
j

1

j

j

= 0

= 0,

u i

[

=

,0

i

=

,3,2

...,

n

]

j

j vg1

= 0,  as

1 �u
jg1 = 0,  for  j = 2, 3, �, n.

.0

Again the coordinate curves of parameter x1 are geodesics. Then s = x1.
If ti is unit tangent vector to a geodesic at any point then

t1 = 1 and  t i  = 0, for i = 2, 3, �,  n.

...(1)

...(2)

�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
m
(cid:222)
(cid:222)
(cid:222)
(cid:222)
Curvature of Curve, Geodesic

179

Now,

and

it =

i
xd
ds

=

i

xd
1xd

1

1

i

dx

dx
2
xd
2

ds

= 1  and

dxi
1
dx

=

0

 for

1�i

=

i

2
xd
12

dx

=

0

  for i = 1, 2,..., n

Also, the differential equation of geodesic is

i

2
xd
2

ds

+

i

kj

j

dx
ds

k

dx
ds

= 0

using above results, we have

i
11 (cid:254)

1

dx
ds

1
dx
ds

i

11
[
]j
,11
]j,11
[

g ij

= 0

= 0

= 0

= 0  as

ijg

0�

1
2

2

g

x

j

1
i

g

x

11
j

= 0,  since

g

11

(cid:222)=
i

g
11
jx

=

0

So,

g j
1
1
x

= 0  for

1�j

...(3)

from equations (1), (2), and (3), we have

11g = 1,  g1j = 0; ( j = 2, 3, �, n),

g j
1
1
x

 = 0, ( j = 2, 3, �, n)

The line element is given by

2ds =

g

ij

dx

j

i

dx

2ds =

g

dx

11

1

+1

dx

g

dx

k

j

dx

jk

2ds =

(

dx

21
)

+

g

dx

jk

k

j

dx

;

  ( j  = 2, 3, �, n, k = 2, 3, �, n)

...(4)

The line element (4) is called geodesic form of the line element.

Note 1: We note that the coordinate curves with parameter  x1  are  orthogonal  to  the  coordinate  curve

xi = ci (i = 1, 2, ..., n) at all points and hence to the hypersurfaces x1 = c at each point.

Note 2: The existence of geodesic form of the line element proves that the hypersurfaces f  = x1 = constant form a

system of parallels i.e., the hypersurfaces f  = x1 = constant are geodesically parallel hypersurfaces.

(cid:222)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:222)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:222)
(cid:222)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
�
�
�
�
180

Tensors and Their Applications

THEOREM 9.5 The necessary and sufficient condition that the hypersurfaces  =f
system of parallel is that ((cid:209)
Proof:  Necessary  Condition
Suppose that hypersurface f

 = constant form a system of parallels then prove that ((cid:209)

)2 = 1.

2) = 1.

  constant  form  a

Let us take the hypersurface  f

  =  0 as the coordinate hypersurface  x1  =   0.  Let  the  geodesics
cutting this hypersurface orthogonally, be taken as coordinate lines of parameter x1. Then the parameters
x1 measures are length along these geodesics from the hypersurface x1 = 0. This implies the existence
of geodesic form of the line element namely

2ds =

(

dx

+21)

g

ij

dx

j

i

dx

...(1)

where i, j = 2, 3,..., n.
From (1), we have

from these values, it follows that

11g = 1,

g i
1

=

0

i
for

.1

11g = 1,

g i
1

=

,0

i
for

1

Now,

so

((cid:209)

((cid:209)
((cid:209)

=f

ij

g

i

x

j

dx

1
x
x

j

=

ij

g

11
i
j

)2 =

=

)2 =
)2 = 1

ij

g

1
x
x
11 =g

1

i

Sufficient  Condition
Suppose that ((cid:209)

)2 = 1 then prove that the hypersurface  f

 = constant from a system of parallels.

Let  us  taken  f

  =  x1  and  orthogonal  trajectories  of  the  hypersurfaces  f

  =  x1  constant  as  the

coordinate lines of parameter x1. Then the hypersurfaces

x1 = constant

xi = constant (i �

 1) are orthogonal to each other. The condition for this g1i  = 0 for

.1�i

Now, given that

(

2)

 = 1

ij

g

ij

g

i

x

1
x

i

x

= 1

= 1

j

x

1
x

j

x

ijg

1
i

1
j

= 1

11g = 1

f
f
�
�
f
f
�
�
f
�
(cid:209)
(cid:215)
f
(cid:209)
d
d
�
�
�
�
f
f
f
f
(cid:209)
(cid:222)
�
f
�
�
f
�
(cid:222)
�
�
�
�
(cid:222)
d
d
(cid:222)
Curvature of Curve, Geodesic

181

Thus

Consequently

11g = 1  and

g i
1

=

0

i
for

.1

11g = 1,

g i
1

=

  ,0

i
for

.1

Therefore, the line element

2ds =

i

dx

dx

j

g

ij

is given by

2ds =

dx

21
)

(

+

g

ik

i

dx

dx

k

;

 (i, k = 2, 3, �, n)

which is geodesic form of the  line element. It means that the hypersurfaces f

 = x1 =  constant

form a system of parallels.

9.8 GEODESICS  IN  EUCLIDEAN  SPACE

Consider an Euclidean space
equation of geodesics in Euclidean space is given by

nS  for n-dimensions. Let y i be the Euclidean coordinates. The differential

i

2
yd
2

ds

+

i

kj

j

dy
ds

k

dy
ds

= 0

...(1)

In case of Euclidean coordinates the fundamental tensor

ijg  is denoted by

ija  and

ijg =

a

ij

d=

=

i
j

=

 ,1
 ,0

 if
i
 if
i

j
j

g

x

ij
k

=

a

x

ij
k

0=

This implies that

k

i

j

=

,[,0

kij

=
0]

 relative to

.nS

Then equation (1) becomes

i

2
yd
2

ds

= 0

Integrating it, we get

dyi
ds

= ai,  where ai is constant of integration.

Again Integrating, we get

iy = ais + bi,  where bi is constant of integration

...(2)

The equation (2) is of the form

=

+
.cmx

y

Hence equation (2) represents a straight line. Since equation (2) is a solution of equation (1) and
nS  are given by equation (2). Hence geodesic  curves in Euclidean

therefore the geodesic relative to

space

nS  are straight lines.

�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
182

Tensors and Their Applications

THEOREM 9.6 Prove that the distance l between two points  P (y i) and Q (y' i)  in Sn is given by

l =

n

(
y

i

)

2

i

y

=

1

i

Proof: We know that geodesics in
taken as

nS  are straight line. Then equation of straight line in

nS   may  be

...(1)

...(2)

Let

( iyP

)

 and Q ( y' i) lie on equation (1). Then

yi = ais + bi

yi = ais + bi,
sa i
i
s
(
y

=

)

=�
i
y

i
sa

+�

i

b

y -
i
Then equation (2) becomes
y -
i
n

i

y

=
= (cid:229)

la i
n

=

1

i

i

(

y

i

y

2)

2

l

ia

(

)

2

=
1

i

But

ia  is the unit tangent vector to the geodesics. Then

n

=

1

i

ia

(

2)

= 1

So,

2l = (cid:229)

n

=

1

i

i

(

y

i

y

2)

l = (cid:229)

n

=
1

i

i

(

y

i

y

2)

EXAMPLE  1

Prove that Pythagoras theorem holds in

nS .

Solution

Consider a triangle  ABC right angled at A i.e.,

� BAC

o90=

.

i
C (y )3

i
A  ( y )1

i
B  ( y )2

Fig. 9.2

(cid:229)
-
�
�
-
�
�
(cid:229)
-
�
(cid:229)
-
�
-
�
Curvature of Curve, Geodesic

Then the lines AB and AC are orthogonal to each other. So,

AB (cid:215)

AC

= 0

or

or

a

ij

(

y

i
2

i
y
1

()

y

i
3

i
y
1

)

= 0

n

=

1

i

(

y

i
2

i
y
1

()

y

i
3

i
y
1

)

= 0

By distance formula, we have

(AB = (cid:229)
2)

(AC = (cid:229)
2)

(BC = (cid:229)
2)

n

=

1

i

n

=

1

i

n

=

1

i

(

y

i
2

2

i
y
1

)

(

y

i
3

i
y
1

2

)

(

i
y
3

y

i
2

2

)

Now, equation (4) can be written as

183

...(1)

...(2)

...(3)

...(4)

(BC = (cid:229)
2)

= (cid:229)

= (cid:229)

= (cid:229)

n

=

1

i

n

=

1

i

n

=

1

i

n

=

1

i

[(

i
y
3

+

i
y
1

)

(

i
y
1

y

i
2

2
)]

[(

i
y
3

i
y
1

2

)

+

(

i
y
1

2

y

i
2

)

+

(2

i
y
3

i
y
1

()

i
y
1

y

i
2

])

(

y

i
3

i
y
1

2

)

+

(

y

i
3

i
y
1

2

)

+

n

=
1

i

n

=
1

i

(

i
y
1

2

y

i
2

)

�+

,02

[from (1)]

(

i
y
1

i
y
2

2

)

2)
(BC =

AC +
2
)

(

(

AB
)

2

Hence Pythagoras theorem holds in

nS .

EXAMPLE 2

Prove that if q

 is any solution of the differential equation ((cid:209)

)2  = f (q ) then the hypersurfaces q

 =

constant constitute a system of parallels.

Solution

Given that

((cid:209)

)2 = f (q )

...(1)

-
-
-
-
(cid:229)
-
-
-
-
-
-
-
-
-
(cid:229)
-
-
(cid:229)
-
-
q
q
184

Tensors and Their Applications

Then prove that the hypersurfaces q
Suppose

 = constant form a system of parallel.

or

Now,

f = (cid:242)

d

)(f

,

  Then, df =

d

)(q

f

d
d

=

=

1

f

)(

 =

ix�

 =

ix�

1
q )(

f

(

2)

=

2

1

f

)(

=

1
)(

(

2)

 =

1
)(

f

f

(

);

 from (1)

f
)2 = 1
This proves that the hypersurfaces  f

((cid:209)

hypersurfaces q

 = constant.

  =  constant  form  a  system  of  parallels  and  therefore  the

EXAMPLE 3

Show  that  it  is  always  possible  to  choose  a  geodesic  coordinates  system  for  any

nV   with  an

arbitrary pole P0.
Solution
Let
value of
equation.

0P  be an arbitrary pole in a
ix  and
0P  are denoted by

nV .  Let us consider general coordinate system
ix  Now consider a new coordinate system
.0

ix . suppose the
jx   defined  by  the

jx =

a

j
m

(

x

m

x

m
0

)

+

1
2

a

j
h

h

ml

i

(

x

x

l
0

m

()

x

x

m
0

)

...(1)

The coefficients

j

ma  being constants and as such that their determinant do not vanish.

Now we shall prove that this new system of coordinated

jx  defined by equation (1) is a geodesic

coordinate system with pole at

Differentiating equation (1) with respect to

0P  i.e., second covariant derivative of
,mx
h

 we get

j

x

=

a

j
m

+

a

j
h

i

(2

x

x

l
0

)

1
2

ml

m

x

j

m

x

x

0

Now, the Jacobian determinant

=

a j
m

at P
0

jx  vanishes at P..

...(2)

...(3)

q
q
q
q
f
q
f
(cid:209)
f
�
q
�
q
�
f
�
q
(cid:209)
f
(cid:209)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
q
(cid:209)
q
q
(cid:209)
q
q
q
f
-
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
Curvature of Curve, Geodesic

185

j

x
m
x

0

=

j
ma

0�

and therefore the transformation given by equation (1) is permissible in the neighbourhood of  0P .
Differentiating equation (2) with respect to

 we get

,jx

2

j

j

x
x

x

m

a j
h

=

0

h

ml

0(cid:254)

But we know that

...(4)

(
j
lmx
,

)

0

=

j

2

x

l
xx

m

h

ml

0

0

j

h

x

x

0

h

ml

0

h

lm

a

j
h

,

  (from (3) and (4))

=

a

j
h

(
, j
lmx

)

0

= 0

Hence equation (1) is a geodesic coordinate system with pole at

0P .

EXAMPLE  4

If the coordinates xi of points on a geodesic are functions of arc lengths  s  and  f

 is any scalar

function of the  x's show that

p
d f
p
ds

=

,

ij

...

l

i
xd
ds

j

xd
ds

l
dx
ds

Solution

Since the coordinates

ix  lie on a geodesic. Then

i

2
xd
2

ds

+

i

kj

j

dx
ds

k

dx
ds

= 0

Here the number of suffices i j...l is p.
We shall prove the theorem by mathematical induction method.

Since x� s are functions of s and  f

 is a scalar function of x� s , we have

df
ds

2
d f
2
ds

=

=

=

i

x

i
,
j

x

i
,
j

x

i

dx
ds

  or

df
ds

 =

,

i

i

dx
ds

j

dx
ds

i

dx
ds

f+

,

i

j

dx
ds

i

2
xd
2

ds

i

kj

j

xd
ds

k

xd
ds

,

i
,

 from (2)

...(1)

...(2)

...(3)

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:215)
(cid:215)
(cid:215)
f
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
f
�
f
�
f
�
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
f
-
�
f
�
186

Tensors and Their Applications

i

dx
ds

j

dx
ds

i

dx
ds

j

dx
ds

,
i
i
x

,
i
j

x

,

m

,

m

m

j

k

j

dx
ds

k

dx
ds

m

ji

i

dx
ds

j

dx
ds

 (adjusting the dummy index.)

,
i
j

x

m

i

j

,

m

i

dx
ds

j

dx
ds

=

=

=

Equations (3) and (4) imply that the equation (1) holds for p = 1 and p = 2.

Suppose that the equation (1) holds for p indices

, 2
rr
1

,...,

pr

 so that

p
d f
p
ds

=

,,
r
1

r
2

,

...,

r
p

r
1

dx
ds

r
1

dx
ds

r
p

dx
ds

Differentiating the equation (5) with respect to s, we get

+ f
1
+
p

1

p

d

ds

=

...

r

p

+
1

r
1

dx
ds

r

p

dx
ds

,

rr
21
r
p

x

(cid:215)+

,

rr
21

...

r
p

r
1

dx
ds

f+

,

rr
21

...

r
p

r
1

2
xd
2
ds

r
2

dx
ds

r

+
1

p

dx

ds

p

r

2
xd
2
ds

r

p

dx
ds

...(4)

...(5)

...(6)

(cid:215)+

substituting value of

r
1

2
xd
2
ds

 etc. from (2) in (6) and adjusting dummy indices, we have

+ f
1
+

p

1

p

d

ds

=

...

r

p

,

rr
21
r

p

+
1

x

,

rm
2

...

r
p

m

rr
1
p

+
1

mrr
,
...
21

m

rr
pp

+

1

r
1

dx
ds

r
2

dx
ds

pr

+
1

dx

ds

=

,
rr
21 ..

rr
pp

+
1

r
1

xd
ds

r
2

xd
ds

+

1

r
p

xd
ds

This shows that the equation (1) holds for next values of  p. But equation (1) holds for p = 1, 2,

... Hence equation (1) holds for all values of p.

EXERCISES

1. Prove that at the pole of a geodesic coordinate system, the components of first covariant derivatives

are ordinary derivatives.

2. If

ix   are  geodesic  coordinates  in  the  neighbourhood  of  a  point  if  they  are  subjected  to  the

transformation

ix =

i

x

1+
6

c

i
jkl

j

x

k

x

l

x

(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
f
-
�
f
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
f
-
�
f
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
f
-
�
f
�
(cid:215)
(cid:215)
(cid:215)
f
(cid:215)
(cid:215)
(cid:215)
�
f
�
(cid:215)
(cid:215)
(cid:215)
(cid:215)
(cid:215)
(cid:215)
(cid:215)
(cid:215)
f
(cid:215)
(cid:215)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
f
(cid:215)
(cid:215)
(cid:215)
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
(cid:239)
(cid:238)
(cid:239)
(cid:237)
(cid:236)
f
-
�
f
�
(cid:215)
(cid:215)
(cid:215)
(cid:215)
(cid:215)
(cid:215)
f
Curvature of Curve, Geodesic

187

where

sC�  are constants then show that

ix  are geodesic coordinates in the neighbourhood of O.

3. Show that the principal normal vector vanishes identically when the given curve is geodesic.

4. Show that the coordinate system

ix =

ix  defined by
i
kj

1
2

+

i

x

kj
xx

is geodesic coordinate system with the pole at the origin.

5. Obtain the equations of geodesics for the metric
2
dy

2ds =

e kt
2

2
dx

+

(

+

2
dz

)

+

dt

2

6. Obtain the differential equations of geodesics for the metric

2ds =

dxxf
)(

2

+

2

+

dy

2
dz

+

1
xf
)(

2

dt

:Ans

2
xd
2
ds

1
2

d
dx

(log

f

)

2
+(cid:247)

dx
ds

1
f

2

2

d
dx

(log

f

)

dt
ds

=

;0

2
yd
2
ds

=

;0

2
zd
2
ds

=

;0

2
td
2
ds

f

)

d

(log
dx

dx
ds

dt
ds

=

0

7. Find the differential equations for the geodesics in a cylindrical and spherical coordinates.
8. Find the rate of divergence of a given curve C from the geodesic which touches it at a given point.

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
�
�
�
�
�
�
-
?
(cid:246)
(cid:231)
?
(cid:230)
-
CHAPTER � 10

PARALLELISM  OF  VECTORS

10.1 PARALLELISM  OF  A  VECTOR  OF  CONSTANT  MAGNITUDE  (LEVI-CIVITA�S

CONCEPT)

Consider a vector field whose direction at any point is that of the Unit Vector ti. In ordinary space, the
field is said to be parallel if the derivative of ti vanishes for all directions ui (say) and at every point of
the field i.e.,

i

t

j

x

j

u

= 0

Similarly in a Riemannian Vn the field is said to be parallel if the derived vector of ti vanishes at

each point for every direction ui at each point of Vn. i.e.,

i
jt, =

0=ju

It can be shown that it is not possible for an arbitrary V n. Consequently we define parallelism of

vectors with respect to a given curve C in a V n.

A vector ui of constant magnitude is parallel with respect to Vn along the curve  C if its derived

vector in the direction of the curve is zero at all points of C i.e.,
j

dx
ds
where s is arc-length of curve C.

u

i
, = 0
j

The equation (1) can be written in expansion form as

i

j

u

x

xd
ds

i

du
ds

+

m

u

i
jm

j

+

m

u

+

m

u

i
jm

i

jm

j

xd
ds

j

xd
ds

j

dx
ds

= 0

= 0

= 0

i

j

u
x

� (1)

� (2)

�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
Parallelism of Vectors

189

This concept of parallelism is due to Levi-Civita. The vector  ui is satisfying the equation (1) is

said to a parallel displacement along the curve

Now, multiplying equation (1) by gil, we get

g

il

u

i
,

j

j

dx
ds

(

ug
il

i
,

j

)

j

xd
ds

(

ug
il

i

,)

j

u

l ,

j

u

ji,

u

x

i
j

u

m

m

ji

du
i
ds

u

m

m

ji

j

dx
ds

j

dx
ds

j

dx
ds

j

dx
ds

j

dx
ds

= 0

= 0

= 0

= 0

= 0

= 0

= 0

or

The equation (2) and (3) can be also written as

and

dui =

m

u

i
jm

j

xd

dui =

u

m

m
ji

j

xd

� (3)

� (4)

� (5)

The  equation  (4)  and  (5)  give  the  increment  in  the  components  ui and  ui  respectively  due  to

displacement dxj along C.

THEOREM 10.1 If two vectors of constant magnitudes undergo parallel displacements along a given
curve then they are inclined at a constant angle.
Proof: Let the vectors ui and vi be of constant magnitudes and undergo parallel displacement along a
curve C, we have (from equation (1), Pg. 188.)

u

i
,

j

i
v
,

j

j

j

xd
ds
xd
sd

=

0

=

0

= 0

� (1)

at each point of C.

Multiplying (1) by gil, we get
dx
ds

ug
il

i
,

)

(

j

j

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:239)
(cid:239)
(cid:254)
(cid:239)
(cid:239)
(cid:253)
(cid:252)
190

Tensors and Their Applications

or

Similarly,

u

l ,

j

u

i,

j

j

dx
ds

j

dx
ds

= 0

= 0

v
i ,

j

j

dx
ds

= 0

Let f

 be the angle between ui and vi then
ui.vi = uv cos q

Differentiating it with respect to arc length s, we get

d
ds

(uv cos q) =

)

i
(
vud
i
ds

=

i
vu

(

,)

i

j

j

dx
ds

uv

sin

d
ds

=

u

i
,

j

j

dx
ds

v
i

+

i
vu

,
ji

j

dx
ds

Using equation (1) and (3), then equation (4) becomes

uv

sin

sin

d
ds
dq
ds

= 0

= 0,

as

u

,0

v

0

Either sinq = 0

or

dq
ds

 = 0

Either q = 0

or

  = constant.

 is constant. Since 0 is also a constant.

THEOREM 10.2 A geodesic is an auto-parallel curve.

Proof: The differential equation of the geodesic is given by (See Pg. 174, eqn. 6)

� (2)

� (3)

� (4)

m

2
xd
2

ds

+

m

kj

j

dx
ds

k

dx
ds

  = 0

d
ds

m

xd
ds

+(cid:247)

m
kj

j

xd
ds

k

xd
ds

 = 0

m

dx
ds

j

xd
ds

+

m
kj

j

xd
ds

k

xd
ds

 = 0

j

x

q
q
-
q
q
-
(cid:222)
q
�
�
(cid:222)
(cid:222)
q
(cid:222)
q

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
Parallelism of Vectors

191

m

dx
ds

+(cid:247)

m
kj

k

dx
ds

j

dx
ds

= 0

j

x

m

xd
ds

j

xd
ds

,

j

= 0

or

t

m
j
,

j

dx
ds

=

0

This shows that the unit tangent vector

dxm
ds

 suffer a parallel displacement along a geodesic curve.This

confirms that geodesic is an auto-parallel curve.

Proved.

10.2 PARALLELISM  OF  A  VECTOR  OF  VARIABLE  MAGNITUDE

Two  vectors  at  a  point  are  said  to  be  parallel  or  to  have  the  same  direction  if  their  corresponding
components are proportional. Consequently the vector vi will be parallel to ui at each point of curve C
provided

where  f

 is a function of arc length s.

vi = f ui

If ui is parallel with respect to Riemannian Vn along the curve C. Then,

u

i
j,

j

dx
ds

= 0

� (1)

� (2)

The equation (1) shows that v i is of variable constant and parallel with respect to Riemannian Vn

so that

v

i
j,

j

xd
ds

(f

=

i

u

,)

j

j

xd
ds

=

(

i

u

,

j

f+

u

i
,

j

)

j

xd
ds

=

,

j

j

xd
ds

i

u

f+

u

i
,

j

j

xd
ds

=

,

j

j

xd
ds

i

u

Since

u

i
,

j

j

xd
ds

=

0

i
v
j,

j

xd
ds

=

=

=

j

xd
ds

i

u

j

x�

iu

iv

df
sd

d
sd

from (1)

�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
f
f
f
f
�
f
f
192

Tensors and Their Applications

vi

=

)

d

(log f
ds

)

d

(log f
ds

i
v
j,

j

dx
ds

� (3)
Hence a vector vi of variable magnitude will be parallel with respect to Vn if equation (3) is satisfied.

= v i f (s) where

f (s) =

Conversely suppose that a vector vi of variable magnitude such that

to show that vi is parallel, with respect to Vn .

i
jv, =

j

dx
ds

=

i
)(sfv

Take

Then

ui = vi

y

(s)

� (5)

u

i
j,

j

xd
ds

u

i
j,

Select  y

 such that

sf
)(

j

dx
ds
d
ds

+

i

(

v

=

),

j

=

i
v
,

j

j

dx
ds

j

dx
ds

y+y

,

j

i
sfv

+y)(

=

vi

=

)(
sf

+

j

x
d
ds

=

.0

j

i

v

i

v

dx
ds
j

dx
ds

� (6)

Then equation (6) becomes

u

i
j,

j

dx
ds

= 0

This equation shows that the vector ui is of constant magnitude and suffers a parallel displacement

along curve C. The equation (5) shows that vi is parallel along C.

Hence necessary and sufficient condition that a vector vi of variable magnitude suffers a parallel

displacement along a curve  C is that
j

EXAMPLE 1

i
v
j,

dx
ds

=

(sfvi

).

Show that the vector vi of variable magnitude suffers a parallel displacement along a curve  C  if

and only if

i
vv

(

i
,
k

i
vv

l
,
k

)

Solution
From equation (4), we have

k

dx
ds

= 0,

i = 1, 2, ..., n.

v

i
j,

j

dx
ds

=

)(sfvi

� (1)

y
�
y
�
�
�
�
�
�
�
y
y
y
y
-
Parallelism of Vectors

Multiplying by v l, we get

l
i
vv
,

j

j

dx
ds

=

l
)(sfvv i

Interchange the indices l and i, we get

dx
ds
Subtract (1) and (2), we get

l
i
vv
,

j

j

= vi vl f (s)

193

� (2)

l
uv

(

i
,
k

i
vv

l
,
k

)

k

dx
ds

= 0 by interchanging dummy indices j and k.

10.3 SUBSPACES  OF  A  RIEMANNIAN  MANIFOLD

Let  V n  be  Riemannian  space  of  n  dimensions  referred  to  coordinates  xi  and  having  the  metric
ay  and having
ds2 = gij dxi dxj. Let Vm be Riemannian space of m dimensions referred to coordinates
 dyb,  where m > n. Let Greek letters a , b, g
the metric ds2
indices i, j, k ... take the values 1, 2, � n.

  take the values 1, 2,  ..., m and Latin

 = axb  dya

If the n independent variables xi are such that the coordinates (

ay ) of points in Vm are expressed
as a function of xi then Vn is immersed in Vm i.e. Vn is a subspace of Vm. Also Vm is called enveloping
space of Vn.

Since the length ds of the element of arc connecting the two points is the same with respect to Vn

or Vm. it follows that

gij dxi dxj =

a

dy

dy

gij dxi dxj =

a

gij =

a

i

dx

dx

j

i

y

x
y

i

x

y

x
y

x

j

j

As dxi and dxj are arbitrary.

This gives relation between gij and aa

.

� (1)

THEOREM 10.3 To show that the angle between any two vectors is the same whether it is calculated
with respect to Vm or Vn.
Proof:  Consider two vectors dxi and dxj defined at any point of Vn and suppose that the same vectors
in Vm are represented by

 is the angle between dxi and

d y  respectively. If  q

ady  and

 then

ixd

cos q =

i

dxg
ij

j

x

dxg
ij

j

i

dx

g

ij

i
xx

j

If f

 is the angle between the vectors dya  and d ya  then

cos f =

a

dy

y

a

dy

dy

a

y

y

� (1)

-
b
a
a
b
�
�
�
�
b
a
a
b
(cid:222)
�
�
�
�
b
a
a
b
b
a
d
d
d
b
a
a
b
b
a
a
b
b
a
a
b
d
d
d
194

Tensors and Their Applications

=

=

a

a

a

y

i

x

i

xd

y

j

x

j

xd

i

xd

y
x

i

y
x

j

j

xd

a

y
x

i

i
x

y
x

j

x

.j

a

y

x

j

y

i
x

y
x

i

y
x

j

i

dx

dx

j

dx

i

dx

j

a

y

i

x

y

j

x

i
xx

.j

cos =

dxg
ij

j

i

dx

dxg
ij

j

i

dx

g

ij

i
xx

j

 (from equation (1), art. 10.3)

Since

g

ij

=

a

from (1) & (2)

y

i

x

y

x

j

� (2)

 = f
Proved.
THEOREM  10.4 If Ua  and ui denote the components of the same vector in Riemannian Vm and Vn
respectively then to show that

cos q = cos f

 (cid:222) q

.

aU =

i

u

y

i

x

Proof: Let the given vector be unit vector at any point P of V n.  Let the component of the same vector
x's and  y's  be  ai  and  Aa  respectively. Let  C be curve passing through  s  in  the  direction  of  the  given
vector then

dx
ds
But the components of a vector of magnitude a are a times the corresponding components of the

aA =

� (1)

y
s

i
x

=

=

a

x

y

y

i

i

i

Unit vector in the same direction. Then

Multiplying (1) by a, we get

aU = aAa ,

ui = aai

Or

aaA =

aU =

i

y

x
y

i

x

(

aa

i

)

u

,i

using (2)

� (2)

Proved.

THEOREM 10.5 To show that there are m-n linearly vector fields normal to a surface V n immersed in
a Riemannian V m.
Proof:  Since Vn is immersed in V m,  the coordinates ya  of points in V m are expressible as functions of
coordinates xi in Vn.

d
�
�
d
�
�
�
�
�
�
�
�
�
�
b
a
a
b
b
a
a
b
b
a
a
b
d
d
�
�
�
�
�
�
�
�
�
�
�
�
b
a
a
b
b
a
a
b
b
a
a
b
f
d
d
�
�
�
�
b
a
a
b
�
�
a
�
�
�
�
�
�
a
a
a
�
�
a
�
�
a
Parallelism of Vectors

Now,

dya
ds
for the curve  xi = s, we have
dya
ds

=

=

y

i

x

y
ix

i

dx
ds

195

...(1)

...(2)

dya
ds

Since

y
ix
in  Vm is tangential to the coordinate curve of transmeter  xi in  Vn. Let the Unit vectors  Na  in  Vm  be
normal to each of the above vector fields of Vn then

 is a vector tangential to the curve in V m.  Then from equation (2) it follows that

a

y

i
x

(

Na

)

N

N

y
ix
y
ix

= 0

= 0

iyNa
,

 = 0

= 0,

i = 1, 2, �, n & a

 = 1, 2, �, m

The equation (3) are n equations in m unknowns

aN  (m > n). The coefficient matrix

...(3)

y
ix

 is

of order m � n and the rank of this matrix is n.

It means that there will be only m � n linearly independent solution of

aN . This shows that there

are m � n linearly independent normals to Vn to V m.

EXAMPLE 2

Show that

[ij, k] =

[

],

y
x

i

y
x

j

y
x

k

+

a

2

y
i
xx

j

x
x

k

where [a

, g ] and

[

kji
,

]

 are the Christoffel�s symbols of first kind relative to metrics

a

a

dy

dy

 and

dxg
ij

i

dx

.j

Solution

Since relation between

ba

 and

ijg  is given by

ijg =

a

y

i

x

y

x

j

Differentiating it with respect to xk, we get

g

x

ij
k

=

a

x

y

i

x

y

x

j

y

x

k

+

a

2

y

i
xx

k

+

a

y

x

j

y

i

x

2

j

x

y

k

x

� (1)

�
�
a
�
�
a
�
�
a
b
a
a
b
�
�
(cid:222)
�
�
a
b
a
b
(cid:222)
a
b
a
b
(cid:222)
�
�
a
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
a
�
�
�
�
�
�
�
�
�
�
�
g
a
b
b
a
a
b
g
b
a
b
b
a
b
a
�
�
�
�
b
a
a
b
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
b
a
a
b
g
a
a
b
g
b
a
g
a
b
196

Tensors and Their Applications

Similarly

and

g

jk
i
x

g

ki
j

x
But we know that

=

=

a

y

a

x

y
x

i

y

i

x

kij
,[

]

=

and,

[

],

=

y
x

j

y
x

k

+

a

2

y

j

x

i

x

y
x

k

+

a

y
x

j

2

k

x

y
x

i

y

x

j

y

x

k

+

a

2

y

k

x

j

x

y

i

x

+

a

y

k

x

2

y

i
xx

j

1
2

1
2

g

jk

i

x

g

y

+

+

ki
j

g

x

g

y

g

x

ij

k

g

y

� (2)

� (3)

� (4)

� (5)

Substituting the value of (1), (2), (3) in equation (4) and using (5) we get,

kij
,[

]

=

[

],

y
x

i

y
x

j

y
x

k

+

a

2

y
i
xx

j

y
x

k

.

10.4 PARALLELISM  IN  A  SUBSPACE

THEOREM  10.6 Let  T a
  and  t i   be  the  components  of  the  same  vector  t  relative  to  Vm  and  Vn
respectively. Let the vector t be defined along a curve C in V n.  If   p i and qa  are derived vectors of t
along C relative to Vn and Vm respectively. Then

y
ix
Proof: Since from equation (1), theorem, (10.4), Pg. 194 we have

= pi

q

Now,

aT =

i

t

y

i

x

dT a
ds

=

i

dt
ds

y

i

x

+

i

t

2

y

i
xx

j

j

dx
ds

ip =

t

i
j,

aq =

T
b,

j

dx
ds
dy
ds

=

=

aq =

T

x

+

T

dy
ds

+

T

T

x

dT
ds

+

T

dy
ds

dy
ds

dy
ds

� (1)

� (2)

� (3)

� (4)

�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
b
a
a
b
b
a
a
b
g
b
a
a
b
g
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
�
b
a
a
b
b
a
a
b
g
b
a
b
g
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
g
a
b
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
g
a
b
b
g
a
a
b
g
�
�
�
�
�
�
�
�
�
�
�
g
a
b
b
a
a
b
g
b
a
�
�
a
a
�
�
a
�
�
�
�
�
a
a
b
a
b
g
b
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
a
�
�
b
g
b
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
a
�
�
b
g
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
a
Parallelism of Vectors

197

Putting the value of

dT a
ds

 and

gT  from (2) and (1) in equation (4), we get

aq =

i

dt
ds

y

i

x

+

i

t

2

y

i
xx

j

j

dx
ds

+

i

t

y

i

dx

dy
ds

=

=

aq =

i

t

j

x

i

j

t
x

i

t

j

x

y

i

x

y
x

i

j

dx
ds

j

dx
ds

+

i

t

+

i

t

2

y

i
xx

j

j

dx
ds

+

i

t

y

i

x

dy
ds

2

y
i
xx

j

j

dx
ds

+

i

t

y
x

i

j

dx
ds

y

x
.

j

y

i

x

j

xd
ds

+

i

t

j

xd
ds

+

2

y

i
xx

j

+

y

x

j

y

i

x

� (5)

But we know that

kij
,[

]

=

[

kji
,

]

=

[

[

],

],

y

i

x

y

x

j

y

x

j

y

x

k

+

a

y

i

x

y

x

k

+

a

2

y

i
xx

j

2

y

i
xx

j

y

x

k

y

x

k

=

a

=

a

y
x

k

y

x

j

y

i

x

y

k

x

+

a

2

y

i
xx

j

y

x

k

y
x

j

+

y
x

i

2

y
i
xx

j

� (6)

Multiplying (5) by

a

a

y
kx

y

x

k

, we get, using (6),

q

=

i

t

j

x

j

dx
ds

y

i

x

a

Or

q

y
kx

j

dx
ds

j

dx
ds

=

=

i

j

t
x

i

t

j

x

=

g

ik

j

dx
ds

t

i
,
j

g

ik

+

i

t

g

ik

+

a

t

i
ja

g

ik

y

x

k

p
ji

+

i

t

j

dx
ds

kij
],[

g

pk

,

since

g

ij

=

a

y

i

x

y

x

j

(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
a
�
�
�
�
�
�
b
g
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
a
�
�
�
�
�
�
�
�
�
b
g
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
a
�
�
�
�
�
�
�
�
�
�
�
b
g
a
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
a
�
�
�
�
�
�
�
�
�
�
�
g
b
a
a
�
�
�
�
�
�
�
�
�
�
�
g
a
b
g
a
a
g
g
b
a
�
�
�
�
�
�
�
�
�
�
�
d
b
g
d
a
a
d
d
g
b
�
�
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
a
d
a
a
d
d
g
b
a
d
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
a
�
�
a
g
b
d
a
d
�
�
d
a
d
a
d
a
d
�
�
�
�
�
�
�
�
d
a
d
a
�
�
d
d
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
�
�
�
b
a
a
b
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
198

Tensors and Their Applications

=

i

(

tg
ik

),

j

j

dx
ds

j

dx
ds
j

dx
ds

=

=

t

t

jk ,

jk ,

= pk

q

q

y
kx
y
kx

or

Properties of Vm

�(7)

Proved.

(i)

If a curve C lies in a subspace Vn of Vm and a vector field in Vn is parallel along the curve C
with regard to Vm, then to show that it is also a parallel with regard to V n.
Proof: If a vector field t is a parallel along C with respect to V m.  then its derived vector qa
vanishes i.e.,
qa  = 0,
Hence equation (7) becomes pk = 0, k = 1, 2, ..., n. This shows that the vector field t is also
parallel along C with respect to V n.

 = 1, 2,... m.

(ii) To show that if a curve  C is a geodesic in a Vn it is a geodesic in any subspace  Vn of V m.

q

Proff : Science,

y
ix
Let t be unit tangent vector to the curve C them
and ti is unit tangent vector to C relative to Vn.

= pi

aT  is unit tangent vector to C relative to Vm

Now,

Also,

pi = 0
aq = 0
aq = 0,

curve C is a geodesic in Vn
curve C is a geodesic in Vm
i

pi = 0, "

This shows that if a curve  C in Vn is a  geodesic relative to Vm then the same curve is also a geodesic
relative to V n.

(iii) A  necessary  and  sufficient  condition  that  a  vector  of  constant  magnitude  be  parallel  with
respect to  Vn  along  C  in  that  subspace,  is  that  its  derived  vector  relative  to  Vm  for  the
direction of the curve be normal to Vn.

Proof: Let t be a vector of constant magnitude. This vector t is parallel along C relative to Vn iff pi =
0, "

i

iff

q

y
ix

= 0

 = 0 implies that

aq  is normal to Vn.

y
,

i

 lying in Vm is tangential to a coordinate curve of parameter xi in V n.

But

q

For

y
ix

=

y
x

i

�
�
a
a
�
�
a
a
a
�
�
a
a
(cid:222)
(cid:222)
a
"
(cid:222)
�
�
a
a
�
�
a
a
a
a
�
�
Parallelism of Vectors

199

These statements prove that a necessary and sufficient condition that a vector of constant
magnitude be parallel along C relative to Vn is that its derived vector i.e., qa  along C relative
to Vm be normal to V n.

(iv) A  necessary  and  sufficient  condition  that  a  curve  be  a  geodesic  in  Vn  is  that  its  principal
subnormal relative to Vm the enveloping space be normal to Vn at all points of the curve.
Proof: In particular let the vector t be unit tangent vector to the curve C. In this case
called principal normal the curve  C. Also  pi =  0, "
relative to V n.
Using result (iii), we get at once the result (iv).

aq  is
i implies that the curve  C  is a geodesic

(v) To prove that the tendency of a vector is the same whether it is calculated with respect to

V m.
Proof: Since, we have

q

y
ix

y

i

x

i

dx
ds

dy
ds

dy
ds

q

dy
ds

q

T

,

= pi

=

p

i

=

p

i

i
dx
ds

i
dx
ds

=

t

i,

j

j

dx
ds

i

dx
ds

i.e., tendency of
i.e., tendency of t along C relative to Vm = tendency of t along C relative to V n.

aT  along C = tendency of ti along C.

10.5 THE  FUNDAMENTAL  THEOREM  OF  RIEMANNIAN  GEOMETRY

  STATEMENT

With a given Riemannian metric (or fundamental tensor) of a Riemannian manifold there is associated
a  symmetric  affine  connection  with  the  property  that  parallel  displacement  (or  transport)  preserves
scalar product.
Proof:  Let C be a curve in Vn. Let pi and qi be two unit vectors defined along C. Suppose that the unit
vectors pi and qi suffer parallel displacement along the curve C in V n,  then we have

p

i
j,

q

i
j,

j

dx
ds

j

dx
ds

= 0

= 0

and

� (1)

� (2)

Let gij be the given fundamental tensor of a Riemannian manifold. Hence, the scalar product of

vectors pi and qi is gij pi qj.

Now,

(

i

qpg

ij

j

),
k

k

dx
ds

= 0

�
�
a
a
�
�
a
a
a
a
a
b
b
a
200

Tensors and Their Applications

g

ij

p

i
,
k

k

dx
ds

j

+

q

i
qpg
ij

j
,
k

k

dx
ds

+(cid:247)

g

,
kij

k

dx
ds

i
qP

j

 = 0

Using equation (1) and (2), the equation (3) becomes,

g

,
kij

k

dx
ds

j

i
qp

= 0

gij, k = 0

(Since pi and qi are unit vectors and

dxk
ds

)0�

g

x

ij
k

g

mj

g

x

ij
k

m

ki

g

im

m

kj

= 0

ik
[

,

j

]

[

jk

i
],

= 0

g

x

ij
k

=

[

ik

j
],

+

[

jk

i
],

Now, using equation (4), we have

since

kij
,[

]

=

So,

g

x

ki
j

g

+

jk
i
x
kji
,
[

].

g

jk
i
x

+

g

x

ki
j

g

x

ij
k

g

x

ij
k

=

[

kji
,

]

+

[

jki
,

]

+

[

ikj
],

+

kij
,[

([�]

ik

i
],

+

[

jk

,

i

])

=

,[2

kij

]

[ij,k] =

1
2

g

jk

i

x

+

g

x

ki
j

g

x

ij

k

But we know that

from (5), we have

k

ji

=

lk

g

ij
],[
l

k

ji

lk

g

1
2

=

+

g

jl

i
x

g

x

li
j

g

ij

l

x

� (3)

� (4)

� (5)

Proved.

EXAMPLE 3

If  t i and  T a

immersed in Vm. Show that

  are  contravariant  components  in  x's and  y's  respectively, of  n  vector  field  in  Vn

Solution

Since we know that

jT, =

ty
,
ij

i

+

ty
,
i

i
,

j

aT =

i

t

y

i

x

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
-
�
�
-
-
�
�
�
�
�
�
-
�
�
�
�
�
�
-
�
�
�
�
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
-
�
�
�
�
a
a
a
�
�
a
Parallelism of Vectors

201

Taking covariant differentiation of both sides, we get

aT =

i yt

.,
j

jT, =

i yt

(

, )
i

,

j

=

a +
i
yt
i
,
,
j

i
(
yt

,
i

)

,

j

jT, =

i
yt

a +
,
ij

i
yt
,
j

,
i

 remains constant along a geodesic.

EXAMPLE 4

Show that

g

ij

i

dx
ds

j

dx
ds

Solution

Let

i =

t

i

dx
ds

. Then

g

ij

i

dx
ds

j

dx
ds

=

i
ttg
ij

j

=

2t

Since we know that geodesics are autoparallel curves. Then
j

t

i
j,

dx
ds

= 0

or

Now,

j

i
t,

jt

= 0

� (1)

dt 2
ds

=

d
ds

(

i
ttg
ij

j

)

=

d
ds

(

tt
i

i

)

=

(

tt
i

i

)

,

j

j

dx
ds

=

(
tt
i

i

)

,

j

j

t

dt 2
ds

=

(

t

i

,

j

i

t

i

)

t

+

t
(

i

i
,

j

t

)

t

i

=

,0

from (1)

Integrating it we get

t2 = constant.

So,

g

ij

i

dx
ds

j

dx
ds

EXAMPLE 5

 remains constant along a geodesic.

If ti are the contravariant components of the unit tangent vector to a congruence of geodesics.

Show that

and also show that

|

t

ji
,

t
+

i

(

t

ji
,

+

t

ij
,

)

= 0

t

ij
,

=
.0|

a
a
a
a
a
a
202

Solution

Tensors and Their Applications

Let t i denote unit tangent vector to a congruence of geodesic so that

j

i
t,

jt

= 0

� (1)

Since geodesics are auto-parallel curves. Then to prove that

(i)

(ii)

i

t

(

t

ji
,

+

t

ij
,

=

)

0

|

t

i

,

j

+

t

ij
,

=
0|

Since

i tt

i tt

i

i

=

2 =t

1

= 1

Taking covariant derivative of both sides, we get

or

itt

(

,)

i

j

= 0

t

i
,

t

j

i

+

i
tt

i

,

j

= 0

Since t is a free index. Then we have

i

t

i
,

t

j

+

t

t

i

,

j

i
t ,2

jt

t ,
i

t

j

i
tg ,
ik

t

j

t ,
jk

t

i

i

i

j

j

i

= 0

= 0

= 0

= 0

= 0

= 0

t

ij t
,

=i

0

i tt

ij
,

=

0

ik t
t ,
i tt

, = 0
ij

from (1)

or

Thus

Adding (2) and (3), we get

i

t

(

t

ji
,

+

t

ij
,

)

= 0

Also,  since ti �
Taking determinants of both sides we get

  0, "

i.

i

|

t

t
(

|

+

t
,
j
i
t +
,
j
i

|)

= 0

ij
,

t

ij
,

|

= 0

EXAMPLE 6

� (2)

� (3)

Solved.

When  the  coordinates  of  a  V 2  are  chosen  so  that  the  fundamental  forms  is
,  prove  that  the  tangent  to  either  family  of  coordinate  curves  suffers  a

+

2

1
dx

dx

(

dx

22
)

2

g

(

+&
)
'xd

12

parallel displacement along a curve of the other family.

(cid:222)
(cid:222)
(cid:222)
(cid:222)
(cid:222)
Parallelism of Vectors

Solution

The metric is given by

Comparing it with

We have,

2ds =

dxg
ij

j

i

dx

i,j = 1,2.

2ds =

dx

21
)

(

+

2

g
12

1
dx

dx

2

+

(

dx

22
)

g11 = 1,

g22 = 1.

203

(1)

In this case, we have coordinate curves of parameters x1 and x2 respectively. The coordinate x1

curve is defined by

xi = ci,
and the coordinate x2 curve is defined by
xi = di,

where ci and di are constants.

i,

except

i = 1.

i,

except

i = 2

(2)

(3)

Let pi and qi be the components of tangents vectors to the curves (2) and (3) respectively. Then

we have

and

So,

pi = dxi = 0,
qi = dxi = 0,

i,
except
i, except i = 2

i = 1

pi = (dxi, 0)

and qi = (0,dxi)

Let t be the unit tangent vector to the curve (2).
Hence

dx i
ds

i

t

=

=

i

p
p

)0,1(=

where

p =

idx

So, we have

p

i
j,

j

dx
ds

= 0

Hence the tangent to the family of coordinates curves (2) suffers a parallel displacement along a

curve of the family of curves (3).

EXERCISES

1. Explain Levi-Civita's concept of parallelism of vectors and prove that any vector which undergoes a

parallel displacement along a geodesic is inclined at a constant angle to the curve.

2. Show that the geodesics is a Riemannian space are given by
k

m

i

2
xd
2

ds

+

m

ki

dx
ds

dx
ds

= 0

Hence prove that geodesics are auto-parallel curves.

"
"
"
"
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
204

Tensors and Their Applications

3. Establish the equivalence of the following definitions of a geodesic.

 It is an auto�parallel curve.
It is a line whose first curvature relative to Vn identically zero.

(i)
(ii)
(iii)   It is the path extremum length between two points on it.

4. If u and v are orthogonal vector fields in a V n, prove that the projection on u

u in its own direction is equal to minus the tendency of v in the direction of u.

 of the desired vector of

5. If the derived vector of a vector u i is zero then to show that vector u i has a constant magnitude along

curve.

6. Prove that any vector which undergoes a parallel displacement along a geodesic is inclined at a

constant angle to the curve.

7. Prove that there are m � n linearly independent vector fields normal to a surface  Vn immersed in a
Riemannian Vm and they may be chosen in a multiply infinite number of ways. But there is only one
vector field normal to the hyperface.

8. Show that the principal normal vector vanishes identically when the given curve in geodesic.
9. Show that if a curve is a geodesic of a space it is a geodesic of any subspace in which it lies.
10. Define parallelism in a subspace of Riemannian manifold. If a curve C lies in a subspace Vn of Vm and
a vector field in Vn is parallel along C with respect to V m. Then show that it is also parallel with
respect to V n.

CHAPTER � 11

RICCI'S  COEFFICIENTS  OF  ROTATION  AND

CONGRUENCE

11.1 RICCI'S  COEFFICIENTS  OF  ROTATION

Let

i
he |   (h=1,  2,  �n) be the unit tangents to the n congruences
e
ke |   has  components

Riemannian  V n, .   The  desired  vector  of  el|i  in  the  direction  of
i
he |  is a scalar invariant, denoted by g

projection of this vector on

lhk, so that

he ,  of  an  orthogonal  ennuple  in  a

e

,|
il

j
e
kj

|

and  the

�(1)

lhk =

e

i
ee
,|
hjil
|

j
k

|

The invariants g
Since i being a dummy index, has freedom of movement. Then equation (1) may be written as

lhk are Ricci's Coefficients of Rotation.

lhk =
The indices l, h and k are not tensor indices. But these indices in g

j
ee
e
,|
kihjil
|
|

lhk are arranged in proper way,
the first index l indicates the congruence whose unit tangent is considered, the second h indicates the
direction of projection and the third k is used for differentiation.

�(2)

THEOREM 11.1 To  prove  thast  the  Riccie's  coefficients  of  rotation  are  skew-symmetric  in  the  first
two indices i.e.,

lhk = �g

lhk

If

i
he |  (h  = 1, 2, �, n) be n unit tangents to n congruences

|he of an orthgonal ennuple in a  nV

Proof:

then

i
ih ee
|
|
l

= 0

convariant differentiation with respect to xj, we get

i
ih ee
|
|
l

,j = 0

e
,|
jih

i
e
|
l

+

i
e
|,
l

e
|
ih

j

= 0

g
g
g
206

Tensors and Their Applications

multiplying by

j

ke |  and summing for j, we get

e
,|
ih

i
i
ee
|
k
l

j

|

+

i
j
eee
|,
|
kih
|
i
j
hlk + g

= 0

lhk = 0
hlk = �g

or

Note: Put l = h in equation (1), we get

or
or

2g

llk = �g
llk = 0
llk = 0

THEOREM 11.2 To prove that

lhk

llk

�(1)

lhke | =
ih

e

,|
il

j
e
kj

|

h

Proof: since we know that

=

e

i
ee
,|
hjil
|

j
k

|

lhk

Multiplying by eh|m and summing for h.

lhke |
mh

=

=

=

i
j
eee
e
,|
hjil
|
|
k
(
i
ee
|
h

j
e
|
kj

|
mh

)

|
mh

h

e
,|
il

h

j
e
e
,|
kjil
|

i
m

 since

i
h ee
|

|
mh

 = i
m

h

=

e

(

,|
il

i
d
mj

)

e

j
k

|

lhke | =

mh

j
e
e
|
kjml
,

|

h

h

Replacing m by i, we get

lhke |
mi

=

e
,|
il

j
e
|
kj

h

11.2 REASON  FOR  THE  NAME  "COEFFICIENTS  OF  ROTATION"
Let Cm be a definite curve of the congruence whose unit tangent is
u1 be a unit vector which coincides with the vector
the curve Cm.
Thus

i
me |   and P0 a fidxed point on it. Let
i
le |  at P0 and undergoes a parallel displayment along

and

iu = i

le |  at P0

i
j
jeu
,
|
h

= 0

� (1)

� (2)

g
g
g
g
(cid:229)
g
g
(cid:229)
g
(cid:229)
(cid:229)
d
(cid:229)
d
(cid:229)
g
(cid:229)
g
Ricci's Coefficients of Rotation and Congruence

207

If q

 is the angle between the vectors ui and

i
he | , we have

cos q = ui

ihe |

Differentiating it with respect to arc length sm along Cm, we get

sin�

sin�

dq
mds

dq
mds

=

i
eu

(

| ),
ih

j
ej
m

|

=

i
eu

(

,|
ih

j

+

i
eu
,
j

|
ih

)

j
e
m

|

i
eu

=

,|
ih

j
e
mj

|

+

i
eu
,
j

j

m

|

at the point P0,q =

, we have

2

�

dq
mds

=

=

=

i
eu

j
e
,|
mjih

i
eu

j
e
,|
mjih

|

|

i
ee
|
l

j
e
;|
mjih
,|

from (1)

=

e

,|
ih

j

i
j
ee
|
m
l

|

�

dq
mds
dq
mds

= g

hlm

= � g

hlm

�(3)

� (4)

In Eucliden space of three dimensions,

curve  Cm.  Hence  the  quantities  g
discovered by Ricci  and hence it is called Ricci's coefficient of rotation.

i
he |  about the
hlmare  called  coefficients  of  rotation  of  the  ennuple.  Since  it  was

 is the arc-rate of rotation of the vector

dq
mds

11.3 CURVATURE  OF  CONGRUENCE
The  first  curvature  vector  pl|  of  curve  of  the  congruence  is  the  derived  vector  of
direction. Where

i
he |   in  its  own
i
he |  (h = 1, 2, �, n) be unit tangents to n congruence eh| of an orthogonal ennuple in

a

.nV

If

i
hp |  be the contravariant component of first curvature vector pl|. Then by definition, we have

i
hp | =

j
i
jh ee
,
|
h

from theorem (11.2), we have

i
hp | =

l

i
hlhe |
l

� (1)

q
q
p
(cid:222)
(cid:229)
g
208

Tensors and Their Applications

The magnitude of

i

hp |  is called curvature of the curve of congruence

|he  and denoted by

|hK .

Now,

2
|hK =

i
ppg
h

ij

|

j
|
h

g

ij

=

g

e

i
l

|

hlh

g

i
e
m

|

hmh

l

m

=

= (cid:229)
= (cid:229)

i
eeg
|
ij
l

i
m

|

hlh

hmh

l m

l

m

l
m

hlm

hmh

,  Since

i
ij eeg
|
l

i
|
m

  =

l
m

hmh

hmh

m

(

2)

hmh

|hK = (cid:229)

2

m
This is the required formula for Kh|.

11.4 GEODESIC  CONGRUENCE

If all the curves of a congruence are geodesics then the congruence is called a geodesic congruence.

THEOREM 11.3 A necessary and sufficient condition that congruence C of an orthogonal ennuple
be  a  geodesic  congruence  is  that  the  tendencies  of  all  the  other  congruences  of  the  ennuple  in  the
direction of C vanish indentically.

To obtain necessary and sufficient conditions that a congruence be a geodesic congruence.
Proof: From equation (1), Pg. 207, we have

Or

i
hp | =

i
hlhe |
l

0

,  h"

 and hence

l
i
hp |  = 0 iff g

hlh = 0, h"

. But

i
lp |  = 0 iff the congruence C is geodesic

Since

i
le
|
congruence.

 Hence C is a geodesic congruence iff g
But

hlh = 0,  h"

.

hlh = � g
So, C is a geodesic congruence iff � g
or C is a geodesic congruence iff g
Hence g

lhh = 0.

lhh
lhh = 0,

h"�

.

lhh = 0 are the necessary and sufficient conditions that congruence with unit tangents

be geodesic congruence. Again g

lhh is the tendency of the vector

i
le | in the direction vector

i
he | .

i
he |

(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
(cid:229)
(cid:229)
(cid:229)
(cid:229)
g
g
(cid:229)
g
g
d
d
g
g
g
(cid:229)
g
�
g
Ricci's Coefficients of Rotation and Congruence

209

Thus a congruence C of an orthogonal ennuple is a geodesic congruence iff the tendency of all

other congruences in the direction of C vanish identically.

11.5 NORMAL  CONGRUENCE

A normal congruence is one which intersects orthogonally a family of hypersurfaces.

THEOREM 11.4 Necessary and sufficient conditions that the congruence
be normal is that

|he  of an orthogonal ennuple

nqp = g

npq

Proof: Consider a congruence  C of curves in a  Vn. Let
hypersurfaces. To determines a normal congruence whose tangent vector is grad f

(

)

,

,

1
xx
,

2

nx

�

 or  (cid:209)

.

 = constant be a family of

Let ti be the convariant components of unit tangent vector to C. The congruence  C is a normal
nx

congruence to a family of hypersurface

 =  constant if

1
xx
,

�

(

)

,

,

2

1,

t

1

=

,

2

t

2

�=

,

n =
n

t

y

 (say)

� (1)

In order that

)1�(n

 differential equations given by equation (1) admit a solution which is not

constant, these must constitute a complete system.

From (1)

,
i

t

i

= y  or  f

,i = yti

iyt =

Differentiating it with respect to

ix�
jx , we get

y
j

x

+

y

t

i

t

x

i
j

=

� 2
i xx �

j

Interchanging indices i and j, we get

y
i

x

+

y

t

j

t

x

j
i

=

� 2
j x
x �

i

Subtracting (2) and (3) we get

y
j

dx

+

y

t

i

t

i
x

j

�

y

t

x

i
j

�

t

x

j
i

+(cid:247)

t

i

y
i

x

y
j

x

+

y

t

j

t

i
dx

j

= 0

�

t

j

y
i

x

= 0

� (2)

� (3)

g
f
f
f
f
f
f
f
(cid:222)
f
�
�
�
�
�
�
f
�
�
�
�
�
f
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
210

Multiplying by tk,

Tensors and Their Applications

ytk

t

dx

i
j

�

t

j

i

x

+(cid:247)

tt
ki

y
j

x

�

ti
ik

y
i

dx

= 0

By cyclic permutation of i, j, k in (4), we get

 and

yt
i

t

x

j

k

�

t

k
j
x

+(cid:247)

tt
j

i

y
k

x

�

ti
ik

y
j

x

 = 0

yt
i

t

k
i
x

�

t

x

i
k

+(cid:247)

tt
jk

y
i

x

�

tt
i

j

y
k

x

 = 0

�(4)

� (5)

� (6)

On adding (4) , (5) and (6) we get

ty

k

t

i
j
x

�

j
i

t

x

+(cid:247)

t

i

t

x

j
k

�

or

t

k

(

t

i

,

j

�

t

,
ij

+

)

t

i

(
t

,
kj

�

t

,
jk

+

)

t

j

as

0�y

, where i, j, k = 1, 2, �, n.

t

k
j
x
(
t

+(cid:247)

t

j

t

k
i
x

�

t

x

i
k

 = 0

)

 = 0

�

t

,
ki

� (7)

,
ik

These are the necessary and sufficient conditions that the given congruence be a normal congruence.
Now  suppose  that  the  congruence  is  one  of  an  orthogonal  ennuple  in  Vn. Let  en|i  be  the  unit

tangents of given congruence C so that  ti = en|i

Then equation (7) becomes
en|k (en| i,j � en| j,i) + en| i (en| j,k � en| k,j) + en| j (en|k,i � en|i,k) = 0

Now, multiplying equation (8) by

R
i
p ee
|
|
q

, we get

� (8)

(

e

|
kn

�

e
,|
in

i
k
ee
q
|
p
where p and q are two new indices chosen from 1, 2, �, n � 1. i.e., p,q and n are unequal.

i
k
ee
|
p
q

i
k
ee
|
p
q

e
|
kjn
,

e
|
jkn
,

e
,|
kin

e
|
ikn
,

e
|
ijn
,

e
|
jn

e
|
in

+

+

�

)

)

(

(

)

j

|

|

But

so, we have

e

k
kn e
|
q

|

k
pin ee
|
|

= q
h
= p
h

 = 0,

q �

n

 = 0,

p �

n

e

|
jn

(

e
|
ikn
,

�

e
|
kjn
,

)

k
i
ee
|
q
p

|

= 0

e

jn
|

(

i
e
ee
qikn
|
,

|

k
p

|

�
e
kjn
,
|
en|j (g

nqp � g

i
ee
|
p

k
q
|

)

= 0

npq) = 0

Since

e
|
ikn
,

k
i
ee
|
q
p

|

= g

npq

�
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�

�
�
�
�
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
�
�
�
�
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
-
d
d
Ricci's Coefficients of Rotation and Congruence

211

� (9)
Conversely if equation (9) in true then we get equation (8). Which implies that equation (7) are

nqp,

nqp � g

npq = 0,
npq = g

en| j

� 0

satisfied bty en|i Hence en| is a normal congruence.

Thus necessary and sufficient conditions that the congruence  en| of an orthogonal ennuple be a

normal congruence are that

nqp = g

npq

(p, q = 1, 2, �, n � 1 such that p �

 q)

THEOREM 11.5 Necessary and sufficient conditions that all the congruences of an orthogonal ennuple
be normal.
Proof: If all the congruences of a orthogonal ennuple are normal. Then
nqp = g

npq (p, q = 1, 2, �, n � 1 such that p �  q)

If the indices h, k, l and unequal then

But due to skew-symmetric property i.e., g
So,

hkl = g

hlk

hlk = � g

lhk.

� (1)

hlk = � g

lhk = � g

from (1)

lkh,
(skew-symmetric property).
from  (1)

klh
khl,

hkl, (skew-symmetric property).

hkl = g
= g
= g
hkl = � g
hkl = 0
hkl = 0
hkl = 0

hkl +
2 g

where (l, h, k = 1, 2, �, n such that h, k, kl   are unequal).

11.6 CURL  OF  CONGRUENCE

The curl of the unit tangent to a congruence of curves is called the curl of congruence.

If en| is a given congruence of curves then

Curl en|i = en|i, j  � en| j,i
If curl en| i = 0 then congruence is irrotational.

THEOREM 11.6 If a congruence of curves satisfy two of the following conditions it will also satisfy
the third
(a)
(b)
(c)

that it be a satisfy the third
that it be a geodesic congruence
that it be irrotational.

Proof: Consider an orthogonal ennuple and
this orthogonal ennnuple.

From theorem (11.2), we have

i
he |  (h = 1, 2, �, n) be n unit tangent to n congruences of

lhke |
ih

j
e
e
,|
kjil
|

=

h

(cid:222)
g
(cid:222)
g
g
g
g
g
g
g
g
(cid:222)
(cid:222)
g
(cid:229)
g
212

Tensors and Their Applications

Putting l = n and j = m, we have

n

=

1

h

nhke |
ih

=

e

m
e
|
kmin

,|

Now, multiplying by ek| j and summing with respect to k from 1 to n, we have

n

,
kh

=
1

n

,
kh

=
1

ee
|
jkih
|

nhk

=

m
ee
e
|
kmin

,|

|
jk

=

m
jmine
,|

Since

m
k ee
|

d=|
jk

m
j

ee
|
jkih
|

nhk

=

jine ,|

� (2)

By definition of curl of congruence, we have
curl en|i = en| i, j  � en| j,i

=

curl en| i =

n

kh
,

=
1

n

,
kh

=
1

ee
jkih
|

|

�

nhk

n

kn
,

=
1

e
e
ikjh
;|

|

nhk

from (2)

ee
|
jkih

|

nhk

n

� (cid:229)

,
kh

=

1

ee
|
|
ihjk

nkh

curl

n

ine | = (cid:229)

 (g

nhk � g

nkh) eh| iek| j

=
1

� (3)

,
kh

This double sum may be separated into two sums as follows.
(i) Let h and k take the values 1, 2, �, n � 1.
(ii) Either h = n or k = n or h = k = n.
Now, the equation (3) becomes

curl en| i =

n

1

,
kh

=
1

n

(

�

)

ee
|
jkih

|

nkh

nhk

+

n

1�

h

=
1

(

�

nhn

)

ee
|
jnih
|

nnh

(

�

)

ee
|
in

|
jn

nkn

nnk

g+
(

�

)

e
|
in

e
|
jn

nnn

nnn

Since we know that g
So,

nnk = g

nnh = g

+

=

k
1
nnn= 0.

curl en|i =

curl en|i =

n

1�

,
kh

=

1

n

1�

kh
,

(

g

nhk

�

g

)

e

|
ih

e

|
jk

nkh

+

n

1�

h

=
1

g

e

|
ih

e
|
jn

�

nhn

n

1�

=

1

k

g

e
|
in

e

|
jk

nkn

(

�

)

ee
|
ih

jk
|

nkh

nhk

+

n

1�

=
1

h

(

ee
|
jnih

|

�

ee
|
jkin

|

)

nhn

� (4)

(cid:229)
g
(cid:229)
g
d
(cid:229)
g
(cid:229)
(cid:229)
g
g
(cid:229)
g
g
g
g
g
g
(cid:229)
(cid:229)
-
g
g
g
(cid:229)
(cid:229)
(cid:229)
(cid:229)
g
g
g
(cid:229)
(cid:229)
Ricci's Coefficients of Rotation and Congruence

213

The first term on R.H.S. of equation (4) vanishes
nhk � g
if
i.e.,
if

nkh = 0

nhk = g
nkh.
|ne  is normal.

i.e., if the congruence

Again the second term of R.H.S of equation (4) vanishes

if

  nhn

= 0

i.e., if hnn

  = 0

i.e., if the congruence

|ne  is a geodesic congruence. Further, if first and second term on right

hand side of equation (4) both vanishes then

curl

ine | = 0

Hence we have proved that if the congruence

|ne  satisfies any two of the following conditions

then it will also satisfy the third.

(a)

(b)

(c)

|ne is a normal congruence
|ne is irrotational
|ne is a geodesic congruence

11.7 CANONICAL  CONGRUENCE

It has been shown that given a congruence of curves, it is possible to choose, in a multiply infinite
number of ways, n � 1 other congruences forming with the given congruence an orthogonal ennuple.
Consider the system of n � 1 congruence discovered by Ricci, and known as the system canonical with
respect to the given congruence.

THEOREM 11.7 Necessary and sufficient conditions that the n � 1  congruences eh| of an orthogonal
ennuple be canonical with respect to en|  are

nhk + g

nkh = 0;

(h, k = 1, 2, �, n � 1, h �

 k ).

Proof: Let the given congruence en| be regarded as nth
to given congruence.

 of the required ennuple. Let en|i be  unit tangent

Let us find a quantity r

 and n quantities ei satisfying the n + 1 equations

ijX = � (en| i, j  + en| j,i)

(

X

ij

�

i

r+

eg
ij

i
ee
|
in
E

|
jn

=

=

0

0

i

,

j

= 1, 2, �, n

where w
Writing equation (2) in expansion form, we have

 is a scalar invariant.

1
ee
1|
n

+

e
n

2|

e

2

+��+

n

e
|
nn

e

= 0

and

(

X

1

j

�

g

1

j

i
e

)

+

(

X

�

2

j

g

2

j

)

e

2

�+

(

X

nj

�

g

nj

)

e

n

r+

e

|
jn

= 0

� (1)

� (2)

� (3)

g
g
g
g
g
(cid:239)
(cid:254)
(cid:239)
(cid:253)
(cid:252)
w
w
w
w
214

Tensors and Their Applications

for j = 1, 2, �, n.

(

X

11

�

g

11

)

1
e

+

(

X

21

�

g

21

)

e

2

+�+

(

X

n
1

�

(

X

12

�

g

12

1
e

)

+

(

X

�

22

g

22

)

e

2

+�+

(

X

n

2

�

n

r+

)

e

n

1

e
n

1|

= 0

n

e
)

r+

n

2

e
n

2|

= 0

g

g

�

M
X
(
n
1
from (3)
+

1
ee
ee
1|
n
n
2|
eliminating r

g
1

)

n

1
e

+

(

X

�

2

n

g

2

n

)

e

2

+�+

(

X

nn

�

g

nn

n

)

e

r+

e
|
nn

= 0

2

�+

ee
|
nn

n

r+

0.

= 0

 and the quantities e1, e2, �en, we have the equation

(
(

X

X

11

12

(

X

1

n

�

�

M
�

)
)

)

g

g

11

12

g

n
1

(
(

X

X

�

�

21

22

g

g

21

22

(

X

�

2

n

g

2

n

)
)

)

e
n

1|

e
n

2|

�

�

�

�

(
(

X

n
1

X

n

2

�

�

g

g

n
1

n

2

(

X

�

nn

g

nn

e
nn
|

)
)

)

e
n

1|

e

n
2|

e
nn
|
0

which is of degree n � 1 in w

n � 1. All roots are real. Let w
denoted by r h and

. Hence there will be n � 1 roots of w

 and these roots be w
h be one of these roots and let the corresponding values of r
i
he |  will satisfy the equation (2), we have

h and

i
he |  respectively. Then r
= 0

i
hin ee
|
|

(

and
X
 Similarly w

ij

r+

i
e
h

|

ij

)

g

e
|
jnh

�
= 0
k be another root of these roots, we have
= 0

i
kin ee
|

|

(

X

ij

�

g

ij

)

r+

i
e
k

|

e
jnh

|

= 0

Multiplying (5) by

j
ke |  and (7)

j

he |   and using (4) and (6), we get

(

X

(

X

ij

ij

�

�

g

ij

h

g

ij

k

)

)

j
i
ee
|
|
k
h

= 0

i
ee
|
k

j
|
h

= 0

1, w
2, �,
 and  ei  be

� (4)

� (5)

� (6)

� (7)

� (8)

� (9)

Since

ijX  and

ijg  are symmetric tensor in i and j. Now, interchanging i and j in equation (9), we

get

(

X

�

g

)

i
ee
|
h

j
|
k

= 0

ij
Subtracting (10) and (8), we get

ij

k

(

k � w

k

�
)
h
k
�  0 as w

Since

j
i
eeg
k
|
h
ij
|
h.
h
j
i
eeg
|
hij
|
k

= 0

= 0

� (10)

� (11)

� (12)

w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
w
�

w
(cid:222)
Ricci's Coefficients of Rotation and Congruence

215

This shows that

j
ke |  unit vectors are orthogonal to each other and hence the congruence
|ke  (h �  k) are orthogonal to each other. Hence the n � 1 congruence eh| (h = 1, 2, �, n) thus

i
he |  and

|he  and
determined form an orthogonal ennuple with en|.
Using equation (12), equation (10) becomes

X

=

ij

ij

j
i
eeX
|
h
|
k
(
e
,|
jin

1
2

= 0

)

e

|
ijn
,

= 0

. Then we have,

+
)j

+

i
e
ee
,
hijn
|
k

|

|

Since from (1),
(
e
,|
jin

1
2

j
i
e
ee
|
hjin
|
k

,|

+

= 0

i
j
e
ee
kijn
|
,
h
nhk + g
nkh = 0;

� (13)
Conversely if equation (13) is true then (n � 1) congruences  eh| of the orthogonal ennuple and
|he

|ne . Hence necessary and sufficient condition that the n � 1 congruences

(h, k = 1, 2, �, n � 1 such that

k

)

h �

canonical with respect to
of an orthogonal ennuple be canonical with respect to en| are
nhk  + g

nkh = 0;

(h, k = 1, 2, �, n such that

h �

)

k

THEOREM 11.8 Necessary and sufficient conditions that n �1 mutually orthogonal congruences

|he
|ne , be canconical with respect to the later are gnhk = 0 where k, h

orthogonal to a normal congruence

= 1, 2,�, n � 1 such that h � k.

Proof: By  theorem  (11.7), Necessary and sufficient conditions that (n � 1) congruences
orthogonal ennuple be canonical with respect to en| are

nhk +  g

nkh = 0

(h, k = 1, 2, �, n � 1 such that h �

 k).

If the congruence en| is normal. Then
nkh

nhk = g

The given condition

g+
nhk
nhk + g
2g

 = 0 becomes

nkh
nhk = 0
nhk = 0
nhk = 0

EXAMPLE 1

|he  of an

Proved.

If en| are the congruences canonical with respect to en| prove that (i) w

h =
(iii) If en| is a geodesic congruence, the congruences canonical with respect to it are given by
i
eg
)
ij

nhk (ii) r

h = g

 = 0

X

=

0

(

ij

1
2

hnn,

Solution

Suppose (n � 1) congruences eh| of an orthogonal ennable in a Vn are canonical with respect to

the cougruence eh|  then

i
hin ee
|

|

= 0

� (1)

g
g
g
g
g
g
g
g
w
-
216

Tensors and Their Applications

� (2)

� (3)

� (4)

and

(

X

ij

�

g

ij

)

h

i
e
h

|

r+

e
jnh

|

= 0

where

Xij =

1
2

 (en| i,j + en| j,i)

Since

i
ne |  unit tangents then
i
eeg
|
hij

j
|
h

= 0

(i) Multiplying equation (2) by

i
he | , we get

(

X

ij

�

g

ij

)

h

j
i
ee
|
|
h
h

r+

e

|
jn

h

e

j
|
h

= 0

j
i
eeX
|
hij
|
h

� w

i
eeg
|
hij

j
|
h

h

= 0

since

e

j
hjn e
|
|

 = 0

(from (1))

j
i
eeX
|
h
ij
h

w�|

h

= 0,

since

j
i
eeg
|
hij
|
h

 = 1

1
2

(
e

,|
in

j

+

e
|
ijn
,

)

i
ee
|
h

j
|
h

�

= 0;

h

from (3)

1
2

(
e

,|
in

i
ee
|
hj

j
|
h

+

j
i
e
ee
,
hijn
|
h

|

|

�)

= 0

h

 (g

nhh + g

1
2

nhh) � w

h = 0
h = g
j
ne | , we get

nhh

(ii) Multiplying (2) by

(

X

ij

�

g

ij

)

h

j
i
ee
|
|
n
h

r+

e

|
jn

h

e

j
|
n

= 0

i
j
eeX
hij
|
n
|

�

i
eeg
|
hij

j
n
|

h

r+

h

1

= 0

from (1)

1
2

(
e

,|
in

j

+

e
|
ijn
,

)

i
ee
|
h

j
|
h

r+

h

= 0,

since

i
h ee
|

j
|
h

 = 0

1
2

(
e

,|
in

j

i
ee
|
h

j
|
n

+

e
|
ijn
,

j
i
ee
|
h
h

|

)

r+

= 0

h

1
2

(g

nhn + g

nnh) + r

h = 0

or

(iii)

1
2

nhn, since

nnh = 0

hnn

h = �

h =

1
2

If en| is a geodesic congruence, then g
1 �
h =
2
h = 0

hnn = 0 from the result (ii), we have

0

w
w
w
w
w
w
�
w
r
g
g
r
g
r
r
Ricci's Coefficients of Rotation and Congruence

217

from equation (2), we get
+
| 0
i

i
e
h

w

X

g

�

)

(

ij

ij

h

this  gives

(

X

ij

�

g

ij

)

e

 = 0 .

e
|
jn

= 0 or

(

X

ij

�

g

ij

)

h

e

i
|
h

 = 0

EXAMPLE 2

Prove that when a manifold admits an orthogonal systyem of n normal congruences then any of

these in canonical with respect to each other congruence of the system.
solution

i
he |  (h = 1, 2, �, n) be unit tangents to n normal congruences of an orthogonal ennuple in a

Let
Vn. So that

lhk = 0, where l, h, k = 1, 2,...n such that l, h, k being unequal.

It is required to show that a congruence eh| is canonical with respect to the congruence ek| (h, k = 1, 2,
�, h �  k). We know that the  n � 1 congruence en| of an orthogonal ennuple be canonical to en| iff

nhk + g

nkh = 0

This condition is satisfied by virtue of equation (1). Hence (n � 1) congruences eh| of an orthogonal

ennuple are canonical to en|.

Similarly we can show that any n � 1 congruences are canonical to the remaining congruence.
It follows from the above results that any one congruence is canonical with respect to each other

congruence of the system.

EXERCISE

1. If  f  is a scalar invariant

n

=
1

h

i
i
,
ee
hij
|
h

|

= D 2f

2. The coefficient of r n � 1 in the expansion of the determinant |f ,ij � r gij| is equal to (cid:209)
3. If eh| are the unit tangents to n mutually orthogonal normal congruences and e11+ be21 is also a

2f .

normal congruence then ae11 � be21 is a normal congruence.

4. Show that

s
h

s
k

�

s
k

Q
s
h

=

l

(

�

)

lkh

lhk

Q
s
l

where Sh denotes the arc length of a curve through a point P of an ennuple and Q is a scalar invariant.
5. If the congruence eh| (h = 1, 2, �  n � 1) of an orthogonal ennuple are normal, prove that they are
|he .

canonical with respect to other congruence

w
w
g
g
(cid:229)
f
�
�
�
�
�
f
�
�
�
�
�
g
g
(cid:229)
CHAPTER � 12

HYPERSURFACES

12.1 INTRODUCTION

We have already studied (Art 10.3 chapter 10) that if m > n then we call Vn to be a subspace of Vm and
consequently Vm is called enveloping space of Vn. We also know that there are m � n linearly independent
normals Na to  Vn. (Art 10.3 Theorem 10.5, chapter 10). If we take  m =  n + 1 then  Vn  is  said  to  be
hypersurface of the enveloping space  Vn + 1

Let Vn be Riemannian space of n dimensions referred to cordinates xi and having the metric ds2 =
gij dxi dxj. Let Vn be Riemannian space of m dimensions referred to coordinates ya and having the metric
, g  � take the values 1, 2, �, m and latin indices
, b
ds2 = aa
 and  gij
i, j, k, �  take the values 1, 2, �n. Then we have, the relation between aa

 dyb . Where m > n. Let Greek letters a

 dya

gij = aa

y

y

i

x

x

j

� (1)

Since  the  function  ya  are invariants for transformations of the coordinates  xi in  Vn,  their  first
covariant derivatives with respect to the metric of Vn are the same as their ordinary derivatives with
respect to the variables xi.

i.e.,

a
iy, =

y
ix

Then equation (1) can be written as

gij = aa

i yy
,

,

j

� (2)

The  vector  of  Vn  +  i  whose  contravariant  components  are

  is  tangential  to  the  curve  of
parameter xi in Vn. Consequently if Na  are the contravariant components of the unit vector normal to
Vn. Then we have

iy,

and

aa

b Nb
iy, = 0,
b N a N b = 1
aa

(i = 1, 2, �, n)

� (3)

� (4)

b
b
b
�
�
�
�
b
a
�
�
a
b
b
a
a
a
Hypersurface

219

12.2GENERALISED  COVARIANT  DIFFERENTIATION
Let C be any curve in Vn and s its arc length. The along this curve the x's and the y's may be expressed
as function of s only. Let ua  and n b  be the components in the  y's of two unit vector fields which are
parallel along C with respect to Vm. Similarly w i the components in x's of a unit vector field which is
parallel along C with respect to Vn. Now, ua is parallel along C relative to Vm then we have,

u

y

u

y

dy
ds

du
ds

�

u

�

u

�

u

u

,

dy
ds

dy
ds

dy
ds

dy
ds

dua
ds

= 0

= 0

= 0

= 0

u

=

dy
ds

similarly v b

  is parallel along C relative to Vm  then

dv
ds

�

v

=

and wi is parallel along C relative to Vn then
i

dwi
ds

j

�

w

=

kj

dy
ds

k

dx
ds

� (1)

� (2)

      � (3)

The Christoffel symbol with Greek indices being formed with respect to the  aa

 and the  y's and

christoffel symbol with Latin indices with respect to the  gij and the  x's.

Let

iA  be a tensor field, defined along C, which is mixed tenser of the second order in the y's and
iA  is scalar invariant and it is a function of s

a covariant vector in the x's. Then the product ua  vb   wi
along c. Its derivative with respect to s is also a scalar invariant.

Differentiating ua  vb  wi

iA  with respect to s, we have

d
dt

i Awvu

(

)

i

=

wvu

i

=

wvu

i

dA

ds

dA

ds

i

+

i

+

du

ds

du
ds

i
Awv

 +

i

dv
ds

i
Awu

+

i

i
Awv

+

i

dv
ds

i
Awu

 +

i

i

dw
ds

j

dw
ds

Avu

i

Avu

j

b
b
a
b
g
b
a
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
g
�
�
b
g
b
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
g
�
�
b
g
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
g
b
g
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
a
b
a
g
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
g
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
b
a
b
a
b
a
b
b
a
a
b
b
a
a
b
b
a
a
b
b
a
a
b
a
b
a
d
a
d
d
b
b
d
a
b
b
a
a
b
b
a
220

Tensors and Their Applications

wvu

=

dA

i

i

+

ds

wvuA

.

i

i

dy
ds

 �

i
wvuA

.

i

dy
ds

i
wvuA

j

�

j
ik

k

xd
ds

, by equation (1), (2) & (3).

wvu

i

=

dA

i

ds

+

A

.

i

dy
ds

�

A

i

dy
ds

�

A

.

j

j

ki

k

dx
ds

= Scalar, along C

Since the outer product ua vb wi is a tensor and hence from quotient law that the expression within
iA  with respect to

iA  and this tensor is called intrinsic derivative of

the bracket is tenser of the type
s.

The expression within the bracket can also be expressed as

k
dx
ds

A

i

k

x

+

A

i

y
,

k

�

A

i

y
,

k

�

A

j

j
ik

Since

dxk
ds

 is arbitrary..

So, by Quotient law the expression within the bracket is a tensor and is called tensor derivative of

iA  with respect to xk. It is denoted by

kiA ;

. Then we have

kiA ;

=

A

i
k
dx

+

A

i

y

,

k

�

A

i

y

,

k

�

A

j

j
ik

.

kiA ;

 is also defined as generalised covariant derivative
Note:� Semi-colon(;) is used to denote tensor differentiation.

iA  with respect to xk .

12.3 LAWS  OF  TENSOR  DIFFERENTIATION

THEOREM 12.1 Tensor differentiation of sums and products obeys the ordinary rules of differentiation.

Proof: Suppose

b BA ,

 and Bg  are tensors in V m.

(i) To prove that

b +
A

(

;)
kB

=

A

+
k B
;

;

k

b + B
A

 be denoted by the tensor

bC .

Let the sum
Now,

b kC ; =

C

k

x

+

C

c
�,
y
k

C
a

ca

,
y

c
k

c

g
b
a
d
b
a
b
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
g
a
g
b
a
a
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
a
b
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
d
a
a
b
g
a
d
g
d
b
a
b
b
a
a
b
a
b
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
g
a
�
�
a
b
g
a
d
g
d
b
a
b
a
b
a
b
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
d
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
g
a
�
a
b
g
a
d
g
d
b
a
b
a
b
a
b
a
b
a
a
b
a
a
b
a
b
a
b
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
a
b
a
b
Hypersurface

221

b +
A

(

;)
kB

 =

(

A

+

B

)

k

dx

+

a

(

A

+

A

)

c
y
,
k

(

A
a

�

+ a
B
a

)

ca

a

c

y

c
,
k

=

A

k

x

B

k

x

+

a

A

ac

c
, �
y
k

A
a

+

a

B

ac

c
, �
y
k

B
a

a
c

a
c

y

c
,
k

 +

c
y
,
k

b +
A

(

;)
kB

 =

A

a
ab
;

+

B

a
b

;
k

 Hence  the result (i)

(ii) Prove that

b +
A

(

B

);

k

=

a
BA
b
g
;
k

+

a
BA
b
g

;

k

Let

we have

b BA

=

gD

Then

gD  is a tensor

a

bg kD ; =

D

k

x

+

a
D
;
k

ac

yc �

a
D
g
a

a

b
c

y

c
, �
k

a
D
b
a

a

g
c

y

c
,
k

(

a
BA
b
g

;)

k

=

(

BA

)

k

dx

+

a
BA

y

c
,
k

 �

BA
a

ac

a

c

c
, �
y
k

BA

a

a

c

y

c
,
k

A

k

dx

+

A

=

c
, �
y
k

A
a

ac

a
c

c
By
,
k

A

 +

B
k
dx

�

B
a

a
c

c
y
,
k

(

;)
kBA

=

BA
;
k

+

BA
a

;

k

Hence the result (ii).

Note:� a

, b

, g , a, c  take values from 1 to m while k take values from 1 to n.

THEOREM 12.2 To show that

a ;

j

= 0

or

To prove that the metric tensor of the enveloping space is generalised covariant constant with respect
to the Christoffel symbol of the subspace.

Proof: We have (see pg. 220)

a

=

;
i

=

a

i

dx

a

i

dx

�

a

, �
y
i

a

y

,
i

(

[

�

]
[
d+b

,

]

)

y
,

i

,

\
a
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
a
b
b
a
b
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
b
a
b
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
b
a
b
a
b
a
g
a
g
a
a
b
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
a
b
g
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
\
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
g
b
g
g
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
a
b
g
a
g
a
a
b
a
b
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
�
�
�
�
�
�
�
�
�
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
�
g
a
b
g
a
b
g
a
g
a
b
a
b
a
b
d
a
g
d
g
b
a
b
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
d
g
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
a
b
g
�
d
a
b
a
b
a
d
�
222

or

or

Tensors and Their Applications

a

=

;
i

a

=

;
i

a

i

dx

a

i

dx

�

�

a

y

a

y

y

,
i

y

i

dx

=

a
ab �
i
x

a
ab
i
x

= 0

Proved.

12.4 GAUSS'S  FORMULA

At a point of a hypersurface Vn of a Riemannian space Vn + 1,  the formula of Gauss are given by

Proof: Since ya  is an invariant for transformation of the x's and its tensor derivative is the same as its
covariant derivative with respect  to the x�s, so that

ijy; =

Nij

a

iy; = a

iy, =

dy
idx

Again tenser derivative of equation (1) with respect to x�s is

ijy; =

(

; )
iy

;

j

=

iy
( ,

)

=

=

(

, �)
y
i

y
,

l

j

dx

y

i

x

j

x

,�
y

l

l
ij

l

ij

+

yy
,
i
,

j

+

yy
,
i

,

j

ijy; =

2

y

i
xx

j

�

y
,

l

l
ij

+

yy
,
,
i

j

Interchanging j and i in (2), we have

jiy; =

jiy; =

2

y

j

x

i

x

2

y

i
xx

j

�

y

,
l

�

y

,
l

l

ji

l

ij

+

yy
,
j

,
i

+

yy
,
l

,

j

[On interchanging b

 and g  in third term of R.H.S. of (3) and using

 =

].

On comparing equation (2) and (3), we have

ijy; =

jiy;

So,

ijy;

is symmetrical with respect to indices i & j.

� (1)

� (2)

� (3)

a
b
d
d
a
b
a
b
�
�
�
a
b
d
d
a
b
a
b
�
(cid:215)
�
�
�
�
�
�
�
a
a
W
a
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
g
b
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
�
�
�
�
g
b
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
g
b
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
g
b
a
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
�
�
�
g
b
a
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
b
g
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
g
b
a
a
a
a
Hypersurface

223

Let gijdxidxj and

a

dy

dy

 be fundamental forms corresponding to Vn and Vn + 1 respectively..

Then

or

gij =

a

y

i

x

y

x

j

gij =

a

i yy
;
;

j

Taking tensor derivative of both sides with respect to xk

gij;k =

a

yy
;
i
;

j

;
k

+

a

yy
;
i
;

jk

+

a

y
;
ik

y
;

j

gij;k = 0

 and

a

 = 0.

;
k

But

we have,

0 =

a

yy
;
i
;

jk

+

ya
;

ik

y
;

j

or

a

yy
;
,
ik

j

+

yya
,
;

i

jk

= 0, using (1), Art. 12.4

By cyclic permutation on i, j, k in (4), we have

a

ab

a
b
yy
;
,
ji
k

+

ba
yya
ab
,
;
j
ki

= 0

and

a

y
;

kj

y
,

i

+

yya

,
k

;
ij

= 0

subtracting equation (4) from the sum of (5) and (6), we get.

or

2

a

ij yy
;

,

k

= 0

a

ij yy
;

,
k

= 0

� (4)

 � (5)

� (6)

� (7)

This shows that

ijy;

 is normal (orthogonal) to

ky,

. Since

ky,

 is tangential to Vn and hence

ijy;

 is

normal to V n.  Then we can write

ij
where Na  is unit vector normal to Vn and W

ijy; =

N W

ij is a symmetric covariant tensor of rank two. Since

(8)

 is a function of x's the tenser W

ijy;

ij is also a function of x's.

The equation (8) are called Gauss�s formula.
From equation (8)

ijy; =

Nij

Nay ij;

=

=

NNaij

2Nij

b
a
a
b
�
�
�
�
b
a
a
b
b
a
a
b
b
a
a
b
b
a
a
b
b
a
a
b
a
b
b
a
a
b
b
a
a
b
b
a
a
b
b
a
a
b
b
a
a
b
b
a
a
b
b
a
a
b
b
a
a
b
a
b
b
a
a
a
a
a
a
W
b
a
b
a
a
b
a
b
W
W
224

or

Nay ij;

=

,ij

N = 1

ij =

Nay ij
;

.

The quadratic differential form

i
dx

dx

j

ij

Tensors and Their Applications

is called the second fundamental form for the hypersurface Vn of Vn + 1. The components of tenser W
are said to be coefficient of second fundamental form.
Note: The quadratic differential form gijdxidxj is called first fundamental form.

ij

12.5 CURVATURE  OF  A  CURVE  IN  A  HYPERSURFACE  AND  NORMAL  CURVATURE
If Ua  and ui be the contravariant components of the vector u relative to Vn and Vn + 1 respectively then
we have (from chapter 10, Theorem 10.4)

U

=

y
i
x

i

u

 =

i
i uy
,

� (1)

Let the derived vector to vector u along C with respect to metric of Vn and Vn + 1 are denoted by

p and q respectively. Then

and

pi

=

u

i
j,

q =

U

b,

=

U

;

j

j

xd
sd

yd
sd

j

dx
ds

, (from equation 1, Art 12.4)

� (2)

Taking the tensor derivative of each side of equation (1) with respect to x's, we have

jU; =

uy
;
ij

i

+

uy
,
i

i
,

j

By Gauss's formula, we have

W=

y
ij;

N

ij

Then

jU; =

uN

ij

i

+

uy
,
i

i
,

j

Putting the value of

ju;

 in equation (2), we have

or

where

q =

(

uN

i

+

ij

uy
,
i

i
,

j

)

j

dx
ds

j

dx
ds

a +

N

i

py
,
i

q =

ip =

i
u
j,

i

u

ij

j

dx
ds

� (3)

b
a
b
a
W
W
b
a
b
a
W
a
�
�
a
a
a
b
a
a
a
a
a
a
a
a
a
a
W
a
a
a
a
W
a
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
W
Hypersurface

225

Now suppose that vector u is a the unit tangent t to the curve C. Then the derived vectors q and
p are the curvature vectors of a C relatively to Vn + 1  and Vn respectively. Then equation (3) becomes.

q =

Kn =

i

dx
ds

ij

j

dx
ds

a +

N

i

py
,
i

i

dx
ds

ij

j

dx
ds

, we get

Taking

� (4)

 pi
...(5)
Kn is called Normal curvature of  Vn  at  any  point  P of the curve  C and  Kn  Na  is  called  normal

q = Kn Na +

iy,

curvature vector of Vn + 1 in the direction of C.
Meaunier's  Theorem

If Ka and Kn are the first curvature of C relative to Vn + 1 and normal curvature of Vn respectively
and w is the angle between  Nr  and  Cr  (C being the unit vector of Vn + 1 then the relation between Ka, Kn
and  w  is given by

Proof: We know that

Kn =

aK

cos

q =

NK
n

a +

i
yp
,

i

...(1)

where

xd
sd
Let Ka and Kg be the first curvatures of C with respect Vn + 1 and Vn respectively then

xd
sd

Kn =

ij

i

j

Ka =

qqa

,

 Kg =

i
ppg
ij

j

Let  w

 be the angle between  Nr  and  Cr .

Then

r (cid:215)
CN r

=

CN

cos

cos

as |N| = |C| = 1

=
CN rr . =

cos
If  br   is  a  unit  vector  of  Vn +  1  in  the  direction  principal  normal  of  Cr

equation (1) becomes

r =

CKa

r +
r
NKbK

n

g

Taking scalar product of equation (3) with  N

r , we have
rr
.
NNKbNK

rr
. +

n

g

rr
. =
CNKa

aK

cos

=

K

g

+(cid:215)
0

K

;1

n

from  (2)

Kn =

aK

cos

...(2)

with  respect  to  Vn  then

� (3)

 � (4)

Proved.

a
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
W
W
a
a
w
a
a
W
b
a
a
b
w
(cid:215)
w
w
w
(cid:215)

w
226

EXAMPLE 1

Tensors and Their Applications

Show that the normal curvature is the difference of squares of geodesic curvatures.

Solution

We know that (from Meurier�s Theorem, equation 3)

g
Taking modulus of both sides, we get

r =

CKa

r +
r
NKbK

n

or

2
aK =

2
K +
g K

2
n

2
nK =

K

2 � g
2
a K

.

Theorem 12.3 To show that the first curvature in Vn +  1 of a geodesic of the hypersurface Vn is the
normal curvature of the hypersurface in the direction of the geodesic.
Proof: From, example 1, we have

2
aK =

2
K +
g K

2
n

If C is a geodesic of Vn  then
Then we have

i

p

(cid:222)=
0

K

g

=

0

2
2
aK =
nK
Ka = Kn.

Dupin's  Theorem

Proved.

The sum of normal curvatures of a hypersurface  Vn for  n mutually orthogonal directions is an

invariant and equal to W

ijgij.

Proof: Let

i
he |  (h = 1, 2, �, n) be unit tangents to n congruences of an orthogonal ennuple in a Vn. Let

Knh be normal curvature of the hypersurface  Vn in the direction of the congruence

|he . Then

Knh =

j
i
ee
|
|
h
h

ij

The sum of normal curvatures for n mutually orthogonal directions of an orthogonal ennuple is a

Vn is

n

=
1

h

nhK

= (cid:229)

n

=

1

h

j
i
ee
|
h
|
h

ij

=

ij

n

h

=
1

j
i
ee
|
h
|
h

ij g ij

= W
= Scalar invariant

Proved.

(cid:222)
W
(cid:229)
W
(cid:229)
W
Hypersurface

12.6 DEFINITIONS

227

(a) First curvature  (or  mean  curvature)  of  the  hypersurface  Vn  at  point  P.
It is defined as the sum of normal curvatures of a hypersurface Vn form mutually orthogonal directions
at P and is denoted by M. Then

M = W

ij g ij

(b) Minimal  Hypersurface
The hypersurface Vn is said to be minimal if M = 0

i.e.,

ij g ij = 0

(c) Principle  normal  curvatures
The maximum and minimum values of  Kn are said to be the principle normal curvatures of  Vn at  P.
Since these maximum and minimum values of Kn correspond to the principal directions of the symmetric
tensor

.

ij

(d) Principal  directions  of  the  hypersurface  at  a  point  P.

The principal directions determined by the symmetric tensor
of the hypersurface at P.
(e) Line  of  curvature  in  Vn
A line of curvature in a hypersurface  Vn is a curve such that its direction at any point is a principal
direction.

 at P are said to be principal directions

ij

Hence we have n congruences of lines of curvature of a V n.

THEOREM 12.4 To show that the mean curvature of a hypersurface is equal to the negative of the
divergence of the unit normal.

or

To show that the first curvature of a hypersurface is equal to the negative of the divergence of the unit
normal.

or

To show that the normal curvature of a hypersurface for any direction is the negative of the tendency
of the unit normal in that direction.
Proof: Let  N  be  the  unit  normal  vector  to  the  hypersurface  Vn  in  y's  and  let  Na   be  its  covariant
components. Let t be the unit tangent vector to N  congruences eh| (h =  1, 2, �,  n) of an orthogonal
ennuple in  Vn and let
|hT  be the contravariant components in  Vn +  1 of  t. Since  t is orthogonal to  N,
therefore

NTh|

= 0

Taking covariant derivative of equation (1) with regard to

y  provides

NT
h
| ,

+

NT
h
|

,

= 0

Multiplying equation (2) by

hT , we get

� (1)

� (2)

NTT

h

| ,

h

|

+

NTT
h

h

|

|

= 0

,

W
W
W
a
a
a
b
b
a
a
a
a
b
b
b
a
b
a
a
b
a
b
228

Tensors and Their Applications

(
NTT
h

| ,
h

)

|

= �

h TTN
,

|

h
|

� (3)

|

Now

 is the first curvature of the curve eh| and

h TT
|,
h
|he . Hence equation (3) implies that the normal component of the first curvature of the

|hT  is the tendency of

direction of
relative to Vn + 1 or the normal curvature of Vn in the direction of the curve

|hT .

,N

|he

.

aN  is the

|he

= � tendency of N in the direction of the curve

|he

� (4)

Taking summation of both of (3) and (4) for h = 1, 2, �, n
we have
i.e., mean curvature or first curvature of a hypersurface = � divergence of the unit normal.

Corollary: To prove that

M = � divn + 1N

Proof: Since N is a vector of unit magnitude  i.e., constant magnitude, its tendency is zero. also by
 N differ only by the tendency of the vector N in its direction. But the
definition, the divn + 1 N and divn
tendency of N is zero hence it follows that

Hence

divn N = � div n + 1 N

M = � div n N = � div n + 1 N.

12.7 EULER'S  THEOREM

Statement

The normal curvature  Kn of Vn  for any direction of  ar  in Vn is given by
n

Kn =

2
cos

K

h

h

h
where Kh are the principal curvature and a
eh|
Proof: The  principal  directions  in  Vn    determined  by  the  symmetric  covariant  tensor  teser  W
given by

=
1
h are the angles between direction of  ar  and  the congruence

(W

�

gK
h

ij

)

p

ij

i
|
h

= 0

where Kh are the roots of the equation

ij Kg�
ij

= 0

ij  are

� (1)

� (2)

and

i
hp |  are the unit tangents to n congruences of lines of curvature.

The roots of the equation (2) are the maximum and minimum values of the quality Kn defined by

ij

i
pp
i

j

i

� (3)

ppg

ij

Kn =

a
b
a
b
a
b
b
a
b
a
b
b
a
b
a
a
(cid:229)
W
W
Hypersurface

Multiplying equation (1) by

i

kp | , (K �

 h), we get

(W

�

gK
n

)

i
pp
|
h

i
k

= 0

ij

ij
The principal directions satisfy the equation
)k

(
h �

= 0

i
pp
|
h

j
|
k

ij

|

229

� (4)

Let

|he  be the unit tangents to the n congruences of lines of curvature. Then the principal curvatures

are given by

Any other unit vector  ar  in Vn is expressible in the form

Kh =

i
i
ee
|
h
h

|

ij

(h = 1, 2, �, n)

or

where

ar =

ai =

cos ha
|

=

e

|
h

cos

h

h

n

i
e
|
h

acos

h

=
h
1
|hea rr (cid:215)

 =

iea

|
ih

� (5)

� (6)

ai being the contravariant components of  ar  and a
normal curvature of Vn for the direction of  ar  is given by

h being the inclination of vector a to eh|.  The

from (6), we get

Kn =

j

i
ij aa

i
e
|
h

cos

h

j
e
|
k

cos

h

h

h

(

i
j
ee
|
h
k

|

)

ij

cos

cos

h

k

(

j
i
ee
|
|
h
h

ij

2

)

cos

h

2
cos

K

h

h

Kn =

=

=

Kn =

ij

n

kh
,

=
1

n

1

=
h
n

=
1

h

� (7)

This is a generalisation of Euler�s Theorem.

12.8 CONJUGATE  DIRECTIONS  AND  ASYMPTOTIC DIRECTIONS IN  A HYPERSURFACE
The directions of two vector,  ar  and  br , at a point in Vn are said to be conjugate if

j

i
ij ba

= 0

� (1)

and two congruences of curves in the hypersurface are said to be conjugate if the directions of the two
curves through any point are conjugate.

W
W
a
(cid:229)
(cid:229)
W
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
a
(cid:247)
(cid:247)
?
(cid:246)
(cid:231)
(cid:231)
?
(cid:230)
a
W
(cid:229)
(cid:229)
W
(cid:229)

a
a
a
W
(cid:229)
a
(cid:229)
W
230

Tensors and Their Applications

A direction in Vn which is self-conjugate is said to be asymptotic and the curves whose direction

are along asymptotic directions are called asymptotic lines.

Therefore the direction the vector  ar  at a point of Vn be asymptotic if

j

i
aa

ij

= 0

The asymptotic lines at a point of a hypersurface satisfies the differential equation

i
dx

j

dx

ij

= 0

� (2)

� (3)

THEOREM 12.5 If a curve C in a hypersurface V n has any two of the following properties it has the
third

(i)
(ii)
(iii)

it is a geodesic in the hypersurface Vn
it is a geodesic in the enveloping space Vn + 1
it is an asymptotic line in the hypersurface V n.

Proof:  Let C be a curve in the hypersurface Vn. The normal curvature Kn of the hypersurface Vn in the
direction of C is given by

where Ka and Kg  are the first curvatures of C relative to enveloping space Vn + 1 and hypersurface Vn
respectively.

2
aK =

2
K +
g K

2
n

� (1)

Suppose C is a geodesic in the hypersurface Vn [i.e., (i) holds] then Kg = 0.
If C is also a geodesic in the enveloping space Vn + 1 [i.e., (ii) holds] then Ka = 0.
Now, using these values in equation (1), we have

2
nK = 0

nK(cid:222)

 = 0

Implies that C is an asymptotic line is the hypersurface Vn i.e., (iii) holds.
Hence we have proved that (i) and (ii)  (cid:222)
Similarly we have proved that
(ii) and (iii)  (cid:222)
(i)
 and  (i) and (iii)  (cid:222)

 (iii)

 (ii)

12.9 TENSOR  DERIVATIVE  OF  THE  UNIT  NORMAL
The  function  ya   are  invariants  for  transformations  of  the  coordinates  xi in  Vn  their  first  covariant
derivatives with respect to the metric of Vn are the same as their ordinary derivatives with respect to the
variables xi.

i.e.,

iy; =

y
,
i

=

y
x

i

� (1)

The unit Normal

N  be the contravariant vector in the  y�s whose tensor derivative with respect

to the x�s is

iN; =

+

N

i

x

yN
,

i

� (2)

W
W
a
�
�
a
a
a
a
d
b
a
(cid:254)
(cid:253)
(cid:252)
(cid:238)
(cid:237)
(cid:236)
d
b
a
�
�
Hypersurface

Since

NNa

= 1

Tensor derivative of this equation with respect to xi gives

NNa

+

;
i

NNa
;
i

= 0

Interchanging a

 and b

 in Ist term, we get

or

NNa

;

i

+

NNa

;
i

= 0

NNa

;

i

+

NNa
;

i

= 0

2

iNNa
;

= 0

iNNa

; = 0

231

� (3)

� (4)

which shows that

iN;

 is orthogonal to the normal and therefore tangential to the hypersurface.

Thus

iN;

 can be expressed in terms to tangential vectors

ky,

 to Vn so that

iN;
k
iA  is a mixed tensor of second order in Vn  to be determined.

where
Since unit normal Na  is orthogonal to tangential vector

i yA ,

=

k

iy,

 in Vn. Then

� (5)

Taking tensor derivative with respect to x j, we get

iyNa

, = 0

yNa
,

j

;

+

i

yNa
,

ij

= 0

since

y
ij,

W=

N

ij

or

or

yNa

;

j

+

Na

,
i

N

ij

= 0

yNa

;

j

W+

i
,

(

NNa

)

= 0

ij

from equation (3),

NNa

= 1

or

or

W+b
j yNa
,
i

;

ki Ag

k
j

W+

= 0

= 0

ij

ij

since

a

y
,

k

y
,

i

=

a

y

k
x

y

i
x

=

g

ki

Multiplying this equation by gim, we get

im

Agg

ki

W+

k
j

im

g

ij

= 0

a
b
a
b
b
a
a
b
b
a
a
b
b
a
a
b
a
b
b
a
b
a
a
b
a
b
a
b
a
b
a
b
a
b
a
b
a
a
a
a
a
a
a
b
a
a
b
b
a
a
b
b
a
a
b
b
b
b
a
a
b
b
a
a
b
W
b
a
a
b
b
a
a
b
b
a
a
b
a
a
b
�
�
�
�
�
�
�
�
�
�
b
a
a
b
a
a
a
b
232

Tensors and Their Applications

m
k

A

k
j

W+

im

g

ij

= 0

m
A
i

W+

im

g

ij

= 0

m
iA =

W�

im

ij g

Substituting this value in equation (5), we get

iN; =

ik
yg
,

k

ij

 =

�

jk
yg
,

k

ji

iN; =

�

jk
yg

,

k

ij

� (6)

This is the required expression for the tensor derivative of

N .

Theorem 12.6 The derived vector of the unit normal with respect to the enveloping space, along a
curve provided it be a line of curvature of the hypersurface.

Proof: Since the tensor derivative of

N  is

iN; =

�

jk
yg

,

k

ij

Consider a unit vector

ie  tangential to the curve C.

Then

i

i eN
;

=

�

ij

jk

i
eyg
,

k

The direction of

i

i eN
;

 is identical with that of

ie

Then

from (2), we have

i

i eN
;

=

�

i

,

i ey
,

(l

 is scalar constant)

� (1)

� (2)

i
jk
eyg
,

k

ij

=

i

i ey
,

Multiplying both sides of this equation by

a

i
jk
aeg

(

ij

y

k
,

y
,

l

)

=

i
yyae

(

,

i

ly
,

)

l
,

since

g

il

b=
a

yy
,
i

,
l

jk
geg

i

kl

ij

=

igel

ii
ij ed
l

=

igel

il

il

i
e

�

il

i
ge

il

= 0

i

il

(

�

l

(l = 1, 2, �, n)
This equation implies that the direction of ei is a principal direction for the symmetric tensor W
il
i.e., ei is a principal direction for the hypersurface  V n.  Hence by definition the curve  C  is  a  line  of
curvature V n.

g )
il

= 0

e

d
a
a
W
a
W
a
a
W
a
a
a
a
W
a
a
W
a
a
a
l
a
W
a
l
b
a
b
b
a
a
b
W
b
a
a
b
l
W
b
a
a
W
l
W
W
Hypersurface

12.10 THE  EQUATION  OF  GAUSS  AND  CODAZZI

since we know that (pg. 86, equation 5)

A

,
jki

�

A
i

,

kj

=

pRA

p
ijk

233

� (1)

where Ai  is  a  covariant  tenser  of  rank  one  and  difference  of  two  tensers
tenser of rank three.

A

,
jki

�

A
,
i

kj

  is  a  covariant

It

iy,
equation (1), we get

  are  components  of  a  covariant  tensor  of  rank  one  in  x's.  Then replacing  Ai  by

y
,

ijk

�

y
,

ikj

 =

p Ry
,

p
ijk

 =

ph

Rgy
,

p

hijk

,  Since

p
ijkR  =

phRg

hijk

where  hijkR

 are Riemann symbols for the tensor

ijg

We know that

and

ijy, =

Nij

iN, =

�

ik
yg
,

k

ij

iy,

  in

� (2)

� (3)

� (4)

Let

eR

 are Riemann symbols for the tensor

ba

 and evaluated at points of the hypersurface

using equation (3) and (4), equation (2) becomes

gy
,
p

ph

[

R

hijk

(�

hj

ik

�

hk

�)]

N

ij

(

�

,
kij

�)

R

ik

,

j

yy
,
i

,

j

y
,

k

� (5)

Multiplying equation (5) by

a

,  and summed with respect to a
ly

. Using the relations

a

y

i y
,
,

l

= 0

bg

 =

lyN , = 0,

and

we get

Multiplying (6) by aa

lijkR

e+
R
b N b  and summing with respect to a

=

�

(

)

ik

lk

lj

ij

and

lyNa

, = 0

b

BNa
ab

b

= 0

we get

, �
kij

e+
R

ik

,

j

yyyN
,
i

j

,

= 0

,

k

yyy
,
l
,

j

i

,
. Using relations

k

y
,

� (6)

� (7)

Hence, The equation (6) are  generalisation of the Gauss Characteristic equation and equation

(7) of the Mainardi-Codazzi equations.

a
a
a
a
a
a
a
a
W
a
a
W
a
g
d
a
e
d
g
a
g
d
e
a
a
W
W
W
W
W
W
b
a
b
b
a
a
b
a
a
b
e
d
g
b
b
g
d
W
W
W
W
a
b
a
b
e
d
g
b
b
g
d
W
W
234

Tensors and Their Applications

12.11 HYPERSURFACES  WITH  INDETERMINATE  LINES  OF  CURVATURE

A point of a hypersurface at which the lines of curvature are indeterminate is called an Umbilical Point.

The lines of curvature may be indeterminate at every point of the hypersurface iff

=

ij

ijgw

where w
The mean curvature M of such a hypersurface is given by

 is an invariant

M =

w =

ij

ijg

 =

ij ggw

ij

 = w n

M
n

.

So that the conditions for indeterminate lines of curvature are expressible as

=

ij

M
n

ijg

,

from  (i)

� (8)

� (9)

If  all  the  geodesics  of  a  hypersurface  Vn  are  also  geodesics  of  an  enveloping  Vn  + 1.  They

hypersurface Vn is called a totally geodesic hypersurface of the hypersurface Vn + 1.

THEOREM 12.7 A totally geodesic hypersurface is a minimal hypersurface and its-lines of curvature
are indeterminate.
Proof: We know that

2
aK =

2
K +
n K

2
g

and a hypersurface is said to be minimal if

M = 0

and the lines of curvature are indeterminate if

=

ij

M
n

ijg

� (1)

� (2)

� (3)

If a hypersurface Vn is totally geodesic then geodesics of Vn are also geodesics of Vn + 1.
i.e.,
Now, from (1),  we have

Ka = 0 = kg

Kn = 0

But normal curvature  Kn is zero for an asymptotic direction. Hence a hypersurface  Vn is  totally

geodesic hypersurface iff the normal curvature Kn zero for all directions in Vn and hence

= 0

ij

M =

ijg = 0

ij

i.e., equation (2) is satisfied.
Hence, the totally geodesic hypersurface is minimal hypersurface.
In this case equation (3) are satisfied hence the lines of curvature are indeterminate.

W
W
(cid:222)
W
W
W
W
Hypersurface

235

12.12 CENTRAL  QUADRATIC  HYPERSURFACE
Let xi be the cartesian in Euclidean space Sn,  so that the components gij of the fundamental tensor are
constants. Let yi be the Riemmannian coordinates. If a fixed point O is taken as a pole and s the distance
of any point P then Riemannian coordinates yi of P with pole O are given by
yi = sx i

� (1)

i is unit tangent in the direction of OP.

Let

ija  be the components in the x�s of a symmetric tensor of the rank two and evaluated at the

pole O. Then the equation

represents a central quadratic hypersurface

i
yay
ij

j

= 1

Substituting the value of equation (1) in equation (2), we get
i

i
sas
ij

= 1

i

ia x
ij

1
= 2
s

� (2)

� (3)

The equation (3) showing that the two values of s are equal in magnitude but opposite in sign.
The positive value of s given by equation (3) is the length of the radius of the quadric (2) for the

direction

i

.

THEOREM 12.8 The sum of the inverse squares of the radii of the quadric for n mutually orthogonal

directions at O is an invariant equal to

ij ga

ij

.

Proof: If

i
he | , (h = 1, 2, �n) are the contravariant components of the unit tangents at O to the curves

of an orthogonal ennuple in Sn. The radius Sh relative to the direction

i
he |  is given by

or

j
i
eea
h
h
ij

|

|

1
= 2
hs

n

=
1

h

) 2�

(

s

h

=

h

=

1

h

j
i
eea
|
h
ij
h

|

a

ij

=

n

=

1

h

i
ee
|
h

j
|
h

n

=
1

h

) 2�

(

s

h

ij

=

ij ga

Proved.

x
x
x
x
x
(cid:229)
(cid:229)
(cid:229)
(cid:229)
236

Tensors and Their Applications

THEOREM 12.9 The equation of hyperplane of contact of the tangent hypercone with vertex at the

point

( )yQ .

Proof: Given (from equation 2, pg. 235)

i
yya
ij

j

= 1

Differentiating it

a

ij

i
ydy

j

+

i
dyya
ij

or

a

ij

i
ydy

j

+

ya
ij

j

dy

a2
ij

i
ydy

a

ij

i
ydy

j

i

j

j

= 0

= 0

= 0

= 0

This shows that dyi is tangential to the quadric. Hence yj is normal to the quadric.
The tangent hyperplane at the point P ( y j) is given by

i

(
Y

�

i
)
yay
ij

i

yYa
ij

i

yYa
ij

j

j

j

= 0

=

i
yya
ij

j

= 1

since

i
yya
ij

j

 = 1

This equation represents the equation of tangent hyperplane at P ( y  i) .
If the tangent hyperplane P ( y j) passes through the point Q (y�i). Then we have

i
yay
ij

j

= 1

� (4)

� (5)

Thus all points of the hyperquadric, the tangent hyperplanes at which pass through Q lie on the
hyperplane  (5)  on  which  yi  is  the  current  point.  This  is  the  hyperplane  of  contact  of  the  tangent

hypercone whose vertex is

iyQ
(

)

.

12.13 POLAR  HYPERPLANE

The polar hyperplane of the point

)~(
iyR

 with respect to quadric (2) is the locus of the vertices of the

hypercones which touch the hyperquadric along its intersections with hyperplanes through R. If
is the vertex of such tangent hypercone, then R lies on the hyperplane of contact of Q so that

iyQ
(

)

i
yay
ij

j

~ = 1

Consequently for all positions of the hyperplane through R, Q  lies on the hyperplane

i
yay
ij

j

~ = 1

� (6)

This is required equations of the polar hyperplane of R and R is the pole of this hyperplane.

Hypersurface

237

12.14 EVOLUTE  OF  A  HYPERSURFACE  IN  EUCLIDEAN  SPACE
Consider a hypersurface  Vn of Euclidean space  Sn +  1 and let xi (i = 1, 2, � n) be coordinates of an
arbitrary point P of Vn whose components relative to Sn + 1 are
n

(
=a

,2,1

)1

+

y

Let Na  be a unit normal vector at P relative to Sn + 1 so that tensor derivate Na  becomes covariant

L

derivative.

So,

and

ayP
Let
(
of Na  such that

)

iN; =

iN,

 =

�

g

ij

jk

y
,

k

ijy; =

ijy,

 =

Nij

+
1

n

=a
1

gij =

yy
,
,
i

j

� (1)

� (2)

� (3)

 be a point on the unit normal Na  such that distance of  P from P is r

 in the direction

y =

y

r+ N

� (4)

Suppose P undergoes a displacement dxi in Vn then the corresponding displacement

yd

 of  P is

given by

ayd

=

(

y
,
i

r+

N

,
i

i

)

dx

+

dN

� (5)

The vector

(

r+

y
,
i

N

,

i

i

)

dx

 is tangential to Vn  whereas Na dr

 is a normal vector. Therefore if the

displacement of

ayP
(

)

 be along the normal to the hypersurface then we have

(

y
,
i

�

N

,

i

)0

dx

i

= 0

Using equation (1) in equation (6), we get

(

y
,
i

�

jk

g

y
,

k

i

)

dx

= 0

Multiplying it by

iy,

 and summing with respect to a

, we get, using equation (3), as

(

g

il

�

jk
gg

lk

ij

i
xd

)

= 0

(

g

il

�

j
l

)

i
xd

ij

= 0

(

g

il

�

)

i

dx

= 0

il

(

g

ij

�

)

ij

i
xd

= 0

� (6)

� (7)

where the roots r

This shows that the directions dxi given by equation (7) are principal directions of the hypersurface
 of the equation |gij � r
ij| = 0 are called principal radii of normal curvature.  The
 satisfying the condition (4) is called evolute of the hypersurface Vn of Sn + 1 where r

ayP
(

)

locus of
is a root of (7). The evolute is also a hypersurface of Sn + 1.

a
a
a
a
W
a
a
a
W
a
a
(cid:229)
a
a
a
a
r
a
a
a
a
a
a
a
r
a
a
r
a
W
r
d
W
r
W
r
W
r
W
238

Tensors and Their Applications

12.15 HYPERSPHERE
The locus of a point in Sn which moves in such way that it is always at a fixed distance R from a fixed
  is  called  a  hypersphere  of  radius  R  and  centre  C.  Therefore  the  equation  of  such  a
point
)
hypersphere is given by

( abC

n

1=a

(

y

�

b

2)

= R2

� (1)

THEOREM 12.10 The Riemannian curvatrure of a hypersphere of radius R is constant and equal to

1
2

.

R
Proof: Let  the  hypersurface  be  a  Vn of    Sn +  1  and  let  its  centre  be  taken  as  origin  of  Euclidean
coordinates in Sn + 1. Then the hypersphere is given by

(y

2)

=  R2,

(a

 = 1, 2, �, n + 1)

For the point in Vn the y's are functions of the coordinates xi on the hypersphere.
Differentiating equation (2) with respect to xi, we get

and again differentiating it with respect to x j, we get.

iyy

, = 0

y

,

j

y
,

i

+

yy
,

ij

= 0

By Gauss formula,

ijy, =
Using (5), equation (4) becomes

Nij

+

g

ij

y

N

ij

= 0

� (2)

� (3)

� (4)

� (5)

� (6)

From equation (3) it follows that

iy,

 is perpendicular to

y . But

iy,

 is tangential to Vm. Hence

y  is normal to Vn.. The equation (2) implies that the components of the unit vector

N  are given by

N =

y
R

y

=

RN

� (7)

Using (7), equation (6) becomes

+

g

ij

NNR

ij

= 0

g

ij

W+
R

ij

(N

2)

= 0

g

W+
ij R

= 0

ij

since

aN

(

2 =

)

1

or

or

(cid:229)
a
a
(cid:229)
a
a
(cid:229)
a
a
a
(cid:229)
(cid:229)
a
a
a
a
a
a
a
a
W
(cid:229)
a
a
a
W
a
a
a
a
a
a
a
a
a
(cid:222)
W
a
a
a
(cid:229)
(cid:229)
a
a
240

Tensors and Their Applications

2. To prove that for a space Vn with positive constant Riemannian curvature K these exists sets of n + 1 real

coordinate

y  satisfying the condition

+

1

n

=a
1

(

y

)

=

1
K

  where R2 =

1
K

Proof: Using (9) in equation (1), we get

(y

2)

=

1
K

EXAMPLE  2

 Proved.

Show that the directions of two lines of curvature at a point of a hypersurface are conjugate.

Solution

The principal direction

i
he |  are given by

j
i
ee
|
|
h
h

ij

= 0

� (1)

The shows that principal directions at a point of a hypersurface are conjugate.
Thus we say that two congruences of lines of curvature are conjugate.

EXAMPLE  3

Show that the normal curvature of hyper surface Vn  in an asymptotic direction vanishes.

Solution

Let  us  consider  a  curve  C in a  Vn. If  C  is  an  asymptotic  line  then  it  satisfies  the  differential

equation

i.e.,

i
dx

j

dx

ij

= 0

i
dx
ds

j

dx
ds

ij

= 0

� (1)

Now the normal curvature Kn of the hypersurface Vn in an asymptotic direction in a Vn is given

by

Kn =

ij

Kn = 0

j

i
dx
dx
ds
ds
from  (1)

i.e.,

EXAMPLE  4

To prove that if the polar hyperplane of the point R passes through a point P then that of P passes

through R.

Solution

Let

( iyP

)

 and

( iyR

)

 be two points. The polar hyperplane of

( iyR

)

 is

a
(cid:229)
a
(cid:229)
a
a
W
W
W
W
242

Tensors and Their Applications

11. Prove that the necessary and sufficient condition that system of hypersurface with unit normal N be

isothermic is that

(

N

div

NNN

�

.

)

= 0.

12. Show that the normal curvature of a subspace in an asymptotic direction is zero.
13. If straight line through a point P in Sn meets a hyperquadric in A and B and the polar hyperplane of

P in Q prove that P, Q are harmonic conjugates to A, B.

14. What are evolutes of a hypersurface in an Euclidean space. Show that the varieties  1
as orthogonal geodesics of V n.

evolute are parallel, having the curves of parameter  1

 = constant in

(cid:209)
r
r
r
INDEX

A

Absolute  131
Addition and Subtraction of Tensors  15
Associated tensor  43
Asymptotic directions  229

B

Bianchi identity  94
Binormal  137

C

Canonical congruence  213
Christoffel�s symbols  55
Completely skew-symmetric  111
Completely symmetric  111
Concept  188
Congruence of curves  49
Conjugate (or Reciprocal) Symmetric Tensor  25
Conjugate directions  229
Conjugate metric tensor  34
Conservative force field  144
Contraction  18
Contravariant tensor of rank r  14
Contravariant Tensor of rank two  9
Contravariant vector  7
Covariant tensor of rank s  14

Covariant tensor of rank two  9
Covariant vector  7
Curl  76
Curl of congruence  211
Curvature  136
Curvature of Congruence  207
Curvature tensor  86

D

Degree of freedom  157
Dextral Index  1
Divergence  75
Divergence Theorem  161
Dummy index  1
Dupin�s Theorem  226

E

Einstein space  103
Einstein tensor  95
Einstein�s Summation Convention  1
Euler�s  condition  171
Euler�s Theorem  228
Evolute  237

F

First curvature  227
First curvature vector of curve  170

244

Tensors and Their Applications

First fundamental tensor  34
Free Index  2
Fundamental tensor  31
Fundamental theorem  199

G

Gauss Characteristic equation  233
Gauss�s formula  222
Gauss�s theorem  164
Generalised Kr�necker delta  112
Generalized coordinates  157
Geodesic congruence  208
Geodesic coordinate system 175
Geodesics  171
Gradient  75
Green�s Theorem  162

H

Hamilton�s  principle  153
Hypersphere  238
Hypersurface  48, 218

I

Inner product of two tensors  18
Integral of energy  155
Intrinsic derivative  131

K

Kinetic energy  144
Kr�necker Delta  2

L

Lagrangean equation  148
Lagrangean function  148
Laplace�s equation  167
Laplacian operator  80, 163
Length of a curve  42
Levi-civita�s concept  188
Line element  31
Line of curvature  227

M

Magnitude of vector  44
Mainardi-Codazzi equations  233
Mean curvature  101
Meaunier�s Theorem  225
Metric Tensor  31
Minimal curve  42
Minimal Hypersurface  227, 234
mixed tensor of rank r + s  14
mixed tensor of rank two  9

N

Newtonian Laws  142
Normal congruence  209
n-ply orthogonal system of hypersurfaces  49
Null curve  42

O

Orthogonal Cartesian coordinates  120
Orthogonal ennuple  49
Osculating plane  136
Outer Product of Tensor  16

P

Parallel vector fields  134
Parallelism  188
Poisson�s equation  166
Polar hyperplane  236
Potential energy  145
Principal directions of the hypersurface  227
Principle normal vector  136
Principle normal curvatures  227
Principle of least action  156
Projective curvature tensor  104

Q

Quotient Law  24

R

Reciprocal base systems  122

Index

Relative Tensor  26
Ricci tensor  88
Ricci�s principal directions  102
Ricci�s coefficients of rotation  205
Ricci�s Theorem  71
Riemann curvature  96
Riemann�s symbol  86
Riemann-Christoffel Tensor  85
Riemannian coordinates  177
Riemannian Geometry  31, 116
Riemannian Metric  31
Riemannian space  31

S

Scalar product of two vectors  44
Schur�s theorem  100
Second fundamental tensors  34
Serret-Frenet formula  138
Simple Pendulum  152
Skew-Symmetric Tensor  20

245

Stoke�s Theorem  164
Straight line  140
Subscripts  1
Superscripts  1
Symmetric Tensors  20

T

Tensor density  26
Torsion  137
Totally geodesic hypersurface  234
Transformation of Coordinates  6

U

Umbral  1
Unit principal normal  171

W

Weyl tensor  104
Work function W  145


