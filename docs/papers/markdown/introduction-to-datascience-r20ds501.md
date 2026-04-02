MALLA REDDY COLLEGE OF ENGINEERING & TECHNOLOGY

Autonomous Institution - UGC, Govt. of India

APPROVED BY AICTE, ACCREDITED BY NBA & NAAC � 'A' GRADE ISO 9001:2015 CERTIFIED

B.TECH

     II � I

R20

2021-2022

Introduction to Data
Science
[R20DS501]

DIGITAL NOTES

 DEPARTMENT OF CSE (DATA SCIENCE, CYBER SECURITY, INTERNET OF THINGS)

Introduction to Datascience
             (R20DS501)

LECTURE NOTES

B.TECH II YEAR � I SEM (R20)

                       (2021-2022)

DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING
(DATA SCIENCE,CYBER SECURITY,INTERNET OF THINGS)

MALLA REDDY COLLEGE OF ENGINEERING & TECHNOLOGY

(Autonomous Institution � UGC, Govt. of India)
(Affiliated to JNTUH, Hyderabad, Approved by AICTE - Accredited by NBA & NAAC � �A� Grade - ISO 9001:2015 Certified)

Maisammaguda, Dhulapally (Post Via. Hakimpet), Secunderabad � 500100, Telangana State, INDIA.

MALLA REDDY COLLEGE OF ENGINEERING AND TECHNOLOGY

II Year B.Tech CSE(DS) � I Sem

(R20DS501) Introduction to Datascience

                                           L/T/P/C
           3  -/-/-3

COURSE OBJECTIVES:

1.  An understanding of the data operations
2.  An  overview  of  simple  statistical  models  and  the  basics  of  machine

learning techniques of regression.

3.  An understanding good practices of data science
4.  Skills in the use of tools such as python, IDE
5.  Understanding of the basics of the Supervised learning

UNIT-1
Introduction,  Toolboxes:  Python,  fundamental
libraries  for  data  Scientists.
Integrated development environment (IDE). Data operations: Reading, selecting,
filtering, manipulating, sorting, grouping, rearranging, ranking, and plotting.
UNIT-2
Descriptive  statistics,  data  preparation.  Exploratory  Data  Analysis  data
summarization,  data  distribution,  measuring  asymmetry.  Sample  and  estimated
mean,  variance  and  standard  score.  Statistical  Inference  frequency  approach,
variability  of  estimates,  hypothesis  testing  using  confidence  intervals,  using  p-
values
UNIT-3
Supervised  Learning:  First  step,  learning  curves,  training-validation  and  test.
Learning models generalities, support vector machines, random forest. Examples
UNIT-4
Regression  analysis,  Regression:
multiple  &  Polynomial  regression,  Sparse  model.  Unsupervised
clustering, similarity and distances, quality measures of clustering, case study.

linear  regression,
learning,

linear  regression  simple

UNIT-5
Network  Analysis,  Graphs,  Social  Networks,  centrality,  drawing  centrality  of
Graphs, PageRank, Ego-Networks, community Detection

TEXT/REFERENCES BOOK:
 1. Introduction to Data Science a Python approach to concepts, Techniques and
Applications, Igual, L;Seghi�, S. Springer, ISBN:978-3-319-50016-4
2. Data Analysis with Python A Modern Approach, David Taieb, Packt Publishing,
ISBN-9781789950069
3. Python Data Analysis, Second Ed., Armando Fandango, Packt Publishing, ISBN:
9781787127487
COURSE OUTCOMES:
1.  Describe what Data Science is and the skill sets needed to be a data scientist
2. Explain the significance of exploratory data analysis (EDA) in data science
3. Ability to learn the supervised learning, SVM
4. Apply basic machine learning algorithms (Linear Regression)
5. Explore the Networks, PageRank

UNIT NO

TOPIC

PAGE NO

I

II

III

IV

Introduction

Introduction, Toolboxes: Python, fundamental
libraries for data Scientists.
Integrated development environment
(IDE).
Data Operations

Descriptive statistics, data
preparation
Descriptive statistics

Exploratory Data Analysis data
summarization
Statistical Inference frequency
approach

Supervised Learning

Supervised Learning

Learning models generalities,
support vector machines
Regression analysis

Regression analysis,

5-10

10-16

17-29

30-33

33-49

50-60

61-75

76-95

linear regression simple linear
regression
Unsupervised learning

   96-112

   113-137

V

Network Analysis

Network Analysis,Graphs

community Detection

   139-159

   160-161

                            UNIT-1

Introduction,  Toolboxes:  Python,  fundamental  libraries  for  data  Scientists.
Integrated development environment (IDE). Data operations: Reading, selecting,
filtering, manipulating, sorting, grouping, rearranging, ranking, and plotting.
     Introduction to Data Science

1.1  What is Data Science?

You  have,  no  doubt,  already  experienced  data  science  in  several  forms.  When
you are looking for information on the web by using a search engine or asking
your  mobile phone  for  directions,  you  are  interacting  with  data  science
products. Data science has been behind resolving some of our most common
daily tasks for several years.

Most of the  scientific methods  that power data science  are not  new and
they have been out there, waiting for applications to be developed, for a long
time.  Statistics  is an  old  science  that  stands  on  the  shoulders  of  eighteenth-
century  giants  such  as  Pierre  Simon  Laplace  (1749�1827)  and  Thomas  Bayes
(1701�1761). Machine learning is younger, but it has already moved beyond
its  infancy  and  can  be  considered  a  well- established  discipline.  Computer
science changed our lives several decades ago and continues to do so; but it
cannot be considered new.

So,  why  is  data  science  seen  as  a  novel  trend  within  business  reviews,  in

technology blogs, and at academic conferences?

The  novelty  of  data  science  is  not  rooted  in  the  latest  scientific  knowledge,
but in a disruptive change in our society that has been caused by the evolution of
technology: datification. Datification is the process of rendering into data aspects
of the world that have never been quantified before. At the personal level, the
list of datified concepts is very long and still growing: business networks, the
lists of books we are reading, the films we enjoy, the food we eat, our physical
activity, our purchases, our driving behavior, and so on. Even our thoughts are
datified when we publish them on our favorite social network; and in a not so
distant  future,  your  gaze  could  be  datified  by wearable  vision  registering
devices.  At  the  business  level,  companies  are  datifying semi-structured  data
that were previously discarded: web activity logs, computer network activity,
machinery signals, etc. Nonstructured data, such as written reports, e-mails, or
voice recordings, are now being stored not only for archive purposes but also
to be analyzed.

However, datification is not the only ingredient of the data science revolution. The
other ingredient is the democratization of data analysis. Large companies such as
Google, Yahoo, IBM, or SAS were the only players in this field when data science
had no name. At the beginning of the century, the huge computational resources
of those  companies  allowed  them  to  take  advantage  of  datification  by  using
analytical techniques  to  develop  innovative  products  and  even  to  take  decisions
about  their  own  business.  Today,  the  analytical  gap  between  those  companies
and  the  rest  of  the  world  (companies  and  people)  is  shrinking.  Access  to  cloud
computing allows any individual to analyze huge amounts of data in short periods
of  time.  Analytical knowledge  is  free  and  most  of  the crucial  algorithms  that  are
needed to create a solution can  be found, because  open-source  development is  the
norm  in  this  field.  As a  result,  the  possibility  of  using  rich  data  to  take  evidence-
based decisions is open to virtually any person or company.

Data science is  commonly  defined  as  a  methodology by  which actionable insights
can be inferred from data. This is a subtle but important difference with respect
to  previous  approaches  to  data  analysis,  such  as  business  intelligence  or
exploratory  statistics.  Performing  data  science  is  a  task  with  an  ambitious
objective: the produc- tion of beliefs informed by data and to be used as the basis
of decision-making. In the absence of data, beliefs are uninformed and decisions,
in the best of cases, are based on best practices or intuition. The representation of
complex  environments  by rich  data  opens  up  the  possibility  of  applying  all  the
scientific knowledge we have regarding how to infer knowledge from data.

In general, data science allows us to adopt four different strategies to explore

the world using data:

1.  Probing reality. Data can be gathered by passive or by active methods. In the
latter case, data represents the response of the world to our actions. Analysis
of those  responses  can  be  extremely  valuable  when  it  comes  to  taking
decisions  about  our  subsequent  actions.  One  of  the  best  examples  of  this
strategy  is  the  use  of  A/B  testing  for  web  development:  What  is  the  best
button  size  and  color? The  best  answer  can  only  be  found  by  probing  the
world.

2.  Pattern discovery. Divide and conquer is an old heuristic used to solve complex
problems; but it is not always easy to decide how to apply this common sense
to problems.  Datified  problems  can  be  analyzed  automatically  to  discover
useful patterns and natural clusters that can greatly simplify their solutions.
The use of this technique to profile users is a critical ingredient today in such
important fields as programmatic advertising or digital marketing.

3.  Predicting future events. Since the early days of statistics, one of the most impor-
tant scientific questions has been how to build  robust data models that are
capa-  ble  of  predicting  future  data  samples.  Predictive  analytics  allows
decisions  to  be  taken  in  response  to  future  events,  not  only  reactively.  Of
course, it is not possible to predict the future in any environment and there will
always be unpre- dictable events; but the identification of predictable events
represents valuable knowledge. For example, predictive analytics can be used

to optimize the tasks

What is Data Science?

3

planned  for  retail  store  staff  during  the  following  week,  by  analyzing  data
such as weather, historic sales, traffic conditions, etc.

4.  Understanding  people  and  the  world.  This  is  an  objective  that  at  the
moment is  beyond  the  scope  of  most  companies  and  people,  but  large
companies and governments are investing considerable amounts of money in
research  areas  such  as  understanding  natural  language,  computer  vision,
psychology  and  neu- roscience.  Scientific  understanding  of  these  areas  is
important  for  data  science because  in  the  end,  in  order  to  take  optimal
decisions,  it  is  necessary  to  know  the  real  processes  that  drive  people�s
decisions  and  behavior.  The  development  of  deep  learning  methods  for
natural  language  understanding  and  for  visual  object recognition  is  a  good
example of this kind of research.

Toolboxes for Data Scientists

Introduction

In  this  chapter,  first  we  introduce  some  of  the  tools  that  data  scientists  use.  The
toolbox of  any  data  scientist,  as  for  any  kind  of  programmer,  is  an  essential
ingredient  for  success  and  enhanced  performance.  Choosing  the right  tools  can
save a lot of time and thereby allow us to focus on data analysis.

The most basic tool to decide on is which programming language we will use.
Many people use only one programming language in their entire life: the first and
only one they learn. For many, learning a new language is an enormous task that,
if at  all  possible,  should  be  undertaken  only  once.  The  problem  is  that  some
languages are  intended  for  developing  high-performance  or  production  code,
such  as  C,  C++, or  Java,  while  others  are  more  focused  on  prototyping  code,
among these the best known are the so-called scripting languages: Ruby, Perl, and
Python.  So,  depending on  the  first  language  you  learned,  certain  tasks  will,  at  the
very  least,  be  rather  tedious.  The  main  problem  of  being  stuck  with  a  single
language is that many basic tools simply will not be available in it, and eventually
you will have either to reimplement them or to create a bridge to use some other
language just for a specific task.

Toolboxes for Data Scientists

In conclusion, you either have to be ready to change to the best language for
each task  and  then glue  the results  together,  or  choose  a very  flexible  language
with  a  rich ecosystem  (e.g.,  third-party  open-source  libraries).  In  this  book  we
have selected Python as the programming language.

Why Python?

Python1  is  a  mature  programming  language  but  it  also  has  excellent  properties
for  newbie  programmers,  making  it  ideal  for  people  who  have  never  programmed
before. Some of the most remarkable of those properties are easy to read code,
suppression of non-mandatory delimiters, dynamic typing, and dynamic memory
usage. Python is an interpreted language, so the code is executed immediately in
the Python con- sole without needing the compilation step to machine language.
Besides  the Python console (which  comes  included  with  any  Python  installation)
you can find other in- teractive consoles, such as IPython,2 which give you a richer
environment in which to execute your Python code.

Currently, Python is one of the most flexible programming languages. One of
its main  characteristics  that  makes  it  so  flexible  is  that  it  can  be  seen  as  a
multiparadigm language.  This  is  especially  useful  for  people  who  already  know  how
to program with other languages, as they can rapidly start programming with Python
in  the  same  way.  For  example,  Java  programmers  will  feel  comfortable  using
Python  as  it  supports  the  object-oriented  paradigm,  or  C  programmers  could  mix
Python  and  C  code  using  cython.  Furthermore,  for  anyone  who  is  used  to
programming in functional languages such  as  Haskell  or  Lisp,  Python  also  has  basic
statements for functional programming in its own core library.

In  this  book,  we  have  decided  to  use  Python  language  because,  as  explained
before,  it  is  a mature language  programming,  easy  for  the  newbies,  and  can  be
used as  a  specific  platform  for  data  scientists,  thanks  to  its  large  ecosystem  of
scientific libraries and its high and vibrant community. Other popular alternatives
to Python for data scientists are R and MATLAB/Octave.

Fundamental Python Libraries for Data Scientists

The Python community is one of the most active programming communities with
a huge  number  of developed  toolboxes.  The most  popular Python  toolboxes  for
any data scientist are NumPy, SciPy, Pandas, and Scikit-Learn.

  Numeric and Scientific Computation: NumPy and SciPy

NumPy3 is the cornerstone toolbox for scientific computing with Python. NumPy
provides,  among  other  things,  support  for  multidimensional  arrays  with  basic
oper- ations on them and useful linear algebra functions. Many toolboxes use the
NumPy array representations as an efficient basic data structure. Meanwhile, SciPy
provides  a  collection  of  numerical  algorithms  and  domain-specific  toolboxes,
including signal processing, optimization, statistics, and much more. Another core
toolbox  in  SciPy is the plotting library Matplotlib. This toolbox has many tools for
data visualization.

SCIKIT-Learn: Machine Learning in Python

Scikit-learn4 is a machine learning library built from NumPy, SciPy, and Matplotlib.
Scikit-learn offers simple and efficient tools for common tasks in data analysis such
as  classification,  regression,  clustering,  dimensionality  reduction,  model
selection, and preprocessing.

PANDAS: Python Data Analysis Library

Pandas5  provides high-performance data structures and data analysis tools. The
key feature of Pandas is a fast and efficient DataFrame object for data manipulation
with integrated indexing. The DataFrame structure can be seen as a spreadsheet
which  offers  very  flexible  ways  of  working  with  it.  You  can  easily  transform  any
dataset in the way you want, by reshaping it and adding or removing columns or
rows.  It  also provides  high-performance  functions  for  aggregating, merging,  and
joining dataset-
s. Pandas also has tools for importing and exporting data from different formats:
comma-separated value (CSV), text files, Microsoft Excel, SQL databases, and the
fast HDF5 format. In many situations, the data you have in such formats will not
be complete or totally structured. For such cases, Pandas offers handling of miss-
ing  data  and  intelligent  data  alignment.  Furthermore,  Pandas  provides  a
convenient Matplotlib interface.

Data Science Ecosystem Installation

Before we can get started on solving our own data-oriented problems, we will need
to set  up our  programming  environment. The first  question  we need to answer
concerns

Toolboxes for Data Scientists

Python language itself. There are currently two different versions of Python: Python
2.X and Python 3.X. The differences between the versions are important, so there
is no  compatibility  between  the  codes,  i.e., code  written in  Python  2.X does  not
work in Python 3.X and vice versa. Python 3.X was introduced in late 2008; by then,
a lot of code and many toolboxes were already deployed using Python 2.X (Python
2.0 was initially introduced in 2000). Therefore, much of the scientific community
did not change to Python 3.0 immediately and they were stuck with Python 2.7. By
now, almost  all  libraries  have  been  ported  to  Python  3.0;  but  Python  2.7  is  still
maintained,  so  one  or  another  version  can  be  chosen.  However,  those  who
already  have  a  large amount  of code  in  2.X  rarely  change to  Python  3.X.  In  our
examples throughout this book we will use Python 2.7.

Once we have chosen one of the Python versions,  the next thing to decide is

whether  we  want  to  install  the  data  scientist  Python  ecosystem  by  individual
tool-  boxes,  or  to  perform  a  bundle  installation  with  all  the  needed  toolboxes
(and  a  lot  more).  For  newbies,  the  second  option  is  recommended.  If  the  first
option is chosen, then it is only necessary to install all the mentioned toolboxes in the
previous section, in exactly that order.

However,

installation

if  a  bundle

is  chosen,  the  Anaconda  Python
distribution6  is  then  a  good  option.  The  Anaconda  distribution  provides
integration  of  all  the  Python  toolboxes  and  applications  needed  for  data
scientists  into  a  single  directory without  mixing  it  with  other  Python  toolboxes
installed  on  the  machine.  It  contain-  s,  of  course,  the  core  toolboxes  and
applications  such  as  NumPy,  Pandas,  SciPy,  Matplotlib,  Scikit-learn,  IPython,
Spyder,  etc.,  but  also  more  specific  tools  for  other related  tasks  such  as  data
visualization, code optimization, and big data processing.

Integrated Development Environments (IDE)

For any programmer, and by extension, for any data scientist, the integrated de-
velopment environment (IDE) is an essential tool. IDEs are designed to maximize
programmer productivity. Thus, over the years this software has evolved in order
to make the coding task less complicated. Choosing the right IDE for each person
is  no  �one-size-fits-all�  programming
is  crucial  and,  unfortunately,  there
environment. The  best  solution  is  to  try  the  most  popular  IDEs  among  the
community and keep whichever fits better in each case.

In general, the basic pieces of any IDE are three: the editor, the compiler, (or
interpreter) and the debugger. Some IDEs can be used in multiple programming
languages,  provided  by  language-specific  plugins,  such  as Netbeans7 or Eclipse.8
Others are only specific for one language or even a specific programming task. In

                       Integrated Development Environments (IDE)

9

the  case  of  Python,  there  are  a large  number  of  specific  IDEs,  both  commercial
(PyCharm,9  WingIDE10  �)  and  open-source.  The  open-source  community  helps
IDEs  to  spring  up,  thus  anyone  can  customize  their  own  environment  and  share  it
with  the  rest  of  the  community.  For  example,  Spyder11  (Scientific  Python
Development  EnviRonment)  is  an  IDE  customized  with  the  task  of  the  data
scientist in mind.

Web Integrated Development Environment (WIDE): Jupyter

With the advent of web applications, a new generation of IDEs for interactive lan-
guages  such  as  Python  has  been  developed.  Starting  in  the  academia  and  e-
learning communities, web-based IDEs were developed considering how not only
your code but also all your environment and executions can be stored in a server.
One of the first applications of  this  kind  of  WIDE  was  developed  by  William Stein in
early  2005  using  Python  2.3  as  part  of  his  SageMath  mathematical  software.  In

SageMath,  a  server can be set up in a center, such as a university or school, and
then  students  can work  on  their  homework  either  in  the  classroom  or  at  home,
starting from exactly the same point they left off. Moreover, students can execute
all the previous steps over and over again, and then change some particular code
cell  (a  segment  of  the  docu-  ment  that  may  content  source  code  that  can  be
executed)  and  execute  the  operation again.  Teachers  can  also  have  access  to
student sessions and review the progress or results of their pupils.

Nowadays,  such  sessions are  called  notebooks  and  they  are  not  only  used in
classrooms  but  also  used  to  show  results  in  presentations  or  on  business
dashboards. The  recent  spread  of  such  notebooks  is  mainly  due  to  IPython.  Since
December  2011,  IPython  has  been  issued  as  a  browser  version  of  its  interactive
console,  called IPython notebook, which shows the Python execution results very
clearly  and  concisely  by  means  of  cells.  Cells  can  contain  content  other  than  code.
For  example,  markdown  (a  wiki  text  language)  cells  can  be  added  to  introduce
algorithms. It is also possible to insert Matplotlib graphics to illustrate examples or
even  web  pages.  Recently,  some  scientific  journals  have  started  to  accept
notebooks in  order to  show  experimental  results, complete with  their  code and
data  sources.  In  this  way,  experiments  can  become  completely  and  absolutely
replicable.

Since  the  project  has  grown  so  much,  IPython  notebook  has  been  separated
from IPython software and now it has become a part of a larger project: Jupyter12.
Jupyter (for  Julia,  Python  and  R)  aims  to  reuse  the  same  WIDE  for  all  these
interpreted  languages  and  not
IPython  notebooks  are
automatically  imported to  the  new  version  when  they  are  opened  with  the
Jupyter platform; but once they

just  Python.  All  old

Get Started with Python for Data Scientists

Throughout this book,  we  will  come across many  practical examples. In this chapter,
we  will  see  a  very  basic  example  to  help  get  started  with  a  data  science
ecosystem from scratch. To execute our examples, we will use Jupyter notebook,
although any other console or IDE can be used.

The Jupyter Notebook Environment

Once  all  the  ecosystem  is  fully  installed,  we  can  start  by  launching  the  Jupyter
notebook  platform.  This  can  be  done  directly  by  typing  the  following  command
on your terminal or command line: $ jupyter notebook

If we chose the bundle installation, we can start the Jupyter notebook platform by

clicking on the Jupyter Notebook icon installed by Anaconda in the start menu or
on the desktop.

The browser will immediately be launched displaying the Jupyter notebook home-
page,  whose  URL  is  http://localhost:8888/tree.  Note  that  a  special port  is  used;
by default it is 8888. As can be seen in Fig. 2.1, this initial page displays a tree view of
a  directory. If we use the command line, the root directory is the same directory
where we  launched  the  Jupyter  notebook.  Otherwise,  if  we  use  the  Anaconda
launcher,  the root  directory  is  the  current  user  directory.  Now,  to  start  a  new

New  Notebooks  Python 2

notebook, we only
need to press
the home page.

button at the top on the right of the

As can be seen  in  Fig. 2.2,  a  blank  notebook  is  created  called  Untitled. First of
all,  we  are  going  to  change  the  name  of  the  notebook  to  something more
appropriate.  To  do  this,  just  click  on  the  notebook  name  and  rename  it:
DataScience-GetStartedExample.

Let us begin by importing those toolboxes that we will need for our program. In the
first cell we put the code to import the Pandas library as pd. This is for convenience;
every  time  we  need  to  use  some  functionality  from  the  Pandas  library,  we  will
write pd  instead  of  pandas.  We  will  also  import  the  two  core  libraries  mentioned
above: the numpy library as np and the matplotlib library as plt.

In []:

import  pan das  as  pd
import   numpy  as  np
i m p o r t   m a t p l o t l i b  . p y p l o t   as  plt

  Get Started with Python for Data Scientists

11

Fig. 2.1

IPython  notebook  home page, displaying a home tree  directory

Fig. 2.2  An empty new notebook

To execute just one cell, we press the �  button or click on

or press
the keys  Ctrl + Enter . While execution is underway, the header of the cell shows the
* mark:

Cell  Run

In [*]:

import  pan das  as  pd
import   numpy  as  np
i m p o r t   m a t p l o t l i b  . p y p l o t   as  plt

12

2  Toolboxes for Data Scientists

While a cell is being executed, no other cell can be executed. If you try to
execute another cell, its execution will not start until the first cell has finished its
execution. Once the execution is finished, the header of the cell will be replaced
by the next number of execution. Since this will be the first cell executed, the
number shown will
be 1. If the process of importing the libraries is correct, no output cell is produced.

In [1]:

import  pan das  as  pd
import   numpy  as  np
i m p o r t   m a t p l o t l i b  . p y p l o t   as  plt

For simplicity, other chapters in this book will avoid writing these imports.

The DataFrame Data Structure

The  key  data structure in Pandas is the DataFrame  object.  A DataFrame is  basically a
tabular data structure,  with  rows  and  columns.  Rows have  a specific index to  access
them, which can be any name or value. In Pandas, the columns are called Series, a
special  type  of data,  which  in  essence  consists  of a list  of several  values,  where
each value has an index. Therefore, the DataFrame data structure can be seen as
a  spreadsheet,  but it is much  more  flexible.  To  understand  how it works,  let  us
see  how to create a DataFrame from a common Python dictionary of lists. First,
we will
create a new cell by clicking
Then, we write in the following code:

or pressing the keys  Ctrl +

Insert Cell Below

Insert

B

.

In [2]:

data  =  { � year �:  [

2010 ,  2011 ,  2012 ,
2010 ,  2011 ,  2012 ,
2010 ,  2011 ,  2012

],
� team �:  [

� FCBar celon a  �,  � FCBar celo na  �,
� FC Bar celo na  �,  � RMa drid  �,
� RMadrid  �,  � RMadrid  �,
� Valen ciaCF  �,  � Vale nciaC F �,
� Vale ncia CF  �

],
� wins �:
� draws �:  [6 ,  7,  4,  5,  4,  7,  8,  10 ,  8] ,
� losses �:  [2 ,  3,  2,  4,  2,  5,  9,  11 ,  11 ]
}

[30 ,  28,  32,  29,  32,  26,  21,  17,  19] ,

foot bal l  =  pd . Da taF ram e  ( data ,  columns   =  [

� year �,  � team �,  � wins �,  � draws �,  � losses �
]

)

In this example, we use the pandas DataFrame object constructor with a dictionary
of lists as argument. The value of each entry in the dictionary is the name of the
column, and the lists are their values.

The  DataFrame  columns  can  be  arranged  at  construction  time  by  entering  a
key- word  columns with  a  list  of  the  names  of  the  columns  ordered  as  we  want.  If
the

with Python for Data Scientists

Get Started

13

column keyword is not present in the constructor, the columns will be arranged
in  alphabetical order. Now, if we execute this cell, the result will be a table like
this:

Out[2]:

year

team
0  2010  FCBarcelona
1  2011  FCBarcelona
2  2012  FCBarcelona
3  2010  RMadrid
4  2011  RMadrid
5  2012  RMadrid
6  2010  ValenciaCF
7  2011  ValenciaCF
8  2012  ValenciaCF

wins  draws
6
30
7
28
4
32
5
29
4
32
7
26
8
21
10
17
8
19

losses
2
3
2
4
2
5
9
11
11

where each entry in the dictionary is a column. The index of each row is created
automatically taking the position of its elements inside the entry lists, starting from 0.
Although  it  is  very  easy  to  create  DataFrames  from  scratch,  most  of  the  time
what we will need to do is import chunks of data into a DataFrame structure, and
we will see how to do this in later examples.

Apart  from  DataFrame  data  structure  creation,  Panda  offers  a  lot  of
functions to  manipulate  them.  Among  other  things,  it  offers  us  functions  for
aggregation,  manipulation,  and  transformation  of  the  data.  In  the  following
sections, we will introduce some of these functions.

Open Government Data Analysis Example Using Pandas

To illustrate how we can use Pandas in a simple real problem, we will start doing
some  basic  analysis  of  government  data.  For  the  sake  of  transparency,  data
produced by government entities must be open, meaning that they can be freely
used, reused, and distributed by anyone. An example of this is the Eurostat, which
is the home of European Commission data. Eurostat�s main role is to process and
publish  compa- rable  statistical  information  at  the  European  level.  The  data  in
Eurostat are provided by each member state and it is free to reuse them, for both
noncommercial and commercial purposes (with some minor exceptions).

Since  the  amount  of  data  in  the Eurostat  database  is  huge,  in  our  first  study
we  are  only  going  to  focus  on  data relative  to indicators  of  educational  funding
by  the  member  states.  Thus,  the  first  thing  to  do  is  to  retrieve  such  data  from
Eurostat. Since open data have to be delivered in a plain text format, CSV (or any
other  delimiter-separated  value)  formats  are  commonly  used  to  store  tabular
data.  In  a  delimiter-separated  value  file,  each  line  is  a  data  record  and  each
record  consist- s  of  one  or  more  fields,  separated  by  the  delimiter  character
(usually  a  comma).  Therefore,  the  data  we  will  use  can  be  found  already
processed  at  book�s  Github  repository  as  educ_figdp_1_Data.csv  file.  Of  course,  it
can also be download- ed as unprocessed tabular data from the Eurostat database
site13 following the path:

In [1]:

Out[1]:

14

2  Toolboxes for Data Scientists

Tables by themes  Population and social conditions  Education and training  Education
Indicators on education finance  Public expenditure on education .

Reading

Let us start reading the data we downloaded. First of all, we have to create a new
notebook  called  Open  Government  Data  Analysis  and  open  it.  Then,  after  ensuring
that  the  educ_figdp_1_Data.csv file  is  stored  in  the  same  directory as our notebook
directory, we will write the following code to read and show the content:

edu  =  pd . read_csv ( � files / ch02 / e d u c_ fi gd p_ 1_ Da ta  . csv �,
na_va lue s  =  �: �,
usecols  =  [" TIME " ," GEO " ," Value " ])

edu

TIME GEO

Value

0  2000 European Union ... NaN
1  2001 European Union ... NaN
2  2002 European Union ... 5.00
3  2003 European Union ... 5.03
... ...  ...
382 2010 Finland
383 2011 Finland
384 rows �  5 columns

...
6.85
6.76

The  way  to  read  CSV  (or  any  other  separated  value,  providing  the  separator
character)  files  in  Pandas  is  by  calling  the  read_csv method.  Besides  the  name of
the file, we add the na_values key argument to this method along with the character
that represents �non available data� in the file. Normally, CSV files have a header
with  the  names  of  the  columns.  If  this  is  the  case,  we  can  use  the  usecols
parameter to select which columns in the file will be used.

In this case, the DataFrame resulting from reading our data is stored in edu. The
output  of  the  execution  shows  that  the  edu  DataFrame  size  is  384  rows  3  columns.
Since the DataFrame is too large to be fully displayed, three dots appear in the middle
of each row.

�

Beside this, Pandas also has functions for reading files with formats such as Excel,
HDF5,  tabulated  files,  or  even  the  content  from  the  clipboard  (read_excel(),
read_hdf(),  read_table(),  read_clipboard()).  Whichever  function  we  use,  the
result of reading a file is stored as a DataFrame structure.

To  see  how  the  data  looks,  we  can  use  the  head()  method,  which  shows  just  the
first five rows. If we use a number as an argument to this method, this will be the
number of rows that will be listed:

.

2.6    Get Started with Python for Data Scientists

15

In [2]:

edu . h e a d  ()

Out[2]:

Value

  TIME GEO
0 2000 European Union ... NaN
1 2001 European Union ... NaN
2 2002 European Union ... 5.00
3 2003 European Union ... 5.03
4 2004 European Union ... 4.95

Similarly, it exists the tail() method, which returns the last five rows by default.

In [3]:

edu . t a i l  ()

Out[3]:

379 2007 Finland 5.90
380 2008 Finland 6.10
381 2009 Finland 6.81
382 2010 Finland 6.85
383 2011 Finland 6.76

If we want to know the names of the columns or the names of the indexes, we
can  use the DataFrame  attributes  columns  and index  respectively. The names of the
columns or indexes can be changed by assigning a new list of the same length to
these attributes. The values of any DataFrame can be retrieved as a Python array
by calling its values attribute.

If  we  just  want  quick  statistical  information  on  all  the  numeric  columns  in  a
DataFrame, we can use the function  describe(). The result shows  the count, the
mean, the standard deviation, the minimum and maximum, and the percentiles,
by default, the 25th, 50th, and 75th, for all the values in each column or series.

In [4]:

edu . des cr ibe ()

Out[4]:

TIME

Value

count 384.000000  361.000000
mean  2005.500000 5.203989
1.021694
std  3.456556
min  2000.000000 2.880000
25%  2002.750000 4.620000
50%  2005.500000 5.060000
75%  2008.250000 5.660000
max  2011.000000 8.810000

Name: Value, dtype: float64

16

2  Toolboxes for Data Scientists

Selecting Data

If we want to select a subset of data from a DataFrame, it is necessary to indicate this
subset  using  square  brackets  ([ ])  after  the  DataFrame.  The  subset  can  be  specified
in several ways. If we want to select only one column from a DataFrame, we only
need  to  put  its  name  between  the  square  brackets.  The  result  will  be  a  Series
data structure, not a DataFrame, because only one column is retrieved.

In [5]:

edu [ � V a l u e  �]

Out[5]:  0

NaN
NaN
5.00
5.03
4.95

1
2
3
4
... ... 380 6.10
381 6.81
382 6.85
383 6.76
Name: Value, dtype: float64

If we want to select a subset of rows from a DataFrame, we can do so by indicating
a range of rows separated by a colon (:) inside the square brackets. This is commonly
known as a slice of rows:

In [6]:

edu [10:1 4]

Out[6]:

TIME GEO

Value

10 2010 European Union (28 countries) 5.41
11 2011 European Union (28 countries) 5.25
12 2000 European Union (27 countries) 4.91
13 2001 European Union (27 countries) 4.99

This  instruction returns  the  slice  of  rows  from  the  10th  to  the  13th  position.
Note that  the  slice  does  not  use  the  index  labels  as  references,  but  the  position.  In
this case, the labels of the rows simply coincide with the position of the rows.

If  we  want  to  select  a  subset  of  columns  and  rows  using  the  labels  as  our

references instead of the positions, we can use ix indexing:

In [7]:

edu . ix [ 9 0: 9 4 ,  [ � T I M E  �, �  GEO � ] ]

2.6    Get Started with Python for Data Scientists

17

Out[7]:

TIME  GEO
90  2006  Belgium
91  2007  Belgium
92  2008  Belgium
93  2009  Belgium
94  2010  Belgium

In [8]:

Out[8]:

This returns all the rows between the indexes specified in the slice before the
comma, and the columns specified as a list after the comma. In this case, ix references
the  index  labels,  which  means  that  ix  does  not  return  the  90th  to  94th  rows,  but  it
returns all the rows between the row labeled 90 and the row labeled 94; thus if
the index 100 is placed between the rows labeled as 90 and 94, this row would
also be returned.

Filtering Data

Another way to select a subset of data is by applying Boolean indexing. This indexing
is commonly known as a filter. For instance, if we want to filter those values less
than or equal to 6.5, we can do it like this:

edu [ edu  [ � V a l u e  �]  >  6 . 5 ] .  t a il  ()

TIME  GEO

218  2002  Cyprus
281  2005  Malta
94
93
95

2010  Belgium
2009  Belgium
2011  Belgium

Value
6.60
6.58
6.58
6.57
6.55

Boolean  indexing  uses  the  result  of  a  Boolean  operation  over  the  data,
returning a  mask  with  True  or  False  for  each  row.  The  rows  marked  True  in  the
mask  will  be  selected.  In  the  previous  example,  the  Boolean  operation
edu[�Value�] >produces a Boolean mask. When an element in the �Value� column
is greater than 6.5, the corresponding value in the mask is set to True, otherwise
it is set to False. Then, when this mask is applied as an index in edu[edu[�Value�] >
6.5],  the  result  is  a  filtered  DataFrame  containing  only  rows  with  values  higher
than 6.5. Of course, any of the usual Boolean operators can be used for filtering:
< (less than),<= (less than or equal to), > (greater than), >= (greater than or equal
to), = (equal to),  and  ! =  (not  equal  to).

Filtering Missing Values

Pandasuses  the  special  value  NaN (not  a  number)  to  represent  missing  values.  In Python,  NaN is  a  special  floating-point  value  returned  by  certain  operations  when

18

2  Toolboxes for Data Scientists

Table 2.1  List of most common aggregation functions

Function

count()

sum()

mean()

median()

min()

max()

prod()

std()

var()

Description

Number of non-null observations

Sum of values

Mean of values

Arithmetic median of values

Minimum

Maximum

Product of values

Unbiased standard deviation

Unbiased variance

one of their results ends in an undefined value. A subtle feature of NaN values is that
two NaN are never equal. Because of this, the only safe way to tell whether a value is
missing in a DataFrame is by using the  isnull() function. Indeed, this function can be
used to filter rows with missing values:

In [9]:

edu [ edu  [" V a l u e  " ]. i s n u l l  () ]. h e a d ()

Out[9]:

TIME  GEO

0  2000  European Union (28 countries)
1  2001  European Union (28 countries)
36  2000  Euro area (18 countries)
37  2001  Euro area (18 countries)
48  2000  Euro area (17 countries)

Value
NaN
NaN
NaN
NaN
NaN

  Manipulating Data

Once we know how to select the desired data, the next thing we need to know is
how to manipulate data. One of the most straightforward things we can do is to
operate with  columns  or rows  using  aggregation functions. Table  2.1 shows  a list  of
the most common aggregation functions. The result of all these functions applied
to  a  row  or column is  always  a  number. Meanwhile, if  a  function is  applied  to  a
DataFrame or a selection  of  rows  and  columns,  then  you  can  specify  if  the  function
should  be  applied to the rows for each column (setting the axis=0 keyword on the
invocation  of  the function), or it should be applied on the columns for each row
(setting the axis=1 keyword on the invocation of the function).

edu . m ax ( a x is   =  0)

In [10]:

2.6    Get Started with Python for Data Scientists

19

Out[10]: TIME

GEO
Value
dtype: object

2011

Spain

8.81

Note  that  these  are  functions  specific  to  Pandas,  not  the  generic  Python
functions. There  are  differences  in  their  implementation.  In  Python,  NaN  values
propagate  through  all  operations  without  raising  an  exception.  In  contrast,
Pandas operations exclude  NaN values  representing  missing  data.  For  example,  the
pandas max function excludes NaN values, thus they are interpreted as missing values,
while the standard Python max function will take the mathematical interpretation of
NaN and return it as the maximum:

In [11]:

print  " Pandas  max  function :" ,  edu [ � Value �]. max ()
print  " Python  max  function :" ,  max ( edu [ � Value �])

Out[11]: Pandas  max  function:  8.81 Python max function:

nan

Beside these aggregation functions, we can apply operations over all the values in
rows,  columns  or  a  selection  of  both.  The  rule  of  thumb  is  that  an  operation
between columns  means  that  it  is  applied  to  each  row  in  that  column  and  an
operation  between rows means that it is applied to each column in that row. For
example we can apply any binary arithmetical operation (+,-,*,/) to an entire row:

s  =  edu  [" V a l u e  " ] / 1 0 0
s. h e a d  ()

NaN
NaN

1
2
3
4
Name: Value, dtype: float64

0.0500
0.0503
0.0495

However, we can apply any function to a DataFrame or Series just setting its name
as  argument  of  the  apply method.  For  example,  in  the  following  code,  we  apply
the  sqrt function  from  the  NumPy  library  to  perform  the  square  root  of  each  value
in the Value column.

s  =  edu [" V a l u e  " ]. a p p l y  ( np . s q r t  )
s. h e a d  ()

NaN
NaN

1
2
3
4
Name: Value, dtype: float64

2.236068
2.242766
2.224860

In [12]:

Out[12]: 0

In [13]:

Out[13]: 0

20

2  Toolboxes for Data Scientists

If we need to design a specific function to apply it, we can write an in-line function,
commonly known as a ?-function. A ?-function is a function without a name. It is
only  necessary  to  specify  the  parameters  it  receives,  between  the  lambda  keyword
and the colon (:). In the next example, only one parameter is needed, which will
be the value of each element in the  Value column. The value the function returns will
be the square of that value.

s  =  edu  [" V a l u e  " ]. a p p l y  ( l a m b d a   d:  d * * 2)
s. head ()

NaN
NaN

1
2
3
4
Name: Value, dtype: float64

25.0000
25.3009
24.5025

In [14]:

Out[14]: 0

Another basic manipulation operation is to set new values in our DataFrame. This
can be done directly using the assign operator (=) over a DataFrame. For example, to
add  a  new  column  to  a  DataFrame,  we  can  assign  a  Series  to  a  selection  of  a
column that  does  not  exist.  This  will  produce  a  new  column  in  the  DataFrame
after  all  the  others.  You  must  be  aware  that  if  a  column  with  the  same  name
already exists, the previous values will be overwritten. In the following example,
we  assign  the  Series that  results  from  dividing  the  Value  column  by  the  maximum
value in the same column to a new column named ValueNorm.

In [15]:

edu [ � Valu eNo rm �] = edu [ � Value �]/ edu [ � Value �]. max ()
edu . tail ()

Out[15]:

TIME GEO

Value ValueNorm

379 2007 Finland 5.90  0.669694
380 2008 Finland 6.10  0.692395
381 2009 Finland 6.81  0.772985
382 2010 Finland 6.85  0.777526
383 2011 Finland 6.76  0.767310

Now, if we want to remove this column from the DataFrame, we can use the drop
function;  this  removes  the  indicated  rows  if  axis=0,  or  the  indicated  columns  if
axis=1. In Pandas, all the functions that change the contents of a DataFrame, such
as the drop function, will normally return a copy of the modified data, instead of
overwriting the DataFrame. Therefore, the original DataFrame is kept. If you do
not want to keep the old values, you can set the keyword inplace to True. By default,
this keyword is set to False, meaning that a copy of the data is returned.

In [16]:

edu . drop ( � Va lue Norm �,  axis  =  1 ,  inplace  =  True )
edu . head ()

2.6    Get Started with Python for Data Scientists

21

Out[16]:

  TIME GEO
0 2000 European Union (28 countries) NaN
1 2001 European Union (28 countries) NaN
2 2002 European Union (28 countries) 5
3 2003 European Union (28 countries) 5.03
4 2004 European Union (28 countries) 4.95

Value

Instead,  if  what  we  want  to  do  is  to  insert  a  new  row  at  the  bottom  of  the
DataFrame, we  can  use  the  Pandas  append function.  This  function  receives  as
argument the  new  row,  which  is  represented  as  a  dictionary  where  the  keys
are  the  name of the  columns  and the values  are the  associated value. You  must  be
aware  to  setting  the  ignore_index flag  in  the  append method  to  True,  otherwise
the  index  0 is  given  to  this  new  row,  which  will  produce  an  error  if  it  already
exists:

edu  =  edu . append ({ " TIME ":  2000 , " Value ":  5.00 , " GEO ":  �a �},

edu . tail ()

ignor e_i ndex   =  True )

In [17]:

Out[17]:

TIME GEO

Value

380 2008 Finland 6.1
381 2009 Finland 6.81
382 2010 Finland 6.85
383 2011 Finland 6.76
384 2000 a

5

Finally,  if  we  want  to  remove  this  row,  we  need  to  use  the  drop  function  again.
Now  we  have  to  set  the  axis  to  0,  and  specify  the  index  of  the  row  we  want  to
remove. Since we want to remove the last row, we can use the max function over
the indexes to determine which row is.

In [18]:

edu . drop ( max ( edu . index ) ,  axis  =  0 ,  inplace  =  True )
edu . tail ()

Out[18]:

TIME GEO

Value

379 2007 Finland 5.9
380 2008 Finland 6.1
381 2009 Finland 6.81
382 2010 Finland 6.85
383 2011 Finland 6.76

The  drop() function  is  also  used  to  remove  missing  values  by  applying  it  over the
result  of the isnull() function. This has  a similar effect to filtering  the  NaN values, as
we explained above, but here the difference is that a copy of the DataFrame without
the NaN values is returned, instead of a view.

In [19]:

eduDrop  =  edu . drop ( edu [" Value " ]. isn ull () ,  axis  =  0)
eduDrop . head ()

22

2  Toolboxes for Data Scientists

Out[19]:

  TIME GEO
2 2002 European Union (28 countries) 5.00
3 2003 European Union (28 countries) 5.03
4 2004 European Union (28 countries) 4.95
5 2005 European Union (28 countries) 4.92
6 2006 European Union (28 countries) 4.91

Value

To  remove  NaN values,  instead  of  the  generic  drop  function,  we  can  use  the
specific dropna() function. If we want to erase any row that contains an NaN value, we
have  to  set  the  how  keyword  to  any.  To  restrict  it  to  a  subset  of  columns,  we  can
specify it using the subset keyword. As we can see below, the result will be the same
as using the drop function:

In [20]:

Out[20]:

eduD rop  = edu . d rop na ( how =  � any �, su bse t  =  [" Value " ])
eduDrop . head ()

  TIME GEO
2 2002 European Union (28 countries) 5.00
3 2003 European Union (28 countries) 5.03
4 2004 European Union (28 countries) 4.95
5 2005 European Union (28 countries) 4.92
6 2006 European Union (28 countries) 4.91

Value

If, instead of removing the rows containing NaN, we want to fill them with another
value, then we can use the  fillna() method, specifying which value has to be used. If
we  want  to  fill  only  some  specific  columns,  we  have  to  set  as  argument  to  the
fillna()  function  a  dictionary  with  the  name  of  the  columns  as  the  key  and  which
character to be used for filling as the value.

In [21]:

eduFi lle d  =  edu . fillna ( value  =  {" Value ":  0})
eduFi lle d . head ()

Out[21]:

TIME  GEO

0  2000  European Union (28 countries)
1  2001  European Union (28 countries)
2  2002  European Union (28 countries)
3  2003  European Union (28 countries)
4  2004  European Union (28 countries)

Value
0.00
0.00
5.00
4.95
4.95

Sorting

Another important functionality we will need when inspecting our data is to sort
by columns.  We  can  sort  a  DataFrame  using  any  column,  using  the  sort function.  If
we want to see the first five rows of data sorted in descending order (i.e., from
the largest to the smallest values) and using the Value column, then we just need to
do this:

2.6    Get Started with Python for Data Scientists

23

In [22]:

edu . sort _va lues ( by  =  � Value �,  asc endi ng  =  False ,

edu . head ()

inpl ace  =  True )

Out[22]:

TIME GEO

Value

130 2010 Denmark 8.81
131 2011 Denmark 8.75
129 2009 Denmark 8.74
121 2001 Denmark 8.44
122 2002 Denmark 8.44

Note that the inplace keyword means that the DataFrame will be overwritten, and
hence  no  new  DataFrame  is  returned.  If  instead  of  ascending  =  False  we  use
ascending = True, the values are sorted in ascending order (i.e., from the smallest
to the largest values).

If  we  want  to  return  to  the  original  order,  we  can  sort  by  an  index  using  the

sort_index function  and  specifying  axis=0:

In [23]:

edu . sort_i ndex ( axis  =  0 ,  asce ndi ng  =  True ,  i nplace  =  True )
edu . head ()

Out[23]:

TIME  GEO

0  2000  European Union ...
1  2001  European Union ...
2  2002  European Union ...
3  2003  European Union ...
4  2004  European Union ...

Value
NaN
NaN
5.00
5.03
4.95

Grouping Data

Another very useful way to inspect data is to group it according to some criteria.
For instance,  in  our  example  it  would  be  nice  to  group  all  the  data  by  country,
regardless of the year. Pandas  has the groupby function that  allows us to do  exactly
this. The value returned by this function is a special grouped DataFrame. To have
a proper DataFrame as a result, it is necessary to apply an aggregation function.
Thus, this function will be applied to all the values in the same group.

For  example,  in  our  case,  if  we  want  a  DataFrame  showing  the mean  of  the
values for each country  over  all the years,  we can  obtain it by grouping  according to
country and using the mean function as the aggregation method for each group.
The result would be a DataFrame with countries as indexes and the mean values as
the column:

group =  edu [[ " GEO " ,  " Value " ]]. groupby ( � GEO �). mean ()
group . head ()

In [24]:

24

2  Toolboxes for Data Scientists

Out[24]:

Value

GEO
Austria
Belgium
Bulgaria
Cyprus
Czech Republic

5.618333
6.189091
4.093333
7.023333
4.16833
Rearranging Data

Up  until  now,  our  indexes  have  been  just  a  numeration  of  rows  without  much
meaning. We can transform the arrangement of our data, redistributing the indexes
and columns for better manipulation of our data, which normally leads to better
performance. We can  rearrange  our  data  using  the pivot_table function.  Here,  we
can  specify which columns will be the new indexes, the new values, and the new
columns.

For  example,  imagine  that  we  want  to  transform  our  DataFrame  to  a
spreadsheet- like  structure  with  the  country  names  as  the  index,  while  the
columns  will  be  the  years  starting  from  2006  and  the  values  will  be  the  previous
Value  column.  To  do  this, first  we need to  filter  out the data  and  then pivot it  in
this way:

f i l t er e d _d a t a   =  edu [ edu [" TIME "]  >  2005]
pivedu  =  pd . pivot_ tabl e ( fil ter ed_d ata ,  valu es  =  � Value �,

In [25]:

pive du . head ()

index  =  [ � GEO �],
columns  =  [ � TIME �])

Out[25]:

2006 2007 2008 2009 2010 2011

TIME
GEO
5.40 5.33 5.47 5.98 5.91 5.80
Austria
5.98 6.00 6.43 6.57 6.58 6.55
Belgium
4.04 3.88 4.44 4.58 4.10 3.82
Bulgaria
Cyprus
7.02 6.95 7.45 7.98 7.92 7.87
Czech Republic 4.42 4.05 3.92 4.36 4.25 4.51

Now  we can use  the  new index  to  select  specific rows  by label,  using the ix

operator:

In [26]:

pivedu . ix [[ � Spain �,� Portugal �],  [2006 ,2011]]

Out[26]:

TIME
GEO
Spain
Portugal

2006  2011

4.26
5.07

4.82
5.27

Pivot also offers the option of providing an argument aggr_function that allows us

to perform an aggregation function between the values if there is more

2.6    Get Started with Python for Data Scientists

25

than one value for the given row and column after the transformation. As usual,
you can design any custom function you want,  just giving its name  or using a  ?-
function.

Ranking Data

Another useful visualization feature is to rank data. For example, we would like to
know  how  each  country  is  ranked  by  year.  To  see  this,  we  will  use  the  pandas  rank
function. But first, we need to clean up our previous pivoted table a bit so that it
only has  real  countries  with  real  data.  To  do  this,  first  we  drop  the  Euro  area
entries and shorten  the  Germany  name  entry,  using  the  rename function  and  then
we  drop  all the rows containing any NaN, using the dropna function.

Now  we  can  perform  the  ranking  using  the  rank  function.  Note  here  that  the
parameter  ascending=False makes  the  ranking  go  from  the  highest  values  to the
lowest  values.  The  Pandas  rank  function  supports  different  tie-breaking  methods,
specified with the method parameter. In our case, we use the first method, in which
ranks  are  assigned  in  the  order  they  appear  in  the  array,  avoiding  gaps  between
ranking.

In [27]:

pi v ed u  =  pi v edu . drop ([

� Euro  area  ( 13  countri es  ) �, � Euro  area  (
15  countrie s  ) �, � Euro  area  ( 17  countri es
) �, � Euro  area  ( 18  countri es  ) �,
� Europ ean  Union
Union
coun tri es ) �

( 27  countri es ) �, � Eur opea n  Union

( 25  cou ntri es  ) �, � Europ ean
( 28

],
axis  =  0)

pi v ed u  = p i ved u . r en a m e ( in dex   = { � G er m an y  ( until  1990  f or m er  t err it or y  of  the  FRG ) �:  � Germany �})
pi v ed u   =  pi v ed u . dro p na ()
pi v ed u . rank ( as c en d i ng   =  False ,  m et ho d  =

� first �). head ()

Out[27]:

2006 2007 2008 2009 2010 2011

TIME
GEO
Austria
Belgium
Bulgaria
Cyprus
2
Czech Republic 19  20  21  21  20  18

10  7
5
4
21  21  20  20  22  21
2

11  7
4
3

8
5

8
5

2

3

2

2

If we want to make a global ranking taking into account all the years,  we can
sum up all the columns and rank the result. Then we can sort the resulting values
to retrieve the top five countries for the last 6 years, in this way:

In [28]:

totalSum  =  piv edu . sum ( axis  =  1)
totalSum . rank ( a scen din g  =  False ,  method  =  � dense �)

. sort_ valu es () . head ()

2  Toolboxes for Data Scientists

26

Out[28]: GEO

Denmark
Cyprus
Finland
Malta
Belgium
dtype: float64

1
2
3
4
5

Notice  that  the  method  keyword  argument  in  the  in  the  rank  function  specifies
how items that compare equals receive ranking. In the case of dense, items that
compare  equals  receive  the  same  ranking  number,  and  the  next  not  equal  item
receives the immediately following ranking number.

Plotting

Pandas DataFrames and Series can be plotted using the  plot function, which uses the
library for graphics Matplotlib. For example, if we want to plot the accumulated
values for each country over the last 6 years, we can take the Series obtained in
the previous example  and  plot it  directly  by calling the  plot function  as shown in the
next cell:

totalSum  =  piv edu . sum ( axis  =  1)

totalSum . plot ( kind  =  � bar �,  style  =  �b �,  alpha  =  0.4 ,

title  =  " Total  Values  for  Country ")

. sort_ valu es ( a sce ndi ng  =  False )

In [29]:

Out[29]:

Note  that  if  we  want  the  bars  ordered  from the  highest  to  the lowest  value,
we  need  to  sort  the  values  in  the  Series  first.  The  parameter  kind  used  in  the  plot
function defines which kind of graphic will be used. In our case, a bar graph. The
parameter  style refers  to  the  style  properties  of  the  graphic,  in  our  case,  the  color

with Python for Data Scientists

Get Started

27

of  bars  is  set  to  b  (blue).  The  alpha  channel  can  be  modified  adding  a  keyword
parameter alpha with a percentage, producing a more translucent  plot.  Finally, using
the title keyword the name of the graphic can be set.

It is also possible to plot a DataFrame directly. In this case, each column is treated
as  a  separated  Series.  For  example,  instead  of  printing  the  accumulated  value
over the years, we can plot the value for each year.

In [30]:

my_co lor s  =  [ �b �,  �r �,  �g �,  �y �,  �m �,  �c �]
ax  =  pi ved u . plot ( kind  =  � barh �,

stacked  =  True ,
color  =  my_co lor s )

ax . le gen d ( loc  =  � ce nte r  left �,  b b o x _ t o _ a n ch o r   =  (1 ,  . 5 ) )

Out[30]:

In  this  case,  we  have  used  a  horizontal  bar  graph  (kind=�barh�)  stacking  all  the
years in the same country bar. This can be done by setting the parameter stacked
to  True.  The number  of  default  colors  in  a  plot  is  only  5,  thus  if  you  have more
than 5 Series to show, you need to specify more colors or otherwise the same set
of colors will be used again. We can set a new set of colors using the keyword color
with  a  list  of  colors.  Basic  colors  have  a  single-character code  assigned  to  each,
for example,  �b�  is  for  blue,  �r�  for  red,  �g�  for  green,  �y�  for  yellow,  �m�  for
magenta, and �c�  for cyan.  When  several  Series  are  shown  in  a  plot,  a  legend  is
created  for  identifying  each  one.  The  name  for  each  Series  is  the  name  of  the
column in the DataFrame. By default, the legend goes inside the plot area. If we
want  to  change  this,  we  can  use  the  legend  function  of  the  axis  object  (this  is  the
object  returned  when  the  plot  function  is  called).  By  using  the  loc  keyword,  we  can
set  the  relative  position  of  the  legend  with  respect  to  the  plot.  It  can  be  a
combination of right or left and upper, lower, or center. With bbox_to_anchor we
can  set  an  absolute  position  with  respect  to  the  plot,  allowing  us  to  put  the
legend outside the graph.

                                                                                    UNIT-2
Descriptive statistics, data preparation. Exploratory Data Analysis data summarization,
data  distribution,  measuring  asymmetry.  Sample  and  estimated  mean,  variance  and
standard  score.  Statistical  Inference  frequency  approach,  variability  of  estimates,
hypothesis testing using confidence intervals, using p-values

              Descriptive Statistics

Descriptive  statistics  helps  to  simplify  large  amounts  of  data  in  a  sensible
way. In  contrast  to  inferential  statistics,  which  will  be  introduced  in  a  later
chapter, in descriptive statistics we do not draw conclusions beyond the data we
are analyzing; neither  do  we  reach  any  conclusions  regarding  hypotheses  we  may
make.  We  do  not try to infer characteristics of the �population� (see below) of the
data, but claim to present quantitative descriptions of it in a manageable form. It
is simply a way to describe the data.

Statistics, and in particular descriptive statistics, is based on two main concepts:

�

a population is a collection of objects, items (�units�) about which information
is sought;

�  a sample is a part of the population that is observed.

Descriptive statistics applies the concepts, measures, and terms that are used
to  describe  the  basic  features  of  the  samples  in  a  study.  These  procedures  are
essential to  provide  summaries  about  the  samples  as  an  approximation  of  the
population.  Together  with  simple  graphics,  they  form  the  basis  of  every
quantitative analysis of data. In order to describe the sample data and to be able
to infer any conclusion, we should go through several steps:

1.  Data preparation:  Given  a  specific  example,  we  need  to  prepare  the  data

for generating statistically valid descriptions.

2.  Descriptive statistics: This generates different statistics to describe and

summa- rize the data concisely and evaluate different ways to visualize them.

30

3    Descriptive Statistics

Data Preparation

One  of  the  first  tasks  when  analyzing  data  is  to  collect  and  prepare  the  data  in  a
format  appropriate for analysis of the samples. The most common steps for data
preparation involve the following operations.

1.  Obtaining the data: Data can be read directly from a file or they might be obtained

by scraping the web.

2.  Parsing  the  data:  The  right  parsing  procedure  depends  on  what  format  the

data are in: plain text, fixed columns, CSV, XML, HTML, etc.

3.  Cleaning the data: Survey responses and other data files are almost always in-
complete. Sometimes, there are multiple codes for things such as, not asked,
did not know, and declined to answer. And there are almost always  errors. A
simple strategy is to remove or ignore incomplete records.

4.  Building data structures: Once you read the data, it is necessary to store them
in a data structure that lends itself to the analysis we are interested in. If the
data  fit into the memory, building a data structure is usually the way to go. If
not,  usually a  database  is  built,  which  is  an  out-of-memory  data  structure.
Most  databases  provide  a  mapping  from  keys  to  values,  so  they  serve  as
dictionaries.

The  Adult  Example

Let us consider a public database called the �Adult� dataset, hosted on the UCI�s
Machine Learning Repository.1 It contains approximately 32,000 observations con-
cerning  different  financial  parameters  related  to  the  US  population:  age,  sex,
marital (marital status of the individual), country, income (Boolean variable: whether
the per- son makes more than $50,000 per annum), education (the highest level of
education achieved by the individual), occupation, capital gain, etc.

We will show that we can explore the data by asking questions like: �Are men
more likely to become high-income professionals than women, i.e., to receive an
income of over $50,000 per annum?�

Data Preparation

First, let us read the data:

In [1]:

file  =  open ( � files  / ch03  / adult  . data �,  �r �)
def  chr_int ( a):

if  a. isdigit  () :  return   int (a)
else :  return   0

data  =  []
for  line  in  file :

data1  =  line  . split  ( �,  �)
if  len ( data1  )  ==  15 :

data . append  ([ chr_int  ( data1  [0])  ,  data1  [1] ,
chr_int  ( data1  [2])  ,  data1  [3] ,
chr_int  ( data1  [4])  ,  data1  [5] ,
data1 [6] ,  data1 [7] ,  data1  [8] ,
data1 [9] ,  chr_int  ( data1  [10]) ,
chr_int  ( data1  [11])  ,
chr_int  ( data1  [12])  ,
data1  [13] ,  data1 [14]

])

Checking the data, we obtain:

In [2]:

print  data  [1:2]

Out[2]:  [[50, �Self-emp-not-inc�, 83311, �Bachelors�, 13,

�Married-civ-spouse�, �Exec-managerial�, �Husband�, �White�, �Male�, 0, 0, 13, �United-
States�, r  <= 50K �]]

One of the easiest ways to manage data in Python is by using the DataFrame
structure,  defined  in  the  Pandas  library,  which  is  a  two-dimensional,  size-
mutable, potentially heterogeneous tabular data structure with labeled axes:

In [3]:

df  =  pd . DataF ra me ( data )
df . columns  =  [

� age �,  � type_ emp loy er  �,  � fnlwgt �,
� education �,  � ed uca tio n_n um  �,  � marital  �,
� occupati on  �,�  relatio nsh ip  �,  � race  �,
� sex �,  � capita l_g ain  �,  � ca pit al_ los s  �,
� hr_ per _we ek  �,  � country �,  � income  �
]

The  command  shape gives  exactly  the  number  of  data  samples  (in  rows,  in  this

case) and features (in columns):

In [4]:

df . shape

Out[4]:  (32561, 15)

32

3    Descriptive Statistics

Thus, we can see that our dataset contains 32,561 data records with 15

features each. Let us count the number of items per country:

In [5]:

counts  =  df . groupby  ( � country  �). size ()
print  counts . head  ()

Out[5]:  country

? 583
Cambodia 19
Vietnam 67
Yugoslavia 16

In [6]:

In [7]:

The first row shows the number of samples with unknown country, followed

by the number of samples corresponding to the first countries in the dataset.

Let us split people according to their gender into two groups: men and women.

ml  =  df [( df . sex  ==  � Male �)]

If we focus on high-income professionals separated by sex, we can do:

ml1  =  df [( df . sex  ==  � Male �)  &  ( df . income  == � >50 K\ n �)

]

fm  =  df [( df . sex  ==  � Female  �)]
fm1  =  df [( df . sex  ==  � Female �)  &  ( df . income  == � >50 K\ n

�)]

Exploratory  Data  Analysis

The  data  that  come  from  performing  a  particular  measurement  on  all  the
subjects in  a  sample  represent  our  observations  for  a  single  characteristic  like
country,  age,  education,  etc.  These  measurements  and  categories  represent  a
sample  distribution  of  the  variable,  which  in  turn  approximately  represents  the
population  distribution  of  the  variable.  One  of  the  main  goals  of  exploratory
data  analysis  is to  visualize  and  summarize  the  sample  distribution,  thereby
allowing us to make tentative assumptions about the population distribution.
Summarizing the Data

The  data  in  general  can  be  categorical  or  quantitative.  For  categorical  data,  a
simple tabulation  of  the  frequency  of  each  category  is  the  best  non-graphical
exploration  for  data  analysis.  For  example,  we  can  ask  ourselves  what  is  the
proportion of high- income professionals in our database:

3.3    Exploratory Data Analysis

33

In [8]:

df 1  =  df [( df . income  == � >50 K\ n �)]
print  � The  rate   of  people   with   high  income  is :  �,
int ( len ( df1 )/ float  ( len ( df )) *100)  ,  �%. �

print  � The  rate  of  men  with  high  income   is :  �,

int ( len ( ml1 )/ float  ( len ( ml )) *100) ,  �%. �

print  � The  rate   of  women   with  high  income  is :  �,

int ( len ( fm1 )/ float  ( len ( fm )) *100)  ,  �%. �

Out[8]:  The rate of people with high income is: 24 %.

The rate of men with high income is: 30 %. The rate of women
with high income is: 10 %.

Given  a  quantitative  variable,  exploratory  data  analysis  is  a  way  to  make
prelim- inary assessments about the population distribution of the variable using
the  data  of  the  observed  samples.  The  characteristics  of  the  population
distribution  of  a  quanti- tative  variable  are  its  mean,  deviation,  histograms,
outliers,  etc.  Our  observed  data represent just a finite set of samples of an often
infinite number of possible samples. The characteristics of our randomly observed
samples are interesting only to the degree that they represent the population of
the data they came from.

Mean
One of the first measurements we use to have a look at the data is to obtain
sample statistics from the data, such as the sample mean [1]. Given a sample of
n  values,
{   }  =
xi , i
2
values,

1 , . . . ,  n,  the  mean,  ?,  is  the  sum  of  the  values  divided  by  the  number  of
in other words:

?

 1
=
n

n

i =1

x .
i

(3.1)

The terms mean and average are often used interchangeably. In fact, the
main distinction between them is that the mean of a sample is the summary
statistic com- puted  by Eq. (3.1),  while  an average is  not strictly defined  and could
be  one  of many summary statistics that can be chosen to describe the central
tendency of a sample.
In our case, we can consider what the average age of men and women samples

in our dataset would be in terms of their mean:

    Descriptive Statistics

In [9]:

print   � The  average   age
ml [ � age �]. mean ()

of  men  is :  �,

print  � The average  age

of  women  is :  �,

fm [ � age �]. mean ()

print  � The average  age

of  high - income   men  is :  �,

ml1 [ � age �]. mean ()

print  � The average  age of high - income women  is : �,

fm1 [ � age �]. mean ()

Out[9]:          The average age of men is: 39.4335474989 The average age of

women is: 36.8582304336
The average age of high-income men is: 44.6257880516
The average age of high-income women is: 42.1255301103

This  difference  in  the  sample  means  can  be  considered  initial  evidence  that

there are differences between men and women with high income!

Comment:  Later,  we  will  work  with  both  concepts:  the  population  mean  and
the sample mean. We should not confuse them! The first is the mean of samples
taken from the population; the second, the mean of the whole population.
Sample Variance
The mean is not usually a sufficient descriptor of the data. We can go further by
knowing two numbers: mean and variance. The variance ?2  describes the spread
of the data and it is defined as follows:
1
n

(3.2)

?2  =

(xi  ? ?)2.
i

?

The term (xi  ?) is called the deviation from the mean, so the variance is the mean
squared  deviation.  The  square  root  of  the  variance,  ?,  is  called  the  standard
deviation. We  consider  the  standard  deviation,  because  the  variance  is  hard  to
interpret (e.g.,  if the units are grams, the variance is in grams squared).

Let us compute the mean and the variance of hours per week men and women

in our dataset work:

In [10]:

ml_mu  =  ml [ � age �]. mean ()
fm_mu  =  fm [ � age �]. mean ()
ml_var   =  ml [ � age �]. var ()
fm_var   =  fm [ � age �]. var ()
ml_std   =  ml [ � age �]. std ()
fm_std   =  fm [ � age �]. std ()
print   � Statistics   of  age  for  men :  mu : �,

ml_mu ,  � var : �,  ml_var ,  � std : �,  ml_std

3.3    Exploratory Data Analysis

35

        Out[10]: Statistics  of  age  for  men:  mu:  39.4335474989  var:  178.773751745 std: 13.3706301925
Statistics of age for women: mu: 36.8582304336 var: 196.383706395 std:
14.0136970994

We can see that the mean number of hours worked per week by women is signif-
icantly  lesser  than  that  worked  by  men,  but  with  much  higher  variance  and
standard deviation.

Sample Median

The  mean  of  the  samples  is  a  good  descriptor,  but  it  has  an  important  drawback:
what will happen if in the sample set there is an error with a value very different
from  the  rest?  For  example,  considering  hours  worked  per  week,  it  would
normally be in a range between 20 and 80; but what would happen if by mistake
there was a value of 1000? An item of data that is significantly different from the
rest  of  the  data  is called  an outlier. In  this case,  the  mean,  ?,  will  be  drastically
changed  towards  the  outlier.  One  solution  to  this  drawback  is  offered  by  the
statistical  median,  ?12,  which is  an  order  statistic  giving  the  middle  value  of  a
sample.  In  this  case,  all  the  values  are  ordered  by  their  magnitude  and  the
median is defined as the value that is in the middle of the ordered list. Hence, it is
a value that is much more robust in the face of outliers.

Let us see, the median age of working men and women in our dataset and the

median age of high-income men and women:

In [11]:

ml_median  =  ml [ � age �]. median ()
fm_median  =  fm [ � age �]. median ()
print  " Median   age  per  men  and  women  :  " ,

ml_median  ,  fm_median

ml_m edi an_ age   =  ml 1 [ � age �]. median ()
fm_m edi an_ age   =  fm 1 [ � age �]. median ()
print  " Median   age  per  men  and  women   with  high -

income  :  " ,

ml_median_ag e  ,  fm_ med ian _ag e

Out[11]: Median age per men and women: 38.0 35.0

Median age per men and women with high-income: 44.0 41.0

As expected, the median age of high-income people is higher than the whole
set of working people, although the difference between men and women in both
sets is the same.

Quantiles and Percentiles
Sometimes  we  are  interested  in  observing  how  sample  data  are  distributed  in
general.  In  this  case,  we  can  order  the  samples  xi  ,  then  find  the  xp  so  that  it
divides the data into two parts, where:

{  }

36

3    Descriptive Statistics

Fig. 3.1  Histogram of the age of  working  men (left) and  women (right)

�  a fraction  p of the data values is less than or equal to xp and
�  the  remaining fraction (1 ?  p) is  greater than  x p.

That value, xp,  is the p-th quantile, or the 100 p-th percentile. For example, a 5-
number summary is defined by the values  xmin, Q1, Q2, Q3, xmax , where Q1 is the
25  p-th percentile, Q2 is the 50  p-th percentile and Q3 is the 75  p-th percentile.
�

�

�

�

Data Distributions

Summarizing data by just looking at their mean, median, and variance can be danger-
ous: very different data can be described by the same statistics. The best thing to
do is  to  validate  the  data  by  inspecting  them.  We  can  have  a  look  at  the  data
distribution,  which  describes  how  often  each  value  appears  (i.e.,  what  is  its
frequency).

The most common representation of a distribution is a histogram, which is a graph
that shows the frequency of each value. Let us show the age of working men and
women separately.

ml_age   =  ml [ � age �]
ml_age  . hist ( normed   =  0 ,  histtype  =  � stepfilled  �,

bins  =  20 )

fm_age   =  fm [ � age �]
fm_age  . hist ( normed   =  0 ,  histtype  =  � stepfilled  �,

bins  =  10 )

The  output can  be seen in  Fig. 3.1. If  we  want  to  compare the histograms,  we

can plot them overlapping in the same graphic as follows:

In [12]:

In [13]:

3.3    Exploratory Data Analysis

37

Fig. 3.2 Histogram of the age of working men (in ochre) and women (in violet) (left). Histogram of the
age of working men (in ochre), women (in blue), and their intersection (in violet) after samples
normalization (right)

In [14]:

import   seaborn  as  sns
fm_age  . hist ( normed   =  0 ,  histtype  =  � stepfilled  �,

alpha   =  .5 ,  bins  =  20 )

ml_age  . hist ( normed   =  0 ,  histtype  =  � stepfilled  �,

alpha  =  .5 ,
color   =  sns . desaturate  (" india nred  " ,

.75) ,

bins  =  10 )

The output can be seen in Fig.  3.2 (left). Note that we are visualizing the absolute
values of the number of people in our dataset according to their age (the abscissa
of the histogram). As a side effect, we can see that there are many more men in
these conditions than women.

We can normalize the frequencies of the histogram by dividing/normalizing by
n,  the  number  of  samples.  The  normalized  histogram  is  called  the  Probability
Mass Function (PMF).

In [15]:

fm_age  . hist ( normed   =  1 ,  histtype  =  � stepfilled  �,

ml_age  . hist ( normed   =  1 ,  histtype  =  � stepfilled  �,

alpha   =  .5 ,  bins  =  20 )

alpha   =  .5 ,  bins  =  10 ,
color   =  sns . desaturate  (" india nred  " ,

.75) )

This  outputs  Fig.  3.2  (right),  where  we  can  observe  a  comparable  range  of  indi-

viduals (men and women).

The  Cumulative  Distribution  Function  (CDF),  or  just  distribution  function,
describes the probability that a real-valued random variable X with a given proba-
bility  distribution  will  be  found  to  have  a  value  less  than  or  equal  to  x .  Let  us  show
the CDF of age distribution for both men and women.

38

3    Descriptive Statistics

Fig. 3.3 The CDF of the age of
working male (in blue)
and female (in red) samples

In [16]:

ml_age  . hist ( normed   =  1 ,  histtype   =  � step �,
cumulative   =  True ,
bins  =  20 )

linewidth   =  3.5 ,

fm_age  . hist ( normed   =  1 ,  histtype = � step �,

cumulative   =  True ,
bins  =  20 ,
color   =  sns . desaturate  (" india nred  " ,

linewidth   =  3.5 ,

.75) )

The output can be seen in Fig. 3.3, which illustrates the CDF of the age distributions

for both men and women.

Outlier Treatment

As mentioned before, outliers are data samples with a value that is far from the
central tendency. Different rules can be defined to detect outliers, as follows:

�  Computing samples that are far from the median.
�  Computing samples whose values exceed the mean by 2 or 3 standard deviations.

For example, in our case, we are interested in the age statistics of men versus
women with high incomes and we can see that in our dataset, the minimum age is
17 years and the maximum is  90 years.  We can consider that some  of these samples
are  due to errors or are not representable. Applying the domain knowledge, we
focus on the median age (37, in our case) up to 72 and down to 22 years old, and
we consider the rest as outliers.

3.3    Exploratory Data Analysis

39

In [17]:

df 2  =  df . drop ( df . index  [

( df . income  ==  � >50 K\ n �)  &
(df[ � age �]  >  df [ � age �]. median  ()  +  35 )  &
(df[ � age �]  >  df [ � age �]. median ()  -15)
])

ml1_age  =  ml 1 [ � age �]
fm1_age  =  fm 1 [ � age �]

ml2_age  =  ml1_age  . drop ( ml1_age . index  [

( ml1_age  >  df [ � age �]. median  ()  +  35 )  &
( ml1_age  >  df [ � age �]. median  ()  -  15 )
])

fm2_age  =  fm1_age  . drop ( fm1_age . index  [

( fm1_age  >  df [ � age �]. median  ()  +  35 )  &
( fm1_age  >  df [ � age �]. median  ()  -  15 )
])

We can check how the mean  and the median  changed once the data  were cleaned:

In [18]:

mu2ml  =  ml2_age  . mean ()
std2ml  =  ml2_age  . std ()
md2ml  =  ml2_age  . median  ()
mu2fm  =  fm2_age  . mean ()
std2fm  =  fm2_age  . std ()
md2fm  =  fm2_age  . median  ()

print  " Men  statistics  :"
print  " Mean :" ,  mu2ml ,  " Std :" ,  std2ml
print   " Median  :" ,  md2ml
print  " Min :" ,  ml2_age . min () ,  " Max :" ,  ml2_age . max ()

print  " Women   statistics  :"
print  " Mean :" ,  mu2fm ,  " Std :" ,  std2fm
print   " Median  :" ,  md2fm
print  " Min :" ,  fm2_age . min () ,  " Max :" ,  fm2_age . max ()

Out[18]: Men  statistics:  Mean:  44.3179821239  Std:  10.0197498572  Median:

44.0 Min: 19 Max: 72
Women  statistics:  Mean:  41.877028181  Std:  10.0364418073  Median:
41.0 Min: 19 Max: 72

Let us visualize how many outliers are removed from the whole data by:

In [19]:

plt . figure ( figsize   =  (13.4 ,  5) )
df . age [( df . income   ==  � >50 K\ n �)]

. plot ( alpha   =  .25 ,  color   =  � blue �)

df 2 . age [( df 2 . income   ==  � >50 K\ n �)]

. plot ( alpha   =  .45 ,  color   =  � red �)

40

3  Descriptive Statistics

Fig. 3.4  The red  shows the cleaned  data  without the considered  outliers (in  blue)

Figure 3.4  shows  the  outliers  in  blue  and  the  rest  of  the  data  in  red.  Visually,

we can confirm that we removed mainly outliers from the dataset.

Next  we  can  see  that  by  removing  the  outliers,  the  difference  between  the
popula- tions (men and women) actually decreased. In our case, there were more
outliers in men than women. If the difference in the mean values before removing
the outliers is 2.5, after removing them it slightly decreased to 2.44:

In [20]:

print  � The  mean  differenc e  with  outliers   is :  %4.2 f.

�

%  ( ml_age  . mean ()  -  fm_age  . mean  () )

print  � The  mean  differen ce   without  outliers   is :

%4.2 f. �

%  ( ml2_age  . mean ()  -  fm2_age . mean () )

Out[20]: The mean difference with outliers is: 2.58.

The mean difference without outliers is: 2.44.

Let us observe the difference of men and women incomes in the cleaned

subset with some more details.

In [21]:

countx ,  divis ionx   =  np . histog ram  ( ml 2 _age ,  normed   =

True )

county ,  divis iony   =  np . histog ram  ( fm 2 _age ,  normed   =

True )

val  =  [( divisionx  [ i]  +  divisi onx  [ i +1]) /2

for  i  in  range  ( len ( divisionx )  -  1) ]

plt . plot ( val ,  countx   -  county ,  �o - �)

The  results  are  shown  in  Fig. 3.5.  One  can  see  that  the  differences  between
male and  female values are  slightly  negative  before  age  42  and  positive  after  it.
Hence, women tend to be promoted (receive more than 50 K) earlier than men.

3.3    Exploratory Data Analysis

41

Fig. 3.5  Differences  in high-income  earner men versus  women as a function  of age

Measuring Asymmetry: Skewness and Pearson�s Median
Skewness Coefficient

For univariate data, the formula for skewness is a statistic that measures the
asym- metry of the set of n data samples, xi :

.

1

i (xi  ? ?3)
?3

g1 =

n

,

(3.3)

where ? is the mean, ? is the standard deviation, and n is the number of data points.
Negative  deviation  indicates  that  the  distribution  �skews  left�  (it  extends
further to  the left  than  to  the right). One  can easily  see  that  the  skewness  for  a
normal  distribution  is  zero,  and  any  symmetric  data  must  have  a  skewness  of
zero. Note that skewness can be affected by outliers! A simpler alternative is to
look at the relationship between the mean ? and the median ?12.

In [22]:

def  skewness ( x):
res  =  0
m  =  x. mean ()
s = x. std ()
for  i  in  x:

res  +=  ( i - m)  *  ( i - m)  *  ( i - m)

res  /=  ( len ( x)  *  s  *  s  *  s)
return   res

print  " Skewness   of  the  male  population   =  " ,

skewness ( ml2_age )

print  " Skewness  of  the  female   population   is  =  " ,

skewness ( fm2_age )

42

3    Descriptive Statistics

Out[22]: Skewness  of  the  male  population  =  0.266444383843 Skewness of the female

population = 0.386333524913

That  is,  the female  population  is more  skewed  than  the  male,  probably  since

men could be most prone to retire later than women.

The Pearson�s median skewness coefficient is a more robust alternative to the

skewness coefficient and is defined as follows:

gp  = 3(? ? ?12)?.

There are many other definitions for skewness that will not be discussed here.
In our  case,  if  we  check  the  Pearson�s  skewness  coefficient  for  both  men  and
women, we can see that the difference between them actually increases:

In [23]:

def  pearson ( x):

return   3 *( x. mean ()  -  x. median  () )* x. std ()

print   " Pearson  � s  coeffic ien t   of  the  male  population

=  " ,

pearson ( ml2_age )

print   " Pearson  � s  coeffi cie nt   of  the  female

population  =  " ,

pearson ( fm2_age )

Out[23]: Pearson�s  coefficient  of  the  male  population  =  9.55830402221 Pearson�s coefficient of the

female population = 26.4067269073

Continuous  Distribution

The  distributions  we  have  considered  up  to  now  are  based  on  empirical
observations and thus are called empirical distributions. As an alternative, we may
be  interested  in  considering  distributions  that  are  defined  by  a  continuous
function and are called continuous distributions  [2].  Remember that  we  defined  the
PMF, f X (x), of a discrete random variable  X  as  f X (x)  P(X    x) for all x . In the case
of  a  continuous  random  variable  X ,  we  speak  of  the  Probability  Density  Function
(PDF), which

=

=

3.3    Exploratory Data Analysis

43

Fig. 3.6  Exponential  CDF (left) and PDF  (right)  with  ?  = 3.00

is  defined  as  FX (x) where  this  satisfies:  FX (x) =

� x    f X (t)?t  for  all  x .  There  are

?
many continuous distributions; here, we will conside
r the most common ones: the
exponential and the normal distributions.

The Exponential Distribution

Exponential  distributions  are  well  known  since  they  describe  the  inter-arrival
time  between  events.  When  the  events  are  equally  likely  to  occur  at  any  time,
the distri- bution of the inter-arrival time tends to an exponential distribution. The
CDF  and  the PDF  of  the  exponential  distribution  are  defined  by  the  following
equations:

CDF(x) = 1 ? e??x ,

PDF(x) = ?e??x .

The parameter ? defines the shape of the distribution. An example is given
in Fig. 3.6. It is easy to show that the mean of the distribution is  1 , the variance is
and the median is  l?
?

Note  that  for  a  small  number  of  samples,  it  is  difficult  to  see  that  the  exact
empirical distribution fits a continuous distribution. The best way to observe this
match  is to  generate  samples  from  the  continuous  distribution  and  see  if  these
samples match the data. As an exercise, you can consider the birthdays of a large
enough  group  of people,  sorting  them  and  computing  the  inter-arrival  time  in
days.  If  you  plot  the  CDF  of  the  inter-arrival  times,  you  will  observe  the
exponential distribution.

?2

There  are  a  lot  of  real-world  events  that  can  be  described  with  this
distribution,  including  the  time  until  a  radioactive  particle  decays;  the  time  it
takes  before  your  next telephone call; and the time until default (on payment to
company debt holders) in  reduced-form  credit  risk modeling.  The  random variable  X
of the lifetime of some
batteries  is  associated  with  a  probability  density  function  of  the  form:  PD F(x) =

x
1
e ?
4

4 e

(x   ?)2
�  ?
2

44

3    Descriptive Statistics

Fig. 3.7  Normal PDF with ? = 6 and ?  = 2

The Normal Distribution

The normal distribution, also called the Gaussian distribution, is the most common
since  it  represents  many  real  phenomena:  economic,  natural,  social,  and  others.
Some well-known examples of real phenomena with a normal distribution are as
follows:

�  The size of living tissue (length, height, weight).
�  The length of inert appendages (hair, nails, teeth) of biological specimens.
�  Different physiological measurements (e.g., blood pressure), etc.

The normal CDF has no closed-form expression and its most common

represen- tation is the PDF:

PDF (x) =  ?

1

(x ??)2
2?2     .

?

e

2??2

The parameter ? defines the shape of the distribution. An example of the PDF

of a normal distribution with ? = 6 and ? = 2 is given in Fig. 3.7.

Kernel Density

In  many  real  problems,  we  may  not  be  interested  in  the  parameters  of  a
particular distribution of  data,  but  just  a  continuous representation  of the  data.
In  this  case, we should estimate the distribution non-parametrically (i.e., making
no assumptions about the form of the underlying distribution) using kernel density
estimation.  Let  us imagine  that  we  have  a  set  of  data  measurements  without
knowing  their  distribution  and  we  need
the  continuous
representation  of  their  distribution.  In  this  case,  we  can  consider  a  Gaussian
kernel to generate the density around the data. Let us  consider  a  set  of  random
data  generated  by  a  bimodal  normal  distribution.  If  we consider  a  Gaussian
kernel around the data, the sum of those kernels can give us

to  estimate

Explorator

y Data Analysis

45

Fig. 3.8  Summed kernel functions around a  random set of points (left) and the kernel density
estimate with the optimal bandwidth (right) for our dataset. Random data shown in blue, kernel
shown in black and summed function shown in red

a continuous function that when normalized would approximate the density of the
distribution:

In [24]:

x1  =  np . random  . normal ( -1 ,  0.5 ,  15 )
x2  =  np . random  . normal  (6 ,  1 ,  10 )
y  =  np .r_[ x1 ,  x2]  #  r_  t ranslate s   slice  objects   to

conc ate nat ion   along  the  first   axis  .

x  =  np . linspace  ( min (y)  ,  max ( y) ,  100)

s  =  0.4  #  Smoothing   para meter

#  Calculate  the  kernels
kernels  =  np . transpose  ([ norm . pdf (x ,  yi ,  s)  for  yi

in  y])

plt . plot (x ,  kernels  ,  �k: �)
plt . plot (x ,  kernels  . sum (1) ,  �r �)
plt . plot  (y ,  np . zeros  ( len (y)) ,  �bo �,  ms  =  10)

Figure  3.8  (left)  shows  the  result  of  the  construction  of  the  continuous

function from the kernel summarization.

In fact, the library SciPy3 implements a Gaussian kernel density estimation that
automatically chooses the appropriate bandwidth parameter for the kernel. Thus,
the final construction of the density estimate will be obtained by:

.

46

In [25]:

3    Descriptive Statistics

from  scipy  . stats   import   kde
density  =  kde . gaus sia n_k de  ( y)
xgrid   =  np . linspace  ( x. min () ,  x. max () ,  200)
plt . hist (y ,  bins   =  28 ,  normed  =  True )
plt . plot ( xgrid ,  density ( xgrid  ) ,  �r-� )

Figure 3.8  (right)  shows  the  result  of  the  kernel  density  estimate  for  our  example.

Estimation

An  important  aspect  when  working  with  statistical  data  is  being  able  to  use
estimates to approximate the values of unknown parameters of the dataset. In this
section,  we will review  different  kinds  of  estimators  (estimated  mean,  variance,
standard score, etc.).

Sample and Estimated Mean, Variance and Standard Scores

In continuation, we will deal with point estimators that are single numerical estimates
of parameters of a population.
Mean
Let us assume that we know that our data are coming from a normal distribution
and the random samples drawn are as follows:

{0.33, ?1.76, 2.34, 0.56, 0.89}.

�

The question is can we guess the mean ? of the distribution? One approximation
is given by the sample mean, x . This process is called estimation and the statistic (e.g.,
the sample mean) is called an estimator. In our case, the sample mean is 0.472, and
it seems  a  logical  choice  to  represent  the  mean  of  the  distribution.  It  is  not  so
evident if we add a sample with a value of 465. In this case, the sample mean will be
77.11, which does not look like the mean of the distribution. The reason is due to
the  fact  that  the  last  value  seems  to  be  an  outlier  compared  to  the  rest  of  the
sample. In order to avoid this effect, we can try first to remove outliers and then to
estimate  the mean; or  we  can  use  the  sample  median  as  an  estimator  of  the
�
mean  of  the  distribution. If  there  are  no  outliers,  the  sample mean  x minimizes
the following mean squared error:

�

?

MSE

=

1

n

� ?

(x

?)2,

where n is the number of times we estimate the
mean. Let us compute the MSE of a set of random
data:

3.4    Estimation

47

In [26]:

NTs  =  200
mu = 0.0
var  =  1.0
err = 0.0
NPs  =  1000
for  i  in  range ( NTs ):

x  =  np . random  . normal  ( mu ,  var ,  NPs )
err  +=  ( x. mean () - mu ) ** 2

print   � MSE :  �,  err / NTests

Out[26]: MSE:  0.00019879541147

Variance

If we ask ourselves what is the variance, ?2, of the distribution of  X , analogously
we can use the sample variance as an estimator. Let us denote by ?2  the sample
variance estimator:

�

1
?2
�  =    ? �i
n

(x

x )2.

For large samples, this estimator works well, but for a small number of

samples it is biased. In those cases, a better estimator is given by:

?� 2  =

     1
n ? 1

(xi  ? x�)2.

Standard Score
In  many  real  problems,  when  we  want  to  compare  data,  or  estimate  their
correlations or  some  other  kind  of  relations,  we  must  avoid  data  that  come  in
different  units.  For  example,  weight  can  come in  kilograms  or grams. Even  data
that come in the same units can still belong to different distributions. We need to
normalize  them  to standard  scores.  Given  a dataset  as  a  series  of  values,  xi  ,  we
convert the data to standard  scores  by  subtracting  the  mean  and  dividing  them  by
the standard deviation:

{  }

(xi  ? ?)
.
?
Note  that  this measure  is  dimensionless  and  its  distribution  has  a mean  of  0
and  variance  of  1.  It  inherits  the  �shape�  of  the  dataset:  if  X  is  normally
distributed, so is Z ; if X  is skewed, so is Z .

zi
=

Covariance, and Pearson�s and Spearman�s Rank Correlation

Variables of data can express relations. For example, countries that tend to invest

in research  also  tend  to  invest  more  in  education  and  health.  This  kind  of
relationship is captured by the covariance.

48

3    Descriptive Statistics

Fig.  3.9  Positive  correlation  between  economic  growth  and  stock  market  returns  worldwide  (left).
Negative correlation between the world oil production and gasoline prices worldwide (right)

Covariance

When two variables share the same tendency, we speak about covariance. Let us
consider two series, xi and  yi  . Let us center the data with respect to their mean:
dxi      xi      ?X  and  d  yi      yi        ?Y  .  It  is  easy  to  show  that  when  xi      and      yi      vary
together, their deviations tend to have the same sign. The covariance is defined
as the mean of the following products:

=  ?

=  ?

{  }

{  }

{  }

{  }

Cov(X, Y)

 1
=
n

n

i
i =1

dx  d y  ,
i

where n is the length of both sets. Still, the covariance itself is hard to interpret.

Correlation and the Pearson�s Correlation

If we normalize the data with respect to their deviation, that leads to the
standard scores; and then multiplying them, we get:

?X
The  mean  of  this  product  is  ? =  1  .n

n

i =1

?i  =

xi  ? ?X  yi  ? ?Y
?Y

.

?i .  Equivalently,  we  can  rewrite  ?  in

terms of the covariance, and thus obtain the Pearson�s correlation:

?

=

Cov(X, Y)

?X ?Y

.

Note  that  the  Pearson�s  correlation  is  always  between  1  and  1,  where
the magnitude depends on the degree of correlation. If the Pearson�s correlation is
?
1 (or 1), it means that the variables are perfectly correlated (positively or
negatively) (see Fig. 3.9). This means that one variable can predict the other very
well. However,

?

+

Estimation

49

Fig. 3.10  Anscombe configurations

=

having ? 0, does not necessarily mean that the variables are not correlated! Pear-
son�s  correlation  captures  correlations  of  first  order,  but  not  nonlinear
correlations. Moreover, it does not work well in the presence of outliers.

Spearman�s Rank Correlation

The Spearman�s rank correlation comes as a solution to the robustness problem
of Pearson�s  correlation  when  the  data  contain  outliers.  The main  idea is  to  use
the  ranks  of  the  sorted  sample  data,  instead  of  the  values  themselves.  For
example, in the list [4, 3, 7, 5], the rank of 4 is 2, since it will appear second in the
ordered  list  ([3,  4,  5,  7]).  Spearman�s  correlation  computes  the  correlation
between the ranks
of  the  data.  For  example,  considering  the  data:  X  [10,  20,  30,  40,  1000],  and
Y      70,   1000,   50,   10,   20 , where we have an outlier in each one set. If we
compute the ranks, they are [1.0, 2.0, 3.0, 4.0, 5.0] and [2.0, 1.0, 3.0, 5.0, 4.0]. As
value  of  the  Pearson�s  coefficient,  we  get  0.28,  which  does  not  show  much

?  ?  ?

= [?  ?

=

]

correlation

between  the  sets.  However,  the  Spearman�s  rank  coefficient,  capturing  the
correlation  between  the  ranks,  gives  as  a  final  value  of  0.80,  confirming  the
correlation between the sets. As an exercise, you can compute the Pearson�s and
the Spearman�s rank correlations for the different Anscombe configurations given in
Fig.  3.10.  Observe  if  linear  and  nonlinear  correlations  can  be  captured  by  the
Pearson�s and the Spearman�s rank correlations.

                       Statistical Inference

Introduction

There is not only one way to address the problem of statistical inference. In fact,
there  are  two  main  approaches  to  statistical  inference:  the  frequentist  and
Bayesian approaches. Their differences are subtle but fundamental:

�

�

In the case of the frequentist approach, the main assumption is that there is a
population,  which  can  be  represented  by  several  parameters,  from  which  we
can obtain  numerous  random  samples.  Population  parameters  are  fixed  but
they  are  not  accessible  to  the  observer.  The  only  way  to  derive  information
about these parameters is to take a sample of the population, to compute the
parameters of the sample,  and  to  use  statistical inference  techniques  to make
probable propositions regarding population parameters.
The  Bayesian  approach  is  based  on  a  consideration  that  data  are  fixed,  not  the
result of a repeatable sampling process, but parameters describing data can be
described probabilistically.  To  this  end,  Bayesian  inference  methods  focus  on
producing  parameter  distributions  that  represent  all  the  knowledge  we  can
extract from the sample and from prior information about the problem.

A  deep  understanding  of  the  differences  between  these  approaches  is  far
beyond the scope of this chapter, but there are many interesting references that
will enable you to learn about it [1]. What is really important is to realize that the
approaches  are  based  on  different  assumptions which  determine  the validity  of
their inferences. The assumptions are related in the first case to a sampling process;
and  to  a  statistical  model  in  the  second  case.  Correct  inference  requires  these
assumptions  to  be  correct.  The  fulfillment  of  this  requirement  is  not  part  of  the
method, but it is the responsibility of the data scientist.

In this chapter, to keep things simple, we will only deal with the first approach,
but we suggest the reader also explores the second approach as it is well worth it!

Statistical Inference: The Frequentist Approach

As  we  have  said,  the  ultimate  objective  of  statistical  inference,  if  we  adopt  the
fre-  quentist  approach,  is  to  produce  probable  propositions  concerning  population

param-  eters  from  analysis  of  a  sample.  The  most
propositions are as follows:
�

Propositions  about point  estimates. A  point  estimate is  a  particular value  that
best approximates  some parameter  of  interest.  For example,  the mean  or  the
variance of the sample.
Propositions about confidence intervals or set estimates. A confidence interval
is a range of values that best represents some parameter of interest.

important  classes  of

�

�  Propositions about the acceptance or rejection of a hypothesis.

In  all  these  cases,  the  production  of  propositions  is  based  on  a  simple
assumption: we  can  estimate  the  probability  that  the  result  represented  by  the
proposition  has  been  caused  by  chance.  The  estimation  of  this  probability  by
sound methods is one of the main topics of statistics.

The  development  of  traditional  statistics  was  limited  by  the  scarcity  of
computa-  tional  resources.  In  fact,  the  only  computational  resources  were
mechanical  devices  and  human  computers,  teams  of  people  devoted  to
undertaking  long  and  tedious  calculations.  Given  these  conditions,  the  main
results  of classical  statistics  are  theo- retical  approximations,  based  on  idealized
models  and  assumptions,  to  measure  the effect  of  chance  on  the  statistic  of
interest. Thus, concepts such as the Central Limit Theorem, the empirical sample
distribution or the t-test are central to understanding this approach.

The development of modern computers has opened an alternative strategy for
measuring chance that is based on simulation; producing computationally inten-
sive  methods  including  resampling  methods  (such  as  bootstrapping),  Markov
chain Monte  Carlo  methods,  etc.  The  most  interesting  characteristic  of  these
methods is that they allow us to treat more realistic models.

Measuring the Variability in Estimates

Estimates  produced  by  descriptive  statistics  are  not equal to  the  truth  but  they
are  better  as  more  data  become  available.  So,  it  makes  sense  to  use  them  as
central elements of our propositions and to measure its variability with respect to the
sample size.

                         Point Estimates

Let  us consider  a  dataset  of  accidents in  Barcelona  in  2013.  This  dataset can  be
downloaded  from  the  OpenDataBCN  website,1  Barcelona  City  Hall�s  open  data
service.  Each  register  in  the  dataset  represents  an  accident  via  a  series  of
features: weekday, hour, address, number of dead and injured people, etc. This
dataset  will  represent our population: the set of all reported traffic accidents in
Barcelona during 2013.

In [1]:

Sampling Distribution of Point Estimates

Let  us  suppose  that  we  are  interested  in  describing  the  daily  number  of  traffic
acci-  dents  in  the  streets  of  Barcelona  in  2013.  If  we  have  access  to  the
population,  the  computation  of  this  parameter  is  a  simple  operation:  the  total
number of accidents divided by 365.

data  =  pd . re ad_csv  (" files / ch04 / A C CI D EN T S _G U _ BC N _ 20 1 3  . csv ")
data [ � Date �]  = data [ u � Dia de mes �]. apply ( lambda  x: str (x))

+  �-�  +
data [ u � Mes  de any �]. apply ( lambda  x: str (x))

data [ � Date �]  =  pd . t o_da tetim e  ( data [ � Date �])
acciden ts  =  data . groupby ([ � Date �]) . size ()

Out[1]:
Mean:
25.9095

B

u
t

n
o
w
,

f
o
r

i
l
l
u
s
t
r
a
t
i
v
e

suppose  that  we  only  have  access  to  a  limited  part  of  the  data  (the
sample):  the  number  of accidents  during  some  days  of  2013.  Can  we
still give an approximation of the population mean?

The most intuitive way to go about providing such a mean is simply
to  take  the  sample  mean.  The  sample  mean  is  a  point  estimate  of  the
population  mean.  If  we  can  only  choose  one  value  to  estimate  the
population mean, then this is our best guess.

The  problem  we  face  is  that  estimates  generally  vary  from  one
sample to another, and  this  sampling  variation  suggests  our  estimate
may  be  close,  but  it  will  not  be  exactly  equal  to  our  parameter  of
interest. How can we measure this variability?

In our example, because we have access to the population, we can
empirically build the sampling distribution of the sample mean2 for a
given number of observations. Then, we can use the sampling
distribution to compute a measure of the variability. In Fig. 4.1, we can

=

see the empirical sample distribution of the mean for s

=

10.000 samples with n

200 observations from our dataset. This empirical distribution has

been built in the following way:    Statistical Inference

Fig. 4.1  Empirical distribution  of  the sample  mean. In red,  the mean  value  of this  distribution

1.  Draw  s  (a  large  number)  independent  samples  {x 1 , . . .,  xs}  from  the

population where each element x j  is composed of {x j }i=1,...,n.
i

2.  Evaluate the sample mean ?� j  =  1  .n
3.  Estimate the sampling distribution of ? by the empirical distribution of the

x j  of each sample.

n

i

i =1
�

sample
replications.

#  popu latio n
df  =  acc ide nts  . to_f ram e  ()
N_test  =  10000
elements  =  200
# mean array of samples
means  =  [ 0 ]  *  N_test
#  sample  g enera tion
for  i  in  range ( N_test ):

rows  =  np . random . choice ( df . index . values ,  elements  )
sampled _df   =  df . ix [ rows ]
means [ i]  =  s ample d_df  . mean ()

p
u
r
p
o
s
e
s
,
In [2]:

l
e
t

u
s

I

n

g
e
n
e
r
a
l
,

te  from  a  sample  of  size  n,  we  define  its  sampling distribution  as  the
distribution of the point estimate based on samples of  size n from its
population.  This  definition  is  valid  for  point  estimates  of  other
population  parameters,  such  as  the  population median  or population
standard  deviation,  but  we will  focus  on  the  analysis  of  the  sample
mean.

The sampling distribution of an estimate plays an important role in
understanding  the  real  meaning  of  propositions  concerning  point
estimates.  It  is  very  useful  to  think of  a  particular  point  estimate  as
being drawn from such a distribution.

The Traditional Approach

In  real  problems,  we  do  not  have  access  to  the  real  population  and
so  estimation of  the  sampling  distribution  of  the  estimate  from  the
empirical distribution of the sample replications is not an option. But
this  problem can  be  solved by  making  use of  some  theoretical results
from traditional statistics.

g
i
v
e
4.3    Measuring the Variability in Estimates
n

55

{  }

SE =

  ?x
?
n

It can be mathematically shown that given n independent observations xi i=1,..,n
a
of  a  population  with  a  standard  deviation  ?x  ,  the  standard  deviation  of  the
sample mean ?x� , or standard error, can be approximated by this formula:
p
o
i
The demonstration of this result is based on the Central Limit Theorem: an
n
old theorem with a history that starts in 1810 when Laplace released his first paper
t
on it. This formula uses the standard deviation of the population ?x , which is not
known, but it can be shown that if it is substituted by its empirical estimate ?x , the
e
estimation is sufficiently good if n > 30  and the population distribution is not
s
skewed. This allows us to estimate the standard error of the sample mean even if
t
we do not have
i
access to the population.
m
So,  how  can  we  give  a  measure  of  the  variability  of  the  sample  mean?  The
a
answer is simple: by giving the empirical standard error of the mean distribution.

�

rows  =  np . random . choice ( df . index . values ,  200)
sampled _df   =  df . ix [ rows ]
est_sig ma_me an   =  sam pled _df  . std () / math . sqrt (200)

print  � Direct  estim ation   of  SE  from  one  sample  of

200  elements : �,  e st_si gma_m ean  [ 0 ]

print � Esti matio n  of the SE  by  sim ulati ng  10000 sa mples of

200  elements : �,  np . array ( means ). std ()

Out[3]:  Direct estimation of SE from one sample of 200 elements: 0.6536 Estimation of the SE by

simulating 10000 samples of 200
elements: 0.6362

Unlike the case of the sample mean, there is no formula for the standard error

of other interesting sample estimates, such as the median.

The Computationally Intensive Approach

Let  us  consider  from  now  that  our  full  dataset  is  a  sample  from  a  hypothetical
population (this is the most common situation when analyzing real data!).

A modern alternative to the traditional approach to statistical inference is the
bootstrapping  method  [2].
In  the  bootstrap,  we  draw  n  observations  with
replacement from the  original  data to  create  a  bootstrap  sample or  resample.  Then,
we  can  calculate the  mean  for  this  resample.  By  repeating  this  process  a  large
number  of  times,  we  can  build  a  good  approximation  of  the  mean  sampling
distribution (see Fig. 4.2).

56

4    Statistical Inference

Fig. 4.2  Mean sampling  distribution  by bootstrapping.  In red,  the  mean  value  of this  distribution

In [4]:

def  m eanB ootst rap  ( X ,  numberb  ):

x  =  [0]* number b
for  i  in range ( n umberb ):
sample  =  [X[j]

for  j
in  np . random . ran dint  ( len ( X) ,  size = len (X) )

]

x[i]  =  np . mean ( sample )

return  x

m  =  m eanBo otst rap  ( accidents ,  10000)
print  " Mean  estimate :" ,  np . mean ( m)

Out[4]:  Mean estimate: 25.9094

The  basic  idea  of  the  bootstrapping  method  is  that  the  observed  sample
contains  sufficient
information  about  the  underlying  distribution.  So,  the
information we can extract  from  resampling  the  sample  is  a  good  approximation  of
what can be expected from resampling the population.

The  bootstrapping  method  can  be  applied  to  other  simple  estimates  such  as
the  median  or  the  variance  and  also  to  more  complex  operations  such  as
estimates of censored data.3

Confidence Intervals

A  point  estimate  ?,  such  as  the  sample  mean,  provides  a  single plausible  value
for a  parameter.  However,  as  we  have  seen,  a  point  estimate  is  rarely  perfect;
usually there is some error in the estimate. That is why we have suggested using the
standard error as a measure of its variability.

Instead  of  that,  a  next  logical  step  would  be  to  provide  a  plausible  range  of
values for the parameter. A plausible range of values for the sample parameter is
called a confidence interval.

We  will  base  the  definition  of  confidence  interval  on  two  ideas:

1.  Our point estimate is the most plausible value of the parameter, so it makes

sense to build the confidence interval around the point estimate.
2.  The plausibility of a range of values can be defined from the sampling

distribution of the estimate.

For  the  case  of  the  mean,  the  Central  Limit  Theorem  states  that  its

sampling distribution is normal:

Theorem 4.1  Given a population with a finite mean ? and a finite non-zero variance ?
2,  the  sampling  distribution  of  the  mean  approaches  a  normal  distribution  with  a
mean of ? and a variance of ? 2/n  as n, the sample size, increases.

In  this  case,  and  in  order  to  define  an  interval,  we  can  make  use  of  a  well-
known result from probability that applies to normal distributions: roughly 95% of
the time our estimate will be within 1.96 standard errors of the true mean of the
distribution. If  the  interval  spreads  out  1.96  standard  errors  from  a  normally
distributed  point  estimate,  intuitively  we  can  say  that  we  are  roughly  95%
confident that we have captured the true parameter.

CI  = [? ? 1.96 � SE , ? + 1.96 � SE ]

In [5]:

m  =  accid ents  . mean ()
se  =  ac cid ent s  . std () / math . sqrt ( len ( ac cid ent s  ))
ci  =  [ m  -  se *1.96 ,  m  +  se *1.96]
print  " Co nfi den ce   int erva l  :" ,  ci

Out[5]:  Confidence interval: [24.975, 26.8440]

Suppose we want to consider confidence intervals where the confidence level
is somewhat higher than 95%: perhaps we would like a confidence level of 99%.
To  create  a  99%  confidence  interval,  change  1.96  in  the  95%  confidence  interval
formula  to  be  2.58  (it  can  be  shown  that  99%  of  the  time  a  normal  random
variable will be within 2.58 standard deviations of the mean).

In general, if the point estimate follows the normal model with standard error SE ,

then a confidence interval for the population parameter is

where z corresponds to the confidence level selected:

? � z � SE

Confidence Level
z Value

90%
1.65

95%
1.96

99%
2.58

99.9%
3.291

This is how we would compute a 95% confidence interval of the sample mean

using bootstrapping:

1.  Repeat the following steps for a large number, s, of times:

a.  Draw  n  observations  with  replacement  from  the  original  data  to  create

a bootstrap sample or resample.
b.  Calculate the mean for the resample.

2.  Calculate the mean of your s  values of the sample statistic. This process

gives you a �bootstrapped� estimate of the sample statistic.

3.  Calculate  the  standard  deviation  of  your  s  values  of  the  sample  statistic.
This process gives you a �bootstrapped� estimate of the SE of the sample
statistic.

4.  Obtain the 2.5th and 97.5th percentiles of your s values of the sample statistic.

In [6]:

m  =  m eanBo otst rap  ( accidents ,  10000)
sample_ mean   =  np . mean ( m)
sample_ se  =  np . std ( m)

print  " Mean  estimate  :" ,  samp le_me an
print  " SE  of  the  estimate  :" ,

sample_ se

ci  =  [ np . per centi le  ( m ,  2.5) ,  np . p erce ntile  ( m ,  97.5) ]
print  " Co nfi denc e   int erv al  :" ,  ci

Out[6]:  Mean estimate: 25.9039

SE of the estimate: 0.4705
Confidence interval: [24.9834, 26.8219]

But What Does �95% Confident� Mean?

The real meaning of �confidence� is not evident and it must be understood from
the point of view of the generating process.

Suppose  we  took  many  (infinite)  samples  from  a  population  and  built  a  95%
confidence interval from each sample. Then about 95% of those intervals would
contain the actual parameter. In Fig. 4.3 we show how many confidence intervals
computed from 100 different samples of 100 elements from our dataset contain
the real  population  mean.  If  this  simulation  could  be  done  with  infinite  different
samples, 5% of those intervals would not contain the true mean.

So,  when  faced  with  a  sample,  the  correct  interpretation  of  a  confidence

interval is as follows:

In 95% of the cases, when I compute the 95% confidence interval from this sample, the
true mean of the population will fall within the interval defined by these bounds: �1.96 �
SE.

We cannot say either that our specific sample contains the true parameter or
that the  interval  has  a  95%  chance  of  containing  the  true  parameter.  That
interpretation  would  not  be  correct  under  the  assumptions  of  traditional

statistics.

Hypothesis Testing

Giving  a  measure  of  the  variability  of  our  estimates  is  one  way  of  producing  a
statistical  proposition  about  the  population,  but  not  the  only  one.  R.A.  Fisher
(1890� 1962) proposed an alternative, known as hypothesis testing, that is based
on the concept of statistical significance.

Let us suppose that a deeper analysis of traffic accidents in Barcelona results in
a difference  between  2010  and  2013.  Of course,  the  difference  could  be  caused
only by chance, because of the variability of both estimates. But it could also be
the  case that  traffic  conditions  were  very  different  in  Barcelona  during  the  two
periods  and, because  of  that,  data  from  the  two  periods  can  be  considered  as
belonging  to  two  different  populations.  Then,  the  relevant  question  is:  Are  the
observed effects real or not?

Technically, the question is usually translated to:  Were the observed effects statis-

tically significant?

The process of determining the statistical significance of an  effect is called hypoth-

esis testing.

This process starts by simplifying the options into two competing hypotheses:

�

�

H0:  The  mean  number  of  daily  traffic  accidents is  the  same  in  2010  and  2013
(there  is  only  one  population,  one  true  mean,  and  2010  and  2013  are  just
different samples from the same population).
HA:  The  mean  number  of  daily  traffic  accidents  in  2010  and  2013  is  different
(2010 and 2013 are two samples from two different populations).

Fig.  4.3  This  graph  shows  100  sample  means  (green  points)  and  its  corresponding  confidence
intervals,  computed  from  100  different  samples  of  100  elements  from  our  dataset.  It  can  be
observed  that a few  of them (those in red)  do  not contain the mean of the population (black
horizontal line)

60

4    Statistical Inference

We call H0 the null hypothesis and it represents a skeptical point of view: the
effect we have observed is due to chance (due to the specific sample bias).  HA is
the alternative hypothesis and it represents the other point of view: the effect is
real.

The general rule of frequentist hypothesis testing: we will not discard H0 (and
hence  we  will  not  consider  HA)  unless  the  observed  effect  is  implausible  under
H0.

Testing Hypotheses Using Confidence Intervals

We can use the concept represented by confidence intervals to measure the
plausi- bility of a hypothesis.

We can illustrate the evaluation of the hypothesis setup by comparing the

mean rate of traffic accidents in Barcelona during 2010 and 2013:

In [7]:

data  =  pd . read_csv (" files / ch04 / A CC ID EN TS _G U_ BC N_ 20 10  . csv " ,
encoding = � latin -1 �)

#  Create  a  new  column  which  is  the  date
data [ � Date �]  =  data [ � Dia  de  mes �]. apply ( lambda  x:  str (x))

+  �-�  +
data [ � Mes  de  any �]. apply ( lambda  x:  str (x))

data2  =  data [ � Date �]
count s20 10  =  data [ � Date �]. valu e_c ount s ()
print  � 2010:  Mean �,  cou nts 2010  . mean ()

data  =  pd . read_csv (" files / ch04 / A CC ID EN TS _G U_ BC N_ 20 13  . csv " ,
encoding = � latin -1 �)

#  Create  a  new  column  which  is  the  date
data [ � Date �]  =  data [ � Dia  de  mes �]. apply ( lambda  x:  str (x))

+  �-�  +
data [ � Mes  de  any �]. apply ( lambda  x:  str (x))

data2  =  data [ � Date �]
count s20 13  =  data [ � Date �]. valu e_c ount s ()
print  � 2013:  Mean �,  cou nts 2013  . mean ()

Out[7]:  2010: Mean 24.8109
2013: Mean 25.9095

This estimate suggests that in 2013 the mean rate of traffic accidents in

Barcelona was higher than it was in 2010. But is this effect statistically significant?
Based on our  sample, the 95%  confidence interval for  the mean rate  of

traffic accidents in Barcelona during 2013 can be calculated as follows:

In [8]:

n  =  len ( c oun ts2 013  )
mean  =  coun ts201 3  . mean ()
s  =  coun ts201 3 . std ()
ci  =  [ mean  -  s *1.96/ np . sqrt ( n) ,  mean  +  s *1.96/ np . sqrt (n)]
print  � 2010  accident   rate  estimate : �,  cou nts20 10  . mean ()
print  � 2013  accident  rate  estimate  : �,  c ount s2013  . mean ()
print  � CI  for  2013: �,ci

4.4    Hypothesis Testing

61

Out[8]:  2010 accident rate estimate: 24.8109
2013 accident rate estimate: 25.9095

CI for 2013: [24.9751, 26.8440]

Because the 2010 accident rate estimate does not fall in the range of plausible
values of 2013, we say the alternative hypothesis cannot be discarded. That is, it
cannot be ruled out that in 2013 the mean rate of traffic accidents in Barcelona
was higher than in 2010.

Interpreting CI Tests

Hypothesis testing is built around rejecting or failing to reject the null hypothesis.
That is, we do not reject H0 unless we have strong evidence against it. But what
precisely does strong evidence mean? As a general rule of thumb, for those cases
where the null hypothesis is actually true, we do not want to incorrectly reject H0
more  than  5%  of  the  time.  This  corresponds  to  a  significance  level  of ?  0.05. In
this case, the correct interpretation of our test is as follows:

=

If we use a 95% confidence interval to test a problem where the null hypothesis is true,
we will make an error whenever the point estimate is at least 1.96 standard errors away
from the population parameter. This happens about 5% of the time (2.5% in each tail).

Testing Hypotheses Using  p-Values

A more advanced notion of statistical significance was developed by R.A. Fisher in
the  1920s  when  he  was  looking  for  a  test  to  decide  whether  variation  in  crop
yields was  due  to  some  specific  intervention  or  merely  random  factors  beyond
experimental control.

Fisher  first  assumed  that  fertilizer  caused no  difference  (null  hypothesis)  and
then calculated  P, the probability that an observed yield in a fertilized field would
occur if fertilizer had no real effect. This probability is called the  p-value.

The  p-value  is  the  probability  of  observing  data  at  least  as  favorable  to  the
alter- native hypothesis as our current dataset, if the null hypothesis is true. We
typically use  a  summary  statistic  of  the  data  to  help  compute  the  p-value  and
evaluate the hypotheses.

Usually, if P is less than 0.05 (the chance of a fluke is less than 5%) the result is

declared statistically significant.

It  must  be  pointed  out  that  this  choice  is  rather  arbitrary  and  should  not  be

taken as a scientific truth.

The goal of classical hypothesis testing is to answer the question, �Given a sample
and  an  apparent  effect,  what  is  the  probability  of  seeing  such  an  effect  by
chance?� Here is how we answer that question:
�

The  first  step  is  to  quantify  the  size  of  the  apparent  effect  by  choosing  a  test
statistic. In  our  case,  the  apparent effect  is  a  difference in accident  rates,  so  a
natural choice for the test statistic is the difference in means between the two
periods.

62

4  Statistical Inference

�

�

�

The second step is to define a null hypothesis, which is a model of the system
based on the assumption that the apparent effect is not real. In our case, the
null hypothesis is that there is no difference between the two periods.
The  third  step is  to  compute  a  p-value,  which is  the  probability  of  seeing  the
apparent  effect  if  the null  hypothesis  is  true. In  our case, we  would compute
the difference in means, then compute the probability of seeing a difference as
big, or bigger, under the null hypothesis.
The last step is to interpret the result. If the  p-value is low, the effect is said to
be statistically  significant,  which  means  that  it  is  unlikely  to  have  occurred  by
chance. In  this  case  we  infer  that  the  effect  is  more  likely  to  appear  in  the  larger
population.

In our case, the test statistic can be easily computed:

In [9]:

m=  len ( c o u n t s 2 0 1 0  )
n=  len ( c o u n t s 2 0 1 3  )
p  =  ( c o u n t s 2 0 1 3  . m e an  ()  -  c o u n t s 2 0 1 0  . m e a n  () )
print  �m: �,  m ,  �n: �,  n
p r i nt   � m e a n   d i f f e r e n c e  :  �,   p

Out[9]:  m: 365 n: 365

mean difference: 1.0986

To approximate the p-value , we can follow the following procedure:

1.  Pool the distributions, generate samples with size n and compute the

difference in the mean.

2.  Generate samples with size n and compute the difference in the mean.
3.  Count how many differences are larger than the observed one.

In [10]:

#  pool ing  di st r ib ut io n s
x  =  co un t s2 01 0
y  =  c ou n ts 20 1 3
po o l   =  np . c o n c a t e n a t e  ([ x ,  y ])
np . r a n d o m  . s h u f f l e  ( p o o l  )

# s a m p l e   g e n e r a t i o n
import  r andom
N  =  10000  #  number   of  sample s
di f f   =  r a n g e  ( N )
for  i  in  r a n g e  ( N ) :

p1  =  [ r a n d o m  . c h o i c e  ( p o o l  )  fo r  _  in  x r a n g e  ( n ) ]
p2  =  [ r a n d o m  . c h o i c e  ( p o o l  )  fo r  _  in  x r a n g e  ( n ) ]
di f f  [ i]  =  ( np . m e a n  ( p1 )  -  np . m e a n  ( p2 ))

Hypothesis Testing

In [11]:

#  counting   di ffere nces   larger  than  the  observed   one
diff2  =  np . array ( diff )
w1  =  np . where ( diff2  >  p) [ 0 ]

print  �p - value  ( Sim ula tio n  ) = �,  len (w1 )/ float (N) ,

�( �, len (w1 )/ float ( N) *100  ,�%) �,  � D iff ere nce   = �,  p

if  ( len (w1) / float ( N))  <  0.05:

print  � The  effect  is  likely �

else :

print  � The  effect  is  not  likely �

Out[11]: p-value  (Simulation)=  0.0485  (  4.85%)  Difference  =  1.098 The effect is likely

Interpreting P-Values

A p-value is the probability of an observed (or more extreme) result arising only
from chance.

If P is less than 0.05, there are two possible conclusions: there is a real effect
or the  result  is  an  improbable  fluke.  Fisher�s  method  offers  no  way  of  knowing
which is the case.

We must not confuse the odds of getting a result (if a hypothesis is true) with
the  odds  of  favoring  the  hypothesis  if  you  observe  that  result.  If  P  is  less  than
0.05, we cannot say that this means that it is 95% certain that the observed effect
is real and could not have arisen by chance. Given an observation E  and a hypothesis
|
H , P(E  H) and  P(H  E) are not the same!

|

Another  common  error  equates  statistical  significance

to  practical
importance/  relevance.  When  working  with  large  datasets,  we  can  detect
statistical significance for small effects that are meaningless in practical terms.

We have defined the effect as a difference in mean as large or larger than ?,

considering the sign. A test like this is called one sided.

If  the relevant question is  whether accident  rates  are different,  then it makes
sense to test the absolute difference in means. This kind of test is called two sided
because it counts both sides of the distribution of differences.

Direct Approach

The  formula  for  the  standard  error  of  the  absolute  difference  in  two  means  is
similar to the formula for other standard errors. Recall that the standard error of
a single mean can be approximated by:

S E x�1

  ?1

=  ?

n1

The standard error of the difference of two sample means can be constructed

from the standard errors of the separate sample means:

S E x�1 ?x�2   =

,
2
?
1
n1

2
?
2
n2

+

This would allow us to define a direct test with the 95% confidence interval.

But Is the Effect E Real?

    Statistical Inference

We  do  not  yet  have  an  answer  for  this  question!  We  have  defined  a  null
hypothesis H0 (the effect is not real) and we have computed the probability of the
observed effect under the null hypothesis, which is  P(E  H0), where E is an effect
as big as or bigger than the apparent effect and a  p-value .

|

|

We have stated that from the frequentist point of view, we cannot consider HA
unless  P(E  H0)  is  less  than  an  arbitrary  value.  But  the  real  answer  to  this  question
must  be  based  on  comparing P(H0  E)  to  P(HA  E),  not  on P(E  H0)! One  possi-  ble
solution  to  these  problems  is  to  use  Bayesian  reasoning;  an  alternative  to  the
frequentist approach.

|

|

|

No  matter  how  many  data  you  have,  you  will  still  depend  on  intuition  to
decide  how  to  interpret,  explain,  and  use  that  data.  Data  cannot  speak  by
themselves.  Data scientists  are  interpreters,  offering  one  interpretation  of  what
the useful narrative story derived from the data is, if there is one at all.

UNIT-3

Supervised Learning: First step, learning curves, training-validation and test.
Learning  models  generalities,  support  vector  machines,  random  forest.
Examples
Supervised Learning

Machine  learning  involves  coding  programs  that  automatically  adjust  their
perfor- mance  in  accordance  with  their  exposure  to  information  in  data.  This
learning is achieved via a parameterized model with tunable parameters that are
automatically  adjusted  according  to  different  performance  criteria.  Machine
learning  can  be  con- sidered  a  subfield  of  artificial  intelligence  (AI)  and  we  can
roughly divide the field into the following three major classes.

1.  Supervised  learning:  Algorithms  which  learn  from  a  training  set  of  labeled
examples (exemplars) to generalize to the set of all possible inputs. Examples
of  techniques  in  supervised  learning:  logistic  regression,  support  vector
machines, decision trees, random forest, etc.

2.  Unsupervised learning: Algorithms that learn from a training set of unlabeled
examples.  Used  to  explore  data  according  to  some  statistical,  geometric  or
sim-  ilarity  criterion.  Examples  of  unsupervised  learning  include  k-means
clustering  and  kernel  density  estimation.  We  will  see  more  on  this  kind  of
techniques in Chap. 7.

3.  Reinforcement

learning:  Algorithms  that

learn  via  reinforcement  from
criticism that provides information on the quality of a solution, but not on how
to  improve it.  Improved  solutions  are  achieved  by  iteratively  exploring  the
solution space.

This chapter focuses on a particular class of supervised machine learning: clas-
sification. As a data scientist, the first step you apply given a certain problem is to
identify  the  question  to  be  answered.  According  to  the  type  of  answer  we  are
seeking, we are directly aiming for a certain set of techniques.

    Supervised Learning

�

If our question is answered by YES/NO, we are facing a classification problem.
Classifiers are also the tools to use if our question admits only a discrete set of
answers, i.e., we want to select from a finite number of choices.

�  Given the results of a clinical test, e.g., does this patient suffer from diabetes?
�  Given a magnetic resonance image, is it a tumor shown in the image?
�  Given the past activity  associated with  a credit card, is  the current

operation fraudulent?

�

If  our  question  is  a  prediction  of  a  real-valued  quantity,  we  are  faced  with  a

regres- sion problem. We will go into details of regression in Chap. 6.

�  Given the description of an apartment, what is the expected market value of

the flat? What will the value be if the apartment has an elevator?

�  Given the past records of user activity on Apps, how long will a certain client

be connected to our App?

�  Given my skills and marks in computer science and maths, what mark will I

achieve in a data science course?

Observe  that  some  problems  can  be  solved  using  both  regression  and
classification.  As  we  will  see  later,  many  classification  algorithms  are  thresholded
regressors.  There is  a  certain  skill  involved  in  designing  the  correct  question  and
this dramatically affects the solution we obtain.

The Problem

In this chapter we use data from the Lending Club1 to develop our understanding
of  machine  learning  concepts.  The  Lending  Club  is  a  peer-to-peer  lending
company.  It  offers  loans  which  are  funded  by  other  people.  In  this  sense,  the
Lending Club acts as a hub connecting borrowers with investors. The client applies
for a loan of a certain amount, and the company assesses the risk of the operation.
If the application is  accepted,  it  may  or  may  not  be  fully  covered.  We  will  focus
on  the  prediction of whether the loan will be fully funded, based on the scoring
of and information related to the application.

We  will  use  the  partial  dataset  of  period  2007�2011.  Framing  the  problem  a
little bit  more,  based  on  the  information  supplied  by  the  customer  asking  for  a
loan, we want to predict whether it will be granted up to a certain threshold thr . The
attributes  we  use  in  this  problem  are related  to  some  of  the  details  of  the  loan
application,  such as  amount  of  the  loan  applied  for  the  borrower,  monthly
payment  to  be  made  by  the  borrower  if  the  loan  is  accepted,  the  borrower�s
annual income, the number of  incidences of delinquency in the borrower�s credit
file, and interest rate of the loan, among others.

In  this  case  we  would  like  to  predict  unsuccessful  accepted  loans.  A  loan
applica- tion is unsuccessful if the funded amount (funded_amnt) or the amount
funded  by  investors  (funded_amnt_inv)  falls  far  short  of  the  requested  loan
amount (loan_amnt). That is,

loan ?  f unded
loan

0.95.

?

First Steps

Note that in this problem we are predicting a binary value: either the loan is fully
funded  or  not.  Classification  is  the  natural  choice  of  machine  learning  tools  for
prediction  with  discrete  known  outcomes.  According  to  the  cardinality  of  the
target set,  one  usually  distinguishes  between  binary  classifiers  when  the  target
output only takes two values, i.e., the classifier answers questions with a yes or a no;
or  multiclass  classifiers,  for  a larger  number  of  classes.  This  issue  is  important  in
that not all methods can naturally handle the multiclass setting.2

:  ?

In a formal way, classification is regarded as the problem of finding a function
h(x)  Rd      K that maps an input space in Rd  onto a discrete set of k target outputs
or  classes  K      1 , . . . ,  k  . In this  setting, the features  are  arranged  as  a  vector x  of d
real-valued numbers.3

= {

}

We  can  encode  both  target  states  in  a  numerical  variable,  e.g.,  a  successful

loan target can take value 1; and it is 1, otherwise.

+

?

Let us check the dataset,4

import  pickle
ofname  =  open ( �./ files / ch05 / data set_s mall  . pkl �,�rb �)
#  x  stores  input  data  and  y  target  values
(x , y)  =  pickle . load ( ofname )

In [1]:

2Several  well-known  techniques  such  as  support  vector  machines  or  adaptive  boosting
(adaboost) are  originally  defined  in  the  binary  case.  Any  binary  classifier  can  be  extended  to  the
multiclass  case  in  two  different  ways.  We  may  either  change  the  formulation  of  the
learning/optimization process. This requires the derivation of a new learning algorithm capable
of  handling  the  new  modeling.  Alternatively,  we  may  adopt  ensemble  techniques.  The  idea
behind  this  latter  approach  is  that  we may  divide  the  multiclass  problem  into  several  binary
problems;  solve  them;  and  then  aggregate  the results.  If  the  reader  is  interested  in  these
techniques,  it  is  a  good  idea  to  look  for:  one-versus-all,  one-versus-one,  or  error  correcting
output codes methods.
3Many problems are described using categorical data. In these cases either we need classifiers that
are  capable  of  coping  with  this  kind  of  data  or  we  need  to  change  the  representation  of  those
variables into numerical values.
4The notebook companion shows the preprocessing steps, from reading the dataset, cleaning and
imputing data, up to saving a subsampled clean version of the original dataset.

A problem in Scikit-learn is modeled as follows:

�

Input data is structured in Numpy arrays. The size of the array is expected to be
[n_samples, n_features]:

�  n_samples:  The  number  of  samples  (n).  Each  sample  is  an  item  to  process
(e.g., classify). A sample can be a document, a picture, an audio file, a video,
an astronomical object, a row in a database or CSV file, or whatever you can
describe with a fixed set of quantitative traits.

�  n_features: The number of features (d) or distinct traits that can be used to
describe  each  item  in  a  quantitative  manner.  Features  are  generally  real-
valued, but may be Boolean, discrete-valued or even categorical.

feature matrix :  X  =

?

?

x11  x12  � � � x1d
? x21  x22  � � � x2d
?
? x31  x32  � � � x3d
. . .
?
.
.
.
?
?
. . .   .
?
?
.
xn1  xn2  � � �  xnd

.

label vector :  yT  =  [y1, y2, y3, � � �  yn]

The number of features must be fixed in advance. However, it can be very

great (e.g., millions of features).

In [2]:

dims  =  x. shape [ 1 ]
N  =  x. shape [ 0 ]
print  � dims :  �  +  str ( dims )  +  �,  samp les :  �  +  str (N)

Out[2]:  dims: 15, samples: 4140

Considering data arranged as in the previous matrices we refer to:

�

the columns as features, attributes, dimensions, regressors, covariates,
predictors, or independent variables;

�  the rows as instances, examples, or samples;
�  the target as the label, outcome, response, or dependent variable.

All objects in Scikit-learn share a uniform and limited API consisting of three

complementary interfaces:

�  an estimator interface for building and fitting models (fit());
�  a predictor interface for making predictions (predict());
�  a transformer interface for converting data (transform()).

Let us apply a classifier using Python�s Scikit-learn libraries,

In [3]:

from  sklearn   import  nei ghbo rs
from  sklearn   import  datasets
#  Create  an  insta nce   of  K - nearest   n eig hbo r  c las si fier
knn  =  nei ghb ors  . K N ei g hb o r sC l a ss i f ie r  ( n _ne igh bor s   =  11 )
#  Train  the  cla ssif ier
knn . fit ( x ,  y)
# Compute  the predi ction  acc ordin g  to the model
yhat  =  knn . predict  ( x)
#  Check  the  result  on  the  last  example
print  � Pre dic ted   value :  �  +  str ( yhat [ -1]) ,
�,  real  target  :  �  +  str ( y [ -1])

Out[3]:  Predicted value: -1.0 , real target: -1.0

The basic measure of  performance of  a classifier is its  accuracy. This is defined  as
the number of correctly predicted examples divided by the total amount of examples.
Accuracy is related to the error as follows: acc = 1 ? err .

acc

=

Number of correct predictions
n

Each  estimator  has  a  score() method  that  invokes  the  default  scoring  metric.

In the case of k-nearest neighbors, this is the classification accuracy.

In [4]:

knn . s c o re  (x , y )

Out[4]:  0.83164251207729467

It looks like a really good result. But how good is it? Let us first understand a little

bit more about the problem by checking the distribution of the labels.

Let us load the dataset and check the distribution of labels:

In [5]:

plt . pie ( np . c_ [ np . sum ( np . where ( y  ==  1 ,  1 ,  0) ) ,

np . sum ( np . where ( y  ==  -1 ,  1 ,  0) ) ][0] ,
labels  =  [ � Not  fully  funded �,� Full  amount �],
colors  =  [ �r �,  �g �], shadow  =  False ,
autopct  =  � %.2 f �  )
plt . gcf () . s et _s iz e _i nc he s  ((7 ,  7) )

with the result observed in Fig. 5.1.

Note  that  there  are far  more  positive  labels  than  negative  ones.  In  this  case,
the  dataset  is  referred  to  as  unbalanced.5  This  has  important  consequences  for  a
classifier as  we  will  see later  on.  In  particular,  a  very  simple  rule  such  as  always
predict the

5The  term  unbalanced  describes  the  condition  of  data  where  the  ratio  between  positives  and
negatives is a small value. In these scenarios, always predicting the majority class usually yields
accurate performance, though it is not very informative. This kind of problems is very common
when  we  want  to  model  unusual  events  such  as  rare  diseases,  the  occurrence  of  a  failure  in
machinery, fraudulent credit card operations, etc. In these scenarios, gathering data from usual
events  is  very  easy  but  collecting  data  from  unusual  events
in  a
comparatively small dataset.

is  difficult  and  results

Fig.  5.1  Pie  chart  showing
the  distribution  of  labels
in the dataset

majority class, will give us good performance. In our problem, always predicting
that  the  loan  will  be  fully  funded  correctly  predicts  81.57%  of  the  samples.
Observe that this value is very close to that obtained using the classifier.

Although  accuracy  is  the  most  normal  metric  for  evaluating  classifiers,  there
are cases when the business value of correctly predicting elements from one class
is  different  from  the  value  for  the  prediction  of  elements  of  another  class.  In
those  cases,  accuracy  is  not  a  good  performance  metric  and  more  detailed
analysis  is  needed.  The  confusion  matrix  enables  us  to  define  different  metrics
considering  such  scenarios.  The  confusion  matrix  considers  the  concepts  of  the
classifier  outcome  and  the  actual  ground  truth  or  gold  standard.  In  a  binary
problem, there are four possible cases:

�

�

�

�

True positives (TP): When the classifier predicts a sample as positive and it really
is positive.
False positives (FP): When the classifier predicts a sample as positive but in fact
it is negative.
True negatives (TN): When the classifier predicts a sample as negative and it really
is negative.
False negatives (FN): When the classifier predicts a sample as negative but in fact
it is positive.

We can summarize this information in a matrix, namely the confusion matrix,

as follows:

5.3    First Steps

73

Gold Standard
Positive  Negative

Positive
Prediction Negative

TP
FN
?
itivity
?

Sens

FP
TN

? Precision
? Negative Predictive Value

Spec

ificity (Recall)

The combination of these elements allows us to define several performance metrics:

�  Accuracy:

accuracy

=

TP + TN

TP + TN + FP + FN

�  Column-wise we find these two partial performance metrics:

�  Sensitivity or Recall:

�  Specificity:

sensitivity

=

TP

Real Positives

specificity

=

TN

Real Negatives

TP

=

TP + FN

TN

=

TN + FP

�  Row-wise we find these two partial performance metrics:

�  Precision or Positive Predictive Value:

precision

=

�  Negative predictive value:

TP
Predicted Positives

=

TP

TP + FP

NPV

TN

=

Predicted Negative

TN

=

TN + FN

These  partial  performance  metrics  allow  us  to  answer  questions  concerning
how  often  a  classifier  predicts  a  particular  class,  e.g.,  what  is  the  rate  of
predictions for  not  fully  funded  loans  that  have  actually  not  been fully funded?
This  question  is  answered  by  recall.  In  contrast,  we  could  ask:  Of  all  the  fully
funded loans predicted by  the  classifier,  how  many  have  been  fully  funded?  This  is
answered by the precision metric.

Let us compute these metrics for our problem.

74

In [6]:

5  Supervised Learning

yh a t   =  knn . p r e d i c t  ( x)
TP  =  np . sum ( np . l o g i c a l _ a n d  ( y h a t   ==  -1 ,  y  == -1) )
TN  =  np . sum ( np . l o g i c a l _ a n d  ( y h at   ==  1 ,  y  ==  1) )
FP  =  np . sum  ( np . l o g i c a l _ a n d  ( y h a t   ==  -1 ,  y  ==  1) )
FN  =  np . s um ( np . l o g i c a l _ a n d  ( y h a t   ==  1 ,  y  ==  -1) )
p r i nt   � TP :  � +  str  ( T P ) ,   �,  FP :  � +  st r ( F P )
p r i nt   � FN  :  � +  str ( F N ) ,   �,  TN :  � +  str ( T N )

Out[6]:  TP:  3370  , FP:  690

FN: 7 ,

TN: 73

Scikit-learn provides us with the confusion matrix,

In [7]:

from  sklearn   import  metrics
metrics . co nf us io n _m at ri x  ( yhat ,  y)
#  sklearn uses a  tra nspo sed c onven tion  for  the  co nfusion
#  matrix  thus  I  change  targets   and  pre dict ions

Out[7]:  3370, 690

7,

73

Let us check the following example. Let us select a nearest neighbor classifier
with the number of neighbors equal to one instead of eleven, as we did before,
and check the training error.

In [8]:

#  Train  a  class ifier   using  . fit ()
knn = ne igh bor s  . K N ei g h bo r s Cl a s si f i er  ( n _ne igh bor s   =  1)
knn . fit ( x ,  y)
yhat  =  knn . pred ict  ( x)

print  " c lass ifica tion   accuracy :"  +

str ( metrics . accu racy_ score  ( yhat ,  y))

print  " confusion  matrix :  \ n"  +

str ( metr ics  . co nf u si on _m at ri x  ( yhat ,  y))

Out[8]:  classification accuracy: 1.0 confusion matrix:

3377  0
0   763

The performance measure is perfect! 100% accuracy and a diagonal confusion
matrix! This looks good. However, up to this point we have checked the classifier
performance  on  the  same  data  it has  been  trained  with. During  exploitation,  in
real applications,  we  will  use  the  classifier  on  data  not  previously  seen.  Let  us
simulate this  effect  by  splitting  the  data  into  two  sets:  one  will  be  used  for
learning (training set) and the other for testing the accuracy (test set).

5.3    First Steps

75

In [9]:

#  Simulate  a  real  case :  Ran domiz e  and  split  data  into
#  two  sub sets  PRC *100\%  for  tra ining   and  the  rest
#  (1 - PRC ) *100\%  for  t esting
perm  =  np . random . perm utati on  ( y. size )
PRC  =  0.7
split_p oint   =  int ( np . ceil ( y. shape [0]* PRC ))

X_train  =  x[ perm [: spl it_po int  ]. ravel () ,:]
y_train  =  y[ perm [: spl it_po int  ]. ravel () ]

X_test  =  x[ perm [ s plit_ point  :]. ravel () ,:]
y_test  =  y[ perm [ sp lit_p oint  :]. ravel () ]

If we check the shapes of the training and test sets we obtain,

Out[9]:  Training shape: (2898, 15), training targets shape: (2898,)

Testing shape: (1242, 15), testing targets shape: (1242,)

With this new partition, let us train the model

In [10]:

# Train  a  clas sifie r   on  tra ining   data
knn = ne igh bor s  . K N ei g h bo r s Cl a s si f i er  ( n _ne igh bor s   =  1)
knn . fit ( X_train  ,  y_ train  )
yhat  =  knn . predict ( X_tra in )

print  "\ n  TRAI NING   STATS :"
print  " c lass ifica tion   accuracy :"  +

str ( metrics  . ac curac y_sc ore  ( yhat ,  y_train  ))

print  " confusion  matrix :  \ n"  +

str ( metrics  . c on fu si o n_ ma tr ix  ( y_train ,  yhat ))

Out[10]: TRAINING  STATS:

classification accuracy: 1.0
confusion matrix:
2355  0
0   543

As expected from the former experiment, we achieve a perfect score. Now let

us see what happens in the simulation with previously unseen data.

In [11]:

# C h ec k   on  t he  t e s t   set
yh a t   =  knn . p r e d i c t  ( X _ t e s t  )
print  " TESTIN G  STATS :"
print  " cl a ss if i ca ti on   a cc ura cy :" ,

metrics  . ac cu r ac y_ s co re  ( yhat ,  y_test  )

p r i nt   " c o n f u s i o n   m a t r i x  :  \ n "  +

str ( me trics  . c o n f u s i o n _ m a tr i x  ( yhat ,  y_t es t ))

Out[11]: TESTING  STATS:

classification accuracy: 0.754428341385
confusion matrix:
865 148
157 72

76

5  Supervised Learning

Observe  that  each  time  we run  the  process  of  randomly  splitting  the  dataset
and  train  a  classifier  we  obtain  a  different  performance.  A  good  simulation  for
approxi- mating the test error is to run this process many times and average the
performances. Let us do this!6

In [12]:

#  Sp itting   done  by  using  the  tools  provide d   by  sk learn :
from  sklearn  . c ro ss _v al id at i on   import  t ra in _t es t _s p li t

PRC  =  0.3
acc = np . zeros ((10 ,) )
for  i  in  xrange (10) :

X_train ,  X_test ,  y_train  ,  y_test  =

tr ai n_ te st _s pl i t  ( x ,  y ,  tes t_s ize   =  PRC )

knn = ne igh bor s  . K N ei g h bo r s Cl a s si f i er  ( n _ne igh bor s   =  1)
knn . fit ( X_train  ,  y_ train  )
yhat  =  knn . predict  ( X_test )
acc [ i]  =  metrics . a ccur acy_s core  ( yhat ,  y_test )

acc . shape  =  (1 ,  10 )
print  " Mean  expected   error :"  +  str ( np . mean ( acc [0]) )

Out[12]: Mean  expected  error:  0.754669887279

As we can see, the resulting error is below 81%, which was the result of the

most naive decision process. What is wrong with this result?

Let us introduce the nomenclature for the quantities we have just computed

and define the following terms.

�

In-sample error  Ein: The in-sample error or training error is the error
measured over all the observed data samples in the training set, i.e.,

   1
E
=in

N

N

e(x  , y  )

i

i

�

i =1
Out-of-sample error Eout: The out-of-sample error or generalization error mea-
sures  the  expected  error  on  unseen  data.  We  can  approximate/simulate  this
quantity by holding back some training data for testing purposes.

Eout  =  Ex,y(e(x, y))

Note  that  the  definition  of  the  instantaneous  error  e(xi  ,  yi  )  is  still  missing.  For
example,  in  classification  we  could  use  the  indicator  function  to  account  for  a
cor- rectly classified sample as follows:

e(x  , y  ) = I [h(x  ) = y  ] =

i

i

i

i

1, if  h(xi ) = yi

0 otherwise.

6sklearn    allows    us    to    easily    automate    the    train/test    splitting    using    the    function
train_test_split(...).

First Steps

77

Fig. 5.2  Comparison  of the  methods  using the accuracy  metric

Observe that:

Eout  ? Ein

Using  the  expected  error  on  the  test  set,  we  can  select  the  best  classifier
for our  application.  This  is  called  model  selection.  In  this  example  we  cover  the
most simplistic setting. Suppose we have a set of different classifiers and want to
select the �best� one. We may use the one that yields the lowest error rate.

In [13]:

from  sklearn  import  tree
from  sklearn   import  svm
PRC  =  0.1
acc_r  =  np . zeros ((10 ,  4) )
for  i  in  xrange (10) :

X_train ,  X_test ,  y_train  ,  y_test  =

tr ai n_ te st _s pl i t  ( x ,  y ,  tes t_s ize   =  PRC )

nn 1  =  nei ghb ors  . KN e i gh b o rs C l as s i fi e r  ( n _ne igh bors   =  1)
nn 3  =  nei ghb ors  . K N e ig h bo r s Cl a s si f i er  ( n_n eig hbor s   =  3)
svc  =  svm . SVC ()
dt  =  tree . D ec i si on T re e Cl as s if i er  ()

nn1 . fit ( X_train  ,  y_train  )
nn3 . fit ( X_train  ,  y_train  )
svc . fit ( X_train  ,  y_train  )
dt . fit ( X_train  , y_t rain )

yhat_nn1   =  nn 1 . p redict ( X_test )
yhat_nn3   =  nn 3 . p redict ( X_test )
yhat_svc   =  svc . pr edict  ( X_test )
yhat_dt   =  dt . predict  ( X_test )

acc_r [ i ][0]  =  metrics . a ccur acy_s core  ( yhat_nn1 ,  y_test )
acc_r [ i ][1]  =  metrics . a ccura cy_sc ore  ( yhat_nn3 ,  y_test )
acc_r [ i ][2]  =  metrics . a ccura cy_sc ore  ( yhat_svc ,  y_test )
acc_r [ i ][3]  =  metrics  . a ccura cy_sc ore  ( yhat_dt ,  y_test )

Figure 5.2  shows  the  results  of  applying  the  code.

78

5    Supervised Learning

This  process  is  one  particular  form  of  a  general  model  selection  technique
called cross-validation.  There  are  other  kinds  of  cross-validation,  such  as  leave-
one-out or K-fold cross-validation.

�

�

In leave-one-out, given N samples, the model is trained with N 1 samples and
tested  with  the  remaining  one.  This  is  repeated  N  times,  once  per  training
sample and the result is averaged.
In K-fold cross-validation, the training set is divided into K nonoverlapping splits.
K-1 splits are used for training and the remaining one used to assess the mean.
This process is repeated K times leaving one split out each time. The results are
then averaged.

?

What Is Learning?

Let us recall the two basic values defined in the last section. We talk of  training error
or in-sample error,  Ein, which refers to the error measured over all the observed
data samples in the training set. We also talk of test error or generalization error,
Eout, as the error expected on unseen data.

We  can  empirically  estimate  the  generalization  error  by  means  of  cross-

validation techniques and observe that:

Eout  ? Ein.

The  goal  of  learning  is  to  minimize  the  generalization  error;  but  how  can  we

guarantee this minimization using only training data?

From the above inequality it is easy to derive a couple of very intuitive ideas.

�  Because Eout is greater than or equal to Ein, it is desirable to have

�

Ein ? 0.
Additionally,  we  also  want  the  training  error  behavior  to  track  the
generalization error  so  that  if  one  minimizes  the  in-sample  error  the  out-of-
sample error follows, i.e.,

Eout  ? Ein.

We can rewrite the second condition as

Ein  ? Eout  ? Ein + ?,

with ?

?

0.

We would like to characterize ? in terms of our problem parameters, i.e.,

the number of samples (N ), dimensionality of the problem (d), etc.

Statistical analysis offers an interesting characterization of this quantity7

7The  reader  should  note  that  there  are  several  bounds  in  machine  learning  to  characterize  the
generalization error. Most of them come from variations of Hoeffding�s inequality.

Learning?

Fig. 5.3  Toy problem data

What Is

79

,

E

?
E  (C)

log C

,

+ Oout

N

where C is a measure of the complexity of the model class we are using. Technically,
we may also refer to this model class as the hypothesis space.

in

Learning Curves

Let  us  simulate  the  effect  of  the  number  of  examples  on  the  training  and  test
errors for a given complexity. This curve is called the learning curve. We will focus
for a moment in a more simple case. Consider the toy problem in Fig. 5.3.

Let us take a classifier and vary the number of examples we feed it for training
purposes,  then  check  the  behavior  of  the  training  and  test  accuracies  as  the
number of examples grows. In this particular case, we will be using a decision tree
with fixed maximum depth.

Observing  the  plot  in  Fig. 5.4,  we  can  see  that:

�
�

As the number of training samples increases, both errors tend to the same
value. When we have few training data, the training error is very small but the
test error is very large.

Now check the learning curve when the degree of complexity is greater in Fig. 5.5.

We simulate this effect by increasing the maximum depth of the tree.

And  if  we  put  both  curves  together,  we  have  the  results  shown  in  Fig. 5.6.
Although both show similar behavior, we can note several differences:

80

5  Supervised Learning

Fig. 5.4  Learning curves (training and test  errors) for a model with a  high degree  of complexity

Fig. 5.5  Learning curves (training and test  errors)  for a  model  with a  low degree of complexity

Fig. 5.6  Learning curves (training and test errors) for models with a low and a high degree of
complexity

Learning

Curves

81

Fig. 5.7  Learning  curves (training and  test errors)  for a  fixed number  of data  samples, as  the
complexity of the decision tree increases

�

�

With a low degree of complexity, the training and test errors converge to the
bias sooner/with fewer data.
Moreover, with  a low  degree of complexity,  the error of convergence is larger
than with increased complexity.

The value both errors converge towards is also called the bias; and the differ-
ence  between  this  value  and  the  test  error  is  called  the  variance.  The
bias/variance decomposition  of  the  learning  curve  is  an  alternative  approach  to
the training and generalization view.

Let us now plot the learning behavior for a fixed number of examples with
respect to the complexity of the model. We may use the same data but now we
will change the maximum depth of the decision tree, which governs the complexity
of the model. Observe  in  Fig. 5.7  that  as  the  complexity  increases  the  training  error
is  reduced; but above a certain level of complexity, the test error also increases.
This effect is

called overfitting. We may enact several cures for overfitting:

�

�

Observe  that  models  are  usually  parameterized  by  some  hyperparameters.
Select- ing the complexity is usually governed by some such parameters. Thus,
we are faced with a model selection problem. A good heuristic for selecting the
model is to  choose  the  value  of  the  hyperparameters  that  yields  the  smallest
estimated test error. Remember that this can be done using cross-validation.
We may also change the formulation of the objective function to penalize complex
models. This is called regularization. Regularization accounts for estimating the
value of ? in our out-of-sample error inequality. In other words, it models the
complexity of the technique. This usually becomes implicit in the algorithm but
has  huge  consequences in real  applications.  The most  common regularization
strategies are as follows:

82

5    Supervised Learning

�  L2 weight regularization: Adding an L2 penalization term to the weights of a
weight-controlled  model  implies  looking  for  solutions  with  small  weight  values.
Intuitively, adding an L2 penalization term can be seen as a surrogate for the
notion  of  smoothness.  In  this  sense,  a  low  complexity  model  means  a  very
smooth model.

�  L1 weight regularization: Adding an L1 regularization term forces sparsity in
the  weights  of  the  model.  In  this  sense,  a  low  complexity  model  means  a
model with few components or few active terms.

These terms are added to the objective function. They trade off with the error
function in the objective and are governed by a hyperparameter. Thus, we still
have to select this parameter by means of model selection.
We can use �ensemble techniques�. A third cure for overfitting is to use ensemble
techniques. The best known are bagging and boosting.

�

Training, Validation and Test

Going back to our problem, we have to select a model and control its complexity
according  to  the  number  of  training  data.  In  order  to  do  this,  we  can  start  by
using a  model  selection  technique.  We  have  seen  model  selection  before  when  we
wanted to compare the performance of different classifiers. In that case, our best
bet  was  to select  the  classifier  with  the  smallest  Eout.  Analogous  to  model
selection,  we  may  think  of  selecting  the  best  hyperparameters  as  choosing  the
classifier  with  parameters  that  performs  the  best.  Thus,  we  may  select  a  set  of
hyperparameter values and use cross-validation to select the best configuration.

The  process  of  selecting  the  best  hyperparameters  is  called  validation.  This
intro- duces a new set into our simulation scheme; we now need to divide the data
we have into three sets: training, validation, and test sets. As we have seen, the
process  of  assessing  the  performance  of  the  classifier  by  estimating  the
generalization  error  is called testing.  And  the process  of  selecting  a model  using
the estimation of the gen- eralization error is called validation. There is a subtle but
critical difference between the  two  and  we  have  to  be  aware  of  it when  dealing
with our problem.

�

�

Test  data  is  used  exclusively  for  assessing  performance  at  the  end  of  the
process and will never be used in the learning process.8
Validation data is used explicitly to select the parameters/models with the best
performance  according  to  an  estimation  of  the  generalization  error.  This  is  a
form of learning.

�  Training data are used to learn the instance of the model from a model class.

.

Validation and Test

Training,

83

In  practice,  we  are  just  given  training  data,  and in  the most  general case  we
explicitly  have  to  tune  some  hyperparameter.  Thus,  how  do  we  select  the
different splits?

How  we  do  this  will  depend  on  the  questions regarding  the  method  that  we

want to answer:

�

�

Let us say that our customer asks us to deliver a classifier for a given problem. If
we just want to provide the best model, then we may use cross-validation on
our training  dataset  and  select  the  model  with  the  best  performance.  In  this
scenario, when we return the trained classifier to our customer, we know that it
is the one that achieves the best performance. But if the customer asks about
the expected performance, we cannot say anything.
A  practical  issue:  once  we  have  selected  the  model,  we  use  the  complete
training set to train the final model.
If  we  want  to  know  about  the  performance  of  our  model,  we  have  to  use
unseen data. Thus, we may proceed in the following way:

1.  Split the original dataset into training and test data. For example, use 30% of
the original dataset for testing purposes. This data is held back and will only
be used to assess the performance of the method.

2.  Use the remaining training  data  to  select  the  hyperparameters  by  means  of

cross- validation.

3.  Train  the  model  with  the  selected  parameter  and  assess  the  performance

using the test dataset.

A  practical  issue:  Observe  that  by  splitting  the  data  into  three  sets,  the
classifier is trained with a smaller fraction of the data.

�

If  we  want  to  make  a  good  comparison  of  classifiers  but  we  do  not  care
about the best parameters, we may use nested cross-validation. Nested cross-
validation runs  two  cross-validation  processes.  An  external  cross-validation  is
used to assess the performance of the classifier and in each loop of the external
cross-validation another  cross-validation  is  run  with  the  remaining  training  set
to select the best parameters.

If  we  want  to  select  the  best  complexity  of  a  decision  tree,  we  can  use  tenfold
cross- validation  checking  for  different  complexity  parameters.  If  we  change  the
maximum depth of the method, we obtain the results in Fig. 5.8.

84

5  Supervised Learning

Fig. 5.8  Box  plot showing accuracy for  different  complexities  of the  decision tree

In [14]:

#  Create  a  10 - fold  cross - va lidat ion  set
kf  =  c ro s s_ va li da ti on  . KFold ( n  =  y. shape [0] ,

n_folds   =  10 ,
shuffle  =  True ,
random_ state   =  0)

#  Search  for  the  param eter   among  the  f oll owi ng  :
C  =  np . arange (2 ,  20 ,)

acc  =  np . zeros ((10 ,  18 ) )
i  =  0
for  train_index  ,  va l_ind ex  in  kf :

X_train  ,  X_val  =  X[ t rai n_i nde x  ],  X[ va l_i nde x  ]
y_train  ,  y_val  =  y[ tr ain _in dex  ],  y[ v al_ ind ex  ]
j  =  0
for  c  in  C:

dt  =  tree . De ci si onT re eCl as sif ie r  (

mi n_ sa mp le s_ le a f   =  1 ,
max_dep th   =  c)

dt . fit ( X_train  ,  y_tr ain  )
yhat  =  dt . predict  ( X_val )
acc [ i ][ j]  =  metrics  . ac curac y_sco re  ( yhat ,  y_val )
j  =  j  +  1

i  =  i  +  1

Checking  Fig. 5.8,  we  can  see  that  the  best  average  accuracy  is  obtained  by
the fifth  model,  a  maximum  depth  of  6.  Although  we  can  report  that  the  best
accuracy  is  estimated  to  be  found  with  a complexity  value  of  6,  we  cannot  say
anything  about the  value  it  will  achieve.  In  order  to  have  an  estimation  of  that
value, we need to run the model on a new set of data that are completely unseen,
both  in  training  and  in  model  selection  (the  model  selection  value  is  positively
biased). Let us put everything together. We will be considering a simple train_test
split for testing purposes and then run cross-validation for model selection.

In [15]:

#  Train _test   split
X_train ,  X_test ,  y_train ,  y_test  =  c ro ss _v a li da ti on

. t ra in _t es t_ s pl it  ( X ,  y ,  tes t_s ize   =  0.20)

#  Create  a  10 - fold  cross - va lidat ion  set
kf  =  c ro ss _v al id a ti on  . KFold ( n  =  y_train . shape [0] ,

n_folds   =  10 ,
shuffle  =  True ,
random_ state   =  0)

#  Search  the  parameter  among  the  foll owin g
C  =  np . arange (2 ,  20 ,)
acc  =  np . zeros ((10 ,  18 ) )
i  =  0
for  train_index  ,  va l_ind ex  in  kf :

X_t ,  X_val  =  X_train  [ tr ain_i ndex  ],  X_train  [ v al_in dex  ]
y_t ,  y_val  =  y_train  [ t rain_ index  ],  y_train  [ val_ in dex  ]
j  =  0
for  c  in  C:

dt  =  tree . De ci si onT re eCl as sif ie r  (

mi n_ sa mp le s_ le a f   =  1 ,
max_dep th  =  c)

dt . fit ( X_t ,  y_t )
yhat  =  dt . predict  ( X_val )
acc [ i ][ j]  =  metrics . accur acy_s core  ( yhat ,  y_val )
j  =  j  +  1

i  =  i  +  1

print  � Mean  accuracy  :  �  +  str ( np . mean ( acc ,  axis  =  0) )
print  � Selected   model  index :  �  +

str ( np . argmax ( np . mean ( acc ,  axis  =  0) ))

Out[15]: Mean  accuracy:  [0.8254832  0.83031158  0.83091854  0.83423816
0.83363939 0.83303516 0.82759983 0.82337022 0.82034725
0.81642795 0.80947567 0.79951316 0.80162614 0.79226695
0.79589324 0.785928 0.78049267 0.78320988]
Selected model index: 3

If  we  run  the  output  of  this  code,  we  observe  that  the  best  accuracy  is
provided  by  the  fourth model. In  this  example  it  is  a model  with  complexity  5.9
The selected model achieves a success rate of 0.83423816 in validation. We then
train the model with the complete training set and verify its test accuracy.

In [16]:

#  Train  the  model  with  the  complete   training   set  with  the

selecte d   comp lexit y

dt  =  tree . D eci si on Tre eC las si fi er  (

mi n_ sa mp le s_ le a f   =  1 ,
max_depth  =  C[ np . argmax ( np . mean ( acc ,  axis  =  0) ) ])

dt . fit ( X_train , y_train  )

#  Test  the  model  with  the  test  set
yhat  =  dt . predict  ( X_test )
print  � Test  accuracy :  �  +

str ( metrics  . ac curac y_sc ore  ( yhat ,  y_test ))

Out[16]: Test  accuracy:  0.826086956522

As expected, the value is slightly reduced; it achieves 0.82608. Finally, the model
is trained with the complete dataset. This will be the model used in exploitation
and we expect to at least achieve an accuracy rate of 0.82608.

In [17]:

#  Train  the  final  model
dt  = tree . D ec is i on Tr ee Cl as si fi e r  ( mi n_ sa mp le s _l ea f   =  1 ,
max_depth  =  C[ np . argmax ( np . mean ( acc ,  axis  =  0) ) ])

dt . fit ( X ,  y)

Two  Learning  Models

Let  us  return  to  our  problem  and  check  the  performance  of  different  models.
There are  many learning  models  in  the machine  learning literature.  However, in
this short introduction we focus on two of the most important and pragmatically
effective approaches10: support vector machines (SVM) and random forests (RF).

Generalities  Concerning  Learning  Models

Before  going  into  some  of  the  details  of  the  models  selected,  let  us  check  the
com- ponents of any learning algorithm. In order to be able to learn, an algorithm
has to define at least three components:

�

The  model  class/hypothesis  space  defines  the  family  of  mathematical  models
that will  be  used.  The  target  decision  boundary  will  be  approximated  from  one
element of this space. For example, we can consider the class of linear models. In
this case our decision boundary will be a line if the problem is defined in R2 and
the model class is the space of all possible lines in R2.

Model classes define the geometric properties of the decision function. There
are different  taxonomies  but  the  best  known  are  the  families  of  linear  and
nonlinear models. These families usually depend on some parameters; and the
solution to a learning problem is the selection of a particular set of parameters, i.e.,
the selection of an instance of a model from the model class space. The model
class space is also called the hypothesis space.
The  selection  of  the  best  model  will  depend  on  our  problem  and  what  we
want to  obtain  from  the  problem.  The  primary  goal  in  learning  is  usually  to
achieve  the  minimum  error/maximum  performance;  but  according  to  what
else  we  want  from  the  algorithm,  we can  come  up  with  different  algorithms.
Other  common  desirable  properties  are  interpretability,  behavior  when  faced
with missing data, fast training, etc.
The  problem  model  formalizes  and  encodes  the  desired  properties  of  the
solution. In  many  cases,  this  formalization  takes  the  form  of  an  optimization
problem.  In  its  most  basic  instantiation,  the  problem  model  can  be  the
minimization  of  an  error function.  The  error function measures  the  difference
between  our  model  and  the  target.  Informally  speaking,  in  a  classification
problem  it  measures  how  �irritated�  we  are  when  our  model  misses  the  right
label for a training sample. For example, in classification, the ideal error function
is the 0�1 loss. This function takes value 1 when we incorrectly classify a training
sample and zero otherwise. In this case, we can interpret  it  by  saying  that we
are only irritated by �one unit of irritation� when one sample is misclassified.
The  problem  model  can  also  be  used  to  impose  other  constraints  on  our
solution,11 such as finding a smooth approximation, a model with a low degree
of small complexity, a sparse solution, etc.
The  learning  algorithm  is  an  optimization/search  method  or  algorithm  that,
given a model class, fits it to the training data according to the error function.
According to the nature of our problem there are many different algorithms. In
general,  we  are  talking  about  finding  the  minimum  error  approximation  or
maximum probable model. In those cases, if the problem is convex/quasi-convex
we  will  typically  use  first-  or  second-order  methods  (i.e.,  gradient  descent,
coordinate  descent,  Newton�s  method,  interior  point  methods,  etc.).  Other
searching techniques such as genetic algorithms or Monte Carlo techniques can
be used if we do not have access to the derivatives of the objective function.

�

�

Support Vector  Machines

SVM  is  a  learning  technique  initially  designed  to  fit  a  linear  boundary  between
the samples  of  a  binary  problem,  ensuring  the  maximum  robustness  in  terms  of
tolerance to isotropic uncertainty. This effect is observed in Fig. 5.9. Note that the
boundary  displayed  has  the  largest  distance  to  the  closest  point  of  both
classes.  Any  other

11Remember the regularization cure for overfitting.

88

Fig. 5.9 Support vector
machine decision
boundary and the support
vectors

5    Supervised Learning

separating  boundary  will  have  a  point  of  a  class  closer  to  it  than  this  one.  The
figure also shows the closest points of the classes to the boundary. These points
are called support vectors. In fact, the boundary only depends on those points. If
we  remove  any  other  point  from  the  dataset,  the  boundary  remains  intact.
However,  in general, if  any  of  these  special  points  is  removed  the  boundary  will
change.

  A Brief Note on Deriving Hard Margin Support Vector Machines In
order to understand the model, we have to be able to approximately
derive its for- mulation. For this purpose it is important to understand  a
couple of things about basic geometry of a hyperplane. A hyperplane in
Rd is defined as an affine combination of the variables: ?   aT x   b     0. A
hyperplane splits the space into two half-spaces. The evaluation of the
equation of the hyperplane on any element belonging to one

+  =

?

of the half-spaces is a positive value. It is a negative value for all the elements in
the other half-space. The distance of a point x  ? Rd  to the hyperplane ? is

d(x, ?)  =

|aT x  + b|

?a?2

Given  a  binary  classification  problem  with  training  data  D = {(xi , yi )},  i  = 1 . . .
N, yi ? {+1, ?1}, consider S ? D the subset of all data points belonging to class +1, S  =
{xi |yi  = +1}, and R = {xi |yi  = ?1} its complement.

5.7   Two Learning Models

89

Then  the  problem  of  finding  a  separating  hyperplane  consists  of  fulfilling  the

following constraints12

aT si  + b > 0  and aT ri  + b < 0  ?si  ? S, ri  ? R.

This  is  a  feasibility  problem  and  it  is  usually  written  in  the  following  way  in

optimization standard notation:

minimize  1

subject to  yi (aT xi  + b) ? 1,  ?xi  ? D
The  solution  of  this  problem  is  not  unique.  Selecting  the  maximum  margin
hyper- plane requires us to add a new constraint to our problem. Remember from
the geom- etry of the hyperplane that the distance of any point to a hyperplane is
given by: d(x, ?) a
=
?a?2

T x +b .

Recall also that we want positive data to be beyond value 1 and negative data

below 1. Thus, what is the distance value we want to maximize?

?

The positive point closest to the boundary is at 1/ a  2 and the negative point
closest  to  the  boundary  data  point  is  also  at  1/  a  2.  Thus,  data  points  from
different classes are at least 2/  a 2 apart.

?  ?

?  ?

?  ?

Recall  that  our  goal  is  to  find  the  separating  hyperplane  with  maximum
margin, i.e.,  with  maximum  distance  between  elements  in  the  different  classes.
Thus,  we  can complete  the  former  formulation  with  our  last  requirement  as
follows:

minimize  ?a?2/2
subject to  yi (aT xi  + b) ? 1,  ?xi  ? D

This formulation has a solution as long as the problem is linearly separable.
In order to deal with misclassifications, we are going to introduce a new set of
variables ?i , that represents the amount of violation in the i -th constraint. If the
constraint  is  already  satisfied,  then  ?i  0;  while  ?i  >  0  otherwise.  Because  ?i  is
related  to  the  errors,  we  would  like  to  keep  this  amount  as  close  to  zero  as
possible. This makes  us  introduce  an  element  in  the  objective  trade-off with  the
maximum margin.

=

12Note  the  strict  inequalities  in  the  formulation.  Informally,  we  can  consider  the  smallest
satisfied constraint, and observe that the rest must be satisfied with a larger value. Thus, we can
arbitrarily set that value to 1 and rewrite the problem as

aT si + b  ? 1 and aT ri  + b  ? ?1.

90

5    Supervised Learning

The new model becomes:

minimize    ?a?2/2 + C

N

i =1

?i

subject to  yi (aT xi  + b) ? 1 ? ?i ,  i  = 1 . . .  N

?i  ? 0

where C is the trade-off parameter that roughly balances the rates of margin and
misclassification. This formulation is also called soft-margin SVM.

The larger the C value is, the more importance one gives to the error, i.e., the
method will be more accurate according to the data at hand, at the cost of being
more sensitive to variations of the data.

The  decision  boundary  of  most  problems  cannot  be  well  approximated  by  a
linear model. In SVM, the extension to the nonlinear case is handled by means of
kernel theory. In a pragmatic way, a kernel can be referred to as any function that
captures the  similarity  between  any  two  samples  in  the  training  set.  The  kernel
has to be a positive semi-definite function as follows:

�  Linear kernel:

�  Polynomial kernel:

�  Radial Basis Function kernel:

k(xi , x j ) = x T x j

i

k(xi , x j ) = (1 + x T x j ) p

i

k(xi , x j ) = e

?

?xi ?x j ?2 2?2

Note that selecting a polynomial or a Radial Basis Function kernel means that
we have  to  adjust  a  second  parameter  p  or  ?,  respectively.  As  a  practical
summary,  the SVM  method  will  depend  on  two  parameters  (C,  ?)  that  have  to  be
chosen carefully using cross-validation to obtain the best performance.

Random Forest

Random Forest (RF) is the other technique that is considered in this work. RF is
an  ensemble  technique.  Ensemble  techniques  rely  on  combining  different
classifiers using some aggregation technique, such as majority voting. As pointed
out  earlier,  ensemble  techniques  usually  have  good  properties  for  combating
overfitting.  In  this case,  the  aggregation  of  classifiers  using  a  voting  technique
reduces  the  variance  of the  final  classifier.  This  increases  the  robustness  of  the
classifier  and usually  achieves a very good classification performance. A critical issue
in  the  ensemble  of  classifiers is  that  for  the  combination  to  be  successful,  the
errors  made  by  the  members  of  the ensemble  should  be  as  uncorrelated  as
possible. This is sometimes referred to in the

literature  as  the  diversity  of  the  classifiers.  As  the  name  suggests,  the  base
classifiers in RF are decision trees.

A Brief Note on Decision Trees

A  decision  tree  is  one  of  the  most  simple  and  intuitive  techniques  in  machine
learning,  based  on  the  divide  and  conquer  paradigm.  The  basic  idea  behind
decision trees is to partition the space into patches and to fit a model to a patch.
There are two questions to answer in order to implement this solution:

�  How do we partition the space?
�  What model shall we use for each patch?

Tackling the first question leads to different strategies for creating decision tree.
However, most techniques share the axis-orthogonal hyperplane partition policy,
i.e.,  a  threshold  in  a  single  feature.  For  example,  in  our  problem  �Does  the
applicant have a home mortgage?�. This is the key that allows the results of this
method  to  be  interpreted.
is
straightforward, each patch is given the value of a label, e.g., the majority label,
and all data falling in that part of the space will be predicted as such.

In  decision  trees,  the  second  question

The  RF  technique  creates  different  trees  over  the  same  training  dataset.  The
word �random� in RF refers to the fact that only a subset of features is available
to each of the trees in its building process. The two most important parameters in
RF are the number of trees in the ensemble and the number of features each tree
is allowed to check.

Ending the Learning Process

With  both  techniques  in  mind,  we  are  going  to  optimize  and  check  the  results
using nested cross-validation. Scikit-learn allows us to do this easily using several
model selection  techniques.  We  will  use  a  grid  search,  GridSearchCV  (a  cross-
validation  using  an  exhaustive  search  over  all  combinations  of  parameters
provided).

92

5    Supervised Learning

In [16]:

para met ers   =  { �C �:  [1 e4,  1 e5,  1 e6 ],

� gamma �:  [1 e -5 ,  1 e -4 ,  1 e -3]}

N_folds  =  5

kf = cr os s_ va li da t io n  . KFold ( n  =  y. shape [0] ,

n_folds  = N_folds  ,
shuffle  =  True ,
random_ state   =  0)

acc  =  np . zeros (( N_folds ,) )
i  =  0
#  We  will  build  the  p redi cted   y  from  the  partial  predict ions

on  the  test  of  each  of  the  folds

yhat  =  y. copy ()
for  train_index ,  t est_i ndex   in  kf :

X_train ,  X_test  =  X[ train_index  ,:] ,  X[ test_index  ,:]
y_train ,  y_test  =  y[ trai n_ind ex  ],  y[ test_ index  ]
scaler  =  Stan dardS cale r  ()
X_train  =  scaler . fi t_tra nsfor m  ( X_train  )
clf  =  svm . SVC ( kernel  =  � rbf �)
clf  =  gr id_se arch  . GridS earch CV  ( clf ,  parameters ,  cv  =  3)
clf . fit ( X_train ,  y_train . ravel () )
X_test  =  scaler . transform  ( X_test )
yhat [ test _inde x  ]  =  clf . predict  ( X_test )

print  metrics  . accu racy _scor e  ( yhat ,  y)
print  met rics  . co nf us io n_ ma tr i x  ( yhat ,  y)

Out[16]: classification  accuracy:  0.856038647343 confusion matrix:

3371 590
6    173

The result obtained  has a large  error in  the non-fully funded class (negative).
This  is  because  the  default  scoring  for  cross-validation  grid-search  is  mean
accuracy.  Depending on our business, this large  error in recall for this class may
be unaccept- able. There are different strategies for diminishing the impact of this
effect.  On  the  one  hand,  we  may  change  the  default  scoring  and  find  the
parameter setting that cor- responds to the maximum average recall. On the other
hand, we could mitigate this effect by imposing a different weight on an error on
the critical  class.  For example, we could look  for  the  best  parameterization  such
than  one  error  on  the  critical  class is  equivalent  to  one  thousand  errors  on  the
noncritical  class.  This  is  important  in  business  scenarios  where  monetization  of
errors can be derived.

A Toy Business Case

Consider that clients using our service yield a profit of 100 units per client (we will use
abstract  units  but  keep
in
euros/dollars). We  design  a  campaign  with  the  goal  of  attracting  investors  in
order to cover all non-fully  funded  loans.  Let  us  assume  that  the  cost  of  the
campaign  is  ?  units per client. With this policy we expect to keep our customers

in  mind  that  this  will  usually  be  accounted

satisfied  and  engaged  with  our  service,  so  they  keep  using  it.  Analyzing  the
confusion  matrix  we  can

93

5.9    A Toy Business Case

Fig. 5.10  Surfaces for two
different campaign and
attraction factors. The
horizontal plane
corresponds to the profit if
no campaign is launched.
The slanted plane is the
profit for a certain
confusion matrix

�

+

+

give precise meaning to different concepts in this campaign. The real positive set
( TP    FN)  consists  of  the  number  of  clients  that  are  fully  funded.  According  to
our  assumption,  each  of  these  clients  generates  a  profit  of  100  units.  The  total
profit is 100 ( TP    FN).  The campaign to attract investors will be cast considering
all  the clients we predict are not fully funded. These are those that the classifier
predict as  negative,  i.e.,  (FN    T  N).  However,  the  campaign  will  only  have  an
effect on the investors/clients that are actually not funded, i.e.,  T N ; and we expect
to attract a certain fraction ?  of them. After deploying our campaign, a simplified
model of the expected profit is as follows:

+

100 � ( T P   + F N )  ? ?(TN  + F N )  + 100?TN

When  optimizing  the  classifier  for  accuracy,  we  do  not  consider  the  business
needs. In this case, optimizing an SVM using cross-validation for different parameters
of  the  C  and ?,  we have  an accuracy of  85.60% and  a  confusion  matrix with  the
following values:

3371. 590.
6.  173.

If we check how the profit changes for different values of ? and ?, we obtain the
plot in  Fig.  5.10.  The  figure  shows  two  hyperplanes.  The  horizontal  plane  is  the
expected  profit  if  the  campaign  is  not  launched,  i.e.,  100  ( TP   FN).  The  other
hyperplane represents the profit of the campaign for different values of ? and ? using
a particular classifier. Remember that the cost of the campaign is given by  ?, and the
success  rate  of  the  campaign  is  represented  by  ?.  For  the  campaign  to  be
successful we would like to select values for both parameters so that the profit of
the  campaign  is  larger than  the  cost  of  launching  it.  Observe  in  the  figure  that
certain costs and attraction rates result in losses.

+

�

We may launch different  classifiers  with different configurations  and toy  with dif-

ferent  weights (2, 4, 8, 16) for  elements of  different  classes in  order to bias the classi
Supervised Learning

Fig. 5.11  3D surfaces  of the profit  obtained for  different classifiers and configurations  of  retention
campaign cost and retention rate. a RF, b SVM with the same cost per class, c SVM with double
cost for the target class, d SVM with a cost for the target class equal to 4, e SVM with a cost for
the target class equal to 8, f SVM with a cost for the target class equal to 16

fier towards obtaining different values for the confusion matrix.13 The weights define

Table 5.1  Different configurations  of classifiers and their respective profit rates and accuracies

Max profit rate (%)

Profit rate at 60% (%)  Accuracy (%)

Random forest

SVM {1 : 1}

SVM {1 : 2}

SVM {1 : 4}

SVM {1 : 8}

SVM {1 : 16}

4.41

4.59

4.52

4.30

10.69

10.68

2.41

2.54

2.50

2.28

3.57

2.88

87.87

85.60

85.60

83.81

52.51

41.40

how  much  a  misclassification
in  one  class  counts  with  respect  to  a
misclassification in another. Figure  5.11 shows the  different landscapes for  different
configurations of the SVM classifier and RF.

In order to frame the problem, we consider a very successful campaign with a

60% investor attraction rate. We can ask several questions in this scenario:

�  What is the maximum amount to be spent on the campaign?
�  How much will I gain?
�  From all possible configurations of the classifier, which is the most profitable?
�  Is it the one with the best accuracy?

Checking the values in Fig. 5.11, we find the results collected in Table 5.1. Observe
that the most profitable campaign with 60% corresponds to a classifier that considers
the cost of mistaking a sample from the non-fully funded class eight times larger
than the one from the other class. Observe also that the accuracy in that case is
much worse than in other configurations.

The  take-home  idea  of  this  section  is  that  business  needs  are  often  not  aligned
with  the notion of accuracy. In such scenarios, the confusion matrix values have
specific meanings. This must be taken into account when tuning the classifier.

96

5    Supervised Learning

may  tackle  many  more  different  settings.  For  example,  we  may  have  different
target labels  for  a  single  example;  this  is called multilabel  learning.  Or,  data can
come  from  streams  or  be  time  dependent;  in  these  settings,  sequential  learning  or
sequence  learning  can  be  the methods  of  choice. Moreover,  each  data example
can be a non- vector or have a variable size, such as a graph, a tree, or a string. In
such  scenarios kernel  learning  or  structural  learning  may  be  used.  During  these  last
years  we  are  also  seeing the revival of neural networks under the name of deep
learning and achieving impressive results  in  different domains  such  as  computer
vision or natural language processing. Nonetheless, all of these methods will behave
as  explained  in  this  chapter  and  most  of  the  lessons  learned  here  can  be  readily
applied to these techniques.

                                                             UNIT-4

Regression  analysis,  Regression:  linear  regression  simple  linear  regression,
multiple  &  Polynomial  regression,  Sparse  model.  Unsupervised  learning,
clustering,  similarity  and  distances,  quality  measures  of  clustering,  case
study.

Regression Analysis

Introduction

In this chapter, we introduce regression analysis and some of its applications in
data science.  Regression  is  related  to  how  to  make  predictions  about  real-world
quantities  such  as,  for  instance,  the  predictions  alluded  to  in  the  following
questions. How does sales volume change with changes in price? How is sales
volume affected by the weather? How does the title of a book affect its sales?
How  does  the  amount  of  a  drug  absorbed  vary  with  the  patient�s  body  weight;
and does this relationship depend on blood pressure? How many customers can
I expect today? At what time should I go home to avoid traffic jams? What is the
chance  of  rain  on  the  next  two  Mondays;  and  what
is  the  expected
temperature?

All these questions have a common structure: they ask for a response that
can be expressed as a combination of one or more (independent) variables
(also called covariates or predictors). The role of regression is to build a model
to predict the response from the variables. This process involves the transition
from data to model. More specifically, the model can be useful in different tasks,
such as the following:
(1) analyzing the behavior of data (the relation between the response and the
vari-  ables),  (2)  predicting  data  values  (whether  continuous  or  discrete),  and
(3) finding important variables for the model.

In order to understand how a regression model can be suitable for tackling
these tasks,  we  will  introduce  three  practical  cases  for  which  we  use  three  real
datasets  and  solve  different  questions.  These  practical  cases  will  motivate  simple
logistic  regression,  as
linear  regression,  multiple
presented in the following sections.

linear  regression,  and

98

6  Regression Analysis

Fig. 6.1  Illustration  of  different  simple  linear  regression  models.  Blue  points  correspond  to  a  set
of  random  points  sampled  from  a  univariate  normal  (Gaussian)  distribution.  Red,  green  and
yellow lines are three different simple linear regression models

Linear Regression

The  objective  of  performing  a  regression  is  to  build  a  model  to  express  the
relation  between  the  response  y  Rn  and  a  combination  of  one  or  more
(independent) vari- ables xi    Rn . [1] The model allows us to predict the response y
from the variables. The simplest model which can be considered is a linear model,
where the response y depends linearly on the d variables xi :

?

?

y = a1x1  + �  � � + ad xd .

(6.1)

The  variables ai  are  termed  the  parameters  or  coefficients  of  the model.
This equation can be rewritten in a more compact matrix form: y = Xw, where

?

?

y1
y2
?  ?
?
?
?  .
?
yn

y =

?

?

x11 . ..  x1d
x21 . . .  x2d

, X =

?
?
?

.
xn1 . . .  xnd

?
?

, w =

?

?

a1
a2
?  ?
?
?
.
?
?
ad

.

Linear regression is the technique for creating these linear models.

Simple Linear Regression

Rn and
Simple  linear  regression  considers  n  samples  of  a  single  variable  x
describes the relationship between the variable and the response with the model:

?

y = a0 + a1x,

(6.2)

where the parameter a0 is called the intercept or the constant term.

Given  a set  of  samples (x, y), such  as the  set illustrated in Fig. 6.1,  we  can create
a linear  model to explain  the data,  as in Eq. (6.2). But  how  do  we  know  which is the

6.2    Linear Regression

99

best  model  (best  parameters)  for  this  particular  set  of  samples?  See  the  three
different models (straight lines in different colors) in Fig. 6.1.

Ordinary least squares (OLS) is the simplest and most common estimator in which
the parameters (a�s) are chosen to minimize the square of the distance between
the predicted values and the actual values with respect to a0, a1:

||a0 + a1x ? y||2  =

2

(a0 + a1x j  ? y j )2.

n

j =1
We  are  concerned  here  with  the  y-axis  distance,  since  it  does  not  consider  the
error in  the  variables.  This  error  expression  is  often  called  the  sum  of  squared
errors of prediction (SSE). The SSE function is quadratic in the parameters, w, with
positive- definite  Hessian,  and therefore  this function  possesses  a  unique global
�  =  �  �
minimum  at w        (a0,  a1).  The  resulting  model  is  represented  as  follows:  y        a0
a1x,  where  the  hats  on  the variables  represent  the fact  that  they  are estimated
from the data available.

� = �  + �

OLS is a popular approach for several reasons. It makes it computationally cheap to
calculate  the  coefficients.  It  is  also  easier  to  interpret  than  the  other  more
sophisticated models. In situations where the goal is to understand a simple model
in detail, rather than  to  estimate the  response  well, it  can  provide insight into  what
the model captures. Finally, in situations where there is a lot of noise, as in many
real  scenarios,  it may be hard  to  find the  true  functional  form,  so  a  constrained
model can perform quite well compared to a complex model which can be more
affected by noise.

Practical Case: Sea Ice Data and Climate Change

In this practical case, we pose the question: Is the climate really changing? More
concretely, we want to show the effect of the climate change by determining whether
the sea ice area (or extent) has decreased over the years. Sea ice area refers to
the total area covered by ice, whereas sea ice extent is the area of ocean with at
least 15% sea ice. Reliable measurement of sea ice edges began with the satellite
era  in  the late 1970s. Before then, sea ice area and extent were monitored less
precisely by a combination of ships, buoys, and aircraft.

We will use the sea ice data from the National Snow & Ice Data Center1 which
provides measurements  of  the  area  and  extend  of  sea  ice at  the  poles  over  the
last  36 years. The center has given access to the archived monthly Sea Ice Index
images and data since 1979 [2]. The archived data reside at an FTP location2 (web-
page  instructions  can  be  followed  easily  to  access  and  download  the  files).  The
ASCII data files tabulate sea ice extent and area (in millions of square kilometers)
by year for a given month.

In  order  to  check  whether  there  is  an  anomaly  in  the  evolution  of  sea  ice
extent  over recent years, we want to build a simple linear regression model and
analyze the fitting; but before we need to perform several processing steps.

.

1

Fig. 6.2  Ice extent data by month

0
0

ession Analysis

6
In [1]:
R
e
(Pandas) as follows:
g
r

First,  we  read  the  data,  previously  downloaded,  and  create  a DataFrame

ice  =  pd . r e a d _ c s v  ( � f i l e s  / c h 06  / S e a I c e  . t xt �,

p r i nt   � s h a p e  : �,  i ce . s h a p e

de li m_ w hi te s pa ce  = True )

Out[1]:  shape: (424, 6)

For  data  cleaning,  we  check  the  values  of  all  the  fields  to  detect  any  potential
error. We find that there is a � 9999� value in the data_type field which should contain
�Goddard� or �NRTSI-G� (the type of the input dataset). So we can easily clean the
data, removing these instances.

?

In [2]:

ic e 2   =  ic e [ ice  . d a t a _ t y p e   !=  � -9 999 �]

Next,  we  visualize  the  data.  The  lmplot() function  from  the  Seaborn  toolbox is
intended for exploring linear relationships of different forms in multidimensional
datasets.  For  instance,  we  can  illustrate  the  relationship  between  the  month  of  the
year (variable) and the extent (response) as follows:

In [3]:

import  Se aborn   as  sns
sns . l m p l o t  ( " m o "  ,  " e x t e n t  " ,  i c e2  )

This  outputs  Fig.  6.2.  We  can  observe  a  monthly  fluctuation  of  the  sea  ice

extent, as would be expected for the different seasons of the year.

We  should  normalize  the  data  before  performing  the  regression  analysis  to
avoid this  fluctuation  and  be  able  to  study  the  evolution  of  the  extent  over  the
years.  To  capture  the  variation  for  a  given  interval  of  time  (month),  we  can
compute the mean

Fig. 6.3

Ice  extent data  by  month  after the normalization

for  the  i-th  interval  of  time  (using  the  period  from  1979  through  2014  for  the
mean extent) ?i , and subtract it from the set of extent values for that month ei .
This  value can be converted to a relative percentage difference by dividing it by
the total average (1979�2014) ?, and then multiplying by 100:

{  }
j

e�i   = 100 ?
j

ei  ? ?i
j

?

, i  = 1, . . . , 12.

We implement this normalization and plot the relationship again as follows:

In [4]:

for  i  in  r a n g e  (1 2 )  :

ic e 2  . e x t e n t  [ i c e 2  . mo  ==  i + 1 ]  =

1 0 0 *(  i c e 2  . e x t e n t  [ i c e 2  . mo  ==  i + 1 ]

-  m o n t h _ m e a n s  [ i + 1 ] )

/ m o n t h _ m e a n s  . m e an  ()

sns . l m p l o t  ( " m o "  ,  " e x t e n t  " ,  i c e2  )

The  new  output  is  in  Fig. 6.3.  We  now  observe  a  comparable  range  of  values

for all months.

Next, the normalized values can be plotted for the entire time series to analyze the
tendency.  We  compute  the  trend  as  a  simple  linear  regression.  We  use  the  lmplot()
function for visualizing linear relationships between the year (variable) and the extent
(response).

In [5]:

sns . l m p l o t  (" y e a r  " ,  " e x t e n t  " ,  i c e 2 )

This  outputs  Fig.  6.4  showing  the  regression  model  fitting  the  extent  data.
This plot  has  two  main  components.  The  first  is  a  scatter  plot,  showing  the
observed  data points.  The  second  is  a  regression  line,  showing  the  estimated
linear model relating

Fig. 6.4  Regression model fitting sea ice extent  data for all months by year using  lmplot

the  two  variables.  The  regression line is  plotted  with  a  95%  confidence  band  to
give an impression of the uncertainty in the model.

In this figure, we can observe that the data show a long-term negative trend
over years. The negative trend can be attributed to global warming, although there
is also a considerable amount of variation from year to year.

Up  until  here,  we  have  qualitatively  shown  the  linear  regression  using  a  useful
visu- alization  tool.  We  can  also  analyze  the  linear  relationship  in  the  data  using  the
Scikit-  learn library, which allows a quantitative evaluation. As was explained in the
previous  chapter,  Scikit-learn  provides  an  object-oriented  interface  centered
around the con- cept of an estimator. The sklearn.linear_model.LinearRegression
estimator  sets  the  state  of  the  estimator  based  on  the  training  data  using  the
function fit.  Moreover,  it  allows  the  user  to  specify  whether  to  fit  an  intercept
term  in  the  object  construction.  This  is  done  by  setting  the  corresponding
constructor arguments of the estimator object as follows:

In [6]:

from sklearn . l in ea r _m od e l  im port L i n e a r R e g r e s s io n
est  =  L i n e a r R e g r e ss i o n  ( f i t_ i n te r c e pt   =  True )

During  the  fitting  process,  the  state  of  the  estimator  is  stored  in  instance
attributes that have a trailing underscore (�_�). For example, the coefficients of a
LinearRegression  estimator  are  stored  in  the  attribute  coef_.  We  fit  a  regres-  sion
model using years as variables (x) and the extent values as the response (y).

In [7]:

x  =  i c e2  [[ � y e a r  � ] ]
y  =  i c e 2  [[ � e x t e n t  � ] ]
est . fi t ( x ,  y )
print  " Coe f fi ci e nt s  :" ,  est . coef_
p r i nt   " I n t e r c e p t  : " ,  es t . i n t e r c e p t _

6.2    Linear Regression

103

Out[7]:  Coefficients: [[-0.45275459]]

Intercept: [ 903.71640207]

.

Estimators that can generate predictions provide an Estimator.predict method.
In  the  case  of  regression,  Estimator.predict  will  return  the  predicted  regression
values.  We  can  evaluate  the  model  fitting  by  computing  the  mean  squared  error
(MSE) and the coefficient of determination (R2) of the model. The coefficient R2 is
?  =
defined as (1    u/v), with u      (y    y)2 and v      (y    y)2, where y is the mean. The
best possible score for R2 is 1.0, lower values are worse (it can also be negative).
These measures can provide a quantitative answer to the question we are facing:
Is there a negative trend in the evolution of sea ice extent over recent years? We
can  perform  this  analysis  for  a  particular  month  or  for  all  months  together,  as
done in the following lines:

  .

? �

? �

=

�

In [8]:

from  sklearn   import  metrics
y_hat  =  est . p redict  ( x)
print  " MSE :" ,  metric s  . m ea n _s qu ar ed _e rr or  ( y_hat ,  y)
print  " R ^2: " ,  metric s  . r2 _score  ( y_hat ,  y)
print  � var : �,  y. var ()

Out[8]:  MSE: 10.5391316398

R2: 0.50678703821
var: 31.98324

The  negative  trend  seen  in  Fig. 6.4 is  validated  by  the  MSE value  which  is  small,
0.1%, and the R2 value which is acceptable, given the variance of the data, 0.3%.
Given the model, we can also predict the extent value for the coming years.
For instance, the predicted extent for January 2025 can be computed as follows:

In [9]:

x  =  [2025]
y_hat  =  model . p redict  ( x)
m  =  1  #  January
y_hat  =  ( y_hat * m onth_ means  . mean () /100)  +  mon th_m e ans  [ m]
print  " P redic tion   of  extent  for  January  2025

( in  millions  of  square  km ):" ,  y_hat

Out[9]:  Prediction of extent for January 2025 (in millions of square km): [12.93603933].

Multiple  Linear  Regression  and Polynomial Regression

As  we  have  seen  in  the  previous  section,  with  simple  linear  regression  we
describe the relationship  between  the variable  and  the response  with  a  straight
line.  In  the  case  of  multiple  linear  regression,  we extend  this idea  by fitting  a  d-
dimensional hyperplane to our d variables, as defined in Eq. (6.1).

Multiple linear regression may seem a very simple model, but even when the
response depends on the variables in nonlinear ways, this model can still be used
by

104

6  Regression Analysis

considering nonlinear transformations ? (�) of the variables:

y  = a1?(x1) + � � �  + ad ?(xd )

This model is called polynomial regression  and it is a popular nonlinear regression
technique  which  models  the  relationship  between  the  response  and  the
variables as an p-th order polynomial. The higher the order of the polynomial, the
more complex the functions you can fit. However, using higher-order polynomial
can  involve computational complexity and overfitting. Overfitting occurs when a
model fits  the  characteristics  of  the  training  data  and  loses  the  capacity  to
generalize from the seen to predict the unseen.

Sparse  Model

Often,  in  real  problems,  there  are  uninformative  variables  in  the  data  which
prevent proper  modeling  of  the  problem  and  thus,  the  building  of  a  correct
regression model. In  such  cases,  a  feature  selection  process  is  crucial  to  select
only  the  informative  features  and  discard  non-informative  ones.  This  can  be
achieved  by  sparse  methods which  use  a  penalization  approach,  such  as  LASSO
(least absolute shrinkage and selection operator) to set some model coefficients
to zero (thereby discarding those variables). Sparsity can be seen as an application
of Occam�s razor: prefer simpler models to complex ones.

Given  the  set  of  samples  (X, y),  the  objective  of  a  sparse  model  is  to  minimize

the SSE through a restriction (or penalty):

1
2n

||Xw ? y||2  + ?||w||1,
2

where ||w||1  is the  L1-norm of the parameter vector w  = ( a 0 , .. .,  ad ).

Practical Case: Prediction of the Price of a New Housing Market

In this practical case we want to solve the question: Can we predict the price of a
new market given any of its attributes?

We will use the Boston housing dataset from Scikit-learn, which provides recorded
measurements of 13 attributes of housing markets around Boston, as well as the
median house price.3  Once  we  load  the dataset (506 instances),  the description
of  the  dataset  can  easily  be  shown  by  printing  the  field  DESCR.  The  data  (x),
feature names, and target (y) are stored in other fields of the dataset.

We  first  consider  the  task  of  predicting  median  house  values  in  the  Boston
area using as the variable one of the attributes, for instance, LSTAT, defined as the
�pro- portion of lower status of the population�.

Seaborn visualization can be used to show this linear relationships easily:

3Copy of UCI ML housing dataset: http://archive.ics.uci.edu/ml/datasets/Housing.

6.2    Linear Regression

105

Fig. 6.5  Scatter  plot  of  Boston  data  (LSTAT versus  price)  and  their  linear  relationship  (using
lmplot)

In [10]:

from  sk le ar n  i mpo rt  da ta se t s
bost on  =  da ta se ts  . l o a d _ b o s t on  ()
X_bo sto n ,  y_ bo st on   =  b ost on  . data ,  bos ton  . ta rge t
prin t  � S hape  of  data : �,  X_ bo st on . shape ,  y _b o st on . s hap e
prin t  � Fe at ur e  nam es : �, bo sto n . f e a t u r e _ na m e s
d f _ b o st o n   =  pd . D a t a F r am e  ( bosto n . data ,

co lu mn s  =  bo sto n . f e a t u r e _ na m e s )

d f _ b o st o n  [ � p ric e �]  =  bo sto n . ta rge t
sns . l mpl ot  (" pri ce  " ,  " L STA T " ,  d f _ b o s t o n  )

Out[10]: Shape  of  data:  (506L,  13L)  (506L,)

Feature names: [�CRIM� �ZN� �INDUS� �CHAS� �NOX� �RM� �AGE�
�DIS� �RAD� �TAX� �PTRATIO� �B� �LSTAT�]

In  Fig. 6.5,  we  can  clearly  see  that  the  relationship  between  price and  LSTAT is
nonlinear, since the straight line is a poor fit. We can examine whether a better fit
can be obtained by including higher-order terms. For example, a quadratic model:
yi ? a0 + a1xi  + a2x2
The  lmplot function  allows  to  easily  change  the  order  of  the  model  as  is  done  in
the next code, which outputs Fig. 6.6, where we observe a better fit.

i

In [11]:

sns . lmplot (" price " ,  " LSTAT " ,  df_boston ,  order  =  2)

To study the relation among multiple variables in a dataset, there are different
options. We can study the relationship between several variables in a dataset by
using  the  functions  corr  and  heatmap  which  allow  to  calculate  a  correlation  matrix
for a dataset and draws a heat map with the correlation values. The heat map is a
matricial  image  which  helps  to  interpret  the  correlations  among  variables.  For  the
sake  of  visualization,  we  do  not  consider  all  the  13  variables  in  the  Boston
housing data, but six: CRIM, per capita crime rate by town; INDUS, proportion of
non-retail

106

6  Regression Analysis

Fig. 6.6 Scatter plot of Boston data (LSTAT versus price) and their polynomial  relationship (using
lmplot with order 2)

business acres per town; NOX, nitric oxide concentrations (parts per 10 million);
RM, average number of rooms per dwelling; AGE, proportion of owner-occupied
units built prior to 1940; and LSTAT. These variables are indicated by their indexes
in the following code:

In [12]:

indexes  =  [0 ,2 ,4 ,5 ,6 ,12]
df 2  =  pd . Dat aFrame  ( boston . data [: , ind exes  ],

columns  =  boston . feat ure_ names  [ indexes  ])

df2 [ � price �]  =  boston . target
corrmat  =  df 2 . corr ()
sns . he atmap  ( corrmat  ,  vmax  =  .8 ,  square  =  True )

Figure  6.7  shows  a  heat  map  representing  the  correlation  between  pairs  of  vari-
ables;  specifically,  the  six  variables  selected  and  the  price  of  houses.  The  color
bar shows  the  range  of  values  used  in  the  matrix.  This  plot  is  a  useful  way  of
summa- rizing the correlation of several variables. It can be seen that LSTAT and RM
are the variables that are most correlated with price.

Another  good  way  to  explore  multiple  variables  is  the  scatter  plot  from
Pandas. The  scatter  plot  is  a  grid  of  plots  of  multiple  variables  one  against  the
others, illus- trating the relationship of each variable with the rest. For the sake of
visualization, we do not consider all the variables, but just three: RM, AGE, and LSTAT
defined by indexes in the following code:

In [13]:

indexes  =[ 5 , 6 ,12]
df 2  =  pd . Dat aFrame  ( boston . data [: , ind exes  ],

df2 [ � price �]  =  boston . target
pd . sc atter _matr ix  ( df2 ,  figsize  =  (12.0 ,  12.0) )

columns  =  boston . feat ure_ names  [ indexes  ])

6.2    Linear Regression

Fig. 6.7 Correlation plot:
heat map representing
the correlation between
seven pairs of variables in
the Boston housing
dataset

107

This  code  outputs  Fig.  6.8,  where  we  obtain  visual  information  concerning  the
density function for every variable, in the diagonal, as well as the scatter plots of
the data  points  for  pairs  of variables.  In  the  last  column, we  can  appreciate  the
relation between  the  three  variables  selected  and  house  prices.  It  can  be  seen  that
RM follows a linear relation with price; whereas AGE does not. LSTAT follows a higher-
order  relation  with  price.  This  plot  gives  us  an  indication  of  how  good  or  bad
every attribute would be as a variable in a linear model.

For  the  evaluation  of  the  prediction  power  of  the  model  with  new  samples,  we
split  the  data  into  a  training  set  and  a  testing  set,  and  we  compute  the  linear
regression  score,  which  returns  the  coefficient  of  determination  R2  of  the
prediction. We can also calculate the MSE.

In [14]:

from  sklearn   import  li near_ model
train_s ize   =  X_b oston  . shape [0]/2
X_train  = X _boston  [: tra in_ siz e  ]
X_test  =  X_boston  [ train _size  :]
y_train  = y_bo ston  [: tr ain _si ze  ]
y_test  =  y_boston [ tra in_si ze :]
print  � Training   and  test ing   set  sizes �,

X_train . shape ,  X_test . shape

regr  =  Li ne ar Re gr e ss io n  ()
regr . fit ( X_train ,  y_ train  )
print  � Coeff  and  int erc ept  : �,

regr . coef_ ,  regr . int ercep t_

print  � Tes ting  Score : �,  regr . score ( X_test ,  y_test )  print  �

Training

MSE :  �,

np . mean (( regr . predict  ( X_tr ain ) - y_train  ) **2)

print  � Testing  MSE :  �,

np . mean (( regr . predict  ( X_test )  -  y_test ) **2)

108

6  Regression Analysis

Fig. 6.8  Scatter  plot  of  Boston  housing dataset

Out[14]: Training  and  testing  set  sizes  (253,  13)  (253,  13)

Coeff and intercept: [ 1.20133313 0.02449686 0.00999508
0.42548672 -8.44272332 8.87767164 -0.04850422 -1.11980855
0.20377571 -0.01597724 -0.65974775 0.01777057 -0.11480104]
-10.0174305829
Testing Score: -2.24420202674
Training MSE: 9.98751732546
Testing MSE: 302.64091133

We  can  see  that  all  the  coefficients  obtained  are  different  from  zero,  meaning
that no variable is discarded. Next, we try to build a sparse model to predict the
price  using  the  most  important  factors  and  discarding  the  non-informative  ones.  To
do this, we can create a LASSO regressor, forcing zero coefficients.

Regression

Linear

109

In [15]:

regr_la sso   =  l inear _mod el  . Lasso ( alpha  =  . 3 )
regr _la sso  . fit ( X_train , y_train  )  print  � Coeff  and  inte rce pt  :

�, r egr_l asso  . coef_

print  � Tesing  Score : �, r egr _la sso  . score ( X_test  ,
y_test )  print  � Trainin g   MSE :  �,

np . mean (( r egr_ las so  . predi ct ( X_ train  )

-  y_tra in  ) **2)

print  � Testing  MSE :  �,

np . mean (( reg r_las so  . predict  ( X_test )  -  y_test ) **2)

Out[15]: Coeff  and  intercept:  [  0.  0.01996512  -0.  0.  -0.  7.69894744

-0.03444803 -0.79380636 0.0735163 -0.0143421 -0.66768539
0.01547437 -0.22181817] -6.18324183615
Testing Score: 0.501127529021
Training MSE: 10.7343110095
Testing MSE: 46.5381680949

It can now be seen that the result of the model fitting for a set of sparse

coefficients is much better than before (using all the variables), with the score
increasing from
?
important for the prediction and in fact they confuse the regressor.

2.24 to 0.5. This demonstrates that four of the initial variables are not

With the LASSO result, we can also emphasize the most important factors for

determining the price of a new market, based on the coefficient values:

In [16]:

ind  =  np . a rgsort  ( np . abs ( r egr _la sso  . coef_ ))
print  � Ordered  variable  ( from  less  to  more  impo rt ant ): �,

boston . feat ure_n ames  [ ind ]

Out[16]: Ordered variable (from  less  to  more  important): [�CRIM�  �INDUS� �CHAS� �NOX� �TAX� �B� �ZN� �AGE�

�RAD� �LSTAT� �PTRATIO� �DIS� �RM�]

There  are  also  other  strategies  for  feature  selection.  For  instance,  we  can
select the k 5 best features, according to the k highest scores, using the function
SelectKBest from Scikit-learn:

=

In [17]:

import  sklearn  . fe at ur e_ se l ec ti on   as  fs
selecto r  =  fs . S elect KBest  ( sc ore_f unc   =  fs . f_regre ssion  ,

selecto r . fi t_tra nsfor m  ( X_train ,  y_train )  per
selector  . fit ( X_train  , y_tr ain )
print  � Se lected   featur es : �,

k  =  5)

zip ( sel ector  . get_ suppo rt  () ,  boston . fe ature _name s  )

Out[17]: Selected  features:  [(False,  �CRIM�),  (False,  �ZN�),  (True,

�INDUS�), (False, �CHAS�), (False, �NOX�), (True, �RM�), (True,
�AGE�), (False, �DIS�), (False, �RAD�), (False, �TAX�), (True, �PTRATIO�), (False, �B�), (True, �LSTAT�)]

The  set  of  selected  features  is  now  different,  since  the  criterion  has  changed.

However, three of the most important features: RM, PTRATIO, and LSTAT.

In order to evaluate the prediction, it could be interesting to visualize the
target and predicted responses in a scatter plot, as it is done in the next code:

110

6  Regression Analysis

Fig. 6.9  Relation  between  true  (x-axis)  and  predicted  (y-axis)  prices

In [18]:

clf  =  L in e ar Re gr es si on  ()
clf . fit ( boston . data ,  boston . target )
pred ict ed   = clf . p redict  ( boston . data )
plt . sca tter ( boston . target ,  predicted ,  alpha  =  0.3)
plt . plot ([0 ,  50] ,  [0 ,  50] ,  �-- k �)
plt . axis ( � tight �)
plt . xlabel ( � True  price  ( $1000s ) �)
plt . ylabel ( � Pr edict ed   price  ( $1000s ) �)

The  output  is  shown  in  Fig.  6.9,  where  we  can  observe  that  the  original
prices are  properly  estimated  by  the  predicted  ones,  except  for  the  higher
values, around
$50.000 (points in the top right corner).

Finally,  it  is  worth  noting  that  we  can  work  with  statistical  evaluation  of  a
linear regression with the OLS toolbox of the Stats Model toolbox.4 This toolbox is
useful to study several statistics concerning the regression model. To know more
about the toolbox, go to the Documentation related to Stats Models.

Logistic Regression

Logistic regression is a type of model of probabilistic statistical classification. It is
used as a binary model to predict a binary response, the outcome of a categorical
dependent variable (i.e., a class label), based on one or more variables.

The form of the logistic function is:

f  x
(  ) =

1

1 + e??x

Logistic

Regression

111

Fig. 6.10  Logistic function for different  lambda values

Fig. 6.11 Linear regression (blue) versus logistic regression (red) for fitting a set of data (black points)
normally distributed across the 0 and 1 y-values

Figure 6.10  illustrates  the  logistic  function  with  different  values  of  ?.  This  function
is  useful  because  it  can  take  as  its  input  any  value  from  negative  infinity  to
positive infinity, whereas the output is restricted to values between 0 and 1 and
hence can be interpreted as a probability.

The  set  of  samples  (X,  y),  illustrated  as  black  points  in  Fig.  6.11,  defines  a  fitting
problem suitable for a logistic regression. The blue and red lines show the fitting
result for linear and logistic models, respectively. In this case, a logistic model can
clearly explain the data; whereas a linear model cannot.

Practical Case: Winning or Losing Football Team

Now,  we  pose  the  question:  What  number  of  goals  makes  a  football  team  the
winner or  the  loser?  More  concretely,  we  want  to  predict  victory  or  defeat  in  a
football match  when  we  are  given  the  number  of  goals  a  team  scores.  To  do  this
we consider

the set of results of the football matches from the Spanish league5 and we build a
classification model with it.

We  first  read  the  data  file  in  a  DataFrame and  select  the  following  columns in
a new  DataFrame: HomeTeam, AwayTeam,  FTHG (home  team goals),  FTAG (away
team  goals),  and  FTR  (H=home  win,  D    draw,  A
  away  win).  We  then  build a  d-
dimensional  vector  of  variables  with  all  the  scores,  x,  and  a  binary  response
indicating victory or defeat, y. For that, we create two extra columns containing
W the number of goals of the winning team and L the number  of goals of  the losing
team and we concatenate these data. Finally, we can compute and visualize a logistic
regression  model  to  predict  the  discrete  value  (victory  or  defeat)  using  these
data.

=

=

In [19]:

from skl ea rn  . li nea r_model  import Log i st i cR eg ressi on  data   =  pd . r ea d_cs v  ( �
files / ch06 / SP 1 . csv �)
s  =  data [[ � HomeTea m �,� A w ay Team �,  � FTHG �,  � FTAG �,  � FTR �] ] def  my_f1 ( row ):

ret urn  max ( row [ � FTHG �],  row [ � FTAG �] ) def  my_f2 ( row ):
ret urn  min ( row [ � FTHG �],  row [ � FTAG �] )  s[ �W �]  =  s. appl y (

my_f1 ,  axis  =  1)
s[ �L �]  =  s. a pply ( my_f2 ,  axis  =  1) x1  =  s[ �W �]. val ues
y1  =  np . ones ( len ( x1 ) ,  dtype  =  np . int ) x2  =  s[ �L �]. values
y2  =  np . zeros ( len ( x2 ) ,  dtype  =  np . int ) x  =  np . conca t ena t e
([ x1 ,  x2 ])
x  =  x [: ,  np . ne wa x is ]
y  =  np . conca t ena t e ([ y1 ,  y2 ]) log reg  =
Log is ti cR eg r es s io n  () log reg . fit (x ,  y)
X_t est  =  np . l i nspa ce ( -5 ,  10 ,  300) def  lr_ model  (x ):

ret urn  1  /  ( 1 + np . exp ( - x) )

loss  =  l r_model ( X_t est * l ogreg . coef _  +  log reg . i nt ercept _  )

. ravel ()

X _tes t2   =  X _tes t [: , np . newa x is  ]
los s pred   = log reg . p redict  ( X _tes t2  ) plt . s ca tter ( x. ravel
() ,  y ,

� bla ck �,

color  =
s  =  100 ,  z order  =  20 ,
alpha  =  0.03)

plt . plot ( X _test ,
plt . plot ( X _test ,

loss ,  color  =
loss pred ,  color  =

� blue �,

lin ewi dth   =  3)

� red �,

lin ewidt h   =  3)

Figure 6.12 shows a scatter plot with transparency so we can appreciate the over-
lapping in the discrete positions of the total numbers of victories and defeats. It
also shows the fitting of the logistic regression model, in blue, and prediction of
the logistic regression model, in red, for the Spanish football league results. With
this  information  we  can  estimate  that  the  cutoff  value  is  1.  This  means  that  a
team, in general, has to score more than one goal to win.

5http://www.football-data.co.uk/mmz4281/1213/SP1.csv.

Fig. 6.12 Fitting of the logistic regression model (blue) and prediction of the logistic regression model
(red) for the Spanish football league results

Unsupervised Learning

Introduction

In machine learning, the problem of unsupervised learning is that of trying to find
hidden  structure  in  unlabeled  data.  Since  the examples  given  to  the learner  are
unla- beled,  there  is  no  error  or  reward  signal  to  evaluate  the  goodness  of  a
potential  solution.  This  distinguishes  unsupervised  from  supervised  learning.
Unsupervised  learning is defined as the task performed by algorithms that learn
from a training set of  unlabeled  or  unannotated  examples, using  the  features  of
the inputs to categorize them according to some geometric or statistical criteria.

Unsupervised learning encompasses many techniques that seek to summarize and
explain  key  features  or  structures  of  the  data.  Many  methods  employed  in
unsuper- vised  learning  are  based  on  data  mining  methods  used  to  preprocess
data.  Most  unsupervised  learning  techniques  can  be  summarized  as  those  that
tackle the follow- ing four groups of problems:

�
�

�

�

Clustering: has as a goal to partition the set of examples into groups.
Dimensionality reduction: aims to reduce the dimensionality of the data. Here,
we encounter techniques such as Principal Component Analysis (PCA),
independent component analysis, and nonnegative matrix factorization.
Outlier detection: has as a purpose to find unusual events (e.g., a
malfunction), that distinguish part of the data from the rest according to certain
criteria.
Novelty detection: deals with cases when changes occur in the data (e.g., in
stream- ing data).

The  most  common  unsupervised  task  is  clustering,  which we  focus  on  in  this

chapter.

Clustering

Clustering is a process of grouping similar objects together; i.e., to partition unlabeled
examples into disjoint subsets of clusters, such that:

�

�

Examples within a cluster are similar (in this case, we speak of  high intraclass
similarity).
Examples in different clusters are different (in this case, we speak of  low interclass
similarity).

When we denote data as similar and dissimilar, we should define a measure for
this similarity/dissimilarity.  Note  that  grouping  similar  data  together  can  help in
discov- ering  new  categories  in  an  unsupervised  manner,  even  when  no  sample
category  labels  are  provided.  Moreover,  two  kinds  of  inputs  can  be  used  for
grouping:

(a)  in  similarity-based  clustering,  the  input  to  the  algorithm  is  an  n  n  dissimilarity

�

matrix or distance matrix;

(b)  in feature-based clustering, the input to the algorithm is an n  D feature matrix
or design matrix, where n is the number of examples in the dataset and D the
dimensionality of each sample.

�

Similarity-based  clustering  allows  easy  inclusion  of  domain-specific  similarity,
while  feature-based  clustering  has  the  advantage  that  it  is  applicable  to
potentially noisy data.

Therefore, several questions regarding the clustering process arise.

�

�

�

�

What is a natural grouping among the objects? We need to define the �groupness�
and the �similarity/distance� between data.
How can we group samples? What are the best procedures? Are they efficient?
Are they fast? Are they deterministic?
How  many  clusters  should  we  look  for  in  the  data?  Shall  we  state  this
number a priori? Should the process be completely data driven or can the user
guide  the  grouping  process?  How  can  we  avoid  �trivial�  clusters?  Should  we
allow  final  clustering results  to  have  very  large  or  very  small  clusters?  Which
methods  work when  the  number  of  samples  is  large?  Which  methods  work
when the number of classes is large?
What constitutes a good grouping? What objective measures can be defined to
evaluate the quality of the clusters?

There  is  not  always  a  single  or  optimal  answer  to  these  questions.  It  used  to  be
said  that  clustering  is  a  �subjective�  issue.  Clustering  will  help  us  to  describe,
analyze, and gain insight into the  data, but the  quality of the partition  depends to  a
great extent on the application and the analyst.

clustering

Similarity and Distances

117

To speak of similar and dissimilar data, we need to introduce a notion of the similarity
of data. There are several ways for modeling of similarity. A simple way to model
this is by means of a Gaussian kernel:

s(a, b)  =  e??d(a,b)

where d(a, b) is  a metric function,  and  ? is  a constant that controls the decay of the
function. Observe that when a  b, the similarity is maximum and equal to one. On
the  contrary,  when  a  is  very  different  to  b,  the  similarity  tends  to  zero.  The
former modeling of the similarity function suggests that we can use the notion of
distance  as  a  surrogate.  The  most  widespread  distance  metric  is  the  Minkowski
distance:

=

d

d(a, b) = (

|ai ? bi|p)1/p

i=1
where  d(a, b)  stands  for  the  distance  between  two  elements  a, b  Rd ,  d is  the
dimensionality of the data, and p is a parameter.

?

The best-known instantiations of this metric are as follows:

�  when p = 2, we have the Euclidean distance,
�  when p = 1, we have the Manhattan distance, and
�  when p = inf, we have the max-distance. In this case, the distance corresponds

to the component |ai ? bi| with the highest value.

What Constitutes a Good Clustering? Defining Metrics to
Measure Clustering Quality

When  performing clustering,  the question  normally  arises:  How  do  we measure
the quality of the clustering result? Note that in unsupervised clustering, we do not
have groundtruth  labels  that  would  allow  us  to  compute  the  accuracy  of  the
algorithm.  Still, there  are  several  procedures  for  assessing  quality.  We  find  two
families  of  techniques: those  that  allow  us  to  compare  clustering  techniques,  and
those  that  check  on  specific  properties  of  the  clustering,  for  example
�compactness�.

Rand Index, Homogeneity, Completeness and V-measure Scores
One  of  the  best-known  methods  for  comparing  the  results  in  clustering
techniques  in statistics is the Rand index  or Rand measure (named after William
M. Rand). The Rand  index  evaluates  the  similarity  between  two  results  of  data
clustering.  Since in  unsupervised  clustering,  class  labels  are  not  known,  we  use
the Rand index  to  compare  the  coincidence  of  different clusterings  obtained  by
later  discuss  the
different  approaches  or  criteria.  As  an  alternative,  we

Silhouette coefficient: instead of

118

7    Unsupervised Learning

comparing different clusterings, this evaluates the compactness of the results of
applying a specific clustering approach.

Given a set of n elements S        o 1 , . . . ,   on  , we can compare two partitions of S1: X
}
= {

X 1 , . . . ,   Xr  , a partition of S into r subsets; and Y      Y 1 , . . . , ,   Ys , a partition of S into s
subsets. Let us use the annotations as follows:

= {

= {

}

}

�

�

�

�

a is the number of pairs of elements in S that are in the same subset in both X  and
Y ;
b is the number of pairs of elements in S that are in different subsets in both X and
Y ;
c  is  the  number  of  pairs  of  elements  in  S  that  are  in  the  same  subset  in  X ,  but
in different subsets in Y ; and
d  is the  number  of  pairs  of  elements in  S  that  are in  different subsets  in  X ,  but
in the same subset in Y .

The Rand index, R, is defined as follows:

ensuring that its value is between 0 and 1.

R

=

a + b

a + b + c + d

,

One  of  the  problems  of  the  Rand  index  is  that  when  given  two  datasets  with
random labelings,  it  does  not  take  a  constant  value  (e.g.,  zero)  as  expected.
Moreover,  when the  number  of  clusters  increases  it  is  desirable  that  the  upper
limit tends to the unity. To solve this problem, a form of the Rand index, called the
Adjusted Rand index, is used  that  adjusts  the Rand  index  with  respect  to  chance
grouping of elements. It is defined as follows:

 n
AR =  2

 n 2
2

(a + d) ? [(a + b)(a + c) + (c + d )(b + d)]
.

[(a + b)(a + c) + (c + d )(b + d)]

Another way for comparing clustering results is the V-measure. Let us first intro-
duce  some  concepts.  We  say  that  a  clustering  result  satisfies  a  homogeneity
criterion if  all  of  its  clusters  contain  only  data  points  which  are  members  of  the
same original (single) class. A clustering result satisfies a completeness criterion if
all  the  data  points that are members of  a given class are elements of  the same
predicted  cluster. Note  that  both  scores  have  real  positive  values  between  0.0
and  1.0,  larger  values  being  desirable.  For  example,  if  we  consider  two  toy
clustering sets (e.g., original and predicted) with four samples and two labels, we
get:

print (" %.3 f"  %  metr ics  . h om og en ei ty _s co r e  ([0 ,  0 ,  1 ,  1] ,

[0 ,  0 ,  0 ,  0]) )

In [1]:

Out[1]:  0.000

.

7.2    Clustering

119

The  homogeneity  is  0  since  the  samples in  the  predicted  cluster  0  come

from original cluster 0 and cluster 1.

In [2]:

print  metrics . c o m p l e t e n e ss _ s c or e  ([0 ,  0 ,  1 ,  1] ,
[1 ,  1 ,  0 ,  0] )

Out[2]:  1.0

The completeness is 1 since all the samples from the original cluster with label
0 go  into  the  same  predicted  cluster  with  label  1,  and  all  the  samples  from  the
original cluster with label 1 go into the same predicted cluster with label 0.

However,  how  can  we  define  a  measure  that  takes  into  account  the
completeness as well as the homogeneity? The V-measure is the harmonic mean
between the homogeneity and the completeness defined as follows:

v  = 2 ? (homogeneity ? completeness)/(homogeneity + completeness).

Note that this metric is not dependent of the absolute values of the labels: a
permutation of the class or cluster label values will not change the score value in
any  way. Moreover,  the metric  is  symmetric  with  respect to  switching  between
the predicted  and  the  original  cluster  label.  This  is  very  useful  to  measure  the
agreement of  two  independent  label  assignment  strategies  applied  to  the  same
dataset  even  when  the  real  groundtruth  is  not  known.  If  class  members  are
completely  split  across  different  clusters,  the  assignment  is  totally  incomplete,
hence the V-measure is null:

In [3]:

print (" %.3 f"  %  m etrics  . v _m ea su re _s c or e  ([0 ,  0 ,  0 ,  0] ,

[0 ,  1 ,  2 ,  3]) )

Out[3]:  0.000

In contrast, clusters that include samples from different classes destroy the

homo- geneity of the labeling, hence:

In [4]:

print (" %.3 f"  %  m etrics  . v _m ea su re _s c or e  ([0 ,  0 ,  1 ,  1] ,

[0 ,  0 ,  0 ,  0]) )

Out[4]:  0.000

In summary, we can say that the advantages of the  V-measure include that it
has  bounded  scores:  0.0  means  the  clustering  is  extremely  bad;  1.0  indicates  a
per- fect clustering result. Moreover, it can be interpreted easily: when analyzing
the V-measure, low completeness or homogeneity explain in which direction the
clus- tering  is  not  performing  well.  Furthermore,  we  do  not  assume  anything
about  the  cluster  structure.  Therefore,  it  can  be  used  to  compare  clustering
algorithms  such as K-means, which assume isotropic blob shapes, with results of
other clustering algorithms such as spectral clustering (see Sect. 7.2.3.2),  which  can
find  clusters  with  �folded�  shapes.  As  a  drawback,  the  previously  introduced
metrics  are  not  normalized  with  regard  to  random  labeling.  This  means  that
depending  on  the  num- ber  of  samples,  clusters  and  groundtruth  classes,  a
completely random labeling will

120

7    Unsupervised Learning

not always yield the same values for homogeneity, completeness and hence, the

V- measure.  In  particular,  random  labeling  will  not  yield  a  zero  score,  and  they  will
tend  further from zero as the number of clusters increases. It can be shown that
this prob- lem can reliably be overcome when the number of samples is high, i.e.,
more than a thousand, and the number of clusters is less than 10. These metrics
require knowl- edge of the groundtruth classes, while in practice this information
is  almost  never  available  or  requires  manual  assignment  by  human  annotators.
Instead, as mentioned before, these metrics can be used to compare the results of
different clusterings.

Silhouette Score

An  alternative  to  the  former  scores  is  to  evaluate  the  final  �shape�  of  the
clustering result. This is the underlying idea behind the Silhouette coefficient. It is
defined as a function of the intracluster distance of a sample in the dataset, a and
the  nearest-  cluster distance, b for each sample.2  Later, we will discuss different
ways  to  compute the  distance  between  clusters.  The  Silhouette  coefficient  for  a
sample i can be written as follows:

Silhouette(i)
=

b ? a

.
max(a, b)

Hence,  if  the  Silhouette  s(i) is  close  to  0,  it  means  that  the  sample  is  on  the  border
of its cluster and the closest one from the rest of the dataset clusters. A negative
value means that the sample is closer to the neighbor cluster. The average of the
Silhouette coefficients of all samples of a given cluster defines the �goodness� of
the  cluster.  A high positive value, i.e., close to 1 would mean a compact cluster,
and vice versa. And the average of the Silhouette coefficients of all clusters gives
idea  of  the  quality of  the  clustering  result.  Note  that  the  Silhouette  coefficient
only makes sense when the number of labels predicted is less than the number of
samples clustered.

The advantage of the Silhouette coefficient is that it is bounded between  1 and
1. Moreover, it is easy to show that the score is higher when clusters are dense
+
and well separated; a logical feature when speaking about clusters. Furthermore,
the Silhouette coefficient is generally higher when clusters are compact.

?

Taxonomies of Clustering Techniques

Within  different  clustering  algorithms,  one  can  find  soft  partition  algorithms,
which assign  a  probability  of  the  data  belonging  to  each  cluster,  and  also  hard
partition algorithms,  where  each  datapoint  is  assigned  precise  membership  of
one  cluster. A  typical  example  of  a  soft  partition  algorithm  is  the  Mixture  of
Gaussians [1], which  can  be  viewed  as  a  density  estimator  method  that  assigns
a  confidence or

2The intracluster distance of sample i is obtained by the distance of the sample to the nearest sample
from  the  same  class,  and  the  nearest-cluster  distance  is  given  by  the  distance  to  the  closest
sample from the cluster nearest to the cluster of sample i.

probability to each point in the space. A Gaussian mixture model is a probabilistic
model that assumes all the data points are generated from a mixture of a finite
number  of  Gaussian  distributions  with  unknown  parameters.  The  universally
used  generative unsupervised  clustering using  a Gaussian mixture  model  is  also
known as EM Clustering. Each point in the dataset has a soft assignment to the K
clusters. One  can  convert  this  soft  probabilistic  assignment  into  membership  by
picking  out  the  most  likely  clusters  (those  with  the  highest  probability  of
assignment).

An alternative to soft algorithms are the hard partition algorithms, which assign a
unique  cluster  value  to  each  element  in  the  feature  space.  According  to  the
grouping process  of  the  hard  partition  algorithm,  there  are  two large  families  of
clustering techniques:

�

�

Partitional  algorithms:  these  start  with  a  random  partition  and  refine  it
iteratively. That  is  why  sometimes  these  algorithms  are  called  �flat�  clustering.  In
this chapter, we will consider two partitional algorithms in detail: K-means and
spectral clus- tering.
Hierarchical  algorithms:  these  organize  the  data  into  hierarchical  structures,
where data can be agglomerated in the bottom-up direction, or split in a top-down
manner. In this chapter, we will discuss and illustrate agglomerative clustering.

A typical hard partition algorithm is K-means clustering. We will now discuss it

in some detail.

K-means Clustering

K-means  algorithm  is  a  hard  partition  algorithm  with  the  goal  of  assigning  each
data point to a single cluster. K-means algorithm divides a set of n samples X into
k disjoint clusters ci, i  1,.  , k, each described by the mean ?i  of the samples in the
cluster. The means are commonly called cluster centroids. The K-means algorithm
assumes that all k groups have equal variance.

=

K-means clustering solves the following minimization problem:

k

k

arg minc

d(x, ?j) = arg minc

||x ? ?j||2

2

(7.1)

j=1 x?cj

j=1 x?cj

= ||  ?

where  ci  is  the  set  of  points  that  belong  to  cluster  i  and  ?i  is  the  center  of  the
class  ci.  K-means  clustering  objective  function  uses  the  square  of  the  Euclidean
distance d(x, ?j) x ?j 2, that is also referred to as the inertia or within-cluster sum-
of-squares. This problem is not trivial to solve (in fact, it is NP-hard problem), so
the algorithm only hopes to find the global minimum, but may become stuck at a
different solution.

||

In  other  words,  we  may  wonder  whether  the  centroids  should  belong  to  the

original set of points:

inertia =

n

i=0

min?j ?c(||xi  ? ?j||2)).

(7.2)

122

7    Unsupervised Learning

The K-means algorithm, also known as Lloyd�s algorithm, is an iterative procedure
that searches for a solution of the K -means clustering problem and works as follows.
First, we need to decide  the number of clusters, k. Then we apply the following
procedure:

1.  Initialize (e.g., randomly) the k cluster centers, called centroids.
2.  Decide the class memberships of the n data samples by assigning them to

the nearest-cluster centroids (e.g., the center of gravity or mean).

3.  Re-estimate the k cluster centers, ci, by assuming the memberships found

above are correct.

4.  If none of the n objects changes its membership from the last iteration, then

exit. Otherwise go to step 2.

Let us illustrate the algorithm in Python. First, we will create three sample

distri- butions:

In [5]:

MAXN  =  40
X  =  np . c o n c a t e n a t e  ([

1.25 * np . r andom . randn ( MAXN ,  2) ,
5  +  1 .5 *  np . r a n d o m  . r a n d n  ( MAXN ,  2) ] )

X  =  np . c o n c a t e n a t e  ([

X ,  [8 ,  3]  +  1 .2 *  np . r a n d o m  . r a n d n  ( MAXN ,  2) ] )

The sample distributions generated are shown in Fig. 7.1 (left). However, the algo-
rithm  is  not  aware  of  their  distribution.  Figure  7.1  (right)  shows  what  the
algorithm sees.  Let  us  assume  that  we  expect  to  have  three  clusters  (k  3)  and
apply the K-means command from the Scikit-learn library:

=

Fig. 7.1

Initial  samples  as  generated  (left), and  samples  seen by  the  algorithm (right)

7.2    Clustering

123

In [6]:

from  sklearn   import  cluster

K  =  3  #  Ass uming   we  have  3  clusters  !
clf  =  cluster  . KMeans ( init  =  � random �,  n_cl uster s   =  K)
clf . fit ( X)

Out[6]:  KMeans(copy_x=True, init=�random�, max_iter=300,

n_clusters=3, n_init=10, n_jobs=1, precompute_distances=True, random_state=None,
tol=0.0001, verbose=0)

Each clustering algorithm in Scikit-learn is used as follows. First, an object from
the  clustering  technique  is  instantiated.  Then  we  can  use  the  fit method  to  adjust
the  learning  parameters.  We  also  find  the  method  predict  that,  given  new  data,
returns the cluster they belong to. For the class, the labels over the training data
can be found in the labels_  attribute or  alternatively they can  be obtained  using the
predict method.

How  many  �mis-clusterings�  do  we  have?  In  order  to  see  this,  we  tessellate
the  space  and  color  all  grid  points  from  the  same  cluster  with  the  same  color.
Then, we overlay the initial sample distributions (see Fig. 7.2). In the ideal case, we
expect that in each partitioned subspace the sample points are of the same color.
However,  as  shown  in  Fig.  7.2,  the  resulting  clustering,  which  is  represented  in  the
figure by the color subspace in gray, does not usually coincide exactly with the initial
distribution, which is represented by the color of the data. For example, in the same
figure, if most of the blue points belong to the same cluster, there are a few ones
that belong to the space occupied by the green data.

When computing the Rand index, we get:

In [7]:

print  ( � The  Adjuste d   Rand  index  is :  %. 2 f �  %

metrics  . ad ju st ed _r an d_ s co re  ( y. ravel () ,  clf . l abels _ ))

Fig. 7.2 Original samples
(dots) generated by three
distributions and the
partition of the space
according to the K-means
clustering

124

7    Unsupervised Learning

Out[7]:  The Adjusted Rand index is: 0.66

Taking into account that the Adjusted Rand index belongs to the interval [0, 1],
the result of 0.66 in our example means that although most of the clusters were
discovered, not 100% of them were; as confirmed by Fig. 7.2.

The inertia can be seen as a measure of how internally coherent the clusters are.

Several issues should be taken into account:

�

�

�

�

The inertia assumes that clusters are isotropic and convex, since the Euclidean
distance  is  applied,  which is  isotropic  with  regard  to  the  different  dimensions
of  the  data.  However,  we  cannot  expect  that  the  data  fulfill  this  assumption  by
default. Hence, the K-means algorithm responds poorly to elongated clusters or
manifolds with irregular shapes.
The  algorithm may  not  ensure  convergence  to  the  global minimum. It can be
shown  that  K-means  will  always  converge  to  a  local  minimum  of  the  inertia
(Eq.  (7.2)).  It  depends  on  the  random  initialization  of  the  seeds,  but  some
seeds can  result  in  a  poor  convergence  rate,  or  convergence  to  suboptimal
clustering. To alleviate the problem of local minima, the K-means computation
is  often  per- formed  several  times,  with  different  centroid  initializations.  One
way to address this  issue  is  the  k-means++ initialization  scheme,  which  has  been
implemented in  Scikit-learn  (use  the  init=�kmeans++�  parameter).  This  parameter
initializes the centroids to be (generally) far from each other, thereby probably
leading to better results than random initialization.
This  algorithm  requires  the  number  of  clusters  to  be  specified.  Different
heuristics  can  be  applied  to  predetermine  the  number  of  seeds  of  the
algorithm.
It  scales  well  to  a  large  number  of  samples  and  has  been  used  across  a  large
range of application areas in many different fields.

In summary, we can conclude that K-means has the advantages of allowing the
easy  use  of  heuristics  to  select  good  seeds;  initialization  of  seeds  by  other
methods; multiple points to be tried. However, in contrast, it still cannot ensure
that  the  local minima  problem  is  overcome;  it is  iterative  and  hence  slow when
there  are  a  lot  of  high-dimensional  samples;  and  it  tends  to  look  for  spherical
clusters.

Spectral  Clustering

Up  to  this  point,  the  clustering  procedure  has  been  considered  as  a  way  to find
data groups following a notion of compactness. Another way of looking at what a
cluster is is provided by connectivity (or similarity). Spectral clustering [2] refers to a
family of methods that use spectral techniques. Specifically, these techniques are
related to the eigendecomposition of an affinity or similarity matrix and solve the
problem  of clustering  according  to  the  connectivity  of  the  data.  Let  us  consider  an
ideal similarity matrix of two clear sets.

Let us denote the similarity matrix, S, as the matrix Sij  s(xi, xj) which gives the
similarity  between  observations  xi  and  xj.  Remember  that  we  can  model
similarity

=

using the Euclidean distance, d(xi, xj)
as follows:

xi
= ||  ?  ||

xj  2, by means of a Gaussian Kernel

s(xi, xj)  = exp(??||xi  ? xj||2),

where ? is  a  parameter.  We  expect  two  points  from  different  clusters  to  be  far
away from  each  other.  However,  if  there  is  a  sequence  of  points  within  the  cluster
that forms a �path� between them, this also would lead to big distance among some
of the points from the same cluster. Hence, we define an affinity matrix A based on
the similarity matrix S, where A contains positive values and is symmetric. This can
be  done,  for example,  by  applying  a  k-nearest  neighbor  that  builds  a  graph
connecting just the k closest data points. The symmetry comes from the fact that
Aij  and  Aji  give  the  distance  between  the  same  points.  Considering  the  affinity
matrix, the clustering can be  seen  as  a  graph  partition  problem,  where  connected
graph  components  correspond to clusters. The graph  obtained by spectral clustering
will  be  partitioned  so  that  graph  edges  connecting  different  clusters  have  low
weights,  and  vice  versa.  Furthermore, we  define  a  degree  matrix  D,  where  each
diagonal  value  is  the  degree  of  the  respective  graph  node  and  all  other  elements
are 0. Finally, we can compute the unnormalized graph Laplacian (U D A) and/or a
normalized version of the Laplacian (L), as follows:

=  ?

�

=  ?

Simple Laplacian: L    I     D?1A, which corresponds to a random walk, being D?1
the transition matrix. Spectral clustering obtains groups of nodes such that the
random walk corresponds to seldom transitions from one group to another.
1
1
Normalized Laplacian: L = D?
UD?
�
2
�  Generalized Laplacian: L = D?1U .

.

2

If  we  assume  that  there  are  k  clusters,  the  next  step  is  to  find  the  k
small- est eigenvectors, without considering the trivial constant eigenvector. Each
row of the matrix formed by the k  smallest eigenvectors of the Laplacian matrix
defines a  transformation  of  the  data  xi.  Thus,  in  this  transformed  space, we can
apply K-means clustering in order to find the final clusters. If we do not know in
advance the number of clusters, k, we can look for sudden changes in the sorted
eigenvalues of the matrix, U , and keep the smallest ones.

Hierarchical Clustering

Another well-known clustering technique of particular interest is hierarchical cluster-
ing.  Hierarchical  clustering  is  comprised  of  a  general  family  of  clustering  algorithms
that construct nested clusters by successive merging or splitting of data. The hier-
archy of clusters is represented as a tree. The tree is usually called a dendrogram.
The root of the dendrogram is the single cluster that contains all the samples; the
leaves  are  the  clusters  containing  only  one  sample  each.  This  is  a  nice  tool,
since it  can  be  straightforwardly  interpreted:  it  �explains�  how  clusters  are
formed  and  visualizes clusters  at different  scales.  The  tree that  results from the
technique shows

the similarity between the samples. Partitioning is computed by selecting a cut
on the tree at a certain level.

In general, there are two types of hierarchical clustering:

�  Top-down divisive clustering applies the following algorithm:

�  Start with all the data in a single cluster.
�  Consider every possible way to divide the cluster into two.
�  Choose the best division.
�  Recursively, it  operates  on  both  sides  until  a  stopping criterion is met.  That
can  be  something  as  follows:  there  are  as  much  clusters  as  data;  the
predetermined number  of  clusters  has  been  reached;  the  maximum  distance
between  all  possible  partition  divisions  is  smaller  than  a  predetermined
threshold; etc.

�  Bottom-up agglomerative clustering applies the following algorithm:

�  Start with each data point in a separate cluster.
�  Repeatedly join the closest pair of clusters.
�  At  each  step,  a  stopping  criterion  is  checked:  there  is  only  one  cluster;  a
prede- termined number of clusters has been reached; the distance between
the closest clusters is greater than a predetermined threshold; etc.

This process of merging forms a binary tree or hierarchy.

When merging  two clusters,  a  question  naturally  arises:  How  to measure  the
similarity  of  two clusters?  There  are  different  ways  to define  this  with  different
results  for  the  agglomerative  clustering.  The  linkage  criterion  determines  the
metric used for the cluster merging strategy:

�

�

�

Maximum or complete linkage minimizes the maximum distance between observa-
tions  of  pairs  of  clusters.  Based  on  the  similarity  of  the  two  least  similar
members of the clusters, this clustering tends to give tight spherical clusters as a
final result. Average  linkage  averages  similarity  between  members,  i.e.,  minimizes
the  average of the distances between all observations of pairs of clusters.
Ward linkage minimizes the sum of squared differences within all clusters. It is
thus a variance-minimizing approach and in this sense is similar to the K-means
objective function, but tackled with an agglomerative hierarchical approach.

Let  us  illustrate  how  the  different  linkages  work  with  an  example.  Let  us

generate three clusters as follows:

In [8]:

MAXN1  =  500
MAXN2  =  400
MAXN3  =  300
X1  =  np . c o n c a t e n a t e  ([

2. 2 5   *  np . r a n d o m  . r a nd n  ( MA XN 1 ,  2) ,
4  +  1 . 7 *  np . r a n d o m  . r a n d n  ( MA XN2  ,  2) ])

X1  =  np . c o n c a t e n a t e  ([

X1 ,  [8 ,  3]  +  1 . 9 *  np . r a n d o m  . r a n d n  ( M AX N3 ,  2) ])

y1  =  np . c o n c a t e n a t e  ([

np . o n e s  (( M AX N1  ,  1) ) ,
2  *  np . o n e s  (( M AX N2 ,  1) ) ])

y1  =  np . c o n c a t e n a t e  ([

y1 ,  3  *  np . o n es  (( M AXN 3 ,  1) ) ]) . r a v e l  ()

y1  =  np . int_ ( y1 )
l a b e l s _ y 1   =  [ �+ �,  �* �,   �o �]
c o l o r s   =  [ �r �,   �g �,  �b �]

Let us apply agglomerative clustering using the different linkages:

In [9]:

from skle arn  . clus ter   import  A gg l o me r a ti v e Cl u st e r in g

for  linkage   in  ( � ward �,  � com plete �,  � a verage  �):

clus ter ing   =  Ag g lo m e ra t i ve C l us t e ri n g  ( linkag e   =  linkage ,

n_clust ers   = 3 )

cluster ing  . fit ( X1 )

x_min  ,  x_max  =  np . min  ( X1 ,  axis  =  0)  ,  np . max  ( X1 , axis

=  0)

X1  =  ( X1  -  x_min  )  /  (  x_max  -  x_min  )
plt . figure  (  figs ize  =( 5  ,  5) )
for  i  in  range  ( X1 . shape  [0])  :

plt . text ( X1 [ i ,  0] ,  X1 [ i ,  1] , la bel s_y 1  [ y1 [ i ] -1] ,

color  =  colors [ y1 [ i ] -1])

plt . title (" \% s  linkage   "  \%  linkage ,  size
plt . ti ght_l ayout  ()

=  20)

plt . show ()

The results of the agglomerative clustering using the different linkages: complete,
average,  and  Ward  are  given  in  Fig.  7.3.  Note  that  agglomerative  clustering
exhibits �rich  get  richer�  behavior  that  can  sometimes  lead  to  uneven  cluster
sizes,  with  average  linkage  being  the  worst  strategy  in  this  respect  and  Ward
linkage giving the most regular sizes. Ward linkage is an attempt to form clusters
that are as compact as possible, since it considers inter- and intra-distances of the
clusters.  Meanwhile,  for  non-Euclidean  metrics,  average  linkage  is  a  good
alternative.  Average  linkage  can  produce  very  unbalanced  clusters,  it  can  even
separate a single data point into a separate cluster. This fact would be useful if we
want  to  detect  outliers,  but  it  may  be  undesirable  when  two  clusters  are  very
close to each other, since it would tend to merge them.

Agglomerative  clustering  can  scale  to  a  large  number  of  samples  when  it  is
used jointly with a connectivity matrix, but it is computationally expensive when no
con-

nectivity  constraints  are  added  between  samples:  it  considers  all  the  possible
merges at each step.

Adding Connectivity Constraints

Sometimes,  we  are  interested  in  introducing  a  connectivity  constraint  into  the
clus- tering  process  so  that  merging  of  nonadjacent  points  is  avoided.  This  can  be
achieved  by  constructing  a  connectivity  matrix  that  defines  which  are  the
neighboring  samples in  the  dataset.  For  instance,  in  the  example  in  Fig.  7.4,  we
want  to  avoid  the  forma- tion of clusters of samples from the different circles. A
sample code to compute agglomerative clustering with connectivity would be as
follows:

Fig.  7.3  Illustration  of  agglomerative  clustering  using  different  linkages:  Ward,  complete,  and
average.  The  symbol  of  each  data  point  corresponds  to  the  original  class  generated  and  the
color corresponds to the cluster obtained

Fig. 7.4  Illustration  of agglomerative clustering  without (top  row) and  with (bottom  row) a connec-
tivity  graph  using  the  three  linkages  (from  left  to  right):  average,  complete,  and  Ward.  The
colors correspond to the clusters obtained

In [10]:

conn ect ivi ty   =  k ne ig h bo rs _g ra ph  ( X ,  30 )
model  = A gg l o me r a ti v e Cl u st e r in g  ( linkage   = � ave rage  �,

connect ivity   =  connectivity  ,  n_cl uster s   =  8)

model . fit ( X)

information  or  can  be
to

A  connectivity  constraint  is  useful  to  impose  a  certain  local  structure,  but  it
also makes  the  algorithm  faster,  especially  when  the  number  of  the  samples  is
large. A connectivity constraint is imposed via  a  connectivity  matrix:  a sparse matrix
that only has elements at the intersection of a row and a column with indexes of
the  dataset  that  should  be  connected.  This  matrix  can  be  constructed  from  a
instance  using
priori
kneighbors_graph
using
image.grid_to_graph to  limit  merging to neighboring pixels in an image, both from
Scikit-learn. This phenomenon can be observed in Fig. 7.4, where in the first row we
see the results of the agglomerative clustering without using a connectivity graph.
The clustering can join data from different circles (e.g., the black cluster). At the
bottom, the three linkages use a connectivity graph and thus two of them avoid
joining  data  points  that  belong  to  different  circles  (except  the  Ward  linkage  that
attempts to form compact and isotropic clusters).

learned
restrict  merging

from  the  data,
nearest
to

for
neighbors

or

Fig.  7.5  Comparison  of  the  different  clustering  techniques  (from  left  to  right):  K-means,  spectral
clustering, and agglomerative clustering with average and Ward linkage on simple compact datasets.
In the first row, the expected number of clusters is k = 2 and in the second row: k  = 4

Comparison of Different Hard Partition Clustering Algorithms Let us

compare the behavior of the different clustering algorithms discussed so far. For
this purpose, we generate three different datasets� configurations:

(a)  4 spherical groups of data;
(b)  a uniform data distribution; and
(c)  a non-flat configuration of data composed of two moon-like groups of data.

An  easy  way  to  generate  these  datasets  is  by  using  Scikit-learn  that  has  predefined
functions for it: datasets.make_blobs(), datasets.ma- ke_ moons(), etc.

We  apply  the  clustering  techniques  discussed  above,  namely  K-means,
agglom- erative  clustering  with  average  linkage,  agglomerative  clustering  with  Ward
linkage, and spectral clustering. Let us test the behavior of the different algorithms
assuming k  2  and  k  4.  Connectivity  is  applied  in  the  algorithms  where  it  is
applicable.

=

=

In the simple case of separated clusters of data and k 4, most of the clustering
algorithms  perform  well,  as  expected (see Fig.  7.5).  The  only  algorithm  that could
not discover the four groups of samples is the average agglomerative clustering.
Since  it  allows  highly  unbalanced  clusters,  the  two  noisy  data  points  that  are
quite separated from the closest two blobs were considered as a different cluster,
while the two central blobs were merged in one cluster. In case of k    2, each of
the methods is obligated to join at least two blobs in a cluster.

=

=

Regarding the uniform distribution of data (see Fig. 7.6), K-means, Ward linkage
agglomerative clustering and spectral clustering tend to yield even and compact
clusters;  while  the  average  linkage  agglomerative  clustering  attempts  to  join
close  points  as  much  as  possible  following  the  �rich  get  richer�  rule.  This
results  in  a

Clustering

131

Fig.  7.6  Comparison  of  the  different  clustering  techniques  (from  left  to  right):  K-means,  spectral
clustering,  and  agglomerative  clustering  with  average  and  Ward  linkage  on  uniformly
distributed data. In the first row, the number of clusters assumed is k  = 2 and in the second row:
k = 4

Fig.  7.7  Comparison  of  the  different  clustering  techniques  (from  left  to  right):  K-means,  spec-
tral  clustering,  and  agglomerative  clustering  with  average  and  Ward  linkage  on  non-flat
geometry  datasets. In the first row, the expected number of clusters is k  = 2 and in the second
row: k = 4

second cluster of a small set of data. This behavior is observed in both cases: k      2
and k      4.

=

=

Regarding  datasets  with  more  complex  geometry,  like  in  the  moon  dataset
(see  Fig.  7.7),  K-means  and  Ward  linkage  agglomerative  clustering  attempt  to
construct  compact  clusters  and  thus  cannot  separate  the  moons.  Due  to  the
connectivity  con-  straint,  the  spectral  clustering  and  the  average  linkage
agglomerative clustering sep- arated both moons in case of k 2, while in case of k
4,  the  average  linkage  agglomerative  clustering  clustered  most  of  datasets
correctly separating some of the noisy data points as two separate single clusters.
In the case of spectral clustering, looking for four clusters, the method splits each
of the two moon datasets into two clusters.

=

=

132

7    Unsupervised Learning

Fig.  7.8  Expenditure  on  different  educational  indicators  for  the  first  five  countries  in  the  Eurostat
dataset

Case Study

In order to illustrate clustering with a real dataset, we will now analyze the indicators
of  spending on  education  among  the  European  Union member  states,  provided
by  the  Eurostat  data  bank.3  The  data  are  organized  by  year  (TIME)  from  2002
until  2011  and  country  (GEO):  (�Albania�,  �Austria�,  �Belgium�,  �Bulgaria�,  etc.).
Twelve indicators  (INDIC_ED)  of  financing  of  education  with  their  corresponding
values (Value) are given: (1) Expenditure on educational institutions from private
sources  as  %  of  gross  domestic  product  (GDP),  for  all  levels  of  education
combined; (2) Expenditure  on educational  institutions  from  public  sources  as  %
of  GDP,  for  all  levels  of  government  combined,  (3)  Expenditure  on  educational
institutions  from  public sources as % of total public expenditure, for all levels of
education combined,
(4) Public subsidies to the private sector as % of GDP, for all levels of education
combined,  (5)  Public  subsidies  to  the  private  sector  as  %  of  total  public
expenditure, for  all  levels  of  education  combined,  etc.  We  can  store  the  12
indicators for a given year (e.g.,  2010) in  a table. Figure  7.8  provides  visualization of
the first five countries in the table.

As we can observe, this is not a clean dataset, since there are values missing.
Some countries have very limited information and should be excluded. Other
countries may still not collect or have access to a few indicators. For these last cases,
we can proceed in two ways: (a) fill in the gaps with some non-informative, non-
biasing data; or (b) drop the features with missing values for the analysis. If we
have many features and only a few have missing values, then it is not very harmful
to drop them. However, if missing values are spread across most of the features, we
eventually have to deal with them. In our case, both options seem reasonable, as
long as the number of missing features for a country is not too large. We will
proceed in both ways at the same time. We  apply  both  options:  filling  the  gap
with  the mean  value  of  the feature  and the dropping option, ignoring the
indicators with missing values. Let us now apply K-means  clustering  to  thesedatin
order  to  partition  the  countries  according  to

.

7.3    Case Study

133

Fig.  7.9  Clustering  of  the  countries  according  to  their  educational  expenditure  using  filled-in  (top
row) and dropped (bottom row) missing values

their  investment  in  education  and  check  their  profiles.  Figure  7.9  shows  the
results  of  this  K-means  clustering.  We  have  sorted  the  data  for  better
visualization.  At a simple  glance,  we can see that the  partitions (top  and  bottom  of
Fig.  7.9)  are  different.  Most  countries  in  cluster  2  in  the  filled-in  dataset
correspond to cluster 0 in the dropped missing values dataset. Analogously, most
of cluster 0 in the filled- in dataset correspond to cluster 1 in the dropped missing
values  dataset;  and  most  countries  from  cluster  1  in  the  filled-in  dataset
correspond to cluster 2 in the dropped

134

7    Unsupervised Learning

Fig. 7.10  Mean expenditure of the different clusters according to the 8 indicators of the indicators-
dropped dataset

set. Still, there are some countries that do not follow this rule. That is, looking at
both clusterings, they may yield similar (up to label permutation) results, but
they will not necessarily always coincide. This is mainly due to two aspects: the
random initialization of the K-means clustering and the fact that each method
works in a different  space  (i.e.,  dropped  data in  8D  space  vs  filled-in  data,
working  in  12D space). Note that we should not consider the assigned absolute
cluster value, since it is irrelevant. The mean expenditure of the different clusters is
shown by different colors  according  to  the  8  indicators  of  the  indicators-dropped
dataset  (see  Fig. 7.10). So, without loss of generality, we continue analyzing the set
obtained by dropping missing values. Let us now check the clusters and check their
profile by looking at the centroids. Visualizing the eight values of the three clusters
(see Fig. 7.10), we can see that cluster 1 spends more on education for the 8
educational indicators, while

cluster 0 is the one with least resources invested in education.

Let us consider a specific country, e.g., Spain and its expenditure on education.
If  we  refine  cluster  0  further  and  check  how  close  members  are  from  this
cluster to  cluster  1,  it  may  give  us  a  hint  as  to  a  possible  ordering.  When
visualizing  the  distance  to  cluster  0  and  1,  we  can  observe  that  Spain,  while  being
from  cluster  0,  has a smaller distance to cluster 1 (see Fig.  7.11). This should make us
realize that using  3 clusters probably does not sufficiently represent the groups of
countries. So we redo the process, but applying k    4: we obtain 4 clusters. This
time cluster  0 includes  the  EU  members  with  medium  expenditure  (Fig.  7.12).  This
reinforce the intuition about Spain being a limit case in the former clustering. The
clusters obtained are as follows:

=

�

Cluster  0:  (�Austria�,
�France�,
�EU13�,
�Germany�,  �Hungary�,  �Latvia�,  �Lithuania�,  �Netherlands�,  �Poland�,  �Portugal�,
�Slovenia�, �Spain�, �Switzerland�, �United Kingdom�, �United States�)

�Estonia�,

�EU15�,

�EU27�,

�EU25�,

7.3    Case Study

135

Fig. 7.11  Distance of countries in cluster 0 to centroids of cluster 0 (in red) and cluster 1 (in blue)

Fig. 7.12  K-means applied  to the Eurostat dataset  grouping the countries into  four clusters

Cluster 1: (�Bulgaria�, �Croatia�, �Czech Republic�, �Italy�, �Japan�, �Romania�, �Slovakia�)

�
�  Cluster 2: (�Cyprus�, �Denmark�, �Iceland�)
�  Cluster 3: (�Belgium�, �Finland�, �Ireland�, �Malta�, �Norway�, �Sweden�)

We  can  repeat  the  process  using  the  alternative  clustering  techniques  and
compare  their  results.  Let  us  first  apply  spectral  clustering.  The  corresponding
code will be as follows:

136

7    Unsupervised Learning

Fig. 7.13  Spectral clustering  applied to  the  European  countries  according  to their  expenditure  on
education

In [11]:

X  =  Sta ndard Scale r  () . fi t_tra nsfor m  ( edudrop  . values )
dist anc es   =  e uc li de an _ di st an ce s  ( edud rop . values )
spectral   =  clus ter  . Sp ec tr al Cl us t er in g  (

n_cl ust ers   =  4 ,  aff ini ty   =  " ne ar es t_ ne ig hb o rs  ")

spectral . fit ( edudrop  . values )
y_pred  =  sp ectral  . lab els_ . astype ( np . int )

The  result  of  this  spectral  clustering  is  shown  in  Fig. 7.13.  Note  that  in  general,
the aim of spectral clustering is to obtain more balanced clusters. In this way, the
predicted  cluster  1  merges clusters  2  and  3  of  the K-means  clustering,  cluster  2
corresponds  to  cluster  1  of  the  K-means  clustering,  cluster  0  mainly  shifts  to
cluster 2, and cluster 3 corresponds to cluster 0 of the K-means.

Applying  agglomerative  clustering,  not  only  we  do  obtain  different  clusters,
but also  we can  see  how  different clusters  are  obtained.  Thus,  in  some  way  it is
giving  us  information  on  which  the  most  similar  pairs  of  countries  and  clusters
are. The corresponding code that applies the agglomerative clustering will be as
follows:

Case Study

137

In [12]:

from  scipy . cluster . hie rarch y  import  linkage ,  dend rogram
from  scipy . s patial  . dis tance   import  pdist

X_train  =  edu drop  . values
dist  =  pdist ( X_train ,  � eu cli dea n  �)
linkage _matr ix   =  linkage  ( dist ,  method  =  � complete �);

plt . figure ( fi gsize   =  (11.3 ,  11.3) )
dend rog ram  ( l ink age _ma tri x  ,  or ien tat ion  =" right " ,

col or_ th re sho ld   =  3 ,
labels   = w r k_ co un tr ie s_ na m es  ,
leaf_fo nt_si ze   =  20 ) ;

plt . ti ght_l ayout  ()

In Scikit-learn, the parameter color_threshold of the command dendro- gram()
colors  all  the  descendent  links  below  a cluster node  k  the same  color  if  k is  the
first  node  below  the  color_threshold.  All  links  connecting  nodes  with  distances
greater  than  or  equal  to  the  threshold  are  colored  blue.  Hence,  using
color_threshold
= 3, the clusters obtained are as follows:

�
�

�
�

Cluster 0: (�Cyprus�, �Denmark�, �Iceland�)
Cluster  1:  (�Bulgaria�,  �Croatia�,  �Czech  Republic�,  �Italy�,  �Japan�,  �Romania�,
�Slovakia�)
Cluster 2: (�Belgium�, �Finland�, �Ireland�, �Malta�, �Norway�, �Sweden�)
�France�,
�EU27�,
�EU13�,
Cluster  3:  (�Austria�,
�Germany�,  �Hungary�,  �Latvia�,  �Lithuania�,  �Netherlands�,  �Poland�,  �Portugal�,
�Slovenia�, �Spain�, �Switzerland�, �United Kingdom�, �United States�)
Note  that,  to  a  high  degree,  they  correspond  to  the  clusters  obtained  by  the  K-
means (except for permutation of cluster labels, which is irrelevant).

�Estonia�,

�EU15�,

�EU25�,

Figure 7.14 shows the construction of the clusters using complete linkage agglom-
erative clustering. Different cuts at different levels of the dendrogram allow us to
obtain different numbers of clusters.

To summarize, we can compare the results of the three clustering approaches. We
cannot  expect  the  results  to  coincide,  since  the  different  approaches  are based
on  different  criteria  for  constructing  clusters.  Nonetheless,  we  can  still  observe
that  in this  case,  K-means  and  the  agglomerative  approaches  gave  the  same
results (up to a permutation of the number of cluster, which is irrelevant); while
spectral  clustering gave  more  evenly  distributed  clusters.  This  later  approach
fused clusters 0 and 2 of the agglomerative clustering in cluster 1, and split cluster
3 of the agglomerative clustering into its clusters 0 and 3. Note that these results
could change when using different distances among data.

138

7    Unsupervised Learning

Fig.  7.14  Agglomerative  clustering  applied  to  cluster  European  countries  according  to  their  expen-
diture on education

Network  Analysis,  Graphs,  Social  Networks,  centrality,  drawing  centrality  of
Graphs, PageRank, Ego-Networks, community Detection

UNIT-5

Network Analysis
Introduction

Network  data  are  generated  when  we  consider  relationships  between  two  or
more  entities  in  the  data,  like  the  highways  connecting  cities,  friendships
between peo- ple or their phone calls. In recent years, a huge number of network
data  are  being  generated  and  analyzed  in  different  fields.  For  instance,  in
sociology there is inter- est in analyzing blog networks, which can be built based
on  their  citations,  to  look  for  divisions  in  their  structures  between  political
orientations. Another example is infectious disease transmission networks, which
are  built in  epidemiological  studies to  find  the  best  way  to  prevent  infection  of
people in  a  territory,  by isolating  cer-  tain areas. Other examples studied in the
field  of  technology  include  interconnected computer  networks  or  power  grids,
which are analyzed to optimize their functioning. We also find examples in academia,
where  we  can  build  co-authorship  networks  and  citation  networks  to  analyze
collaborations among Universities.

Structuring data  as  networks  can  facilitate  the  study  of  the  data  for  different
goals; for example, to discover the weaknesses of a structure. That could be the
objective of  a  biologist  studying  a  community  of  plants  and  trying  to  establish
which  of  its  properties  promote  quick  transmission  of  a  disease.  A  contrasting
objective  would  be to  find  and  exploit  structures  that  work  efficiently  for  the
transmission  of  messages  across  the  network.  This  may  be  the  goal  of  an
advertising agent trying to find the best strategy for spreading publicity.

How  to  analyze  networks  and  extract  the  features  we  want  to  study  are
some of the issues we consider in this chapter. In particular, we introduce some
basic  concepts  related  with  networks,  such  as  connected  components,  centrality
measures, ego-networks, and PageRank. We present some useful Python tools for
the analysis of networks and discuss some of the visualization options. In order to
motivate  and illustrate  the  concepts,  we  perform  social  network  analysis  using  real
data.  We  present  a  practical  case  based  on  a  public  dataset  which  consists  of  a
set of interconnected

Facebook friendship networks. We formulate multiple questions at different
levels: the local/member level, the community level, and the global level.
In general, some of the questions we try to solve are the following:

�
�

�

�

�

�

�
�

�
�
�

�

What type of network are we dealing with?
Which is the most representative member of the network in terms of being the
most connected to the rest of the members?
Which is the most representative member of the network in terms of being the
most circulated on the paths between the rest of the members?
Which is the most representative member of the network in terms of proximity
to the rest of the members?
Which is the most representative member of the network in terms of being the
most accessible from any location in the network?
There are many ways of calculating the representativeness or importance of a
member,  each  one  with  a  different  meaning,  so:  how  can  we  illustrate  them
and compare them?
Are there different communities in the network? If so, how many?
Does any member of the network belong to more than one community? That
is,  is  there  any  overlap  between  the  communities?  How  much  overlap?  How
can we illustrate this overlap?
Which is the largest community in the network?
Which is the most dense community (in terms of connections)?
How can we automatically detect the communities in the
network?
Is there any difference between automatically detected communities and real
ones (manually labeled by users)?

Basic Definitions in Graphs

Graph is the mathematical term used to refer to a network. Thus, the field that
studies networks is called graph theory and it provides the tools necessary to analyze
networks. Leonhard Euler defined the first graph in 1735, as an abstraction of one
of the problems posed by mathematicians of the time regarding Konigsberg, a city
with two islands created by the River Pregel, which was crossed by seven bridges.
The problem was: is it possible to walk through the town of Konigsberg crossing
each bridge once and only once? Euler represented the land areas as nodes and the
bridges  connecting  them  as edges  of  a graph  and  proved  that  the  walk  was  not
possible for this particular graph.

A  graph  is  defined  as  a  set  of  nodes,  which  are  an  abstraction  of  any  entities
(parts of a city, persons, etc.), and the connecting links between pairs  of nodes called
edges or relationships. The edge between two nodes can be  directed  or undirected.A
directed edge means that the edge points from one node to the other and not the
other way  round.  An  example  of  a  directed  relationship  is  �a  person  knows  another
person�. An edge has a direction when person A knows person B, and not the reverse

direction

Definitions in Graphs

Fig.  8.1  Simple  undirected
labeled  graph  with  5
nodes and5 edges

Basic

143

if B does not know A (which is usual for many fans and celebrities). An undirected
edge  means  that  there  is  a  symmetric  relationship.  An  example  is  �a  person
shook  hands  with  another  person�;  in  this  case,  the  relationship,  unavoidably,
involves both persons and there is no directionality. Depending on whether the edges
of  a  graph  are  directed  or  undirected,  the graph  is  called  a  directed  graph  or  an
undirected graph, respectively.

The  degree  of  a  node  is  the  number  of  edges  that  connect  to  it.  Figure  8.1
shows an example of an undirected graph with 5 nodes and 5 edges. The degree
of node C is 1, while the degree of nodes A, D and E is 2 and for node B it is 3. If a
network is directed, then nodes have two different degrees, the in-degree, which
is  the  number of  incoming  edges,  and  the  out-degree,  which  is  the  number  of
outgoing edges.

In  some cases,  there  is  information we  would  like  to  add  to  graphs  to  model
properties  of  the  entities  that  the  nodes  represent  or  their  relationships.  We  could
add strengths or weights to the links between the nodes, to represent some real-
world measure. For instance, the length of the highways connecting the cities in a
network. In this case, the graph is called a weighted graph.

Some other elementary concepts that are useful in graph analysis are those
we explain in what follows. We define a path in a network to be a sequence of
nodes connected by edges. Moreover, many applications of graphs require
shortest paths to be computed. The shortest path problem is the problem of
finding a path between two nodes in a graph such that the length of the path or
the sum of the weights of edges in the path is minimized. In the  example in Fig. 8.1,
the paths (C, A, B, E)  and (C, A, B, D, E) are those between nodes C and E. This
graph is unweighted, so the shortest path between C and E is the one that follows
the fewer edges: (C, A, B, E). A graph is said to be connected if for every pair of
nodes, there is a path between them. A graph is fully connected or complete if
each pair of nodes is connected by an edge. A connected component or simply a
component of a graph is a subset of its nodes such that every node in the subset has
a path to every other one. In the example of  Fig. 8.1,  the  graph  has  one  connected
component.  A  subgraph  is  a  subset  of  the nodes of a graph and all the edges
linking those nodes. Any group of nodes can form

a subgraph.

Social Network Analysis

Social network analysis processes social data structured in graphs. It involves the
extraction of several characteristics and graphics to describe the main properties
of the  network.  Some  general  properties  of  networks,  such  as  the  shape  of  the
network  degree  distribution  (defined  bellow)  or  the  average  path  length,
determine  the  type of  network,  such  as  a  small-world  network  or  a  scale-free
network.  A  small-world  network is a type of graph in which most nodes are not
neighbors of one another, but most nodes can be reached from every other node
in a small number of steps. This is the so-called small-world phenomenon which
can  be  interpreted  by  the  fact  that  strangers  are  linked  by  a  short  chain  of
acquaintances.  In  a  small-world  network,  people  usually  form  communities  or
small  groups  where  everyone  knows  every-  one  else.  Such  communities can  be
seen as complete graphs. In addition, most the community members have a few
relationships  with  people  outside  that  community. However,  some  people  are
connected to a large number of communities. These may be celebrities  and  such
people  are  considered  as  the  hubs  that  are  responsible  for  the  small-world
phenomenon.  Many  small-world  networks  are  also  scale-free  net-  works.  In  a
scale-free  network  the  node  degree  distribution  follows  a  power  law  (a
relationship function between two quantities x and y defined as y      xn, where n
is a constant). The name scale-free comes from the fact that power laws have the
same  functional  form  at  all  scales,  i.e.,  their  shape  does  not  change  on
multiplication by  a scale  factor.  Thus,  by  definition,  a  scale-free  network  has  many
nodes  with  a  very  few  connections  and  a  small  number  of  nodes  with  many
connections.  This  structure  is typical  of  the  World  Wide  Web  and  other  social
networks. In  the following  sections, we  illustrate  this  and  other  graph  properties
that are useful in social network analysis.

=

In [1]:

Basics in NetworkX

NetworkX1  is  a  Python  toolbox  for  the  creation,  manipulation  and  study  of  the
struc- ture,  dynamics  and  functions  of  complex  networks.  After  importing  the
toolbox, we can create an undirected graph with 5 nodes by adding the edges, as
is done in the following code. The output is the graph in Fig. 8.1.

import   ne two rk x  as  nx
G  =  nx . Graph ()
G. add _ed ge  ( �A �,  �B �);
G. add _ed ge  ( �A �,  �C �);
G. add _ed ge  ( �B �,  �D �);
G. add _ed ge  ( �B �,  �E �);
G. add _ed ge  ( �D �,  �E �);
nx . d r aw _n et w or kx  ( G)

To create a directed graph we would use nx.DiGraph().

Practical Case: Facebook Dataset

For our practical case we consider data from the Facebook network. In particular, we
use the data Social circles: Facebook2 from the Stanford Large Network Dataset3
(SNAP)  collection.  The  SNAP  collection  has  links  to  a  great  variety  of  networks
such  as  Facebook-style  social  networks,  citation  networks,  Twitter  networks  or
open communities  like  Live  Journal.  The  Facebook  dataset consists  of  a  network
repre-  senting  friendship  between  Facebook  users.  The  Facebook  data  was
anonymized  by replacing  the  internal  Facebook  identifiers  for  each  user  with  a
new value.

The  network  corresponds  to  an  undirected  and  unweighted  graph  that
contains  users  of  Facebook  (nodes)  and  their  friendship  relations  (edges).  The
Facebook dataset is defined by an edge list in a plain text file with one edge per
line.

Let  us  load  the  Facebook  network  and  start  extracting  the  basic  information
from the  graph,  including  the  numbers  of  nodes  and  edges,  and  the  average
degree:

fb  =  nx . read_ edgel ist  (" files / ch08 / fa ce boo k_ com bi n ed  . txt ")
fb_n ,  fb_k  =  fb . order () ,  fb . size ()
fb_avg_ deg   =  fb_k  /  fb_n
print  � Nodes :  �,  fb_n
print  � Edges :  �,  fb_k
print  � Average   degree :  �,  fb_av g_de g

In [2]:

Out[2]:  Nodes: 4039

Edges: 88234
Average degree: 21

The  Facebook  dataset  has  a  total  of  4,039  users  and  88,234  friendship
connections,  with  an  average  degree  of  21.  In  order  to  better  understand  the
graph,  let  us  compute the  degree  distribution  of  the  graph.  If  the  graph  were
directed, we would need to generate two distributions: one for the in-degree and
another  for  the  out-degree.  A  way  to  illustrate  the  degree  distribution  is  by
computing  the  histogram  of  degrees  and  plotting  it,  as  the  following  code  does
with the output shown in Fig. 8.2:

In [3]:

d e g r e e s   =  fb . d e g r e e  () . v a l u e s  ()
d e g r e e _ h i s t   =  pl t . h i s t  ( d e g r e e s  ,  10 0 )

The  graph in  Fig. 8.2  is  a  power-law  distribution. Thus,  we  can  say that the  Face-

book network is a scale-free network.

Next, let us find out if the Facebook dataset contains more than one

connected component (previously defined in Sect. 8.2):

In [4]:

print  �#  co nnec ted   c ompo nents   of  Fa ceboo k  network  :  �,

nx . n um b er _c o nn e ct ed _ co m po ne n ts  ( fb )

    Out[4]:  # connected components of Facebook network: 1

As it can  be seen, there is only  one connected component in the Facebook network.
Thus, the Facebook network is a connected graph (see definition in Sect.  8.2). We can
try  to  divide  the  graph  into  different  connected  components,  which  can  be
potential communities (see Sect. 8.6). To do that, we can remove one node from
the graph (this operation also involves removing the edges linking the node) and
see  if  the  number  of  connected  components  of  the  graph  changes.  In  the
following code, we prune the graph by removing node �0� (arbitrarily selected) and
compute  the  number of  connected  components  of  the  pruned  version  of  the
graph:

In [5]:

fb_prun  =  nx . read_ edge list  (

" files / ch08 / f ac eb o ok _c om bi ne d  . txt ")

fb_prun . r emove _node  ( �0 �)
print  � Rem ain ing   nodes : �, f b_prun  . nu mb er _o f _n od es  ()
print  � New  #  c onn ect ed   com pon ent s  : �,

nx . nu m b er _ c on n e ct e d _c o mp o n en t s  ( fb_pru n  )

Out[5]:  Remaining nodes: 4038

New # connected components: 19

Now there are 19 connected components, but let us see how big the biggest is

and how small the smallest is:

In [6]:

f b _ c o m p o n e n t s   =  nx . c o n n e c t e d _ c o m p o n e n t s  ( f b _ p r u n  )
p r i nt   � S i z e s   of  th e  c o n n e c t e d   c o m p o n e n t s  � ,

[ len ( c )   fo r  c  in  f b _ c o m p o n e n t s  ]

Out[6]:  Sizes of the connected components [4015, 1, 3, 2, 2, 1, 1, 1,

1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]

This simple example shows that removing a node splits the graph into multiple
components. You can see that there is one large connected component and the
rest  are  almost  all  isolated  nodes.  The  isolated  nodes  in  the  pruned  graph
were  only Fig. 8.2  Degree histogram  distribution

connected  to  node  �0�  in  the  original  graph  and  when  that  node  was  removed
they  were  converted  into  connected  components  of  size  1.  These  nodes,  only
connected to one neighbor, are probably not important nodes in the structure of
the graph. We can generalize the analysis by studying the centrality of the nodes.
The next section is devoted to explore this concept.

Centrality

influential,  have  greater  access  to

The centrality of a node measures its relative importance within the graph. In this
section we focus on undirected graphs. Centrality concepts were first developed
in social  network  analysis.  The  first  studies  indicated  that  central  nodes  are
probably  more
information,  and  can
communicate their opinions to others more efficiently [1]. Thus, the applications
of  centrality  concepts in  a  social  network  include  identifying  the  most  influential
people, the most informed people, or the most communicative people. In practice,
what  centrality  means  will  depend  on  the  application  and  the  meaning  of  the
entities  represented  as  nodes  in  the data  and  the  connections  between  those
nodes.  Various  measures  of  the  centrality  of  a  node  have  been  proposed.  We
present  four  of  the  best-known  measures:  degree  centrality,  betweenness
centrality, closeness centrality, and eigenvector centrality.

Degree centrality is defined as the number of edges of the node. So the more
ties  a node  has,  the  more  central  the  node  is.  To  achieve  a  normalized  degree
centrality  of  a  node,  the  measure  is  divided  by  the  total  number  of  graph  nodes  (n)
without  counting  this  particular  one  (n  1).  The  normalized  measure  provides
proportions and allows us to compare it among graphs. Degree centrality is related
to the capacity of a node to capture  any information  that is  floating  through  the
network. In social networks, connections are associated with positive aspects such
as knowledge or friendship.

?

Betweenness centrality quantifies the number of times a node is crossed along
the  shortest  path/s  between  any  other  pair  of  nodes.  For  the  normalized
measure  this number is divided by the total number of shortest paths for every
pair of nodes. Intuitively, if  we  think  of  a  public  bus  transportation  network,  the
bus  stop  (node)  with  the  highest  betweenness  has  the  most  traffic.  In  social
networks,  a  person  with high  betweenness  has  more  power  in  the  sense  that
more  people  depend  on  him/her to  make  connections  with  other  people  or  to
access  information  from  other  people. Comparing  this  measure  with  degree
centrality,  we  can  say  that  degree  centrality  depends  only  on  the  node�s
neighbors; thus, it is more local than the betweenness centrality, which depends
on the connection properties of every pair of nodes in the graph, except pairs with
the  node  in  question  itself.  The  equivalent  measure  exists  for  edges.  The
betweenness  centrality  of  an  edge  is  the  proportion  of  the  shortest  paths
between all node pairs which pass through it.

Closeness  centrality  tries  to  quantify  the  position  a  node  occupies  in  the

network based on a distance calculation. The distance metric used between a pair
of nodes is defined by the length of its shortest path. The closeness of a node is
inversely  proportional  to  the  length  of  the  average  shortest  path  between  that
node and all the

other nodes in the graph. In this case, we interpret a central node as being close
to, and able to communicate quickly with, the other nodes in a social network.

Eigenvector  centrality  defines  a  relative  score  for  a  node  based  on  its
connections  and  considering  that  connections  from  high  centrality  nodes
contribute  more  to  the score  of  the  node  than  connections  from  low  centrality
nodes.  It  is  a  measure  of  the influence  of  a  node  in  a  network,  in  the  following
sense: it measures the extent to which a node is connected to influential nodes.
Accordingly, an important node is connected to important neighbors.

=

Let  us  illustrate  the  centrality  measures  with  an  example.  In  Fig.  8.3,  we
show an undirected star graph with n      8 nodes. Node C is obviously important,
since it can  exchange  information  with more  nodes  than  the  others.  The  degree
centrality measures  this  idea.  In  this  star  network,  node  C  has  a  degree
centrality  of  7  or  1 if  we  consider  the  normalized  measure,  whereas  all  other
nodes have a degree of 1 or 1/7 if we consider the normalized measure. Another
reason why node C is more important than the others in this star network is that it
lies between each of the other pairs of nodes, and no other node lies between C
and any other node. If node C wants to contact F, C can do it directly; whereas if
node F wants to contact B, it must go through C. This gives node C the capacity to
broke/prevent contact among other nodes and to isolate nodes from information.
The  betweenness  centrality  is  underneath  this  idea.  In  this  example,  the
betweenness centrality of the node C is 28, computed as (n  1)(n  2)/2,  while the
rest of nodes have a betweenness of 0. The final reason why we can say node C is
superior in the star network is because C is closer to more nodes than any other
node is. In the example, node C is at a distance of 1 from all other 7 nodes and each
other  node  is  at  a  distance  2  from  all  other  nodes,  except  C.  So,  node  C  has
closeness centrality of 1/7, while the rest of nodes have a closeness of 1/13. The
normalized measures, computed by dividing by  n 1, are 1 for C and 7/13 for the
other nodes.

�

?

?

An important concept in social network analysis is that of a hub node, which is
defined as a node with high degree centrality and betweenness centrality. When
a hub governs a very centralized network, the network can be easily fragmented
by removing that hub.

Coming back to the Facebook example, let us compute the degree centrality of
Facebook graph nodes. In the  code below we show the  user identifier of the 10
most central nodes together with their normalized degree centrality measure. We
also  show  the  degree  histogram  to  extract  some  more  information  from  the
shape  of  the distribution.  It  might  be  useful  to  represent  distributions  using
logarithmic scale. We

Fig. 8.3  Star graph example

8.4    Centrality

149

In [7]:

do  that  with  the  matplotlib.loglog() function.  Figure  8.4  shows  the  degree
centrality histogram in linear and logarithmic scales as computed in the box
bellow.

degree_ cent_ fb   =  nx . deg re e_ cen tr ali ty  ( fb )
print  � Facebook   degree  cent ralit y :  �,

sorted ( de gree_ cent_ fb  . items () ,

key  =  lambda  x:  x [1] ,
reverse  =  True ) [:10]

degree_ hist   =  plt . hist ( list ( d egree _cent _fb  . values () ) ,  100)
plt . loglog ( degre e_his t  [1][1:] ,

degree_ hist  [0] ,  �b �,  marker  =  �o �)

Out[7]:  Facebook degree centrality: [(u�107�, 0.258791480931154),

(u�1684�, 0.1961367013372957), (u�1912�, 0.18697374938088163),
(u�3437�, 0.13546310054482416), (u�0�, 0.08593363051015354),
(u�2543�, 0.07280832095096582), (u�2347�, 0.07206537890044576),
(u�1888�, 0.0629024269440317), (u�1800�, 0.06067360079247152),
(u�1663�, 0.058197127290737984)]

The  previous  plots  show  us  that  there  is  an  interesting  (large)  set  of  nodes
which corresponds  to  low  degrees.  The  representation  using  a  logarithmic  scale
(right-hand graphic in Fig. 8.4) is useful to distinguish the members of this set of
nodes, which are clearly visible as a straight line at low values for the x-axis (upper
left-hand part of the logarithmic plot). We can conclude that most of the nodes in
the  graph  have  low  degree  centrality;  only  a  few  of  them  have  high  degree
centrality.  These latter  nodes  can  be properly  seen  as  the points  in  the  bottom
right-hand part of the logarithmic plot.

The  next  code  computes  the  betweenness,  closeness,  and  eigenvector

centrality and prints the top 10 central nodes for each measure.

Fig. 8.4  Degree centrality  histogram shown using  a linear scale (left) and a  log scale for both the
x- and y-axis (right)

8    Network Analysis

150

In [8]:

between ness_ fb   =  nx . b e tw ee n ne s s_ ce n tr a li ty  ( fb )
closene ss_fb   =  nx . c l os en e ss _ ce nt r al i ty  ( fb )
ei ge nc en tr al it y _f b   =  nx . e ig en ve c to r_ ce nt ra li ty  ( fb )
print  � Fac ebook   be tween ness   ce ntral ity  : �,
sorted ( be tween ness_ fb  . items () ,

key  =  lambda  x:  x [1] ,
reverse  =  True ) [:10]

print  � F ace boo k  clo sen ess   c ent ral ity  : �,
sorted ( clo senes s_fb  . items () ,
key  =  lambda  x:  x [1] ,
reverse  =  True ) [:10]

print  � Face book  eigen vect or   c entr ality  : �,

sorted ( ei ge nc en tr al it y _f b  . items () ,

key  =  lambda  x:  x [1] ,
reverse  =  True ) [:10]

Out[8]:  Facebook betweenness centrality: [(u�107�, 0.4805180785560141), (u�1684�,
0.33779744973019843), (u�3437�, 0.23611535735892616),
(u�1912�, 0.2292953395868727), (u�1085�, 0.1490150921166526),
(u�0�, 0.1463059214744276), (u�698�, 0.11533045020560861),
(u�567�, 0.09631033121856114), (u�58�, 0.08436020590796521),
(u�428�, 0.06430906239323908)]

Out[8]:  Facebook closeness centrality: [(u�107�, 0.45969945355191255), (u�58�, 0.3974018305284913),

(u�428�, 0.3948371956585509),
(u�563�, 0.3939127889961955), (u�1684�, 0.39360561458231796),
(u�171�, 0.37049270575282134), (u�348�, 0.36991572004397216),
(u�483�, 0.3698479575013739), (u�414�, 0.3695433330282786),
(u�376�, 0.36655773420479304)]
Facebook eigenvector centrality: [(u�1912�, 0.09540688873596524), (u�2266�,
0.08698328226321951), (u�2206�, 0.08605240174265624),
(u�2233�, 0.08517341350597836), (u�2464�, 0.08427878364685948),
(u�2142�, 0.08419312450068105), (u�2218�, 0.08415574433673866),
(u�2078�, 0.08413617905810111), (u�2123�, 0.08367142125897363),
(u�1993�, 0.08353243711860482)]

As can be seen in the previous results, each measure gives a different ordering
of the  nodes.  The  node  �107�  is  the  most  central  node  for  degree  (see  box  Out
[7]), betweenness, and closeness centrality,  while it is not among the 10 most central
nodes  for  eigenvector  centrality.  The  second  most  central  node  is  different  for
closeness  and  eigenvector  centralities;  while  the  third  most  central  node  is
different for all four centrality measures.

Another interesting measure is the current flow betweenness centrality, also called
random  walk  betweenness  centrality,  of  a  node.  It  can  be  defined  as  the
probability of  passing  through  the  node  in  question  on  a  random  walk  starting
and  ending  at  some  node.  In  this  way,  the  betweenness  is  not  computed  as  a
function  of  shortest  paths,  but  of  all  paths.  This  makes  sense  for  some  social
networks  where  messages may  get  to  their  final  destination  not  by the  shortest
path,  but  by  a  random  path,  as  in  the  case  of  gossip  floating  through  a  social
network for example.

Computing  the  current  flow  betweenness  centrality  can  take  a  while,  so  we

In [9]:

will  work  with  a  trimmed  Facebook  network  instead  of  the  original  one. In  fact,
we can

pose the question: What happen if we only consider the graph nodes with more
than the average degree of the network (21)? We can trim the graph using degree
centrality values.  To  do  this,  in  the  next  code,  we  define  a  function  to  trim  the
graph based on the degree centrality of the graph nodes. We set the threshold to
21 connections:

def  t r im _ de gr e e_ c en tr a li t y  ( graph ,  degree  =  0.01) :

g  =  graph . copy ()
d = nx . d eg re e_ ce nt ra l it y  ( g)
for  n  in  g. nodes () :

if  d[ n]  <=  degree :

g. r emove _nod e  ( n)

return  g

thr  =  21.0/( fb . order ()  -  1.0)

print  � Degree  cen trali ty   thre shold  : �,  thr

fb_t rim med   = t r i m_ d e gr e e _c e n tr a li t y  ( fb , degree  = thr )
print  � R ema ini ng   #  nodes : �,  len ( f b_t rim med  )

Out[9]:  Degree centrality threshold: 0.00520059435364 Remaining # nodes:

2226

In [10]:

The  new  graph  is  much  smaller;  we  have  removed  almost  half  of  the  nodes

(we have moved from 4,039 to 2,226 nodes).

The  current  flow  betweenness  centrality  measure  needs  connected  graphs,  as
does any betweenness centrality measure, so we should first extract a connected
compo-  nent  from  the  trimmed  Facebook  network  and  then  compute  the
measure:

fb_subg raph   =  list ( nx . co nn e ct e d_ co m po n en t_ s ub g ra p hs  (

fb_trim ed ))

print  �#  s ubgr aphs   found : �,  size ( fb_s ubgra ph  )
print  �#  nodes  in  the  first  s ubgraph  : �,

len ( fb _subg raph  [0])

between ness   =  nx . b et w ee n ne ss _ ce n tr al i ty  ( fb_ subgr a ph  [0])
print  � Trimmed  FB  betw eenne ss  :  �,

sorted ( betw eenne ss  . items () ,  key  =  lambda  x:  x [1] ,

current _flow   =  nx . c u rr e nt _f l ow _ be tw e en n es s_ c en t ra li ty  (

reverse  =  True ) [:10]

fb_subg raph [0])

print  � Trim med   FB  cur rent   flow  be twee nne ss  : �,

sorted ( curr ent_f low  . items () ,  key  =  lambda  x:  x [1] ,
reverse  =  True ) [:10]

152

Fig.  8.5  The  Facebook
network
a
random layout

with

8    Network Analysis

Out[10]: #  subgraphs  found:  2

# nodes in the first subgraph: 2225
Trimmed FB betweenness: [(u�107�, 0.5469164906683255),
(u�1684�, 0.3133966633778371), (u�1912�, 0.19965597457246995),
(u�3437�, 0.13002843874261014), (u�1577�, 0.1274607407928195),
(u�1085�, 0.11517250980098293), (u�1718�, 0.08916631761105698),
(u�428�, 0.0638271827912378), (u�1465�, 0.057995900747731755),
(u�567�, 0.05414376521577943)]
Trimmed FB current flow betweenness: [(u�107�,
0.2858892136334576), (u�1718�, 0.2678396761785764), (u�1684�,
0.1585162194931393), (u�1085�, 0.1572155780323929), (u�1405�,
0.1253563113363113), (u�3437�, 0.10482568101478178), (u�1912�,
0.09369897700970155), (u�1577�, 0.08897207040045449), (u�136�,
0.07052866082249776), (u�1505�, 0.06152347046861114)]

As  can  be  seen,  there  are  similarities  in  the  10  most  central  nodes  for  the
between- ness and current flow betweenness centralities. In particular, seven up
to ten are the same nodes, even if they are differently ordered.

Drawing  Centrality  in  Graphs

In  this  section  we  focus  on  graph  visualization,  which  can  help  in  the  network
data understanding and usability.

The visualization of a network with a large amount of nodes is a complex task.
Different layouts can be used to try to build a proper visualization. For instance,
we can  draw  the  Facebook  graph  using  the  random  layout  (nx.random_layout),
but this is a bad option, as can be seen in Fig. 8.5. Other alternatives can be more
useful. In the box below, we use the Spring layout, as it is used in the default function
(nx.draw),  but  with  more  iterations.  The  function  nx.spring_layout  returns  the
position of the nodes using the Fruchterman�Reingold force-directed algorithm.

8.4    Centrality

Fig. 8.6 The Facebook
network drawn using the
Spring layout and degree
centrality to define the node
size

153

This  algorithm  distributes  the  graph  nodes  in  such  a way  that  all  the  edges  are
more or  less  equally  long  and  they  cross  themselves  as  few  times  as  possible.
Moreover, we  can  change  the  size  of  the  nodes  to  that defined by  their degree
centrality.  As  can  be  seen  in  the  code,  the  degree  centrality  is  normalized  to
values  between  0  and  1,  and  multiplied  by  a  constant  to  make  the  sizes
appropriate for the format of the figure:

In [11]:

pos_fb  =  nx . sp ring_ layou t  ( fb , it erati ons   =  1000)

nsize  =  np . array ([ v  for  v  in  degre e_cen t_fb  . values () ])

nsize  = 500*( nsize  -  min ( nsize )) /( max ( nsize )  -  min ( nsize ))

nodes  =  nx . d ra w_ ne tw or kx _n o de s  ( fb ,  pos  =  pos_fb ,

node _si ze   =  nsize )

edges  =  nx . d ra w_ ne t wo rk x_ ed ge s  ( fb ,  pos  =  pos_fb  ,

alpha  =  . 1 )

The resulting graph visualization is shown in Fig. 8.6. This illustration allows us
to understand the network better. Now we can distinguish several groups of nodes
or �communities� clearly in the graph. Moreover, the larger nodes are  the more
central nodes, which are highly connected of the Facebook graph.

We can also use the betweenness centrality to define the size of the nodes. In this
way, we obtain a new illustration stressing the nodes with higher betweenness, which
are  those  with  a  large  influence  on  the  transfer  of  information  through  the
network. The new graph is shown in Fig.  8.7. As expected, the central nodes are
now those connecting the different communities.

Generally  different  centrality  metrics  will  be  positively  correlated,  but  when
they are  not,  there  is  probably  something  interesting  about  the  network  nodes.  For
instance,  if you can spot nodes with high betweenness but relatively low degree,
these are the nodes with few links but which are crucial for network flow. We can
also look for

154

Fig. 8.7 The Facebook
network drawn using
the Spring layout and
betweenness centrality
to define the node size

8    Network Analysis

the  opposite  effect:  nodes  with  high  degree  but  relatively  low  betweenness.
These nodes are those with redundant communication.

Changing the centrality measure to closeness and eigenvector,  we obtain the
graphs  in  Figs.  8.8  and  8.9,  respectively.  As  can  be  seen,  the  central  nodes
are also different for these measures. With this or other visualizations you will be
able to  discern  different  types  of  nodes.  You  can  probably  see  nodes  with  high
closeness centrality  but  low  degree;  these  are  essential  nodes  linked  to  a  few
important  or  active  nodes.  If  the  opposite  occurs,  if  there  are  nodes  with  high
degree centrality but low closeness, these can be interpreted as nodes embedded
in a community that is far removed from the rest of the network.

In other examples of social networks, you could find nodes with high closeness
centrality  but  low  betweenness;  these  are  nodes  near  many  people,  but  since
there  may  be  multiple  paths  in  the  network,  they  are  not  the  only  ones  to  be
near  many  people.  Finally,  it  is  usually  difficult  to  find  nodes  with  high
betweenness but low closeness, since this would mean that the node in question
monopolized the links from a small number of people to many others.

PageRank

PageRank  is  an  algorithm  related  to  the  concept  of  eigenvector  centrality  in
directed graphs.  It  is  used  to rate  webpages  objectively  and  effectively measure
the  attention devoted to them. PageRank was invented by Larry Page and Sergey
Brin, and became a Google trademark in 1998 [2].

Assigning the importance of a webpage is a subjective task, which depends on the
interests  and  knowledge  of  the  persons  that  browse  the  webpages.  However,
there are ways to objectively rank the relative importance of webpages.

Centrality

155

Fig. 8.8 The Facebook
network drawn using the
Spring layout and closeness
centrality to define the
node size

Fig. 8.9 The Facebook
network drawn using
the Spring layout and
eigenvector centrality
to define the node size

We  consider  the  directed  graph  formed  by  nodes  corresponding  to  the
webpages and edges corresponding to the hyperlinks. Intuitively, a hyperlink to a
page counts as a vote of support and a page has a high rank if the sum of the ranks
of  its  incoming edges  is  high.  This  considers  both  cases  when  a  page  has  many
incoming  links  and  when  a  page  has  a  few  highly  ranked  incoming  links.
Nowadays,  a  variant  of  the  algorithm  is  used  by  Google.  It  does  not  only  use
information  on  the  number  of  edges  pointing into  and  out  of  a  website,  but  uses
many more variables.

We can describe the PageRank algorithm from a probabilistic point of view. The
rank of page Pi is the probability that a surfer on the Internet who starts visiting a
random  page  and  follows  links,  visits  the  page Pi .  With  more  details,  we  consider

that  the  weights  assigned  to  the  edges  of  a  network  by  its  transition  matrix,  M,  are
the  probabilities  that  the  surfer  goes  from  one  webpage  to  another.  We  can
understand the

Fig. 8.10 The Facebook
network drawn using the
Spring layout and PageRank
to define the node size

n

=
=

rank computation as a random walk through the network. We start with an initial equal
probability  for  each  page:  v0  =  ( 1 , . . . ,   1 ),  where  n  is  the  number  of  nodes.  Then
n
we can compute the probability that each page is visited after one step by applying
the transition matrix: v1      Mv. The probability that each page will be visited after
k  steps  is  given  by  vk      Mka.  After  several  steps,  the  sequence  converges  to  a
unique  probabilistic  vector  a?  which  is  the  PageRank  vector.  The  i -th  element  of
this vector is the probability that at each moment the surfer visits page Pi . We need a
nonambiguous  definition  of  the  rank  of  a  page  for  any  directed  web  graph.
However,
in  the  Internet,  we  can  expect  to  find  pages  that  do  not  contain  outgoing  links
and this configuration can lead to certain problems to the explained procedure.
In  order to  overcome  this  problem,  the  algorithm  fixes  a  positive  constant  p
between  0  and  1  (a  typical  value  for  p  is  0.85)  and  redefines  the  transition
matrix of the graph by
R  =  (1 ?  p) M  +  p  B,  where  B  =  1 I ,  and  I  is  the  identity  matrix.  Therefore,  a
n
node with no outgoing edges has probability  n  of moving to any other node.

Let us compute the PageRank vector of the Facebook network and use it to define

  p

the size of the nodes, as was done in box In [11].

In [12]:

pr  =  nx . pagerank  ( fb ,  alpha  =  0.85)
nsize  =  np . array ([ v  for  v  in  pr . values () ])
nsize  =  500*( nsize  - min ( nsize )) /( max ( nsize ) - min ( nsize ))
nodes  =  nx . dr aw _ne tw ork x_ no des  ( fb ,

edges  =  nx . d r aw _n et wo rk x_ ed g es  ( fb ,

pos  =  pos_fb  ,
alpha  =  . 1 )

pos  =  pos_fb  ,
node _si ze   =  nsize )

The  code  above  outputs the graph  in  Fig.  8.10,  that emphasizes  some  of  the
nodes with high PageRank. Looking the graph carefully one can realize that there
is one large node per community.

8.5  Ego-Networks

Ego-networks are subnetworks of neighbors that are centered on a certain node.
In Facebook and LinkedIn, these are described as �your network". Every person in
an ego-network has her/his own ego-network and can only access the nodes in it.
All  ego-networks  interlock  to  form  the  whole  social  network.  The  ego-network
definition  depends  on  the  network  distance  considered.  In  the  basic  case,  a
distance  of  1,  a  link means  that  person  A  is  a  friends  of  person  B,  a  distance  of  2
means  that  a  person,  C, is a friend of a friend of A, and a distance of 3 means that
another person, D, is a friend of  a friend  of a  friend  of  A. Knowing  the  size  of  an
ego-network  is  important  when it  comes  to  understanding  the  reach  of  the
information  that  a  person can  transmit  or have  access  to.  Figure  8.11  shows  an
example of an ego-network. The blue node is the ego, while the rest of the nodes
are red.

Our  Facebook  network  was  manually  labeled  by  users  into  a  set  of  10  ego-
networks.  The  public  dataset  includes  the  information  of  these  10  manually
defined ego-networks. In particular, we have available the list of the 10 ego nodes:
�0�, �107�, �348�,  �414�,  �686�,  �1684�,  �1912�,  �3437�,  �3980�  and  their  connections.
These  ego-networks  are  interconnected  to  form  the  fully  connected  graph  we
have been analyzing in previous sections.

In Sect. 8.4 we saw that node �107� is the most central node of the Facebook
network  for  three  of  the  four  centrality  measures  computed.  So,  let  us  extract
the ego-networks of the popular node �107� with a distance of 1 and 2, and compute
their sizes. NetworkX has a function devoted to this task:

ego_107  =  nx . ego_gr aph  ( fb ,  � 107 �)
print  �#  nodes  of  ego  graph  107: �,

len ( ego_ 107  )

print  �#  nodes  of  ego  graph  107  with  radius  up  to  2: �,

len ( nx . eg o_g rap h  ( fb ,  � 107 �,  radius  =  2) )

In [13]:

Fig. 8.11 Example of an ego-
network. The blue node is
the ego

Out[13]: #  nodes  of  ego  graph  107:  1046

# nodes of ego graph 107 with radius up to 2: 2687

The  ego-network  size is  1,046  with  a  distance  of  1,  but  when  we expand  the
distance to 2, node �107� is able to reach up to 2,687 nodes. That is quite a large
ego-network, containing more than half of the total number of nodes.

Since  the  dataset  also  provides  the  previously  labeled  ego-networks,  we  can
com- pute the actual size of the ego-network following the user labeling. We can
access the ego-networks by simply importing os.path and reading the edge list corre-
sponding, for instance, to node �107�, as in the following code:

In [14]:

i m p o r t   os . p a th
ego_id   =  107
G_107  =  nx . re ad _e d ge li s t  (

os . p a t h  . j oi n  ( � f i l es  / c h 0 8  / f a c e b o o k  �,

� { 0 }.  e d g es  �. f o r m a t  ( e g o _ i d  )) ,

node ty pe  =  int )

p r i nt   � N o d e s   of  the   ego   g r ap h   1 0 7:  �,  l en ( G _ 1 0 7  )

Out[14]: Nodes  of  the  ego  graph  107:  1034

As can be seen, the size of the previously defined ego-network of node �107� is
slightly different from the ego-network automatically computed using NetworkX.
This is due to the fact that the manual labeling is not necessarily referred to the
subgraph of neighbors at a distance of 1.

We  can  now  answer  some  other  questions  about  the  structure  of  the
Facebook  network  and  compare  the  10  different  ego-networks  among  them.
First,  we  can  compute  which  the  most  densely  connected  ego-network  is  from
the total of 10. To do that, in the code below, we compute the number of edges in
every ego-network and select the network with the maximum number:

In [15]:

ego_ids   =  (  0 ,  107 ,  348 ,

414 ,  686 ,  698 ,
1684 ,  1912 ,  3437 ,  3980)

ego_ siz es   =  zeros ((10 ,  1) )
i  =  0
#  Fill  the  � ego_sizes �  vector  with  the  size  (#  edges )  of  the

10  ego - networks   in  egoids

for  id  in  ego_ids  :

G  =  nx . re ad_ed gelis t  (

os . path . join ( � files / ch08 / facebook  �,

� {0}. edges �. format ( id )) ,

nodetype  =  int )
ego_siz es  [ i]  =  G. size ()
i  =  i  +  1

[ i_max , j] = ( eg o_s ize s   ==  e go_ siz es  . max () ). nonzero  ()
ego_max  =  ego_ids [ i_max ]
print � The most densel y  conn ected  ego - netwo rk  is \

that  of  node : �,  ego_max

G  =  nx . r ead_e dgeli st  (

os . path . join ( � files / ch08 / facebook  �,

� {0}. edges �. format ( ego_max )) ,

nodetype  =  int )

print  � Nodes :  �,  G. order ()
print  � Edges :  �,  G. size ()
print  � Averag e  degree :  �,  G_k  /  G_n

Out[15]: The  most  densely  connected  ego-network  is  that  of  node:  1912 Nodes: 747

Edges: 30025
Average degree: 40

The most densely connected ego-network is that of node �1912�, which has an
average degree of 40. We can also compute which is the largest (in number of nodes)
ego-network, changing the measure of sizes from  G.size()  by  G.order(). In this case,
we  obtain  that  the  largest  ego-network  is  that  of  node  �107�,  which  has  1,034
nodes and an average degree of 25.

Next let us work out how much intersection exists between the ego-networks
in the  Facebook  network.  To  do  this,  in  the  code  below,  we  add  a  field  �egonet�  for
every  node and store an array with the ego-networks the node belongs to. Then,
having the length of these arrays, we compute the number of nodes that belong to
1, 2, 3, 4 and more than 4 ego-networks:

In [16]:

# Add a field � egonet � to the nodes of the whole face bo ok

network .

#  Default  value  egonet  =  [] ,  meaning  that  this  node  does  not

belong  to  any  ego - netowrk

for  i  in  fb . nodes ()  :

fb . node [ str (i) ][ � egonet �]  =  []

#  Fill  the  � egonet �  field  with  one  of  the  10  ego  values  in

ego_ids :

for  id  in  ego_ids  :

G  =  nx . re ad_ed gelis t  (

os . path . join ( � files / ch08 / facebook  �,

� {0}. edges �. format ( id )) ,

nodetype  =  int )

print  id
for  n  in  G. nodes ()  :

if  ( fb . node [ str (n) ][ � egonet �]  ==  [])  :

fb . node [ str (n) ][ � egonet �]  =  [ id ]

else  :

fb . node [ str (n) ][ � egonet �]. append ( id )

#  Compute  the  inter secti ons  :
S  =  [ len (x[ � egonet �])  for  x  in  fb . node . values () ]

print
sum ( equa l (S ,
sum ( equa l (S ,
print
sum ( equa l (S ,
print
sum ( equa l (S ,
print
print �# nodes into more than  4 ego - network  : � ,\
sum ( equa l (S ,
print

�#
�#
�#
�#
�#
sum ( greater  ( S ,  4) )

ego - netwo rk  :
ego - netwo rk  :
ego - netwo rk  :
ego - netwo rk  :
ego - netwo rk  :

nodes
nodes
nodes
nodes
nodes

into
into
into
into
into

�,
�,
�,
�,
�,

0
1
2
3
4

0) )
1) )
2) )
3) )
4) )

Out[16]: #  nodes  into  0  ego-network:  80

# nodes into 1 ego-network: 3844
# nodes into 2 ego-network: 102
# nodes into 3 ego-network: 11
# nodes into 4 ego-network: 2
# nodes into more than 4 ego-network: 0

As can be seen, there is an intersection between the ego-networks in the Facebook
network,  since  some  of  the  nodes  belong  to  more  than  1  and  up  to  4  ego-
networks simultaneously.

We can also try to visualize the different ego-networks. In the following code,
we draw the ego-networks using different colors on the whole Facebook network
and  we  obtain  the  graph  in  Fig.  8.12.  As  can  be  seen,  the  ego-networks  clearly
form groups of nodes that can be seen as communities.

Networks

Fig. 8.12 The Facebook
network drawn using the
Spring layout and different
colors to separate the
ego-networks

Ego-

161

In [17]:

� eg ocolor �  to  the  nodes  of  the  whole  fa cebook  net w ork .

#  Add  a  field
#  D efa ult   value  eg ocolor  r  =0 ,  mea ning   that   this  node
does  not  belong  to  any  ego - netow rk

in  fb . nodes ()

for

i

: fb . node [ str (i) ][ � eg ocol or �]

=  0

#  Fill  the  � eg ocol or �  field  with  a  dif f erent  color  number for  each  ego - netw ork  in

eg o_i ds :

i dCol or  =  1
for  id  in  eg o_i ds  :

G  =  nx . rea d_edg el i st (

os . path . join ( � files / ch08 / f a cebook �,

for  n

in  G. nodes ()
fb . node [ str (n) ][ � eg ocolor �] =  idColor i dCol or  +=  1

:

� {0}. edg es �. f ormat ( id )) , nodet ype  =  int )

col ors  =  [ x[ � eg ocolor �]  for  x  in  fb . node . values () ]

nsize  =  np . a rray ([ v  for  v  in  deg ree_cent _f b  . val ues () ])

nsiz e  = 5 00 *( nsiz e

- min ( nsiz e )) /( max ( ns iz e ) - min ( nsiz e )) nodes  =  nx .

dra w _net w orkx _n ode s  (

fb ,  pos  =  pos_fb ,
cmap  =  plt . g et _cmap ( � Pai red �) , no de_ col or   =  colors ,
node_si ze  =  nsize ,
w it h_l a bel s  =  False )

edges = nx . d ra w_ ne tw or kx _e dg es  ( fb ,  pos  =  pos _fb ,  alpha   =

.1)

However, the graph in Fig. 8.12 does not illustrate how much overlap is there
between the ego-networks. To do that, we can visualize the intersection between
ego-networks using a Venn or an Euler diagram. Both diagrams are useful in order to
see  how  networks  are  related.  Figure  8.13  shows  the  Venn  diagram  of  the
Facebook network.  This  powerful  and  complex  graph  cannot  be  easily  built  in
Python tool-

8    Network Analysis

162

Fig. 8.13  Venn diagram. The
area is weighted according
to the number of friends in
each ego-network and the
intersection between ego-
networks is related to the
number of common users

boxes like NetworkX or Matplotlib. In order to create it, we have used a JavaScript
visualization library called D3.JS.4

Community Detection

A  community  in  a  network  can  be  seen  as  a  set  of  nodes  of  the  network  that  is
densely  connected  internally.  The  detection  of  communities  in  a  network  is  a
difficult task since the number and sizes of communities are usually unknown [3].
Several  methods  for  community  detection  have  been  developed.  Here,  we
apply one  of  the  methods  to  automatically  extract  communities  from  the  Facebook
network.  We  import  the  Community  toolbox5  which  implements  the  Louvain
method  for  community  detection.  In  the  code  below,  we  compute  the  best
partition and plot the resulting communities in the whole Facebook network with
different colors, as we did in box In [17]. The resulting graph is shown in Fig. 8.14.

In [18]:

import  commu nity   pa rtiti on  =  c ommun ity  . b est_ parti tion  ( fb )

print  "#

communi ties   found :" ,  max ( partition . values () )  colo rs2  =
[ part ition  . get ( node )  for  node  in  fb . nodes () ]  nsize  =  np .

array ([ v

for  v  in  deg ree_ cent_ fb  . values () ])  nsize  =  500*( nsize
min ( nsize )) /( max ( nsize ) -  min ( nsize ))  nodes  =
nx . dr aw _ ne tw or kx _n od es  (

-

fb ,  pos  =  pos_fb ,
cmap  =  plt . get_cmap  ( � Paired �),
node_co lor   =  colors2 ,
node_si ze   =  nsize ,
with_la bels  =  False )

edges  =  nx . d ra w_ ne tw or kx _ ed ge s  ( fb ,  pos  =  pos_fb ,  alpha  =  .1)

.

Community Detection

Fig. 8.14 The Facebook
network drawn using the
Spring layout and different
colors to separate the
communities found

163

Out[18]: #  communities  found:  15

As can be seen, the 15 communities found automatically are similar to the 10 ego-
networks  loaded  from  the  dataset  (Fig.  8.12).  However,  some  of  the  10  ego-
networks are  subdivided into  several  communities  now.  This  discrepancy  can be
due  to  the  fact  that  the  ego-networks  are  manually  annotated  based  on  more
properties  of  the  nodes, whereas communities  are extracted based  only  on  the
graph information.


