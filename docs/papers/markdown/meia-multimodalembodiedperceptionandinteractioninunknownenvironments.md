MEIA: Multimodal Embodied Perception and Interaction in Unknown
Environments

Yang Liu1 , Xinshuai Song1 , Kaixuan Jiang1 , Weixing Chen1 , Jingzhou Luo1 , Guanbin
Li1 and Liang Lin1
1Sun Yat-sen University
liuy856@mail.sysu.edu.cn, songxsh@mail2.sysu.edu.cn, jiangkx3@mail2.sysu.edu.cn,
chen867820261@gmail.com, luojzh5@mail2.sysu.edu.cn, liguanbin@mail.sysu.edu.cn,
linliang@ieee.org

4
2
0
2

l
u
J

9
2

]

V
C
.
s
c
[

3
v
0
9
2
0
0
.
2
0
4
2
:
v
i
X
r
a

Abstract

With the surge in the development of large lan-
guage models, embodied intelligence has attracted
increasing attention. Nevertheless, prior works on
embodied intelligence typically encode scene or
historical memory in an unimodal manner, either
visual or linguistic, which complicates the align-
ment of the model�s action planning with embod-
ied control. To overcome this limitation, we intro-
duce the Multimodal Embodied Interactive Agent
(MEIA), capable of translating high-level tasks ex-
pressed in natural language into a sequence of exe-
cutable actions. Specifically, we propose a novel
Multimodal Environment Memory (MEM) mod-
ule, facilitating the integration of embodied con-
trol with large models through the visual-language
memory of scenes. This capability enables MEIA
to generate executable action plans based on di-
verse requirements and the robot�s capabilities.
Furthermore, we construct an embodied question
answering dataset based on a dynamic virtual cafe
environment with the help of the large language
model.
In this virtual environment, we conduct
several experiments, utilizing multiple large mod-
els through zero-shot learning, and carefully design
scenarios for various situations. The experimental
results showcase the promising performance of our
MEIA in various embodied interactive tasks.

1 Introduction
Picture yourself entering a coffee shop staffed by robots.
Upon recognizing your arrival and that of your friends, a
robot approaches, greets you, and guides you to suitable
seats based on the group size. Subsequently, it proceeds to
the bar, takes your coffee and dessert orders, and then pre-
pares your order. Meanwhile, you request the robot to ad-
just the environment by lowering the curtains and decreas-
ing the air conditioning due to the rising outside temperature
and warmth inside the cafe. Additionally, it is able to re-
spond to questions you asked based on the environment. This
scenario illustrates the agent�s responsibilities in vision-and-
language navigation,
instruction decomposition and plan-
ning, as depicted in Fig.1. These tasks demand diverse ca-

pabilities, including visual recognition [Liu et al., 2018c;
Liu et al., 2023d; Yan et al., 2023], target navigation [Liu
et al., 2022b], language comprehension [Liu et al., 2023c;
Wei et al., 2023], commonsense reasoning [Liu et al., 2023b;
Chen et al., 2023], task planning [Zhu et al., 2022; Wang et
al., 2023a; Lin et al., 2023], and object manipulation [Tang
et al., 2023]. Termed embodied AI, it stands as a promi-
nent research focus in the realms of artificial intelligence and
robotics. Unlike conventional AI learning from datasets com-
prised mainly of internet-curated images [Liu et al., 2016;
Liu et al., 2019; Xu et al., 2022], videos [Liu et al., 2022a;
Liu et al., 2021; Liu et al., 2018a; Liu et al., 2018b], or text
[Liu et al., 2023e], embodied AI involves learning from self-
centered data, akin to human experiences [Duan et al., 2022].
Agents must collect vital visual information and engage with
the physical world based on real-time observations in embod-
ied AI tasks [Liu et al., 2024].

In recent years, large language models (LLMs) and vision
language models (VLMs) have received widespread atten-
tion. Leveraging massive training data on the web, LLMs
such as GPT-4 [OpenAI, 2023a], PaLM [Chowdhery et al.,
2023], and LLaMA [Touvron et al., 2023] have demonstrated
powerful capabilities in tasks like dialogue and reasoning.
However, the challenge arises when LLMs attempt simple
planning and reasoning in the physical world as the lack
of physical grounding. Some works [Driess et al., 2023;
Singh et al., 2023; Song et al., 2023] attempt to integrate vi-
sual information into LLMs, utilizing LLMs to execute em-
bodied tasks and resulting in good performance. Other works
[Hu et al., 2023] choose to employ VLMs for embodied tasks,
which can accomplish tasks by directly integrating obser-
vation images of the environment into reasoning and plan-
ning processes, showing impressive performance in zero-shot
scene understanding and reasoning. However, several issues
remain: 1) how to better store environmental information to
provide richer content for large models, and 2) whether LLMs
and VLMs can work collaboratively to solve embodied tasks.
To address the above challenges, we propose a novel em-
bodied model MEIA based on LLMs and VLMs, which ex-
cels in performing a variety of embodied AI tasks such as em-
bodied question answering and embodied control in a physi-
cal environment. Our MEIA comprises three essential mod-
ules that work collaboratively, including i) vision module uti-
lized to generate environmental language memory and envi-

Figure 1: We propose MEIA, a model that decomposes high-level language instructions into a series of executable actions.

ronmental image memory based on the surroundings, ii) con-
trol module used to dialogue with guests and perform em-
bodied operations such as seating guidance and cleaning up,
iii) large model responsible for reasoning and action plan-
ning. Since a good environmental memory storage method
aids agents in remembering the environment and making cor-
rect decisions to facilitate task completion, we propose a
novel Multi-modal Environmental Memory (MEM) module
incorporating object descriptions and floor plan of the en-
vironment. To further enhance the accuracy and feasibility
of model output, we meticulously designed prompts to guide
the LLMs and VLMs. In addition, we also build an embod-
ied question answering dataset with a thousand pieces of data
based on the simulator provided by Dataa Robotics, which
provides a cafe scene that is close to reality and achieves
abundant human-machine interaction design. We evaluate our
MEIA in the simulator and on the dataset we constructed. The
contributions can be summarized as follows:

� We introduce an embodied model MEIA that seamlessly
integrates large language models and vision language
models. Our MEIA can translate high-level tasks into
executable action sequences.

� We propose a Multi-modal Environment Memory
(MEM) module that equips the MEIA with more de-
tailed embodied knowledge, enhancing the understand-
ing of the physical environment and experience.

� We utilize a large language model to effectively generate
data with high quality, constructing an embodied ques-
tion answering dataset.

� Our MEIA exhibits robust zero-shot learning capabili-
ties and is capable of completing the embodied task with
good performance.

2 Related Work
2.1 Vision Language Model
The VLMs excel in simultaneous comprehension of both im-
ages and text, which can be used in tasks such as caption
generation, visual question answering and commonsense rea-
soning. Like UNITER [Chen et al., 2020] and BLIP [Li et
al., 2022], previous works mostly rely on neural networks
such as visual transformer (ViT) and recurrent neural network
(RNN), to construct vision language models. These models
utilize self-attention mechanisms to establish alignment be-
tween text and images. Inspired by the success of large lan-
guage models, there has been a growing interest in construct-
ing large-scale vision language models. Some works [Li et
al., 2023; Dai et al., 2023; Liu et al., 2023a] combine pre-
trained LLMs with visual reasoning, leveraging visual mod-
els. In this work, we select GPT-4-Vision as our vision lan-
guage model due to its superior performance.

2.2 Memory of Environment
The retention of environmental observations is crucial in em-
bodied tasks, as the absence of grounding in the physical
environment might lead to inaccurate responses and unfea-
sible planning for LLMs and VLMs. LLM-Planner [Song
et al., 2023], VELMA [Schumann et al., 2023], NavGPT
[Zhou et al., 2023] and LLM-Grounder [Yang et al., 2023]
take the textual descriptions of visual observations as input
or attach them to prompts. However, pure text environmen-
tal memory will lose a great deal of detail because of the
constraints of text description. Some works focus on stor-
ing environmental information through graphs constructed in
various forms, such as 2D top-down maps [Min et al., 2021;
Inoue and Ohashi, 2022], 3D voxel maps [Tan et al., 2023;
Murray and Cakmak, 2022], 3D maps [Hong et al., 2023b],

�Vision-and-Language NavigationI would like to order a cup of hot cocoa. I�m so sorry that we don�t sell hot cocoa.Then I would like a cup of coffee.One cup of coffee. I will prepare it for you as soon as possible.Go straight and turn right, there is a clear sign at the door of the restroom.Where is the restroom?Observations in EnvironmentEmbodied Question AnsweringInstruction Decomposition and PlanningVisual Understanding and DialogControl GuidanceWelcome. A table for how many sir?Two.Please follow me.Is there any vacuum cup in the cafe?Yes.etc. Inspired by these works, we introduce a novel memory
module:
the Multi-modal Environmental Memory (MEM),
including object IDs and coordinates as environmental lan-
guage memory and visual observation pictures as environ-
mental image memory.

2.3 Embodied AI Tasks
In embodied AI, agents learn through interactions with en-
vironments from an egocentric perception similar to humans
[Duan et al., 2022]. Embodied AI tasks mainly encompass
visual exploration and navigation, embodied planning, em-
bodied control, and embodied question answering. As the
most intricate task, embodied question answering [Das et al.,
2018] requires the agent to answer questions related to the
physical world based on information obtained from visual ex-
ploration and navigation. Additionally, the agent needs to in-
teract with the environment through embodied planning and
embodied control to obtain more information. Unlike existing
works [Driess et al., 2023; Singh et al., 2023; Hu et al., 2023;
Ding et al., 2023; Yaoxian et al., 2023] that utilize either
LLMs or VLMs to solve these tasks, we seamlessly integrate
both LLMs and VLMs. We leverage LLMs to understand cus-
tomers� needs, and decompose and plan tasks expressed in
language. Simultaneously, VLMs aid in achieving environ-
mental space understanding, location navigation, robot mo-
tion control, and embodied question answering.

3 Simulation Configuration
Our MEIA is performed in the simulator provided by Dataa
Robotics, which is built on the UE5 and provides a cafe
human-computer interaction scene with rich details. In this
section, we introduce the configuration of the scene and robot.

3.1 Scene Configuration
In terms of scene configuration, the simulator offers a cafe en-
vironment designed to closely emulate reality. Distinguishing
itself from other simulators like Habitat [Savva et al., 2019],
it uses high-resolution models as well as precise materials
and textures, ensuring an accurate representation of object
dynamics and mechanical behavior through a sophisticated
physics engine and making the appearance and physical char-
acteristics of objects in the simulator environment very close
to the real world. In addition, the simulator provides a robot
control system and object interaction to achieve a rich human-
machine interaction design.

The various objects in the simulator provide strong sup-
port for robot navigation and operation training. In the cafe
scene, there are a total of 13 categories of nearly 800 ob-
jects, including interactive objects such as coffee machines,
kettles, floor scrubbers, air conditioners, light switches, etc.
Additionally, 73 categories of objects can be added. The cafe
scene can be seen in Fig.1.

3.2 Robot Configuration
The robot in the simulation environment are designed based
on Dataa Robotics� Ginger humanoid robot intelligent agent.
It has a feature-rich environment interaction interface and
supports robot joint control, chassis speed control, and multi-
ple sensor simulations.

The visual capabilities of the robot include:

� Head, chest, and waist cameras capture RGB images,

� Head, chest, and waist cameras capture depth images,

� Head camera captures segmentation images.

The control capabilities of the robot include:

� Precise control of 21 joints in the robot�s torso and arms,

and 10 joints in its hands.

� Precise movement to specified coordinates.

� Joint movement control (IK control) to specified coordi-

nates.

� The ability to grasp, suction, and release cylindrical ob-

jects.

In addition, the robot can display the content of the con-
versation. It can also perform complex interactive operations
such as turning on/off lights, moving tables and chairs, push-
ing carts, using vacuum cleaners, using coffee machines, and
operating air conditioning control panels, as shown in Fig.1.

4 MEIA: An Agent for Multimodal Embodied

Interaction and Control

The structure of MEIA is shown in Fig.2. MEIA consists of
three modules: the vision module, the control module, and
the large model. The multimodal memory obtained by the
vision module is the core of MEIA. It serves as a bridge be-
tween the control module and the large model, enabling the
large model to generate highly executable action sequences
for diverse needs and hand them over to the control module
for execution. The details of MEIA is as follows.

4.1 Multimodal Environment Memory
The embodied control and the large model cannot be di-
rectly interacted due to the domain gap. The integration of
embodied control and the large model has always been the
focus of embodied AI. Previous works [Song et al., 2023;
Zhou et al., 2023] usually implement memory of scenes or
historical trajectories in a single-modal form of visual or nat-
ural language modality, making the plans given by the large
model difficult to execute by embodied agents. Recently,
some works attempt to delineate the scope of action plans
generated by large models to achieve the executability of the
plans [Wake et al., 2023a]. To address this issue, we propose
the multimodal environment memory (MEM) module (Fig.2),
which stores the names and coordinates of key items in the
scene recorded in natural language and the two-dimensional
plan of the environment as memory, serving as a guidance for
the large model to achieve highly executable action plans un-
der diverse needs. The details of the construction of MEM
are as follows.

Environmental Language Memory. The visual interface
of the robot can obtain different modalities of images, includ-
ing RGB images, depth images, and segmentation images.
Segmentation images set the pixels corresponding to differ-
ent key items in the scene to their ID values to achieve item
segmentation. We use the K-Means algorithm to calculate the

Figure 2: The structure diagram of MEIA. MEIA is implemented through three functional modules: the vision module, the
control module, and the large model. The multimodal environmental memory generated by the vision module will serve as a
bridge between the control module and the large model, enabling them to work collaboratively to complete tasks, and achieving
efficient integration of large model perception, memory, and embodied control.

Algorithm 1 Center positioning algorithm
Input: Pixel coordinate set list C, threshold ?
Output: Pixel coordinate center c
1: Initialize K-Means algorithm cls, set number of cluster

centers n clusters = 2.

system. Additionally, it has Euler angles ?g and a translation
vector Tg.

The rotation matrix Rg is obtained by calculating the Euler

angles:

2: Calculate cluster centers cluster = clf.f it(C)
3: Calculate the distance between cluster centers

(cid:113)(cid:80)2

d =

i=1(cluster0[i] ? cluster1[i])2.

c = cluster[0].

4: if d > ? then
5:
6: else
7:
8: end if

c = (cluster[0] + cluster[1])/2.

pixel coordinates of the key items. The algorithm is shown in
Algorithm 1.

After obtaining the pixel coordinates, we calculate the
camera coordinates. With camera intrinsic parameters de-
noted as fx, fy, cx, cy, and given object pixel coordinates
(i, j) along with the corresponding object depth at that pixel,
represented by depth, the calculation proceeds as follows:
(i ? cx)
fx
(j ? cy)
fy
zc = depth

?
xc =
????
????

? depth

? depth

yc =

(1)

Given the camera coordinates (xc, yc, zc), for the camera
extrinsic parameters, we have a rotation matrix R and a trans-
lation vector T :

(xg, yg, zg) = R ? [xc, yc, zc] + T

(2)
When obtaining the coordinates (xg, yg, zg) of the ginger
coordinate system, it should be noted that the ginger coordi-
nate system is opposite to the x-axis of the world coordinate

Rx =

Ry =

Rz =

0

(cid:35)

cos? ?sin?
cos?
sin?

0

(cid:34)1
0
0
(cid:34) cos?
0

0
1
?sin? 0
(cid:34)cos? ?sin?
cos?
sin?
0
0

(cid:35)

sin?
0
cos?

(cid:35)
0
0
1

Rg = Rz ? Ry ? Rx

Finally, the world coordinates are obtained:

(xw, yw, zw) = Rg ? [xg, ?yg, zg] + Tg

(3)

(4)

(5)

(6)

(7)

Environmental Image Memory. The environmental im-
age memory is an environmental plane map obtained by fur-
ther projecting the 3D point cloud image constructed by the
robot using the visual information gathered during the explo-
ration process.

The robot obtains environmental visual information by
calling the RGBD camera to capture RGB images and depth
images. For the RGB image and depth image correspond-
ing to the same viewing angle, with the help of Open3D,
an open-source library that supports rapid processing of 3D
data, a three-dimensional point cloud with color information
is constructed based on the internal parameter information
of the camera. However, directly superimposing 3D point
clouds obtained from multiple sets of RGBD images will

Vision ModuleSegmentation & Targeting�object�: {�Walnut�: 21, 'VacuumCup': 7, ...},�object_coordinates�: {'Walnut': (-208, 920), 'VacuumCup': (159, 1226),  ...}�object�: {...},  �object_co-ordinates�: {...}  environmental language memoryenvironmental image memory�conversation comprehension & instruction summarization�instruction decomposition & action planning�Image Spatial Understanding and Reasoning�Visual comprehension and dialogue�Visual Q&A�Embodied Q&AVLMLarge ModelLLMTask planningPromptsControl Modulemake coffeemop floorwipe tablenavigateQuestion: Is there any vacuum cup in the cafe?Answer: Yes.ExploreControl interface Answer questionsUpdate  Multimodal Environment Memorycontrol module and task requirements. We encapsulate the
robot�s capabilities and design flexible control interfaces:

� Move to target: specify the name of the item and move

to the item based on environmental memory.

� Produce and grab milk: generate a glass of milk on the
bar table, recognize and guide by the vision module to
grab the milk.

� Make coffee: operate the coffee machine to make coffee.

� Pour water: operate the kettle to pour water.

� Grab bread: grab bread from the cabinet.

� Control air conditioning: control the air conditioning ac-

cording to visual feedback, guided by VLM.

� Mop the floor: use a mop to mop the floor.

� Wipe the table: wipe the table with a towel.

� Control curtains: control the curtains to open or close.

� Control lighting: control the lighting to turn on or off.

� Straighten the chair: straighten the misplaced chair.

Our design for embodied control capability is versatile, al-
lowing the robot to possess a range of abilities. It is not re-
stricted to a single type of skill but is equipped to successfully
accomplish a variety of tasks in complex scenarios. Our em-
bodied control module not only provides �tools� for the large
model but also integrates multimodal environmental memory
to execute the action sequence planned by the large model to
complete the task.

4.3 Embodied Question Answering
Dataset. We construct a new dataset based on the simula-
tor. First, different scenarios are constructed using the GPT-
4 Turbo, with several reference examples provided. Under
the guidance of prompt, the GPT-4 Turbo randomly com-
bines the objext IDs and the location coordinates within the
allowed range to produce an output. According to the out-
put, the generation function is further called to generate cor-
responding items in the environment, achieving the effect of
random environment construction. Next, for different types
of questions, templates are designed to facilitate the large-
scale generation of questions. The templates are shown in
Table 1. Then, based on the randomly constructed environ-
ment, prompt is used to instruct the GPT-4 Turbo to generate
questions and corresponding answers, which are combined
with the scenes and question types as a piece of data in the
dataset. For each scenario, three question-answer pairs are
generated for each template. The dataset contains 70 gener-
ated scenes, with a total of 1050 pieces of data, all recorded
in a json file. Ultimately, we conduct manual inspection and
correction of the data, and ensure data balance by controlling
the distribution of answers in the dataset.

Embodied Question Answering. Questions, multimodal
environment memories, and newly explored information
serve as inputs to the multimodal large language model GPT-
4 Turbo with vision. Upon reaching a location, the robot ob-
serves the environment from four directions (front, rear, left,
and right) to gather new information. Acknowledging that the

Figure 3: Construction process of an environmental floor
plan.

cause all point cloud information to overlap and block each
other. Therefore, fusing the 3D point clouds obtained from
each set of images also requires calculations similar to the
world coordinates of the objects, applying rotation and trans-
lation operations based on the camera�s posture.

Additionally, outliers in the point cloud are identified and
removed based on statistical analysis. For each point in the
point cloud, we select the closest n points and calculate their
average distance from the point. Simultaneously, we calcu-
late the global average distance disg and global standard de-
viation stdg, which represent the average distance from each
point in the point cloud to any other point and the standard
deviation of these distances, respectively. If a point�s average
distance is greater than disg + stdr ? stdg, it is considered
an outlier, where stdr specifies the multiple of the standard
deviation.

Although the 3D point cloud contains richer and more ac-
curate information than 2D images, processing and analyz-
ing three-dimensional images directly can be challenging for
large models. As a result, we project the 3D point cloud onto
the plane where the floor is located. The environmental plane
map can to some extent display the exploration status of the
current environment, providing an overall understanding of
the cafe�s environment and determining whether the environ-
ment has been fully explored. As the exploration progresses,
the 3D point cloud is enriched and the environmental plane
map contains more information, as shown in Fig.3.

By combining object IDs and coordinates with environ-
mental plane map, we obtain multimodal environment mem-
ory. The MEM is designed as the global memory so that the
robot can update multimodal memory when it calls the visual
interface, to achieve flexible response for complex environ-
ment.

4.2 Embodied Control

Embodied control is the fundamental ability of robots to com-
plete tasks, and its design is determined jointly by the robot

(a)(b)(c)(d)type

location

comparing

existence

template
What is the item on the same table as the < obj >?
Are the < obj1 > and the < obj2 > on the same table?
Is the < obj1 > closer to the < obj2 > than to the < obj3 >?
Is there any < obj > in the cafe?
Is there anything in the cafe that I can use to < do something >?

Table 1: Question Templates

plans generated by the model may not always be feasible, pre-
viously unsuccessful plans are also fed back into the model to
prevent the repetition of similar invalid plans. Prompted by
the designed queries, GPT-4 Turbo with vision first assesses
whether the existing information is adequate for providing an
accurate answer to the question. In cases of insufficiency, the
model devises action plans to direct the robot in moving and
interacting to acquire new observations. Then, it generates
output based on the new input, thereby creating a closed loop.
If the existing information is sufficient, the model synthesizes
the available information to answer the question, concluding
the closed loop.

4.4 Large Model

Large model is the command center of MEIA. As shown in
Fig.2, we comprehensively utilize LLMs and VLMs, combin-
ing different large models for various tasks to fully leverage
the strengths of each model.

LLM. For LLM, we use GPT-3.5 Turbo and GPT-4
Turbo. GPT-3.5 Turbo is used to summarize and generate
natural language instructions for the conversation between
the robot and the guest, as this task is relatively easy. On
the other hand, GPT-4 Turbo is employed for the intricate
task of decomposing and planning instructions, demanding
robust comprehension, planning, and reasoning abilities from
the LLM.

VLM. GPT-4 Turbo with vision has superior capability in
both visual and language understanding compared to open-
source models. In our experiments, it can form spatial un-
derstanding of complex images and offer independent judg-
ments, thereby generating content of high quality. Therefore,
we use GPT-4 Turbo with vision to implement guest seating
guidance task (visual language navigation), spatial-aware dia-
logue interaction task (understanding the space inside the cafe
and completing dialogue requirements), and control guidance
task (determining which action to execute). Furthermore, we
also use the model for autonomous decision-making to solve
embodied question answering problems, as mentioned in Sec-
tion 4.3.

Inspired by the spirit of Chain-of-Thought (CoT) [Yao et
al., 2022] and In-Context Learning (ICL) [Dong et al., 2022],
we design specific prompts to make different large models
provide accurate and appropriate solutions for different sit-
uations. The detailed prompt design can be seen in the ap-
pendix. By assigning different sub-tasks to different large
models, the large models can decompose and plan various
tasks based on the control module�s capabilities and multi-
modal environmental memory, enabling the robot to interact
with guests like humans and meet their requirements.

Model

miniGPT4 [Zhu et al., 2023]
CogVLM [Wang et al., 2023b]
CogAgent [Hong et al., 2023a]
GPT4-V [OpenAI, 2023b]

Visual
question
answering
44%
52%
46%
98%

Spatial
understanding
and reasoning
8%
38%
32%
94%

Table 2: Sub-task Evaluation.

5 Experimental Results
We design a plot pipeline for our solution based on reality,
systematically evaluating and comparing various models for
different tasks. Additionally, we also test the performance
of the model on the embodied question answering dataset we
construct. The details are outlined below.

5.1 Experimental Settings
According to the real interaction between the waiter and the
customer in a cafe, we design a closed-loop plot pipeline.

In the closed-loop plot pipeline, the robot�s tasks are di-

vided into three parts:

Environmental Exploration: The robot explores the en-
vironment and forms an initial multimodal environmental
memory.

Human-Robot Interaction: The robot chooses an appro-
priate location to wait for guests, interacts with the guests,
answers their questions, and understands their needs.

Task Understanding and Execution: The robot compre-
hends the guests� requirements, forms an action plan accord-
ingly, and executes tasks to fulfill the guests� needs. During
this period, the robot agent may encounter abnormal prompts
(such as garbage on the ground), and it needs to solve the
abnormal situation. We illustrate three tasks in Fig.4.

Once the customer�s needs are all fulfilled, the agent transi-
tions back to the Human-Robot Interaction task again. In ad-
dition, we use GPT-3.5 to randomly sample from executable
subtasks and generate colloquial instructions to simulate au-
thentic human-machine interaction scenarios, serving as eval-
uation cases to assess the performance of different models.

5.2 Results and Comparison
We add state-of-the-art models and methods [Zhu et al., 2023;
Wang et al., 2023b; Huang et al., 2022; Wake et al., 2023b]
to our experimental comparisons. The outputs of large mod-
els are obtained through zero-shot learning via prompt engi-
neering. Specific experimental details can be found in the
appendix we submitted.

Metrics. We use Executable Success Rate (ESR) to judge

the performance of the models, which formulated as:

ESR =

ne
N

(8)

Where the ne is the number of successful executions and N
is the total number of tasks. When the output or plan given by
the model can be successfully executed and complete the task
in the simulator, it is counted as a success. Besides, we use
Success Rate Weighted by Step Length (SSL) to measure

Figure 4: Task execution. We present three tasks, each of which has a brief description of the implementation process.

the accuracy and efficiency of the plans given by different
models and methods:

SSL =

N
(cid:88)

si ?

1
N

lc
max(lg, lp)

(9)

where N is the total number of tasks, si is the success rate
of planning execution, lg is the length of the grounding plan-
ning, lp is the length of the planning generated by large mod-
els, and lc is the length of the correct step in the planning.

Sub-task Evaluation
The design of sub-task evaluation is based on the abilities re-
quired by robot agent in experimental plots, including visual
question answering, and spatial understanding and reasoning.
We use the latest large visual language models, miniGPT4
[Zhu et al., 2023], CogVLM [Wang et al., 2023b] and Co-
gAgent [Hong et al., 2023a] for the sub-task evaluation. For
each task, we evaluate each model 50 times.

Visual question answering. For the visual question an-
swering scenario, we select the scene that the robot needs to
identify the air conditioner controller and select the correct
control button according to the instructions.

Spatial understanding and reasoning. We select visual
navigation tasks to assess the spatial understanding and rea-
soning ability. The robot needs to select the appropriate lo-
cation in the image based on the scene and customer require-
ments. The evaluation results are shown in Table 2.

MiniGPT-4 has strong language capabilities. It can effec-
tively understand the content of the prompt and provide out-
puts that are basically in line with the format. Nevertheless,

its visual capability is limited, posing challenges in compre-
hending low-resolution and intricate images, thereby hinder-
ing effective task completion. CogVLM�s visual ability is
relatively prominent. While it can effectively identify various
targets on complex images and have an appropriate under-
standing of them, its grasp of prompts is insufficient, leading
to difficulties in generating the required responses. CogA-
gent, built on the CogVLM design, shows no significant dif-
ference in accuracy compared to CogVLM in our evaluations.
However, CogAgent�s answers are quite stronger in terms of
executability than CogVLM. The performance of GPT4-V
is significantly better than these open-source models, and it
maintains extremely high performance in various tasks. This
will also be the basis for our subsequent experiments.

Instruction Planning Evaluation
To compare MEIA with existing methods, we design an in-
struction planning evaluation. We use GPT3.5 to sample
events and tasks in the cafe scenario and generate natural lan-
guage instructions that include a certain number of sub-tasks,
which are used as evaluation cases. To compare the effect of
the number of sub-tasks in instructions on the model planning
performance, we divide the generated instructions into two
types: long instructions containing 3 to 5 sub-tasks and short
instructions containing 2 to 3 sub-tasks. We generate 20 sam-
ples for each task and conduct experiments in the simulator to
evaluate the models� ESR and SSL metrics. The instruction
generation prompt and some of the evaluation examples and
results can be found in the appendix.

We evaluate the recent models Language-Planner [Huang

Here's your yogurt.The chair is out of order.It's a little hot in the cafe.What can I do for you?I would like to order a bottle of yogurt.Complete Orders: The agent obtains the customers� orders by conducting a conversation with them. Here we take yogurt as an example. The agent first navigates to where the yogurt is placed, and then grabs the yogurt. Next, the agent brings the yogurt to the customers.Adjust Temperature: After receiving instructions for a lower temperature in the cafe, the agent navigates to the location of the air conditioning controller. The agent then looks down to identify the controller and presses the correct button.Straighten Chairs: Recognizing that the chair is arranged in a cluttered manner, the agent approaches to the incorrectly placed chair and straightens it.Model

Language-Planner [Huang et al., 2022]
Robot-Prompt [Wake et al., 2023b]
MEIA w/o MEM
MEIA w/o Large Model*
MEIA

Short instructions
ESR(b)
71.67%
78.33%
70.00%
79.16%
97.50%

SSL
33.92%
71.80%
64.12%
77.17%
97.00%

ESR(a)
40%
55%
50%
70%
95%

Task

Long instructions
ESR(b)
64.67%
80.25%
56.33%
84.58%
94.17%

SSL
43.61%
73.59%
47.13%
78.51%
92.82%

ESR(a)
25%
40%
15%
55%
85%

Table 3: The result for instruction planning evaluation. ESR(a) is the ESR for instruction and ESR(b) is the ESR for sub-tasks
in instruction. For MEIA w/o Large Model, We replaced GPT4 with GPT3.5.

performance is mainly reflected in the completion rate of in-
structions, and has little effect on the completion rate of sub-
tasks and the accuracy of planning. Even with the increase
of instructions, the model�s planning tends to be more stable,
which leads to a slight increase in success rate. As shown in
Fig.5, the differences between ESR of instructions and SSL
reflect the precision of model planning. Due to Language-
Planner generating plans based on the similarity of examples
and actions, its planning often results in confusion and redun-
dancy, leading to particularly low SSL. Robot-Prompt lacks
logical understanding of actions, causing it to occasionally
lose key actions, resulting in a decrease in SSL.

Full Pipeline Evaluation
We evaluate the full plot pipeline. Since each evaluation con-
tains more than ten sub-tasks, existing models or methods
alone cannot complete the pipeline we design. Therefore,
we conduct experiments with MEIA using large models such
as GPT3.5, GPT4, and GPT4-V. The process of task execu-
tion can be seen in Fig.6. In multi-round experiments(for 20
rounds), MEIA�s ESR reaches 95%. This means that MEIA
has extremely high accuracy and executability in planning.

We also study cases of failure. In one such case, the model
makes a mistake in responding to �the table is dirty� by ne-
glecting the initial step of �taking a towel� and directly going
to the table, which reflects the large model�s insufficient un-
derstanding of the relation between embodied control ability
and real-world scenarios.

Ablation Study
We conduct an ablation study on Multimodal Environment
Memory(MEM) and Large Model of MEIA to validate their
effectiveness. We conduct experiments using the same set-
tings as in the instruction planning evaluation, and the results
can be seen in Table 3 and Fig.5.

We find that when we change the large model from GPT4
to GPT3.5, the performance of the model consistently de-
creases, reflecting the importance of the large model on in-
struction planning. When we eliminate the multimodal envi-
ronmental memory, the model�s performance exhibits a more
pronounced decline. Particularly concerning the ESR of in-
structions, it shows a greater regression compared to down-
grading the large model. At the same time, the gap between
the ESR of subtask and SSL has also widened significantly,
reflecting a decrease in the precision of planning. We at-
tribute it to the lack of multimodal environmental memory.
The absence of multimodal environmental memory compels
the large model to place more emphasis on the content of in-
structions. However, the precision of the instruction descrip-

Figure 5: The result for instruction planning evaluation. -s
for short instructions and -l for long instructions.

et al., 2022] and Robot-Prompt [Wake et al., 2023b].
Language-Planner uses LLM for planning, adding the most
relevant example from the example set along with the cor-
responding actions and the task to be completed as prompt,
while Robot-Prompt uses the method of CoT to form high-
performance action planning through multi-round dialogue
and user feedback. We make appropriate adjustments to the
Language-Planner and Robot-Prompt based on the simulator
environment, robot capabilities, and task requirements. The
hyperparameter CUTOFF for Language-Planner is set to 0.6,
and the number of examples is set to 10. We format the 3D en-
vironment of the cafe according to the requirements of Robot-
Prompt. For more experimental results on hyperparameters
and text format descriptions of the cafe environment, please
refer to the appendix. The evaluation is conducted using sam-
ples generated by GPT3.5, as shown in Table 3.

As shown in Table 3, MEIA�s performance far exceeds that
of Language-Planner and Robot-Prompt. We attribute it to
our multimodal environment memory. Multimodal environ-
ment memory enables the model to have a more logical un-
derstanding of the environment, thereby ensuring the order
and coherence of planning. This is reflected in the fact that
the difference between MEIA�s ESR of instructions and that
of other models is much greater than the difference in their
ESRs of sub-tasks. The impact of instruction length on model

In (b) and (c), the agent waits for the customers and guides them to suitable seats.

Figure 6: Full Pipeline Evaluation.
In (a) the agent explores the environment and constructs multimodal environmental
In (d) the agent conducts a
memory.
conversation with customers and summarizes the conversation content to extract their needs. In (e) and (f) the agent plans and
executes a sequence of actions to complete the order. In (g) and (h), the agent fulfills other tasks according to the instructions
like cleaning the floors and wiping the table.

tion cannot support the robot to successfully execute tasks,
which affects the task execution success rate of the robot.

5.3 Embodied Question Answering Evaluation
In addition, we also evaluate the model on the embodied ques-
tion answering task.

We randomly select 6 scenes from the dataset, with a to-
tal of 90 questions, and conduct experiments on the model in
the form of single-round question answering and multi-round
question answering. In the single-round question answering
task, the robot starts with an empty environment memory
when receiving a question and explores from scratch each
time. In the multi-round question answering task, the envi-
ronmental information explored to answer previous questions
is available for generating answers to the current question.
Each set of question-answer pairs for each scene is consid-
ered as one testing round.

The experimental results are shown in the Table 4. Metrics
used to evaluate the performance of the model include av-
erage accuracy, average exploration count, average unreach-
able planning count, and average path length. The aver-
age accuracy(ACC) measures the proportion of correct an-
swers among multiple questions. The average exploration
count(EC) counts the number of times the robot moves to
the next exploration position planned by the model, excluding
the initial exploration conducted by the robot at the starting
position. The average unreachable planning count(UPC)
tallies the number of positions in the environment that the
model plans for the robot but are inaccessible. The average
path length(PL) measures the total distance in centimeters
that the robot navigates in the environment from receiving
the question to generating the answer.

According to Table 4, it can be observed that in the multi-
round question answering task, the model achieves better re-
sults in terms of average exploration count, average unreach-
able planning count, and average path length while maintain-
ing comparable or even improved average accuracy compared
to the single-round question answering task. Particularly, the

single-round
multi-round

EC UPC
ACC
3.5
70%
22
1
72% 19.3

PL
4769.9
3800.7

Table 4: Comparison of results between single-round Q&A
and multiple-rounds Q&A.

average unreachable planning count is reduced by more than
70%. This result suggests the effectiveness of environment
memory and historical collision information in the model�s
planning process. By leveraging the environment memory
and historical collision information obtained through explo-
ration, the model can better understand the overall environ-
ment, plan action, and expedite the exploration of the required
environmental information for answering questions.

Ablation Study
In order to further verify the effectiveness of the memory
module, we present the results of the ablation experiment on
the embodied question answering task. We test four mod-
els respectively: MEIA, MEIA without multimodal envi-
ronment memory(MEM), MEIA without environmental lan-
guage memory(ELM), and MEIA without environmental im-
age memory(EIM). For each model, we calculate the average
accuracy for different types of problems. The result is shown
in Table 5.

location
MEIA w/o MEM 38.9%
MEIA w/o ELM 41.7%
50.0%
MEIA w/o EIM
55.6%
MEIA

comparing
50.0%
50.0%
55.6%
66.7%

total
existence
56.7%
77.8%
57.8%
77.8%
80.6%
63.3%
86.1% 70.0%

Table 5: Comparison of average accuracy for different types
of questions in the ablation experiment.

Obviously, MEIA performs the best among the four mod-
els. The absence of either environmental language mem-

Please follow me.I would like to order a dessert and a bottle of yogurt.Okay. I will prepare it for you as soon as possible.Go straight and turn right, there is a clear sign at the door of the restroom.Where is the restroom?Here's your yogurt.(a)(b)(c)(d)(e)(f)(g)(h)ory or environmental image memory has an impact on the
model�s performance.
Furthermore, adding environmen-
tal image memory without environmental language memory
does not significantly improve the model�s performance.

It can also be observed from the result that the models
perform the worst on location-type questions, followed by
comparing-type questions. To some extent, it reflects the lim-
ited spatial reasoning ability of MEIA, and there is room for
improvement in its overall performance.

6 Conclusion
In this work, we propose the MEIA for accomplishing em-
bodied tasks and build an embodied question answering
dataset with a thousand pieces of data. In the physical en-
vironment, the agent gathers egocentric environmental infor-
mation, executes embodied control and completes embod-
ied tasks. We innovatively introduce multi-modal memory to
store global memory and update it in real time, thereby inte-
grating embodied control with large models. Experiments in-
dicate that MEIA demonstrates the promising ability to solve
various embodied tasks.

References
[Chen et al., 2020] Yen-Chun Chen, Linjie Li, Licheng Yu,
Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng,
and Jingjing Liu. Uniter: Universal image-text representa-
tion learning. In European conference on computer vision,
pages 104�120. Springer, 2020.

[Chen et al., 2023] Weixing Chen, Yang Liu, Ce Wang,
Guanbin Li, Jiarui Zhu, and Liang Lin. Visual-linguistic
causal intervention for radiology report generation. arXiv
preprint arXiv:2303.09117, 2023.

[Chowdhery et al., 2023] Aakanksha Chowdhery, Sharan
Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra,
Adam Roberts, Paul Barham, Hyung Won Chung, Charles
Sutton, Sebastian Gehrmann, et al. Palm: Scaling lan-
Journal of Machine
guage modeling with pathways.
Learning Research, 24(240):1�113, 2023.

[Dai et al., 2023] Wenliang Dai, Junnan Li, Dongxu Li, An-
thony Meng Huat Tiong, Junqi Zhao, Weisheng Wang,
Instructblip:
Boyang Li, Pascale Fung, and Steven Hoi.
Towards general-purpose vision-language models with in-
struction tuning, 2023.

[Das et al., 2018] Abhishek Das, Samyak Datta, Georgia
Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. Em-
In Proceedings of the IEEE
bodied question answering.
conference on computer vision and pattern recognition,
pages 1�10, 2018.

[Ding et al., 2023] Yan Ding, Xiaohan Zhang, Chris Paxton,
and Shiqi Zhang. Leveraging commonsense knowledge
from large language models for task and motion planning.
In RSS 2023 Workshop on Learning for Task and Motion
Planning, 2023.

[Dong et al., 2022] Qingxiu Dong, Lei Li, Damai Dai,
Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing

Xu, Lei Li, and Zhifang Sui. A survey for in-context learn-
ing. arXiv:2301.00234, Dec 2022.

[Driess et al., 2023] Danny Driess, Fei Xia, Mehdi SM Saj-
jadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter,
Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe
Yu, et al. Palm-e: An embodied multimodal language
model. arXiv preprint arXiv:2303.03378, 2023.

[Duan et al., 2022] Jiafei Duan, Samson Yu, Hui Li Tan,
Hongyuan Zhu, and Cheston Tan. A survey of embod-
ied ai: From simulators to research tasks. IEEE Transac-
tions on Emerging Topics in Computational Intelligence,
6(2):230�244, 2022.

[Hong et al., 2023a] Wenyi Hong, Weihan Wang, Qingsong
Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang,
Zihan Wang, Yuxiao Dong, Ming Ding, and Jie Tang.
language model for gui agents.
Cogagent: A visual
arXiv:2312.08914, Dec 2023.

[Hong et al., 2023b] Yining Hong, Haoyu Zhen, Peihao
Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and
Chuang Gan. 3d-llm: Injecting the 3d world into large
language models. arXiv preprint arXiv:2307.12981, 2023.
[Hu et al., 2023] Yingdong Hu, Fanqi Lin, Tong Zhang,
Li Yi, and Yang Gao. Look before you leap: Unveiling
the power of gpt-4v in robotic vision-language planning.
arXiv preprint arXiv:2311.17842, 2023.

[Huang et al., 2022] Wenlong Huang,

Pieter Abbeel,
Deepak Pathak, and Igor Mordatch. Language models as
zero-shot planners: Extracting actionable knowledge for
embodied agents. arXiv:2201.07207, 2022.

[Inoue and Ohashi, 2022] Yuki Inoue and Hiroki Ohashi.
Prompter: Utilizing large language model prompting for
arXiv
a data efficient embodied instruction following.
preprint arXiv:2211.03267, 2022.

[Li et al., 2022] Junnan Li, Dongxu Li, Caiming Xiong, and
Steven Hoi. Blip: Bootstrapping language-image pre-
training for unified vision-language understanding and
In International Conference on Machine
generation.
Learning, pages 12888�12900. PMLR, 2022.

[Li et al., 2023] Junnan Li, Dongxu Li, Silvio Savarese, and
Steven Hoi. Blip-2: Bootstrapping language-image pre-
training with frozen image encoders and large language
models. arXiv preprint arXiv:2301.12597, 2023.

[Lin et al., 2023] Junfan Lin, Yuying Zhu, Lingbo Liu, Yang
Liu, Guanbin Li, and Liang Lin. Denselight: Efficient
control for large-scale traffic signals with dense feedback.
In Edith Elkind, editor, Proceedings of the Thirty-Second
International Joint Conference on Artificial Intelligence,
IJCAI-23, pages 6058�6066. International Joint Confer-
ences on Artificial Intelligence Organization, 8 2023. AI
for Good.

[Liu et al., 2016] Yang Liu, Jing Li, ZhaoYang Lu, Tao
Yang, and ZiJian Liu. Combining multiple features for
cross-domain face sketch recognition. In Biometric Recog-
nition: 11th Chinese Conference, CCBR 2016, Chengdu,
China, October 14-16, 2016, Proceedings 11, pages 139�
146. Springer, 2016.

[Liu et al., 2018a] Yang Liu, Zhaoyang Lu, Jing Li, and Tao
Yang. Hierarchically learned view-invariant representa-
IEEE Trans-
tions for cross-view action recognition.
actions on Circuits and Systems for Video Technology,
29(8):2416�2430, 2018.

[Liu et al., 2018b] Yang Liu, Zhaoyang Lu, Jing Li, Tao
Yang, and Chao Yao. Global temporal representation
IEEE Signal
based cnns for infrared action recognition.
Processing Letters, 25(6):848�852, 2018.

[Liu et al., 2018c] Yang Liu, Zhaoyang Lu, Jing Li, Chao
Yao, and Yanzi Deng. Transferable feature representation
for visible-to-infrared cross-dataset human action recogni-
tion. Complexity, 2018:1�20, 2018.

[Liu et al., 2019] Yang Liu, Zhaoyang Lu, Jing Li, Tao Yang,
and Chao Yao. Deep image-to-video adaptation and fusion
IEEE Transactions on
networks for action recognition.
Image Processing, 29:3168�3182, 2019.

[Liu et al., 2021] Yang Liu, Keze Wang, Guanbin Li, and
Liang Lin. Semantics-aware adaptive knowledge distilla-
tion for sensor-to-vision action recognition. IEEE Trans-
actions on Image Processing, 30:5573�5588, 2021.

[Liu et al., 2022a] Yang Liu, Keze Wang, Lingbo Liu,
Haoyuan Lan, and Liang Lin. Tcgl: Temporal contrastive
graph for self-supervised video representation learning.
IEEE Transactions on Image Processing, 31:1978�1993,
2022.

[Liu et al., 2022b] Yang Liu, Yu-Shen Wei, Hong Yan,
Guan-Bin Li, and Liang Lin. Causal reasoning meets vi-
sual representation learning: A prospective study. Ma-
chine Intelligence Research, 19(6):485�511, 2022.

[Liu et al., 2023a] Haotian Liu, Chunyuan Li, Qingyang Wu,
arXiv

and Yong Jae Lee. Visual instruction tuning.
preprint arXiv:2304.08485, 2023.

[Liu et al., 2023b] Yang Liu, Weixing Chen, Guanbin Li,
and Liang Lin. Causalvlr: A toolbox and benchmark
arXiv preprint
for visual-linguistic causal reasoning.
arXiv:2306.17462, 2023.

[Liu et al., 2023c] Yang Liu, Guanbin Li, and Liang Lin.
Cross-modal causal relational reasoning for event-level vi-
IEEE Transactions on Pattern
sual question answering.
Analysis and Machine Intelligence, 2023.

[Liu et al., 2023d] Yang Liu, Ying Tan, and Haoyuan Lan.
Self-supervised contrastive learning for audio-visual ac-
tion recognition. In 2023 IEEE International Conference
on Image Processing (ICIP), pages 1000�1004. IEEE,
2023.

[Liu et al., 2023e] Yang Liu, Ying Tan, Jingzhou Luo, and
Weixing Chen. Vcd: Visual causality discovery for cross-
In Chinese Conference on
modal question reasoning.
Pattern Recognition and Computer Vision (PRCV), pages
309�322. Springer, 2023.

[Liu et al., 2024] Yang Liu, Weixing Chen, Yongjie Bai,
Guanbin Li, Wen Gao, and Liang Lin. Aligning cyber
space with physical world: A comprehensive survey on
embodied ai. arXiv preprint arXiv:2407.06886, 2024.

[Min et al., 2021] So Yeon Min, Devendra Singh Chaplot,
Pradeep Ravikumar, Yonatan Bisk, and Ruslan Salakhut-
dinov. Film: Following instructions in language with mod-
ular methods. arXiv preprint arXiv:2110.07342, 2021.
[Murray and Cakmak, 2022] Michael Murray and Maya
Cakmak.
Following natural language instructions for
household tasks with landmark guided search and rein-
forced pose adjustment. IEEE Robotics and Automation
Letters, 7(3):6870�6877, 2022.

[OpenAI, 2023a] OpenAI. Gpt-4 technical report, 2023.
[OpenAI, 2023b] OpenAI. Gpt-4v(ision) system card, 2023.
[Savva et al., 2019] Manolis Savva, Abhishek Kadian, Olek-
sandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain,
Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik,
Devi Parikh, and Dhruv Batra. Habitat: A platform for
In 2019 IEEE/CVF International
embodied ai research.
Conference on Computer Vision (ICCV), Oct 2019.

[Schumann et al., 2023] Raphael Schumann, Wanrong Zhu,
Weixi Feng, Tsu-Jui Fu, Stefan Riezler, and William Yang
Wang. Velma: Verbalization embodiment of llm agents
for vision and language navigation in street view. arXiv
preprint arXiv:2307.06082, 2023.

[Singh et al., 2023] Ishika Singh, Valts Blukis, Arsalan
Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay,
Dieter Fox, Jesse Thomason, and Animesh Garg. Prog-
prompt: Generating situated robot task plans using large
language models. In 2023 IEEE International Conference
on Robotics and Automation (ICRA), pages 11523�11530.
IEEE, 2023.

[Song et al., 2023] Chan Hee Song, Jiaman Wu, Clayton
Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su.
Llm-planner: Few-shot grounded planning for embodied
agents with large language models. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 2998�3009, 2023.

[Tan et al., 2023] Sinan Tan, Mengmeng Ge, Di Guo, Huap-
ing Liu, and Fuchun Sun. Knowledge-based embodied
question answering. IEEE Transactions on Pattern Analy-
sis and Machine Intelligence, 2023.

[Tang et al., 2023] Ziyi Tang, Ruilin Wang, Weixing Chen,
Keze Wang, Yang Liu, Tianshui Chen, and Liang Lin.
Towards causalgpt: A multi-agent approach for faithful
knowledge reasoning via promoting causal consistency in
llms. arXiv preprint arXiv:2308.11914, 2023.

[Touvron et al., 2023] Hugo Touvron, Thibaut Lavril, Gau-
tier Izacard, Xavier Martinet, Marie-Anne Lachaux, Tim-
oth�ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Ham-
bro, Faisal Azhar, et al. Llama: Open and efficient founda-
tion language models. arXiv preprint arXiv:2302.13971,
2023.

[Wake et al., 2023a] Naoki Wake,

Atsushi Kanehira,
Kazuhiro Sasabuchi,
and Katsushi
Ikeuchi. Chatgpt empowered long-step robot control in
various environments: A case application. arXiv preprint
arXiv:2304.03893v6, Apr 2023.

Jun Takamatsu,

vision-language understanding with advanced large lan-
guage models. arXiv preprint arXiv:2304.10592, 2023.

[Wake et al., 2023b] Naoki Wake, Atsushi Kanehira,
Kazuhiro Sasabuchi,
and Katsushi
Ikeuchi. Chatgpt empowered long-step robot control in
various environments: A case application. IEEE Access,
11:95060�95078, 2023.

Jun Takamatsu,

[Wang et al., 2023a] Kuo Wang, LingBo Liu, Yang Liu,
GuanBin Li, Fan Zhou, and Liang Lin. Urban regional
Information Sci-
function guided traffic flow prediction.
ences, 634:308�320, 2023.

[Wang et al., 2023b] Weihan Wang, Qingsong Lv, Wenmeng
Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi
Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual
arXiv preprint
expert for pretrained language models.
arXiv:2311.03079, 2023.

[Wei et al., 2023] Yushen Wei, Yang Liu, Hong Yan, Guan-
bin Li, and Liang Lin. Visual causal scene refinement
for video question answering. MM �23, page 377�386,
New York, NY, USA, 2023. Association for Computing
Machinery.

[Xu et al., 2022] Xiaoqian Xu, Pengxu Wei, Weikai Chen,
Yang Liu, Mingzhi Mao, Liang Lin, and Guanbin Li. Dual
adversarial adaptation for cross-device real-world image
super-resolution. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
5667�5676, 2022.

[Yan et al., 2023] Hong Yan, Yang Liu, Yushen Wei, Zhen
Li, Guanbin Li, and Liang Lin. Skeletonmae: graph-based
masked autoencoder for skeleton sequence pre-training. In
Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 5606�5618, 2023.

[Yang et al., 2023] Jianing Yang, Xuweiyi Chen, Shengyi
Qian, Nikhil Madaan, Madhavan Iyengar, David F Fouhey,
and Joyce Chai. Llm-grounder: Open-vocabulary 3d vi-
sual grounding with large language model as an agent.
arXiv preprint arXiv:2309.12311, 2023.

[Yao et al., 2022] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan
Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language mod-
els. arXiv:2210.03629, Oct 2022.

[Yaoxian et al., 2023] Song Yaoxian, Sun Penglei, Liu
Haoyu, Li Zhixu, Song Wei, Xiao Yanghua, and
Scene-driven multimodal knowledge
Zhou Xiaofang.
arXiv preprint
graph construction for embodied ai.
arXiv:2311.03783, 2023.

[Zhou et al., 2023] Gengze Zhou, Yicong Hong, and Qi Wu.
Navgpt: Explicit reasoning in vision-and-language nav-
arXiv preprint
igation with large language models.
arXiv:2305.16986, 2023.

[Zhu et al., 2022] Yuying Zhu, Yang Zhang, Lingbo Liu,
Yang Liu, Guanbin Li, Mingzhi Mao, and Liang Lin.
Hybrid-order representation learning for electricity theft
IEEE Transactions on Industrial Informatics,
detection.
19(2):1248�1259, 2022.

[Zhu et al., 2023] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xi-
ang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing


