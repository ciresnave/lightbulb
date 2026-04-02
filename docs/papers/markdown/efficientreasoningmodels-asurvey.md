5
2
0
2

r
p
A
5
1

]
L
C
.
s
c
[

1
v
3
0
9
0
1
.
4
0
5
2
:
v
i
X
r
a

Efficient Reasoning Models: A Survey

Sicheng Feng
National University of Singapore, Singapore
Nankai University, Tianjin, China

Gongfan Fang
National University of Singapore, Singapore

Xinyin Ma
National University of Singapore, Singapore

Xinchao Wang∗
National University of Singapore, Singapore

sicheng@mail.nankai.edu.cn

gongfan@u.nus.edu

maxinyin@u.nus.edu

xinchao@nus.edu.sg

Abstract

Reasoning models have demonstrated remarkable progress in solving complex and logic-
intensive tasks by generating extended Chain-of-Thoughts (CoTs) prior to arriving at a
final answer. Yet, the emergence of this "slow-thinking" paradigm, with numerous tokens
generated in sequence, inevitably introduces substantial computational overhead. To this
end, it highlights an urgent need for effective acceleration. This survey aims to provide
a comprehensive overview of recent advances in efficient reasoning. It categorizes existing
works into three key directions: (1) shorter – compressing lengthy CoTs into concise yet
effective reasoning chains; (2) smaller – developing compact language models with strong
reasoning capabilities through techniques such as knowledge distillation, other model com-
pression techniques, and reinforcement learning; and (3) faster – designing efficient decoding
strategies to accelerate inference. A curated collection of papers discussed in this survey is
available in our GitHub repository1.

1 Introduction

Recent reasoning-oriented models, or Large Reasoning Models (LRMs) (Guo et al., 2025; Jaech et al., 2024),
have achieved remarkable performance on complex reasoning tasks by generating long Chain-of-Thoughts
(CoTs), enabling effective problem-solving in domains such as mathematics and coding (Sprague et al., 2024).
However, while LRMs significantly improve performance on reasoning tasks, they also cause substantial
overhead. Compared to standard LLMs, reasoning models lead to redundancy across multiple dimensions.

A salient characteristic of reasoning models is their tendency to overthink by generating excessively long
reasoning chains (Chen et al., 2024c; Sui et al., 2025a), which has naturally motivated efforts to improve
efficiency by shortening reasoning paths. Meanwhile, recent studies (Wu et al., 2025c; Yang et al., 2025c;
Jin et al., 2024b) challenge the assumption that longer CoTs always lead to better performance, showing
even negative returns. To address this kind of CoT length redundancy, a range of methods have been
proposed: reinforcement learning (RL) with length penalties (Luo et al., 2025a; Aggarwal & Welleck, 2025),
supervised fine-tuning (SFT) on variable-length CoT data (Ma et al., 2025; Xia et al., 2025), and prompt-
driven strategies that either guide reasoning paths or route inputs to more efficient solutions (Ding et al., 2024;
Aytes et al., 2025). Furthermore, latent reasoning performs the process in latent space without generating
explicit CoTs, making reasoning chains more concise (Hao et al., 2024; Su et al., 2025).

∗Corresponding author
1https://github.com/fscdc/Awesome-Efficient-Reasoning-Models. We will keep it updated with new research.

1

[Content continues with full survey paper text - truncated for length]

30