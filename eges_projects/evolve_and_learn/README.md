# Generational replacement vs Elitism and Learning in Evolutionary Robotics

**Abstract**: Evolutionary Robotics offers the possibility to design
robots to solve a specific task automatically by optimizing their
morphology and control together. However, this co-optimization
of body and control is challenging, because controllers need some
time to adapt to the evolving morphology - which may make
it difficult for new and promising designs to enter the evolving
population. A solution to this is to add intra-life learning, defined
as an additional controller optimization loop, to each individual in
the evolving population. A related problem is the lack of diversity
often seen in evolving populations as evolution narrows down the
search to a few promising designs too quickly. Using an explicit
diversity-selecting approach can be used to preserve diversity, but
one has to define a diversity measure in that case. Alternatively,
this problem can be mitigated by implementing full generational
replacement, where all robots are replaced by a new population
in each generation. This solution for increasing diversity usually
comes at the cost of lower performance compared to using
elitism. In this work, we show that combining such generational
replacement with intra-life learning is a way to increase diversity,
while retaining performance. We also highlight the importance of
performance metrics when studying learning in morphologically
evolving robots, showing that evaluating according to function
evaluations versus according to generations of evolution can give
different conclusions.

[Link to playlist](https://www.youtube.com/playlist?list=PL_uDzdHnUKPoOueT6LCqIe208O5ZBM-CJ) of best performing robots. [Link to data](https://zenodo.org/records/14501099).

See core folder of this repository for installation instructions.
