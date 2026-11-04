#!/bin/sh
# GLM A/B: HEAD / the NL4 AVX-512 rows two rows ahead + the vec_dot prefetch / the prefetch alone, 3 rounds
W=/home/ubuntu/work/nemotron
GR=/home/ubuntu/lanes/int3/tools/gr.sh
O=$W/opt/glmab2
mkdir -p $O
for i in 1 2 3; do
  $GR $W/opt/glm5_run.head $O/head$i.out > /dev/null 2>&1
  $GR $W/build/glm5_run $O/new$i.out > /dev/null 2>&1
  $GR $W/build/glm5_run $O/pf$i.out -- GUILD_NL4_512=0 > /dev/null 2>&1
done
