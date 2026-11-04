#!/bin/sh
# GLM-5.3-Flash, the 5-token prompt, 256 tokens --spec: --ctx 32768 against --ctx 262144 (the same placement, so the
# same ids and speed?), each with the old and the new attention kernel alternated; then the 2,051-token prompt's
# phases (GUILD_GLM_TIMING)
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
D="$(dirname "$0")"
cd $D
for c in 32768 262144; do
  echo "$(date -u +%H:%M:%S) start c$c" >> progress
  GUILD_GLM_TIMING=1 $B $M 785,6722,315,9621,374 256 --spec --ctx $c --ab GUILD_GLM_ATTEND_LEGACY=1,0 --reps 3 \
    > c$c.out 2> c$c.log
  echo "$(date -u +%H:%M:%S) c$c rc $?" >> progress
done
GUILD_GLM_TIMING=1 $B $M @/home/ubuntu/lanes/g3/prompts/ctx2051.ids 2 --ctx 32768 > pf.out 2> pf.log
echo done >> $D/progress
