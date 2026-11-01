#!/bin/sh
# GLM-5.3-Flash at a 262,144-cell context, a 127,997-token prompt, then 64 tokens --spec from a snapshot with the old
# and the new attention kernel (two rounds): the decode step at 128K and where it goes
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
cd "$(dirname "$0")"
GUILD_GLM_TIMING=1 GLM5_RUN_STEPS=1 $B $M @ctx128000.ids 64 --spec --ctx 262144 --ab GUILD_GLM_ATTEND_LEGACY=1,0 --reps 2 \
  > ab.out 2> ab.log
echo "rc $?" >> progress
