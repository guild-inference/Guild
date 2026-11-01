#!/bin/sh
# the final kernels: glm5_kernels_test, then GLM-5.3-Flash at 262,144 cells with the 32K prompt, the row cache on,
# decode with the old and the new attention kernel alternated (and the ids against the all-VRAM run's)
D="$(dirname "$0")"
cd "$D"
/home/ubuntu/work/nemotron/build-cu/glm5_kernels_test > test.out 2>&1; echo "test rc $?" >> progress
/home/ubuntu/work/nemotron/build-cu/glm5_kernels_test attend > attend.out 2>&1
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
GUILD_GLM_TIMING=1 GLM5_RUN_STEPS=1 $B $M @/home/ubuntu/lanes/g3/prompts/ctx32000.ids 64 --spec --ctx 262144 \
  --logits final.f32 --ab GUILD_GLM_ATTEND_LEGACY=1,0 --reps 3 > ab.out 2> ab.log
echo "ab rc $?" >> progress
