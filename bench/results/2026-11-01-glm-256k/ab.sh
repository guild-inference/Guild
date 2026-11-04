#!/bin/sh
# GLM-5.3-Flash at a 262,144-token context, the 32K prompt: the prompt with the union gather (against 678.6 s with
# each query's rows), then decode with the old and the new attention kernel from the same snapshot, alternated
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
cd "$(dirname "$0")"
GLM5_RUN_STEPS=1 $B $M @/home/ubuntu/lanes/g3/prompts/ctx32000.ids 64 --spec --ctx 262144 --logits host256u.f32 \
  --ab GUILD_GLM_ATTEND_LEGACY=1,0 --reps 3 > ab.out 2> ab.log
echo "ab rc $?" >> progress
