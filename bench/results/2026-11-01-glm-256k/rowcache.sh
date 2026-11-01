#!/bin/sh
# GLM-5.3-Flash at a 262,144-cell context, the 32K prompt: decode with each token's rows copied whole against the
# row cache (GUILD_GLM_ROW_CACHE=0 / 1, one process from a snapshot, alternated); the ids must match the
# all-VRAM run's
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
cd "$(dirname "$0")"
GUILD_GLM_TIMING=1 GLM5_RUN_STEPS=1 $B $M @/home/ubuntu/lanes/g3/prompts/ctx32000.ids 64 --spec --ctx 262144 \
  --logits host256rc.f32 --ab GUILD_GLM_ROW_CACHE=0,1 --reps 3 > ab.out 2> ab.log
echo "ab rc $?" >> progress
