#!/bin/sh
# GLM-5.3-Flash's prompt with the IQ4_XS down rows on AVX-512 BW or AVX-2 (GUILD_IQ4XS_512), one process, alternated
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
cd "$(dirname "$0")"
GUILD_GLM_TIMING=1 $B $M 0 16 --ctx 32768 --prompts prompts.txt --ab GUILD_IQ4XS_512=0,1 --reps 3 > ab.out 2> ab.log
echo "rc $?" >> progress
