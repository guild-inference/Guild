#!/bin/sh
# GLM-5.3-Flash decode (5-token prompt, 256 tokens, --spec), one process: the q_b rows' 2-byte loads and the KDA state
# written twice (GLM_Q6K_W4=0 GUILD_GLM_KDA_MID1=0) against 16-byte windows and one write, alternated four times
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
cd "$(dirname "$0")"
GUILD_GLM_TIMING=1 $B $M 785,6722,315,9621,374 256 --spec --ctx 32768 --ab GLM_Q6K_W4=0,1 --ab GUILD_GLM_KDA_MID1=0,1 \
  --reps 4 > ab.out 2> ab.log
echo "rc $?" >> progress
