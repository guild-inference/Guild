#!/bin/sh
# GLM-5.3-Flash decode: down items of 16 (the default) against 32 and 64 rows a group (GUILD_MOE_DRB), alternated
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
cd "$(dirname "$0")"
GUILD_GLM_TIMING=1 $B $M 785,6722,315,9621,374 256 --spec --ctx 32768 --ab GUILD_MOE_DRB=64,16 --reps 6 > drb64b.out 2> drb64b.log
echo "rc $?" >> progress
