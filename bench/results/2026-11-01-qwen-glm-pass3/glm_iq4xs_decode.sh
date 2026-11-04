#!/bin/sh
# GLM-5.3-Flash decode (5-token prompt, 256 tokens, --spec): the IQ4_XS down rows through ggml-cpu's AVX-2 dot
# (GUILD_IQ4XS_512=0) or the AVX-512 rows (both ggml's dot to the bit), one process, alternated
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
cd "$(dirname "$0")"
GUILD_GLM_TIMING=1 $B $M 785,6722,315,9621,374 256 --spec --ctx 32768 --ab GUILD_IQ4XS_512=0,1 --reps 4 > ab.out 2> ab.log
echo "rc $?" >> progress
