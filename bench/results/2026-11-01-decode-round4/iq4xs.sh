#!/bin/sh
# GLM-5.3-Flash: the IQ4_XS down rows through ggml-cpu's dot (decode) and the AVX-2 rows (prompt) against two rows to an
# AVX-512BW register (GUILD_IQ4XS_512=0 / 1), each in one process alternated
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
cd "$(dirname "$0")"
GUILD_GLM_TIMING=1 $B $M 785,6722,315,9621,374 256 --spec --ctx 32768 --ab GUILD_IQ4XS_512=0,1 --reps 4 > dec.out 2> dec.log
echo "dec rc $?" >> progress
GUILD_GLM_TIMING=1 $B $M 0 16 --ctx 32768 --prompts ../glmiq4/prompts.txt --ab GUILD_IQ4XS_512=0,1 --reps 3 > pf.out 2> pf.log
echo "pf rc $?" >> progress
