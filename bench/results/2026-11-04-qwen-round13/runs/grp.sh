#!/bin/sh
# one matrix kind at a time as Q4_K (with the imatrix), the rest the GGUF's Q8_0: each kind's share of q4_k's KL
O=/home/ubuntu/work/nemotron/opt/r17
M=/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PK=/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs
PR=/home/ubuntu/Guild/data/expert-profile.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
cd /home/ubuntu/Guild
: > $O/grp.progress
for g in attn_qkv attn_gate ssm_out attn_q attn_output; do for t in 1 2; do
  GUILD_HC=q8_0 GUILD_DENSE_IMATRIX=$IM GUILD_DENSE_MAP=$g=q4_k $O/qkm $M $PK $PR $O/text$t.txt $O/g_$g$t.f16 $O/native$t.f16 > $O/g_$g$t.out 2> $O/g_$g$t.err
  echo "$g $t $(grep -o 'top-1.*max [0-9.]*' $O/g_$g$t.out) $(grep -o '[0-9]* dense matrices re-quantized[^;]*' $O/g_$g$t.err)" >> $O/grp.progress
done; done
echo done >> $O/grp.progress
