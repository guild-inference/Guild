#!/bin/sh
# mixed maps (with the imatrix): the attention output kept, the least sensitive kinds lower
O=/home/ubuntu/work/nemotron/opt/r17
M=/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PK=/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs
PR=/home/ubuntu/Guild/data/expert-profile.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
cd /home/ubuntu/Guild
: > $O/map2.progress
run() { # label dense map
  for t in 1 2; do
    GUILD_HC=q8_0 GUILD_DENSE_IMATRIX=$IM GUILD_DENSE=$2 GUILD_DENSE_MAP=$3 $O/qkm $M $PK $PR $O/text$t.txt $O/m_$1$t.f16 $O/native$t.f16 > $O/m_$1$t.out 2> $O/m_$1$t.err
    echo "$1 $t $(grep -o 'top-1.*max [0-9.]*' $O/m_$1$t.out)" >> $O/map2.progress
  done
}





run F q5_k attn_output=q8_0,attn_gate=q4_k,attn_q=q4_k,ssm_out=q6_k
run H q5_k attn_output=q8_0,attn_gate=q4_k,attn_q=q4_k,attn_qkv=q4_k
run I q5_k attn_output=q6_k,attn_gate=q4_k,attn_q=q4_k
echo done >> $O/map2.progress
