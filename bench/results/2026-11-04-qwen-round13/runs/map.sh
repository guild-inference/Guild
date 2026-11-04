#!/bin/sh
# mixed maps (with the imatrix): the attention output kept, the least sensitive kinds lower
O=/home/ubuntu/work/nemotron/opt/r17
M=/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PK=/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs
PR=/home/ubuntu/Guild/data/expert-profile.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
cd /home/ubuntu/Guild
: > $O/map.progress
run() { # label dense map
  for t in 1 2; do
    GUILD_HC=q8_0 GUILD_DENSE_IMATRIX=$IM GUILD_DENSE=$2 GUILD_DENSE_MAP=$3 $O/qkm $M $PK $PR $O/text$t.txt $O/m_$1$t.f16 $O/native$t.f16 > $O/m_$1$t.out 2> $O/m_$1$t.err
    echo "$1 $t $(grep -o 'top-1.*max [0-9.]*' $O/m_$1$t.out)" >> $O/map.progress
  done
}
run A q5_k attn_output=q8_0
run B q5_k attn_output=q6_k
run C q5_k attn_output=q8_0,attn_gate=q4_k,attn_q=q4_k
run D q6_k attn_output=q8_0
run E q4_k attn_output=q8_0
echo done >> $O/map.progress
