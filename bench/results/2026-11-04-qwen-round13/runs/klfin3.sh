#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r17
M=/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PK=/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs
PR=/home/ubuntu/Guild/data/expert-profile.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
cd /home/ubuntu/Guild
E="GUILD_HC=q8_0 GUILD_DENSE=q4_k GUILD_DENSE_MAP=ssm_out=q5_k,attn_output=q8_0 GUILD_DENSE_IMATRIX=$IM QWEN_KL_PREFIX=1024"
env $E QWEN_KL_PREFILL=0 $O/qkfin $M $PK $PR $O/text2.txt $O/pv2.f16 > $O/pv2.out 2> $O/pv2.err
env $E QWEN_KL_PREFILL=1 GUILD_PREFILL_DENSE_MMQ=0 $O/qkfin $M $PK $PR $O/text2.txt $O/pc2.f16 $O/pv2.f16 > $O/pc2.out 2> $O/pc2.err
env $E QWEN_KL_PREFILL=1 GUILD_PREFILL_DENSE_MMQ=1 $O/qkfin $M $PK $PR $O/text2.txt $O/pm2.f16 $O/pv2.f16 > $O/pm2.out 2> $O/pm2.err
rm -f $O/pc2.f16 $O/pm2.f16
echo "text 2 cublas: $(grep -o 'top-1.*max [0-9.]*' $O/pc2.out)" > $O/klfin3.progress
echo "text 2 mmq:    $(grep -o 'top-1.*max [0-9.]*' $O/pm2.out)" >> $O/klfin3.progress
