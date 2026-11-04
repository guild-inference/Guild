#!/bin/sh
# klfin.sh's cuBLAS and MMQ arms again (pv?.f16 was written); each output removed once compared
O=/home/ubuntu/work/nemotron/opt/r17
M=/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PK=/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs
PR=/home/ubuntu/Guild/data/expert-profile.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
cd /home/ubuntu/Guild
: > $O/klfin.progress
for t in 1 2; do
  E="GUILD_HC=q8_0 GUILD_DENSE=q4_k GUILD_DENSE_MAP=ssm_out=q5_k,attn_output=q8_0 GUILD_DENSE_IMATRIX=$IM QWEN_KL_PREFIX=1024 QWEN_KL_PREFILL=1"
  env $E GUILD_PREFILL_DENSE_MMQ=0 $O/qkfin $M $PK $PR $O/text$t.txt $O/pc$t.f16 $O/pv$t.f16 > $O/pc$t.out 2> $O/pc$t.err
  env $E GUILD_PREFILL_DENSE_MMQ=1 $O/qkfin $M $PK $PR $O/text$t.txt $O/pm$t.f16 $O/pv$t.f16 > $O/pm$t.out 2> $O/pm$t.err
  rm -f $O/pc$t.f16 $O/pm$t.f16
  echo "text $t cublas: $(grep -o 'top-1.*max [0-9.]*' $O/pc$t.out)" >> $O/klfin.progress
  echo "text $t mmq:    $(grep -o 'top-1.*max [0-9.]*' $O/pm$t.out)" >> $O/klfin.progress
done
echo done >> $O/klfin.progress
