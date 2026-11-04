#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r16
M=/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PK=/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs
PR=/home/ubuntu/Guild/data/expert-profile.bin
: > $O/kl2.progress
for t in 1 2; do
  cd /home/ubuntu/Guild
  GUILD_DENSE=q5_k GUILD_HC=q8_0 QWEN_KL_PREFIX=1024 QWEN_KL_PREFILL=1 GUILD_PREFILL_DENSE_MMQ=1 $O/qkh $M $PK $PR /tmp/claude-1000/-home-ubuntu-Guild/fc412f01-5572-5649-9611-4ca7c5573b54/scratchpad/kl/text$t.txt $O/kD$t.f16 $O/kA$t.f16 > $O/kD$t.out 2>$O/kD$t.err
  echo "text $t" >> $O/kl2.progress; cat $O/kD$t.out >> $O/kl2.progress
done
echo done >> $O/kl2.progress
