#!/bin/sh
# KL baseline: the GGUF's own dense matrices (no requantization), --hc q8_0 (the GGUF's own); then q6_k / q5_k / q4_k
# requantized without an importance matrix, each against it.  Decode's arithmetic (qwen_kl: the verifier, 4 rows a window)
O=/home/ubuntu/work/nemotron/opt/r17
M=/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PK=/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs
PR=/home/ubuntu/Guild/data/expert-profile.bin
cd /home/ubuntu/Guild
: > $O/base.progress
for t in 1 2; do
  GUILD_HC=q8_0 $O/qk $M $PK $PR $O/text$t.txt $O/native$t.f16 > $O/native$t.out 2> $O/native$t.err
  echo "native $t $(cat $O/native$t.out)" >> $O/base.progress
  for q in q6_k q5_k q4_k; do
    GUILD_HC=q8_0 GUILD_DENSE=$q $O/qk $M $PK $PR $O/text$t.txt $O/${q}_$t.f16 $O/native$t.f16 > $O/${q}_$t.out 2> $O/${q}_$t.err
    echo "$q $t $(tail -1 $O/${q}_$t.out)" >> $O/base.progress
  done
done
echo done >> $O/base.progress
