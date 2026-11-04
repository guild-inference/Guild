#!/bin/sh
# the prompt path's numbers: text 1 and 2, the first 1024 tokens through the verifier (A), the cuBLAS prompt path (B)
# or the dense-MMQ prompt path (C), then every later position's prediction under teacher forcing
O=/home/ubuntu/work/nemotron/opt/r16
M=/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PK=/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs
PR=/home/ubuntu/Guild/data/expert-profile.bin
: > $O/kl.progress
for t in 1 2; do
  T=/tmp/claude-1000/-home-ubuntu-Guild/fc412f01-5572-5649-9611-4ca7c5573b54/scratchpad/kl/text$t.txt
  cd /home/ubuntu/Guild
  GUILD_DENSE=q5_k GUILD_HC=q8_0 QWEN_KL_PREFIX=1024 QWEN_KL_PREFILL=0 $O/qk $M $PK $PR $T $O/kA$t.f16 > $O/kA$t.out 2>$O/kA$t.err
  GUILD_DENSE=q5_k GUILD_HC=q8_0 QWEN_KL_PREFIX=1024 QWEN_KL_PREFILL=1 GUILD_PREFILL_DENSE_MMQ=0 $O/qk $M $PK $PR $T $O/kB$t.f16 $O/kA$t.f16 > $O/kB$t.out 2>$O/kB$t.err
  GUILD_DENSE=q5_k GUILD_HC=q8_0 QWEN_KL_PREFIX=1024 QWEN_KL_PREFILL=1 GUILD_PREFILL_DENSE_MMQ=1 $O/qk $M $PK $PR $T $O/kC$t.f16 $O/kA$t.f16 > $O/kC$t.out 2>$O/kC$t.err
  echo "text $t" >> $O/kl.progress; cat $O/kA$t.out $O/kB$t.out $O/kC$t.out >> $O/kl.progress
done
echo done >> $O/kl.progress
