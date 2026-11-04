#!/bin/sh
# the KL noise floor: the GGUF's own weights with the experts' rows summed on the GPU instead of the CPU
O=/home/ubuntu/work/nemotron/opt/r17
M=/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PK=/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs
PR=/home/ubuntu/Guild/data/expert-profile.bin
cd /home/ubuntu/Guild
: > $O/floor2.progress
for t in 1 2; do
  GUILD_HC=bf16 $O/qkm $M $PK $PR $O/text$t.txt $O/floor_hc$t.f16 $O/native$t.f16 > $O/floor_hc$t.out 2> $O/floor_hc$t.err
  echo "floor $t $(grep -o 'top-1.*max [0-9.]*' $O/floor_hc$t.out)" >> $O/floor2.progress
done
echo done >> $O/floor2.progress
