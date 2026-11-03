#!/bin/sh
# The all-models bench, second part: Drama 3 with the en0 voice from its WAV; Ornith-1.5-35B on the bench's prose and
# code prompts (tools/bench_runtime.py, its server settings: MTP --spec 4, and plain); Qwen with setup's config, plain
O=/home/ubuntu/work/nemotron/opt/all
G=/home/ubuntu/lanes/bin/gpu-run
step() { echo "$(date +%H:%M:%S) $*" >> $O/progress2; }
: > $O/progress2
TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
V=/mnt/models-ssd/Strata-data/drama3/voices
for seed in 11 12; do
  (cd /home/ubuntu/Guild && $G $O/gg --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $seed --wav $O/drama3_en0wav_$seed.wav --ref-wav $V/en0.wav --ref-text "$(cat $V/en0.txt)" --say "$TEXT" > $O/drama3_en0wav_$seed.log 2>&1); step drama3_en0wav_$seed
done
OR() { # label, extra bench args
  (cd /home/ubuntu/Guild && $G .venv/bin/python tools/bench_runtime.py --exe $O/gg --model /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/ornith-1.5-35b --mtp /mnt/models-ssd/Strata-data/mtp/ornith-rt --context 32768 --resident-budget-gib 24 --runs 2 --max-new 256 --out $O/b-$1.json "$@" > $O/$1.out 2>&1); }
OR ornith_mtp --spec 4; step ornith_mtp
shift 0
(cd /home/ubuntu/Guild && $G .venv/bin/python tools/bench_runtime.py --exe $O/gg --model /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/ornith-1.5-35b --mtp /mnt/models-ssd/Strata-data/mtp/ornith-rt --context 32768 --resident-budget-gib 24 --runs 2 --max-new 256 --out $O/b-ornith_plain.json --spec 4 --extra-args --decode-strategy normal > $O/ornith_plain.out 2>&1); step ornith_plain
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
QWEN_RUNS=2 QWEN_BENCH_EXTRA="--spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0 --decode-strategy normal" $O/qb.sh $O/gg qwen_plain > $O/qwen_plain.sum 2>&1; step qwen_plain
step done
