#!/bin/sh
# arms.sh <kind> <exe-dir> <outdir> <label> [VAR=value ...]: one A/B arm WITHOUT taking the machine lock (the caller holds
# it for the whole hold).  kinds: qdec (Qwen decode bench, prose+code 2x256), odec (Ornith), d11/d12 (Drama 3 seed),
# nem (Nemotron clips), sl (StartLux), b3/b7 (batch slots), qpr (Qwen prompts pT.json, no KV cache).
K="$1"; X="$2"; O="$3"; LB="$4"; shift 4
mkdir -p "$O"
PY=/home/ubuntu/Guild/.venv/bin/python; BR=/home/ubuntu/work/nemotron/guild/tools/bench_runtime.py
PV=/home/ubuntu/lanes/r20/pf-tail-ring/verify
cd /home/ubuntu/Guild
echo "== $K $LB $(date -u +%T) $X $*" >> "$O/arms.log"
case "$K" in
qdec|qpr)
  EX=""; RUNS=2; MN=256
  [ "$K" = qpr ] && { EX="--prompts $PV/pT.json --timeout 3000 --max-new 32 --warmup 1"; RUNS=1; set -- GUILD_NO_KV_CACHE=1 "$@"; }
  env "$@" $PY $BR --exe "$X/guild-generate" \
   --model /mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
   --pack /mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs --profile data/expert-profile.bin \
   --mtp /mnt/models-ssd/Strata-data/mtp/rt --context 32768 --resident-budget-gib 56 \
   --runs $RUNS --max-new $MN --out "$O/b-$LB.json" --prefill auto --spec 8 --spec-min-p 0.5 $EX \
   --extra-args --dflash /home/ubuntu/work/dflash/drafter --dflash-vocab /home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin \
   --dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix /home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf \
   --hc q8_0 --prefill-mmq dense > "$O/$LB.out" 2>&1 ;;
odec)
  env "$@" $PY $BR --exe "$X/guild-generate" \
   --model /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/ornith-1.5-35b \
   --mtp /mnt/models-ssd/Strata-data/mtp/ornith-rt --context 32768 --resident-budget-gib 24 --runs 2 --max-new 256 --spec 4 \
   --out "$O/b-$LB.json" > "$O/$LB.out" 2>&1 ;;
d11|d12)
  SEED=${K#d}
  TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
  env -u GUILD_DRAMA3_FUSE "$@" "$X/guild-generate" --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $SEED --wav "$O/$LB.wav" --say "$TEXT" > "$O/$LB.log" 2>&1 ;;
nem)
  env "$@" "$X/transcribe_run" --model /home/ubuntu/models/nemotron-3.5-transcribe-guild/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio /home/ubuntu/models/nemotron-3.5-transcribe-guild/audio --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json --mtp 3 --repeat 6 --clips /home/ubuntu/work/nemotron/perf/clips.jsonl --out "$O/$LB.jsonl" > "$O/$LB.log" 2>&1 ;;
sl)
  env "$@" "$X/decision_run" --native /home/ubuntu/models/startlux-decision-35b-a3b-gguf/StartLux-Decision-35B-A3B-Q8_0.gguf --pack /home/ubuntu/models/startlux-decision-35b-a3b-gguf/pack-Q8_0 --resident-budget-gib 40 --requests /home/ubuntu/work/startlux/perf/perf-short.jsonl --out "$O/$LB.jsonl" --repeat 5 --fresh > "$O/$LB.log" 2>&1 ;;
b3|b7)
  N=${K#b}
  env "$@" BATCH_TEST_LOG="$O/$LB.log" $PY /home/ubuntu/work/par/guild/tools/batch_test.py --exe "$X/guild-generate" --config /home/ubuntu/work/par/cfg/qwen32k.json --batch $N --n $N --max-new 96 > "$O/$LB.out" 2>&1 ;;
*) echo "unknown kind $K"; exit 2 ;;
esac
echo "rc $? $(date -u +%T)" >> "$O/arms.log"
