#!/bin/sh
# qb.sh <exe> <label> [VAR=value ...]: one Qwen decode bench (server settings, prose + code, QWEN_RUNS runs x 256),
# stdout to <label>.out, the engine's stderr whole in b-<label>.log.  Extra bench_runtime args in QWEN_BENCH_EXTRA.
EXE="$1"; LABEL="$2"; shift 2
OUT=/home/ubuntu/work/nemotron/opt/r14
cd /home/ubuntu/Guild
env "$@" /home/ubuntu/lanes/bin/gpu-run .venv/bin/python /home/ubuntu/work/nemotron/guild/tools/bench_runtime.py --exe "$EXE" \
 --model /mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
 --pack /mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs --profile data/expert-profile.bin \
 --mtp ${QWEN_MTP:-/mnt/models-ssd/Strata-data/mtp/rt} --context ${QWEN_CONTEXT:-32768} --resident-budget-gib 56 \
 --runs ${QWEN_RUNS:-3} --max-new 256 --out "$OUT/b-$LABEL.json" $QWEN_BENCH_EXTRA > "$OUT/$LABEL.out" 2>&1
grep -E "ms/window" "$OUT/b-$LABEL.log" | sed -E 's/.*windows, ([0-9.]+) tokens\/window, ([0-9.]+) ms\/window.*CPU dispatch ([0-9.]+).*draft ([0-9.]+).*/tok\/win \1  ms\/win \2  cpu \3  draft \4/'
grep SUMMARY "$OUT/$LABEL.out"
