#!/bin/sh
# check2.sh [outdir]: check.sh with a step list.  STEPS (default all) picks among
# qwen ornith drama3 tower nemotron startlux slots btlong prompts ctest; I (default int/build) the build.
# Every step takes the machine lock itself.  GLM is glm.sh, run last (a GLM load evicts the page cache).
I=${I:-/home/ubuntu/lanes/r20/int/build}; O=${1:-/home/ubuntu/lanes/r20/int/chk2}; L=/home/ubuntu/lanes/bin
PY=/home/ubuntu/Guild/.venv/bin/python; T=/home/ubuntu/work/par/guild/tools
PV=/home/ubuntu/lanes/r20/pf-tail-ring/verify; DT=/mnt/models-ssd/Strata-data/drama3/tower; DR=/home/ubuntu/lanes/r20/d3-tower-fuse/ref.npz.d
STEPS=${STEPS:-"qwen ornith drama3 tower nemotron startlux slots btlong prompts ctest"}
mkdir -p $O
st() { echo "$1 $(date -u +%T)" >> $O/progress; }
want() { case " $STEPS " in *" $1 "*) st $1; return 0 ;; esac; return 1; }
if want qwen; then
$L/qwen-bench-r18 $I/guild-generate $O q1 > $O/q1.summary 2>&1
fi
if want ornith; then
$L/ornith-bench-r18 $I/guild-generate $O o1 > $O/o1.summary 2>&1
fi
if want drama3; then
$L/drama3-run-r18 $I/guild-generate $O d11 11 > $O/d11.summary 2>&1
$L/drama3-run-r18 $I/guild-generate $O d12 12 > $O/d12.summary 2>&1
for f in 0 ks ksawf; do
  $L/drama3-run-r18 $I/guild-generate $O d11-$f 11 GUILD_DRAMA3_FUSE=$f > $O/d11-$f.summary 2>&1
done
fi
if want tower; then
$L/gpu-run sh -c "for f in unset 0 ks ksawf; do if [ \$f = unset ]; then env -u GUILD_DRAMA3_FUSE $I/drama3_tower_test $DT $DR; else GUILD_DRAMA3_FUSE=\$f $I/drama3_tower_test $DT $DR; fi > $O/tower_\$f.txt 2>&1; echo rc \$? >> $O/tower_\$f.txt; done"
fi
if want nemotron; then
$L/gpu-run $I/transcribe_run --model /home/ubuntu/models/nemotron-3.5-transcribe-guild/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio /home/ubuntu/models/nemotron-3.5-transcribe-guild/audio --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json --mtp 3 --repeat 6 --clips /home/ubuntu/work/nemotron/perf/clips.jsonl --out $O/nem.jsonl > $O/nem.log 2>&1
fi
if want startlux; then
$L/gpu-run $I/decision_run --native /home/ubuntu/models/startlux-decision-35b-a3b-gguf/StartLux-Decision-35B-A3B-Q8_0.gguf --pack /home/ubuntu/models/startlux-decision-35b-a3b-gguf/pack-Q8_0 --resident-budget-gib 40 --requests /home/ubuntu/work/startlux/perf/perf-short.jsonl --out $O/sl.jsonl --repeat 5 --fresh > $O/sl.log 2>&1
fi
if want slots; then
BATCH_TEST_LOG=$O/b6.log $L/gpu-run $PY $T/batch_test.py --exe $I/guild-generate --config /home/ubuntu/work/par/cfg/qwen32k.json --batch 7 --n 7 --max-new 96 > $O/b6.out 2>&1
BATCH_TEST_LOG=$O/b3.log $L/gpu-run $PY $T/batch_test.py --exe $I/guild-generate --config /home/ubuntu/work/par/cfg/qwen32k.json --batch 3 --n 3 --max-new 96 > $O/b3.out 2>&1
fi
if want btlong; then
(cd /home/ubuntu/Guild && GUILD_NO_KV_CACHE=1 BATCH_TEST_LOG=$O/btl.log $L/gpu-run $PY $PV/bt_long.py --exe $I/guild-generate --config /home/ubuntu/work/par/cfg/qwen32k.json --batch 3 --n 3 --max-new 96 > $O/btl.out 2>&1)
fi
if want prompts; then
QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $PV/pT.json --timeout 3000 --max-new 32 --warmup 1" $L/gpu-run $PV/qb.sh $I/guild-generate $O/pr pT GUILD_NO_KV_CACHE=1 > $O/pT.summary 2>&1
QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts /home/ubuntu/lanes/r20/scout/prompt/pD.json --timeout 3000 --max-new 32 --warmup 1" $L/gpu-run $PV/qb.sh $I/guild-generate $O/pr pD GUILD_NO_KV_CACHE=1 > $O/pD.summary 2>&1
QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts /home/ubuntu/lanes/r18/scout/qwen-prompt/pB.json --timeout 3000 --max-new 32 --warmup 1" $L/gpu-run $PV/qb.sh $I/guild-generate $O/pr pB GUILD_NO_KV_CACHE=1 > $O/pB.summary 2>&1
QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $PV/pM.json --timeout 3000 --max-new 64 --warmup 1" $L/gpu-run $PV/qb_mtp.sh $I/guild-generate $O/pr pM GUILD_NO_KV_CACHE=1 > $O/pM.summary 2>&1
fi
if want ctest; then
$L/gpu-run sh -c "cd $I && ctest --timeout 1200 --output-on-failure > $O/ctest.log 2>&1; echo ctest rc \$? >> $O/ctest.log"
fi
st done
