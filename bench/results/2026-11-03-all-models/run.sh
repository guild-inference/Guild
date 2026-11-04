#!/bin/sh
# Every model on this PC, the branch's head (38ac0b6) with its pinned kernels, each measured as before (one GPU job at
# a time through gpu-run / glm-run).
O=/home/ubuntu/work/nemotron/opt/all
G=/home/ubuntu/lanes/bin/gpu-run
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
step() { echo "$(date +%H:%M:%S) $*" >> $O/progress; }
: > $O/progress

# 1. Qwen3.8-Flash-Next UD-IQ4_XS with setup's run config (DFlash: Q4_K weights, the multilingual head, --spec 8;
#    --dense q5_k --hc q8_0): six prompts, 2 runs x 256 tokens; a ~4K-token prompt (no prefix cache), 2 runs
Q="--spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0"
QWEN_RUNS=2 QWEN_BENCH_EXTRA="$Q" $O/qb.sh $O/gg qwen_short > $O/qwen_short.sum 2>&1; step qwen_short
QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prompts $O/more.json $Q" $O/qb.sh $O/gg qwen_more > $O/qwen_more.sum 2>&1; step qwen_more
QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prompts $O/qwen4k.json --timeout 3000 $Q" $O/qb.sh $O/gg qwen_4k GUILD_NO_KV_CACHE=1 > $O/qwen_4k.sum 2>&1; step qwen_4k

# 2. GLM-5.3-Flash UD-IQ4_XS, its defaults (NextN, the attention as Q6_K): four chat prompts x 256 tokens with drafts
#    and without (--plain), 256K context; a 1,974-token prompt
P=/home/ubuntu/work/dflash/glm/prompts.txt
/home/ubuntu/lanes/bin/glm-run $O/glm5_run $O/glm_chat.out 0 256 --ctx 262144 --spec --prompts $P --plain -- GLM5_RUN_IGNORE_EOS=1 > /dev/null 2>&1; step glm_chat
/home/ubuntu/lanes/bin/glm-run $O/glm5_run $O/glm_prompt.out text:/tmp/claude-1000/-home-ubuntu-Guild/fc412f01-5572-5649-9611-4ca7c5573b54/scratchpad/kl/text1.txt 32 --tokenizer /home/ubuntu/models/GLM-5.3-Flash/pack/tokenizer --ctx 262144 --spec > /dev/null 2>&1; step glm_prompt

# 3. Ornith-1.5-35B Q4_K_M: a 1,024-token prompt (batch 256) and 256 tokens, plain and with its MTP layer (--spec 4)
for s in 0 4; do
  (cd /home/ubuntu/Guild && env GUILD_NO_KV_CACHE=1 $G $O/ornith_prefill_bench --prompt 1024 --generate 256 --runs 3 --batch 256 --context 4096 --spec $s --json-summary $O/ornith_spec$s.json > $O/ornith_spec$s.log 2>&1); step ornith_spec$s
done

# 4. StartLux-Decision Q8_0: the short requests, 5 runs each, fresh state
W=/home/ubuntu/models/startlux-decision-35b-a3b-gguf
(cd /home/ubuntu/Guild && $G $O/decision_run --native $W/StartLux-Decision-35B-A3B-Q8_0.gguf --pack $W/pack-Q8_0 --resident-budget-gib 40 --requests /home/ubuntu/work/startlux/perf/perf-short.jsonl --out $O/startlux.jsonl --repeat 5 --fresh > $O/startlux.log 2>&1); step startlux

# 5. Nemotron 3.5 Transcribe Q8_0: the 5 s and 29.6 s clips, greedy and with 3 MTP drafts, 6 runs (the first a warm-up)
M=/home/ubuntu/models/nemotron-3.5-transcribe-guild
for mode in greedy mtp3; do
  case $mode in greedy) E="--no-mtp";; *) E="--mtp 3";; esac
  (cd /home/ubuntu/Guild && $G $O/transcribe_run --model $M/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio $M/audio --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json $E --repeat 6 --clips /home/ubuntu/work/nemotron/perf/clips.jsonl --out $O/nemotron_$mode.jsonl > $O/nemotron_$mode.log 2>&1); step nemotron_$mode
done

# 6. Fish Audio Drama 3 Q4_K_M: the 290-character passage, seeds 11 and 12, no voice and the en0 voice
TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
V=/mnt/models-ssd/Strata-data/drama3/voices
for voice in none en0; do for seed in 11 12; do
  if [ $voice = en0 ]; then R="--ref-codes $V/en0.codes"; RT="$(cat $V/en0.txt)"; else R=""; RT=""; fi
  (cd /home/ubuntu/Guild && $G $O/gg --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $seed --wav $O/drama3_${voice}_$seed.wav $R ${RT:+--ref-text "$RT"} --say "$TEXT" > $O/drama3_${voice}_$seed.log 2>&1); step drama3_${voice}_$seed
done; done
step done
