#!/bin/sh
# the 60 gate clips greedy and with 2 and 3 drafts; every clip's ids against guild5-greedy (the committed engine's)
G=/home/ubuntu/work/nemotron/gate
M=/home/ubuntu/models/nemotron-3.5-transcribe-guild
R=${R:-/home/ubuntu/work/nemotron/build/transcribe_run}
O=/home/ubuntu/work/nemotron/opt
T=${TAG:-new}
PY=/home/ubuntu/Guild/.venv/bin/python
for mode in ${MODES:-greedy mtp2 mtp3}; do
  case $mode in greedy) E="--no-mtp";; *) E="--mtp ${mode#mtp}";; esac
  $R --model $M/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio $M/audio \
    --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json $E --clips $G/clips.jsonl \
    --out $O/$T-gate-$mode.jsonl > $O/$T-gate-$mode.log 2>&1
  echo "$mode rc $?" >> $O/$T-gate.progress
  $PY /home/ubuntu/work/nemotron/guild/tools/transcribe_gate.py --ref $G/guild5-greedy.jsonl --got $O/$T-gate-$mode.jsonl \
    --clips $G/clips.jsonl --same-as $G/guild5-greedy.jsonl > $O/$T-gate-$mode.cmp 2>&1
  echo "$mode cmp rc $?" >> $O/$T-gate.progress
done
echo done >> $O/$T-gate.progress
