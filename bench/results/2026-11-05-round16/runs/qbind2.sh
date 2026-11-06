#!/bin/sh
# Qwen3.8 decode with node 0's memory mostly page cache: before (gpool) and after (gbind) the bound first touch.
# Before each run, 111 GB of other models' files are put in node 0's page cache again.
O=/home/ubuntu/work/nemotron/opt/r20
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
F="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
FILES="/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf /home/ubuntu/models/drama3/Drama3-BF16.gguf"
: > $O/qbind2.progress
for r in 3 4; do for a in bind nobind; do
  E=$O/gbind; [ $a = nobind ] && E=$O/gpool
  python3 -I -c "
import os,sys
for p in sys.argv[1:]:
    fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)" $FILES
  numactl -N 0 -m 0 cat $FILES > /dev/null
  FREE0=$(numastat -m | awk '/MemFree/{print $2}')
  (QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $E qb${a}_$r > $O/qb${a}_$r.sum 2>&1) &
  B=$!; : > $O/qb${a}_$r.mon
  while kill -0 $B 2>/dev/null; do P=$(pgrep -x $(basename $E) | head -1); [ -n "$P" ] && numastat -p $P | tail -1 >> $O/qb${a}_$r.mon; sleep 10; done
  MON=$(sort -k4 -n $O/qb${a}_$r.mon | tail -1)
  echo "$a round $r node0 free before ${FREE0} MB | $(grep SUMMARY $O/qb${a}_$r.sum) $(python3 -c "import json; print([hash(tuple(x['output_ids']))%100000 for x in json.load(open('$O/b-qb${a}_$r.json'))['runs']])") | $(grep -o 'resident NUMA split: [0-9]*/[0-9]*' $O/b-qb${a}_$r.log) | engine MB by node: $MON" >> $O/qbind2.progress
done; done
echo done >> $O/qbind2.progress
