#!/bin/sh
# GLM decode after its files left the page cache and other models' files filled it (the state of base1)
O=/home/ubuntu/work/nemotron/opt/r20/glm
G=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS
FILL="/home/ubuntu/models/drama3/Drama3-BF16.gguf /home/ubuntu/models/startlux-decision-35b-a3b-gguf/StartLux-Decision-35B-A3B-Q8_0.gguf /home/ubuntu/models/nemotron-3.5-transcribe-guild/Nemotron-3.5-Transcribe-LM-Q8_0.gguf /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf"
python3 -I -c "
import os,sys
for p in sys.argv[1:]:
    fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)" $G/*.gguf
cat $FILL > /dev/null
: > $O/repro.mon
numastat -m | grep -E "MemFree|FilePages" >> $O/repro.mon
grep -E "^thp_collapse_alloc |^thp_fault_alloc |^thp_fault_fallback |^compact_stall |^pswpin |^pswpout " /proc/vmstat | tr '\n' ' ' >> $O/repro.mon; echo >> $O/repro.mon
(/home/ubuntu/lanes/bin/glm-run $O/gr $O/base3.out > /dev/null 2>&1) &
B=$!
while kill -0 $B 2>/dev/null; do
  P=$(pgrep -x gr | head -1)
  if [ -n "$P" ]; then
    echo "$(date +%T) steps $(grep -c 'step ' $O/base3.out) majflt $(awk '{print $12}' /proc/$P/stat) rd $(awk '/^read_bytes/{print $2}' /proc/$P/io) $(grep -E 'VmSwap|AnonHugePages' /proc/$P/status 2>/dev/null | tr -s ' ' | tr '\n' ' ') $(grep -E '^thp_collapse_alloc |^thp_fault_fallback |^compact_stall |^pswpin |^pswpout ' /proc/vmstat | tr '\n' ' ')" >> $O/repro.mon
  fi
  sleep 10
done
echo done >> $O/repro.mon
