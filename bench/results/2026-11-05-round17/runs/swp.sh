#!/bin/sh
# GLM decode with swap traffic and the engine's memory sampled every 10 s
O=/home/ubuntu/work/nemotron/opt/r20/glm
: > $O/swp.mon
(/home/ubuntu/lanes/bin/glm-run $O/gr $O/base2.out > /dev/null 2>&1) &
B=$!
while kill -0 $B 2>/dev/null; do
  P=$(pgrep -x gr | head -1)
  echo "$(date +%T) $(grep -E '^pswpin|^pswpout' /proc/vmstat | awk '{printf "%s %s ", $1, $2}') $(grep -c 'per step' $O/base2.out 2>/dev/null) $( [ -n "$P" ] && numastat -p $P | tail -1) swap-of-proc $( [ -n "$P" ] && grep VmSwap /proc/$P/status)" >> $O/swp.mon
  sleep 10
done
echo done >> $O/swp.mon
