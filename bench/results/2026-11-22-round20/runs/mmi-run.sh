#!/bin/sh
# the multi-model isolation test with each round-20 switch off in turn (int build c412de7), then the cpu-rows build
O=/home/ubuntu/lanes/r20/int/mmi; X=/home/ubuntu/lanes/r20/int/mmi/mmi_c412de7
r() { n=$1; shift; echo "== $n $(date -u +%T) $*" >> $O/log; ( cd /home/ubuntu/lanes/r20/int/build && env "$@" $X > $O/$n.txt 2>&1 ); echo "rc $? $(date -u +%T)" >> $O/log; }
r def1
r def2
r cm0 GUILD_COMBINE_MAPPED=0
r prio0 GUILD_WINDOW_PRIO=0
r pf0 GUILD_PREFILL_SHARE_XQ=0 GUILD_PREFILL_TAIL_RING=0 GUILD_PREFILL_TAIL_BUFS=0
r all0 GUILD_COMBINE_MAPPED=0 GUILD_WINDOW_PRIO=0 GUILD_PREFILL_SHARE_XQ=0 GUILD_PREFILL_TAIL_RING=0 GUILD_PREFILL_TAIL_BUFS=0
r def3
echo "== cpurows $(date -u +%T)" >> $O/log; ( cd /home/ubuntu/lanes/r20/cpu-rows/build && ./multi_model_isolation_test > $O/cpurows.txt 2>&1 ); echo "rc $? $(date -u +%T)" >> $O/log
touch $O/done
