#!/bin/sh
# second bisect hold: the base build, single-lane builds, the new int build, cpu-rows with its switches off
O=/home/ubuntu/lanes/r20/int/mmi
r() { n=$1; x=$2; shift 2; echo "== $n $(date -u +%T) $x $*" >> $O/log2; ( cd /home/ubuntu/lanes/r20/int/build && env "$@" $O/$x > $O/$n.txt 2>&1 ); echo "rc $? $(date -u +%T)" >> $O/log2; }
r base mmi_base
r pftail mmi_pftail_d8d9214
r glm mmi_glm_1d991ba
r int4f mmi_4f3ce6d
r cpuoff mmi_cpurows_9f34736 GUILD_KQ_ROWS512=0 GUILD_POOL_IQ4XS=0 GUILD_IQ4NL_PF=2 GUILD_Q8_PAIR=0
r cpudef mmi_cpurows_9f34736
r hc0 mmi_c412de7 GUILD_HOST_COMBINE=0
r base2 mmi_base
touch $O/done2
