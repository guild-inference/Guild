#!/bin/sh
B=/home/ubuntu/work/nemotron/build/q8_0_rows512_test
for nt in 1 2 16; do for pf in 0 2 4 0 2 4; do printf "pf %s " $pf; GUILD_Q8_PF=$pf $B --dram 32 $nt | tail -1; done; done
