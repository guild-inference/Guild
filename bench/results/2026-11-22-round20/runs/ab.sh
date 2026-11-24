#!/bin/sh
# ab.sh [outdir]: round 20's integration A/B, base20 (round 19's binaries) against lane20/int's build.  Arms alternate
# B I I B (two pairs) inside one gpu-run hold per model; Drama 3 also runs the GUILD_DRAMA3_FUSE arms that settle its
# default.  Every run's bits are checked afterwards (abcmp.py).  GLM is glm.sh, run after this.
A=/home/ubuntu/lanes/r20/int/abk/arms.sh; B=/home/ubuntu/lanes/r20/base20; I=${I:-/home/ubuntu/lanes/r20/int/build}
O=${1:-/home/ubuntu/lanes/r20/int/ab}; G=/home/ubuntu/lanes/bin/gpu-run
STEPS=${STEPS:-"qdec odec d3 nem sl slots qpr"}
mkdir -p $O; echo "I=$I $(git -C /home/ubuntu/lanes/r20/int/guild rev-parse --short HEAD)" >> $O/progress
want() { case " $STEPS " in *" $1 "*) echo "$1 $(date -u +%T)" >> $O/progress; return 0 ;; esac; return 1; }
abba() { # abba <kind> <prefix> [VAR=value ... for the int arm]
  k=$1; p=$2; shift 2
  $G sh -c "$A $k $B $O $p-b1; $A $k $I $O $p-i1 $*; $A $k $I $O $p-i2 $*; $A $k $B $O $p-b2"
}
want qdec && abba qdec q
want odec && abba odec o
if want d3; then
  for s in 11 12; do
    $G sh -c "$A d$s $B $O d$s-b1; $A d$s $I $O d$s-i1; $A d$s $I $O d$s-ks1 GUILD_DRAMA3_FUSE=ks; \
      $A d$s $I $O d$s-ksawf1 GUILD_DRAMA3_FUSE=ksawf; $A d$s $I $O d$s-f01 GUILD_DRAMA3_FUSE=0; \
      $A d$s $I $O d$s-f02 GUILD_DRAMA3_FUSE=0; $A d$s $I $O d$s-ksawf2 GUILD_DRAMA3_FUSE=ksawf; \
      $A d$s $I $O d$s-ks2 GUILD_DRAMA3_FUSE=ks; $A d$s $I $O d$s-i2; $A d$s $B $O d$s-b2"
  done
fi
want nem && abba nem n
want sl && abba sl s
if want slots; then abba b3 b3; abba b7 b7; fi
want qpr && abba qpr p
echo "done $(date -u +%T)" >> $O/progress
