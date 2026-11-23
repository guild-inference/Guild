#!/bin/sh
# glm.sh [outdir]: GLM-5.3-Flash at integration, run last (a load takes 12-22 min and evicts the page cache).  One
# glm5_run load, the glm-attend3 verifier's prompts (a 3K-token one and the 5-token one), arms C (GUILD_GLM_ATTEND3=0,
# GUILD_GLM_FUSE2 without bit 4: round 19's GPU path) and A (the defaults), 3 rounds (C A, A C, C A).  Then the ids:
# the 5-token prompt's against base20's glm1.out, the 3K one's against the verifier's (md5 of the stripped id line).
I=${I:-/home/ubuntu/lanes/r20/int/build}; O=${1:-/home/ubuntu/lanes/r20/int/glm}; V=/home/ubuntu/lanes/r20/glm-attend3/verify
mkdir -p $O
date -u +%FT%TZ > $O/t0
/home/ubuntu/lanes/bin/glm-run $I/glm5_run $O/glm-ab.out 0 256 --ctx 32768 --spec --prompts $V/prompts.txt \
  --ab GUILD_GLM_ATTEND3=0,1 --ab GUILD_GLM_FUSE2=0xfffffffb,0xffffffff --reps 3 -- GLM5_RUN_STEPS=1 > $O/glm-ab.summary 2>&1
echo $? > $O/rc
date -u +%FT%TZ >> $O/t0
python3 $V/parse_ab.py $O/glm-ab.out /home/ubuntu/lanes/r20/base20/bl/glm1.out > $O/parse.txt 2>&1
python3 - $O/glm-ab.out > $O/ids.txt <<'PY'
import hashlib, re, sys
want = {'short': '7aee0c4c56f9d17cedd399684372d687', 'long': 'ddad1ae1b3b5177642afbf122af32925'}
ok = True; n = 0
for l in open(sys.argv[1]).read().splitlines():
    m = re.match(r"^ids (\S+) (\S+) round (\d+): ?(.*)$", l)
    if m:
        n += 1; h = hashlib.md5(m.group(4).strip().encode()).hexdigest(); good = h == want.get(m.group(1))
        ok &= good; print(m.group(1), m.group(2), 'round', m.group(3), h, 'OK' if good else 'DIFFER')
print('GLM ids', 'PASS' if ok and n == 12 else 'FAIL', f'({n} generations)')
PY
grep -E "VRAM|FFNs" $O/glm-ab.out | head -4 > $O/vram.txt
