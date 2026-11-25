#!/usr/bin/env bash
# Baseline tie re-check on GPU 6 (67): config B -> M3 tuned -> config B at n 4, L 44,729 (as P6 F-ABA).
B=~/inkling-cmp-engine/s2b; cd $B; S=$B/scripts
T=base45-n4-g6; mkdir -p $T
bash $S/cfgb6.sh $T --n 4 --L 44729 --R 1,8
echo "step --mode step --secs 24 --discard 2 --rounds 2 --n 4 --L 44729 --merge 1 --fused 0 --split-g 4 --split-s 4 --arms bf16:d:1,bf16:d:8" > $T/cmds.txt
bash $S/m3run6.sh $T
bash $S/cfgb6.sh $T --n 4 --L 44729 --R 1,8
echo "base45 done $(date -u +%FT%TZ)" >> $T/chain.log
