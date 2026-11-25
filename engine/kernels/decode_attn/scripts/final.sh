#!/usr/bin/env bash
# S2b final grid on GPU 6 (67), full discipline: per cell config B (2 rounds, sandwich) -> ours (2 rounds, sandwich,
# baseline arm = R 1) -> config B again. 24 s cells, first 2 s dropped. Usage: final.sh <tag-prefix> "[label:]<n>:<L>:<dg>" ...
S=~/inkling-cmp-engine/s2b; cd $S; P=$1; shift
ARMS=${ARMS:-bf16:head:1:0:7:A/B3,bf16:head:8:0:7:A/B3,e4m3/bf16:head:8:0:7:A/B3}
for cell in "$@"; do
  IFS=: read a b c d <<< "$cell"
  if [ -n "$d" ]; then n=$b; L=$c; dg=$d; else n=$a; L=$b; dg=$c; fi
  T=$P-n$n-L$L; mkdir -p $T
  export CFGB_PY=/s2b/cfgb_attn_kinds.py
  bash scripts/cfgb6.sh $T --n $n --L $L --R 1,8 --distinct-global $dg --kinds step
  echo "step --mode step --secs 24 --discard 2 --rounds 2 --n $n --L $L --kv-budget-gb ${BUDGET:-42} --mem-cap-gb 58 --arms $ARMS" > $T/cmds.txt
  bash scripts/run6.sh $T
  bash scripts/cfgb6.sh $T --n $n --L $L --R 1,8 --distinct-global $dg --kinds step
  echo "cell $cell done $(date -u +%T)" >> $S/$P.log
done
