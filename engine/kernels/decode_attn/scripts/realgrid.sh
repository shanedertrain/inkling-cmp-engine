#!/usr/bin/env bash
# Task 2 (timing confound): ours and config B on the SAME real-KV dump (K, V, q of captured executions tiled to length;
# rel / gate / sink synthetic on both sides). Per cell config B -> ours -> config B, 24 s cells, 2 rounds, sandwich.
# Usage: realgrid.sh <tag-prefix> "<label>:<n>:<L>:<dg>" ...
S=~/inkling-cmp-engine/s2b; cd $S; P=$1; shift
ARMS=${ARMS:-bf16:head:1:0:7:A/B3,bf16:head:8:0:7:A/B3,e4m3/bf16:head:8:0:7:A/B3}
export CFGB_PY=/s2b/cfgb_attn_kinds.py
for cell in "$@"; do
  IFS=: read lab n L dg <<< "$cell"
  T=$P-$lab; mkdir -p $T
  bash scripts/cfgb6.sh $T --n $n --L $L --R 1,8 --distinct-global $dg --kinds step --kvdump /dump
  echo "step --mode step --secs 24 --discard 2 --rounds 2 --n $n --L $L --kv-budget-gb ${BUDGET:-42} --mem-cap-gb 58 --kvdump /dump --arms $ARMS" > $T/cmds.txt
  bash scripts/run6.sh $T
  bash scripts/cfgb6.sh $T --n $n --L $L --R 1,8 --distinct-global $dg --kinds step --kvdump /dump
  echo "cell $cell done $(date -u +%T)" >> $S/$P.log
done
