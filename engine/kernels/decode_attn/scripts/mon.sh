#!/usr/bin/env bash
# mon.sh <TAG>: wait until the S2b container for TAG is gone, then summarise
S=~/inkling-cmp-engine/s2b
until ! docker ps --format '{{.Names}}' | grep -qx ink-eng-s2b-g6; do sleep 5; done
cat $S/$1/exit_codes.txt 2>/dev/null; python3 -I $S/scripts/summ.py $S/$1
