#!/bin/bash
cd /home/claude/work
while pgrep -f "src/sweep.py pn" >/dev/null; do sleep 30; done
for c in '16 3 w16d3' '64 4 w64d4'; do set -- $c
  [ -f sweep/pn_${3}_scratch.json ] || python3 src/dl2.py "{\"reps\":3,\"threads\":2,\"seg\":\"arch\",\"arch\":\"pointnet\",\"width\":$1,\"depth\":$2}" pn_${3}_scratch > sweep/pn_${3}_scratch.log 2>&1
done
