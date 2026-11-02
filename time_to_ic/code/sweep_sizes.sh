#!/bin/bash
# width/depth/kernel variants can't load the 32x4 pretrained checkpoint; run them from scratch once the main CNN sweep is done
cd /home/claude/work
while pgrep -f "src/sweep.py cnn" >/dev/null; do sleep 30; done
for c in '16 3 7 w16d3' '64 4 7 w64d4' '32 5 7 w32d5' '32 4 15 k15'; do set -- $c
  [ -f sweep/cnn_${4}_scratch.json ] || python3 src/dl2.py "{\"reps\":3,\"threads\":2,\"seg\":\"arch\",\"arch\":\"cnn\",\"width\":$1,\"depth\":$2,\"kernel\":$3}" cnn_${4}_scratch > sweep/cnn_${4}_scratch.log 2>&1
done
