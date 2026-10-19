#!/bin/bash
# Poll a distil labs job every 30 s (up to 45 min) and append its status to progress.log.
# usage: watch.sh <group> <id> <label>   e.g. watch.sh slm <slm-id> train-fast
# env: DISTIL_BIN (default: distil on PATH), WORK_DIR (default: current directory)
DISTIL=${DISTIL_BIN:-distil}
D=${WORK_DIR:-.}
for i in $(seq 90); do
  s=$("$DISTIL" "$1" status --output json "$2" 2>/dev/null | python3 -c "import json,sys
try:
 d=json.load(sys.stdin); print(d.get('status') or d)
except Exception: print('NO_ANSWER')")
  echo "$(date +%H:%M:%S) $3 $s" >> "$D/progress.log"
  case "$s" in *SUCCESS*|*FAILURE*|*STOPPED*|*FAILED*) echo "$3 DONE $s"; exit 0;; esac
  sleep 30
done
echo "$3 TIMEOUT $s"
