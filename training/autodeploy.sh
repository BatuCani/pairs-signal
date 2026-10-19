#!/bin/bash
# Wait for a distil labs training job, deploy the trained model, open an inference endpoint
# and link an existing distil labs API key (by name) to it.
# usage: autodeploy.sh <slm-id> <endpoint-name>
# env: DISTIL_BIN (default: distil on PATH), WORK_DIR (default: current directory),
#      DISTIL_API_KEY_NAME (name of a key created with `distil api-keys create`, default: pairs-demo-key)
DISTIL=${DISTIL_BIN:-distil}
D=${WORK_DIR:-.}
KEY_NAME=${DISTIL_API_KEY_NAME:-pairs-demo-key}
SLM=$1; NAME=$2
log(){ echo "$(date +%H:%M:%S) $*" | tee -a "$D/deploy.log"; }
for i in $(seq 120); do
  s=$("$DISTIL" slm status --output json "$SLM" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status'))" 2>/dev/null)
  case "$s" in *SUCCESS*) break;; *FAIL*|*STOP*) log "slm $s"; exit 1;; esac
  sleep 15
done
log "slm $SLM SUCCESS, deploying"
dep=$("$DISTIL" deployment create-from-slm --output json "$SLM" 2>&1 | tee "$D/deploy-$NAME.json" | python3 -c "import json,sys;print(json.load(sys.stdin).get('id'))" 2>/dev/null)
log "deployment $dep"
[ -n "$dep" ] && [ "$dep" != None ] || { log "deploy create failed: $(head -c 300 "$D/deploy-$NAME.json")"; exit 1; }
for i in $(seq 80); do
  ds=$("$DISTIL" deployment status --output json "$dep" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('deployment_status') or d.get('status'))" 2>/dev/null)
  log "deployment status $ds"
  case "$ds" in *RUNNING*|*READY*|*ACTIVE*|*SUCCESS*|*DEPLOYED*) break;; *FAIL*) exit 1;; esac
  sleep 15
done
ep=$("$DISTIL" inference-endpoint create-from-deployment --output json --name "$NAME" "$dep" 2>&1 | tee "$D/endpoint-$NAME.json")
log "endpoint: $(echo "$ep" | head -c 400)"
"$DISTIL" inference-endpoint link-api-key "$NAME" "$KEY_NAME" 2>&1 | tee -a "$D/deploy.log"
log "DONE"
