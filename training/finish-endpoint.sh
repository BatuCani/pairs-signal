#!/bin/bash
# Finish a deployment that autodeploy.sh started: wait for JOB_SUCCESS, open the inference endpoint,
# link the API key and list all endpoints.
# usage: finish-endpoint.sh <deployment-id> [endpoint-name]   (the hackathon run used news-gate-batu-v1)
# env: DISTIL_BIN (default: distil on PATH), WORK_DIR (default: current directory),
#      DISTIL_API_KEY_NAME (default: pairs-demo-key)
DISTIL=${DISTIL_BIN:-distil}
D=${WORK_DIR:-.}
KEY_NAME=${DISTIL_API_KEY_NAME:-pairs-demo-key}
DEP=${1:?usage: finish-endpoint.sh <deployment-id> [endpoint-name]}; NAME=${2:-news-gate-batu-v1}
for i in $(seq 80); do
  out=$("$DISTIL" deployment status --output json "$DEP" 2>&1)
  echo "$(date +%H:%M:%S) $(echo $out | tr -d '\n ')" >> "$D/deploy.log"
  echo "$out" | grep -q '"deployment_status": "JOB_SUCCESS"' && break
  echo "$out" | grep -qi 'FAIL' && { echo FAILED; exit 1; }
  sleep 15
done
"$DISTIL" inference-endpoint create-from-deployment --output json --name "$NAME" "$DEP" > "$D/endpoint.json" 2>&1
cat "$D/endpoint.json" >> "$D/deploy.log"
"$DISTIL" inference-endpoint link-api-key "$NAME" "$KEY_NAME" >> "$D/deploy.log" 2>&1
"$DISTIL" inference-endpoint list --output json >> "$D/deploy.log" 2>&1
echo ENDPOINT-DONE
