#!/bin/bash
# Peek at recent honeypot log events flowing through GCP Pub/Sub.
#
# Uses a persistent "viewer" subscription (created on first run) rather than
# a fresh one each time. Messages are auto-acked on pull, so this behaves
# like a tail of what's arrived since the last check, not a stable "last N"
# history -- Pub/Sub is a delivery queue, not a queryable log store. Once
# the BigQuery sink exists, `bq query ... ORDER BY timestamp DESC LIMIT N`
# is the better tool for that.
set -e

PROJECT_ID="mineral-droplet-160709"
TOPIC="bear-trap-honeypot-logs"
SUBSCRIPTION="bear-trap-log-viewer"
LIMIT="${1:-10}"

if ! gcloud pubsub subscriptions describe "$SUBSCRIPTION" --project="$PROJECT_ID" &>/dev/null; then
    echo "Creating persistent viewer subscription '$SUBSCRIPTION'..." >&2
    gcloud pubsub subscriptions create "$SUBSCRIPTION" --topic="$TOPIC" --project="$PROJECT_ID" >&2
fi

gcloud pubsub subscriptions pull "$SUBSCRIPTION" --limit="$LIMIT" --auto-ack --project="$PROJECT_ID" --format=json \
    | python3 -c "
import base64
import json
import sys

data = json.load(sys.stdin)
if not data:
    print('No new messages since the last check.')
for item in data:
    raw = base64.b64decode(item.get('message', {}).get('data', ''))
    try:
        print(json.dumps(json.loads(raw), indent=2))
    except Exception:
        print(raw.decode('utf-8', errors='replace'))
    print()
"
