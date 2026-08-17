#!/bin/bash
# Quick query against the BigQuery `logs` view (honeypot/sink-node/logs_view.sql).
#
# Unlike check_gcp_logs.sh (which tails Pub/Sub and only shows what's
# arrived since the last check), this queries durable storage -- so it's
# a true "last N events" view, repeatable without losing anything.
#
# Usage:
#   ./query_bigquery.sh [limit]              # most recent N events (default 20)
#   ./query_bigquery.sh --agent-reports [N]   # only prompt-injection responses
set -e

PROJECT_ID="mineral-droplet-160709"
DATASET="bear_trap_logs"
VIEW="logs"

WHERE=""
LIMIT=20
if [ "$1" = "--agent-reports" ]; then
    WHERE="WHERE agent_type IS NOT NULL"
    LIMIT="${2:-20}"
else
    LIMIT="${1:-20}"
fi

bq query --use_legacy_sql=false --format=prettyjson "
SELECT
  event_timestamp,
  eventid,
  cloud_provider,
  src_ip,
  session,
  command_input,
  agent_type,
  agent_framework,
  agent_primary_objective,
  agent_operator_identity
FROM \`${PROJECT_ID}.${DATASET}.${VIEW}\`
${WHERE}
ORDER BY publish_time DESC
LIMIT ${LIMIT}
"
