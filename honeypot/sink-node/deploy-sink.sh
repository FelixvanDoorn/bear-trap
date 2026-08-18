#!/bin/bash
# Idempotent setup for the GCP sink: BigQuery dataset/table, a Pub/Sub
# subscription that streams the honeypot-logs topic straight into it, IAM
# for the Pub/Sub service agent to write, and the query-friendly view on
# top. Safe to re-run -- every step checks before creating.
#
# Requires: gcloud + bq CLIs, authenticated as an account with permissions
# on this project. Never run automatically by CI -- see deploy-gcp.yml.
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0;37m'

PROJECT_ID="mineral-droplet-160709"
# Fixed for this project, not secret -- hardcoded rather than looked up via
# `gcloud projects describe` so this script doesn't need
# resourcemanager.projects.get / Cloud Resource Manager API enabled, which
# the CI service account intentionally doesn't have (least privilege).
PROJECT_NUMBER="549783972519"
TOPIC="bear-trap-honeypot-logs"
DATASET="bear_trap_logs"
TABLE="raw_events"
VIEW="logs"
SUBSCRIPTION="bear-trap-bigquery-sink"

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}    Bear Trap GCP Sink Orchestration Engine                           ${NC}"
echo -e "${GREEN}======================================================================${NC}"

echo -e "${YELLOW}[*] Verifying active gcloud account/project...${NC}"
ACTIVE_ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
ACTIVE_PROJECT="$(gcloud config get-value project 2>/dev/null)"
echo -e "   Account: ${GREEN}${ACTIVE_ACCOUNT}${NC}"
echo -e "   Project: ${GREEN}${ACTIVE_PROJECT}${NC}"
if [ "$ACTIVE_PROJECT" != "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Active gcloud project ($ACTIVE_PROJECT) does not match expected ($PROJECT_ID).${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# BigQuery dataset
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[*] Checking BigQuery dataset ${DATASET}...${NC}"
if bq --headless show --dataset "${PROJECT_ID}:${DATASET}" &>/dev/null; then
    echo -e "${GREEN}✔ Dataset already exists.${NC}"
else
    bq --headless mk --dataset --location=US "${PROJECT_ID}:${DATASET}"
    echo -e "${GREEN}✔ Dataset created.${NC}"
fi

# ------------------------------------------------------------------------------
# BigQuery table
# ------------------------------------------------------------------------------
# Deliberately minimal and schema-stable: the raw message lands whole in
# `data` (JSON), so wildly varying Cowrie event shapes (cowrie.client.kex
# has ~8 fields nothing else has, cowrie.login.success has username/
# password, etc.) can never fail to write due to a schema mismatch or get
# silently dropped -- there's no per-field column list to fall out of sync
# with. Querying happens through the `logs` view instead (logs_view.sql).
echo -e "${YELLOW}[*] Checking BigQuery table ${TABLE}...${NC}"
if bq --headless show "${PROJECT_ID}:${DATASET}.${TABLE}" &>/dev/null; then
    echo -e "${GREEN}✔ Table already exists.${NC}"
else
    bq --headless mk --table "${PROJECT_ID}:${DATASET}.${TABLE}" \
        subscription_name:STRING,message_id:STRING,publish_time:TIMESTAMP,attributes:JSON,data:JSON
    echo -e "${GREEN}✔ Table created.${NC}"
fi

# ------------------------------------------------------------------------------
# IAM: let the Pub/Sub service agent write into the dataset
# ------------------------------------------------------------------------------
PUBSUB_SA="service-${PROJECT_NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com"
echo -e "${YELLOW}[*] Granting write access on ${DATASET} to ${PUBSUB_SA}...${NC}"
# `bq add-iam-policy-binding` on a dataset requires allowlisting we don't
# have -- use the classic dataset ACL mechanism instead (WRITER role is
# the ACL-model equivalent of roles/bigquery.dataEditor). Idempotent: only
# patches if the entry isn't already present.
#
# Fetch once and reuse, both to avoid a redundant second `bq show` call and
# so a failure here gives a real diagnostic instead of an opaque Python
# JSONDecodeError. When authenticated via Workload Identity Federation, bq
# prints "WARNING: --scopes flag may not work as expected... account type
# external_account" to STDOUT (not stderr, so 2>/dev/null doesn't touch
# it) ahead of the actual JSON -- strip anything before the first `{` to
# stay robust to this and any similar future noise, rather than matching
# this one specific warning string.
DATASET_JSON="$(bq --headless show --format=prettyjson "${PROJECT_ID}:${DATASET}" 2>/dev/null | sed -n '/^{/,$p')"
case "$DATASET_JSON" in
    \{*) ;; # looks like a JSON object, proceed
    *)
        echo -e "${RED}❌ 'bq show --format=prettyjson ${PROJECT_ID}:${DATASET}' didn't return JSON.${NC}"
        echo -e "${RED}Raw output was:${NC}"
        echo "$DATASET_JSON"
        exit 1
        ;;
esac

ALREADY_GRANTED="$(echo "$DATASET_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
sa = '${PUBSUB_SA}'
print('yes' if any(a.get('userByEmail') == sa for a in d.get('access', [])) else 'no')
")"

if [ "$ALREADY_GRANTED" = "yes" ]; then
    echo -e "${GREEN}✔ Access already granted.${NC}"
else
    TMP_ACCESS="$(mktemp)"
    echo "$DATASET_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
access = d.get('access', [])
access.append({'role': 'WRITER', 'userByEmail': '${PUBSUB_SA}'})
json.dump({'access': access}, sys.stdout, indent=2)
" > "$TMP_ACCESS"
    bq --headless update --source "$TMP_ACCESS" "${PROJECT_ID}:${DATASET}"
    rm -f "$TMP_ACCESS"
    echo -e "${GREEN}✔ Access granted.${NC}"
fi

# ------------------------------------------------------------------------------
# Pub/Sub subscription -> BigQuery
# ------------------------------------------------------------------------------
# No --use-table-schema / --use-topic-schema: that's the "wide fixed-column
# table" approach, and it fails (or silently drops fields, if
# --drop-unknown-fields is set) on any message field the table doesn't
# have a column for. Given how much the event shape varies here, plain
# metadata-only mode is the only option that can't lose data.
echo -e "${YELLOW}[*] Checking Pub/Sub subscription ${SUBSCRIPTION}...${NC}"
if gcloud pubsub subscriptions describe "$SUBSCRIPTION" --project="$PROJECT_ID" &>/dev/null; then
    echo -e "${GREEN}✔ Subscription already exists.${NC}"
else
    gcloud pubsub subscriptions create "$SUBSCRIPTION" \
        --topic="$TOPIC" \
        --project="$PROJECT_ID" \
        --bigquery-table="${PROJECT_ID}:${DATASET}.${TABLE}" \
        --write-metadata
    echo -e "${GREEN}✔ Subscription created.${NC}"
fi

# ------------------------------------------------------------------------------
# Query view
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[*] Applying logs_view.sql...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bq --headless query --use_legacy_sql=false < "${SCRIPT_DIR}/logs_view.sql"
echo -e "${GREEN}✔ View ${VIEW} applied.${NC}"

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}🚀 GCP SINK SETUP COMPLETE!${NC}"
echo -e "   Dataset:      ${YELLOW}${DATASET}${NC}"
echo -e "   Raw table:    ${YELLOW}${TABLE}${NC}"
echo -e "   Query view:   ${YELLOW}${VIEW}${NC}"
echo -e "   Subscription: ${YELLOW}${SUBSCRIPTION}${NC}"
echo -e "${GREEN}======================================================================${NC}"
