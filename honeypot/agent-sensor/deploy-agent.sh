#!/bin/bash
# Exit immediately if any command returns a non-zero exit status
set -e

# ANSI Color Coding for clean CLI terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0;37m' # No Color

# Not secrets -- just resource identifiers. The actual credential (gcp_key.json)
# is provisioned separately and never touches this script or git. Single source
# of truth: nothing downstream re-derives these from anywhere else.
GCP_PROJECT_ID="mineral-droplet-160709"
GCP_PUBSUB_TOPIC="bear-trap-honeypot-logs"

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}    Cowrie & Vector Azure Agent-Sensor Node Orchestration Engine       ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# ------------------------------------------------------------------------------
# PHASE 0: SELF-UPDATE
# ------------------------------------------------------------------------------
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$REPO_ROOT" ]; then
    echo -e "${YELLOW}[*] Pulling latest changes...${NC}"
    git -C "$REPO_ROOT" pull
fi

# ------------------------------------------------------------------------------
# PHASE 1: CLOUD ENVIRONMENT VALIDATION
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[*] Validating cloud execution environment via Azure IMDS...${NC}"
METADATA=$(curl -s -H "Metadata:true" --max-time 2 "http://169.254.169.254/metadata/instance?api-version=2021-02-01" || true)

if [ -z "$METADATA" ]; then
    echo -e "${RED}❌ CRITICAL CONFIGURATION FAULT: Azure Metadata Service unreachable.${NC}"
    echo -e "${RED}This script must be executed exclusively inside an Azure VM. Terminating.${NC}"
    exit 1
fi

AZURE_VM_NAME=$(echo "$METADATA" | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)
AZURE_PUBLIC_IP=$(curl -s -H "Metadata:true" --max-time 2 "http://169.254.169.254/metadata/instance/network?api-version=2021-02-01" \
    | grep -o '"publicIpAddress":"[^"]*"' | head -1 | cut -d'"' -f4)

echo -e "${GREEN}✔ Azure Cloud Environment Validated.${NC}"
echo -e "   VM Name:     ${GREEN}$AZURE_VM_NAME${NC}"
echo -e "   Public IP:   ${GREEN}$AZURE_PUBLIC_IP${NC}\n"

# ------------------------------------------------------------------------------
# PHASE 2: .ENV PROVISIONING (only what Docker Compose actually reads)
# ------------------------------------------------------------------------------
if [ ! -f .env ]; then
    echo -e "${YELLOW}[*] Writing local environment profile (.env)...${NC}"
    cat << EOF > .env
# ==============================================================================
# Automated Azure Agent-Sensor Node Environment Variables (Local Deployment Only)
# Compiled on: $(date -u)
# ==============================================================================
GCP_PROJECT_ID='$GCP_PROJECT_ID'
GCP_PUBSUB_TOPIC='$GCP_PUBSUB_TOPIC'
EOF
    chmod 600 .env
    echo -e "${GREEN}✔ Local state successfully initialized and locked down (.env).${NC}\n"
else
    echo -e "${GREEN}✔ Local environment profile (.env) discovered. Skipping.${NC}\n"
fi

# ------------------------------------------------------------------------------
# PHASE 3: AUTOMATED SYSTEM PACKAGE MANIFEST VERIFICATION
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[*] Inspecting system runtimes and host dependency trees...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker runtime not discovered. Initiating host upgrades and engine installation...${NC}"
    sudo apt-get update -y
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common gnupg

    # Clean, non-deprecated GPG implementation pathing
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo systemctl enable docker
    sudo systemctl start docker
fi

if ! sudo docker compose version &> /dev/null; then
    echo -e "${YELLOW}Docker Compose plugin missing. Bundling component dependencies...${NC}"
    sudo apt-get update -y && sudo apt-get install -y docker-compose-plugin
fi

echo -e "${GREEN}✔ Host runtime dependencies verified successfully.${NC}\n"

# ------------------------------------------------------------------------------
# PHASE 4: CONFIGURATION SANITIZATION & ORCHESTRATION UPSTAGES
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[*] Enforcing configuration file boundaries and launching containers...${NC}"

# Ensure default template text maps are globally readable
chmod 644 cowrie.cfg 2>/dev/null || true

if [ -f userdb.txt ]; then
    chmod 644 userdb.txt
fi

if [ -f vector.toml ]; then
    chmod 664 vector.toml
fi

# Custom Cowrie command-injection layer -- must stay world-readable so the
# container's non-root cowrie user can load it.
if [ -d handlers ]; then
    chmod 644 handlers/*.py handlers/*.yaml handlers/*.txt 2>/dev/null || true
fi

# The GCP service account key is provisioned out-of-band (copied onto this host
# directly, never stored in git) -- fail clearly if it's missing rather than
# letting Vector silently fail to authenticate after containers start.
if [ ! -f gcp_key.json ]; then
    echo -e "${RED}❌ CRITICAL CONFIGURATION FAULT: gcp_key.json not found in $(pwd).${NC}"
    echo -e "${RED}Copy the GCP service account key for the Vector log shipper here before deploying.${NC}"
    exit 1
fi
chmod 600 gcp_key.json

# Refresh engine cache maps and boot the architecture.
# --force-recreate matters here: `up -d` alone won't recreate a running
# container just because a bind-mounted host file (cowrie.cfg, userdb.txt,
# vector.toml, handlers/*) changed -- Compose only reacts to changes it tracks itself.
sudo docker compose pull
sudo docker compose up -d --force-recreate

# ------------------------------------------------------------------------------
# PHASE 5: RUNTIME SMOKE CHECK
# ------------------------------------------------------------------------------
# The pytest/ruff CI validation exercises base_injector.py in isolation, not
# inside Cowrie's own interpreter -- this catches load-time failures (bad
# commands.yaml, import errors in the mounted handler files) that would
# otherwise only surface the first time an attacker actually hits the honeypot.
echo -e "${YELLOW}[*] Waiting for cowrie_detector to settle, then checking for startup errors...${NC}"
sleep 5
if sudo docker compose logs cowrie_detector --tail 100 | grep -iq "traceback"; then
    echo -e "${RED}⚠ Traceback detected in cowrie_detector startup logs -- check the custom command handlers.${NC}"
    sudo docker compose logs cowrie_detector --tail 100 | grep -i -A 20 "traceback"
    exit 1
else
    echo -e "${GREEN}✔ No startup tracebacks detected.${NC}"
fi

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}🚀 DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
echo -e "   Agent-Sensor Node IP: ${YELLOW}$AZURE_PUBLIC_IP${NC}"
echo -e "   Target GCP Sink ID:   ${YELLOW}$GCP_PROJECT_ID${NC}"
echo -e "   GCP Ingest Topic:     ${YELLOW}$GCP_PUBSUB_TOPIC${NC}"
echo -e "${GREEN}======================================================================${NC}"
