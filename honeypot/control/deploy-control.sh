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
echo -e "${GREEN}    Cowrie & Vector AWS Control Node Orchestration Engine             ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# ------------------------------------------------------------------------------
# PHASE 0: SELF-UPDATE
# ------------------------------------------------------------------------------
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$REPO_ROOT" ]; then
    echo -e "${YELLOW}[*] Pulling latest changes...${NC}"
    # This host should always exactly mirror origin/main -- a plain `pull`
    # refuses to run if anything here was ever hand-edited (e.g. during
    # debugging via scp), so reset hard instead of merging.
    git -C "$REPO_ROOT" fetch origin main
    git -C "$REPO_ROOT" reset --hard origin/main
fi

# ------------------------------------------------------------------------------
# PHASE 1: CLOUD ENVIRONMENT VALIDATION
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[*] Validating cloud execution environment via IMDSv2...${NC}"
TOKEN=$(curl -s -S -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" --max-time 2 || true)

if [ -z "$TOKEN" ]; then
    echo -e "${RED}❌ CRITICAL CONFIGURATION FAULT: AWS Metadata Service unreachable.${NC}"
    echo -e "${RED}This script must be executed exclusively inside an AWS EC2 instance. Terminating.${NC}"
    exit 1
fi

AWS_INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
AWS_PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)

echo -e "${GREEN}✔ AWS Cloud Environment Validated.${NC}"
echo -e "   Instance ID: ${GREEN}$AWS_INSTANCE_ID${NC}"
echo -e "   Public IP:   ${GREEN}$AWS_PUBLIC_IP${NC}\n"

# ------------------------------------------------------------------------------
# PHASE 2: .ENV PROVISIONING (only what Docker Compose actually reads)
# ------------------------------------------------------------------------------
if [ ! -f .env ]; then
    echo -e "${YELLOW}[*] Writing local environment profile (.env)...${NC}"
    cat << EOF > .env
# ==============================================================================
# Automated AWS Control Node Environment Variables (Local Deployment Only)
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

# Secure Vector's configuration specifically to avoid ownership conflicts
if [ -f vector.toml ]; then
    chmod 664 vector.toml
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
# vector.toml) changed -- Compose only reacts to changes it tracks itself.
sudo docker compose pull
sudo docker compose up -d --force-recreate

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}🚀 DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
echo -e "   Control Node IP:     ${YELLOW}$AWS_PUBLIC_IP${NC}"
echo -e "   Target GCP Sink ID:  ${YELLOW}$GCP_PROJECT_ID${NC}"
echo -e "   GCP Ingest Topic:    ${YELLOW}$GCP_PUBSUB_TOPIC${NC}"
echo -e "${GREEN}======================================================================${NC}"
