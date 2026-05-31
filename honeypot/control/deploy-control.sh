#!/bin/bash
# Exit immediately if any command returns a non-zero exit status
set -e

# ANSI Color Coding for clean CLI terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0;37m' # No Color

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}    Cowrie & Vector AWS Control Node Orchestration Engine             ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# ------------------------------------------------------------------------------
# PHASE 1: SELF-HEALING CONFIGURATION DEPLOYMENT (.env Validation)
# ------------------------------------------------------------------------------
if [ ! -f .env ]; then
    echo -e "${YELLOW}[!] Local environment profile (.env) not discovered.${NC}"
    echo -e "${YELLOW}[*] Activating cloud validation and configuration prompt sequence...${NC}\n"

    # 1. AWS Cloud Verification via IMDSv2
    echo -e "${YELLOW}[1/3] Validating cloud execution environment via IMDSv2...${NC}"
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

    # 2. Interactive Input Sequence for GCP Ingestion Infrastructure
    echo -e "${YELLOW}[2/3] Configuring GCP Central Logging Sink Target...${NC}"
    read -p "Enter Target GCP Project ID: " GCP_ID
    read -p "Enter Target GCP Pub/Sub Topic: " GCP_TOPIC

    echo -e "${YELLOW}Please paste the absolute contents of your GCP Service Account Key JSON file.${NC}"
    echo -e "${YELLOW}Press Ctrl+D when finished pasting:${NC}"
    GCP_RAW_JSON=$(cat)

    # Validate inputs aren't empty
    if [ -z "$GCP_ID" ] || [ -z "$GCP_TOPIC" ] || [ -z "$GCP_RAW_JSON" ]; then
        echo -e "${RED}❌ ERROR: GCP pipeline attributes cannot be initialized with blank values.${NC}"
        exit 1
    fi

    # Flatten the JSON string so Docker can safely pass it as a single-line environment variable
    GCP_CLEAN_JSON=$(echo "$GCP_RAW_JSON" | tr -d '\n' | tr -d '\r')

    # 3. Dynamic Compilation of the Local Secrets Template
    echo -e "\n${YELLOW}[3/3] Compiling local secrets profile...${NC}"
    cat << EOF > .env
# ==============================================================================
# Automated AWS Control Node Environment Variables (Local Deployment Only)
# Compiled on: $(date -u)
# ==============================================================================
AWS_INSTANCE_ID=$AWS_INSTANCE_ID
AWS_PUBLIC_IP=$AWS_PUBLIC_IP
GCP_PROJECT_ID=$GCP_ID
GCP_PUBSUB_TOPIC=$GCP_TOPIC
GOOGLE_APPLICATION_CREDENTIALS_DATA=$GCP_CLEAN_JSON
EOF

    # Restrict read/write operations on the file strictly to the root owner
    chmod 600 .env
    echo -e "${GREEN}✔ Local state successfully initialized and locked down (.env).${NC}\n"
else
    echo -e "${GREEN}✔ Local environment profile (.env) discovered. Skipping configuration wizard.${NC}"
fi

# Load variables into the current script execution context
source .env

# ------------------------------------------------------------------------------
# PHASE 2: AUTOMATED SYSTEM PACKAGE MANIFEST VERIFICATION
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[*] Inspecting system runtimes and host dependency trees...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker runtime not discovered. Initiating host upgrades and engine installation...${NC}"
    sudo apt-get update -y
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo systemctl enable docker
    sudo systemctl start docker
fi

if ! docker compose version &> /dev/null; then
    echo -e "${YELLOW}Docker Compose plugin missing. Bundling component dependencies...${NC}"
    sudo apt-get update -y && sudo apt-get install -y docker-compose-plugin
fi

echo -e "${GREEN}✔ Host runtime dependencies verified successfully.${NC}\n"

# ------------------------------------------------------------------------------
# PHASE 3: CONFIGURATION SANITIZATION & ORCHESTRATION UPSTAGES
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[*] Enforcing configuration file boundaries and launching containers...${NC}"

# Ensure default template text maps are globally readable but not writeable by containers
chmod 644 cowrie.cfg vector.toml 2>/dev/null || true
if [ -f userdb.txt ]; then chmod 644 userdb.txt; fi

# Refresh engine cache maps and boot the architecture
sudo docker compose pull
sudo docker compose up -d

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}🚀 DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
echo -e "   Control Node IP:     ${YELLOW}$AWS_PUBLIC_IP${NC}"
echo -e "   Target GCP Sink ID:  ${YELLOW}$GCP_PROJECT_ID${NC}"
echo -e "   GCP Ingest Topic:    ${YELLOW}$GCP_PUBSUB_TOPIC${NC}"
echo -e "${GREEN}======