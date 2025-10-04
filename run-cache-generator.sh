#!/bin/bash
source .env

echo "Creating instance and running cache generation..."

# Clean up any existing instance
gcloud compute instances delete cache-generator --zone=us-central1-a --quiet 2>/dev/null || true

# Create instance with startup script
gcloud compute instances create cache-generator \
  --machine-type=c4-standard-32 \
  --zone=us-central1-a \
  --quiet

# Wait for instance to be ready for SSH
echo "Waiting for SSH to be available..."
until gcloud compute ssh cache-generator --zone=us-central1-a --command="echo 'SSH ready'" &>/dev/null; do
  sleep 5
  echo "Still waiting..."
done

# Cleanup function
cleanup() {
  echo "Cleaning up instance..."
  gcloud compute instances delete cache-generator --zone=us-central1-a --quiet 2>/dev/null
  exit
}

# Set trap for script interruption
trap cleanup INT TERM

# Now run our commands
gcloud compute ssh cache-generator --zone=us-central1-a --command="
  export GH_PAT='${GH_PAT}' && \
  export CLAUDE_API_KEY='${CLAUDE_API_KEY}' && \
  sudo apt-get update && \
  sudo apt-get install -y git python3 python3-pip python3-venv && \
  python3 -m venv /tmp/venv && \
  source /tmp/venv/bin/activate && \
  echo \"Using PAT (first 10 chars): \${GH_PAT:0:10}\" && \
  git config --global credential.helper store && \
  echo "https://${GH_PAT}:x-oauth-basic@github.com" > ~/.git-credentials && \
  git clone https://github.com/obliojoe/who-should-lose.git && \
  cd who-should-lose/scripts/generate_cache && \
  pip3 install -r requirements.txt && \
  python3 generate_cache.py --simulations 100000
" 

# Clean up instance after script completes or if it fails
echo "Cleaning up instance..."
gcloud compute instances delete cache-generator --zone=us-central1-a --quiet 