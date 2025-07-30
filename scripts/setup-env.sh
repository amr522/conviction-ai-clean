#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install Git hook
if [ -d ".git/hooks" ]; then
  cp scripts/pre-push .git/hooks/pre-push
  chmod +x .git/hooks/pre-push
  echo "✅ pre-push Git hook installed"
fi

echo "✅ Virtualenv created and dependencies installed."
