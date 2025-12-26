#!/bin/bash
# Deploy to HuggingFace Space only
set -e

cd "$(dirname "$0")/.."

echo "=== Deploying to HuggingFace Space ==="
hf upload KarimElgammal/analytics-cup-research . . --repo-type=space \
    --exclude="docs/assets/*" --exclude=".git/*" --exclude=".venv/*" \
    --exclude="site/*" --exclude="__pycache__/*" --exclude="*.pyc" \
    --exclude=".rate_limits.json" --exclude="github_token.txt" --exclude="hf_token.txt"

echo ""
echo "Done! Space: https://huggingface.co/spaces/KarimElgammal/analytics-cup-research"
