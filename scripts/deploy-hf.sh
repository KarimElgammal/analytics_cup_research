#!/bin/bash
# Deploy to HuggingFace Space only
set -e

cd "$(dirname "$0")/.."

# Add HF frontmatter to README temporarily
HF_FRONTMATTER="---
title: Archetype Comparison Tool
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: \"1.40.0\"
app_file: app.py
pinned: false
---

"

# Backup original README
cp README.md README.md.bak

# Add frontmatter
echo "$HF_FRONTMATTER$(cat README.md)" > README.md

echo "=== Deploying to HuggingFace Space ==="
hf upload KarimElgammal/analytics-cup-research . . --repo-type=space \
    --exclude="docs/assets/*" --exclude=".git/*" --exclude=".venv/*" \
    --exclude="site/*" --exclude="__pycache__/*" --exclude="*.pyc" \
    --exclude=".rate_limits.json" --exclude="github_token.txt" --exclude="hf_token.txt" \
    --exclude="README.md.bak"

# Restore original README
mv README.md.bak README.md

echo ""
echo "Done! Space: https://huggingface.co/spaces/KarimElgammal/analytics-cup-research"
