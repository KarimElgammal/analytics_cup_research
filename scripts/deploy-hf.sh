#!/bin/bash
set -e
cd "$(dirname "$0")/.."

HF_FRONTMATTER="---
title: Finding Alvarez in the A-League
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: \"1.40.0\"
app_file: app.py
pinned: false
---

"

cp README.md README.md.bak
echo "$HF_FRONTMATTER$(cat README.md)" > README.md

hf upload KarimElgammal/analytics-cup-research . . --repo-type=space \
    --exclude="docs/assets/*" --exclude=".git/*" --exclude=".venv/*" \
    --exclude="site/*" --exclude="__pycache__/*" --exclude="*.pyc" \
    --exclude=".rate_limits.json" --exclude="github_token.txt" --exclude="hf_token.txt" \
    --exclude="README.md.bak"

mv README.md.bak README.md

echo "Done! HuggingFace: https://huggingface.co/spaces/KarimElgammal/analytics-cup-research"
