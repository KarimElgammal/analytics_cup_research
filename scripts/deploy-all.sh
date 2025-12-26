#!/bin/bash
# Deploy everything: GitHub, Docs, and HuggingFace Space
set -e

cd "$(dirname "$0")/.."

echo "=== Deploying to GitHub ==="
git add -A
git commit -m "${1:-Update}" --allow-empty || true
git push origin main

echo ""
echo "=== Deploying Documentation ==="
PYTHONPATH=. uv run mkdocs gh-deploy --force

echo ""
echo "=== Deploying to HuggingFace Space ==="
hf upload KarimElgammal/analytics-cup-research . . --repo-type=space \
    --exclude="docs/assets/*" --exclude=".git/*" --exclude=".venv/*" \
    --exclude="site/*" --exclude="__pycache__/*" --exclude="*.pyc" \
    --exclude=".rate_limits.json" --exclude="github_token.txt" --exclude="hf_token.txt"

echo ""
echo "=== All deployments complete! ==="
echo "GitHub: https://github.com/KarimElgammal/analytics_cup_research"
echo "Docs: https://karimelgammal.github.io/analytics_cup_research/"
echo "Space: https://huggingface.co/spaces/KarimElgammal/analytics-cup-research"
