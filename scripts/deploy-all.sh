#!/bin/bash
# Deploy everything: GitHub, Docs, and HuggingFace Space
set -e

echo "=== Deploying to GitHub ==="
git add -A
git commit -m "${1:-Update}" --allow-empty || true
git push origin main

echo ""
echo "=== Deploying Documentation ==="
PYTHONPATH=. uv run mkdocs gh-deploy --force

echo ""
echo "=== Deploying to HuggingFace Space ==="
git push hf main

echo ""
echo "=== All deployments complete! ==="
echo "GitHub: https://github.com/KarimElgammal/analytics_cup_research"
echo "Docs: https://karimelgammal.github.io/analytics_cup_research/"
echo "Space: https://huggingface.co/spaces/KarimElgammal/analytics-cup-research"
