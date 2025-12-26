#!/bin/bash
# Deploy to GitHub and rebuild docs
set -e

echo "=== Pushing to GitHub ==="
git add -A
git commit -m "${1:-Update}" --allow-empty || true
git push origin main

echo ""
echo "=== Deploying Documentation ==="
PYTHONPATH=. uv run mkdocs gh-deploy --force

echo ""
echo "Done!"
echo "GitHub: https://github.com/KarimElgammal/analytics_cup_research"
echo "Docs: https://karimelgammal.github.io/analytics_cup_research/"
