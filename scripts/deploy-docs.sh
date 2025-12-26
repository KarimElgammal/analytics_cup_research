#!/bin/bash
# Deploy documentation only (no git commit)
set -e

cd "$(dirname "$0")/.."

echo "=== Building and Deploying Documentation ==="
PYTHONPATH=. uv run mkdocs gh-deploy --force

echo ""
echo "Done! Docs: https://karimelgammal.github.io/analytics_cup_research/"
