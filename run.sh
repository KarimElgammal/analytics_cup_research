#!/bin/bash
# Run the Archetype Comparison Streamlit app
# installs uv and dependencies

set -e
cd "$(dirname "$0")"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt --quiet

# Run Streamlit app
echo "Starting Archetype Comparison Tool..."
uv run streamlit run app.py "$@"
