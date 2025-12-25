#!/bin/bash
# Run script for Analytics Cup Research Track submission

set -e

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Run Streamlit app (if main.py exists)
if [ -f "main.py" ]; then
    echo "Starting Streamlit app..."
    uv run streamlit run main.py
else
    echo "No main.py found. Run the Jupyter notebook instead:"
    echo "  uv run jupyter notebook submission.ipynb"
fi
