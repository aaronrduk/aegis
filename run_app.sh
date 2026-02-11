#!/bin/bash
# Script to run the SVAMITVA Streamlit Application

# Path to virtual environment python
PYTHON_EXEC="./venv/bin/python3"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Virtual environment not found at ./venv"
    echo "Please run 'bash quickstart.sh' or 'python3 -m venv venv && source venv/bin/activate && pip install -r requirements_minimal.txt' first."
    exit 1
fi

echo "Starting SVAMITVA Feature Extraction System..."
echo "Using python: $PYTHON_EXEC"

# Run Streamlit
$PYTHON_EXEC -m streamlit run app.py "$@"
