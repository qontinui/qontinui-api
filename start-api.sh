#!/bin/bash

# Start Qontinui API Service

echo "Starting Qontinui API Service..."

# Activate virtual environment (if it exists)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the API server on port 8001
uvicorn main:app --reload --host 0.0.0.0 --port 8001
