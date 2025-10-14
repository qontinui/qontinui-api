#!/bin/bash

# Qontinui API startup script

echo "Starting Qontinui API Service..."

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set defaults if not defined
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the API service
echo "Starting FastAPI server on http://${HOST}:${PORT}"
echo "API documentation available at http://${HOST}:${PORT}/docs"

uvicorn main:app --reload --host ${HOST} --port ${PORT}
