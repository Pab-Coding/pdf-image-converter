#!/usr/bin/env bash
set -euo pipefail

# cd to script directory
cd "$(dirname "$0")"

VENV=".venv"
PYTHON=${PYTHON:-python3}

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  $PYTHON -m venv "$VENV"
fi

source "$VENV/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo "Starting server at http://127.0.0.1:8000 ..."
exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
