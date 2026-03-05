#!/bin/bash
# Daily purge script for Svend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Run purge
python3 manage.py purge_old_data --days 30
