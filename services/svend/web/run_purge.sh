#!/bin/bash
# Daily purge script for Svend

cd /home/eric/kjerne/services/svend/web

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Run purge
python3 manage.py purge_old_data --days 30
