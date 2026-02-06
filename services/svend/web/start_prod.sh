#!/bin/bash
# Production startup script for Svend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="/home/eric/.local/bin:$PATH"

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Start gunicorn (bound to localhost - Cloudflare Tunnel handles external)
echo "Starting gunicorn on 127.0.0.1:8000..."
exec /home/eric/.local/bin/gunicorn svend.wsgi:application \
    --bind 127.0.0.1:8000 \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile /var/log/svend/access.log \
    --error-logfile /var/log/svend/error.log \
    --capture-output
