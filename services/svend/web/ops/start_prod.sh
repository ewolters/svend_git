#!/bin/bash
# Production startup script for Svend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Load field-level encryption key
KEYFILE="$HOME/.svend_encryption_key"
if [ -f "$KEYFILE" ]; then
    export SVEND_FIELD_ENCRYPTION_KEY=$(cat "$KEYFILE")
fi

# Start gunicorn (bound to localhost - Cloudflare Tunnel handles external)
echo "Starting gunicorn on 127.0.0.1:8000..."
exec gunicorn svend.wsgi:application \
    --bind 127.0.0.1:8000 \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile /var/log/svend/access.log \
    --error-logfile /var/log/svend/error.log \
    --capture-output
