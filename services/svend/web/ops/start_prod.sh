#!/bin/bash
# Production startup script for Svend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Always cd to the web root (one level up from ops/)
WEB_ROOT="${SCRIPT_DIR%/ops}"
cd "$WEB_ROOT"

export PATH="$HOME/.local/bin:$PATH"

# Secrets loaded via systemd EnvironmentFile=/etc/svend/env
# Fallback: source .env if running outside systemd (e.g., dev/debug)
if [ -z "${SVEND_SECRET_KEY:-}" ] && [ -f .env ]; then
    echo "WARNING: Loading .env fallback — secrets should be in /etc/svend/env"
    set -a
    source .env
    set +a
fi

# Load field-level encryption key from separate keyfile if not already set
KEYFILE="$HOME/.svend_encryption_key"
if [ -z "${SVEND_FIELD_ENCRYPTION_KEY:-}" ] && [ -f "$KEYFILE" ]; then
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
