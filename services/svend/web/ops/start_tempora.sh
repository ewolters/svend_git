#!/bin/bash
# Tempora scheduler startup script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Always cd to the web root (one level up from ops/)
WEB_ROOT="${SCRIPT_DIR%/ops}"
cd "$WEB_ROOT"

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

echo "Starting Tempora scheduler (single-node)..."
exec python3 manage.py tempora_server --single-node --workers 2
