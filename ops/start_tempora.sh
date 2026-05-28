#!/bin/bash
# Tempora scheduler startup script
#
# DEPRECATED (pending sunset, 2026-05-28): superseded by ~/tempora-service/
# (`manage.py run_scheduler`). This script targets the old in-repo scheduler in
# the sunset ~/kjerne tree and is no longer used. See syn/sched/DEPRECATED.md.

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
