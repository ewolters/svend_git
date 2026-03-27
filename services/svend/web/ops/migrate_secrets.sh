#!/bin/bash
# migrate_secrets.sh — Move secrets from project .env to /etc/svend/env
#
# What this does:
#   1. Creates /etc/svend/ (root-owned)
#   2. Copies .env → /etc/svend/env with root:eric 640 permissions
#   3. Moves .env → .env.migrated (backup, not deleted)
#   4. Updates systemd to use EnvironmentFile=/etc/svend/env
#
# Why:
#   .env in the project dir means any directory traversal, accidental
#   git add, or backup tool that copies the project tree leaks everything.
#   /etc/svend/env is outside the project, root-owned, group-readable by
#   eric only (for the systemd service running as User=eric).
#
# Run as: sudo bash ops/migrate_secrets.sh
# Rollback: sudo cp /etc/svend/env .env && sudo chown eric:eric .env

set -euo pipefail

WEB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SRC="${WEB_ROOT}/.env"
ENV_PROD_SRC="${WEB_ROOT}/.env.production"
DEST_DIR="/etc/svend"
DEST_FILE="${DEST_DIR}/env"

# ── Preflight ────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run with sudo"
    exit 1
fi

if [[ ! -f "$ENV_SRC" ]]; then
    echo "ERROR: No .env found at $ENV_SRC"
    exit 1
fi

# ── Create destination ───────────────────────────────────────────────
echo "[1/5] Creating $DEST_DIR"
mkdir -p "$DEST_DIR"
chmod 750 "$DEST_DIR"
chown root:eric "$DEST_DIR"

# ── Copy secrets ─────────────────────────────────────────────────────
echo "[2/5] Copying .env → $DEST_FILE"
cp "$ENV_SRC" "$DEST_FILE"

# If .env.production has additional vars, append them (skip duplicates)
if [[ -f "$ENV_PROD_SRC" ]]; then
    echo "       Merging .env.production overrides..."
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        # Extract key
        key="${line%%=*}"
        # Only add if not already present
        if ! grep -q "^${key}=" "$DEST_FILE" 2>/dev/null; then
            echo "$line" >> "$DEST_FILE"
        fi
    done < "$ENV_PROD_SRC"
fi

# ── Lock down permissions ────────────────────────────────────────────
echo "[3/5] Setting permissions: root:eric 640"
chown root:eric "$DEST_FILE"
chmod 640 "$DEST_FILE"

# ── Move originals out of project dir ────────────────────────────────
echo "[4/5] Archiving original .env files"
if [[ -f "$ENV_SRC" ]]; then
    mv "$ENV_SRC" "${ENV_SRC}.migrated"
    echo "       .env → .env.migrated"
fi
if [[ -f "$ENV_PROD_SRC" ]]; then
    mv "$ENV_PROD_SRC" "${ENV_PROD_SRC}.migrated"
    echo "       .env.production → .env.production.migrated"
fi

# ── Reload systemd ───────────────────────────────────────────────────
echo "[5/5] Reloading systemd daemon"
systemctl daemon-reload

echo ""
echo "Done. Secrets are now at: $DEST_FILE"
echo ""
echo "Next steps:"
echo "  1. Restart svend:    sudo systemctl restart svend"
echo "  2. Verify it works:  curl -s http://127.0.0.1:8000/api/health/"
echo "  3. Delete backups:   sudo rm ${ENV_SRC}.migrated ${ENV_PROD_SRC}.migrated"
echo ""
echo "Rollback if broken:"
echo "  sudo cp $DEST_FILE ${ENV_SRC} && sudo chown eric:eric ${ENV_SRC}"
echo "  sudo systemctl restart svend"
