#!/bin/bash
# Encrypted database backup for Svend
#
# Creates a gzip-compressed, AES-256 encrypted pg_dump.
# Retention: 30 days. Run daily via systemd timer.
#
# Usage:
#   ./backup_db.sh              # Create backup
#   ./backup_db.sh --restore <file>  # Restore from backup

set -euo pipefail

BACKUP_DIR="$HOME/backups/svend"
DB_NAME="svend"
DB_USER="svend"
RETENTION_DAYS=30
KEYFILE="$HOME/.svend_encryption_key"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/svend_${TIMESTAMP}.sql.gz.enc"

mkdir -p "$BACKUP_DIR"

# ── Restore mode ──────────────────────────────────────────────────────
if [[ "${1:-}" == "--restore" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "Usage: $0 --restore <backup_file>"
        exit 1
    fi

    RESTORE_FILE="$2"
    if [[ ! -f "$RESTORE_FILE" ]]; then
        echo "Error: File not found: $RESTORE_FILE"
        exit 1
    fi

    echo "Restoring from: $RESTORE_FILE"
    echo "WARNING: This will overwrite the current database!"
    read -p "Continue? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Aborted."
        exit 0
    fi

    openssl enc -aes-256-cbc -d -pbkdf2 \
        -pass "file:${KEYFILE}" \
        -in "$RESTORE_FILE" \
    | gunzip \
    | psql -U "$DB_USER" -d "$DB_NAME"

    echo "Restore complete."
    exit 0
fi

# ── Backup mode ───────────────────────────────────────────────────────
if [[ ! -f "$KEYFILE" ]]; then
    echo "Error: Encryption key not found at $KEYFILE"
    exit 1
fi

echo "Backing up database: $DB_NAME"

pg_dump -U "$DB_USER" "$DB_NAME" \
    | gzip \
    | openssl enc -aes-256-cbc -pbkdf2 \
        -pass "file:${KEYFILE}" \
        -out "$BACKUP_FILE"

# Verify the backup is non-empty
if [[ ! -s "$BACKUP_FILE" ]]; then
    echo "Error: Backup file is empty"
    rm -f "$BACKUP_FILE"
    exit 1
fi

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup created: $BACKUP_FILE ($SIZE)"

# ── Prune old backups ─────────────────────────────────────────────────
PRUNED=$(find "$BACKUP_DIR" -name "svend_*.sql.gz.enc" -mtime +${RETENTION_DAYS} -delete -print | wc -l)
if [[ "$PRUNED" -gt 0 ]]; then
    echo "Pruned $PRUNED backup(s) older than ${RETENTION_DAYS} days"
fi

echo "Done."
