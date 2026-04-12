#!/bin/bash
# Encrypted database backup for Svend
#
# Creates a gzip-compressed, AES-256 encrypted pg_dump.
# Retention: 30 days. Run daily via systemd timer.
# Off-site: Pushes encrypted backup to Backblaze B2 (if configured).
#
# Usage:
#   ./backup_db.sh              # Create backup
#   ./backup_db.sh --restore <file>  # Restore from backup

set -euo pipefail

BACKUP_DIR="$HOME/backups/svend"
DB_NAME="svend"
DB_USER="svend"
DB_HOST="127.0.0.1"
RETENTION_DAYS=30
KEYFILE="$HOME/.svend_encryption_key"
PGPASSFILE="$HOME/.pgpass"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/svend_${TIMESTAMP}.sql.gz.enc"

# Backblaze B2 off-site backup (set B2_BUCKET to enable)
B2_BUCKET="${B2_BUCKET:-}"

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
    | psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME"

    echo "Restore complete."
    exit 0
fi

# ── Backup mode ───────────────────────────────────────────────────────
if [[ ! -f "$KEYFILE" ]]; then
    echo "Error: Encryption key not found at $KEYFILE"
    exit 1
fi

echo "Backing up database: $DB_NAME"

pg_dump -h "$DB_HOST" -U "$DB_USER" "$DB_NAME" \
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

# ── Push to Backblaze B2 ──────────────────────────────────────────────
if [[ -n "$B2_BUCKET" ]] && command -v b2 &>/dev/null; then
    echo "Uploading to B2: $B2_BUCKET"
    FILENAME=$(basename "$BACKUP_FILE")
    if b2 upload-file "$B2_BUCKET" "$BACKUP_FILE" "$FILENAME"; then
        echo "B2 upload complete: $FILENAME"
    else
        echo "WARNING: B2 upload failed — local backup is safe"
    fi

    # B2 retention: use bucket lifecycle rules (set once during setup)
    # b2 update-bucket --lifecycleRules '[{"daysFromUploadingToHiding":30,"daysFromHidingToDeleting":1,"fileNamePrefix":"svend_"}]' $B2_BUCKET
    # No script-side pruning needed — B2 handles it automatically
elif [[ -n "$B2_BUCKET" ]]; then
    echo "WARNING: B2_BUCKET set but b2 CLI not found — skipping off-site backup"
fi

# ── Prune old local backups ───────────────────────────────────────────
PRUNED=$(find "$BACKUP_DIR" -name "svend_*.sql.gz.enc" -mtime +${RETENTION_DAYS} -delete -print | wc -l)
if [[ "$PRUNED" -gt 0 ]]; then
    echo "Pruned $PRUNED local backup(s) older than ${RETENTION_DAYS} days"
fi

echo "Done."
