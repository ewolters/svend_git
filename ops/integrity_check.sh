#!/bin/bash
# integrity_check.sh — Daily file integrity monitor for Svend
#
# Compares current SHA-256 hashes against baseline.
# Logs changes to /var/log/svend/integrity.log.
# Exits non-zero if any file changed (systemd will record failure).

set -euo pipefail

BASELINE="/var/lib/svend-integrity/baseline.sha256"
LOGFILE="/var/log/svend/integrity.log"
TIMESTAMP=$(date -Is)

mkdir -p "$(dirname "$LOGFILE")"

if [[ ! -f "$BASELINE" ]]; then
    echo "[$TIMESTAMP] ERROR: No baseline found at $BASELINE" >> "$LOGFILE"
    exit 1
fi

# Check all hashes
CHANGES=0
MISSING=0

while IFS="  " read -r expected_hash filepath; do
    if [[ ! -f "$filepath" ]]; then
        echo "[$TIMESTAMP] MISSING: $filepath" >> "$LOGFILE"
        MISSING=$((MISSING + 1))
        continue
    fi

    current_hash=$(sha256sum "$filepath" | awk '{print $1}')
    if [[ "$current_hash" != "$expected_hash" ]]; then
        echo "[$TIMESTAMP] CHANGED: $filepath (expected: ${expected_hash:0:16}... got: ${current_hash:0:16}...)" >> "$LOGFILE"
        CHANGES=$((CHANGES + 1))
    fi
done < "$BASELINE"

if [[ "$CHANGES" -gt 0 ]] || [[ "$MISSING" -gt 0 ]]; then
    echo "[$TIMESTAMP] ALERT: $CHANGES file(s) changed, $MISSING file(s) missing" >> "$LOGFILE"
    # Also write to syslog so fail2ban or monitoring can pick it up
    logger -t svend-integrity -p auth.warning "ALERT: $CHANGES file(s) changed, $MISSING missing — check $LOGFILE"
    exit 1
fi

echo "[$TIMESTAMP] OK: All files match baseline" >> "$LOGFILE"
exit 0
