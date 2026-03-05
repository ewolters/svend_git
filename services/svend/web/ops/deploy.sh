#!/bin/bash
# ops/deploy.sh — Automated deployment for Svend
# FEAT-080 | INIT-010 | OPS-001 compliant
#
# Usage:
#   ./ops/deploy.sh --cr <uuid>     # Standard deploy linked to ChangeRequest
#   ./ops/deploy.sh                  # Quick deploy (warns about missing CR)
#   ./ops/deploy.sh --dry-run        # Pre-flight + show what would change
#
# Prerequisites (one-time):
#   sudo visudo -f /etc/sudoers.d/svend-deploy
#   eric ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart svend, /usr/bin/systemctl restart tempora

set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────
DEPLOY_DIR="/home/eric/kjerne/services/svend/web"
OPS_DIR="${DEPLOY_DIR}/ops"
HEALTH_URL="http://127.0.0.1:8000/api/health/"
BRANCH="main"
DEPLOY_START=$(date +%s)
PREV_COMMIT=""
NEW_COMMIT=""
NO_CHANGES=false
ROLLBACK_DONE=false
DEPLOY_STARTED=false

# ── Color helpers ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
step()  { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Parse arguments ──────────────────────────────────────────────────
CR_ID=""
DRY_RUN=false

usage() {
    echo "Usage: $0 [--cr <uuid>] [--dry-run] [--help]"
    echo ""
    echo "Options:"
    echo "  --cr <uuid>   Link deployment to a ChangeRequest"
    echo "  --dry-run     Pre-flight checks only, show what would change"
    echo "  --help        Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --cr)     CR_ID="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help)   usage; exit 0 ;;
        *)        fail "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

# ── Rollback ─────────────────────────────────────────────────────────
rollback() {
    if [[ "$ROLLBACK_DONE" == "true" ]]; then return; fi
    ROLLBACK_DONE=true

    fail "DEPLOYMENT FAILED — initiating rollback"
    cd "$DEPLOY_DIR"

    if [[ -z "$PREV_COMMIT" ]]; then
        fail "No previous commit recorded — cannot rollback"
        return 1
    fi

    step "Rolling back to ${PREV_COMMIT:0:12}"

    git checkout "$BRANCH" 2>/dev/null || true
    git reset --hard "$PREV_COMMIT" 2>/dev/null || true
    warn "Reset to ${PREV_COMMIT:0:12}"

    info "Re-running migrations..."
    python3 manage.py migrate --noinput 2>/dev/null || warn "Migration rollback had issues"

    info "Re-collecting static files..."
    python3 manage.py collectstatic --noinput 2>/dev/null || warn "Collectstatic during rollback had issues"

    info "Restarting services..."
    sudo systemctl restart svend 2>/dev/null || warn "Failed to restart svend"
    sudo systemctl restart tempora 2>/dev/null || warn "Failed to restart tempora"

    sleep 5

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")
    if [[ "$HTTP_CODE" == "200" ]]; then
        warn "Rollback successful — service healthy at ${PREV_COMMIT:0:12}"
    else
        fail "ROLLBACK FAILED — service unhealthy. Manual intervention required."
        fail "Previous commit: $PREV_COMMIT"
        fail "Check: sudo journalctl -u svend.service -n 50"
    fi

    # Log rollback via ORM
    if [[ -n "$CR_ID" ]]; then
        python3 manage.py shell -c "
from syn.audit.models import ChangeLog, ChangeRequest
try:
    cr = ChangeRequest.objects.get(id='${CR_ID}')
    ChangeLog.objects.create(
        change_request=cr, actor='ops/deploy.sh', action='rolled_back',
        details={'rolled_back_to': '${PREV_COMMIT}', 'failed_commit': '${NEW_COMMIT}'},
        message='Automated rollback after failed deployment',
    )
except Exception:
    pass
" 2>/dev/null || true
    fi

    fail "Rollback complete. Investigate before re-deploying."
}

# ── Cleanup trap ─────────────────────────────────────────────────────
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]] && [[ "$DEPLOY_STARTED" == "true" ]]; then
        rollback
    fi

    local total=$(( $(date +%s) - DEPLOY_START ))
    echo ""
    if [[ $exit_code -eq 0 ]]; then
        ok "Deployment completed in ${total}s"
    else
        fail "Deployment failed after ${total}s"
    fi
}
trap cleanup EXIT

# ══════════════════════════════════════════════════════════════════════
# Step 1: Pre-flight checks
# ══════════════════════════════════════════════════════════════════════
step "Pre-flight checks"

cd "$DEPLOY_DIR"

if [[ $EUID -eq 0 ]]; then
    fail "Do not run as root (OPS-001 §13.1)"
    exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    fail "Not on $BRANCH branch (on: $CURRENT_BRANCH)"
    exit 1
fi

if ! git diff-index --quiet HEAD --; then
    fail "Uncommitted changes detected. Commit or stash first."
    exit 1
fi

PREV_COMMIT=$(git rev-parse HEAD)
info "Current commit: ${PREV_COMMIT:0:12}"

if [[ ! -x "$OPS_DIR/backup_db.sh" ]]; then
    fail "backup_db.sh not found or not executable"
    exit 1
fi

if [[ ! -f "$HOME/.svend_encryption_key" ]]; then
    fail "Encryption key not found (OPS-001 §11.1)"
    exit 1
fi

if [[ -n "$CR_ID" ]]; then
    info "Linked to CR: $CR_ID"
else
    warn "No --cr provided — deployment will not be linked to a ChangeRequest"
fi

ok "Pre-flight passed"

# ══════════════════════════════════════════════════════════════════════
# Step 2: Backup database
# ══════════════════════════════════════════════════════════════════════
step "Database backup"

if [[ "$DRY_RUN" == "true" ]]; then
    info "[DRY RUN] Would run backup_db.sh"
else
    "$OPS_DIR/backup_db.sh"
    ok "Backup complete"
fi

# ══════════════════════════════════════════════════════════════════════
# Step 3: Git pull
# ══════════════════════════════════════════════════════════════════════
step "Pulling latest from origin/$BRANCH"

git fetch origin "$BRANCH"

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [[ "$LOCAL" == "$REMOTE" ]]; then
    warn "Already up to date"
    NO_CHANGES=true
    NEW_COMMIT="$LOCAL"
else
    info "$(git log --oneline "${LOCAL}..${REMOTE}" | wc -l) new commit(s):"
    git log --oneline "${LOCAL}..${REMOTE}"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    info "[DRY RUN] Would pull $(git log --oneline "${LOCAL}..${REMOTE}" 2>/dev/null | wc -l) commit(s)"
    echo ""
    ok "Dry run complete — no changes made"
    exit 0
fi

# From here on, changes are being made — enable rollback
DEPLOY_STARTED=true

git pull --ff-only origin "$BRANCH"
NEW_COMMIT=$(git rev-parse HEAD)
info "Now at: ${NEW_COMMIT:0:12}"

ok "Pull complete"

# ══════════════════════════════════════════════════════════════════════
# Step 4: Pip install (conditional)
# ══════════════════════════════════════════════════════════════════════
step "Checking dependencies"

if [[ "$NO_CHANGES" == "true" ]]; then
    info "No changes — skipping pip install"
elif git diff "${PREV_COMMIT}..${NEW_COMMIT}" --name-only | grep -qE '(pyproject\.toml|requirements|setup\.cfg)'; then
    info "Dependency files changed — installing"
    pip install -e ".[prod]"
    ok "Dependencies updated"
else
    info "No dependency changes — skipping"
fi

# ══════════════════════════════════════════════════════════════════════
# Step 5: Migrate
# ══════════════════════════════════════════════════════════════════════
step "Database migrations"

if python3 manage.py migrate --check 2>/dev/null; then
    info "No pending migrations"
else
    info "Pending migrations — applying"
    python3 manage.py migrate --noinput
    ok "Migrations applied"
fi

# ══════════════════════════════════════════════════════════════════════
# Step 6: Collectstatic (OPS-001 §13.5 — never skip)
# ══════════════════════════════════════════════════════════════════════
step "Collecting static files"

python3 manage.py collectstatic --noinput --verbosity 0
ok "Static files collected"

# ══════════════════════════════════════════════════════════════════════
# Step 7: Django deploy check
# ══════════════════════════════════════════════════════════════════════
step "Django deploy checks"

python3 manage.py check --deploy 2>&1 || warn "Deploy check reported warnings (non-blocking)"
ok "Deploy checks complete"

# ══════════════════════════════════════════════════════════════════════
# Step 8: Restart services
# ══════════════════════════════════════════════════════════════════════
step "Restarting services"

sudo systemctl restart svend
info "svend.service restarted"

sudo systemctl restart tempora
info "tempora.service restarted"

ok "Services restarted"

# ══════════════════════════════════════════════════════════════════════
# Step 9: Health check
# ══════════════════════════════════════════════════════════════════════
step "Health check"

sleep 5

HEALTH_ATTEMPTS=3
HEALTH_OK=false

for i in $(seq 1 $HEALTH_ATTEMPTS); do
    HTTP_CODE=$(curl -s -o /tmp/svend_health_response -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")

    if [[ "$HTTP_CODE" == "200" ]]; then
        RESPONSE=$(cat /tmp/svend_health_response)
        if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
            HEALTH_OK=true
            break
        else
            warn "Health returned 200 but unexpected body: $RESPONSE"
        fi
    else
        warn "Health check attempt $i/$HEALTH_ATTEMPTS returned HTTP $HTTP_CODE"
    fi

    if [[ $i -lt $HEALTH_ATTEMPTS ]]; then
        sleep 3
    fi
done

rm -f /tmp/svend_health_response

if [[ "$HEALTH_OK" != "true" ]]; then
    fail "Health check failed after $HEALTH_ATTEMPTS attempts"
    rollback
    exit 1
fi

ok "Health check passed (HTTP 200, status=ok)"

# ══════════════════════════════════════════════════════════════════════
# Step 10: Log deployment
# ══════════════════════════════════════════════════════════════════════
step "Logging deployment"

DEPLOY_END=$(date +%s)
DEPLOY_DURATION=$((DEPLOY_END - DEPLOY_START))

if [[ -n "$CR_ID" ]]; then
    python3 manage.py shell -c "
from syn.audit.models import ChangeLog, ChangeRequest
cr = ChangeRequest.objects.get(id='${CR_ID}')
ChangeLog.objects.create(
    change_request=cr, actor='ops/deploy.sh', action='completed',
    from_state=cr.status, to_state='completed',
    details={
        'deploy_type': 'automated', 'prev_commit': '${PREV_COMMIT}',
        'new_commit': '${NEW_COMMIT}', 'duration_seconds': ${DEPLOY_DURATION},
    },
    message='Deployed ${PREV_COMMIT:0:12}..${NEW_COMMIT:0:12} (${DEPLOY_DURATION}s)',
)
shas = cr.commit_shas or []
if '${NEW_COMMIT}' not in shas:
    shas.append('${NEW_COMMIT}')
    cr.commit_shas = shas
    cr.save(update_fields=['commit_shas'])
print('Logged to CR ${CR_ID}')
" 2>/dev/null || warn "Failed to log deployment to CR"
    ok "Deployment logged to CR $CR_ID"
else
    warn "No --cr provided — deployment not linked to a ChangeRequest"
    info "To link retroactively: update CR commit_shas via dashboard"
fi

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}── Deploy Summary ──${NC}"
echo "  Previous:  ${PREV_COMMIT:0:12}"
echo "  Deployed:  ${NEW_COMMIT:0:12}"
if [[ "$PREV_COMMIT" != "$NEW_COMMIT" ]]; then
    COMMIT_COUNT=$(git log --oneline "${PREV_COMMIT}..${NEW_COMMIT}" | wc -l)
    echo "  Commits:   $COMMIT_COUNT"
fi
echo "  Duration:  ${DEPLOY_DURATION}s"
echo "  CR:        ${CR_ID:-none}"
echo "  Health:    OK"
