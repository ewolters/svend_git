#!/bin/bash
# harden.sh — Server hardening for solo-operated Svend production server
#
# What this does:
#   1. SSH: key-only auth, no root login, AllowUsers eric
#   2. UFW: deny all inbound, allow SSH + localhost services
#   3. Fail2ban: deploy Svend jail config
#   4. Kernel: basic sysctl hardening
#   5. Cleanup: shred stale unencrypted DB backups
#   6. File integrity: install baseline + daily checker
#
# Run as: sudo bash ops/harden.sh
#
# IMPORTANT: Make sure you have SSH key access BEFORE running this.
#            Password auth will be disabled. If you lock yourself out,
#            you'll need console access from your VPS provider.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/ops}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[HARDEN]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ $EUID -ne 0 ]]; then
    err "Run with sudo"
    exit 1
fi

# ── Preflight: verify SSH key exists ─────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Svend Server Hardening"
echo "═══════════════════════════════════════════════════════════════"
echo ""

SSH_KEYS=$(find /home/eric/.ssh/ -name "authorized_keys" -exec cat {} \; 2>/dev/null | grep -c "^ssh-" || true)
if [[ "$SSH_KEYS" -eq 0 ]]; then
    err "No SSH public keys found in /home/eric/.ssh/authorized_keys"
    err "Add your key first: ssh-copy-id eric@<this-server>"
    err "Aborting — password auth would be disabled with no key access."
    exit 1
fi
log "Found $SSH_KEYS SSH key(s) — safe to disable password auth"

# ── 1. SSH Hardening ─────────────────────────────────────────────────
log ""
log "═══ [1/6] SSH Hardening ═══"

SSHD_CONFIG="/etc/ssh/sshd_config"
SSHD_BACKUP="/etc/ssh/sshd_config.bak.$(date +%Y%m%d)"

cp "$SSHD_CONFIG" "$SSHD_BACKUP"
log "Backed up sshd_config → $SSHD_BACKUP"

# Write hardened sshd config drop-in (cleaner than sed)
cat > /etc/ssh/sshd_config.d/99-svend-hardening.conf << 'SSHEOF'
# Svend server hardening — applied by harden.sh
# To revert: sudo rm /etc/ssh/sshd_config.d/99-svend-hardening.conf

# Key-only authentication
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes

# No root login
PermitRootLogin no

# Only eric can SSH in
AllowUsers eric

# Brute-force mitigation
MaxAuthTries 3
LoginGraceTime 30

# Disable unused features
X11Forwarding no
AllowTcpForwarding no
AllowAgentForwarding no
PermitTunnel no

# Logging
LogLevel VERBOSE
SSHEOF

# Test config before reloading
if sshd -t 2>/dev/null; then
    systemctl reload sshd
    log "SSH hardened and reloaded"
else
    err "sshd config test failed — reverting"
    rm /etc/ssh/sshd_config.d/99-svend-hardening.conf
    exit 1
fi

# ── 2. UFW Firewall ──────────────────────────────────────────────────
log ""
log "═══ [2/6] UFW Firewall ═══"

# Reset quietly, set defaults
ufw --force reset > /dev/null 2>&1
ufw default deny incoming > /dev/null
ufw default allow outgoing > /dev/null

# SSH — rate limited
ufw limit 22/tcp comment "SSH (rate limited)"

# Cloudflare Tunnel connects outbound, so no inbound HTTP rules needed.
# Port 80 listener is cloudflared — if it needs inbound, it's localhost.

ufw --force enable
log "UFW enabled: deny all inbound, allow SSH (rate-limited)"
ufw status verbose

# ── 3. Fail2ban ──────────────────────────────────────────────────────
log ""
log "═══ [3/6] Fail2ban ═══"

cp "${SCRIPT_DIR}/fail2ban-svend.conf" /etc/fail2ban/jail.local
# Update banaction to use iptables since UFW is now the firewall
sed -i 's/banaction = ufw/banaction = iptables-multiport/' /etc/fail2ban/jail.local

systemctl enable fail2ban
systemctl restart fail2ban
log "Fail2ban deployed with SSH jail (3 tries → 24h ban)"
fail2ban-client status sshd 2>/dev/null || warn "fail2ban sshd jail status check failed — verify manually"

# ── 4. Kernel Hardening (sysctl) ─────────────────────────────────────
log ""
log "═══ [4/6] Kernel Hardening ═══"

cat > /etc/sysctl.d/99-svend-hardening.conf << 'SYSEOF'
# Svend server hardening — applied by harden.sh

# Prevent IP spoofing
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP redirects (MITM prevention)
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv6.conf.all.accept_redirects = 0

# Ignore broadcast pings (smurf attack prevention)
net.ipv4.icmp_echo_ignore_broadcasts = 1

# SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048

# Log suspicious packets
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Disable IPv4 forwarding (not a router)
net.ipv4.ip_forward = 0

# Restrict kernel pointer exposure
kernel.kptr_restrict = 2

# Restrict dmesg access
kernel.dmesg_restrict = 1

# Restrict ptrace (only parent can trace children)
kernel.yama.ptrace_scope = 1
SYSEOF

sysctl --system > /dev/null 2>&1
log "Kernel parameters hardened"

# ── 5. Cleanup Stale Backups ─────────────────────────────────────────
log ""
log "═══ [5/6] Cleanup Stale Unencrypted Backups ═══"

STALE_COUNT=0
for f in "${REPO_ROOT}"/db_backup_iso_*.sql; do
    if [[ -f "$f" ]]; then
        SIZE=$(du -h "$f" | cut -f1)
        # Overwrite before deleting (basic shred)
        shred -n 1 -z "$f" 2>/dev/null || true
        rm -f "$f"
        log "Shredded: $(basename "$f") ($SIZE)"
        STALE_COUNT=$((STALE_COUNT + 1))
    fi
done

# Also check for unencrypted backup in backups dir
for f in /home/eric/backups/svend/*.sql.gz; do
    if [[ -f "$f" ]]; then
        SIZE=$(du -h "$f" | cut -f1)
        shred -n 1 -z "$f" 2>/dev/null || true
        rm -f "$f"
        log "Shredded unencrypted backup: $(basename "$f") ($SIZE)"
        STALE_COUNT=$((STALE_COUNT + 1))
    fi
done

if [[ "$STALE_COUNT" -eq 0 ]]; then
    log "No stale unencrypted backups found"
else
    log "Cleaned $STALE_COUNT stale backup(s)"
fi

# ── 6. File Integrity Baseline ───────────────────────────────────────
log ""
log "═══ [6/6] File Integrity Monitor ═══"

INTEGRITY_DIR="/var/lib/svend-integrity"
mkdir -p "$INTEGRITY_DIR"
chown root:root "$INTEGRITY_DIR"
chmod 700 "$INTEGRITY_DIR"

# Generate baseline hashes of critical files
BASELINE="${INTEGRITY_DIR}/baseline.sha256"
{
    # System config
    sha256sum /etc/ssh/sshd_config 2>/dev/null
    sha256sum /etc/ssh/sshd_config.d/99-svend-hardening.conf 2>/dev/null
    sha256sum /etc/fail2ban/jail.local 2>/dev/null
    sha256sum /etc/sysctl.d/99-svend-hardening.conf 2>/dev/null
    # Svend ops
    sha256sum "${SCRIPT_DIR}/start_prod.sh" 2>/dev/null
    sha256sum "${SCRIPT_DIR}/svend.service" 2>/dev/null
    sha256sum "${SCRIPT_DIR}/tempora.service" 2>/dev/null
    sha256sum "${SCRIPT_DIR}/backup_db.sh" 2>/dev/null
    # Django settings
    sha256sum "${REPO_ROOT}/svend/settings.py" 2>/dev/null
    sha256sum "${REPO_ROOT}/svend/urls.py" 2>/dev/null
    sha256sum "${REPO_ROOT}/svend_config/config.py" 2>/dev/null
    # Auth
    sha256sum "${REPO_ROOT}/accounts/permissions.py" 2>/dev/null
    sha256sum "${REPO_ROOT}/accounts/models.py" 2>/dev/null
    # Secrets config
    sha256sum /etc/svend/env 2>/dev/null
} > "$BASELINE" 2>/dev/null || true

chmod 600 "$BASELINE"
HASH_COUNT=$(wc -l < "$BASELINE")
log "Integrity baseline created: $HASH_COUNT files tracked"
log "Baseline at: $BASELINE"

# Install systemd timer for daily checks
cp "${SCRIPT_DIR}/integrity_check.sh" /usr/local/bin/svend-integrity-check
chmod 755 /usr/local/bin/svend-integrity-check

cp "${SCRIPT_DIR}/svend-integrity.service" /etc/systemd/system/
cp "${SCRIPT_DIR}/svend-integrity.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable svend-integrity.timer
systemctl start svend-integrity.timer
log "Integrity checker installed as daily systemd timer"

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Hardening Complete"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  [1] SSH:        Key-only, no root, AllowUsers eric, MaxAuth 3"
echo "  [2] UFW:        Deny all inbound, SSH rate-limited"
echo "  [3] Fail2ban:   SSH jail active (3 tries → 24h ban)"
echo "  [4] Kernel:     Anti-spoofing, SYN flood protection, restricted ptrace"
echo "  [5] Backups:    Stale unencrypted dumps shredded"
echo "  [6] Integrity:  Daily hash check of critical files"
echo ""
echo "  IMPORTANT: Test SSH in a NEW terminal before closing this one!"
echo "    ssh eric@<this-server>"
echo ""
echo "  If locked out, use VPS provider console to:"
echo "    sudo rm /etc/ssh/sshd_config.d/99-svend-hardening.conf"
echo "    sudo systemctl reload sshd"
echo ""
