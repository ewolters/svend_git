# Business Continuity and Disaster Recovery Plan

**Policy ID:** BCDR-001
**Version:** 1.0
**Effective Date:** 2026-03-03
**Owner:** Eric (Founder)
**Review Cycle:** Annual
**Parent Policy:** [Information Security Policy](information-security.md)

---

## 1. Purpose

Ensure the Svend platform can recover from disruptions — from minor service interruptions to catastrophic data loss — with defined recovery objectives and tested procedures.

## 2. Recovery Objectives

| Metric | Target | Current Capability |
|---|---|---|
| **RTO** (Recovery Time Objective) | < 4 hours | ~1-2 hours (service restart + backup restore) |
| **RPO** (Recovery Point Objective) | < 24 hours | 24 hours (daily backups) |
| **MTTR** (Mean Time to Repair) | < 1 hour for service issues | ~15 minutes for simple restarts |

## 3. Backup Strategy

### 3.1 Database Backups

| Parameter | Value |
|---|---|
| Tool | `pg_dump` via `backup_db.sh` |
| Schedule | Daily (systemd timer: `svend-backup.timer`) |
| Encryption | AES-256-CBC with PBKDF2 key derivation |
| Storage | `/home/eric/backups/svend/` |
| Retention | 30 days (automated pruning) |
| Off-site | **Gap** — not yet implemented (target: Backblaze B2) |

### 3.2 Source Code

| Parameter | Value |
|---|---|
| Primary | Git repository on server |
| Remote | GitHub (private repository) |
| Frequency | Push on every deployment |

### 3.3 File Storage (User Uploads)

| Parameter | Value |
|---|---|
| Storage | Local filesystem with `EncryptedFileSystemStorage` |
| Backup | Not separately backed up (gap — include in daily backup) |
| Off-site | Not yet implemented |

### 3.4 Configuration

| Parameter | Value |
|---|---|
| Non-secret config | In git repository |
| Secrets (env vars) | Documented locations; manual backup |
| Encryption key | `~/.svend_encryption_key` — manual secure backup |
| SSL certificates | Auto-provisioned by Caddy/Let's Encrypt |

## 4. Disaster Scenarios and Recovery Procedures

### Scenario 1: Application Crash / Hang

**Symptoms:** 502/503 errors; no response from application
**RTO:** < 5 minutes

1. Check Gunicorn status: `systemctl status svend`
2. Restart service: `sudo systemctl restart svend`
3. Check logs: `tail -100 /var/log/svend/error.log`
4. If restart fails, check disk space, memory, database connectivity
5. Log incident

### Scenario 2: Database Corruption / Failure

**Symptoms:** 500 errors with database-related tracebacks
**RTO:** < 1 hour

1. Check PostgreSQL status: `systemctl status postgresql`
2. Attempt restart: `sudo systemctl restart postgresql`
3. If corrupt, stop application: `sudo systemctl stop svend`
4. Restore from latest backup:
   ```bash
   # Decrypt backup
   openssl enc -aes-256-cbc -d -pbkdf2 \
     -in /home/eric/backups/svend/LATEST.sql.gz.enc \
     -out /tmp/restore.sql.gz
   gunzip /tmp/restore.sql.gz

   # Restore
   psql -U svend -d svend_db < /tmp/restore.sql

   # Clean up
   rm /tmp/restore.sql
   ```
5. Restart application: `sudo systemctl restart svend`
6. Verify data integrity
7. Log incident with data loss assessment (RPO = time since last backup)

### Scenario 3: Server Disk Failure

**Symptoms:** I/O errors; read-only filesystem; service failures
**RTO:** < 4 hours (new server provisioning)

1. Provision new server (or replace disk)
2. Install Ubuntu 22.04, PostgreSQL, Python, Caddy
3. Clone repository from GitHub
4. Restore environment variables and encryption key (from secure backup)
5. Restore database from off-site backup (when available) or local backup
6. Configure systemd services, Caddy, fail2ban
7. Update Cloudflare Tunnel to point to new server
8. Verify all services operational

### Scenario 4: Encryption Key Compromise

**Symptoms:** Suspected unauthorized access to `~/.svend_encryption_key`
**RTO:** < 2 hours

1. Generate new encryption key
2. Run re-encryption migration for all encrypted fields
3. Re-encrypt user uploads
4. Generate new encrypted backups (old backups remain readable with old key)
5. Rotate all API keys and secrets
6. Invalidate all user sessions
7. Document incident

### Scenario 5: Cloudflare Tunnel Failure

**Symptoms:** Site unreachable from internet; server responds locally
**RTO:** < 30 minutes

1. Check Cloudflare status page
2. Verify tunnel service: `systemctl status cloudflared`
3. Restart tunnel if needed
4. If Cloudflare outage: wait (no bypass — Cloudflare is the only ingress)
5. Consider: emergency direct Caddy HTTPS exposure (last resort, temporary)

### Scenario 6: Complete Data Loss (Catastrophic)

**Symptoms:** Both server and backups destroyed
**RTO:** < 8 hours | **RPO:** Last off-site backup or git push

1. Provision new server
2. Clone from GitHub (source code restored)
3. Restore from off-site backup (when available)
4. If no off-site backup: database is lost — customers notified
5. Restore encryption key from secure backup
6. Full rebuild of infrastructure

**Mitigation:** Off-site backup implementation is the top BCDR priority.

## 5. Off-Site Backup Roadmap

| Phase | Action | Status |
|---|---|---|
| 1 | Evaluate Backblaze B2 vs AWS S3 for encrypted backup storage | Not started |
| 2 | Add `b2` CLI sync to `backup_db.sh` | Not started |
| 3 | Include user upload files in backup set | Not started |
| 4 | Test full restore from off-site backup | Not started |
| 5 | Document restore procedure for off-site source | Not started |

## 6. Testing Schedule

| Test | Frequency | Last Tested | Next Due |
|---|---|---|---|
| Backup restore (local) | Quarterly | Not tested | Overdue |
| Backup restore (off-site) | Quarterly | N/A | After off-site implemented |
| Service restart recovery | Monthly | Ongoing (incidental) | N/A |
| Full disaster recovery drill | Annual | Not tested | Overdue |
| Encryption key rotation | Annual | Not tested | Overdue |

## 7. Dependencies

| Component | Single Point of Failure? | Mitigation |
|---|---|---|
| Server hardware | Yes | Off-site backups + documented rebuild procedure |
| PostgreSQL | Yes | Encrypted daily backups |
| Encryption key | Yes | Secure off-site copy (manual) |
| Cloudflare | Yes (for external access) | Accept — Cloudflare has 99.99% SLA |
| GitHub | No (local git is primary) | Local + remote copies |
| Eric (operator) | Yes | Document all procedures (this file); AI collaborator can guide recovery |
