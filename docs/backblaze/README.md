# Backblaze B2 — Off-Site Backup

**Purpose:** Encrypted off-site database backups for disaster recovery.
**Set up:** 2026-03-03
**Cost:** $0/mo (free tier: 10 GB storage, 1 GB/day download)

---

## Account

- **Provider:** Backblaze B2 Cloud Storage
- **Dashboard:** https://secure.backblaze.com/b2_buckets.htm
- **Account ID:** `22d8c24a3745`
- **Region:** US East (us-east-005)

## Bucket

- **Name:** `svend-backups`
- **Bucket ID:** `1292bd285cd2b44a93c70415`
- **Type:** Private (allPrivate)
- **Encryption:** None (files are pre-encrypted with AES-256-CBC before upload)
- **Lifecycle:** Files auto-delete 30 days after upload (`svend_` prefix rule)

## Application Key

- **Key Name:** `svend-backup`
- **Key ID:** `00522d8c24a37450000000001`
- **Scoped to:** `svend-backups` bucket only
- **Capabilities:** listBuckets, listFiles, readFiles, writeFiles, deleteFiles

Key is stored in the B2 CLI config at `~/.config/b2/account_info`.

## How It Works

```
pg_dump → gzip → openssl AES-256-CBC → local file → b2 upload → Backblaze B2
```

1. `backup_db.sh` runs daily at 3 AM via systemd timer (`svend-backup.timer`)
2. Creates an encrypted dump: `~/backups/svend/svend_YYYYMMDD_HHMMSS.sql.gz.enc`
3. If `B2_BUCKET` env var is set, uploads the encrypted file to B2
4. B2 lifecycle rule auto-deletes files older than 30 days
5. Local pruning also deletes local copies older than 30 days

## Files on Server

| File | Purpose |
|------|---------|
| `~/kjerne/services/svend/web/ops/backup_db.sh` | Backup script (backup + restore + B2 upload) |
| `~/kjerne/services/svend/web/ops/svend-backup.service` | systemd one-shot service |
| `~/kjerne/services/svend/web/ops/svend-backup.timer` | systemd daily timer (3 AM) |
| `~/.svend_b2_env` | `B2_BUCKET=svend-backups` (read by systemd EnvironmentFile) |
| `~/.svend_encryption_key` | Fernet key used for AES-256-CBC encryption |
| `~/.pgpass` | PostgreSQL password file (`127.0.0.1:5432:svend:svend:<password>`) |
| `~/.config/b2/account_info` | B2 CLI auth token (created by `b2 authorize-account`) |
| `~/backups/svend/` | Local backup directory |

## Common Commands

### Check backup status
```bash
journalctl -u svend-backup -n 30
```

### List files in B2
```bash
b2 ls svend-backups
```

### Manual backup + upload
```bash
B2_BUCKET=svend-backups ~/kjerne/services/svend/web/ops/backup_db.sh
```

### Download a backup from B2
```bash
b2 download-file-by-name svend-backups svend_20260303_104632.sql.gz.enc ./restore.sql.gz.enc
```

### Restore from a downloaded backup
```bash
~/kjerne/services/svend/web/ops/backup_db.sh --restore ./restore.sql.gz.enc
```

### Re-authorize B2 CLI (if token expires)
```bash
b2 authorize-account <keyID> <applicationKey>
```
Key ID and application key are in the Backblaze dashboard under **App Keys**.

### Check bucket lifecycle rules
```bash
b2 get-bucket svend-backups
```

## Disaster Recovery Scenario

If the server is lost:

1. Install PostgreSQL, create `svend` database and user
2. Install b2 CLI: `pip3 install b2`
3. Authorize: `b2 authorize-account <keyID> <applicationKey>`
4. Download latest backup: `b2 ls svend-backups` then `b2 download-file-by-name ...`
5. Copy encryption key (`~/.svend_encryption_key`) from secure backup
6. Restore: `./backup_db.sh --restore <file>`

**Critical dependency:** The encryption key (`~/.svend_encryption_key`) is required to decrypt backups. If this key is lost, backups are unrecoverable. Store a copy somewhere safe outside the server.
