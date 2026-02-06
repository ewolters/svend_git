#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kjerne Lab Management Tool

Mini-git for managing dev/prod separation with quality gates.

Usage:
    kjerne snapshot "message"      # Create snapshot
    kjerne log                     # View snapshot history
    kjerne diff [service]          # View changes since last snapshot
    kjerne validate <service>      # Run all quality checks
    kjerne deploy <service>        # Deploy to prod
    kjerne rollback <service>      # Rollback prod to previous snapshot
    kjerne lint [service]          # Run linter
    kjerne test [service]          # Run tests
    kjerne security [service]      # Security scan
    kjerne status                  # Show lab status
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# Configuration
KJERNE_ROOT = Path.home() / "kjerne"
KJERNE_META = KJERNE_ROOT / ".kjerne"
SNAPSHOTS_DIR = KJERNE_META / "snapshots"
DIFFS_DIR = KJERNE_META / "diffs"
PROD_ROOT = Path.home() / "prod"
CONFIG_FILE = KJERNE_META / "config.json"


@dataclass
class Config:
    version: str = "1.0.0"
    created: str = ""
    services: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class Snapshot:
    id: str
    timestamp: str
    message: str
    services: list
    author: str
    files: dict  # path -> sha256

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def generate_id() -> str:
    """Generate short unique ID."""
    import random
    import string
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))


def hash_file(path: Path) -> str:
    """SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_service_files(service_path: Path) -> dict:
    """Get all files in a service with their hashes."""
    files = {}
    if not service_path.exists():
        return files

    for path in service_path.rglob("*"):
        if path.is_file():
            # Skip __pycache__, .pyc, etc.
            if "__pycache__" in str(path) or path.suffix == ".pyc":
                continue
            rel_path = str(path.relative_to(service_path))
            files[rel_path] = hash_file(path)
    return files


def get_all_services() -> list:
    """List all services in the lab."""
    services_dir = KJERNE_ROOT / "services"
    if not services_dir.exists():
        return []
    return [d.name for d in services_dir.iterdir() if d.is_dir()]


def get_latest_snapshot(service: str = None) -> Optional[Snapshot]:
    """Get most recent snapshot, optionally filtered by service."""
    if not SNAPSHOTS_DIR.exists():
        return None

    snapshots = []
    for snap_dir in SNAPSHOTS_DIR.iterdir():
        manifest_file = snap_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                data = json.load(f)
            if service is None or service in data.get("services", []):
                snapshots.append((snap_dir.name, Snapshot.from_dict(data)))

    if not snapshots:
        return None

    # Sort by timestamp (directory name starts with timestamp)
    snapshots.sort(key=lambda x: x[0], reverse=True)
    return snapshots[0][1]


# =============================================================================
# Commands
# =============================================================================

def cmd_status(args):
    """Show lab status."""
    config = Config.load()

    print("=" * 50)
    print("KJERNE LAB STATUS")
    print("=" * 50)
    print()
    print(f"Lab version: {config.version}")
    print(f"Dev path: {KJERNE_ROOT}")
    print(f"Prod path: {PROD_ROOT}")
    print()

    services = get_all_services()
    if services:
        print("Services:")
        for svc in services:
            svc_path = KJERNE_ROOT / "services" / svc
            version = "?"
            init_file = svc_path / "__init__.py"
            if init_file.exists():
                content = init_file.read_text()
                for line in content.split("\n"):
                    if "__version__" in line:
                        version = line.split("=")[1].strip().strip("'\"")
                        break
            prod_exists = (PROD_ROOT / "services" / svc).exists()
            prod_status = "[DEPLOYED]" if prod_exists else "[dev only]"
            print(f"  - {svc} v{version} {prod_status}")
    else:
        print("No services found.")

    print()
    latest = get_latest_snapshot()
    if latest:
        print(f"Latest snapshot: {latest.id} ({latest.timestamp})")
        print(f"  Message: {latest.message}")
    else:
        print("No snapshots yet.")

    print("=" * 50)


def cmd_snapshot(args):
    """Create a snapshot of current state."""
    message = args.message
    services = args.services or get_all_services()

    if not services:
        print("No services to snapshot.")
        return 1

    # Create snapshot
    snap_id = generate_id()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    snap_dir = SNAPSHOTS_DIR / f"{timestamp}_{snap_id}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    all_files = {}
    for svc in services:
        svc_path = KJERNE_ROOT / "services" / svc
        if not svc_path.exists():
            print(f"Warning: Service {svc} not found, skipping.")
            continue

        # Get file hashes
        files = get_service_files(svc_path)
        for path, hash_val in files.items():
            all_files[f"services/{svc}/{path}"] = hash_val

        # Create tarball
        tar_path = snap_dir / f"services_{svc}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(svc_path, arcname=svc)

    # Also snapshot core/
    core_path = KJERNE_ROOT / "core"
    if core_path.exists():
        core_files = get_service_files(core_path)
        for path, hash_val in core_files.items():
            all_files[f"core/{path}"] = hash_val

        tar_path = snap_dir / "core.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(core_path, arcname="core")

    # Create manifest
    snapshot = Snapshot(
        id=snap_id,
        timestamp=datetime.now().isoformat(),
        message=message,
        services=services,
        author=os.environ.get("USER", "unknown"),
        files=all_files,
    )

    manifest_path = snap_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(snapshot.to_dict(), f, indent=2)

    print(f"Snapshot created: {snap_id}")
    print(f"  Services: {', '.join(services)}")
    print(f"  Files: {len(all_files)}")
    print(f"  Location: {snap_dir}")

    return 0


def cmd_log(args):
    """Show snapshot history."""
    if not SNAPSHOTS_DIR.exists():
        print("No snapshots yet.")
        return 0

    snapshots = []
    for snap_dir in SNAPSHOTS_DIR.iterdir():
        manifest_file = snap_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                data = json.load(f)
            snapshots.append(data)

    if not snapshots:
        print("No snapshots found.")
        return 0

    # Sort by timestamp
    snapshots.sort(key=lambda x: x["timestamp"], reverse=True)

    limit = args.limit or 10
    print(f"Last {min(limit, len(snapshots))} snapshots:")
    print()

    for snap in snapshots[:limit]:
        ts = snap["timestamp"][:19].replace("T", " ")
        print(f"[{snap['id']}] {ts}")
        print(f"  {snap['message']}")
        print(f"  Services: {', '.join(snap['services'])}")
        print()

    return 0


def cmd_diff(args):
    """Show changes since last snapshot."""
    service = args.service

    latest = get_latest_snapshot(service)
    if not latest:
        print("No previous snapshot to compare against.")
        return 1

    print(f"Changes since snapshot {latest.id} ({latest.timestamp[:19]}):")
    print()

    # Get current files
    current_files = {}

    if service:
        svc_path = KJERNE_ROOT / "services" / service
        if svc_path.exists():
            files = get_service_files(svc_path)
            for path, hash_val in files.items():
                current_files[f"services/{service}/{path}"] = hash_val
    else:
        # All services + core
        for svc in get_all_services():
            svc_path = KJERNE_ROOT / "services" / svc
            files = get_service_files(svc_path)
            for path, hash_val in files.items():
                current_files[f"services/{svc}/{path}"] = hash_val

        core_path = KJERNE_ROOT / "core"
        if core_path.exists():
            files = get_service_files(core_path)
            for path, hash_val in files.items():
                current_files[f"core/{path}"] = hash_val

    # Compare
    old_files = latest.files
    added = set(current_files.keys()) - set(old_files.keys())
    removed = set(old_files.keys()) - set(current_files.keys())
    modified = {
        f for f in set(current_files.keys()) & set(old_files.keys())
        if current_files[f] != old_files[f]
    }

    if not added and not removed and not modified:
        print("No changes.")
        return 0

    if added:
        print(f"Added ({len(added)}):")
        for f in sorted(added)[:20]:
            print(f"  + {f}")
        if len(added) > 20:
            print(f"  ... and {len(added) - 20} more")
        print()

    if removed:
        print(f"Removed ({len(removed)}):")
        for f in sorted(removed)[:20]:
            print(f"  - {f}")
        if len(removed) > 20:
            print(f"  ... and {len(removed) - 20} more")
        print()

    if modified:
        print(f"Modified ({len(modified)}):")
        for f in sorted(modified)[:20]:
            print(f"  ~ {f}")
        if len(modified) > 20:
            print(f"  ... and {len(modified) - 20} more")
        print()

    print(f"Summary: +{len(added)} -{len(removed)} ~{len(modified)}")
    return 0


def cmd_lint(args):
    """Run linter on service(s)."""
    service = args.service
    paths = []

    if service:
        svc_path = KJERNE_ROOT / "services" / service
        if not svc_path.exists():
            print(f"Service not found: {service}")
            return 1
        paths.append(str(svc_path))
    else:
        for svc in get_all_services():
            paths.append(str(KJERNE_ROOT / "services" / svc))
        core_path = KJERNE_ROOT / "core"
        if core_path.exists():
            paths.append(str(core_path))

    if not paths:
        print("Nothing to lint.")
        return 0

    print("Running lint checks...")
    print()

    errors = 0
    for path in paths:
        # Check Python syntax
        for py_file in Path(path).rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                with open(py_file) as f:
                    compile(f.read(), py_file, "exec")
            except SyntaxError as e:
                print(f"SYNTAX ERROR: {py_file}:{e.lineno}")
                print(f"  {e.msg}")
                errors += 1

    if errors:
        print()
        print(f"FAILED: {errors} syntax error(s)")
        return 1

    print("All syntax checks passed.")
    return 0


def cmd_test(args):
    """Run tests for service(s)."""
    service = args.service

    if service:
        test_path = KJERNE_ROOT / "services" / service / "tests"
        if not test_path.exists():
            print(f"No tests found for {service}")
            return 1
        paths = [str(test_path)]
    else:
        paths = []
        for svc in get_all_services():
            test_path = KJERNE_ROOT / "services" / svc / "tests"
            if test_path.exists():
                paths.append(str(test_path))

    if not paths:
        print("No tests found.")
        return 0

    print("Running tests...")
    print()

    # Use pytest if available, else unittest
    try:
        result = subprocess.run(
            ["python", "-m", "pytest"] + paths + ["-v", "--tb=short"],
            cwd=str(KJERNE_ROOT),
        )
        return result.returncode
    except FileNotFoundError:
        result = subprocess.run(
            ["python", "-m", "unittest", "discover"] + paths,
            cwd=str(KJERNE_ROOT),
        )
        return result.returncode


def cmd_security(args):
    """Run security scan on service(s)."""
    service = args.service

    if service:
        svc_path = KJERNE_ROOT / "services" / service
        if not svc_path.exists():
            print(f"Service not found: {service}")
            return 1
        paths = [svc_path]
    else:
        paths = [KJERNE_ROOT / "services" / svc for svc in get_all_services()]

    print("Running security scan...")
    print()

    # Try to import the security analyzer from core
    sys.path.insert(0, str(KJERNE_ROOT))
    try:
        from core.security import SecurityAnalyzer
        analyzer = SecurityAnalyzer()
    except ImportError:
        # Fallback: basic pattern matching
        print("Warning: SecurityAnalyzer not found, using basic checks.")
        analyzer = None

    issues = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}

    for svc_path in paths:
        for py_file in svc_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()

            if analyzer:
                report = analyzer.analyze(content)
                for issue in report.issues:
                    issues[issue.severity.value.upper()].append(
                        f"{py_file.name}:{issue.line_number} - {issue.message}"
                    )
            else:
                # Basic checks
                dangerous = [
                    ("eval(", "CRITICAL", "eval() usage"),
                    ("exec(", "CRITICAL", "exec() usage"),
                    ("os.system(", "HIGH", "os.system() usage"),
                    ("shell=True", "HIGH", "shell=True in subprocess"),
                    ("password =", "MEDIUM", "Possible hardcoded password"),
                ]
                for pattern, level, msg in dangerous:
                    if pattern in content:
                        line_num = next(
                            (i + 1 for i, line in enumerate(content.split("\n"))
                             if pattern in line),
                            0
                        )
                        issues[level].append(f"{py_file.name}:{line_num} - {msg}")

    # Report
    total = sum(len(v) for v in issues.values())
    if total == 0:
        print("No security issues found.")
        return 0

    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if issues[level]:
            print(f"{level} ({len(issues[level])}):")
            for issue in issues[level][:10]:
                print(f"  {issue}")
            if len(issues[level]) > 10:
                print(f"  ... and {len(issues[level]) - 10} more")
            print()

    # Determine return code
    config = Config.load()
    block_level = config.settings.get("security_block_level", "CRITICAL")

    if block_level == "CRITICAL" and issues["CRITICAL"]:
        print("BLOCKED: Critical security issues found.")
        return 1
    if block_level == "HIGH" and (issues["CRITICAL"] or issues["HIGH"]):
        print("BLOCKED: High or critical security issues found.")
        return 1

    print("Security scan complete (warnings only).")
    return 0


def cmd_validate(args):
    """Full validation before deploy."""
    service = args.service

    print(f"Validating: {service}")
    print("=" * 50)
    print()

    # 1. Lint
    print("[1/3] Lint check...")
    lint_args = argparse.Namespace(service=service)
    if cmd_lint(lint_args) != 0:
        print()
        print("VALIDATION FAILED: Lint errors")
        return 1
    print()

    # 2. Tests
    print("[2/3] Running tests...")
    test_args = argparse.Namespace(service=service)
    if cmd_test(test_args) != 0:
        print()
        print("VALIDATION FAILED: Tests failed")
        return 1
    print()

    # 3. Security
    print("[3/3] Security scan...")
    sec_args = argparse.Namespace(service=service)
    if cmd_security(sec_args) != 0:
        print()
        print("VALIDATION FAILED: Security issues")
        return 1
    print()

    print("=" * 50)
    print("VALIDATION PASSED")
    print(f"Service {service} is ready for deployment.")
    print("=" * 50)
    return 0


def cmd_deploy(args):
    """Deploy service to prod."""
    service = args.service
    force = args.force

    svc_path = KJERNE_ROOT / "services" / service
    if not svc_path.exists():
        print(f"Service not found: {service}")
        return 1

    # Validate first (unless --force)
    if not force:
        print("Running pre-deploy validation...")
        print()
        val_args = argparse.Namespace(service=service)
        if cmd_validate(val_args) != 0:
            print()
            print("Deploy cancelled. Use --force to override.")
            return 1
        print()

    # Create snapshot before deploy
    print("Creating pre-deploy snapshot...")
    snap_args = argparse.Namespace(
        message=f"Pre-deploy: {service}",
        services=[service]
    )
    cmd_snapshot(snap_args)
    print()

    # Copy to prod
    prod_svc_path = PROD_ROOT / "services" / service
    prod_svc_path.parent.mkdir(parents=True, exist_ok=True)

    if prod_svc_path.exists():
        print(f"Removing old prod version...")
        shutil.rmtree(prod_svc_path)

    print(f"Deploying {service} to {prod_svc_path}...")
    shutil.copytree(svc_path, prod_svc_path)

    # Also copy core/ if it exists
    core_src = KJERNE_ROOT / "core"
    core_dst = PROD_ROOT / "core"
    if core_src.exists():
        if core_dst.exists():
            shutil.rmtree(core_dst)
        shutil.copytree(core_src, core_dst)
        print(f"Deployed core/ to {core_dst}")

    print()
    print("=" * 50)
    print(f"DEPLOYED: {service}")
    print(f"Location: {prod_svc_path}")
    print("=" * 50)
    return 0


def cmd_rollback(args):
    """Rollback prod to previous snapshot."""
    service = args.service

    # Find snapshots for this service
    if not SNAPSHOTS_DIR.exists():
        print("No snapshots available for rollback.")
        return 1

    snapshots = []
    for snap_dir in sorted(SNAPSHOTS_DIR.iterdir(), reverse=True):
        manifest_file = snap_dir / "manifest.json"
        tar_file = snap_dir / f"services_{service}.tar.gz"
        if manifest_file.exists() and tar_file.exists():
            with open(manifest_file) as f:
                data = json.load(f)
            if service in data.get("services", []):
                snapshots.append((snap_dir, data))

    if not snapshots:
        print(f"No snapshots found for service: {service}")
        return 1

    # Skip the most recent (that's current), get the one before
    if len(snapshots) < 2:
        print("Only one snapshot available, cannot rollback further.")
        return 1

    snap_dir, snap_data = snapshots[1]  # Second most recent
    print(f"Rolling back to snapshot: {snap_data['id']}")
    print(f"  From: {snap_data['timestamp']}")
    print(f"  Message: {snap_data['message']}")
    print()

    # Extract to prod
    prod_svc_path = PROD_ROOT / "services" / service
    if prod_svc_path.exists():
        shutil.rmtree(prod_svc_path)

    prod_svc_path.parent.mkdir(parents=True, exist_ok=True)

    tar_file = snap_dir / f"services_{service}.tar.gz"
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall(prod_svc_path.parent)

    # The tar extracts to service name, rename if needed
    extracted = prod_svc_path.parent / service
    if extracted != prod_svc_path and extracted.exists():
        if prod_svc_path.exists():
            shutil.rmtree(prod_svc_path)
        shutil.move(str(extracted), str(prod_svc_path))

    print(f"ROLLBACK COMPLETE: {service}")
    print(f"Restored from: {snap_data['id']}")
    return 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kjerne Lab Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    subparsers.add_parser("status", help="Show lab status")

    # snapshot
    snap_parser = subparsers.add_parser("snapshot", help="Create snapshot")
    snap_parser.add_argument("message", help="Snapshot message")
    snap_parser.add_argument("--services", "-s", nargs="+", help="Services to snapshot")

    # log
    log_parser = subparsers.add_parser("log", help="View snapshot history")
    log_parser.add_argument("--limit", "-n", type=int, help="Number of entries")

    # diff
    diff_parser = subparsers.add_parser("diff", help="View changes since last snapshot")
    diff_parser.add_argument("service", nargs="?", help="Service to diff")

    # lint
    lint_parser = subparsers.add_parser("lint", help="Run linter")
    lint_parser.add_argument("service", nargs="?", help="Service to lint")

    # test
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("service", nargs="?", help="Service to test")

    # security
    sec_parser = subparsers.add_parser("security", help="Security scan")
    sec_parser.add_argument("service", nargs="?", help="Service to scan")

    # validate
    val_parser = subparsers.add_parser("validate", help="Full validation")
    val_parser.add_argument("service", help="Service to validate")

    # deploy
    dep_parser = subparsers.add_parser("deploy", help="Deploy to prod")
    dep_parser.add_argument("service", help="Service to deploy")
    dep_parser.add_argument("--force", "-f", action="store_true", help="Skip validation")

    # rollback
    roll_parser = subparsers.add_parser("rollback", help="Rollback prod")
    roll_parser.add_argument("service", help="Service to rollback")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "status": cmd_status,
        "snapshot": cmd_snapshot,
        "log": cmd_log,
        "diff": cmd_diff,
        "lint": cmd_lint,
        "test": cmd_test,
        "security": cmd_security,
        "validate": cmd_validate,
        "deploy": cmd_deploy,
        "rollback": cmd_rollback,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
