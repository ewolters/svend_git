#!/usr/bin/env python3
"""Pre-commit hook: prevent secrets from entering git history.

SEC-001 §7 enforcement — secrets must never be hardcoded in source.
Once a secret enters git history, it's effectively permanent — even
force-push and BFG can't guarantee removal from all clones.

Scans staged file contents (not working copy) for patterns that
indicate hardcoded credentials. Pure regex — no Django or DB needed.

Exit 0 = clean, Exit 1 = secrets detected.
"""

import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Patterns that indicate hardcoded secrets
# ---------------------------------------------------------------------------

SECRET_PATTERNS = [
    # Anthropic / OpenAI API keys
    (re.compile(r"""['"]sk-ant-[a-zA-Z0-9_-]{20,}['"]"""), "Anthropic API key"),
    (re.compile(r"""['"]sk-[a-zA-Z0-9]{20,}['"]"""), "OpenAI/API key (sk- prefix)"),
    # Stripe keys
    (re.compile(r"""['"]sk_live_[a-zA-Z0-9]{20,}['"]"""), "Stripe live secret key"),
    (re.compile(r"""['"]sk_test_[a-zA-Z0-9]{20,}['"]"""), "Stripe test secret key"),
    (
        re.compile(r"""['"]pk_live_[a-zA-Z0-9]{20,}['"]"""),
        "Stripe live publishable key",
    ),
    (re.compile(r"""['"]whsec_[a-zA-Z0-9]{20,}['"]"""), "Stripe webhook secret"),
    # Generic secret assignment patterns
    (
        re.compile(
            r"""(?:SECRET_KEY|ENCRYPTION_KEY|JWT_SECRET)\s*=\s*['"][^'"]{10,}['"]"""
        ),
        "Hardcoded secret key",
    ),
    (
        re.compile(
            r"""(?:password|passwd|pwd)\s*=\s*['"][^'"]{6,}['"]""", re.IGNORECASE
        ),
        "Hardcoded password",
    ),
    # Database URLs with credentials
    (
        re.compile(r"""postgres(?:ql)?://\w+:[^@\s'"]+@(?!localhost|127\.0\.0\.1)"""),
        "Database URL with remote credentials",
    ),
    # AWS keys
    (re.compile(r"""['"]AKIA[0-9A-Z]{16}['"]"""), "AWS access key ID"),
    # Private keys
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"), "Private key"),
]

# Files where secret-like patterns are expected (test fixtures, docs, examples)
EXEMPT_PATHS = {
    ".env.example",
    "CLAUDE.md",
    "check_secrets.py",  # This file itself
    "check_cr.py",  # CR hook references secret patterns in docs
    "conftest.py",  # Test fixtures use test passwords
}

# Paths where password patterns are expected (test files use test credentials)
EXEMPT_PATH_PREFIXES = (
    "tests",
    "test_",
)

# File extensions to scan (skip binaries, images, etc.)
SCAN_EXTENSIONS = {
    ".py",
    ".js",
    ".html",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".md",
    ".txt",
    ".sh",
    ".env",
}


def _get_staged_files():
    """Return staged files (added/modified)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        return []


def main():
    staged = _get_staged_files()
    if not staged:
        sys.exit(0)

    violations = []

    for filepath in staged:
        filename = filepath.split("/")[-1]

        # Skip exempt files
        if filename in EXEMPT_PATHS:
            continue

        # Skip non-text files
        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
        if ext not in SCAN_EXTENSIONS:
            continue

        try:
            result = subprocess.run(
                ["git", "show", f":{filepath}"],
                capture_output=True,
                text=True,
            )
            content = result.stdout
        except Exception:
            continue

        # Test files are exempt from password patterns (test credentials are not secrets)
        is_test_file = any(p in filename for p in EXEMPT_PATH_PREFIXES)

        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#") or stripped.startswith("//"):
                continue

            for pattern, description in SECRET_PATTERNS:
                # Skip password patterns in test files
                if is_test_file and "password" in description.lower():
                    continue
                if pattern.search(line):
                    # Redact the actual secret in output
                    safe_line = (
                        stripped[:40] + "..." if len(stripped) > 40 else stripped
                    )
                    violations.append((filepath, i, description, safe_line))

    if violations:
        print()
        print("=" * 70)
        print("  SEC-001 VIOLATION: Potential secret(s) detected in staged files")
        print()
        for filepath, line, description, content in violations:
            print(f"  {filepath}:{line}")
            print(f"    {description}")
            print(f"    {content}")
            print()
        print("  Secrets must come from /etc/svend/env, never hardcoded.")
        print("  If this is a false positive, add the file to EXEMPT_PATHS")
        print("  in ops/hooks/check_secrets.py.")
        print("=" * 70)
        print()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
