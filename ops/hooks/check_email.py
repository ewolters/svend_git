#!/usr/bin/env python3
"""Pre-commit hook: block direct django.core.mail.send_mail() usage.

All email MUST go through notifications.email_service.EmailService.
Direct send_mail() imports create scattered, untraceable email paths.

Exempt: email_service.py itself (the only authorized caller).

Exit 0 = clean, Exit 1 = violations found.
"""

import re
import subprocess
import sys

PATTERN = re.compile(r"from django\.core\.mail import.*send_mail")

EXEMPT_FILES = {
    "email_service.py",
}


def _get_staged_python_files():
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return [f for f in files if f.endswith(".py")]
    except Exception:
        return []


def main():
    staged = _get_staged_python_files()
    if not staged:
        sys.exit(0)

    violations = []

    for filepath in staged:
        filename = filepath.split("/")[-1]
        if filename in EXEMPT_FILES:
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

        for i, line in enumerate(content.splitlines(), 1):
            if PATTERN.search(line):
                violations.append((filepath, i, line.strip()[:80]))

    if violations:
        print()
        print("=" * 70)
        print("  EMAIL SERVICE VIOLATION: Direct send_mail() import detected")
        print()
        for filepath, line, content in violations:
            print(f"  {filepath}:{line}")
            print(f"    {content}")
        print()
        print("  All email must go through notifications.email_service.")
        print("  Use: from notifications.email_service import email_service")
        print("  Then: email_service.send(to=..., subject=..., body_html=...)")
        print("=" * 70)
        print()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
