#!/usr/bin/env python3
"""Pre-commit hook: detect non-atomic read-modify-write patterns in models.

CRIT-01/02/03 regression prevention — catches `self.field += N; self.save()`
patterns that cause race conditions under concurrent requests. The fix is to
use F() expressions: `Model.objects.filter(pk=self.pk).update(field=F('field') + N)`.

Only scans staged model files. Exit 0 = clean, Exit 1 = violations found.
"""

import re
import subprocess
import sys


def _get_staged_model_files():
    """Return staged models.py files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return [f for f in files if f.endswith("models.py")]
    except Exception:
        return []


# Patterns that indicate in-memory increment before save
# Matches: self.field += value, self.field -= value, self.field = self.field + value
INCREMENT_PATTERN = re.compile(
    r"self\.\w+\s*[+\-]=\s*"  # self.field += N or self.field -= N
)

# Also catch: self.field = self.field + N
REASSIGN_PATTERN = re.compile(r"self\.(\w+)\s*=\s*self\.\1\s*[+\-]")

# Allowlist: patterns that are safe (e.g., in-memory-only fields, list/dict ops)
SAFE_SUFFIXES = {
    ".append(",
    ".extend(",
    ".pop(",
    ".update(",
    ".get(",
    ".items()",
    ".keys()",
    ".values()",
}


def main():
    staged = _get_staged_model_files()
    if not staged:
        sys.exit(0)

    violations = []

    for filepath in staged:
        try:
            result = subprocess.run(
                ["git", "show", f":{filepath}"],
                capture_output=True,
                text=True,
            )
            content = result.stdout
        except Exception:
            continue

        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            # Check for in-memory increment patterns
            if INCREMENT_PATTERN.search(stripped) or REASSIGN_PATTERN.search(stripped):
                # noqa: nonatomic suppresses this warning
                if "# noqa: nonatomic" in line or "# noqa:nonatomic" in line:
                    continue
                # Check if there's a self.save() nearby (within 5 lines)
                nearby = "\n".join(lines[i : min(i + 5, len(lines))])
                if "self.save(" in nearby or ".save(update_fields=" in nearby:
                    # Not a safe pattern (list/dict operation)
                    if not any(safe in stripped for safe in SAFE_SUFFIXES):
                        violations.append((filepath, i, stripped[:120]))

    if violations:
        print()
        print("=" * 70)
        print("  AUDIT WARNING: Non-atomic read-modify-write pattern(s) detected")
        print()
        for filepath, line, content in violations:
            print(f"  {filepath}:{line}")
            print(f"    {content}")
            print()
        print("  These patterns cause race conditions under concurrent requests.")
        print("  Fix: use F() expressions instead:")
        print("    Model.objects.filter(pk=self.pk).update(field=F('field') + N)")
        print()
        print("  If this is intentional (e.g., in-memory only), add a # noqa: nonatomic comment.")
        print("=" * 70)
        print()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
