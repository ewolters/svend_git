"""
Code Security Analyzer

Checks generated code for common security vulnerabilities:
- SQL injection patterns
- Command injection
- Path traversal
- Hardcoded secrets
- Unsafe deserialization
- XSS patterns
- SSRF patterns
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SecuritySeverity(Enum):
    """Severity levels for security issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """A security issue found in code."""
    severity: SecuritySeverity
    category: str
    message: str
    line_number: int = 0
    code_snippet: str = ""
    recommendation: str = ""
    cwe_id: str = ""  # Common Weakness Enumeration ID

    def __str__(self):
        icon = {
            SecuritySeverity.LOW: "ℹ️",
            SecuritySeverity.MEDIUM: "⚠️",
            SecuritySeverity.HIGH: "❌",
            SecuritySeverity.CRITICAL: "🚨",
        }.get(self.severity, "•")
        return f"{icon} [{self.severity.value.upper()}] {self.category}: {self.message}"


@dataclass
class SecurityReport:
    """Result of security analysis."""
    issues: list[SecurityIssue] = field(default_factory=list)
    passed: bool = True
    score: float = 100.0  # 0-100

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == SecuritySeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == SecuritySeverity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == SecuritySeverity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == SecuritySeverity.LOW)

    def summary(self) -> str:
        """Generate text summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            "=" * 50,
            f"SECURITY ANALYSIS: {status}",
            "=" * 50,
            "",
            f"Score: {self.score:.0f}/100",
            f"Issues: {len(self.issues)}",
            f"  Critical: {self.critical_count}",
            f"  High: {self.high_count}",
            f"  Medium: {self.medium_count}",
            f"  Low: {self.low_count}",
            "",
        ]

        if self.issues:
            lines.append("## Issues Found")
            lines.append("")
            for issue in self.issues:
                lines.append(f"  {issue}")
                if issue.recommendation:
                    lines.append(f"    Fix: {issue.recommendation}")
                if issue.cwe_id:
                    lines.append(f"    Ref: CWE-{issue.cwe_id}")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": [
                {
                    "severity": i.severity.value,
                    "category": i.category,
                    "message": i.message,
                    "line_number": i.line_number,
                    "cwe_id": i.cwe_id,
                }
                for i in self.issues
            ],
        }


class SecurityAnalyzer:
    """
    Static security analysis for Python code.

    Checks for OWASP Top 10 style vulnerabilities adapted for Python.
    """

    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        (r'execute\s*\(\s*["\'][^"\']*%s', "String formatting in SQL query"),
        (r'execute\s*\(\s*["\'][^"\']*\+', "String concatenation in SQL query"),
        (r'execute\s*\(\s*f["\']', "F-string in SQL execute"),
        (r'cursor\.execute\s*\(\s*["\'][^"\']*\.format', ".format() in SQL query"),
        (r'executemany\s*\(\s*f["\']', "F-string in executemany"),
    ]

    # Command Injection patterns
    COMMAND_INJECTION_PATTERNS = [
        (r'os\.system\s*\(', "os.system() - use subprocess with shell=False"),
        (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "subprocess with shell=True"),
        (r'subprocess\.run\s*\([^)]*shell\s*=\s*True', "subprocess.run with shell=True"),
        (r'subprocess\.Popen\s*\([^)]*shell\s*=\s*True', "Popen with shell=True"),
        (r'eval\s*\(', "eval() can execute arbitrary code"),
        (r'exec\s*\(', "exec() can execute arbitrary code"),
        (r'__import__\s*\(', "__import__() can load arbitrary modules"),
    ]

    # Path Traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        (r'open\s*\(\s*[^,)]+\+', "Unsanitized path concatenation in open()"),
        (r'open\s*\(\s*f["\'][^"\']*\{', "F-string path in open()"),
        (r'os\.path\.join\s*\([^)]*\+', "Unsanitized input in path.join"),
        (r'\.read\s*\(\s*\).*request', "Reading file from request input"),
    ]

    # Hardcoded Secrets patterns
    SECRET_PATTERNS = [
        (r'password\s*=\s*["\'][^"\']{4,}["\']', "Hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded API key"),
        (r'secret\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret"),
        (r'token\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded token"),
        (r'AWS_SECRET_ACCESS_KEY\s*=\s*["\']', "Hardcoded AWS secret"),
        (r'PRIVATE_KEY\s*=\s*["\']', "Hardcoded private key"),
        (r'-----BEGIN.*PRIVATE KEY-----', "Embedded private key"),
    ]

    # Unsafe Deserialization patterns
    DESERIALIZATION_PATTERNS = [
        (r'pickle\.load', "pickle.load can execute arbitrary code"),
        (r'pickle\.loads', "pickle.loads can execute arbitrary code"),
        (r'yaml\.load\s*\([^)]*(?!Loader)', "yaml.load without safe Loader"),
        (r'marshal\.load', "marshal.load is unsafe"),
    ]

    # XSS patterns (for web output)
    XSS_PATTERNS = [
        (r'\.format\s*\([^)]*request', "User input in format string"),
        (r'f["\'][^"\']*\{[^}]*request', "Request data in f-string"),
        (r'Response\s*\([^)]*\+', "String concatenation in Response"),
        (r'render_template_string\s*\(', "render_template_string is unsafe"),
    ]

    # SSRF patterns
    SSRF_PATTERNS = [
        (r'requests\.get\s*\([^)]*request\.(args|form|data)', "User input in requests URL"),
        (r'urllib\.request\.urlopen\s*\([^)]*\+', "Dynamic URL in urlopen"),
        (r'urllib\.request\.urlopen\s*\(.*request', "User input in urlopen"),
    ]

    # Cryptography issues
    CRYPTO_PATTERNS = [
        (r'from\s+Crypto\.Cipher\s+import\s+DES', "DES is deprecated"),
        (r'from\s+Crypto\.Hash\s+import\s+MD5', "MD5 is weak for security"),
        (r'hashlib\.md5', "MD5 is weak for security purposes"),
        (r'hashlib\.sha1', "SHA1 is weak for security purposes"),
        (r'random\.random', "random module not cryptographically secure"),
        (r'random\.randint', "random module not cryptographically secure"),
    ]

    # Other dangerous patterns
    OTHER_PATTERNS = [
        (r'DEBUG\s*=\s*True', "Debug mode enabled"),
        (r'verify\s*=\s*False', "SSL verification disabled"),
        (r'assert\s+', "Assert statements removed in optimized mode"),
        (r'except\s*:', "Bare except catches all exceptions"),
    ]

    def analyze(self, code: str) -> SecurityReport:
        """
        Analyze code for security issues.

        Args:
            code: Python source code to analyze

        Returns:
            SecurityReport with findings
        """
        issues = []
        lines = code.split('\n')

        # Check each pattern category
        issues.extend(self._check_patterns(
            code, lines, self.SQL_INJECTION_PATTERNS,
            SecuritySeverity.HIGH, "SQL Injection", "CWE-89"
        ))
        issues.extend(self._check_patterns(
            code, lines, self.COMMAND_INJECTION_PATTERNS,
            SecuritySeverity.CRITICAL, "Command Injection", "CWE-78"
        ))
        issues.extend(self._check_patterns(
            code, lines, self.PATH_TRAVERSAL_PATTERNS,
            SecuritySeverity.HIGH, "Path Traversal", "CWE-22"
        ))
        issues.extend(self._check_patterns(
            code, lines, self.SECRET_PATTERNS,
            SecuritySeverity.HIGH, "Hardcoded Secret", "CWE-798"
        ))
        issues.extend(self._check_patterns(
            code, lines, self.DESERIALIZATION_PATTERNS,
            SecuritySeverity.CRITICAL, "Unsafe Deserialization", "CWE-502"
        ))
        issues.extend(self._check_patterns(
            code, lines, self.XSS_PATTERNS,
            SecuritySeverity.MEDIUM, "XSS Risk", "CWE-79"
        ))
        issues.extend(self._check_patterns(
            code, lines, self.SSRF_PATTERNS,
            SecuritySeverity.MEDIUM, "SSRF Risk", "CWE-918"
        ))
        issues.extend(self._check_patterns(
            code, lines, self.CRYPTO_PATTERNS,
            SecuritySeverity.MEDIUM, "Weak Cryptography", "CWE-327"
        ))
        issues.extend(self._check_patterns(
            code, lines, self.OTHER_PATTERNS,
            SecuritySeverity.LOW, "Best Practice", ""
        ))

        # Calculate score
        score = 100.0
        for issue in issues:
            if issue.severity == SecuritySeverity.CRITICAL:
                score -= 25
            elif issue.severity == SecuritySeverity.HIGH:
                score -= 15
            elif issue.severity == SecuritySeverity.MEDIUM:
                score -= 8
            else:
                score -= 2

        score = max(0, score)

        # Determine if passed
        passed = (
            all(i.severity != SecuritySeverity.CRITICAL for i in issues)
            and sum(1 for i in issues if i.severity == SecuritySeverity.HIGH) <= 1
        )

        return SecurityReport(
            issues=issues,
            passed=passed,
            score=score,
        )

    def _check_patterns(
        self,
        code: str,
        lines: list[str],
        patterns: list[tuple[str, str]],
        severity: SecuritySeverity,
        category: str,
        cwe_id: str,
    ) -> list[SecurityIssue]:
        """Check code against a list of patterns."""
        issues = []

        for pattern, message in patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                # Find line number
                line_num = code[:match.start()].count('\n') + 1

                # Get code snippet
                if 0 < line_num <= len(lines):
                    snippet = lines[line_num - 1].strip()
                else:
                    snippet = match.group(0)[:50]

                # Generate recommendation
                recommendation = self._get_recommendation(category, message)

                issues.append(SecurityIssue(
                    severity=severity,
                    category=category,
                    message=message,
                    line_number=line_num,
                    code_snippet=snippet,
                    recommendation=recommendation,
                    cwe_id=cwe_id,
                ))

        return issues

    def _get_recommendation(self, category: str, message: str) -> str:
        """Get a recommendation for fixing the issue."""
        recommendations = {
            "SQL Injection": "Use parameterized queries: cursor.execute('SELECT * FROM t WHERE id = ?', (id,))",
            "Command Injection": "Use subprocess with shell=False and pass args as list",
            "Path Traversal": "Validate and sanitize paths, use os.path.realpath and check against base path",
            "Hardcoded Secret": "Use environment variables or a secrets manager",
            "Unsafe Deserialization": "Use safe loaders (yaml.safe_load) or JSON instead",
            "XSS Risk": "Escape user input, use template auto-escaping",
            "SSRF Risk": "Validate and whitelist allowed URLs/hosts",
            "Weak Cryptography": "Use modern algorithms: SHA-256+, AES, secrets module",
            "Best Practice": "Review and apply security best practices",
        }
        return recommendations.get(category, "Review this code for security implications")


def check_security(code: str) -> SecurityReport:
    """Quick helper to check code security."""
    analyzer = SecurityAnalyzer()
    return analyzer.analyze(code)
