"""
Codebase Context

Reads and understands project structure to provide context for code generation.
This is what makes the difference between "generate a function" and
"generate a function that fits into THIS codebase".
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class FileInfo:
    """Information about a file in the project."""
    path: Path
    relative_path: str
    extension: str
    size_bytes: int
    preview: str = ""  # First N lines
    symbols: list[str] = field(default_factory=list)  # Functions, classes found


@dataclass
class ProjectContext:
    """Context about a project/codebase."""
    root: Path
    files: list[FileInfo] = field(default_factory=list)
    structure: str = ""  # Tree view
    summary: str = ""  # LLM-generated summary

    @property
    def file_count(self) -> int:
        return len(self.files)

    def get_file(self, relative_path: str) -> FileInfo | None:
        for f in self.files:
            if f.relative_path == relative_path:
                return f
        return None


class CodebaseReader:
    """
    Reads and indexes a codebase for context.

    Key capabilities:
    - Tree structure visualization
    - File discovery (respects .gitignore patterns)
    - Symbol extraction (functions, classes)
    - Content preview for relevant files
    """

    # Common patterns to ignore
    IGNORE_PATTERNS = {
        '__pycache__', '.git', '.venv', 'venv', 'node_modules',
        '.pytest_cache', '.mypy_cache', '*.pyc', '*.pyo',
        '.env', '.DS_Store', '*.egg-info', 'dist', 'build',
    }

    # Extensions we care about
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go',
        '.rs', '.cpp', '.c', '.h', '.hpp', '.rb', '.php',
        '.swift', '.kt', '.scala', '.sql', '.sh', '.bash',
    }

    CONFIG_EXTENSIONS = {
        '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
        '.env.example', 'Dockerfile', '.dockerignore',
    }

    DOC_EXTENSIONS = {
        '.md', '.rst', '.txt',
    }

    def __init__(self, max_file_size: int = 100_000, preview_lines: int = 50):
        self.max_file_size = max_file_size
        self.preview_lines = preview_lines

    def read_project(self, root: Path | str) -> ProjectContext:
        """Read a project and build context."""
        root = Path(root).resolve()

        if not root.exists():
            raise ValueError(f"Project root does not exist: {root}")

        context = ProjectContext(root=root)

        # Collect files
        for file_info in self._discover_files(root):
            context.files.append(file_info)

        # Build tree structure
        context.structure = self._build_tree(root, context.files)

        return context

    def _discover_files(self, root: Path) -> Iterator[FileInfo]:
        """Discover relevant files in the project."""
        for path in root.rglob('*'):
            if not path.is_file():
                continue

            # Check ignore patterns
            if self._should_ignore(path):
                continue

            # Check extension
            ext = path.suffix.lower()
            if ext not in self.CODE_EXTENSIONS | self.CONFIG_EXTENSIONS | self.DOC_EXTENSIONS:
                # Also allow extensionless files like Makefile, Dockerfile
                if path.name not in {'Makefile', 'Dockerfile', 'Procfile', 'Gemfile'}:
                    continue

            # Check size
            try:
                size = path.stat().st_size
                if size > self.max_file_size:
                    continue
            except OSError:
                continue

            # Build file info
            relative = path.relative_to(root)
            preview = self._get_preview(path)
            symbols = self._extract_symbols(path) if ext == '.py' else []

            yield FileInfo(
                path=path,
                relative_path=str(relative),
                extension=ext,
                size_bytes=size,
                preview=preview,
                symbols=symbols,
            )

    def _should_ignore(self, path: Path) -> bool:
        """Check if path matches ignore patterns."""
        parts = path.parts
        for pattern in self.IGNORE_PATTERNS:
            if pattern.startswith('*'):
                # Wildcard extension
                if path.suffix == pattern[1:]:
                    return True
            else:
                # Directory or file name
                if pattern in parts:
                    return True
        return False

    def _get_preview(self, path: Path) -> str:
        """Get preview of file contents."""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= self.preview_lines:
                        break
                    lines.append(line.rstrip())
                return '\n'.join(lines)
        except Exception:
            return ""

    def _extract_symbols(self, path: Path) -> list[str]:
        """Extract function and class names from Python files."""
        import ast

        try:
            with open(path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            symbols = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.append(f"def {node.name}")
                elif isinstance(node, ast.ClassDef):
                    symbols.append(f"class {node.name}")
                elif isinstance(node, ast.AsyncFunctionDef):
                    symbols.append(f"async def {node.name}")

            return symbols
        except Exception:
            return []

    def _build_tree(self, root: Path, files: list[FileInfo], max_depth: int = 4) -> str:
        """Build a tree view of the project structure."""
        lines = [root.name + "/"]

        # Group files by directory
        dirs: dict[str, list[str]] = {}
        for f in files:
            parts = Path(f.relative_path).parts
            if len(parts) == 1:
                dir_key = ""
            else:
                dir_key = str(Path(*parts[:-1]))

            if dir_key not in dirs:
                dirs[dir_key] = []
            dirs[dir_key].append(parts[-1])

        # Build tree
        shown_dirs = set()
        for f in sorted(files, key=lambda x: x.relative_path):
            parts = Path(f.relative_path).parts
            depth = len(parts)

            if depth > max_depth:
                continue

            # Show directory entries
            for i in range(1, len(parts)):
                dir_path = str(Path(*parts[:i]))
                if dir_path not in shown_dirs:
                    indent = "  " * (i)
                    lines.append(f"{indent}{parts[i-1]}/")
                    shown_dirs.add(dir_path)

            # Show file
            indent = "  " * depth
            lines.append(f"{indent}{parts[-1]}")

        return '\n'.join(lines[:100])  # Limit output

    def get_relevant_context(self, context: ProjectContext, query: str, max_files: int = 5) -> str:
        """Get context relevant to a specific query/task."""
        # Simple relevance: keyword matching in file names and symbols
        query_words = set(query.lower().split())

        scored_files = []
        for f in context.files:
            score = 0

            # Check file name
            name_words = set(f.relative_path.lower().replace('_', ' ').replace('/', ' ').split())
            score += len(query_words & name_words) * 2

            # Check symbols
            for symbol in f.symbols:
                symbol_words = set(symbol.lower().replace('_', ' ').split())
                score += len(query_words & symbol_words)

            if score > 0:
                scored_files.append((score, f))

        # Sort by relevance
        scored_files.sort(key=lambda x: x[0], reverse=True)

        # Build context string
        lines = ["## Relevant Files\n"]
        for _, f in scored_files[:max_files]:
            lines.append(f"### {f.relative_path}")
            if f.symbols:
                lines.append(f"Symbols: {', '.join(f.symbols[:10])}")
            lines.append("```")
            lines.append(f.preview[:2000])
            lines.append("```\n")

        return '\n'.join(lines)


def format_context_for_prompt(context: ProjectContext, task: str, max_tokens: int = 2000) -> str:
    """Format project context for inclusion in an LLM prompt."""
    lines = [
        "## Project Structure\n",
        "```",
        context.structure,
        "```\n",
    ]

    # Add relevant file context
    reader = CodebaseReader()
    relevant = reader.get_relevant_context(context, task, max_files=3)
    lines.append(relevant)

    result = '\n'.join(lines)

    # Rough token limit (4 chars per token estimate)
    if len(result) > max_tokens * 4:
        result = result[:max_tokens * 4] + "\n... (truncated)"

    return result
