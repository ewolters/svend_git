"""
Text and Code Statistics

Quick statistics for text and code files:
- Word count, character count, line count
- Vocabulary analysis
- Code-specific stats (functions, classes, imports)
"""

import re
import ast
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter


@dataclass
class TextStats:
    """Statistics for text content."""
    char_count: int
    char_count_no_spaces: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    line_count: int

    unique_words: int
    avg_word_length: float
    avg_sentence_length: float

    top_words: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "characters": self.char_count,
            "characters_no_spaces": self.char_count_no_spaces,
            "words": self.word_count,
            "sentences": self.sentence_count,
            "paragraphs": self.paragraph_count,
            "lines": self.line_count,
            "unique_words": self.unique_words,
            "avg_word_length": round(self.avg_word_length, 2),
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "vocabulary_richness": round(self.unique_words / max(self.word_count, 1), 3),
            "top_words": self.top_words[:10],
        }


@dataclass
class CodeStats:
    """Statistics for Python code."""
    total_lines: int
    code_lines: int  # Non-blank, non-comment
    blank_lines: int
    comment_lines: int
    docstring_lines: int

    function_count: int
    class_count: int
    import_count: int

    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "blank_lines": self.blank_lines,
            "comment_lines": self.comment_lines,
            "docstring_lines": self.docstring_lines,
            "code_percentage": round(self.code_lines / max(self.total_lines, 1) * 100, 1),
            "function_count": self.function_count,
            "class_count": self.class_count,
            "import_count": self.import_count,
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
        }


def analyze_text(text: str) -> TextStats:
    """Analyze text and return statistics."""
    # Character counts
    char_count = len(text)
    char_count_no_spaces = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))

    # Word count
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    word_count = len(words)

    # Unique words
    unique_words = len(set(words))

    # Top words (excluding common words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'this', 'that', 'these', 'those', 'it', 'its'}
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    top_words = Counter(filtered_words).most_common(10)

    # Sentence count
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Paragraph count
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    paragraph_count = len(paragraphs)

    # Line count
    line_count = len(text.split('\n'))

    # Averages
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    avg_sentence_length = word_count / max(sentence_count, 1)

    return TextStats(
        char_count=char_count,
        char_count_no_spaces=char_count_no_spaces,
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        line_count=line_count,
        unique_words=unique_words,
        avg_word_length=avg_word_length,
        avg_sentence_length=avg_sentence_length,
        top_words=top_words,
    )


def analyze_code(code: str) -> CodeStats:
    """Analyze Python code and return statistics."""
    lines = code.split('\n')
    total_lines = len(lines)

    # Line classification
    blank_lines = 0
    comment_lines = 0
    docstring_lines = 0
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            blank_lines += 1
            continue

        # Docstring handling
        if in_docstring:
            docstring_lines += 1
            if stripped.endswith(docstring_char):
                in_docstring = False
            continue

        if stripped.startswith('"""') or stripped.startswith("'''"):
            docstring_char = stripped[:3]
            docstring_lines += 1
            if not stripped.endswith(docstring_char) or len(stripped) == 3:
                in_docstring = True
            continue

        # Comments
        if stripped.startswith('#'):
            comment_lines += 1

    code_lines = total_lines - blank_lines - comment_lines - docstring_lines

    # Parse AST for structure
    functions = []
    classes = []
    imports = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(f"async {node.name}")
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
    except SyntaxError:
        pass  # Skip AST analysis on syntax error

    return CodeStats(
        total_lines=total_lines,
        code_lines=code_lines,
        blank_lines=blank_lines,
        comment_lines=comment_lines,
        docstring_lines=docstring_lines,
        function_count=len(functions),
        class_count=len(classes),
        import_count=len(imports),
        functions=functions,
        classes=classes,
        imports=imports,
    )


# CLI
def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Analyze text or code statistics")
    parser.add_argument("input", type=Path, help="File to analyze")
    parser.add_argument("--type", choices=["text", "code", "auto"], default="auto",
                        help="Analysis type")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    content = args.input.read_text()

    # Auto-detect type
    analysis_type = args.type
    if analysis_type == "auto":
        if args.input.suffix in ['.py', '.pyw']:
            analysis_type = "code"
        else:
            analysis_type = "text"

    if analysis_type == "code":
        result = analyze_code(content)
    else:
        result = analyze_text(content)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Statistics: {args.input.name}")
        print("=" * 50)

        if isinstance(result, CodeStats):
            print(f"Total lines: {result.total_lines}")
            print(f"Code lines: {result.code_lines} ({result.code_lines/max(result.total_lines,1)*100:.1f}%)")
            print(f"Blank lines: {result.blank_lines}")
            print(f"Comment lines: {result.comment_lines}")
            print(f"Docstring lines: {result.docstring_lines}")
            print("-" * 50)
            print(f"Functions: {result.function_count}")
            print(f"Classes: {result.class_count}")
            print(f"Imports: {result.import_count}")
            if result.functions:
                print(f"\nFunctions: {', '.join(result.functions[:10])}")
            if result.classes:
                print(f"Classes: {', '.join(result.classes[:10])}")
        else:
            print(f"Characters: {result.char_count} ({result.char_count_no_spaces} without spaces)")
            print(f"Words: {result.word_count}")
            print(f"Unique words: {result.unique_words}")
            print(f"Sentences: {result.sentence_count}")
            print(f"Paragraphs: {result.paragraph_count}")
            print(f"Lines: {result.line_count}")
            print("-" * 50)
            print(f"Avg word length: {result.avg_word_length:.1f} chars")
            print(f"Avg sentence length: {result.avg_sentence_length:.1f} words")
            print(f"Vocabulary richness: {result.unique_words/max(result.word_count,1):.1%}")
            if result.top_words:
                print(f"\nTop words: {', '.join(f'{w}({c})' for w,c in result.top_words[:5])}")


if __name__ == "__main__":
    main()
