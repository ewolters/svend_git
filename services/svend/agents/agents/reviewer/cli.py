#!/usr/bin/env python3
"""
Document Reviewer CLI

Usage:
    python -m reviewer.cli document.md
    python -m reviewer.cli document.md --type technical
    python -m reviewer.cli document.md --type academic --output review.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/eric/Desktop/agents')

from reviewer.agent import DocumentReviewer, Severity


def main():
    parser = argparse.ArgumentParser(
        description="Document Reviewer - General-purpose document review"
    )
    parser.add_argument("document", type=Path, help="Document to review (text or markdown)")
    parser.add_argument(
        "--type", "-t",
        choices=["technical", "business", "academic", "general"],
        default="general",
        help="Document type for checklist selection"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path for review report (markdown)"
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Use LLM for deeper review"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to use for LLM review"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output findings as JSON"
    )

    args = parser.parse_args()

    # Read document
    if not args.document.exists():
        print(f"Error: Document not found: {args.document}")
        return 1

    document = args.document.read_text()
    title = args.document.stem

    # Initialize LLM if requested
    llm = None
    if args.with_llm:
        print(f"Loading {args.model}...")
        try:
            sys.path.insert(0, '/home/eric/Desktop/experiments/neuro_symbolic')
            from local_llm import TransformersLLM
            llm = TransformersLLM(model_name=args.model)
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            print("Continuing without LLM review...")

    # Create reviewer
    reviewer = DocumentReviewer(llm=llm)

    print("=" * 70)
    print("DOCUMENT REVIEWER")
    print("=" * 70)
    print(f"\nDocument: {args.document}")
    print(f"Type: {args.type}")
    print(f"Size: {len(document)} characters, {len(document.split())} words")
    print("\n" + "-" * 70)
    print("Reviewing...")

    # Run review
    result = reviewer.review(document, title=title, doc_type=args.type)

    # Output
    print("\n" + "=" * 70)
    print("REVIEW RESULTS")
    print("=" * 70)

    if args.json:
        import json
        output = {
            "document": title,
            "overall_score": result.overall_score,
            "scores": result.scores,
            "findings": [f.to_dict() for f in result.findings],
            "summary": result.summary,
        }
        print(json.dumps(output, indent=2))
    else:
        markdown = result.to_markdown()
        print(markdown)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        if args.json:
            import json
            output_path.write_text(json.dumps(output, indent=2))
        else:
            output_path.write_text(markdown)
        print(f"\n✓ Review saved to: {output_path}")

    # Quick stats
    print("\n" + "-" * 70)
    print("REVIEW STATS")
    print("-" * 70)
    print(f"Overall Score: {result.overall_score:.0%}")
    print(f"Total Findings: {len(result.findings)}")

    by_severity = {}
    for f in result.findings:
        by_severity[f.severity.value] = by_severity.get(f.severity.value, 0) + 1

    print("By Severity:")
    for sev in ["critical", "major", "minor", "suggestion"]:
        count = by_severity.get(sev, 0)
        if count > 0:
            print(f"  {sev}: {count}")

    print("\nDimension Scores:")
    for dim, score in sorted(result.scores.items(), key=lambda x: x[1]):
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  {dim:15} [{bar}] {score:.0%}")

    # Exit code based on critical findings
    critical = by_severity.get("critical", 0)
    if critical > 0:
        print(f"\n⚠ {critical} CRITICAL findings require attention")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
