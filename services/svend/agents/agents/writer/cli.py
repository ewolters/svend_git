#!/usr/bin/env python3
"""
Writer Agent CLI

Usage:
    python -m writer.cli "AI in Healthcare" --type technical_report
    python -m writer.cli "CRISPR Applications" --type grant_proposal --tone formal
    python -m writer.cli "Market Opportunity" --type executive_summary --length brief
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/eric/Desktop/agents')

from writer.agent import WriterAgent, DocumentRequest, DocumentType


def main():
    parser = argparse.ArgumentParser(
        description="Writer Agent - Generate structured documents"
    )
    parser.add_argument("topic", help="Document topic")
    parser.add_argument(
        "--type", "-t",
        choices=[t.value for t in DocumentType],
        default="technical_report",
        help="Document type"
    )
    parser.add_argument(
        "--tone",
        choices=["formal", "casual", "academic", "business"],
        default="formal",
        help="Writing tone"
    )
    parser.add_argument(
        "--length", "-l",
        choices=["brief", "standard", "detailed"],
        default="standard",
        help="Document length"
    )
    parser.add_argument(
        "--audience", "-a",
        default="general",
        help="Target audience"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (markdown)"
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Use local LLM for generation"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to use"
    )
    parser.add_argument(
        "--notes", "-n",
        nargs="*",
        default=[],
        help="Additional notes to include"
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        help="Custom section titles (overrides template)"
    )

    args = parser.parse_args()

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
            print("Continuing with mock generation...")

    # Create agent
    agent = WriterAgent(llm=llm)

    # Create request
    doc_type = DocumentType(args.type)
    request = DocumentRequest(
        topic=args.topic,
        doc_type=doc_type,
        tone=args.tone,
        length=args.length,
        audience=args.audience,
        notes=args.notes,
        outline=args.sections if args.sections else [],
    )

    print("=" * 70)
    print("WRITER AGENT")
    print("=" * 70)
    print(f"\nTopic: {request.topic}")
    print(f"Type: {request.doc_type.value}")
    print(f"Tone: {request.tone}")
    print(f"Length: {request.length}")
    print(f"Audience: {request.audience}")
    print("\n" + "-" * 70)
    print("Generating document...")

    # Generate
    document = agent.write(request)

    # Output
    print("\n" + "=" * 70)
    print("GENERATED DOCUMENT")
    print("=" * 70)

    markdown = document.to_markdown()
    print(markdown)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(markdown)
        print(f"\nSaved to: {output_path}")

    # Stats
    print("\n" + "-" * 70)
    print("DOCUMENT STATS")
    print("-" * 70)
    print(f"Title: {document.title}")
    print(f"Sections: {len(document.sections)}")
    print(f"Word count: {document.word_count}")
    print(f"Citations: {len(document.citations)}")

    # Intent tracking
    print("\n" + "=" * 70)
    print("INTENT TRACKING")
    print(agent.intent_tracker.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
