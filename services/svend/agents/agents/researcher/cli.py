#!/usr/bin/env python3
"""
Research Agent CLI

Usage:
    python -m researcher.cli "What is the current state of quantum computing?"
    python -m researcher.cli --focus scientific "CRISPR gene editing applications"
    python -m researcher.cli --focus market --depth thorough "Electric vehicle market"
    python -m researcher.cli --output report.md "Climate change mitigation strategies"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/eric/Desktop/agents')

from researcher.agent import ResearchAgent, ResearchQuery


def main():
    parser = argparse.ArgumentParser(
        description="Research Agent - Thorough multi-source research"
    )
    parser.add_argument("question", help="Research question to investigate")
    parser.add_argument(
        "--focus", "-f",
        choices=["scientific", "market", "general"],
        default="general",
        help="Research focus area"
    )
    parser.add_argument(
        "--depth", "-d",
        choices=["quick", "standard", "thorough"],
        default="standard",
        help="Research depth (affects source count and iterations)"
    )
    parser.add_argument(
        "--max-sources", "-s",
        type=int,
        default=10,
        help="Maximum number of sources"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (markdown)"
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Use local LLM for synthesis"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to use for synthesis"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of markdown"
    )
    parser.add_argument(
        "--brave-key",
        help="Brave Search API key (or set BRAVE_API_KEY env var)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock search instead of real APIs"
    )

    args = parser.parse_args()

    # Get Brave API key from args, environment, or .env file
    import os
    from pathlib import Path

    brave_key = args.brave_key or os.environ.get("BRAVE_API_KEY")

    # Try loading from .env if not set
    if not brave_key:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("BRAVE_API_KEY="):
                    brave_key = line.split("=", 1)[1].strip()
                    break

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
            print("Continuing with mock synthesis...")

    # Create agent
    agent = ResearchAgent(
        llm=llm,
        brave_api_key=brave_key,
        use_real_search=not args.mock
    )

    # Create query
    query = ResearchQuery(
        question=args.question,
        depth=args.depth,
        focus=args.focus,
        max_sources=args.max_sources,
    )

    print("=" * 70)
    print("RESEARCH AGENT")
    print("=" * 70)
    print(f"\nQuestion: {query.question}")
    print(f"Focus: {query.focus}")
    print(f"Depth: {query.depth}")

    # Show search mode
    if args.mock:
        print("Search: Mock (for testing)")
    else:
        sources = ["Semantic Scholar", "arXiv"]
        if brave_key:
            sources.append("Brave Search")
        else:
            sources.append("DuckDuckGo")
        print(f"Search: {', '.join(sources)}")

    print("\n" + "-" * 70)
    print("Researching...")

    # Run research
    findings = agent.run(query)

    # Output
    print("\n" + "=" * 70)
    print("RESEARCH FINDINGS")
    print("=" * 70)

    if args.json:
        import json
        output = {
            "query": findings.query,
            "summary": findings.summary,
            "sections": findings.sections,
            "sources": [
                {
                    "id": s.id,
                    "title": s.title,
                    "url": s.url,
                    "type": s.source_type.value,
                    "credibility": s.credibility_score,
                }
                for s in findings.sources
            ],
            "confidence": findings.confidence,
            "gaps": findings.gaps,
        }
        print(json.dumps(output, indent=2))
    else:
        # Markdown output
        markdown = findings.to_markdown()
        print(markdown)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        if args.json:
            import json
            output_path.write_text(json.dumps(output, indent=2))
        else:
            output_path.write_text(markdown)
        print(f"\nSaved to: {output_path}")

    # Summary stats
    print("\n" + "-" * 70)
    print("RESEARCH STATS")
    print("-" * 70)
    print(f"Sources found: {len(findings.sources)}")

    # Source type breakdown
    type_counts = {}
    for s in findings.sources:
        t = s.source_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    print("Source breakdown:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    print(f"\nDiversity score: {agent.source_manager.get_diversity_score():.0%}")
    print(f"Credibility score: {agent.source_manager.get_credibility_score():.0%}")
    print(f"Overall confidence: {findings.confidence:.0%}")

    if findings.gaps:
        print(f"\nResearch gaps identified: {len(findings.gaps)}")
        for gap in findings.gaps:
            print(f"  - {gap}")

    # Intent tracking
    print("\n" + "=" * 70)
    print("INTENT TRACKING")
    print(agent.intent_tracker.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
