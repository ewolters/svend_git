#!/usr/bin/env python3
"""
Business Plan Guide CLI

Interactive interview-style guide for creating business plans.

Usage:
    python -m guide.cli
    python -m guide.cli --output my_business_plan.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/eric/Desktop/agents')

from guide.business_plan import BusinessPlanGuide


def print_header():
    """Print welcome header."""
    print("=" * 70)
    print("SVEND BUSINESS PLAN GUIDE")
    print("=" * 70)
    print()
    print("This guide will help you create a comprehensive business plan")
    print("by asking you structured questions about your business.")
    print()
    print("Commands:")
    print("  - Type your answer and press Enter")
    print("  - Type 'skip' to skip optional questions")
    print("  - Type 'back' to go back (not implemented yet)")
    print("  - Type 'quit' to exit (progress will be lost)")
    print("  - Type 'progress' to see your progress")
    print()
    print("-" * 70)


def print_section_header(section):
    """Print section header."""
    print()
    print("=" * 70)
    print(f"SECTION: {section.title.upper()}")
    print("-" * 70)
    print(section.description)
    print("=" * 70)
    print()


def print_progress(guide):
    """Print progress bar and stats."""
    progress = guide.get_progress()
    pct = progress["percent_complete"]
    filled = int(pct / 5)  # 20 char bar
    bar = "█" * filled + "░" * (20 - filled)

    print()
    print(f"Progress: [{bar}] {pct:.0f}%")
    print(f"Section {progress['current_section']}/{progress['total_sections']} | "
          f"Questions: {progress['answered']}/{progress['total_questions']} answered, "
          f"{progress['skipped']} skipped")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Business Plan Guide"
    )
    parser.add_argument(
        "--output", "-o",
        default="business_plan.md",
        help="Output file path for the generated business plan"
    )

    args = parser.parse_args()

    # Initialize guide
    guide = BusinessPlanGuide()

    print_header()
    input("Press Enter to begin...")

    # Start interview
    question = guide.start()
    current_section = None

    while question:
        # Check if we've moved to a new section
        section = guide.get_current_section()
        if section and section != current_section:
            print_section_header(section)
            current_section = section

        # Display question
        print()
        print(guide.format_question(question))
        print()

        # Get input
        try:
            answer = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            return 1

        # Handle commands
        if answer.lower() == 'quit':
            confirm = input("Are you sure you want to quit? Progress will be lost. (y/n): ")
            if confirm.lower() in ['y', 'yes']:
                print("Goodbye!")
                return 0
            continue

        if answer.lower() == 'progress':
            print_progress(guide)
            continue

        if answer.lower() == 'skip':
            success, message, question = guide.skip_question()
            if not success:
                print(f"⚠ {message}")
            else:
                print("✓ Skipped")
            continue

        # Submit answer
        success, message, question = guide.submit_answer(answer)

        if not success:
            print(f"⚠ {message}")
            # question stays the same for retry
        else:
            print("✓ Recorded")

    # Interview complete
    print()
    print("=" * 70)
    print("INTERVIEW COMPLETE!")
    print("=" * 70)
    print()

    print_progress(guide)

    # Generate business plan
    print("Generating your business plan...")
    print()

    markdown = guide.to_business_plan_markdown()

    # Preview
    print("-" * 70)
    print("PREVIEW (first 50 lines):")
    print("-" * 70)
    preview_lines = markdown.split("\n")[:50]
    print("\n".join(preview_lines))
    if len(markdown.split("\n")) > 50:
        print("\n... (truncated)")
    print("-" * 70)

    # Save
    output_path = Path(args.output)
    output_path.write_text(markdown)
    print(f"\n✓ Business plan saved to: {output_path}")

    # Also save JSON
    result = guide.synthesize()
    json_path = output_path.with_suffix(".json")
    import json
    json_path.write_text(json.dumps(result.synthesized_output, indent=2))
    print(f"✓ Structured data saved to: {json_path}")

    print()
    print("Thank you for using SVEND Business Plan Guide!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
