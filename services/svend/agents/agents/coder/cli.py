#!/usr/bin/env python3
"""
Coder Agent CLI

Usage:
    python -m coder.cli "Create a function that calculates fibonacci"
    python -m coder.cli --with-llm "Create a REST API endpoint"
"""

import argparse
import sys

sys.path.insert(0, '/home/eric/Desktop/agents')

from coder.agent import CodingAgent, CodingTask


def main():
    parser = argparse.ArgumentParser(description="Coding Agent with anti-drift")
    parser.add_argument("task", help="Description of what to code")
    parser.add_argument("--with-llm", action="store_true", help="Use local LLM (Qwen)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model to use")
    parser.add_argument("--constraints", nargs="*", default=[], help="Constraints to apply")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max refinement iterations")
    parser.add_argument("--project", "-p", help="Project path for codebase context")

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

    # Create agent with optional project context
    agent = CodingAgent(
        llm=llm,
        max_iterations=args.max_iterations,
        project_path=args.project
    )

    # Show context info
    if agent.project_context:
        print(f"Loaded project context: {agent.project_context.file_count} files")
        print(f"Structure:\n{agent.project_context.structure[:500]}...")

    # Create task
    task = CodingTask(
        description=args.task,
        constraints=args.constraints,
    )

    print("=" * 60)
    print("CODING AGENT")
    print("=" * 60)
    print(f"\nTask: {task.description}")
    if task.constraints:
        print(f"Constraints: {', '.join(task.constraints)}")
    print("\n" + "-" * 60)

    # Run
    result = agent.run(task)

    # Output
    print("\nGENERATED CODE:")
    print("-" * 60)
    print(result.code)
    print("-" * 60)

    print(f"\nVERIFICATION:")
    print(result.verification_summary)

    # Show Bayesian quality breakdown
    if result.quality_assessment:
        from core.reasoning import CodeReasoner
        reasoner = CodeReasoner()
        print("\nQUALITY ASSESSMENT (Bayesian):")
        print("-" * 40)
        print(reasoner.format_assessment(result.quality_assessment))

    print(f"\nINTENT ALIGNMENT: {result.intent_alignment:.0%}")
    print(f"ITERATIONS: {result.iterations}")

    if result.reasoning:
        print(f"\nREASONING:")
        print(result.reasoning[:500])

    # Show intent tracker summary
    print("\n" + "=" * 60)
    print("INTENT TRACKING:")
    print(agent.intent_tracker.summary())

    return 0 if result.verified else 1


if __name__ == "__main__":
    sys.exit(main())
