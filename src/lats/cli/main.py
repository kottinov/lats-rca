"""Command-line interface for LATS.

This module provides the CLI entry point for running LATS from the command line.
"""

import sys
from typing import NoReturn

from lats.config.settings import Settings, get_settings
from lats.core import LanguageAgentTreeSearch
from lats.models import LATSConfig


def main() -> NoReturn:
    """Main CLI entry point.

    Usage:
        python -m lats "Your question here"
        python -m lats "Your question" --show-rollouts

    Exit codes:
        0: Success
        1: Error (invalid arguments, configuration error, etc.)
        2: Execution error (search failed, API error, etc.)
    """
    if len(sys.argv) < 2:
        print("Usage: python -m lats <question> [--show-rollouts]")
        print("Example: python -m lats 'What is Python?'")
        sys.exit(1)

    question = sys.argv[1]
    show_rollouts = "--show-rollouts" in sys.argv

    try:
        settings: Settings = get_settings()

        config = LATSConfig(
            model=settings.openai_model,
            n_candidates=settings.lats_n_candidates,
            max_depth=settings.lats_max_depth,
            exploration_weight=settings.lats_exploration_weight,
        )

        lats = LanguageAgentTreeSearch(config=config)

        print(f"Running LATS for question: {question}")
        print(f"Configuration: {config}")
        print("-" * 80)

        solution, trajectory = lats.run(question, print_rollouts=show_rollouts)

        print("-" * 80)
        print(f"Solution found at depth {solution.depth}")
        print(f"Score: {solution.reflection.score}/10")
        print(f"Reflection: {solution.reflection.reflections}")
        print(f"Is solved: {solution.is_solved}")
        print(f"Trajectory length: {len(trajectory)} messages")

        print("\nSolution messages:")
        for i, msg in enumerate(solution.messages):
            msg_type = type(msg).__name__
            content = str(msg.content)[:200]
            print(f"  {i+1}. [{msg_type}] {content}")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
