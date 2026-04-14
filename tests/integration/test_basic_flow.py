"""Basic integration test to verify the refactored code works."""

import pytest
from langchain_core.messages import AIMessage

from lats import LATSConfig, LanguageAgentTreeSearch
from lats.models import Reflection


def test_imports_work() -> None:
    """Test that all imports from refactored package work."""
    from lats import LATSConfig, LanguageAgentTreeSearch, Reflection, SearchNode

    assert LATSConfig is not None
    assert LanguageAgentTreeSearch is not None
    assert Reflection is not None
    assert SearchNode is not None


def test_config_creation() -> None:
    """Test that config can be created and validated."""
    config = LATSConfig(
        model="gpt-4o-mini",
        n_candidates=3,
        max_depth=3,
        exploration_weight=1.41,
    )

    config.validate()  # Should not raise

    assert config.model == "gpt-4o-mini"
    assert config.n_candidates == 3
    assert config.max_depth == 3


def test_search_node_creation() -> None:
    """Test that search nodes can be created."""
    from lats.models import SearchNode

    reflection = Reflection(
        reflections="Test reflection",
        evidence_quality=7,
        diagnostic_completeness=7,
        internal_consistency=7,
        found_solution=False,
    )

    node = SearchNode(
        messages=[AIMessage(content="Test message")], reflection=reflection
    )

    assert node.depth == 1
    assert node.visits > 0  # Backpropagation happened
    assert node.is_terminal is True


def test_policy_functions() -> None:
    """Test that policy functions work."""
    from lats.core.policies import select_leaf, should_continue
    from lats.models import SearchNode

    reflection = Reflection(
        reflections="Test reflection",
        evidence_quality=7,
        diagnostic_completeness=7,
        internal_consistency=7,
        found_solution=False,
    )

    root = SearchNode(
        messages=[AIMessage(content="Root")], reflection=reflection
    )

    leaf = select_leaf(root, exploration_weight=1.41)
    assert leaf is root

    decision = should_continue(root, max_depth=5)
    assert decision == "expand"

    solved_reflection = Reflection(
        reflections="Perfect",
        evidence_quality=10,
        diagnostic_completeness=10,
        internal_consistency=10,
        found_solution=True,
    )
    solved_node = SearchNode(
        messages=[AIMessage(content="Solved")], reflection=solved_reflection
    )

    decision = should_continue(solved_node, max_depth=5)
    assert decision == "__end__"


def test_lats_initialization() -> None:
    """Test that LATS can be initialized without errors."""
    config = LATSConfig(model="gpt-4o-mini", n_candidates=2, max_depth=2)

    try:
        lats = LanguageAgentTreeSearch(config=config)
        assert lats is not None
        assert lats.config == config
        assert lats.llm is not None
        assert lats.graph is not None
    except Exception as e:
        if "TAVILY_API_KEY" in str(e) or "OPENAI_API_KEY" in str(e):
            pytest.skip(f"Skipping due to missing API keys: {e}")
        else:
            raise


@pytest.mark.skip(reason="Requires real API keys and makes actual API calls")
def test_end_to_end_search() -> None:
    """End-to-end test with real LATS execution.

    This test is skipped by default because it requires real API keys
    and makes actual LLM API calls. Run manually with:
        pytest tests/integration/test_basic_flow.py::test_end_to_end_search -v
    """
    config = LATSConfig(
        model="gpt-4o-mini", n_candidates=2, max_depth=2
    )

    lats = LanguageAgentTreeSearch(config=config)

    solution, trajectory = lats.run("What is 2+2?", print_rollouts=True)

    assert solution is not None
    assert solution.depth > 0
    assert len(trajectory) > 0
    assert solution.reflection.score >= 0
