"""Unit tests for SearchNode model."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from lats.models.node import SearchNode
from lats.models.reflection import Reflection


@pytest.fixture
def sample_reflection() -> Reflection:
    """Provide a sample reflection for testing."""
    return Reflection(
        reflections="Good starting point",
        evidence_quality=7,
        diagnostic_completeness=7,
        internal_consistency=7,
        found_solution=False,
    )


@pytest.fixture
def root_node(sample_reflection: Reflection) -> SearchNode:
    """Provide a root node for testing."""
    return SearchNode(
        messages=[AIMessage(content="Hello")], reflection=sample_reflection
    )


class TestSearchNode:
    """Tests for SearchNode dataclass."""

    def test_create_root_node(self, sample_reflection: Reflection) -> None:
        """Test creating a root node."""
        node = SearchNode(
            messages=[AIMessage(content="Test")], reflection=sample_reflection
        )

        assert node.depth == 1
        assert node.parent is None
        assert len(node.children) == 0
        assert node.is_terminal is True
        assert node.height == 1

    def test_create_child_node(
        self, root_node: SearchNode, sample_reflection: Reflection
    ) -> None:
        """Test creating a child node."""
        child = SearchNode(
            messages=[AIMessage(content="Child")],
            reflection=sample_reflection,
            parent=root_node,
        )

        assert child.depth == 2
        assert child.parent is root_node
        assert child.is_terminal is True

    def test_node_properties(
        self, root_node: SearchNode, sample_reflection: Reflection
    ) -> None:
        """Test node properties."""
        # Root node
        assert root_node.is_terminal is True
        assert root_node.is_solved is False
        assert root_node.height == 1

        # Add a child
        child = SearchNode(
            messages=[AIMessage(content="Child")],
            reflection=sample_reflection,
            parent=root_node,
        )
        root_node.children.append(child)

        # Root is no longer terminal
        assert root_node.is_terminal is False
        assert root_node.height == 2

    def test_backpropagation(self, sample_reflection: Reflection) -> None:
        """Test that backpropagation updates ancestors."""
        # Create a chain: root -> child -> grandchild
        root = SearchNode(
            messages=[AIMessage(content="Root")], reflection=sample_reflection
        )

        child = SearchNode(
            messages=[AIMessage(content="Child")],
            reflection=sample_reflection,
            parent=root,
        )

        grandchild = SearchNode(
            messages=[AIMessage(content="Grandchild")],
            reflection=sample_reflection,
            parent=child,
        )

        # All nodes should have been visited during backpropagation
        assert root.visits > 0
        assert child.visits > 0
        assert grandchild.visits > 0

    def test_get_messages(self, root_node: SearchNode) -> None:
        """Test getting messages with and without reflections."""
        messages_with = root_node.get_messages(include_reflections=True)
        messages_without = root_node.get_messages(include_reflections=False)

        assert len(messages_with) == 2  # Original message + reflection
        assert len(messages_without) == 1  # Just original message

        assert isinstance(messages_with[-1], HumanMessage)  # Reflection as message
        assert isinstance(messages_without[0], AIMessage)  # Original message

    def test_get_trajectory(
        self, root_node: SearchNode, sample_reflection: Reflection
    ) -> None:
        """Test getting full trajectory from root to leaf."""
        child = SearchNode(
            messages=[AIMessage(content="Child")],
            reflection=sample_reflection,
            parent=root_node,
        )

        grandchild = SearchNode(
            messages=[AIMessage(content="Grandchild")],
            reflection=sample_reflection,
            parent=child,
        )

        trajectory = grandchild.get_trajectory(include_reflections=False)

        # Should contain messages from root -> child -> grandchild
        assert len(trajectory) == 3
        assert trajectory[0].content == "Hello"
        assert trajectory[1].content == "Child"
        assert trajectory[2].content == "Grandchild"

    def test_uct_calculation(
        self, root_node: SearchNode, sample_reflection: Reflection
    ) -> None:
        """Test UCT value calculation."""
        child = SearchNode(
            messages=[AIMessage(content="Child")],
            reflection=sample_reflection,
            parent=root_node,
        )
        root_node.children.append(child)

        # Child should be able to compute UCT
        uct = child.upper_confidence_bound(exploration_weight=1.41)
        assert uct > 0

    def test_uct_root_raises_error(self, root_node: SearchNode) -> None:
        """Test that computing UCT for root raises error."""
        with pytest.raises(ValueError, match="Cannot compute UCT for the root"):
            root_node.upper_confidence_bound()

    def test_solved_propagation(self, sample_reflection: Reflection) -> None:
        """Test that solved status propagates to ancestors."""
        root = SearchNode(
            messages=[AIMessage(content="Root")], reflection=sample_reflection
        )

        child = SearchNode(
            messages=[AIMessage(content="Child")],
            reflection=sample_reflection,
            parent=root,
        )

        # Create a solved grandchild
        solved_reflection = Reflection(
            reflections="Perfect answer",
            evidence_quality=10,
            diagnostic_completeness=10,
            internal_consistency=10,
            found_solution=True,
        )
        grandchild = SearchNode(
            messages=[AIMessage(content="Solved!")],
            reflection=solved_reflection,
            parent=child,
        )

        # All ancestors should be marked as solved
        assert grandchild.is_solved is True
        assert child.is_solved is True
        assert root.is_solved is True

    def test_get_best_solution(self, sample_reflection: Reflection) -> None:
        """Test finding the best solution in the tree."""
        root = SearchNode(
            messages=[AIMessage(content="Root")], reflection=sample_reflection
        )

        high_score = Reflection(
            reflections="Great",
            evidence_quality=9,
            diagnostic_completeness=9,
            internal_consistency=9,
            found_solution=True,
        )
        low_score = Reflection(
            reflections="Okay",
            evidence_quality=5,
            diagnostic_completeness=5,
            internal_consistency=5,
            found_solution=False,
        )

        child1 = SearchNode(
            messages=[AIMessage(content="Good")],
            reflection=high_score,
            parent=root,
        )
        root.children.append(child1)

        child2 = SearchNode(
            messages=[AIMessage(content="Bad")],
            reflection=low_score,
            parent=root,
        )
        root.children.append(child2)

        best = root.get_best_solution()

        # Should prefer the solved node with higher score
        assert best.reflection.score == 9
        assert best.is_solved is True
