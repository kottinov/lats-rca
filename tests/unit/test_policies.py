"""Unit tests for policy functions."""

import pytest
from langchain_core.messages import AIMessage

from lats.core.policies import select_leaf, should_continue
from lats.models.node import SearchNode
from lats.models.reflection import Reflection


@pytest.fixture
def sample_reflection() -> Reflection:
    """Provide a sample reflection for testing."""
    return Reflection(
        reflections="Good starting point", score=7, found_solution=False
    )


@pytest.fixture
def tree_with_children(sample_reflection: Reflection) -> SearchNode:
    """Create a tree with multiple children for testing selection."""
    root = SearchNode(
        messages=[AIMessage(content="Root")], reflection=sample_reflection
    )

    # Create children with different values
    for i in range(3):
        child = SearchNode(
            messages=[AIMessage(content=f"Child {i}")],
            reflection=sample_reflection,
            parent=root,
        )
        root.children.append(child)

        # Give different values to test UCT selection
        child.value = float(i)
        child.visits = i + 1

    return root


class TestSelectLeaf:
    """Tests for select_leaf policy function."""

    def test_select_leaf_terminal(self, sample_reflection: Reflection) -> None:
        """Test selecting leaf from a terminal node."""
        root = SearchNode(
            messages=[AIMessage(content="Root")], reflection=sample_reflection
        )

        leaf = select_leaf(root, exploration_weight=1.41)

        assert leaf is root
        assert leaf.is_terminal is True

    def test_select_leaf_with_children(
        self, tree_with_children: SearchNode
    ) -> None:
        """Test selecting leaf with children."""
        leaf = select_leaf(tree_with_children, exploration_weight=1.41)

        # Should select one of the children (they're all leaves)
        assert leaf.parent is tree_with_children
        assert leaf.is_terminal is True

    def test_select_leaf_deep_tree(self, sample_reflection: Reflection) -> None:
        """Test selecting leaf in a deep tree."""
        # Create a chain: root -> child -> grandchild
        root = SearchNode(
            messages=[AIMessage(content="Root")], reflection=sample_reflection
        )

        child = SearchNode(
            messages=[AIMessage(content="Child")],
            reflection=sample_reflection,
            parent=root,
        )
        root.children.append(child)

        grandchild = SearchNode(
            messages=[AIMessage(content="Grandchild")],
            reflection=sample_reflection,
            parent=child,
        )
        child.children.append(grandchild)

        leaf = select_leaf(root, exploration_weight=1.41)

        # Should traverse to the deepest leaf
        assert leaf is grandchild
        assert leaf.depth == 3


class TestShouldContinue:
    """Tests for should_continue policy function."""

    def test_continue_when_not_solved(self, sample_reflection: Reflection) -> None:
        """Test that search continues when not solved."""
        root = SearchNode(
            messages=[AIMessage(content="Root")], reflection=sample_reflection
        )

        decision = should_continue(root, max_depth=5)

        assert decision == "expand"

    def test_stop_when_solved(self, sample_reflection: Reflection) -> None:
        """Test that search stops when solution found."""
        solved_reflection = Reflection(
            reflections="Perfect", score=10, found_solution=True
        )
        root = SearchNode(
            messages=[AIMessage(content="Solved")], reflection=solved_reflection
        )

        decision = should_continue(root, max_depth=5)

        assert decision == "__end__"

    def test_stop_at_max_depth(self, sample_reflection: Reflection) -> None:
        """Test that search stops at max depth."""
        # Create a tree with depth = max_depth
        root = SearchNode(
            messages=[AIMessage(content="Root")], reflection=sample_reflection
        )

        # Add children to reach max depth
        child = SearchNode(
            messages=[AIMessage(content="Child")],
            reflection=sample_reflection,
            parent=root,
        )
        root.children.append(child)

        grandchild = SearchNode(
            messages=[AIMessage(content="Grandchild")],
            reflection=sample_reflection,
            parent=child,
        )
        child.children.append(grandchild)

        # Tree height is 3, so max_depth=2 should stop
        decision = should_continue(root, max_depth=2)

        assert decision == "__end__"

    def test_continue_below_max_depth(
        self, sample_reflection: Reflection
    ) -> None:
        """Test that search continues below max depth."""
        root = SearchNode(
            messages=[AIMessage(content="Root")], reflection=sample_reflection
        )

        child = SearchNode(
            messages=[AIMessage(content="Child")],
            reflection=sample_reflection,
            parent=root,
        )
        root.children.append(child)

        # Tree height is 2, max_depth=5 should continue
        decision = should_continue(root, max_depth=5)

        assert decision == "expand"
