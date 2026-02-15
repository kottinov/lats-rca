"""Policy functions for LATS tree search.

This module contains pure functions that implement the tree search policies,
following functional programming principles for testability and clarity.

The main policies are:
- select_leaf: UCT-based selection for choosing which node to expand
- should_continue: Termination condition for the search

These functions are extracted from the main search class to make them
easily testable and to separate the algorithm logic from the orchestration.
"""

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lats.models.node import SearchNode

Decision = Literal["expand", "__end__"]


def select_leaf(root: "SearchNode", exploration_weight: float) -> "SearchNode":
    """Select the best leaf node using Upper Confidence Bound for Trees (UCT).

    This function implements the selection phase of Monte Carlo Tree Search,
    traversing from the root to a leaf node by always choosing the child
    with the highest UCT value.

    The UCT formula balances exploitation (choosing high-value nodes) with
    exploration (visiting less-explored nodes):

        UCT(node) = avg_reward + C * sqrt(ln(parent_visits) / node_visits)

    Where:
    - avg_reward: Average reward from all simulations through this node
    - C: Exploration weight (typically sqrt(2) ≈ 1.41)
    - parent_visits: Number of times the parent has been visited
    - node_visits: Number of times this node has been visited

    The exploration term encourages visiting less-explored nodes, while
    the exploitation term favors nodes with high average rewards.

    Args:
        root: Root node to start selection from
        exploration_weight: UCT exploration weight (C parameter).
                          Higher values encourage more exploration.
                          Typical value is sqrt(2) ≈ 1.41.

    Returns:
        The selected leaf node (a node with no children)

    Example:
        >>> root = SearchNode(...)
        >>> leaf = select_leaf(root, exploration_weight=1.41)
        >>> assert leaf.is_terminal  # Leaf has no children

    Note:
        This is a pure function with no side effects. It only traverses
        the tree and returns a reference to an existing node.
    """
    node = root

    while node.children:
        node = max(
            node.children,
            key=lambda child: child.upper_confidence_bound(
                exploration_weight=exploration_weight
            ),
        )

    return node


def should_continue(root: "SearchNode", max_depth: int) -> Decision:
    """Determine whether the search should continue or terminate.

    This function implements the termination conditions for LATS:
    1. Stop if a solution has been found (root.is_solved = True)
    2. Stop if maximum tree depth has been reached
    3. Otherwise, continue expanding

    Args:
        root: Root node of the search tree
        max_depth: Maximum allowed tree depth

    Returns:
        "expand" if search should continue, or "__end__" if it should terminate

    Example:
        >>> root = SearchNode(...)
        >>> decision = should_continue(root, max_depth=5)
        >>> if decision == "expand":
        ...     # Continue search
        >>> else:
        ...     # Terminate and extract solution

    Note:
        The return value is a Literal type that matches LangGraph's
        conditional edge routing system, where "__end__" is a special
        value indicating graph termination.
    """
    if root.is_solved:
        return "__end__"

    if root.height >= max_depth:
        return "__end__"

    return "expand"
