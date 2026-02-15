"""Search node model for LATS tree.

This module defines the SearchNode dataclass that represents a node
in the Monte Carlo Tree Search, containing messages, reflections,
and tree structure information.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage

from lats.core.types import Depth, RewardValue, UCTValue, VisitCount
from lats.models.reflection import Reflection


@dataclass
class SearchNode:
    """A node in the LATS search tree.

    Each node represents a state in the search, containing:
    - Messages: The conversation history at this state
    - Reflection: Self-critique of this state
    - Tree structure: Parent and children relationships
    - MCTS statistics: Value and visit count for UCT

    Attributes:
        messages: Conversation messages at this node
        reflection: Self-critique reflection for this node
        parent: Parent node (None for root)
        children: Child nodes generated from this node
        value: Cumulative reward value (updated via backpropagation)
        visits: Number of times this node has been visited
        depth: Depth in the tree (1 for root, increments for children)

    Example:
        >>> from langchain_core.messages import AIMessage
        >>> reflection = Reflection(reflections="Good start", score=7, found_solution=False)
        >>> node = SearchNode(messages=[AIMessage(content="Hello")], reflection=reflection)
        >>> node.depth
        1
        >>> node.is_terminal
        True
    """

    messages: list[BaseMessage]
    reflection: Reflection
    parent: SearchNode | None = None
    children: list[SearchNode] = field(default_factory=list)
    value: float = 0.0
    visits: int = 0
    depth: int = field(init=False)
    _is_solved: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Initialize computed fields and perform backpropagation.

        This method:
        1. Computes the node's depth based on parent
        2. Marks the node as solved if reflection indicates solution
        3. Propagates solved status to ancestors
        4. Backpropagates the reflection score
        """
        self.depth = self.parent.depth + 1 if self.parent else 1
        self._is_solved = self.reflection.found_solution

        if self._is_solved:
            self._mark_ancestors_solved()

        self.backpropagate(self.reflection.normalized_score)

    @property
    def is_solved(self) -> bool:
        """Return whether this node represents a solved state.

        A node is solved if either:
        - Its reflection indicates a solution was found, or
        - One of its descendants found a solution (propagated up)

        Returns:
            True if this node or any descendant found a solution
        """
        return self._is_solved

    @property
    def is_terminal(self) -> bool:
        """Return whether this is a terminal (leaf) node.

        Returns:
            True if this node has no children
        """
        return not self.children

    @property
    def height(self) -> int:
        """Return the height of the subtree rooted at this node.

        Height is defined as:
        - 1 for leaf nodes
        - 1 + max(child heights) for internal nodes

        Returns:
            Height of the subtree

        Example:
            >>> node = SearchNode(...)
            >>> node.height  # Leaf node
            1
            >>> child = SearchNode(parent=node, ...)
            >>> node.height  # Parent of leaf
            2
        """
        if not self.children:
            return 1
        return 1 + max(child.height for child in self.children)

    @property
    def depth_typed(self) -> Depth:
        """Return depth as a domain type.

        Returns:
            Typed depth value
        """
        return Depth(self.depth)

    @property
    def visits_typed(self) -> VisitCount:
        """Return visits as a domain type.

        Returns:
            Typed visit count
        """
        return VisitCount(self.visits)

    def upper_confidence_bound(self, exploration_weight: float = 1.0) -> UCTValue:
        """Calculate the Upper Confidence Bound (UCT) value for this node.

        The UCT formula balances exploitation and exploration:

            UCT = avg_reward + C * sqrt(ln(parent_visits) / node_visits)

        Where:
        - avg_reward = value / visits (exploitation)
        - C = exploration_weight (typically sqrt(2) ≈ 1.41)
        - The second term encourages exploring less-visited nodes

        Args:
            exploration_weight: Weight for the exploration term (C parameter)

        Returns:
            UCT value for this node

        Raises:
            ValueError: If called on root node (which has no parent)

        Example:
            >>> node = SearchNode(parent=parent, ...)
            >>> uct = node.upper_confidence_bound(exploration_weight=1.41)
            >>> # Higher UCT = this node should be explored next
        """
        if self.parent is None:
            raise ValueError("Cannot compute UCT for the root node")

        if self.visits == 0:
            # unexplored nodes: return current value (encourages exploration)
            return UCTValue(self.value)

        # exploitation term: average reward
        average_reward = self.value / self.visits

        # exploration term: sqrt(ln(N) / n)
        exploration = math.sqrt(math.log(self.parent.visits) / self.visits)

        return UCTValue(average_reward + exploration_weight * exploration)

    def backpropagate(self, reward: float) -> None:
        """Propagate reward value up the tree to all ancestors.

        This method updates the value and visit count for this node and
        all ancestors. The value is updated using incremental averaging
        to maintain numerical stability.

        Args:
            reward: Reward value to propagate (typically normalized score)

        Example:
            >>> node = SearchNode(...)
            >>> node.backpropagate(0.8)
            >>> # Updates this node and all ancestors
        """
        reward_typed = RewardValue(reward)
        node: SearchNode | None = self

        while node is not None:
            node.visits += 1
            # incremental average: new_avg = old_avg + (new_value - old_avg) / n
            # equivalent to: (old_avg * (n-1) + new_value) / n
            node.value = (node.value * (node.visits - 1) + reward_typed) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages for this node, optionally including reflection.

        Args:
            include_reflections: Whether to include reflection as a message

        Returns:
            List of messages for this node

        Example:
            >>> node = SearchNode(...)
            >>> messages = node.get_messages(include_reflections=True)
            >>> # Returns: [message1, message2, reflection_message]
        """
        if include_reflections:
            return [*self.messages, self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get the full trajectory from root to this node.

        The trajectory is the sequence of all messages from the root node
        to this node, following parent pointers.

        Args:
            include_reflections: Whether to include reflections in the trajectory

        Returns:
            List of all messages from root to this node

        Example:
            >>> leaf = SearchNode(...)
            >>> trajectory = leaf.get_trajectory()
            >>> # Returns messages from root -> parent -> ... -> leaf
        """
        trajectory: list[BaseMessage] = []
        node: SearchNode | None = self

        while node is not None:
            node_messages = node.get_messages(include_reflections=include_reflections)
            trajectory.extend(node_messages[::-1])
            node = node.parent

        return trajectory[::-1]

    def get_best_solution(self) -> SearchNode:
        """Find the best solution node in the tree.

        The best solution is defined as the terminal node with the highest
        value among all nodes marked as solved. If no solved nodes exist,
        returns the terminal node with the highest value.

        Returns:
            The best solution node

        Example:
            >>> root = SearchNode(...)
            >>> # After expanding the tree...
            >>> best = root.get_best_solution()
            >>> assert best.is_terminal
        """
        all_nodes = [self, *self._get_all_children()]

        return max(
            all_nodes,
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )

    def _get_all_children(self) -> list[SearchNode]:
        """Get all descendant nodes using breadth-first search.

        Returns:
            List of all descendant nodes

        Example:
            >>> root = SearchNode(...)
            >>> all_children = root._get_all_children()
            >>> # Returns all nodes in the tree except root
        """
        all_nodes: list[SearchNode] = []
        queue: deque[SearchNode] = deque([self])

        while queue:
            node = queue.popleft()
            all_nodes.extend(node.children)
            queue.extend(node.children)

        return all_nodes

    def _mark_ancestors_solved(self) -> None:
        """Mark all ancestor nodes as solved.

        When a node finds a solution, we propagate this information
        up to all ancestors so they also know a solution exists in
        their subtree.
        """
        node = self.parent
        while node is not None:
            node._is_solved = True
            node = node.parent
