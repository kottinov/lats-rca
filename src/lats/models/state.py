"""State model for LATS LangGraph execution.

This module defines the TreeState TypedDict used by LangGraph to
manage the search tree state during execution.
"""

from typing_extensions import TypedDict

from lats.models.node import SearchNode


class TreeState(TypedDict):
    """State maintained during LATS graph execution.

    This TypedDict defines the structure of the state passed between
    nodes in the LangGraph execution graph.

    Attributes:
        root: Root node of the search tree
        input: User's input question or task

    Example:
        >>> state: TreeState = {"root": root_node, "input": "What is Python?"}
        >>> # Pass to LangGraph nodes
    """

    root: SearchNode
    input: str
