"""Constants and enums for LATS.

This module defines constants and enumerations used throughout the codebase,
following the principle of having a single source of truth for configuration values.
"""

from enum import Enum


class NodeStatus(str, Enum):
    """Status of a node in the search tree.

    Nodes can be in different states during the search process:
    - PENDING: Created but not yet expanded
    - EXPANDED: Child nodes have been generated
    - TERMINAL: Leaf node that cannot be expanded further
    - SOLVED: Node represents a solution to the problem
    """

    PENDING = "pending"
    EXPANDED = "expanded"
    TERMINAL = "terminal"
    SOLVED = "solved"


class ReflectionType(str, Enum):
    """Type of reflection performed by the agent.

    Different types of reflections provide different kinds of insights:
    - QUALITY: Assess the quality of the current trajectory
    - STRATEGY: Suggest alternative search strategies
    - ERROR: Identify potential errors or issues
    - IMPROVEMENT: Suggest specific improvements
    """

    QUALITY = "quality"
    STRATEGY = "strategy"
    ERROR = "error"
    IMPROVEMENT = "improvement"


# Default LATS configuration values
DEFAULT_MAX_DEPTH = 5
DEFAULT_N_CANDIDATES = 5
DEFAULT_EXPLORATION_WEIGHT = 1.41  # sqrt(2)
DEFAULT_MIN_VISITS_FOR_EXPANSION = 1
DEFAULT_REFLECTION_ENABLED = True

# Score ranges
MIN_SCORE = 0
MAX_SCORE = 10
SCORE_THRESHOLD_SOLVED = 9  # nodes with score >= this are considered solved

# UCT constants
UCT_UNEXPLORED_VALUE = float("inf")  # value for unexplored nodes in UCT
UCT_MIN_VISITS_FOR_UCT = 1  # minimum visits before UCT formula applies

# Logging
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "json"
