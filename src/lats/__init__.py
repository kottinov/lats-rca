"""LATS: Language Agent Tree Search.

A production-grade implementation of Language Agent Tree Search combining
Monte Carlo Tree Search with language model reflection and self-improvement.

Example:
    >>> from lats import LanguageAgentTreeSearch, LATSConfig
    >>> config = LATSConfig(model="gpt-4o", n_candidates=5, max_depth=5)
    >>> lats = LanguageAgentTreeSearch(config=config)
    >>> solution, trajectory = lats.run("What is Python?")
    >>> print(solution.reflection.score)
    9
"""

from lats.core import LanguageAgentTreeSearch, select_leaf, should_continue
from lats.exceptions import LATSConfigError
from lats.models import (
    AgentCompleteness,
    AgentConfidence,
    AgentRunResult,
    CorrelationLabel,
    EvidenceCount,
    LATSConfig,
    Reflection,
    SearchNode,
    SupervisorResult,
    TreeState,
)
from lats.orchestration import RCASupervisor

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "AgentCompleteness",
    "AgentConfidence",
    "AgentRunResult",
    "CorrelationLabel",
    "EvidenceCount",
    "LATSConfigError",
    "LanguageAgentTreeSearch",
    "LATSConfig",
    "RCASupervisor",
    "SearchNode",
    "Reflection",
    "SupervisorResult",
    "TreeState",
    "select_leaf",
    "should_continue",
]
