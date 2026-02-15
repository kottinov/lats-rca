"""Configuration constants for multi-agent RCA.

This module centralizes all configuration values used by the multi-agent
system, following the Robust Python principle of avoiding magic numbers
and making configuration explicit and discoverable.
"""

from dataclasses import dataclass

from lats.models.agent import AgentCompleteness, AgentConfidence

DEFAULT_CONFIDENCE_THRESHOLD = AgentConfidence(0.7)
"""Default minimum confidence before escalating to additional agents."""

DEFAULT_COMPLETENESS_THRESHOLD = AgentCompleteness(0.6)
"""Default minimum completeness before escalating to additional agents."""

MAX_EVIDENCE_ITEMS = 10
"""Maximum number of evidence items to consider in agent scoring.

This cap prevents unbounded evidence accumulation from skewing
agent confidence scores disproportionately.
"""

STRONG_CORRELATION_THRESHOLD = 6
"""Minimum token overlap for strong correlation between agent findings."""

WEAK_CORRELATION_THRESHOLD = 2
"""Minimum token overlap for weak correlation between agent findings."""

CORRELATION_STOP_WORDS = frozenset({
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "agent",
    "summary",
    "reflection",
    "confidence",
})
"""Words to ignore when computing correlation between agent summaries.

Using frozenset ensures immutability and communicates intent that this
set should not be modified at runtime.
"""

ERROR_SIGNALS = frozenset({
    "error",
    "failure",
    "timeout",
    "incident",
    "degraded",
    "exception",
    "fault",
})
"""Keywords indicating error or failure conditions."""

NORMAL_SIGNALS = frozenset({
    "normal",
    "healthy",
    "stable",
    "no anomaly",
    "no issue",
    "nominal",
})
"""Keywords indicating normal operational state."""


@dataclass(frozen=True, slots=True)
class HandoffThresholds:
    """Thresholds for deciding whether escalation is needed.

    This frozen dataclass encapsulates the decision criteria for when
    a subagent should escalate to the supervisor for additional investigation.

    Attributes:
        confidence: Minimum confidence level (0.0-1.0)
        completeness: Minimum completeness level (0.0-1.0)

    Example:
        >>> thresholds = HandoffThresholds(
        ...     confidence=AgentConfidence(0.8),
        ...     completeness=AgentCompleteness(0.7)
        ... )
        >>> thresholds.confidence
        AgentConfidence(0.8)
    """

    confidence: AgentConfidence = DEFAULT_CONFIDENCE_THRESHOLD
    completeness: AgentCompleteness = DEFAULT_COMPLETENESS_THRESHOLD

    def __post_init__(self) -> None:
        """Validate threshold values.

        Raises:
            ValueError: If confidence or completeness is out of range [0.0, 1.0]
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence threshold must be in [0.0, 1.0], got {self.confidence}"
            )

        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError(
                f"Completeness threshold must be in [0.0, 1.0], got {self.completeness}"
            )

    def should_escalate(
        self,
        confidence: AgentConfidence,
        completeness: AgentCompleteness,
    ) -> bool:
        """Determine if metrics warrant escalation to supervisor.

        Args:
            confidence: Agent's confidence in its findings
            completeness: Agent's assessment of investigation completeness

        Returns:
            True if escalation is warranted, False otherwise

        Example:
            >>> thresholds = HandoffThresholds()
            >>> thresholds.should_escalate(
            ...     AgentConfidence(0.5),
            ...     AgentCompleteness(0.8)
            ... )
            True
        """
        return (
            confidence < self.confidence
            or completeness < self.completeness
        )