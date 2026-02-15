"""Agent result models for multi-agent RCA orchestration.

This module defines the data models used for communication between
specialized LATS subagents and the supervisor orchestrator. Following
Robust Python principles, we use frozen dataclasses for immutability
and NewType for domain-specific type safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

AgentConfidence = NewType("AgentConfidence", float)
"""Confidence score from agent reflection (0.0-1.0 range)."""

AgentCompleteness = NewType("AgentCompleteness", float)
"""Completeness score indicating investigation depth (0.0-1.0 range)."""

EvidenceCount = NewType("EvidenceCount", int)
"""Number of evidence items collected during investigation."""


class CorrelationLabel(str, Enum):
    """Classification of correlation between agent findings.

    This enum provides a type-safe way to represent the relationship
    between findings from different diagnostic modalities (logs vs metrics).

    Attributes:
        STRONG_CORRELATION: Findings strongly support each other
        WEAK_CORRELATION: Findings somewhat support each other
        NO_CORRELATION: No meaningful relationship between findings
        CONTRADICTORY: Findings directly contradict each other

    Example:
        >>> correlation = CorrelationLabel.STRONG_CORRELATION
        >>> correlation.value
        'strong_correlation'
    """

    STRONG_CORRELATION = "strong_correlation"
    WEAK_CORRELATION = "weak_correlation"
    NO_CORRELATION = "no_correlation"
    CONTRADICTORY = "contradictory"


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Compact output exposed by a subagent to the supervisor.

    This frozen dataclass ensures that agent results cannot be modified
    after creation, preventing bugs from accidental mutation. It serves
    as the primary communication interface between specialized agents
    and the supervisor orchestrator.

    Attributes:
        agent_name: Identifier for the agent that produced this result
        summary: Human-readable summary of findings
        confidence: Agent's confidence in its findings (0.0-1.0)
        completeness: How complete the investigation was (0.0-1.0)
        evidence_count: Number of evidence items collected
        escalate: Whether this finding requires escalation to other agents

    Example:
        >>> result = AgentRunResult(
        ...     agent_name="log_agent",
        ...     summary="High error rate in authentication service",
        ...     confidence=AgentConfidence(0.85),
        ...     completeness=AgentCompleteness(0.90),
        ...     evidence_count=EvidenceCount(7),
        ...     escalate=False
        ... )
        >>> result.confidence
        AgentConfidence(0.85)
    """

    agent_name: str
    summary: str
    confidence: AgentConfidence
    completeness: AgentCompleteness
    evidence_count: EvidenceCount
    escalate: bool

    def __post_init__(self) -> None:
        """Validate agent result values.

        Raises:
            ValueError: If confidence or completeness is out of range [0.0, 1.0]
                       or if evidence_count is negative
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )

        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError(
                f"Completeness must be in [0.0, 1.0], got {self.completeness}"
            )

        if self.evidence_count < 0:
            raise ValueError(
                f"Evidence count must be non-negative, got {self.evidence_count}"
            )

        if not self.agent_name:
            raise ValueError("Agent name cannot be empty")


@dataclass(frozen=True, slots=True)
class SupervisorResult:
    """Final result from the orchestration supervisor.

    This represents the complete output of a multi-agent RCA investigation,
    including findings from individual agents and their correlation.

    Attributes:
        query: Original diagnostic query
        log_result: Findings from log-based investigation
        metrics_result: Findings from metrics-based investigation (if escalated)
        correlation: Relationship between log and metrics findings
        final_summary: Consolidated summary of all findings

    Example:
        >>> from lats.models.agent import AgentConfidence, AgentCompleteness, EvidenceCount
        >>> log_result = AgentRunResult(
        ...     agent_name="log_agent",
        ...     summary="Database connection errors",
        ...     confidence=AgentConfidence(0.8),
        ...     completeness=AgentCompleteness(0.7),
        ...     evidence_count=EvidenceCount(5),
        ...     escalate=True
        ... )
        >>> result = SupervisorResult(
        ...     query="Why is checkout failing?",
        ...     log_result=log_result,
        ...     metrics_result=None,
        ...     correlation=CorrelationLabel.NO_CORRELATION,
        ...     final_summary="Investigation complete"
        ... )
        >>> result.correlation
        <CorrelationLabel.NO_CORRELATION: 'no_correlation'>
    """

    query: str
    log_result: AgentRunResult
    metrics_result: AgentRunResult | None
    correlation: CorrelationLabel
    final_summary: str

    def __post_init__(self) -> None:
        """Validate supervisor result.

        Raises:
            ValueError: If query or final_summary is empty
        """
        if not self.query:
            raise ValueError("Query cannot be empty")

        if not self.final_summary:
            raise ValueError("Final summary cannot be empty")


def validate_confidence(value: float) -> AgentConfidence:
    """Validate and convert a float to AgentConfidence.

    Args:
        value: Confidence value to validate

    Returns:
        Validated AgentConfidence

    Raises:
        ValueError: If value is outside [0.0, 1.0] range

    Examples:
        >>> validate_confidence(0.75)
        AgentConfidence(0.75)
        >>> validate_confidence(1.5)
        Traceback (most recent call last):
            ...
        ValueError: Confidence must be in [0.0, 1.0], got 1.5
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Confidence must be in [0.0, 1.0], got {value}")
    return AgentConfidence(value)


def validate_completeness(value: float) -> AgentCompleteness:
    """Validate and convert a float to AgentCompleteness.

    Args:
        value: Completeness value to validate

    Returns:
        Validated AgentCompleteness

    Raises:
        ValueError: If value is outside [0.0, 1.0] range

    Examples:
        >>> validate_completeness(0.90)
        AgentCompleteness(0.9)
        >>> validate_completeness(-0.1)
        Traceback (most recent call last):
            ...
        ValueError: Completeness must be in [0.0, 1.0], got -0.1
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Completeness must be in [0.0, 1.0], got {value}")
    return AgentCompleteness(value)


def validate_evidence_count(value: int) -> EvidenceCount:
    """Validate and convert an int to EvidenceCount.

    Args:
        value: Evidence count to validate

    Returns:
        Validated EvidenceCount

    Raises:
        ValueError: If value is negative

    Examples:
        >>> validate_evidence_count(10)
        EvidenceCount(10)
        >>> validate_evidence_count(-1)
        Traceback (most recent call last):
            ...
        ValueError: Evidence count must be non-negative, got -1
    """
    if value < 0:
        raise ValueError(f"Evidence count must be non-negative, got {value}")
    return EvidenceCount(value)