"""Agent-specific exceptions for LATS-RCA.

This module defines the exception hierarchy for multi-agent RCA operations,
following Robust Python Chapter 9 principles for clear error communication.
"""

from __future__ import annotations

from lats.exceptions.core import LATSError


class AgentError(LATSError):
    """Base exception for agent-related errors.

    Raised when an agent encounters an error during investigation.
    """


class AgentValidationError(AgentError):
    """Raised when agent result validation fails.

    Example:
        >>> raise AgentValidationError("Confidence out of range: 1.5")
        Traceback (most recent call last):
            ...
        AgentValidationError: Confidence out of range: 1.5
    """


class EscalationError(AgentError):
    """Raised when agent escalation logic fails.

    Example:
        >>> raise EscalationError("No metrics agent available for escalation")
        Traceback (most recent call last):
            ...
        EscalationError: No metrics agent available for escalation
    """


class CorrelationError(AgentError):
    """Raised when correlation analysis fails.

    Example:
        >>> raise CorrelationError("Cannot correlate: missing metrics result")
        Traceback (most recent call last):
            ...
        CorrelationError: Cannot correlate: missing metrics result
    """


class SearchError(AgentError):
    """Raised when LATS search execution fails.

    Example:
        >>> raise SearchError("Search runner failed: timeout")
        Traceback (most recent call last):
            ...
        SearchError: Search runner failed: timeout
    """