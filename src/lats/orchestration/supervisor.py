"""Supervisor orchestration for multi-agent LATS RCA.

This module implements the supervisor pattern for coordinating multiple
specialized LATS agents. The supervisor handles handoff decisions, correlation
analysis, and final report generation.

Following Robust Python principles:
- Uses Protocols for dependency injection
- Centralizes configuration in config module
- Provides clear type signatures
- Documents all public APIs with examples
"""

from __future__ import annotations

import re
from typing import Protocol

from lats.config.agent import (
    CORRELATION_STOP_WORDS,
    ERROR_SIGNALS,
    NORMAL_SIGNALS,
    STRONG_CORRELATION_THRESHOLD,
    WEAK_CORRELATION_THRESHOLD,
)
from lats.models.agent import (
    AgentCompleteness,
    AgentConfidence,
    AgentRunResult,
    CorrelationLabel,
    SupervisorResult,
)


class RCAWorker(Protocol):
    """Protocol for supervisor-compatible subagents.

    This protocol defines the minimal interface required for an agent
    to participate in supervised RCA investigation.
    """

    def run(self, query: str) -> AgentRunResult:
        """Run a subagent investigation.

        Args:
            query: Diagnostic query to investigate

        Returns:
            Agent result with findings and metrics
        """
        ...


class RCASupervisor:
    """Supervisor that coordinates log and metrics LATS subagents.

    The supervisor implements a gated escalation pattern:
    1. Always runs the log agent first
    2. Evaluates confidence and completeness
    3. Escalates to metrics agent if thresholds not met
    4. Correlates findings across modalities
    5. Generates final consolidated report

    The supervisor itself does NOT perform tree search; it only orchestrates
    specialized agents and synthesizes their findings.

    Attributes:
        log_agent: Agent for log-based investigation
        metrics_agent: Agent for metrics-based investigation
        confidence_threshold: Minimum confidence before escalation
        completeness_threshold: Minimum completeness before escalation

    Example:
        >>> from lats.agents.subagents import LogLATSAgent, MetricsLATSAgent
        >>> supervisor = RCASupervisor(
        ...     log_agent=LogLATSAgent(),
        ...     metrics_agent=MetricsLATSAgent()
        ... )
        >>> result = supervisor.run("Why is checkout failing?")
        >>> result.correlation
        <CorrelationLabel.STRONG_CORRELATION: 'strong_correlation'>
    """

    def __init__(
        self,
        log_agent: RCAWorker,
        metrics_agent: RCAWorker,
        *,
        confidence_threshold: float = 0.7,
        completeness_threshold: float = 0.6,
    ) -> None:
        """Initialize RCA supervisor.

        Args:
            log_agent: Agent for log-based diagnostics
            metrics_agent: Agent for metrics-based diagnostics
            confidence_threshold: Minimum confidence to skip escalation
            completeness_threshold: Minimum completeness to skip escalation

        Raises:
            ValueError: If thresholds are out of range [0.0, 1.0]
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"Confidence threshold must be in [0.0, 1.0], got {confidence_threshold}"
            )

        if not 0.0 <= completeness_threshold <= 1.0:
            raise ValueError(
                f"Completeness threshold must be in [0.0, 1.0], got {completeness_threshold}"
            )

        self.log_agent = log_agent
        self.metrics_agent = metrics_agent
        self.confidence_threshold = AgentConfidence(confidence_threshold)
        self.completeness_threshold = AgentCompleteness(completeness_threshold)

    def run(self, query: str) -> SupervisorResult:
        """Run the full supervisor-controlled RCA flow.

        This orchestrates the complete multi-agent investigation:
        1. Execute log agent investigation
        2. Check if escalation is needed based on quality metrics
        3. If needed, augment context and run metrics agent
        4. Correlate findings from both modalities
        5. Generate consolidated final summary

        Args:
            query: Diagnostic question to investigate

        Returns:
            SupervisorResult containing all findings and correlation

        Raises:
            ValueError: If query is empty

        Example:
            >>> supervisor = RCASupervisor(log_agent, metrics_agent)
            >>> result = supervisor.run("Database timeouts increasing")
            >>> result.final_summary
            'Final correlation: strong_correlation...'
        """
        if not query:
            raise ValueError("Query cannot be empty")

        log_result = self.log_agent.run(query)

        should_escalate = (
            log_result.escalate
            or log_result.confidence < self.confidence_threshold
            or log_result.completeness < self.completeness_threshold
        )

        metrics_result: AgentRunResult | None = None
        if should_escalate:
            metrics_query = self._build_metrics_query(query, log_result)
            metrics_result = self.metrics_agent.run(metrics_query)

        correlation = self._correlate(log_result, metrics_result)

        final_summary = self._build_final_summary(
            log_result, metrics_result, correlation
        )

        return SupervisorResult(
            query=query,
            log_result=log_result,
            metrics_result=metrics_result,
            correlation=correlation,
            final_summary=final_summary,
        )

    @staticmethod
    def _build_metrics_query(
        original_query: str, log_result: AgentRunResult
    ) -> str:
        """Build enriched query for metrics agent with log context.

        The metrics agent receives both the original query and a summary
        of log findings to enable cross-modal validation.

        Args:
            original_query: Original diagnostic question
            log_result: Results from log agent investigation

        Returns:
            Enriched query with log context
        """
        return (
            f"{original_query}\n\n"
            "Context from log agent:\n"
            f"{log_result.summary}"
        )

    def _correlate(
        self,
        log_result: AgentRunResult,
        metrics_result: AgentRunResult | None,
    ) -> CorrelationLabel:
        """Correlate findings from log and metrics agents.

        Correlation analysis uses two signals:
        1. Token overlap: shared diagnostic terms between summaries
        2. Polarity conflict: contradictory error/normal signals

        Args:
            log_result: Findings from log agent
            metrics_result: Findings from metrics agent (None if not run)

        Returns:
            Correlation classification

        Example:
            >>> correlation = supervisor._correlate(log_result, metrics_result)
            >>> correlation
            <CorrelationLabel.STRONG_CORRELATION: 'strong_correlation'>
        """
        if metrics_result is None:
            return CorrelationLabel.NO_CORRELATION

        if self._is_polarity_conflict(log_result.summary, metrics_result.summary):
            return CorrelationLabel.CONTRADICTORY

        log_tokens = self._tokenize(log_result.summary)
        metrics_tokens = self._tokenize(metrics_result.summary)
        overlap = len(log_tokens.intersection(metrics_tokens))

        if overlap >= STRONG_CORRELATION_THRESHOLD:
            return CorrelationLabel.STRONG_CORRELATION

        if overlap >= WEAK_CORRELATION_THRESHOLD:
            return CorrelationLabel.WEAK_CORRELATION

        return CorrelationLabel.NO_CORRELATION

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text for correlation analysis.

        Extracts significant tokens (3+ characters) and filters out
        common stop words that don't carry diagnostic meaning.

        Args:
            text: Text to tokenize

        Returns:
            Set of significant tokens (lowercased)

        Example:
            >>> tokens = RCASupervisor._tokenize("Database connection timeout")
            >>> "database" in tokens
            True
            >>> "the" in tokens
            False
        """
        all_tokens = set(re.findall(r"[a-zA-Z_]{3,}", text.lower()))

        return {
            token for token in all_tokens
            if token not in CORRELATION_STOP_WORDS
        }

    @staticmethod
    def _is_polarity_conflict(log_summary: str, metrics_summary: str) -> bool:
        """Detect contradictory error/normal signals between summaries.

        A polarity conflict occurs when one summary indicates errors while
        the other indicates normal operation, suggesting misdiagnosis.

        Args:
            log_summary: Summary from log agent
            metrics_summary: Summary from metrics agent

        Returns:
            True if summaries have contradictory polarity

        Example:
            >>> conflict = RCASupervisor._is_polarity_conflict(
            ...     "severe errors detected",
            ...     "all metrics normal"
            ... )
            >>> conflict
            True
        """
        log_text = log_summary.lower()
        metrics_text = metrics_summary.lower()

        log_has_error = any(signal in log_text for signal in ERROR_SIGNALS)
        log_has_normal = any(signal in log_text for signal in NORMAL_SIGNALS)
        metrics_has_error = any(signal in metrics_text for signal in ERROR_SIGNALS)
        metrics_has_normal = any(signal in metrics_text for signal in NORMAL_SIGNALS)

        return (
            (log_has_error and metrics_has_normal)
            or (metrics_has_error and log_has_normal)
        )

    @staticmethod
    def _build_final_summary(
        log_result: AgentRunResult,
        metrics_result: AgentRunResult | None,
        correlation: CorrelationLabel,
    ) -> str:
        """Build consolidated final summary from all findings.

        The final summary includes:
        - Correlation classification
        - Confidence scores from each agent
        - Complete findings from each modality

        Args:
            log_result: Findings from log agent
            metrics_result: Findings from metrics agent (None if not run)
            correlation: Correlation classification

        Returns:
            Formatted final summary

        Example:
            >>> summary = supervisor._build_final_summary(
            ...     log_result, metrics_result, CorrelationLabel.STRONG_CORRELATION
            ... )
            >>> "Final correlation: strong_correlation" in summary
            True
        """
        if metrics_result is None:
            return (
                f"Final correlation: {correlation.value}\n"
                "Single-agent investigation (log only, no escalation)\n\n"
                f"{log_result.summary}"
            )

        return (
            f"Final correlation: {correlation.value}\n"
            f"Multi-agent investigation complete\n"
            f"Log confidence: {log_result.confidence:.2f}, "
            f"Metrics confidence: {metrics_result.confidence:.2f}\n\n"
            f"Log Agent Findings\n"
            f"{log_result.summary}\n\n"
            f"Metrics Agent Findings\n"
            f"{metrics_result.summary}"
        )