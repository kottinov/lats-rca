"""Specialized LATS subagents for RCA.

This module implements the specialized subagents that perform modality-specific
root cause analysis using LATS tree search. Each subagent focuses on a specific
telemetry type (logs, metrics) and reports findings to the supervisor.

Following Robust Python principles, this module uses:
- Protocols for dependency injection
- NewType for domain-specific type safety
- Frozen dataclasses where appropriate
- Explicit configuration constants from config module
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from lats.config.agent import HandoffThresholds, MAX_EVIDENCE_ITEMS
from lats.core.search import LanguageAgentTreeSearch
from lats.models.agent import (
    AgentRunResult,
    EvidenceCount,
    validate_completeness,
    validate_confidence,
    validate_evidence_count,
)
from lats.models.config import LATSConfig
from lats.models.reflection import Reflection
from lats.tools.logs import LOG_TOOLS
from lats.tools.metrics import METRICS_TOOLS


class SearchSolution(Protocol):
    """Protocol for search solution nodes consumed by subagents.

    This protocol defines the minimal interface required from a search
    tree node to extract diagnostic findings.

    Attributes:
        reflection: Self-critique reflection containing the solution quality
    """

    reflection: Reflection


class SearchRunner(Protocol):
    """Protocol for search engines used by subagents.

    This protocol allows dependency injection of different search
    implementations while maintaining type safety.
    """

    def run(
        self, question: str, print_rollouts: bool = False
    ) -> tuple[SearchSolution, list[BaseMessage]]:
        """Run search for a question and return solution plus trajectory.

        Args:
            question: Diagnostic question to investigate
            print_rollouts: Whether to print rollout information

        Returns:
            Tuple of (solution node, message trajectory)
        """
        ...


SearchFactory = Callable[[LATSConfig], SearchRunner]
"""Factory function for creating search runner instances."""


class LATSSubAgent:
    """Reusable LATS-backed subagent for a single modality.

    This class implements a specialized diagnostic agent that uses
    Language Agent Tree Search to investigate failures in a specific
    telemetry modality (logs or metrics).

    The agent augments queries with modality-specific context, runs
    LATS search, and produces a supervisor-compatible result summary.

    Attributes:
        agent_name: Identifier for this agent instance
        modality_prompt: Context prompt for this modality
        tools: LangChain tools available to this agent
        config: LATS search configuration
        thresholds: Escalation decision thresholds
        search_runner: Search engine instance

    Example:
        >>> from lats.config.agent import HandoffThresholds
        >>> from lats.tools.logs import LOG_TOOLS
        >>> agent = LATSSubAgent(
        ...     agent_name="log_agent",
        ...     modality_prompt="Focus on log evidence",
        ...     tools=LOG_TOOLS,
        ...     config=LATSConfig(),
        ...     thresholds=HandoffThresholds()
        ... )
        >>> result = agent.run("Why is the service failing?")
        >>> result.agent_name
        'log_agent'
    """

    def __init__(
        self,
        agent_name: str,
        modality_prompt: str,
        tools: list[BaseTool],
        config: LATSConfig | None = None,
        *,
        thresholds: HandoffThresholds | None = None,
        search_factory: SearchFactory | None = None,
    ) -> None:
        """Initialize a LATS subagent.

        Args:
            agent_name: Identifier for this agent
            modality_prompt: Modality-specific instruction prompt
            tools: List of LangChain tools for this agent's modality
            config: LATS configuration (uses default if None)
            thresholds: Escalation thresholds (uses default if None)
            search_factory: Factory for creating search runners

        Raises:
            ValueError: If agent_name or modality_prompt is empty
        """
        if not agent_name:
            raise ValueError("Agent name cannot be empty")

        if not modality_prompt:
            raise ValueError("Modality prompt cannot be empty")

        self.agent_name = agent_name
        self.modality_prompt = modality_prompt
        self.tools = tools
        self.config = config or LATSConfig()
        self.thresholds = thresholds or HandoffThresholds()

        tool_node = ToolNode(tools=self.tools)
        factory = search_factory or (
            lambda cfg: LanguageAgentTreeSearch(config=cfg, tool_node=tool_node)
        )
        self.search_runner = factory(self.config)

    def run(self, query: str) -> AgentRunResult:
        """Run modality-specialized LATS search and return supervisor-safe summary.

        This method executes the complete diagnostic workflow:
        1. Augments query with modality-specific context
        2. Runs LATS tree search to explore hypotheses
        3. Extracts and validates solution metrics
        4. Determines if escalation is needed
        5. Builds human-readable summary

        Args:
            query: Diagnostic question to investigate

        Returns:
            AgentRunResult containing findings and metrics

        Raises:
            ValueError: If query is empty

        Example:
            >>> agent = LogLATSAgent()
            >>> result = agent.run("High latency in checkout")
            >>> result.confidence > 0.0
            True
        """
        if not query:
            raise ValueError("Query cannot be empty")

        augmented_query = self._augment_query(query)
        solution, trajectory = self.search_runner.run(augmented_query)

        reflection = solution.reflection
        confidence = validate_confidence(float(reflection.normalized_score))
        completeness = validate_completeness(
            float(reflection.normalized_diagnostic_completeness)
        )
        evidence_count = self._count_evidence(trajectory)
        summary = self._build_summary(solution, trajectory)
        escalate = self.thresholds.should_escalate(confidence, completeness)

        return AgentRunResult(
            agent_name=self.agent_name,
            summary=summary,
            confidence=confidence,
            completeness=completeness,
            evidence_count=evidence_count,
            escalate=escalate,
        )

    def _augment_query(self, query: str) -> str:
        """Augment query with modality-specific context.

        Args:
            query: Original diagnostic query

        Returns:
            Query augmented with modality instructions
        """
        return f"{self.modality_prompt}\n\n{query}"

    @staticmethod
    def _count_evidence(trajectory: list[BaseMessage]) -> EvidenceCount:
        """Count evidence items in trajectory.

        Evidence items are tool messages, which represent data retrieved
        from external sources (logs, metrics, traces, etc.).

        Args:
            trajectory: Message history from search

        Returns:
            Number of evidence items (capped at MAX_EVIDENCE_ITEMS)
        """
        evidence_items = sum(
            1 for message in trajectory if isinstance(message, ToolMessage)
        )
        capped_count = min(evidence_items, MAX_EVIDENCE_ITEMS)
        return validate_evidence_count(capped_count)

    def _build_summary(
        self, solution: SearchSolution, trajectory: list[BaseMessage]
    ) -> str:
        """Build human-readable summary of agent findings.

        The summary includes:
        - Agent identifier
        - Last AI-generated content (the conclusion)
        - Reflection critique
        - Confidence score

        Args:
            solution: Search solution node
            trajectory: Message history

        Returns:
            Formatted summary string
        """
        ai_contents = [
            str(message.content).strip()
            for message in trajectory
            if isinstance(message, AIMessage) and str(message.content).strip()
        ]

        last_ai = (
            ai_contents[-1] if ai_contents else "No conclusion generated."
        )

        reflection_text = solution.reflection.reflections
        score = solution.reflection.normalized_score

        return (
            f"{self.agent_name} findings:\n"
            f"{last_ai}\n\n"
            f"Reflection: {reflection_text}\n"
            f"Confidence: {score:.2f}"
        )


class LogLATSAgent(LATSSubAgent):
    """LATS subagent specialized for log-driven diagnosis.

    This agent focuses on analyzing structured logs, error messages,
    and service-level symptoms to identify root causes.

    Uses observability tools for file listing, reading, and log searching.

    Example:
        >>> agent = LogLATSAgent()
        >>> result = agent.run("Database connection failures")
        >>> result.agent_name
        'log_agent'
    """

    def __init__(
        self,
        config: LATSConfig | None = None,
        *,
        tools: list[BaseTool] | None = None,
        thresholds: HandoffThresholds | None = None,
        search_factory: SearchFactory | None = None,
    ) -> None:
        """Initialize log diagnostics agent.

        Args:
            config: LATS configuration (uses default if None)
            tools: Log analysis tools (uses LOG_TOOLS if None)
            thresholds: Escalation thresholds (uses default if None)
            search_factory: Factory for creating search runners
        """
        super().__init__(
            agent_name="log_agent",
            modality_prompt=(
                "You are the LOG diagnostics agent. Focus on log evidence, "
                "error messages, stack traces, and service-level symptoms. "
                "Identify patterns in log timestamps, error codes, and affected services."
            ),
            tools=tools or LOG_TOOLS,
            config=config,
            thresholds=thresholds,
            search_factory=search_factory,
        )


class MetricsLATSAgent(LATSSubAgent):
    """LATS subagent specialized for metrics-driven validation.

    This agent focuses on analyzing time-series metrics, resource
    utilization, and quantitative validation of hypotheses.

    Uses observability tools for loading and querying metrics data.

    Example:
        >>> agent = MetricsLATSAgent()
        >>> result = agent.run("CPU throttling detected")
        >>> result.agent_name
        'metrics_agent'
    """

    def __init__(
        self,
        config: LATSConfig | None = None,
        *,
        tools: list[BaseTool] | None = None,
        thresholds: HandoffThresholds | None = None,
        search_factory: SearchFactory | None = None,
    ) -> None:
        """Initialize metrics diagnostics agent.

        Args:
            config: LATS configuration (uses default if None)
            tools: Metrics analysis tools (uses METRICS_TOOLS if None)
            thresholds: Escalation thresholds (uses default if None)
            search_factory: Factory for creating search runners
        """
        super().__init__(
            agent_name="metrics_agent",
            modality_prompt=(
                "You are the METRICS diagnostics agent. Focus on time-series evidence, "
                "resource utilization patterns, and quantitative validation of hypotheses. "
                "Analyze CPU, memory, disk, network metrics and correlate with incidents."
            ),
            tools=tools or METRICS_TOOLS,
            config=config,
            thresholds=thresholds,
            search_factory=search_factory,
        )
