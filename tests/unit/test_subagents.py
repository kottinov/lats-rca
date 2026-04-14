"""Unit tests for specialized LATS subagents."""

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from lats.agents import LogLATSAgent, MetricsLATSAgent
from lats.models import Reflection, SearchNode


class FakeSearchRunner:
    """Deterministic search runner used for testing subagents."""

    def __init__(
        self,
        evidence_quality: int = 7,
        diagnostic_completeness: int = 7,
        internal_consistency: int = 7,
    ) -> None:
        self.evidence_quality = evidence_quality
        self.diagnostic_completeness = diagnostic_completeness
        self.internal_consistency = internal_consistency
        self.last_question: str | None = None

    def run(
        self, question: str, print_rollouts: bool = False
    ) -> tuple[SearchNode, list[BaseMessage]]:
        del print_rollouts
        self.last_question = question

        reflection = Reflection(
            reflections="Test reflection for modality analysis",
            evidence_quality=self.evidence_quality,
            diagnostic_completeness=self.diagnostic_completeness,
            internal_consistency=self.internal_consistency,
            found_solution=self.evidence_quality >= 8
            and self.diagnostic_completeness >= 8
            and self.internal_consistency >= 8,
        )
        messages: list[BaseMessage] = [
            AIMessage(content="Candidate analysis"),
            ToolMessage(content="Evidence item", tool_call_id="tool-1"),
        ]
        solution = SearchNode(messages=messages, reflection=reflection)
        return solution, messages


def test_log_subagent_escalates_for_low_confidence() -> None:
    runner = FakeSearchRunner(
        evidence_quality=5, diagnostic_completeness=4, internal_consistency=6,
    )
    agent = LogLATSAgent(search_factory=lambda _cfg: runner)

    result = agent.run("Investigate login failures")

    assert result.agent_name == "log_agent"
    assert result.confidence == round((5 + 4 + 6) / 3) / 10  # 0.5
    assert result.completeness == 0.4  # diagnostic_completeness / 10
    assert result.evidence_count == 1
    assert result.escalate is True
    assert runner.last_question is not None
    assert "LOG diagnostics agent" in runner.last_question


def test_metrics_subagent_does_not_escalate_for_high_confidence() -> None:
    runner = FakeSearchRunner(
        evidence_quality=9, diagnostic_completeness=9, internal_consistency=9,
    )
    agent = MetricsLATSAgent(search_factory=lambda _cfg: runner)

    result = agent.run("Validate resource saturation hypothesis")

    assert result.agent_name == "metrics_agent"
    assert result.confidence == 0.9
    assert result.completeness == 0.9
    assert result.escalate is False
    assert runner.last_question is not None
    assert "METRICS diagnostics agent" in runner.last_question