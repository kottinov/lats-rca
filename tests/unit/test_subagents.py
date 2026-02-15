"""Unit tests for specialized LATS subagents."""

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from lats.agents import LogLATSAgent, MetricsLATSAgent
from lats.agents.subagents import HandoffSynthesis
from lats.models import Reflection, SearchNode


class FakeSearchRunner:
    """Deterministic search runner used for testing subagents."""

    def __init__(self, score: int) -> None:
        self.score = score
        self.last_question: str | None = None

    def run(
        self, question: str, print_rollouts: bool = False
    ) -> tuple[SearchNode, list[BaseMessage]]:
        del print_rollouts
        self.last_question = question

        reflection = Reflection(
            reflections="Test reflection for modality analysis",
            score=self.score,
            found_solution=self.score >= 8,
        )
        messages: list[BaseMessage] = [
            AIMessage(content="Candidate analysis"),
            ToolMessage(content="Evidence item", tool_call_id="tool-1"),
        ]
        solution = SearchNode(messages=messages, reflection=reflection)
        return solution, messages


class FakeHandoffSynthesizer:
    """Deterministic structured handoff synthesizer for tests."""

    def __init__(self, c: float) -> None:
        self.c = c

    def invoke(self, input: str) -> HandoffSynthesis:
        del input
        return HandoffSynthesis(summary="Structured handoff summary.", c=self.c)


def test_log_subagent_escalates_for_low_confidence() -> None:
    runner = FakeSearchRunner(score=5)
    synthesizer = FakeHandoffSynthesizer(c=0.5)
    agent = LogLATSAgent(
        search_factory=lambda _cfg: runner,
        handoff_synthesizer=synthesizer,
    )

    result = agent.run("Investigate login failures")

    assert result.agent_name == "log_agent"
    assert result.r == 0.5
    assert result.c == 0.5
    assert result.evidence_count == 1
    assert result.escalate is True
    assert result.summary == "Structured handoff summary."
    assert runner.last_question is not None
    assert "LOG diagnostics agent" in runner.last_question


def test_metrics_subagent_does_not_escalate_for_high_confidence() -> None:
    runner = FakeSearchRunner(score=9)
    synthesizer = FakeHandoffSynthesizer(c=0.92)
    agent = MetricsLATSAgent(
        search_factory=lambda _cfg: runner,
        handoff_synthesizer=synthesizer,
    )

    result = agent.run("Validate resource saturation hypothesis")

    assert result.agent_name == "metrics_agent"
    assert result.r == 0.9
    assert result.c == 0.92
    assert result.escalate is False
    assert result.summary == "Structured handoff summary."
    assert runner.last_question is not None
    assert "METRICS diagnostics agent" in runner.last_question
