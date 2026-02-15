"""Unit tests for RCA supervisor orchestration."""

from dataclasses import dataclass

from lats.models import AgentRunResult
from lats.orchestration import RCASupervisor


@dataclass
class StubWorker:
    """Simple worker stub that captures incoming queries."""

    result: AgentRunResult
    call_count: int = 0
    last_query: str | None = None

    def run(self, query: str) -> AgentRunResult:
        self.call_count += 1
        self.last_query = query
        return self.result


def test_supervisor_skips_metrics_when_log_is_confident() -> None:
    log_worker = StubWorker(
        result=AgentRunResult(
            agent_name="log_agent",
            summary="Timeout errors in auth service.",
            r=0.85,
            c=0.8,
            evidence_count=4,
            escalate=False,
        )
    )
    metrics_worker = StubWorker(
        result=AgentRunResult(
            agent_name="metrics_agent",
            summary="CPU remains stable.",
            r=0.9,
            c=0.9,
            evidence_count=3,
            escalate=False,
        )
    )

    supervisor = RCASupervisor(log_agent=log_worker, metrics_agent=metrics_worker)
    result = supervisor.run("Investigate auth outage")

    assert log_worker.call_count == 1
    assert metrics_worker.call_count == 0
    assert result.metrics_result is None
    assert result.correlation == "no_correlation"


def test_supervisor_escalates_and_passes_log_summary() -> None:
    log_worker = StubWorker(
        result=AgentRunResult(
            agent_name="log_agent",
            summary="Error bursts in token endpoint with timeout traces.",
            r=0.45,
            c=0.45,
            evidence_count=5,
            escalate=True,
        )
    )
    metrics_worker = StubWorker(
        result=AgentRunResult(
            agent_name="metrics_agent",
            summary="Metrics show timeout spikes and error correlation in auth pods.",
            r=0.8,
            c=0.75,
            evidence_count=4,
            escalate=False,
        )
    )

    supervisor = RCASupervisor(log_agent=log_worker, metrics_agent=metrics_worker)
    result = supervisor.run("Investigate auth outage")

    assert metrics_worker.call_count == 1
    assert metrics_worker.last_query is not None
    assert "Context from log agent summary" in metrics_worker.last_query
    assert log_worker.result.summary in metrics_worker.last_query
    assert result.metrics_result is not None
    assert result.correlation in {"strong_correlation", "weak_correlation", "no_correlation"}
