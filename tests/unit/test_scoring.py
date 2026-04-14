"""Unit tests for action signature extraction and self-consistency scoring."""

import pytest
from langchain_core.messages import AIMessage

from lats.core.scoring import (
    ActionSignature,
    _NO_TOOL_SIGNATURE,
    compute_combined_reward,
    compute_self_consistency,
    extract_candidate_signature,
)


def _ai(tool_calls: list[dict] | None = None) -> AIMessage:
    """Helper to build an AIMessage with optional tool calls."""
    return AIMessage(content="analysis", tool_calls=tool_calls or [])


def _tc(name: str, **args: str) -> dict:
    """Helper to build a tool call dict."""
    return {"name": name, "args": args, "id": f"id-{name}", "type": "tool_call"}


# --- extract_candidate_signature ---


class TestExtractCandidateSignature:
    def test_no_tool_calls(self) -> None:
        sig = extract_candidate_signature(_ai())
        assert sig == (_NO_TOOL_SIGNATURE,)

    def test_single_tool(self) -> None:
        msg = _ai([_tc("grep_file", pattern="ERROR", path="/var/log")])
        sig = extract_candidate_signature(msg)
        assert len(sig) == 1
        assert sig[0].tool_name == "grep_file"
        assert sig[0].arg_keys == frozenset({"pattern", "path"})

    def test_multiple_tools_sorted(self) -> None:
        msg = _ai([
            _tc("read_file", path="/a.log"),
            _tc("grep_file", pattern="err"),
        ])
        sig = extract_candidate_signature(msg)
        assert len(sig) == 2
        assert sig[0].tool_name == "grep_file"
        assert sig[1].tool_name == "read_file"

    def test_same_tool_different_values_same_signature(self) -> None:
        msg1 = _ai([_tc("grep_file", pattern="ERROR", path="/var/log")])
        msg2 = _ai([_tc("grep_file", pattern="timeout", path="/app/log")])
        assert extract_candidate_signature(msg1) == extract_candidate_signature(msg2)

    def test_same_tool_different_keys_different_signature(self) -> None:
        msg1 = _ai([_tc("grep_file", pattern="ERROR")])
        msg2 = _ai([_tc("grep_file", pattern="ERROR", path="/var/log")])
        assert extract_candidate_signature(msg1) != extract_candidate_signature(msg2)

    def test_tool_with_no_args(self) -> None:
        msg = _ai([{"name": "list_files", "args": {}, "id": "id-1", "type": "tool_call"}])
        sig = extract_candidate_signature(msg)
        assert sig[0] == ActionSignature(tool_name="list_files", arg_keys=frozenset())


# --- compute_self_consistency ---


class TestComputeSelfConsistency:
    def test_all_same_signatures(self) -> None:
        candidates = [
            _ai([_tc("grep_file", pattern="err")]),
            _ai([_tc("grep_file", pattern="timeout")]),
            _ai([_tc("grep_file", pattern="crash")]),
        ]
        scores = compute_self_consistency(candidates)
        assert scores == [1.0, 1.0, 1.0]

    def test_all_different_signatures(self) -> None:
        candidates = [
            _ai([_tc("grep_file", pattern="err")]),
            _ai([_tc("read_file", path="/a")]),
            _ai([_tc("list_files", directory="/")]),
        ]
        scores = compute_self_consistency(candidates)
        assert scores == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_majority_pattern(self) -> None:
        candidates = [
            _ai([_tc("grep_file", pattern="a")]),
            _ai([_tc("grep_file", pattern="b")]),
            _ai([_tc("grep_file", pattern="c")]),
            _ai([_tc("grep_file", pattern="d")]),
            _ai([_tc("read_file", path="/x")]),  # outlier
        ]
        scores = compute_self_consistency(candidates)
        assert scores[:4] == [0.8, 0.8, 0.8, 0.8]
        assert scores[4] == pytest.approx(0.2)

    def test_single_candidate(self) -> None:
        scores = compute_self_consistency([_ai([_tc("grep_file", pattern="x")])])
        assert scores == [1.0]

    def test_empty(self) -> None:
        assert compute_self_consistency([]) == []

    def test_no_tool_calls_all_consistent(self) -> None:
        candidates = [_ai(), _ai(), _ai()]
        scores = compute_self_consistency(candidates)
        assert scores == [1.0, 1.0, 1.0]


# --- compute_combined_reward ---


class TestComputeCombinedReward:
    def test_pure_reflection(self) -> None:
        assert compute_combined_reward(0.8, 0.5, alpha=1.0) == pytest.approx(0.8)

    def test_pure_consistency(self) -> None:
        assert compute_combined_reward(0.8, 0.5, alpha=0.0) == pytest.approx(0.5)

    def test_default_alpha(self) -> None:
        result = compute_combined_reward(0.8, 0.4)
        expected = 0.7 * 0.8 + 0.3 * 0.4
        assert result == pytest.approx(expected)

    def test_equal_scores(self) -> None:
        assert compute_combined_reward(0.6, 0.6, alpha=0.5) == pytest.approx(0.6)
