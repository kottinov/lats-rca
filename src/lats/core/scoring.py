"""Action signature extraction and self-consistency scoring.

This module implements action signatures — a canonical representation of
tool-call strategy — and self-consistency scoring, which measures how much
independent search candidates converge on the same diagnostic actions.

The self-consistency score is combined with the reflection score to form
the final reward signal backpropagated through the MCTS tree.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage


@dataclass(frozen=True, slots=True)
class ActionSignature:
    """Canonical identifier for a tool-call action.

    Two actions share the same signature when they invoke the same tool
    with the same argument names, regardless of argument values.  This
    captures *strategic* convergence (e.g., "both candidates chose to
    grep logs") without requiring identical queries.

    Attributes:
        tool_name: Name of the invoked tool.
        arg_keys: Frozen set of argument key names.
    """

    tool_name: str
    arg_keys: frozenset[str]


_NO_TOOL_SIGNATURE = ActionSignature(tool_name="__no_tool__", arg_keys=frozenset())
"""Sentinel signature for candidates that make no tool calls."""


def extract_candidate_signature(
    candidate: AIMessage,
) -> tuple[ActionSignature, ...]:
    """Extract the composite action signature from a candidate response.

    For candidates with multiple tool calls the individual signatures are
    sorted to produce a canonical ordering, so ``(grep_file, read_file)``
    matches regardless of call order.

    Args:
        candidate: An AIMessage potentially containing tool calls.

    Returns:
        Tuple of ActionSignature(s).  A candidate with no tool calls
        returns ``(_NO_TOOL_SIGNATURE,)``.
    """
    if not candidate.tool_calls:
        return (_NO_TOOL_SIGNATURE,)

    sigs = [
        ActionSignature(
            tool_name=tc["name"],
            arg_keys=frozenset(tc["args"].keys()),
        )
        for tc in candidate.tool_calls
    ]
    return tuple(sorted(sigs, key=lambda s: (s.tool_name, sorted(s.arg_keys))))


def compute_self_consistency(candidates: Sequence[AIMessage]) -> list[float]:
    """Compute per-candidate self-consistency scores.

    Self-consistency measures how often a candidate's action signature
    appears among its siblings.  If 4 out of 5 candidates share the
    same signature, each of the 4 receives ``0.8`` and the outlier
    receives ``0.2``.

    Args:
        candidates: Candidate AIMessages from a single expansion step.

    Returns:
        List of float scores in ``[0.0, 1.0]``, one per candidate,
        in the same order as *candidates*.
    """
    n = len(candidates)
    if n == 0:
        return []

    composite_sigs = [extract_candidate_signature(c) for c in candidates]
    counts: Counter[tuple[ActionSignature, ...]] = Counter(composite_sigs)

    return [counts[sig] / n for sig in composite_sigs]


def compute_combined_reward(
    reflection_score: float,
    self_consistency: float,
    alpha: float = 0.7,
) -> float:
    """Combine reflection score and self-consistency into a single reward.

    ``reward = alpha * reflection_score + (1 - alpha) * self_consistency``

    Setting ``alpha=1.0`` recovers pure reflection scoring (backward
    compatible with the original implementation).

    Args:
        reflection_score: Normalized reflection score in ``[0.0, 1.0]``.
        self_consistency: Self-consistency score in ``[0.0, 1.0]``.
        alpha: Weight for the reflection component.

    Returns:
        Combined reward value.
    """
    return alpha * reflection_score + (1.0 - alpha) * self_consistency
