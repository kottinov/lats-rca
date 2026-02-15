"""Domain-specific type definitions for LATS.

This module defines NewType wrappers for domain concepts, providing type safety
at compile time and preventing mixing of semantically different values.

Following "Robust Python" principles, we use NewType to create distinct types
for concepts like Score, Depth, and VisitCount, even though they're all integers
at runtime. This catches bugs where we might accidentally add a score to a depth.
"""

from typing import NewType

Depth = NewType("Depth", int)
"""Depth of a node in the search tree (0 = root)."""

VisitCount = NewType("VisitCount", int)
"""Number of times a node has been visited during MCTS."""

Score = NewType("Score", int)
"""Raw score value (typically 0-10 range)."""

NormalizedScore = NewType("NormalizedScore", float)
"""Normalized score value (0.0-1.0 range)."""

RewardValue = NewType("RewardValue", float)
"""Reward value used in backpropagation (can be negative)."""

ExplorationWeight = NewType("ExplorationWeight", float)
"""Exploration weight (C) in UCT formula: typically sqrt(2) ≈ 1.41."""

UCTValue = NewType("UCTValue", float)
"""Upper Confidence Bound value used for node selection."""


def validate_score(score: int) -> Score:
    """Validate and convert an integer to a Score.

    Args:
        score: Raw score value to validate

    Returns:
        Validated Score

    Raises:
        ValueError: If score is outside valid range [0, 10]

    Examples:
        >>> validate_score(5)
        Score(5)
        >>> validate_score(15)
        Traceback (most recent call last):
            ...
        ValueError: Score must be in [0, 10], got 15
    """
    if not 0 <= score <= 10:
        raise ValueError(f"Score must be in [0, 10], got {score}")
    return Score(score)


def normalize_score(score: int, min_score: int = 0, max_score: int = 10) -> NormalizedScore:
    """Normalize a score to [0.0, 1.0] range.

    Args:
        score: Raw score value
        min_score: Minimum possible score (default: 0)
        max_score: Maximum possible score (default: 10)

    Returns:
        Normalized score in [0.0, 1.0] range

    Raises:
        ValueError: If min_score >= max_score

    Examples:
        >>> normalize_score(5)
        NormalizedScore(0.5)
        >>> normalize_score(10)
        NormalizedScore(1.0)
    """
    if min_score >= max_score:
        raise ValueError(f"min_score must be < max_score, got {min_score} >= {max_score}")

    normalized = (score - min_score) / (max_score - min_score)
    return NormalizedScore(max(0.0, min(1.0, normalized)))


def validate_depth(depth: int, max_depth: int) -> Depth:
    """Validate a depth value.

    Args:
        depth: Depth value to validate
        max_depth: Maximum allowed depth

    Returns:
        Validated Depth

    Raises:
        ValueError: If depth is negative or exceeds max_depth

    Examples:
        >>> validate_depth(3, max_depth=5)
        Depth(3)
        >>> validate_depth(-1, max_depth=5)
        Traceback (most recent call last):
            ...
        ValueError: Depth must be non-negative, got -1
    """
    if depth < 0:
        raise ValueError(f"Depth must be non-negative, got {depth}")
    if depth > max_depth:
        raise ValueError(f"Depth {depth} exceeds max_depth {max_depth}")
    return Depth(depth)


def validate_visit_count(count: int) -> VisitCount:
    """Validate a visit count.

    Args:
        count: Visit count to validate

    Returns:
        Validated VisitCount

    Raises:
        ValueError: If count is negative

    Examples:
        >>> validate_visit_count(5)
        VisitCount(5)
        >>> validate_visit_count(-1)
        Traceback (most recent call last):
            ...
        ValueError: Visit count must be non-negative, got -1
    """
    if count < 0:
        raise ValueError(f"Visit count must be non-negative, got {count}")
    return VisitCount(count)


def validate_exploration_weight(weight: float) -> ExplorationWeight:
    """Validate exploration weight for UCT.

    The exploration weight should typically be positive. Common values
    are sqrt(2) ≈ 1.41 for standard UCT.

    Args:
        weight: Exploration weight to validate

    Returns:
        Validated ExplorationWeight

    Raises:
        ValueError: If weight is negative

    Examples:
        >>> validate_exploration_weight(1.41)
        ExplorationWeight(1.41)
        >>> validate_exploration_weight(-1.0)
        Traceback (most recent call last):
            ...
        ValueError: Exploration weight must be non-negative, got -1.0
    """
    if weight < 0:
        raise ValueError(f"Exploration weight must be non-negative, got {weight}")
    return ExplorationWeight(weight)
