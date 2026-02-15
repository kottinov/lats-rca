"""Unit tests for domain types."""

import pytest

from lats.core.types import (
    normalize_score,
    validate_depth,
    validate_exploration_weight,
    validate_score,
    validate_visit_count,
)


class TestValidateScore:
    """Tests for score validation."""

    def test_valid_score(self) -> None:
        """Test that valid scores are accepted."""
        assert validate_score(0) == 0
        assert validate_score(5) == 5
        assert validate_score(10) == 10

    def test_invalid_score_negative(self) -> None:
        """Test that negative scores are rejected."""
        with pytest.raises(ValueError, match="Score must be in"):
            validate_score(-1)

    def test_invalid_score_too_high(self) -> None:
        """Test that scores > 10 are rejected."""
        with pytest.raises(ValueError, match="Score must be in"):
            validate_score(11)


class TestNormalizeScore:
    """Tests for score normalization."""

    def test_normalize_min_score(self) -> None:
        """Test normalization of minimum score."""
        assert normalize_score(0) == 0.0

    def test_normalize_max_score(self) -> None:
        """Test normalization of maximum score."""
        assert normalize_score(10) == 1.0

    def test_normalize_mid_score(self) -> None:
        """Test normalization of middle score."""
        assert normalize_score(5) == 0.5

    def test_normalize_custom_range(self) -> None:
        """Test normalization with custom range."""
        result = normalize_score(50, min_score=0, max_score=100)
        assert result == 0.5

    def test_normalize_invalid_range(self) -> None:
        """Test that invalid ranges are rejected."""
        with pytest.raises(ValueError, match="min_score must be"):
            normalize_score(5, min_score=10, max_score=5)


class TestValidateDepth:
    """Tests for depth validation."""

    def test_valid_depth(self) -> None:
        """Test that valid depths are accepted."""
        assert validate_depth(0, max_depth=5) == 0
        assert validate_depth(3, max_depth=5) == 3
        assert validate_depth(5, max_depth=5) == 5

    def test_negative_depth(self) -> None:
        """Test that negative depths are rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_depth(-1, max_depth=5)

    def test_depth_exceeds_max(self) -> None:
        """Test that depths exceeding max are rejected."""
        with pytest.raises(ValueError, match="exceeds max_depth"):
            validate_depth(10, max_depth=5)


class TestValidateVisitCount:
    """Tests for visit count validation."""

    def test_valid_visit_count(self) -> None:
        """Test that valid visit counts are accepted."""
        assert validate_visit_count(0) == 0
        assert validate_visit_count(10) == 10
        assert validate_visit_count(1000) == 1000

    def test_negative_visit_count(self) -> None:
        """Test that negative visit counts are rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_visit_count(-1)


class TestValidateExplorationWeight:
    """Tests for exploration weight validation."""

    def test_valid_exploration_weight(self) -> None:
        """Test that valid weights are accepted."""
        assert validate_exploration_weight(0.0) == 0.0
        assert validate_exploration_weight(1.41) == 1.41
        assert validate_exploration_weight(2.0) == 2.0

    def test_negative_exploration_weight(self) -> None:
        """Test that negative weights are rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_exploration_weight(-1.0)
