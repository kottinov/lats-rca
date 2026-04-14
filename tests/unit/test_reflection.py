"""Unit tests for Reflection model."""

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from lats.models.reflection import Reflection


def _make_reflection(**overrides):
    """Helper to create a Reflection with sensible defaults."""
    defaults = dict(
        reflections="Solid analysis",
        evidence_quality=7,
        diagnostic_completeness=6,
        internal_consistency=8,
        found_solution=False,
    )
    defaults.update(overrides)
    return Reflection(**defaults)


class TestReflection:
    """Tests for Reflection Pydantic model."""

    def test_create_reflection(self) -> None:
        """Test creating a valid reflection with three dimensions."""
        r = _make_reflection(found_solution=True)
        assert r.evidence_quality == 7
        assert r.diagnostic_completeness == 6
        assert r.internal_consistency == 8
        assert r.found_solution is True

    def test_overall_score_is_mean(self) -> None:
        """Overall score is the rounded mean of three dimensions."""
        r = _make_reflection(evidence_quality=7, diagnostic_completeness=6, internal_consistency=8)
        assert r.score == round((7 + 6 + 8) / 3)  # 7

    def test_overall_score_rounds_correctly(self) -> None:
        """Verify rounding at the .5 boundary."""
        r = _make_reflection(evidence_quality=7, diagnostic_completeness=7, internal_consistency=8)
        assert r.score == 7

        r2 = _make_reflection(evidence_quality=8, diagnostic_completeness=7, internal_consistency=8)
        assert r2.score == 8

    def test_normalized_score(self) -> None:
        """Test score normalization of overall score."""
        r = _make_reflection(evidence_quality=7, diagnostic_completeness=7, internal_consistency=7)
        assert r.normalized_score == 0.7

        r_min = _make_reflection(evidence_quality=0, diagnostic_completeness=0, internal_consistency=0)
        assert r_min.normalized_score == 0.0

        r_max = _make_reflection(evidence_quality=10, diagnostic_completeness=10, internal_consistency=10)
        assert r_max.normalized_score == 1.0

    def test_normalized_dimensions(self) -> None:
        """Test per-dimension normalization."""
        r = _make_reflection(evidence_quality=5, diagnostic_completeness=8, internal_consistency=10)
        assert r.normalized_evidence_quality == 0.5
        assert r.normalized_diagnostic_completeness == 0.8
        assert r.normalized_internal_consistency == 1.0

    def test_score_typed(self) -> None:
        """Test typed score property."""
        r = _make_reflection(evidence_quality=8, diagnostic_completeness=8, internal_consistency=8)
        assert int(r.score_typed) == 8

    def test_as_message(self) -> None:
        """Test converting reflection to message."""
        r = _make_reflection(reflections="Needs improvement")
        message = r.as_message()

        assert isinstance(message, HumanMessage)
        assert "Reasoning: Needs improvement" in message.content
        assert "Evidence quality: 7" in message.content
        assert "Diagnostic completeness: 6" in message.content
        assert "Internal consistency: 8" in message.content

    def test_invalid_evidence_quality_too_low(self) -> None:
        with pytest.raises(ValidationError):
            _make_reflection(evidence_quality=-1)

    def test_invalid_evidence_quality_too_high(self) -> None:
        with pytest.raises(ValidationError):
            _make_reflection(evidence_quality=11)

    def test_invalid_diagnostic_completeness_too_low(self) -> None:
        with pytest.raises(ValidationError):
            _make_reflection(diagnostic_completeness=-1)

    def test_invalid_internal_consistency_too_high(self) -> None:
        with pytest.raises(ValidationError):
            _make_reflection(internal_consistency=11)

    def test_empty_reflections(self) -> None:
        """Test that empty reflections are rejected."""
        with pytest.raises(ValidationError):
            _make_reflection(reflections="")