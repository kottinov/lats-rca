"""Unit tests for Reflection model."""

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from lats.models.reflection import Reflection


class TestReflection:
    """Tests for Reflection Pydantic model."""

    def test_create_reflection(self) -> None:
        """Test creating a valid reflection."""
        reflection = Reflection(
            reflections="Good answer with detailed explanation",
            score=8,
            found_solution=True,
        )
        assert reflection.reflections == "Good answer with detailed explanation"
        assert reflection.score == 8
        assert reflection.found_solution is True

    def test_normalized_score(self) -> None:
        """Test score normalization."""
        reflection = Reflection(reflections="Test", score=7, found_solution=False)
        assert reflection.normalized_score == 0.7

        reflection_min = Reflection(reflections="Test", score=0, found_solution=False)
        assert reflection_min.normalized_score == 0.0

        reflection_max = Reflection(reflections="Test", score=10, found_solution=False)
        assert reflection_max.normalized_score == 1.0

    def test_score_typed(self) -> None:
        """Test typed score property."""
        reflection = Reflection(reflections="Test", score=8, found_solution=False)
        score_typed = reflection.score_typed
        assert int(score_typed) == 8

    def test_as_message(self) -> None:
        """Test converting reflection to message."""
        reflection = Reflection(
            reflections="Needs improvement", score=5, found_solution=False
        )
        message = reflection.as_message()

        assert isinstance(message, HumanMessage)
        assert "Reasoning: Needs improvement" in message.content
        assert "Score: 5" in message.content

    def test_invalid_score_too_low(self) -> None:
        """Test that scores below 0 are rejected."""
        with pytest.raises(ValidationError):
            Reflection(reflections="Test", score=-1, found_solution=False)

    def test_invalid_score_too_high(self) -> None:
        """Test that scores above 10 are rejected."""
        with pytest.raises(ValidationError):
            Reflection(reflections="Test", score=11, found_solution=False)

    def test_empty_reflections(self) -> None:
        """Test that empty reflections are rejected."""
        with pytest.raises(ValidationError):
            Reflection(reflections="", score=5, found_solution=False)
