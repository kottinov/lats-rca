"""Reflection model for LATS self-critique.

This module defines the Reflection Pydantic model used to capture
the agent's self-evaluation of candidate responses.
"""

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from lats.config.constants import MAX_SCORE, MIN_SCORE
from lats.core.types import Score, normalize_score, validate_score


class Reflection(BaseModel):
    """Self-critique signal used as reward during tree search.

    The reflection captures the agent's assessment of a candidate response,
    including qualitative feedback and a quantitative score.

    Attributes:
        reflections: Textual critique of the response quality
        score: Numerical score from 0-10
        found_solution: Whether this response fully solves the task

    Example:
        >>> reflection = Reflection(
        ...     reflections="Good answer but lacks depth",
        ...     score=7,
        ...     found_solution=False
        ... )
        >>> reflection.normalized_score
        0.7
    """

    reflections: str = Field(
        description=(
            "The critique and reflections on the sufficiency, superfluency, "
            "and general quality of the response"
        ),
        min_length=1,
    )

    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        ge=MIN_SCORE,
        le=MAX_SCORE,
    )

    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    @field_validator("score")
    @classmethod
    def validate_score_value(cls, v: int) -> int:
        """Validate score is in valid range.

        Args:
            v: Score value to validate

        Returns:
            Validated score

        Raises:
            ValueError: If score is out of range
        """
        validate_score(v)
        return v

    def as_message(self) -> HumanMessage:
        """Convert reflection to a message for inclusion in chat history.

        Returns:
            HumanMessage containing the reflection text and score

        Example:
            >>> reflection = Reflection(reflections="Good", score=8, found_solution=False)
            >>> msg = reflection.as_message()
            >>> msg.content
            'Reasoning: Good\\nScore: 8'
        """
        return HumanMessage(content=f"Reasoning: {self.reflections}\nScore: {self.score}")

    @property
    def normalized_score(self) -> float:
        """Return the score normalized to [0.0, 1.0] range.

        Returns:
            Normalized score value

        Example:
            >>> reflection = Reflection(reflections="Good", score=8, found_solution=False)
            >>> reflection.normalized_score
            0.8
        """
        return float(normalize_score(self.score, MIN_SCORE, MAX_SCORE))

    @property
    def score_typed(self) -> Score:
        """Return the score as a domain type.

        Returns:
            Score domain type

        Example:
            >>> reflection = Reflection(reflections="Good", score=8, found_solution=False)
            >>> reflection.score_typed
            Score(8)
        """
        return Score(self.score)
