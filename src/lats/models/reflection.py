"""Reflection model for LATS self-critique.

This module defines the Reflection Pydantic model used to capture
the agent's self-evaluation of candidate responses along three
explicit quality dimensions: evidence quality, diagnostic completeness,
and internal consistency.
"""

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from lats.config.constants import MAX_SCORE, MIN_SCORE
from lats.core.types import Score, normalize_score, validate_score


def _validate_dimension(v: int) -> int:
    """Validate a dimension score is within [MIN_SCORE, MAX_SCORE]."""
    validate_score(v)
    return v


class Reflection(BaseModel):
    """Self-critique signal used as reward during tree search.

    The reflection scores the candidate response along three dimensions:
    - **Evidence quality**: Are the retrieved artefacts relevant and sufficient?
    - **Diagnostic completeness**: Does the response cover all plausible
      hypotheses and rule them in/out systematically?
    - **Internal consistency**: Are the claims logically coherent and free
      of contradictions?

    The overall ``score`` is the arithmetic mean of the three dimensions,
    rounded to the nearest integer.

    Attributes:
        reflections: Textual critique of the response quality
        evidence_quality: Score 0-10 for relevance and sufficiency of evidence
        diagnostic_completeness: Score 0-10 for hypothesis coverage
        internal_consistency: Score 0-10 for logical coherence
        found_solution: Whether this response fully solves the task

    Example:
        >>> reflection = Reflection(
        ...     reflections="Good answer but lacks depth",
        ...     evidence_quality=7,
        ...     diagnostic_completeness=6,
        ...     internal_consistency=8,
        ...     found_solution=False,
        ... )
        >>> reflection.score
        7
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

    evidence_quality: int = Field(
        description=(
            "Score from 0-10: Are the retrieved artefacts (logs, metrics, traces) "
            "relevant to the failure and sufficient to support the diagnosis?"
        ),
        ge=MIN_SCORE,
        le=MAX_SCORE,
    )

    diagnostic_completeness: int = Field(
        description=(
            "Score from 0-10: Does the response enumerate plausible root-cause "
            "hypotheses and systematically confirm or rule out each one?"
        ),
        ge=MIN_SCORE,
        le=MAX_SCORE,
    )

    internal_consistency: int = Field(
        description=(
            "Score from 0-10: Are the claims logically coherent, free of "
            "contradictions, and properly supported by the cited evidence?"
        ),
        ge=MIN_SCORE,
        le=MAX_SCORE,
    )

    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    @field_validator("evidence_quality", "diagnostic_completeness", "internal_consistency")
    @classmethod
    def validate_dimension_value(cls, v: int) -> int:
        """Validate a dimension score is in valid range."""
        return _validate_dimension(v)

    @property
    def score(self) -> int:
        """Overall score: arithmetic mean of the three dimensions, rounded.

        Returns:
            Integer score in [0, 10]
        """
        return round(
            (self.evidence_quality + self.diagnostic_completeness + self.internal_consistency) / 3
        )

    def as_message(self) -> HumanMessage:
        """Convert reflection to a message for inclusion in chat history.

        Returns:
            HumanMessage containing the reflection text and dimension scores
        """
        return HumanMessage(
            content=(
                f"Reasoning: {self.reflections}\n"
                f"Evidence quality: {self.evidence_quality} | "
                f"Diagnostic completeness: {self.diagnostic_completeness} | "
                f"Internal consistency: {self.internal_consistency} | "
                f"Overall: {self.score}"
            )
        )

    @property
    def normalized_score(self) -> float:
        """Return the overall score normalized to [0.0, 1.0] range."""
        return float(normalize_score(self.score, MIN_SCORE, MAX_SCORE))

    @property
    def normalized_evidence_quality(self) -> float:
        """Return evidence quality normalized to [0.0, 1.0]."""
        return float(normalize_score(self.evidence_quality, MIN_SCORE, MAX_SCORE))

    @property
    def normalized_diagnostic_completeness(self) -> float:
        """Return diagnostic completeness normalized to [0.0, 1.0]."""
        return float(normalize_score(self.diagnostic_completeness, MIN_SCORE, MAX_SCORE))

    @property
    def normalized_internal_consistency(self) -> float:
        """Return internal consistency normalized to [0.0, 1.0]."""
        return float(normalize_score(self.internal_consistency, MIN_SCORE, MAX_SCORE))

    @property
    def score_typed(self) -> Score:
        """Return the overall score as a domain type."""
        return Score(self.score)
