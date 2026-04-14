"""LATS configuration model.

This module defines the LATSConfig dataclass that holds algorithm parameters.
The config is immutable (frozen) to prevent accidental modifications during search.
"""

from dataclasses import dataclass

from lats.exceptions import LATSConfigError


@dataclass(frozen=True, slots=True)
class LATSConfig:
    """Configuration for the LATS algorithm.

    This frozen dataclass ensures that configuration values cannot be modified
    after initialization, preventing bugs from accidental mutation.

    Attributes:
        model: OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini")
        n_candidates: Number of candidate responses to generate at each node
        max_depth: Maximum depth of the search tree
        max_search_results: Maximum number of search results per query
        exploration_weight: UCT exploration weight (C parameter)

    Example:
        >>> config = LATSConfig(model="gpt-4o", n_candidates=5, max_depth=5)
        >>> config.validate()  # Raises LATSConfigError if invalid
        >>> config.n_candidates = 10  # Error: frozen dataclass
        Traceback (most recent call last):
            ...
        dataclasses.FrozenInstanceError: cannot assign to field 'n_candidates'
    """

    model: str = "gpt-4o"
    n_candidates: int = 5
    max_depth: int = 5
    max_search_results: int = 5
    exploration_weight: float = 1.0
    consistency_weight: float = 0.7

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            LATSConfigError: If any configuration value is invalid

        Example:
            >>> config = LATSConfig(n_candidates=-1)
            >>> config.validate()
            Traceback (most recent call last):
                ...
            lats.exceptions.config.LATSConfigError: n_candidates must be >= 1
        """
        if self.n_candidates < 1:
            raise LATSConfigError(
                "n_candidates must be >= 1",
                context={"n_candidates": self.n_candidates},
            )

        if self.max_depth < 1:
            raise LATSConfigError(
                "max_depth must be >= 1",
                context={"max_depth": self.max_depth},
            )

        if self.max_search_results < 1:
            raise LATSConfigError(
                "max_search_results must be >= 1",
                context={"max_search_results": self.max_search_results},
            )

        if self.exploration_weight < 0:
            raise LATSConfigError(
                "exploration_weight must be >= 0",
                context={"exploration_weight": self.exploration_weight},
            )

        if not 0.0 <= self.consistency_weight <= 1.0:
            raise LATSConfigError(
                "consistency_weight must be in [0.0, 1.0]",
                context={"consistency_weight": self.consistency_weight},
            )
