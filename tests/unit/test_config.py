"""Unit tests for configuration models."""

import pytest

from lats.exceptions import LATSConfigError
from lats.models.config import LATSConfig


class TestLATSConfig:
    """Tests for LATSConfig dataclass."""

    def test_default_config(self) -> None:
        """Test that default configuration is valid."""
        config = LATSConfig()
        assert config.model == "gpt-4o"
        assert config.n_candidates == 5
        assert config.max_depth == 5
        assert config.max_search_results == 5
        assert config.exploration_weight == 1.0

    def test_custom_config(self) -> None:
        """Test creating custom configuration."""
        config = LATSConfig(
            model="gpt-4o-mini",
            n_candidates=3,
            max_depth=3,
            max_search_results=10,
            exploration_weight=1.41,
        )
        assert config.model == "gpt-4o-mini"
        assert config.n_candidates == 3
        assert config.max_depth == 3
        assert config.max_search_results == 10
        assert config.exploration_weight == 1.41

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable after creation."""
        config = LATSConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.n_candidates = 10  # type: ignore[misc]

    def test_validate_invalid_n_candidates(self) -> None:
        """Test validation of n_candidates."""
        config = LATSConfig(n_candidates=0)
        with pytest.raises(LATSConfigError, match="n_candidates must be"):
            config.validate()

    def test_validate_invalid_max_depth(self) -> None:
        """Test validation of max_depth."""
        config = LATSConfig(max_depth=-1)
        with pytest.raises(LATSConfigError, match="max_depth must be"):
            config.validate()

    def test_validate_invalid_max_search_results(self) -> None:
        """Test validation of max_search_results."""
        config = LATSConfig(max_search_results=0)
        with pytest.raises(LATSConfigError, match="max_search_results must be"):
            config.validate()

    def test_validate_invalid_exploration_weight(self) -> None:
        """Test validation of exploration_weight."""
        config = LATSConfig(exploration_weight=-0.5)
        with pytest.raises(LATSConfigError, match="exploration_weight must be"):
            config.validate()

    def test_validate_invalid_consistency_weight_too_high(self) -> None:
        """Test validation of consistency_weight above 1.0."""
        config = LATSConfig(consistency_weight=1.5)
        with pytest.raises(LATSConfigError, match="consistency_weight must be"):
            config.validate()

    def test_validate_invalid_consistency_weight_negative(self) -> None:
        """Test validation of negative consistency_weight."""
        config = LATSConfig(consistency_weight=-0.1)
        with pytest.raises(LATSConfigError, match="consistency_weight must be"):
            config.validate()

    def test_validate_valid_config(self) -> None:
        """Test that valid config passes validation."""
        config = LATSConfig()
        config.validate()  # Should not raise
