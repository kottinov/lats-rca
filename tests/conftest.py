"""Pytest configuration and shared fixtures.

This module provides fixtures used across the test suite, including
test data, mocked components, and configuration helpers.
"""

import os
from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture(scope="session")
def test_env_vars() -> dict[str, str]:
    """Provide test environment variables.

    Returns:
        Dictionary of environment variables for testing
    """
    return {
        "OPENAI_API_KEY": "sk-test-key-for-testing",
        "TAVILY_API_KEY": "tvly-test-key-for-testing",
        "OPENAI_MODEL": "gpt-4o-mini",
        "LATS_N_CANDIDATES": "3",
        "LATS_MAX_DEPTH": "3",
        "LOG_LEVEL": "DEBUG",
    }


@pytest.fixture(autouse=True)
def set_test_env(test_env_vars: dict[str, str]) -> Generator[None, None, None]:
    """Automatically set test environment variables for all tests.

    This fixture runs before each test and ensures environment variables
    are set for configuration loading.

    Args:
        test_env_vars: Test environment variables from fixture
    """
    original_env = os.environ.copy()
    os.environ.update(test_env_vars)
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Provide a mock LLM response structure.

    Returns:
        Dictionary mimicking LangChain LLM response
    """
    return {
        "content": "This is a test response",
        "additional_kwargs": {},
        "response_metadata": {
            "model": "gpt-4o-mini",
            "finish_reason": "stop",
        },
    }


@pytest.fixture
def sample_reflections() -> list[str]:
    """Provide sample reflection text for testing.

    Returns:
        List of reflection strings
    """
    return [
        "The search should focus on recent academic papers.",
        "Consider verifying facts from multiple sources.",
        "The current approach lacks depth in technical details.",
    ]
