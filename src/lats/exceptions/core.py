"""Core exception hierarchy for LATS.

This module defines the base exception class for all LATS-specific errors,
following the principle of creating a clear exception hierarchy for better
error handling and debugging.
"""


class LATSError(Exception):
    """Base exception for all LATS-specific errors.

    All custom exceptions in the LATS codebase should inherit from this class.
    This allows for catching all LATS-related errors with a single except clause
    while still maintaining specific error types for different failure modes.

    Attributes:
        message: Human-readable error message
        context: Optional dictionary with additional error context
    """

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        """Initialize the LATS error.

        Args:
            message: Human-readable error description
            context: Optional dictionary containing additional error context
                    (e.g., node state, configuration values, etc.)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        """Return formatted error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message

    def __repr__(self) -> str:
        """Return detailed error representation."""
        return f"{self.__class__.__name__}({self.message!r}, context={self.context!r})"
