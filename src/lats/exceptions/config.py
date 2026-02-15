"""Configuration-related exceptions.

This module contains exceptions specifically related to configuration
errors, such as missing environment variables or invalid settings.
"""

from lats.exceptions.core import LATSError


class LATSConfigError(LATSError):
    """Raised when there is a configuration error.

    This exception is raised when the LATS configuration is invalid,
    such as when required settings are missing or have invalid values.

    Examples:
        >>> raise LATSConfigError("Invalid max_depth", context={"max_depth": -1})
        LATSConfigError: Invalid max_depth (max_depth=-1)
    """

    pass


class MissingEnvironmentError(LATSConfigError):
    """Raised when a required environment variable is missing.

    This is a specialized configuration error for missing environment variables,
    which is a common failure mode in applications that rely on external configuration.

    Attributes:
        variable_name: The name of the missing environment variable

    Examples:
        >>> raise MissingEnvironmentError("OPENAI_API_KEY")
        MissingEnvironmentError: Required environment variable 'OPENAI_API_KEY' is not set
    """

    def __init__(self, variable_name: str) -> None:
        """Initialize the missing environment error.

        Args:
            variable_name: The name of the missing environment variable
        """
        self.variable_name = variable_name
        message = (
            f"Required environment variable '{variable_name}' is not set. "
            f"Please set it in your .env file or environment."
        )
        super().__init__(message, context={"variable": variable_name})
