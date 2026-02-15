"""Exception hierarchy for LATS."""

from lats.exceptions.agent import (
    AgentError,
    AgentValidationError,
    CorrelationError,
    EscalationError,
    SearchError,
)
from lats.exceptions.config import LATSConfigError, MissingEnvironmentError
from lats.exceptions.core import LATSError

__all__ = [
    "AgentError",
    "AgentValidationError",
    "CorrelationError",
    "EscalationError",
    "LATSConfigError",
    "LATSError",
    "MissingEnvironmentError",
    "SearchError",
]
