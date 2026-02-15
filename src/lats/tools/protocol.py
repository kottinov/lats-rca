"""Tool protocol for LATS.

This module defines the Protocol (structural typing) for tools used in LATS.
Using Protocol instead of ABC provides flexibility for integrating with
LangChain tools while maintaining type safety.

The Protocol approach follows Python's "duck typing" philosophy:
if it quacks like a Tool, it is a Tool.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Protocol for tools that can be used by LATS agents.

    This protocol defines the interface that all tools must implement.
    Tools can be search engines, calculators, databases, or any other
    external resource that agents can use to gather information.

    Using Protocol instead of ABC allows for structural subtyping,
    meaning any class that implements these methods can be used as a Tool
    without explicitly inheriting from a base class.

    The @runtime_checkable decorator allows isinstance() checks at runtime,
    enabling validation of tool objects during execution.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does

    Example:
        >>> class SearchTool:
        ...     @property
        ...     def name(self) -> str:
        ...         return "search"
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Search the web"
        ...
        ...     def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        ...         return {"result": "search results"}
        ...
        ...     def batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ...         return [self.invoke(i) for i in inputs]
        ...
        >>> tool = SearchTool()
        >>> isinstance(tool, Tool)
        True
    """

    @property
    def name(self) -> str:
        """Return the tool's unique identifier.

        The name is used to identify and route tool calls from agents.
        It should be lowercase, alphanumeric, and use underscores for spaces.

        Returns:
            Unique tool identifier (e.g., "web_search", "calculator")
        """
        ...

    @property
    def description(self) -> str:
        """Return a description of what the tool does.

        The description is used by LLMs to decide when to use the tool.
        It should be clear, concise, and specify:
        - What the tool does
        - What inputs it expects
        - What outputs it produces

        Returns:
            Human-readable tool description
        """
        ...

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with given input.

        This is the main method for tool execution. It should:
        - Validate the input parameters
        - Execute the tool's logic
        - Return results in a structured format
        - Handle errors gracefully

        Args:
            input: Dictionary containing tool input parameters.
                   Structure depends on the specific tool.

        Returns:
            Dictionary containing tool results. The structure depends
            on the specific tool but should always be JSON-serializable.

        Raises:
            ValueError: If input is invalid or missing required parameters
            RuntimeError: If tool execution fails

        Example:
            >>> tool.invoke({"query": "Python programming"})
            {"results": [...], "num_results": 10}
        """
        ...

    def batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute the tool with multiple inputs in batch.

        Batch execution can be more efficient than multiple invoke() calls,
        especially for tools that make network requests or database queries.

        The default implementation can simply call invoke() for each input,
        but tools can override this for optimized batch processing.

        Args:
            inputs: List of input dictionaries, each following the same
                   structure as invoke() expects

        Returns:
            List of result dictionaries, one for each input, in the same order

        Raises:
            ValueError: If any input is invalid
            RuntimeError: If batch execution fails

        Example:
            >>> tool.batch([
            ...     {"query": "Python"},
            ...     {"query": "JavaScript"}
            ... ])
            [{"results": [...]}, {"results": [...]}]
        """
        ...


class ToolError(Exception):
    """Base exception for tool-related errors.

    This exception is raised when a tool encounters an error during execution.
    It provides context about what went wrong and suggestions for fixing it.

    Attributes:
        tool_name: Name of the tool that failed
        message: Human-readable error message
        context: Additional error context
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the tool error.

        Args:
            tool_name: Name of the tool that encountered the error
            message: Description of what went wrong
            context: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.tool_name = tool_name
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        """Return formatted error message."""
        ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        if ctx:
            return f"Tool '{self.tool_name}' error: {self.message} ({ctx})"
        return f"Tool '{self.tool_name}' error: {self.message}"
