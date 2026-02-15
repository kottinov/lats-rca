"""Log analysis tools for LATS-RCA.

This module provides LangChain tools for searching and analyzing log files
during root cause analysis investigations. Tools support both structured
and unstructured log formats.

Following Robust Python principles:
- Type annotations on all functions
- Clear error handling with descriptive messages
- Immutable configuration where possible
- Documented return formats
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from langchain_core.tools import tool


def _format_dir_entry(path: Path) -> str:
    """Format a directory entry for display.

    Args:
        path: Path to format

    Returns:
        Formatted string with type and name

    Example:
        >>> _format_dir_entry(Path("/tmp/test.log"))
        'FILE: test.log'
    """
    kind = "DIR" if path.is_dir() else "FILE"
    return f"{kind}: {path.name}"


@tool
def list_files(directory_path: str) -> str:
    """List files and subdirectories for a given path.

    Args:
        directory_path: Path to directory to list

    Returns:
        Formatted listing of directory contents, or error message

    Example:
        >>> list_files("/var/log")
        'DIR: system\\nFILE: app.log\\nFILE: error.log'
    """
    directory = Path(directory_path).expanduser().resolve()

    if not directory.exists():
        return f"Error: directory '{directory}' does not exist"

    if not directory.is_dir():
        return f"Error: '{directory}' is not a directory"

    entries = sorted(
        directory.iterdir(),
        key=lambda p: (not p.is_dir(), p.name.lower())
    )

    if not entries:
        return "Empty directory"

    return "\n".join(_format_dir_entry(path) for path in entries)


@tool
def read_file(
    file_path: str,
    max_lines: int = -1,
    max_chars: int = -1,
) -> str:
    """Read file contents with UTF-8 fallback and error handling.

    This tool reads complete files by default. Use limits for large files
    or prefer grep_file/search_directory for targeted searches.

    Args:
        file_path: Path to the file to read
        max_lines: Maximum lines to return (-1 for unlimited)
        max_chars: Maximum characters to return (-1 for unlimited)

    Returns:
        File contents (possibly truncated) or error message

    Example:
        >>> read_file("/var/log/app.log", max_lines=100)
        'line 1\\nline 2\\n...\\n[Truncated: showing first 100 of 500 lines...]'
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return f"Error: file '{path}' does not exist"

    if not path.is_file():
        return f"Error: '{path}' is not a regular file"

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()

        if max_lines > 0 and len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            content += (
                f"\n\n[Truncated: showing first {max_lines} of {len(lines)} lines. "
                "Use grep_file to search for specific patterns.]"
            )

        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars]
            content += (
                f"\n\n[Truncated at {max_chars} characters. "
                "Use grep_file to search for specific patterns.]"
            )

        return content

    except OSError as exc:
        return f"Error reading '{path}': {exc}"


def _read_file_contents(file_path: str) -> str:
    """Internal helper to read file contents without limits.

    Args:
        file_path: Path to file to read

    Returns:
        File contents or error message
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return f"Error: file '{path}' does not exist"

    if not path.is_file():
        return f"Error: '{path}' is not a regular file"

    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return f"Error reading '{path}': {exc}"


@tool
def grep_file(file_path: str, keyword: str) -> str:
    """Return matching lines for a keyword in a file.

    Performs case-insensitive search and returns line numbers with matches.

    Args:
        file_path: Path to file to search
        keyword: Search term (case-insensitive)

    Returns:
        Formatted matches with line numbers, or error/no-match message

    Example:
        >>> grep_file("/var/log/app.log", "error")
        '/var/log/app.log:42: ERROR connection timeout\\n/var/log/app.log:87: ERROR ...'
    """
    contents = _read_file_contents(file_path)

    if contents.startswith("Error:"):
        return contents

    keyword_lower = keyword.lower()
    matches: list[str] = []

    for line_number, line in enumerate(contents.splitlines(), start=1):
        if keyword_lower in line.lower():
            matches.append(f"{Path(file_path)}:{line_number}: {line}")

    if not matches:
        return f"No matches for '{keyword}'"

    return "\n".join(matches)


def _iter_candidate_files(directory: Path) -> Iterable[Path]:
    """Iterate over all files in a directory tree.

    Args:
        directory: Root directory to traverse

    Yields:
        Path objects for each file (not directories)
    """
    for root, _, files in os.walk(directory):
        for file_name in files:
            yield Path(root) / file_name


@tool
def search_directory(directory_path: str, keyword: str) -> str:
    """Search for a keyword across files in a directory tree.

    Recursively searches all files, limiting results per file to keep
    output manageable. Case-insensitive search.

    Args:
        directory_path: Root directory to search
        keyword: Search term (case-insensitive)

    Returns:
        Formatted matches with file paths and line numbers, or error message

    Example:
        >>> search_directory("/var/log", "timeout")
        '  /var/log/app.log:42: connection timeout\\n  ... (5 more matches)'
    """
    directory = Path(directory_path).expanduser().resolve()

    if not directory.exists():
        return f"Error: directory '{directory}' does not exist"

    if not directory.is_dir():
        return f"Error: '{directory}' is not a directory"

    keyword_lower = keyword.lower()
    results: list[str] = []

    for file_path in _iter_candidate_files(directory):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        matching_lines: list[str] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if keyword_lower in line.lower():
                matching_lines.append(f"  {file_path}:{line_number}: {line}")

        if matching_lines:
            results.extend(matching_lines[:10])
            if len(matching_lines) > 10:
                remaining = len(matching_lines) - 10
                results.append(f"  ... ({remaining} more matches in {file_path.name})")

    if not results:
        return f"No matches for '{keyword}' in directory tree"

    return "\n".join(results)


LOG_TOOLS = [list_files, read_file, grep_file, search_directory]

__all__ = [
    "LOG_TOOLS",
    "grep_file",
    "list_files",
    "read_file",
    "search_directory",
]
