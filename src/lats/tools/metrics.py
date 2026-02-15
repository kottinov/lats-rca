"""Metrics query tools for LATS-RCA.

This module provides LangChain tools for loading and searching metrics
from CSV files. The agent can explore metrics data to validate hypotheses
during root cause analysis.

Following Robust Python principles:
- Type annotations throughout
- Clear error handling
- Domain-specific types
"""

from __future__ import annotations

from typing import NewType

import pandas as pd
from langchain_core.tools import tool

ScenarioName = NewType("ScenarioName", str)
"""Name of a test scenario in the metrics dataset."""

MetricName = NewType("MetricName", str)
"""Name of a metric column."""


class MetricsError(Exception):
    """Base exception for metrics tool errors."""


class DataLoadError(MetricsError):
    """Raised when CSV data cannot be loaded."""


def _load_dataframe(csv_path: str) -> pd.DataFrame:
    """Load metrics CSV with error handling.

    Args:
        csv_path: Path to CSV file

    Returns:
        Loaded DataFrame

    Raises:
        DataLoadError: If CSV cannot be loaded
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise DataLoadError(f"CSV file '{csv_path}' is empty")
        return df
    except FileNotFoundError as exc:
        raise DataLoadError(f"CSV file '{csv_path}' not found") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"CSV file '{csv_path}' contains no data") from exc
    except Exception as exc:
        raise DataLoadError(
            f"Failed to load metrics CSV '{csv_path}': {exc}"
        ) from exc


@tool
def load_metrics_csv(csv_path: str) -> str:
    """Load a metrics CSV and return available columns and scenarios.

    Use this tool first to understand what metrics are available in the dataset.

    Args:
        csv_path: Path to metrics CSV file

    Returns:
        Summary of available columns and scenarios

    Example:
        >>> load_metrics_csv("metrics.csv")
        'Loaded 500 rows with 25 columns\\n\\nAvailable scenarios:\\n  - correct: 100...'
    """
    try:
        df = _load_dataframe(csv_path)
    except DataLoadError as exc:
        return f"Error: {exc}"

    lines = [
        f"Loaded {len(df)} rows with {len(df.columns)} columns",
        "",
        "Available columns (metrics):",
    ]

    columns_to_show = df.columns[:10]
    for col in columns_to_show:
        lines.append(f"  - {col}")

    if len(df.columns) > 10:
        lines.append(f"  ... and {len(df.columns) - 10} more columns")

    if "test_name" in df.columns:
        scenarios = df["test_name"].value_counts()
        lines.extend(["", "Available scenarios:"])
        for scenario, count in scenarios.items():
            lines.append(f"  - {scenario}: {count} data points")

    return "\n".join(lines)


@tool
def query_metrics(
    csv_path: str,
    scenario: str,
    metric_name: str,
    operation: str = "mean",
) -> str:
    """Query a specific metric for a scenario.

    Args:
        csv_path: Path to metrics CSV
        scenario: Scenario name (use load_metrics_csv to see available scenarios)
        metric_name: Metric column name to query
        operation: Operation to perform: "mean", "min", "max", "std", "count"

    Returns:
        Query result or error message

    Example:
        >>> query_metrics("metrics.csv", "high_load", "go_goroutines", "mean")
        'go_goroutines for scenario \\'high_load\\':\\n  mean: 1245.67'
    """
    try:
        df = _load_dataframe(csv_path)
    except DataLoadError as exc:
        return f"Error: {exc}"

    if "test_name" in df.columns:
        scenario_data = df[df["test_name"] == scenario]
        if scenario_data.empty:
            available = df["test_name"].unique()
            return (
                f"Error: Scenario '{scenario}' not found. "
                f"Available: {', '.join(available)}"
            )
    else:
        scenario_data = df

    if metric_name not in scenario_data.columns:
        return f"Error: Metric '{metric_name}' not found in columns"

    metric_values = scenario_data[metric_name]

    operations_map = {
        "mean": metric_values.mean,
        "min": metric_values.min,
        "max": metric_values.max,
        "std": metric_values.std,
        "count": metric_values.count,
    }

    if operation not in operations_map:
        return (
            f"Error: Unknown operation '{operation}'. "
            f"Available: {', '.join(operations_map.keys())}"
        )

    result = operations_map[operation]()

    return (
        f"{metric_name} for scenario '{scenario}':\n"
        f"  {operation}: {result:.2f}"
    )


@tool
def compare_metric_across_scenarios(
    csv_path: str,
    metric_name: str,
    scenarios: str,
) -> str:
    """Compare a metric across multiple scenarios.

    Args:
        csv_path: Path to metrics CSV
        metric_name: Metric column name to compare
        scenarios: Comma-separated scenario names (e.g., "correct,high_load")

    Returns:
        Comparison table or error message

    Example:
        >>> compare_metric_across_scenarios(
        ...     "metrics.csv",
        ...     "go_goroutines",
        ...     "correct,high_load"
        ... )
        'go_goroutines comparison:\\n  correct: mean=100.5, max=150\\n  high_load: mean=1200.3...'
    """
    try:
        df = _load_dataframe(csv_path)
    except DataLoadError as exc:
        return f"Error: {exc}"

    if "test_name" not in df.columns:
        return "Error: CSV must have 'test_name' column for scenario comparison"

    if metric_name not in df.columns:
        return f"Error: Metric '{metric_name}' not found in columns"

    scenario_list = [s.strip() for s in scenarios.split(",")]
    lines = [f"{metric_name} comparison:"]

    for scenario in scenario_list:
        scenario_data = df[df["test_name"] == scenario]

        if scenario_data.empty:
            lines.append(f"  {scenario}: NOT FOUND")
            continue

        metric_values = scenario_data[metric_name]
        mean_val = metric_values.mean()
        min_val = metric_values.min()
        max_val = metric_values.max()

        lines.append(
            f"  {scenario}: mean={mean_val:.2f}, min={min_val:.2f}, max={max_val:.2f}"
        )

    return "\n".join(lines)


@tool
def search_metrics_by_threshold(
    csv_path: str,
    metric_name: str,
    threshold: float,
    operation: str = "greater",
) -> str:
    """Find scenarios where a metric exceeds a threshold.

    Args:
        csv_path: Path to metrics CSV
        metric_name: Metric column name to check
        threshold: Threshold value
        operation: Comparison operation: "greater", "less", "equal"

    Returns:
        List of matching scenarios or error message

    Example:
        >>> search_metrics_by_threshold("metrics.csv", "go_goroutines", 1000, "greater")
        'Scenarios where go_goroutines > 1000:\\n  - high_load: 85 data points exceed...'
    """
    try:
        df = _load_dataframe(csv_path)
    except DataLoadError as exc:
        return f"Error: {exc}"

    if metric_name not in df.columns:
        return f"Error: Metric '{metric_name}' not found in columns"

    if operation == "greater":
        filtered = df[df[metric_name] > threshold]
        op_symbol = ">"
    elif operation == "less":
        filtered = df[df[metric_name] < threshold]
        op_symbol = "<"
    elif operation == "equal":
        filtered = df[df[metric_name] == threshold]
        op_symbol = "=="
    else:
        return f"Error: Unknown operation '{operation}'. Use: greater, less, equal"

    if filtered.empty:
        return f"No data points where {metric_name} {op_symbol} {threshold}"

    lines = [f"Scenarios where {metric_name} {op_symbol} {threshold}:"]

    if "test_name" in filtered.columns:
        scenario_counts = filtered["test_name"].value_counts()
        for scenario, count in scenario_counts.items():
            lines.append(f"  - {scenario}: {count} data points exceed threshold")
    else:
        lines.append(f"  {len(filtered)} total data points match")

    return "\n".join(lines)


METRICS_TOOLS = [
    load_metrics_csv,
    query_metrics,
    compare_metric_across_scenarios,
    search_metrics_by_threshold,
]

__all__ = [
    "METRICS_TOOLS",
    "DataLoadError",
    "MetricsError",
    "compare_metric_across_scenarios",
    "load_metrics_csv",
    "query_metrics",
    "search_metrics_by_threshold",
]
