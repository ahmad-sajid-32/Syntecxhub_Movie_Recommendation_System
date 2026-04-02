"""
File: src/utils/helpers.py

Purpose:
- Provide shared utility functions used across the project.
- Centralize logging setup, path handling, JSON helpers, and simple DataFrame summaries.
- Keep common support logic out of the model and recommendation modules.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


LOGGER_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure application logging once.

    Why this matters:
    Without this, each script may log differently or not log at all.
    """
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(level=level, format=LOGGER_FORMAT)
    else:
        root_logger.setLevel(level)


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists and return its resolved path.
    """
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_directory(file_path: str | Path) -> Path:
    """
    Ensure the parent directory for a file path exists.

    Returns the resolved file path.
    """
    resolved_file_path = Path(file_path).resolve()
    resolved_file_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_file_path


def save_json(data: Any, file_path: str | Path, indent: int = 2) -> Path:
    """
    Save Python data as JSON.
    """
    output_file = ensure_parent_directory(file_path)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

    return output_file


def load_json(file_path: str | Path) -> Any:
    """
    Load JSON data from disk.
    """
    input_file = Path(file_path).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"JSON file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def dataframe_overview(df: pd.DataFrame, name: str = "dataframe") -> dict[str, Any]:
    """
    Build a small summary of a DataFrame.

    This is useful in scripts and notebooks where quick inspection matters.
    """
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def print_dataframe_overview(df: pd.DataFrame, name: str = "dataframe") -> None:
    """
    Print a simple DataFrame summary.
    """
    overview = dataframe_overview(df, name=name)
    print(f"DataFrame: {overview['name']}")
    print(f"Rows: {overview['rows']}")
    print(f"Columns: {overview['columns']}")
    print(f"Missing values total: {overview['missing_values_total']}")
    print(f"Duplicate rows: {overview['duplicate_rows']}")
    print(f"Column names: {overview['column_names']}")


def preview_records(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Return a small preview of DataFrame rows as records.

    This is useful for JSON output or debugging.
    """
    preview_df = df.copy()

    if columns is not None:
        preview_df = preview_df[columns]

    return preview_df.head(limit).to_dict(orient="records")


@contextmanager
def timed_block(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """
    Measure execution time for a block of code.

    Example:
        with timed_block("preprocessing", logger):
            ...
    """
    active_logger = logger or logging.getLogger(__name__)
    start_time = time.perf_counter()
    active_logger.info("Started: %s", label)

    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        active_logger.info("Finished: %s | elapsed_seconds=%.4f", label, elapsed_seconds)
