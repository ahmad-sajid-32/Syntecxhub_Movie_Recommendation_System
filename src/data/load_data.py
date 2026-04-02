"""
File: src/data/load_data.py

Purpose:
- Load the core MovieLens CSV files from data/raw.
- Validate that all required files exist before the project runs.
- Apply clean, predictable dtypes so later steps do not break on bad inference.
- Provide a single bundle object that the rest of the project can use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


logger = logging.getLogger(__name__)


REQUIRED_FILES: Dict[str, str] = {
    "movies": "movies.csv",
    "ratings": "ratings.csv",
    "tags": "tags.csv",
    "links": "links.csv",
}


@dataclass(frozen=True)
class DatasetBundle:
    """
    Holds all core MovieLens tables in one place.

    This keeps the rest of the code simple.
    Instead of passing four separate DataFrames everywhere,
    we pass one object.
    """

    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame


def resolve_data_paths(data_dir: str | Path = "data/raw") -> dict[str, Path]:
    """
    Build absolute paths for all required dataset files.

    Args:
        data_dir:
            Folder where the MovieLens CSV files are stored.

    Returns:
        A dictionary like:
        {
            "movies": Path(...),
            "ratings": Path(...),
            "tags": Path(...),
            "links": Path(...),
        }
    """
    base_path = Path(data_dir).resolve()
    logger.info("Resolving dataset paths from directory: %s", base_path)

    return {
        dataset_name: base_path / filename
        for dataset_name, filename in REQUIRED_FILES.items()
    }


def validate_required_files(paths: dict[str, Path]) -> None:
    """
    Make sure all required files exist before reading anything.

    Why this matters:
    If even one file is missing, later code will fail in a confusing way.
    This function fails early with a clear error message.
    """
    missing_files = [name for name, file_path in paths.items() if not file_path.exists()]

    if missing_files:
        missing_details = "\n".join(
            f"- {name}: {paths[name]}" for name in missing_files
        )
        error_message = (
            "Missing required MovieLens dataset files.\n"
            "Expected files:\n"
            f"{missing_details}\n"
            "Put the CSV files inside the configured data/raw folder."
        )
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    logger.info("All required dataset files were found successfully.")


def load_movies(file_path: str | Path) -> pd.DataFrame:
    """
    Load movies.csv.

    Expected columns:
    - movieId
    - title
    - genres
    """
    logger.info("Loading movies from: %s", file_path)

    movies = pd.read_csv(
        file_path,
        dtype={
            "movieId": "int64",
            "title": "string",
            "genres": "string",
        },
    )

    logger.info("Loaded movies.csv with %d rows.", len(movies))
    return movies


def load_ratings(file_path: str | Path, convert_timestamps: bool = True) -> pd.DataFrame:
    """
    Load ratings.csv.

    Expected columns:
    - userId
    - movieId
    - rating
    - timestamp
    """
    logger.info("Loading ratings from: %s", file_path)

    ratings = pd.read_csv(
        file_path,
        dtype={
            "userId": "int64",
            "movieId": "int64",
            "rating": "float64",
            "timestamp": "int64",
        },
    )

    if convert_timestamps:
        ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)

    logger.info("Loaded ratings.csv with %d rows.", len(ratings))
    return ratings


def load_tags(file_path: str | Path, convert_timestamps: bool = True) -> pd.DataFrame:
    """
    Load tags.csv.

    Expected columns:
    - userId
    - movieId
    - tag
    - timestamp
    """
    logger.info("Loading tags from: %s", file_path)

    tags = pd.read_csv(
        file_path,
        dtype={
            "userId": "int64",
            "movieId": "int64",
            "tag": "string",
            "timestamp": "int64",
        },
    )

    if convert_timestamps:
        tags["tagged_at"] = pd.to_datetime(tags["timestamp"], unit="s", utc=True)

    logger.info("Loaded tags.csv with %d rows.", len(tags))
    return tags


def load_links(file_path: str | Path) -> pd.DataFrame:
    """
    Load links.csv.

    Expected columns:
    - movieId
    - imdbId
    - tmdbId

    Important:
    imdbId and tmdbId can contain missing values,
    so we use pandas nullable integer type Int64.
    """
    logger.info("Loading links from: %s", file_path)

    links = pd.read_csv(
        file_path,
        dtype={
            "movieId": "int64",
            "imdbId": "Int64",
            "tmdbId": "Int64",
        },
    )

    logger.info("Loaded links.csv with %d rows.", len(links))
    return links


def load_movielens_dataset(
    data_dir: str | Path = "data/raw",
    convert_timestamps: bool = True,
) -> DatasetBundle:
    """
    Main entry point for loading the full MovieLens dataset.

    Args:
        data_dir:
            Folder containing movies.csv, ratings.csv, tags.csv, and links.csv.
        convert_timestamps:
            If True, create readable datetime columns for ratings and tags.

    Returns:
        DatasetBundle containing all four DataFrames.
    """
    logger.info("Starting MovieLens dataset load process.")

    paths = resolve_data_paths(data_dir=data_dir)
    validate_required_files(paths=paths)

    movies = load_movies(paths["movies"])
    ratings = load_ratings(
        file_path=paths["ratings"],
        convert_timestamps=convert_timestamps,
    )
    tags = load_tags(
        file_path=paths["tags"],
        convert_timestamps=convert_timestamps,
    )
    links = load_links(paths["links"])

    logger.info(
        "MovieLens dataset loaded successfully. "
        "movies=%d, ratings=%d, tags=%d, links=%d",
        len(movies),
        len(ratings),
        len(tags),
        len(links),
    )

    return DatasetBundle(
        movies=movies,
        ratings=ratings,
        tags=tags,
        links=links,
    )


def build_dataset_summary(dataset: DatasetBundle) -> dict[str, int]:
    """
    Create a small summary dictionary for quick checks.

    This is useful in notebooks, scripts, and logs.
    """
    summary = {
        "movies_count": int(len(dataset.movies)),
        "ratings_count": int(len(dataset.ratings)),
        "tags_count": int(len(dataset.tags)),
        "links_count": int(len(dataset.links)),
        "unique_users": int(dataset.ratings["userId"].nunique()),
        "unique_movies_in_ratings": int(dataset.ratings["movieId"].nunique()),
    }

    logger.info("Dataset summary created: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    loaded_dataset = load_movielens_dataset()
    summary = build_dataset_summary(loaded_dataset)

    print("MovieLens dataset loaded successfully.")
    for key, value in summary.items():
        print(f"{key}: {value}")