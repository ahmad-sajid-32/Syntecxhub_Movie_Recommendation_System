"""
File: src/data/preprocess.py

Purpose:
- Clean and standardize MovieLens movie metadata.
- Parse titles and release years from movies.csv.
- Normalize genres and user tags into machine-friendly text.
- Merge movies, tags, links, and rating summaries into one clean table.
- Build the final metadata text column that the recommender will use later.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.data.load_data import DatasetBundle, load_movielens_dataset


logger = logging.getLogger(__name__)


TITLE_YEAR_PATTERN = re.compile(r"^(?P<title>.*)\s\((?P<year>\d{4})\)$")


def clean_text(value: object) -> str:
    """
    Normalize text so the recommender gets cleaner input.

    Rules:
    - convert to lowercase
    - replace non-alphanumeric characters with spaces
    - collapse repeated spaces
    - return empty string for null-like values
    """
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_title_and_year(raw_title: object) -> tuple[str, pd._libs.missing.NAType | int]:
    """
    Split a MovieLens title into:
    - clean display title without the year suffix
    - release year if it exists at the very end, otherwise NA

    Example:
    - "Toy Story (1995)" -> ("Toy Story", 1995)
    - "Heat" -> ("Heat", NA)
    """
    if raw_title is None or pd.isna(raw_title):
        return "", pd.NA

    title_text = str(raw_title).strip()
    match = TITLE_YEAR_PATTERN.match(title_text)

    if match:
        clean_display_title = match.group("title").strip()
        release_year = int(match.group("year"))
        return clean_display_title, release_year

    return title_text, pd.NA


def normalize_genres(raw_genres: object) -> list[str]:
    """
    Convert the raw MovieLens genres string into a clean token list.

    Example:
    - "Adventure|Animation|Children|Comedy|Fantasy"
      becomes
      ["adventure", "animation", "children", "comedy", "fantasy"]

    "(no genres listed)" becomes an empty list.
    """
    if raw_genres is None or pd.isna(raw_genres):
        return []

    genres_text = str(raw_genres).strip()

    if genres_text == "(no genres listed)":
        return []

    normalized_genres: list[str] = []

    for genre in genres_text.split("|"):
        cleaned_genre = clean_text(genre)
        if cleaned_gene := cleaned_genre:
            normalized_genres.append(cleaned_gene)

    return normalized_genres


def build_genres_text(genres_list: list[str]) -> str:
    """
    Convert a genre list back into one space-separated text field.

    This text is later used by TF-IDF.
    """
    if not genres_list:
        return ""
    return " ".join(genres_list)


def preprocess_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean tags and aggregate them per movie.

    Output columns:
    - movieId
    - tag_count
    - tags_text

    Why aggregate:
    The recommender works at movie level, not row-per-tag level.
    """
    logger.info("Starting tags preprocessing.")

    if tags_df.empty:
        logger.warning("tags.csv is empty. Returning empty aggregated tags DataFrame.")
        return pd.DataFrame(
            columns=["movieId", "tag_count", "tags_text"]
        ).astype({"movieId": "int64", "tag_count": "int64", "tags_text": "string"})

    tags = tags_df.copy()

    tags["tag_clean"] = tags["tag"].apply(clean_text)
    tags = tags[tags["tag_clean"] != ""].copy()

    if tags.empty:
        logger.warning("All tags became empty after cleaning.")
        return pd.DataFrame(
            columns=["movieId", "tag_count", "tags_text"]
        ).astype({"movieId": "int64", "tag_count": "int64", "tags_text": "string"})

    aggregated = (
        tags.groupby("movieId", as_index=False)["tag_clean"]
        .agg(
            tag_count="count",
            tags_text=lambda values: " ".join(sorted(set(values))),
        )
        .rename(columns={"tag_clean": "tags_text"})
    )

    aggregated["tags_text"] = aggregated["tags_text"].astype("string")
    aggregated["tag_count"] = aggregated["tag_count"].astype("int64")

    logger.info("Finished tags preprocessing. Movies with tags: %d", len(aggregated))
    return aggregated


def build_rating_summary(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple movie-level rating statistics.

    Output columns:
    - movieId
    - rating_count
    - rating_mean
    - rating_median
    """
    logger.info("Building rating summary.")

    if ratings_df.empty:
        logger.warning("ratings.csv is empty. Returning empty rating summary.")
        return pd.DataFrame(
            columns=["movieId", "rating_count", "rating_mean", "rating_median"]
        ).astype(
            {
                "movieId": "int64",
                "rating_count": "int64",
                "rating_mean": "float64",
                "rating_median": "float64",
            }
        )

    rating_summary = (
        ratings_df.groupby("movieId", as_index=False)["rating"]
        .agg(
            rating_count="count",
            rating_mean="mean",
            rating_median="median",
        )
    )

    rating_summary["rating_count"] = rating_summary["rating_count"].astype("int64")
    rating_summary["rating_mean"] = rating_summary["rating_mean"].astype("float64")
    rating_summary["rating_median"] = rating_summary["rating_median"].astype("float64")

    logger.info("Built rating summary for %d movies.", len(rating_summary))
    return rating_summary


def preprocess_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the movies table and derive useful metadata.

    Output columns added:
    - clean_title
    - release_year
    - title_text
    - genres_list
    - genres_text
    """
    logger.info("Starting movies preprocessing.")

    movies = movies_df.copy()

    extracted = movies["title"].apply(extract_title_and_year)
    movies["clean_title"] = extracted.apply(lambda item: item[0]).astype("string")
    movies["release_year"] = extracted.apply(lambda item: item[1]).astype("Int64")

    movies["title_text"] = movies["clean_title"].apply(clean_text).astype("string")
    movies["genres_list"] = movies["genres"].apply(normalize_genres)
    movies["genres_text"] = movies["genres_list"].apply(build_genres_text).astype("string")

    logger.info("Finished movies preprocessing. Total movies: %d", len(movies))
    return movies


def build_metadata_text(row: pd.Series) -> str:
    """
    Build the final text blob used by the recommender.

    Design choice:
    - title_text once
    - genres_text twice
    - tags_text once

    Why genres twice:
    Genres are reliable in MovieLens.
    Tags are useful, but noisy and sparse.
    Repeating genres gives them a little extra importance without adding complexity.
    """
    title_text = row.get("title_text", "") or ""
    genres_text = row.get("genres_text", "") or ""
    tags_text = row.get("tags_text", "") or ""

    parts = [
        title_text,
        genres_text,
        genres_text,
        tags_text,
    ]

    combined = " ".join(part for part in parts if str(part).strip())
    return clean_text(combined)


def preprocess_movielens_metadata(dataset: DatasetBundle) -> pd.DataFrame:
    """
    Main preprocessing pipeline.

    It merges:
    - cleaned movies
    - aggregated tags
    - links
    - rating summary

    Final output is one movie-level table ready for feature building.
    """
    logger.info("Starting full MovieLens metadata preprocessing pipeline.")

    movies = preprocess_movies(dataset.movies)
    tags_agg = preprocess_tags(dataset.tags)
    rating_summary = build_rating_summary(dataset.ratings)

    processed = movies.merge(tags_agg, on="movieId", how="left")
    processed = processed.merge(dataset.links.copy(), on="movieId", how="left")
    processed = processed.merge(rating_summary, on="movieId", how="left")

    processed["tag_count"] = processed["tag_count"].fillna(0).astype("int64")
    processed["tags_text"] = processed["tags_text"].fillna("").astype("string")
    processed["rating_count"] = processed["rating_count"].fillna(0).astype("int64")
    processed["rating_mean"] = processed["rating_mean"].astype("float64")
    processed["rating_median"] = processed["rating_median"].astype("float64")
    processed["has_tmdb_mapping"] = processed["tmdbId"].notna()

    processed["metadata_text"] = processed.apply(build_metadata_text, axis=1).astype("string")

    processed = processed[
        [
            "movieId",
            "title",
            "clean_title",
            "release_year",
            "genres",
            "genres_list",
            "genres_text",
            "tags_text",
            "tag_count",
            "imdbId",
            "tmdbId",
            "has_tmdb_mapping",
            "rating_count",
            "rating_mean",
            "rating_median",
            "metadata_text",
        ]
    ].copy()

    logger.info(
        "Finished preprocessing pipeline. Output rows=%d, columns=%d",
        len(processed),
        len(processed.columns),
    )

    return processed


def save_processed_movies(
    processed_df: pd.DataFrame,
    output_path: str | Path = "data/processed/movies_metadata.csv",
) -> Path:
    """
    Save the processed movie metadata to disk.

    The parent folder is created automatically if missing.
    """
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    export_df = processed_df.copy()

    # Lists do not serialize nicely to plain CSV unless converted.
    export_df["genres_list"] = export_df["genres_list"].apply(
        lambda values: "|".join(values) if isinstance(values, list) else ""
    )

    export_df.to_csv(output_file, index=False)

    logger.info("Saved processed metadata to: %s", output_file)
    return output_file


def run_preprocessing_pipeline(
    data_dir: str | Path = "data/raw",
    output_path: str | Path = "data/processed/movies_metadata.csv",
) -> pd.DataFrame:
    """
    Convenience function for scripts and manual runs.

    It:
    1. loads raw MovieLens data
    2. preprocesses it
    3. saves the result
    4. returns the processed DataFrame
    """
    logger.info("Running preprocessing pipeline with data_dir=%s", data_dir)

    dataset = load_movielens_dataset(data_dir=data_dir)
    processed = preprocess_movielens_metadata(dataset=dataset)
    save_processed_movies(processed_df=processed, output_path=output_path)

    logger.info("Preprocessing pipeline completed successfully.")
    return processed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    processed_movies = run_preprocessing_pipeline()

    print("Preprocessing completed successfully.")
    print(f"Processed rows: {len(processed_movies)}")
    print("Sample columns:")
    print(processed_movies.head(5)[["movieId", "title", "clean_title", "release_year", "metadata_text"]])