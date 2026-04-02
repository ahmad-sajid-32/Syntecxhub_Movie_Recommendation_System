"""
File: src/recommender/recommend.py

Purpose:
- Load saved recommendation artifacts.
- Find movies by exact or partial title.
- Return top-N similar movies using cosine similarity.
- Keep the recommendation interface simple and predictable.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.build_features import FeatureArtifacts, load_processed_movies


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendationResult:
    """
    Represents one recommended movie.

    This keeps output structured instead of returning loose dictionaries.
    """

    movie_id: int
    title: str
    clean_title: str
    release_year: int | None
    similarity_score: float
    genres: str
    rating_count: int
    rating_mean: float | None
    metadata_text: str


@dataclass(frozen=True)
class RecommenderAssets:
    """
    Holds everything needed to generate recommendations.

    This object is the runtime package for the recommender.
    """

    movies_df: pd.DataFrame
    similarity_matrix: np.ndarray
    vectorizer: TfidfVectorizer
    movie_id_to_index: dict[int, int]
    index_to_movie_id: dict[int, int]


def validate_recommender_assets(assets: RecommenderAssets) -> None:
    """
    Validate that the recommender assets are internally consistent.

    Why this matters:
    If the matrix row count does not match the movie table row count,
    every recommendation becomes wrong.
    """
    movies_count = len(assets.movies_df)
    similarity_shape = assets.similarity_matrix.shape

    if movies_count == 0:
        raise ValueError("movies_df is empty. Cannot generate recommendations.")

    if similarity_shape[0] != similarity_shape[1]:
        raise ValueError(
            f"Similarity matrix must be square. Received shape={similarity_shape}."
        )

    if similarity_shape[0] != movies_count:
        raise ValueError(
            "Mismatch between movies_df row count and similarity matrix size. "
            f"movies_df rows={movies_count}, similarity_matrix shape={similarity_shape}."
        )

    if len(assets.movie_id_to_index) != movies_count:
        raise ValueError(
            "movie_id_to_index size does not match movies_df row count. "
            f"map size={len(assets.movie_id_to_index)}, movies_df rows={movies_count}."
        )

    if len(assets.index_to_movie_id) != movies_count:
        raise ValueError(
            "index_to_movie_id size does not match movies_df row count. "
            f"map size={len(assets.index_to_movie_id)}, movies_df rows={movies_count}."
        )

    logger.info(
        "Recommender assets validation passed. movies=%d similarity_shape=%s",
        movies_count,
        similarity_shape,
    )


def load_pickle_file(file_path: str | Path) -> Any:
    """
    Load a pickle file with a clear error if missing.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Required artifact file not found: {path}")

    with path.open("rb") as file:
        return pickle.load(file)


def load_recommender_assets(
    processed_movies_path: str | Path = "data/processed/movies_metadata.csv",
    artifact_dir: str | Path = "artifacts",
) -> RecommenderAssets:
    """
    Load all saved assets required for recommendation.

    Expected files inside artifact_dir:
    - tfidf_vectorizer.pkl
    - tfidf_matrix.npz
    - similarity_matrix.npy
    - movie_index_maps.pkl

    Important:
    tfidf_matrix is loaded mainly for artifact completeness verification.
    It is not needed directly for simple similarity lookup in this file.
    """
    artifact_base = Path(artifact_dir).resolve()
    logger.info("Loading recommender assets from artifact directory: %s", artifact_base)

    movies_df = load_processed_movies(file_path=processed_movies_path)

    vectorizer = load_pickle_file(artifact_base / "tfidf_vectorizer.pkl")
    _ = load_npz(artifact_base / "tfidf_matrix.npz")
    similarity_matrix = np.load(artifact_base / "similarity_matrix.npy")
    index_maps = load_pickle_file(artifact_base / "movie_index_maps.pkl")

    assets = RecommenderAssets(
        movies_df=movies_df,
        similarity_matrix=similarity_matrix,
        vectorizer=vectorizer,
        movie_id_to_index=index_maps["movie_id_to_index"],
        index_to_movie_id=index_maps["index_to_movie_id"],
    )

    validate_recommender_assets(assets)
    logger.info("Recommender assets loaded successfully.")
    return assets


def normalize_title_for_search(title: str) -> str:
    """
    Normalize a title string for safer matching.
    """
    return " ".join(str(title).strip().lower().split())


def find_movies_by_partial_title(
    movies_df: pd.DataFrame,
    query: str,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Find candidate movies using case-insensitive partial matching.

    Search fields:
    - clean_title
    - title

    This is a helper for ambiguous user input.
    """
    if not query or not query.strip():
        raise ValueError("Query title cannot be empty.")

    normalized_query = normalize_title_for_search(query)

    clean_title_series = (
        movies_df["clean_title"].fillna("").astype("string").str.lower().str.strip()
    )
    raw_title_series = (
        movies_df["title"].fillna("").astype("string").str.lower().str.strip()
    )

    mask = clean_title_series.str.contains(normalized_query, regex=False) | raw_title_series.str.contains(
        normalized_query, regex=False
    )

    matches = movies_df.loc[mask].copy()

    if matches.empty:
        logger.warning("No movies found for partial title query: %s", query)
        return matches

    sort_columns: list[str] = []
    ascending: list[bool] = []

    if "rating_count" in matches.columns:
        sort_columns.append("rating_count")
        ascending.append(False)

    if "rating_mean" in matches.columns:
        sort_columns.append("rating_mean")
        ascending.append(False)

    if "release_year" in matches.columns:
        sort_columns.append("release_year")
        ascending.append(False)

    if sort_columns:
        matches = matches.sort_values(by=sort_columns, ascending=ascending)

    matches = matches.head(limit).reset_index(drop=True)

    logger.info(
        "Found %d candidate matches for partial title query='%s'.",
        len(matches),
        query,
    )
    return matches


def resolve_movie_id_from_title(
    movies_df: pd.DataFrame,
    title: str,
    prefer_exact: bool = True,
) -> int:
    """
    Resolve a movieId from a title query.

    Matching strategy:
    1. exact match on clean_title
    2. exact match on title
    3. partial match fallback

    If multiple exact matches exist, pick the one with:
    - higher rating_count
    - then higher rating_mean
    - then newer release_year

    This is a deliberate tie-break strategy.
    """
    if not title or not title.strip():
        raise ValueError("Title cannot be empty.")

    normalized_query = normalize_title_for_search(title)
    working_df = movies_df.copy()

    working_df["clean_title_norm"] = (
        working_df["clean_title"].fillna("").astype("string").str.lower().str.strip()
    )
    working_df["title_norm"] = (
        working_df["title"].fillna("").astype("string").str.lower().str.strip()
    )

    if prefer_exact:
        exact_matches = working_df[
            (working_df["clean_title_norm"] == normalized_query)
            | (working_df["title_norm"] == normalized_query)
        ].copy()

        if not exact_matches.empty:
            sort_columns: list[str] = []
            ascending: list[bool] = []

            if "rating_count" in exact_matches.columns:
                sort_columns.append("rating_count")
                ascending.append(False)

            if "rating_mean" in exact_matches.columns:
                sort_columns.append("rating_mean")
                ascending.append(False)

            if "release_year" in exact_matches.columns:
                sort_columns.append("release_year")
                ascending.append(False)

            if sort_columns:
                exact_matches = exact_matches.sort_values(by=sort_columns, ascending=ascending)

            selected_movie_id = int(exact_matches.iloc[0]["movieId"])
            logger.info(
                "Resolved movie title '%s' by exact match to movieId=%d.",
                title,
                selected_movie_id,
            )
            return selected_movie_id

    partial_matches = find_movies_by_partial_title(working_df, query=title, limit=1)

    if partial_matches.empty:
        raise ValueError(f"No movie found for title query: {title}")

    selected_movie_id = int(partial_matches.iloc[0]["movieId"])
    logger.info(
        "Resolved movie title '%s' by partial match to movieId=%d.",
        title,
        selected_movie_id,
    )
    return selected_movie_id


def get_movie_row_by_id(movies_df: pd.DataFrame, movie_id: int) -> pd.Series:
    """
    Get the movie row for a specific movieId.
    """
    matches = movies_df.loc[movies_df["movieId"] == movie_id]

    if matches.empty:
        raise ValueError(f"movieId {movie_id} was not found in movies_df.")

    return matches.iloc[0]


def to_nullable_int(value: object) -> int | None:
    """
    Convert pandas/NumPy nullable integer-like values into plain Python int or None.
    """
    if value is None or pd.isna(value):
        return None
    return int(value)


def to_nullable_float(value: object) -> float | None:
    """
    Convert pandas/NumPy nullable float-like values into plain Python float or None.
    """
    if value is None or pd.isna(value):
        return None
    return float(value)


def build_recommendation_result(row: pd.Series, similarity_score: float) -> RecommendationResult:
    """
    Convert a movie row into a structured RecommendationResult.
    """
    return RecommendationResult(
        movie_id=int(row["movieId"]),
        title=str(row.get("title", "")),
        clean_title=str(row.get("clean_title", "")),
        release_year=to_nullable_int(row.get("release_year")),
        similarity_score=float(similarity_score),
        genres=str(row.get("genres", "")),
        rating_count=int(row.get("rating_count", 0)) if not pd.isna(row.get("rating_count", 0)) else 0,
        rating_mean=to_nullable_float(row.get("rating_mean")),
        metadata_text=str(row.get("metadata_text", "")),
    )


def get_similar_movies_by_movie_id(
    assets: RecommenderAssets,
    movie_id: int,
    top_n: int = 10,
    exclude_input_movie: bool = True,
    min_rating_count: int = 0,
) -> list[RecommendationResult]:
    """
    Get top-N similar movies for a given movieId.

    Logic:
    - find the movie's row index
    - read its similarity scores against all movies
    - sort descending
    - optionally exclude the seed movie itself
    - optionally filter weakly rated movies
    """
    validate_recommender_assets(assets)

    if movie_id not in assets.movie_id_to_index:
        raise ValueError(f"movieId {movie_id} was not found in the recommender index.")

    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    source_index = assets.movie_id_to_index[movie_id]
    similarity_scores = assets.similarity_matrix[source_index]

    ranked_indices = np.argsort(similarity_scores)[::-1]

    recommendations: list[RecommendationResult] = []

    for candidate_index in ranked_indices:
        candidate_movie_id = assets.index_to_movie_id[int(candidate_index)]

        if exclude_input_movie and candidate_movie_id == movie_id:
            continue

        candidate_row = assets.movies_df.iloc[int(candidate_index)]

        rating_count_value = candidate_row.get("rating_count", 0)
        rating_count = 0 if pd.isna(rating_count_value) else int(rating_count_value)

        if rating_count < min_rating_count:
            continue

        result = build_recommendation_result(
            row=candidate_row,
            similarity_score=float(similarity_scores[int(candidate_index)]),
        )
        recommendations.append(result)

        if len(recommendations) >= top_n:
            break

    logger.info(
        "Generated %d recommendations for movieId=%d with top_n=%d.",
        len(recommendations),
        movie_id,
        top_n,
    )
    return recommendations


def get_similar_movies_by_title(
    assets: RecommenderAssets,
    title: str,
    top_n: int = 10,
    exclude_input_movie: bool = True,
    min_rating_count: int = 0,
) -> list[RecommendationResult]:
    """
    Get top-N similar movies using a title query.
    """
    movie_id = resolve_movie_id_from_title(assets.movies_df, title=title)
    return get_similar_movies_by_movie_id(
        assets=assets,
        movie_id=movie_id,
        top_n=top_n,
        exclude_input_movie=exclude_input_movie,
        min_rating_count=min_rating_count,
    )


def get_candidate_titles(
    assets: RecommenderAssets,
    query: str,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Return possible movie matches for user inspection.

    This is useful when a title is ambiguous.
    """
    matches = find_movies_by_partial_title(assets.movies_df, query=query, limit=limit)

    if matches.empty:
        return matches

    selected_columns = [
        column
        for column in ["movieId", "title", "clean_title", "release_year", "genres", "rating_count", "rating_mean"]
        if column in matches.columns
    ]

    return matches[selected_columns].copy()


def recommendation_results_to_dataframe(
    results: list[RecommendationResult],
) -> pd.DataFrame:
    """
    Convert recommendation results into a DataFrame for notebooks or printing.
    """
    if not results:
        return pd.DataFrame(
            columns=[
                "movie_id",
                "title",
                "clean_title",
                "release_year",
                "similarity_score",
                "genres",
                "rating_count",
                "rating_mean",
                "metadata_text",
            ]
        )

    return pd.DataFrame([result.__dict__ for result in results])


def build_runtime_assets_from_feature_artifacts(
    artifacts: FeatureArtifacts,
) -> RecommenderAssets:
    """
    Convert FeatureArtifacts into runtime recommender assets.

    This is useful when working in-memory inside notebooks or scripts.
    """
    assets = RecommenderAssets(
        movies_df=artifacts.movies_df.copy(),
        similarity_matrix=artifacts.similarity_matrix,
        vectorizer=artifacts.vectorizer,
        movie_id_to_index=artifacts.movie_id_to_index,
        index_to_movie_id=artifacts.index_to_movie_id,
    )

    validate_recommender_assets(assets)
    return assets


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    assets = load_recommender_assets()

    seed_title = "Toy Story"
    print(f"Seed title: {seed_title}")
    print("-" * 80)

    try:
        recommendations = get_similar_movies_by_title(
            assets=assets,
            title=seed_title,
            top_n=10,
            exclude_input_movie=True,
            min_rating_count=10,
        )
        recommendations_df = recommendation_results_to_dataframe(recommendations)

        if recommendations_df.empty:
            print("No recommendations found.")
        else:
            print(
                recommendations_df[
                    [
                        "title",
                        "release_year",
                        "similarity_score",
                        "genres",
                        "rating_count",
                        "rating_mean",
                    ]
                ].to_string(index=False)
            )
    except ValueError as error:
        print(f"Recommendation error: {error}")