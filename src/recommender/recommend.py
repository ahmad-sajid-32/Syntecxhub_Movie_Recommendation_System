"""
File: src/recommender/recommend.py

Purpose:
- Load saved recommendation artifacts.
- Find movies by exact or partial title.
- Return top-N similar movies using on-demand TF-IDF scoring.
- Keep the recommendation interface simple and predictable.

Important design change:
- This file no longer depends on similarity_matrix.npy.
- Recommendations are scored on demand from the sparse TF-IDF matrix.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import linear_kernel

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

    Important design change:
    - tfidf_matrix is now the primary runtime artifact
    - full dense similarity_matrix is no longer required
    """

    movies_df: pd.DataFrame
    tfidf_matrix: csr_matrix
    movie_id_to_index: dict[int, int]
    index_to_movie_id: dict[int, int]


def validate_recommender_assets(assets: RecommenderAssets) -> None:
    """
    Validate that the recommender assets are internally consistent.

    Why this matters:
    If the TF-IDF matrix row count does not match the movie table row count,
    every recommendation becomes wrong.
    """
    movies_count = len(assets.movies_df)
    tfidf_shape = assets.tfidf_matrix.shape

    if movies_count == 0:
        raise ValueError("movies_df is empty. Cannot generate recommendations.")

    if tfidf_shape[0] != movies_count:
        raise ValueError(
            "Mismatch between movies_df row count and TF-IDF matrix size. "
            f"movies_df rows={movies_count}, tfidf_matrix shape={tfidf_shape}."
        )

    if tfidf_shape[1] == 0:
        raise ValueError("TF-IDF matrix has zero columns. Cannot score recommendations.")

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
        "Recommender assets validation passed. movies=%d tfidf_shape=%s",
        movies_count,
        tfidf_shape,
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


def load_tfidf_matrix(file_path: str | Path) -> csr_matrix:
    """
    Load the saved sparse TF-IDF matrix.

    Row slicing is part of runtime scoring, so CSR format is preferred.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Required TF-IDF matrix file not found: {path}")

    tfidf_matrix = load_npz(path)

    if not isinstance(tfidf_matrix, csr_matrix):
        logger.info("Converting loaded TF-IDF matrix to CSR format.")
        tfidf_matrix = tfidf_matrix.tocsr()

    return tfidf_matrix


def load_recommender_assets(
    processed_movies_path: str | Path = "data/processed/movies_metadata.csv",
    artifact_dir: str | Path = "artifacts",
) -> RecommenderAssets:
    """
    Load the saved assets required for recommendation.

    Expected files inside artifact_dir:
    - tfidf_matrix.npz
    - movie_index_maps.pkl

    Notes:
    - similarity_matrix.npy is no longer required
    - tfidf_vectorizer.pkl is not required for title-to-title runtime lookup
    """
    artifact_base = Path(artifact_dir).resolve()
    logger.info("Loading recommender assets from artifact directory: %s", artifact_base)

    movies_df = load_processed_movies(file_path=processed_movies_path)
    tfidf_matrix = load_tfidf_matrix(artifact_base / "tfidf_matrix.npz")
    index_maps = load_pickle_file(artifact_base / "movie_index_maps.pkl")

    assets = RecommenderAssets(
        movies_df=movies_df,
        tfidf_matrix=tfidf_matrix,
        movie_id_to_index=index_maps["movie_id_to_index"],
        index_to_movie_id=index_maps["index_to_movie_id"],
    )

    validate_recommender_assets(assets)
    logger.info("Recommender assets loaded successfully.")
    return assets


def load_movies_for_title_search(
    processed_movies_path: str | Path = "data/processed/movies_metadata.csv",
) -> pd.DataFrame:
    """
    Load only processed movie metadata.

    This is the lightweight path for candidate title lookup.
    It does not load TF-IDF artifacts.
    """
    return load_processed_movies(file_path=processed_movies_path)


def normalize_title_for_search(title: str) -> str:
    """
    Normalize a title string for safer matching.
    """
    return " ".join(str(title).strip().lower().split())


def prepare_title_search_frame(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create normalized title columns used by exact and partial matching.
    """
    working_df = movies_df.copy()

    working_df["clean_title_norm"] = (
        working_df["clean_title"].fillna("").astype("string").str.lower().str.strip()
    )
    working_df["title_norm"] = (
        working_df["title"].fillna("").astype("string").str.lower().str.strip()
    )

    return working_df


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
    working_df = prepare_title_search_frame(movies_df)

    mask = (
        working_df["clean_title_norm"].str.contains(normalized_query, regex=False)
        | working_df["title_norm"].str.contains(normalized_query, regex=False)
    )

    matches = working_df.loc[mask].copy()

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
    """
    if not title or not title.strip():
        raise ValueError("Title cannot be empty.")

    normalized_query = normalize_title_for_search(title)
    working_df = prepare_title_search_frame(movies_df)

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
                exact_matches = exact_matches.sort_values(
                    by=sort_columns,
                    ascending=ascending,
                )

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


def build_recommendation_result(
    row: pd.Series,
    similarity_score: float,
) -> RecommendationResult:
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
        rating_count=int(row.get("rating_count", 0))
        if not pd.isna(row.get("rating_count", 0))
        else 0,
        rating_mean=to_nullable_float(row.get("rating_mean")),
        metadata_text=str(row.get("metadata_text", "")),
    )


def compute_similarity_scores(
    assets: RecommenderAssets,
    source_index: int,
) -> np.ndarray:
    """
    Compute one-to-all similarity scores for a single movie row.

    Why linear_kernel:
    TF-IDF vectors are L2-normalized by default, so linear_kernel gives the
    same ranking behavior as cosine similarity here without requiring a saved
    dense NxN similarity matrix.
    """
    source_vector = assets.tfidf_matrix[source_index]
    similarity_scores = linear_kernel(source_vector, assets.tfidf_matrix).ravel()
    return similarity_scores


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
    - compute one-to-all similarity from the TF-IDF matrix
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
    similarity_scores = compute_similarity_scores(assets=assets, source_index=source_index)
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
    source: RecommenderAssets | pd.DataFrame,
    query: str,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Return possible movie matches for user inspection.

    This accepts either:
    - RecommenderAssets
    - processed movies DataFrame

    Why:
    Candidate title lookup should work even when TF-IDF artifacts are not loaded.
    """
    if isinstance(source, RecommenderAssets):
        movies_df = source.movies_df
    elif isinstance(source, pd.DataFrame):
        movies_df = source
    else:
        raise TypeError(
            "source must be either RecommenderAssets or a pandas DataFrame."
        )

    matches = find_movies_by_partial_title(movies_df, query=query, limit=limit)

    if matches.empty:
        return matches

    selected_columns = [
        column
        for column in [
            "movieId",
            "title",
            "clean_title",
            "release_year",
            "genres",
            "rating_count",
            "rating_mean",
        ]
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
        tfidf_matrix=artifacts.tfidf_matrix,
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