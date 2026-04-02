"""
File: src/features/build_features.py

Purpose:
- Convert cleaned movie metadata text into TF-IDF features.
- Save the fitted vectorizer and sparse TF-IDF matrix for reuse.
- Build movie ID/index maps for runtime lookup.
- Optionally build a dense movie-to-movie similarity matrix for debugging only.
- Return a clean bundle that later recommendation code can use directly.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


DEFAULT_TEXT_COLUMN = "metadata_text"
DEFAULT_ARTIFACT_DIR = "artifacts"
SIMILARITY_ARTIFACT_FILENAME = "similarity_matrix.npy"


@dataclass(frozen=True)
class FeatureArtifacts:
    """
    Holds the main feature objects produced by this step.

    Important design change:
    - tfidf_matrix is now the primary runtime artifact
    - similarity_matrix is optional and disabled by default

    Why:
    The recommender only needs one-to-all scoring at query time.
    Saving a full dense NxN similarity matrix is wasteful for runtime.
    """

    movies_df: pd.DataFrame
    tfidf_matrix: csr_matrix
    vectorizer: TfidfVectorizer
    movie_id_to_index: dict[int, int]
    index_to_movie_id: dict[int, int]
    similarity_matrix: np.ndarray | None = None


def validate_processed_movies_dataframe(
    movies_df: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> None:
    """
    Validate that the processed DataFrame contains the columns needed
    for feature building.

    This fails early with a clear message instead of breaking later
    inside scikit-learn with a confusing stack trace.
    """
    required_columns = {"movieId", "title", text_column}
    missing_columns = required_columns - set(movies_df.columns)

    if missing_columns:
        error_message = (
            "Processed movies DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if movies_df.empty:
        error_message = "Processed movies DataFrame is empty. Cannot build features."
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info(
        "Processed movies DataFrame validation passed. rows=%d columns=%d",
        len(movies_df),
        len(movies_df.columns),
    )


def prepare_text_corpus(
    movies_df: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> pd.Series:
    """
    Prepare the text corpus that TF-IDF will consume.

    Rules:
    - fill missing values
    - convert everything to string
    - strip extra spaces
    - keep empty rows if they exist

    Why this matters:
    TF-IDF expects clean text input. Empty or null-like values create noise.
    """
    logger.info("Preparing text corpus from column: %s", text_column)

    corpus = (
        movies_df[text_column]
        .fillna("")
        .astype("string")
        .str.strip()
        .replace(r"\s+", " ", regex=True)
    )

    empty_rows = corpus.eq("").sum()
    if empty_rows > 0:
        logger.warning(
            "Found %d rows with empty text in '%s'. They will remain but produce sparse vectors.",
            int(empty_rows),
            text_column,
        )

    return corpus


def build_tfidf_vectorizer(
    max_features: int | None = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.8,
    stop_words: str | list[str] | None = "english",
) -> TfidfVectorizer:
    """
    Create the TF-IDF vectorizer.

    Chosen defaults:
    - unigrams + bigrams: captures single words and short phrases
    - english stop words: removes common useless words
    - sublinear_tf: softens repeated term explosion
    - max_features: keeps the model compact and predictable

    This is a practical baseline, not a magic setting.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )

    logger.info(
        "Created TF-IDF vectorizer with max_features=%s, ngram_range=%s, min_df=%d, max_df=%s",
        max_features,
        ngram_range,
        min_df,
        max_df,
    )
    return vectorizer


def fit_tfidf_features(
    corpus: pd.Series,
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Fit TF-IDF on the movie metadata text and return the sparse feature matrix.
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()

    logger.info("Fitting TF-IDF on corpus with %d documents.", len(corpus))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    logger.info(
        "TF-IDF fitting completed. matrix_shape=%s vocabulary_size=%d",
        tfidf_matrix.shape,
        len(vectorizer.vocabulary_),
    )

    return vectorizer, tfidf_matrix


def build_cosine_similarity_matrix(
    tfidf_matrix: csr_matrix,
    dtype: Any = np.float32,
) -> np.ndarray:
    """
    Build the full movie-to-movie cosine similarity matrix.

    Important:
    - This is now optional and disabled by default.
    - It exists only for debugging, inspection, or comparison work.
    - It should not be part of the normal runtime artifact path.

    Output:
    - square dense matrix of shape (n_movies, n_movies)

    Note:
    We cast to float32 to reduce size when this optional artifact is built.
    """
    logger.warning(
        "Building full dense similarity matrix. "
        "This is expensive and should not be enabled for normal runtime flow."
    )
    logger.info("Building cosine similarity matrix for shape=%s", tfidf_matrix.shape)

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(
        dtype,
        copy=False,
    )

    logger.info(
        "Cosine similarity matrix built successfully. shape=%s dtype=%s",
        similarity_matrix.shape,
        similarity_matrix.dtype,
    )

    return similarity_matrix


def build_movie_index_maps(
    movies_df: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Build two-way index maps between movieId and row index.

    Why this matters:
    Runtime scoring works with row positions, not movie IDs.
    This bridge is mandatory.
    """
    movie_ids = movies_df["movieId"].astype("int64").tolist()

    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    index_to_movie_id = {idx: movie_id for idx, movie_id in enumerate(movie_ids)}

    logger.info("Built movie index maps for %d movies.", len(movie_ids))
    return movie_id_to_index, index_to_movie_id


def build_feature_artifacts(
    movies_df: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
    vectorizer: TfidfVectorizer | None = None,
    build_similarity_matrix: bool = False,
) -> FeatureArtifacts:
    """
    Main entry point for feature generation.

    It:
    1. validates the input
    2. prepares the text
    3. fits TF-IDF
    4. builds ID/index maps
    5. optionally builds a dense similarity matrix
    6. returns one clean artifact bundle

    Important design choice:
    build_similarity_matrix defaults to False.
    """
    logger.info(
        "Starting feature artifact build process. build_similarity_matrix=%s",
        build_similarity_matrix,
    )

    validate_processed_movies_dataframe(movies_df=movies_df, text_column=text_column)
    corpus = prepare_text_corpus(movies_df=movies_df, text_column=text_column)
    fitted_vectorizer, tfidf_matrix = fit_tfidf_features(
        corpus=corpus,
        vectorizer=vectorizer,
    )
    movie_id_to_index, index_to_movie_id = build_movie_index_maps(movies_df=movies_df)

    similarity_matrix: np.ndarray | None = None
    if build_similarity_matrix:
        similarity_matrix = build_cosine_similarity_matrix(tfidf_matrix=tfidf_matrix)

    logger.info("Feature artifact build process completed successfully.")

    return FeatureArtifacts(
        movies_df=movies_df.copy(),
        tfidf_matrix=tfidf_matrix,
        vectorizer=fitted_vectorizer,
        movie_id_to_index=movie_id_to_index,
        index_to_movie_id=index_to_movie_id,
        similarity_matrix=similarity_matrix,
    )


def load_processed_movies(
    file_path: str | Path = "data/processed/movies_metadata.csv",
) -> pd.DataFrame:
    """
    Load the processed movie metadata CSV produced by preprocess.py.
    """
    input_file = Path(file_path).resolve()
    logger.info("Loading processed movies from: %s", input_file)

    if not input_file.exists():
        error_message = (
            f"Processed file not found: {input_file}\n"
            "Run the preprocessing pipeline first so movies_metadata.csv exists."
        )
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    movies_df = pd.read_csv(
        input_file,
        dtype={
            "movieId": "int64",
            "title": "string",
            "clean_title": "string",
            "release_year": "Int64",
            "genres": "string",
            "genres_list": "string",
            "genres_text": "string",
            "tags_text": "string",
            "tag_count": "int64",
            "imdbId": "Int64",
            "tmdbId": "Int64",
            "has_tmdb_mapping": "boolean",
            "rating_count": "int64",
            "rating_mean": "float64",
            "rating_median": "float64",
            "metadata_text": "string",
        },
    )

    logger.info("Loaded processed movies file with %d rows.", len(movies_df))
    return movies_df


def save_vectorizer(
    vectorizer: TfidfVectorizer,
    output_path: str | Path = f"{DEFAULT_ARTIFACT_DIR}/tfidf_vectorizer.pkl",
) -> Path:
    """
    Save the fitted vectorizer using pickle.

    Why pickle:
    TfidfVectorizer is a Python object, not a CSV.
    """
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("wb") as file:
        pickle.dump(vectorizer, file)

    logger.info("Saved TF-IDF vectorizer to: %s", output_file)
    return output_file


def save_tfidf_matrix(
    tfidf_matrix: csr_matrix,
    output_path: str | Path = f"{DEFAULT_ARTIFACT_DIR}/tfidf_matrix.npz",
) -> Path:
    """
    Save the sparse TF-IDF matrix in compressed .npz format.
    """
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    save_npz(output_file, tfidf_matrix)

    logger.info("Saved TF-IDF matrix to: %s", output_file)
    return output_file


def save_similarity_matrix(
    similarity_matrix: np.ndarray,
    output_path: str | Path = f"{DEFAULT_ARTIFACT_DIR}/{SIMILARITY_ARTIFACT_FILENAME}",
) -> Path:
    """
    Save the dense cosine similarity matrix as a NumPy file.

    Important:
    This should only be used when the optional similarity matrix
    was explicitly requested.
    """
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_file, similarity_matrix)

    logger.info("Saved similarity matrix to: %s", output_file)
    return output_file


def save_movie_index_maps(
    movie_id_to_index: dict[int, int],
    index_to_movie_id: dict[int, int],
    output_path: str | Path = f"{DEFAULT_ARTIFACT_DIR}/movie_index_maps.pkl",
) -> Path:
    """
    Save the ID/index maps for later recommendation lookups.
    """
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": index_to_movie_id,
    }

    with output_file.open("wb") as file:
        pickle.dump(payload, file)

    logger.info("Saved movie index maps to: %s", output_file)
    return output_file


def remove_stale_similarity_artifact(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> Path | None:
    """
    Remove an old similarity_matrix.npy artifact if it exists.

    Why this matters:
    After the runtime design changed, old dense artifacts can remain on disk
    and create confusion. If the optional similarity matrix was not built,
    that stale file should be removed.
    """
    artifact_base = Path(artifact_dir).resolve()
    similarity_file = artifact_base / SIMILARITY_ARTIFACT_FILENAME

    if not similarity_file.exists():
        return None

    similarity_file.unlink()
    logger.info("Removed stale similarity artifact: %s", similarity_file)
    return similarity_file


def save_feature_artifacts(
    artifacts: FeatureArtifacts,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Path]:
    """
    Save all feature artifacts to disk.

    Default output files:
    - tfidf_vectorizer.pkl
    - tfidf_matrix.npz
    - movie_index_maps.pkl

    Optional output file:
    - similarity_matrix.npy

    Important design change:
    similarity_matrix.npy is not saved unless it was explicitly built.
    """
    artifact_base = Path(artifact_dir).resolve()
    artifact_base.mkdir(parents=True, exist_ok=True)

    logger.info("Saving feature artifacts to directory: %s", artifact_base)

    saved_paths = {
        "vectorizer": save_vectorizer(
            vectorizer=artifacts.vectorizer,
            output_path=artifact_base / "tfidf_vectorizer.pkl",
        ),
        "tfidf_matrix": save_tfidf_matrix(
            tfidf_matrix=artifacts.tfidf_matrix,
            output_path=artifact_base / "tfidf_matrix.npz",
        ),
        "movie_index_maps": save_movie_index_maps(
            movie_id_to_index=artifacts.movie_id_to_index,
            index_to_movie_id=artifacts.index_to_movie_id,
            output_path=artifact_base / "movie_index_maps.pkl",
        ),
    }

    if artifacts.similarity_matrix is not None:
        saved_paths["similarity_matrix"] = save_similarity_matrix(
            similarity_matrix=artifacts.similarity_matrix,
            output_path=artifact_base / SIMILARITY_ARTIFACT_FILENAME,
        )
    else:
        remove_stale_similarity_artifact(artifact_dir=artifact_base)

    logger.info("Feature artifacts saved successfully.")
    return saved_paths


def run_feature_pipeline(
    processed_movies_path: str | Path = "data/processed/movies_metadata.csv",
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    build_similarity_matrix: bool = False,
) -> FeatureArtifacts:
    """
    Convenience function for manual runs and scripts.

    It:
    1. loads processed movie metadata
    2. builds TF-IDF features
    3. optionally builds a dense similarity matrix
    4. saves artifacts
    5. returns the artifact bundle

    Important design choice:
    build_similarity_matrix defaults to False.
    """
    logger.info(
        "Running feature pipeline with processed_movies_path=%s artifact_dir=%s build_similarity_matrix=%s",
        processed_movies_path,
        artifact_dir,
        build_similarity_matrix,
    )

    movies_df = load_processed_movies(file_path=processed_movies_path)
    artifacts = build_feature_artifacts(
        movies_df=movies_df,
        text_column=text_column,
        build_similarity_matrix=build_similarity_matrix,
    )
    save_feature_artifacts(artifacts=artifacts, artifact_dir=artifact_dir)

    logger.info("Feature pipeline completed successfully.")
    return artifacts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    built_artifacts = run_feature_pipeline()

    print("Feature pipeline completed successfully.")
    print(f"Movies: {len(built_artifacts.movies_df)}")
    print(f"TF-IDF matrix shape: {built_artifacts.tfidf_matrix.shape}")
    if built_artifacts.similarity_matrix is None:
        print("Similarity matrix: not built (disabled by default)")
    else:
        print(f"Similarity matrix shape: {built_artifacts.similarity_matrix.shape}")
    print(f"Vocabulary size: {len(built_artifacts.vectorizer.vocabulary_)}")