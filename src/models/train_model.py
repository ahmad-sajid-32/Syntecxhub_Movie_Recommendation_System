"""
File: src/models/train_model.py

Purpose:
- Run the full training/build pipeline for the Movie Recommendation System.
- Execute data loading, preprocessing, feature generation, and artifact saving.
- Produce a training summary that can be used for verification and reporting.
- Run a few sample recommendation checks to confirm the pipeline works end to end.

Important design change:
- The dense similarity matrix is now optional.
- The default runtime strategy is sparse TF-IDF + on-demand scoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_data import DatasetBundle, build_dataset_summary, load_movielens_dataset
from src.data.preprocess import preprocess_movielens_metadata, save_processed_movies
from src.features.build_features import (
    FeatureArtifacts,
    build_feature_artifacts,
    save_feature_artifacts,
)
from src.recommender.recommend import (
    build_runtime_assets_from_feature_artifacts,
    get_candidate_titles,
    get_similar_movies_by_title,
    recommendation_results_to_dataframe,
)


logger = logging.getLogger(__name__)


DEFAULT_DATA_DIR = "data/raw"
DEFAULT_PROCESSED_OUTPUT_PATH = "data/processed/movies_metadata.csv"
DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_SUMMARY_OUTPUT_PATH = "artifacts/training_summary.json"


@dataclass(frozen=True)
class TrainingConfig:
    """
    Holds the pipeline configuration.

    Why this exists:
    The training pipeline has multiple paths and settings.
    Keeping them in one object makes the pipeline easier to reason about.
    """

    data_dir: str | Path = DEFAULT_DATA_DIR
    processed_output_path: str | Path = DEFAULT_PROCESSED_OUTPUT_PATH
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT_PATH
    text_column: str = "metadata_text"
    sample_titles: tuple[str, ...] = (
        "Toy Story",
        "Jumanji",
        "Heat",
    )
    recommendation_top_n: int = 5
    min_rating_count: int = 10
    build_similarity_matrix: bool = False


@dataclass(frozen=True)
class TrainingSummary:
    """
    Structured record of what the pipeline produced.

    This summary is useful for:
    - debugging
    - proving the pipeline worked
    - internship submission evidence
    """

    raw_dataset_summary: dict[str, int]
    processed_movies_count: int
    processed_columns: list[str]
    tfidf_matrix_shape: tuple[int, int]
    similarity_matrix_built: bool
    similarity_matrix_shape: tuple[int, int] | None
    vocabulary_size: int
    saved_artifacts: dict[str, str]
    sample_recommendations: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class TrainingOutput:
    """
    Final output of the training pipeline.
    """

    dataset: DatasetBundle
    processed_movies_df: pd.DataFrame
    feature_artifacts: FeatureArtifacts
    saved_artifact_paths: dict[str, Path]
    summary: TrainingSummary


def ensure_parent_directory(file_path: str | Path) -> Path:
    """
    Make sure the output file's parent directory exists.
    """
    resolved_path = Path(file_path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def run_data_loading_step(config: TrainingConfig) -> DatasetBundle:
    """
    Load the raw MovieLens dataset.
    """
    logger.info("Running data loading step with data_dir=%s", config.data_dir)

    dataset = load_movielens_dataset(data_dir=config.data_dir)

    dataset_summary = build_dataset_summary(dataset)
    logger.info("Raw dataset summary: %s", dataset_summary)

    return dataset


def run_preprocessing_step(
    dataset: DatasetBundle,
    config: TrainingConfig,
) -> pd.DataFrame:
    """
    Preprocess raw dataset into one clean movie-level metadata table.
    """
    logger.info("Running preprocessing step.")

    processed_movies_df = preprocess_movielens_metadata(dataset=dataset)
    save_processed_movies(
        processed_df=processed_movies_df,
        output_path=config.processed_output_path,
    )

    logger.info(
        "Preprocessing step completed. processed_movies_count=%d",
        len(processed_movies_df),
    )

    return processed_movies_df


def run_feature_build_step(
    processed_movies_df: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[FeatureArtifacts, dict[str, Path]]:
    """
    Build TF-IDF artifacts from the processed movies table.

    Important design change:
    - dense similarity matrix is optional
    - default runtime path uses sparse TF-IDF only
    """
    logger.info(
        "Running feature build step with build_similarity_matrix=%s.",
        config.build_similarity_matrix,
    )

    artifacts = build_feature_artifacts(
        movies_df=processed_movies_df,
        text_column=config.text_column,
        build_similarity_matrix=config.build_similarity_matrix,
    )
    saved_artifact_paths = save_feature_artifacts(
        artifacts=artifacts,
        artifact_dir=config.artifact_dir,
    )

    if artifacts.similarity_matrix is None:
        logger.info(
            "Feature build step completed. tfidf_shape=%s similarity_matrix=not_built",
            artifacts.tfidf_matrix.shape,
        )
    else:
        logger.info(
            "Feature build step completed. tfidf_shape=%s similarity_shape=%s",
            artifacts.tfidf_matrix.shape,
            artifacts.similarity_matrix.shape,
        )

    return artifacts, saved_artifact_paths


def build_sample_recommendations(
    feature_artifacts: FeatureArtifacts,
    sample_titles: tuple[str, ...],
    recommendation_top_n: int,
    min_rating_count: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    Generate a few sample recommendation outputs.

    Why this matters:
    A training pipeline that only builds files is incomplete.
    We also need proof that the recommender returns sensible results.
    """
    logger.info(
        "Building sample recommendations for titles=%s",
        list(sample_titles),
    )

    runtime_assets = build_runtime_assets_from_feature_artifacts(feature_artifacts)
    sample_outputs: dict[str, list[dict[str, Any]]] = {}

    for title in sample_titles:
        try:
            recommendations = get_similar_movies_by_title(
                assets=runtime_assets,
                title=title,
                top_n=recommendation_top_n,
                exclude_input_movie=True,
                min_rating_count=min_rating_count,
            )
            recommendations_df = recommendation_results_to_dataframe(recommendations)

            selected_columns = [
                column
                for column in [
                    "movie_id",
                    "title",
                    "clean_title",
                    "release_year",
                    "similarity_score",
                    "genres",
                    "rating_count",
                    "rating_mean",
                ]
                if column in recommendations_df.columns
            ]

            sample_outputs[title] = recommendations_df[selected_columns].to_dict(
                orient="records"
            )

            logger.info(
                "Built %d sample recommendations for title='%s'.",
                len(sample_outputs[title]),
                title,
            )

        except ValueError as error:
            logger.warning(
                "Failed to generate sample recommendations for title='%s'. reason=%s",
                title,
                error,
            )

            candidate_titles = get_candidate_titles(runtime_assets, query=title, limit=5)
            sample_outputs[title] = [
                {
                    "error": str(error),
                    "candidate_titles": candidate_titles.to_dict(orient="records"),
                }
            ]

    return sample_outputs


def build_training_summary(
    dataset: DatasetBundle,
    processed_movies_df: pd.DataFrame,
    feature_artifacts: FeatureArtifacts,
    saved_artifact_paths: dict[str, Path],
    config: TrainingConfig,
) -> TrainingSummary:
    """
    Create a structured summary of the entire pipeline run.
    """
    logger.info("Building training summary.")

    raw_dataset_summary = build_dataset_summary(dataset)

    sample_recommendations = build_sample_recommendations(
        feature_artifacts=feature_artifacts,
        sample_titles=config.sample_titles,
        recommendation_top_n=config.recommendation_top_n,
        min_rating_count=config.min_rating_count,
    )

    similarity_matrix_built = feature_artifacts.similarity_matrix is not None
    similarity_matrix_shape: tuple[int, int] | None = None

    if similarity_matrix_built:
        similarity_matrix_shape = tuple(feature_artifacts.similarity_matrix.shape)

    summary = TrainingSummary(
        raw_dataset_summary=raw_dataset_summary,
        processed_movies_count=int(len(processed_movies_df)),
        processed_columns=processed_movies_df.columns.tolist(),
        tfidf_matrix_shape=tuple(feature_artifacts.tfidf_matrix.shape),
        similarity_matrix_built=similarity_matrix_built,
        similarity_matrix_shape=similarity_matrix_shape,
        vocabulary_size=int(len(feature_artifacts.vectorizer.vocabulary_)),
        saved_artifacts={name: str(path) for name, path in saved_artifact_paths.items()},
        sample_recommendations=sample_recommendations,
    )

    logger.info("Training summary built successfully.")
    return summary


def save_training_summary(
    summary: TrainingSummary,
    output_path: str | Path,
) -> Path:
    """
    Save the training summary as JSON.

    Why JSON:
    It is simple, readable, and good for reporting.
    """
    output_file = ensure_parent_directory(output_path)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2, ensure_ascii=False)

    logger.info("Saved training summary to: %s", output_file)
    return output_file


def run_training_pipeline(
    config: TrainingConfig | None = None,
) -> TrainingOutput:
    """
    Main pipeline entry point.

    Steps:
    1. load raw dataset
    2. preprocess metadata
    3. build TF-IDF artifacts
    4. build summary and sample recommendation checks
    5. save summary

    Important:
    Dense similarity matrix generation is optional and disabled by default.
    """
    if config is None:
        config = TrainingConfig()

    logger.info("Starting full training pipeline with config=%s", config)

    dataset = run_data_loading_step(config=config)
    processed_movies_df = run_preprocessing_step(dataset=dataset, config=config)
    feature_artifacts, saved_artifact_paths = run_feature_build_step(
        processed_movies_df=processed_movies_df,
        config=config,
    )
    summary = build_training_summary(
        dataset=dataset,
        processed_movies_df=processed_movies_df,
        feature_artifacts=feature_artifacts,
        saved_artifact_paths=saved_artifact_paths,
        config=config,
    )
    save_training_summary(summary=summary, output_path=config.summary_output_path)

    logger.info("Training pipeline completed successfully.")

    return TrainingOutput(
        dataset=dataset,
        processed_movies_df=processed_movies_df,
        feature_artifacts=feature_artifacts,
        saved_artifact_paths=saved_artifact_paths,
        summary=summary,
    )


def print_training_summary(summary: TrainingSummary) -> None:
    """
    Print a human-readable version of the training summary.
    """
    print("Training pipeline completed successfully.")
    print("-" * 80)
    print("Raw dataset summary:")
    for key, value in summary.raw_dataset_summary.items():
        print(f"{key}: {value}")

    print("-" * 80)
    print(f"Processed movies count: {summary.processed_movies_count}")
    print(f"TF-IDF matrix shape: {summary.tfidf_matrix_shape}")
    print(f"Similarity matrix built: {summary.similarity_matrix_built}")
    if summary.similarity_matrix_shape is None:
        print("Similarity matrix shape: not built")
    else:
        print(f"Similarity matrix shape: {summary.similarity_matrix_shape}")
    print(f"Vocabulary size: {summary.vocabulary_size}")

    print("-" * 80)
    print("Saved artifacts:")
    for name, path in summary.saved_artifacts.items():
        print(f"{name}: {path}")

    print("-" * 80)
    print("Sample recommendations:")
    for seed_title, recommendations in summary.sample_recommendations.items():
        print(f"\nSeed title: {seed_title}")
        if not recommendations:
            print("  No sample recommendations available.")
            continue

        for item in recommendations:
            if "error" in item:
                print(f"  Error: {item['error']}")
                if "candidate_titles" in item:
                    print("  Candidate titles:")
                    for candidate in item["candidate_titles"]:
                        print(f"    - {candidate}")
                continue

            print(
                "  - "
                f"{item.get('title', '')} | "
                f"year={item.get('release_year')} | "
                f"score={item.get('similarity_score')} | "
                f"ratings={item.get('rating_count')} | "
                f"mean={item.get('rating_mean')}"
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    training_output = run_training_pipeline()
    print_training_summary(training_output.summary)