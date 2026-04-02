"""
File: main.py

Purpose:
- Provide one clean command-line entry point for the whole project.
- Run the full training pipeline.
- Query movie candidates for ambiguous titles.
- Generate recommendations from the saved artifacts.
- Optionally start the Flask API server.

Why this file exists:
Right now the project has working modules, but no single controlled entry point.
This file becomes the one place from which the project can actually be operated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.app import create_app  # noqa: E402
from src.models.train_model import (  # noqa: E402
    TrainingConfig,
    print_training_summary,
    run_training_pipeline,
)
from src.recommender.recommend import (  # noqa: E402
    get_candidate_titles,
    get_similar_movies_by_title,
    load_recommender_assets,
    recommendation_results_to_dataframe,
)
from src.utils.helpers import configure_logging, timed_block  # noqa: E402


logger = logging.getLogger(__name__)


VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


def resolve_project_path(value: str | Path) -> Path:
    """
    Resolve a path against the project root.

    Rules:
    - absolute paths stay absolute
    - relative paths are resolved from the project root
    """
    candidate = Path(value)

    if candidate.is_absolute():
        return candidate.resolve()

    return (PROJECT_ROOT / candidate).resolve()


def parse_log_level(value: str) -> int:
    """
    Convert a log level string into a logging module constant.
    """
    normalized = value.strip().upper()

    if normalized not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level '{value}'. Allowed values: {sorted(VALID_LOG_LEVELS)}"
        )

    return getattr(logging, normalized)


def parse_sample_titles(value: str) -> tuple[str, ...]:
    """
    Parse a comma-separated list of sample titles.

    Example:
    "Toy Story,Jumanji,Heat"
    ->
    ("Toy Story", "Jumanji", "Heat")
    """
    titles = tuple(part.strip() for part in value.split(",") if part.strip())

    if not titles:
        raise ValueError("At least one sample title must be provided.")

    return titles


def print_dataframe(df: pd.DataFrame) -> None:
    """
    Print a DataFrame safely for terminal output.
    """
    if df.empty:
        print("No rows found.")
        return

    print(df.to_string(index=False))


def run_train_command(args: argparse.Namespace) -> int:
    """
    Run the full project pipeline:
    1. load raw data
    2. preprocess metadata
    3. build TF-IDF + similarity artifacts
    4. save outputs
    5. print a training summary
    """
    config = TrainingConfig(
        data_dir=resolve_project_path(args.data_dir),
        processed_output_path=resolve_project_path(args.processed_output_path),
        artifact_dir=resolve_project_path(args.artifact_dir),
        summary_output_path=resolve_project_path(args.summary_output_path),
        text_column=args.text_column,
        sample_titles=parse_sample_titles(args.sample_titles),
        recommendation_top_n=args.recommendation_top_n,
        min_rating_count=args.min_rating_count,
    )

    with timed_block("full_training_pipeline", logger):
        output = run_training_pipeline(config=config)

    print_training_summary(output.summary)
    return 0


def run_candidates_command(args: argparse.Namespace) -> int:
    """
    Show likely movie matches for a partial title query.

    This is useful when the seed title is ambiguous.
    """
    processed_movies_path = resolve_project_path(args.processed_movies_path)
    artifact_dir = resolve_project_path(args.artifact_dir)

    with timed_block("candidate_lookup", logger):
        assets = load_recommender_assets(
            processed_movies_path=processed_movies_path,
            artifact_dir=artifact_dir,
        )
        matches = get_candidate_titles(
            assets=assets,
            query=args.query,
            limit=args.limit,
        )

    print_dataframe(matches)
    return 0


def run_recommend_command(args: argparse.Namespace) -> int:
    """
    Generate top-N recommendations for a movie title.
    """
    processed_movies_path = resolve_project_path(args.processed_movies_path)
    artifact_dir = resolve_project_path(args.artifact_dir)

    with timed_block("recommendation_lookup", logger):
        assets = load_recommender_assets(
            processed_movies_path=processed_movies_path,
            artifact_dir=artifact_dir,
        )
        results = get_similar_movies_by_title(
            assets=assets,
            title=args.title,
            top_n=args.top_n,
            exclude_input_movie=not args.include_input_movie,
            min_rating_count=args.min_rating_count,
        )
        results_df = recommendation_results_to_dataframe(results)

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
        if column in results_df.columns
    ]

    print_dataframe(results_df[selected_columns])
    return 0


def run_serve_command(args: argparse.Namespace) -> int:
    """
    Start the Flask API server.

    Important:
    This command serves the API directly from the project.
    It does not use `flask run`.
    """
    app = create_app()
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the project.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Syntecxhub Movie Recommendation System CLI",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level. Allowed values: CRITICAL, ERROR, WARNING, INFO, DEBUG",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Run the full preprocessing + feature build + artifact generation pipeline.",
    )
    train_parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Path to the raw MovieLens dataset directory.",
    )
    train_parser.add_argument(
        "--processed-output-path",
        default="data/processed/movies_metadata.csv",
        help="Output path for processed movie metadata.",
    )
    train_parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory where vectorizer, matrices, and maps will be saved.",
    )
    train_parser.add_argument(
        "--summary-output-path",
        default="artifacts/training_summary.json",
        help="Output path for the training summary JSON file.",
    )
    train_parser.add_argument(
        "--text-column",
        default="metadata_text",
        help="Processed text column used to build TF-IDF features.",
    )
    train_parser.add_argument(
        "--sample-titles",
        default="Toy Story,Jumanji,Heat",
        help="Comma-separated sample titles used for sanity-check recommendations.",
    )
    train_parser.add_argument(
        "--recommendation-top-n",
        type=int,
        default=5,
        help="Number of sample recommendations to generate in the training summary.",
    )
    train_parser.add_argument(
        "--min-rating-count",
        type=int,
        default=10,
        help="Minimum rating count used when generating sample recommendations.",
    )
    train_parser.set_defaults(func=run_train_command)

    candidates_parser = subparsers.add_parser(
        "candidates",
        help="Find candidate movie titles for an ambiguous query.",
    )
    candidates_parser.add_argument(
        "--query",
        required=True,
        help="Partial movie title to search for.",
    )
    candidates_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of candidate matches to return.",
    )
    candidates_parser.add_argument(
        "--processed-movies-path",
        default="data/processed/movies_metadata.csv",
        help="Path to the processed movie metadata CSV.",
    )
    candidates_parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory containing saved recommender artifacts.",
    )
    candidates_parser.set_defaults(func=run_candidates_command)

    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Generate recommendations for a movie title.",
    )
    recommend_parser.add_argument(
        "--title",
        required=True,
        help="Movie title used as the recommendation seed.",
    )
    recommend_parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations to return.",
    )
    recommend_parser.add_argument(
        "--min-rating-count",
        type=int,
        default=10,
        help="Minimum rating count filter for recommended movies.",
    )
    recommend_parser.add_argument(
        "--include-input-movie",
        action="store_true",
        help="Include the seed movie itself in the returned results.",
    )
    recommend_parser.add_argument(
        "--processed-movies-path",
        default="data/processed/movies_metadata.csv",
        help="Path to the processed movie metadata CSV.",
    )
    recommend_parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory containing saved recommender artifacts.",
    )
    recommend_parser.set_defaults(func=run_recommend_command)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the Flask API server.",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Flask app.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the Flask app.",
    )
    serve_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode.",
    )
    serve_parser.set_defaults(func=run_serve_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Main CLI entry point.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        configure_logging(parse_log_level(args.log_level))
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    try:
        return int(args.func(args))
    except ValueError as error:
        logger.error("Validation error: %s", error)
        print(f"Validation error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        logger.error("File error: %s", error)
        print(f"File error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        print("Execution interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        logger.exception("Unhandled application error.")
        print(f"Unhandled error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())