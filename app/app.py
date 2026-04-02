"""
File: app/app.py

Purpose:
- Provide a Flask API for the Movie Recommendation System.
- Expose lightweight health and readiness endpoints.
- Keep candidate title search cheap by loading only processed movie metadata.
- Load recommender artifacts lazily only for recommendation requests.

Important design changes:
- /health is now lightweight and does not force model loading.
- /ready performs a heavier readiness check by loading recommender assets.
- /candidates uses only processed metadata, not TF-IDF artifacts.
- /recommend uses sparse TF-IDF runtime assets loaded lazily and cached.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.recommender.recommend import (  # noqa: E402
    get_candidate_titles,
    get_similar_movies_by_title,
    load_movies_for_title_search,
    load_recommender_assets,
    recommendation_results_to_dataframe,
)
from src.utils.helpers import configure_logging  # noqa: E402


logger = logging.getLogger(__name__)


load_dotenv(PROJECT_ROOT / ".env")


def resolve_runtime_path(value: str | None, default_relative_path: str) -> Path:
    """
    Resolve a runtime path from environment or fallback default.

    Rules:
    - if env value is missing, use the default relative path under the project root
    - if env value is relative, resolve it from the project root
    - if env value is absolute, use it directly
    """
    if not value:
        return (PROJECT_ROOT / default_relative_path).resolve()

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()

    return (PROJECT_ROOT / candidate).resolve()


def parse_bool_env(value: str | None, default: bool = False) -> bool:
    """
    Parse a boolean-like environment value safely.

    Accepted truthy values:
    - true
    - 1
    - yes
    - on
    """
    if value is None:
        return default

    return value.strip().lower() in {"true", "1", "yes", "on"}


def get_processed_movies_path() -> Path:
    """
    Resolve the processed movie metadata path once per call site.
    """
    return resolve_runtime_path(
        os.getenv("PROCESSED_MOVIES_PATH"),
        "data/processed/movies_metadata.csv",
    )


def get_artifact_dir() -> Path:
    """
    Resolve the artifact directory once per call site.
    """
    return resolve_runtime_path(
        os.getenv("ARTIFACT_DIR"),
        "artifacts",
    )


@lru_cache(maxsize=1)
def get_movies_df():
    """
    Load processed movie metadata once and reuse it across requests.

    Why this exists:
    Candidate title lookup only needs the processed movie table.
    It should not pay the cost of loading TF-IDF artifacts.
    """
    processed_movies_path = get_processed_movies_path()

    logger.info(
        "Loading processed movie metadata for lightweight title search. processed_movies_path=%s",
        processed_movies_path,
    )

    return load_movies_for_title_search(processed_movies_path=processed_movies_path)


@lru_cache(maxsize=1)
def get_assets():
    """
    Load recommender runtime assets once and reuse them across requests.

    Important:
    This is heavier than get_movies_df() because it loads the sparse TF-IDF matrix.
    It is used only by recommendation and readiness paths.
    """
    processed_movies_path = get_processed_movies_path()
    artifact_dir = get_artifact_dir()

    logger.info(
        "Loading recommender assets with processed_movies_path=%s artifact_dir=%s",
        processed_movies_path,
        artifact_dir,
    )

    return load_recommender_assets(
        processed_movies_path=processed_movies_path,
        artifact_dir=artifact_dir,
    )


def create_app() -> Flask:
    """
    Create and configure the Flask app.
    """
    configure_logging()
    app = Flask(__name__)

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "Syntecxhub Movie Recommendation System",
                "status": "ok",
                "architecture": {
                    "recommender_type": "content_based",
                    "runtime_scoring": "on_demand_tfidf",
                },
                "endpoints": {
                    "health": "/health",
                    "ready": "/ready",
                    "recommend": "/recommend?title=Toy+Story&top_n=10&min_rating_count=10",
                    "candidates": "/candidates?query=toy",
                },
            }
        )

    @app.get("/health")
    def health():
        """
        Lightweight liveness check.

        Important:
        This endpoint must not force recommender asset loading.
        """
        try:
            processed_movies_path = get_processed_movies_path()
            artifact_dir = get_artifact_dir()

            return jsonify(
                {
                    "status": "ok",
                    "service": "Syntecxhub Movie Recommendation System",
                    "liveness": "alive",
                    "processed_movies_path": str(processed_movies_path),
                    "artifact_dir": str(artifact_dir),
                    "movies_cache_loaded": get_movies_df.cache_info().currsize > 0,
                    "assets_cache_loaded": get_assets.cache_info().currsize > 0,
                }
            )
        except Exception as error:  # pragma: no cover - runtime safety
            logger.exception("Health check failed.")
            return jsonify({"status": "error", "message": str(error)}), 500

    @app.get("/ready")
    def ready():
        """
        Heavier readiness check.

        Important:
        Unlike /health, this endpoint is allowed to load recommender assets
        and confirm they are usable.
        """
        try:
            assets = get_assets()
            return jsonify(
                {
                    "status": "ok",
                    "readiness": "ready",
                    "movies_count": int(len(assets.movies_df)),
                    "tfidf_matrix_shape": list(assets.tfidf_matrix.shape),
                }
            )
        except Exception as error:  # pragma: no cover - runtime safety
            logger.exception("Readiness check failed.")
            return jsonify({"status": "error", "message": str(error)}), 500

    @app.get("/candidates")
    def candidates():
        """
        Candidate title search.

        Important:
        This endpoint intentionally uses only processed movie metadata.
        It does not load TF-IDF artifacts.
        """
        query = request.args.get("query", "", type=str).strip()
        limit = request.args.get("limit", default=10, type=int)

        if not query:
            return jsonify({"error": "Missing required query parameter: query"}), 400

        if limit <= 0:
            return jsonify({"error": "limit must be greater than 0"}), 400

        try:
            movies_df = get_movies_df()
            matches = get_candidate_titles(source=movies_df, query=query, limit=limit)

            return jsonify(
                {
                    "query": query,
                    "limit": limit,
                    "count": int(len(matches)),
                    "matches": matches.to_dict(orient="records"),
                }
            )
        except ValueError as error:
            logger.warning("Candidate lookup rejected. query=%s reason=%s", query, error)
            return jsonify({"error": str(error)}), 400
        except Exception as error:  # pragma: no cover - runtime safety
            logger.exception("Candidate lookup failed.")
            return jsonify({"error": str(error)}), 500

    @app.get("/recommend")
    def recommend():
        """
        Recommendation endpoint.

        This endpoint uses lazily loaded recommender assets and computes
        one-to-all similarity scores on demand from the sparse TF-IDF matrix.
        """
        title = request.args.get("title", "", type=str).strip()
        top_n = request.args.get("top_n", default=10, type=int)
        min_rating_count = request.args.get("min_rating_count", default=10, type=int)
        include_input_movie = request.args.get(
            "include_input_movie",
            default=False,
            type=parse_bool_env,
        )

        if not title:
            return jsonify({"error": "Missing required query parameter: title"}), 400

        if top_n <= 0:
            return jsonify({"error": "top_n must be greater than 0"}), 400

        if min_rating_count < 0:
            return jsonify({"error": "min_rating_count cannot be negative"}), 400

        try:
            assets = get_assets()
            results = get_similar_movies_by_title(
                assets=assets,
                title=title,
                top_n=top_n,
                exclude_input_movie=not include_input_movie,
                min_rating_count=min_rating_count,
            )
            results_df = recommendation_results_to_dataframe(results)

            return jsonify(
                {
                    "query": title,
                    "top_n": top_n,
                    "min_rating_count": min_rating_count,
                    "include_input_movie": include_input_movie,
                    "count": int(len(results_df)),
                    "recommendations": results_df.to_dict(orient="records"),
                }
            )

        except ValueError as error:
            logger.warning(
                "Recommendation request rejected. title=%s reason=%s",
                title,
                error,
            )
            return jsonify({"error": str(error)}), 400
        except Exception as error:  # pragma: no cover - runtime safety
            logger.exception("Recommendation request failed.")
            return jsonify({"error": str(error)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = parse_bool_env(os.getenv("FLASK_DEBUG"), default=False)

    app.run(host=host, port=port, debug=debug)