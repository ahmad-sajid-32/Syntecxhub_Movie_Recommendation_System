"""
File: app/app.py

Purpose:
- Provide a minimal Flask API for the Movie Recommendation System.
- Expose health and recommendation endpoints.
- Load recommender artifacts lazily so the app stays simple and fast to start.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.recommender.recommend import (  # noqa: E402
    get_candidate_titles,
    get_similar_movies_by_title,
    load_recommender_assets,
    recommendation_results_to_dataframe,
)
from src.utils.helpers import configure_logging  # noqa: E402


logger = logging.getLogger(__name__)


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


@lru_cache(maxsize=1)
def get_assets():
    """
    Load recommender assets once and reuse them across requests.

    This avoids reloading the similarity matrix on every API call.
    """
    processed_movies_path = resolve_runtime_path(
        os.getenv("PROCESSED_MOVIES_PATH"),
        "data/processed/movies_metadata.csv",
    )
    artifact_dir = resolve_runtime_path(
        os.getenv("ARTIFACT_DIR"),
        "artifacts",
    )

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
                "endpoints": {
                    "health": "/health",
                    "recommend": "/recommend?title=Toy+Story&top_n=10&min_rating_count=10",
                    "candidates": "/candidates?query=toy",
                },
            }
        )

    @app.get("/health")
    def health():
        try:
            assets = get_assets()
            return jsonify(
                {
                    "status": "ok",
                    "movies_count": int(len(assets.movies_df)),
                    "similarity_matrix_shape": list(assets.similarity_matrix.shape),
                }
            )
        except Exception as error:  # pragma: no cover - runtime safety
            logger.exception("Health check failed.")
            return jsonify({"status": "error", "message": str(error)}), 500

    @app.get("/candidates")
    def candidates():
        query = request.args.get("query", "", type=str).strip()
        limit = request.args.get("limit", default=10, type=int)

        if not query:
            return jsonify({"error": "Missing required query parameter: query"}), 400

        try:
            assets = get_assets()
            matches = get_candidate_titles(assets=assets, query=query, limit=limit)
            return jsonify(
                {
                    "query": query,
                    "count": int(len(matches)),
                    "matches": matches.to_dict(orient="records"),
                }
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Exception as error:  # pragma: no cover - runtime safety
            logger.exception("Candidate lookup failed.")
            return jsonify({"error": str(error)}), 500

    @app.get("/recommend")
    def recommend():
        title = request.args.get("title", "", type=str).strip()
        top_n = request.args.get("top_n", default=10, type=int)
        min_rating_count = request.args.get("min_rating_count", default=10, type=int)

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
                exclude_input_movie=True,
                min_rating_count=min_rating_count,
            )
            results_df = recommendation_results_to_dataframe(results)

            return jsonify(
                {
                    "query": title,
                    "top_n": top_n,
                    "min_rating_count": min_rating_count,
                    "count": int(len(results_df)),
                    "recommendations": results_df.to_dict(orient="records"),
                }
            )

        except ValueError as error:
            logger.warning("Recommendation request rejected. title=%s reason=%s", title, error)
            return jsonify({"error": str(error)}), 400
        except Exception as error:  # pragma: no cover - runtime safety
            logger.exception("Recommendation request failed.")
            return jsonify({"error": str(error)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"

    app.run(host=host, port=port, debug=debug)
