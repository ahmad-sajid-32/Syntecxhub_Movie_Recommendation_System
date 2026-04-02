# Syntecxhub Movie Recommendation System

A content-based Movie Recommendation System built for the Syntecxhub internship task.

This project uses the **MovieLens Latest Small** dataset to:

- load and validate movie data
- clean and preprocess metadata
- build TF-IDF text features
- generate movie recommendations by title
- expose a simple Flask API for testing

The current implementation is a **content-based recommender**, not collaborative filtering.

---

## 1. Project Goal

The goal of this project is to recommend movies that are similar to a given movie based on metadata.

Instead of asking:

> What did similar users watch?

this project asks:

> What movies look similar based on their content?

That content is built mainly from:

- movie title
- genres
- user tags

Those text fields are cleaned and merged into one metadata field, then transformed into numbers using **TF-IDF**.

The recommender then compares the seed movie against all other movies and returns the most similar ones.

---

## 2. Current Architecture

This project now follows an optimized runtime design.

### Old idea

The earlier version built and stored a **full dense movie-to-movie similarity matrix**.

### Current idea

The current version stores the **sparse TF-IDF matrix** and computes **one-to-all similarity on demand** when a recommendation request is made.

This is better because:

- it avoids storing a huge dense similarity artifact by default
- it reduces unnecessary disk usage
- it reduces startup and load overhead
- it keeps the recommendation method the same from the user's point of view

### Current flow

1. Load raw MovieLens CSV files
2. Validate file presence and schema assumptions
3. Clean movie metadata
4. Aggregate user tags per movie
5. Build rating summary statistics
6. Create one final `metadata_text` field per movie
7. Convert text into TF-IDF vectors
8. Save sparse TF-IDF artifacts
9. Resolve a movie by title
10. Compute one-to-all similarity scores on demand
11. Return top-N similar movies

This is appropriate for the internship scope because it is:

- explainable
- easy to verify
- modular
- efficient enough for MovieLens small
- simple to demo

---

## 3. Tech Stack

- **Python 3.x**
- **pandas** for data handling
- **numpy** for numerical operations
- **scikit-learn** for TF-IDF and similarity scoring
- **scipy** for sparse matrix persistence
- **matplotlib** for EDA plots
- **Flask** for the simple API
- **Jupyter Notebook** for exploratory analysis
- **python-dotenv** for environment variable loading

---

## 4. Project Structure

```text
Syntecxhub_Movie_Recommendation_System/
│   .env.example
│   .gitignore
│   main.py
│   README.md
│   requirements.txt
│
├───app
│       app.py
│
├───artifacts
│
├───data
│   ├───processed
│   └───raw
│       ├───links.csv
│       ├───movies.csv
│       ├───ratings.csv
│       └───tags.csv
│
├───notebooks
│       eda.ipynb
│
├───src
│   │   __init__.py
│   │
│   ├───data
│   │       __init__.py
│   │       load_data.py
│   │       preprocess.py
│   │
│   ├───features
│   │       __init__.py
│   │       build_features.py
│   │
│   ├───models
│   │       __init__.py
│   │       train_model.py
│   │
│   ├───recommender
│   │       __init__.py
│   │       recommend.py
│   │
│   └───utils
│           __init__.py
│           helpers.py
│
└───tests
        __init__.py
```

---

## 5. Dataset

This project uses the **MovieLens Latest Small** dataset.

Required raw files:

- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`

These files must be placed inside:

```text
data/raw/
```

Expected raw data layout:

```text
data/raw/
├── movies.csv
├── ratings.csv
├── tags.csv
└── links.csv
```

If these files are missing, the loader fails early with a clear error.

---

## 6. Installation and Setup

### Step 1: Create and activate virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Step 2: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Confirm dataset exists

```powershell
dir data\raw
```

You should see:

- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`

---

## 7. Environment Configuration

The Flask app reads environment variables from a local `.env` file.

Create `.env` using `.env.example` as a reference.

Example values:

```env
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_DEBUG=false
PROCESSED_MOVIES_PATH=data/processed/movies_metadata.csv
ARTIFACT_DIR=artifacts
```

### What these values mean

- `FLASK_HOST`
  Host interface for the Flask API

- `FLASK_PORT`
  Port used by the Flask API

- `FLASK_DEBUG`
  Enables or disables Flask debug mode

- `PROCESSED_MOVIES_PATH`
  Path to the processed movie metadata CSV

- `ARTIFACT_DIR`
  Directory containing runtime artifacts such as the TF-IDF matrix and index maps

---

## 8. How the Pipeline Works

### 8.1 Data Loading

File: `src/data/load_data.py`

This module:

- resolves the raw dataset paths
- checks that all required files exist
- loads the CSV files with controlled dtypes
- converts timestamps into readable datetime fields
- returns the dataset in a structured bundle

### 8.2 Preprocessing

File: `src/data/preprocess.py`

This module:

- extracts release year from titles like `Toy Story (1995)`
- normalizes genres into clean text tokens
- cleans user tags
- aggregates tags per movie
- creates rating statistics like count and mean
- builds the final `metadata_text` column

Example idea:

A movie like:

- title: `Toy Story`
- genres: `Adventure|Animation|Children|Comedy|Fantasy`
- tags: `pixar`, `toys`, `funny`

becomes a cleaned metadata text field that the recommender can use.

### 8.3 Feature Building

File: `src/features/build_features.py`

This module:

- loads processed movie metadata
- converts `metadata_text` into TF-IDF vectors
- saves reusable runtime artifacts such as:
  - TF-IDF vectorizer
  - TF-IDF matrix
  - movie index maps

- optionally builds and saves a dense similarity matrix for debugging only

### 8.4 Recommendation Engine

File: `src/recommender/recommend.py`

This module:

- loads saved runtime artifacts
- resolves a movie from title or partial title
- computes one-to-all similarity scores on demand
- returns top-N similar movies
- supports filtering by minimum rating count

### 8.5 Training / Build Pipeline

File: `src/models/train_model.py`

This module runs the full project pipeline:

- load raw data
- preprocess metadata
- build TF-IDF artifacts
- optionally build a dense similarity matrix
- save artifacts
- generate sample recommendation checks
- write a training summary JSON file

### 8.6 CLI Entry Point

File: `main.py`

This is the main project entry point and exposes commands for:

- training the pipeline
- finding candidate titles
- generating recommendations
- running the Flask API

### 8.7 Flask API

File: `app/app.py`

This module provides a simple HTTP layer for testing the recommender.

Current endpoints:

- `/`
- `/health`
- `/ready`
- `/candidates`
- `/recommend`

---

## 9. Running the Project

## 9.1 Run the full pipeline

```powershell
python main.py train
```

This command will:

- load raw data
- preprocess metadata
- build TF-IDF features
- save artifacts to `artifacts/`
- save processed metadata to `data/processed/`
- generate a training summary

### Optional custom training command

```powershell
python main.py train `
  --data-dir data/raw `
  --processed-output-path data/processed/movies_metadata.csv `
  --artifact-dir artifacts `
  --summary-output-path artifacts/training_summary.json `
  --sample-titles "Toy Story,Jumanji,Heat" `
  --recommendation-top-n 5 `
  --min-rating-count 10
```

### Optional dense similarity build

This is **not** the normal runtime path, but you can still build the dense similarity matrix explicitly for debugging or comparison:

```powershell
python main.py train --build-similarity-matrix
```

---

## 9.2 Find candidate movie titles

Use this when a title may be ambiguous.

```powershell
python main.py candidates --query "toy" --limit 10
```

Important:
This is a **lightweight metadata-only path**.
It does **not** load TF-IDF runtime artifacts.

This helps find the correct movie before asking for recommendations.

---

## 9.3 Generate recommendations by title

```powershell
python main.py recommend --title "Toy Story" --top-n 10 --min-rating-count 10
```

Example meaning:

- `--title "Toy Story"` = seed movie
- `--top-n 10` = return 10 recommendations
- `--min-rating-count 10` = ignore weak low-signal movies with too few ratings

If you want to include the seed movie itself in the results:

```powershell
python main.py recommend --title "Toy Story" --top-n 10 --include-input-movie
```

Important:
This command loads the sparse TF-IDF runtime artifacts and computes similarity on demand.

---

## 9.4 Run the Flask API

```powershell
python main.py serve
```

Custom host and port:

```powershell
python main.py serve --host 127.0.0.1 --port 5000
```

Debug mode:

```powershell
python main.py serve --debug
```

---

## 10. API Usage

## 10.1 Root endpoint

```http
GET /
```

Purpose:

- shows service information and available endpoints

---

## 10.2 Health check

```http
GET /health
```

Purpose:

- confirms the API is alive
- does **not** force recommender artifact loading

This is a lightweight liveness endpoint.

---

## 10.3 Readiness check

```http
GET /ready
```

Purpose:

- confirms the recommender assets can be loaded and used
- returns runtime readiness information

This is a heavier endpoint than `/health`.

---

## 10.4 Candidate title search

```http
GET /candidates?query=toy
```

Purpose:

- returns likely movie matches for a partial query

Example:

```text
/candidates?query=toy
```

Optional limit:

```text
/candidates?query=toy&limit=5
```

---

## 10.5 Recommendations

```http
GET /recommend?title=Toy%20Story&top_n=10&min_rating_count=10
```

Purpose:

- returns top similar movies for the given title

Example:

```text
/recommend?title=Toy%20Story&top_n=10&min_rating_count=10
```

Optional inclusion of the seed movie:

```text
/recommend?title=Toy%20Story&top_n=10&include_input_movie=true
```

---

## 11. Output Files

After running the training pipeline, the project generates files such as:

### Processed data

```text
data/processed/movies_metadata.csv
```

### Default artifacts

```text
artifacts/tfidf_vectorizer.pkl
artifacts/tfidf_matrix.npz
artifacts/movie_index_maps.pkl
artifacts/training_summary.json
```

### Optional artifact

```text
artifacts/similarity_matrix.npy
```

This file is generated **only if** you run:

```powershell
python main.py train --build-similarity-matrix
```

### What they mean

- `movies_metadata.csv`
  Cleaned movie-level dataset used for feature generation

- `tfidf_vectorizer.pkl`
  Saved TF-IDF model object

- `tfidf_matrix.npz`
  Sparse matrix of text features used for runtime scoring

- `movie_index_maps.pkl`
  Mapping between `movieId` and TF-IDF row positions

- `training_summary.json`
  Pipeline summary and sample recommendation output

- `similarity_matrix.npy`
  Optional dense movie-to-movie similarity scores for debugging or comparison only

---

## 12. Example Recommendation Flow

User asks for recommendations for:

```text
Toy Story
```

The system will:

1. resolve the movie row for `Toy Story`
2. get its TF-IDF row index
3. compute similarity scores between that row and all other movie rows
4. sort movies by highest similarity
5. skip the input movie unless explicitly included
6. optionally filter out low-rating-count movies
7. return the top results

This is why the system feels like:

> show me movies that look like this movie

not:

> show me what similar users watched

---

## 13. Why TF-IDF + On-Demand Similarity

This method is used because it is:

- simple
- fast enough for the current dataset
- explainable
- correct for a first internship recommender
- more efficient than storing a huge dense similarity matrix by default

### TF-IDF

TF-IDF converts words into numbers.

Important words get more value.
Common useless words get less value.

### On-demand similarity scoring

Instead of loading a full saved similarity matrix for all movies, the system computes similarity only for the current seed movie against the TF-IDF matrix.

That is more efficient for this project because recommendation requests only need one movie’s scores at a time.

---

## 14. Current Limitations

This project is correct for the internship task, but it has limitations.

### 14.1 No collaborative filtering

The current system does not learn user-user or user-item behavior.

### 14.2 Metadata is limited

MovieLens small gives:

- title
- genres
- user tags
- ratings
- links

It does not provide rich plot summaries in the base CSV files.

### 14.3 Candidate title matching is heuristic

Exact matching is attempted first, then partial matching.
This is reasonable, but not perfect for all edge cases.

### 14.4 No hybrid ranking

The current recommender mainly uses content similarity.
It does not combine:

- popularity weighting
- collaborative filtering
- semantic embeddings
- reranking

### 14.5 API is still simple

The current Flask app is a lightweight demo API.
It is not production-hardened.

---

## 15. Possible Future Improvements

These are logical future directions for the project:

- enrich metadata using TMDB
- add plot overview text and keyword metadata
- add collaborative filtering baseline
- build a hybrid recommender
- add unit tests
- add better ranking logic using popularity and quality priors
- add evaluation metrics beyond qualitative examples
- package the API for deployment
- persist normalized title columns directly into processed metadata
- use a faster runtime storage format such as Parquet if needed later

---

## 16. EDA Notebook

Notebook file:

```text
notebooks/eda.ipynb
```

This notebook is intended for:

- raw dataset inspection
- missing value checks
- rating distribution review
- genre frequency inspection
- previewing processed metadata
- quick recommendation sanity checks

Use it for analysis and screenshots for internship submission.

---

## 17. Logging and Utilities

Helper file:

```text
src/utils/helpers.py
```

This module provides shared utilities such as:

- logging setup
- safe directory creation
- JSON save/load helpers
- simple DataFrame summaries
- timing helpers

These utilities keep repeated support logic out of the core modules.

---

## 18. Recommended Execution Order

Use the project in this order:

### First-time build

```powershell
python main.py train
```

### Then test ambiguous titles

```powershell
python main.py candidates --query "heat"
```

### Then get recommendations

```powershell
python main.py recommend --title "Heat" --top-n 10 --min-rating-count 10
```

### Then run API if needed

```powershell
python main.py serve
```

This order matters.
If you skip training first, runtime artifacts will not exist.

---

## 19. Troubleshooting

### Problem: raw dataset files not found

Cause:

- CSV files are not inside `data/raw/`

Fix:

- move `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv` into `data/raw/`

---

### Problem: processed file not found

Cause:

- preprocessing/training has not been run yet

Fix:

```powershell
python main.py train
```

---

### Problem: recommender artifacts not found

Cause:

- training has not been run yet
- artifact directory is wrong
- `.env` path configuration is wrong

Fix:

- run `python main.py train`
- verify `artifacts/` exists
- verify `PROCESSED_MOVIES_PATH` and `ARTIFACT_DIR` values

---

### Problem: no movie found for title query

Cause:

- title does not match exactly
- spelling is off
- the title is ambiguous

Fix:

```powershell
python main.py candidates --query "your partial title"
```

---

### Problem: recommendation results look weak

Cause:

- content-based similarity is limited by metadata quality
- MovieLens tags can be sparse or noisy

Fix:

- start with movies that have stronger rating/tag coverage
- later improve metadata enrichment

---

### Problem: `/health` returns ok but recommendations still fail

Cause:

- `/health` is only a liveness check
- it does not guarantee that recommender assets are loaded successfully

Fix:

- use `/ready` to verify recommender readiness

---

## 20. Internship Relevance

This project directly satisfies the main technical expectations of the internship movie recommendation task:

- dataset usage
- EDA
- metadata cleaning
- content-based recommendation
- qualitative recommendation examples
- optional Flask packaging

So this implementation is aligned with the assignment and is not random scope expansion.

---

## 21. Author

Author Name:

```text
Ahmad Sajid
```

Project Name:

```text
Syntecxhub_Movie_Recommendation_System
```
