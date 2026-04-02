# Syntecxhub Movie Recommendation System

A content-based Movie Recommendation System built for the Syntecxhub internship task.

This project uses the **MovieLens small dataset** to:

- load and validate movie data
- clean and preprocess metadata
- build TF-IDF text features
- compute cosine similarity between movies
- return similar movie recommendations by title
- expose a simple Flask API for testing

The current implementation is a **content-based recommender**, not collaborative filtering.

---

## 1. Project Goal

The goal of this project is to recommend movies that are similar to a given movie based on metadata.

Instead of asking, “What did other users watch?”, this project asks, “What movies look similar based on their content?”

That content is built mainly from:

- movie title
- genres
- user tags

Those text fields are cleaned and merged into one metadata field, then transformed into numbers using **TF-IDF**, and compared using **cosine similarity**.

---

## 2. Current Approach

This project follows a simple and correct pipeline:

1. **Load raw MovieLens CSV files**
2. **Validate file presence and schema assumptions**
3. **Clean movie metadata**
4. **Aggregate user tags per movie**
5. **Build rating summary statistics**
6. **Create one final `metadata_text` field per movie**
7. **Convert text into TF-IDF vectors**
8. **Build a movie-to-movie cosine similarity matrix**
9. **Return top-N similar movies for a given title**
10. **Optionally expose the recommender through Flask**

This is appropriate for the current internship scope because it is:

- explainable
- easy to verify
- fast enough for MovieLens small
- simple to demo

---

## 3. Tech Stack

- **Python 3.x**
- **pandas** for data handling
- **numpy** for numerical operations
- **scikit-learn** for TF-IDF and cosine similarity
- **matplotlib** for EDA plots
- **Flask** for the simple API
- **Jupyter Notebook** for exploratory analysis

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

So the expected raw data layout is:

```text
data/raw/
├── movies.csv
├── ratings.csv
├── tags.csv
└── links.csv
```

If these files are missing, the loader will fail early with a clear error.

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

## 7. How the Pipeline Works

### 7.1 Data Loading

File: `src/data/load_data.py`

This module:

- resolves the raw dataset paths
- checks that all required files exist
- loads the CSV files with controlled dtypes
- converts timestamps into readable datetime fields
- returns the dataset in a structured bundle

### 7.2 Preprocessing

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

becomes a cleaned metadata text field that the recommender can understand.

### 7.3 Feature Building

File: `src/features/build_features.py`

This module:

- loads processed movie metadata
- converts `metadata_text` into TF-IDF vectors
- computes cosine similarity between all movies
- saves reusable artifacts such as:
  - TF-IDF vectorizer
  - TF-IDF matrix
  - similarity matrix
  - movie index maps

### 7.4 Recommendation Engine

File: `src/recommender/recommend.py`

This module:

- loads saved artifacts
- resolves a movie from title or partial title
- retrieves similarity scores
- returns top-N similar movies
- supports filtering by minimum rating count

### 7.5 Training / Build Pipeline

File: `src/models/train_model.py`

This module runs the full project pipeline:

- load raw data
- preprocess metadata
- build features
- save artifacts
- generate sample recommendation checks
- write a training summary JSON file

### 7.6 CLI Entry Point

File: `main.py`

This is the main project entry point and exposes commands for:

- training the pipeline
- finding candidate titles
- generating recommendations
- running the Flask API

### 7.7 Flask API

File: `app/app.py`

This module provides a simple HTTP layer for testing the recommender.

Expected endpoints:

- `/health`
- `/candidates`
- `/recommend`

---

## 8. Running the Project

## 8.1 Run the full pipeline

```powershell
python main.py train
```

This command will:

- load raw data
- preprocess metadata
- build TF-IDF features
- build similarity matrix
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

---

## 8.2 Find candidate movie titles

Use this when a title may be ambiguous.

```powershell
python main.py candidates --query "toy" --limit 10
```

This helps find the correct movie before asking for recommendations.

---

## 8.3 Generate recommendations by title

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

---

## 8.4 Run the Flask API

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

## 9. API Usage

## 9.1 Health check

```http
GET /health
```

Purpose:

- confirms the API is alive

---

## 9.2 Candidate title search

```http
GET /candidates?query=toy
```

Purpose:

- returns likely movie matches for a partial query

Example:

```text
/candidates?query=toy
```

---

## 9.3 Recommendations

```http
GET /recommend?title=Toy%20Story&top_n=10&min_rating_count=10
```

Purpose:

- returns top similar movies for the given title

Example:

```text
/recommend?title=Toy%20Story&top_n=10&min_rating_count=10
```

---

## 10. Output Files

After running the training pipeline, the project generates files such as:

### Processed data

```text
data/processed/movies_metadata.csv
```

### Artifacts

```text
artifacts/tfidf_vectorizer.pkl
artifacts/tfidf_matrix.npz
artifacts/similarity_matrix.npy
artifacts/movie_index_maps.pkl
artifacts/training_summary.json
```

What they mean:

- `movies_metadata.csv`
  Cleaned movie-level dataset used for feature generation

- `tfidf_vectorizer.pkl`
  Saved TF-IDF model

- `tfidf_matrix.npz`
  Sparse matrix of text features

- `similarity_matrix.npy`
  Dense movie-to-movie similarity scores

- `movie_index_maps.pkl`
  Mapping between `movieId` and matrix row positions

- `training_summary.json`
  Pipeline summary and sample recommendation output

---

## 11. Example Recommendation Flow

User asks for recommendations for:

```text
Toy Story
```

The system will:

1. find the movie row for `Toy Story`
2. get its matrix index
3. read similarity scores against all other movies
4. sort movies by highest similarity
5. skip the input movie unless explicitly included
6. optionally filter out low-rating-count movies
7. return the top results

This is why the system feels like:
“show me movies that look like this movie”

not:
“show me what similar users watched”

---

## 12. Why TF-IDF + Cosine Similarity

This method is used because it is:

- simple
- fast
- explainable
- correct for a first internship recommender

### TF-IDF

TF-IDF converts words into numbers.

Important words get more value.
Common useless words get less value.

### Cosine Similarity

Cosine similarity measures how close two movie text vectors are.

If two movies have very similar cleaned metadata, their cosine similarity score will be high.

This is why the recommender can find movies with similar genres, tags, and title patterns.

---

## 13. Current Limitations

This project is correct for the internship task, but it has limitations.

### 13.1 No collaborative filtering

The current system does not learn user-user or user-item behavior.

### 13.2 Metadata is limited

MovieLens small gives:

- title
- genres
- user tags
- ratings
- links

It does not provide rich plot summaries in the base CSV files.

### 13.3 Similarity matrix is dense

A full dense similarity matrix is acceptable for MovieLens small.
It would become a bad design for a large-scale production dataset.

### 13.4 No hybrid ranking

The current recommender mainly uses content similarity.
It does not combine:

- popularity weighting
- collaborative filtering
- semantic embeddings
- reranking

### 13.5 Title resolution is still heuristic

Exact matching is attempted first, then partial matching.
This is reasonable, but not perfect for all edge cases.

---

## 14. Possible Future Improvements

These are logical future directions for the project:

- enrich metadata using TMDB
- add plot overview text and keyword metadata
- add collaborative filtering baseline
- build a hybrid recommender
- add unit tests
- add model evaluation metrics beyond qualitative examples
- add better ranking logic using popularity and quality priors
- package the API for deployment

---

## 15. EDA Notebook

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

## 16. Logging and Utilities

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

## 17. Recommended Execution Order

Use the project in this order:

### First time run

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
If you skip training first, recommendation artifacts will not exist.

---

## 18. Troubleshooting

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

## 19. Internship Relevance

This project directly satisfies the main technical expectations of the internship movie recommendation task:

- dataset usage
- EDA
- metadata cleaning
- content-based recommendation
- qualitative recommendation examples
- optional Flask packaging

So this implementation is aligned with the assignment and is not random project scope expansion.

---

## 20. Author

Project Name:

```text
Syntecxhub_Movie_Recommendation_System
```

Internship:

```text
Syntecxhub Internship Program
```

```

```
