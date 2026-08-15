# 🎵 Music Recommender

A content-based music recommendation system built as a hands-on introduction to machine learning. Given a song you like, it recommends similar songs based on their audio characteristics — danceability, energy, tempo, acousticness, and more — using cosine similarity.

This was built as a one-week learning project, prioritizing understanding core ML concepts over production polish.

---

## How it works

Most music recommenders you've used (Spotify, YouTube Music) rely on **collaborative filtering** — "people who listened to X also listened to Y." That requires huge amounts of user listening data, which isn't available for a project like this.

Instead, this project uses **content-based filtering**: it recommends songs that *sound* similar, based on their actual audio properties, regardless of who else has listened to them.

### The pipeline

1. **Feature extraction** — Each song is represented as a vector of 9 numeric audio features:
   `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

2. **Scaling** — These features have very different ranges (e.g. `loudness` is roughly -60 to 0 dB, `tempo` is roughly 50-200 BPM, `danceability` is 0-1). Without scaling, features with larger numbers would dominate the similarity calculation. All features are scaled to a common 0-1 range using `MinMaxScaler`.

3. **Cosine similarity** — To find songs similar to a given track, the system computes the cosine similarity between that song's feature vector and every other song's vector. Cosine similarity measures the angle between two vectors — it captures how similar the *pattern* of a song's features is, not just the raw magnitude.

4. **On-demand computation** — With ~81,000 songs, precomputing similarity between every pair of songs would require an 81,000 × 81,000 matrix (tens of billions of values) — this crashes with a memory error on a normal machine. Instead, similarity is computed **on demand**: only between the one song you're asking about and every other song (81,000 × 1), which is cheap and fast.

5. **Ranking** — Songs are sorted by similarity score, and the top N most similar tracks are returned (excluding the song itself).

### A note on genre

The recommendations are based purely on audio characteristics — not on the `track_genre` label in the dataset. This means recommendations can span multiple genres if the underlying sound (tempo, energy, mood) is similar. This is expected behavior, not a bug — genre is a human/cultural label, while this system is matching on the raw acoustic profile of a song.

---

## Dataset

**Source:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets) on Kaggle (uploaded by user "Yash")

- ~114,000 tracks originally, ~81,000 after cleaning
- Columns include: `track_name`, `artists`, `album_name`, `track_genre`, `popularity`, `duration_ms`, `explicit`, and the 9 audio features listed above

> **Note:** Spotify deprecated public API access to audio features and recommendations for new apps in November 2024, so this project uses a static, pre-collected dataset rather than live API calls.

---

## Project structure

```
music-recommender/
├── data/
│   └── spotify_tracks.csv       # Raw dataset
├── notebooks/
│   └── *.ipynb                  # Exploration, cleaning, and testing
├── src/
│   ├── data_loader.py           # Loads and cleans the dataset
│   └── recommender.py           # Core recommendation logic
├── app.py                       # Command-line interface
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd music-recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies: `pandas`, `scikit-learn`

### 3. Add the dataset

Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets) from Kaggle and place the CSV file in the `data/` folder as `spotify_tracks.csv`.

---

## Running the project

From the project root:

```bash
python app.py
```

You'll be prompted to:
1. Search for a song by (partial) name
2. Pick the exact song from the list of matches (handles duplicate titles by different artists)
3. View 10 recommended songs based on audio similarity

Example session:

```
Search for a song (or 'quit' to exit): blinding lights

Found 6 matches:
  1. Blinding Lights — Kidz Bop Kids
  2. Blinding Lights — The Weeknd
  ...

Pick a number (or press Enter to search again): 2

Because you picked 'Blinding Lights' by The Weeknd, you might like:
  1. The Mother We Share — CHVRCHES [synth-pop]
  ...
```

Type `quit` at any prompt to exit.

---

## Key design decisions & lessons learned

- **On-demand similarity over precomputed matrix:** The initial approach tried to precompute a full pairwise similarity matrix for all songs, which caused a `MemoryError`. Switching to computing similarity only between the query song and all others (rather than all songs against all songs) solved this while keeping recommendation quality identical.

- **Artist disambiguation:** Many songs share the same title across different artists (e.g. six different songs are titled "Blinding Lights"). Without accounting for this, the system would silently pick whichever matching song happens to appear first in the dataset — leading to irrelevant recommendations with no error or warning. The search-then-select flow in `app.py`, combined with an optional `artist` parameter in `get_recommendations()`, resolves this.

- **Scope:** Content-based filtering with cosine similarity was chosen deliberately over collaborative filtering or neural approaches, since it's achievable in a one-week timeframe without requiring user interaction data, while still covering core ML concepts: feature engineering, scaling, and similarity metrics.

---

## Possible future improvements

- Web interface (e.g. Streamlit) instead of command-line
- Incorporate `track_genre` as a feature or filter
- Weight features (e.g. prioritize `energy` and `valence` over `liveness`)
- Allow recommendations based on multiple seed songs at once
- Deploy as a small web app

---

## Tech stack

- **Python 3.14**
- **pandas** — data loading and cleaning
- **scikit-learn** — feature scaling (`MinMaxScaler`) and similarity computation (`cosine_similarity`)