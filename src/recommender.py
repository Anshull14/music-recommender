import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# The audio features we'll use to measure song similarity
FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Scale audio features to 0-1 range so no single feature
    (like tempo, which has much bigger numbers) dominates the similarity calculation.
    """
    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(df[FEATURES])
    return feature_matrix

def get_recommendations(song_name: str, df: pd.DataFrame, feature_matrix: np.ndarray, 
                         artist: str = None, top_n: int = 10) -> pd.DataFrame:
    """
    Given a song name, return the top_n most similar songs.
    If multiple songs share the name, pass `artist` to disambiguate.
    """
    matches = df[df['track_name'].str.lower() == song_name.lower()]
    
    if artist:
        matches = matches[matches['artists'].str.lower().str.contains(artist.lower(), na=False)]
    
    if matches.empty:
        raise ValueError(f"Song '{song_name}' (artist: {artist}) not found in dataset.")
    
    if len(matches) > 1:
        print(f"Warning: {len(matches)} matches found, using the first one: "
              f"'{matches.iloc[0]['track_name']}' by {matches.iloc[0]['artists']}")
    
    idx = matches.index[0]
    query_vector = feature_matrix[idx].reshape(1, -1)
    scores = cosine_similarity(query_vector, feature_matrix)[0]
    
    scored_indices = list(enumerate(scores))
    scored_indices = sorted(scored_indices, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    song_indices = [i for i, _ in scored_indices]
    return df.iloc[song_indices][['track_name', 'artists', 'track_genre']]

def search_song(query: str, df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """
    Case-insensitive partial match search, useful for autocomplete/search box.
    """
    matches = df[df['track_name'].str.lower().str.contains(query.lower(), na=False)]
    return matches[['track_name', 'artists']].drop_duplicates().head(limit)