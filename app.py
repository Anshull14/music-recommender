"""
Simple command-line interface for the music recommendation system.
Run with: python app.py
"""

import sys
sys.path.append('src')

from data_loader import load_data
from recommender import build_feature_matrix, get_recommendations, search_song


def main():
    print("Loading dataset...")
    df = load_data('data/spotify_tracks.csv')
    feature_matrix = build_feature_matrix(df)
    print(f"Loaded {len(df)} songs. Ready!\n")

    while True:
        query = input("\nSearch for a song (or 'quit' to exit): ").strip()

        if query.lower() == 'quit':
            print("Goodbye!")
            break

        if not query:
            continue

        # Step 1: search for matches
        matches = search_song(query, df, limit=10)

        if matches.empty:
            print(f"No songs found matching '{query}'. Try again.")
            continue

        # Step 2: show matches, numbered
        print(f"\nFound {len(matches)} matches:")
        for i, (_, row) in enumerate(matches.iterrows()):
            print(f"  {i + 1}. {row['track_name']} — {row['artists']}")

        # Step 3: let user pick one
        choice = input("\nPick a number (or press Enter to search again): ").strip()

        if not choice:
            continue

        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(matches):
                print("Invalid number, try again.")
                continue
        except ValueError:
            print("Please enter a number.")
            continue

        chosen_song = matches.iloc[choice_idx]
        song_name = chosen_song['track_name']
        artist_name = chosen_song['artists'].split(';')[0]  # use first artist for disambiguation

        # Step 4: get and display recommendations
        print(f"\nBecause you picked '{song_name}' by {artist_name}, you might like:\n")
        recs = get_recommendations(song_name, df, feature_matrix, artist=artist_name, top_n=10)

        for i, (_, row) in enumerate(recs.iterrows()):
            print(f"  {i + 1}. {row['track_name']} — {row['artists']} [{row['track_genre']}]")


if __name__ == "__main__":
    main()