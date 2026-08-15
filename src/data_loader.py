import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load the Spotify tracks dataset and do basic cleaning.
    """
    df = pd.read_csv(path)
    
    # Drop the junk index column from the CSV export
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Drop rows missing essential info
    df = df.dropna(subset=['track_name', 'artists'])
    
    # Drop exact duplicate songs (same track name + artist)
    df = df.drop_duplicates(subset=['track_name', 'artists'])
    
    return df.reset_index(drop=True)