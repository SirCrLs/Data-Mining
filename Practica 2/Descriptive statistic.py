import numpy as np
import pandas as pd

df = pd.read_csv("data_clean.csv")

# TRACK DATAFRAME
trackdf = df[[
    'track_popularity',
    'track_duration_min',
    'explicit'
]].copy()

# FALSE = 0 // TRUE = 1
trackdf['explicit'] = trackdf['explicit'].astype(int)


# ARTIST DATAFRAME
artistdf = df[[
    'artist_name',
    'artist_popularity',
    'artist_followers',
    'artist_genres'
]].copy()

#borrar artistas duplicados
artistdf = artistdf.drop_duplicates(
    subset='artist_name',
    keep='first'
)

#Cantidad de generos  /// N/A = 0
artistdf['artist_genres'] = (
    artistdf['artist_genres']
    .apply(lambda x: 0
           if pd.isna(x) or str(x).strip().lower() in ['unknown', 'n/a', '']
           else len([g.strip() for g in str(x).split(',') if g.strip() != '']))
)

albumdf = df[['album_name',
              'album_total_tracks',
              'album_type']].copy()

albumdf = albumdf.drop_duplicates(
    subset='album_name',
    keep='first'
)

# album  = 0 single = 1 otro = 2
albumdf['album_type'] = (
    albumdf['album_type']
    .apply(lambda x: 0 if x == 'album' else 1 if x == 'single' else 2 )
)

print("---- TRACKS STATISTICS ----")
print(trackdf.describe())
print("---- ARTISTS STATISTICS ----")
print(artistdf.describe())
print("---- ALBUM STATISTICS ----")
print(albumdf.describe())
