import pandas as pd
from datetime import datetime

df_raw = pd.read_csv('data.csv')


#Columnas
cols = ["track_id", "track_name", "track_number", "track_popularity","track_duration_ms","explicit",
        "artist_name", "artist_popularity", "artist_followers", "artist_genres","album_id","album_name",
        "album_release_date","album_total_tracks","album_type"]

df = df_raw[cols].dropna()

#Normalizacion
df['track_popularity'] = df['track_popularity'].apply(
    lambda x: 100 if x > 100 else 0 if x < 0 else x
)

df['artist_popularity'] = df['artist_popularity'].apply(
    lambda x: 100 if x > 100 else 0 if x < 0 else x
)

df['artist_genres'] = (
    df['artist_genres']
    .fillna('')
    .str.strip('[]')
    .str.replace("'", "", regex=False)
)

df['artist_genres'] = df['artist_genres'].apply(
    lambda x: x if x != '' else 'N/A'
)
#Minutos
df['track_duration_ms'] = df['track_duration_ms']/60000

df = df.rename(columns={'track_duration_ms': 'track_duration_min',})

def fix_date(x):
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip()
    if x.isdigit() and len(x) == 4:
        return f"{x}-01-01"
    return x

df['album_release_date'] = df['album_release_date'].apply(fix_date)

df['album_release_date'] = pd.to_datetime(df['album_release_date'],errors='coerce')
df = df.dropna(subset=['album_release_date'])

print(f"Datos limpios: {df.shape[0]} registros válidos")

#Ordenar por fecha
df.sort_values(by=['album_release_date'],ascending=False, inplace=True)

df.to_csv('data_clean.csv', index=False, encoding='utf-8')



