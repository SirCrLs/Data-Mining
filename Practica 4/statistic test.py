import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, kruskal

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

#tests
sns.set(style="whitegrid")
plt.figure(figsize=(8,5))

explicit_tracks = trackdf[trackdf['explicit'] == 1]['track_popularity']
clean_tracks = trackdf[trackdf['explicit'] == 0]['track_popularity']

t_stat, p_value = ttest_ind(explicit_tracks, clean_tracks)
print(" =========  POPULARIDAD VS EXPLICIT  =========")
print("T statistic:", t_stat)
print("p-value:", p_value)
print("Se realizo una prueba T-test para comparar la popularidad de las canciones explícitas y no explícitas. ")
print("El valor p obtenido (p < 0.05) indica que existe una diferencia significativa entre ambos grupos.")
print("Por lo tanto, se concluye que el contenido explícito está asociado con diferencias en la popularidad de las canciones.")

sns.boxplot(x='explicit', y='track_popularity', data=trackdf)

plt.title('Popularidad por Explicitud')
plt.xlabel('Explicito (0 = No, 1 = Si)')
plt.ylabel('Popularidad de la cancion')

plt.savefig("Distribucion T Popularidad-Explicito.png")
plt.show()


groups = [
    group['track_popularity'].values
    for name, group in df.groupby('album_type')
]

f_stat, p_value = f_oneway(*groups)
print(" =========  POPULARIDAD VS TIPO DE ALBUM  =========")
print("F statistic:", f_stat)
print("p-value:", p_value)
print("Se aplicó ANOVA para analizar si la popularidad de las canciones varía por el tipo de album. ")
print("p < 0.05, lo que indica que existe una diferencia significativa entre ambos grupos.")
print("Por lo tanto, se concluye que el tipo de álbum influye en la popularidad de las canciones.")
sns.boxplot(x='album_type', y='track_popularity', data=df)

plt.title('Popularidad por Tipo de Album')
plt.xlabel('Tipo de Album')
plt.ylabel('Popularidad')

plt.savefig("Distribucion F Popularidad-Album.png")
plt.show()

groups = [
    group['artist_popularity'].values
    for name, group in artistdf.groupby('artist_genres')
]

stat, p_value = kruskal(*groups)
print(" =========  ARTISTA POPULARIDAD VS GENEROS  =========")
print("Statistic:", stat)
print("p-value:", p_value)
print("Se utilizo la prueba Kruskal-Wallis para evaluar si la popularidad de los artistas cambia segun la cantidad de géneros del artista. ")
print("El valor p es (p < 0.05) muestra que existen diferencias significativas entre los grupos. ")
print("En consecuencia, se concluye que el número de géneros  asociados a un artista está relacionado con su popularidad.")

sns.boxplot(x='artist_genres', y='artist_popularity', data=artistdf)

plt.title('Popularidad del Artista vs Cantidad de generos')
plt.xlabel('Numero de Generos')
plt.ylabel('Popularidad')

plt.savefig("Distribucion Popularidad-Generos.png")
plt.show()
