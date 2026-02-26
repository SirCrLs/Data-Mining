import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_clean.csv')

plt.style.use('_mpl-gallery-nogrid')

# EXPLICIT => TRUE / FALSE
label = ["False", "True"]
aux = df["explicit"].value_counts()
x = [0,0] #x[0] = false.count // x[1] = true.count
x[0],x[1] = int(aux.iloc[0]),int(aux.iloc[1])


plt.pie(x,labels=label,autopct='%1.3f%%',startangle=90,colors=['blue','orange'] )
plt.title('Es explícito?', fontsize=14, pad=20)
plt.savefig('pieExplicit.png', bbox_inches='tight')
# plt.show()

# GENRES
aux = df["artist_genres"].str.split(", ")
aux = aux.explode().value_counts()
x = []
label = []
for i in aux.keys():
    label.append(i)
for key in label:
    x.append(int(aux[key]))
label = label[0:20]
x = x[0:20]

plt.figure(figsize=(10, 8))
plt.barh(label, x, color='skyblue')

plt.title('Top 20 Generos mas Frecuentes')
plt.xlabel('Número de Repeticiones')
plt.ylabel('Género')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('top20g.png', bbox_inches='tight')
#plt.show()

# top 10 artiusta

aux = df[["artist_name","artist_popularity"]]
aux = aux.sort_values(by=['artist_popularity'], ascending=False)
aux = aux.drop_duplicates(subset=['artist_name'], keep='first')
label = []
x = []
for name in aux["artist_name"]:
    label.append(name)
for i in aux["artist_popularity"]:
    x.append(i)
x = x[0:10]
label = label[0:10]

plt.figure(figsize=(10, 8))
plt.bar(label, x, color='skyblue')

plt.title('Top 10 Artistas por popularidad')
plt.xlabel('Artista')
plt.ylabel('Popularidad')
plt.xticks(rotation=45, ha='right')
plt.ylim(80,100)

plt.tight_layout()
plt.savefig('top10ArtistasPopularidad.png', bbox_inches='tight')

#Top 10 artistas followers
aux = df[["artist_name","artist_followers"]]
aux = aux.sort_values(by=['artist_followers'], ascending=False)
aux = aux.drop_duplicates(subset=['artist_name'], keep='first')
label = []
x = []
for name in aux["artist_name"]:
    label.append(name)
for i in aux["artist_followers"]:
    x.append(i)
x = x[0:10]
label = label[0:10]

plt.figure(figsize=(10, 8))
plt.bar(label, x, color='green')

plt.title('Top 10 Artistas por followers (Millones)')
plt.xlabel('Artista')
plt.ylabel('Followers')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('top10ArtistasFollowers.png', bbox_inches='tight')


#top 10 canciones
aux = df[["track_name","track_popularity"]]
aux = aux.sort_values(by=['track_popularity'], ascending=False)
aux = aux.drop_duplicates(subset=['track_name'], keep='first')
label = []
x=[]
for name in aux["track_name"]:
    label.append(name)
for i in aux["track_popularity"]:
    x.append(i)
x = x[0:10]
label = label[0:10]

plt.figure(figsize=(10, 8))
plt.bar(label, x, color='skyblue')
plt.title('Top 10 Canciones por popularidad')
plt.xlabel('Cancion')
plt.ylabel('Popularidad')
plt.xticks(rotation=45, ha='right')
plt.ylim(80,100)

plt.savefig('top10CancionesPopularidad.png', bbox_inches='tight')

#tipo de album
aux = df[["album_name","album_type"]]
aux.drop_duplicates(subset=['album_name'], keep='first')
aux = aux["album_type"].value_counts()

label = ["Album", "Single","Compilation"]
x=[int(aux.iloc[0]),int(aux.iloc[1]),int(aux.iloc[2])]
plt.figure(figsize=(10, 8))
plt.title('Tipo de album', fontsize=24, pad=20)
plt.pie(x,labels=label,autopct='%1.3f%%',startangle=90 )

plt.savefig('pieAlbumType.png', bbox_inches='tight')
