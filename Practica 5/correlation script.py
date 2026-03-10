import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data_clean.csv")

# FALSE = 0 // TRUE = 1
df['explicit'] = df['explicit'].astype(int)

#Solo anio
df["album_release_date"] = pd.to_datetime(df["album_release_date"])
df["album_release_date"] = df["album_release_date"].dt.year

# Columnas  para el analisis
FEATURES = [
    "artist_popularity",
    "artist_followers",
    "track_duration_min",
    "track_number",
    "album_total_tracks",
    "album_release_date",
    "explicit",
]
TARGET = "track_popularity"

cols_needed = FEATURES + [TARGET]
df = df[cols_needed]

# matriz de correlacion
corr_matrix = df.corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
)
plt.title("Matriz de Correlacion", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlacion matriz.png", dpi=150)
plt.close()

# Correlacion específica con track_popularity
print("Correlacion con track_popularity:")
print(corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False).to_string())
print("Conclusión: Solo artist_popularity tiene una relación medianamente útil. "
      "\nEl resto aporta muy poco o nada.\n")


# regresion lineal simple
X_simple = df[["artist_popularity"]]
y = df[TARGET]

reg_simple = LinearRegression()
reg_simple.fit(X_simple, y)

print("==== Regresión Lineal Simple ====")
print(f"  Variable: artist_popularity -> {TARGET}")
print(f"  Coeficiente: {reg_simple.coef_[0]:.4f}")
print(f"  Intercepto:              {reg_simple.intercept_:.4f}")
print(f"  R^2 :      {reg_simple.score(X_simple, y):.4f}")
print("Conclusion: El R^2 de 0.207 significa que artist_popularity explica "
      "\nsolo el 20.7% de la variación en track_popularity.\n")

# grafico de dispersion
plt.figure(figsize=(8, 5))
plt.scatter(X_simple, y, alpha=0.3, color="steelblue", label="Datos")
x_range = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
plt.plot(x_range, reg_simple.predict(x_range), color="red", linewidth=2, label="Regresion")
plt.xlabel("artist_popularity")
plt.ylabel("track_popularity")
plt.title("Regresion Lineal Simple", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("regresion simple.png", dpi=150)
plt.close()

#regresion lineal multiple
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

reg_multi = LinearRegression()
reg_multi.fit(X_train_sc, y_train)

y_pred = reg_multi.predict(X_test_sc)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("==== regresion lineal multiple ====")
print(f"  Variables: {FEATURES}")
print(f"  R^2 :  {r2:.4f}")
print(f"  RMSE : {rmse:.4f}")
print("Conclusion: Agregar todas las variables apenas mejora el R2 de 0.207 a 0.236. "
      "\nEso confirma que las demás variables casi no aportan informacion nueva."
      "\nEl RMSE de 21.2 significa que el modelo se equivoca en promedio +-21 puntos al predecir track_popularity")

# Importancia de coeficientes
coef_df = pd.DataFrame({
    "Feature":     FEATURES,
    "Coeficiente": reg_multi.coef_
}).sort_values("Coeficiente", key=abs, ascending=False)

# grafico de coeficientes
plt.figure(figsize=(8, 5))
colors = ["tomato" if c < 0 else "steelblue" for c in coef_df["Coeficiente"]]
plt.barh(coef_df["Feature"], coef_df["Coeficiente"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coeficiente estandarizado")
plt.title("Importancia de Variables / Regresión Multiple", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("regresion coeficientes.png", dpi=150)
plt.close()

# Gráfico real vs predicho
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color="mediumpurple")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Real vs Predicho", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("realvspredicho.png", dpi=150)
plt.close()
