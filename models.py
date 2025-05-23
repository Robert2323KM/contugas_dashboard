"""
Re-exporta K-Means + IsolationForest en el entorno local (Py 3.10).
Crea / sobre-escribe todo en la carpeta models/.
"""
import pathlib, pandas as pd, joblib, json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

BASE = pathlib.Path(__file__).resolve().parent
DATA = "Datos.xlsx"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

# ---------- 1. leer Excel completo ----------
xl = pd.ExcelFile(DATA)
frames = []
for sheet in xl.sheet_names:
    df = pd.read_excel(DATA, sheet_name=sheet, parse_dates=["Fecha"])
    df["cliente"] = sheet
    frames.append(df)
df_raw = pd.concat(frames, ignore_index=True)

# ---------- 2. segmentación K-Means ----------
agg = df_raw.groupby("cliente").agg({
    "Presion":      ["mean", "median", "std"],
    "Temperatura":  ["mean", "median", "std"],
    "Volumen":      ["mean", "median", "std"],
})
agg.columns = ["_".join(c) for c in agg.columns]

scaler_km = StandardScaler()
Xs = scaler_km.fit_transform(agg.values)

best_k, best_sc = 0, -1
#for k in range(2, 10):
#    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
#    sc = silhouette_score(Xs, km_tmp.fit_predict(Xs))
#    if sc > best_sc:
#        best_k, best_sc = k, sc
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Xs)
agg["segmento"] = kmeans.labels_

joblib.dump(scaler_km, MODELS / "scaler_kmeans.pkl")
joblib.dump(kmeans,    MODELS / "kmeans.pkl")

seg_map = dict(zip(agg.index, agg["segmento"]))
with open(MODELS / "cliente_segmento.json", "w") as fp:
    json.dump(seg_map, fp, indent=2)

df_raw["segmento"] = df_raw["cliente"].map(seg_map)

# ---------- 3. Isolation Forest por segmento ----------
for seg, grp in df_raw.groupby("segmento"):
    X = grp[["Presion", "Temperatura", "Volumen"]].dropna()
    if len(X) < 20:
        print(f"Segmento {seg} - pocos datos → omitido")
        continue
    scaler_iso = StandardScaler()
    Xs = scaler_iso.fit_transform(X)
    iso = IsolationForest(n_estimators=300, contamination=0.005, random_state=42).fit(Xs)
    joblib.dump(scaler_iso, MODELS / f"segment_{seg}_scaler.pkl")
    joblib.dump(iso,        MODELS / f"segment_{seg}_iforest.pkl")
    print(f"✅ Segmento {seg} exportado")

print("Modelos re-exportados en", MODELS)
