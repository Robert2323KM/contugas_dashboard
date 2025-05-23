import pathlib, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler

MODELS  = pathlib.Path("models")
PARQUET = pathlib.Path("data/historico.parquet")

cliente = "CLIENTE1"

# 1) carga datos históricos del cliente
df = pd.read_parquet(PARQUET)
df = df[df["cliente"] == cliente].sort_values("Fecha")

# 2) carga scaler + Isolation Forest de su segmento
seg_map = json.load(open(MODELS / "cliente_segmento.json"))
seg     = seg_map[cliente]
scaler  = joblib.load(MODELS / f"segment_{seg}_scaler.pkl")
iso     = joblib.load(MODELS / f"segment_{seg}_iforest.pkl")

# 3) calcula score y flag
X  = scaler.transform(df[["Presion","Temperatura","Volumen"]])
df["score"]   = -iso.decision_function(X)
df["anomaly"] = iso.predict(X) == -1

# 4) gráfico
plt.figure(figsize=(12,4))
plt.plot(df["Fecha"], df["score"], label="Score")
plt.scatter(df[df["anomaly"]]["Fecha"],
            df[df["anomaly"]]["score"],
            color="red", label="Anómalo")
plt.axhline(df["score"].quantile(.95), ls="--", color="gray")
plt.title(f"Isolation Forest – {cliente} (segmento {seg})")
plt.ylabel("Rareza (score)")
plt.legend(); plt.tight_layout()
plt.show()
