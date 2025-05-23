import pathlib, json, io, traceback
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, conlist

from contugas_detector import ContugasDetector

# ───────────────────────── rutas ─────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent
MODEL_DIR  = BASE_DIR / "models"
PARQUET    = BASE_DIR / "data" / "historico.parquet"        # único archivo

# 🔹 detector global
_detector: Optional[ContugasDetector] = None

# ───────────────────────── FastAPI ───────────────────────
app = FastAPI(
    title="Contugas Local API",
    description="Detección de anomalías + auto-segmentación y gráfico por cliente",
    version="0.3.0",
)

# ───────────────────────── esquemas ──────────────────────
class StatsVector(BaseModel):
    values: conlist(float, min_items=9, max_items=9)
    def as_np(self):
        return np.array([self.values], dtype=float)

class Reading(BaseModel):
    cliente_id:  str = Field(..., example="CLIENTE1")
    presion:     float
    temperatura: float
    volumen:     float
    stats_vec:   Optional[StatsVector] = None       # requerido si cliente nuevo

class ReadingBatch(BaseModel):
    readings: List[Reading]

# ───────────────────────── startup ───────────────────────
@app.on_event("startup")
def startup_event():
    global _detector
    print("🛠️  startup_event BEGIN")
    print("   • MODEL_DIR exists? ", MODEL_DIR.exists())
    try:
        print("   • Instanciando ContugasDetector…")
        _detector = ContugasDetector(MODEL_DIR)
        print("✅  Detector cargado correctamente")
    except Exception as e:
        print("❌  Error al instanciar detector:", e)
        raise
    print("🛠️  startup_event END")

# ───────────────────────── endpoints core ────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "detector_loaded": _detector is not None}

@app.post("/predict")
async def predict_one(r: Reading):
    if _detector is None:
        raise HTTPException(500, "Detector no está cargado aún")
    try:
        vec = r.stats_vec.as_np() if r.stats_vec else None
        flag, score, seg = _detector.predict(
            r.cliente_id, r.presion, r.temperatura, r.volumen, vec
        )
        return {"cliente_id": r.cliente_id, "segmento": seg,
                "anomaly": flag, "score": score}
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(500, str(exc))

@app.post("/batch_predict")
async def batch_predict(batch: ReadingBatch):
    if _detector is None:
        raise HTTPException(500, "Detector no está cargado aún")
    results = []
    for r in batch.readings:
        vec = r.stats_vec.as_np() if r.stats_vec else None
        flag, score, seg = _detector.predict(
            r.cliente_id, r.presion, r.temperatura, r.volumen, vec
        )
        results.append({"cliente_id": r.cliente_id, "segmento": seg,
                        "anomaly": flag, "score": score})
    return {"results": results}

# ───────────────────────── helper gráfico ────────────────
def _build_plot(cliente_id: str) -> StreamingResponse:
    # 1) leer histórico
    if not PARQUET.exists():
        raise FileNotFoundError("historico.parquet no existe")
    df = pd.read_parquet(PARQUET)
    df = df[df["cliente"] == cliente_id].sort_values("Fecha")
    if df.empty:
        raise KeyError(f"{cliente_id} no existe en Parquet")

    # 2) cargar modelos del segmento
    seg_map = json.load(open(MODEL_DIR / "cliente_segmento.json"))
    if cliente_id not in seg_map:
        raise KeyError(f"{cliente_id} no está registrado en los modelos")
    seg      = seg_map[cliente_id]
    scaler   = joblib.load(MODEL_DIR / f"segment_{seg}_scaler.pkl")
    iso      = joblib.load(MODEL_DIR / f"segment_{seg}_iforest.pkl")

    X = scaler.transform(df[["Presion", "Temperatura", "Volumen"]])
    df["score"]   = -iso.decision_function(X)
    df["anomaly"] = iso.predict(X) == -1

    # 3) gráfico
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(df["Fecha"], df["score"], label="Score")
    ax.scatter(df[df["anomaly"]]["Fecha"],
               df[df["anomaly"]]["score"],
               color="red", label="Anómalo")
    ax.set_title(f"Isolation Forest – {cliente_id}  (seg {seg})")
    ax.set_ylabel("Rareza"); ax.grid(True); ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# ───────────────────────── endpoint gráfico ──────────────
@app.get(
    "/plot/{cliente_id}",
    responses={404: {"description": "Cliente no encontrado"}},
    response_class=StreamingResponse,
)
async def plot_cliente(cliente_id: str):
    try:
        return _build_plot(cliente_id)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(404, str(exc))

def _append_rows(records: list[dict]) -> int:
    if not records:
        return 0
    df_new = pd.DataFrame(records)
    if "Fecha" in df_new.columns:
        df_new["Fecha"] = pd.to_datetime(df_new["Fecha"], errors="coerce")
    if PARQUET.exists():
        old = pd.read_parquet(PARQUET, engine="pyarrow")
        df_out = pd.concat([old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(PARQUET, index=False, engine="pyarrow")
    return len(df_new)

# ───────────────────────── esquemas de ingesta ───────────────
class IngestRow(BaseModel):
    Fecha:        Optional[str]            # ISO8601 opcional
    Presion:      float
    Temperatura:  float
    Volumen:      float
    cliente:      str

class IngestBatch(BaseModel):
    rows: List[IngestRow]

# ───────────────────────── endpoint de ingesta ───────────────
@app.post("/ingest")
async def ingest_data(batch: IngestBatch):
    rows = [r.dict() for r in batch.rows]
    inserted = _append_rows(rows)
    return {"inserted": inserted, "parquet": str(PARQUET)}