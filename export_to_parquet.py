# parquet_io.py
"""
I/O utilidades para Contugas
----------------------------
• build_master_parquet()  → genera historico.parquet desde Datos.xlsx
• append_rows(records)    → añade nuevas filas al Parquet
"""
import pathlib, pandas as pd
from typing import List, Dict

BASE   = pathlib.Path(__file__).resolve().parent
DATA_X = "Datos.xlsx"
PARQUET = BASE / "data" / "historico.parquet"   # único archivo “master”

# ──────────────────────────────────────────────────────────────
def build_master_parquet():
    """Lee cada hoja del Excel y crea/reescribe historico.parquet."""
    xl = pd.ExcelFile(DATA_X)
    frames = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(DATA_X, sheet_name=sheet, parse_dates=["Fecha"])
        df["cliente"] = sheet
        frames.append(df)
    master = pd.concat(frames, ignore_index=True)
    master.to_parquet(PARQUET, index=False, engine="pyarrow")
    print(f"✅  {PARQUET.name} generado ({len(master):,} filas)")

# ──────────────────────────────────────────────────────────────
def append_rows(records: List[Dict]) -> int:
    """
    Añade registros al Parquet. `records` es una lista de dicts, p.ej.:

    records = [
        {"Fecha":"2025-05-27 08:00","Presion":11.9,"Temperatura":24.8,
         "Volumen":52.3,"cliente":"CLIENTE1"},
        {"Fecha":"2025-05-27 08:00","Presion":9.8,"Temperatura":23.1,
         "Volumen":47.6,"cliente":"NEW_01"}
    ]
    """
    if not records:
        return 0

    df_new = pd.DataFrame(records)
    # normaliza tipos
    if "Fecha" in df_new.columns:
        df_new["Fecha"] = pd.to_datetime(df_new["Fecha"], errors="coerce")

    if PARQUET.exists():
        old = pd.read_parquet(PARQUET, engine="pyarrow")
        df_out = pd.concat([old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_parquet(PARQUET, index=False, engine="pyarrow")
    print(f"⬆️  {len(df_new)} filas añadidas  →  {PARQUET.name}")
    return len(df_new)

# ── test rápido ───────────────────────────────────────────────
if __name__ == "__main__":
    # 1) construir master (solo la primera vez o si cambió Datos.xlsx)
    build_master_parquet()

    # 2) simular llegada de nuevas lecturas
#     nuevos = [
#         {"Fecha":"2025-05-28 09:00","Presion":11.5,"Temperatura":24.4,
#          "Volumen":51.0,"cliente":"CLIENTE1"},
#         {"Fecha":"2025-05-28 09:00","Presion":10.2,"Temperatura":23.0,
#          "Volumen":49.3,"cliente":"NEW_02"}
#     ]
#     append_rows(nuevos)
