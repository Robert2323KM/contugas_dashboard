# streamlit_app.py – versión 0.5  ✨ mejoras UI + lote ingest
"""
• Vista cliente: líneas P/T/V + score IF
• Vista global: KPIs, líneas globales y tabla últimas N lecturas
• Ingestar:  ➜ formulario individual  ➜ upload JSON/CSV (lote)
"""

import os, io, json, datetime as dt, pathlib, requests, pandas as pd, streamlit as st, matplotlib.pyplot as plt, joblib, numpy as np, altair as alt

API_URL   = os.getenv("CONTUGAS_API", "http://127.0.0.1:8000")
BASE_DIR  = pathlib.Path(__file__).resolve().parent
PARQUET   = BASE_DIR / "data" / "historico.parquet"
MODEL_DIR = BASE_DIR / "models"

# ---------- utils ---------------------------------------------------------
@st.cache_data(ttl=60)
def load_parquet():
    if PARQUET.exists():
        return pd.read_parquet(PARQUET).sort_values("Fecha")
    return pd.DataFrame()

def call_api(path: str, method="GET", json=None):
    try:
        r = requests.request(method, f"{API_URL}{path}", json=json, timeout=300)
        r.raise_for_status(); return r
    except Exception as e:
        st.error(f"Error API: {e}")
        return None

# ---------- sidebar -------------------------------------------------------
st.set_page_config(page_title="Contugas", layout="wide")
st.sidebar.title("Contugas Dashboard")
menu = st.sidebar.radio("Navegación", ["Cliente", "Global", "Ingestar"])

df = load_parquet()
clientes = sorted(df["cliente"].unique()) if not df.empty else []

# ===================== 1. Vista por cliente ===============================
if menu == "Cliente":
    if not clientes:
        st.warning("Parquet vacío. Ingesta datos primero."); st.stop()

    cliente = st.sidebar.selectbox("Cliente", clientes)
    st.header(f"Cliente {cliente}")

    df_c = df[df["cliente"] == cliente]
    if df_c.empty:
        st.info("Sin datos de este cliente"); st.stop()

    # --- líneas P/T/V -----------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(11,6), sharex=True)
    ax[0].plot(df_c["Fecha"], df_c["Presion"], label="Presión"); ax[0].set_ylabel("Presión")
    ax[1].plot(df_c["Fecha"], df_c["Temperatura"], color="orange"); ax[1].set_ylabel("Temp")
    ax[2].plot(df_c["Fecha"], df_c["Volumen"], color="green"); ax[2].set_ylabel("Volumen")
    for a in ax: a.grid(True)
    st.pyplot(fig)

    # --- gráfico Isolation Forest via API ---------------------------------
    col_left, col_right = st.columns([2,1])
    with col_left:
        st.subheader("Score anomalía (Isolation Forest)")
    r = call_api(f"/plot/{cliente}")
    if r is not None:
        col_left.image(r.content, use_column_width=True)

# ===================== 2. Vista global ===================================
if menu == "Global":
    st.header("Resumen global")
    if df.empty:
        st.warning("Sin datos"); st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes", len(clientes))
    col2.metric("Lecturas", len(df))
    col3.metric("Última lectura", df["Fecha"].max().strftime("%Y-%m-%d"))

    st.markdown("### Distribución de clientes por segmento")
    try:
        seg_map = json.load(open(MODEL_DIR/"cliente_segmento.json"))
        seg_series = pd.Series(seg_map).value_counts().sort_index().reset_index()
        seg_series.columns = ["segmento","count"]
        bar = alt.Chart(seg_series).mark_bar().encode(x="segmento:O", y="count:Q")
        st.altair_chart(bar, use_container_width=True)
    except FileNotFoundError:
        st.info("Mapeo de segmentos no encontrado")

    st.markdown("### Tendencia global (medias diarias)")
    df_day = df.set_index("Fecha").groupby(pd.Grouper(freq="D")).mean(numeric_only=True).reset_index()
    if not df_day.empty:
        line = (
            alt.Chart(df_day)
            .transform_fold(
                ["Presion", "Temperatura", "Volumen"],
                as_=["variable", "value"]  # ← aquí estaba el error
            )
            .mark_line()
            .encode(x="Fecha:T", y="value:Q", color="variable:N")
        )
        st.altair_chart(line, use_container_width=True)

    st.markdown("### Últimas 150 lecturas")
    st.dataframe(df.tail(150), use_container_width=True)

# ===================== 3. Ingestar lecturas ===============================
if menu == "Ingestar":
    st.header("Nueva lectura / lote")

    tabs = st.tabs(["Individual", "Lote JSON/CSV"])

    # -------- individual form --------
    with tabs[0]:
        with st.form("ingesta_form"):
            cliente_id = st.text_input("Cliente", "CLIENTE1")
            col_d, col_t = st.columns(2)
            fecha_d = col_d.date_input("Fecha", dt.date.today())
            fecha_t = col_t.time_input("Hora", dt.datetime.now().time())
            presion  = st.number_input("Presión", value=0.0)
            temperatura = st.number_input("Temperatura", value=0.0)
            volumen = st.number_input("Volumen", value=0.0)
            submitted = st.form_submit_button("Enviar y predecir")

        if submitted:
            fecha = dt.datetime.combine(fecha_d, fecha_t)
            payload = {
                "rows": [{
                    "Fecha": fecha.isoformat(sep=" ", timespec="minutes"),
                    "Presion": presion,
                    "Temperatura": temperatura,
                    "Volumen": volumen,
                    "cliente": cliente_id
                }]
            }
            r_ing = call_api("/ingest", method="POST", json=payload)
            if r_ing is not None:
                st.success(f"Insertadas {r_ing.json()['inserted']} filas")
                pred_payload = {"cliente_id": cliente_id, "presion": presion,
                                "temperatura": temperatura, "volumen": volumen}
                r_pred = call_api("/predict", method="POST", json=pred_payload)
                if r_pred is not None:
                    st.json(r_pred.json())
                    load_parquet.clear()

    # -------- batch upload --------
    with tabs[1]:
        st.markdown("Sube **JSON** con estructura `/ingest` o un **CSV** con columnas Fecha, Presion, Temperatura, Volumen, cliente.")
        file = st.file_uploader("Archivo", type=["json","csv"])
        if file:
            if file.type == "application/json":
                payload = json.load(file)
                if "rows" not in payload:
                    st.error("JSON debe contener clave 'rows'.")
                else:
                    if st.button("Enviar lote JSON"):
                        r_ing = call_api("/ingest", method="POST", json=payload)
                        if r_ing is not None:
                            st.success(f"Insertadas {r_ing.json()['inserted']} filas")
                            load_parquet.clear()
            elif file.type in ("text/csv","application/vnd.ms-excel"):
                df_csv = pd.read_csv(file)
                rows = df_csv.to_dict(orient="records")
                if st.button("Enviar lote CSV"):
                    r_ing = call_api("/ingest", method="POST", json={"rows": rows})
                    if r_ing is not None:
                        st.success(f"Insertadas {r_ing.json()['inserted']} filas")
                        load_parquet.clear()
