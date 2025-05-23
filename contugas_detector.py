"""
ContugasDetector – solo-pkl, 100 % Python 3.8+
Carga K-Means + IsolationForest y actualiza mapeo cliente→segmento.
"""
import pathlib, json, threading, sys, importlib
from typing import Optional, Union

# ── parche para pickles antiguos que apuntan a “numpy._core” ─────
import numpy as np
try:
    sys.modules.setdefault("numpy._core", importlib.import_module("numpy.core"))
except ModuleNotFoundError:
    pass

import joblib
from sklearn.preprocessing import StandardScaler

# -------- parámetros globales -----------------------------------
SCORE_THRESHOLD = 0.10     # ↑ score (> threshold) ⇒ anómalo (si predict==1)
CACHE_LOCK = threading.Lock()

class ContugasDetector:
    """Carga artefactos y ofrece .predict(...)."""

    def __init__(self, model_dir: Union[str, pathlib.Path]):
        self.dir = pathlib.Path(model_dir)

        # ── artefactos K-Means ──
        self.scaler_km: StandardScaler = joblib.load(self.dir / "scaler_kmeans.pkl")
        self.kmeans                    = joblib.load(self.dir / "kmeans.pkl")

        # mapeo cliente→segmento (JSON)
        self.seg_map_path = self.dir / "cliente_segmento.json"
        self._load_seg_map()

        # caché de {seg: (scaler, iso)}
        self._cache: dict[int, tuple[StandardScaler, object]] = {}

    # ────────────────────────────────────────────────────────────
    def _load_seg_map(self):
        if self.seg_map_path.exists():
            with open(self.seg_map_path) as fp:
                self.seg_map: dict[str, int] = json.load(fp)
        else:
            self.seg_map = {}

    def _save_seg_map(self):
        with open(self.seg_map_path, "w") as fp:
            json.dump(self.seg_map, fp, indent=2)

    # ────────────────────────────────────────────────────────────
    def _load_segment_models(self, seg: int):
        with CACHE_LOCK:
            if seg in self._cache:
                return self._cache[seg]

            scaler = joblib.load(self.dir / f"segment_{seg}_scaler.pkl")
            iso    = joblib.load(self.dir / f"segment_{seg}_iforest.pkl")
            self._cache[seg] = (scaler, iso)
            return scaler, iso

    # ────────────────────────────────────────────────────────────
    def _assign_segment(self, stats_vec: np.ndarray) -> int:
        xs = self.scaler_km.transform(stats_vec)
        return int(self.kmeans.predict(xs)[0])

    # ───────────────────────── PUBLIC API ──────────────────────
    def predict(
        self,
        cliente_id: str,
        pres: float,
        temp: float,
        vol: float,
        stats_vec: Optional[np.ndarray] = None,
    ):
        """
        Devuelve (is_anomaly, score, segmento)
        - Si cliente_id ya está en el JSON → usa ese segmento.
        - Si es nuevo → requiere stats_vec (1×9) para asignarle uno.
        """
        # ── 1. segmento ───────────────────────────────────────
        if cliente_id in self.seg_map:
            seg = self.seg_map[cliente_id]
        else:
            if stats_vec is None:
                raise ValueError("Cliente nuevo: 'stats_vec' (1×9) obligatorio")
            seg = self._assign_segment(stats_vec)
            self.seg_map[cliente_id] = seg
            self._save_seg_map()

        # ── 2. predicción Isolation Forest ────────────────────
        scaler, iso = self._load_segment_models(seg)
        X = scaler.transform([[pres, temp, vol]])

        score      = -iso.decision_function(X)[0]  # ↑ score ⇒ ↑ rareza
        iso_flag   = iso.predict(X)[0] == -1        # True si IF lo marca anómalo
        thresh_flag = score > SCORE_THRESHOLD       # True si supera umbral

        is_anom = iso_flag or thresh_flag
        return bool(is_anom), float(score), seg
