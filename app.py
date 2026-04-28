"""FastAPI-сервер для ML-системы оценки квартир.

- POST /api/predict — предсказание цены и класса по параметрам квартиры.
- GET  /api/health  — health-check.
- GET  /            — раздача index.html (фронт).
- /static/*         — раздача CSS/JS.

Запуск:
    uvicorn app:app --host 0.0.0.0 --port 8000

Перед запуском должен быть выполнен train.py — нужны models/*.pkl и metrics.json.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Принудительный UTF-8 для stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).parent
MODEL_PATH    = ROOT / "models" / "price_model.pkl"
CLUSTER_PATH  = ROOT / "models" / "cluster_model.pkl"
DATA_PATH     = ROOT / "data"   / "apartments_astana.csv"
METRICS_PATH  = ROOT / "metrics.json"
STATIC_DIR    = ROOT / "static"


app = FastAPI(title="Apartments ML API", version="1.0")

_state: Dict[str, Any] = {}


def load_artifacts() -> Dict[str, Any]:
    """Lazy-загрузка моделей и справочника district_rank."""
    if "price_model" in _state:
        return _state

    if not MODEL_PATH.exists() or not CLUSTER_PATH.exists() or not METRICS_PATH.exists():
        raise RuntimeError(
            "Артефакты не найдены. Сначала запусти `python train.py`."
        )

    _state["price_model"]   = joblib.load(MODEL_PATH)
    _state["cluster_model"] = joblib.load(CLUSTER_PATH)

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    _state["cluster_to_class"] = metrics["clustering"]["cluster_to_class"]
    _state["regression_test_metrics"] = metrics["regression"].get("test", {})
    _state["final_model_name"] = metrics["regression"].get("final_model", "?")
    _state["optimal_k"] = metrics["clustering"].get("optimal_k")

    # district_rank — то же кодирование, что в train.py:train_clustering
    df = pd.read_csv(DATA_PATH)
    df["price_per_m2"] = df["price"] / df["area"]
    rank = (
        df.groupby("district")["price_per_m2"].mean()
          .rank(method="dense").astype(int)
    )
    _state["district_rank"] = rank.to_dict()
    return _state


class Apartment(BaseModel):
    """Входные параметры квартиры — диапазоны соответствуют ТЗ §2."""
    area: float           = Field(..., ge=25,   le=200,  description="Площадь, м²")
    rooms: int            = Field(..., ge=1,    le=5,    description="Комнат")
    floor: int            = Field(..., ge=1,    le=25,   description="Этаж")
    total_floors: int     = Field(..., ge=5,    le=25,   description="Этажей в доме")
    year_built: int       = Field(..., ge=1960, le=2025, description="Год постройки")
    district: str         = Field(..., description="Район")
    material: str         = Field(..., description="Материал стен")
    renovation: str       = Field(..., description="Тип ремонта")
    dist_to_center: float = Field(..., ge=0.5,  le=25,   description="До центра, км")


class PredictionResponse(BaseModel):
    price_mln:     float
    price_per_m2:  float
    cluster_id:    int
    class_name:    str
    model_used:    str
    optimal_k:     int


@app.post("/api/predict", response_model=PredictionResponse)
def predict(apt: Apartment) -> PredictionResponse:
    try:
        s = load_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if apt.floor > apt.total_floors:
        raise HTTPException(
            status_code=400,
            detail=f"floor ({apt.floor}) не может быть больше total_floors ({apt.total_floors})",
        )

    apt_df = pd.DataFrame([apt.model_dump()])

    # Регрессия
    price = float(s["price_model"].predict(apt_df)[0])

    # Кластеризация — собираем фичи как в train_clustering
    rank = s["district_rank"].get(apt.district)
    if rank is None:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный район: {apt.district}. "
                   f"Допустимые: {list(s['district_rank'].keys())}",
        )

    cluster_features = pd.DataFrame([{
        "price_per_m2":   price / apt.area,
        "area":           apt.area,
        "dist_to_center": apt.dist_to_center,
        "district_rank":  float(rank),
    }])
    cluster_id = int(s["cluster_model"].predict(cluster_features)[0])
    class_name = s["cluster_to_class"].get(str(cluster_id), f"класс {cluster_id}")

    return PredictionResponse(
        price_mln=round(price, 2),
        price_per_m2=round(price / apt.area, 3),
        cluster_id=cluster_id,
        class_name=class_name,
        model_used=s["final_model_name"],
        optimal_k=s["optimal_k"],
    )


@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Возвращает статус сервера и метрики модели на test."""
    try:
        s = load_artifacts()
        return {
            "status": "ok",
            "model_used": s["final_model_name"],
            "test_metrics": s["regression_test_metrics"],
            "optimal_k": s["optimal_k"],
            "districts": sorted(s["district_rank"].keys()),
        }
    except RuntimeError as e:
        return {"status": "no-models", "detail": str(e)}


# ---------- Раздача статики и фронта ----------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    index_html = STATIC_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=404, detail="index.html не найден")
    return FileResponse(str(index_html))
