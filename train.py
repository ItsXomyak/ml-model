"""Главный обучающий скрипт ML-системы оценки квартир.

Покрывает Задания 2–5 рубежного контроля:
- Задание 2: split 60/20/20 (функция split_data)
- Задание 3: базовая регрессия + MSE/R² (функция train_regression)
- Задание 4: k-means + elbow method (функция train_clustering)
- Задание 5: Lasso/Ridge регуляризация (функция train_regularized)

Запуск: `python train.py`. При отсутствии CSV — данные генерируются автоматически.
"""

import json
import os
import sys
from typing import Tuple, Dict, Any, List

# Принудительный UTF-8 для stdout/stderr (важно для Windows-консоли)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Константы путей и параметров
DATA_PATH    = "data/apartments_astana.csv"
MODELS_DIR   = "models"
PRICE_MODEL  = os.path.join(MODELS_DIR, "price_model.pkl")
CLUSTER_MODEL = os.path.join(MODELS_DIR, "cluster_model.pkl")
METRICS_PATH = "metrics.json"
PLOTS_DIR    = "plots"
ELBOW_PLOT   = os.path.join(PLOTS_DIR, "elbow_plot.png")

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "area", "rooms", "floor", "total_floors",
    "year_built", "dist_to_center"
]
CATEGORICAL_FEATURES = ["district", "material", "renovation"]
TARGET = "price"

CLUSTER_NAMES = ["эконом", "комфорт", "бизнес", "премиум"]


# ---------- Загрузка данных ----------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Загрузка CSV с квартирами. Если файла нет — генерируем."""
    if not os.path.exists(path):
        print(f"[load_data] {path} не найден, запускаю генерацию данных...")
        from generate_data import generate_dataset
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = generate_dataset(n=3000, seed=RANDOM_STATE)
        df.to_csv(path, index=False, encoding="utf-8")
    df = pd.read_csv(path)
    print(f"[load_data] Загружено {len(df)} строк, {df.shape[1]} колонок")
    return df


# ---------- Препроцессинг ----------
def build_preprocessor() -> ColumnTransformer:
    """Сборка препроцессора: StandardScaler для числовых,
    OneHotEncoder для категориальных признаков."""
    # sklearn 1.2+ использует sparse_output, более старые — sparse
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", ohe,               CATEGORICAL_FEATURES),
        ]
    )


# ---------- Задание 2: разделение данных ----------
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                          pd.Series, pd.Series, pd.Series]:
    """Двухступенчатый split 60/20/20 train/val/test."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Шаг 1: 60% train vs 40% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE
    )
    # Шаг 2: 50/50 от temp = по 20% от исходного
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    print(f"[split_data] train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------- Задание 3: базовая регрессия ----------
def _fit_and_score(pipe: Pipeline, X_train, y_train, X_val, y_val) -> Dict[str, float]:
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_val   = pipe.predict(X_val)
    return {
        "mse_train": float(mean_squared_error(y_train, y_pred_train)),
        "r2_train":  float(r2_score(y_train, y_pred_train)),
        "mse_val":   float(mean_squared_error(y_val, y_pred_val)),
        "r2_val":    float(r2_score(y_val, y_pred_val)),
    }


def train_regression(X_train, y_train, X_val, y_val) -> Tuple[Pipeline, Dict[str, Any]]:
    """Базовые модели регрессии (LinearRegression + RandomForestRegressor),
    возврат лучшей по MSE на val. Метрики: MSE и R².
    """
    candidates = [
        ("LinearRegression", Pipeline([
            ("preproc", build_preprocessor()),
            ("model", LinearRegression()),
        ])),
        ("RandomForest(d=10)", Pipeline([
            ("preproc", build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=10,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])),
        ("RandomForest(d=15)", Pipeline([
            ("preproc", build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=15,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])),
    ]

    results = []
    for name, pipe in candidates:
        m = _fit_and_score(pipe, X_train, y_train, X_val, y_val)
        m["name"] = name
        m["pipeline"] = pipe
        results.append(m)
        print(f"[regression] {name} — "
              f"MSE_val={m['mse_val']:.4f}, R²_val={m['r2_val']:.4f} "
              f"(train: MSE={m['mse_train']:.4f}, R²={m['r2_train']:.4f})")

    best = min(results, key=lambda r: r["mse_val"])
    print(f"[regression] Лучшая базовая модель: {best['name']} — "
          f"R²_val={best['r2_val']:.4f}")

    summary = {
        "best_baseline": best["name"],
        "all_runs": [
            {k: v for k, v in r.items() if k != "pipeline"} for r in results
        ],
        "mse_train": best["mse_train"],
        "r2_train":  best["r2_train"],
        "mse_val":   best["mse_val"],
        "r2_val":    best["r2_val"],
    }
    return best["pipeline"], summary


# ---------- Задание 5: Lasso / Ridge ----------
def train_regularized(X_train, y_train, X_val, y_val) -> Tuple[Pipeline, Dict[str, Any]]:
    """Подбор alpha для Lasso и Ridge по val. Возврат лучшей модели."""
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    results = []

    for name, ModelCls in [("Lasso", Lasso), ("Ridge", Ridge)]:
        for alpha in alphas:
            pipe = Pipeline([
                ("preproc", build_preprocessor()),
                ("model", ModelCls(alpha=alpha, random_state=RANDOM_STATE,
                                   max_iter=10000)),
            ])
            pipe.fit(X_train, y_train)
            y_pred_val = pipe.predict(X_val)
            mse = float(mean_squared_error(y_val, y_pred_val))
            r2 = float(r2_score(y_val, y_pred_val))
            results.append({
                "model": name, "alpha": alpha,
                "mse_val": mse, "r2_val": r2,
                "pipeline": pipe,
            })
            print(f"[regularized] {name}(alpha={alpha}) — "
                  f"MSE_val={mse:.4f}, R²_val={r2:.4f}")

    # Лучшая модель — минимум MSE_val
    best = min(results, key=lambda r: r["mse_val"])
    print(f"[regularized] Лучшая: {best['model']}(alpha={best['alpha']}) — "
          f"R²_val={best['r2_val']:.4f}")

    summary = {
        "best_model": best["model"],
        "best_alpha": best["alpha"],
        "best_mse_val": best["mse_val"],
        "best_r2_val":  best["r2_val"],
        "all_runs": [
            {k: v for k, v in r.items() if k != "pipeline"} for r in results
        ],
    }
    return best["pipeline"], summary


# ---------- Задание 4: k-means + elbow ----------
def find_elbow(inertias: List[float]) -> int:
    """Находит точку локтя по максимальному замедлению темпа убывания inertia.

    Алгоритм:
      1. Считаем drops[i] = inertia(K=i) − inertia(K=i+1).
      2. Считаем ratio[i] = drops[i+1] / drops[i] — насколько следующий
         «шаг улучшения» меньше предыдущего.
      3. Первый шаг (K=1 → K=2) почти всегда доминирует, поэтому первое
         отношение пропускаем и ищем минимум среди остальных.
      4. K, ПОСЛЕ которого темп резко падает — оптимальное K локтя.

    Этот подход устойчивее классического «перпендикулярного расстояния»,
    которое смещается к ранним K из-за доминирующего первого падения.
    """
    if len(inertias) < 3:
        return 1

    drops = [inertias[i - 1] - inertias[i] for i in range(1, len(inertias))]

    # Если первый drop = 0 или curve плоская — возврат K=1
    if drops[0] <= 0:
        return 1

    # Ratio[i] = drops[i+1] / drops[i] (i от 0 до len(drops)-2)
    ratios = []
    for i in range(1, len(drops)):
        prev = drops[i - 1]
        ratios.append(drops[i] / prev if prev > 0 else 1.0)

    if len(ratios) < 2:
        # Слишком короткая кривая, fallback на перпендикулярную дистанцию
        return _perpendicular_elbow(inertias)

    # Пропускаем первое отношение (slow-down после dominant K=1→K=2 шага)
    # и ищем минимум среди остальных = точку максимального замедления.
    candidate_ratios = ratios[1:]
    min_idx_in_candidates = int(np.argmin(candidate_ratios))

    # candidate_ratios[m] = ratios[m+1]
    # ratios[i] = drops[i+1] / drops[i]
    # drops[i] = drop K=(i+1) → K=(i+2)
    # При min ratios[i] (i = min_idx_in_candidates+1) последний «большой»
    # drop был drops[i] = drop K=(i+1) → K=(i+2). Значит elbow K = i+2.
    elbow_k = (min_idx_in_candidates + 1) + 2
    return elbow_k


def _perpendicular_elbow(inertias: List[float]) -> int:
    """Резервный elbow finder через перпендикулярную дистанцию от линии."""
    n = len(inertias)
    p1 = np.array([0, inertias[0]], dtype=float)
    p2 = np.array([n - 1, inertias[-1]], dtype=float)
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 1
    distances = []
    for i, val in enumerate(inertias):
        p = np.array([i, val], dtype=float)
        d = abs(line_vec[0] * (p1[1] - p[1]) - (p1[0] - p[0]) * line_vec[1]) / line_len
        distances.append(d)
    return int(np.argmax(distances)) + 1


def save_elbow_plot(K_range: List[int], inertias: List[float],
                    silhouettes: List[float], optimal_k: int,
                    path: str = ELBOW_PLOT) -> None:
    """Сохранение графика elbow + silhouette в PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # без GUI
        import matplotlib.pyplot as plt
    except ImportError:
        print("[clustering] matplotlib не установлен, график не сохранён")
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("Количество кластеров K")
    ax1.set_ylabel("Inertia (WCSS)", color=color1)
    ax1.plot(K_range, inertias, marker="o", color=color1, label="Inertia")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axvline(optimal_k, color="red", linestyle="--",
                label=f"Оптимальное K={optimal_k}")

    ax2 = ax1.twinx()
    color2 = "tab:green"
    ax2.set_ylabel("Silhouette score", color=color2)
    # silhouette не считается для K=1, выравниваем по K_range[1:]
    ax2.plot(K_range[1:], silhouettes, marker="s", color=color2,
             label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Elbow method + Silhouette для k-means кластеризации квартир")
    fig.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[clustering] График elbow сохранён → {path}")


def train_clustering(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    """k-means с elbow method для классификации квартир по классам.

    Признаки: цена за м², площадь, расстояние до центра, район (label-encoding
    по ранговой средней цене за м²). Label-encoding выбран вместо OHE,
    т.к. 5 dummy-колонок добавляли искусственную геометрию в признаковое
    пространство и elbow выходил на K = N районов вместо K = N классов жилья.
    """
    # Закодировать район как 1 числовой признак: rank по средней цене за м²
    df_tmp = df.copy()
    df_tmp["price_per_m2"] = df_tmp["price"] / df_tmp["area"]
    district_rank = (
        df_tmp.groupby("district")["price_per_m2"].mean()
              .rank(method="dense").astype(int)
    )
    district_encoded = df["district"].map(district_rank).astype(float)

    df_feat = pd.DataFrame({
        "price_per_m2":   df["price"] / df["area"],
        "area":           df["area"],
        "dist_to_center": df["dist_to_center"],
        "district_rank":  district_encoded,
    })

    # Масштабирование — обязательно для k-means (зависит от расстояний)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat)

    # Перебор K от 1 до 10
    K_range = list(range(1, 11))
    inertias: List[float] = []
    silhouettes: List[float] = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        if k > 1:
            silhouettes.append(float(silhouette_score(X_scaled, labels)))
        print(f"[clustering] K={k}: inertia={km.inertia_:.2f}"
              + (f", silhouette={silhouettes[-1]:.4f}" if k > 1 else ""))

    optimal_k = find_elbow(inertias)
    print(f"[clustering] Оптимальное K (elbow) = {optimal_k}")

    # Финальная модель с оптимальным K
    final_km = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = final_km.fit_predict(X_scaled)

    # Маппинг кластеров на классы по средней цене за м²
    df_map = df_feat.copy()
    df_map["cluster"] = cluster_labels
    cluster_means = df_map.groupby("cluster")["price_per_m2"].mean().sort_values()
    sorted_clusters = cluster_means.index.tolist()  # от дешёвого к дорогому

    # Если K совпадает с длиной списка — используем именованные классы,
    # иначе обобщённые имена «класс N»
    if optimal_k == len(CLUSTER_NAMES):
        names = CLUSTER_NAMES
    else:
        names = [f"класс {i+1}" for i in range(optimal_k)]

    cluster_to_class = {int(cl): names[i] for i, cl in enumerate(sorted_clusters)}
    print(f"[clustering] Маппинг кластеров: {cluster_to_class}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_elbow_plot(K_range, inertias, silhouettes, optimal_k)

    # Сохраняем pipeline (scaler + KMeans) единым артефактом
    cluster_pipeline = Pipeline([
        ("scaler", scaler),
        ("kmeans", final_km),
    ])

    summary = {
        "K_range":         K_range,
        "inertias":        inertias,
        "silhouettes":     silhouettes,
        "optimal_k":       int(optimal_k),
        "cluster_to_class": {str(k): v for k, v in cluster_to_class.items()},
        "feature_columns": df_feat.columns.tolist(),
    }
    return cluster_pipeline, summary


# ---------- Финальная оценка + сохранение ----------
def evaluate(model: Pipeline, X_test, y_test, label: str = "test") -> Dict[str, float]:
    """Финальные метрики на тесте."""
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2  = float(r2_score(y_test, y_pred))
    print(f"[evaluate] {label}: MSE={mse:.4f}, R²={r2:.4f}")
    return {"mse": mse, "r2": r2}


def save_artifacts(price_model: Pipeline, cluster_model: Pipeline,
                   metrics: Dict[str, Any]) -> None:
    """Сохранение моделей в pkl и метрик в JSON."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(price_model, PRICE_MODEL)
    joblib.dump(cluster_model, CLUSTER_MODEL)
    print(f"[save] Модель регрессии   → {PRICE_MODEL}")
    print(f"[save] Модель кластеризации → {CLUSTER_MODEL}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[save] Метрики → {METRICS_PATH}")


# ---------- main ----------
def main() -> None:
    print("=" * 70)
    print("ML-СИСТЕМА ОЦЕНКИ КВАРТИР АСТАНЫ — старт")
    print("=" * 70)

    # Загрузка
    df = load_data()

    # Шаг 3: split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Шаг 4: базовая регрессия
    base_model, base_metrics = train_regression(X_train, y_train, X_val, y_val)

    # Шаг 5: регуляризация
    reg_model, reg_summary = train_regularized(X_train, y_train, X_val, y_val)

    # Финальный выбор: лучшая по val между baseline и regularized
    if reg_summary["best_mse_val"] < base_metrics["mse_val"]:
        final_model = reg_model
        final_name  = f"{reg_summary['best_model']}(alpha={reg_summary['best_alpha']})"
    else:
        final_model = base_model
        final_name  = base_metrics["best_baseline"]
    print(f"[main] Финальная модель регрессии: {final_name}")

    # Финальная оценка на test
    test_metrics = evaluate(final_model, X_test, y_test, label="test")

    # Шаг 6: кластеризация
    cluster_model, cluster_summary = train_clustering(df)

    # Сборка metrics.json
    metrics = {
        "regression": {
            "baseline":     base_metrics,
            "regularized":  reg_summary,
            "final_model":  final_name,
            "test":         test_metrics,
        },
        "clustering": cluster_summary,
        "config": {
            "random_state":       RANDOM_STATE,
            "rows_total":         int(len(df)),
            "split":              {"train": 0.6, "val": 0.2, "test": 0.2},
            "numeric_features":   NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
    }

    save_artifacts(final_model, cluster_model, metrics)

    print("=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Финальная модель: {final_name}")
    print(f"R² на test: {test_metrics['r2']:.4f}")
    print(f"MSE на test: {test_metrics['mse']:.4f}")
    print(f"Оптимальное K: {cluster_summary['optimal_k']}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
