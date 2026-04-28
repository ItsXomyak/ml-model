"""Генератор ключевых графиков ML-проекта.

9 PNG-файлов в plots/ + extended_metrics.json:
    01_price_distribution.png      — гистограмма цены (мультимодальность 4 сегментов)
    02_correlation_heatmap.png     — Pearson корреляции числовых признаков
    03_boxplots_by_category.png    — цена по district / material / renovation
    04_predicted_vs_actual.png     — точность регрессии train + test
    05_residuals.png               — остатки vs прогноз + распределение ошибок
    06_feature_importance.png      — важности признаков (RandomForest)
    07_overfit_curve.png           — train/val R² vs max_depth (иллюстрация overfit)
    08_clusters_2d.png             — 4 кластера в осях area × price/m²
    09_confusion_matrix.png        — латентный сегмент × предсказанный класс

Запуск:
    python make_plots.py

Перед запуском:
    python generate_data.py
    python train.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT      = Path(__file__).parent
DATA_PATH = ROOT / "data" / "apartments_astana.csv"
PRICE_PATH   = ROOT / "models" / "price_model.pkl"
CLUSTER_PATH = ROOT / "models" / "cluster_model.pkl"
METRICS_PATH = ROOT / "metrics.json"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
NUMERIC_FEATURES = ["area", "rooms", "floor", "total_floors", "year_built", "dist_to_center"]
CATEGORICAL_FEATURES = ["district", "material", "renovation"]
TARGET = "price"

CLASS_COLORS = {
    "эконом":   "#3b82f6",
    "комфорт":  "#10b981",
    "бизнес":   "#f59e0b",
    "премиум":  "#ef4444",
}
SEGMENTS = ["эконом", "комфорт", "бизнес", "премиум"]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.family":      "DejaVu Sans",
        "font.size":        10,
    })


def save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ---------- 01. Распределение цены ----------
def plot_01_price_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["price"], bins=60, color="#3b82f6", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Цена (млн ₸)")
    ax.set_ylabel("Количество квартир")
    ax.set_title("Распределение цены — видна мультимодальность 4 латентных сегментов")

    # KDE поверх
    counts, bin_edges = np.histogram(df["price"], bins=200, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed = np.convolve(counts, np.ones(5) / 5, mode="same")
    ax2 = ax.twinx()
    ax2.plot(centers, smoothed, color="#ef4444", lw=2, alpha=0.85, label="Плотность")
    ax2.set_ylabel("Плотность")
    ax2.grid(False)
    ax2.legend(loc="upper right")

    save(fig, "01_price_distribution.png")


# ---------- 02. Pearson correlation ----------
def plot_02_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.grid(False)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Корреляции между числовыми признаками")
    save(fig, "02_correlation_heatmap.png")


# ---------- 03. Boxplots по категориям ----------
def plot_03_boxplots_by_category(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, CATEGORICAL_FEATURES):
        order = df.groupby(col)["price"].median().sort_values().index.tolist()
        data = [df.loc[df[col] == c, "price"].values for c in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3b82f6")
            patch.set_alpha(0.6)
        ax.set_title(f"Цена vs {col}")
        ax.set_ylabel("Цена (млн ₸)")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Распределение цены по категориальным признакам",
                 fontsize=12, y=1.02)
    save(fig, "03_boxplots_by_category.png")


# ---------- 04. Predicted vs Actual ----------
def plot_04_predicted_vs_actual(price_model, X_train, y_train, X_test, y_test) -> None:
    pred_train = price_model.predict(X_train)
    pred_test  = price_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_train, pred_train, alpha=0.3, s=12,
               c="#3b82f6", label=f"Train (n={len(y_train)})")
    ax.scatter(y_test, pred_test, alpha=0.7, s=18,
               c="#ef4444", label=f"Test (n={len(y_test)})")

    lo = min(y_train.min(), y_test.min())
    hi = max(y_train.max(), y_test.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="Идеал y = x")

    r2_train = r2_score(y_train, pred_train)
    r2_test  = r2_score(y_test, pred_test)
    ax.set_xlabel("Реальная цена (млн ₸)")
    ax.set_ylabel("Предсказанная цена (млн ₸)")
    ax.set_title(f"Predicted vs Actual\n"
                 f"R² train = {r2_train:.4f}  |  R² test = {r2_test:.4f}")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    save(fig, "04_predicted_vs_actual.png")


# ---------- 05. Residuals ----------
def plot_05_residuals(price_model, X_test, y_test) -> None:
    pred = price_model.predict(X_test)
    resid = y_test - pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1 = axes[0]
    ax1.scatter(pred, resid, alpha=0.5, s=15, c="#3b82f6")
    ax1.axhline(0, color="red", lw=1.5, linestyle="--")
    ax1.set_xlabel("Предсказание (млн ₸)")
    ax1.set_ylabel("Остаток y_true − y_pred (млн ₸)")
    ax1.set_title("Residuals vs Predicted")

    ax2 = axes[1]
    ax2.hist(resid, bins=40, color="#10b981", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="red", lw=1.5, linestyle="--")
    ax2.set_xlabel("Остаток (млн ₸)")
    ax2.set_ylabel("Частота")
    ax2.set_title(f"Распределение остатков "
                  f"(μ={resid.mean():.2f}, σ={resid.std():.2f})")

    save(fig, "05_residuals.png")


# ---------- 06. Feature importance ----------
def plot_06_feature_importance(price_model) -> None:
    if not hasattr(price_model, "named_steps"):
        return
    model = price_model.named_steps.get("model")
    preproc = price_model.named_steps.get("preproc")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return

    try:
        feature_names = preproc.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(len(importances))]

    idx = np.argsort(importances)[::-1][:15]
    top_imp = importances[idx]
    top_names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_imp))[::-1], top_imp, color="#3b82f6", alpha=0.85)
    ax.set_yticks(range(len(top_imp))[::-1])
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{len(top_imp)} важнейших признаков "
                 f"({type(model).__name__})")
    save(fig, "06_feature_importance.png")


# ---------- 07. Overfit curve по max_depth ----------
def plot_07_overfit_curve(X_train, y_train) -> None:
    pipe = Pipeline([
        ("preproc", _build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    depths = [2, 4, 6, 8, 10, 12, 15, 20, 25, None]
    depth_labels = [d if d is not None else 30 for d in depths]

    train_scores, val_scores = validation_curve(
        pipe, X_train, y_train,
        param_name="model__max_depth",
        param_range=depths,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(depth_labels, train_mean, "o-", color="#3b82f6",
            label="Train R²", lw=2)
    ax.plot(depth_labels, val_mean, "s-", color="#ef4444",
            label="Val R² (3-fold CV)", lw=2)
    ax.fill_between(depth_labels, train_mean, val_mean, alpha=0.1, color="gray",
                     label="Overfit-разрыв")
    ax.set_xlabel("max_depth (None отображается как 30)")
    ax.set_ylabel("R²")
    ax.set_title("Иллюстрация overfit-разрыва по max_depth (Задание 6)")
    ax.legend()
    save(fig, "07_overfit_curve.png")


# ---------- 08. Кластеры в исходных осях ----------
def plot_08_clusters_2d(df: pd.DataFrame, cluster_pipeline,
                         cluster_to_class: dict) -> None:
    df_feat = _build_cluster_features(df)
    cluster_labels = cluster_pipeline.predict(df_feat)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cid in sorted(set(cluster_labels)):
        cls_name = cluster_to_class.get(str(cid), f"кластер {cid}")
        mask = cluster_labels == cid
        ax.scatter(df.loc[mask, "area"], df_feat.loc[mask, "price_per_m2"],
                   s=15, alpha=0.6,
                   color=CLASS_COLORS.get(cls_name, "gray"),
                   label=f"{cls_name} ({mask.sum()})")

    ax.set_xlabel("Площадь, м²")
    ax.set_ylabel("Цена за м² (млн ₸ / м²)")
    ax.set_title("4 кластера квартир в осях площадь × цена за м²")
    ax.legend()
    save(fig, "08_clusters_2d.png")


# ---------- 09. Confusion matrix ----------
def plot_09_confusion_matrix(df_with_seg: pd.DataFrame, cluster_pipeline,
                              cluster_to_class: dict) -> dict:
    df_feat = _build_cluster_features(df_with_seg)
    cluster_labels = cluster_pipeline.predict(df_feat)
    predicted_class = [cluster_to_class.get(str(cid), f"класс {cid}")
                        for cid in cluster_labels]

    ct = pd.crosstab(df_with_seg["segment"], pd.Series(predicted_class, name="predicted"))
    order = [s for s in SEGMENTS if s in ct.index]
    cols  = [s for s in SEGMENTS if s in ct.columns]
    if order and cols:
        ct = ct.loc[order, cols]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_yticks(range(ct.shape[0]))
    ax.set_xticklabels(ct.columns, rotation=20)
    ax.set_yticklabels(ct.index)
    ax.set_xlabel("Предсказанный кластером класс")
    ax.set_ylabel("Истинный (латентный) сегмент")

    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            color = "white" if val > ct.values.max() / 2 else "black"
            ax.text(j, i, val, ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    diag = np.array([ct.loc[s, s] if s in ct.columns and s in ct.index else 0
                      for s in (set(ct.index) & set(ct.columns))])
    purity = diag.sum() / ct.values.sum() if ct.values.sum() > 0 else 0
    ax.set_title(f"Confusion matrix: латентная истина vs k-means\n"
                 f"Cluster Purity = {purity:.3f}")

    fig.colorbar(im, ax=ax, label="Количество квартир")
    ax.grid(False)
    save(fig, "09_confusion_matrix.png")

    return {"purity": float(purity), "matrix": ct.to_dict()}


# ---------- Helpers ----------
def _build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["price_per_m2"] = tmp["price"] / tmp["area"]
    rank = (tmp.groupby("district")["price_per_m2"].mean()
              .rank(method="dense").astype(int))
    return pd.DataFrame({
        "price_per_m2":   tmp["price_per_m2"].values,
        "area":           df["area"].values,
        "dist_to_center": df["dist_to_center"].values,
        "district_rank":  df["district"].map(rank).astype(float).values,
    })


def _build_preprocessor():
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", ohe,               CATEGORICAL_FEATURES),
    ])


# ---------- main ----------
def main() -> None:
    setup_style()
    print("=" * 60)
    print("ГЕНЕРАЦИЯ КЛЮЧЕВЫХ ГРАФИКОВ — старт")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} не найден. Запусти generate_data.py")
    if not PRICE_PATH.exists() or not CLUSTER_PATH.exists():
        raise FileNotFoundError("Модели не найдены. Запусти train.py")

    df = pd.read_csv(DATA_PATH)
    price_model = joblib.load(PRICE_PATH)
    cluster_model = joblib.load(CLUSTER_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cluster_to_class = metrics["clustering"]["cluster_to_class"]

    print("[setup] Регенерирую датасет с латентными сегментами для confusion matrix...")
    from generate_data import generate_dataset
    df_with_seg = generate_dataset(n=len(df), seed=RANDOM_STATE,
                                    include_segment=True)

    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

    print("\n[1/9] Гистограмма цены")
    plot_01_price_distribution(df)

    print("[2/9] Pearson корреляции")
    plot_02_correlation_heatmap(df)

    print("[3/9] Boxplots по категориям")
    plot_03_boxplots_by_category(df)

    print("[4/9] Predicted vs Actual")
    plot_04_predicted_vs_actual(price_model, X_train, y_train, X_test, y_test)

    print("[5/9] Residuals")
    plot_05_residuals(price_model, X_test, y_test)

    print("[6/9] Feature importance")
    plot_06_feature_importance(price_model)

    print("[7/9] Overfit curve по max_depth")
    plot_07_overfit_curve(X_train, y_train)

    print("[8/9] Кластеры в исходных осях")
    plot_08_clusters_2d(df, cluster_model, cluster_to_class)

    print("[9/9] Confusion matrix")
    cm_summary = plot_09_confusion_matrix(df_with_seg, cluster_model, cluster_to_class)

    extended = {"cluster_purity": cm_summary.get("purity")}
    out_path = PLOTS_DIR / "extended_metrics.json"
    out_path.write_text(json.dumps(extended, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\n[save] extended_metrics.json → {out_path}")

    print("=" * 60)
    print(f"ГОТОВО — 9 графиков в {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
