"""Генератор расширенного набора графиков для ML-проекта.

Производит 15 PNG-файлов в папке plots/:
    01_price_distribution.png      — гистограмма цены + KDE
    02_correlation_heatmap.png     — корреляции числовых признаков
    03_boxplots_by_category.png    — цена по district / renovation / material
    04_predicted_vs_actual.png     — точность регрессии train + test
    05_residuals.png               — остатки vs предсказания
    06_qq_plot.png                 — нормальность остатков
    07_feature_importance.png      — важности признаков из RandomForest
    08_lasso_path.png              — траектории коэффициентов Lasso по alpha
    09_learning_curve.png          — train/val метрики vs размер выборки
    10_validation_curve_alpha.png  — overfit curve по Lasso alpha
    11_validation_curve_depth.png  — overfit curve по max_depth RF
    12_clusters_2d_pca.png         — кластеры в проекции PCA на 2D
    13_clusters_2d_native.png      — кластеры в осях price/m² × area
    14_confusion_matrix.png        — латентный сегмент × предсказанный кластер
    15_per_segment_metrics.png     — R² и MAE по сегментам жилья

Запуск:
    python make_plots.py

Перед запуском должны быть выполнены:
    python generate_data.py
    python train.py
"""

import json
import sys
from pathlib import Path

# UTF-8 для Windows-консоли
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import (
    learning_curve, train_test_split, validation_curve
)
from sklearn.pipeline import Pipeline


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

# Палитра 4 классов (от дешёвого к дорогому)
CLASS_COLORS = {
    "эконом":   "#3b82f6",   # blue
    "комфорт":  "#10b981",   # green
    "бизнес":   "#f59e0b",   # orange
    "премиум":  "#ef4444",   # red
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
def plot_price_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["price"], bins=60, color="#3b82f6", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Цена (млн ₸)")
    ax.set_ylabel("Количество квартир")
    ax.set_title("Распределение цены в датасете (видна мультимодальность 4 сегментов)")

    # Простой KDE через гистограмму (без scipy)
    ax2 = ax.twinx()
    counts, bin_edges = np.histogram(df["price"], bins=200, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Сглаживание скользящим средним
    window = 5
    smoothed = np.convolve(counts, np.ones(window) / window, mode="same")
    ax2.plot(centers, smoothed, color="#ef4444", lw=2, alpha=0.8, label="Плотность")
    ax2.set_ylabel("Плотность")
    ax2.grid(False)
    ax2.legend(loc="upper right")

    save(fig, "01_price_distribution.png")


# ---------- 02. Корреляции числовых признаков ----------
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = NUMERIC_FEATURES + ["price"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.grid(False)

    # Аннотации
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
def plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, CATEGORICAL_FEATURES):
        # Сортируем категории по медианной цене
        order = df.groupby(col)["price"].median().sort_values().index.tolist()
        data = [df.loc[df[col] == c, "price"].values for c in order]
        bp = ax.boxplot(data, labels=order, patch_artist=True)
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
def plot_predicted_vs_actual(price_model, X_train, y_train, X_test, y_test) -> None:
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
def plot_residuals(price_model, X_test, y_test) -> None:
    pred = price_model.predict(X_test)
    resid = y_test - pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(pred, resid, alpha=0.5, s=15, c="#3b82f6")
    ax1.axhline(0, color="red", lw=1.5, linestyle="--")
    ax1.set_xlabel("Предсказание (млн ₸)")
    ax1.set_ylabel("Остаток y_true − y_pred (млн ₸)")
    ax1.set_title("Residuals vs Predicted")

    # Histogram of residuals
    ax2 = axes[1]
    ax2.hist(resid, bins=40, color="#10b981", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="red", lw=1.5, linestyle="--")
    ax2.set_xlabel("Остаток (млн ₸)")
    ax2.set_ylabel("Частота")
    ax2.set_title(f"Распределение остатков (μ={resid.mean():.2f}, σ={resid.std():.2f})")

    save(fig, "05_residuals.png")


# ---------- 06. QQ-plot ----------
def plot_qq(price_model, X_test, y_test) -> None:
    resid = (y_test - price_model.predict(X_test)).values
    resid_sorted = np.sort(resid)
    n = len(resid_sorted)

    # Теоретические квантили нормального распределения
    probs = (np.arange(1, n + 1) - 0.5) / n
    # Approximation of normal inverse CDF (без scipy)
    theoretical = np.sqrt(2) * _erfinv(2 * probs - 1)
    theoretical = theoretical * resid.std() + resid.mean()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(theoretical, resid_sorted, s=12, alpha=0.6, c="#3b82f6")
    lo, hi = theoretical.min(), theoretical.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Нормальное распределение")
    ax.set_xlabel("Теоретические квантили (нормальное)")
    ax.set_ylabel("Реальные квантили остатков")
    ax.set_title("Q-Q plot: проверка нормальности остатков")
    ax.legend()
    save(fig, "06_qq_plot.png")


def _erfinv(x: np.ndarray) -> np.ndarray:
    """Аппроксимация обратной функции ошибок (без scipy)."""
    # Winitzki's approximation
    a = 0.147
    ln = np.log(1 - x ** 2)
    first = 2 / (np.pi * a) + ln / 2
    return np.sign(x) * np.sqrt(np.sqrt(first ** 2 - ln / a) - first)


# ---------- 07. Feature importance ----------
def plot_feature_importance(price_model) -> None:
    # Достаём из Pipeline
    if not hasattr(price_model, "named_steps"):
        print("  ⚠ price_model не Pipeline, пропуск feature_importance")
        return
    model = price_model.named_steps.get("model")
    preproc = price_model.named_steps.get("preproc")

    if not hasattr(model, "feature_importances_"):
        # LinearRegression не имеет feature_importances_; используем |coef_|
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            print("  ⚠ модель не поддерживает feature_importances_")
            return
    else:
        importances = model.feature_importances_

    try:
        feature_names = preproc.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(len(importances))]

    # Сортировка top-15
    idx = np.argsort(importances)[::-1][:15]
    top_imp = importances[idx]
    top_names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_imp))[::-1], top_imp, color="#3b82f6", alpha=0.8)
    ax.set_yticks(range(len(top_imp))[::-1])
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{len(top_imp)} важнейших признаков ({type(model).__name__})")
    save(fig, "07_feature_importance.png")


# ---------- 08. Lasso path ----------
def plot_lasso_path(X_train, y_train, preproc_factory) -> None:
    preproc = preproc_factory()
    X_proc = preproc.fit_transform(X_train)
    feature_names = preproc.get_feature_names_out().tolist()

    alphas = np.logspace(-3, 1.5, 40)
    coefs = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=20000, random_state=RANDOM_STATE)
        lasso.fit(X_proc, y_train)
        coefs.append(lasso.coef_)
    coefs = np.array(coefs)  # (n_alphas, n_features)

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.tab20
    for i, name in enumerate(feature_names):
        ax.plot(alphas, coefs[:, i], lw=1.5, alpha=0.85,
                label=name, color=cmap(i % 20))

    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("alpha (сила регуляризации, log scale)")
    ax.set_ylabel("Значение коэффициента")
    ax.set_title("Lasso path: как коэффициенты зануляются с ростом alpha")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, ncol=1, frameon=False)
    save(fig, "08_lasso_path.png")


# ---------- 09. Learning curve ----------
def plot_learning_curve(price_model_factory, X, y) -> None:
    pipe = price_model_factory()
    train_sizes = np.linspace(0.1, 1.0, 8)

    sizes, train_scores, val_scores = learning_curve(
        pipe, X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(sizes, train_mean, "o-", color="#3b82f6", label="Train R²", lw=2)
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#3b82f6")
    ax.plot(sizes, val_mean, "s-", color="#ef4444", label="Val R² (5-fold CV)", lw=2)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#ef4444")

    ax.set_xlabel("Размер train выборки")
    ax.set_ylabel("R²")
    ax.set_title("Learning curve: как метрики меняются с ростом данных")
    ax.legend()
    save(fig, "09_learning_curve.png")


# ---------- 10. Validation curve по alpha ----------
def plot_validation_curve_alpha(X_train, y_train, preproc_factory) -> None:
    pipe = Pipeline([
        ("preproc", preproc_factory()),
        ("model", Lasso(max_iter=20000, random_state=RANDOM_STATE)),
    ])
    alphas = np.logspace(-3, 1.5, 20)

    train_scores, val_scores = validation_curve(
        pipe, X_train, y_train,
        param_name="model__alpha",
        param_range=alphas,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(alphas, train_mean, "o-", color="#3b82f6", label="Train R²", lw=2)
    ax.plot(alphas, val_mean,   "s-", color="#ef4444", label="Val R² (3-fold CV)", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (Lasso, log scale)")
    ax.set_ylabel("R²")
    ax.set_title("Validation curve: подбор alpha для Lasso")
    ax.legend()
    save(fig, "10_validation_curve_alpha.png")


# ---------- 11. Validation curve по max_depth ----------
def plot_validation_curve_depth(X_train, y_train, preproc_factory) -> None:
    pipe = Pipeline([
        ("preproc", preproc_factory()),
        ("model", RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    depths = [2, 4, 6, 8, 10, 12, 15, 20, 25, None]
    depth_labels = [d if d is not None else 30 for d in depths]  # None=∞

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
    ax.plot(depth_labels, train_mean, "o-", color="#3b82f6", label="Train R²", lw=2)
    ax.plot(depth_labels, val_mean,   "s-", color="#ef4444", label="Val R² (3-fold CV)", lw=2)
    ax.set_xlabel("max_depth (None показано как 30)")
    ax.set_ylabel("R²")
    ax.set_title("Validation curve: иллюстрация overfit-разрыва по max_depth")
    ax.legend()
    save(fig, "11_validation_curve_depth.png")


# ---------- 12. Кластеры в проекции PCA на 2D ----------
def plot_clusters_pca(df: pd.DataFrame, cluster_pipeline,
                      cluster_to_class: dict) -> None:
    df_feat, _ = _build_cluster_features(df)
    scaler = cluster_pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(df_feat)
    cluster_labels = cluster_pipeline.named_steps["kmeans"].predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in sorted(set(cluster_labels)):
        cls_name = cluster_to_class.get(str(cid), f"кластер {cid}")
        mask = cluster_labels == cid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=15, alpha=0.6,
                   color=CLASS_COLORS.get(cls_name, "gray"),
                   label=f"{cls_name} ({mask.sum()})")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% дисперсии)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% дисперсии)")
    ax.set_title("Кластеры квартир в проекции PCA на 2D")
    ax.legend()
    save(fig, "12_clusters_2d_pca.png")


# ---------- 13. Кластеры в осях price/m² × area ----------
def plot_clusters_native(df: pd.DataFrame, cluster_pipeline,
                          cluster_to_class: dict) -> None:
    df_feat, _ = _build_cluster_features(df)
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
    ax.set_title("Кластеры квартир в исходных осях")
    ax.legend()
    save(fig, "13_clusters_2d_native.png")


# ---------- 14. Confusion matrix: латентный сегмент × кластер ----------
def plot_confusion_matrix(df_with_seg: pd.DataFrame, cluster_pipeline,
                           cluster_to_class: dict) -> dict:
    df_feat, _ = _build_cluster_features(df_with_seg)
    cluster_labels = cluster_pipeline.predict(df_feat)
    predicted_class = [cluster_to_class.get(str(cid), f"класс {cid}")
                        for cid in cluster_labels]

    # Cross-tab
    ct = pd.crosstab(df_with_seg["segment"], pd.Series(predicted_class, name="predicted"))
    # Упорядочим строки и колонки в едином порядке от эконом к премиум
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

    # Cell-level annotations
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            color = "white" if val > ct.values.max() / 2 else "black"
            ax.text(j, i, val, ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    # Cluster purity
    diag = np.array([ct.loc[s, s] if s in ct.columns and s in ct.index else 0
                      for s in (set(ct.index) & set(ct.columns))])
    purity = diag.sum() / ct.values.sum() if ct.values.sum() > 0 else 0
    ax.set_title(f"Confusion matrix: истина vs k-means\n"
                 f"Purity (доля совпадений) = {purity:.3f}")

    fig.colorbar(im, ax=ax, label="Количество квартир")
    ax.grid(False)
    save(fig, "14_confusion_matrix.png")

    return {"purity": float(purity), "matrix": ct.to_dict()}


# ---------- 15. Per-segment metrics ----------
def plot_per_segment_metrics(price_model, df_with_seg: pd.DataFrame) -> dict:
    X = df_with_seg.drop(columns=["price", "segment"])
    y = df_with_seg["price"]

    # Используем тот же двухступенчатый split, что в train.py
    X_tr, X_temp, y_tr, y_temp, seg_tr, seg_temp = train_test_split(
        X, y, df_with_seg["segment"], test_size=0.40, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test, seg_val, seg_test = train_test_split(
        X_temp, y_temp, seg_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    pred_test = price_model.predict(X_test)

    rows = []
    for seg in SEGMENTS:
        mask = (seg_test == seg).values
        if mask.sum() < 2:
            continue
        rows.append({
            "segment": seg,
            "n":       int(mask.sum()),
            "r2":      float(r2_score(y_test[mask], pred_test[mask])),
            "mse":     float(mean_squared_error(y_test[mask], pred_test[mask])),
            "mae":     float(mean_absolute_error(y_test[mask], pred_test[mask])),
        })

    if not rows:
        return {}

    df_metrics = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # R² по сегментам
    ax = axes[0]
    colors = [CLASS_COLORS[s] for s in df_metrics["segment"]]
    bars = ax.bar(df_metrics["segment"], df_metrics["r2"], color=colors, alpha=0.85)
    ax.set_ylabel("R²")
    ax.set_title("R² на test по сегментам")
    ax.set_ylim(0, 1.05)
    for b, v in zip(bars, df_metrics["r2"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10)

    # MAE по сегментам
    ax = axes[1]
    bars = ax.bar(df_metrics["segment"], df_metrics["mae"], color=colors, alpha=0.85)
    ax.set_ylabel("MAE (млн ₸)")
    ax.set_title("MAE на test по сегментам")
    for b, v in zip(bars, df_metrics["mae"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=10)

    # Размер выборки по сегментам
    ax = axes[2]
    bars = ax.bar(df_metrics["segment"], df_metrics["n"], color=colors, alpha=0.85)
    ax.set_ylabel("Размер test-выборки")
    ax.set_title("Количество квартир в test по сегментам")
    for b, v in zip(bars, df_metrics["n"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                ha="center", fontsize=10)

    fig.suptitle("Метрики регрессии по сегментам жилья (на test)",
                 fontsize=12, y=1.02)
    save(fig, "15_per_segment_metrics.png")

    return {row["segment"]: row for row in rows}


# ---------- Helpers ----------
def _build_cluster_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Та же подготовка фичей для кластеризации, что в train.py."""
    tmp = df.copy()
    tmp["price_per_m2"] = tmp["price"] / tmp["area"]
    rank = (tmp.groupby("district")["price_per_m2"].mean()
              .rank(method="dense").astype(int))
    out = pd.DataFrame({
        "price_per_m2":   tmp["price_per_m2"].values,
        "area":           df["area"].values,
        "dist_to_center": df["dist_to_center"].values,
        "district_rank":  df["district"].map(rank).astype(float).values,
    })
    return out, rank.to_dict()


def _build_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", ohe,               CATEGORICAL_FEATURES),
    ])


def _build_pipeline_factory(model_class):
    def factory():
        return Pipeline([
            ("preproc", _build_preprocessor()),
            ("model",   model_class(
                n_estimators=200, max_depth=15,
                random_state=RANDOM_STATE, n_jobs=-1,
            ) if model_class is RandomForestRegressor else model_class()),
        ])
    return factory


# ---------- main ----------
def main() -> None:
    setup_style()
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ГРАФИКОВ — старт")
    print("=" * 60)

    # Загрузка
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} не найден. Запусти generate_data.py")
    if not PRICE_PATH.exists() or not CLUSTER_PATH.exists():
        raise FileNotFoundError("Модели не найдены. Запусти train.py")

    df = pd.read_csv(DATA_PATH)
    price_model = joblib.load(PRICE_PATH)
    cluster_model = joblib.load(CLUSTER_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cluster_to_class = metrics["clustering"]["cluster_to_class"]

    # Дополнительный датасет с латентным сегментом для confusion matrix
    print("[setup] Регенерирую датасет с сегментами для оценки качества кластеризации...")
    from generate_data import generate_dataset
    df_with_seg = generate_dataset(n=len(df), seed=RANDOM_STATE,
                                    include_segment=True)

    # Split — тот же что в train.py
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

    # ----- Графики -----
    print("\n[1/15] Гистограмма цены")
    plot_price_distribution(df)

    print("[2/15] Корреляции")
    plot_correlation_heatmap(df)

    print("[3/15] Boxplots по категориям")
    plot_boxplots(df)

    print("[4/15] Predicted vs Actual")
    plot_predicted_vs_actual(price_model, X_train, y_train, X_test, y_test)

    print("[5/15] Residuals")
    plot_residuals(price_model, X_test, y_test)

    print("[6/15] Q-Q plot")
    plot_qq(price_model, X_test, y_test)

    print("[7/15] Feature importance")
    plot_feature_importance(price_model)

    print("[8/15] Lasso path")
    plot_lasso_path(X_train, y_train, _build_preprocessor)

    print("[9/15] Learning curve")
    plot_learning_curve(_build_pipeline_factory(RandomForestRegressor), X, y)

    print("[10/15] Validation curve по alpha")
    plot_validation_curve_alpha(X_train, y_train, _build_preprocessor)

    print("[11/15] Validation curve по max_depth")
    plot_validation_curve_depth(X_train, y_train, _build_preprocessor)

    print("[12/15] Кластеры PCA")
    plot_clusters_pca(df, cluster_model, cluster_to_class)

    print("[13/15] Кластеры в исходных осях")
    plot_clusters_native(df, cluster_model, cluster_to_class)

    print("[14/15] Confusion matrix")
    cm_summary = plot_confusion_matrix(df_with_seg, cluster_model, cluster_to_class)

    print("[15/15] Per-segment metrics")
    seg_summary = plot_per_segment_metrics(price_model, df_with_seg)

    # Сохраняем расширенный summary метрик
    extended = {
        "cluster_purity":     cm_summary.get("purity"),
        "per_segment":        seg_summary,
    }
    out_path = PLOTS_DIR / "extended_metrics.json"
    out_path.write_text(json.dumps(extended, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\n[save] extended_metrics.json → {out_path}")

    print("=" * 60)
    print(f"ГОТОВО — графики в {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
