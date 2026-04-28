"""Exploratory Data Analysis (EDA) — описательный и корреляционный анализ датасета.

Производит:
    plots/16_numeric_histograms.png   — гистограммы всех числовых признаков
    plots/17_categorical_counts.png   — частоты категориальных признаков
    plots/18_spearman_correlation.png — Spearman correlation heatmap (нелинейные связи)
    plots/19_scatter_matrix.png       — попарные scatter ключевых признаков с price
    plots/20_outliers_boxplots.png    — boxplots для детекции выбросов
    plots/21_price_vs_features.png    — связь цены с каждым числовым признаком

    plots/descriptive_stats.csv       — описательная статистика (count, mean, std, min, max, квантили)
    plots/categorical_counts.csv      — count + % для категориальных
    plots/correlation_matrix.csv      — Pearson и Spearman корреляции

Запуск:
    python eda.py

Перед запуском: python generate_data.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT      = Path(__file__).parent
DATA_PATH = ROOT / "data" / "apartments_astana.csv"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = ["area", "rooms", "floor", "total_floors", "year_built", "dist_to_center"]
CATEGORICAL_FEATURES = ["district", "material", "renovation"]
TARGET = "price"


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


# ---------- Описательная статистика ----------
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Расширенная describe(): + skewness, kurtosis, missing %."""
    numeric = df.select_dtypes(include=[np.number])
    stats = numeric.describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    stats["skewness"] = numeric.skew()
    stats["kurtosis"] = numeric.kurtosis()
    stats["missing"]  = numeric.isna().sum()
    stats["missing_pct"] = (numeric.isna().sum() / len(numeric) * 100).round(2)
    stats["unique"]    = numeric.nunique()
    stats["range"]     = numeric.max() - numeric.min()
    stats["iqr"]       = numeric.quantile(0.75) - numeric.quantile(0.25)
    stats["cv"]        = (numeric.std() / numeric.mean()).round(3)  # coefficient of variation

    # Округление для читабельности
    stats = stats.round(3)
    return stats


def categorical_stats(df: pd.DataFrame) -> dict:
    """Counts и процентное распределение для категориальных признаков."""
    out = {}
    for col in CATEGORICAL_FEATURES:
        counts = df[col].value_counts()
        pct = (counts / len(df) * 100).round(2)
        result = pd.DataFrame({"count": counts, "percent": pct})
        out[col] = result
    return out


# ---------- 16. Гистограммы числовых признаков ----------
def plot_numeric_histograms(df: pd.DataFrame) -> None:
    n = len(NUMERIC_FEATURES)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_FEATURES):
        ax = axes[i]
        ax.hist(df[col], bins=30, color="#3b82f6", alpha=0.75, edgecolor="white")
        ax.axvline(df[col].mean(), color="#ef4444", lw=2, linestyle="--",
                   label=f"Mean = {df[col].mean():.2f}")
        ax.axvline(df[col].median(), color="#f59e0b", lw=2, linestyle=":",
                   label=f"Median = {df[col].median():.2f}")
        ax.set_xlabel(col)
        ax.set_ylabel("Частота")
        ax.set_title(f"{col} (skew={df[col].skew():.2f})")
        ax.legend(fontsize=8)

    fig.suptitle("Распределения числовых признаков",
                 fontsize=13, fontweight="bold", y=1.00)
    save(fig, "16_numeric_histograms.png")


# ---------- 17. Counts категориальных признаков ----------
def plot_categorical_counts(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col in zip(axes, CATEGORICAL_FEATURES):
        counts = df[col].value_counts().sort_values()
        bars = ax.barh(counts.index, counts.values, color="#10b981", alpha=0.85)
        ax.set_xlabel("Количество квартир")
        ax.set_title(f"{col} (n_unique={df[col].nunique()})")
        # Аннотации с %
        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            pct = val / total * 100
            ax.text(val + total * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val} ({pct:.1f}%)", va="center", fontsize=9)
        ax.set_xlim(0, counts.max() * 1.18)

    fig.suptitle("Распределение категориальных признаков",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "17_categorical_counts.png")


# ---------- 18. Spearman correlation ----------
def plot_spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].corr(method="spearman")

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

    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Spearman correlation (улавливает монотонные нелинейные связи)")
    save(fig, "18_spearman_correlation.png")
    return corr


# ---------- 19. Scatter matrix ----------
def plot_scatter_matrix(df: pd.DataFrame) -> None:
    """Pairplot ключевых признаков (area, dist_to_center, year_built, price)."""
    cols = ["area", "dist_to_center", "year_built", "price"]
    n = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(13, 13))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                # Диагональ — гистограмма
                ax.hist(df[ci], bins=25, color="#3b82f6", alpha=0.7, edgecolor="white")
                ax.set_title(ci, fontsize=10, pad=5)
                ax.tick_params(labelleft=False)
            else:
                # Off-diagonal — scatter с раскраской по price
                sc = ax.scatter(df[cj], df[ci], c=df[TARGET], cmap="viridis",
                                s=4, alpha=0.5)
                ax.tick_params(labelsize=7)

            if i == n - 1:
                ax.set_xlabel(cj, fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ci, fontsize=9)
            else:
                ax.set_yticklabels([])

    fig.suptitle("Scatter matrix ключевых признаков (цвет = цена)",
                 fontsize=13, fontweight="bold", y=0.995)
    save(fig, "19_scatter_matrix.png")


# ---------- 20. Outliers boxplots ----------
def plot_outliers(df: pd.DataFrame) -> dict:
    """Boxplots для всех числовых признаков + price. Подсчёт outliers по IQR."""
    cols = NUMERIC_FEATURES + [TARGET]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    outliers_count = {}
    for i, col in enumerate(cols):
        ax = axes[i]
        bp = ax.boxplot(df[col], vert=True, patch_artist=True,
                        widths=0.5, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "#ef4444",
                                   "markeredgecolor": "white", "markersize": 7})
        for patch in bp["boxes"]:
            patch.set_facecolor("#3b82f6")
            patch.set_alpha(0.65)

        # Подсчёт outliers по IQR-методу
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        outliers_count[col] = int(n_out)

        ax.set_title(f"{col}\nOutliers: {n_out} ({n_out/len(df)*100:.1f}%)",
                     fontsize=10)
        ax.set_xticks([])

    # Скрываем последний пустой subplot
    if len(cols) < len(axes):
        axes[-1].axis("off")

    fig.suptitle("Boxplots для детекции выбросов (IQR метод)",
                 fontsize=13, fontweight="bold", y=1.00)
    save(fig, "20_outliers_boxplots.png")
    return outliers_count


# ---------- 21. Price vs each feature ----------
def plot_price_vs_features(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_FEATURES):
        ax = axes[i]
        ax.scatter(df[col], df[TARGET], alpha=0.4, s=10, color="#3b82f6")

        # Линия тренда (полиномиальная 2-ой степени)
        try:
            z = np.polyfit(df[col], df[TARGET], 2)
            xs = np.linspace(df[col].min(), df[col].max(), 100)
            ax.plot(xs, np.polyval(z, xs), color="#ef4444", lw=2,
                    label=f"Тренд (poly-2)")
        except Exception:
            pass

        # Pearson и Spearman
        pearson = df[col].corr(df[TARGET], method="pearson")
        spearman = df[col].corr(df[TARGET], method="spearman")

        ax.set_xlabel(col)
        ax.set_ylabel("Цена (млн ₸)")
        ax.set_title(f"{col} → price\nPearson: {pearson:.3f} | Spearman: {spearman:.3f}",
                     fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Связь каждого числового признака с целевой переменной",
                 fontsize=13, fontweight="bold", y=1.00)
    save(fig, "21_price_vs_features.png")


# ---------- main ----------
def main() -> None:
    setup_style()
    print("=" * 60)
    print("EDA — описательный и корреляционный анализ")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} не найден. Запусти `python generate_data.py`."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"\n[load] Загружено {len(df)} строк, {df.shape[1]} колонок")
    print(f"[load] Колонки: {list(df.columns)}")
    print(f"[load] Пропусков: {df.isna().sum().sum()} (всего)")
    print(f"[load] Дубликатов: {df.duplicated().sum()}")

    # ----- Описательная статистика -----
    print("\n" + "─" * 60)
    print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА (числовые)")
    print("─" * 60)
    stats = descriptive_stats(df)
    print(stats[["count", "mean", "std", "min", "50%", "max",
                 "skewness", "kurtosis", "cv"]].to_string())
    stats.to_csv(PLOTS_DIR / "descriptive_stats.csv", encoding="utf-8")
    print(f"\n[save] descriptive_stats.csv → {PLOTS_DIR}/descriptive_stats.csv")

    # ----- Категориальные счётчики -----
    print("\n" + "─" * 60)
    print("КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ")
    print("─" * 60)
    cat_stats = categorical_stats(df)
    cat_combined = []
    for col, table in cat_stats.items():
        print(f"\n[{col}]")
        print(table.to_string())
        df_table = table.reset_index().rename(columns={"index": "value"})
        df_table.insert(0, "feature", col)
        cat_combined.append(df_table)
    pd.concat(cat_combined).to_csv(
        PLOTS_DIR / "categorical_counts.csv", index=False, encoding="utf-8"
    )
    print(f"\n[save] categorical_counts.csv")

    # ----- Графики -----
    print("\n" + "─" * 60)
    print("ГРАФИКИ EDA")
    print("─" * 60)

    print("[16/21] Гистограммы числовых")
    plot_numeric_histograms(df)

    print("[17/21] Counts категориальных")
    plot_categorical_counts(df)

    print("[18/21] Spearman correlation")
    spearman = plot_spearman_correlation(df)

    print("[19/21] Scatter matrix")
    plot_scatter_matrix(df)

    print("[20/21] Outliers boxplots")
    outliers = plot_outliers(df)

    print("[21/21] Price vs features")
    plot_price_vs_features(df)

    # ----- Корреляция: Pearson + Spearman в один CSV -----
    cols = NUMERIC_FEATURES + [TARGET]
    pearson = df[cols].corr(method="pearson")
    corr_combined = pd.concat({
        "pearson":  pearson,
        "spearman": spearman,
    }, axis=0)
    corr_combined.to_csv(PLOTS_DIR / "correlation_matrix.csv", encoding="utf-8")
    print(f"\n[save] correlation_matrix.csv (Pearson + Spearman)")

    # ----- Краткое резюме -----
    print("\n" + "=" * 60)
    print("КРАТКОЕ РЕЗЮМЕ EDA")
    print("=" * 60)

    # Топ корреляций с price
    price_corr_p = pearson[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print(f"\nТоп-3 признака по |Pearson| с price:")
    for feat in price_corr_p.head(3).index:
        print(f"  {feat:20s}: Pearson={pearson.loc[feat, TARGET]:+.3f}, "
              f"Spearman={spearman.loc[feat, TARGET]:+.3f}")

    print(f"\nOutliers по IQR (>1.5·IQR от Q1/Q3):")
    for col, n in outliers.items():
        marker = " ⚠" if n > 50 else ""
        print(f"  {col:20s}: {n:4d} ({n/len(df)*100:.1f}%){marker}")

    print(f"\nКатегориальные распределения:")
    for col in CATEGORICAL_FEATURES:
        top = df[col].value_counts().head(1)
        print(f"  {col:12s}: {df[col].nunique()} уникальных, "
              f"топ: {top.index[0]} ({top.values[0]} = {top.values[0]/len(df)*100:.1f}%)")

    # ----- Сохраняем JSON-summary -----
    summary = {
        "n_rows":          int(len(df)),
        "n_columns":       int(df.shape[1]),
        "missing_total":   int(df.isna().sum().sum()),
        "duplicates":      int(df.duplicated().sum()),
        "outliers":        outliers,
        "top_pearson_with_price":  pearson[TARGET].drop(TARGET).to_dict(),
        "top_spearman_with_price": spearman[TARGET].drop(TARGET).to_dict(),
        "categorical_unique": {col: int(df[col].nunique()) for col in CATEGORICAL_FEATURES},
        "price_range":     [float(df[TARGET].min()), float(df[TARGET].max())],
        "price_mean":      float(df[TARGET].mean()),
        "price_median":    float(df[TARGET].median()),
        "price_std":       float(df[TARGET].std()),
    }
    summary_path = PLOTS_DIR / "eda_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[save] eda_summary.json → {summary_path}")

    print("=" * 60)
    print(f"EDA ЗАВЕРШЁН — все артефакты в {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
