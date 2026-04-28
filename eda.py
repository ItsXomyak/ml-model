"""Exploratory Data Analysis — описательная и корреляционная статистика.

Производит CSV/JSON в plots/ (без графиков — основные графики делает make_plots.py):
    plots/descriptive_stats.csv   — count, mean, std, min/max, квантили,
                                    skewness, kurtosis, IQR, CV, range
    plots/categorical_counts.csv  — count + percent для категориальных
    plots/correlation_matrix.csv  — Pearson + Spearman корреляции
    plots/eda_summary.json        — краткая JSON-сводка

В stdout выводит:
    - размеры датасета, пропуски, дубликаты
    - таблицу описательной статистики
    - распределение категориальных
    - топ корреляций с price
    - подсчёт outliers по IQR

Запуск:
    python eda.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd


ROOT      = Path(__file__).parent
DATA_PATH = ROOT / "data" / "apartments_astana.csv"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = ["area", "rooms", "floor", "total_floors", "year_built", "dist_to_center"]
CATEGORICAL_FEATURES = ["district", "material", "renovation"]
TARGET = "price"


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    stats = numeric.describe().T
    stats["skewness"]    = numeric.skew()
    stats["kurtosis"]    = numeric.kurtosis()
    stats["missing"]     = numeric.isna().sum()
    stats["missing_pct"] = (numeric.isna().sum() / len(numeric) * 100).round(2)
    stats["unique"]      = numeric.nunique()
    stats["range"]       = numeric.max() - numeric.min()
    stats["iqr"]         = numeric.quantile(0.75) - numeric.quantile(0.25)
    stats["cv"]          = (numeric.std() / numeric.mean()).round(3)
    return stats.round(3)


def categorical_stats(df: pd.DataFrame) -> dict:
    out = {}
    for col in CATEGORICAL_FEATURES:
        counts = df[col].value_counts()
        pct = (counts / len(df) * 100).round(2)
        out[col] = pd.DataFrame({"count": counts, "percent": pct})
    return out


def count_outliers_iqr(df: pd.DataFrame, columns: list) -> dict:
    """Подсчёт выбросов по правилу IQR (>1.5·IQR от Q1/Q3)."""
    out = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        out[col] = int(n)
    return out


def main() -> None:
    print("=" * 60)
    print("EDA — описательный и корреляционный анализ")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} не найден. Запусти `python generate_data.py`."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"\n[load] Строк:        {len(df)}")
    print(f"[load] Колонок:      {df.shape[1]} ({list(df.columns)})")
    print(f"[load] Пропусков:    {df.isna().sum().sum()}")
    print(f"[load] Дубликатов:   {df.duplicated().sum()}")

    # ----- Описательная статистика -----
    print("\n" + "─" * 60)
    print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА (числовые)")
    print("─" * 60)
    stats = descriptive_stats(df)
    print(stats[["count", "mean", "std", "min", "50%", "max",
                 "skewness", "kurtosis", "cv"]].to_string())
    stats.to_csv(PLOTS_DIR / "descriptive_stats.csv", encoding="utf-8")
    print(f"\n[save] {PLOTS_DIR}/descriptive_stats.csv")

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
    print(f"\n[save] {PLOTS_DIR}/categorical_counts.csv")

    # ----- Корреляции -----
    cols = NUMERIC_FEATURES + [TARGET]
    pearson  = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")
    pd.concat({"pearson": pearson, "spearman": spearman}, axis=0).to_csv(
        PLOTS_DIR / "correlation_matrix.csv", encoding="utf-8"
    )
    print(f"[save] {PLOTS_DIR}/correlation_matrix.csv (Pearson + Spearman)")

    # ----- Outliers -----
    outliers = count_outliers_iqr(df, NUMERIC_FEATURES + [TARGET])

    # ----- Резюме -----
    print("\n" + "=" * 60)
    print("РЕЗЮМЕ")
    print("=" * 60)

    price_corr_p = pearson[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print(f"\nТоп-3 драйвера цены (по |Pearson|):")
    for feat in price_corr_p.head(3).index:
        print(f"  {feat:20s}  Pearson={pearson.loc[feat, TARGET]:+.3f}  "
              f"Spearman={spearman.loc[feat, TARGET]:+.3f}")

    print(f"\nOutliers по IQR:")
    for col, n in outliers.items():
        marker = " ⚠" if n > 50 else ""
        print(f"  {col:20s}  {n:4d}  ({n/len(df)*100:.1f}%){marker}")

    print(f"\nКатегориальные:")
    for col in CATEGORICAL_FEATURES:
        top = df[col].value_counts().head(1)
        print(f"  {col:12s}  {df[col].nunique()} уникальных, "
              f"топ: {top.index[0]} ({top.values[0]/len(df)*100:.1f}%)")

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
    print(f"\n[save] {summary_path}")

    print("=" * 60)
    print("EDA ЗАВЕРШЁН")
    print("=" * 60)


if __name__ == "__main__":
    main()
