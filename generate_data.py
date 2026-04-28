"""Генератор синтетического датасета квартир г. Астаны.

Создаёт CSV с признаками квартир и целевой переменной (цена).
Внутри генерации заложены 4 латентных сегмента (эконом / комфорт / бизнес /
премиум) — это нужно, чтобы k-means с elbow method уверенно находил K = 4
в задаче кластеризации. Сами метки сегментов в датасет НЕ попадают —
кластеризатор должен их обнаружить сам.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Принудительный UTF-8 для stdout/stderr (важно для Windows-консоли)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ------ Параметры по 4 латентным сегментам ------
# Каждый сегмент = свой профиль распределений признаков и множитель цены.

SEGMENTS = ["эконом", "комфорт", "бизнес", "премиум"]
# Близкие к равномерным веса — чтобы все 4 кластера имели сопоставимую
# плотность точек и k-means чётко находил K = 4 через elbow method.
SEGMENT_WEIGHTS = [0.30, 0.28, 0.25, 0.17]

# Базовая цена за м² (млн ₸) — монотонные шаги для чёткого K=4
SEGMENT_BASE_PPS = {
    "эконом":   0.50,
    "комфорт":  1.00,
    "бизнес":   1.45,
    "премиум":  1.85,
}

# Площадь — непересекающиеся диапазоны с gap. Премиум-диапазон сжат специально
# (110-130), чтобы при базовой цене 1.85 цена не клиппировалась к 250 (max ТЗ).
# Иначе clipping искажает price/m² у премиум вниз и ломает сортировку кластеров.
SEGMENT_AREA = {
    "эконом":   (30, 45),
    "комфорт":  (60, 80),
    "бизнес":   (90, 110),
    "премиум":  (115, 130),
}

# Расстояние до центра — непересекающиеся диапазоны
SEGMENT_DIST = {
    "эконом":   (15, 22),
    "комфорт":  (8, 12),
    "бизнес":   (3, 6),
    "премиум":  (0.7, 2.5),
}

# Распределение районов внутри сегмента
SEGMENT_DISTRICT_WEIGHTS = {
    "эконом":   {"Нура": 0.55, "Сарыарка": 0.40, "Байконыр": 0.05, "Алматы": 0.00, "Есиль": 0.00},
    "комфорт":  {"Нура": 0.05, "Сарыарка": 0.50, "Байконыр": 0.40, "Алматы": 0.05, "Есиль": 0.00},
    "бизнес":   {"Нура": 0.00, "Сарыарка": 0.05, "Байконыр": 0.45, "Алматы": 0.45, "Есиль": 0.05},
    "премиум":  {"Нура": 0.00, "Сарыарка": 0.00, "Байконыр": 0.05, "Алматы": 0.40, "Есиль": 0.55},
}

SEGMENT_RENOVATION_WEIGHTS = {
    "эконом":   {"без ремонта": 0.40, "косметический": 0.45, "евро": 0.13, "дизайнерский": 0.02},
    "комфорт":  {"без ремонта": 0.10, "косметический": 0.40, "евро": 0.45, "дизайнерский": 0.05},
    "бизнес":   {"без ремонта": 0.05, "косметический": 0.20, "евро": 0.55, "дизайнерский": 0.20},
    "премиум":  {"без ремонта": 0.02, "косметический": 0.05, "евро": 0.43, "дизайнерский": 0.50},
}

SEGMENT_MATERIAL_WEIGHTS = {
    "эконом":   {"панель": 0.65, "кирпич": 0.30, "монолит": 0.05},
    "комфорт":  {"панель": 0.30, "кирпич": 0.50, "монолит": 0.20},
    "бизнес":   {"панель": 0.05, "кирпич": 0.40, "монолит": 0.55},
    "премиум":  {"панель": 0.01, "кирпич": 0.20, "монолит": 0.79},
}

# Множители стоимости (внешние, поверх сегментного base_pps).
# Диапазоны намеренно сжаты: широкие множители «размазывали» price/m²
# между сегментами и мешали elbow method чётко находить K = 4.
MATERIAL_MULT = {
    "панель":   0.96,
    "кирпич":   1.00,
    "монолит":  1.05,
}

RENOVATION_MULT = {
    "без ремонта":     0.92,
    "косметический":   0.97,
    "евро":            1.04,
    "дизайнерский":    1.10,
}


def _choice_from_dict(weights: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Случайный выбор ключей словаря согласно его весам."""
    keys = list(weights.keys())
    p = np.array(list(weights.values()), dtype=float)
    p = p / p.sum()
    return rng.choice(keys, size=n, p=p)


def generate_dataset(n: int = 3000, seed: int = 42,
                     include_segment: bool = False) -> pd.DataFrame:
    """Генерация синтетического датасета квартир.

    Пайплайн:
        1. Каждая квартира получает скрытый сегмент (эконом/.../премиум) согласно SEGMENT_WEIGHTS.
        2. Признаки квартиры тянутся из распределений своего сегмента.
        3. Цена считается как функция площади, base_pps сегмента, мультипликаторов
           ремонта/материала, года, расстояния, этажа + гауссовский шум.
        4. Метка сегмента в финальный CSV не попадает — это «истина», скрытая
           от модели; кластеризатор должен восстановить эти 4 группы сам.

    Параметры:
        include_segment: если True, в DataFrame добавляется колонка `segment`
            с латентным классом квартиры (для confusion matrix в make_plots.py).
            Не сохраняй в production CSV — это «утечка».
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # 1. Сегменты
    segments = rng.choice(SEGMENTS, size=n, p=SEGMENT_WEIGHTS)

    # 2. Признаки в зависимости от сегмента
    area = np.empty(n)
    dist_to_center = np.empty(n)
    district = np.empty(n, dtype=object)
    material = np.empty(n, dtype=object)
    renovation = np.empty(n, dtype=object)
    base_pps = np.empty(n)

    for seg in SEGMENTS:
        mask = (segments == seg)
        k = mask.sum()
        if k == 0:
            continue
        area_lo, area_hi = SEGMENT_AREA[seg]
        dist_lo, dist_hi = SEGMENT_DIST[seg]

        area[mask] = rng.uniform(area_lo, area_hi, size=k).round(1)
        dist_to_center[mask] = rng.uniform(dist_lo, dist_hi, size=k).round(2)
        district[mask] = _choice_from_dict(SEGMENT_DISTRICT_WEIGHTS[seg], k, rng)
        material[mask] = _choice_from_dict(SEGMENT_MATERIAL_WEIGHTS[seg], k, rng)
        renovation[mask] = _choice_from_dict(SEGMENT_RENOVATION_WEIGHTS[seg], k, rng)
        base_pps[mask] = SEGMENT_BASE_PPS[seg]

    # Числовые «общие» признаки
    # rooms — полу-независимый от area: базовое количество тянем из площади,
    # затем добавляем рандомный сдвиг ±1, чтобы признак не был детерминирован.
    rooms_base = np.round(area / 30).astype(int)
    rooms_jitter = rng.integers(-1, 2, size=n)  # -1, 0, или +1
    rooms = np.clip(rooms_base + rooms_jitter, 1, 5)
    total_floors = rng.integers(5, 26, size=n)
    floor = np.array([rng.integers(1, tf + 1) for tf in total_floors])
    year_built = rng.integers(1960, 2026, size=n)

    # 3. Множители цены
    mat_mult = np.array([MATERIAL_MULT[m] for m in material])
    ren_mult = np.array([RENOVATION_MULT[r] for r in renovation])
    # Сжатые диапазоны множителей (сохраняют информативность для регрессии,
    # но не размывают сегменты в кластеризационном пространстве).
    year_factor = 0.90 + (year_built - 1960) / (2025 - 1960) * 0.20  # 0.90 → 1.10
    dist_factor = 1.00 - (dist_to_center - 0.5) / (25 - 0.5) * 0.15  # 1.00 → 0.85
    floor_factor = np.where(
        (floor == 1) | (floor == total_floors), 0.97, 1.0
    )

    # rooms_factor: студии (1 комната, area>50) и просторные планировки
    # (мало комнат на большую площадь) — премиальнее. Тесные планировки
    # (много комнат на малую площадь) — дешевле. Эффект ±5%.
    expected_rooms = area / 30
    rooms_deviation = rooms - expected_rooms       # >0 = тесно, <0 = просторно
    rooms_factor = 1.0 - 0.04 * rooms_deviation    # лес поймает signal по rooms

    clean_price = (
        area * base_pps * mat_mult * ren_mult
        * year_factor * dist_factor * floor_factor * rooms_factor
    )

    # Гауссовский шум ~5% (баланс между реалистичностью и разделимостью кластеров)
    noise = rng.normal(loc=1.0, scale=0.05, size=n)
    price = (clean_price * noise).clip(15, 250).round(2)

    df = pd.DataFrame({
        "area":            area,
        "rooms":           rooms,
        "floor":           floor,
        "total_floors":    total_floors,
        "year_built":      year_built,
        "district":        district,
        "material":        material,
        "renovation":      renovation,
        "dist_to_center":  dist_to_center,
        "price":           price,
    })
    if include_segment:
        df["segment"] = segments
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор синтетики квартир Астаны")
    parser.add_argument("--n", type=int, default=3000, help="Количество строк")
    parser.add_argument(
        "--out", type=str, default="data/apartments_astana.csv",
        help="Путь к выходному CSV"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = generate_dataset(n=args.n, seed=42)
    df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"[generate_data] Сгенерировано {len(df)} строк → {args.out}")
    print(f"[generate_data] Цена: min={df['price'].min():.2f}, "
          f"max={df['price'].max():.2f}, mean={df['price'].mean():.2f}")


if __name__ == "__main__":
    main()
