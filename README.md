# ML-система оценки квартир Астаны

Рубежный контроль по дисциплине ML. Решает две связанные задачи на синтетическом датасете рынка недвижимости г. Астаны:

1. **Регрессия** — предсказание стоимости квартиры по её характеристикам (LinearRegression / Lasso / Ridge).
2. **Кластеризация** — автоматическое определение класса квартиры (эконом / комфорт / бизнес / премиум) методом k-means с автоматическим подбором числа кластеров через elbow method.

Покрывает все 8 заданий рубежного контроля (Блоки 1–4) и упакован в Docker-контейнер.

---

## Структура проекта

```
ml-model/
├── data/
│   └── apartments_astana.csv       # генерируется
├── models/                          # создаётся при запуске
│   ├── price_model.pkl
│   └── cluster_model.pkl
├── static/                          # фронт (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── generate_data.py                 # генератор синтетики
├── train.py                         # главный скрипт (Задания 2,3,4,5)
├── app.py                           # FastAPI-сервер: API + раздача статики
├── theory.md                        # теория (Задания 1, 5, 6)
├── theory_full.docx                 # расширенная теория (всё в одном файле)
├── plots/                           # графики + статистика
│   ├── graphs_explained.md          # ПОДРОБНЫЕ пояснения каждого графика
│   ├── elbow_plot.png               # K=4 (генерируется train.py)
│   ├── 01_price_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_boxplots_by_category.png
│   ├── 04_predicted_vs_actual.png
│   ├── 05_residuals.png
│   ├── 06_feature_importance.png
│   ├── 07_overfit_curve.png
│   ├── 08_clusters_2d.png
│   ├── 09_confusion_matrix.png
│   ├── descriptive_stats.csv
│   ├── categorical_counts.csv
│   ├── correlation_matrix.csv
│   ├── eda_summary.json
│   └── extended_metrics.json
├── eda.py                           # описательный анализ
├── make_plots.py                    # генератор графиков
├── requirements.txt
├── Dockerfile                       # Задание 7
├── README.md                        # Задание 8
└── metrics.json                     # вывод метрик
```

---

## Запуск через Docker (Задание 8)

```bash
docker build -t ml-model-v1 .
docker run --rm ml-model-v1
```

Контейнер при запуске:
1. Сгенерирует синтетический датасет (3000 строк).
2. Разделит данные 60/20/20.
3. Обучит базовую регрессию + Lasso/Ridge с подбором alpha.
4. Запустит k-means + elbow method.
5. Сохранит модели и `metrics.json` (внутри контейнера).
6. Выведет метрики в stdout.

Чтобы получить артефакты на хост:

```bash
docker run --rm -v "$(pwd)/output:/app/output" ml-model-v1 \
  sh -c "python train.py && cp -r models metrics.json elbow_plot.png output/"
```

---

## Локальный запуск (без Docker)

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Генерация датасета (опционально — train.py сделает это сам)
python generate_data.py

# 3. Обучение моделей
python train.py
```

После запуска появятся:
- `data/apartments_astana.csv` — сгенерированный датасет.
- `models/price_model.pkl` — обученная модель регрессии.
- `models/cluster_model.pkl` — обученная модель кластеризации.
- `metrics.json` — все метрики.
- `elbow_plot.png` — график elbow + silhouette.

---

## Описательный анализ + графики

```bash
# Описательная статистика и корреляции (CSV/JSON в plots/)
python eda.py

# 9 ключевых графиков
python make_plots.py
```

**Графики (PNG):**

| Файл | Что показывает |
|------|----------------|
| `plots/elbow_plot.png` | Inertia + Silhouette → выбор K=4 |
| `plots/01_price_distribution.png` | Мультимодальность 4 сегментов |
| `plots/02_correlation_heatmap.png` | Pearson корреляции |
| `plots/03_boxplots_by_category.png` | Цена по district/material/renovation |
| `plots/04_predicted_vs_actual.png` | Точность регрессии (главный) |
| `plots/05_residuals.png` | Анализ остатков |
| `plots/06_feature_importance.png` | Важности признаков из RF |
| `plots/07_overfit_curve.png` | Train/val R² по max_depth (Задание 6) |
| `plots/08_clusters_2d.png` | 4 кластера в осях area × price/m² |
| `plots/09_confusion_matrix.png` | Cluster Purity 0.987 |

**Дополнительно в `plots/`:**

- `descriptive_stats.csv` — describe() + skew/kurtosis/CV/IQR/range
- `categorical_counts.csv` — counts + % для district/material/renovation
- `correlation_matrix.csv` — Pearson + Spearman
- `eda_summary.json` — JSON-сводка EDA
- `extended_metrics.json` — Cluster Purity

Полные пояснения к каждому графику: `plots/graphs_explained.md`.

---

## Веб-интерфейс (FastAPI + статика)

Поверх обученной модели крутится FastAPI-сервер с фронтом — форма ввода параметров квартиры → возвращает цену и класс.

```bash
# 1. Сначала обучи модель (если ещё не сделал)
python train.py

# 2. Запуск сервера
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# 3. Открой в браузере
# http://localhost:8000
```

**Endpoints:**

| Метод | Путь            | Описание                                |
|-------|-----------------|-----------------------------------------|
| GET   | `/`             | Веб-форма (фронт)                       |
| GET   | `/static/*`     | Раздача CSS / JS                        |
| GET   | `/api/health`   | Статус сервера + метрики модели         |
| POST  | `/api/predict`  | Предсказание цены и класса по JSON      |

**Пример запроса к API:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json; charset=utf-8" \
  --data-binary @- <<'EOF'
{
  "area": 75,
  "rooms": 3,
  "floor": 8,
  "total_floors": 16,
  "year_built": 2020,
  "district": "Есиль",
  "material": "монолит",
  "renovation": "евро",
  "dist_to_center": 4.5
}
EOF
```

Ответ:
```json
{
  "price_mln": 128.79,
  "price_per_m2": 1.717,
  "cluster_id": 2,
  "class_name": "премиум",
  "model_used": "RandomForest(d=15)",
  "optimal_k": 4
}
```

**Документация API** (Swagger UI): `http://localhost:8000/docs`

---

## Ожидаемые метрики

При синтетике из 3000 строк и `random_state=42`:

- **R² на test ≥ 0.85** для регрессии.
- **MSE на test** — порядок единиц (млн ₸²).
- **Оптимальное K** — обычно 3–4 (видно по точке локтя).

Запуск дважды должен давать идентичные метрики до 4-го знака (воспроизводимость).

---

## Соответствие заданиям

| Задание | Описание                              | Где реализовано                              |
|---------|---------------------------------------|----------------------------------------------|
| 1       | Теория: val vs test                   | `theory.md`, секция 1                        |
| 2       | Split 60/20/20                        | `train.py`, функция `split_data()`           |
| 3       | Регрессия + MSE и R²                  | `train.py`, функция `train_regression()`     |
| 4       | k-means + elbow method                | `train.py`, функция `train_clustering()`     |
| 5       | Lasso vs Ridge + код                  | `theory.md` + `train.py:train_regularized()` |
| 6       | 3 шага при overfit                    | `theory.md`, секция 3                        |
| 7       | Dockerfile                            | `Dockerfile`                                 |
| 8       | Команды build/run                     | `README.md`                                  |

---

## Зависимости

- Python 3.9+
- pandas, numpy, scikit-learn, joblib, matplotlib

Полный список в `requirements.txt`.
