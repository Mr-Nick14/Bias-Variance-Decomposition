# Bias-Variance Decomposition

Проект начался с вопроса о том, насколько хорошо учебная картина
bias-variance trade-off видна в обычном численном эксперименте. Интересно
было сравнить несколько разных семейств регрессии на одних и тех же данных,
а затем проверить, как картина меняется с размером выборки и уровнем шума.

![Bias-variance trade-off](reports/figures/complexity_tradeoff.png)

## Идея эксперимента

На синтетических данных известны истинная функция и дисперсия шума, поэтому
для squared loss можно напрямую оценить

```text
E[(Y - f_hat_D(X))²] = Bias² + Variance + Noise
```

Используются 50 независимых обучающих выборок по 160 объектов и общая сетка
оценки, которая не участвует в обучении. Polynomial Ridge, Decision Tree,
KNN, Random Forest и MLP получают одинаковые выборки. Это уменьшает случайный
разброс при сравнении моделей.

Оценка Bias² исправлена на конечное число Monte Carlo повторов. Для реальных
данных классическое разложение недоступно, поэтому там сравниваются repeated
CV scores и bootstrap variance на общем test split.

## Результаты

| Модель | Выбранная сложность | Expected MSE |
| --- | --- | --- |
| Polynomial Ridge | degree 7 | 0.1305 |
| MLP | width 256 | 0.1312 |
| KNN | 12 neighbors | 0.1428 |
| Random Forest | depth 4 | 0.1558 |
| Decision Tree | depth 5 | 0.1802 |

У дерева глубины 6 увеличение train set с 40 до 320 объектов снижает variance
с 0.134 до 0.050. GridSearchCV и Monte Carlo выбирают одинаковую сложность
для четырех семейств из пяти.

Ширина MLP на синтетике приводит к плато, а не к заметному росту variance.
Рост нестабильности появляется при добавлении скрытых слоев и на Diabetes.
На реальных данных Ridge получает RMSE 54.86 на Diabetes, а MLP получает
RMSE 0.570 на выборке California Housing.

## Структура

```text
research.ipynb                  ход исследования и промежуточные наблюдения
src/bias_variance_project/
    core.py                     математическое разложение
    experiments.py              синтетические эксперименты
    real_data.py                CV и bootstrap на реальных данных
    plotting.py                 графики
scripts/                        запуск экспериментов и сборка отчета
reports/                        рассчитанные таблицы и рисунки
report/report.tex               итоговый анализ
```

## Запуск

Нужны Python 3.11 или 3.12, [uv](https://docs.astral.sh/uv/) и XeLaTeX.

```bash
git clone git@github.com:Mr-Nick14/Bias-Variance-Decomposition.git
cd Bias-Variance-Decomposition
uv sync --locked --all-groups
uv run python scripts/run_experiments.py
uv run jupyter execute research.ipynb --inplace
uv run python scripts/build_report.py
```

Первый полный запуск загружает California Housing в локальный каталог
`data/scikit_learn`. Для короткой проверки без загрузки данных есть команда

```bash
uv run python scripts/run_experiments.py --fast
```

Проверки запускаются через `uv run pytest` и `uv run ruff check .`.

### Docker

Для быстрого запуска без локальной настройки Python можно собрать контейнер

```bash
docker build -t bias-variance-lab .
docker run --rm bias-variance-lab
```

По умолчанию выполняется короткий эксперимент. Полный запуск можно передать
как другую команду. Кеш California Housing хранится в отдельном volume, а
новые таблицы и графики записываются в локальный каталог `reports`.

```bash
docker run --rm \
  -v bias-variance-data:/workspace/data \
  -v "$PWD/reports:/workspace/reports" \
  bias-variance-lab \
  .venv/bin/python scripts/run_experiments.py
```

В этом образе нет XeLaTeX и системных шрифтов отчета. PDF собирается локально
через `uv run python scripts/build_report.py`.

## Ограничения

Monte Carlo интервалы отражают конечное число prediction vectors, а не полный
повтор всего исследования. Variance Random Forest и MLP включает случайность
обучающего алгоритма. На реальных данных ошибка среднего прогноза и bootstrap
variance наблюдаются отдельно, но Bias² и irreducible noise разделить нельзя.
