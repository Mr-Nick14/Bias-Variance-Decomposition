# Bias-Variance Decomposition

Исследовательский ML-проект о том, как сложность регрессионной модели,
размер обучающей выборки и шум меняют bias, variance и итоговую ошибку.

![Bias-variance trade-off](reports/figures/complexity_tradeoff.png)

## Что исследуется

Для squared loss на синтетических данных проверяется разложение

```text
E[(Y - f_hat_D(X))²] = Bias² + Variance + Noise
```

Polynomial Ridge, Decision Tree, KNN, Random Forest и MLP обучаются на одних
и тех же 50 независимых выборках. Оценка Bias² исправлена на Monte Carlo
добавку `variance / n_repeats`. Полосы на графике показывают неопределенность
из-за конечного числа повторов.

На Diabetes сложность меняется отдельно для Ridge, дерева, KNN и MLP. Для MLP
также исследуются ширина, глубина и L2 regularisation. California Housing
используется как второй реальный набор.

## Основные результаты

- Polynomial Ridge степени 7 дает expected MSE 0.1305
- MLP ширины 256 дает 0.1312, но дальнейший рост ширины выходит на плато
- Для дерева глубины 6 рост train size с 40 до 320 снижает variance с 0.134 до 0.050
- CV и Monte Carlo выбирают одинаковую сложность для четырех семейств из пяти
- На Diabetes Ridge получает RMSE 54.86, MLP и Gradient Boosting находятся рядом
- На California Housing MLP получает RMSE 0.570, Random Forest получает 0.612

Широкая однослойная MLP на синтетике не показывает обязательного роста
variance. Заметный компромисс появляется при увеличении глубины и на
многомерном Diabetes. Поэтому выводы в отчете разделяют подтвержденные
траектории и случаи, где ошибка лишь выходит на плато.

## Реальные данные и bootstrap

Истинная regression function для реальных данных неизвестна. Проект не
называет расстояние до другой модели оценкой bias. Вместо этого для общей
prediction matrix проверяется точное squared-loss тождество

```text
mean bootstrap MSE = MSE of mean prediction + bootstrap variance
```

Первая часть включает систематическую ошибку относительно наблюдаемых targets,
test noise и эффект split. Вторая измеряет нестабильность прогноза при
перевыборке. Эти величины не являются классическим разделением Bias² и Noise.

## Структура

```text
research.ipynb
report/
    report.tex
    report.pdf
reports/
    figures/
    tables/
scripts/
    run_experiments.py
    build_report.py
src/bias_variance_project/
    core.py
    experiments.py
    plotting.py
tests/
```

`core.py` показывает прямую реализацию Monte Carlo decomposition.
`experiments.py` содержит отдельные исследовательские вопросы.
`plotting.py` строит графики из рассчитанных таблиц.

## Установка

Нужны Python 3.11 или 3.12, [uv](https://docs.astral.sh/uv/) и XeLaTeX.

```bash
git clone git@github.com:Mr-Nick14/Bias-Variance-Decomposition.git
cd Bias-Variance-Decomposition
uv sync --locked --all-groups
```

California Housing скачивается средствами scikit-learn при первом полном
запуске и сохраняется в `data/scikit_learn`. Каталог cache не добавляется в
Git. В анализ попадает фиксированная выборка из 5000 строк.

## Запуск

Полный пересчет таблиц и графиков

```bash
uv run python scripts/run_experiments.py
```

Быстрая локальная проверка без загрузки California Housing

```bash
uv run python scripts/run_experiments.py --fast
```

Выполнение notebook сверху вниз

```bash
uv run jupyter execute research.ipynb --inplace
```

Сборка PDF из `report/report.tex`

```bash
uv run python scripts/build_report.py
```

Пересчет результатов перед сборкой PDF

```bash
uv run python scripts/build_report.py --refresh
```

Те же действия доступны через `make run`, `make fast`, `make notebook`,
`make report` и `make report-refresh`.

## Воспроизводимость

Train data, model initialization, test noise, bootstrap и CV splits используют
разные seed sequences. Модели сравниваются на общих Monte Carlo samples,
CV folds и bootstrap indices. Масштабирование находится внутри sklearn
Pipeline и не видит validation или test part.

Таблицы содержат время обучения. Оно помогает сравнить вычислительную цену
моделей внутри одного запуска, но меняется между компьютерами.

## Проверки

```bash
uv run pytest
uv run ruff check .
```

Проверяются seed schedules, corrected Bias², приближенное synthetic
decomposition, точное bootstrap identity и создание основных артефактов.

## Ограничения

- Monte Carlo и bootstrap оценки зависят от числа повторов
- Полосы отражают перевыборку prediction vectors, а не полный повтор исследования
- Variance Random Forest и MLP включает случайность обучающего алгоритма
- Оси сложности разных семейств нельзя сравнивать напрямую
- На реальных данных bias и irreducible noise раздельно не наблюдаются
- Wall-clock время зависит от оборудования и фоновой нагрузки
