"""Experiments on Diabetes and California Housing."""

import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .experiments import MODEL_SEED, SPLIT_SEED

BOOTSTRAP_SEED = 4_000


def load_real_datasets(
    *,
    include_california: bool = True,
    data_home: Path | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    diabetes = load_diabetes(as_frame=True)
    datasets = {
        "Diabetes": (
            diabetes.data.copy(),
            diabetes.target.rename("disease_progression"),
        ),
    }
    if include_california:
        california = fetch_california_housing(data_home=data_home, as_frame=True)
        rng = np.random.default_rng(SPLIT_SEED)
        selected = np.sort(rng.choice(len(california.data), size=5_000, replace=False))
        datasets["California Housing"] = (
            california.data.iloc[selected].reset_index(drop=True),
            california.target.iloc[selected].reset_index(drop=True).rename("MedHouseVal"),
        )
    return datasets


def target_correlations(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    correlations = x.corrwith(y).dropna()
    return correlations.loc[correlations.abs().sort_values(ascending=False).index]


def real_data_overview(
    *, include_california: bool = True, data_home: Path | None = None
) -> pd.DataFrame:
    rows = []
    datasets = load_real_datasets(
        include_california=include_california,
        data_home=data_home,
    )
    for name, (x, y) in datasets.items():
        q1, q3 = np.quantile(y, [0.25, 0.75])
        iqr = q3 - q1
        rows.append(
            {
                "dataset": name,
                "rows": len(x),
                "features": x.shape[1],
                "missing_values": int(x.isna().sum().sum() + y.isna().sum()),
                "target_mean": float(y.mean()),
                "target_std": float(y.std()),
                "target_iqr_outliers": int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum()),
            }
        )
    return pd.DataFrame(rows)


def real_model_zoo(random_state: int = MODEL_SEED) -> dict[str, object]:
    return {
        "Dummy": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=7,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.03,
            random_state=random_state,
        ),
        "MLP": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    alpha=1e-3,
                    early_stopping=True,
                    max_iter=800,
                    random_state=random_state,
                ),
            ),
            transformer=StandardScaler(),
        ),
    }


def evaluate_real_datasets(
    *, fast: bool = False, data_home: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    datasets = load_real_datasets(include_california=not fast, data_home=data_home)
    for dataset_name, (x, y) in datasets.items():
        cv = RepeatedKFold(
            n_splits=5,
            n_repeats=1 if fast else 3,
            random_state=SPLIT_SEED,
        )
        splits = list(cv.split(x, y))
        for model_name, model in real_model_zoo().items():
            with warnings.catch_warnings():
                if model_name == "MLP":
                    warnings.simplefilter("ignore", ConvergenceWarning)
                scores = cross_validate(
                    model,
                    x,
                    y,
                    cv=splits,
                    scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                    n_jobs=1,
                )

            for fold, (mse, r2, fit_time) in enumerate(
                zip(
                    -scores["test_mse"],
                    scores["test_r2"],
                    scores["fit_time"],
                    strict=True,
                )
            ):
                raw_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "fold": fold,
                        "rmse": float(np.sqrt(mse)),
                        "r2": float(r2),
                        "fit_time_sec": float(fit_time),
                    }
                )

    raw = pd.DataFrame(raw_rows)
    summary = (
        raw.groupby(["dataset", "model"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            fit_time_sec=("fit_time_sec", "mean"),
        )
        .sort_values(["dataset", "rmse_mean"])
        .reset_index(drop=True)
    )
    return raw, summary


def bootstrap_indices(n_samples: int, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_samples, size=(n_bootstrap, n_samples))


def _fit_bootstrap_predictions(
    base_model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    sample_rows: np.ndarray,
) -> np.ndarray:
    predictions = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for indices in sample_rows:
            model = clone(base_model)
            model.fit(x_train.iloc[indices], y_train.iloc[indices])
            predictions.append(model.predict(x_test))
    return np.asarray(predictions)


def _bootstrap_error_terms(
    prediction_matrix: np.ndarray,
    y_test: pd.Series,
) -> dict[str, float]:
    """Apply the exact squared-loss identity on a fixed test set."""

    mean_prediction = prediction_matrix.mean(axis=0)
    mean_prediction_mse = float(mean_squared_error(y_test, mean_prediction))
    bootstrap_variance = float(np.mean(np.var(prediction_matrix, axis=0, ddof=0)))
    mean_bootstrap_mse = float(np.mean((prediction_matrix - y_test.to_numpy()) ** 2))
    return {
        "mean_prediction_mse": mean_prediction_mse,
        "bootstrap_variance": bootstrap_variance,
        "mean_bootstrap_mse": mean_bootstrap_mse,
        "identity_gap": mean_bootstrap_mse - mean_prediction_mse - bootstrap_variance,
    }


def bootstrap_mse_identity(*, fast: bool = False, data_home: Path | None = None) -> pd.DataFrame:
    n_bootstrap = 10 if fast else 60
    rows = []
    datasets = load_real_datasets(include_california=not fast, data_home=data_home)
    for dataset_number, (dataset_name, (x, y)) in enumerate(datasets.items()):
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=SPLIT_SEED,
        )
        sample_rows = bootstrap_indices(
            len(x_train),
            n_bootstrap,
            BOOTSTRAP_SEED + dataset_number,
        )
        for model_name, base_model in real_model_zoo().items():
            started = perf_counter()
            predictions = _fit_bootstrap_predictions(
                base_model,
                x_train,
                y_train,
                x_test,
                sample_rows,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    **_bootstrap_error_terms(predictions, y_test),
                    "n_bootstrap": n_bootstrap,
                    "fit_time_sec": perf_counter() - started,
                }
            )
    return pd.DataFrame(rows)


def diabetes_complexity_experiment(*, fast: bool = False) -> pd.DataFrame:
    x, y = load_real_datasets(include_california=False)["Diabetes"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=SPLIT_SEED
    )
    n_bootstrap = 6 if fast else 30
    sample_rows = bootstrap_indices(len(x_train), n_bootstrap, BOOTSTRAP_SEED)
    settings = []
    for alpha in [0.1, 10.0, 100.0] if fast else [0.01, 0.1, 1.0, 10.0, 100.0]:
        settings.append(("Ridge", "alpha", alpha, None, None))
    for depth in [1, 4, 8] if fast else [1, 2, 3, 5, 8, 12]:
        settings.append(("Decision Tree", "max_depth", depth, None, None))
    for neighbors in [3, 12, 35] if fast else [2, 5, 10, 20, 35]:
        settings.append(("KNN", "n_neighbors", neighbors, None, None))
    widths = [16, 64] if fast else [16, 64, 128]
    alphas = [1e-3] if fast else [1e-5, 1e-3, 1e-1]
    for alpha in alphas:
        for width in widths:
            settings.append(("MLP", "hidden_width", width, alpha, 2))

    rows = []
    for model_name, complexity_label, complexity, alpha, depth in settings:
        if model_name == "Ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=complexity))
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor(
                max_depth=int(complexity),
                min_samples_leaf=5,
                random_state=MODEL_SEED,
            )
        elif model_name == "KNN":
            model = make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=int(complexity)),
            )
        else:
            model = TransformedTargetRegressor(
                regressor=make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(int(complexity),) * int(depth),
                        activation="relu",
                        alpha=float(alpha),
                        early_stopping=True,
                        max_iter=800,
                        random_state=MODEL_SEED,
                    ),
                ),
                transformer=StandardScaler(),
            )

        started = perf_counter()
        predictions = _fit_bootstrap_predictions(
            model,
            x_train,
            y_train,
            x_test,
            sample_rows,
        )
        rows.append(
            {
                "model": model_name,
                "complexity_label": complexity_label,
                "complexity": complexity,
                "alpha": alpha,
                "depth": depth,
                **_bootstrap_error_terms(predictions, y_test),
                "n_bootstrap": n_bootstrap,
                "fit_time_sec": perf_counter() - started,
            }
        )
    return pd.DataFrame(rows)
