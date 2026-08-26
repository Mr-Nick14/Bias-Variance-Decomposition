"""Synthetic and real-data experiments used in the project."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes, load_linnerud
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .core import (
    bias_variance_decomposition,
    generate_synthetic,
    generate_training_sets,
)

DATA_SEED = 1_000
MODEL_SEED = 2_000
TEST_NOISE_SEED = 3_000
BOOTSTRAP_SEED = 4_000
SPLIT_SEED = 5_000
X_DOMAIN = (-3.0, 3.0)


def make_synthetic_model(name: str, complexity: int, random_state: int):
    """Build one model used in the synthetic complexity study."""

    if name == "Polynomial Ridge":
        return make_pipeline(
            PolynomialFeatures(degree=complexity, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1e-3),
        )
    if name == "Decision Tree":
        return DecisionTreeRegressor(max_depth=complexity, random_state=random_state)
    if name == "KNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=complexity),
        )
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=60,
            max_depth=complexity,
            max_features=1.0,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "MLP":
        return TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(complexity,),
                    activation="tanh",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=3e-3,
                    max_iter=800,
                    n_iter_no_change=60,
                    tol=1e-5,
                    random_state=random_state,
                ),
            ),
            transformer=StandardScaler(),
        )
    raise ValueError(f"Unknown model name {name!r}")


def complexity_grids(fast: bool = False) -> dict[str, tuple[str, list[int]]]:
    if fast:
        return {
            "Polynomial Ridge": ("degree", [1, 5, 8]),
            "Decision Tree": ("max_depth", [1, 4, 8]),
            "KNN": ("n_neighbors", [1, 7, 35]),
            "Random Forest": ("max_depth", [1, 4, 10]),
            "MLP": ("hidden_width", [4, 16, 32]),
        }
    return {
        "Polynomial Ridge": ("degree", [1, 2, 3, 4, 5, 7, 9, 12]),
        "Decision Tree": ("max_depth", [1, 2, 3, 4, 5, 7, 10, 14]),
        "KNN": ("n_neighbors", [1, 2, 3, 5, 8, 12, 20, 35, 60]),
        "Random Forest": ("max_depth", [1, 2, 3, 4, 6, 8, 12]),
        "MLP": ("hidden_width", [2, 4, 8, 16, 32, 64]),
    }


def run_complexity_experiment(
    *,
    fast: bool = False,
    n_train: int = 160,
    sigma: float = 0.35,
    n_repeats: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Compare model complexity on paired Monte Carlo training samples."""

    n_repeats = n_repeats or (8 if fast else 50)
    x_eval = np.linspace(*X_DOMAIN, 240).reshape(-1, 1)
    training_sets = generate_training_sets(n_repeats, n_train, sigma, DATA_SEED)
    rows = []
    representative_predictions = {}

    for model_name, (complexity_label, values) in complexity_grids(fast).items():
        model_rows = []
        predictions_by_value = {}
        for value in values:
            def make_model(seed, name=model_name, level=value):
                return make_synthetic_model(name, level, seed)

            if model_name == "MLP":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    result, predictions = bias_variance_decomposition(
                        make_model,
                        training_sets=training_sets,
                        x_eval=x_eval,
                        sigma=sigma,
                        model_seed=MODEL_SEED,
                        test_noise_seed=TEST_NOISE_SEED,
                    )
            else:
                result, predictions = bias_variance_decomposition(
                    make_model,
                    training_sets=training_sets,
                    x_eval=x_eval,
                    sigma=sigma,
                    model_seed=MODEL_SEED,
                    test_noise_seed=TEST_NOISE_SEED,
                )

            row = {
                "model": model_name,
                "complexity_label": complexity_label,
                "complexity": value,
                "n_train": n_train,
                "sigma": sigma,
                "n_repeats": n_repeats,
                **result,
            }
            rows.append(row)
            model_rows.append(row)
            predictions_by_value[value] = predictions

        best = min(model_rows, key=lambda row: row["expected_mse"])
        representative_predictions[model_name] = predictions_by_value[best["complexity"]]

    return pd.DataFrame(rows), representative_predictions, x_eval


def best_complexity_summary(results: pd.DataFrame) -> pd.DataFrame:
    indices = results.groupby("model")["expected_mse"].idxmin()
    columns = [
        "model",
        "complexity_label",
        "complexity",
        "bias2",
        "variance",
        "noise",
        "expected_mse",
        "empirical_mse",
        "decomposition_gap",
    ]
    return results.loc[indices, columns].sort_values("expected_mse").reset_index(drop=True)


def tune_classical_models(*, fast: bool = False) -> pd.DataFrame:
    """Tune four classical models on one synthetic sample with paired CV folds."""

    x_train, y_train = generate_synthetic(
        n_samples=420 if fast else 700,
        sigma=0.35,
        seed=DATA_SEED,
    )
    searches = {
        "Polynomial Ridge": (
            make_pipeline(PolynomialFeatures(), StandardScaler(), Ridge()),
            {
                "polynomialfeatures__degree": [1, 2, 3, 5, 7],
                "ridge__alpha": [1e-3, 1e-2, 1e-1, 1.0],
            },
        ),
        "KNN": (
            make_pipeline(StandardScaler(), KNeighborsRegressor()),
            {"kneighborsregressor__n_neighbors": [2, 3, 5, 8, 12, 20]},
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=MODEL_SEED),
            {"max_depth": [2, 3, 4, 6, 8, 12], "min_samples_leaf": [1, 3, 6]},
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=50 if fast else 140,
                n_jobs=-1,
                random_state=MODEL_SEED,
            ),
            {"max_depth": [2, 4, 7, 12], "min_samples_leaf": [1, 3, 6]},
        ),
    }
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED).split(x_train))
    rows = []
    for name, (estimator, param_grid) in searches.items():
        search = GridSearchCV(
            estimator,
            param_grid,
            scoring="neg_mean_squared_error",
            cv=folds,
            n_jobs=1,
        )
        search.fit(x_train, y_train)
        rows.append(
            {
                "model": name,
                "cv_folds": 5,
                "best_cv_rmse": float(np.sqrt(-search.best_score_)),
                "best_params": str(search.best_params_),
            }
        )
    return pd.DataFrame(rows).sort_values("best_cv_rmse").reset_index(drop=True)


def mlp_training_history(*, epochs: int = 180) -> pd.DataFrame:
    """Record train and validation metrics after each MLP epoch."""

    x, y = generate_synthetic(500, 0.35, DATA_SEED)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=SPLIT_SEED,
    )
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    x_train_scaled = x_scaler.transform(x_train)
    x_val_scaled = x_scaler.transform(x_val)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="tanh",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=3e-3,
        max_iter=1,
        warm_start=True,
        random_state=MODEL_SEED,
    )

    rows = []
    with warnings.catch_warnings():
        # One iteration per fit is intentional because each fit represents one epoch.
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, epochs + 1):
            model.fit(x_train_scaled, y_train_scaled)
            train_prediction = y_scaler.inverse_transform(
                model.predict(x_train_scaled).reshape(-1, 1)
            ).reshape(-1)
            val_prediction = y_scaler.inverse_transform(
                model.predict(x_val_scaled).reshape(-1, 1)
            ).reshape(-1)
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(model.loss_),
                    "train_rmse": float(mean_squared_error(y_train, train_prediction) ** 0.5),
                    "val_rmse": float(mean_squared_error(y_val, val_prediction) ** 0.5),
                }
            )
    return pd.DataFrame(rows)


def load_real_datasets() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Load the two offline regression datasets used in the study."""

    diabetes = load_diabetes(as_frame=True)
    linnerud = load_linnerud(as_frame=True)
    return {
        "Diabetes": (
            diabetes.data.copy(),
            diabetes.target.rename("disease_progression"),
        ),
        "Linnerud-Weight": (
            linnerud.data.copy(),
            linnerud.target["Weight"].rename("Weight"),
        ),
    }


def target_correlations(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Return feature correlations with the target, excluding the target itself."""

    features = x.drop(columns=[y.name], errors="ignore")
    correlations = features.corrwith(y).dropna()
    return correlations.loc[correlations.abs().sort_values(ascending=False).index]


def real_data_overview() -> pd.DataFrame:
    rows = []
    for name, (x, y) in load_real_datasets().items():
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
                "target_iqr_outliers": int(
                    ((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def real_model_zoo(random_state: int = MODEL_SEED) -> dict[str, object]:
    """Return fixed model configurations used on both real datasets."""

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


def evaluate_real_datasets(*, fast: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed configurations on identical repeated CV splits."""

    raw_rows = []
    for dataset_name, (x, y) in load_real_datasets().items():
        cv = RepeatedKFold(
            n_splits=5,
            n_repeats=1 if fast else 3,
            random_state=SPLIT_SEED,
        )
        splits = list(cv.split(x, y))
        for model_name, model in real_model_zoo().items():
            if model_name == "MLP":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    scores = cross_validate(
                        model,
                        x,
                        y,
                        cv=splits,
                        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                        n_jobs=1,
                    )
            else:
                scores = cross_validate(
                    model,
                    x,
                    y,
                    cv=splits,
                    scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                    n_jobs=1,
                )
            for fold, (mse, r2) in enumerate(
                zip(-scores["test_mse"], scores["test_r2"], strict=True)
            ):
                raw_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "fold": fold,
                        "rmse": float(np.sqrt(mse)),
                        "r2": float(r2),
                    }
                )

    raw = pd.DataFrame(raw_rows)
    summary = (
        raw.groupby(["dataset", "model"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
        )
        .sort_values(["dataset", "rmse_mean"])
        .reset_index(drop=True)
    )
    return raw, summary


def bootstrap_indices(n_samples: int, n_bootstrap: int, seed: int) -> np.ndarray:
    """Generate bootstrap rows that can be reused for every compared model."""

    if n_samples < 2 or n_bootstrap < 2:
        raise ValueError("n_samples and n_bootstrap must be at least 2")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_samples, size=(n_bootstrap, n_samples))


def bootstrap_proxy_decomposition(*, fast: bool = False) -> pd.DataFrame:
    """Estimate paired bootstrap variance and reference-model proxy bias."""

    n_bootstrap = 10 if fast else 60
    rows = []
    for dataset_number, (dataset_name, (x, y)) in enumerate(
        load_real_datasets().items()
    ):
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
        reference = GradientBoostingRegressor(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            random_state=MODEL_SEED,
        ).fit(x_train, y_train)
        reference_prediction = reference.predict(x_test)
        residual_noise_proxy = mean_squared_error(y_test, reference_prediction)

        for model_name, base_model in real_model_zoo().items():
            predictions = []
            for indices in sample_rows:
                model = clone(base_model)
                if model_name == "MLP":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        model.fit(x_train.iloc[indices], y_train.iloc[indices])
                else:
                    model.fit(x_train.iloc[indices], y_train.iloc[indices])
                predictions.append(model.predict(x_test))

            prediction_matrix = np.asarray(predictions)
            mean_prediction = prediction_matrix.mean(axis=0)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "proxy_bias2": float(
                        np.mean((mean_prediction - reference_prediction) ** 2)
                    ),
                    "bootstrap_variance": float(
                        np.mean(np.var(prediction_matrix, axis=0, ddof=0))
                    ),
                    "residual_noise_proxy": float(residual_noise_proxy),
                    "test_mse": float(mean_squared_error(y_test, mean_prediction)),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def run_train_size_and_noise_experiments(*, fast: bool = False) -> pd.DataFrame:
    """Measure how training size and observation noise affect one tree."""

    x_eval = np.linspace(*X_DOMAIN, 200).reshape(-1, 1)
    repeats = 10 if fast else 40
    rows = []
    for n_train in ([40, 80, 160, 320] if not fast else [40, 160, 320]):
        training_sets = generate_training_sets(repeats, n_train, 0.35, DATA_SEED)
        result, _ = bias_variance_decomposition(
            lambda seed: make_synthetic_model("Decision Tree", 6, seed),
            training_sets=training_sets,
            x_eval=x_eval,
            sigma=0.35,
            model_seed=MODEL_SEED,
            test_noise_seed=TEST_NOISE_SEED,
        )
        rows.append(
            {
                "study": "sample_size",
                "value": n_train,
                "value_label": "n_train",
                **result,
            }
        )

    for sigma in ([0.1, 0.2, 0.35, 0.6, 0.9] if not fast else [0.1, 0.35, 0.9]):
        training_sets = generate_training_sets(repeats, 160, sigma, DATA_SEED)
        result, _ = bias_variance_decomposition(
            lambda seed: make_synthetic_model("Decision Tree", 6, seed),
            training_sets=training_sets,
            x_eval=x_eval,
            sigma=sigma,
            model_seed=MODEL_SEED,
            test_noise_seed=TEST_NOISE_SEED,
        )
        rows.append(
            {
                "study": "noise",
                "value": sigma,
                "value_label": "sigma",
                **result,
            }
        )
    return pd.DataFrame(rows)
