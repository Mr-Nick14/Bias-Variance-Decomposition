import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from bias_variance_project.core import generate_synthetic
from bias_variance_project.experiments import (
    DATA_SEED,
    MODEL_SEED,
    make_synthetic_model,
)
from bias_variance_project.real_data import bootstrap_indices, bootstrap_mse_identity
from scripts.run_experiments import run_experiments


def test_bootstrap_indices_are_reproducible_and_reusable_between_models():
    first = bootstrap_indices(n_samples=18, n_bootstrap=7, seed=51)
    second = bootstrap_indices(n_samples=18, n_bootstrap=7, seed=51)

    assert first.shape == (7, 18)
    np.testing.assert_array_equal(first, second)
    assert ((first >= 0) & (first < 18)).all()


def test_wide_mlp_does_not_stop_after_the_default_short_patience():
    x_train, y_train = generate_synthetic(160, 0.35, DATA_SEED + 2)
    model = make_synthetic_model("MLP", 64, MODEL_SEED + 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)

    fitted_mlp = model.regressor_.named_steps["mlpregressor"]
    assert fitted_mlp.n_iter_ > 60


def test_bootstrap_squared_loss_identity_is_exact_on_fixed_targets():
    results = bootstrap_mse_identity(fast=True)

    assert results["identity_gap"].abs().max() < 1e-10
    np.testing.assert_allclose(
        results["mean_bootstrap_mse"],
        results["mean_prediction_mse"] + results["bootstrap_variance"],
    )


def test_fast_run_creates_finite_tables_and_core_figures(tmp_path):
    run_experiments(tmp_path, fast=True)

    expected_tables = {
        "best_complexity_summary.csv",
        "complexity_results.csv",
        "cv_complexity_selection.csv",
        "diabetes_complexity.csv",
        "mlp_capacity_results.csv",
        "real_bootstrap_identity.csv",
        "real_model_summary.csv",
        "train_size_noise_results.csv",
    }
    expected_figures = {
        "complexity_tradeoff.png",
        "diabetes_complexity.png",
        "mlp_capacity.png",
        "mlp_training_history.png",
        "real_bootstrap_identity.png",
        "real_model_comparison.png",
        "train_size_noise_effects.png",
    }

    for name in expected_tables:
        table = pd.read_csv(tmp_path / "reports" / "tables" / name)
        assert not table.empty
        numeric = table.select_dtypes(include="number").to_numpy()
        assert np.isfinite(numeric[~np.isnan(numeric)]).all()

    for name in expected_figures:
        path = tmp_path / "reports" / "figures" / name
        assert path.exists()
        assert path.stat().st_size > 0
