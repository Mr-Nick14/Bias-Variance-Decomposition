import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from bias_variance_project.core import generate_synthetic
from bias_variance_project.experiments import (
    DATA_SEED,
    MODEL_SEED,
    bootstrap_indices,
    make_synthetic_model,
    target_correlations,
)
from scripts.run_experiments import run_experiments


def test_bootstrap_indices_are_reproducible_and_reusable_between_models():
    first = bootstrap_indices(n_samples=18, n_bootstrap=7, seed=51)
    second = bootstrap_indices(n_samples=18, n_bootstrap=7, seed=51)

    assert first.shape == (7, 18)
    np.testing.assert_array_equal(first, second)
    assert ((first >= 0) & (first < 18)).all()


def test_target_correlations_do_not_include_target_as_a_feature():
    from sklearn.datasets import load_diabetes

    dataset = load_diabetes(as_frame=True)
    correlations = target_correlations(dataset.data, dataset.target)

    assert dataset.target.name not in correlations.index
    assert np.isfinite(correlations.to_numpy()).all()


def test_wide_mlp_does_not_stop_after_the_default_short_patience():
    x_train, y_train = generate_synthetic(160, 0.35, DATA_SEED + 2)
    model = make_synthetic_model("MLP", 64, MODEL_SEED + 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)

    fitted_mlp = model.regressor_.named_steps["mlpregressor"]
    assert fitted_mlp.n_iter_ > 60


def test_fast_run_creates_finite_tables_and_core_figures(tmp_path):
    run_experiments(tmp_path, fast=True)

    expected_tables = {
        "best_complexity_summary.csv",
        "complexity_results.csv",
        "real_model_summary.csv",
        "real_proxy_decomposition.csv",
        "train_size_noise_results.csv",
    }
    expected_figures = {
        "complexity_tradeoff.png",
        "mlp_training_history.png",
        "real_model_comparison.png",
        "real_proxy_decomposition.png",
        "train_size_noise_effects.png",
    }

    for name in expected_tables:
        table = __import__("pandas").read_csv(tmp_path / "reports" / "tables" / name)
        assert not table.empty
        assert np.isfinite(table.select_dtypes(include="number").to_numpy()).all()

    for name in expected_figures:
        path = tmp_path / "reports" / "figures" / name
        assert path.exists()
        assert path.stat().st_size > 0
