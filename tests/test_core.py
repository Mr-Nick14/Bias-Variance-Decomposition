import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from bias_variance_project.core import (
    bias_variance_decomposition,
    generate_synthetic,
    generate_training_sets,
)


def test_synthetic_data_is_reproducible_and_has_expected_shapes():
    x_first, y_first = generate_synthetic(n_samples=24, sigma=0.2, seed=17)
    x_second, y_second = generate_synthetic(n_samples=24, sigma=0.2, seed=17)

    assert x_first.shape == (24, 1)
    assert y_first.shape == (24,)
    np.testing.assert_array_equal(x_first, x_second)
    np.testing.assert_array_equal(y_first, y_second)


def test_training_sets_use_a_reproducible_data_seed_schedule():
    first = generate_training_sets(
        n_repeats=4,
        n_train=20,
        sigma=0.3,
        data_seed=100,
    )
    second = generate_training_sets(
        n_repeats=4,
        n_train=20,
        sigma=0.3,
        data_seed=100,
    )

    assert len(first) == 4
    for (x_first, y_first), (x_second, y_second) in zip(first, second, strict=True):
        np.testing.assert_array_equal(x_first, x_second)
        np.testing.assert_array_equal(y_first, y_second)


def test_monte_carlo_decomposition_matches_squared_error_identity():
    training_sets = generate_training_sets(
        n_repeats=160,
        n_train=80,
        sigma=0.25,
        data_seed=300,
    )
    x_eval = np.linspace(-3.0, 3.0, 120).reshape(-1, 1)

    result, predictions = bias_variance_decomposition(
        lambda _seed: make_pipeline(
            PolynomialFeatures(degree=5, include_bias=False),
            Ridge(alpha=1e-3),
        ),
        training_sets=training_sets,
        x_eval=x_eval,
        sigma=0.25,
        model_seed=800,
        test_noise_seed=1_200,
    )

    assert predictions.shape == (160, 120)
    assert abs(result["decomposition_gap"]) < 0.015
    assert result["expected_mse"] == (
        result["bias2"] + result["variance"] + result["noise"]
    )
    assert np.isfinite(list(result.values())).all()


def test_decomposition_is_reproducible_with_fixed_seed_schedules():
    training_sets = generate_training_sets(
        n_repeats=12,
        n_train=40,
        sigma=0.2,
        data_seed=90,
    )
    x_eval = np.linspace(-2.0, 2.0, 30).reshape(-1, 1)

    def run_once():
        return bias_variance_decomposition(
            lambda seed: Ridge(alpha=0.1, random_state=seed),
            training_sets=training_sets,
            x_eval=x_eval,
            sigma=0.2,
            model_seed=180,
            test_noise_seed=270,
        )

    first_result, first_predictions = run_once()
    second_result, second_predictions = run_once()

    assert first_result == second_result
    np.testing.assert_array_equal(first_predictions, second_predictions)


def test_model_randomness_uses_its_own_seed_schedule():
    seen_seeds = []

    class MeanRegressor:
        def fit(self, _x, y):
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, x):
            return np.full(len(x), self.mean_)

    def make_model(seed):
        seen_seeds.append(seed)
        return MeanRegressor()

    training_sets = generate_training_sets(4, 12, 0.1, data_seed=10)
    bias_variance_decomposition(
        make_model,
        training_sets=training_sets,
        x_eval=np.linspace(-1.0, 1.0, 8),
        sigma=0.1,
        model_seed=700,
        test_noise_seed=900,
    )

    assert seen_seeds == [700, 701, 702, 703]
