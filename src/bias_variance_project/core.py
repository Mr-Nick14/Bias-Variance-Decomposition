"""Mathematical core of the synthetic bias-variance experiments."""

from collections.abc import Callable

import numpy as np


def true_function(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sin(2.0 * x) + 0.2 * x**2


def generate_synthetic(
    n_samples: int,
    sigma: float,
    seed: int,
    x_domain: tuple[float, float] = (-3.0, 3.0),
) -> tuple[np.ndarray, np.ndarray]:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_domain, size=n_samples)
    y = true_function(x) + rng.normal(0.0, sigma, size=n_samples)
    return x.reshape(-1, 1), y


def generate_training_sets(
    n_repeats: int,
    n_train: int,
    sigma: float,
    data_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [generate_synthetic(n_train, sigma, data_seed + repeat) for repeat in range(n_repeats)]


def bias_variance_decomposition(
    make_model: Callable[[int], object],
    *,
    training_sets: list[tuple[np.ndarray, np.ndarray]],
    x_eval: np.ndarray,
    sigma: float,
    model_seed: int,
    test_noise_seed: int,
    interval_seed: int = 4_000,
    n_interval_resamples: int = 300,
) -> tuple[dict[str, float], np.ndarray]:
    """Estimate the squared-loss decomposition on one fixed evaluation grid.

    The squared bias is corrected for the finite-repeat Monte Carlo term. If
    ``m`` prediction vectors are averaged, the naive squared distance between
    their mean and the true function contains approximately ``variance / m``.
    """

    if len(training_sets) < 2:
        raise ValueError("training_sets must contain at least two samples")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    x_eval = np.asarray(x_eval, dtype=float).reshape(-1, 1)
    f_true = true_function(x_eval[:, 0])
    predictions = np.empty((len(training_sets), len(x_eval)), dtype=float)
    noisy_targets = np.empty_like(predictions)
    test_rng = np.random.default_rng(test_noise_seed)

    for repeat, (x_train, y_train) in enumerate(training_sets):
        # A separate model seed keeps algorithm randomness independent of the
        # training sample that happens to have the same repeat number.
        model = make_model(model_seed + repeat)
        model.fit(x_train, y_train)
        predictions[repeat] = np.asarray(model.predict(x_eval)).reshape(-1)

        # Empirical MSE gets fresh noise instead of reusing noise seen in fit.
        noisy_targets[repeat] = f_true + test_rng.normal(0.0, sigma, len(x_eval))

    mean_prediction = predictions.mean(axis=0)
    raw_bias2 = float(np.mean((mean_prediction - f_true) ** 2))
    variance = float(np.mean(np.var(predictions, axis=0, ddof=1)))
    bias2_mc_correction = variance / len(training_sets)
    bias2 = raw_bias2 - bias2_mc_correction
    noise = float(sigma**2)
    expected_mse = bias2 + variance + noise
    empirical_mse = float(np.mean((noisy_targets - predictions) ** 2))

    interval_rng = np.random.default_rng(interval_seed)
    interval_rows = np.empty((n_interval_resamples, 2), dtype=float)
    for sample_number in range(n_interval_resamples):
        indices = interval_rng.integers(0, len(predictions), size=len(predictions))
        sample = predictions[indices]
        sample_variance = float(np.mean(np.var(sample, axis=0, ddof=1)))
        sample_raw_bias2 = float(np.mean((sample.mean(axis=0) - f_true) ** 2))
        interval_rows[sample_number] = (
            sample_raw_bias2 - sample_variance / len(sample),
            sample_variance,
        )

    bias2_se, variance_se = interval_rows.std(axis=0, ddof=1)
    bias2_low = bias2 - 1.96 * bias2_se
    bias2_high = bias2 + 1.96 * bias2_se
    variance_low = variance - 1.96 * variance_se
    variance_high = variance + 1.96 * variance_se

    result = {
        "bias2": bias2,
        "bias2_raw": raw_bias2,
        "bias2_mc_correction": bias2_mc_correction,
        "bias2_ci_low": float(bias2_low),
        "bias2_ci_high": float(bias2_high),
        "variance": variance,
        "variance_ci_low": float(variance_low),
        "variance_ci_high": float(variance_high),
        "noise": noise,
        "expected_mse": expected_mse,
        "empirical_mse": empirical_mse,
        "decomposition_gap": empirical_mse - expected_mse,
    }
    return result, predictions
