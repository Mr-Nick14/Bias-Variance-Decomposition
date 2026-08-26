"""Mathematical core of the synthetic bias-variance experiments."""

from collections.abc import Callable

import numpy as np


def true_function(x: np.ndarray) -> np.ndarray:
    """Return the regression function used to generate synthetic targets."""

    x = np.asarray(x, dtype=float)
    return np.sin(2.0 * x) + 0.2 * x**2


def generate_synthetic(
    n_samples: int,
    sigma: float,
    seed: int,
    x_domain: tuple[float, float] = (-3.0, 3.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one sample from Y = f(X) + epsilon."""

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
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
    """Generate a reusable sequence of independent training samples."""

    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2")

    return [
        generate_synthetic(n_train, sigma, data_seed + repeat)
        for repeat in range(n_repeats)
    ]


def bias_variance_decomposition(
    make_model: Callable[[int], object],
    *,
    training_sets: list[tuple[np.ndarray, np.ndarray]],
    x_eval: np.ndarray,
    sigma: float,
    model_seed: int,
    test_noise_seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Estimate the squared-loss decomposition on one fixed evaluation grid."""

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
        # Model randomness has a separate schedule from training-data randomness.
        model = make_model(model_seed + repeat)
        model.fit(x_train, y_train)
        predictions[repeat] = np.asarray(model.predict(x_eval)).reshape(-1)

        # Test noise is independent of every training sample and model fit.
        noisy_targets[repeat] = f_true + test_rng.normal(0.0, sigma, len(x_eval))

    mean_prediction = predictions.mean(axis=0)
    bias2 = float(np.mean((mean_prediction - f_true) ** 2))
    variance = float(np.mean(np.var(predictions, axis=0, ddof=0)))
    noise = float(sigma**2)
    expected_mse = bias2 + variance + noise
    empirical_mse = float(np.mean((noisy_targets - predictions) ** 2))

    result = {
        "bias2": bias2,
        "variance": variance,
        "noise": noise,
        "expected_mse": expected_mse,
        "empirical_mse": empirical_mse,
        "decomposition_gap": empirical_mse - expected_mse,
    }
    return result, predictions
