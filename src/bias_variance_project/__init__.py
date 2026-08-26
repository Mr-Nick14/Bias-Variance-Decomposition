"""Experiments for empirical bias-variance decomposition."""

from .core import (
    bias_variance_decomposition,
    generate_synthetic,
    generate_training_sets,
    true_function,
)

__all__ = [
    "bias_variance_decomposition",
    "generate_synthetic",
    "generate_training_sets",
    "true_function",
]
