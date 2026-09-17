"""Plots for the synthetic and real-data experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .core import generate_synthetic, true_function
from .real_data import load_real_datasets, target_correlations

PALETTE = {
    "bias2": "#D55E00",
    "variance": "#0072B2",
    "noise": "#999999",
    "expected_mse": "#009E73",
    "empirical_mse": "#CC79A7",
}


sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_synthetic_eda(path: Path, *, seed: int = 42) -> None:
    x, y = generate_synthetic(500, 0.35, seed)
    x = x.reshape(-1)
    grid = np.linspace(-3, 3, 400)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].scatter(x, y, s=14, alpha=0.35, color="#0072B2", label="observations")
    axes[0].plot(grid, true_function(grid), color="#D55E00", lw=2.5, label="true f(x)")
    axes[0].set(xlabel="x", ylabel="y", title="Synthetic regression data")
    axes[0].legend()
    sns.histplot(y, bins=25, kde=True, ax=axes[1], color="#009E73")
    axes[1].set(title="Target distribution", xlabel="y")
    ordered_y = np.sort(y)
    axes[2].plot(ordered_y, np.linspace(0, 1, len(ordered_y)), color="#009E73")
    axes[2].set(title="Empirical target CDF", xlabel="y", ylabel="cumulative share")
    fig.suptitle("EDA for Y = sin(2X) + 0.2X² + ε, ε ~ N(0, 0.35²)", y=1.03)
    fig.tight_layout()
    save_figure(fig, path)


def plot_complexity_grid(results: pd.DataFrame, path: Path) -> None:
    models = list(results["model"].drop_duplicates())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    for ax, model in zip(axes.flat, models, strict=False):
        part = results[results["model"] == model].sort_values("complexity")
        for metric in ["bias2", "variance", "noise", "expected_mse", "empirical_mse"]:
            ax.plot(
                part["complexity"],
                part[metric],
                marker="o",
                ms=4,
                lw=1.8,
                color=PALETTE[metric],
                label=metric.replace("_", " "),
                alpha=0.9 if metric != "empirical_mse" else 0.65,
            )
        ax.fill_between(
            part["complexity"],
            part["bias2_ci_low"],
            part["bias2_ci_high"],
            color=PALETTE["bias2"],
            alpha=0.12,
        )
        ax.fill_between(
            part["complexity"],
            part["variance_ci_low"],
            part["variance_ci_high"],
            color=PALETTE["variance"],
            alpha=0.12,
        )
        label = part["complexity_label"].iloc[0]
        ax.set(title=model, xlabel=label, ylabel="integrated squared error")
        if model == "KNN":
            ax.text(
                0.98,
                0.95,
                "flexibility decreases →",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Bias-variance trade-off across five model families", fontsize=16)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(fig, path)


def plot_representative_predictions(
    predictions: dict[str, np.ndarray], x_eval: np.ndarray, path: Path
) -> None:
    models = list(predictions)
    x = x_eval.reshape(-1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models, strict=False):
        matrix = predictions[model]
        for curve in matrix[: min(12, len(matrix))]:
            ax.plot(x, curve, color="#56B4E9", lw=0.7, alpha=0.22)
        ax.plot(x, matrix.mean(axis=0), color="#0072B2", lw=2.2, label="mean prediction")
        ax.plot(x, true_function(x), color="#D55E00", ls="--", lw=2, label="true f(x)")
        ax.set(title=model, xlabel="x", ylabel="prediction")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Predictions over independent training samples (best configurations)", fontsize=16)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, path)


def plot_mlp_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(history["epoch"], history["train_loss"], color="#0072B2")
    axes[0].set(title="Optimisation curve", xlabel="epoch", ylabel="regularised train loss")
    axes[1].plot(history["epoch"], history["train_rmse"], color="#009E73", label="train RMSE")
    axes[1].plot(history["epoch"], history["val_rmse"], color="#D55E00", label="validation RMSE")
    axes[1].axhline(0.35, color="#999999", ls=":", lw=1.5, label="noise sigma=0.35")
    best_epoch = int(history.loc[history["val_rmse"].idxmin(), "epoch"])
    axes[1].axvline(best_epoch, color="#555555", ls="--", lw=1, label=f"best epoch={best_epoch}")
    axes[1].set(title="Generalisation during training", xlabel="epoch", ylabel="RMSE")
    axes[1].legend()
    fig.suptitle("MLP (64, 64), tanh, Adam, L2 weight decay alpha=1e-3")
    fig.tight_layout()
    save_figure(fig, path)


def plot_mlp_capacity(results: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    width_part = results[results["study"] == "width_alpha"]
    for alpha, part in width_part.groupby("alpha"):
        part = part.sort_values("width")
        axes[0].plot(
            part["width"],
            part["expected_mse"],
            marker="o",
            label=f"alpha={alpha:g}",
        )
    axes[0].set(
        title="Width and L2 regularisation",
        xlabel="hidden width",
        ylabel="expected MSE",
    )
    axes[0].legend()

    depth_part = results[results["study"] == "depth"].sort_values("depth")
    for metric in ["bias2", "variance", "expected_mse"]:
        axes[1].plot(
            depth_part["depth"],
            depth_part[metric],
            marker="o",
            color=PALETTE[metric],
            label=metric.replace("_", " "),
        )
    axes[1].set(
        title="Depth at width 64 and alpha=1e-3",
        xlabel="hidden layers",
        ylabel="integrated squared error",
    )
    axes[1].legend()
    fig.suptitle("MLP capacity is controlled by width, depth and L2")
    fig.tight_layout()
    save_figure(fig, path)


def plot_real_eda(
    output_dir: Path, *, include_california: bool = True, data_home: Path | None = None
) -> list[Path]:
    paths = []
    for name, (x, y) in load_real_datasets(
        include_california=include_california, data_home=data_home
    ).items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
        sns.histplot(y, kde=True, bins=min(20, max(7, len(y) // 4)), ax=axes[0], color="#0072B2")
        axes[0].set(title=f"{name} target", xlabel=y.name)
        ordered_y = np.sort(y.to_numpy())
        axes[1].plot(
            ordered_y,
            np.linspace(0, 1, len(ordered_y)),
            color="#009E73",
        )
        axes[1].set(title="Empirical target CDF", xlabel=y.name, ylabel="cumulative share")
        target_corr = target_correlations(x, y).head(10).to_frame("correlation")
        colors = np.where(target_corr["correlation"] >= 0, "#0072B2", "#D55E00")
        axes[2].barh(target_corr.index[::-1], target_corr["correlation"][::-1], color=colors[::-1])
        axes[2].axvline(0, color="#555555", lw=0.8)
        axes[2].set(title="Feature-target correlations", xlabel="correlation")
        fig.suptitle(f"Exploratory view - {name}", fontsize=15)
        fig.tight_layout()
        slug = name.lower().replace("-", "_").replace(" ", "_")
        path = output_dir / f"eda_{slug}.png"
        save_figure(fig, path)
        paths.append(path)
    return paths


def plot_real_model_comparison(summary: pd.DataFrame, path: Path) -> None:
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 4.5), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets, strict=True):
        part = summary[summary["dataset"] == dataset].sort_values("rmse_mean", ascending=False)
        y_pos = np.arange(len(part))
        ax.barh(y_pos, part["rmse_mean"], xerr=part["rmse_std"], color="#56B4E9", alpha=0.85)
        ax.set_yticks(y_pos, labels=part["model"])
        ax.set(title=dataset, xlabel="5-fold CV RMSE (lower is better)")
    fig.suptitle("Real datasets with identical repeated cross-validation splits", fontsize=15)
    fig.tight_layout()
    save_figure(fig, path)


def plot_real_bootstrap_identity(results: pd.DataFrame, path: Path) -> None:
    datasets = list(results["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 4.8), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets, strict=True):
        part = results[results["dataset"] == dataset].sort_values("mean_bootstrap_mse")
        x_pos = np.arange(len(part))
        ax.bar(
            x_pos,
            part["mean_prediction_mse"],
            label="MSE of mean prediction",
            color="#D55E00",
        )
        ax.bar(
            x_pos,
            part["bootstrap_variance"],
            bottom=part["mean_prediction_mse"],
            label="bootstrap variance",
            color="#0072B2",
        )
        ax.scatter(
            x_pos,
            part["mean_bootstrap_mse"],
            marker="_",
            s=260,
            linewidth=2,
            color="#111111",
            label="mean bootstrap MSE",
            zorder=3,
        )
        ax.set_xticks(x_pos, labels=part["model"], rotation=35, ha="right")
        ax.set(title=dataset, ylabel="test squared error")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Exact bootstrap squared-loss identity on fixed test sets", fontsize=15)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    save_figure(fig, path)


def plot_diabetes_complexity(results: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for ax, model in zip(
        axes.flat,
        ["Ridge", "Decision Tree", "KNN", "MLP"],
        strict=True,
    ):
        part = results[results["model"] == model]
        if model == "MLP":
            for alpha, alpha_part in part.groupby("alpha"):
                alpha_part = alpha_part.sort_values("complexity")
                ax.plot(
                    alpha_part["complexity"],
                    alpha_part["mean_bootstrap_mse"],
                    marker="o",
                    label=f"MSE, alpha={alpha:g}",
                )
                ax.plot(
                    alpha_part["complexity"],
                    alpha_part["bootstrap_variance"],
                    marker=".",
                    ls="--",
                    alpha=0.75,
                    label=f"variance, alpha={alpha:g}",
                )
        else:
            part = part.sort_values("complexity")
            ax.plot(
                part["complexity"],
                part["mean_bootstrap_mse"],
                marker="o",
                color="#009E73",
                label="mean bootstrap MSE",
            )
            ax.plot(
                part["complexity"],
                part["bootstrap_variance"],
                marker="o",
                ls="--",
                color="#0072B2",
                label="bootstrap variance",
            )
        ax.set(
            title=model,
            xlabel=part["complexity_label"].iloc[0],
            ylabel="test squared error",
        )
        ax.legend(fontsize=8)
    fig.suptitle("Complexity and bootstrap stability on Diabetes", fontsize=15)
    fig.tight_layout()
    save_figure(fig, path)


def plot_train_size_and_noise(results: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    for ax, study, title in zip(
        axes,
        ["sample_size", "noise"],
        ["Effect of training-set size", "Effect of observation noise"],
        strict=True,
    ):
        part = results[results["study"] == study].sort_values("value")
        for metric in ["bias2", "variance", "noise", "expected_mse"]:
            ax.plot(part["value"], part[metric], marker="o", label=metric, color=PALETTE[metric])
        ax.set(title=title, xlabel=part["value_label"].iloc[0], ylabel="squared error")
    axes[1].legend(loc="upper left")
    fig.suptitle("Decision Tree with max_depth=6")
    fig.tight_layout()
    save_figure(fig, path)
