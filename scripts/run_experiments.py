"""Run the project experiments and save their tables and figures."""

import argparse
from pathlib import Path

import pandas as pd

from bias_variance_project.experiments import (
    best_complexity_summary,
    compare_cv_with_decomposition,
    mlp_training_history,
    run_complexity_experiment,
    run_mlp_capacity_experiment,
    run_train_size_and_noise_experiments,
)
from bias_variance_project.plotting import (
    plot_complexity_grid,
    plot_diabetes_complexity,
    plot_mlp_capacity,
    plot_mlp_history,
    plot_real_bootstrap_identity,
    plot_real_eda,
    plot_real_model_comparison,
    plot_representative_predictions,
    plot_synthetic_eda,
    plot_train_size_and_noise,
)
from bias_variance_project.real_data import (
    bootstrap_mse_identity,
    diabetes_complexity_experiment,
    evaluate_real_datasets,
    real_data_overview,
)


def save_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def run_experiments(project_root: Path, *, fast: bool = False) -> None:
    figures = project_root / "reports" / "figures"
    tables = project_root / "reports" / "tables"
    data_home = project_root / "data" / "scikit_learn"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    print("Running synthetic complexity experiments")
    plot_synthetic_eda(figures / "synthetic_eda.png")
    complexity, representative, x_eval = run_complexity_experiment(fast=fast)
    save_table(complexity, tables / "complexity_results.csv")
    save_table(best_complexity_summary(complexity), tables / "best_complexity_summary.csv")
    plot_complexity_grid(complexity, figures / "complexity_tradeoff.png")
    plot_representative_predictions(
        representative,
        x_eval,
        figures / "representative_predictions.png",
    )

    print("Running cross-validation and training-size experiments")
    tuning = compare_cv_with_decomposition(complexity, fast=fast)
    train_size_noise = run_train_size_and_noise_experiments(fast=fast)
    save_table(tuning, tables / "cv_complexity_selection.csv")
    save_table(train_size_noise, tables / "train_size_noise_results.csv")
    plot_train_size_and_noise(
        train_size_noise,
        figures / "train_size_noise_effects.png",
    )

    print("Recording MLP training history")
    capacity = run_mlp_capacity_experiment(fast=fast)
    history = mlp_training_history(epochs=120 if fast else 1_000)
    save_table(capacity, tables / "mlp_capacity_results.csv")
    save_table(history, tables / "mlp_training_history.csv")
    plot_mlp_capacity(capacity, figures / "mlp_capacity.png")
    plot_mlp_history(history, figures / "mlp_training_history.png")

    print("Running real-data cross-validation and bootstrap analysis")
    overview = real_data_overview(include_california=not fast, data_home=data_home)
    real_scores, real_summary = evaluate_real_datasets(fast=fast, data_home=data_home)
    bootstrap = bootstrap_mse_identity(fast=fast, data_home=data_home)
    diabetes_complexity = diabetes_complexity_experiment(fast=fast)
    save_table(overview, tables / "real_data_overview.csv")
    save_table(real_scores, tables / "real_cv_scores.csv")
    save_table(real_summary, tables / "real_model_summary.csv")
    save_table(bootstrap, tables / "real_bootstrap_identity.csv")
    save_table(diabetes_complexity, tables / "diabetes_complexity.csv")
    plot_real_eda(
        figures,
        include_california=not fast,
        data_home=data_home,
    )
    plot_real_model_comparison(real_summary, figures / "real_model_comparison.png")
    plot_real_bootstrap_identity(
        bootstrap,
        figures / "real_bootstrap_identity.png",
    )
    plot_diabetes_complexity(
        diabetes_complexity,
        figures / "diabetes_complexity.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Run a smaller smoke configuration")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    run_experiments(args.project_root.resolve(), fast=args.fast)


if __name__ == "__main__":
    main()
