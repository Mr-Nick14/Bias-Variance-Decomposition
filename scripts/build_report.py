"""Build the LaTeX report from current experiment outputs."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def latex_escape(value: object) -> str:
    text = str(value)
    for source, replacement in {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }.items():
        text = text.replace(source, replacement)
    return text


def write_latex_table(
    path: Path,
    headers: list[str],
    rows: list[list[str]],
    columns: str,
) -> None:
    lines = [
        r"\begin{center}",
        r"\small",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(project_root: Path) -> None:
    tables = project_root / "reports" / "tables"
    generated = project_root / "report" / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(tables / "best_complexity_summary.csv")
    best["setting"] = best.apply(
        lambda row: f"{row['complexity_label']}={row['complexity']:g}",
        axis=1,
    )
    best_rows = [
        [
            latex_escape(row.model),
            latex_escape(row.setting),
            f"{row.bias2:.3f}",
            f"{row.variance:.3f}",
            f"{row.expected_mse:.3f}",
        ]
        for row in best.itertuples()
    ]
    write_latex_table(
        generated / "best_models.tex",
        ["Модель", "Параметр", "Bias$^2$", "Variance", "Expected MSE"],
        best_rows,
        "llrrr",
    )

    real = pd.read_csv(tables / "real_model_summary.csv")
    real_rows = [
        [
            latex_escape(row.dataset),
            latex_escape(row.model),
            f"{row.rmse_mean:.3f}",
            f"{row.rmse_std:.3f}",
            f"{row.r2_mean:.3f}",
        ]
        for row in real.itertuples()
    ]
    write_latex_table(
        generated / "real_models.tex",
        ["Датасет", "Модель", "RMSE", "RMSE std", "$R^2$"],
        real_rows,
        "llrrr",
    )


def check_inputs(project_root: Path) -> None:
    required = [
        project_root / "reports" / "tables" / "best_complexity_summary.csv",
        project_root / "reports" / "tables" / "real_model_summary.csv",
        project_root / "reports" / "figures" / "synthetic_eda.png",
        project_root / "reports" / "figures" / "complexity_tradeoff.png",
        project_root / "reports" / "figures" / "train_size_noise_effects.png",
        project_root / "reports" / "figures" / "mlp_training_history.png",
        project_root / "reports" / "figures" / "real_model_comparison.png",
        project_root / "reports" / "figures" / "real_proxy_decomposition.png",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        names = "\n".join(str(path) for path in missing)
        raise FileNotFoundError("Missing report inputs\n" + names)


def compile_report(project_root: Path) -> Path:
    xelatex = shutil.which("xelatex")
    if xelatex is None:
        raise RuntimeError("xelatex is required to build report/report.pdf")

    build_dir = project_root / "tmp" / "pdfs" / "latex-build"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True)
    command = [
        xelatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={build_dir}",
        str(project_root / "report" / "report.tex"),
    ]
    environment = os.environ.copy()
    source_commit = subprocess.run(
        ["git", "log", "-1", "--format=%at", "--", "report/report.tex"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    environment["SOURCE_DATE_EPOCH"] = source_commit.stdout.strip() or "946684800"

    try:
        for _ in range(2):
            completed = subprocess.run(
                command,
                cwd=project_root,
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            if completed.returncode != 0:
                log = completed.stdout + "\n" + completed.stderr
                raise RuntimeError("xelatex failed\n" + log[-8000:])
            if "Overfull \\hbox" in completed.stdout or "Overfull \\vbox" in completed.stdout:
                raise RuntimeError("xelatex reported an overfull layout box")

        output = project_root / "report" / "report.pdf"
        shutil.copy2(build_dir / "report.pdf", output)
        return output
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
        pdf_tmp = project_root / "tmp" / "pdfs"
        if pdf_tmp.exists() and not any(pdf_tmp.iterdir()):
            pdf_tmp.rmdir()


def build_report(project_root: Path, *, refresh: bool = False, fast: bool = False) -> Path:
    project_root = project_root.resolve()
    if refresh:
        command = [
            sys.executable,
            str(project_root / "scripts" / "run_experiments.py"),
            "--project-root",
            str(project_root),
        ]
        if fast:
            command.append("--fast")
        subprocess.run(command, cwd=project_root, check=True)

    check_inputs(project_root)
    write_tables(project_root)
    return compile_report(project_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    print(build_report(args.project_root, refresh=args.refresh, fast=args.fast))


if __name__ == "__main__":
    main()
