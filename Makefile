.PHONY: install run fast notebook report report-refresh all

install:
	uv sync --all-groups

run:
	uv run python scripts/run_experiments.py

fast:
	uv run python scripts/run_experiments.py --fast

notebook:
	uv run jupyter execute research.ipynb --inplace

report:
	uv run python scripts/build_report.py

report-refresh:
	uv run python scripts/build_report.py --refresh

all: run notebook report
