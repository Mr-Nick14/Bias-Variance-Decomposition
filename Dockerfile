FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /workspace

ENV MPLCONFIGDIR=/tmp/matplotlib \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --all-groups --no-install-project

COPY . .
RUN uv sync --locked --all-groups

CMD [".venv/bin/python", "scripts/run_experiments.py", "--fast"]
