# check=error=true
FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /app

ENV PYTHONUNBUFFERED=1

ENV UV_COMPILE_BYTECODE=1

ENV UV_LINK_MODE=copy

ENV UV_NO_DEV=1

ENV UV_TOOL_BIN_DIR=/usr/local/bin

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY --chown=nonroot:nonroot . /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

RUN chown -R nonroot:nonroot /app \
    && mkdir -p /data /output \
    && chown -R nonroot:nonroot /data /output

ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p /data /output

ENTRYPOINT ["wristpy"]

USER nonroot

CMD ["/data", "--output", "/output", "--output-filetype", ".csv", "--calibrator", "none", \
     "--activity-metric", "enmo", "--epoch-length", "5", "--nonwear-algorithm", "ggir"]
