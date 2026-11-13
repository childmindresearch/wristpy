FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/

WORKDIR /app
COPY . /app/

RUN uv sync --locked --no-dev --no-cache \
    && mkdir -p /data /output

ENTRYPOINT ["/app/.venv/bin/wristpy"]

CMD ["/data", "--output", "/output", "--output-filetype", ".csv", "--calibrator", "none", \
     "--activity-metric", "enmo", "--epoch-length", "5", "--nonwear-algorithm", "ggir"]
