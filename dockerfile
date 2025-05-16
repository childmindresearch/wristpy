FROM python:3.11-buster

WORKDIR /app
COPY . /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main

RUN mkdir -p /data /output

ENTRYPOINT ["poetry", "run", "wristpy"]

CMD ["/data", "--output", "/output", "--output-filetype", ".csv", "--calibrator", "none", \
     "--activity-metric", "enmo", "--epoch-length", "5", "--nonwear-algorithm", "ggir"]
