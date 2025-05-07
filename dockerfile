FROM python:3.11-buster

WORKDIR /app
COPY . /app/


ENV INPUT_DIR=/data
ENV OUTPUT_DIR=/output
ENV OUTPUT_TYPE=.csv
ENV CALIBRATOR=none
ENV ACTIVITY_METRIC=enmo
ENV EPOCH_LENGTH=5
ENV NONWEAR=ggir
ENV THRESHOLDS=""


RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main

RUN mkdir -p $INPUT_DIR $OUTPUT_DIR

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
