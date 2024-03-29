FROM python:3.11-buster

WORKDIR /app

ADD . /app

RUN mkdir /app/data
RUN mkdir /app/output
RUN pip install --no-cache-dir .

CMD ["python", "/app/batch_process_test.py"]
