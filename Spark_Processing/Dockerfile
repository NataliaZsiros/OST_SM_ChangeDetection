FROM jupyter/pyspark-notebook

WORKDIR /app

COPY . /app/

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt