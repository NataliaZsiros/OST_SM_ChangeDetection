FROM python:3.9.6

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y openjdk-11-jdk-headless:arm64 && \
    apt-get clean && update-alternatives --config java

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY kafka_Spark_intergration.py /app/

CMD ["python", "kafka_Spark_intergration.py"]
