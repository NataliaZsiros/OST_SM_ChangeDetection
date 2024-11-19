# Use the official Python image from the Docker Hub
FROM python:3.9.6

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY kafka_data_stream_conversion.py /app/

CMD ["python", "kafka_data_stream_conversion.py"]  
