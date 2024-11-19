FROM python:3.9.6

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY saving_InfluxDB.py /app/

CMD ["python", "saving_InfluxDB.py"]  
