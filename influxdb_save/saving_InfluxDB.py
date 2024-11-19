from confluent_kafka import Consumer, KafkaError
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time

kafka_config = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'my-group'
}

consumer = Consumer(kafka_config)
consumer.subscribe(['input'])  

url = "http://influxdb:8086"
token = "TxqYqsyBImk-hYtUMvFPZCCZJ8odQSAlDhEJEwRVPw0NYEpBtTrcBLwyQysVMFws50YrDi0YrIbGgnLX5xNKAw=="  
org = "ChangeDetection_org"
bucket = "ChangeDetection"
username = "admin"
password = "password"

while True:
    try:
        client = InfluxDBClient(url=url, token=token, org=org)
        print(client.ping())  # Test connection
        break
    except Exception as e:
        print(f"Retrying InfluxDB connection... {e}")
        time.sleep(5)

write_api = client.write_api(write_options=SYNCHRONOUS)


try:
    while True:
        msg = consumer.poll(timeout=1.0)  

        if msg is None:
            continue  
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print(f"End of partition reached {msg.topic()} [{msg.partition()}]")
            elif msg.error():
                print(f"Kafka Error: {msg.error()}")
            continue

        message_str = msg.value().decode('utf-8')
        print(f"Received message: {message_str}")

        pairs = [item.split('=') for item in message_str.split(',')]
        new_row = {key: value for key, value in pairs}

        point = Point("Traffic")
        for key, value in new_row.items():
            try:
                formatted_value = float(value)
                # Ensure the float is in standard decimal format
                formatted_value = f"{formatted_value:.8f}"
                point = point.field(key, float(formatted_value))
            except ValueError:
                point = point.field(key, value)

        # Write the Point to InfluxDB
        write_api.write(bucket=bucket, org=org, record=point)
        print("Data written to InfluxDB.")

except KeyboardInterrupt:
    print("Consumer interrupted by user.")
finally:
    consumer.close()
