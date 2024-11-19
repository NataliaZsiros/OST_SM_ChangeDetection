import pandas as pd
from confluent_kafka import Producer
import time

pd.set_option('display.float_format', '{:.8f}'.format)

broker = 'kafka:9092' 
topic = 'input'  

producer = Producer({'bootstrap.servers': broker})

def send_data_to_kafka(file_path):
    df = pd.read_csv(file_path)

    for index, row in df.iterrows():
        message_value = ','.join([f"{key}={value}" for key, value in row.items()])
        print(message_value) 
        producer.produce(topic=topic, value=message_value, key=str(row['StartTime']))
        time.sleep(2)

    producer.flush()

csv_file_path = 'data/wustl_iiot_2021.csv' 

send_data_to_kafka(csv_file_path)
