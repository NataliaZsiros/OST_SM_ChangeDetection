import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
import time

pd.set_option('display.float_format', '{:.8f}'.format)

def create_topic(broker, topic, num_partitions, replication_factor):
    admin_client = AdminClient({'bootstrap.servers': broker})
    topic_list = [NewTopic(topic, num_partitions=num_partitions, replication_factor=replication_factor)]
    
    futures = admin_client.create_topics(topic_list)
    for topic, future in futures.items():
        try:
            future.result()  
            print(f"Topic {topic} created successfully")
        except Exception as e:
            print(f"Failed to create topic {topic}: {e}")

broker = 'kafka:9092' 
topic = 'input'  

create_topic(broker, topic, num_partitions=8, replication_factor=1)

producer = Producer({'bootstrap.servers': broker})

def send_data_to_kafka(file_path):
    df = pd.read_csv(file_path)
    #df = df[df['SrcAddr'] == '192.168.0.20']

    for index, row in df.iterrows():
        message_value = ','.join([f"{key}={value}" for key, value in row.items()])
        print(message_value) 
        producer.produce(topic=topic, value=message_value, key=str(row['StartTime']))
        time.sleep(2)

    producer.flush()

csv_file_path = 'data/wustl_iiot_2021.csv' 

send_data_to_kafka(csv_file_path)
