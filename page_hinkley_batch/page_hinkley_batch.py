from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, split, expr, min, max, avg, stddev
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
from menelaus.change_detection import PageHinkley
import numpy as np
import time


# Initialize Page-Hinkley Detector
ph = PageHinkley(delta=0.001, threshold=1, direction="negative", burn_in=1)

# Function to extract key-value pairs into columns
def extract_key_value_columns(df, keys):
    for key in keys:
        key_col = f"{key}_pair"
        df = df.withColumn(key_col, expr(f"filter(key_value_pairs, x -> x like '{key}=%')")[0])
        if key not in ["StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "Target"]:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(DoubleType()))
        else:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(StringType()))
        df = df.drop(key_col)
    return df

# Function to apply Page-Hinkley
def detect_change(value):
    ph.update(value)
    return ph.drift_state

# Batch Processing
def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return
    batch_pd = batch_df.toPandas()
    for _, row in batch_pd.iterrows():
        print("-" * 40)
        print(f"  DstJitter: {row['DstJitter']}")
        print(f"  result: {row['result']}")
        print(f"  Target: {row['Target']}")
        print(f"  current_timestamp: {row['current_timestamp']}")
        print("-" * 40)


# Start Spark Session and Kafka Stream
while True:
    try:
        spark = SparkSession.builder \
            .appName("KafkaStreamProcessor") \
            .master("local[*]") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
            .getOrCreate()

        kafka_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "kafka:9092") \
            .option("subscribe", "input") \
            .option("startingOffsets", "earliest") \
            .load()
        break
    except Exception as e:
        print(f"Retrying Kafka connection: {e}")
        time.sleep(5)

parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING) as raw_value") \
    .withColumn("key_value_pairs", split(col("raw_value"), ","))

data_columns = ["DstJitter", "Target"]

parsed_stream = extract_key_value_columns(parsed_stream, data_columns)
parsed_stream = parsed_stream.drop("key_value_pairs", "raw_value")

parsed_stream_with_timestamp = parsed_stream.withColumn("current_timestamp", current_timestamp())

change_detection_udf = udf(detect_change, StringType())

# Stream Transformations
vectorized_stream = parsed_stream_with_timestamp.select("current_timestamp", "DstJitter", "Target")

stream_with_detection = vectorized_stream.withColumn("result", change_detection_udf(col("DstJitter")))

query = stream_with_detection.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("update") \
    .trigger(processingTime='10 seconds') \
    .start()

query.awaitTermination()
