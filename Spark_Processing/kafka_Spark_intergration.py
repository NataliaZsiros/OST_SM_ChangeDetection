from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, current_timestamp, split, expr
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
from menelaus.change_detection import PageHinkley
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import joblib
import numpy as np
import time

# Define a mapping function to split key-value pairs into columns
def extract_key_value_columns(df, keys):
    for key in keys:
        key_col = f"{key}_pair"
        df = df.withColumn(key_col, expr(f"filter(key_value_pairs, x -> x like '{key}=%')")[0])
        
        # Step 2: Extract value after the '=' sign
        if key not in ["StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic",]:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(DoubleType()))  # Cast to numeric for calculations
        else:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(StringType()))  # Cast to string these will be dropped
        
        # Drop intermediate key-value column
        df = df.drop(key_col)
    return df

# connecting Apache Spark to the Kafka topic 'input' if fails it waits 5 second and re-tries it
while True:
    try:
        # Initialize Spark Session
        spark = SparkSession.builder \
            .appName("KafkaStreamProcessor") \
            .master("local[*]") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
            .getOrCreate()

        # Read Kafka Stream
        kafka_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "kafka:9092") \
            .option("subscribe", "input") \
            .option("startingOffsets", "earliest") \
            .load()
        break
    except Exception as e:
        print(f"Retrying Kafka connection... {e}")
        time.sleep(5)

print("Connection is complete!")

# splitting the string on commas to have the data into key-value pairs eg. 'Mean=1.5'
parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING) as raw_value") \
    .withColumn("key_value_pairs", split(col("raw_value"), ",")) 

# List of keys to extract (match with the fields you care about)
data_columns = [
    "StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "Mean", "Sport", "Dport",
    "SrcPkts", "DstPkts", "TotPkts", "DstBytes", "SrcBytes", "TotBytes", "SrcLoad",
    "DstLoad", "Load", "SrcRate", "DstRate", "Rate", "SrcLoss", "DstLoss", "Loss", 
    "pLoss", "SrcJitter", "DstJitter", "SIntPkt", "DIntPkt", "Proto", "Dur", "TcpRtt",
    "IdleTime", "Sum", "Min", "Max", "sDSb", "sTtl", "dTtl", "sIpId", "dIpId", 
    "SAppBytes", "DAppBytes", "TotAppByte", "SynAck", "RunTime", "sTos", "SrcJitAct",
    "DstJitAct", "Target"
]

# Apply the mapping function to extract key-value pairs into separate columns
parsed_stream = extract_key_value_columns(parsed_stream, data_columns)

# dropping unique string columns and also the target and traffic which wouldn't be available in real life
parsed_stream = parsed_stream.drop("StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "Target", "key_value_pairs", "raw_value")

# Add a current timestamp to each row - we might not need this but it was said that the systems needs a timestamp to function properly
parsed_stream_with_timestamp = parsed_stream.withColumn("current_timestamp", current_timestamp())


#IMPLEMENT YOU CHANGE DETECTION FUNCTIONS HERE AND THE STATISTICAL CALCULATIONS


# Output the results to the console
query = parsed_stream_with_timestamp.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()
