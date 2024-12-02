from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, current_timestamp, split, expr
from pyspark.sql.functions import min, max, avg, stddev
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
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

parsed_stream_with_timestamp.printSchema()

# Define the batch processing function
def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        print(f"Batch {batch_id} is empty. Skipping.")
        return

    print(f"Processing batch {batch_id}...")

    # Define the numerical columns for aggregation
    numerical_columns = ['DstJitter', 'TcpRtt', 'DIntPkt', 'DstPkts', 'DstBytes', 
                   'SynAck', 'DstRate', 'DstLoad', 'sTtl', 'DstLoss', 
                   'SAppBytes', 'pLoss', 'Sport', 'dTtl', 'Dport']
    #Enable below 2 rows to show all numerical features statistic
    #excludes = ["StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "Target"]
    #numerical_columns = list(filter(lambda x: x not in excludes, data_columns))

    # Ensure all numerical columns are present in the batch
    missing_columns = [col for col in numerical_columns if col not in batch_df.columns]
    if missing_columns:
        print(f"Missing columns in batch {batch_id}: {missing_columns}")
        return

    # Perform aggregation and show statistics
    stats_df = batch_df.agg(
        *[min(col(c)).alias(f"Min_{c}") for c in numerical_columns],
        *[max(col(c)).alias(f"Max_{c}") for c in numerical_columns],
        *[avg(col(c)).alias(f"Avg_{c}") for c in numerical_columns],
        *[stddev(col(c)).alias(f"Std_{c}") for c in numerical_columns]
    )

    # Convert to Pandas for better formatting
    stats_pd = stats_df.toPandas()

    # Print formatted statistics for each column
    print(f"\n--- Statistics for Batch {batch_id} ---\n")
    
    for column in numerical_columns:
        print(f"--- {column} ---")
        print(f"  Min: {stats_pd[f'Min_{column}'][0]:.4f}")
        print(f"  Max: {stats_pd[f'Max_{column}'][0]:.4f}")
        print(f"  Avg: {stats_pd[f'Avg_{column}'][0]:.4f}")
        print(f"  Std Dev: {stats_pd[f'Std_{column}'][0]:.4f}")
        print("-" * 40)


# Write stream and process each batch using foreachBatch
query = parsed_stream_with_timestamp.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("update") \
	.trigger(processingTime='10 seconds') \
    .start()

query.awaitTermination()

'''
root
 |-- Mean: double (nullable = true)
 |-- Sport: double (nullable = true)
 |-- Dport: double (nullable = true)
 |-- SrcPkts: double (nullable = true)
 |-- DstPkts: double (nullable = true)
 |-- TotPkts: double (nullable = true)
 |-- DstBytes: double (nullable = true)
 |-- SrcBytes: double (nullable = true)
 |-- TotBytes: double (nullable = true)
 |-- SrcLoad: double (nullable = true)
 |-- DstLoad: double (nullable = true)
 |-- Load: double (nullable = true)
 |-- SrcRate: double (nullable = true)
 |-- DstRate: double (nullable = true)
 |-- Rate: double (nullable = true)
 |-- SrcLoss: double (nullable = true)
 |-- DstLoss: double (nullable = true)
 |-- Loss: double (nullable = true)
 |-- pLoss: double (nullable = true)
 |-- SrcJitter: double (nullable = true)
 |-- DstJitter: double (nullable = true)
 |-- SIntPkt: double (nullable = true)
 |-- DIntPkt: double (nullable = true)
 |-- Proto: double (nullable = true)
 |-- Dur: double (nullable = true)
 |-- TcpRtt: double (nullable = true)
 |-- IdleTime: double (nullable = true)
 |-- Sum: double (nullable = true)
 |-- Min: double (nullable = true)
 |-- Max: double (nullable = true)
 |-- sDSb: double (nullable = true)
 |-- sTtl: double (nullable = true)
 |-- dTtl: double (nullable = true)
 |-- sIpId: double (nullable = true)
 |-- dIpId: double (nullable = true)
 |-- SAppBytes: double (nullable = true)
 |-- DAppBytes: double (nullable = true)
 |-- TotAppByte: double (nullable = true)
 |-- SynAck: double (nullable = true)
 |-- RunTime: double (nullable = true)
 |-- sTos: double (nullable = true)
 |-- SrcJitAct: double (nullable = true)
 |-- DstJitAct: double (nullable = true)
 |-- current_timestamp: timestamp (nullable = false)
'''


'''
Processing batch 1...

--- Statistics for Batch 1 ---

--- DstJitter ---
  Min: 0.0000
  Max: 64.7336
  Avg: 11.2845
  Std Dev: 9.4958
----------------------------------------
--- TcpRtt ---
  Min: 0.0000
  Max: 0.0014
  Avg: 0.0007
  Std Dev: 0.0004
----------------------------------------
--- DIntPkt ---
  Min: 0.0000
  Max: 25.2349
  Avg: 6.6685
  Std Dev: 3.9156
----------------------------------------
--- DstPkts ---
  Min: 0.0000
  Max: 204.0000
  Avg: 11.1905
  Std Dev: 30.6039
----------------------------------------
--- DstBytes ---
  Min: 0.0000
  Max: 71672.0000
  Avg: 2109.2857
  Std Dev: 10996.9803
----------------------------------------
--- SynAck ---
  Min: 0.0000
  Max: 0.0014
  Avg: 0.0007
  Std Dev: 0.0004
----------------------------------------
--- DstRate ---
  Min: 0.0000
  Max: 162.9779
  Avg: 113.7342
  Std Dev: 49.5837
----------------------------------------
--- DstLoad ---
  Min: 0.0000
  Max: 115812.9922
  Avg: 60213.9518
  Std Dev: 26123.5919
----------------------------------------
--- sTtl ---
  Min: 41.0000
  Max: 241.0000
  Avg: 130.6190
  Std Dev: 28.6220
----------------------------------------
--- DstLoss ---
  Min: 0.0000
  Max: 2.0000
  Avg: 1.6667
  Std Dev: 0.7544
----------------------------------------
--- SAppBytes ---
  Min: 0.0000
  Max: 20000.0000
  Avg: 622.9048
  Std Dev: 3113.8272
----------------------------------------
--- pLoss ---
  Min: 0.0000
  Max: 50.0000
  Avg: 19.4625
  Std Dev: 9.6692
----------------------------------------
--- Sport ---
  Min: 1740.0000
  Max: 65307.0000
  Avg: 52505.4524
  Std Dev: 12655.6499
----------------------------------------
--- dTtl ---
  Min: 0.0000
  Max: 128.0000
  Avg: 56.3810
  Std Dev: 25.2953
----------------------------------------
--- Dport ---
  Min: 80.0000
  Max: 48310.0000
  Avg: 2283.5952
  Std Dev: 8356.4972
---------------------------------------- 
'''
