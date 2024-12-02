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
    # Count the number of records in the batch
    record_count = batch_df.count()
    print(f"Number of records in batch {batch_id}: {record_count}")

    # Define the numerical columns for aggregation
    numerical_columns = ['DstJitter', 'TcpRtt', 'DIntPkt', 'DstPkts', 'DstBytes', 
                   'SynAck', 'DstRate', 'DstLoad', 'sTtl', 'DstLoss', 
                   'SAppBytes', 'pLoss', 'Sport', 'dTtl', 'Dport']
    #Enable below 2 rows to show all features statistic
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
Number of records in batch 1: 73

--- Statistics for Batch 1 ---

--- DstJitter ---
  Min: 0.0000
  Max: 15.0418
  Avg: 10.7956
  Std Dev: 4.0240
----------------------------------------
--- TcpRtt ---
  Min: 0.0000
  Max: 0.0017
  Avg: 0.0007
  Std Dev: 0.0004
----------------------------------------
--- DIntPkt ---
  Min: 0.0000
  Max: 8.4150
  Avg: 6.5747
  Std Dev: 2.3482
----------------------------------------
--- DstPkts ---
  Min: 0.0000
  Max: 8.0000
  Avg: 6.9863
  Std Dev: 2.5193
----------------------------------------
--- DstBytes ---
  Min: 0.0000
  Max: 510.0000
  Avg: 443.9178
  Std Dev: 159.8887
----------------------------------------
--- SynAck ---
  Min: 0.0000
  Max: 0.0016
  Avg: 0.0007
  Std Dev: 0.0004
----------------------------------------
--- DstRate ---
  Min: 0.0000
  Max: 165.7495
  Avg: 119.1929
  Std Dev: 42.6098
----------------------------------------
--- DstLoad ---
  Min: 0.0000
  Max: 84863.7500
  Avg: 60669.5039
  Std Dev: 21704.5544
----------------------------------------
--- sTtl ---
  Min: 0.0000
  Max: 213.0000
  Avg: 124.6712
  Std Dev: 29.3991
----------------------------------------
--- DstLoss ---
  Min: 0.0000
  Max: 2.0000
  Avg: 1.7808
  Std Dev: 0.6291
----------------------------------------
--- SAppBytes ---
  Min: 0.0000
  Max: 3200.0000
  Avg: 71.3425
  Std Dev: 372.8801
----------------------------------------
--- pLoss ---
  Min: 0.0000
  Max: 50.0000
  Avg: 19.8526
  Std Dev: 8.5793
----------------------------------------
--- Sport ---
  Min: 0.0000
  Max: 65224.0000
  Avg: 53922.6164
  Std Dev: 12792.9839
----------------------------------------
--- dTtl ---
  Min: 0.0000
  Max: 64.0000
  Avg: 56.9863
  Std Dev: 20.1305
----------------------------------------
--- Dport ---
  Min: 0.0000
  Max: 32908.0000
  Avg: 1026.1233
  Std Dev: 3887.8545
----------------------------------------
Processing batch 2...
Number of records in batch 2: 4

--- Statistics for Batch 2 ---

--- DstJitter ---
  Min: 12.1134
  Max: 13.4353
  Avg: 12.5460
  Std Dev: 0.6046
----------------------------------------
--- TcpRtt ---
  Min: 0.0007
  Max: 0.0013
  Avg: 0.0008
  Std Dev: 0.0003
----------------------------------------
--- DIntPkt ---
  Min: 7.2401
  Max: 7.4829
  Avg: 7.3651
  Std Dev: 0.1079
----------------------------------------
--- DstPkts ---
  Min: 8.0000
  Max: 8.0000
  Avg: 8.0000
  Std Dev: 0.0000
----------------------------------------
--- DstBytes ---
  Min: 508.0000
  Max: 508.0000
  Avg: 508.0000
  Std Dev: 0.0000
----------------------------------------
--- SynAck ---
  Min: 0.0007
  Max: 0.0013
  Avg: 0.0008
  Std Dev: 0.0003
----------------------------------------
--- DstRate ---
  Min: 131.8913
  Max: 134.7605
  Avg: 133.5202
  Std Dev: 1.3859
----------------------------------------
--- DstLoad ---
  Min: 67076.1562
  Max: 68535.3438
  Avg: 67904.5820
  Std Dev: 704.8284
----------------------------------------
--- sTtl ---
  Min: 128.0000
  Max: 128.0000
  Avg: 128.0000
  Std Dev: 0.0000
----------------------------------------
--- DstLoss ---
  Min: 2.0000
  Max: 2.0000
  Avg: 2.0000
  Std Dev: 0.0000
----------------------------------------
--- SAppBytes ---
  Min: 24.0000
  Max: 24.0000
  Avg: 24.0000
  Std Dev: 0.0000
----------------------------------------
--- pLoss ---
  Min: 18.1818
  Max: 18.1818
  Avg: 18.1818
  Std Dev: 0.0000
----------------------------------------
--- Sport ---
  Min: 52259.0000
  Max: 58240.0000
  Avg: 55221.2500
  Std Dev: 2810.4925
----------------------------------------
--- dTtl ---
  Min: 64.0000
  Max: 64.0000
  Avg: 64.0000
  Std Dev: 0.0000
----------------------------------------
--- Dport ---
  Min: 502.0000
  Max: 502.0000
  Avg: 502.0000
  Std Dev: 0.0000
----------------------------------------
Processing batch 3...
Number of records in batch 3: 5

--- Statistics for Batch 3 ---

--- DstJitter ---
  Min: 7.9140
  Max: 13.5789
  Avg: 11.1232
  Std Dev: 2.2976
----------------------------------------
--- TcpRtt ---
  Min: 0.0006
  Max: 0.0015
  Avg: 0.0010
  Std Dev: 0.0004
----------------------------------------
--- DIntPkt ---
  Min: 6.4676
  Max: 7.8622
  Avg: 7.3460
  Std Dev: 0.5426
----------------------------------------
--- DstPkts ---
  Min: 6.0000
  Max: 8.0000
  Avg: 7.2000
  Std Dev: 1.0954
----------------------------------------
--- DstBytes ---
  Min: 384.0000
  Max: 508.0000
  Avg: 458.4000
  Std Dev: 67.9176
----------------------------------------
--- SynAck ---
  Min: 0.0006
  Max: 0.0013
  Avg: 0.0010
  Std Dev: 0.0003
----------------------------------------
--- DstRate ---
  Min: 124.2205
  Max: 151.7312
  Avg: 133.9342
  Std Dev: 10.8812
----------------------------------------
--- DstLoad ---
  Min: 63600.9023
  Max: 77686.3984
  Avg: 68304.3484
  Std Dev: 5656.3678
----------------------------------------
--- sTtl ---
  Min: 128.0000
  Max: 128.0000
  Avg: 128.0000
  Std Dev: 0.0000
----------------------------------------
--- DstLoss ---
  Min: 2.0000
  Max: 2.0000
  Avg: 2.0000
  Std Dev: 0.0000
----------------------------------------
--- SAppBytes ---
  Min: 24.0000
  Max: 24.0000
  Avg: 24.0000
  Std Dev: 0.0000
----------------------------------------
--- pLoss ---
  Min: 18.1818
  Max: 25.0000
  Avg: 20.9091
  Std Dev: 3.7345
----------------------------------------
--- Sport ---
  Min: 50561.0000
  Max: 64932.0000
  Avg: 60210.4000
  Std Dev: 5787.8642
----------------------------------------
--- dTtl ---
  Min: 64.0000
  Max: 64.0000
  Avg: 64.0000
  Std Dev: 0.0000
----------------------------------------
--- Dport ---
  Min: 502.0000
  Max: 502.0000
  Avg: 502.0000
  Std Dev: 0.0000
----------------------------------------
Processing batch 4...
Number of records in batch 4: 5

--- Statistics for Batch 4 ---

--- DstJitter ---
  Min: 11.3636
  Max: 14.0355
  Avg: 12.2890
  Std Dev: 1.0411
----------------------------------------
--- TcpRtt ---
  Min: 0.0006
  Max: 0.0013
  Avg: 0.0009
  Std Dev: 0.0003
----------------------------------------
--- DIntPkt ---
  Min: 7.2181
  Max: 7.7709
  Avg: 7.3831
  Std Dev: 0.2362
----------------------------------------
--- DstPkts ---
  Min: 8.0000
  Max: 8.0000
  Avg: 8.0000
  Std Dev: 0.0000
----------------------------------------
--- DstBytes ---
  Min: 508.0000
  Max: 508.0000
  Avg: 508.0000
  Std Dev: 0.0000
----------------------------------------
--- SynAck ---
  Min: 0.0006
  Max: 0.0012
  Avg: 0.0009
  Std Dev: 0.0003
----------------------------------------
--- DstRate ---
  Min: 126.5983
  Max: 136.2239
  Avg: 133.0640
  Std Dev: 3.8410
----------------------------------------
--- DstLoad ---
  Min: 64384.2773
  Max: 69279.5703
  Avg: 67672.5508
  Std Dev: 1953.4299
----------------------------------------
--- sTtl ---
  Min: 128.0000
  Max: 128.0000
  Avg: 128.0000
  Std Dev: 0.0000
----------------------------------------
--- DstLoss ---
  Min: 2.0000
  Max: 2.0000
  Avg: 2.0000
  Std Dev: 0.0000
----------------------------------------
--- SAppBytes ---
  Min: 24.0000
  Max: 24.0000
  Avg: 24.0000
  Std Dev: 0.0000
----------------------------------------
--- pLoss ---
  Min: 18.1818
  Max: 18.1818
  Avg: 18.1818
  Std Dev: 0.0000
----------------------------------------
--- Sport ---
  Min: 49441.0000
  Max: 62763.0000
  Avg: 56759.8000
  Std Dev: 5925.5144
----------------------------------------
--- dTtl ---
  Min: 64.0000
  Max: 64.0000
  Avg: 64.0000
  Std Dev: 0.0000
----------------------------------------
--- Dport ---
  Min: 502.0000
  Max: 502.0000
  Avg: 502.0000
  Std Dev: 0.0000
----------------------------------------
'''
