from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, current_timestamp, split, expr
from pyspark.sql.functions import min, max, avg, stddev
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import joblib
import numpy as np
import time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import WriteOptions

# InfluxDB connection details
INFLUXDB_URL = "http://influxdb:8086"
INFLUXDB_TOKEN = "9FEx1XT4dRY-7H65r2ByRsz-XTlvaGlMN9itr9fMWxdw_K6TK7n7skk9p-wr55aZ3rf8sWnEZ24fSrwEd7V0qQ=="
INFLUXDB_ORG = "ChangeDetection_org"
INFLUXDB_BUCKET = "ChangeDetection"

# Initialize InfluxDB Client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

# Configure the write options
write_options = WriteOptions(batch_size=1000, flush_interval=10000)

# Create the Write API with the defined write options
write_api = client.write_api(write_options=write_options)

# Define a mapping function to split key-value pairs into columns
def extract_key_value_columns(df, keys):
    for key in keys:
        key_col = f"{key}_pair"
        df = df.withColumn(key_col, expr(f"filter(key_value_pairs, x -> x like '{key}=%')")[0])
        
        # Step 2: Extract value after the '=' sign
        if key not in ["StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic"]:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(DoubleType()))  # Cast to numeric for calculations
        else:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(StringType()))  # Cast to string these will be dropped
        
        # Drop intermediate key-value column
        df = df.drop(key_col)
    return df

# Connecting Apache Spark to the Kafka topic 'input' if fails it waits 5 second and retries
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

# Splitting the string on commas to have the data into key-value pairs eg. 'Mean=1.5'
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

# Dropping unique string columns and also the target and traffic which wouldn't be available in real life
parsed_stream = parsed_stream.drop("StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "Target", "key_value_pairs", "raw_value")

# Add a current timestamp to each row - we might not need this but it was said that the system needs a timestamp to function properly
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

    # Write to InfluxDB with measurements and fields
    for _, row in stats_pd.iterrows():
        point = Point("batch_statistics") \
            .tag("batch_id", batch_id) \
            .field("Min_DstJitter", row["Min_DstJitter"]) \
            .field("Max_DstJitter", row["Max_DstJitter"]) \
            .field("Avg_DstJitter", row["Avg_DstJitter"]) \
            .field("Std_DstJitter", row["Std_DstJitter"]) \
            .field("Min_TcpRtt", row["Min_TcpRtt"]) \
            .field("Max_TcpRtt", row["Max_TcpRtt"]) \
            .field("Avg_TcpRtt", row["Avg_TcpRtt"]) \
            .field("Std_TcpRtt", row["Std_TcpRtt"]) \
            .field("Min_DstPkts", row["Min_DstPkts"]) \
            .field("Max_DstPkts", row["Max_DstPkts"]) \
            .field("Avg_DstPkts", row["Avg_DstPkts"]) \
            .field("Std_DstPkts", row["Std_DstPkts"]) \
            # Add other fields for each metric you want to track
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

    print(f"Batch {batch_id} data written to InfluxDB.")

# Write stream and process each batch using foreachBatch
query = parsed_stream_with_timestamp.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("update") \
    .trigger(processingTime='10 seconds') \
    .start()

query.awaitTermination()
