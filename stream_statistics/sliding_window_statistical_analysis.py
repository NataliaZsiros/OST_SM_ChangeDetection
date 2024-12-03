from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, current_timestamp, split, expr, window, count, format_number
from pyspark.sql.functions import min, max, avg, stddev
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import time

# Define a mapping function to split key-value pairs into columns
def extract_key_value_columns(df, keys):
    for key in keys:
        key_col = f"{key}_pair"
        df = df.withColumn(key_col, expr(f"filter(key_value_pairs, x -> x like '{key}=%')")[0])
        
        if key not in ["StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic"]:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(DoubleType()))  # Numeric
        else:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(StringType()))  # String
        
        df = df.drop(key_col)
    return df

# Connect to Kafka
while True:
    try:
        spark = SparkSession.builder \
            .appName("SlidingWindowStreamProcessor") \
            .master("local[*]") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
            .getOrCreate()

        # Increase the max fields displayed in Spark's console output
        spark.conf.set("spark.sql.debug.maxToStringFields", "1000")

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

# Parse the Kafka stream
parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING) as raw_value") \
    .withColumn("key_value_pairs", split(col("raw_value"), ","))

# List of keys to extract
data_columns = [
    "StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "Mean", "Sport", "Dport",
    "SrcPkts", "DstPkts", "TotPkts", "DstBytes", "SrcBytes", "TotBytes", "SrcLoad",
    "DstLoad", "Load", "SrcRate", "DstRate", "Rate", "SrcLoss", "DstLoss", "Loss", 
    "pLoss", "SrcJitter", "DstJitter", "SIntPkt", "DIntPkt", "Proto", "Dur", "TcpRtt",
    "IdleTime", "Sum", "Min", "Max", "sDSb", "sTtl", "dTtl", "sIpId", "dIpId", 
    "SAppBytes", "DAppBytes", "TotAppByte", "SynAck", "RunTime", "sTos", "SrcJitAct",
    "DstJitAct", "Target"
]

# Extract key-value pairs into columns
parsed_stream = extract_key_value_columns(parsed_stream, data_columns)

# Drop unnecessary columns
parsed_stream = parsed_stream.drop("StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "Target", "key_value_pairs", "raw_value")

# Add a current timestamp column
parsed_stream_with_timestamp = parsed_stream.withColumn("current_timestamp", current_timestamp())

# Define the sliding window
windowed_stream = parsed_stream_with_timestamp.groupBy(
    window(col("current_timestamp"), "60 seconds", "30 seconds")
).agg(
    *[min(col(c)).alias(f"Min_{c}") for c in ['DstJitter']],
    *[max(col(c)).alias(f"Max_{c}") for c in ['DstJitter']],
    *[avg(col(c)).alias(f"Avg_{c}") for c in ['DstJitter']],
    *[stddev(col(c)).alias(f"Std_{c}") for c in ['DstJitter']],
    count("*").alias("count")  # Add count of records in each window
)

# Show both start and end of the window separately, without concatenating
windowed_stream = windowed_stream.withColumn(
    "window_start", col("window.start").cast("string")
).withColumn(
    "window_end", col("window.end").cast("string")
).drop("window")

# Apply .4f formatting to numerical columns (Min, Max, Avg, Std for DstJitter)
windowed_stream = windowed_stream.withColumn(
    "Min_DstJitter", format_number(col("Min_DstJitter"), 4)
).withColumn(
    "Max_DstJitter", format_number(col("Max_DstJitter"), 4)
).withColumn(
    "Avg_DstJitter", format_number(col("Avg_DstJitter"), 4)
).withColumn(
    "Std_DstJitter", format_number(col("Std_DstJitter"), 4)
)

# Reorder columns to place window_start and window_end in front
windowed_stream = windowed_stream.select(
    "window_start", "window_end", 
    "Min_DstJitter", "Max_DstJitter", "Avg_DstJitter", "Std_DstJitter", "count"
)

# Write the windowed stream to the console
query = windowed_stream.writeStream \
    .outputMode("update") \
    .format("console") \
    .start()

query.awaitTermination()

'''
-------------------------------------------
Batch: 1
-------------------------------------------
+-------------------+-------------------+-------------+-------------+-------------+-------------+------+
|       window_start|         window_end|Min_DstJitter|Max_DstJitter|Avg_DstJitter|Std_DstJitter| count|
+-------------------+-------------------+-------------+-------------+-------------+-------------+------+
|2024-12-03 15:58:00|2024-12-03 15:59:00|       0.0000|  23,471.7092|      14.1538|     149.7848|312031|
|2024-12-03 15:58:30|2024-12-03 15:59:30|       0.0000|      73.2248|      16.1735|      18.6661|    20|
+-------------------+-------------------+-------------+-------------+-------------+-------------+------+

-------------------------------------------
Batch: 2
-------------------------------------------
+-------------------+-------------------+-------------+-------------+-------------+-------------+-----+
|       window_start|         window_end|Min_DstJitter|Max_DstJitter|Avg_DstJitter|Std_DstJitter|count|
+-------------------+-------------------+-------------+-------------+-------------+-------------+-----+
|2024-12-03 15:59:00|2024-12-03 16:00:00|       0.0000|      12.8251|       9.6938|       4.1033|    9|
|2024-12-03 15:58:30|2024-12-03 15:59:30|       0.0000|      73.2248|      14.1626|      15.8287|   29|
+-------------------+-------------------+-------------+-------------+-------------+-------------+-----+

-------------------------------------------
Batch: 3
-------------------------------------------
+-------------------+-------------------+-------------+-------------+-------------+-------------+-----+
|       window_start|         window_end|Min_DstJitter|Max_DstJitter|Avg_DstJitter|Std_DstJitter|count|
+-------------------+-------------------+-------------+-------------+-------------+-------------+-----+
|2024-12-03 15:59:00|2024-12-03 16:00:00|       0.0000|      14.5169|       9.9842|       4.1916|   17|
|2024-12-03 15:58:30|2024-12-03 15:59:30|       0.0000|      73.2248|      13.3298|      14.1942|   37|
+-------------------+-------------------+-------------+-------------+-------------+-------------+-----+
'''
