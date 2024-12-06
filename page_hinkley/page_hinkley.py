from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, split, expr
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
from menelaus.change_detection import PageHinkley
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import when, col, lit
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from pyspark.sql.functions import to_json, struct
import joblib
import numpy as np
import time

print('Starting the script:')
# Check if the PCA model has been trained in the batch processing part and the model is available

url = "http://influxdb:8086"
token = "9FEx1XT4dRY-7H65r2ByRsz-XTlvaGlMN9itr9fMWxdw_K6TK7n7skk9p-wr55aZ3rf8sWnEZ24fSrwEd7V0qQ=="  
org = "ChangeDetection_org"
bucket = "ChangeDetection"
username = "admin"
password = "password"

print('Connecting to InfluxDB...')
while True:
    try:
        client = InfluxDBClient(url=url, token=token, org=org)
        print(client.ping())  # Test connection
        break
    except Exception as e:
        print(f"Retrying InfluxDB connection... {e}")
        time.sleep(5)

write_api = client.write_api(write_options=SYNCHRONOUS)

print('Trying to access the model...')
while True:
    try:
        pca_model = joblib.load('/app/model/pca_model.pkl')
        break
    except Exception as e:
        print('Error: ', e)
        print('Waiting 10 seconds...')
        time.sleep(10)

print("PCA model is available")

# Initialize Page-Hinkley Detector
ph = PageHinkley(delta=0.001, threshold=1, direction="negative", burn_in=1)

###########################################################################
#Functions

# Combine features into a single vector column

# Define a mapping function to split key-value pairs into columns
def extract_key_value_columns(df, keys):
    for key in keys:
        key_col = f"{key}_pair"
        df = df.withColumn(key_col, expr(f"filter(key_value_pairs, x -> x like '{key}=%')")[0])
        
        # Step 2: Extract value after the '=' sign
        if key not in ["StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic",]:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(DoubleType()))  # Cast to numeric for calculations
        else:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(StringType()))  # Cast to numeric for calculations
        
        # Drop intermediate key-value column
        df = df.drop(key_col)
    return df

def vectorize_features(*args):
    return Vectors.dense(args)

# Apply Page-Hinkley for Change Detection
def detect_change(value):
    """Applies Page-Hinkley algorithm to detect change."""
    ph.update(value)
    return ph.drift_state  # Outputs: NO_DRIFT or DRIFT

# Apply PCA using a Scikit-learn model
def apply_pca(features):
    """Apply saved PCA model to Spark Vector."""
    array_features = np.array(features.toArray()).reshape(1, -1)

    reduced = pca_model.transform(array_features)
    return float(reduced[0][0])

# Writing to InfluxDB for visualisation
def write_to_influxdb(batch_df, batch_id):
    """
    Write a batch of data to InfluxDB.
    """
    # Convert the Spark DataFrame to Pandas 
    batch_pd = batch_df.toPandas()
    print(batch_pd.columns)
    
    for index, row in batch_pd.iterrows():
        # Add here the additional change detection method results (if it detected change or not)
        # and add the calculated true positive, false positive and false negative values
        point = Point("PageHinkleyResults") \
            .field("DstJitter", row["DstJitter"]) \
            .field("TP", row["TP"]) \
            .field("FP", row["FP"]) \
            .field("FN", row["FN"]) \
            .field("target", row["Target"]) \
            .tag("result", row["result"]) \
            .time(row["current_timestamp"], write_precision="ms")
        
        # Write the point to InfluxDB
        write_api.write(bucket=bucket, org=org, record=point)

###########################################################################

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

parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING) as raw_value") \
    .withColumn("key_value_pairs", split(col("raw_value"), ","))  # Split by comma to get individual key-value pairs

# List of keys to extract 
data_columns = ["DstJitter", "Target"]

data_columns_only_float = [
    "Mean", "Sport", "Dport", "SrcPkts", "DstPkts", "TotPkts", "DstBytes", "SrcBytes", "TotBytes", "SrcLoad",
    "DstLoad", "Load", "SrcRate", "DstRate", "Rate", "SrcLoss", "DstLoss", "Loss", 
    "pLoss", "SrcJitter", "DstJitter", "SIntPkt", "DIntPkt", "Proto", "Dur", "TcpRtt",
    "IdleTime", "Sum", "Min", "Max", "sDSb", "sTtl", "dTtl", "SAppBytes", "DAppBytes", "TotAppByte", "SynAck", 
    "RunTime", "sTos", "SrcJitAct", "DstJitAct"]

# Apply the mapping function to extract key-value pairs into separate columns
parsed_stream = extract_key_value_columns(parsed_stream, data_columns)

# I did not delete the Target column for the estimation
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

#Adding the True Positive, False Positive and False Negative values to the dataframe
stream_with_metrics = stream_with_detection.withColumn(
    "TP", when((col("result") == "drift") & (col("Target") == 1), lit(1)).otherwise(lit(0))
).withColumn(
    "FP", when((col("result") == "drift") & (col("Target") == 0), lit(1)).otherwise(lit(0))
).withColumn(
    "FN", when((col("result").isNull()) & (col("Target") == 1), lit(1)).otherwise(lit(0))
)

stream_with_metrics.printSchema()

#writing the results to the console and also writing it to influxdb
query = stream_with_metrics.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .foreachBatch(write_to_influxdb) \
    .start()


query.awaitTermination()
