from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, split, expr
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
from menelaus.change_detection import ADWIN
from pyspark.sql.functions import when, col, lit
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import joblib
import numpy as np
import time

# Load Logistic Regression Model
while True:
    try:
        logreg_model = joblib.load('/app/model/logreg_model.pkl')
        break
    except Exception as e:
        print('Error loading logistic regression model:', e)
        time.sleep(10)

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

# Initialize ADWIN Detector
adwin = ADWIN(delta=0.001) 

###########################################################################
# Helper Functions
###########################################################################

def detect_adwin_change(value):
    """
    Applies ADWIN to detect change based on prediction errors.
    """
    adwin.update(value)
    return "drift" if adwin.drift_state else "no_drift"

def make_prediction(features):
    """
    Make predictions using the logistic regression model.
    """
    features_array = np.array(features.toArray()).reshape(1, -1)
    prediction = logreg_model.predict(features_array)
    return float(prediction)

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
        point = Point("ADWINResults") \
            .field("error", row["error"]) \
            .field("prediction", row["prediction"]) \
            .field("TP", row["TP"]) \
            .field("FP", row["FP"]) \
            .field("FN", row["FN"]) \
            .field("target", row["Target"]) \
            .tag("result", row["result"]) \
            .time(row["current_timestamp"], write_precision="ms")
        
        # Write the point to InfluxDB
        write_api.write(bucket=bucket, org=org, record=point)

###########################################################################
# Spark Streaming
###########################################################################

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

# Extract key-value pairs
parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING) as raw_value") \
    .withColumn("key_value_pairs", split(col("raw_value"), ","))

# Define keys to extract
data_columns = [
    "Mean", "Sport", "Dport", "SrcPkts", "DstPkts", "TotPkts", "DstBytes", "SrcBytes", "TotBytes", "SrcLoad",
    "DstLoad", "Load", "SrcRate", "DstRate", "Rate", "SrcLoss", "DstLoss", "Loss", 
    "pLoss", "SrcJitter", "DstJitter", "SIntPkt", "DIntPkt", "Proto", "Dur", "TcpRtt",
    "IdleTime", "Sum", "Min", "Max", "sDSb", "sTtl", "dTtl", "SAppBytes", "DAppBytes", "TotAppByte", "SynAck", 
    "RunTime", "sTos", "SrcJitAct", "DstJitAct", "Target"]

# Function to extract columns from raw key-value pairs
def extract_key_value_columns(df, keys):
    for key in keys:
        df = df.withColumn(key, expr(f"split(filter(key_value_pairs, x -> x like '{key}=%')[0], '=')[1]").cast(DoubleType()))
    return df

parsed_stream = extract_key_value_columns(parsed_stream, data_columns)

# Add timestamp
parsed_stream_with_timestamp = parsed_stream.withColumn("current_timestamp", current_timestamp())

# Vectorize features
vectorize_udf = udf(lambda *args: Vectors.dense(args), VectorUDT())
vectorized_stream = parsed_stream_with_timestamp.withColumn(
    "features", vectorize_udf(*[col(f"{i}") for i in data_columns[:-1]])
).select("current_timestamp", "features", "Target")

# Make predictions
prediction_udf = udf(make_prediction, DoubleType())
predicted_stream = vectorized_stream.withColumn("prediction", prediction_udf(col("features")))

# Calculate prediction error
predicted_stream = predicted_stream.withColumn("error", (col("prediction") - col("Target")).cast(DoubleType()))

# Apply Menelaus ADWIN change detection
adwin_udf = udf(detect_adwin_change, StringType())
stream_with_change_detection = predicted_stream.withColumn(
    "result", adwin_udf(col("error"))
)

stream_with_change_detection = stream_with_change_detection.drop("features")

#Adding the True Positive, False Positive and False Negative values to the dataframe
stream_with_metrics = stream_with_change_detection.withColumn(
    "TP", when((col("result") == "drift") & (col("Target") == 1), lit(1)).otherwise(lit(0))
).withColumn(
    "FP", when((col("result") == "drift") & (col("Target") == 0), lit(1)).otherwise(lit(0))
).withColumn(
    "FN", when((col("result") == "no_drift") & (col("Target") == 1), lit(1)).otherwise(lit(0))
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

'''
Batch: 22
-------------------------------------------
+-----------------------+------+----------+-----+------------+
|current_timestamp      |Target|prediction|error|adwin_result|
+-----------------------+------+----------+-----+------------+
|2024-12-02 22:53:06.904|0.0   |0.0       |0.0  |no_drift    |
+-----------------------+------+----------+-----+------------+

24/12/02 22:53:10 WARN KafkaDataConsumer: KafkaDataConsumer is not running in UninterruptibleThread. It may hang when KafkaDataConsumer's methods are interrupted because of KAFKA-1894
-------------------------------------------
Batch: 23
-------------------------------------------
+-----------------------+------+----------+-----+------------+
|current_timestamp      |Target|prediction|error|adwin_result|
+-----------------------+------+----------+-----+------------+
|2024-12-02 22:53:09.395|0.0   |1.0       |1.0  |drift    |
+-----------------------+------+----------+-----+------------+

Batch: 4
-------------------------------------------
+-----------------------+------+----------+------------+
|current_timestamp      |Target|prediction|adwin_result|
+-----------------------+------+----------+------------+
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|1.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|1.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
|2024-12-04 02:02:00.044|0.0   |0.0       |NULL        |
+-----------------------+------+----------+------------+
only showing top 20 rows


-------------------------------------------
Batch: 1
-------------------------------------------
+------------------------------------------+------+-----+--------------+-----------------+------------+
|window                                    |Target|count|avg_prediction|stddev_prediction|adwin_result|
+------------------------------------------+------+-----+--------------+-----------------+------------+
|{2024-12-04 02:10:00, 2024-12-04 02:15:00}|0.0   |189  |0.0           |0.0              |no_drift    |
|{2024-12-04 02:10:00, 2024-12-04 02:15:00}|1.0   |16   |0.0           |0.0              |no_drift    |
+------------------------------------------+------+-----+--------------+-----------------+------------+
'''
