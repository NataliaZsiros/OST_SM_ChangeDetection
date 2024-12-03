from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, split, expr
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
from menelaus.change_detection import ADWIN
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
    "adwin_result", adwin_udf(col("error"))
)

stream_with_change_detection = stream_with_change_detection.drop("features")

# Output to console
query = stream_with_change_detection.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
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
|2024-12-02 22:53:09.395|0.0   |0.0       |0.0  |no_drift    |
+-----------------------+------+----------+-----+------------+
'''
