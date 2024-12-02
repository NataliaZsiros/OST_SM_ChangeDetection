from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, current_timestamp, split, expr
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
from menelaus.change_detection import PageHinkley
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import when, col, lit, avg
import joblib
import numpy as np
import time

print('Starting the script:')
# Check if the PCA model has been trained in the batch processing part and the model is available

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

data_columns_only_float = [
    "Mean", "Sport", "Dport", "SrcPkts", "DstPkts", "TotPkts", "DstBytes", "SrcBytes", "TotBytes", "SrcLoad",
    "DstLoad", "Load", "SrcRate", "DstRate", "Rate", "SrcLoss", "DstLoss", "Loss", 
    "pLoss", "SrcJitter", "DstJitter", "SIntPkt", "DIntPkt", "Proto", "Dur", "TcpRtt",
    "IdleTime", "Sum", "Min", "Max", "sDSb", "sTtl", "dTtl", "SAppBytes", "DAppBytes", "TotAppByte", "SynAck", 
    "RunTime", "sTos", "SrcJitAct", "DstJitAct"]

# Apply the mapping function to extract key-value pairs into separate columns
parsed_stream = extract_key_value_columns(parsed_stream, data_columns)

# I did not delete the Target column for the estimation
parsed_stream = parsed_stream.drop("StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "key_value_pairs", \
                                   "raw_value", 'sIpId', 'dIpId')

# Add a current timestamp to each row
parsed_stream_with_timestamp = parsed_stream.withColumn("current_timestamp", current_timestamp())

vectorize_udf = udf(vectorize_features, VectorUDT())

vectorized_stream = parsed_stream_with_timestamp.withColumn(
    "features", vectorize_udf(*[col(f"{i}") for i in data_columns_only_float])
).select("current_timestamp", "features", "Target")

pca_udf = udf(apply_pca, DoubleType())

reduced_stream = vectorized_stream.withColumn(
    "reduced_dimension", pca_udf(col("features"))
)

change_detection_udf = udf(detect_change, StringType())

stream_with_change_detection = reduced_stream.withColumn(
    "change_detection", change_detection_udf(col("reduced_dimension"))
)

stream_with_change_detection = stream_with_change_detection.drop("features")

stream_with_metrics = stream_with_change_detection.withColumn(
    "TP", when((col("change_detection") == "drift") & (col("Target") == 1), lit(1)).otherwise(lit(0))
).withColumn(
    "FP", when((col("change_detection") == "drift") & (col("Target") == 0), lit(1)).otherwise(lit(0))
).withColumn(
    "FN", when((col("change_detection").isNull()) & (col("Target") == 1), lit(1)).otherwise(lit(0))
)

stream_with_metrics.printSchema()

# This doesn't work yet so I commented it
'''metrics_aggregated = stream_with_metrics.groupBy().agg(
    (when((sum(col("TP")) + sum(col("FP"))) > 0, sum(col("TP")) / (sum(col("TP")) + sum(col("FP")))).otherwise(0)).alias("Precision"),
    (when((sum(col("TP")) + sum(col("FN"))) > 0, sum(col("TP")) / (sum(col("TP")) + sum(col("FN")))).otherwise(0)).alias("Recall"),
    (
        when(
            ((sum(col("TP")) + sum(col("FP"))) > 0) & ((sum(col("TP")) + sum(col("FN"))) > 0),
            2 * (sum(col("TP")) / (sum(col("TP")) + sum(col("FP")))) * (sum(col("TP")) / (sum(col("TP")) + sum(col("FN")))) /
            ((sum(col("TP")) / (sum(col("TP")) + sum(col("FP")))) + (sum(col("TP")) / (sum(col("TP")) + sum(col("FN")))))
        ).otherwise(0)
    ).alias("F1_Score")
)

query_metrics = metrics_aggregated.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .start()'''

# Output the results to the console
query = stream_with_metrics.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()
#query_metrics.awaitTermination()
