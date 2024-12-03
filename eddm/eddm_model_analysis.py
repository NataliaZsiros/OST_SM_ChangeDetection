import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, current_timestamp, split, expr, when, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT
from menelaus.concept_drift import EDDM
import numpy as np
import time

print('Starting the script...')

# Initialize EDDM Detector
eddm = EDDM(warning_thresh=0.95, drift_thresh=0.9)

# Load the pre-trained Logistic regression model
print("Loading the Logistic regression model...")
logreg_model = joblib.load('/app/model/logreg_model.pkl')
print("Model loaded successfully!")

###########################################################################
# Functions

# Combine features into a single vector column
def extract_key_value_columns(df, keys):
    for key in keys:
        key_col = f"{key}_pair"
        df = df.withColumn(key_col, expr(f"filter(key_value_pairs, x -> x like '{key}=%')")[0])
        
        # Extract value after the '=' sign
        if key not in ["StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic"]:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(DoubleType()))  # Cast to numeric for calculations
        else:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(StringType()))  # Leave as string
        df = df.drop(key_col)
    return df

def vectorize_features(*args):
    return Vectors.dense(args)

# Predict using the Logistic regression model
def predict_lr(features):
    """Predicts using the loaded Logistic regression model."""
    # Convert PySpark VectorUDT to a NumPy array
    features_array = np.array(features.toArray()).reshape(1, -1)
    return float(logreg_model.predict(features_array)[0])  # Assuming regression for continuous output

# Apply EDDM for Change Detection
def detect_change(y_true, y_pred):
    """Applies EDDM algorithm to detect change."""
    # EDDM expects a sequence of true and predicted values; it's meant to be updated in an ongoing fashion
    eddm.update(y_true, y_pred)
    return eddm.drift_state  # Outputs: 'drift', 'warning', or 'no_drift'

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
    "RunTime", "sTos", "SrcJitAct", "DstJitAct"
]

# Apply the mapping function to extract key-value pairs into separate columns
parsed_stream = extract_key_value_columns(parsed_stream, data_columns)

# Drop unnecessary columns
parsed_stream = parsed_stream.drop("StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "key_value_pairs", \
                                   "raw_value", 'sIpId', 'dIpId')

# Add a current timestamp to each row
parsed_stream_with_timestamp = parsed_stream.withColumn("current_timestamp", current_timestamp())

vectorize_udf = udf(vectorize_features, VectorUDT())

vectorized_stream = parsed_stream_with_timestamp.withColumn(
    "features", vectorize_udf(*[col(f"{i}") for i in data_columns_only_float])
).select("current_timestamp", "features", "Target")

predict_udf = udf(predict_lr, DoubleType())

predicted_stream = vectorized_stream.withColumn(
    "prediction", predict_udf(col("features"))
)

# Calculate error
predicted_stream = predicted_stream.withColumn(
    "error", expr("abs(Target - prediction)")
)

# Apply change detection
change_detection_udf = udf(lambda y_true, y_pred: detect_change(y_true, y_pred), StringType())

stream_with_change_detection = predicted_stream.withColumn(
    "eddm_result", change_detection_udf(col("Target"), col("prediction"))
)


stream_with_change_detection = stream_with_change_detection.withColumn(
    "eddm_result", when(col("eddm_result").isNull(), "no_drift").otherwise(col("eddm_result"))
)

# Display schema
stream_with_metrics = stream_with_change_detection.select(
    "current_timestamp", "Target", "prediction", "error", "eddm_result"
)

stream_with_metrics.printSchema()

# Output the results to the console
query = stream_with_metrics.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()



