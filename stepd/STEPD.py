import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, current_timestamp, split, expr, when, lit
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
import time
from menelaus.concept_drift import STEPD
from influxdb_client import InfluxDBClient, Point   
from influxdb_client.client.write_api import SYNCHRONOUS

print('Starting the script...')

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

print('Loading the Random Forest model...')

rf_model = joblib.load('/app/model/rf_model.pkl')
rf_features = joblib.load('/app/model/rf_features.pkl')

stepd = STEPD()

###########################################################################
# Functions

def extract_key_value_columns(df, keys):
    for key in keys:
        key_col = f"{key}_pair"
        df = df.withColumn(key_col, expr(f"filter(key_value_pairs, x -> x like '{key}=%')")[0])

        if key not in ["StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic"]:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(DoubleType()))
        else:
            df = df.withColumn(key, expr(f"split({key_col}, '=')[1]").cast(StringType()))
        df = df.drop(key_col)
    return df

def vectorize_features(*args):
    return Vectors.dense(args)

def predict_rf(features):
    """Predicts using the loaded Random Forest model."""
    features_array = np.array(features.toArray()).reshape(1, -1)
    return float(rf_model.predict(features_array)[0])

def detect_stepd_change(y_true, y_pred):
    stepd.update(y_true, y_pred)
    return stepd.drift_state

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
        point = Point("STEPDResults") \
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

while True:
    try:
        spark = SparkSession.builder \
            .appName("KafkaStreamProcessor") \
            .master("local[*]") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
            .getOrCreate()

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
    .withColumn("key_value_pairs", split(col("raw_value"), ","))

data_columns = [
    "StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "Mean", "Sport", "Dport",
    "SrcPkts", "DstPkts", "TotPkts", "DstBytes", "SrcBytes", "TotBytes", "SrcLoad",
    "DstLoad", "Load", "SrcRate", "DstRate", "Rate", "SrcLoss", "DstLoss", "Loss", 
    "pLoss", "SrcJitter", "DstJitter", "SIntPkt", "DIntPkt", "Proto", "Dur", "TcpRtt",
    "IdleTime", "Sum", "Min", "Max", "sDSb", "sTtl", "dTtl", "sIpId", "dIpId", 
    "SAppBytes", "DAppBytes", "TotAppByte", "SynAck", "RunTime", "sTos", "SrcJitAct",
    "DstJitAct", "Target"
]

data_columns_only_float = rf_features

parsed_stream = extract_key_value_columns(parsed_stream, data_columns)
parsed_stream = parsed_stream.drop("StartTime", "LastTime", "SrcAddr", "DstAddr", "Traffic", "key_value_pairs", \
                                   "raw_value", 'sIpId', 'dIpId')

parsed_stream_with_timestamp = parsed_stream.withColumn("current_timestamp", current_timestamp())

vectorize_udf = udf(vectorize_features, VectorUDT())

vectorized_stream = parsed_stream_with_timestamp.withColumn(
    "features", vectorize_udf(*[col(f"{i}") for i in data_columns_only_float])
).select("current_timestamp", "features", "Target")

predict_udf = udf(predict_rf, DoubleType())

predicted_stream = vectorized_stream.withColumn(
    "prediction", predict_udf(col("features"))
)

# predicted_stream = predicted_stream.withColumn(
#     "error", expr("abs(Target - prediction)")
# )

stepd_udf = udf(detect_stepd_change, StringType())

# Apply the STEPD change detection
stream_with_change_detection = predicted_stream.withColumn(
    "result", stepd_udf(col("Target"), col("prediction"))
)

stream_with_change_detection = stream_with_change_detection.withColumn(
    "result", when(col("result").isNull(), "no_drift").otherwise(col("result"))
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

query = stream_with_metrics.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .foreachBatch(write_to_influxdb) \
    .start()

query.awaitTermination()
