from pyspark.sql import SparkSession
from pyspark.sql.functions import col
 
spark = SparkSession.builder \\
    .appName('KafkaStreamProcessor') \\
    .master('local[*]') \\
    .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0') \\
    .getOrCreate()
 
kafka_stream = spark.readStream \\
    .format('kafka') \\
    .option('kafka.bootstrap.servers', 'kafka:9092') \\
    .option('subscribe', 'iiot_raw_data') \\
    .load()
 
decoded_stream = kafka_stream.selectExpr('CAST(value AS STRING)')
 
query = decoded_stream.writeStream \\
    .outputMode('append') \\
    .format('console') \\
    .start()
 
query.awaitTermination()
