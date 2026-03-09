# spark_connection.py
import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession

load_dotenv()

user = os.getenv("MONGO_USER")
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")
db_name = os.getenv("MONGO_DB")
collection = os.getenv("MONGO_COLLECTION")
app_name = os.getenv("MONGO_APPNAME", "App")

uri = (
    f"mongodb+srv://{user}:{password}@{cluster}/"
    f"?retryWrites=true&w=majority&appName={app_name}"
)

def get_spark_session():
    spark = (
        SparkSession.builder
        .appName("SparkMongoMapReduce")
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
        .config("spark.mongodb.read.connection.uri", uri)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read
        .format("mongodb")
        .option("database", db_name)
        .option("collection", collection)
        .load()
    )

    return spark, df, collection
