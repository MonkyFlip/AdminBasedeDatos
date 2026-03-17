import os
from pyspark.sql import SparkSession
from dotenv import load_dotenv

load_dotenv()

def get_spark_df(app_name):
    uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/{os.getenv('DB_NAME')}"
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.mongodb.read.connection.uri", uri)
        .getOrCreate()
    )
    return spark.read.format("mongodb") \
        .option("collection", os.getenv("COLLECTION_NAME")) \
        .load()
