# 09_mapreduce.py
import streamlit as st
import os
from pyspark.sql import SparkSession
from dotenv import load_dotenv

load_dotenv()

spark = SparkSession.builder \
    .appName("MapReduceGym") \
    .config("spark.mongodb.read.connection.uri",
            f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/{os.getenv('DB_NAME')}") \
    .getOrCreate()

df = spark.read.format("mongodb").option("collection", os.getenv("COLLECTION_NAME")).load()
st.dataframe(df.groupBy("tipo_membresia").count().toPandas())
