import streamlit as st
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from dotenv import load_dotenv

load_dotenv()

spark = SparkSession.builder \
    .appName("KMeans") \
    .config("spark.mongodb.read.connection.uri",
            f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/{os.getenv('DB_NAME')}") \
    .getOrCreate()

df = spark.read.format("mongodb").option("collection", os.getenv("COLLECTION_NAME")).load()

assembler = VectorAssembler(inputCols=["paginas", "precio"], outputCol="features")
data = assembler.transform(df)

model = KMeans(k=3).fit(data)
st.dataframe(model.summary.predictions.select("paginas","precio","prediction").limit(20).toPandas())
