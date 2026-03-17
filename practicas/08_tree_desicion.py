# 08_tree.py
import streamlit as st
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from dotenv import load_dotenv

load_dotenv()

spark = SparkSession.builder \
    .appName("TreeGym") \
    .config("spark.mongodb.read.connection.uri",
            f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/{os.getenv('DB_NAME')}") \
    .getOrCreate()

df = spark.read.format("mongodb").option("collection", os.getenv("COLLECTION_NAME")).load()

data = VectorAssembler(inputCols=["peso","precio_mensual"], outputCol="features").transform(df)
model = DecisionTreeClassifier(labelCol="asiste_regularmente").fit(data)

st.write("Profundidad:", model.depth)
