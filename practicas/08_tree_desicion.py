import streamlit as st
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from dotenv import load_dotenv

load_dotenv()

spark = SparkSession.builder \
    .appName("DecisionTree") \
    .config("spark.mongodb.read.connection.uri",
            f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/{os.getenv('DB_NAME')}") \
    .getOrCreate()

df = spark.read.format("mongodb").option("collection", os.getenv("COLLECTION_NAME")).load()

assembler = VectorAssembler(inputCols=["paginas","precio"], outputCol="features")
data = assembler.transform(df)

model = DecisionTreeClassifier(labelCol="disponible").fit(data)
st.write(model.depth, model.numNodes)
