import streamlit as st
import os
from pyspark.sql import SparkSession
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

st.title("MapReduce: Clientes por Tipo de Membresía")
st.write(
    "En esta práctica se utiliza el paradigma MapReduce para contar "
    "cuántos clientes existen por cada tipo de membresía del gimnasio."
)

# Spark session
spark = SparkSession.builder \
    .appName("MapReduceGym") \
    .config(
        "spark.mongodb.read.connection.uri",
        f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/{os.getenv('DB_NAME')}"
    ) \
    .getOrCreate()

# Leer datos
df = spark.read.format("mongodb") \
    .option("collection", os.getenv("COLLECTION_NAME")) \
    .load()

# MapReduce
result = df.groupBy("tipo_membresia").count()
pdf = result.toPandas()

# Mostrar tabla
st.subheader("Conteo de clientes por tipo de membresía")
st.dataframe(pdf)

# Gráfica
fig, ax = plt.subplots()
ax.bar(pdf["tipo_membresia"], pdf["count"], color="mediumseagreen")
ax.set_xlabel("Tipo de membresía")
ax.set_ylabel("Cantidad de clientes")
ax.set_title("Distribución de clientes por tipo de membresía")

st.pyplot(fig)
