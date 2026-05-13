# Dashboard de Ventas con Spark + MongoDB + Seaborn + Streamlit

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, avg, col
from dotenv import load_dotenv
from pathlib import Path
import os
from urllib.parse import quote_plus
import pandas as pd
import streamlit as st

# 1. Cargar variables de entorno
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

user = os.getenv("MONGO_USER")
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
database = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")

mongo_uri = f"mongodb+srv://{user}:{password}@{cluster}"

# 2. Configurar Spark con conector MongoDB
spark = SparkSession.builder \
    .appName("BigDataVentas") \
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.13:10.3.0") \
    .config("spark.mongodb.read.connection.uri", mongo_uri) \
    .config("spark.mongodb.read.database", database) \
    .config("spark.mongodb.read.collection", collection_name) \
    .getOrCreate()

# 3. Leer datos desde MongoDB
df = spark.read.format("mongodb").load()

# 4. Agregación en Spark
resultado = df.groupBy("producto") \
    .agg(
        sum("cantidad").alias("total_cantidad"),
        avg("precio").alias("promedio_precio"),
        sum(col("cantidad") * col("precio")).alias("ingresos_totales")
    ) \
    .orderBy("ingresos_totales", ascending=False)

# 5. Convertir a Pandas
pdf = resultado.toPandas()

# 6. Exportar resultados
pdf.to_csv("ventas_resumen.csv", index=False) #no se requiere librerias adicionales ya que es universal y ligero
pdf.to_excel("ventas_resumen.xlsx", index=False) #se requiere la instalación de openpyxl para exportar a excel

# 7. Dashboard con Streamlit
st.title(" Dashboard de Ventas")

st.write("### Datos agregados")
st.dataframe(pdf)

# Gráfico 1: Cantidad total vendida por producto
st.write("### Total de ventas por producto")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(x="producto", y="total_cantidad", data=pdf, palette="viridis", ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Gráfico 2: Promedio de precio por producto
st.write("### Promedio de precio por producto")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(x="producto", y="promedio_precio", data=pdf, palette="magma", ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# Gráfico 3: Ingresos totales por producto
st.write("### Ingresos totales por producto")
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.barplot(x="producto", y="ingresos_totales", data=pdf, palette="coolwarm", ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

st.success(" Resultados exportados a 'ventas_resumen.csv' y 'ventas_resumen.xlsx'")