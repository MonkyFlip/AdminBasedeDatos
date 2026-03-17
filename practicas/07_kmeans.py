import streamlit as st
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

st.title("K-Means: Segmentación de Clientes del Gimnasio")
st.write(
    "En esta práctica se agrupan los clientes del gimnasio utilizando "
    "su peso y el precio mensual que pagan, con el objetivo de identificar "
    "perfiles similares de clientes."
)

# Spark session
spark = SparkSession.builder \
    .appName("KMeansGym") \
    .config(
        "spark.mongodb.read.connection.uri",
        f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/{os.getenv('DB_NAME')}"
    ) \
    .getOrCreate()

# Leer datos
df = spark.read.format("mongodb") \
    .option("collection", os.getenv("COLLECTION_NAME")) \
    .load()

# Preparar datos
assembler = VectorAssembler(
    inputCols=["peso", "precio_mensual"],
    outputCol="features"
)
data = assembler.transform(df)

# K-Means
kmeans = KMeans(k=3, seed=42)
model = kmeans.fit(data)

result = model.summary.predictions.select(
    "peso", "precio_mensual", "prediction"
)

# Mostrar tabla
st.subheader("Clientes agrupados por cluster")
st.dataframe(result.limit(20).toPandas())

# Gráfica
pdf = result.toPandas()

fig, ax = plt.subplots()
scatter = ax.scatter(
    pdf["peso"],
    pdf["precio_mensual"],
    c=pdf["prediction"],
    cmap="viridis",
    alpha=0.6
)

ax.set_xlabel("Peso")
ax.set_ylabel("Precio mensual")
ax.set_title("Clusters de clientes según peso y precio mensual")
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)

st.pyplot(fig)
