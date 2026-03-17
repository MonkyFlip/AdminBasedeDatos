import streamlit as st
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.functions import col
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

st.title("Árbol de Decisión: Asistencia al Gimnasio")
st.write(
    "En esta práctica se utiliza un árbol de decisión para clasificar "
    "si un cliente asiste regularmente al gimnasio, utilizando su peso "
    "y el precio mensual que paga."
)

# Spark session
spark = SparkSession.builder \
    .appName("DecisionTreeGym") \
    .config(
        "spark.mongodb.read.connection.uri",
        f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/{os.getenv('DB_NAME')}"
    ) \
    .getOrCreate()

# Leer datos
df = spark.read.format("mongodb") \
    .option("collection", os.getenv("COLLECTION_NAME")) \
    .load()

# Convertir booleano a entero
df = df.withColumn(
    "asiste_regularmente_num",
    col("asiste_regularmente").cast("integer")
)

# Preparar datos
assembler = VectorAssembler(
    inputCols=["peso", "precio_mensual"],
    outputCol="features"
)
data = assembler.transform(df)

# Modelo
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="asiste_regularmente_num"
)
model = dt.fit(data)

# Mostrar métricas
st.subheader("Información del modelo")
st.write("Profundidad del árbol:", model.depth)
st.write("Número de nodos:", model.numNodes)

# Mostrar distribución de clases
pdf = df.select("asiste_regularmente_num").toPandas()

fig, ax = plt.subplots()
pdf["asiste_regularmente_num"].value_counts().plot(
    kind="bar",
    ax=ax,
    color=["skyblue", "salmon"]
)
ax.set_xlabel("Asiste regularmente (0 = No, 1 = Sí)")
ax.set_ylabel("Cantidad de clientes")
ax.set_title("Distribución de asistencia al gimnasio")

st.pyplot(fig)
