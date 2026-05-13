from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, avg
from dotenv import load_dotenv
from pathlib import Path
import os
from urllib.parse import quote_plus

# Cargar .env desde la raíz del proyecto
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Leer variables
user = os.getenv("MONGO_USER")
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
database = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")

if not all([user, password, cluster, database, collection_name]):
    raise ValueError("Faltan variables en el archivo .env")

# Construir URI segura
mongo_uri = f"mongodb+srv://{user}:{password}@{cluster}"

# Configurar Spark con el conector de MongoDB
spark = SparkSession.builder \
    .appName("BigDataVentas") \
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.13:10.3.0") \
    .config("spark.mongodb.read.connection.uri", mongo_uri) \
    .config("spark.mongodb.read.database", database) \
    .config("spark.mongodb.read.collection", collection_name) \
    .getOrCreate()

# Leer datos de MongoDB
df = spark.read.format("mongodb").load()

# Mostrar esquema para verificar
df.printSchema()

# Agregación: total de ventas y promedio de precio por producto
resultado = df.groupBy("producto") \
    .agg(
        sum("cantidad").alias("total_cantidad"),
        avg("precio").alias("promedio_precio")
    ) \
    .orderBy("total_cantidad", ascending=False)

# Mostrar resultados
resultado.show()