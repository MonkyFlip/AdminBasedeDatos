from pyspark.sql import SparkSession
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

mongo_uri = f"mongodb+srv://{user}:{password}@{cluster}"

spark = SparkSession.builder \
    .appName("BigDataVentas") \
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.13:10.3.0") \
    .config("spark.mongodb.read.connection.uri", mongo_uri) \
    .config("spark.mongodb.read.database", database) \
    .config("spark.mongodb.read.collection", collection_name) \
    .config("spark.mongodb.write.connection.uri", mongo_uri) \
    .getOrCreate()

# Leer datos de MongoDB
df = spark.read.format("mongodb").load()

# Mostrar los primeros 10 registros
df.show(10)

# Conteo por producto
df.groupBy("producto").count().show()