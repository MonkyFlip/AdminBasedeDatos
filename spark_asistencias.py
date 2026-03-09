from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, to_date
from spark_config import DB_NAME, MONGO_URI
from dotenv import load_dotenv
import os

# ENV
load_dotenv()

COLECCION = os.getenv("MONGO_COLLECTION")
if not COLECCION:
    raise RuntimeError("MONGO_COLLECTION no está definida.")

# SPARK
spark = (
    SparkSession.builder
    .appName("AnalisisMongo")
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("- Spark conectado")
print(f"- Base de datos: {DB_NAME}")
print("- Colección cargada: " + COLECCION)

# LEER DATOS
df = (
    spark.read
    .format("mongodb")
    .option("database", DB_NAME)
    .option("collection", COLECCION)
    .load()
)

# NORMALIZAR FECHA
df = df.withColumn("fecha_dia", to_date(col("fecha")))

# ANÁLISIS
resumen = (
    df.groupBy("fecha_dia")
      .agg(count("*").alias("total_registros"))
      .orderBy(col("total_registros").desc())
)

# RESULTADOS
print("\n" + "=" * 50)
print(" RESUMEN POR DÍA")
print("=" * 50)

resumen.show(10, truncate=False)

spark.stop()
