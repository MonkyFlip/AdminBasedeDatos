from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

spark = SparkSession.builder \
    .appName("AnalisisRiesgoUsuarios") \
    .getOrCreate()

datos = [
    ("admin", "FAIL"),
    ("admin", "FAIL"),
    ("admin", "FAIL"),
    ("user1", "FAIL"),
    ("user1", "OK"),
    ("root", "FAIL"),
    ("root", "FAIL"),
    ("root", "FAIL"),
    ("root", "FAIL"),
]

df = spark.createDataFrame(datos, ["usuario", "estatus"])

# Contar fallos por usuario
conteo = df.filter(col("estatus") == "FAIL") \
           .groupBy("usuario") \
           .agg(count("*").alias("fallos"))

# Clasificación de riesgo
riesgo = conteo.withColumn(
    "nivel_riesgo",
    when(col("fallos") >= 4, "ALTO")
    .when(col("fallos") >= 2, "MEDIO")
    .otherwise("BAJO")
)

print("\n=== CLASIFICACIÓN DE RIESGO ===")
riesgo.show()

spark.stop()