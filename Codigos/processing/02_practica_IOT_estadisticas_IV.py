from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, min, stddev, col

spark = SparkSession.builder \
    .appName("IoT_Estadistico") \
    .getOrCreate()

datos = [
    ("sensor1", 28),
    ("sensor1", 30),
    ("sensor1", 27),
    ("sensor2", 22),
    ("sensor2", 35),
    ("sensor2", 45),
    ("sensor3", 40),
    ("sensor3", 42),
    ("sensor3", 39),
]
                                   # x              y 
df = spark.createDataFrame(datos, ["sensor", "temperatura"])

# Métricas por sensor
metricas = df.groupBy("sensor").agg(
    avg("temperatura").alias("temperatura promedio"),
    max("temperatura").alias("temperatura maxima"),
    min("temperatura").alias("temperatura minima"),
    stddev("temperatura").alias("desviacion")
)

print("\n=== MÉTRICAS POR SENSOR ===")
metricas.show()

# Detectar temperaturas fuera de rango (>40)
alertas = df.filter(col("temperatura") > 40)

print("\n=== ALERTAS ===")
alertas.show()

spark.stop()