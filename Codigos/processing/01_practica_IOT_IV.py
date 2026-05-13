from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, min, col

#1.- Crear sesión Spark
spark = SparkSession.builder \
    .appName("IoT_BigData") \
    .getOrCreate()

#2.- Crear datos simulados IoT en memoria
datos_iot = [
    ("sensor1", 28, 60),
    ("sensor1", 30, 65),
    ("sensor2", 22, 55),
    ("sensor2", 35, 80),
    ("sensor3", 40, 90),
    ("sensor1", 27, 58),
    ("sensor1", 45, 70)
]

columnas = ["dispositivo", "temperatura", "humedad"]

df = spark.createDataFrame(datos_iot, columnas)

print("\n=== DATOS IoT GENERADOS ===")
df.show()

#3.- Promedio de temperatura por sensor
promedio = df.groupBy("dispositivo") \
             .agg(avg("temperatura").alias("temp_promedio"))

print("\n=== PROMEDIO POR SENSOR ===")
promedio.show()

#4.- Temperatura máxima general
maxima = df.agg(max("temperatura").alias("temp_max"))
print("\n=== TEMPERATURA MÁXIMA ===")
maxima.show()

#5.- Detectar temperaturas críticas (>35)
alertas = df.filter(col("temperatura") > 40)

print("\n=== ALERTAS DE TEMPERATURA CRÍTICA ===")
alertas.show()

spark.stop()