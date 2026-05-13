from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, col

spark = SparkSession.builder.appName("ConsumoIoT").getOrCreate()

datos = [
    ("sensor1", 5),
    ("sensor1", 7),
    ("sensor2", 10),
    ("sensor2", 15),
    ("sensor3", 3),
    ("sensor3", 4),
]

df = spark.createDataFrame(datos, ["sensor", "consumo_kwh"])

consumo_total = df.groupBy("sensor").agg(
    sum("consumo_kwh").alias("consumo_total")
)

print("\n=== CONSUMO TOTAL POR SENSOR ===")
consumo_total.show()

# Detectar sensores con consumo alto (>15 kWh)
alertas = consumo_total.filter(col("consumo_total") > 15)

print("\n=== SENSORES CON CONSUMO ALTO ===")
alertas.show()

spark.stop()