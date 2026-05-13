from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Crear sesión Spark
spark = SparkSession.builder \
    .appName("DeteccionFuerzaBruta") \
    .getOrCreate()

# Datos simulados (más realistas)
datos = [
    ("192.168.1.10", "OK"),
    ("192.168.1.11", "FAIL"),
    ("192.168.1.11", "FAIL"),
    ("192.168.1.11", "FAIL"),
    ("192.168.1.50", "FAIL"),
    ("192.168.1.50", "FAIL"),
    ("192.168.1.50", "FAIL"),
    ("192.168.1.50", "FAIL"),
    ("192.168.1.50", "FAIL"),
    ("192.168.1.99", "OK"),
]

df = spark.createDataFrame(datos, ["ip", "estatus"])

print("\n=== LOGS ===")
df.show()

# Contar intentos fallidos por IP
fallos_ip = df.filter(col("estatus") == "FAIL") \
              .groupBy("ip") \
              .agg(count("*").alias("total_fallos"))

print("\n=== FALLAS POR IP ===")
fallos_ip.show()

# Detectar IPs con más de 3 fallos
ataques = fallos_ip.filter(col("total_fallos") > 3)

print("\n=== POSIBLE ATAQUE DETECTADO ===")
ataques.show()

spark.stop()