from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# 1. Crear sesión Spark
spark = SparkSession.builder \
    .appName("MonitoreoSeguridad") \
    .getOrCreate()

# 2. Crear datos simulados en memoria
datos = [
    ("192.168.1.10", "admin", "OK", 1),
    ("192.168.1.11", "hector", "FAIL", 3),
    ("192.168.1.11", "hector", "FAIL", 4),
    ("192.168.1.50", "root", "FAIL", 6),
    ("192.168.1.50", "root", "FAIL", 7),
    ("192.168.1.15", "user1", "OK", 1),
    ("192.168.1.99", "admin", "FAIL", 8),
]

columnas = ["ip", "usuario", "estatus", "intentos"]

df = spark.createDataFrame(datos, columnas)

print("\n=== LOGS GENERADOS ===")
df.show()

# 3. Filtrar intentos fallidos
fallidos = df.filter(col("estatus") == "FAIL")

print("\n=== INTENTOS FALLIDOS ===")
fallidos.show()

# 4. Detectar IPs sospechosas (>5 intentos)
sospechosos = fallidos.filter(col("intentos") > 5)

print("\n=== POSIBLES ATAQUES DETECTADOS ===")
sospechosos.show()

# 5. Conteo de fallos por usuario
conteo = fallidos.groupBy("usuario") \
                 .agg(count("*").alias("total_fallos"))

print("\n=== CONTEO DE FALLOS POR USUARIO ===")
conteo.show()

spark.stop()