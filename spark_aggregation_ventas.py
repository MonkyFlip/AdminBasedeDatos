from pyspark.sql import SparkSession
from spark_config import MONGO_URI, DB_NAME
from dotenv import load_dotenv
import os
import json

# ──────────────────────────────────────────────
# CARGAR ENTORNO
# ──────────────────────────────────────────────
load_dotenv()
COL = os.getenv("MONGO_COLLECTION")

# ──────────────────────────────────────────────
# PIPELINE DE AGGREGATION
# ──────────────────────────────────────────────
pipeline = [
    {
        "$group": {
            "_id": "$producto",
            "total_ventas": {"$sum": 1},
            "ingresos_totales": {"$sum": "$total"},
            "precio_promedio": {"$avg": "$total"}
        }
    },
    {
        "$sort": {"ingresos_totales": -1}
    }
]

# ──────────────────────────────────────────────
# SPARK SESSION
# ──────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("AggregationVentasGym")
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ──────────────────────────────────────────────
# CONTEXTO DEL ANÁLISIS
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("== ANÁLISIS DE VENTAS DEL GIMNASIO ==")
print("=" * 70)
print(f"- Base de datos: {DB_NAME}")
print(f"- Colección analizada: {COL}")
print("- Análisis: ventas por producto (cantidad, ingresos y promedio)")
print("=" * 70 + "\n")

# ──────────────────────────────────────────────
# EJECUTAR AGGREGATION
# ──────────────────────────────────────────────
df = (
    spark.read
    .format("mongodb")
    .option("database", DB_NAME)
    .option("collection", COL)
    .option("pipeline", json.dumps(pipeline))
    .load()
)

# ──────────────────────────────────────────────
# RESULTADOS
# ──────────────────────────────────────────────
print("-> RESULTADOS")
print("Cada fila representa un producto vendido en el gimnasio.\n")

df.show(20, truncate=False)

print("\n-> Total de productos distintos analizados:", df.count())

spark.stop()
print("\n<+> Aggregation finalizada correctamente")
