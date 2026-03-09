from pyspark.sql import SparkSession
from spark_connection import uri, db_name

# ──────────────────────────────────────────────
# SPARK
# ──────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("CreateMongoDBFull")
    .config("spark.mongodb.write.connection.uri", uri)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print(f"✅ Spark conectado")
print(f"📦 Base de datos: {db_name}")

# ──────────────────────────────────────────────
# COLECCIONES A CREAR
# ──────────────────────────────────────────────
collections = {
    "roles": ["nombre"],
    "usuarios": ["email", "activo"],
    "miembros": ["estado"],
    "membresias": ["nombre", "precio"],
    "productos": ["nombre", "precio", "stock"],
    "ventas": ["total"]
}

# ──────────────────────────────────────────────
# CREAR COLECCIONES + 5 REGISTROS
# ──────────────────────────────────────────────
for collection, fields in collections.items():
    data = []

    for i in range(1, 6):
        row = {field: f"{field}_{i}" for field in fields}
        data.append(row)

    df = spark.createDataFrame(data)

    df.write \
      .format("mongodb") \
      .mode("append") \
      .option("database", db_name) \
      .option("collection", collection) \
      .save()

    print(f"📁 Colección creada y poblada: {collection} (5 registros)")

# ──────────────────────────────────────────────
# FINAL
# ──────────────────────────────────────────────
spark.stop()
print("\n🚀 Base de datos creada con todas las colecciones y datos")
