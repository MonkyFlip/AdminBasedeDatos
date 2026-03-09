from pyspark.sql import SparkSession
from spark_connection import uri, db_name
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker("es_MX")

NUM = 5000
DIAS = 180

spark = (
    SparkSession.builder
    .appName("PoblarGymMongo_FULL_OPT")
    .config("spark.mongodb.write.connection.uri", uri)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("✅ Spark conectado")
print(f"📦 Base de datos: {db_name}")

# ──────────────────────────────────────────────
# ROLES
# ──────────────────────────────────────────────
roles = [{"nombre": r} for r in ["admin", "coach", "recepcion", "cliente"]]

spark.createDataFrame(roles).write.format("mongodb") \
    .mode("overwrite") \
    .option("database", db_name) \
    .option("collection", "roles") \
    .save()

# ──────────────────────────────────────────────
# MEMBRESIAS
# ──────────────────────────────────────────────
membresias = [
    {"nombre": "Mensual", "precio": 500},
    {"nombre": "Trimestral", "precio": 1350},
    {"nombre": "Anual", "precio": 4800}
]

spark.createDataFrame(membresias).write.format("mongodb") \
    .mode("overwrite") \
    .option("database", db_name) \
    .option("collection", "membresias") \
    .save()

# ──────────────────────────────────────────────
# PRODUCTOS
# ──────────────────────────────────────────────
productos = []

for _ in range(NUM):
    productos.append({
        "nombre": random.choice(["Proteína", "Creatina", "Playera Gym", "Toalla", "Shaker"]),
        "precio": random.randint(200, 900),
        "stock": random.randint(20, 300)
    })

spark.createDataFrame(productos).write.format("mongodb") \
    .mode("overwrite") \
    .option("database", db_name) \
    .option("collection", "productos") \
    .save()

# ──────────────────────────────────────────────
# USUARIOS
# ──────────────────────────────────────────────
usuarios = []

for _ in range(NUM):
    usuarios.append({
        "nombre": fake.name(),
        "email": fake.email(),
        "activo": True,
        "fecha_creacion": fake.date_time_this_year()
    })

spark.createDataFrame(usuarios).write.format("mongodb") \
    .mode("overwrite") \
    .option("database", db_name) \
    .option("collection", "usuarios") \
    .save()

# ──────────────────────────────────────────────
# MIEMBROS
# ──────────────────────────────────────────────
miembros = []

for _ in range(NUM):
    miembros.append({
        "sexo": random.choice(["M", "F"]),
        "estado": "Activo",
        "membresia": random.choice(["Mensual", "Trimestral", "Anual"])
    })

spark.createDataFrame(miembros).write.format("mongodb") \
    .mode("overwrite") \
    .option("database", db_name) \
    .option("collection", "miembros") \
    .save()

# ──────────────────────────────────────────────
# ASISTENCIAS (MASIVAS)
# ──────────────────────────────────────────────
asistencias = []
inicio = datetime.now() - timedelta(days=DIAS)

for _ in range(NUM * 15):
    asistencias.append({
        "fecha": inicio + timedelta(days=random.randint(0, DIAS))
    })

spark.createDataFrame(asistencias).write.format("mongodb") \
    .mode("overwrite") \
    .option("database", db_name) \
    .option("collection", "asistencias") \
    .save()

# ──────────────────────────────────────────────
# VENTAS
# ──────────────────────────────────────────────
ventas = []

for _ in range(NUM):
    total = random.randint(200, 1500)
    ventas.append({
        "producto": random.choice(["Proteína", "Creatina", "Playera Gym", "Toalla", "Shaker"]),
        "metodo_pago": random.choice(["Efectivo", "Tarjeta", "Transferencia"]),
        "total": total,
        "fecha": fake.date_time_this_year()
    })

spark.createDataFrame(ventas).write.format("mongodb") \
    .mode("overwrite") \
    .option("database", db_name) \
    .option("collection", "ventas") \
    .save()

spark.stop()
print("\n🚀 Base de datos Gym poblada COMPLETAMENTE con datos realistas")
