from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from spark_connection import uri, db_name

spark = (
    SparkSession.builder
    .appName("PopulateMongoDB100K")
    .config("spark.mongodb.write.connection.uri", uri)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

def write(df, col):
    df.write.format("mongodb") \
        .option("database", db_name) \
        .option("collection", col) \
        .mode("append") \
        .save()
    print(f"   ✅ {col}")

print("🚀 Poblando base con ~100K registros")

# ──────────────────────────────────────────────
# USUARIOS (email UNIQUE)
# ──────────────────────────────────────────────
usuarios = (
    spark.range(10000)
    .withColumn("email", expr("concat('user', id, '@gym.com')"))
    .withColumn("activo", expr("rand() > 0.1"))
)
write(usuarios, "usuarios")

# ──────────────────────────────────────────────
# MIEMBROS
# ──────────────────────────────────────────────
miembros = (
    spark.range(10000)
    .withColumn("estado", expr("CASE WHEN rand()>0.2 THEN 'Activo' ELSE 'Inactivo' END"))
)
write(miembros, "miembros")

# ──────────────────────────────────────────────
# PRODUCTOS
# ──────────────────────────────────────────────
productos = (
    spark.range(50)
    .withColumn("precio", expr("round(rand()*400+50,2)"))
    .withColumn("stock", expr("cast(rand()*200 as int)"))
)
write(productos, "productos")

# ──────────────────────────────────────────────
# VENTAS (fecha indexada)
# ──────────────────────────────────────────────
ventas = (
    spark.range(20000)
    .withColumn("total", expr("round(rand()*900+50,2)"))
    .withColumn("fecha", expr("date_sub(current_date(), cast(rand()*365 as int))"))
)
write(ventas, "ventas")

# ──────────────────────────────────────────────
# DETALLE VENTA
# ──────────────────────────────────────────────
detalle = (
    spark.range(25000)
    .withColumn("cantidad", expr("cast(rand()*5+1 as int)"))
    .withColumn("subtotal", expr("round(rand()*300+20,2)"))
)
write(detalle, "detalle_venta")

# ──────────────────────────────────────────────
# PAGOS (metodo_pago indexado)
# ──────────────────────────────────────────────
pagos = (
    spark.range(10000)
    .withColumn("monto", expr("round(rand()*900+50,2)"))
    .withColumn(
        "metodo_pago",
        expr("""
            CASE
                WHEN rand()<0.4 THEN 'Efectivo'
                WHEN rand()<0.7 THEN 'Tarjeta'
                ELSE 'Transferencia'
            END
        """)
    )
)
write(pagos, "pagos")

# ──────────────────────────────────────────────
# ASISTENCIAS (fecha indexada)
# ──────────────────────────────────────────────
asistencias = (
    spark.range(8000)
    .withColumn("fecha", expr("date_sub(current_date(), cast(rand()*180 as int))"))
)
write(asistencias, "asistencias")

# ──────────────────────────────────────────────
# PROGRESO FÍSICO (bmi indexado)
# ──────────────────────────────────────────────
progreso = (
    spark.range(5000)
    .withColumn("peso", expr("round(rand()*40+60,2)"))
    .withColumn("bmi", expr("round(rand()*10+18,2)"))
)
write(progreso, "progreso_fisico")

# ──────────────────────────────────────────────
# RUTINAS
# ──────────────────────────────────────────────
rutinas = (
    spark.range(2000)
    .withColumn(
        "dificultad",
        expr("""
            CASE
                WHEN rand()<0.33 THEN 'Principiante'
                WHEN rand()<0.66 THEN 'Intermedio'
                ELSE 'Avanzado'
            END
        """)
    )
)
write(rutinas, "rutinas")

# ──────────────────────────────────────────────
# RUTINA DÍAS / EJERCICIOS
# ──────────────────────────────────────────────
write(spark.range(6000), "rutina_dias")
write(spark.range(12000), "rutina_ejercicios")

# ──────────────────────────────────────────────
# RELACIONES (SIN ÍNDICES ÚNICOS)
# ──────────────────────────────────────────────
write(spark.range(10000), "miembro_membresia")
write(spark.range(5000), "correo_miembro")
write(spark.range(2000), "correos_enviados")

# ──────────────────────────────────────────────
# MIEMBRO_RUTINA (ÍNDICE COMPUESTO UNIQUE)
# ──────────────────────────────────────────────
miembro_rutina = (
    spark.range(5000)
    .withColumn("id_miembro", expr("cast(id % 10000 as int)"))
    .withColumn("id_rutina", expr("cast(id % 2000 as int)"))
)
write(miembro_rutina, "miembro_rutina")

# ──────────────────────────────────────────────
# SESIONES (estado indexado)
# ──────────────────────────────────────────────
sesiones = (
    spark.range(3000)
    .withColumn(
        "estado",
        expr("CASE WHEN rand()<0.7 THEN 'completed' ELSE 'cancelled' END")
    )
)
write(sesiones, "sesiones")

# ──────────────────────────────────────────────
# PLANES ALIMENTICIOS
# ──────────────────────────────────────────────
write(spark.range(3000), "planes_alimenticios")
write(spark.range(9000), "plan_recetas")

# ──────────────────────────────────────────────
# ENTRENADORES
# ──────────────────────────────────────────────
write(spark.range(50), "perfil_entrenador")
write(spark.range(150), "certificaciones_entrenador")
write(spark.range(200), "logros_entrenador")
write(spark.range(3000), "evaluaciones_entrenador")

spark.stop()
print("🎉 Base poblada correctamente con ~100K registros")
