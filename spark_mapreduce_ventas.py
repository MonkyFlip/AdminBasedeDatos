from spark_connection import get_spark_session
from pyspark.sql.functions import sum, count, avg

# get_spark_session():
# - crea la sesión Spark
# - conecta con MongoDB
# - carga la colección indicada (ventas)
# - devuelve el DataFrame listo para trabajar

spark, df, collection = get_spark_session()

print("=== MAPREDUCE: ANALISIS DE VENTAS POR PRODUCTO ===")
print(f"Coleccion analizada: {collection}")
print("Operacion: suma de ingresos, conteo de ventas y promedio por producto\n")

# MAP
# Spark toma cada documento de la colección ventas y extrae:
# producto -> total
#
# Ejemplo:
# Proteína -> 350
# Proteína -> 420
# Creatina -> 300

# SHUFFLE
# Spark redistribuye los datos entre nodos agrupando por la clave "producto"

# REDUCE
# Spark aplica las funciones de agregación por cada grupo

resultado = (
    df.groupBy("producto")
      .agg(
          count("*").alias("total_ventas"),
          sum("total").alias("ingresos_totales"),
          avg("total").alias("precio_promedio")
      )
      .orderBy("ingresos_totales", ascending=False)
)

# Mostrar resultado final
resultado.show(truncate=False)

spark.stop()
