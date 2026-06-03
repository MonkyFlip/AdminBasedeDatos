"""
Práctica: Análisis de Datos — ETL con PySpark
Actividad 5 (variante PySpark) + Actividad 7 con MLlib
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# Rutas relativas al script
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────
# Inicializar Spark
# ─────────────────────────────────
spark = (SparkSession.builder
         .appName("PracticaETL")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.sql.shuffle.partitions", "4")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("✓ Spark inicializado:", spark.version)

# ─────────────────────────────────────────────────────────
# PASO 1 — EXTRACCIÓN
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PASO 1 — EXTRACCIÓN")
print("=" * 55)

df_ventas    = spark.read.csv(str(DATA_DIR / "ventas.csv"),              header=True, inferSchema=True)
df_productos = spark.read.parquet(str(DATA_DIR / "productos_clean.parquet"))
df_clientes  = spark.read.parquet(str(DATA_DIR / "clientes_clean.parquet"))

print(f"\n[ventas]    schema:")
df_ventas.printSchema()
print(f"  Registros: {df_ventas.count()}  |  Columnas: {len(df_ventas.columns)}")
print(f"\n[productos] registros: {df_productos.count()}")
print(f"[clientes]  registros: {df_clientes.count()}")

# ─────────────────────────────────────────────────────────
# PASO 2 — TRANSFORMACIÓN
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PASO 2 — TRANSFORMACIÓN (PySpark)")
print("=" * 55)

# 2.1 Duplicados
antes = df_ventas.count()
df_ventas = df_ventas.dropDuplicates()
print(f"\n  Duplicados eliminados: {antes - df_ventas.count()}")

# 2.2 Nulos — imputar con mediana (approxQuantile)
mediana_cant   = df_ventas.approxQuantile("cantidad",        [0.5], 0.01)[0]
mediana_precio = df_ventas.approxQuantile("precio_unitario", [0.5], 0.01)[0]

df_ventas = df_ventas.fillna({"cantidad": mediana_cant, "precio_unitario": mediana_precio})

print(f"  cantidad imputada con mediana={mediana_cant}")
print(f"  precio_unitario imputada con mediana={mediana_precio:.2f}")

# 2.3 Cast y columna derivada
df_ventas = (df_ventas
    .withColumn("cantidad",        F.col("cantidad").cast(IntegerType()))
    .withColumn("precio_unitario", F.col("precio_unitario").cast(DoubleType()))
    .withColumn("descuento",       F.col("descuento").cast(DoubleType()))
    .withColumn("total",           F.col("cantidad") * F.col("precio_unitario") * (1 - F.col("descuento")))
    .withColumn("fecha",           F.to_date("fecha", "yyyy-MM-dd")))

print("\n--- DataFrame ventas (limpio) ---")
df_ventas.show(5, truncate=False)
df_ventas.printSchema()

# ─────────────────────────────────────────────────────────
# PASO 3 — CARGA
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PASO 3 — CARGA")
print("=" * 55)

df_ventas.write.mode("overwrite").parquet(str(DATA_DIR / "spark_ventas_clean.parquet"))
print("✓ ventas limpias guardadas en Parquet (PySpark)")

# ─────────────────────────────────────────────────────────
# ANÁLISIS EXPLORATORIO (Actividad 6)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ANÁLISIS EXPLORATORIO — PySpark")
print("=" * 55)

df_p = df_productos.withColumn("valor_inventario", F.col("precio") * F.col("inventario"))

print("\n¿Producto más caro?")
df_p.orderBy(F.col("precio").desc()).select("nombre", "categoria", "precio").show(1, truncate=False)

print("¿Producto con mayor inventario?")
df_p.orderBy(F.col("inventario").desc()).select("nombre", "categoria", "inventario").show(1, truncate=False)

print("¿Mayor valor económico en inventario?")
df_p.orderBy(F.col("valor_inventario").desc()).select("nombre", "categoria", "precio", "inventario", "valor_inventario").show(1, truncate=False)

print("Ventas totales por categoría:")
(df_ventas.groupBy("categoria")
    .agg(F.round(F.sum("total"), 2).alias("total_ventas"),
         F.count("*").alias("num_ventas"))
    .orderBy(F.col("total_ventas").desc())
    .show())

# ─────────────────────────────────────────────────────────
# ACTIVIDAD 7 — PREDICCIÓN CON MLlib
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ACTIVIDAD 7 — REGRESIÓN (PySpark MLlib)")
print("=" * 55)

df_ml = df_p.dropna(subset=["precio", "inventario", "categoria", "valor_inventario"])

indexer   = StringIndexer(inputCol="categoria", outputCol="cat_idx")
assembler = VectorAssembler(inputCols=["precio", "inventario", "cat_idx"], outputCol="features")

lr_spark  = LinearRegression(featuresCol="features", labelCol="valor_inventario", maxIter=50, regParam=0.1)
gbt_spark = GBTRegressor(featuresCol="features",     labelCol="valor_inventario", maxIter=30, maxDepth=4)

train, test = df_ml.randomSplit([0.75, 0.25], seed=42)

evaluator_mae = RegressionEvaluator(labelCol="valor_inventario", predictionCol="prediction", metricName="mae")
evaluator_r2  = RegressionEvaluator(labelCol="valor_inventario", predictionCol="prediction", metricName="r2")

for model_name, estimator in [("LinearRegression", lr_spark), ("GBTRegressor", gbt_spark)]:
    pipeline = Pipeline(stages=[indexer, assembler, estimator])
    modelo   = pipeline.fit(train)
    preds    = modelo.transform(test)

    print(f"\n{model_name}:")
    print(f"  MAE = {evaluator_mae.evaluate(preds):,.2f}  |  R² = {evaluator_r2.evaluate(preds):.4f}")

    if model_name == "GBTRegressor":
        print("\n--- DataFrame de predicciones (GBT) ---")
        (preds.select("nombre", "categoria", "precio", "inventario",
                      F.round("valor_inventario", 2).alias("valor_real"),
                      F.round("prediction", 2).alias("prediccion"))
              .show(10, truncate=False))

spark.stop()
print("\n✓ PySpark ETL + MLlib completado")