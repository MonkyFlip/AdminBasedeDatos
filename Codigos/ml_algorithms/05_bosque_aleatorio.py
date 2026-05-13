# IMPORTACIÓN DE LIBRERÍAS
from config.mongo_spark_conexion import get_spark_session

from pyspark.sql.functions import when, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# INICIALIZACIÓN DE SPARK
spark, df, _ = get_spark_session()

# LIMPIEZA DE DATOS (PROFESIONAL)
# Rellenamos valores nulos (evita errores en producción)

df = df.fillna({
    "cantidad": 0,
    "precio": 0,
    "ingreso": 0
})

# CREACIÓN DE VARIABLE OBJETIVO (LABEL)
df = df.withColumn(
    "categoria",
    when(col("ingreso") > 50000, 1).otherwise(0)
)

# DIVISIÓN DE DATOS
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# DEFINICIÓN DE ETAPAS DEL PIPELINE

# 1. Vectorización (features)
assembler = VectorAssembler(
    inputCols=["cantidad", "precio", "ingreso"],
    outputCol="features",
    handleInvalid="skip"  # evita errores por datos inesperados
)

# 2. Modelo Random Forest
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="categoria",
    numTrees=50,
    maxDepth=5,
    seed=42
)

# CREACIÓN DEL PIPELINE
pipeline = Pipeline(stages=[assembler, rf])

# ENTRENAMIENTO
print("\n")
print(" Entrenando pipeline...")

model = pipeline.fit(train_data)

print("\n")
print(" Pipeline entrenado correctamente")

# PREDICCIONES
predictions = model.transform(test_data)

print("\n")
print(" Ejemplo de predicciones:")
print("\n")
predictions.select(
    "cantidad",
    "precio",
    "ingreso",
    "prediction",
    "categoria",
    "probability"
).show(10)

# EVALUACIÓN DEL MODELO
evaluator = BinaryClassificationEvaluator(
    labelCol="categoria",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print("\n")
print(f" AUC del modelo: {auc}")

# MATRIZ DE RESULTADOS
print("\n")
print(" Comparación predicción vs real:")
print("\n")
predictions.groupBy("categoria", "prediction").count().show()

# IMPORTANCIA DE VARIABLES
# Extraemos el modelo desde el pipeline

rf_model = model.stages[-1]
print("\n")
print(" Importancia de variables:")
print("\n")

for i, imp in enumerate(rf_model.featureImportances):
    print(f"Feature {i}: {imp}")

# GUARDAR MODELO (PRODUCCIÓN)
model.write().overwrite().save("modelo_pipeline_rf")

print("\n")
print(" Modelo guardado correctamente")

# CIERRE DE SPARK
spark.stop()
print("\n")
print(" Sesión Spark finalizada")