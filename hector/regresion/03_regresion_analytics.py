# LABORATORIO DE MACHINE LEARNING CON SPARK
# COMPARACIÓN DE MODELOS DE REGRESIÓN


# 1 IMPORTACIÓN DE LIBRERÍAS
# Función personalizada que crea la sesión Spark
# y además conecta con la base de datos (MongoDB o similar)

from config.mongo_spark_conexion_sinnulos import get_spark_session


# VectorAssembler convierte varias columnas numéricas
# en un solo vector llamado "features"
# Spark ML requiere este formato para entrenar modelos

from pyspark.ml.feature import VectorAssembler


# PolynomialExpansion crea nuevas variables elevadas
# al cuadrado, cubo, etc.
# Sirve para detectar relaciones no lineales

from pyspark.ml.feature import PolynomialExpansion


# LinearRegression es el algoritmo que utilizará Spark
# para entrenar modelos de regresión

from pyspark.ml.regression import LinearRegression


# RegressionEvaluator mide qué tan bueno es el modelo

from pyspark.ml.evaluation import RegressionEvaluator


# Cross Validation permite probar diferentes
# combinaciones de parámetros automáticamente

from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder


# Librerías para crear gráficas

import matplotlib.pyplot as plt
import pandas as pd


# 2 CREAR SESIÓN SPARK Y CARGAR DATOS

# Se crea la sesión Spark y se cargan los datos

spark, df, _ = get_spark_session()

print("Dataset cargado")

# Mostrar algunos registros
df.show()

# 3 DIVISIÓN DEL DATASET

# El dataset se divide en dos partes:
#
# 80% entrenamiento
# 20% prueba
#
# Esto evita que el modelo memorice los datos

train, test = df.randomSplit([0.8,0.2], seed=42)



# 4 MÉTRICA DE EVALUACIÓN

# Usaremos R² (coeficiente de determinación)

evaluator = RegressionEvaluator(

    labelCol="ingreso",

    predictionCol="prediction",

    metricName="r2"
)

# Interpretación de R²
#
# 0.9 - 1.0  excelente modelo
# 0.7 - 0.9  buen modelo
# 0.5 - 0.7  modelo aceptable
# < 0.5      modelo pobre


# 5 FUNCIÓN PARA CREAR GRÁFICAS

def graficar(predictions, titulo):

    # Convertimos Spark DataFrame a Pandas
    pdf = predictions.select(
        "cantidad",
        "ingreso",
        "prediction"
    ).toPandas()

    # Crear figura
    plt.figure()

    # Valores reales
    plt.scatter(
        pdf["cantidad"],
        pdf["ingreso"],
        label="Real"
    )

    # Valores predichos
    plt.scatter(
        pdf["cantidad"],
        pdf["prediction"],
        label="Predicción"
    )

    plt.title(titulo)

    plt.xlabel("Cantidad")
    plt.ylabel("Ingreso")

    plt.legend()

    plt.show()



# MODELO 1 REGRESIÓN LINEAL SIMPLE

print("MODELO 1 REGRESIÓN LINEAL SIMPLE")

# Solo usa una variable independiente

assembler_simple = VectorAssembler(

    inputCols=["cantidad"],

    outputCol="features"

)

train_simple = assembler_simple.transform(train)
test_simple = assembler_simple.transform(test)

lr_simple = LinearRegression(

    featuresCol="features",

    labelCol="ingreso"
)

model_simple = lr_simple.fit(train_simple)

pred_simple = model_simple.transform(test_simple)

r2_simple = evaluator.evaluate(pred_simple)

print("R2:", r2_simple)

graficar(pred_simple,"Regresión Lineal Simple")



# MODELO 2 REGRESIÓN LINEAL MÚLTIPLE

print("MODELO 2 REGRESIÓN LINEAL MÚLTIPLE")

# Utiliza varias variables independientes

assembler_multiple = VectorAssembler(

    inputCols=["cantidad","precio"],

    outputCol="features"

)

train_multiple = assembler_multiple.transform(train)
test_multiple = assembler_multiple.transform(test)

lr_multiple = LinearRegression(

    featuresCol="features",

    labelCol="ingreso"
)

model_multiple = lr_multiple.fit(train_multiple)

pred_multiple = model_multiple.transform(test_multiple)

r2_multiple = evaluator.evaluate(pred_multiple)

print("R2:", r2_multiple)

graficar(pred_multiple,"Regresión Lineal Multiple")



# MODELO 3 RIDGE REGRESSION

print("MODELO 3 RIDGE REGRESSION")

# Ridge agrega una penalización para evitar coeficientes grandes

ridge = LinearRegression(

    featuresCol="features",

    labelCol="ingreso",

    regParam=0.5,

    elasticNetParam=0
)

model_ridge = ridge.fit(train_multiple)

pred_ridge = model_ridge.transform(test_multiple)

r2_ridge = evaluator.evaluate(pred_ridge)

print("R2:", r2_ridge)

graficar(pred_ridge,"Ridge Regression")



# MODELO 4 LASSO REGRESSION

print("MODELO 4 LASSO REGRESSION")

# Lasso puede eliminar variables poco importantes

lasso = LinearRegression(

    featuresCol="features",

    labelCol="ingreso",

    regParam=0.5,

    elasticNetParam=1
)

model_lasso = lasso.fit(train_multiple)

pred_lasso = model_lasso.transform(test_multiple)

r2_lasso = evaluator.evaluate(pred_lasso)

print("R2:", r2_lasso)

graficar(pred_lasso,"Lasso Regression")


# MODELO 5 REGRESIÓN POLINÓMICA

print("MODELO 5 REGRESIÓN POLINÓMICA")

poly = PolynomialExpansion(

    inputCol="features",

    outputCol="poly_features",

    degree=2
)

train_poly = poly.transform(train_multiple)
test_poly = poly.transform(test_multiple)

lr_poly = LinearRegression(

    featuresCol="poly_features",

    labelCol="ingreso"
)

model_poly = lr_poly.fit(train_poly)

pred_poly = model_poly.transform(test_poly)

r2_poly = evaluator.evaluate(pred_poly)

print("R2:", r2_poly)

graficar(pred_poly,"Regresión Polinómica")



# MODELO 6 CROSS VALIDATION

print("MODELO 6 CROSS VALIDATION")

lr_cv = LinearRegression(

    featuresCol="features",

    labelCol="ingreso"
)

# Grid de parámetros que se probarán

paramGrid = ParamGridBuilder() \
    .addGrid(lr_cv.regParam,[0.01,0.1,1]) \
    .addGrid(lr_cv.elasticNetParam,[0,0.5,1]) \
    .build()

cv = CrossValidator(

    estimator=lr_cv,

    estimatorParamMaps=paramGrid,

    evaluator=evaluator,

    numFolds=3
)

model_cv = cv.fit(train_multiple)

pred_cv = model_cv.transform(test_multiple)

r2_cv = evaluator.evaluate(pred_cv)

print("R2:", r2_cv)

graficar(pred_cv,"Cross Validation")



# COMPARACIÓN FINAL

print("COMPARACIÓN FINAL")

resultados = {

"Lineal Simple":r2_simple,
"Lineal Multiple":r2_multiple,
"Ridge":r2_ridge,
"Lasso":r2_lasso,
"Polinomial":r2_poly,
"Cross Validation":r2_cv

}

for k,v in resultados.items():

    print(k,"→",round(v,4))


mejor = max(resultados, key=resultados.get)

print("MEJOR MODELO:", mejor)



# FINALIZAR SPARK

spark.stop()