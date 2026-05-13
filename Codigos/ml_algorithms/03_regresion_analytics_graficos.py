# LABORATORIO DE MACHINE LEARNING CON SPARK
# Ejemplo educativo para explicar distintos modelos de regresión

# 1 SOLUCIÓN AL ERROR DE TKINTER

import matplotlib                 # importa la librería de gráficos matplotlib
matplotlib.use("Agg")             # cambia el motor gráfico para evitar errores en Linux o Spark


# 2 IMPORTACIÓN DE LIBRERÍAS NECESARIAS

from config.mongo_spark_conexion_sinnulos import get_spark_session
# importa una función personalizada que:
# crea la sesión de Spark
# conecta con MongoDB
# devuelve el DataFrame con los datos

from pyspark.ml.feature import VectorAssembler
# VectorAssembler convierte columnas numéricas en un vector
# necesario para que Spark pueda entrenar modelos de ML

from pyspark.ml.feature import PolynomialExpansion
# permite crear variables polinómicas (por ejemplo x²)

from pyspark.ml.regression import LinearRegression
# algoritmo de regresión lineal disponible en Spark ML

from pyspark.ml.evaluation import RegressionEvaluator
# clase utilizada para evaluar qué tan bueno es un modelo

from pyspark.ml.tuning import CrossValidator
# permite aplicar validación cruzada para encontrar el mejor modelo

from pyspark.ml.tuning import ParamGridBuilder
# permite definir combinaciones de hiperparámetros a evaluar

import matplotlib.pyplot as plt
# módulo de matplotlib que permite crear gráficas

import pandas as pd
# librería usada para manipular datos cuando se convierten desde Spark


# 3 CREAR SESIÓN SPARK Y CARGAR DATOS

spark, df, _ = get_spark_session()
# ejecuta la función que crea la sesión Spark y carga el dataset
# spark → sesión de procesamiento distribuido
# df → DataFrame con los datos
# _ → variable ignorada (por ejemplo conexión a base de datos)

print("\nDataset cargado correctamente")
# imprime mensaje en consola para confirmar carga de datos

df.show()
# muestra algunas filas del dataset para inspección rápida


# 4 DIVISIÓN DEL DATASET

train, test = df.randomSplit([0.8,0.2], seed=42)
# divide el dataset en dos partes
# 80% para entrenamiento del modelo
# 20% para evaluación del modelo


# 5 MÉTRICA DE EVALUACIÓN

evaluator = RegressionEvaluator(
    labelCol="ingreso",          # columna que contiene el valor real
    predictionCol="prediction",  # columna que contendrá la predicción del modelo
    metricName="r2"              # métrica utilizada: R²
)

# R² mide qué tan bien el modelo explica los datos
# valores cercanos a 1 indican mejor modelo


# 6 FUNCIÓN PARA GENERAR GRÁFICAS

def graficar(predictions, titulo):
# función que recibe:
# predictions → DataFrame con predicciones
# titulo → nombre de la gráfica

    pdf = predictions.select(
        "cantidad",              # variable independiente
        "ingreso",               # valor real
        "prediction"             # valor predicho
    ).toPandas()
    # convierte el DataFrame de Spark a Pandas para poder graficarlo

    plt.figure()
    # crea una nueva figura para la gráfica

    plt.scatter(
        pdf["cantidad"],         # eje X
        pdf["ingreso"],          # eje Y
        label="Real"             # etiqueta de la serie
    )

    plt.scatter(
        pdf["cantidad"],         # eje X
        pdf["prediction"],       # eje Y
        label="Predicción"
    )

    plt.title(titulo)
    # asigna título a la gráfica

    plt.xlabel("Cantidad")
    # etiqueta eje X

    plt.ylabel("Ingreso")
    # etiqueta eje Y

    plt.legend()
    # muestra leyenda

    plt.savefig(f"{titulo}.png")
    # guarda la gráfica en archivo


# MODELO 1 REGRESIÓN LINEAL SIMPLE

print("\nMODELO 1 REGRESIÓN LINEAL SIMPLE")
# imprime título del modelo

assembler_simple = VectorAssembler(
    inputCols=["cantidad"],     # variable independiente
    outputCol="features"        # nombre de la columna vectorial
)

train_simple = assembler_simple.transform(train)
# crea columna features en dataset de entrenamiento

test_simple = assembler_simple.transform(test)
# crea columna features en dataset de prueba

lr_simple = LinearRegression(
    featuresCol="features",     # columna con variables independientes
    labelCol="ingreso"          # variable dependiente
)

model_simple = lr_simple.fit(train_simple)
# entrena el modelo con datos de entrenamiento

pred_simple = model_simple.transform(test_simple)
# genera predicciones usando el modelo entrenado

r2_simple = evaluator.evaluate(pred_simple)
# calcula el valor de R² del modelo

print("R2:", r2_simple)
# muestra el valor de R² en consola

graficar(pred_simple,"regresion_lineal_simple")
# genera gráfica del modelo


# MODELO 2 REGRESIÓN LINEAL MÚLTIPLE

print("\nMODELO 2 REGRESIÓN LINEAL MÚLTIPLE")

assembler_multiple = VectorAssembler(
    inputCols=["cantidad","precio"],  # dos variables independientes
    outputCol="features"
)

train_multiple = assembler_multiple.transform(train)
# crea vector de características para entrenamiento

test_multiple = assembler_multiple.transform(test)
# crea vector de características para prueba

lr_multiple = LinearRegression(
    featuresCol="features",
    labelCol="ingreso"
)

model_multiple = lr_multiple.fit(train_multiple)
# entrena modelo con dos variables

pred_multiple = model_multiple.transform(test_multiple)
# genera predicciones

r2_multiple = evaluator.evaluate(pred_multiple)
# calcula R²

print("R2:", r2_multiple)

graficar(pred_multiple,"regresion_lineal_multiple")


# MODELO 3 RIDGE REGRESSION

print("\nMODELO 3 RIDGE REGRESSION")

ridge = LinearRegression(
    featuresCol="features",
    labelCol="ingreso",
    regParam=0.5,        # parámetro de regularización
    elasticNetParam=0    # 0 indica Ridge
)

model_ridge = ridge.fit(train_multiple)
# entrena modelo ridge

pred_ridge = model_ridge.transform(test_multiple)
# genera predicciones

r2_ridge = evaluator.evaluate(pred_ridge)
# evalúa modelo

print("R2:", r2_ridge)

graficar(pred_ridge,"ridge_regression")


# MODELO 4 LASSO REGRESSION

print("\nMODELO 4 LASSO REGRESSION")

lasso = LinearRegression(
    featuresCol="features",
    labelCol="ingreso",
    regParam=0.5,        # intensidad de regularización
    elasticNetParam=1    # 1 indica Lasso
)

model_lasso = lasso.fit(train_multiple)
# entrena modelo

pred_lasso = model_lasso.transform(test_multiple)
# genera predicciones

r2_lasso = evaluator.evaluate(pred_lasso)
# calcula R²

print("R2:", r2_lasso)

graficar(pred_lasso,"lasso_regression")


# MODELO 5 REGRESIÓN POLINÓMICA

print("\nMODELO 5 REGRESIÓN POLINÓMICA")

poly = PolynomialExpansion(
    inputCol="features",        # vector original
    outputCol="poly_features",  # vector expandido
    degree=2                    # grado del polinomio
)

train_poly = poly.transform(train_multiple)
# crea variables polinómicas en entrenamiento

test_poly = poly.transform(test_multiple)
# crea variables polinómicas en prueba

lr_poly = LinearRegression(
    featuresCol="poly_features",
    labelCol="ingreso"
)

model_poly = lr_poly.fit(train_poly)
# entrena modelo

pred_poly = model_poly.transform(test_poly)
# genera predicciones

r2_poly = evaluator.evaluate(pred_poly)
# evalúa modelo

print("R2:", r2_poly)

graficar(pred_poly,"regresion_polinomica")


# MODELO 6 VALIDACIÓN CRUZADA

print("\nMODELO 6 CROSS VALIDATION")

lr_cv = LinearRegression(
    featuresCol="features",
    labelCol="ingreso"
)

paramGrid = ParamGridBuilder() \
    .addGrid(lr_cv.regParam,[0.01,0.1,1]) \
    .addGrid(lr_cv.elasticNetParam,[0,0.5,1]) \
    .build()
# define combinaciones de hiperparámetros a evaluar

cv = CrossValidator(
    estimator=lr_cv,        # modelo base
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3              # número de divisiones del dataset
)

model_cv = cv.fit(train_multiple)
# entrena el modelo probando varias combinaciones

pred_cv = model_cv.transform(test_multiple)
# genera predicciones

r2_cv = evaluator.evaluate(pred_cv)
# evalúa el mejor modelo

print("R2:", r2_cv)

graficar(pred_cv,"cross_validation")


# COMPARACIÓN FINAL DE MODELOS

print("\nCOMPARACIÓN FINAL")

resultados = {
"Lineal Simple":r2_simple,
"Lineal Multiple":r2_multiple,
"Ridge":r2_ridge,
"Lasso":r2_lasso,
"Polinomial":r2_poly,
"Cross Validation":r2_cv
}
# diccionario que guarda los resultados de todos los modelos

for modelo,valor in resultados.items():
# recorre cada modelo

    print(modelo,"→",round(valor,4))
    # imprime nombre y valor R²

mejor = max(resultados, key=resultados.get)
# identifica el modelo con mayor R²

print("\nMEJOR MODELO:", mejor)


# FINALIZAR SESIÓN SPARK

spark.stop()
# cierra la sesión Spark y libera recursos