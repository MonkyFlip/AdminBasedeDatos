# Decision Tree con PySpark + MongoDB

# Importamos la función que ya creaste para conectar con MongoDB
from config.mongo_spark_conexion import get_spark_session

# Librerías de Machine Learning de Spark
from pyspark.ml.classification import DecisionTreeClassifier

# Funciones para manipulación de datos
from pyspark.sql.functions import col, when

# Evaluador del modelo
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Cargar datos desde MongoDB usando Spark

spark, df, df_vector = get_spark_session()

print("Datos cargados desde MongoDB")

df.show()

#Explicación

#Tu función ya hizo:

# 1 conexión a MongoDB
# 2 limpieza de datos
# 3 creación de features

#Por lo tanto ya tenemos un DataFrame listo para análisis.

#Ejemplo:

#producto	cantidad	precio	ingreso
#laptop	       2	    1000	2000
#mouse	       5	     20	     100


# Crear variable objetivo (label)

df_ml = df_vector.withColumn(
    "label",
    when(col("ingreso") > 500, 1).otherwise(0)
)

df_ml.select("producto","cantidad","precio","ingreso","label").show()

#Explicación para alumnos

#Creamos una regla de negocio simple:

#ingreso	    label
#mayor a 50000	 1
#menor o igual	 0

#Esto significa:

#1 = venta alta
#0 = venta baja

# Dividir dataset

train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=42)

print("Datos de entrenamiento:", train_data.count())
print("Datos de prueba:", test_data.count())

#Explicación

#80%  entrenar modelo
#20%  probar modelo

#Esto evita sobreajuste (overfitting).

# Crear modelo de árbol de decisión

dt = DecisionTreeClassifier(
    featuresCol="features",   # variables predictoras
    labelCol="label",         # variable objetivo
    maxDepth=4                # profundidad del árbol
)

print("Modelo Decision Tree creado")

#Explicación

#El modelo usará:

#features = [cantidad, precio, ingreso]

#para predecir:

#label = nivel de venta


# Entrenamiento del modelo

modelo = dt.fit(train_data)

print("Modelo entrenado correctamente")


#Explicación

#Aquí el algoritmo aprende patrones en los datos.

#Ejemplo de regla que podría aprender:

#Si ingreso > label  venta alta
#Si ingreso <= 500 venta baja


# Realizar predicciones

predicciones = modelo.transform(test_data)

predicciones.select(
    "producto",
    "features",
    "label",
    "prediction"
).show()

#Explicación

#El modelo predice la clase del producto.

#Ejemplo:

#producto	label	prediction
#laptop	      1	        1
#mouse	      0	        0


# Evaluación del modelo

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predicciones)

print("Precisión del modelo:", accuracy)

#Explicación

#La accuracy indica qué porcentaje de predicciones fueron correctas.

#Ejemplo:

#Precisión del modelo: 0.90

#Significa 90% de aciertos.


# Mostrar estructura del árbol

print("Estructura del árbol")

print(modelo.toDebugString)

#DecisionTreeClassificationModel

#If (feature 2 <= 500)
#  Predict: 0
#Else
#  Predict: 1

#Donde:

#feature 0 = cantidad
#feature 1 = precio
#feature 2 = ingreso
