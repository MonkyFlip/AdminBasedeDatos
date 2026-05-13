# DASHBOARD BIG DATA + MACHINE LEARNING
# Este script crea un dashboard educativo que muestra:
# 1) Visualización de ventas
# 2) Entrenamiento de modelos ML
# 3) Evaluación del modelo
# 4) Explicación de predicciones


# IMPORTACIÓN DE LIBRERÍAS DEL SISTEMA

import sys                          # Permite modificar rutas del sistema
import os                           # Permite trabajar con rutas del sistema operativo

# Agregamos la carpeta raíz del proyecto al PATH
# Esto permite importar módulos personalizados del proyecto
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# LIBRERÍAS PARA EL DASHBOARD

import streamlit as st              # Framework para crear dashboards web en Python
import pandas as pd                 # Librería para manipulación de datos tipo tabla
import plotly.express as px         # Librería para gráficas interactivas


# LIBRERÍAS DE SPARK PARA BIG DATA

from pyspark.sql.functions import col, when   # Funciones para manipular columnas
from pyspark.ml.feature import VectorAssembler # Convierte columnas en vector de features
from pyspark.ml.classification import DecisionTreeClassifier  # Modelo árbol de decisión
from pyspark.ml.evaluation import MulticlassClassificationEvaluator # Evaluación del modelo


# LIBRERÍAS DE MACHINE LEARNING ADICIONALES

from sklearn.metrics import confusion_matrix   # Calcula matriz de confusión
import seaborn as sns                          # Visualización estadística
import matplotlib.pyplot as plt                # Librería base de gráficas


# LIBRERÍAS PARA EXPLICABILIDAD DEL MODELO

import shap                                    # Explica predicciones de modelos
import xgboost as xgb                          # Algoritmo avanzado de ML


# FUNCIÓN PERSONALIZADA PARA CONECTAR A MONGODB

# Esta función fue creada previamente en el proyecto
# Permite obtener:
# - sesión Spark
# - dataframe cargado desde MongoDB
from config.mongo_spark_conexion_sinnulos import get_spark_session


# CONFIGURACIÓN DEL DASHBOARD

# Configuración visual de la aplicación
st.set_page_config(
    page_title="Big Data ML Dashboard",  # Título de la pestaña del navegador
    layout="wide"                        # Usa todo el ancho de la pantalla
)

# Título principal del dashboard
st.title("Dashboard Big Data + Machine Learning")


# CARGA DE DATOS DESDE MONGODB

# Se obtiene la sesión Spark y el dataframe cargado desde MongoDB
spark, df, df_vector = get_spark_session()


# FEATURE ENGINEERING

# Creamos una nueva columna llamada "ingreso"
# ingreso = cantidad * precio
df = df.withColumn(
    "ingreso",
    col("cantidad") * col("precio")
)

# Creamos la variable objetivo del modelo
# Si ingreso > 50000 → etiqueta 1 (venta alta)
# Si ingreso <= 50000 → etiqueta 0 (venta baja)
df = df.withColumn(
    "label",
    when(col("ingreso") > 50000, 1).otherwise(0)
)

# Guardamos el dataframe en memoria para acelerar operaciones
df.cache()

# Mensaje en el dashboard
st.success("Datos cargados correctamente desde MongoDB")


# FILTRO DE PRODUCTOS

# Creamos sección lateral de filtros
st.sidebar.header("Filtros")

# Obtenemos lista de productos únicos
productos = [row["producto"] for row in df.select("producto").distinct().collect()]

# Creamos menú desplegable
producto = st.sidebar.selectbox(
    "Producto",
    ["Todos"] + productos
)

# Aplicamos filtro si el usuario selecciona un producto
if producto != "Todos":
    df = df.filter(col("producto") == producto)


# AGREGACIÓN DE DATOS PARA GRÁFICAS

# Agrupamos ingresos por producto usando Spark
ventas = df.groupBy("producto") \
           .sum("ingreso") \
           .withColumnRenamed("sum(ingreso)", "ingreso")

# Convertimos el resultado a pandas
ventas_pd = ventas.toPandas()


# GRÁFICA INTERACTIVA CON PLOTLY

st.subheader(" Ingresos por producto")

# Creamos gráfica de barras interactiva
fig = px.bar(
    ventas_pd,              # dataframe
    x="producto",           # eje X
    y="ingreso",            # eje Y
    color="producto",       # color por categoría
    title="Ingresos por producto"
)

# Mostramos la gráfica en el dashboard
st.plotly_chart(fig, use_container_width=True)


# SAMPLING PARA VISUALIZACIÓN

# Tomamos una muestra del dataset (10%)
df_sample = df.sample(fraction=0.1)

# Convertimos la muestra a pandas
pdf = df_sample.toPandas()

st.subheader("Vista de muestra del dataset")

# Mostramos la tabla
st.dataframe(pdf)


# PREPARACIÓN DE DATOS PARA MACHINE LEARNING

# VectorAssembler combina varias columnas en una sola columna vectorial
assembler = VectorAssembler(
    inputCols=["cantidad", "precio", "ingreso"], # variables predictoras
    outputCol="features",                        # columna resultado
    handleInvalid="skip"                         # evita errores por datos inválidos
)

# Aplicamos transformación
df_ml = assembler.transform(df)

# Dataset final para ML
dataset = df_ml.select("features", "label")

# Dividimos dataset en entrenamiento y prueba
train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=42)


# ENTRENAMIENTO DEL MODELO

# Creamos modelo Decision Tree
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=6
)

# Entrenamos el modelo
model = dt.fit(train_data)

# Realizamos predicciones
predictions = model.transform(test_data)


# EVALUACIÓN DEL MODELO

st.subheader("Evaluación del modelo")

# Evaluador de precisión
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# Calculamos accuracy
accuracy = evaluator.evaluate(predictions)

# Mostramos métrica
st.metric("Accuracy", round(accuracy, 3))


# MATRIZ DE CONFUSIÓN

st.subheader(" Matriz de confusión")

# Convertimos predicciones a pandas
pred_pd = predictions.select("label", "prediction").toPandas()

# Calculamos matriz
cm = confusion_matrix(
    pred_pd["label"],
    pred_pd["prediction"]
)

# Crear figura
fig_cm, ax = plt.subplots()

# Graficar matriz
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

ax.set_xlabel("Predicción")
ax.set_ylabel("Valor real")

# Mostrar gráfica
st.pyplot(fig_cm)


# VISUALIZACIÓN DEL ÁRBOL

st.subheader(" Árbol de decisión aprendido")

# Mostrar reglas del árbol
st.text(model.toDebugString)


# EXPLICABILIDAD DEL MODELO CON SHAP

st.subheader("Explicación del modelo (SHAP)")

# Seleccionamos variables para explicabilidad
X = pdf[["cantidad","precio","ingreso"]]
y = pdf["label"]

# Entrenamos modelo XGBoost para explicación
xgb_model = xgb.XGBClassifier()

xgb_model.fit(X,y)

# Creamos objeto explicador
explainer = shap.TreeExplainer(xgb_model)

# Calculamos importancia de variables
shap_values = explainer.shap_values(X)

# Gráfica resumen
fig = shap.summary_plot(
    shap_values,
    X,
    show=False
)

# Mostrar en dashboard
st.pyplot(fig)


# PREDICCIÓN INTERACTIVA

st.subheader(" Predicción interactiva")

# Input del usuario
cantidad = st.number_input("Cantidad", 1, 10, 3)

precio = st.number_input("Precio", 1000, 50000, 10000)

# Calculamos ingreso
ingreso = cantidad * precio

st.write("Ingreso calculado:", ingreso)

# Botón para predicción
if st.button("Predecir"):

    # Creamos dataframe con el nuevo registro
    new_data = spark.createDataFrame(
        [(cantidad, precio, ingreso)],
        ["cantidad", "precio", "ingreso"]
    )

    # Convertimos a features
    new_data = assembler.transform(new_data)

    # Ejecutamos predicción
    result = model.transform(new_data)

    # Extraemos resultado
    pred = result.select("prediction").collect()[0][0]

    # Mostrar resultado
    if pred == 1:
        st.success("Venta ALTA")
    else:
        st.warning("Venta BAJA")