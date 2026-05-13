# Estas librerías permiten manipular el path del proyecto
# para que Streamlit pueda encontrar otros módulos del sistema
import sys
import os

# Agregamos la carpeta raíz del proyecto al PATH de Python
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Librería para crear aplicaciones web de manera sencilla
import streamlit as st

# Librería para manipulación de datos en formato tabla
import pandas as pd

# Librería para crear gráficas
import matplotlib.pyplot as plt


# Componentes de Machine Learning en Spark
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Función que creamos previamente para conectarnos a MongoDB
from config.mongo_spark_conexion_sinnulos import get_spark_session



# CONFIGURACIÓN DEL DASHBOARD

# Configuración general de la aplicación
st.set_page_config(
    page_title="Dashboard Ventas ML",
    layout="wide"
)

# Título principal
st.title(" Dashboard de Ventas con Machine Learning")

st.write(
    """
    Este dashboard permite analizar ventas y entrenar
    un modelo de Machine Learning usando Árboles de Decisión.
    """
)



# CARGAR DATOS DESDE MONGODB

# Obtenemos la sesión de Spark y los datos
spark, df, df_vector = get_spark_session()

# Creamos una nueva columna llamada ingreso
# ingreso = cantidad * precio
df = df.withColumn(
    "ingreso",
    col("cantidad") * col("precio")
)

# Creamos la variable objetivo (label)
# Si ingreso > 50000 → venta alta
# Si ingreso <= 50000 → venta baja
df = df.withColumn(
    "label",
    when(col("ingreso") > 50000, 1).otherwise(0)
)

# Convertimos el DataFrame de Spark a Pandas
# Esto facilita mostrar los datos en Streamlit
pandas_df = df.toPandas()



# FILTRO DE PRODUCTOS

# Creamos un menú lateral
st.sidebar.header("Filtros")

# Obtenemos lista de productos
productos = pandas_df["producto"].unique()

# Creamos selector
producto_filtro = st.sidebar.selectbox(
    "Seleccionar producto",
    ["Todos"] + list(productos)
)

# Aplicamos filtro si el usuario selecciona un producto
if producto_filtro != "Todos":
    pandas_df = pandas_df[pandas_df["producto"] == producto_filtro]



# TABLA DE DATOS

st.subheader("Datos de ventas")

# Mostramos los datos en formato tabla
st.dataframe(pandas_df)



# GRÁFICA DE VENTAS POR PRODUCTO

st.subheader(" Ventas por producto")

# Agrupamos ingresos por producto
ventas_producto = pandas_df.groupby("producto")["ingreso"].sum()

# Creamos figura
fig, ax = plt.subplots()

# Gráfica de barras
ventas_producto.plot(
    kind="bar",
    ax=ax
)

# Mostramos la gráfica en Streamlit
st.pyplot(fig)



# PREPARACIÓN DE DATOS PARA MACHINE LEARNING

# Convertimos nuevamente a DataFrame de Spark
df = spark.createDataFrame(pandas_df)

# VectorAssembler combina columnas en un vector de características
assembler = VectorAssembler(
    inputCols=["cantidad", "precio", "ingreso"],
    outputCol="features",
    handleInvalid="skip"
)

# Aplicamos transformación
df_ml = assembler.transform(df)

# Dataset final para ML
dataset = df_ml.select("features", "label")

# Dividimos datos en entrenamiento y prueba
train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=42)



# ENTRENAMIENTO DEL MODELO

# Creamos modelo de árbol de decisión
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5
)

# Entrenamos el modelo
model = dt.fit(train_data)

# Realizamos predicciones
predictions = model.transform(test_data)



# MÉTRICAS DEL MODELO

st.subheader("Métricas del modelo")

# Evaluador para medir desempeño
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# Calculamos accuracy
accuracy = evaluator.evaluate(predictions)

# Creamos columnas para mostrar métricas
col1, col2 = st.columns(2)

# Mostramos precisión
col1.metric("Accuracy", round(accuracy, 3))

# Número total de registros
total = pandas_df.shape[0]

col2.metric("Registros", total)



# VISUALIZACIÓN DEL ÁRBOL

st.subheader(" Árbol de decisión aprendido")

# Mostramos las reglas del árbol
st.text(model.toDebugString)



# PREDICCIÓN INTERACTIVA

st.subheader("Predicción de ventas")

# Input de usuario
cantidad = st.number_input(
    "Cantidad",
    min_value=1,
    max_value=10,
    value=3
)

precio = st.number_input(
    "Precio",
    min_value=1000,
    max_value=50000,
    value=10000
)

# Calculamos ingreso
ingreso = cantidad * precio

st.write("Ingreso calculado:", ingreso)


# Botón de predicción
if st.button("Predecir"):

    # Creamos nuevo registro
    new_data = spark.createDataFrame(
        [(cantidad, precio, ingreso)],
        ["cantidad", "precio", "ingreso"]
    )

    # Convertimos a features
    new_data = assembler.transform(new_data)

    # Predicción
    prediction = model.transform(new_data)

    result = prediction.select("prediction").collect()[0][0]

    # Mostrar resultado
    if result == 1:
        st.success("Venta ALTA")
    else:
        st.warning("Venta BAJA")