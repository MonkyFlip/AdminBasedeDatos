import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# DASHBOARD PROFESIONAL DE ANALÍTICA PREDICTIVA
# Este dashboard permite:
# - Filtrar productos
# - Visualizar variación de datos
# - Ejecutar modelos de Machine Learning
# - Comparar resultados de los modelos

# LIBRERÍAS

import streamlit as st

# Conexión con Spark y base de datos
from config.mongo_spark_conexion_sinnulos import get_spark_session

# Función que ejecuta los modelos de regresión
from ml_algorithms.regresion_analytics_modelos_dash import ejecutar_modelos

# Funciones de visualización
from processing.regresion_analytics_graficos_dash import (
    grafica_dispersion,
    grafica_distribucion,
    grafica_precio_vs_ingreso,
    grafica_modelos
)

# CONFIGURACIÓN DEL DASHBOARD

st.set_page_config(
    page_title="Analytics Modelos de Regresión",
    layout="wide"
)

st.title("Plataforma de Analítica Predictiva con Spark")

# CONEXIÓN A SPARK

# get_spark_session devuelve 3 valores
# spark  sesión de Spark
# _      objeto que no usaremos
# df     dataset principal

spark, _, df = get_spark_session()

# FILTRO INTERACTIVO

st.sidebar.header("Filtros de Datos")

# Obtenemos lista de productos únicos
productos = df.select("producto").distinct().toPandas()

# Selector de producto
producto = st.sidebar.selectbox(
    "Seleccionar Producto",
    ["Todos"] + list(productos["producto"])
)

# Aplicar filtro
if producto != "Todos":

    df_filtrado = df.filter(df.producto == producto)

else:

    df_filtrado = df

# CONVERSIÓN A PANDAS PARA VISUALIZACIÓN

df_pandas = df_filtrado.toPandas()

# VISUALIZACIONES DE DATOS

st.subheader("Análisis Exploratorio de Datos")

col1, col2 = st.columns(2)

# GRÁFICA 1
# Relación Cantidad vs Ingreso

with col1:

    fig = grafica_dispersion(df_pandas)

    st.plotly_chart(fig, use_container_width=True)

# GRÁFICA 2
# Relación Precio vs Ingreso

with col2:

    fig = grafica_precio_vs_ingreso(df_pandas)

    st.plotly_chart(fig, use_container_width=True)

# GRÁFICA 3
# Distribución de ingresos

st.plotly_chart(
    grafica_distribucion(df_pandas),
    use_container_width=True
)

# EJECUCIÓN DE MODELOS DE MACHINE LEARNING

st.subheader("Modelos de Machine Learning")

resultados, predicciones = ejecutar_modelos(df_filtrado)

# COMPARACIÓN DE MODELOS

st.plotly_chart(
    grafica_modelos(resultados),
    use_container_width=True
)

# TABLA DE RESULTADOS

st.subheader("Resultados de Evaluación")

st.dataframe(resultados)

# CERRAR SESIÓN SPARK

spark.stop()