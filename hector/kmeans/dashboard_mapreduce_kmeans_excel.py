# AJUSTE DE PATH (para arquitectura modular)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# IMPORTACIONES
import streamlit as st
from config.mongo_spark_conexion import get_spark_session
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import sum as spark_sum, col, to_date
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# CONFIGURACIÓN DE LA APP
st.set_page_config(page_title="Dashboard Analítico KMeans", layout="wide")
st.title(" Dashboard Inteligente - Segmentación con KMeans")

# CARGA DE DATOS (cacheada para rendimiento)
@st.cache_resource
def load_data():
    spark, df, _ = get_spark_session()
    return spark, df

spark, df = load_data()

# Convertir fecha correctamente
df = df.withColumn("fecha", to_date(col("fecha")))

# PRE-CÁLCULO DE VALORES MIN/MAX (optimización)
stats = df.agg(
    spark_sum("precio").alias("sum_precio")
)

min_precio = float(df.agg({"precio": "min"}).collect()[0][0])
max_precio = float(df.agg({"precio": "max"}).collect()[0][0])
min_ingreso = float(df.agg({"ingreso": "min"}).collect()[0][0])
max_ingreso = float(df.agg({"ingreso": "max"}).collect()[0][0])

# FILTROS DINÁMICOS
st.sidebar.header(" Filtros")

precio_min, precio_max = st.sidebar.slider(
    "Rango de Precio",
    min_precio,
    max_precio,
    (min_precio, max_precio)
)

ingreso_min, ingreso_max = st.sidebar.slider(
    "Rango de Ingreso",
    min_ingreso,
    max_ingreso,
    (min_ingreso, max_ingreso)
)

df_filtrado = df.filter(
    (col("precio") >= precio_min) &
    (col("precio") <= precio_max) &
    (col("ingreso") >= ingreso_min) &
    (col("ingreso") <= ingreso_max)
)

conteo = df_filtrado.count()
st.write(f"### Registros después del filtro: {conteo}")

# KMEANS SOBRE DATOS FILTRADOS
st.sidebar.header("⚙ Configuración KMeans")
k = st.sidebar.slider("Número de Clusters (K)", 2, 6, 3)

if conteo < k:
    st.warning("No hay suficientes datos para el número de clusters seleccionado.")
    st.stop()

# Vectorización dinámica
assembler = VectorAssembler(
    inputCols=["precio", "cantidad", "ingreso"],
    outputCol="features"
)

df_vector = assembler.transform(df_filtrado)

# Entrenamiento del modelo
kmeans = KMeans(k=k, seed=1, featuresCol="features")
model = kmeans.fit(df_vector)

result = model.transform(df_vector)

# Evaluación
evaluator = ClusteringEvaluator()
score = evaluator.evaluate(result)

st.subheader(" Resultado de Clustering")
st.metric("Silhouette Score", round(score, 4))

# VISUALIZACIÓN 2D
st.subheader(" Clusters (Precio vs Ingreso)")

result_pd = result.select(
    "producto", "precio", "cantidad", "ingreso", "prediction"
).toPandas()

fig = plt.figure(figsize=(8,6))

for cluster in result_pd["prediction"].unique():
    subset = result_pd[result_pd["prediction"] == cluster]
    plt.scatter(
        subset["precio"],
        subset["ingreso"],
        label=f"Cluster {cluster}"
    )

plt.xlabel("Precio")
plt.ylabel("Ingreso")
plt.legend()
st.pyplot(fig)

# ANÁLISIS TEMPORAL
st.subheader(" Ingreso por Fecha")

temporal = df_filtrado.groupBy("fecha").agg(
    spark_sum("ingreso").alias("total_ingreso")
).orderBy("fecha").toPandas()

fig2 = plt.figure(figsize=(8,5))
plt.plot(temporal["fecha"], temporal["total_ingreso"])
plt.xticks(rotation=45)
plt.xlabel("Fecha")
plt.ylabel("Ingreso")
st.pyplot(fig2)

# PARETO 80/20
st.subheader(" Análisis Pareto 80/20")

pareto = df_filtrado.groupBy("producto").agg(
    spark_sum("ingreso").alias("total_ingreso")
).orderBy(col("total_ingreso").desc()).toPandas()

pareto["porcentaje"] = pareto["total_ingreso"] / pareto["total_ingreso"].sum()
pareto["acumulado"] = pareto["porcentaje"].cumsum()

productos_80 = pareto[pareto["acumulado"] <= 0.8]

st.write(f"Productos que generan el 80% del ingreso: {len(productos_80)}")

fig3 = plt.figure(figsize=(10,5))
plt.bar(pareto["producto"], pareto["acumulado"])
plt.xticks(rotation=90)
st.pyplot(fig3)

# EXPORTACIÓN A EXCEL
st.subheader("⬇ Exportar Resultados")

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()

excel_data = to_excel(result_pd)

st.download_button(
    label="Descargar Clusters en Excel",
    data=excel_data,
    file_name="clusters_kmeans.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.success(" Dashboard Analítico Empresarial con Machine Learning Distribuido")