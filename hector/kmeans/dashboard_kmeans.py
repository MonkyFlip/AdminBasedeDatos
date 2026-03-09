import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from config.mongo_spark_conexion_sinnulos import get_spark_session

# CONFIGURACIÓN GENERAL

st.set_page_config(
    page_title="Dashboard Ejecutivo - Segmentación ML",
    layout="wide"
)

st.title(" Dashboard Ejecutivo - Segmentación con KMeans")
st.markdown("Machine Learning aplicado a análisis de productos")

# SLIDER DINÁMICO PARA K

k_value = st.sidebar.slider(
    "Número de Clusters (K)",
    min_value=2,
    max_value=8,
    value=3
)

# FUNCIÓN PRINCIPAL

@st.cache_resource
def cargar_y_entrenar(k):

    spark, df, df_vector = get_spark_session()

    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(df_vector)
    result = model.transform(df_vector)

    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(result)

    pdf = result.select(
        "producto",
        "cantidad",
        "precio",
        "ingreso",
        "prediction"
    ).toPandas()

    spark.stop()

    return pdf, silhouette


data, silhouette_score = cargar_y_entrenar(k_value)

# MÉTRICAS EJECUTIVAS

col1, col2, col3, col4 = st.columns(4)

col1.metric("Productos Totales", len(data))
col2.metric("Ingreso Total", round(data["ingreso"].sum(), 2))
col3.metric("Ingreso Promedio", round(data["ingreso"].mean(), 2))
col4.metric("Silhouette Score", round(silhouette_score, 4))

# FILTRO DE CLUSTERS

cluster_filter = st.sidebar.multiselect(
    "Filtrar por Cluster",
    sorted(data["prediction"].unique()),
    default=sorted(data["prediction"].unique())
)

data_filtrada = data[data["prediction"].isin(cluster_filter)]

# GRÁFICA INTERACTIVA 2D

st.subheader(" Precio vs Ingreso (Interactivo)")

fig = px.scatter(
    data_filtrada,
    x="precio",
    y="ingreso",
    color="prediction",
    hover_data=["producto", "cantidad"],
    title="Segmentación de Productos"
)

st.plotly_chart(fig, use_container_width=True)

# GRÁFICA 3D INTERACTIVA

st.subheader(" Visualización 3D")

fig_3d = px.scatter_3d(
    data_filtrada,
    x="precio",
    y="cantidad",
    z="ingreso",
    color="prediction",
    hover_name="producto"
)

st.plotly_chart(fig_3d, use_container_width=True)

# MÉTODO DEL CODO

st.subheader("📉 Método del Codo")

def metodo_del_codo():

    spark, df, df_vector = get_spark_session()

    wcss = []

    for k in range(2, 9):
        kmeans = KMeans(k=k, seed=1)
        model = kmeans.fit(df_vector)
        wcss.append(model.summary.trainingCost)

    spark.stop()

    return wcss


wcss_values = metodo_del_codo()

fig_elbow = go.Figure()

fig_elbow.add_trace(
    go.Scatter(
        x=list(range(2, 9)),
        y=wcss_values,
        mode="lines+markers"
    )
)

fig_elbow.update_layout(
    xaxis_title="Número de Clusters (K)",
    yaxis_title="WCSS",
    title="Método del Codo"
)

st.plotly_chart(fig_elbow, use_container_width=True)

# INTERPRETACIÓN AUTOMÁTICA

st.subheader(" Interpretación Ejecutiva")

resumen = data.groupby("prediction").mean(numeric_only=True)

for cluster in resumen.index:

    ingreso = resumen.loc[cluster, "ingreso"]
    precio = resumen.loc[cluster, "precio"]

    st.markdown(f"### Cluster {cluster}")
    st.write(f"Ingreso promedio: {round(ingreso,2)}")
    st.write(f"Precio promedio: {round(precio,2)}")

    if ingreso == resumen["ingreso"].max():
        st.success("Segmento Estratégico (Mayor ingreso)")
    elif precio == resumen["precio"].max():
        st.info("Segmento Premium")
    else:
        st.warning("Segmento Operativo")