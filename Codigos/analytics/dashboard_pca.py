# ============================================================
# DASHBOARD PROFESIONAL TIPO POWER BI
# PCA + CLUSTERING + IA + LOGIN
# ============================================================

# ================== IMPORTACIONES ==================
import sys
import os

# Agregar ruta raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st  # Framework web para dashboards
import pandas as pd  # Manejo de datos
import plotly.express as px  # Visualizaciones interactivas

# PySpark ML
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, avg
from pyspark.ml.functions import vector_to_array

# Conexión a datos (tu función existente)
from config.mongo_spark_conexion import get_spark_session

# ================== CONFIGURACIÓN UI ==================
st.set_page_config(page_title="Dashboard Inteligente", layout="wide")

# Título principal
st.title(" Dashboard Inteligente: PCA + Clustering + IA")

# Sidebar (tipo Power BI)
st.sidebar.header(" Configuración")

# Selector de clusters
k_clusters = st.sidebar.slider("Número de clusters", 2, 6, 3)

# ================== FUNCIÓN IA AVANZADA ==================
def interpretar_clusters_avanzado(perfil_df):
    """
    Genera interpretación inteligente tipo analista
    """

    interpretaciones = []

    # Promedios globales
    avg_global = perfil_df.mean(numeric_only=True)

    # Iterar cada cluster
    for _, row in perfil_df.iterrows():

        cluster = int(row["cluster"])
        ingreso = row["avg_ingreso"]
        precio = row["avg_precio"]
        cantidad = row["avg_cantidad"]

        texto = f" Cluster {cluster}:\n"

        # ===== Comparaciones =====
        if ingreso > avg_global["avg_ingreso"]:
            texto += "- Ingreso superior al promedio\n"
        else:
            texto += "- Ingreso inferior al promedio\n"

        if precio > avg_global["avg_precio"]:
            texto += "- Compra productos caros\n"
        else:
            texto += "- Compra productos económicos\n"

        if cantidad > avg_global["avg_cantidad"]:
            texto += "- Compra en volumen\n"
        else:
            texto += "- Compra poco\n"

        # ===== Perfil =====
        if ingreso > avg_global["avg_ingreso"] and precio > avg_global["avg_precio"]:
            perfil = "Cliente Premium"
        elif ingreso < avg_global["avg_ingreso"] and cantidad > avg_global["avg_cantidad"]:
            perfil = "Cazador de Ofertas"
        else:
            perfil = "Cliente Estándar"

        texto += f"\n Perfil: {perfil}\n"

        # ===== Recomendaciones =====
        texto += "\n Recomendaciones:\n"

        if perfil == "Cliente Premium":
            texto += "- Productos premium\n- Programas VIP\n"
        elif perfil == "Cazador de Ofertas":
            texto += "- Descuentos\n- Promociones\n"
        else:
            texto += "- Estrategias mixtas\n"

        interpretaciones.append(texto)

    return interpretaciones

# ================== BOTÓN PRINCIPAL ==================
if st.button(" Ejecutar análisis"):

    # ================== 1. CARGA ==================
    spark, df, _ = get_spark_session()

    # ================== 2. LIMPIEZA ==================
    df = df.fillna({
        "cantidad": 0,
        "precio": 0,
        "ingreso": 0
    })

    # ================== 3. FEATURES ==================
    assembler = VectorAssembler(
        inputCols=["cantidad", "precio", "ingreso"],
        outputCol="features",
        handleInvalid="skip"
    )

    df_vector = assembler.transform(df)

    # ================== 4. ESCALADO ==================
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )

    df_scaled = scaler.fit(df_vector).transform(df_vector)

    # ================== 5. PCA ==================
    pca = PCA(k=2, inputCol="scaledFeatures", outputCol="pcaFeatures")
    pca_model = pca.fit(df_scaled)
    df_pca = pca_model.transform(df_scaled)

    # Varianza
    varianza = pca_model.explainedVariance

    col1, col2 = st.columns(2)
    col1.metric("Varianza PC1", f"{round(varianza[0]*100,2)}%")
    col2.metric("Varianza PC2", f"{round(varianza[1]*100,2)}%")

    # ================== 6. KMEANS ==================
    kmeans = KMeans(k=k_clusters, featuresCol="pcaFeatures", predictionCol="cluster")
    model = kmeans.fit(df_pca)
    df_cluster = model.transform(df_pca)

    # ================== 7. EXTRAER COMPONENTES ==================
    df_cluster = df_cluster.withColumn(
        "pcaArray", vector_to_array(col("pcaFeatures"))
    )

    df_cluster = df_cluster.withColumn("PC1", col("pcaArray")[0])
    df_cluster = df_cluster.withColumn("PC2", col("pcaArray")[1])

    # Convertir a pandas
    pdf = df_cluster.select("PC1", "PC2", "cluster",
                            "cantidad", "precio", "ingreso").toPandas()

    # ================== 8. VISUALIZACIÓN ==================
    st.subheader(" Mapa de clusters")

    fig = px.scatter(
        pdf,
        x="PC1",
        y="PC2",
        color=pdf["cluster"].astype(str),
        title="Segmentación de clientes"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================== 9. PERFIL ==================
    perfil = df_cluster.groupBy("cluster").agg(
        avg("cantidad").alias("avg_cantidad"),
        avg("precio").alias("avg_precio"),
        avg("ingreso").alias("avg_ingreso")
    ).toPandas()

    st.subheader(" Perfil de clusters")
    st.dataframe(perfil)

    # ================== 10. IA ==================
    st.subheader(" Interpretación inteligente")

    interpretaciones = interpretar_clusters_avanzado(perfil)

    for texto in interpretaciones:
        st.code(texto)

    # ================== 11. DATOS ==================
    st.subheader(" Datos")
    st.dataframe(pdf.head(100))

    spark.stop()

# ============================================================
# FIN DEL SISTEMA
# ============================================================
