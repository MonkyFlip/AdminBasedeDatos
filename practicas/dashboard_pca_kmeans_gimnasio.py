# ============================================================
# DASHBOARD PROFESIONAL TIPO POWER BI
# PCA + CLUSTERING + IA (GIMNASIO)
# ============================================================

import sys
import os

# Agregar ruta raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px

from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, avg
from pyspark.ml.functions import vector_to_array

# Conexión común a Mongo + Spark
from common.mongo_spark import get_spark_df

# ================== CONFIGURACIÓN UI ==================
st.set_page_config(page_title="Dashboard Inteligente - Gimnasio", layout="wide")

st.title("🏋️ Dashboard Inteligente: PCA + Clustering (Gimnasio)")

st.sidebar.header("⚙️ Configuración")

k_clusters = st.sidebar.slider("Número de clusters", 2, 6, 3)

# ================== FUNCIÓN IA ==================
def interpretar_clusters(perfil_df):
    interpretaciones = []
    avg_global = perfil_df.mean(numeric_only=True)

    for _, row in perfil_df.iterrows():
        cluster = int(row["cluster"])
        peso = row["avg_peso"]
        precio = row["avg_precio"]
        meses = row["avg_meses"]

        texto = f"Cluster {cluster}:\n"

        texto += "- Peso alto\n" if peso > avg_global["avg_peso"] else "- Peso bajo\n"
        texto += "- Membresía cara\n" if precio > avg_global["avg_precio"] else "- Membresía económica\n"
        texto += "- Cliente antiguo\n" if meses > avg_global["avg_meses"] else "- Cliente reciente\n"

        if precio > avg_global["avg_precio"] and meses > avg_global["avg_meses"]:
            perfil = "Cliente Premium"
        elif precio < avg_global["avg_precio"] and meses < avg_global["avg_meses"]:
            perfil = "Cliente Nuevo"
        else:
            perfil = "Cliente Estándar"

        texto += f"\nPerfil: {perfil}\n"

        texto += "\nRecomendaciones:\n"
        if perfil == "Cliente Premium":
            texto += "- Programas VIP\n- Entrenador personal\n"
        elif perfil == "Cliente Nuevo":
            texto += "- Promociones\n- Seguimiento inicial\n"
        else:
            texto += "- Planes mixtos\n"

        interpretaciones.append(texto)

    return interpretaciones

# ================== BOTÓN PRINCIPAL ==================
if st.button("🚀 Ejecutar análisis"):

    # ================== 1. CARGA ==================
    df = get_spark_df("Dashboard_PCA_Gym")

    # ================== 2. LIMPIEZA ==================
    df = df.fillna({
        "peso": 0,
        "precio_mensual": 0,
        "meses_inscrito": 0
    })

    # ================== 3. FEATURES ==================
    assembler = VectorAssembler(
        inputCols=["peso", "precio_mensual", "meses_inscrito"],
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

    varianza = pca_model.explainedVariance

    col1, col2 = st.columns(2)
    col1.metric("Varianza PC1", f"{round(varianza[0]*100,2)}%")
    col2.metric("Varianza PC2", f"{round(varianza[1]*100,2)}%")

    # ================== 6. KMEANS ==================
    kmeans = KMeans(
        k=k_clusters,
        featuresCol="pcaFeatures",
        predictionCol="cluster",
        seed=42
    )

    model = kmeans.fit(df_pca)
    df_cluster = model.transform(df_pca)

    # ================== 7. COMPONENTES ==================
    df_cluster = df_cluster.withColumn(
        "pcaArray", vector_to_array(col("pcaFeatures"))
    )

    df_cluster = df_cluster.withColumn("PC1", col("pcaArray")[0])
    df_cluster = df_cluster.withColumn("PC2", col("pcaArray")[1])

    pdf = df_cluster.select(
        "PC1", "PC2", "cluster",
        "peso", "precio_mensual", "meses_inscrito"
    ).toPandas()

    # ================== 8. VISUALIZACIÓN ==================
    st.subheader("📊 Mapa de clusters")

    fig = px.scatter(
        pdf,
        x="PC1",
        y="PC2",
        color=pdf["cluster"].astype(str),
        title="Segmentación de clientes del gimnasio"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================== 9. PERFIL ==================
    perfil = df_cluster.groupBy("cluster").agg(
        avg("peso").alias("avg_peso"),
        avg("precio_mensual").alias("avg_precio"),
        avg("meses_inscrito").alias("avg_meses")
    ).toPandas()

    st.subheader("📋 Perfil de clusters")
    st.dataframe(perfil)

    # ================== 10. IA ==================
    st.subheader("🤖 Interpretación inteligente")

    interpretaciones = interpretar_clusters(perfil)

    for texto in interpretaciones:
        st.code(texto)

    # ================== 11. DATOS ==================
    st.subheader("🧾 Datos (muestra)")
    st.dataframe(pdf.head(100))

# ============================================================
# FIN DEL DASHBOARD
# ============================================================
