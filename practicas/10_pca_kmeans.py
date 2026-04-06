import streamlit as st
from common.mongo_spark import get_spark_df

from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

import pandas as pd
import matplotlib.pyplot as plt

# ================== INTERFAZ ==================

st.title("PCA + K-Means: Segmentación Avanzada de Clientes del Gimnasio")

st.write(
    "En esta práctica se aplica PCA para reducir dimensiones y posteriormente "
    "K-Means para identificar patrones de clientes del gimnasio."
)

# ================== 1. CARGA DE DATOS ==================

df = get_spark_df("PCA_KMeans_Gym")

df = df.fillna({
    "peso": 0,
    "edad": 0,
    "meses_inscrito": 0
})

# ================== 2. VECTOR DE CARACTERÍSTICAS ==================

assembler = VectorAssembler(
    inputCols=["peso", "edad", "meses_inscrito"],
    outputCol="features",
    handleInvalid="skip"
)

df_vector = assembler.transform(df)

# ================== 3. ESCALADO ==================

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

# ================== 4. PCA ==================

pca = PCA(
    k=2,
    inputCol="scaledFeatures",
    outputCol="pcaFeatures"
)

pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)

st.subheader("Varianza explicada por cada componente")
st.write(pca_model.explainedVariance.toArray())

# ================== 5. K-MEANS ==================

kmeans = KMeans(
    k=3,
    featuresCol="pcaFeatures",
    predictionCol="cluster",
    seed=42
)

kmeans_model = kmeans.fit(df_pca)
df_cluster = kmeans_model.transform(df_pca)

# ================== 6. CONVERSIÓN DenseVector ==================

vector_to_array = udf(
    lambda v: v.toArray().tolist(),
    ArrayType(DoubleType())
)

df_final = df_cluster.withColumn(
    "pcaArray",
    vector_to_array("pcaFeatures")
)

# ================== 7. A PANDAS ==================

pdf = df_final.select("pcaArray", "cluster").toPandas()

pdf["PC1"] = pdf["pcaArray"].apply(lambda x: x[0])
pdf["PC2"] = pdf["pcaArray"].apply(lambda x: x[1])

# ================== 8. VISUALIZACIÓN ==================

st.subheader("Visualización de Clusters en Espacio PCA")

fig, ax = plt.subplots()

scatter = ax.scatter(
    pdf["PC1"],
    pdf["PC2"],
    c=pdf["cluster"],
    cmap="viridis",
    alpha=0.6
)

ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
ax.set_title("Clusters de Clientes del Gimnasio (PCA + K-Means)")

legend = ax.legend(
    *scatter.legend_elements(),
    title="Cluster"
)
ax.add_artist(legend)

st.pyplot(fig)
