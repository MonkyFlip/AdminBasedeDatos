# =========================================================
# PCA + CLUSTERING (KMEANS) EN PYSPARK
# =========================================================
# OBJETIVO:
# Reducir dimensiones (PCA) y encontrar patrones (clusters)
# =========================================================


# ================== IMPORTACIONES ==================

# Función personalizada para iniciar Spark y cargar datos desde MongoDB
from config.mongo_spark_conexion import get_spark_session

# Librerías de Machine Learning en PySpark
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

# Funciones para manipular datos en Spark
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

# Librerías para visualización
import pandas as pd
import plotly.express as px
import plotly.io as pio


# ================== 1. INICIAR SPARK ==================

# Se inicia la sesión de Spark y se cargan los datos
# df = datos originales
# df_vector = datos ya convertidos a formato vector (columna "features")
spark, df, df_vector = get_spark_session()

# 1. Limpiar datos (recomendado)
df = df.fillna({
    "cantidad": 0,
    "precio": 0,
    "ingreso": 0
})

# 2. Crear vector correctamente
assembler = VectorAssembler(
    inputCols=["cantidad", "precio", "ingreso"],
    outputCol="features",
    handleInvalid="skip"
)

df_vector = assembler.transform(df)

# Mensaje de control para verificar que todo cargó correctamente
print(" Datos cargados correctamente")


# ================== 2. ESCALADO ==================

# IMPORTANTE:
# PCA es sensible a la escala, por lo que debemos estandarizar los datos

print(" Escalando datos...")

# Se define el escalador
scaler = StandardScaler(
    inputCol="features",        # columna original con variables
    outputCol="scaledFeatures", # nueva columna con datos escalados
    withStd=True,               # divide por desviación estándar
    withMean=True               # centra los datos en media 0
)

# Se entrena el modelo de escalado (calcula media y desviación)
scaler_model = scaler.fit(df_vector)

# Se aplica la transformación (estandarización)
df_scaled = scaler_model.transform(df_vector)

print(" Datos escalados correctamente")


# ================== 3. PCA ==================

# PCA reduce dimensiones creando nuevas variables (componentes principales)

print(" Aplicando PCA...")

# Se define el modelo PCA
pca = PCA(
    k=2,                       # número de componentes finales (2D para visualización)
    inputCol="scaledFeatures", # datos de entrada (ya escalados)
    outputCol="pcaFeatures"    # nueva columna con componentes principales
)

# Se entrena el modelo PCA
pca_model = pca.fit(df_scaled)

# Se transforma el dataset → ahora tiene PC1 y PC2
df_pca = pca_model.transform(df_scaled)

print(" PCA aplicado correctamente")


# ================== 4. VARIANZA EXPLICADA ==================

# Indica cuánta información conserva cada componente
print("\n Varianza explicada:")
print(pca_model.explainedVariance)


# ================== 5. CLUSTERING ==================

# Aplicamos KMeans para agrupar los datos

print("\n Aplicando KMeans...")

# Definición del modelo KMeans
kmeans = KMeans(
    k=3,                       # número de clusters (grupos)
    featuresCol="pcaFeatures", # usamos los datos reducidos
    predictionCol="cluster"    # nueva columna con el cluster asignado
)

# Entrenamos el modelo
kmeans_model = kmeans.fit(df_pca)

# Asignamos cada punto a un cluster
df_cluster = kmeans_model.transform(df_pca)

print(" Clustering aplicado correctamente")


# ================== 6. CENTROIDES ==================

# Los centroides representan el "promedio" de cada cluster
centroids = kmeans_model.clusterCenters()

print("\n Centroides de clusters:")

# Recorremos cada centroide
for i, c in enumerate(centroids):
    print(f"Cluster {i}: {c}")


# ================== 7. INTERPRETACIÓN AUTOMÁTICA ==================

# Función para traducir números a significado
def interpretar_cluster(centroid):
    
    # Separamos componentes principales
    pc1, pc2 = centroid

    descripcion = []

    # Interpretación del primer componente
    if pc1 > 0:
        descripcion.append("valores ALTOS en variables principales")
    else:
        descripcion.append("valores BAJOS en variables principales")

    # Interpretación del segundo componente
    if pc2 > 0:
        descripcion.append("segunda componente POSITIVA")
    else:
        descripcion.append("segunda componente NEGATIVA")

    # Unimos la descripción
    return ", ".join(descripcion)


print("\n Interpretación de clusters:")

# Aplicamos interpretación a cada cluster
for i, c in enumerate(centroids):
    interpretacion = interpretar_cluster(c)
    print(f"Cluster {i}: {interpretacion}")


# ================== 8. COMPONENTES PCA ==================

# Muestra cuánto aporta cada variable original a los componentes
print("\n Pesos de los componentes principales:")
print(pca_model.pc)


# ================== 9. SOLUCIÓN ERROR DenseVector ==================

# PROBLEMA:
# Pandas no entiende el tipo DenseVector de Spark

# SOLUCIÓN:
# Convertimos DenseVector → lista de Python

vector_to_array = udf(
    lambda v: v.toArray().tolist(),  # convierte vector a lista
    ArrayType(DoubleType())          # tipo de dato resultante
)

# Creamos nueva columna con formato compatible
df_final = df_cluster.withColumn("pcaArray", vector_to_array("pcaFeatures"))


# ================== 10. CONVERSIÓN A PANDAS ==================

print("\n Generando DataFrame para visualización...")

# Convertimos a Pandas (ya sin error)
pdf = df_final.select("pcaArray", "cluster").toPandas()

# Separamos las componentes principales
pdf["PC1"] = pdf["pcaArray"].apply(lambda x: x[0])
pdf["PC2"] = pdf["pcaArray"].apply(lambda x: x[1])


# ================== 11. VISUALIZACIÓN ==================

print(" Generando gráfica...")

# Creamos gráfica de dispersión
fig = px.scatter(
    pdf,
    x="PC1",                         # eje X
    y="PC2",                         # eje Y
    color=pdf["cluster"].astype(str),# color por cluster
    title="PCA + KMeans Clustering"
)

# Mostramos gráfica
pio.renderers.default = "browser"

fig.show()


# ================== 12. CIERRE ==================

# Cerramos sesión de Spark para liberar recursos
spark.stop()

print("\n Proceso finalizado correctamente")