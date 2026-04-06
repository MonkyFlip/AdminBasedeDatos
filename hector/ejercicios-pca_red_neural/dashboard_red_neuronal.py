#  SISTEMA COMPLETO: SPARK + DEEP LEARNING + DASHBOARD

# Este sistema hace:
# 1. Procesa datos con Spark
# 2. Prepara variables (features)
# 3. Convierte datos a formato PyTorch
# 4. Entrena una red neuronal
# 5. Visualiza resultados
# 6. Explica automáticamente la predicción


# 1. IMPORTACIONES

import streamlit as st  
# Streamlit permite crear aplicaciones web interactivas (dashboard)

import pandas as pd  
# Pandas sirve para manejar datos en formato tabla (DataFrame)

import plotly.express as px  
# Plotly permite crear gráficas interactivas tipo Power BI


# LIBRERÍAS DE SPARK (PROCESAMIENTO DE DATOS)

from pyspark.sql import SparkSession  
# Permite crear una sesión de Spark (motor distribuido)

from pyspark.ml.feature import VectorAssembler  
# Convierte columnas en un vector (necesario para ML)

from pyspark.ml.feature import StandardScaler  
# Escala los datos (media=0, desviación=1)

# IMPORTANTE (SOLUCIÓN ERROR ARROW)
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

# LIBRERÍAS DE DEEP LEARNING (PYTORCH)

import torch  
# Librería principal de deep learning

import torch.nn as nn  
# Permite construir redes neuronales

import torch.optim as optim  
# Contiene algoritmos para entrenar el modelo

#Spark para preparar datos, PyTorch para aprender patrones y Streamlit para visualizar e interpretar resultados en tiempo real.”


# 2. CONFIGURACIÓN DE LA APP

st.set_page_config(
    page_title="Sistema IA",  # Nombre de la pestaña
    layout="wide"                    # Diseño ancho 
)

st.title(" Sistema Inteligente: Spark + Deep Learning")

st.markdown("""
Este dashboard muestra el flujo completo de un sistema de inteligencia artificial:

1. Datos en Spark  
2. Procesamiento (features)  
3. Conversión a tensor  
4. Entrenamiento de red neuronal  
5. Predicción  
6. Interpretación automática  
""")


#  3. CREAR SESIÓN SPARK

@st.cache_resource  
# Evita recrear Spark cada vez que interactúas (optimiza rendimiento)
def crear_spark():
    # Usamos paréntesis (evita error de "\" con comentarios)
    return (
        SparkSession.builder
        .appName("App_DeepLearning")
        .getOrCreate()
    )

spark = crear_spark()


# 4. CREAR DATOS (SIMULACIÓN)

# Creamos un dataset de ejemplo
data = [
    (1, 10, 0),
    (2, 20, 0),
    (3, 15, 1),
    (4, 25, 1),
    (5, 30, 1),
    (6, 35, 2),
    (7, 40, 2),
    (8, 45, 2),
    (9, 50, 2),
    (10, 55, 2)
]

# Creamos DataFrame de Spark
df = spark.createDataFrame(data, ["cantidad", "precio", "label"])

# Mostramos datos
st.subheader(" Datos originales (Spark)")
st.dataframe(df.toPandas())


# 5. FEATURE ENGINEERING (SPARK)

# VectorAssembler une varias columnas en un solo vector
assembler = VectorAssembler(
    inputCols=["cantidad", "precio"],  # columnas de entrada
    outputCol="features"               # nueva columna vector
)

df_vector = assembler.transform(df)
# Ahora tenemos una columna llamada "features" tipo vector


# StandardScaler normaliza los datos
scaler = StandardScaler(
    inputCol="features",        # entrada
    outputCol="scaledFeatures", # salida escalada
    withMean=True,              # centra datos en 0
    withStd=True                # divide por desviación estándar
)

# Entrenamos el escalador
scaler_model = scaler.fit(df_vector)

# Aplicamos transformación
df_scaled = scaler_model.transform(df_vector)

# 6. CORRECCIÓN ARROW (CLAVE)

# Convertimos DenseVector → array
df_fixed = df_scaled.withColumn(
    "features_array",
    vector_to_array("scaledFeatures")
)

# Extraemos cada dimensión
df_fixed = df_fixed.withColumn("f1", col("features_array")[0])
df_fixed = df_fixed.withColumn("f2", col("features_array")[1])

st.subheader(" Datos escalados (compatibles)")
st.dataframe(df_fixed.select("f1", "f2", "label").toPandas())

# VISUALIZACIÓN

fig = px.scatter(
    df_fixed.toPandas(),
    x="f1",
    y="f2",
    color="label",
    title="Distribución de datos"
)

st.plotly_chart(fig, use_container_width=True)

# 6. CONVERSIÓN A PANDAS (PUENTE)

# PyTorch no entiende Spark a convertimos a Pandas
pdf = df_scaled.select("scaledFeatures", "label").toPandas()

# Convertimos vector Spark a lista Python
X = pdf["scaledFeatures"].apply(lambda x: x.toArray()).tolist()

# Variable objetivo
y = pdf["label"].values

# Convertimos a tensores (formato requerido por PyTorch)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

#¿QUÉ ES UN TENSOR? Un tensor es simplemente una estructura de datos numérica multidimensional.
# En palabras simples: Un tensor es como una matriz generalizada que puede tener más dimensiones.

# 0D (Escalar)
    #5 un solo numero
# 1D (Vector)
    #[1, 2, 3] lista de numeros
# 2D (Matriz)
    #[
    #[1, 2, 3],
    #[4, 5, 6]
    #]
    #tabla filas y columnas.
# 3D (Tensor)
    #[
    #[[1,2],[3,4]],
    #[[5,6],[7,8]]
    #]
    #como un cubo de datos.
#¿POR QUÉ SON TAN IMPORTANTES?

# La base del Deep Learning
# Lo que procesa la red neuronal
# Lo que viaja entre capas

# 7. DEFINICIÓN DE RED NEURONAL

class RedNeuronal(nn.Module):

    def __init__(self):
        super().__init__()

        # Arquitectura de la red
        self.net = nn.Sequential(

            nn.Linear(2, 16),  # Entrada (2 variables) → 16 neuronas
            nn.ReLU(),         # Función de activación

            nn.Linear(16, 8),  # Capa oculta
            nn.ReLU(),

            nn.Linear(8, 3)    # Salida → 3 clases
        )

    # Forward pass (flujo de datos)
    def forward(self, x):
        return self.net(x)


# Creamos modelo
model = RedNeuronal()


# 8. CONFIGURACIÓN DE ENTRENAMIENTO

criterion = nn.CrossEntropyLoss()  
# Función de error para clasificación

optimizer = optim.Adam(model.parameters(), lr=0.01)  
# Algoritmo que ajusta los pesos

epochs = st.sidebar.slider("Épocas", 10, 200, 100)


# 9. ENTRENAMIENTO

if st.button("Entrenar modelo"):

    loss_history = []  # Guardar evolución del error

    for epoch in range(epochs):

        outputs = model(X_tensor)      # Predicción
        loss = criterion(outputs, y_tensor)  # Error

        optimizer.zero_grad()  # Reiniciar gradientes
        loss.backward()        # Backpropagation
        optimizer.step()       # Actualizar pesos

        loss_history.append(loss.item())

    st.success("Modelo entrenado correctamente")


    # 10. VISUALIZAR APRENDIZAJE
    
    df_loss = pd.DataFrame({
        "Epoch": list(range(epochs)),
        "Loss": loss_history
    })

    fig = px.line(df_loss, x="Epoch", y="Loss", title="Curva de aprendizaje")
    st.plotly_chart(fig, use_container_width=True)


    # 11. PREDICCIÓN NUEVA
    
    st.subheader(" Predicción en tiempo real")

    cantidad = st.number_input("Cantidad", 1, 20, 5)
    precio = st.number_input("Precio", 1, 100, 30)

    # IMPORTANTE: MISMO FLUJO QUE ENTRENAMIENTO

    nuevo_df = spark.createDataFrame([(cantidad, precio)], ["cantidad", "precio"])

    nuevo_vector = assembler.transform(nuevo_df)
    nuevo_scaled = scaler_model.transform(nuevo_vector)

    nuevo_pdf = nuevo_scaled.select("scaledFeatures").toPandas()

    nuevo_X = torch.tensor(
        nuevo_pdf["scaledFeatures"].apply(lambda x: x.toArray()).tolist(),
        dtype=torch.float32
    )

    # Predicción
    with torch.no_grad():
        pred = model(nuevo_X)
        prob = torch.softmax(pred, dim=1)
        clase = torch.argmax(prob).item()

    st.write(f"Clase predicha: {clase}")
    st.write(f"Probabilidades: {prob.numpy()}")


    # 12. IA EXPLICATIVA
    
    st.subheader(" Interpretación automática")

    if clase == 0:
        texto = "Cliente de bajo valor: precios bajos y compras pequeñas."
    elif clase == 1:
        texto = "Cliente medio: comportamiento equilibrado."
    else:
        texto = "Cliente de alto valor: compras grandes y precios altos."

    st.success(texto)


# 13. RESUMEN FINAL

st.markdown("""
---
###  RESUMEN DEL SISTEMA

 Spark: procesamiento de datos  
 Feature Engineering: preparación  
 PyTorch: aprendizaje  
 Streamlit: visualización  
 IA: interpretación  

Esto es un pipeline real de inteligencia artificial
""")


#1. Spark Se usa para procesar datos grandes
#2. VectorAssembler Convierte columnas en un vector (formato ML)
#3. StandardScaler Normaliza datos → evita sesgos
#4. Pandas Puente entre Spark y PyTorch
#5. Tensor Estructura base del deep learning
#6. Red neuronal Aprende patrones complejos
#7. Entrenamiento Ajusta pesos para minimizar error
#8. Evaluación Mide qué tan bueno es el modelo
#9. Predicción Aplica el modelo a nuevos datos