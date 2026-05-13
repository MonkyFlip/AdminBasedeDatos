# ============================================================
# SISTEMA DEEP LEARNING: SPARK + PYTORCH
# ============================================================
# OBJETIVO:
# ✔ Procesar datos con Spark
# ✔ Preparar features
# ✔ Entrenar red neuronal con PyTorch
# ✔ Evaluar modelo
# ✔ Hacer predicciones nuevas
# ============================================================


# 1. IMPORTACIONES
# Permite crear una sesión de Spark (motor distribuido)
from pyspark.sql import SparkSession  

# Convierte columnas en un vector (requisito para ML en Spark)
from pyspark.ml.feature import VectorAssembler  

# Escala los datos (media=0, desviación=1)
from pyspark.ml.feature import StandardScaler  

# Librería principal de Deep Learning
import torch  

# Módulo base para crear redes neuronales
import torch.nn as nn  

# Optimizadores (algoritmos para entrenar el modelo)
import torch.optim as optim  

# Manejo de datos tipo DataFrame
import pandas as pd  

# Dividir datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split  


# 2. CREAR SESIÓN SPARK

# Se crea o reutiliza una sesión de Spark
spark = SparkSession.builder \
    .appName("Spark con PyTorch para redes neurales") \
    .getOrCreate()


# 3. CREAR DATASET (SIMULADO)

# Datos: (cantidad, precio, clase)
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

# Nombres de columnas
columns = ["cantidad", "precio", "label"]

# Crear DataFrame en Spark
df = spark.createDataFrame(data, columns)


# 4. CREAR FEATURES (VECTOR)

# Une columnas numéricas en un vector
assembler = VectorAssembler(
    inputCols=["cantidad", "precio"],  # variables de entrada
    outputCol="features"               # columna vectorial
)

# Aplica transformación
df_vector = assembler.transform(df)


# 5. ESCALAR DATOS

# Escalador: normaliza datos
scaler = StandardScaler(
    inputCol="features",        # columna original
    outputCol="scaledFeatures", # columna escalada
    withMean=True,              # centra en 0
    withStd=True                # divide por desviación
)

# Entrena el escalador
scaler_model = scaler.fit(df_vector)

# Aplica escalado
df_scaled = scaler_model.transform(df_vector)


# 6. CONVERTIR A PANDAS (PUENTE A PYTORCH)

# PyTorch no trabaja con Spark lo convertimos
pdf = df_scaled.select("scaledFeatures", "label").toPandas()

# Convertimos vector Spark a lista
X = pdf["scaledFeatures"].apply(lambda x: x.toArray()).tolist()

# Variable objetivo
y = pdf["label"].values


# 7. TRAIN / TEST

# Dividir datos (80% entrenamiento / 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Un tensor es simplemente una estructura de datos numérica multidimensional.
#En palabras simples: Un tensor es como una matriz generalizada que puede tener más dimensiones.

#ejemplos
#5

#1d vector [1, 2, 3]

#2d matriz
#[
# [1, 2, 3],
# [4, 5, 6]
#]

#3d como cubo de datos
#[
#  [[1,2],[3,4]],
#  [[5,6],[7,8]]
#]

#¿POR QUÉ SON TAN IMPORTANTES?

#Los tensores son:

# La base del Deep Learning
# Lo que procesa la red neuronal
# Lo que viaja entre capas

#Los tensores pueden correr en:

#CPU
#GPU (aceleración)

# Convertir a tensores (estructura base de PyTorch)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# 8. DEFINIR RED NEURONAL

# Clase de red neuronal
class RedNeuronal(nn.Module):

    # Constructor
    def __init__(self):
        super().__init__()

        # Arquitectura del modelo
        self.net = nn.Sequential(
            nn.Linear(2, 16),  # Entrada (2 variables) → 16 neuronas
            nn.ReLU(),         # Activación

            nn.Linear(16, 8),  # Capa oculta
            nn.ReLU(),

            nn.Linear(8, 3)    # Salida → 3 clases
        )

    # Forward pass (cómo fluye la información)
    def forward(self, x):
        return self.net(x)


# Crear modelo
model = RedNeuronal()


# 9. CONFIGURAR ENTRENAMIENTO

# Función de pérdida (clasificación)
criterion = nn.CrossEntropyLoss()

# Optimizador (ajusta pesos)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 10. ENTRENAMIENTO

epochs = 100

for epoch in range(epochs):

    # Predicción
    outputs = model(X_train)

    # Error
    loss = criterion(outputs, y_train)

    # Resetear gradientes
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Actualizar pesos
    optimizer.step()

    # Mostrar progreso
    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")


# 11. EVALUACIÓN

# Desactivar gradientes
with torch.no_grad():

    outputs = model(X_test)

    # Obtener clase predicha
    _, predicted = torch.max(outputs, 1)

    # Precisión
    accuracy = (predicted == y_test).sum().item() / len(y_test)

print(f"Precisión: {accuracy * 100:.2f}%")


# 12. PREDICCIÓN NUEVA 

# dato nuevo
nuevo = [[5, 30]]

print("Dato original:", nuevo)

# Convertir a DataFrame Spark
nuevo_df = spark.createDataFrame(nuevo, ["cantidad", "precio"])

# Aplicar mismas transformaciones
nuevo_vector = assembler.transform(nuevo_df)
nuevo_scaled = scaler_model.transform(nuevo_vector)

# Convertir a Pandas
nuevo_pdf = nuevo_scaled.select("scaledFeatures").toPandas()

# Convertir a tensor
nuevo_X = torch.tensor(
    nuevo_pdf["scaledFeatures"].apply(lambda x: x.toArray()).tolist(),
    dtype=torch.float32
)

print("Dato escalado:", nuevo_pdf)


# 13. PREDICCIÓN + PROBABILIDADES

with torch.no_grad():

    pred = model(nuevo_X)

    # Convertir a probabilidades
    prob = torch.softmax(pred, dim=1)

    # Clase final
    clase = torch.argmax(prob).item()

print(f"Clase predicha: {clase}")
print(f"Probabilidades: {prob.numpy()}")


# 14. INTERPRETACIÓN (IA SIMPLE)

labels_map = {
    0: "Ingreso bajo",
    1: "Ingreso medio",
    2: "Ingreso alto"
}

print(f"Interpretación: {labels_map[clase]}")


# 15. FINALIZAR

# Cerrar Spark
spark.stop()

