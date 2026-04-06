import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from common.mongo_spark import get_spark_df

from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.sql.functions import col

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split


# ================== UI ==================
st.title("Deep Learning: Spark + PyTorch (Gimnasio)")
st.write(
    "Objetivo: usar Spark para preparar y escalar variables, y PyTorch para entrenar "
    "una red neuronal que predice el **tipo de membresía** del cliente."
)

# ================== 1) Cargar datos desde Mongo con Spark ==================
df = get_spark_df("Spark_PyTorch_Gym")

# Limpieza mínima
df = df.fillna({
    "peso": 0,
    "altura": 0,
    "edad": 0,
    "meses_inscrito": 0,
    "precio_mensual": 0,
    "tipo_membresia": "Basica"
})

# ================== 2) Label numérico (tipo_membresia -> label) ==================
indexer = StringIndexer(inputCol="tipo_membresia", outputCol="label", handleInvalid="keep")
indexer_model = indexer.fit(df)
df_labeled = indexer_model.transform(df_labeled := df)

labels = indexer_model.labels  # orden de clases
num_classes = len(labels)

st.subheader("Clases detectadas")
st.write({i: labels[i] for i in range(num_classes)})

# ================== 3) Features + escalado ==================
feature_cols = ["peso", "altura", "edad", "meses_inscrito", "precio_mensual"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)
df_vector = assembler.transform(df_labeled)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True
)
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

# ================== 4) Spark -> Pandas -> PyTorch ==================
pdf = df_scaled.select("scaledFeatures", "label").toPandas()

X = pdf["scaledFeatures"].apply(lambda v: v.toArray()).tolist()
y = pdf["label"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

input_dim = len(feature_cols)

# ================== 5) Red neuronal ==================
class RedNeuronal(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim)
        )

    def forward(self, x):
        return self.net(x)

model = RedNeuronal(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ================== 6) Entrenamiento ==================
st.subheader("Entrenamiento")

epochs = st.slider("Épocas", min_value=20, max_value=300, value=120, step=10)

loss_history = []
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

fig, ax = plt.subplots()
ax.plot(range(1, epochs + 1), loss_history, color="purple")
ax.set_xlabel("Época")
ax.set_ylabel("Loss")
ax.set_title("Pérdida durante el entrenamiento")
st.pyplot(fig)

# ================== 7) Evaluación ==================
st.subheader("Evaluación")

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test) if len(y_test) else 0.0

st.write(f"Accuracy en test: {accuracy * 100:.2f}%")

# ================== 8) Predicción interactiva ==================
st.subheader("Predicción de un nuevo cliente")

c1, c2 = st.columns(2)

with c1:
    peso = st.slider("Peso", 40.0, 140.0, 80.0, 0.1)
    altura = st.slider("Altura", 1.40, 2.10, 1.70, 0.01)
    edad = st.slider("Edad", 16, 80, 28, 1)

with c2:
    meses = st.slider("Meses inscrito", 0, 60, 10, 1)
    precio = st.slider("Precio mensual", 200, 1000, 450, 10)

nuevo = [[peso, altura, edad, meses, precio]]
nuevo_df = df_scaled.sql_ctx.createDataFrame(nuevo, feature_cols)

nuevo_vector = assembler.transform(nuevo_df)
nuevo_scaled = scaler_model.transform(nuevo_vector)

nuevo_pdf = nuevo_scaled.select("scaledFeatures").toPandas()
nuevo_X = torch.tensor(
    nuevo_pdf["scaledFeatures"].apply(lambda v: v.toArray()).tolist(),
    dtype=torch.float32
)

with torch.no_grad():
    logits = model(nuevo_X)
    prob = torch.softmax(logits, dim=1).numpy()[0]
    clase = int(prob.argmax())

st.write(f"Clase predicha: **{labels[clase]}**")

prob_df = pd.DataFrame({
    "Membresía": labels,
    "Probabilidad": prob
}).sort_values("Probabilidad", ascending=False)

st.dataframe(prob_df, use_container_width=True)

fig2, ax2 = plt.subplots()
ax2.bar(prob_df["Membresía"], prob_df["Probabilidad"], color="teal")
ax2.set_ylim(0, 1)
ax2.set_xlabel("Tipo de membresía")
ax2.set_ylabel("Probabilidad")
ax2.set_title("Probabilidades por clase")
st.pyplot(fig2)
