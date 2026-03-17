import streamlit as st
import os
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()

# Conexión a MongoDB
uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
client = MongoClient(uri)
df = pd.DataFrame(
    list(
        client[os.getenv("DB_NAME")]
        [os.getenv("COLLECTION_NAME")]
        .find({}, {"_id": 0})
    )
)

st.title("Validación Cruzada")

# Variables
X = df[["peso"]]
y = df["precio_mensual"]

# Validación cruzada
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5)

promedio = scores.mean()

st.write("Resultados por fold:", scores)
st.write("Promedio de desempeño del modelo:", promedio)

# Gráfica
fig, ax = plt.subplots()
ax.bar(range(1, len(scores) + 1), scores, color="skyblue", label="Fold")
ax.axhline(promedio, color="red", linestyle="--", label="Promedio")
ax.set_xlabel("Fold")
ax.set_ylabel("R²")
ax.set_title("Desempeño del modelo en Validación Cruzada")
ax.legend()

st.pyplot(fig)
