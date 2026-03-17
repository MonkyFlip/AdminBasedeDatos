import streamlit as st
import os
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import PolynomialFeatures
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

st.title("Regresión Polinómica: Peso vs Precio Mensual")

# Variables
X = df[["peso"]]
y = df["precio_mensual"]

# Transformación polinómica
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Modelo
model = LinearRegression()
model.fit(X_poly, y)

# Ordenar para graficar correctamente
X_sorted = np.sort(X.values, axis=0)
X_sorted_poly = poly.transform(X_sorted)
y_pred = model.predict(X_sorted_poly)

# Gráfica
fig, ax = plt.subplots()
ax.scatter(X, y, color="blue", alpha=0.5, label="Datos reales")
ax.plot(X_sorted, y_pred, color="red", label="Modelo polinómico")
ax.set_xlabel("Peso")
ax.set_ylabel("Precio mensual")
ax.legend()

st.pyplot(fig)
