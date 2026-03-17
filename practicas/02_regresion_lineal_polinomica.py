# 02_regresion_polinomica.py
import streamlit as st
import os, pandas as pd, numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
df = pd.DataFrame(list(
    MongoClient(uri)[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")].find({}, {"_id":0})
))

st.title("Regresión Polinómica: Peso vs Precio")

X = df[["peso"]]
y = df["precio_mensual"]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression().fit(X_poly, y)

X_sorted = np.sort(X.values, axis=0)
X_sorted_poly = poly.transform(X_sorted)
y_pred = model.predict(X_sorted_poly)

fig, ax = plt.subplots()
ax.scatter(X, y, alpha=0.5, label="Datos reales")
ax.plot(X_sorted, y_pred, color="red", label="Modelo polinómico")
ax.set_xlabel("Peso")
ax.set_ylabel("Precio mensual")
ax.legend()

st.pyplot(fig)
