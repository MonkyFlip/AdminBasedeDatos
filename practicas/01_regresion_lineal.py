# 01_regresion_lineal.py
import streamlit as st
import os, pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
df = pd.DataFrame(list(MongoClient(uri)[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")].find({}, {"_id":0})))

st.title("Regresión Lineal: Peso vs Precio")

X = df[["peso"]]
y = df["precio_mensual"]

model = LinearRegression().fit(X, y)

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, model.predict(X))
st.pyplot(fig)
