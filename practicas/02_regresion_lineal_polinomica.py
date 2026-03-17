# 02_regresion_polinomica.py
import streamlit as st
import os, pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

load_dotenv()

uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
df = pd.DataFrame(list(MongoClient(uri)[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")].find({}, {"_id":0})))

st.title("Regresión Polinómica")

X = df[["peso"]]
y = df["precio_mensual"]

X_poly = PolynomialFeatures(2).fit_transform(X)
model = LinearRegression().fit(X_poly, y)

st.write("Coeficientes:", model.coef_)
