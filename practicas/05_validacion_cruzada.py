# 05_validacion_cruzada.py
import streamlit as st
import os, pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

load_dotenv()

uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
df = pd.DataFrame(list(
    MongoClient(uri)[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")].find({}, {"_id":0})
))

st.title("Validación Cruzada")

scores = cross_val_score(
    LinearRegression(),
    df[["peso"]],
    df["precio_mensual"],
    cv=5
)

st.write("Promedio de desempeño del modelo:", scores.mean())
