import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import os
from urllib.parse import quote_plus
import pandas as pd
import matplotlib.pyplot as plt

# Cargar .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

user = os.getenv("MONGO_USER")
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
database = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")

mongo_uri = f"mongodb+srv://{user}:{password}@{cluster}"

client = MongoClient(mongo_uri)
db = client[database]
coleccion = db[collection_name]

st.title("Dashboard Big Data Ventas")

# Leer datos
data = list(coleccion.find())
df = pd.DataFrame(data)

if not df.empty:

    df["ingreso"] = df["cantidad"] * df["precio"]

    st.subheader("Vista previa de datos")
    st.dataframe(df.head())

    # Métricas generales
    st.subheader("Métricas Generales")
    st.metric("Total registros", len(df))
    st.metric("Ingreso total", f"${df['ingreso'].sum():,.2f}")

    # Agrupación
    resumen = df.groupby("producto").agg(
        total_vendido=("cantidad", "sum"),
        ingreso_total=("ingreso", "sum"),
        precio_promedio=("precio", "mean")
    ).reset_index()

    st.subheader("Resumen por producto")
    st.dataframe(resumen)

    # Gráfico
    fig, ax = plt.subplots()
    ax.bar(resumen["producto"], resumen["ingreso_total"])
    ax.set_title("Ingresos por Producto")
    ax.set_xlabel("Producto")
    ax.set_ylabel("Ingreso Total")
    st.pyplot(fig)

else:
    st.warning("No hay datos en la colección.")