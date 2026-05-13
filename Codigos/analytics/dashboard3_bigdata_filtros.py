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

st.title("Dashboard Avanzado Big Data Ventas")

data = list(coleccion.find())
df = pd.DataFrame(data)

if not df.empty:

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ingreso"] = df["cantidad"] * df["precio"]

    # ---------------- FILTROS ----------------
    st.sidebar.header("Filtros")

    productos = df["producto"].unique()
    producto_seleccionado = st.sidebar.multiselect(
        "Selecciona producto(s)",
        productos,
        default=productos
    )

    precio_min, precio_max = st.sidebar.slider(
        "Rango de Precio",
        int(df["precio"].min()),
        int(df["precio"].max()),
        (int(df["precio"].min()), int(df["precio"].max()))
    )

    ingreso_min, ingreso_max = st.sidebar.slider(
        "Rango de Ingreso",
        int(df["ingreso"].min()),
        int(df["ingreso"].max()),
        (int(df["ingreso"].min()), int(df["ingreso"].max()))
    )

    fecha_inicio, fecha_fin = st.sidebar.date_input(
        "Rango de Fechas",
        [df["fecha"].min(), df["fecha"].max()]
    )

    # Aplicar filtros
    df_filtrado = df[
        (df["producto"].isin(producto_seleccionado)) &
        (df["precio"].between(precio_min, precio_max)) &
        (df["ingreso"].between(ingreso_min, ingreso_max)) &
        (df["fecha"].dt.date.between(fecha_inicio, fecha_fin))
    ]

    # ---------------- MÉTRICAS ----------------
    st.subheader("Métricas")
    st.metric("Total registros", len(df_filtrado))
    st.metric("Ingreso total", f"${df_filtrado['ingreso'].sum():,.2f}")

    # ---------------- AGRUPACIÓN ----------------
    resumen = df_filtrado.groupby("producto").agg(
        total_vendido=("cantidad", "sum"),
        ingreso_total=("ingreso", "sum"),
        precio_promedio=("precio", "mean")
    ).reset_index()

    st.subheader("Resumen por producto")
    st.dataframe(resumen)

    # ---------------- GRÁFICO BARRAS ----------------
    fig1, ax1 = plt.subplots()
    ax1.bar(resumen["producto"], resumen["ingreso_total"])
    ax1.set_title("Ingresos por Producto")
    st.pyplot(fig1)

    # ---------------- GRÁFICO TEMPORAL ----------------
    ventas_por_dia = df_filtrado.groupby(
        df_filtrado["fecha"].dt.date
    )["ingreso"].sum().reset_index()

    fig2, ax2 = plt.subplots()
    ax2.plot(ventas_por_dia["fecha"], ventas_por_dia["ingreso"])
    ax2.set_title("Ingresos por Día")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Ingreso")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

else:
    st.warning("No hay datos disponibles.")