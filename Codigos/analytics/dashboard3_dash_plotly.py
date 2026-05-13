# Dashboard de Ventas con Spark + MongoDB + Dash + Plotly

import dash
from dash import dcc, html
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, avg, col
from dotenv import load_dotenv
from pathlib import Path
import os
from urllib.parse import quote_plus

# 1. Cargar variables de entorno

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

user = os.getenv("MONGO_USER")
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
database = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")

mongo_uri = f"mongodb+srv://{user}:{password}@{cluster}"

# 2. Configurar Spark con conector MongoDB

spark = SparkSession.builder \
    .appName("BigDataVentasDashboard") \
    .config("spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.13:10.3.0") \
    .config("spark.mongodb.read.connection.uri", mongo_uri) \
    .config("spark.mongodb.read.database", database) \
    .config("spark.mongodb.read.collection", collection_name) \
    .getOrCreate()

# 3. Leer datos desde MongoDB

df = spark.read.format("mongodb").load()

# 4. Agregación en Spark

resultado = df.groupBy("producto") \
    .agg(
        sum("cantidad").alias("total_cantidad"),
        avg("precio").alias("promedio_precio"),
        sum(col("cantidad") * col("precio")).alias("ingresos_totales")
    ) \
    .orderBy("ingresos_totales", ascending=False)

# 5. Convertir a Pandas

pdf = resultado.toPandas()

# 6. Crear gráficos con Plotly

fig_total = px.bar(
    pdf, x="producto", y="total_cantidad",
    title="Total de ventas por producto",
    color="producto", text_auto=True
)
fig_total.update_traces(textposition="outside")

fig_precio = px.bar(
    pdf, x="producto", y="promedio_precio",
    title="Promedio de precio por producto",
    color="producto", text_auto=True
)
fig_precio.update_traces(textposition="outside")

fig_ingresos = px.bar(
    pdf, x="producto", y="ingresos_totales",
    title="Ingresos totales por producto",
    color="producto", text_auto=True
)
fig_ingresos.update_traces(textposition="outside")

# 7. Crear la app Dash

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(" Dashboard de Ventas - Big Data"),
    html.H3("Datos agregados desde MongoDB con Spark"),
    dcc.Graph(figure=fig_total),
    dcc.Graph(figure=fig_precio),
    dcc.Graph(figure=fig_ingresos)
])

# 8. Ejecutar servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8051, debug=True, use_reloader=False)

#if __name__ == "__main__":
#   app.run(host="0.0.0.0", port=8050, debug=False)

#la direccion de host http://127.0.0.1:8051

#si el puerto falla se debe liberar uno o cambiar el puerto

#- El modo debug es útil para desarrollo porque recarga automáticamente el servidor al cambiar el código.
#- Sin embargo, en algunos entornos (WSL, notebooks, procesos secundarios) el reloader no funciona porque 
#- depende de señales del sistema que solo están disponibles en el hilo principal.
#- En producción siempre se recomienda debug=False.
