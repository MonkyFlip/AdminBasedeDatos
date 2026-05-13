from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote_plus

# Cargar .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Variables de entorno
user = os.getenv("MONGO_USER")
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
database = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")

mongo_uri = f"mongodb+srv://{user}:{password}@{cluster}"
client = MongoClient(mongo_uri)
db = client[database]
coleccion = db[collection_name]

# Pipeline de agregación
pipeline = [
    {
        "$group": {
            "_id": "$producto",
            "total_cantidad": {"$sum": "$cantidad"},
            "promedio_precio": {"$avg": "$precio"},
            "ventas_registradas": {"$sum": 1},
        }
    },
    {
        "$sort": {"total_cantidad": -1}
    }
]

# ¿Por qué? 
# $sum: 1 cuenta documentos.
# $avg calcula promedio.

# equivalente en SQL
# SELECT producto,
#       SUM(cantidad) AS total_cantidad,
#       AVG(precio) AS promedio_precio,
#       COUNT(*) AS ventas_registradas,
#       SUM(precio * cantidad) AS total_ingresos
# FROM ventas
# GROUP BY producto
# ORDER BY total_cantidad DESC;

resultados = list(coleccion.aggregate(pipeline))

# Mostrar resultados
for r in resultados:
    print(f"Producto: {r['_id']}")
    print(f"  Total vendido: {r['total_cantidad']}")
    print(f"  Promedio precio: {r['promedio_precio']:.2f}")
    print(f"  Ventas registradas: {r['ventas_registradas']}")
    print("-" * 40)

## El problema ahora es de calidad de datos.
## El error:
## TypeError: unsupported format string passed to NoneType.__format__
## significa que estás intentando hacer:
## {r['promedio_precio']:.2f}
## pero r['promedio_precio'] es None, y None no puede formatearse como número decimal.
## Y vemos claramente por qué:
## Producto: None
##   Total vendido: 0
## Hay registros donde el producto es NULL en tu dataset hay que filtrar en producto, precio, cantidad.

## Solucion Agrega un $match antes del $group para limpiar los datos.

## Porque en ingeniería de datos:
## Primero se limpia
## Luego se transforma
## Después se agrega
## Nunca debes confiar en que la base esté limpia.


