import random
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import os
from urllib.parse import quote_plus

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

productos = ["Laptop", "Tablet", "Celular", "Monitor"]

# Generar fechas en los últimos 90 días
for _ in range(5000):

    fecha_random = datetime.now() - timedelta(days=random.randint(0, 90))

    venta = {
        "producto": random.choice(productos),
        "cantidad": random.randint(1,5),
        "precio": random.randint(5000,20000),
        "fecha": fecha_random
    }

    coleccion.insert_one(venta)

print("Datos generados con fechas correctamente")