import random
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import os
from urllib.parse import quote_plus

# Cargar .env desde la raíz del proyecto
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Leer variables
user = os.getenv("MONGO_USER")
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
database = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")

if not all([user, password, cluster, database, collection_name]):
    raise ValueError("Faltan variables en el archivo .env")

# Construir URI segura
mongo_uri = f"mongodb+srv://{user}:{password}@{cluster}"

client = MongoClient(mongo_uri)
db = client[database]
coleccion = db[collection_name]

productos = ["Laptop", "Tablet", "Celular", "Monitor"]

for _ in range(1000):
    venta = {
        "producto": random.choice(productos),
        "cantidad": random.randint(1,5),
        "precio": random.randint(5000,20000)
    }
    coleccion.insert_one(venta)

print("Datos generados correctamente")