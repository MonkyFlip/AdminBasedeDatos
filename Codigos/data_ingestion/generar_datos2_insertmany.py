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

# Conectar al cluster
client = MongoClient(mongo_uri)

# Validar conexión
try:
    client.admin.command("ping")
    print(" Conexión exitosa a MongoDB")
except Exception as e:
    raise ConnectionError(f"Error al conectar con MongoDB: {e}")

# Seleccionar base de datos y colección
db = client[database]
coleccion = db[collection_name]

# Lista de productos
productos = ["Laptop", "Tablet", "Celular", "Monitor"]

# Generar datos de prueba
ventas = []
for _ in range(2000):
    venta = {
        "producto": random.choice(productos),
        "cantidad": random.randint(1, 5),
        "precio": random.randint(5000, 20000)
    }
    ventas.append(venta)

# Inserción masiva
resultado = coleccion.insert_many(ventas)

# Crear índices recomendados
coleccion.create_index("producto")
coleccion.create_index("precio")

print(f" {len(resultado.inserted_ids)} documentos insertados correctamente")