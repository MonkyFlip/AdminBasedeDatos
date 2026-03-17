import os
import random
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONEXIÓN A BASE DE PRÁCTICAS
# ──────────────────────────────────────────────
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
cluster = os.getenv("DB_CLUSTER")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")

uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"
client = MongoClient(uri)

db = client[db_name]
collection = db[collection_name]

print("📚 Creando base biblioteca_db y colección biblioteca...")

# Limpieza segura
collection.drop()

categorias = [
    "Ciencia",
    "Historia",
    "Tecnología",
    "Literatura",
    "Matemáticas",
    "Filosofía",
    "Arte"
]

libros = []

for i in range(1000):
    libros.append({
        "titulo": f"Libro {i}",
        "autor": f"Autor {random.randint(1, 200)}",
        "anio": random.randint(1990, 2024),
        "paginas": random.randint(80, 900),
        "precio": round(random.uniform(150, 1200), 2),
        "categoria": random.choice(categorias),
        "disponible": random.choice([True, False])
    })

collection.insert_many(libros)

print("✅ Base biblioteca_db creada")
print("✅ Colección biblioteca poblada con 1000 registros")

client.close()
