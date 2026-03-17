import os
import random
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
client = MongoClient(uri)

db = client[os.getenv("DB_NAME")]
col = db[os.getenv("COLLECTION_NAME")]

col.drop()

tipos = ["Basica", "Premium", "VIP"]

clientes = []

for _ in range(500):
    clientes.append({
        "edad": random.randint(18, 65),
        "peso": round(random.uniform(55, 120), 1),
        "altura": round(random.uniform(1.50, 1.95), 2),
        "meses_inscrito": random.randint(1, 48),
        "precio_mensual": random.choice([350, 450, 600]),
        "asiste_regularmente": random.choice([True, False]),
        "tipo_membresia": random.choice(tipos)
    })

col.insert_many(clientes)
print("Base gimnasio_db creada con 500 clientes")
