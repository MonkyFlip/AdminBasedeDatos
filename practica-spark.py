from pymongo import MongoClient
from datetime import datetime
import random

# Cadena de conexión proporcionada
uri = "mongodb+srv://hector1985:Aime131985@utvt.qqqotrr.mongodb.net/?appName=UTVT"

# Conexión al cluster
client = MongoClient(uri)

# Base de datos y colección
db = client["ventas_db"]              # La base sigue siendo ventas_db
coleccion = db["alumnos_cursos"]      # Nueva colección

# -----------------------------
# Generar 2000 documentos
# -----------------------------

nombres = ["Ana", "Miguel", "Luis", "Carla", "Jorge", "Sofía", "Daniel", "María", "Héctor", "Aime"]
apellidos = ["García", "López", "Martínez", "Hernández", "Santos", "Ramírez", "Flores", "Torres"]
cursos = ["IA", "Big Data", "Ciberseguridad", "Desarrollo Web", "Redes", "Videojuegos"]
niveles = ["Básico", "Intermedio", "Avanzado"]

documentos = []

for i in range(2000):
    nombre = random.choice(nombres)
    apellido = random.choice(apellidos)

    doc = {
        "alumno_id": i + 1,
        "nombre": nombre,
        "apellido": apellido,
        "email": f"{nombre.lower()}.{apellido.lower()}{i}@example.com",
        "curso": random.choice(cursos),
        "nivel": random.choice(niveles),
        "fecha_registro": datetime.utcnow(),
        "calificacion": round(random.uniform(6.0, 10.0), 1),
        "activo": random.choice([True, False])
    }

    documentos.append(doc)

# Insertar en MongoDB
resultado = coleccion.insert_many(documentos)

print(f"Se insertaron {len(resultado.inserted_ids)} documentos en la colección 'alumnos_cursos'.")

# Mostrar algunos documentos
print("\nEjemplo de documentos insertados:")
for doc in coleccion.find().limit(5):
    print(doc)
