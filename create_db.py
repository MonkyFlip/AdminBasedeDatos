import os
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONEXIÓN
# ──────────────────────────────────────────────
user = os.getenv("MONGO_USER")
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")
db_name = os.getenv("MONGO_DB")

uri = f"mongodb+srv://{user}:{password}@{cluster}/{db_name}?retryWrites=true&w=majority"
client = MongoClient(uri)
db = client[db_name]

print("🧹 Limpiando base de datos...")
for col in db.list_collection_names():
    db[col].drop()
    print(f"   ❌ {col}")

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def create(name, validator=None):
    opts = {}
    if validator:
        opts["validator"] = validator
        opts["validationAction"] = "warn"
    db.create_collection(name, **opts)
    print(f"   ✅ {name}")

# ──────────────────────────────────────────────
# VALIDADORES BÁSICOS
# ──────────────────────────────────────────────
V_SIMPLE = {"$jsonSchema": {"bsonType": "object"}}

# ──────────────────────────────────────────────
# COLECCIONES
# ──────────────────────────────────────────────
collections = [
    "roles","usuarios","miembros","membresias","productos","ventas",
    "correos_enviados","asistencias","pagos","progreso_fisico","rutinas",
    "rutina_dias","rutina_ejercicios","miembro_membresia","correo_miembro",
    "detalle_venta","sesiones","tipos_dieta","recetas","planes_alimenticios",
    "plan_recetas","perfil_entrenador","certificaciones_entrenador",
    "logros_entrenador","evaluaciones_entrenador","miembro_rutina"
]

print("\n📦 Creando colecciones...")
for c in collections:
    create(c, V_SIMPLE)

# ──────────────────────────────────────────────
# ÍNDICES IMPORTANTES
# ──────────────────────────────────────────────
print("\n🔍 Creando índices...")
db.usuarios.create_index("email", unique=True)
db.ventas.create_index("fecha")
db.pagos.create_index("metodo_pago")
db.asistencias.create_index("fecha")
db.progreso_fisico.create_index("bmi")
db.sesiones.create_index("estado")
db.miembro_rutina.create_index(
    [("id_miembro", ASCENDING), ("id_rutina", ASCENDING)],
    unique=True
)

# ──────────────────────────────────────────────
# CATÁLOGOS
# ──────────────────────────────────────────────
print("\n📋 Insertando catálogos...")

db.roles.insert_many([
    {"nombre":"Administrador"},
    {"nombre":"Entrenador"},
    {"nombre":"Recepcionista"},
    {"nombre":"Miembro"}
])

db.membresias.insert_many([
    {"nombre":"Básica","duracion_meses":1,"precio":80},
    {"nombre":"Premium","duracion_meses":1,"precio":100},
    {"nombre":"Anual","duracion_meses":12,"precio":500},
    {"nombre":"VIP","duracion_meses":1,"precio":120}
])

db.tipos_dieta.insert_many([
    {"nombre":"Hipercalórica"},
    {"nombre":"Déficit"},
    {"nombre":"Vegana"},
    {"nombre":"Keto"},
    {"nombre":"Mediterránea"},
    {"nombre":"Paleo"},
    {"nombre":"Flexible"}
])

print("\n🎉 Base creada correctamente")
client.close()
