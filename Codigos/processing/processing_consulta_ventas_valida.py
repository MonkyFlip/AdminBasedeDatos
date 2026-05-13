from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote_plus

# 1.- Cargar variables de entorno

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

user = os.getenv("MONGO_USER")
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
database = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")

mongo_uri = f"mongodb+srv://{user}:{password}@{cluster}"

# 2.- Conexión a MongoDB

client = MongoClient(mongo_uri)
db = client[database]
coleccion = db[collection_name]

print(" Conectado correctamente a MongoDB\n")

# 3.- Pipeline de agregación
# Un pipeline es un concepto que describe un proceso en el que las tareas se ejecutan de manera secuencial y organizada, 
# como si fueran pasos en una línea de ensamblaje. 
# La idea es que cada etapa recibe una entrada, la procesa y pasa el resultado a la siguiente etapa, creando un flujo continuo de trabajo.
# Usos principales de un pipeline
# - En programación y DevOps: se refiere a la automatización de pasos como compilación, pruebas y despliegue de software.
        # DevOps es una metodología de desarrollo de software que busca integrar y automatizar el trabajo de los equipos de desarrollo (Dev) 
        # y de operaciones de TI (Ops). 
        # Su objetivo es acelerar la entrega de aplicaciones y servicios de calidad, 
        # fomentando la colaboración y eliminando los silos tradicionales entre áreas que antes trabajaban separadas.
# - En ciencia de datos: organiza fases como limpieza de datos, entrenamiento de modelos y evaluación.
# - En negocios y ventas: describe el recorrido de un cliente potencial desde el primer contacto hasta la compra.
# - En hardware (procesadores): se usa para dividir la ejecución de instrucciones en etapas, mejorando la eficiencia.
# Beneficios
# - Mayor eficiencia: las tareas fluyen sin interrupciones.
# - Automatización: reduce errores humanos.
# - Escalabilidad: permite manejar procesos más complejos.
# - Visibilidad: facilita entender en qué etapa está cada tarea o proyecto.
# En pocas palabras, un pipeline es una forma de estructurar y optimizar procesos para que sean más claros, rápidos y confiables.


pipeline = [

    # Paso 1: Limpiar datos
    {
        "$match": {
            "producto": {"$ne": None},
            "precio": {"$ne": None},
            "cantidad": {"$ne": None}
        }
    },

    # Paso 2: Agrupar y calcular métricas
    {
        "$group": {
            "_id": "$producto",

            "total_cantidad": {
                "$sum": "$cantidad"
            },

            "promedio_precio": {
                "$avg": "$precio"
            },

            "ventas_registradas": {
                "$sum": 1
            },

            "total_ingresos": {
                "$sum": {
                    "$multiply": ["$precio", "$cantidad"]   # $multiply multiplica campos del documento actual.
                                                            # $sum acumula el resultado de esa multiplicación.
                }
            }
        }
    },

    # Paso 3: Ordenar por cantidad vendida
    {
        "$sort": {
            "total_cantidad": -1
        }
    }
]

# 4.- Ejecutar pipeline

resultados = list(coleccion.aggregate(pipeline))

# 5.- Mostrar resultados

for r in resultados:

    promedio = r.get("promedio_precio", 0) or 0
    total_ingresos = r.get("total_ingresos", 0) or 0

    print(f"Producto: {r['_id']}")
    print(f"  Total vendido: {r['total_cantidad']}")
    print(f"  Promedio precio: {promedio:.2f}")
    print(f"  Ventas registradas: {r['ventas_registradas']}")
    print(f"  Total de ingresos: {total_ingresos:.2f}")
    print("-" * 40)

print("\n Proceso finalizado correctamente.")