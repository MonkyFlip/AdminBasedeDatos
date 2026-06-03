"""
Práctica: Análisis de Datos - ETL Pipeline
Parte II y III: Implementación ETL + Minería de Datos
Entorno: pandas, PySpark, scikit-learn
"""

# ─────────────────────────────────────────────
# PASO 1 — EXTRACCIÓN: Generar datasets sintéticos
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
from faker import Faker
import random
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# Rutas relativas al script — funciona desde cualquier directorio
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

fake = Faker("es_MX")
random.seed(42)
np.random.seed(42)

# --- ventas.csv ---
categorias = ["Electrónica", "Ropa", "Alimentos", "Hogar", "Deportes"]
productos_cat = {
    "Electrónica": ["Laptop", "Auriculares", "Teclado", "Monitor", "Webcam"],
    "Ropa":        ["Playera", "Pantalón", "Chamarra", "Calcetines", "Gorra"],
    "Alimentos":   ["Cereal", "Café", "Yogurt", "Arroz", "Avena"],
    "Hogar":       ["Lámpara", "Cojín", "Tapete", "Cortinas", "Reloj"],
    "Deportes":    ["Pelota", "Guantes", "Tenis", "Pesas", "Cuerda"]
}

n_ventas = 500
cat_list  = random.choices(categorias, k=n_ventas)
prod_list = [random.choice(productos_cat[c]) for c in cat_list]

df_ventas = pd.DataFrame({
    "venta_id":        range(1, n_ventas + 1),
    "fecha":           [fake.date_between(start_date="-1y", end_date="today") for _ in range(n_ventas)],
    "cliente_id":      [random.randint(1, 100) for _ in range(n_ventas)],
    "producto":        prod_list,
    "categoria":       cat_list,
    "cantidad":        [random.choice([None, random.randint(1, 20)]) for _ in range(n_ventas)],
    "precio_unitario": [round(random.uniform(10, 5000), 2) if random.random() > 0.05 else None for _ in range(n_ventas)],
    "descuento":       [round(random.uniform(0, 0.3), 2) for _ in range(n_ventas)],
})
# Insertar duplicados intencionales
df_ventas = pd.concat([df_ventas, df_ventas.sample(20)], ignore_index=True)
df_ventas.to_csv(str(DATA_DIR / "ventas.csv"), index=False)

# --- productos.xlsx ---
n_prod = 40
productos_flat = [p for ps in productos_cat.values() for p in ps]
df_productos = pd.DataFrame({
    "producto_id":  range(1, n_prod + 1),
    "nombre":       random.choices(productos_flat, k=n_prod),
    "categoria":    random.choices(categorias, k=n_prod),
    "precio":       [round(random.uniform(50, 8000), 2) if random.random() > 0.04 else None for _ in range(n_prod)],
    "inventario":   [random.randint(0, 500) if random.random() > 0.06 else None for _ in range(n_prod)],
    "proveedor":    [fake.company() for _ in range(n_prod)],
    "fecha_alta":   [fake.date_between(start_date="-3y", end_date="today") for _ in range(n_prod)],
})
df_productos.to_excel(str(DATA_DIR / "productos.xlsx"), index=False)

# --- clientes.json ---
import json
clientes = []
for i in range(1, 101):
    clientes.append({
        "cliente_id":  i,
        "nombre":      fake.name() if random.random() > 0.03 else None,
        "email":       fake.email() if random.random() > 0.05 else "sin_email",
        "ciudad":      fake.city(),
        "edad":        random.randint(18, 75) if random.random() > 0.07 else None,
        "genero":      random.choice(["M", "F", "M", "F", None]),
        "registro":    str(fake.date_between(start_date="-5y", end_date="today")),
    })
with open(str(DATA_DIR / "clientes.json"), "w", encoding="utf-8") as f:
    json.dump(clientes, f, ensure_ascii=False, indent=2)

# --- comentarios.txt ---
sentimientos = ["positivo", "negativo", "neutro"]
with open(str(DATA_DIR / "comentarios.txt"), "w", encoding="utf-8") as f:
    f.write("=== Comentarios de Clientes ===\n\n")
    for i in range(1, 51):
        sent = random.choice(sentimientos)
        if sent == "positivo":
            texto = fake.sentence(nb_words=10) + " Excelente producto, lo recomiendo."
        elif sent == "negativo":
            texto = fake.sentence(nb_words=8) + " No cumplió mis expectativas."
        else:
            texto = fake.sentence(nb_words=12)
        f.write(f"[{i}] {fake.name()} ({sent}): {texto}\n")

print("✓ Datos generados correctamente")
print(f"  ventas.csv     → {len(df_ventas)} registros (con duplicados)")
print(f"  productos.xlsx → {len(df_productos)} registros")
print(f"  clientes.json  → {len(clientes)} registros")
print(f"  comentarios.txt→ 50 comentarios\n")

# ─────────────────────────────────────────────────────────
# PASO 2 — TRANSFORMACIÓN con pandas (ETL básico)
# ─────────────────────────────────────────────────────────
print("=" * 55)
print("PASO 2 — TRANSFORMACIÓN (pandas)")
print("=" * 55)

# 2.1 Cargar
df_v = pd.read_csv(str(DATA_DIR / "ventas.csv"))
df_p = pd.read_excel(str(DATA_DIR / "productos.xlsx"))
df_c = pd.DataFrame(json.load(open(str(DATA_DIR / "clientes.json"))))

print(f"\n[ventas] shape antes: {df_v.shape}")
print(df_v.dtypes)

# 2.2 Duplicados
antes = len(df_v)
df_v = df_v.drop_duplicates()
print(f"\n  Duplicados eliminados: {antes - len(df_v)}")

# 2.3 Nulos — imputar con mediana (numéricos) y moda (categóricos)
for col in ["cantidad", "precio_unitario"]:
    mediana = df_v[col].median()
    nulos = df_v[col].isna().sum()
    df_v[col] = df_v[col].fillna(mediana)
    print(f"  {col}: {nulos} nulos imputados con mediana={mediana:.2f}")

for col in ["precio", "inventario"]:
    mediana = df_p[col].median()
    nulos = df_p[col].isna().sum()
    df_p[col] = df_p[col].fillna(mediana)
    print(f"  productos.{col}: {nulos} nulos imputados con mediana={mediana:.2f}")

# Clientes
df_c["nombre"]  = df_c["nombre"].fillna("Desconocido")
df_c["email"]   = df_c["email"].replace("sin_email", np.nan).fillna("no-email@placeholder.com")
df_c["edad"]    = df_c["edad"].fillna(df_c["edad"].median())
df_c["genero"]  = df_c["genero"].fillna(df_c["genero"].mode()[0])

# 2.4 Corrección de tipos
df_v["fecha"] = pd.to_datetime(df_v["fecha"])
df_v["total"] = df_v["cantidad"] * df_v["precio_unitario"] * (1 - df_v["descuento"])

print(f"\n[ventas] shape después: {df_v.shape}")
print(df_v.head(3).to_string())

# ─────────────────────────────────────────────────────────
# PASO 3 — CARGA: guardar en Parquet (columnar, eficiente)
# ─────────────────────────────────────────────────────────
import pyarrow as pa
import pyarrow.parquet as pq

pq.write_table(pa.Table.from_pandas(df_v), str(DATA_DIR / "ventas_clean.parquet"),   coerce_timestamps="us")
pq.write_table(pa.Table.from_pandas(df_p), str(DATA_DIR / "productos_clean.parquet"), coerce_timestamps="us")
pq.write_table(pa.Table.from_pandas(df_c), str(DATA_DIR / "clientes_clean.parquet"),  coerce_timestamps="us")
print("\n✓ Datos limpios guardados en formato Parquet (compatible Spark)")

# ─────────────────────────────────────────────────────────
# PARTE III — ANÁLISIS EXPLORATORIO (Actividad 6)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PARTE III — ANÁLISIS EXPLORATORIO")
print("=" * 55)

df_p["valor_inventario"] = df_p["precio"] * df_p["inventario"]

prod_caro  = df_p.loc[df_p["precio"].idxmax()]
prod_stock = df_p.loc[df_p["inventario"].idxmax()]
prod_valor = df_p.loc[df_p["valor_inventario"].idxmax()]

print(f"\n¿Cuál es el producto más caro?")
print(f"  → {prod_caro['nombre']} ({prod_caro['categoria']}) — ${prod_caro['precio']:.2f}")

print(f"\n¿Cuál tiene mayor inventario?")
print(f"  → {prod_stock['nombre']} ({prod_stock['categoria']}) — {int(prod_stock['inventario'])} unidades")

print(f"\n¿Cuál representa mayor valor económico en inventario?")
print(f"  → {prod_valor['nombre']} ({prod_valor['categoria']}) — ${prod_valor['valor_inventario']:,.2f}")

print("\n--- Estadísticas generales de productos ---")
print(df_p[["precio", "inventario", "valor_inventario"]].describe().round(2).to_string())

print("\n--- Ventas totales por categoría (Top 5) ---")
print(df_v.groupby("categoria")["total"].sum().sort_values(ascending=False).head().to_string())

# ─────────────────────────────────────────────────────────
# ACTIVIDAD 7 — PREDICCIÓN: Regresión con scikit-learn
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ACTIVIDAD 7 — PREDICCIÓN SIMPLE (scikit-learn)")
print("=" * 55)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

df_ml = df_p[["precio", "inventario", "categoria", "valor_inventario"]].copy().dropna()
le = LabelEncoder()
df_ml["cat_enc"] = le.fit_transform(df_ml["categoria"])

X = df_ml[["precio", "inventario", "cat_enc"]]
y = df_ml["valor_inventario"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print(f"\nLinearRegression  → MAE: {mean_absolute_error(y_test, y_pred_lr):,.2f}  | R²: {r2_score(y_test, y_pred_lr):.4f}")
print(f"GradientBoosting  → MAE: {mean_absolute_error(y_test, y_pred_gb):,.2f}  | R²: {r2_score(y_test, y_pred_gb):.4f}")

df_resultados = X_test.copy()
df_resultados["categoria"]       = le.inverse_transform(df_resultados["cat_enc"])
df_resultados["valor_real"]      = y_test.values
df_resultados["pred_lineal"]     = y_pred_lr.round(2)
df_resultados["pred_gradboost"]  = y_pred_gb.round(2)
df_resultados["error_lineal"]    = (df_resultados["pred_lineal"]    - df_resultados["valor_real"]).abs().round(2)
df_resultados["error_gradboost"] = (df_resultados["pred_gradboost"] - df_resultados["valor_real"]).abs().round(2)
df_resultados = df_resultados.drop(columns="cat_enc").reset_index(drop=True)

print("\n--- DataFrame de predicciones (muestra) ---")
print(df_resultados.head(10).to_string())

print("\n✓ Pipeline completo ejecutado")