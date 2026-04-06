# ============================================================
# SISTEMA COMPLETO: SPARK + DEEP LEARNING + DASHBOARD (GIMNASIO)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

from common.mongo_spark import get_spark_df


# ================== CONFIG UI ==================
st.set_page_config(page_title="Sistema IA - Gimnasio", layout="wide")

st.title("🏋️ Sistema Inteligente: Spark + Deep Learning (Gimnasio)")
st.markdown(
    """
Este dashboard muestra un pipeline real de IA:

1. Datos en Spark (MongoDB)  
2. Feature engineering + escalado  
3. Conversión a tensores (PyTorch)  
4. Entrenamiento de red neuronal  
5. Predicción en tiempo real  
6. Interpretación automática  
"""
)

st.sidebar.header("⚙️ Configuración")
epochs = st.sidebar.slider("Épocas", 10, 300, 120, 10)
lr = st.sidebar.select_slider("Learning rate", options=[0.001, 0.003, 0.005, 0.01, 0.02], value=0.01)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)


# ================== CARGA SPARK (CACHE) ==================
@st.cache_resource
def cargar_df_spark():
    return get_spark_df("Dashboard_Spark_PyTorch_Gym")

df = cargar_df_spark()

# ================== LIMPIEZA ==================
df = df.fillna({
    "peso": 0,
    "altura": 0,
    "edad": 0,
    "meses_inscrito": 0,
    "precio_mensual": 0,
    "tipo_membresia": "Basica"
})

st.subheader("📌 Datos originales (muestra)")
st.dataframe(df.select("peso", "altura", "edad", "meses_inscrito", "precio_mensual", "tipo_membresia").limit(30).toPandas())


# ================== LABEL (STRING -> NUM) ==================
indexer = StringIndexer(inputCol="tipo_membresia", outputCol="label", handleInvalid="keep")
indexer_model = indexer.fit(df)
df_labeled = indexer_model.transform(df)

labels = indexer_model.labels
num_classes = len(labels)

c1, c2 = st.columns(2)
with c1:
    st.subheader("🧾 Clases detectadas")
    st.write({i: labels[i] for i in range(num_classes)})
with c2:
    st.subheader("📊 Conteo por clase")
    st.dataframe(df_labeled.groupBy("tipo_membresia").count().toPandas())


# ================== FEATURES + ESCALADO ==================
feature_cols = ["peso", "altura", "edad", "meses_inscrito", "precio_mensual"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)
df_vector = assembler.transform(df_labeled)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True
)
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

# ================== FIX ARROW / VISUALIZACIÓN FEATURES ==================
df_fixed = df_scaled.withColumn("features_array", vector_to_array(col("scaledFeatures")))
for i, name in enumerate(feature_cols):
    df_fixed = df_fixed.withColumn(f"f{i+1}", col("features_array")[i])

st.subheader("🧪 Datos escalados (muestra)")
st.dataframe(df_fixed.select([f"f{i+1}" for i in range(len(feature_cols))] + ["label"]).limit(30).toPandas())

st.subheader("🗺️ Visualización 2D de features escaladas")
pdf_plot = df_fixed.select("f1", "f2", "label").limit(2000).toPandas()
fig_scatter = px.scatter(
    pdf_plot,
    x="f1",
    y="f2",
    color=pdf_plot["label"].astype(str),
    title="Distribución (2 features escaladas) coloreado por clase"
)
st.plotly_chart(fig_scatter, use_container_width=True)


# ================== SPARK -> PANDAS -> PYTORCH ==================
pdf = df_scaled.select("scaledFeatures", "label").toPandas()
X = pdf["scaledFeatures"].apply(lambda v: v.toArray()).tolist()
y = pdf["label"].astype(int).values

stratify = y if len(set(y)) > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=42, stratify=stratify
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

input_dim = len(feature_cols)


# ================== RED NEURONAL ==================
class RedNeuronal(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ================== ENTRENAMIENTO (BOTÓN) ==================
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []

st.subheader("🧠 Entrenamiento")

if st.button("🚀 Entrenar modelo"):
    model = RedNeuronal(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(lr))

    loss_history = []
    for epoch in range(epochs):
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    st.session_state.model = model
    st.session_state.loss_history = loss_history
    st.session_state.trained = True

    st.success("Modelo entrenado correctamente ✅")

if st.session_state.trained:
    df_loss = pd.DataFrame({"Epoch": list(range(1, len(st.session_state.loss_history) + 1)),
                            "Loss": st.session_state.loss_history})
    fig_loss = px.line(df_loss, x="Epoch", y="Loss", title="Curva de aprendizaje (Loss)")
    st.plotly_chart(fig_loss, use_container_width=True)

    # ================== EVALUACIÓN ==================
    with torch.no_grad():
        outputs = st.session_state.model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_test_t).sum().item() / len(y_test_t) if len(y_test_t) else 0.0

    st.subheader("✅ Evaluación")
    st.metric("Accuracy (test)", f"{acc*100:.2f}%")

    # ================== PREDICCIÓN EN TIEMPO REAL ==================
    st.subheader("🔮 Predicción en tiempo real")

    a, b = st.columns(2)
    with a:
        peso = st.slider("Peso", 40.0, 140.0, 80.0, 0.1)
        altura = st.slider("Altura", 1.40, 2.10, 1.70, 0.01)
        edad = st.slider("Edad", 16, 80, 28, 1)
    with b:
        meses = st.slider("Meses inscrito", 0, 60, 10, 1)
        precio = st.slider("Precio mensual", 200, 1000, 450, 10)

    nuevo = [(float(peso), float(altura), int(edad), int(meses), int(precio))]
    nuevo_df = df_scaled.sql_ctx.createDataFrame(nuevo, feature_cols)

    nuevo_vector = assembler.transform(nuevo_df)
    nuevo_scaled = scaler_model.transform(nuevo_vector)

    nuevo_pdf = nuevo_scaled.select("scaledFeatures").toPandas()
    nuevo_X = torch.tensor(
        nuevo_pdf["scaledFeatures"].apply(lambda v: v.toArray()).tolist(),
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = st.session_state.model(nuevo_X)
        prob = torch.softmax(logits, dim=1).numpy()[0]
        clase = int(prob.argmax())

    st.write(f"Clase predicha: **{labels[clase]}**")

    prob_df = pd.DataFrame({"Membresía": labels, "Probabilidad": prob}).sort_values("Probabilidad", ascending=False)
    st.dataframe(prob_df, use_container_width=True)

    fig_prob = px.bar(prob_df, x="Membresía", y="Probabilidad", title="Probabilidades por clase", range_y=[0, 1])
    st.plotly_chart(fig_prob, use_container_width=True)

    # ================== IA EXPLICATIVA ==================
    st.subheader("🤖 Interpretación automática")

    top = prob_df.iloc[0]
    texto = (
        f"El modelo predice **{labels[clase]}** con probabilidad **{top['Probabilidad']:.2f}**.\n\n"
        "Interpretación rápida:\n"
    )

    if labels[clase].lower().startswith("vip") or "premium" in labels[clase].lower():
        texto += "- Perfil con mayor disposición a pagar.\n- Recomendación: upsell (entrenador personal, nutrición, paquetes).\n"
    elif "bas" in labels[clase].lower() or "eco" in labels[clase].lower():
        texto += "- Perfil sensible al precio.\n- Recomendación: promociones, planes trimestrales, retención.\n"
    else:
        texto += "- Perfil intermedio.\n- Recomendación: ofertas mixtas y seguimiento para aumentar permanencia.\n"

    st.success(texto)

else:
    st.info("Entrena el modelo para habilitar evaluación y predicción.")


st.markdown(
    """
---
### 🧾 Resumen del sistema

- **Spark:** carga y procesamiento de datos desde MongoDB  
- **Feature engineering:** vectorización + escalado  
- **PyTorch:** red neuronal para clasificación  
- **Streamlit:** dashboard interactivo + visualización + explicación automática  
"""
)
