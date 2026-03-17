import streamlit as st
import os
import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# Conexión a MongoDB
uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
client = MongoClient(uri)
df = pd.DataFrame(
    list(
        client[os.getenv("DB_NAME")]
        [os.getenv("COLLECTION_NAME")]
        .find({}, {"_id": 0})
    )
)

st.title("Regresión Lineal Múltiple")

# Variables independientes y dependiente
X = df[["peso", "edad", "meses_inscrito"]]
y = df["precio_mensual"]

# Modelo
model = LinearRegression()
model.fit(X, y)

# Predicciones
y_pred = model.predict(X)

# Mostrar coeficientes
st.subheader("Coeficientes del modelo")
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": model.coef_
})
st.dataframe(coef_df)

# Gráfica Real vs Predicho
fig, ax = plt.subplots()
ax.scatter(y, y_pred, alpha=0.5)
ax.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
ax.set_xlabel("Precio real")
ax.set_ylabel("Precio predicho")
ax.set_title("Valores reales vs valores predichos")

st.pyplot(fig)
