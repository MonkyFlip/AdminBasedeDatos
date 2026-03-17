import streamlit as st
import os
import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

load_dotenv()

uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
df = pd.DataFrame(list(MongoClient(uri)[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")].find({}, {"_id":0})))

st.title("Regresión Lineal Múltiple")

X = df[["paginas", "anio"]]
y = df["precio"]

model = LinearRegression()
model.fit(X, y)

st.write(model.coef_)
