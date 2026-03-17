# 03_ridge.py
import streamlit as st
import os, pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import Ridge
from dotenv import load_dotenv

load_dotenv()

uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
df = pd.DataFrame(list(MongoClient(uri)[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")].find({}, {"_id":0})))

st.title("Regresión Ridge")

model = Ridge(alpha=1.0).fit(df[["peso"]], df["precio_mensual"])
st.write(model.coef_)
