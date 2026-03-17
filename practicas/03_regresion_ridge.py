import streamlit as st
import os
import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import Ridge
from dotenv import load_dotenv

load_dotenv()

uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
df = pd.DataFrame(list(MongoClient(uri)[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")].find({}, {"_id":0})))

st.title("Regresión Ridge")

X = df[["paginas"]]
y = df["precio"]

model = Ridge(alpha=1.0)
model.fit(X, y)

st.write(model.coef_)
