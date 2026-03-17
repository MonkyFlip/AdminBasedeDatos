import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def get_dataframe():
    uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_CLUSTER')}/"
    client = MongoClient(uri)
    db = client[os.getenv("DB_NAME")]
    col = db[os.getenv("COLLECTION_NAME")]
    return pd.DataFrame(list(col.find({}, {"_id": 0})))
