import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("MONGO_DB")
COL = os.getenv("MONGO_COLLECTION")

MONGO_URI = (
    f"mongodb+srv://{os.getenv('MONGO_USER')}:"
    f"{os.getenv('MONGO_PASSWORD')}@"
    f"{os.getenv('MONGO_CLUSTER')}/"
    f"?retryWrites=true&w=majority"
)