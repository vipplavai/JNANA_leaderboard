
import json
from pymongo import MongoClient
import streamlit as st

def populate_reference_data():
    """Populate MongoDB with reference data from samples_1000.json"""
    
    # MongoDB connection
    MONGO_URI = st.secrets.get("mongo_uri", "mongodb://localhost:27017")
    client = MongoClient(MONGO_URI)
    db = client["Leaderboard"]
    ref_collection = db["reference_samples"]
    
    # Check if data already exists
    if ref_collection.count_documents({}) > 0:
        print(f"Reference collection already has {ref_collection.count_documents({})} documents")
        return
    
    # Load reference data
    try:
        with open("data/samples_1000.json", "r", encoding="utf-8") as f:
            ref_data = json.load(f)
        
        print(f"Loaded {len(ref_data)} reference samples")
        
        # Insert into MongoDB
        ref_collection.insert_many(ref_data)
        print(f"Successfully inserted {len(ref_data)} reference samples into MongoDB")
        
    except FileNotFoundError:
        print("Error: data/samples_1000.json not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_reference_data()
