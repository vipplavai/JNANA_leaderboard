
"""
Database operations for MongoDB connection and reference data handling
"""
import json
import streamlit as st
from pymongo import MongoClient
from typing import Dict, Tuple, Optional

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.ref_collection = None
        self.submissions_collection = None
        self.mongodb_available = False
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup MongoDB connection with fallback"""
        try:
            mongo_uri = st.secrets.get("mongo_uri", "mongodb://localhost:27017")
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client["Leaderboard"]
            self.ref_collection = self.db["reference_samples"]
            self.submissions_collection = self.db["submissions"]
            self.mongodb_available = True
            st.success("✅ MongoDB connected successfully")
        except Exception as e:
            st.warning(f"⚠️ MongoDB connection failed: {e}")
            st.info("📝 Running in local mode - submissions will not be saved")
            self.mongodb_available = False
    
    def get_reference_data(self) -> Dict:
        """Load and return reference data lookup dictionary"""
        if self.mongodb_available and self.ref_collection is not None:
            try:
                # Try to load from MongoDB first
                ref_data = list(self.ref_collection.find({}))
                
                # If MongoDB is empty, populate from local file
                if not ref_data:
                    with open("data/samples_1000.json", "r", encoding="utf-8") as f:
                        ref_data = json.load(f)
                    
                    # Insert into MongoDB for future use
                    self.ref_collection.insert_many(ref_data)
                    st.success("✅ Populated 1000 reference samples into MongoDB")
                
                return self._create_lookup_dict(ref_data)
                
            except Exception as e:
                st.warning(f"⚠️ MongoDB error, falling back to local file: {e}")
        
        # Fallback to local file
        return self._load_local_reference_data()
    
    def _create_lookup_dict(self, ref_data) -> Dict:
        """Create optimized lookup dictionary from reference data"""
        lookup_dict = {}
        for item in ref_data:
            try:
                content_id = int(item.get("content_id", 0))
                qa_index = int(item.get("qa_index", 0))
                content_text = item.get("content_text", "")
                lookup_dict[(content_id, qa_index)] = content_text
            except (ValueError, TypeError):
                continue  # Skip malformed entries
        return lookup_dict
    
    def _load_local_reference_data(self) -> Dict:
        """Load reference data from local file"""
        try:
            with open("data/samples_1000.json", "r", encoding="utf-8") as f:
                ref_data = json.load(f)
            return self._create_lookup_dict(ref_data)
        except Exception as e:
            st.error(f"❌ Error loading reference data: {e}")
            return {}
    
    def save_submission(self, submission_data: Dict) -> bool:
        """Save submission to database"""
        if not self.mongodb_available:
            st.error("❌ Cannot save submission - MongoDB not available")
            return False
        
        try:
            self.submissions_collection.insert_one(submission_data)
            return True
        except Exception as e:
            st.error(f"❌ Error saving submission: {e}")
            return False
    
    def load_submissions(self) -> list:
        """Load all submissions from database"""
        if not self.mongodb_available:
            return []
        
        try:
            return list(self.submissions_collection.find({}).sort("timestamp", -1))
        except Exception as e:
            st.error(f"❌ Error loading submissions: {e}")
            return []
    
    def check_duplicate_submission(self, model: str, author: str, sha1_hash: str) -> bool:
        """Check if submission already exists"""
        if not self.mongodb_available:
            return False
        
        try:
            existing = self.submissions_collection.find_one({
                "model": model,
                "author": author,
                "sha1_hash": sha1_hash
            })
            return existing is not None
        except Exception as e:
            st.error(f"❌ Error checking duplicate: {e}")
            return False
