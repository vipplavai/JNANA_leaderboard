
"""
Submission handling and processing logic
"""
import json
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from validate import clean_and_validate_submission
from metrics import compute_metrics

class SubmissionHandler:
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def process_submission(self, form_data: Dict) -> bool:
        """Process uploaded submission file"""
        uploaded_file = form_data.get('uploaded_file')
        model_name = form_data.get('model_name')
        author_name = form_data.get('author_name')
        
        if not uploaded_file:
            return False
        
        if not model_name or not author_name:
            st.sidebar.error("❌ Model name and author are required.")
            return False
        
        try:
            # Parse uploaded file
            raw_bytes = uploaded_file.read()
            parsed_data = json.loads(raw_bytes)
            
            if not isinstance(parsed_data, list):
                st.sidebar.error("❌ Submission file must be a list of JSON objects.")
                return False
            
            # Validate submission
            cleaned_data, validation_errors, sha1_hash = clean_and_validate_submission(parsed_data)
            
            if validation_errors:
                self._display_validation_errors(validation_errors)
                return False
            
            # Check for duplicates
            if self.db_manager.check_duplicate_submission(model_name, author_name, sha1_hash):
                st.sidebar.warning("⚠️ Identical submission already exists!")
                return False
            
            # Process submission
            return self._save_submission(form_data, cleaned_data, sha1_hash)
            
        except json.JSONDecodeError:
            st.sidebar.error("❌ Invalid JSON format.")
            return False
        except Exception as e:
            st.sidebar.error(f"❌ Error processing submission: {str(e)}")
            return False
    
    def _display_validation_errors(self, errors: List[str]):
        """Display validation errors in sidebar"""
        st.sidebar.error("❌ Validation failed!")
        for err in errors[:5]:
            st.sidebar.write(err)
        if len(errors) > 5:
            st.sidebar.warning(f"...and {len(errors)-5} more errors")
    
    def _save_submission(self, form_data: Dict, cleaned_data: List[Dict], sha1_hash: str) -> bool:
        """Save validated submission to database"""
        try:
            metrics = compute_metrics(cleaned_data)
            
            submission_meta = {
                "model": form_data['model_name'],
                "author": form_data['author_name'],
                "timestamp": datetime.utcnow(),
                "version_tag": form_data.get('version_tag') or None,
                "notes": form_data.get('notes') or None,
                "sha1_hash": sha1_hash,
                "metrics": metrics,
                "results": cleaned_data
            }
            
            if self.db_manager.save_submission(submission_meta):
                st.sidebar.success("✅ Submission uploaded successfully!")
                return True
            else:
                return False
                
        except Exception as e:
            st.sidebar.error(f"❌ Error saving submission: {str(e)}")
            return False
    
    def prepare_leaderboard_data(self, submissions: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Prepare leaderboard data and all submissions data"""
        leaderboard_rows = []
        all_data = {}
        
        for sub in submissions:
            try:
                results = sub.get("results", [])
                if not results:
                    continue
                
                # Prepare submission data
                submission_data = self._prepare_submission_data(sub, results)
                all_data[submission_data["display_name"]] = submission_data["data"]
                
                # Prepare leaderboard row
                leaderboard_row = self._prepare_leaderboard_row(sub, results)
                leaderboard_rows.append(leaderboard_row)
                
            except Exception as e:
                st.error(f"Error processing submission: {e}")
                continue
        
        return leaderboard_rows, all_data
    
    def _prepare_submission_data(self, submission: Dict, results: List[Dict]) -> Dict:
        """Prepare submission data for explorer"""
        import pandas as pd
        
        df = pd.DataFrame(results)
        df["breakdown"] = df["type"]
        
        # Create display name
        model_name = submission.get("model", "Unknown")
        author_name = submission.get("author", "Unknown")
        version_tag = submission.get("version_tag", "")
        timestamp = submission["timestamp"].strftime("%Y-%m-%d %H:%M")
        
        display_name = f"🤖 {model_name}"
        if version_tag:
            display_name += f" v{version_tag}"
        display_name += f" | 👤 {author_name} | 📅 {timestamp}"
        
        return {
            "display_name": display_name,
            "data": {
                "data": df,
                "metadata": submission
            }
        }
    
    def _prepare_leaderboard_row(self, submission: Dict, results: List[Dict]) -> Dict:
        """Prepare leaderboard row data"""
        metrics = submission.get("metrics", {})
        timestamp = submission["timestamp"].strftime("%Y-%m-%d %H:%M")
        
        return {
            "Model": submission.get("model", "Unknown"),
            "Author": submission.get("author", "Unknown"),
            "Version": submission.get("version_tag", "N/A"),
            "Samples": metrics.get("total", len(results)),
            "EM (%)": metrics.get("em", 0.0),
            "F1 (%)": metrics.get("f1", 0.0),
            "Answered (%)": metrics.get("answered", 0.0),
            "Hallucinated (%)": metrics.get("hallucinated", 0.0),
            "Faithful Correct (%)": metrics.get("faithful_correct", 0.0),
            "Faithful Incorrect (%)": metrics.get("faithful_incorrect", 0.0),
            "Empty (%)": metrics.get("empty", 0.0),
            "Timestamp": timestamp
        }
