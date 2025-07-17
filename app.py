import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pymongo import MongoClient
from typing import List, Dict

# Import validation and metrics functions
from validate import clean_and_validate_submission
from metrics import compute_metrics
from database import DatabaseManager
from ui_components import UIComponents
from submission_handler import SubmissionHandler

"""
JNANA Telugu QA Leaderboard - Main Application
A comprehensive leaderboard system for evaluating Telugu question-answering models
"""

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize database and other components"""
    db_manager = DatabaseManager()
    submission_handler = SubmissionHandler(db_manager)
    return db_manager, submission_handler

# Main application
def main():
    # Initialize components
    db_manager, submission_handler = initialize_components()

    # Render header
    UIComponents.render_header()

    # Load reference data with caching
    @st.cache_data(ttl=3600)
    def get_reference_lookup():
        return db_manager.get_reference_data()

    ref_lookup = get_reference_lookup()

    # Render status information
    UIComponents.render_status_info(ref_lookup, db_manager.mongodb_available)

    # Handle submission form
    form_data = UIComponents.render_submission_form()

    # Process submission if uploaded
    if form_data['uploaded_file'] and "uploaded" not in st.session_state:
        if submission_handler.process_submission(form_data):
            st.session_state["uploaded"] = True
            st.rerun()

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Load submissions and prepare leaderboard
    submissions = db_manager.load_submissions()
    leaderboard_rows, all_data = submission_handler.prepare_leaderboard_data(submissions)

    # Render leaderboard
    UIComponents.render_leaderboard(leaderboard_rows)

    # Render sample explorer
    UIComponents.render_sample_explorer(all_data, ref_lookup)

if __name__ == "__main__":
    main()