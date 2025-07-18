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

@st.cache_resource
def initialize_components():
    """Initialize database and other components"""
    db_manager = DatabaseManager()
    submission_handler = SubmissionHandler(db_manager)
    return db_manager, submission_handler

def main():
    # Initialize
    db_manager, submission_handler = initialize_components()
    UIComponents.render_header()

    # Load reference data silently
    @st.cache_data(ttl=3600)
    def get_reference_lookup():
        return db_manager.get_reference_data()

    ref_lookup = get_reference_lookup()

    # Render submission form in main area
    form_data = UIComponents.render_submission_form()

    if form_data.get('submitted') and form_data.get('uploaded_file'):
        if submission_handler.process_submission(form_data):
            st.success("✅ Submission successful!")
            st.rerun()

    st.markdown("---")

    # Load and display data
    submissions = db_manager.load_submissions()
    leaderboard_rows, all_data = submission_handler.prepare_leaderboard_data(submissions)

    UIComponents.render_leaderboard(leaderboard_rows)
    UIComponents.render_sample_explorer(all_data, ref_lookup)

if __name__ == "__main__":
    main()