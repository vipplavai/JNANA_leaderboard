
"""
UI components for the leaderboard application
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

class UIComponents:
    @staticmethod
    def render_header():
        """Render application header"""
        st.set_page_config(page_title="JNANA QA Leaderboard", layout="wide")
        st.title("📊 JNANA Telugu QA Leaderboard")
        
        st.markdown("""
        **Welcome to the official JNANA QA Leaderboard!**

        This leaderboard evaluates Telugu short-answer question-answering models using a curated 1000-sample benchmark.

        📎 **Download Evaluation Dataset**: [samples_1000.json](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json)

        🔒 **Open Leaderboard**: All submissions are publicly viewable but cannot be edited or deleted to ensure leaderboard integrity.
        """)
    
    @staticmethod
    def render_submission_form() -> Dict:
        """Render submission form in sidebar"""
        st.sidebar.header("📥 Submit Your Model Output")
        
        form_data = {
            'model_name': st.sidebar.text_input("Model Name (required)"),
            'author_name': st.sidebar.text_input("Your Name or Alias (required)"),
            'version_tag': st.sidebar.text_input("Version Tag (optional)", placeholder="v1.0"),
            'notes': st.sidebar.text_area("Notes (optional)", placeholder="Brief description of your model"),
            'uploaded_file': st.sidebar.file_uploader("Upload result JSON file", type="json")
        }
        
        return form_data
    
    @staticmethod
    def render_leaderboard(leaderboard_data: List[Dict]):
        """Render leaderboard table"""
        st.subheader("🏆 Leaderboard")
        show_advanced = st.toggle("Show Advanced Metrics", value=False)
        
        if not leaderboard_data:
            st.info("No submissions yet.")
            return
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        if show_advanced:
            # Add advanced metrics columns if needed
            advanced_cols = ["FAA (%)", "F1-EM Gap", "Overconfident EM (%)", 
                           "Robust Answer Rate (%)", "Avg Answer Length"]
            for col in advanced_cols:
                if col not in leaderboard_df.columns:
                    leaderboard_df[col] = 0.0
        
        st.dataframe(leaderboard_df, use_container_width=True)
    
    @staticmethod
    def render_sample_explorer(all_data: Dict, ref_lookup: Dict):
        """Render sample explorer interface"""
        st.subheader("🔍 Sample Explorer")
        st.markdown("""
        ℹ️ **How to Use:**
        - Choose a submission from the dropdown (shows model name, author, version, and timestamp).
        - Filter samples by type: hallucinated, faithful_correct, etc.
        - Use the slider to browse examples.
        - This is a read-only view of all submissions in the leaderboard.
        """)
        
        if not all_data:
            st.info("No submissions available yet. Submit your first model to see results here!")
            return
        
        selected_submission = st.selectbox(
            "Choose a submission to explore", 
            ["None"] + list(all_data.keys()),
            help="Select a model submission to explore its predictions"
        )
        
        if selected_submission == "None":
            return
        
        UIComponents._render_submission_details(all_data[selected_submission], ref_lookup)
    
    @staticmethod
    def _render_submission_details(submission_info: Dict, ref_lookup: Dict):
        """Render details for selected submission"""
        try:
            df = submission_info["data"]
            metadata = submission_info["metadata"]
            
            # Show submission metadata
            col1, col2, col3, col4 = st.columns(4)
            metrics = metadata.get('metrics', {})
            
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("EM Score", f"{metrics.get('em', 0):.1f}%")
            with col3:
                st.metric("F1 Score", f"{metrics.get('f1', 0):.1f}%")
            with col4:
                # Calculate context coverage
                context_coverage = UIComponents._calculate_context_coverage(df, ref_lookup)
                st.metric("Context Coverage", f"{context_coverage:.1f}%")
            
            # Add notes if available
            if metadata.get("notes"):
                st.info(f"**Notes**: {metadata['notes']}")
            
            # Filter controls
            tag_filter = st.selectbox("Breakdown Filter", ["all"] + sorted(df["breakdown"].unique()))
            if tag_filter != "all":
                df = df[df["breakdown"] == tag_filter]
            
            if df.empty:
                st.warning("No samples for this filter.")
                return
            
            # Sample browser
            i = st.slider("Sample Index", 0, len(df) - 1, 0)
            row = df.iloc[i]
            
            # Get context
            context_text = UIComponents._get_context_for_sample(
                row["content_id"], row["qa_index"], ref_lookup
            )
            
            # Display sample details
            st.markdown(f"**Q{row['qa_index']}**: {row['question']}")
            st.markdown(f"**Gold Answer**: {row['gold_answer']}")
            st.markdown(f"**Prediction**: {row['prediction']}")
            st.markdown(f"**F1**: {row['f1_score']:.2f} | EM: {row['exact_match']} | Hallucinated: {row['hallucinated']}")
            st.markdown(f"**Type**: {row['breakdown']}")
            st.markdown("---")
            st.markdown(f"**Context**:\n\n{context_text}")
            
        except Exception as e:
            st.error(f"Error displaying submission details: {e}")
    
    @staticmethod
    def _calculate_context_coverage(df: pd.DataFrame, ref_lookup: Dict) -> float:
        """Calculate context coverage percentage"""
        if len(df) == 0:
            return 0.0
        
        context_available = 0
        for _, row in df.iterrows():
            key = (int(row["content_id"]), int(row["qa_index"]))
            if key in ref_lookup and ref_lookup[key]:
                context_available += 1
        
        return (context_available / len(df)) * 100
    
    @staticmethod
    def _get_context_for_sample(content_id: int, qa_index: int, ref_lookup: Dict) -> str:
        """Get context for a specific sample"""
        try:
            key = (int(content_id), int(qa_index))
            return ref_lookup.get(key, "[Context not available]")
        except Exception as e:
            return f"[Error loading context: {e}]"
    
    @staticmethod
    def render_status_info(ref_lookup: Dict, mongodb_available: bool):
        """Render status information in sidebar"""
        if ref_lookup:
            st.sidebar.success(f"📚 {len(ref_lookup)} reference samples loaded")
        else:
            st.sidebar.error("⚠️ No reference data available")
        
        if not mongodb_available:
            st.sidebar.warning("⚠️ MongoDB not available - running in local mode")
