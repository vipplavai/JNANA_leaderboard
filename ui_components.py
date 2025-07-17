"""
UI Components for the JNANA QA Leaderboard
"""
import streamlit as st
import pandas as pd
from typing import Dict, List

class UIComponents:
    @staticmethod
    def render_header():
        st.set_page_config(page_title="JNANA QA Leaderboard", layout="wide")
        st.title("📊 JNANA Telugu QA Leaderboard")
        st.markdown("""
            **Welcome to the official JNANA QA Leaderboard!**

            This leaderboard evaluates Telugu short-answer question-answering models using a curated 1000-sample benchmark.

            📎 **Download Evaluation Dataset**: [samples_1000.json](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json)

            🔒 **Open Leaderboard**: All submissions are publicly viewable but cannot be edited or deleted to ensure leaderboard integrity.
            """)

    @staticmethod
    def render_status_info(ref_lookup, mongodb_available):
        if ref_lookup:
            st.sidebar.success(f"📚 {len(ref_lookup)} reference samples loaded")
        else:
            st.sidebar.error("⚠️ No reference data available")

    @staticmethod
    def render_submission_form():
        st.sidebar.header("📥 Submit Your Model Output")
        model_name = st.sidebar.text_input("Model Name (required)")
        author_name = st.sidebar.text_input("Your Name or Alias (required)")
        version_tag = st.sidebar.text_input("Version Tag (optional)", placeholder="v1.0")
        notes = st.sidebar.text_area("Notes (optional)", placeholder="Brief description of your model")
        uploaded_file = st.sidebar.file_uploader("Upload result JSON file", type="json")

        return {
            "model_name": model_name,
            "author_name": author_name,
            "version_tag": version_tag,
            "notes": notes,
            "uploaded_file": uploaded_file
        }

    @staticmethod
    def render_leaderboard(leaderboard_rows):
        st.subheader("🏆 Leaderboard")
        show_advanced = st.toggle("Show Advanced Metrics", value=False)

        if leaderboard_rows:
            leaderboard_df = pd.DataFrame(leaderboard_rows)
            st.dataframe(leaderboard_df, use_container_width=True)
        else:
            st.info("No submissions yet.")

    @staticmethod
    def render_sample_explorer(all_data, ref_lookup):
        st.subheader("🔍 Sample Explorer")
        st.markdown("""
        ℹ️ **How to Use:**
        - Choose a submission from the dropdown (shows model name, author, version, and timestamp).
        - Filter samples by type: hallucinated, faithful_correct, etc.
        - Use the slider to browse examples.
        - This is a read-only view of all submissions in the leaderboard.
        """)

        if all_data:
            st.success(f"✅ Loaded {len(all_data)} submissions for exploration")

            selected_submission = st.selectbox(
                "Choose a submission to explore", 
                ["None"] + list(all_data.keys()),
                help="Select a model submission to explore its predictions"
            )

            if selected_submission != "None":
                UIComponents._display_submission_details(selected_submission, all_data, ref_lookup)
        else:
            st.info("No submissions available yet. Submit your first model to see results here!")

    @staticmethod
    def _display_submission_details(selected_submission: str, all_data: Dict, ref_lookup: Dict):
        """Display details for a selected submission"""
        try:
            submission_info = all_data[selected_submission]
            df = submission_info["data"]
            metadata = submission_info["metadata"]

            st.info(f"📊 Processing {len(df)} samples from selected submission...")
            
            # Debug info
            if len(ref_lookup) > 0:
                st.success(f"✅ Reference lookup contains {len(ref_lookup)} entries")
            else:
                st.error("❌ Reference lookup is empty!")

            # Calculate context coverage
            context_coverage = UIComponents._calculate_context_coverage(df, ref_lookup)

            # Show submission metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("EM Score", f"{metadata.get('metrics', {}).get('em', 0):.1f}%")
            with col3:
                st.metric("F1 Score", f"{metadata.get('metrics', {}).get('f1', 0):.1f}%")
            with col4:
                st.metric("Context Coverage", f"{context_coverage:.1f}%")

            # Add notes if available
            if metadata.get("notes"):
                st.info(f"**Notes**: {metadata['notes']}")

            # Filter and display samples
            tag_filter = st.selectbox("Breakdown Filter", ["all"] + sorted(df["breakdown"].unique()))
            if tag_filter != "all":
                df = df[df["breakdown"] == tag_filter]

            if df.empty:
                st.warning("No samples for this filter.")
            else:
                i = st.slider("Sample Index", 0, len(df) - 1, 0)
                row = df.iloc[i]

                # Get context for sample
                context_text = UIComponents._get_context_for_sample(
                    row["content_id"], row["qa_index"], ref_lookup
                )

                # Display sample details
                st.markdown(f"**Q{row['qa_index']}**: {row['question']}")
                st.markdown(f"**Gold Answer**: {row['gold_answer']}")
                st.markdown(f"**Prediction**: {row['prediction']}")
                st.markdown(f"**F1**: {row['f1_score']:.2f} | EM: {row['exact_match']} | Hallucinated: {row['hallucinated']}")
                st.markdown(f"**Type**: {row['breakdown']}")
                
                # Display context in an expandable section
                with st.expander("📖 View Context", expanded=True):
                    if context_text and context_text != "[Context not available]":
                        st.text_area("Context Text", value=context_text, height=200, disabled=True)
                    else:
                        st.warning(f"⚠️ Context not found for content_id: {row['content_id']}, qa_index: {row['qa_index']}")
                        st.info("Available reference data keys (first 10):")
                        sample_keys = list(ref_lookup.keys())[:10]
                        st.write(sample_keys)

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
            context = ref_lookup.get(key, None)
            
            if context is None:
                return "[Context not available]"
            elif context.strip() == "":
                return "[Empty context]"
            else:
                return context.strip()
        except Exception as e:
            return f"[Error loading context: {e}]"