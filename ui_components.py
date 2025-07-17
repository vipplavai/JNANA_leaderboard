"""
UI Components for the JNANA QA Leaderboard
"""
import streamlit as st
import pandas as pd
from typing import Dict, List

class UIComponents:
    @staticmethod
    def render_header():
        st.set_page_config(
            page_title="JNANA QA Leaderboard", 
            layout="wide",
            page_icon="🏆"
        )

        # Minimal styling
        st.markdown("""
        <style>
        .stDeployButton { display: none; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)

        # Simple header
        st.title("🏆 JNANA Telugu QA Leaderboard")
        st.markdown("📥 [Download samples_1000.json](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json) and submit your results")
        st.divider()

    @staticmethod
    def render_submission_form():
        st.sidebar.header("Submit Results")

        with st.sidebar.form("submission_form"):
            model_name = st.text_input("Model Name*")
            author_name = st.text_input("Your Name*")
            version_tag = st.text_input("Version (optional)")
            notes = st.text_area("Notes (optional)")
            uploaded_file = st.file_uploader("Upload JSON", type="json")
            submitted = st.form_submit_button("Submit")

            if submitted:
                return {
                    "model_name": model_name,
                    "author_name": author_name,
                    "version_tag": version_tag,
                    "notes": notes,
                    "uploaded_file": uploaded_file,
                    "submitted": True
                }

        return {"submitted": False}

    @staticmethod
    def render_leaderboard(leaderboard_rows):
        st.header("Leaderboard")

        if leaderboard_rows:
            df = pd.DataFrame(leaderboard_rows)
            # Show only key metrics
            key_cols = ["Model", "Author", "EM (%)", "F1 (%)"]
            if all(col in df.columns for col in key_cols):
                st.dataframe(df[key_cols], use_container_width=True, hide_index=True)
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No submissions yet")

    @staticmethod
    def render_sample_explorer(all_data, ref_lookup):
        st.header("Sample Explorer")

        if all_data:
            selected = st.selectbox("Choose submission", [""] + list(all_data.keys()))

            if selected:
                UIComponents._show_samples(selected, all_data, ref_lookup)
        else:
            st.info("No submissions to explore")

    @staticmethod
    def _show_samples(selected_submission: str, all_data: Dict, ref_lookup: Dict):
        submission_info = all_data[selected_submission]
        df = submission_info["data"]
        metadata = submission_info["metadata"]

        # Simple stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(df))
        with col2:
            em_score = metadata.get('metrics', {}).get('em', 0)
            st.metric("EM", f"{em_score:.1f}%")
        with col3:
            f1_score = metadata.get('metrics', {}).get('f1', 0)
            st.metric("F1", f"{f1_score:.1f}%")

        # Filter and browse
        tag_filter = st.selectbox("Filter", ["all"] + sorted(df["breakdown"].unique()))

        if tag_filter != "all":
            df = df[df["breakdown"] == tag_filter]

        if not df.empty:
            sample_idx = st.slider("Sample", 0, len(df) - 1, 0)
            row = df.iloc[sample_idx]

            # Show sample
            st.markdown(f"**Q:** {row['question']}")
            st.markdown(f"**Expected:** {row['gold_answer']}")
            st.markdown(f"**Predicted:** {row['prediction']}")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"F1: {row['f1_score']:.3f}")
            with col2:
                st.write(f"EM: {'✅' if row['exact_match'] else '❌'}")

            # Show context if available
            if st.checkbox("Show Context"):
                context = UIComponents._get_context(row["content_id"], row["qa_index"], ref_lookup)
                if context and context != "[Context not available]":
                    st.text_area("Context", context, height=150)
                else:
                    st.warning("Context not available")

    @staticmethod
    def _get_context(content_id: int, qa_index: int, ref_lookup: Dict) -> str:
        try:
            key = (int(content_id), int(qa_index))
            return ref_lookup.get(key, "[Context not available]")
        except:
            return "[Context not available]"