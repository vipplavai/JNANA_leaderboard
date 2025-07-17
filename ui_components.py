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

        # Simple, clean styling
        st.markdown("""
        <style>
        .main { padding-top: 2rem; }
        .stDeployButton { display: none; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Simple header
        st.title("🏆 JNANA Telugu QA Leaderboard")
        st.markdown("Evaluate your Telugu question-answering models on our curated benchmark.")

        # Simple download link
        st.markdown("""
        **📥 Get Started:** [Download samples_1000.json](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json) 
        and submit your model's predictions below.
        """)
        st.divider()

    @staticmethod
    def render_submission_form():
        st.sidebar.header("📤 Submit Results")

        with st.sidebar.form("submission_form"):
            model_name = st.text_input("Model Name*", placeholder="e.g., GPT-4, BERT-Telugu")
            author_name = st.text_input("Your Name*", placeholder="Your name or team")
            version_tag = st.text_input("Version", placeholder="e.g., v1.0 (optional)")
            notes = st.text_area("Notes", placeholder="Brief description (optional)")

            uploaded_file = st.file_uploader("Upload JSON Results", type="json")
            submitted = st.form_submit_button("Submit", use_container_width=True)

            if submitted:
                return {
                    "model_name": model_name,
                    "author_name": author_name,
                    "version_tag": version_tag,
                    "notes": notes,
                    "uploaded_file": uploaded_file,
                    "submitted": True
                }

        return {
            "model_name": "",
            "author_name": "",
            "version_tag": "",
            "notes": "",
            "uploaded_file": None,
            "submitted": False
        }

    @staticmethod
    def render_leaderboard(leaderboard_rows):
        st.header("🏆 Leaderboard")

        if leaderboard_rows:
            df = pd.DataFrame(leaderboard_rows)

            # Show main metrics
            main_cols = ["Model", "Author", "EM (%)", "F1 (%)", "Samples"]
            if all(col in df.columns for col in main_cols):
                st.dataframe(
                    df[main_cols], 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Show all metrics in expander
            with st.expander("View All Metrics"):
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No submissions yet. Be the first to submit!")

    @staticmethod
    def render_sample_explorer(all_data, ref_lookup):
        st.header("🔍 Sample Explorer")

        if all_data:
            selected_submission = st.selectbox(
                "Choose submission to explore", 
                [""] + list(all_data.keys())
            )

            if selected_submission:
                UIComponents._display_submission_details(selected_submission, all_data, ref_lookup)
        else:
            st.info("No submissions to explore yet.")

    @staticmethod
    def _display_submission_details(selected_submission: str, all_data: Dict, ref_lookup: Dict):
        try:
            submission_info = all_data[selected_submission]
            df = submission_info["data"]
            metadata = submission_info["metadata"]

            # Simple metrics display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Samples", len(df))
            with col2:
                em_score = metadata.get('metrics', {}).get('em', 0)
                st.metric("EM Score", f"{em_score:.1f}%")
            with col3:
                f1_score = metadata.get('metrics', {}).get('f1', 0)
                st.metric("F1 Score", f"{f1_score:.1f}%")
            with col4:
                context_coverage = UIComponents._calculate_context_coverage(df, ref_lookup)
                st.metric("Context Coverage", f"{context_coverage:.1f}%")

            # Sample filtering
            st.subheader("Browse Samples")
            tag_filter = st.selectbox(
                "Filter by type", 
                ["all"] + sorted(df["breakdown"].unique())
            )

            # Apply filter
            if tag_filter != "all":
                df = df[df["breakdown"] == tag_filter]

            if df.empty:
                st.warning("No samples found for this filter.")
            else:
                # Sample navigation
                sample_idx = st.slider("Sample", 0, len(df) - 1, 0)
                row = df.iloc[sample_idx]

                # Simple sample display
                UIComponents._render_sample_card(row, ref_lookup)

        except Exception as e:
            st.error(f"Error loading submission: {e}")

    @staticmethod
    def _render_sample_card(row, ref_lookup):
        # Question and answers
        st.markdown(f"**Question:** {row['question']}")
        st.markdown(f"**Expected:** {row['gold_answer']}")
        st.markdown(f"**Predicted:** {row['prediction']}")

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("F1", f"{row['f1_score']:.3f}")
        with col2:
            st.metric("Exact Match", "✅" if row['exact_match'] else "❌")
        with col3:
            st.metric("Type", row['breakdown'].replace('_', ' ').title())

        # Context
        if st.checkbox("Show Context"):
            context_text = UIComponents._get_context_for_sample(
                row["content_id"], row["qa_index"], ref_lookup
            )
            if context_text and context_text not in ["[Context not available]", "[Empty context]"]:
                st.text_area("Context", context_text, height=200)
            else:
                st.warning("Context not available for this sample")

    @staticmethod
    def _calculate_context_coverage(df: pd.DataFrame, ref_lookup: Dict) -> float:
        if len(df) == 0:
            return 0.0
        context_available = sum(1 for _, row in df.iterrows() 
                              if (int(row["content_id"]), int(row["qa_index"])) in ref_lookup)
        return (context_available / len(df)) * 100

    @staticmethod
    def _get_context_for_sample(content_id: int, qa_index: int, ref_lookup: Dict) -> str:
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
            return f"[Error: {e}]"