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

        # Clean styling
        st.markdown("""
        <style>
        .stDeployButton { display: none; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin: 10px 0;
        }
        .explanation-box {
            background: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #0066cc;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header with instructions
        st.title("🏆 JNANA Telugu QA Leaderboard")
        st.markdown("**Evaluate your Telugu question-answering models on our curated 1000-sample benchmark**")

        # GitHub link for guidelines
        st.info("📖 **Complete submission guidelines and documentation available on [GitHub](https://github.com/vipplavai/JNANA_leaderboard)**")

        st.markdown("---")

    @staticmethod
    def render_submission_form():
        st.sidebar.header("📤 Submit Your Results")

        # Instructions in sidebar
        st.sidebar.markdown("""
        **Steps:**
        1. Download [samples_1000.json](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json)
        2. Run your model on the dataset
        3. Format results as JSON array
        4. Upload below
        """)

        st.sidebar.markdown("---")

        with st.sidebar.form("submission_form"):
            model_name = st.text_input("🤖 Model Name*", placeholder="e.g., GPT-4, Gemini-Pro")
            author_name = st.text_input("👤 Your Name*", placeholder="e.g., John Doe")
            version_tag = st.text_input("🏷️ Version", placeholder="e.g., v1.0")
            notes = st.text_area("📝 Notes", placeholder="Brief description of your model")
            uploaded_file = st.file_uploader("📁 Upload Results JSON", type="json")
            submitted = st.form_submit_button("🚀 Submit to Leaderboard")

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
    def render_metrics_explanation():
        # Direct to GitHub for metrics info
        st.info("📊 **Understanding Metrics**: Visit our [GitHub Documentation](https://github.com/vipplavai/JNANA_leaderboard) for detailed metric explanations.")

    @staticmethod
    def render_leaderboard(leaderboard_rows):
        st.header("🏆 Current Leaderboard")
        UIComponents.render_metrics_explanation()

        if leaderboard_rows:
            df = pd.DataFrame(leaderboard_rows)

            # Display comprehensive metrics
            st.subheader("📊 Detailed Performance Metrics")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Top performers summary
            if len(df) > 0:
                st.subheader("🥇 Top Performers")
                col1, col2, col3 = st.columns(3)

                with col1:
                    top_em = df.loc[df['EM (%)'].idxmax()]
                    st.metric("🎯 Best EM Score", f"{top_em['EM (%)']}%", f"{top_em['Model']}")

                with col2:
                    top_f1 = df.loc[df['F1 (%)'].idxmax()]
                    st.metric("🔍 Best F1 Score", f"{top_f1['F1 (%)']}%", f"{top_f1['Model']}")

                with col3:
                    top_faithful = df.loc[df['Faithful Correct (%)'].idxmax()]
                    st.metric("✅ Most Faithful", f"{top_faithful['Faithful Correct (%)']}%", f"{top_faithful['Model']}")

        else:
            st.info("🔄 No submissions yet. Be the first to submit your model results!")

    @staticmethod
    def render_sample_explorer(all_data, ref_lookup):
        st.header("🔍 Sample Explorer")

        # Simple info box for GitHub
        st.info("ℹ️ **Sample Explorer Guide**: For detailed usage instructions, visit our [GitHub Documentation](https://github.com/vipplavai/JNANA_leaderboard)")

        if all_data:
            # Dynamic submission selector
            submission_options = [""] + [f"📊 {key}" for key in all_data.keys()]
            selected_raw = st.selectbox("Choose a submission to explore", submission_options)

            if selected_raw:
                selected = selected_raw.replace("📊 ", "")
                UIComponents._show_samples(selected, all_data, ref_lookup)
        else:
            st.info("📥 No submissions available for exploration. Submit your model results to start exploring!")

    @staticmethod
    def _show_samples(selected_submission: str, all_data: Dict, ref_lookup: Dict):
        submission_info = all_data[selected_submission]
        df = submission_info["data"]
        metadata = submission_info["metadata"]

        # Enhanced metrics display
        st.subheader(f"📈 Performance Overview")

        # Core metrics in prominent display
        col1, col2, col3, col4 = st.columns(4)
        metrics = metadata.get('metrics', {})

        with col1:
            st.metric("📊 Total Samples", metrics.get('total', len(df)))
        with col2:
            st.metric("🎯 EM Score", f"{metrics.get('em', 0):.1f}%")
        with col3:
            st.metric("🔍 F1 Score", f"{metrics.get('f1', 0):.1f}%")
        with col4:
            st.metric("✅ Faithful Correct", f"{metrics.get('faithful_correct', 0):.1f}%")

        # Additional metrics in expandable section
        with st.expander("📊 Detailed Metrics Breakdown"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📝 Answered", f"{metrics.get('answered', 0):.1f}%")
                st.metric("🚫 Hallucinated", f"{metrics.get('hallucinated', 0):.1f}%")
                st.metric("📭 Empty", f"{metrics.get('empty', 0):.1f}%")

            with col2:
                st.metric("❌ Faithful Incorrect", f"{metrics.get('faithful_incorrect', 0):.1f}%")
                st.metric("🎭 FAA Score", f"{metrics.get('faa', 0):.1f}%")
                st.metric("📏 F1-EM Gap", f"{metrics.get('f1_em_gap', 0):.1f}%")

            with col3:
                st.metric("⚠️ Overconfident EM", f"{metrics.get('overconfident_em', 0):.1f}%")
                st.metric("💪 Robust Answer Rate", f"{metrics.get('robust_answer_rate', 0):.1f}%")
                st.metric("📊 Avg Answer Length", f"{metrics.get('avg_answer_length', 0):.1f} words")

        st.markdown("---")

        # Sample filtering and browsing
        st.subheader("🔍 Browse Individual Samples")

        col1, col2 = st.columns([2, 1])
        with col1:
            tag_filter = st.selectbox("🏷️ Filter by type", ["all"] + sorted(df["breakdown"].unique()))
        with col2:
            if tag_filter != "all":
                filtered_count = len(df[df["breakdown"] == tag_filter])
                st.metric("Filtered Count", filtered_count)

        # Apply filter
        if tag_filter != "all":
            df = df[df["breakdown"] == tag_filter]

        if not df.empty:
            # Sample navigation
            sample_idx = st.slider("📍 Sample Index", 0, len(df) - 1, 0)
            row = df.iloc[sample_idx]

            # Sample display
            st.markdown("### 📝 Sample Details")

            # Question and answers
            st.markdown(f"**❓ Question:** {row['question']}")
            st.markdown(f"**🎯 Expected Answer:** {row['gold_answer']}")
            st.markdown(f"**🤖 Model Prediction:** {row['prediction']}")

            # Metrics and type
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔍 F1 Score", f"{row['f1_score']:.3f}")
            with col2:
                em_symbol = "✅" if row['exact_match'] else "❌"
                st.metric("🎯 Exact Match", em_symbol)
            with col3:
                type_symbols = {
                    "faithful_correct": "✅",
                    "faithful_incorrect": "❌", 
                    "hallucinated": "🚫",
                    "empty": "📭"
                }
                symbol = type_symbols.get(row.get('type', 'unknown'), "❓")
                st.metric("🏷️ Type", f"{symbol} {row.get('type', 'unknown')}")

            # Context display
            if st.checkbox("📖 Show Context", key=f"context_{sample_idx}"):
                context = UIComponents._get_context(row["content_id"], row["qa_index"], ref_lookup)
                if context and context != "[Context not available]":
                    st.text_area("📄 Source Context", context, height=200, key=f"context_text_{sample_idx}")
                else:
                    st.warning("⚠️ Context not available for this sample")

        else:
            st.info(f"🔍 No samples found for filter: {tag_filter}")

    @staticmethod
    def _get_context(content_id: int, qa_index: int, ref_lookup: Dict) -> str:
        try:
            key = (int(content_id), int(qa_index))
            return ref_lookup.get(key, "[Context not available]")
        except:
            return "[Context not available]"