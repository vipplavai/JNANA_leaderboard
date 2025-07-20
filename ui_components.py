
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

        # Optimized styling - removed unused styles
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
        
        </style>
        """, unsafe_allow_html=True)

        # Initialize session state once
        UIComponents._init_session_state()
        
        st.title("🏆 JNANA Telugu QA Leaderboard")
        st.markdown("**Evaluate your Telugu question-answering models on our curated 1000-sample benchmark**")

        # GitHub link for guidelines
        st.info("📖 **Scroll to the bottom of the page for submitting,Complete submission guidelines and documentation available on [GitHub](https://github.com/vipplavai/JNANA_leaderboard)**")

        st.markdown("---")

    @staticmethod
    def _init_session_state():
        """Initialize all session state variables in one place"""
        pass

    

    @staticmethod
    def render_submission_form():
        # Submission form in main area with clean layout
        st.header("📤 Submit Your Results")
        
        # Instructions in expandable section
        with st.expander("📋 Submission Instructions", expanded=False):
            st.markdown("""
            **Steps to Submit:**
            1. Download [samples_1000.json](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json)
            2. Run your model on the dataset
            3. Format results as JSON array
            4. Upload using the form below
            """)

        # Submission form in main area
        with st.form("submission_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("🤖 Model Name*", placeholder="e.g., GPT-4, Gemini-Pro")
                author_name = st.text_input("👤 Your Name*", placeholder="e.g., John Doe")
            
            with col2:
                version_tag = st.text_input("🏷️ Version", placeholder="e.g., v1.0")
                uploaded_file = st.file_uploader("📁 Upload Results JSON", type="json")
            
            notes = st.text_area("📝 Notes", placeholder="Brief description of your model")
            
            col_submit1, col_submit2, col_submit3 = st.columns([2, 1, 2])
            with col_submit2:
                submitted = st.form_submit_button("🚀 Submit to Leaderboard", use_container_width=True)

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
        # Simplified metrics info
        st.info("📊 **Understanding Metrics**: Detailed explanations of all metrics are available in our documentation.")

    @staticmethod
    def render_leaderboard(leaderboard_rows):
        st.header("🏆 Current Leaderboard")
        UIComponents.render_metrics_explanation()

        if leaderboard_rows:
            df = pd.DataFrame(leaderboard_rows)

            # Display comprehensive metrics
            st.subheader("📊 Detailed Performance Metrics")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Optimized top performers summary
            if len(df) > 0:
                UIComponents._render_top_performers(df)

        else:
            st.info("🔄 No submissions yet. Be the first to submit your model results!")

    @staticmethod
    def _render_top_performers(df):
        """Optimized top performers display"""
        st.subheader("🥇 Top Performers")
        
        # Pre-calculate top performers to avoid repeated operations
        top_performers = {
            'em': df.loc[df['EM (%)'].idxmax()],
            'f1': df.loc[df['F1 (%)'].idxmax()],
            'faithful': df.loc[df['Faithful Correct (%)'].idxmax()]
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_em = top_performers['em']
            st.metric("🎯 Best EM Score", f"{top_em['EM (%)']}%", f"{top_em['Model']}")

        with col2:
            top_f1 = top_performers['f1']
            st.metric("🔍 Best F1 Score", f"{top_f1['F1 (%)']}%", f"{top_f1['Model']}")

        with col3:
            top_faithful = top_performers['faithful']
            st.metric("✅ Most Faithful", f"{top_faithful['Faithful Correct (%)']}%", f"{top_faithful['Model']}")

    @staticmethod
    def render_sample_explorer(all_data, ref_lookup):
        st.header("🔍 Sample Explorer")

        # Simplified info box
        st.info("ℹ️ **Sample Explorer**: Browse through individual model predictions and analyze performance patterns.")

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
        metrics = metadata.get('metrics', {})

        # Optimized metrics display
        UIComponents._render_core_metrics(metrics, len(df))
        UIComponents._render_detailed_metrics(metrics)

        st.markdown("---")

        # Sample filtering and browsing
        st.subheader("🔍 Browse Individual Samples")
        filtered_df = UIComponents._render_sample_filters(df)

        if not filtered_df.empty:
            UIComponents._render_sample_details(filtered_df, ref_lookup)
        else:
            st.info("🔍 No samples found for the selected filter.")

    @staticmethod
    def _render_core_metrics(metrics: Dict, total_samples: int):
        """Render core metrics in a clean layout"""
        st.subheader("📈 Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Samples", metrics.get('total', total_samples))
        with col2:
            st.metric("🎯 EM Score", f"{metrics.get('em', 0):.1f}%")
        with col3:
            st.metric("🔍 F1 Score", f"{metrics.get('f1', 0):.1f}%")
        with col4:
            st.metric("✅ Faithful Correct", f"{metrics.get('faithful_correct', 0):.1f}%")

    @staticmethod
    def _render_detailed_metrics(metrics: Dict):
        """Render detailed metrics in expandable section"""
        with st.expander("📊 Detailed Metrics Breakdown"):
            # Group related metrics for better organization
            metric_groups = [
                [
                    ("📝 Answered", "answered"),
                    ("🚫 Hallucinated", "hallucinated"),
                    ("📭 Empty", "empty")
                ],
                [
                    ("❌ Faithful Incorrect", "faithful_incorrect"),
                    ("🎭 FAA Score", "faa"),
                    ("📏 F1-EM Gap", "f1_em_gap")
                ],
                [
                    ("⚠️ Overconfident EM", "overconfident_em"),
                    ("💪 Robust Answer Rate", "robust_answer_rate"),
                    ("📊 Avg Answer Length", "avg_answer_length")
                ]
            ]
            
            cols = st.columns(3)
            for i, group in enumerate(metric_groups):
                with cols[i]:
                    for label, key in group:
                        value = metrics.get(key, 0)
                        unit = " words" if key == "avg_answer_length" else "%"
                        st.metric(label, f"{value:.1f}{unit}")

    @staticmethod
    def _render_sample_filters(df):
        """Render sample filtering controls and return filtered DataFrame"""
        col1, col2 = st.columns([2, 1])
        with col1:
            tag_filter = st.selectbox("🏷️ Filter by type", ["all"] + sorted(df["breakdown"].unique()))
        
        # Apply filter
        filtered_df = df if tag_filter == "all" else df[df["breakdown"] == tag_filter]
        
        with col2:
            if tag_filter != "all":
                st.metric("Filtered Count", len(filtered_df))
        
        return filtered_df

    @staticmethod
    def _render_sample_details(df, ref_lookup: Dict):
        """Render individual sample details"""
        # Sample navigation with dropdown (much clearer than slider)
        sample_options = [f"Sample {i+1} (ID: {row['content_id']}-{row['qa_index']})" for i, row in df.iterrows()]
        selected_sample = st.selectbox("📍 Choose Sample to View", sample_options, key="sample_selector")
        sample_idx = sample_options.index(selected_sample) if selected_sample else 0
        row = df.iloc[sample_idx]

        # Sample display
        st.markdown("### 📝 Sample Details")

        # Question and answers
        st.markdown(f"**❓ Question:** {row['question']}")
        st.markdown(f"**🎯 Expected Answer:** {row['gold_answer']}")
        st.markdown(f"**🤖 Model Prediction:** {row['prediction']}")

        # Metrics and type
        UIComponents._render_sample_metrics(row)

        # Context display
        if st.checkbox("📖 Show Context", key=f"context_{sample_idx}"):
            context = UIComponents._get_context(row["content_id"], row["qa_index"], ref_lookup)
            if context and context != "[Context not available]":
                st.text_area("📄 Source Context", context, height=200, key=f"context_text_{sample_idx}")
            else:
                st.warning("⚠️ Context not available for this sample")

    @staticmethod
    def _render_sample_metrics(row):
        """Render individual sample metrics"""
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

    @staticmethod
    def _get_context(content_id: int, qa_index: int, ref_lookup: Dict) -> str:
        """Get context with improved error handling"""
        try:
            key = (int(content_id), int(qa_index))
            return ref_lookup.get(key, "[Context not available]")
        except (ValueError, TypeError):
            return "[Context not available]"
