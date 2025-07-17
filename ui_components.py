
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
            initial_sidebar_state="expanded",
            page_icon="📊"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        
        .sample-card {
            background: white;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .context-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        .status-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
            margin: 0.25rem;
        }
        
        .badge-correct {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-incorrect {
            background: #f8d7da;
            color: #721c24;
        }
        
        .badge-hallucinated {
            background: #fff3cd;
            color: #856404;
        }
        
        .badge-empty {
            background: #e2e3e5;
            color: #383d41;
        }
        
        .download-button {
            background: #28a745;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            margin: 0.5rem;
        }
        
        .sidebar .stSelectbox > div > div {
            background: #f8f9fa;
        }
        
        .stMetric > div {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header with gradient background
        st.markdown("""
        <div class="main-header">
            <h1>📊 JNANA Telugu QA Leaderboard</h1>
            <p style="font-size: 1.2rem; margin-top: 1rem;">
                Evaluating Telugu Question-Answering Models with a Curated 1000-Sample Benchmark
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Info section with better formatting
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px; margin-bottom: 2rem;">
                <h3>🚀 Quick Start</h3>
                <p>
                    <a href="https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json" 
                       class="download-button" target="_blank">
                        📎 Download Evaluation Dataset
                    </a>
                </p>
                <p><small>🔒 <strong>Open Leaderboard</strong>: All submissions are publicly viewable but cannot be edited or deleted to ensure leaderboard integrity.</small></p>
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def render_status_info(ref_lookup, mongodb_available):
        st.sidebar.markdown("### 📊 System Status")
        
        if ref_lookup:
            st.sidebar.success(f"📚 {len(ref_lookup)} reference samples loaded")
        else:
            st.sidebar.error("⚠️ No reference data available")
        
        if mongodb_available:
            st.sidebar.success("🔗 MongoDB connected")
        else:
            st.sidebar.warning("💾 Local mode (no persistence)")

    @staticmethod
    def render_submission_form():
        st.sidebar.markdown("### 📥 Submit Your Model Output")
        
        with st.sidebar.form("submission_form"):
            model_name = st.text_input("🤖 Model Name", placeholder="e.g., GPT-4, BERT-Telugu")
            author_name = st.text_input("👤 Your Name or Alias", placeholder="e.g., John Doe")
            version_tag = st.text_input("🏷️ Version Tag", placeholder="v1.0")
            notes = st.text_area("📝 Notes", placeholder="Brief description of your model approach")
            uploaded_file = st.file_uploader(
                "📁 Upload result JSON file", 
                type="json",
                help="Upload your model predictions in the required JSON format"
            )
            
            submitted = st.form_submit_button("🚀 Submit Results", use_container_width=True)
            
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
        st.markdown("## 🏆 Leaderboard")
        
        if leaderboard_rows:
            # Add tabs for different views
            tab1, tab2 = st.tabs(["📊 Main Metrics", "🔍 Detailed Analysis"])
            
            with tab1:
                # Main metrics view
                df = pd.DataFrame(leaderboard_rows)
                main_cols = ["Model", "Author", "Version", "EM (%)", "F1 (%)", "Answered (%)", "Hallucinated (%)"]
                if all(col in df.columns for col in main_cols):
                    st.dataframe(
                        df[main_cols].style.format({
                            "EM (%)": "{:.1f}",
                            "F1 (%)": "{:.1f}",
                            "Answered (%)": "{:.1f}",
                            "Hallucinated (%)": "{:.1f}"
                        }),
                        use_container_width=True
                    )
                else:
                    st.dataframe(df, use_container_width=True)
            
            with tab2:
                # Detailed analysis view
                st.dataframe(
                    df.style.format({
                        col: "{:.1f}" for col in df.columns if "(%)" in col
                    }),
                    use_container_width=True
                )
        else:
            st.info("🎯 No submissions yet. Be the first to submit your model!")

    @staticmethod
    def render_sample_explorer(all_data, ref_lookup):
        st.markdown("## 🔍 Sample Explorer")
        
        # Instructions with better formatting
        with st.expander("ℹ️ How to Use Sample Explorer", expanded=False):
            st.markdown("""
            **Steps:**
            1. **Choose a submission** from the dropdown (shows model name, author, version, and timestamp)
            2. **Filter samples** by prediction type: hallucinated, faithful_correct, etc.
            3. **Browse examples** using the slider to see different QA pairs
            4. **View context** to understand the source material for each question
            
            *This is a read-only view of all submissions in the leaderboard.*
            """)

        if all_data:
            st.success(f"✅ Found {len(all_data)} submissions for exploration")

            selected_submission = st.selectbox(
                "🎯 Choose a submission to explore", 
                ["None"] + list(all_data.keys()),
                help="Select a model submission to explore its predictions"
            )

            if selected_submission != "None":
                UIComponents._display_submission_details(selected_submission, all_data, ref_lookup)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px;">
                <h3>🎯 No submissions available yet</h3>
                <p>Submit your first model to see results here!</p>
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def _display_submission_details(selected_submission: str, all_data: Dict, ref_lookup: Dict):
        """Display details for a selected submission with enhanced styling"""
        try:
            submission_info = all_data[selected_submission]
            df = submission_info["data"]
            metadata = submission_info["metadata"]

            # Header with submission info
            st.markdown("### 📊 Submission Overview")
            
            # Calculate context coverage
            context_coverage = UIComponents._calculate_context_coverage(df, ref_lookup)

            # Metrics cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📝 Total Samples", len(df))
            with col2:
                em_score = metadata.get('metrics', {}).get('em', 0)
                st.metric("🎯 EM Score", f"{em_score:.1f}%")
            with col3:
                f1_score = metadata.get('metrics', {}).get('f1', 0)
                st.metric("📈 F1 Score", f"{f1_score:.1f}%")
            with col4:
                st.metric("🔍 Context Coverage", f"{context_coverage:.1f}%")

            # Additional metrics in expandable section
            with st.expander("📊 Detailed Metrics", expanded=False):
                metrics = metadata.get("metrics", {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Answered", f"{metrics.get('answered', 0):.1f}%")
                    st.metric("🚫 Hallucinated", f"{metrics.get('hallucinated', 0):.1f}%")
                with col2:
                    st.metric("✅ Faithful Correct", f"{metrics.get('faithful_correct', 0):.1f}%")
                    st.metric("❌ Faithful Incorrect", f"{metrics.get('faithful_incorrect', 0):.1f}%")
                with col3:
                    st.metric("⚪ Empty", f"{metrics.get('empty', 0):.1f}%")

            # Notes section
            if metadata.get("notes"):
                st.markdown(f"""
                <div class="metric-card">
                    <strong>📝 Notes:</strong> {metadata['notes']}
                </div>
                """, unsafe_allow_html=True)

            # Sample filtering and navigation
            st.markdown("### 🔍 Sample Analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                tag_filter = st.selectbox(
                    "Filter by prediction type", 
                    ["all"] + sorted(df["breakdown"].unique()),
                    help="Filter samples by their prediction category"
                )
            with col2:
                if tag_filter != "all":
                    filtered_count = len(df[df["breakdown"] == tag_filter])
                    st.metric("Filtered Samples", filtered_count)

            # Apply filter
            if tag_filter != "all":
                df = df[df["breakdown"] == tag_filter]

            if df.empty:
                st.warning("🔍 No samples found for this filter.")
            else:
                # Sample navigation
                sample_idx = st.slider(
                    "Navigate through samples", 
                    0, len(df) - 1, 0,
                    help="Use the slider to browse through different samples"
                )
                row = df.iloc[sample_idx]

                # Sample display card
                UIComponents._render_sample_card(row, ref_lookup, sample_idx + 1, len(df))

        except Exception as e:
            st.error(f"❌ Error displaying submission details: {e}")

    @staticmethod
    def _render_sample_card(row, ref_lookup, current_idx, total_samples):
        """Render individual sample card with enhanced styling"""
        # Get context for sample
        context_text = UIComponents._get_context_for_sample(
            row["content_id"], row["qa_index"], ref_lookup
        )

        # Sample header
        st.markdown(f"""
        <div class="sample-card">
            <h4>Sample {current_idx} of {total_samples}</h4>
        </div>
        """, unsafe_allow_html=True)

        # Question and answers
        st.markdown(f"**❓ Question:** {row['question']}")
        st.markdown(f"**✅ Gold Answer:** {row['gold_answer']}")
        st.markdown(f"**🤖 Prediction:** {row['prediction']}")
        
        # Metrics and status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("F1 Score", f"{row['f1_score']:.3f}")
        with col2:
            em_status = "✅ Yes" if row['exact_match'] else "❌ No"
            st.metric("Exact Match", em_status)
        with col3:
            hall_status = "⚠️ Yes" if row['hallucinated'] else "✅ No"
            st.metric("Hallucinated", hall_status)
        with col4:
            # Type badge
            type_color = {
                "faithful_correct": "badge-correct",
                "faithful_incorrect": "badge-incorrect", 
                "hallucinated": "badge-hallucinated",
                "empty": "badge-empty"
            }.get(row['breakdown'], "badge-empty")
            
            st.markdown(f"""
            <div class="status-badge {type_color}">
                {row['breakdown'].replace('_', ' ').title()}
            </div>
            """, unsafe_allow_html=True)

        # Context display
        with st.expander("📖 View Context", expanded=True):
            if context_text and context_text not in ["[Context not available]", "[Empty context]"]:
                st.markdown(f"""
                <div class="context-box">
                    {context_text}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Context not found for content_id: {row['content_id']}, qa_index: {row['qa_index']}")
                
                # Debug information
                with st.expander("🔧 Debug Information", expanded=False):
                    lookup_key = (int(row['content_id']), int(row['qa_index']))
                    st.code(f"Looking for key: {lookup_key}")
                    
                    if len(ref_lookup) > 0:
                        st.info("Sample of available reference data keys:")
                        sample_keys = list(ref_lookup.keys())[:10]
                        for key in sample_keys:
                            st.code(f"  {key}")
                    else:
                        st.error("❌ Reference lookup dictionary is empty!")

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
