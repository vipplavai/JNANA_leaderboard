
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
            page_icon="🏆"
        )
        
        # Elegant, minimal CSS
        st.markdown("""
        <style>
        /* Main styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        /* Cards and components */
        .elegant-card {
            background: white;
            border: none;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin: 1rem 0;
        }
        
        .sample-display {
            background: #f8f9fc;
            border: 1px solid #e8ecf3;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .context-viewer {
            background: linear-gradient(145deg, #f1f3f4, #ffffff);
            border: 1px solid #dadce0;
            border-radius: 8px;
            padding: 1.5rem;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            color: #3c4043;
        }
        
        /* Status badges */
        .status-tag {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.2rem;
        }
        
        .tag-excellent { background: #e8f5e8; color: #2d5a2d; }
        .tag-good { background: #fff3cd; color: #856404; }
        .tag-poor { background: #f8d7da; color: #721c24; }
        .tag-neutral { background: #e2e3e5; color: #495057; }
        
        /* Download section */
        .download-section {
            background: linear-gradient(135deg, #f8f9fa, #ffffff);
            border: 2px dashed #dee2e6;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin: 2rem 0;
        }
        
        .download-btn {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 0.8rem 2rem;
            border-radius: 8px;
            text-decoration: none;
            display: inline-block;
            font-weight: 600;
            transition: transform 0.2s;
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        }
        
        .download-btn:hover {
            transform: translateY(-2px);
            text-decoration: none;
            color: white;
        }
        
        /* Metrics styling */
        .metric-row {
            background: linear-gradient(135deg, #f8f9fa, #ffffff);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* Hide Streamlit elements */
        .stDeployButton { display: none; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        
        /* Sidebar styling */
        .css-1d391kg { padding-top: 1rem; }
        </style>
        """, unsafe_allow_html=True)
        
        # Clean, elegant header
        st.markdown("""
        <div class="main-header">
            <div class="hero-title">🏆 JNANA Telugu QA</div>
            <div class="hero-subtitle">Telugu Question-Answering Model Evaluation Platform</div>
        </div>
        """, unsafe_allow_html=True)

        # Clean download section
        st.markdown("""
        <div class="download-section">
            <h3 style="margin-bottom: 1rem; color: #495057;">📥 Get Started</h3>
            <a href="https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json" 
               class="download-btn" target="_blank">
                📊 Download Evaluation Dataset
            </a>
            <p style="margin-top: 1rem; color: #6c757d; font-size: 0.9rem;">
                1000 curated Telugu QA pairs • JSON format • Ready for evaluation
            </p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_submission_form():
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e1e5e9; margin-bottom: 1.5rem;">
            <h3 style="color: #495057; margin: 0;">🚀 Submit Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.sidebar.form("submission_form"):
            model_name = st.text_input(
                "Model Name*", 
                placeholder="GPT-4, BERT-Telugu, Custom Model...",
                help="Name of your model or approach"
            )
            author_name = st.text_input(
                "Your Name*", 
                placeholder="Your name or team name"
            )
            version_tag = st.text_input(
                "Version", 
                placeholder="v1.0, final, baseline...",
                help="Optional version identifier"
            )
            notes = st.text_area(
                "Description", 
                placeholder="Brief description of your approach, training data, or methodology...",
                help="Optional notes about your submission"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload Results JSON", 
                type="json",
                help="JSON file with your model's predictions"
            )
            
            submitted = st.form_submit_button(
                "📤 Submit to Leaderboard", 
                use_container_width=True,
                type="primary"
            )
            
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
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: #495057; font-weight: 600;">🏆 Leaderboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if leaderboard_rows:
            df = pd.DataFrame(leaderboard_rows)
            
            # Show key metrics in a clean table
            main_cols = ["Model", "Author", "Version", "EM (%)", "F1 (%)", "Answered (%)", "Hallucinated (%)"]
            if all(col in df.columns for col in main_cols):
                st.dataframe(
                    df[main_cols].style.format({
                        "EM (%)": "{:.1f}%",
                        "F1 (%)": "{:.1f}%", 
                        "Answered (%)": "{:.1f}%",
                        "Hallucinated (%)": "{:.1f}%"
                    }).background_gradient(subset=["EM (%)", "F1 (%)"], cmap="RdYlGn"),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                
            # Optional detailed view
            with st.expander("📊 View All Metrics", expanded=False):
                st.dataframe(
                    df.style.format({
                        col: "{:.1f}%" for col in df.columns if "(%)" in col
                    }),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.markdown("""
            <div class="elegant-card" style="text-align: center;">
                <h3 style="color: #6c757d;">🎯 No submissions yet</h3>
                <p style="color: #6c757d;">Be the first to submit your model and claim the top spot!</p>
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def render_sample_explorer(all_data, ref_lookup):
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0 2rem 0;">
            <h2 style="color: #495057; font-weight: 600;">🔍 Sample Explorer</h2>
            <p style="color: #6c757d; margin-top: 0.5rem;">Dive deep into model predictions and analyze performance patterns</p>
        </div>
        """, unsafe_allow_html=True)

        if all_data:
            selected_submission = st.selectbox(
                "Choose a submission to explore", 
                ["Select a model submission..."] + list(all_data.keys()),
                help="Select a model submission to explore its predictions"
            )

            if selected_submission != "Select a model submission...":
                UIComponents._display_submission_details(selected_submission, all_data, ref_lookup)
        else:
            st.markdown("""
            <div class="elegant-card" style="text-align: center;">
                <h3 style="color: #6c757d;">🎯 No submissions available yet</h3>
                <p style="color: #6c757d;">Submit your first model to start exploring results!</p>
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
        """Render individual sample card with clean, elegant styling"""
        # Get context for sample
        context_text = UIComponents._get_context_for_sample(
            row["content_id"], row["qa_index"], ref_lookup
        )

        # Sample display
        st.markdown(f"""
        <div class="sample-display">
            <h4 style="color: #495057; margin-bottom: 1.5rem;">📄 Sample {current_idx} of {total_samples}</h4>
            
            <div style="margin-bottom: 1rem;">
                <strong style="color: #6c757d;">Question:</strong><br>
                <span style="font-size: 1.1rem; color: #212529;">{row['question']}</span>
            </div>
            
            <div style="margin-bottom: 1rem;">
                <strong style="color: #28a745;">Expected Answer:</strong><br>
                <span style="color: #28a745; font-weight: 500;">{row['gold_answer']}</span>
            </div>
            
            <div style="margin-bottom: 1.5rem;">
                <strong style="color: #007bff;">Model Prediction:</strong><br>
                <span style="color: #007bff; font-weight: 500;">{row['prediction']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics in clean cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 F1 Score", f"{row['f1_score']:.3f}")
        with col2:
            em_value = "✅ Match" if row['exact_match'] else "❌ No Match"
            st.metric("🔍 Exact Match", em_value)
        with col3:
            hall_value = "⚠️ Yes" if row['hallucinated'] else "✅ No"
            st.metric("🚨 Hallucinated", hall_value)
        with col4:
            # Clean type badge
            type_styles = {
                "faithful_correct": "tag-excellent",
                "faithful_incorrect": "tag-poor", 
                "hallucinated": "tag-good",
                "empty": "tag-neutral"
            }
            type_class = type_styles.get(row['breakdown'], "tag-neutral")
            type_label = row['breakdown'].replace('_', ' ').title()
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <span class="status-tag {type_class}">{type_label}</span>
            </div>
            """, unsafe_allow_html=True)

        # Context display with cleaner design
        with st.expander("📖 View Source Context", expanded=False):
            if context_text and context_text not in ["[Context not available]", "[Empty context]"]:
                st.markdown(f"""
                <div class="context-viewer">
                    {context_text}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Context not available for this sample (ID: {row['content_id']}, Index: {row['qa_index']})")
                
                # Minimal debug info
                if st.checkbox("Show debug info", key=f"debug_{current_idx}"):
                    lookup_key = (int(row['content_id']), int(row['qa_index']))
                    st.code(f"Looking for: {lookup_key}")
                    st.write(f"Available keys sample: {list(ref_lookup.keys())[:5]}...")

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
