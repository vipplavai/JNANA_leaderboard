
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pymongo import MongoClient
from typing import List, Dict


# ---------------------------
# Metric Computation
# ---------------------------
def compute_metrics(data: List[Dict]) -> Dict:
    total = len(data)
    if total == 0:
        return {key: 0.0 for key in [
            "total", "em", "f1", "answered", "hallucinated", "faithful_correct",
            "faithful_incorrect", "empty", "faa", "f1_em_gap",
            "overconfident_em", "robust_answer_rate", "avg_answer_length"
        ]}

    em = sum(1 for d in data if d.get("exact_match")) / total * 100
    f1 = sum(d.get("f1_score", 0) for d in data) / total
    answered = sum(1 for d in data if d.get("answerable")) / total * 100
    hallucinated = sum(1 for d in data if d.get("hallucinated")) / total * 100
    faithful_correct = sum(1 for d in data if d.get("type") == "faithful_correct") / total * 100
    faithful_incorrect = sum(1 for d in data if d.get("type") == "faithful_incorrect") / total * 100
    empty = sum(1 for d in data if d.get("type") == "empty") / total * 100
    faa = faithful_correct
    f1_em_gap = f1 - em
    overconfident_em = sum(1 for d in data if d.get("hallucinated") and d.get("exact_match")) / total * 100
    robust_answer_rate = max(answered - hallucinated, 0.0)
    avg_answer_length = sum(len(str(d.get("prediction", "")).split()) for d in data) / total

    return {
        "total": total,
        "em": round(em, 2),
        "f1": round(f1, 2),
        "answered": round(answered, 2),
        "hallucinated": round(hallucinated, 2),
        "faithful_correct": round(faithful_correct, 2),
        "faithful_incorrect": round(faithful_incorrect, 2),
        "empty": round(empty, 2),
        "faa": round(faa, 2),
        "f1_em_gap": round(f1_em_gap, 2),
        "overconfident_em": round(overconfident_em, 2),
        "robust_answer_rate": round(robust_answer_rate, 2),
        "avg_answer_length": round(avg_answer_length, 2)
    }

# ---------------------------
# MongoDB Setup
# ---------------------------
try:
    MONGO_URI = st.secrets.get("mongo_uri", "mongodb://localhost:27017")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test the connection
    client.admin.command('ping')
    db = client["Leaderboard"]
    ref_collection = db["reference_samples"]
    submissions_collection = db["submissions"]
    MONGODB_AVAILABLE = True
except Exception as e:
    st.warning(f"⚠️ MongoDB connection failed: {e}")
    st.info("📝 Running in local mode - submissions will not be saved")
    MONGODB_AVAILABLE = False
    ref_collection = None
    submissions_collection = None

# ---------------------------
# Streamlit UI Config
# ---------------------------
st.set_page_config(page_title="JNANA QA Leaderboard", layout="wide")
st.title("📊 JNANA Telugu QA Leaderboard")

# ---------------------------
# About Section
# ---------------------------
st.markdown("""
**Welcome to the official JNANA QA Leaderboard!**

This leaderboard evaluates Telugu short-answer question-answering models using a curated 1000-sample benchmark.

📎 **Download Evaluation Dataset**: [samples_1000.json](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json)

🔒 **Open Leaderboard**: All submissions are publicly viewable but cannot be edited or deleted to ensure leaderboard integrity.
""")

# ---------------------------
# Reference Data Cache
# ---------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour since reference data doesn't change
def get_ref_lookup():
    """Load and cache reference data from MongoDB or local file"""
    if MONGODB_AVAILABLE and ref_collection is not None:
        try:
            # Try to load from MongoDB first
            ref_data = list(ref_collection.find({}))
            
            # If MongoDB is empty, populate from local file
            if not ref_data:
                with open("data/samples_1000.json", "r", encoding="utf-8") as f:
                    ref_data = json.load(f)
                
                # Insert into MongoDB for future use
                ref_collection.insert_many(ref_data)
                st.success("✅ Populated 1000 reference samples into MongoDB")
            
            # Create optimized lookup dictionary
            lookup_dict = {}
            for item in ref_data:
                try:
                    content_id = int(item.get("content_id", 0))
                    qa_index = int(item.get("qa_index", 0))
                    content_text = item.get("content_text", "")
                    lookup_dict[(content_id, qa_index)] = content_text
                except (ValueError, TypeError):
                    continue  # Skip malformed entries
            
            return lookup_dict
            
        except Exception as e:
            st.warning(f"⚠️ MongoDB error, falling back to local file: {e}")
    
    # Fallback to local file
    try:
        with open("data/samples_1000.json", "r", encoding="utf-8") as f:
            ref_data = json.load(f)
        
        # Create lookup with proper error handling
        lookup_dict = {}
        for item in ref_data:
            try:
                content_id = int(item.get("content_id", 0))
                qa_index = int(item.get("qa_index", 0))
                content_text = item.get("content_text", "")
                lookup_dict[(content_id, qa_index)] = content_text
            except (ValueError, TypeError):
                continue  # Skip malformed entries
                
        return lookup_dict
    except Exception as local_e:
        st.error(f"❌ Error loading reference data: {local_e}")
        return {}

# Initialize reference lookup
ref_lookup = get_ref_lookup()

# Show reference data status
if ref_lookup:
    st.sidebar.success(f"📚 {len(ref_lookup)} reference samples loaded")
else:
    st.sidebar.error("⚠️ No reference data available")

# ---------------------------
# Upload Submission
# ---------------------------
st.sidebar.header("📥 Submit Your Model Output")
model_name = st.sidebar.text_input("Model Name (required)")
author_name = st.sidebar.text_input("Your Name or Alias (required)")
version_tag = st.sidebar.text_input("Version Tag (optional)", placeholder="v1.0")
notes = st.sidebar.text_area("Notes (optional)", placeholder="Brief description of your model")
uploaded_file = st.sidebar.file_uploader("Upload result JSON file", type="json")

if uploaded_file and "uploaded" not in st.session_state:
    if not model_name or not author_name:
        st.sidebar.error("❌ Model name and author are required.")
    else:
        raw_bytes = uploaded_file.read()
        try:
            parsed_data = json.loads(raw_bytes)
            if not isinstance(parsed_data, list):
                st.sidebar.error("❌ Submission file must be a list of JSON objects.")
            else:
                cleaned_data, validation_errors, sha1_hash = clean_and_validate_submission(parsed_data)
                
                if validation_errors:
                    st.sidebar.error("❌ Validation failed!")
                    for err in validation_errors[:5]:
                        st.sidebar.write(err)
                    if len(validation_errors) > 5:
                        st.sidebar.warning(f"...and {len(validation_errors)-5} more errors")
                else:
                    # Check for duplicate submission
                    existing = submissions_collection.find_one({
                        "model": model_name,
                        "author": author_name,
                        "sha1_hash": sha1_hash
                    })
                    
                    if existing:
                        st.sidebar.warning("⚠️ Identical submission already exists!")
                    else:
                        metrics = compute_metrics(cleaned_data)
                        meta = {
                            "model": model_name,
                            "author": author_name,
                            "timestamp": datetime.utcnow(),
                            "version_tag": version_tag if version_tag else None,
                            "notes": notes if notes else None,
                            "sha1_hash": sha1_hash,
                            "metrics": metrics,
                            "results": cleaned_data
                        }
                        try:
                            submissions_collection.insert_one(meta)
                            st.session_state["uploaded"] = True
                            st.sidebar.success("✅ Submission uploaded successfully!")
                            st.rerun()
                        except Exception as e:
                            st.sidebar.error(f"❌ MongoDB error: {str(e)}")
        except json.JSONDecodeError:
            st.sidebar.error("❌ Invalid JSON format.")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# ---------------------------
# Load Submissions from MongoDB
# ---------------------------
def load_submissions():
    """Load submissions without trying to embed context - context will be loaded separately"""
    try:
        submissions = list(submissions_collection.find({}).sort("timestamp", -1))  # Sort by newest first
        leaderboard_rows, all_data = [], {}
        
        for sub in submissions:
            # Just load the submission data without trying to add context here
            df = pd.DataFrame(sub["results"])
            df["breakdown"] = df["type"]
            
            # Create a readable submission identifier
            model_name = sub.get("model", "Unknown")
            author_name = sub.get("author", "Unknown")
            version_tag = sub.get("version_tag", "")
            timestamp = sub["timestamp"].strftime("%Y-%m-%d %H:%M")
            
            # Create display name for dropdown with proper model name formatting
            display_name = f"🤖 {model_name}"
            if version_tag:
                display_name += f" v{version_tag}"
            display_name += f" | 👤 {author_name} | 📅 {timestamp}"
            
            all_data[display_name] = {
                "data": df,
                "metadata": sub
            }
            
            m = sub.get("metrics", {})
            leaderboard_rows.append({
                "Model": model_name,
                "Author": author_name,
                "Version": version_tag if version_tag else "N/A",
                "Samples": m.get("total", len(df)),
                "EM (%)": m.get("em", 0.0),
                "F1 (%)": m.get("f1", 0.0),
                "Answered (%)": m.get("answered", 0.0),
                "Hallucinated (%)": m.get("hallucinated", 0.0),
                "Faithful Correct (%)": m.get("faithful_correct", 0.0),
                "Faithful Incorrect (%)": m.get("faithful_incorrect", 0.0),
                "Empty (%)": m.get("empty", 0.0),
                "Timestamp": timestamp
            })
        
        return leaderboard_rows, all_data
    except Exception as e:
        st.error(f"Error loading submissions from MongoDB: {e}")
        return [], {}

# Add a refresh button to clear cache and reload data
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Load submissions data fresh each time to ensure proper context loading
leaderboard_rows, all_data = load_submissions()

# ---------------------------
# Leaderboard
# ---------------------------
st.subheader("🏆 Leaderboard")
show_advanced = st.toggle("Show Advanced Metrics", value=False)

if leaderboard_rows:
    leaderboard_df = pd.DataFrame(leaderboard_rows)
    
    if show_advanced:
        # Get fresh submissions data for advanced metrics
        try:
            submissions = list(submissions_collection.find({}).sort("timestamp", -1))
            for i, sub in enumerate(submissions):
                if i < len(leaderboard_df):
                    m = sub.get("metrics", {})
                    leaderboard_df.loc[i, "FAA (%)"] = m.get("faa", 0.0)
                    leaderboard_df.loc[i, "F1-EM Gap"] = m.get("f1_em_gap", 0.0)
                    leaderboard_df.loc[i, "Overconfident EM (%)"] = m.get("overconfident_em", 0.0)
                    leaderboard_df.loc[i, "Robust Answer Rate (%)"] = m.get("robust_answer_rate", 0.0)
                    leaderboard_df.loc[i, "Avg Answer Length"] = m.get("avg_answer_length", 0.0)
        except Exception as e:
            st.warning(f"Could not load advanced metrics: {e}")
    
    st.dataframe(leaderboard_df, use_container_width=True)
else:
    st.info("No submissions yet.")

# ---------------------------
# Sample Explorer
# ---------------------------
st.subheader("🔍 Sample Explorer")
st.markdown("""
ℹ️ **How to Use:**
- Choose a submission from the dropdown (shows model name, author, version, and timestamp).
- Filter samples by type: hallucinated, faithful_correct, etc.
- Use the slider to browse examples.
- This is a read-only view of all submissions in the leaderboard.
""")

if all_data:
    selected_submission = st.selectbox(
        "Choose a submission to explore", 
        ["None"] + list(all_data.keys()),
        help="Select a model submission to explore its predictions"
    )

    if selected_submission != "None":
        submission_info = all_data[selected_submission]
        df = submission_info["data"]
        metadata = submission_info["metadata"]
        
        # Load context directly from reference_samples collection for each sample
        def get_context_for_sample(content_id, qa_index):
            """Get context directly from reference_samples collection"""
            try:
                ref_doc = ref_collection.find_one({
                    "content_id": int(content_id),
                    "qa_index": int(qa_index)
                })
                if ref_doc and ref_doc.get("content_text"):
                    return ref_doc["content_text"]
                else:
                    return "[Context not available]"
            except Exception as e:
                return f"[Error loading context: {e}]"
        
        # Calculate context coverage by checking actual reference collection
        total_samples = len(df)
        context_available = 0
        for _, row in df.iterrows():
            context = get_context_for_sample(row["content_id"], row["qa_index"])
            if context != "[Context not available]" and not context.startswith("[Error"):
                context_available += 1
        
        context_coverage = (context_available / total_samples * 100) if total_samples > 0 else 0
        
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
        
        tag_filter = st.selectbox("Breakdown Filter", ["all"] + sorted(df["breakdown"].unique()))
        if tag_filter != "all":
            df = df[df["breakdown"] == tag_filter]

        if df.empty:
            st.warning("No samples for this filter.")
        else:
            i = st.slider("Sample Index", 0, len(df) - 1, 0)
            row = df.iloc[i]
            
            # Get context directly from reference_samples collection
            context_text = get_context_for_sample(row["content_id"], row["qa_index"])
            
            st.markdown(f"**Q{row['qa_index']}**: {row['question']}")
            st.markdown(f"**Gold Answer**: {row['gold_answer']}")
            st.markdown(f"**Prediction**: {row['prediction']}")
            st.markdown(f"**F1**: {row['f1_score']:.2f} | EM: {row['exact_match']} | Hallucinated: {row['hallucinated']}")
            st.markdown(f"**Type**: {row['breakdown']}")
            st.markdown("---")
            st.markdown(f"**Context**:\n\n{context_text}")
else:
    st.info("No submissions available yet. Submit your first model to see results here!")
