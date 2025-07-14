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
MONGO_URI = st.secrets["mongo_uri"] if "mongo_uri" in st.secrets else "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["Leaderboard"]
ref_collection = db["reference_samples"]
submissions_collection = db["submissions"]

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
""")

# ---------------------------
# Reference Cache
# ---------------------------
<<<<<<< HEAD
ref_cursor = ref_collection.find({})
ref_lookup = {(item["content_id"], item["qa_index"]): item.get("content_text", "") for item in ref_cursor}
=======
<<<<<<< HEAD
try:
    ref_cursor = ref_collection.find({})
    ref_lookup = {(item["content_id"], item["qa_index"]): item.get("content_text", "") for item in ref_cursor}
except Exception as e:
    st.error(f"Failed to load reference data: {e}")
    ref_lookup = {}
=======
@st.cache_data
def get_ref_lookup():
    return {
        (item["content_id"], item["qa_index"]): item.get("content_text", "")
        for item in ref_collection.find({})
    }

ref_lookup = get_ref_lookup()
>>>>>>> 1a4335e (17th)
>>>>>>> a9be370 (17th)

# ---------------------------
# Upload Submission
# ---------------------------
st.sidebar.header("📥 Submit Your Model Output")
model_name = st.sidebar.text_input("Model Name (optional)")
author_name = st.sidebar.text_input("Your Name or Alias (optional)")
uploaded_file = st.sidebar.file_uploader("Upload result JSON file", type="json")

REQUIRED_FIELDS = {
    "content_id", "qa_index", "question", "gold_answer", "prediction",
    "exact_match", "f1_score", "answerable", "hallucinated", "type"
}

def validate_submission(data):
<<<<<<< HEAD
    """Check if each record has the correct structure."""
=======
<<<<<<< HEAD
    """Check if each record has the correct structure and data types."""
=======
>>>>>>> 1a4335e (17th)
>>>>>>> a9be370 (17th)
    errors = []
    for i, item in enumerate(data):
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            errors.append(f"❌ Record {i} missing fields: {missing}")
        
        # Validate data types
        try:
            if not isinstance(item.get("exact_match"), bool):
                errors.append(f"❌ Record {i}: exact_match must be boolean")
            if not isinstance(item.get("f1_score"), (int, float)):
                errors.append(f"❌ Record {i}: f1_score must be numeric")
            if not isinstance(item.get("answerable"), bool):
                errors.append(f"❌ Record {i}: answerable must be boolean")
            if not isinstance(item.get("hallucinated"), bool):
                errors.append(f"❌ Record {i}: hallucinated must be boolean")
        except Exception:
            errors.append(f"❌ Record {i}: invalid data types")
    return errors

if uploaded_file and "uploaded" not in st.session_state:
    raw_bytes = uploaded_file.read()
    try:
        parsed_data = json.loads(raw_bytes)
        if not isinstance(parsed_data, list):
            st.sidebar.error("❌ Submission file must be a list of JSON objects.")
        else:
            validation_errors = validate_submission(parsed_data)
            if validation_errors:
                st.sidebar.error("Validation failed!")
                for err in validation_errors[:5]:
                    st.sidebar.write(err)
                if len(validation_errors) > 5:
                    st.sidebar.warning(f"...and {len(validation_errors)-5} more errors")
            else:
<<<<<<< HEAD
                # Save submission to MongoDB
<<<<<<< HEAD
                timestamp = datetime.utcnow()
                meta = {
                    "model": model_name or "unnamed_model",
                    "author": author_name or "anonymous",
                    "timestamp": timestamp,
                    "results": parsed_data
                }
                submissions_collection.insert_one(meta)
                st.sidebar.success("✅ Submission uploaded and validated successfully!")
                st.rerun()
=======
                try:
                    timestamp = datetime.utcnow()
                    meta = {
                        "model": model_name or "unnamed_model",
                        "author": author_name or "anonymous",
                        "timestamp": timestamp,
                        "results": parsed_data
                    }
                    submissions_collection.insert_one(meta)
                    st.sidebar.success("✅ Submission uploaded and validated successfully!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to save submission: {e}")
=======
                metrics = compute_metrics(parsed_data)
                meta = {
                    "model": model_name or "unnamed_model",
                    "author": author_name or "anonymous",
                    "timestamp": datetime.utcnow(),
                    "metrics": metrics,
                    "results": parsed_data
                }
                submissions_collection.insert_one(meta)
                st.session_state["uploaded"] = True
                st.sidebar.success("✅ Submission uploaded successfully!")
                st.rerun()
>>>>>>> 1a4335e (17th)
>>>>>>> a9be370 (17th)
    except json.JSONDecodeError:
        st.sidebar.error("❌ Invalid JSON format.")

# ---------------------------
# Load Submissions
# ---------------------------
<<<<<<< HEAD
submissions = list(submissions_collection.find({}))
=======
<<<<<<< HEAD
try:
    submissions = list(submissions_collection.find({}))
except Exception as e:
    st.error(f"Failed to load submissions: {e}")
    submissions = []

>>>>>>> a9be370 (17th)
leaderboard_rows = []
all_data = {}
=======
submissions = list(submissions_collection.find({}))
leaderboard_rows, all_data = [], {}
>>>>>>> 1a4335e (17th)

for sub in submissions:
    df = pd.DataFrame(sub["results"])
    df["breakdown"] = df["type"]
    df["content_text"] = df.apply(
        lambda row: ref_lookup.get((row["content_id"], row["qa_index"]), "[context not available]"),
        axis=1
    )
    sub_id = str(sub["_id"])
    all_data[sub_id] = df
    m = sub.get("metrics", {})

    leaderboard_rows.append({
<<<<<<< HEAD
        "Model": sub["model"],
        "Author": sub["author"],
        "Samples": len(df),
        "EM (%)": round(df["exact_match"].mean() * 100, 2),
        "F1 (%)": round(df["f1_score"].mean() * 100, 2),
        "Answered (%)": round(df["answerable"].mean() * 100, 2),
        "Hallucinated (%)": round(df["hallucinated"].mean() * 100, 2),
        "Faithful Correct (%)": round((df["breakdown"] == "faithful_correct").mean() * 100, 2),
        "Faithful Incorrect (%)": breakdown.get("faithful_incorrect", 0.0),
        "Hallucinated Breakdown (%)": breakdown.get("hallucinated", 0.0),
        "Empty (%)": breakdown.get("empty", 0.0),
=======
        "Model": sub.get("model", "N/A"),
        "Author": sub.get("author", "N/A"),
        "Samples": m.get("total", 1000),
        "EM (%)": m.get("em", 0.0),
        "F1 (%)": m.get("f1", 0.0),
        "Answered (%)": m.get("answered", 0.0),
        "Hallucinated (%)": m.get("hallucinated", 0.0),
        "Faithful Correct (%)": m.get("faithful_correct", 0.0),
        "Faithful Incorrect (%)": m.get("faithful_incorrect", 0.0),
        "Empty (%)": m.get("empty", 0.0),
>>>>>>> 1a4335e (17th)
        "Timestamp": sub["timestamp"].strftime("%Y-%m-%d %H:%M")
    })

# ---------------------------
# Leaderboard
# ---------------------------
st.subheader("🏆 Leaderboard")
if leaderboard_rows:
    leaderboard_df = pd.DataFrame(leaderboard_rows)
    st.dataframe(leaderboard_df, use_container_width=True)
else:
    st.info("No submissions yet.")

# ---------------------------
# Sample Explorer
# ---------------------------
st.subheader("🔍 Sample Explorer")
st.markdown("""
ℹ️ **How to Use:**
- Choose a submission.
- Filter samples by type: hallucinated, faithful_correct, etc.
- Use the slider to browse examples.
""")

selected_id = st.selectbox("Choose a submission to explore", ["None"] + list(all_data.keys()))

if selected_id != "None":
    df = all_data[selected_id]
    tag_filter = st.selectbox("Breakdown Filter", ["all"] + sorted(df["breakdown"].unique()))
    if tag_filter != "all":
        df = df[df["breakdown"] == tag_filter]

    if df.empty:
        st.warning("No samples for this filter.")
    else:
        i = st.slider("Sample Index", 0, len(df) - 1, 0)
        row = df.iloc[i]
        st.markdown(f"**Q{row['qa_index']}**: {row['question']}")
        st.markdown(f"**Gold Answer**: {row['gold_answer']}")
        st.markdown(f"**Prediction**: {row['prediction']}")
        st.markdown(f"**F1**: {row['f1_score']:.2f} | EM: {row['exact_match']} | Hallucinated: {row['hallucinated']}")
        st.markdown(f"**Type**: {row['breakdown']}")
        st.markdown("---")
        st.markdown(f"**Context**:\n\n{row['content_text']}")
