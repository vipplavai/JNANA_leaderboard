import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pymongo import MongoClient

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

### Metrics Explained:
- **EM (%)** – Exact string match with the gold answer.
- **F1 (%)** – Overlap between predicted and true answers.
- **Answered (%)** – Percentage of questions with any non-empty answer.
- **Hallucinated (%)** – Answers that are not grounded in the given context.
- **Faithful Correct (%)** – Exact match and grounded.
- **Faithful Incorrect (%)** – Answer is in the context but incorrect.
- **Empty (%)** – No answer was returned.
""")

# ---------------------------
# Load Reference Samples from MongoDB
# ---------------------------
try:
    ref_cursor = ref_collection.find({})
    ref_lookup = {(item["content_id"], item["qa_index"]): item.get("content_text", "") for item in ref_cursor}
except Exception as e:
    st.error(f"Failed to load reference data: {e}")
    ref_lookup = {}

# ---------------------------
# File Upload Section with Validation
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
    """Check if each record has the correct structure and data types."""
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

if uploaded_file:
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
                # Save submission to MongoDB
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
    except json.JSONDecodeError:
        st.sidebar.error("❌ Invalid JSON format. Please check your file.")

# ---------------------------
# Load Submissions from MongoDB
# ---------------------------
try:
    submissions = list(submissions_collection.find({}))
except Exception as e:
    st.error(f"Failed to load submissions: {e}")
    submissions = []

leaderboard_rows = []
all_data = {}

for sub in submissions:
    df = pd.DataFrame(sub["results"])
    if df.empty:
        continue
    df["breakdown"] = df["type"]

    if "content_text" not in df.columns:
        df["content_text"] = df.apply(
            lambda row: ref_lookup.get((row["content_id"], row["qa_index"]), "[context not available]"),
            axis=1
        )

    sub_id = str(sub["_id"])
    all_data[sub_id] = df
    breakdown = df["breakdown"].value_counts(normalize=True).mul(100).round(2).to_dict()

    leaderboard_rows.append({
        "Model": sub["model"],
        "Author": sub["author"],
        "Samples": len(df),
        "EM (%)": round(df["exact_match"].mean() * 100, 2),
        "F1 (%)": round(df["f1_score"].mean() * 100, 2),
        "Answered (%)": round(df["answerable"].mean() * 100, 2),
        "Hallucinated (%)": round(df["hallucinated"].mean() * 100, 2),
        "Faithful Correct (%)": round((df["breakdown"] == "faithful_correct").mean() * 100, 2),
        "Faithful Incorrect (%)": breakdown.get("faithful_incorrect", 0.0),
        "Empty (%)": breakdown.get("empty", 0.0),
        "Timestamp": sub["timestamp"].strftime("%Y-%m-%d %H:%M")
    })

# ---------------------------
# Leaderboard View
# ---------------------------
st.subheader("🏆 Leaderboard")
if leaderboard_rows:
    leaderboard_df = pd.DataFrame(leaderboard_rows)
    st.dataframe(leaderboard_df)
else:
    st.info("No submissions found yet.")

# ---------------------------
# Sample Explorer
# ---------------------------
st.subheader("🔍 Sample Explorer")
st.markdown("""
ℹ️ **How to Use:**
- Choose a submission.
- Filter samples by type: hallucinated, faithful_correct, etc.
- Use the slider to browse through examples.
- Each sample includes the question, gold answer, model prediction, and context.
""")

selected_id = st.selectbox("Choose a submission to explore", ["None"] + list(all_data.keys()))

if selected_id != "None":
    df = all_data[selected_id]
    tag_filter = st.selectbox("Breakdown Filter", ["all"] + sorted(df["breakdown"].unique()))
    if tag_filter != "all":
        df = df[df["breakdown"] == tag_filter]

    if len(df) == 0:
        st.warning("No samples found for this filter.")
    else:
        index = st.slider("Sample index", 0, len(df)-1, 0)
        sample = df.iloc[index]
        st.markdown(f"**Q{sample['qa_index']}**: {sample['question']}")
        st.markdown(f"**Gold Answer**: {sample['gold_answer']}")
        st.markdown(f"**Prediction**: {sample['prediction']}")
        st.markdown(f"**F1**: {sample['f1_score']:.2f} | EM: {sample['exact_match']} | Hallucinated: {sample['hallucinated']}")
        st.markdown(f"**Type**: {sample['breakdown']}")
        st.markdown("---")
        st.markdown(f"**Context:**\n\n{sample['content_text']}")
