import streamlit as st
import pandas as pd
import json
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
# Streamlit Config
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
# Load Reference Data
# ---------------------------
ref_data = list(ref_collection.find({}))
ref_lookup = {(r["content_id"], r["qa_index"]): r.get("content_text", "") for r in ref_data}

# ---------------------------
# File Upload + Validation
# ---------------------------
st.sidebar.header("📥 Submit Your Model Output")
model_name = st.sidebar.text_input("Model Name (optional)")
author_name = st.sidebar.text_input("Your Name or Alias (optional)")
uploaded_file = st.sidebar.file_uploader("Upload result JSON file", type="json")

REQUIRED_KEYS = {"content_id", "qa_index", "question", "gold_answer", "prediction",
                 "exact_match", "f1_score", "answerable", "hallucinated", "type"}

if uploaded_file:
    raw_json = uploaded_file.read()
    try:
        data = json.loads(raw_json)
        if isinstance(data, list) and all(REQUIRED_KEYS.issubset(d.keys()) for d in data):
            # Save backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = model_name if model_name else "unnamed_model"
            author = author_name if author_name else "anonymous"
            filename = f"{name.replace(' ', '_')}_{author.replace(' ', '_')}_{timestamp}.json"
            with open(f"submissions/{filename}", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Add metadata and insert into MongoDB
            for d in data:
                d["model_name"] = name
                d["author"] = author
                d["timestamp"] = timestamp
            submissions_collection.insert_many(data)

            st.sidebar.success("✅ Submission validated and stored successfully!")
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid JSON structure. Please check required keys.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load JSON: {e}")

# ---------------------------
# Load Submissions from MongoDB
# ---------------------------
all_data = {}
leaderboard_rows = []

for meta in submissions_collection.aggregate([
    {"$group": {
        "_id": {"model_name": "$model_name", "author": "$author", "timestamp": "$timestamp"},
        "count": {"$sum": 1}
    }},
    {"$sort": {"_id.timestamp": -1}}
]):
    m = meta["_id"]
    query = {"model_name": m["model_name"], "author": m["author"], "timestamp": m["timestamp"]}
    records = list(submissions_collection.find(query))
    df = pd.DataFrame(records)

    df["breakdown"] = df["type"]
    if "content_text" not in df.columns:
        df["content_text"] = df.apply(
            lambda row: ref_lookup.get((row["content_id"], row["qa_index"]), "[context not available]"),
            axis=1
        )

    file_id = f"{m['model_name']}_{m['author']}_{m['timestamp']}"
    all_data[file_id] = df

    breakdown = df["breakdown"].value_counts(normalize=True).mul(100).round(2).to_dict()
    leaderboard_rows.append({
        "Model": m["model_name"],
        "Author": m["author"],
        "Time": m["timestamp"],
        "Samples": len(df),
        "EM (%)": round(df["exact_match"].mean() * 100, 2),
        "F1 (%)": round(df["f1_score"].mean() * 100, 2),
        "Answered (%)": round(df["answerable"].mean() * 100, 2),
        "Hallucinated (%)": round(df["hallucinated"].mean() * 100, 2),
        "Faithful Correct (%)": round((df["breakdown"] == "faithful_correct").mean() * 100, 2),
        "Faithful Incorrect (%)": breakdown.get("faithful_incorrect", 0.0),
        "Hallucinated Breakdown (%)": breakdown.get("hallucinated", 0.0),
        "Empty (%)": breakdown.get("empty", 0.0),
    })

# ---------------------------
# Leaderboard Table
# ---------------------------
st.subheader("🏆 Leaderboard")
if leaderboard_rows:
    leaderboard_df = pd.DataFrame(leaderboard_rows)
    st.dataframe(leaderboard_df)
else:
    st.info("No submissions yet. Submit your model in the sidebar!")

# ---------------------------
# Sample Explorer
# ---------------------------
st.subheader("🔍 Sample Explorer")

st.markdown("""
ℹ️ **How to Use:**
- Choose a model submission.
- Filter samples by error type.
- Use the slider to explore examples.
- You'll see question, gold answer, model prediction, and context.
""")

selected_file = st.selectbox("Choose a submission to explore", ["None"] + list(all_data.keys()))
if selected_file != "None":
    df = all_data[selected_file]
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
