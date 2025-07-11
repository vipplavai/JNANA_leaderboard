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
ref_cursor = ref_collection.find({})
ref_lookup = {(item["content_id"], item["qa_index"]): item.get("content_text", "") for item in ref_cursor}

# ---------------------------
# File Upload Section
# ---------------------------
SUBMISSION_DIR = "submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

st.sidebar.header("📥 Submit Your Model Output")
model_name = st.sidebar.text_input("Model Name (optional)")
author_name = st.sidebar.text_input("Your Name or Alias (optional)")
uploaded_file = st.sidebar.file_uploader("Upload result JSON file", type="json")

if uploaded_file:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = model_name if model_name else "unnamed_model"
    author = author_name if author_name else "anonymous"
    filename = f"{name.replace(' ', '_')}_{author.replace(' ', '_')}_{timestamp}.json"
    save_path = os.path.join(SUBMISSION_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"✅ Uploaded and saved as: {filename}")
    st.rerun()

# ---------------------------
# Load All Submissions
# ---------------------------
def load_submission(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    except Exception:
        return None

submission_files = [f for f in os.listdir(SUBMISSION_DIR) if f.endswith(".json")]
leaderboard_rows = []
all_data = {}

for file in submission_files:
    df = load_submission(os.path.join(SUBMISSION_DIR, file))
    if df is not None and len(df) > 0:
        if "type" in df.columns:
            df["breakdown"] = df["type"]
        else:
            df["breakdown"] = df.apply(
                lambda row: "hallucinated" if row["hallucinated"] else (
                    "empty" if not row["prediction"].strip() else (
                        "faithful_correct" if row["exact_match"] else "faithful_incorrect"
                    )
                ), axis=1
            )

        # Add context if missing
        if "content_text" not in df.columns:
            df["content_text"] = df.apply(
                lambda row: ref_lookup.get((row["content_id"], row["qa_index"]), "[context not available]"),
                axis=1
            )

        all_data[file] = df
        breakdown = df["breakdown"].value_counts(normalize=True).mul(100).round(2).to_dict()

        leaderboard_rows.append({
            "Filename": file,
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
    st.info("No submissions found yet. Upload your first model in the sidebar!")

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

selected_file = st.selectbox("Choose a submission to explore", ["None"] + submission_files)

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
        context = sample.get("content_text", "[context not available]")
        st.markdown(f"**Context:**\n\n{context}")
