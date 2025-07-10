import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

st.set_page_config(page_title="JNANA QA Leaderboard", layout="wide")
st.title("📊 JNANA Telugu QA Leaderboard")

# ---------------------------
# About Section
# ---------------------------
st.markdown("""
**Welcome to the official JNANA QA Leaderboard!**

This leaderboard evaluates Telugu short-answer question-answering models using a curated 1000-sample benchmark.

**Metrics Explained:**
- **EM (%)** – Exact string match with the gold answer.
- **F1 (%)** – Overlap between predicted and true answers.
- **Answered (%)** – Percentage of questions with any non-empty answer.
- **Hallucinated (%)** – Answers that are not grounded in the given context.
- **Faithful Correct (%)** – Exact match and grounded.
- **Faithful Incorrect (%)** – Answer is in the context but incorrect.
- **Empty (%)** – No answer was returned.

We encourage the community to upload their model predictions in the prescribed JSON format and track progress over time.
""")

# ---------------------------
# Setup paths
# ---------------------------
SUBMISSION_DIR = "submissions"
REFERENCE_FILE = "data/samples_1000.json"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ---------------------------
# Load Reference Data
# ---------------------------
with open(REFERENCE_FILE, "r", encoding="utf-8") as ref_file:
    ref_data = json.load(ref_file)
ref_lookup = {
    (item["content_id"], item["qa_index"]): item.get("content_text", "")
    for item in ref_data
}

# ---------------------------
# Handle File Upload
# ---------------------------
st.sidebar.header("📥 Submit Your Model Output")
model_name = st.sidebar.text_input("Model Name (required)")
author_name = st.sidebar.text_input("Your Name or Alias")
uploaded_file = st.sidebar.file_uploader("Upload result JSON file", type="json")

if uploaded_file and model_name:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name.replace(' ', '_')}_{author_name.replace(' ', '_')}_{timestamp}.json"
    save_path = os.path.join(SUBMISSION_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"✅ Uploaded and saved as: {filename}")
    st.rerun()
elif uploaded_file and not model_name:
    st.sidebar.warning("⚠️ Please enter a model name before uploading")

# ---------------------------
# Load All Submissions
# ---------------------------
def load_submission(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        return None

submission_files = [f for f in os.listdir(SUBMISSION_DIR) if f.endswith(".json")]
leaderboard_rows = []
all_data = {}

for file in submission_files:
    df = load_submission(os.path.join(SUBMISSION_DIR, file))
    if df is not None and len(df) > 0:
        if "hallucination_type" in df.columns:
            df['breakdown'] = df['hallucination_type']
        else:
            df['breakdown'] = df.apply(
                lambda row: "hallucinated" if row['hallucinated'] else (
                    "empty" if not row['prediction'].strip() else (
                        "faithful_correct" if row['exact_match'] else "faithful_incorrect"
                    )
                ), axis=1
            )

        # Add context if missing using reference
        if "content_text" not in df.columns:
            df["content_text"] = df.apply(
                lambda row: ref_lookup.get((row["content_id"], row["qa_index"]), "[context not available]"), axis=1
            )
        all_data[file] = df

        breakdown_counts = df['breakdown'].value_counts(normalize=True).mul(100).round(2).to_dict()

        leaderboard_rows.append({
            "Filename": file,
            "Samples": len(df),
            "EM (%)": round(df['exact_match'].mean() * 100, 2),
            "F1 (%)": round(df['f1_score'].mean() * 100, 2),
            "Answered (%)": round(df['answerable'].mean() * 100, 2),
            "Hallucinated (%)": round(df['hallucinated'].mean() * 100, 2),
            "Faithful Correct (%)": round((df['breakdown'] == 'faithful_correct').mean() * 100, 2),
            "Empty (%)": breakdown_counts.get("empty", 0.0),
            "Hallucinated Breakdown (%)": breakdown_counts.get("hallucinated", 0.0),
            "Faithful Incorrect (%)": breakdown_counts.get("faithful_incorrect", 0.0)
        })

# ---------------------------
# Leaderboard View
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
selected_file = st.selectbox("Choose a submission to explore", ["None"] + submission_files)

if selected_file != "None":
    df = all_data[selected_file]
    tag_filter = st.selectbox("Breakdown Filter", ["all"] + sorted(df['breakdown'].unique()))
    if tag_filter != "all":
        df = df[df['breakdown'] == tag_filter]

    if len(df) == 0:
        st.warning("No samples found for this filter.")
    else:
        index = st.slider("Sample index", 0, len(df)-1, 0)
        sample = df.iloc[index]

        st.markdown(f"**Q{sample['qa_index']}**: {sample['question']}")
        st.markdown(f"**Gold Answer**: {sample['gold_answer']}")
        st.markdown(f"**Prediction**: {sample['prediction']}")
        st.markdown(f"**F1**: {sample['f1_score']:.2f} | EM: {sample['exact_match']} | Hallucinated: {sample['hallucinated']}")
        st.markdown(f"**Breakdown Tag**: {sample['breakdown']}")
        st.markdown(f"---")
        context = sample.get("content_text", "[context not available]")
        st.markdown(f"**Context:**\n\n{context}")
