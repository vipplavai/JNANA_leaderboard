# 📊 JNANA QA Leaderboard

Welcome to the **JNANA Telugu QA Leaderboard**, a centralized benchmark to evaluate Telugu short-form question-answering (QA) models on faithfulness, precision, and hallucination metrics.

---

## 📂 Dataset

We provide a 1000-sample evaluation dataset carefully curated from verified content and manually audited. Each sample includes:

* `content_id`
* `qa_index`
* `question`
* `gold_answer`
* `content_text`

📥 **Download the evaluation dataset:**
[➡️ samples\_1000.json](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json)

---

## ✅ Submission Format

To participate in the leaderboard, run inference on the provided dataset and prepare a JSON file with the **following schema**:

```json
[
  {
    "content_id": 101,
    "qa_index": 0,
    "question": "...",
    "gold_answer": "...",
    "prediction": "...",
    "exact_match": 0,
    "f1_score": 0.0,
    "answerable": 1,
    "hallucinated": false,
    "hallucination_type": "faithful_incorrect"
  },
  ...
]
```

* `exact_match` → 1 if prediction matches gold exactly
* `f1_score` → computed F1 score between gold and prediction
* `answerable` → 1 if prediction is not empty
* `hallucinated` → true if prediction is not grounded in the context
* `hallucination_type`:

  * `faithful_correct`
  * `faithful_incorrect`
  * `hallucinated`
  * `empty`

📌 Ensure `content_id` and `qa_index` match the original dataset.

---

## 🧪 Metrics Tracked

Each submission is automatically evaluated on the following:

| Metric                   | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `EM (%)`                 | Exact string match                               |
| `F1 (%)`                 | Token-level F1 between gold and prediction       |
| `Answered (%)`           | % of samples with non-empty predictions          |
| `Hallucinated (%)`       | % of predictions not grounded in context         |
| `Faithful Correct (%)`   | Prediction matches gold and appears in context   |
| `Faithful Incorrect (%)` | Prediction is in context but does not match gold |
| `Empty (%)`              | Model gave no answer                             |

---

## 🖥️ Sample Explorer Guide

Once submissions are uploaded:

* Use the dropdown to select your model submission.
* Filter by breakdown category: `hallucinated`, `faithful_correct`, etc.
* Use the slider to browse individual QA pairs.
* See prediction, gold answer, and full content context.

---

## 🚀 Hosting & Deployment

The leaderboard is built with **Streamlit** and supports:

* Local deployment
* Streamlit Cloud hosting
* Automatic submission directory parsing

---

## 📝 How to Contribute

1. Clone the repository
2. Place your results JSON in `submissions/`
3. Run the app:

```bash
streamlit run app.py
```

4. Or upload on Streamlit Cloud once deployed

---

## 🔗 Useful Links

* 📄 [Leaderboard Web App](https://jnana-leaderboard.streamlit.app/) *(once hosted)*
* 📥 [Download Dataset](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json)
* 🧾 [Sample Result File](https://github.com/vipplavai/JNANA_leaderboard/blob/main/submissions/gemma2b_eval_cleaned.json)

---

## 📬 Contact

For questions or collaborations, reach out via GitHub Issues or [@vipplavai](https://github.com/vipplavai)

---

Let’s build faithful Telugu QA systems — together!
