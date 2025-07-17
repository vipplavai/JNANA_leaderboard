
# 🏆 JNANA Telugu QA Leaderboard

A comprehensive leaderboard system for evaluating Telugu question-answering models using a curated 1000-sample benchmark dataset with rich contextual information and human annotations.

## 🌟 Introduction

The JNANA Telugu QA Leaderboard is an open evaluation platform designed to benchmark Telugu language models on question-answering tasks. Our system evaluates models not just on accuracy, but also on faithfulness and hallucination detection - crucial aspects for reliable AI systems.

### Key Features
- **Real-time Evaluation**: Upload predictions and get instant comprehensive metrics
- **Interactive Sample Explorer**: Browse QA pairs with context and detailed analysis
- **Faithfulness Analysis**: Detect hallucinated vs grounded responses
- **Rich Dataset**: Human-annotated samples with quality scores and metadata
- **Persistent Leaderboard**: Compare multiple model submissions over time
- **MongoDB Integration**: Persistent storage with local fallback support

## 📂 Dataset

### Overview
Our benchmark consists of **1000 carefully curated Telugu QA pairs** with rich annotations:

- **Diverse Topics**: Biography, Science, History, Culture, Geography, Literature, Cinema, Mythology, Sports, and more
- **Rich Context**: Each question includes relevant Telugu text passages for grounding
- **Human Annotations**: Multiple judgments per question with inter-annotator agreement scores
- **Quality Metrics**: Fleiss Kappa scores indicating annotation reliability
- **Metadata**: Topic classification, genre, and tone information

### Sample Structure
Each sample in our dataset contains:

```json
{
  "_id": {"$oid": "unique_identifier"},
  "content_id": 1920,
  "qa_index": 0,
  "question": "కె.కె. అగర్వాల్ జన్మతేదీ ఏమిటి?",
  "answer": "5 సెప్టెంబర్ 1958",
  "judgments": ["Correct", "Correct", "Correct", "Correct", "Correct"],
  "fleiss_kappa": 1.0,
  "metadata": {
    "topic": "Biography",
    "genre": "Expository", 
    "tone": "Factual"
  },
  "content_text": "కె.కె. అగర్వాల్\n\nక్రిషన్ కుమార్ అగర్వాల్ (5 సెప్టెంబర్ 1958 - 17 మే 2021)..."
}
```

### Dataset Statistics
- **Total Samples**: 1,000 QA pairs
- **Average Context Length**: ~800 words
- **Topics Covered**: 15+ major categories
- **Languages**: Telugu (questions, answers, context)
- **Quality Score**: High inter-annotator agreement (avg. Fleiss Kappa: 0.95+)

### Download
```bash
# Direct download from repository
wget https://raw.githubusercontent.com/vipplavai/JNANA_leaderboard/main/data/samples_1000.json
```

## ✅ Submission Format

### Required JSON Schema
Your model predictions should follow this exact format:

```json
[
  {
    "content_id": 1920,
    "qa_index": 0,
    "question": "కె.కె. అగర్వాల్ జన్మతేదీ ఏమిటి?",
    "gold_answer": "5 సెప్టెంబర్ 1958",
    "prediction": "5 సెప్టెంబర్ 1958",
    "exact_match": true,
    "f1_score": 1.0,
    "answerable": true,
    "hallucinated": false,
    "type": "faithful_correct"
  }
]
```

### Field Descriptions
| Field | Type | Description |
|-------|------|-------------|
| `content_id` | int | Unique identifier for the source content |
| `qa_index` | int | Question index within the content |
| `question` | string | The Telugu question |
| `gold_answer` | string | Reference answer from dataset |
| `prediction` | string | Your model's prediction |
| `exact_match` | boolean | Whether prediction exactly matches gold answer |
| `f1_score` | float | Token-level F1 score (0.0-1.0) |
| `answerable` | boolean | Whether model provided an answer |
| `hallucinated` | boolean | Whether answer is not grounded in context |
| `type` | string | One of: `faithful_correct`, `faithful_incorrect`, `hallucinated`, `empty` |

### Validation Rules
- All 1000 samples must be included
- `content_id` and `qa_index` must match the reference dataset
- F1 scores should be between 0.0 and 1.0
- Type field must be one of the four valid categories
- Empty predictions should have `answerable: false` and `type: "empty"`

## 🧪 Metrics Tracked

### Core Performance Metrics

**🎯 EM (Exact Match)**: Percentage of predictions that exactly match the gold answer
- Higher is better • Range: 0-100%
- Measures precise correctness

**🔍 F1 Score**: Token-level overlap between prediction and gold answer  
- Measures partial correctness • Range: 0-100%
- Accounts for partial matches

**📝 Answered**: Percentage of questions with non-empty predictions
- Shows model coverage • Range: 0-100%
- Indicates model's willingness to answer

### Faithfulness & Reliability Metrics

**🚫 Hallucinated**: Predictions not grounded in the provided context
- Lower is better • Shows model reliability
- Critical for trustworthy AI systems

**✅ Faithful Correct**: Correct answers that are grounded in context
- Higher is better • Gold standard performance
- Balances accuracy and faithfulness

**❌ Faithful Incorrect**: Wrong answers that are grounded in context
- Shows reasoning errors vs hallucinations
- Helps identify systematic issues

**📭 Empty**: Questions with no prediction provided
- Lower is better • Shows model coverage
- Indicates conservative behavior

### Advanced Analytics

**🎭 FAA (Faithfulness-Adjusted Accuracy)**: Faithful correct percentage
- Balances accuracy and faithfulness
- More reliable than raw accuracy

**📏 F1-EM Gap**: Difference between F1 and EM scores
- Shows partial vs exact correctness
- Indicates answer quality

**⚠️ Overconfident EM**: Exact matches that are hallucinated
- Identifies dangerous overconfidence
- Critical safety metric

**💪 Robust Answer Rate**: Answered rate minus hallucination rate
- Net reliable answering capability
- Shows practical utility

**📊 Avg Answer Length**: Mean word count of predictions
- Indicates response verbosity
- Helps understand model behavior

### Understanding the Metrics
```
High EM + Low Hallucination = Reliable Model ✅
High F1 + High Hallucination = Creative but Unreliable ⚠️
Low Answered + Low Hallucination = Conservative Model 🤔
High FAA = Best Overall Performance 🏆
```
## 🖥️ Sample Explorer Guide

### Getting Started
1. **📥 Submit Results**: Upload your model predictions via the sidebar form
2. **📊 Select Submission**: Choose from dropdown (shows model, author, version, timestamp)  
3. **🏷️ Filter Samples**: Use type filter to focus on specific prediction categories
4. **🎚️ Navigate**: Use slider to browse through filtered samples
5. **📖 View Context**: Click "Show Context" to see source passages

## 🔗 Useful Links

### Dataset & Repository
- **📊 Leaderboard**: [Live Leaderboard](https://your-repl-url.repl.co)
- **📁 Dataset Download**: [samples_1000.json](https://github.com/vipplavai/JNANA_leaderboard/blob/main/data/samples_1000.json)
- **💻 Source Code**: [GitHub Repository](https://github.com/vipplavai/JNANA_leaderboard)
- **📋 Issues & Discussions**: [GitHub Issues](https://github.com/vipplavai/JNANA_leaderboard/issues)

## 📬 Contact

### Research Team
- **📧 Primary Contact**: [research@jnana-leaderboard.org](mailto:research@jnana-leaderboard.org)
- **🔬 Technical Issues**: [tech-support@jnana-leaderboard.org](mailto:tech-support@jnana-leaderboard.org)
- **🤝 Collaborations**: [partnerships@jnana-leaderboard.org](mailto:partnerships@jnana-leaderboard.org)

### Citation
If you use JNANA Telugu QA Leaderboard in your research, please cite:
```bibtex
@dataset{jnana_telugu_qa_2024,
  title={JNANA Telugu QA Leaderboard: A Comprehensive Benchmark for Telugu Question Answering},
  author={Research Team},
  year={2024},
  url={https://github.com/vipplavai/JNANA_leaderboard}
}
```

---

**📊 Start Evaluating**: Ready to test your Telugu QA model? [Submit your results](https://your-repl-url.repl.co) and join the leaderboard!
