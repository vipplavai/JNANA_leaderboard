
# 🏆 JNANA Telugu QA Leaderboard

A comprehensive leaderboard system for evaluating Telugu question-answering models using a curated 1000-sample benchmark dataset with rich contextual information and human annotations.

## 🌟 Introduction

The JNANA Telugu QA Leaderboard is an open evaluation platform designed to benchmark Telugu language models on question-answering tasks. Our system evaluates models not just on accuracy, but also on faithfulness and hallucination detection - crucial aspects for reliable AI systems.

### Why JNANA Telugu QA Leaderboard?

- **🎯 Comprehensive Evaluation**: Beyond simple accuracy, we measure faithfulness, hallucination detection, and model reliability
- **📚 Rich Dataset**: 1000 carefully curated Telugu QA pairs with human annotations and quality scores
- **🔍 Interactive Analysis**: Explore model predictions with detailed sample-by-sample analysis
- **🏆 Fair Comparison**: Standardized evaluation protocol for consistent model comparison
- **🌐 Open Platform**: Transparent evaluation process with public dataset and methodology

### Key Features
- Real-time evaluation with instant comprehensive metrics
- Interactive sample explorer for detailed analysis
- Faithfulness analysis to detect hallucinated vs grounded responses
- Persistent leaderboard with historical model comparisons
- Rich metadata including topics, genres, and annotation quality scores

## 📂 Dataset

Our benchmark consists of **1000 carefully curated Telugu QA pairs** covering diverse topics including Biography, Science, History, Culture, Geography, Literature, Cinema, Mythology, and Sports.

### Dataset Highlights
- **Human-Annotated**: Multiple judgments per question with inter-annotator agreement scores
- **Rich Context**: Each question includes relevant Telugu text passages for grounding
- **Quality Assured**: Fleiss Kappa scores indicating annotation reliability
- **Diverse Coverage**: 15+ major topic categories with balanced representation

### Sample Structure
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

### Dataset Download
```bash
# Download the complete dataset
wget https://raw.githubusercontent.com/vipplavai/JNANA_leaderboard/main/data/samples_1000.json
```

## 📋 Submission Guidelines

### Step 1: Prepare Your Results
Your model predictions must follow this exact JSON format for all 1000 samples:

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

### Step 2: Field Requirements

| Field | Type | Description | Requirements |
|-------|------|-------------|--------------|
| `content_id` | integer | Source content identifier | Must match dataset |
| `qa_index` | integer | Question index within content | Must match dataset |
| `question` | string | Telugu question text | Exact copy from dataset |
| `gold_answer` | string | Reference answer | Exact copy from dataset |
| `prediction` | string | Your model's answer | Your model's output |
| `exact_match` | boolean | Perfect answer match | `true` if prediction == gold_answer |
| `f1_score` | float | Token overlap score | Range: 0.0-1.0 |
| `answerable` | boolean | Model provided answer | `false` for empty predictions |
| `hallucinated` | boolean | Answer not grounded in context | Your evaluation |
| `type` | string | Prediction category | See categories below |

### Step 3: Prediction Categories

**📝 `faithful_correct`**: Correct answer grounded in the provided context
**❌ `faithful_incorrect`**: Wrong answer but grounded in the provided context  
**🚫 `hallucinated`**: Answer not supported by the context (regardless of correctness)
**📭 `empty`**: No answer provided by the model

### Step 4: Validation Rules
- ✅ Include all 1000 samples from the dataset
- ✅ Ensure `content_id` and `qa_index` match exactly
- ✅ F1 scores must be between 0.0 and 1.0
- ✅ Use only the four valid `type` categories
- ✅ Set `answerable: false` for empty predictions
- ✅ Ensure JSON is valid and properly formatted

### Step 5: Upload Process
1. **📁 Save** your results as a `.json` file
2. **🌐 Visit** the leaderboard interface
3. **📜 Scroll down** to find the submission panel at the bottom of the page
4. **📤 Upload** via the submission form
5. **✅ Verify** validation passes
6. **🏆 View** your results on the leaderboard

## 📊 Metrics Explanation

### Core Performance Metrics

**🎯 EM (Exact Match) - Percentage of perfect predictions**
- Calculation: `(Exact matches / Total samples) × 100`
- Range: 0-100% (higher is better)
- Measures: Precise correctness

**🔍 F1 Score - Token-level overlap with gold answers**
- Calculation: Harmonic mean of precision and recall
- Range: 0-100% (higher is better)  
- Measures: Partial correctness, handles spelling variations

**📝 Answered Rate - Percentage of non-empty predictions**
- Calculation: `(Non-empty predictions / Total samples) × 100`
- Range: 0-100% (higher shows coverage)
- Measures: Model's willingness to answer

### Faithfulness & Reliability Metrics

**🚫 Hallucination Rate - Predictions not grounded in context**
- Calculation: `(Hallucinated predictions / Total samples) × 100`
- Range: 0-100% (lower is better)
- Measures: Model reliability and groundedness

**✅ Faithful Correct - Accurate and grounded predictions**
- Calculation: `(Faithful correct / Total samples) × 100`
- Range: 0-100% (higher is better)
- Measures: Gold standard performance

**❌ Faithful Incorrect - Wrong but grounded predictions**
- Calculation: `(Faithful incorrect / Total samples) × 100`
- Range: 0-100% (shows reasoning vs hallucination errors)
- Measures: Systematic reasoning issues

**📭 Empty Rate - Questions with no prediction**
- Calculation: `(Empty predictions / Total samples) × 100`
- Range: 0-100% (lower shows better coverage)
- Measures: Conservative behavior

### Advanced Analytics

**🎭 FAA (Faithfulness-Adjusted Accuracy)**
- Calculation: Same as Faithful Correct percentage
- Purpose: Balances accuracy with reliability
- Interpretation: More trustworthy than raw accuracy

**📏 F1-EM Gap**
- Calculation: `F1 Score - EM Score`
- Purpose: Shows partial vs exact correctness
- Interpretation: Large gap indicates approximate answers

**⚠️ Overconfident EM**
- Calculation: `(Hallucinated exact matches / Total samples) × 100`
- Purpose: Identifies dangerous overconfidence
- Interpretation: Should be close to 0%

**💪 Robust Answer Rate**
- Calculation: `Answered Rate - Hallucination Rate`
- Purpose: Net reliable answering capability  
- Interpretation: Practical model utility

**📊 Average Answer Length**
- Calculation: Mean word count of all predictions
- Purpose: Understanding response verbosity
- Interpretation: Varies by model strategy



## 🔍 Sample Explorer

Analyze individual model predictions and understand performance patterns through our interactive explorer.

### Key Features
- **📊 Compare Submissions**: Select and analyze different model results
- **🏷️ Smart Filtering**: Focus on specific prediction types (faithful_correct, hallucinated, etc.)
- **📖 Context Analysis**: View source passages alongside questions and predictions
- **📈 Detailed Metrics**: Examine F1 scores, exact matches, and prediction categories

### How to Use
1. Submit your model results using the form below
2. Select your submission from the explorer dropdown
3. Filter by prediction type to analyze specific behaviors
4. Browse individual samples to understand model reasoning

## 📬 Contact

### Research Team
- **📧 General Inquiries**: [research@jnana-leaderboard.org](mailto:research@jnana-leaderboard.org)
- **🔬 Technical Support**: [tech-support@jnana-leaderboard.org](mailto:tech-support@jnana-leaderboard.org)  
- **🤝 Collaborations**: [partnerships@jnana-leaderboard.org](mailto:partnerships@jnana-leaderboard.org)

### Community Links
- **💻 GitHub Repository**: [vipplavai/JNANA_leaderboard](https://github.com/vipplavai/JNANA_leaderboard)
- **🐛 Report Issues**: [GitHub Issues](https://github.com/vipplavai/JNANA_leaderboard/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/vipplavai/JNANA_leaderboard/discussions)

## 📜 Citation

If you use the JNANA Telugu QA Leaderboard in your research, please cite:

```bibtex
@dataset{jnana_telugu_qa_2024,
  title={JNANA Telugu QA Leaderboard: A Comprehensive Benchmark for Telugu Question Answering},
  author={Research Team},
  year={2024},
  url={https://github.com/vipplavai/JNANA_leaderboard},
  note={A curated benchmark of 1000 Telugu QA pairs with human annotations}
}
```

---

**🚀 Ready to evaluate your Telugu QA model?** Download the dataset, prepare your results, and submit to join the leaderboard!
