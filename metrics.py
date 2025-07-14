from typing import List, Dict

def compute_metrics(data: List[Dict]) -> Dict:
    total = len(data)
    if total == 0:
        return {
            "total": 0,
            "em": 0.0,
            "f1": 0.0,
            "answered": 0.0,
            "hallucinated": 0.0,
            "faithful_correct": 0.0,
            "faithful_incorrect": 0.0,
            "empty": 0.0,
            "faa": 0.0,
            "f1_em_gap": 0.0,
            "overconfident_em": 0.0,
            "robust_answer_rate": 0.0,
            "avg_answer_length": 0.0
        }

    em = sum(1 for d in data if d.get("exact_match")) / total * 100
    f1 = sum(d.get("f1_score", 0) for d in data) / total
    answered = sum(1 for d in data if d.get("answerable")) / total * 100
    hallucinated = sum(1 for d in data if d.get("hallucinated")) / total * 100
    faithful_correct = sum(1 for d in data if d.get("type") == "faithful_correct") / total * 100
    faithful_incorrect = sum(1 for d in data if d.get("type") == "faithful_incorrect") / total * 100
    empty = sum(1 for d in data if d.get("type") == "empty") / total * 100
    faa = faithful_correct  # Faithfulness-Adjusted Accuracy

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
