
import hashlib
import json
from typing import List, Dict, Tuple

REQUIRED_FIELDS = {
    "content_id", "qa_index", "question", "gold_answer", "prediction",
    "exact_match", "f1_score", "answerable", "hallucinated", "type"
}

VALID_TYPES = {"faithful_correct", "faithful_incorrect", "hallucinated", "empty"}

def clean_and_validate_submission(data: List[Dict]) -> Tuple[List[Dict], List[str], str]:
    """
    Clean and validate submission data.
    Returns: (cleaned_data, errors, sha1_hash)
    """
    errors = []
    cleaned_data = []
    
    for i, record in enumerate(data):
        # Check required fields
        missing_fields = REQUIRED_FIELDS - set(record.keys())
        if missing_fields:
            errors.append(f"Record {i}: Missing fields {missing_fields}")
            continue
            
        # Clean and validate each record
        cleaned_record = {}
        
        # Handle content_id and qa_index
        try:
            cleaned_record["content_id"] = int(record["content_id"])
            cleaned_record["qa_index"] = int(record["qa_index"])
        except (ValueError, TypeError):
            errors.append(f"Record {i}: content_id and qa_index must be integers")
            continue
            
        # Handle string fields
        for field in ["question", "gold_answer", "prediction"]:
            cleaned_record[field] = str(record[field])
            
        # Handle boolean fields with type coercion
        for field in ["exact_match", "answerable", "hallucinated"]:
            value = record[field]
            if isinstance(value, bool):
                cleaned_record[field] = value
            elif isinstance(value, str):
                cleaned_record[field] = value.lower() in ("true", "1", "yes")
            elif isinstance(value, int):
                cleaned_record[field] = bool(value)
            else:
                errors.append(f"Record {i}: {field} must be boolean-like")
                continue
                
        # Handle f1_score
        try:
            cleaned_record["f1_score"] = float(record["f1_score"])
        except (ValueError, TypeError):
            errors.append(f"Record {i}: f1_score must be a number")
            continue
            
        # Handle type field
        if record["type"] not in VALID_TYPES:
            errors.append(f"Record {i}: type must be one of {VALID_TYPES}")
            continue
        cleaned_record["type"] = record["type"]
        
        cleaned_data.append(cleaned_record)
    
    # Generate SHA1 hash
    sorted_data = sorted(cleaned_data, key=lambda x: (x["content_id"], x["qa_index"]))
    data_str = json.dumps(sorted_data, sort_keys=True)
    sha1_hash = hashlib.sha1(data_str.encode()).hexdigest()
    
    return cleaned_data, errors, sha1_hash
