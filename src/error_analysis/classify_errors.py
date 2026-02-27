# error_analysis/classify_errors.py
def classify_fp(pred):
    return "background_fp"

def classify_fn(gt):
    return "missed_object"

def classify_tp(pred, gt):
    if pred["class_id"] != gt["class_id"]:
        return "wrong_class"
    return "correct"
