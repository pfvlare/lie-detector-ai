def get_prediction_label(label, score):
    if label == "NEGATIVE" and score > 0.7:
        return "Lie Detected"
    return "Truth Detected"
