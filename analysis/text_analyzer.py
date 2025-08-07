from transformers import pipeline

# Inicializa o pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_text(text):
    if not text.strip():
        return {"error": "Texto vazio."}
    result = classifier(text)
    return {
        "label": result[0]['label'],
        "score": round(result[0]['score'], 4)
    }
