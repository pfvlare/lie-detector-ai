import torch  # <-- ESSENCIAL
from transformers import pipeline

# Pipeline de análise de sentimentos
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # força CPU, importante para ambientes como Streamlit Cloud
)

def analyze_text(text):
    if not text.strip():
        return {"error": "Texto vazio."}
    result = classifier(text)
    return {
        "label": result[0]['label'],
        "score": round(result[0]['score'], 4)
    }
