def get_prediction_label(result):
    if "label" in result:
        if result["label"] == "NEGATIVE":
            return "⚠️ Risco de mentira detectado"
        else:
            return "✅ Provavelmente verdadeiro"
    return "ℹ️ Análise inconclusiva"
