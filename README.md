# 🧠 Lie Detector AI

> Detecção de possíveis mentiras em **texto** e **áudio** com NLP e análise de voz.

## 🔍 Sobre o Projeto

Este app usa **transformers**, **análise de sentimentos** e **extração de padrões de voz** para analisar depoimentos, detectando indícios de mentira de forma intuitiva, responsiva e visualmente moderna.

---

## 🎯 Funcionalidades

- ✍️ Análise semântica de texto (DistilBERT)
- 🎤 Análise de voz com pyAudioAnalysis
- 📊 Resultados em tempo real
- 🧑‍💻 Interface com Streamlit moderna e acessível

---

## 🛠️ Tecnologias Usadas

- Python 3.11
- [Streamlit](https://streamlit.io/)
- Transformers (`HuggingFace`)
- pyAudioAnalysis
- UX/UI com CSS customizado

---

## 📦 Instalação

```bash
git clone https://github.com/pfvlare/lie-detector-ai.git
cd lie-detector-ai
pip install -r requirements.txt
streamlit run app.py
