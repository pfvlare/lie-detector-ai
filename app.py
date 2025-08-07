import streamlit as st
from transformers import pipeline
import torch
import whisper
import tempfile
import os
from utils.helpers import get_prediction_label

# ========== CONFIGURAÇÃO DE PÁGINA ==========
st.set_page_config(page_title="Lie Detector AI", layout="wide")

# ========== FONTE E ESTILO ==========
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

body {
    transition: background-color 0.5s ease;
}
footer { visibility: hidden; }
#rodape {
    position: fixed;
    bottom: 10px;
    width: 100%;
    text-align: center;
    font-size: 0.8rem;
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)

# ========== MODO CLARO/ESCURO ==========
tema = st.sidebar.radio("🧠 Tema", ["🌙 Escuro", "☀️ Claro"])
if tema == "☀️ Claro":
    st.markdown("""
        <style>
            body {
                background-color: #f5f5f5;
                color: #000;
            }
            .stTextInput > div > div > input, .stTextArea textarea {
                background-color: #fff !important;
                color: #000 !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body {
                background-color: #0e1117;
                color: #fafafa;
            }
            .stTextInput > div > div > input, .stTextArea textarea {
                background-color: #1e1e1e !important;
                color: #fff !important;
            }
        </style>
    """, unsafe_allow_html=True)

# ========== LOGO ==========
st.sidebar.image("assets/logo.png", use_container_width=True)

# ========== SIDEBAR ==========
st.sidebar.title("🎯 Detecção de Mentiras")
st.sidebar.markdown("Envie um depoimento em texto ou voz para analisar sinais de possível mentira.")

# ========== TÍTULO ==========
st.markdown("## 🎤 Lie Detector: Verificador de Veracidade por Texto e Voz")
st.markdown("Analise automaticamente se um **texto** ou **áudio** apresenta sinais de possível mentira com IA. Este app utiliza análise de sentimentos para identificar conteúdos **negativos ou suspeitos** com base no tom emocional.")

# ========== MODELO ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if device == "cuda" else -1)

# ========== COLUNAS ==========
col1, col2 = st.columns(2)

# ========== ANÁLISE DE TEXTO ==========
with col1:
    st.subheader("📝 Análise de Texto")
    user_input = st.text_area("Cole ou digite o depoimento aqui:")

    if st.button("Analisar Texto"):
        if user_input.strip() != "":
            result = classifier(user_input)
            if result:
                label = result[0]["label"]
                score = result[0]["score"]
                interpretacao = get_prediction_label(label, score)

                if interpretacao == "Lie Detected":
                    st.error("🔔 Risco de mentira detectado")
                else:
                    st.success("✅ Nenhum sinal de mentira detectado")

                st.json(result[0])
            else:
                st.error("❌ O modelo não retornou resultados.")
        else:
            st.warning("⚠ Digite um texto para analisar.")

# ========== ANÁLISE DE ÁUDIO ==========
with col2:
    st.subheader("🔊 Análise de Áudio")
    st.markdown("Envie um arquivo de áudio (.wav)")
    audio_file = st.file_uploader("Arraste ou selecione um arquivo .wav", type=["wav"])

    if st.button("Analisar Áudio"):
        if audio_file is not None:
            with st.spinner("🎧 Transcrevendo áudio..."):
                model = whisper.load_model("base")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                result = model.transcribe(tmp_path, language="en")
                os.remove(tmp_path)
                texto_transcrito = result["text"]

            st.text_area("🗣 Texto Transcrito:", value=texto_transcrito, height=150)

            if texto_transcrito.strip() != "":
                with st.spinner("🧠 Analisando sentimento..."):
                    result = classifier(texto_transcrito)
                    if result:
                        label = result[0]["label"]
                        score = result[0]["score"]
                        interpretacao = get_prediction_label(label, score)

                        if interpretacao == "Lie Detected":
                            st.error("🔔 Risco de mentira detectado no áudio")
                        else:
                            st.success("✅ Nenhum sinal de mentira no áudio")

                        st.json(result[0])
                    else:
                        st.error("❌ O modelo não retornou resultados.")
        else:
            st.warning("⚠ Por favor, envie um arquivo .wav para continuar.")

# ========== RODAPÉ ==========
st.markdown("""
<div id="rodape">
    © 2025 Dev @ Larissa Campos – <a href="https://github.com/pfvlare" target="_blank">@pfvlare</a>
</div>
""", unsafe_allow_html=True)
