import streamlit as st
from analysis.text_analyzer import analyze_text
from analysis.audio_analyzer import analyze_audio
from utils.helpers import get_prediction_label

st.set_page_config(
    page_title="Lie Detector AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("assets/logo.png", use_column_width=True)
st.sidebar.markdown("## 🎯 Detecção de Mentiras")
st.sidebar.markdown("Envie um depoimento em texto ou voz para analisar sinais de possível mentira.")

# Título principal
st.markdown("## 🎤 Lie Detector: Verificador de Veracidade por Texto e Voz")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Análise de Texto")
    user_text = st.text_area("Cole ou digite o depoimento aqui:", height=200)
    if st.button("Analisar Texto"):
        with st.spinner("Analisando..."):
            result = analyze_text(user_text)
            label = get_prediction_label(result)
            st.success(f"Resultado: {label}")
            st.json(result)

with col2:
    st.markdown("### 🔊 Análise de Áudio")
    audio_file = st.file_uploader("Envie um arquivo de áudio (.wav)", type=["wav"])
    if audio_file and st.button("Analisar Áudio"):
        with st.spinner("Analisando áudio..."):
            result = analyze_audio(audio_file)
            label = get_prediction_label(result)
            st.success(f"Resultado: {label}")
            st.json(result)

st.markdown("---")
st.caption("🧠 Este app utiliza modelos de NLP e análise de áudio para identificar sinais de possíveis mentiras.")
