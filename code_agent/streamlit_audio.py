from audio_recorder_streamlit import audio_recorder
import streamlit as st
import soundfile as sf
from langchain_voyageai.embeddings import VoyageAIEmbeddings

st.title("Call recorder")
recorded_audio=audio_recorder()
if recorded_audio:
    st.write(recorded_audio)

