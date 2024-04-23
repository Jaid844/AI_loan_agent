from audio_recorder_streamlit import audio_recorder
import streamlit as st
import soundfile as sf


st.title("Call recorder")
recorded_audio=audio_recorder()

