import streamlit as st
import sound
import os
import pickle
from settings import *
import extract
import librosa
import numpy as np
import lightgbm
from lightgbm import LGBMClassifier

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")
animation_symbol = "❄"
st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    """,
    unsafe_allow_html=True,
)

data = None
sample_rate = None
sound = sound.Sound()

st.title('MACHINE LEARNING PROJECT')
st.header('SPEECH EMOTION DETECTION')
st.image('Flowchart (3).png', caption='End to End Machine Learning Pipeline')
type_select = st.selectbox("Do You want to record audio?",options=option)
if type_select == 'no':
    audio = st.file_uploader("Upload Audio file in WAV format",type=allowed_audio)
if type_select == 'yes':
    btn1 = st.button("Record")
    if btn1:
        with st.spinner(f'Recording for {5} seconds ....'):
            sound.record()
        st.success("Recording completed")
    btn2 = st.button('Play')
    if btn2:
        # sound.play()
        try:
            audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")

Explore = st.button('Explore')
if Explore:
    if type_select == 'no':
        try:
            data, sample_rate = librosa.load(audio, res_type='kaiser_fast')
            extract.create_spectrogram(data, sample_rate)
            extract.create_waveplot(data, sample_rate)
        except:
            st.write('Error')
    if type_select == 'yes':
        try:
            data, sample_rate = librosa.load(WAVE_OUTPUT_FILE, res_type='kaiser_fast')
            extract.create_spectrogram(data, sample_rate)
            extract.create_waveplot(data, sample_rate)
        except:
            st.write('Error')
select = st.selectbox("Model Type: ",options = options)
btn = st.button("PREDICT")
if btn:
    if type_select == 'no':
        model = Models_matrix[select]
        try:
            data, sample_rate = librosa.load(audio, res_type='kaiser_fast')
            data = extract.extract_features(data, sample_rate)
            output = dtree.predict(data)
            if model == LGBM:
                st.text(Emotions[np.argmax(output)])
            else:
                st.text(Emotions[output[0]])
        except:
            st.write('Error')
    if type_select == 'yes':
        model = Models_matrix[select]
        try:
            data, sample_rate = librosa.load(WAVE_OUTPUT_FILE, res_type='kaiser_fast')
            data = extract.extract_features(data, sample_rate)
            output = model.predict(data)
            if model == LGBM:
                st.text(Emotions[np.argmax(output)])
            else:
                st.text(Emotions[output[0]])
        except:
            st.write('Error')
