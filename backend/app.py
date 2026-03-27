import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import cv2
from inference import load_model, predict

st.set_page_config(page_title="VSR System", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@300;400;600;700;900&display=swap');
:root {
    --bg-primary: #020408;
    --bg-card: #0a1628;
    --accent-cyan: #00f5ff;
    --accent-green: #00ff88;
    --text-dim: #4a7a8a;
    --grid-color: rgba(0, 245, 255, 0.04);
}
html, body, [data-testid="stAppViewContainer"], * {
    background-color: var(--bg-primary) !important;
    color: #e0f7ff !important;
    font-family: 'Roboto Slab', serif !important;
}
[data-testid="stAppViewContainer"] {
    background-image:
        linear-gradient(var(--grid-color) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
    background-size: 40px 40px;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }
.header-banner {
    text-align: center;
    padding: 2rem 0 1.5rem;
    position: relative;
    margin-bottom: 2rem;
}
.header-banner::before {
    content: '';
    position: absolute;
    bottom: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-green), transparent);
}
.header-title {
    font-family: 'Roboto Slab', serif;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 0.2em;
    color: #00ff88;
    filter: drop-shadow(0 0 20px rgba(0, 255, 136, 0.4));
    margin: 0;
}
.box-label {
    font-family: 'Roboto Slab', serif;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.3em;
    margin-bottom: 0.7rem;
    text-transform: uppercase;
}
.output-box {
    background: var(--bg-card);
    border: 1px solid rgba(0,245,255,0.18);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 3px;
    padding: 1.2rem 1.4rem;
    min-height: 140px;
    font-family: 'Roboto Slab', serif;
    font-size: 1.1rem;
    color: var(--accent-cyan);
}
.cursor-blink {
    display: inline-block;
    width: 9px; height: 1rem;
    background: var(--accent-cyan);
    vertical-align: middle;
    margin-left: 4px;
    animation: blink 1s step-end infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.output-placeholder { color: var(--text-dim); font-size: 0.8rem; letter-spacing: 0.1em; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

model = get_model()


st.markdown("""
<div class="header-banner">
    <div class="header-title">VSR : Visual Speech Recognition System</div>
</div>
""", unsafe_allow_html=True)


if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'running' not in st.session_state:
    st.session_state.running = False

col_cam, col_out = st.columns([1.4, 0.9], gap="large")

with col_cam:
    st.markdown('<div class="box-label">// CAMERA FEED</div>', unsafe_allow_html=True)
    cam_placeholder = st.empty()
    pred_placeholder = st.empty()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("▶ START", use_container_width=True):
            st.session_state.running = True
    with btn_col2:
        if st.button("■ STOP", use_container_width=True):
            st.session_state.running = False

with col_out:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown('<div class="box-label">// TRANSCRIPTION OUTPUT</div>', unsafe_allow_html=True)
    output_placeholder = st.empty()


def show_output(prediction):
    if prediction:
        output_placeholder.markdown(f"""
        <div class="output-box">
            {prediction}<span class="cursor-blink"></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        output_placeholder.markdown("""
        <div class="output-box">
            <span class="output-placeholder">AWAITING LIP MOVEMENT DATA...</span>
            <span class="cursor-blink"></span>
        </div>
        """, unsafe_allow_html=True)

show_output(st.session_state.prediction)
if st.session_state.running:
    cap = cv2.VideoCapture(0)

    while st.session_state.running:
        frames_collected = []

        for _ in range(75):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            c = (0, 245, 255)
            L = 22
            for pt, dx, dy in [((10,10),1,1),((w-10,10),-1,1),((10,h-10),1,-1),((w-10,h-10),-1,-1)]:
                x, y = pt
                cv2.line(frame, (x,y), (x+dx*L, y), c, 2)
                cv2.line(frame, (x,y), (x, y+dy*L), c, 2)
            cv2.circle(frame, (w-28, 22), 5, (0,0,220), -1)
            cv2.putText(frame, "REC", (w-20,27), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,180), 1)
            cv2.putText(frame, "LIP DETECTION: ACTIVE", (12, h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0,245,255), 1)

            frames_collected.append(frame.copy())

            if len(frames_collected) % 5 == 0:
                import numpy as np
                rgb = frame[:, :, ::-1]  
                cam_placeholder.image(rgb, channels="RGB", width=700)

        # Predict on collected frames
        if len(frames_collected) > 0:
            result = predict(model, frames_collected)
            st.session_state.prediction = result
            show_output(result)

    cap.release()
    cam_placeholder.empty()