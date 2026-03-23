import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2

st.set_page_config(page_title="LipSync AI", layout="wide")

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

/* ── Header ── */
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
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
}
.header-title {
    font-family: 'Roboto Slab', serif;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 0.2em;
    background: linear-gradient(135deg, #00f5ff 0%, #00ff88 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.4));
    margin: 0;
}
.header-sub {
    font-family: 'Roboto Slab', serif;
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 0.4em;
    margin-top: 0.5rem;
}

/* ── Status pills ── */
.status-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 2.5rem;
}
.status-pill {
    font-family: 'Roboto Slab', serif;
    font-size: 0.72rem;
    padding: 0.3rem 1rem;
    border-radius: 2px;
    letter-spacing: 0.1em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.pill-online { border: 1px solid #00ff88; color: #00ff88; background: rgba(0,255,136,0.07); }
.pill-model  { border: 1px solid #00f5ff; color: #00f5ff; background: rgba(0,245,255,0.07); }
.dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.dot-green { background: #00ff88; box-shadow: 0 0 6px #00ff88; animation: pulse 2s infinite; }
.dot-cyan  { background: #00f5ff; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Camera column: center the webrtc widget ── */
.cam-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.box-label {
    font-family: 'Roboto Slab', serif;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.3em;
    margin-bottom: 0.7rem;
    text-transform: uppercase;
}

/* ── Output box ── */
.output-box {
    background: var(--bg-card);
    border: 1px solid rgba(0,245,255,0.18);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 3px;
    padding: 1.2rem 1.4rem;
    min-height: 140px;
    max-width: 420px;
    font-family: 'Roboto Slab', serif;
    font-size: 0.95rem;
    color: var(--accent-cyan);
    position: relative;
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

/* ── Right column layout ── */
.right-col {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    height: 100%;
    padding-top: 2rem;
}

/* webrtc widget sizing */
video { border-radius: 3px; }

#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="header-banner">
    <div class="header-title">LIPSYNC · AI</div>
    <div class="header-sub">VISUAL SPEECH RECOGNITION SYSTEM · V0.1 · DEEP LEARNING</div>
</div>
<div class="status-row">
    <span class="status-pill pill-online"><span class="dot dot-green"></span> SYSTEM ONLINE</span>
    <span class="status-pill pill-model"><span class="dot dot-cyan"></span> MODEL: NOT LOADED</span>
</div>
""", unsafe_allow_html=True)

# ── Layout ──
col_cam, col_out = st.columns([1.4, 0.9], gap="large")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        # corner brackets HUD
        c = (0, 245, 255)
        L = 22
        for pt, dx, dy in [((10,10),1,1),((w-10,10),-1,1),((10,h-10),1,-1),((w-10,h-10),-1,-1)]:
            x, y = pt
            cv2.line(img, (x,y), (x+dx*L, y), c, 2)
            cv2.line(img, (x,y), (x, y+dy*L), c, 2)
        cv2.circle(img, (w-28, 22), 5, (0,0,220), -1)
        cv2.putText(img, "REC", (w-20,27), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,180), 1)
        cv2.putText(img, "LIP DETECTION: ACTIVE", (12, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0,245,255), 1)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_cam:
    st.markdown('<div class="box-label">// CAMERA FEED</div>', unsafe_allow_html=True)
    webrtc_streamer(
        key="lipsync",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        media_stream_constraints={"video": True, "audio": False},
    )

with col_out:
    # push output box to bottom
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown('<div class="box-label">// TRANSCRIPTION OUTPUT</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="output-box">
        <span class="output-placeholder">AWAITING LIP MOVEMENT DATA...</span>
        <span class="cursor-blink"></span>
    </div>
    """, unsafe_allow_html=True)