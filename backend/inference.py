import cv2
import torch
import numpy as np
from model import LipReadingModel
from dataset import GRID_VOCAB, idx2word, FRAME_COUNT, IMG_H, IMG_W
import os

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    FACE_MESH = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                       refine_landmarks=True,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[Warning] MediaPipe not found. Using fixed crop fallback.")

LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                 95, 185, 40, 39, 37, 0, 267, 269, 270, 409]

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'best_model.pt')
VOCAB_SIZE  = len(GRID_VOCAB)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = LipReadingModel(vocab_size=VOCAB_SIZE).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def extract_lip_mediapipe(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = FACE_MESH.process(rgb)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        xs = [int(lm[i].x * w) for i in LIP_LANDMARKS]
        ys = [int(lm[i].y * h) for i in LIP_LANDMARKS]

        x1, x2 = max(min(xs) - 10, 0), min(max(xs) + 10, w)
        y1, y2 = max(min(ys) - 10, 0), min(max(ys) + 10, h)

        lip = frame_bgr[y1:y2, x1:x2]
        if lip.size == 0:
            return None
        return lip

    return None

def extract_lip_fixed(frame_bgr):
    h, w = frame_bgr.shape[:2]
    return frame_bgr[int(h*0.55):int(h*0.85), int(w*0.25):int(w*0.75)]

#
def preprocess_frames(frames):
    processed = []
    for frame in frames:
        lip = None
        if MEDIAPIPE_AVAILABLE:
            lip = extract_lip_mediapipe(frame)
        if lip is None:
            lip = extract_lip_fixed(frame)

        gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_W, IMG_H))
        gray = gray.astype(np.float32) / 255.0
        gray = (gray - 0.5) / 0.5
        processed.append(gray)

    while len(processed) < FRAME_COUNT:
        processed.append(processed[-1])
    processed = processed[:FRAME_COUNT]

    tensor = torch.tensor(np.array(processed)).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

def ctc_greedy_decode(output):
    pred = output.argmax(dim=2).permute(1, 0)
    results = []
    for seq in pred:
        decoded = []
        prev = -1
        for token in seq:
            t = token.item()
            if t != prev and t != 0:
                decoded.append(idx2word.get(t, '?'))
            prev = t
        results.append(' '.join(decoded))
    return results

def predict(model, frames):
    if len(frames) == 0:
        return "NO FRAMES"
    tensor = preprocess_frames(frames)
    with torch.no_grad():
        output = model(tensor)
        result = ctc_greedy_decode(output.cpu())
    return result[0].upper() if result[0] else "..."