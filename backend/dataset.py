import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    FACE_MESH = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                       refine_landmarks=True,
                                       min_detection_confidence=0.5)
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[Warning] MediaPipe not found. Using fixed crop fallback.")

FRAME_COUNT  = 75
IMG_H, IMG_W = 64, 128
DATA_ROOT    = r'C:\Users\lenovo\Desktop\PROJECTS\2year\4sem\VSR-DL\VSR-4thsem\data'

GRID_VOCAB = [
    '<blank>',
    'bin', 'blue', 'green', 'red', 'white',
    'at', 'by', 'in', 'with',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z',
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine',
    'again', 'now', 'please', 'soon'
]

word2idx = {w: i for i, w in enumerate(GRID_VOCAB)}
idx2word = {i: w for i, w in enumerate(GRID_VOCAB)}

LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                 95, 185, 40, 39, 37, 0, 267, 269, 270, 409]

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
    lip = frame_bgr[int(h*0.55):int(h*0.85), int(w*0.25):int(w*0.75)]
    return lip

def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lip = None
        if MEDIAPIPE_AVAILABLE:
            lip = extract_lip_mediapipe(frame)
        if lip is None:
            lip = extract_lip_fixed(frame)

        gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_W, IMG_H))
        frames.append(gray)

    cap.release()

    if len(frames) == 0:
        return np.zeros((FRAME_COUNT, IMG_H, IMG_W), dtype=np.float32)

    while len(frames) < FRAME_COUNT:
        frames.append(frames[-1])
    frames = frames[:FRAME_COUNT]

    frames = np.array(frames, dtype=np.float32)
    frames = frames / 255.0
    frames = (frames - 0.5) / 0.5
    return frames

def parse_align(align_path):
    words = []
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                word = parts[2].lower()
                if word not in ('sil', 'sp'):
                    words.append(word)
    return words


def words_to_tensor(words):
    indices = [word2idx.get(w, 0) for w in words]
    return torch.tensor(indices, dtype=torch.long)


class GRIDDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, speakers=None):
        self.samples = []

        all_speakers = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and d.startswith('s')
        ])

        if speakers:
            all_speakers = [s for s in all_speakers if s in speakers]

        for spk in all_speakers:
            video_dir = os.path.join(data_root, spk, 'video')
            align_dir = os.path.join(data_root, spk, 'align')

            if not os.path.exists(video_dir):
                continue

            for fname in os.listdir(video_dir):
                if fname.endswith('.mpg'):
                    vid_path   = os.path.join(video_dir, fname)
                    align_path = os.path.join(align_dir, fname.replace('.mpg', '.align'))
                    if os.path.exists(align_path):
                        self.samples.append((vid_path, align_path))

        print(f"[Dataset] Total samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_path, align_path = self.samples[idx]
        frames = video_to_frames(vid_path)
        frames = torch.tensor(frames).unsqueeze(0)
        words  = parse_align(align_path)
        label  = words_to_tensor(words)
        return frames, label, len(label)


def collate_fn(batch):
    frames, labels, label_lens = zip(*batch)
    frames     = torch.stack(frames)
    label_lens = torch.tensor(label_lens, dtype=torch.long)
    labels     = torch.cat(labels)
    return frames, labels, label_lens

def get_dataloaders(data_root=DATA_ROOT, batch_size=4, speakers=None):
    full_ds = GRIDDataset(data_root, speakers)

    total      = len(full_ds)
    val_size   = int(total * 0.2)
    train_size = total - val_size

    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    print(f"[Dataset] Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    return train_loader, val_loader

if __name__ == '__main__':
    print("Vocabulary size:", len(GRID_VOCAB))
    print("MediaPipe available:", MEDIAPIPE_AVAILABLE)

    fake_frames = np.random.rand(FRAME_COUNT, IMG_H, IMG_W).astype(np.float32)
    fake_tensor = torch.tensor(fake_frames).unsqueeze(0)
    print("Fake frame tensor shape:", fake_tensor.shape)

    fake_words = ['bin', 'blue', 'at', 'f', 'two', 'now']
    label = words_to_tensor(fake_words)
    print("Decoded back:", [idx2word[i.item()] for i in label])
    print()

    print("Testing real data...")
    train_loader, val_loader = get_dataloaders(speakers=['s1'])
    frames, labels, lens = next(iter(train_loader))
    print("Batch frames shape:", frames.shape)
    print("Labels:", labels)