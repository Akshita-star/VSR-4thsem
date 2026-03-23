import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─── Config ───────────────────────────────────────────────────────────────────
FRAME_COUNT  = 75       # fixed frames per video
IMG_H, IMG_W = 64, 128  # lip region size
DATA_ROOT    = "data"   # root folder jahan s1, s2... folders hain

# GRID ke 51 words ka vocabulary
GRID_VOCAB = [
    '<blank>',  # CTC blank token (index 0)
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


# ─── Utility: Video → Lip Frames ──────────────────────────────────────────────
def video_to_frames(video_path):
    """
    Video file se frames extract karo aur lip region crop karo.
    Returns: numpy array of shape (FRAME_COUNT, IMG_H, IMG_W)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Grayscale convert
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Lip region crop — GRID videos mein face centered hota hai
        # Approximate lip region (lower half of face center)
        h, w = gray.shape
        lip = gray[int(h*0.55):int(h*0.85), int(w*0.25):int(w*0.75)]
        lip = cv2.resize(lip, (IMG_W, IMG_H))
        frames.append(lip)

    cap.release()

    # Pad ya trim to fixed FRAME_COUNT
    if len(frames) < FRAME_COUNT:
        # Repeat last frame to pad
        while len(frames) < FRAME_COUNT:
            frames.append(frames[-1] if frames else np.zeros((IMG_H, IMG_W)))
    else:
        frames = frames[:FRAME_COUNT]

    frames = np.array(frames, dtype=np.float32)   # (75, 64, 128)
    frames = frames / 255.0                         # normalize 0-1
    frames = (frames - 0.5) / 0.5                  # normalize -1 to 1
    return frames


# ─── Utility: Align file → Label ──────────────────────────────────────────────
def parse_align(align_path):
    """
    GRID .align file padhke words ki list return karo.
    Format: start_time end_time word
    Example line: 0 23750 sil  (sil = silence, ignore karo)
    """
    words = []
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                word = parts[2].lower()
                if word not in ('sil', 'sp'):   # silence skip
                    words.append(word)
    return words


def words_to_tensor(words):
    """Word list ko index tensor mein convert karo."""
    indices = [word2idx.get(w, 0) for w in words]
    return torch.tensor(indices, dtype=torch.long)


# ─── Dataset Class ────────────────────────────────────────────────────────────
class GRIDDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, speakers=None, split='train'):
        """
        data_root : 'data/' folder path
        speakers  : list like ['s1','s2'] — None matlab sab speakers
        split     : 'train' ya 'val'
        """
        self.samples = []   # list of (video_path, align_path)

        all_speakers = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and d.startswith('s')
        ])

        if speakers:
            all_speakers = [s for s in all_speakers if s in speakers]

        # 80/20 train-val split
        split_idx = int(len(all_speakers) * 0.8)
        if split == 'train':
            use_speakers = all_speakers[:split_idx]
        else:
            use_speakers = all_speakers[split_idx:]

        for spk in use_speakers:
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

        print(f"[Dataset] {split} — {len(self.samples)} samples loaded from {len(use_speakers)} speakers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_path, align_path = self.samples[idx]

        frames = video_to_frames(vid_path)             # (75, 64, 128)
        frames = torch.tensor(frames).unsqueeze(0)     # (1, 75, 64, 128) — channel dim add

        words  = parse_align(align_path)
        label  = words_to_tensor(words)                # (num_words,)

        return frames, label, len(label)


# ─── Collate Function (variable length labels handle karne ke liye) ────────────
def collate_fn(batch):
    frames, labels, label_lens = zip(*batch)

    frames     = torch.stack(frames)                          # (B, 1, 75, 64, 128)
    label_lens = torch.tensor(label_lens, dtype=torch.long)
    labels     = torch.cat(labels)                            # flat for CTC

    return frames, labels, label_lens


# ─── DataLoader helper ────────────────────────────────────────────────────────
def get_dataloaders(data_root=DATA_ROOT, batch_size=8, speakers=None):
    train_ds = GRIDDataset(data_root, speakers, split='train')
    val_ds   = GRIDDataset(data_root, speakers, split='val')

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    return train_loader, val_loader


# ─── Quick test (dataset download ke bina bhi test ho sake) ───────────────────
if __name__ == '__main__':
    print("Vocabulary size:", len(GRID_VOCAB))
    print("Sample words:", GRID_VOCAB[1:10])
    print()

    # Fake data se test karo
    fake_frames = np.random.rand(FRAME_COUNT, IMG_H, IMG_W).astype(np.float32)
    fake_tensor = torch.tensor(fake_frames).unsqueeze(0)
    print("Fake frame tensor shape:", fake_tensor.shape)
    # Expected: torch.Size([1, 75, 64, 128])

    fake_words = ['bin', 'blue', 'at', 'f', 'two', 'now']
    label = words_to_tensor(fake_words)
    print("Fake label tensor:", label)
    print("Decoded back:", [idx2word[i.item()] for i in label])