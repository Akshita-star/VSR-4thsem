import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LipReadingModel
from dataset import GRID_VOCAB, get_dataloaders, idx2word
from jiwer import wer, cer
from torchinfo import summary

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'best_model.pt')
DATA_ROOT  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model load ──
model = LipReadingModel(vocab_size=len(GRID_VOCAB))
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()
model.to(DEVICE)

# ── CTC Greedy Decode ──
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

# ── Word Accuracy ──
def word_accuracy(preds, targets):
    correct = 0
    total = len(preds)
    for p, t in zip(preds, targets):
        if p.strip() == t.strip():
            correct += 1
    
    print("\n── Sample Predictions ──")
    for i in range(min(5, len(preds))):
        print(f"  Pred   : {preds[i]}")
        print(f"  Target : {targets[i]}")
        print()
    
    return (correct / total * 100) if total > 0 else 0

# ── Load val data ──
print("Loading validation data...")
_, val_loader = get_dataloaders(data_root=DATA_ROOT, batch_size=4, speakers=['s1'])

all_preds   = []
all_targets = []

print("Running inference on validation set...")
with torch.no_grad():
    for frames, labels, label_lens in val_loader:
        frames = frames.to(DEVICE)
        output = model(frames)
        preds  = ctc_greedy_decode(output.cpu())

        idx = 0
        for length in label_lens:
            target_words = [idx2word.get(labels[idx+i].item(), '?') for i in range(length)]
            all_targets.append(' '.join(target_words))
            idx += length
        all_preds.extend(preds)

acc = word_accuracy(all_preds, all_targets)

# ── Print Summary ──
word_error_rate = wer(all_targets, all_preds)
char_error_rate = cer(all_targets, all_preds)

print("=" * 55)
print("   VSR — VISUAL LIP READING MODEL SUMMARY")
print("=" * 55)
print(f"  Best Val Loss     : {checkpoint['val_loss']:.4f}")
print(f"  Best Epoch        : {checkpoint['epoch']}")
print(f"  Device            : {DEVICE}")
print(f"  Dataset           : GRID Corpus (s1 — 1000 samples)")
print(f"  Train/Val Split   : 800 / 200")
print(f"  WER (Word Error Rate)  : {word_error_rate*100:.2f}%")
print(f"  CER (Char Error Rate)  : {char_error_rate*100:.2f}%")
print(f"  Word Accuracy          : {(1-word_error_rate)*100:.2f}%")
print("=" * 55)

summary(model, input_size=(2, 1, 75, 64, 128),
        col_names=["input_size", "output_size", "num_params"], depth=4)