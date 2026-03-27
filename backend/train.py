import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_dataloaders, GRID_VOCAB, idx2word
from model import LipReadingModel
DATA_ROOT  = r'C:\Users\lenovo\Desktop\PROJECTS\2year\4sem\VSR-DL\VSR-4thsem\data'
BATCH_SIZE = 10
EPOCHS     = 20
LR         = 1e-3
VOCAB_SIZE = len(GRID_VOCAB)
SAVE_DIR   = 'checkpoints'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(SAVE_DIR, exist_ok=True)

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


def train():
    print(f"Device     : {DEVICE}")
    print(f"Vocab size : {VOCAB_SIZE}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print()

    # Data
    train_loader, val_loader = get_dataloaders(
        data_root  = DATA_ROOT,
        batch_size = BATCH_SIZE,
        speakers   = ['s1']
    )

    # Model
    model     = LipReadingModel(vocab_size=VOCAB_SIZE).to(DEVICE)
    total_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_p:,}\n")

    # Loss + Optimizer + Scheduler
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):

   
        model.train()
        train_loss = 0.0

        for batch_idx, (frames, labels, label_lens) in enumerate(train_loader):
            frames     = frames.to(DEVICE)
            labels     = labels.to(DEVICE)
            label_lens = label_lens.to(DEVICE)

            optimizer.zero_grad()

            output     = model(frames)   
            input_lens = torch.full((frames.size(0),), 75, dtype=torch.long).to(DEVICE)

            loss = ctc_loss(output.log_softmax(2), labels, input_lens, label_lens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

           
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train = train_loss / len(train_loader)

     
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for frames, labels, label_lens in val_loader:
                frames     = frames.to(DEVICE)
                labels     = labels.to(DEVICE)
                label_lens = label_lens.to(DEVICE)

                output     = model(frames)
                input_lens = torch.full((frames.size(0),), 75, dtype=torch.long).to(DEVICE)

                loss = ctc_loss(output.log_softmax(2), labels, input_lens, label_lens)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(f"\nEpoch [{epoch:2d}/{EPOCHS}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

   
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'val_loss'   : best_val_loss
            }, os.path.join(SAVE_DIR, 'best_model.pt'))
            print(f"   Best model saved! Val Loss: {best_val_loss:.4f}")

      
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                frames, _, _ = next(iter(val_loader))
                out   = model(frames.to(DEVICE))
                preds = ctc_greedy_decode(out.cpu())
                print(f"  Sample predictions: {preds[:2]}")

        print("-" * 55)

    print("\nTraining complete!")
    print(f"Best model: {SAVE_DIR}/best_model.pt")


if __name__ == '__main__':
    train()