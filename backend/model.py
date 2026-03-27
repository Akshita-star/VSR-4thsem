import torch
import torch.nn as nn

VOCAB_SIZE   = 51
FRAME_COUNT  = 75
IMG_H, IMG_W = 64, 128

class LipReadingModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE):
        super(LipReadingModel, self).__init__()

    
        self.cnn = nn.Sequential(

            # Layer 1
            nn.Conv3d(1, 32, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            # (B, 32, 75, 16, 32)

            # Layer 2
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
          

            nn.Dropout3d(p=0.2)
        )

        cnn_out_size = 64 * 8 * 16
        self.bilstm = nn.LSTM(
            input_size    = cnn_out_size,
            hidden_size   = 128,           
            num_layers    = 1,             
            batch_first   = True,
            bidirectional = True,          
            dropout       = 0.0           
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, vocab_size)     
        )

    def forward(self, x):
        B = x.size(0)
        x = self.cnn(x)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.contiguous().view(B, FRAME_COUNT, -1)
    
        x, _ = self.bilstm(x)
       
        x = self.classifier(x)
       
        x = x.permute(1, 0, 2)
        return x


if __name__ == '__main__':
    model = LipReadingModel()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 50)
    print("   LIP READING MODEL — Simple Version")
    print("=" * 50)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")

    x   = torch.randn(2, 1, 75, 64, 128)
    out = model(x)
    print(f"  Input  shape    : {list(x.shape)}")
    print(f"  Output shape    : {list(out.shape)}")
    print("=" * 50)

    try:
        from torchinfo import summary
        summary(model, input_size=(2, 1, 75, 64, 128),
                col_names=["input_size", "output_size", "num_params"], depth=3)
    except:
        print(model)