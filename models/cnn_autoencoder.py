import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: Conv -> ReLU -> MaxPool, twice
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )

        # Decoder: Conv -> Upsample, Conv -> Upsample, Conv -> Sigmoid
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),   # (32,7,7) -> (16,7,7)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),   # (16,7,7) -> (16,14,14)

            nn.Conv2d(16, 8, kernel_size=3, padding=1),    # (16,14,14) -> (8,14,14)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),   # (8,14,14) -> (8,28,28)

            nn.Conv2d(8, 1, kernel_size=3, padding=1),     # (8,28,28) -> (1,28,28)
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.encoder(x)     # downsample to (32,7,7)
        x = self.decoder(x)     # upsample back to (1,28,28)
        return x