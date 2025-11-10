import torch
import torch.nn as nn

class MLP_AE(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        
        # encode
        z = self.encoder(x)
        # decode
        out = self.decoder(z)
        # reshape
        out = out.view(-1, 1, 28, 28)

        return out