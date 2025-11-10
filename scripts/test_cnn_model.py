import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from models.cnn_autoencoder import CNNAutoencoder
from utils.dataset import get_dataLoaders

if __name__ == "__main__":
    model = CNNAutoencoder()
    train_loader, _ = get_dataLoaders(batch_size=8)

    images, _ = next(iter(train_loader))

    outputs = model(images)

    print("Input:", images.shape)      # expect [8,1,28,28]
    print("Output:", outputs.shape)    # expect [8,1,28,28]

    # Check latent shape
    with torch.no_grad():
        latent = model.encoder(images)
    print("Latent:", latent.shape)      # expect [8,32,7,7]
