import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from models.mlp_autoencoder import MLP_AE
from utils.dataset import get_dataLoaders


if __name__ == "__main__":
    # Step 1: get one batch from loader
    train_loader, _ = get_dataLoaders(batch_size=8)
    images, _ = next(iter(train_loader))  # images shape: [8,1,28,28]

    # Step 2: create model instance
    model = MLP_AE()

    # Step 3: run forward pass
    outputs = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)

    # checking latent shape
    with torch.no_grad():
        latent = model.encoder(images)
        print("Latent shape:", latent.shape)