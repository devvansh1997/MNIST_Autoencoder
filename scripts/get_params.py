import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.cnn_autoencoder import CNNAutoencoder
from models.mlp_autoencoder import MLP_AE

# parameter counts
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Instantiate models (no need for device)
    mlp = MLP_AE()
    cnn = CNNAutoencoder()

    print("\n=== MLP Autoencoder Parameters ===")
    print(f"Encoder params: {count_params(mlp.encoder):,}")
    print(f"Decoder params: {count_params(mlp.decoder):,}")
    print(f"Total params:   {count_params(mlp):,}")

    print("\n=== CNN Autoencoder Parameters ===")
    print(f"Encoder params: {count_params(cnn.encoder):,}")
    print(f"Decoder params: {count_params(cnn.decoder):,}")
    print(f"Total params:   {count_params(cnn):,}")