import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import matplotlib.pyplot as plt

from models.cnn_autoencoder import CNNAutoencoder
from utils.dataset import get_dataLoaders
def get_samples_by_class(dataloader, num_each=2):
    samples = {i: [] for i in range(10)}
    for images, labels in dataloader:
        for img, lbl in zip(images, labels):
            lbl = int(lbl.item())
            if len(samples[lbl]) < num_each:
                samples[lbl].append(img)
        if all(len(v) == num_each for v in samples.values()):
            break
    return samples

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    _, test_loader = get_dataLoaders(batch_size=128)

    model = CNNAutoencoder().to(device)
    model.load_state_dict(torch.load("cnn_autoencoder.pth", map_location=device))
    model.eval()

    samples = get_samples_by_class(test_loader)

    fig, axes = plt.subplots(4, 10, figsize=(14, 6))

    for digit in range(10):
        for j in range(2):
            img = samples[digit][j].to(device).unsqueeze(0)
            with torch.no_grad():
                recon = model(img).cpu().squeeze().numpy()

            # original
            axes[j, digit].imshow(samples[digit][j].squeeze(), cmap="gray")
            axes[j, digit].set_title(str(digit))
            axes[j, digit].axis("off")

            # reconstruction
            axes[j+2, digit].imshow(recon, cmap="gray")
            axes[j+2, digit].axis("off")

    plt.tight_layout()
    out_path = "outputs/cnn_reconstructions.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")