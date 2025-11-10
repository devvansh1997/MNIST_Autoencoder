import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import matplotlib.pyplot as plt
from models.mlp_autoencoder import MLP_AE
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

    model = MLP_AE().to(device)
    model.load_state_dict(torch.load("mlp_autoencoder.pth", map_location=device))
    model.eval()

    samples = get_samples_by_class(test_loader, num_each=2)

    fig, axes = plt.subplots(4, 10, figsize=(14, 6))

    # first two rows = originals
    # last two rows = reconstructions

    for digit in range(10):
        for j in range(2):
            img = samples[digit][j].to(device).unsqueeze(0)   # (1,1,28,28)
            with torch.no_grad():
                recon = model(img).cpu().squeeze().numpy()

            axes[0 + j, digit].imshow(samples[digit][j].squeeze(), cmap="gray")
            axes[0 + j, digit].set_title(str(digit))

            axes[2 + j, digit].imshow(recon, cmap="gray")

            axes[0 + j, digit].axis("off")
            axes[2 + j, digit].axis("off")

    plt.tight_layout()
    out_path = "outputs/mlp_reconstructions.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
