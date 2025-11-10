import sys, os

# Add project root to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import torch
from utils.dataset import get_dataLoaders

# build loaders
train_loader, test_loader = get_dataLoaders(batch_size=16)

# pick first batch
batch = next(iter(train_loader))
images, labels = batch

print(f"Image batch shape -> {images.shape}") # [16, 1, 28, 28]
print(f"Labels batch shape -> {labels.shape}") # [16]

# visualize first 8 images
fig, axes = plt.subplots(2, 8, figsize=(10, 3))

# matplot subplot magic 
# practice this more
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")
    ax.set_title(str(labels[i].item()))
    ax.axis("off")

plt.show()