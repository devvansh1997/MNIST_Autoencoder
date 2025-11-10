import yaml
import wandb
import torch

from utils.dataset import get_dataLoaders
from utils.train import train_one_epoch, evaluate
from models.mlp_autoencoder import MLP_AE
from models.cnn_autoencoder import CNNAutoencoder


# -------------------------
# Load config
# -------------------------
with open("./configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

# wandb setup
if cfg.get("log_wandb", False):
    wandb.init(
        project=cfg["wandb_project"],
        name=cfg["wandb_run_name"],
        config=cfg
    )

# -------------------------
# Device selection
# -------------------------
if cfg["device"] == "cpu":
    device = "cpu"
elif cfg["device"] == "mps":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    # auto
    device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# -------------------------
# Dataloaders
# -------------------------
train_loader, test_loader = get_dataLoaders(batch_size=cfg["batch_size"])
print("--- Dataset Loaded ---")

# -------------------------
# Model selection
# -------------------------
if cfg["model_type"] == "mlp":
    model = MLP_AE().to(device)
elif cfg["model_type"] == "cnn":
    model = CNNAutoencoder().to(device)
else:
    raise ValueError("Unknown model type")

# Loss + Optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

# -------------------------
# Training Loop
# -------------------------
for epoch in range(cfg["epochs"]):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    test_loss = evaluate(model, test_loader, loss_fn, device)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # wandb logging
    if cfg.get("log_wandb", False):
        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch": epoch + 1
        })

# -------------------------
# Save model
# -------------------------
save_name = f"{cfg['model_type']}_autoencoder.pth"
torch.save(model.state_dict(), save_name)
print(f"Saved model as {save_name}")
