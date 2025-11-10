import wandb
import yaml
import torch
from utils.dataset import get_dataLoaders

# loading configs
# with open("./config/default.yaml") as f:
#     cfg = yaml.safe_load(f)
#     print("config file - loaded")

# if cfg['log_wandb']:
#     wandb.init(
#         project="mnist_autoencoders",
#         config=cfg
#     )

train_loader, test_loader = get_dataLoaders(batch_size=128)
print("Dataset Loaded")