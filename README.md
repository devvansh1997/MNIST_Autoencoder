# MNIST Autoencoders (MLP vs CNN)

This project implements and compares two autoencoder architectures for the MNIST dataset:
1. **Fully Connected Autoencoder (MLP)**
2. **Convolutional Autoencoder (CNN)**

Both models are trained to reconstruct handwritten digit images, and the goal is to study:
- Reconstruction performance  
- Parameter efficiency  
- Impact of spatial structure (CNN) vs flattened input (MLP)  
- Loss curves and convergence behavior  

This project was developed with modular, easy-to-understand PyTorch code, full configuration control through YAML, and experiment tracking via Weights & Biases (W&B).

---

## ✅ Project Structure

│
├── config/
│ └── default.yaml # configuration file for model, training, wandb
│
├── models/
│ ├── mlp_autoencoder.py # fully connected autoencoder
│ └── cnn_autoencoder.py # convolutional autoencoder
│
├── utils/
│ ├── dataset.py # MNIST dataloaders
│ └── train.py # training + evaluation loops
│
├── scripts/
│ ├── verify_data.py # sanity check for data loading
│ ├── test_mlp_model.py # shape test for MLP AE
│ ├── test_cnn_model.py # shape test for CNN AE
│ ├── reconstruct_mlp.py # generate reconstructions (MLP)
│ ├── reconstruct_cnn.py # generate reconstructions (CNN)
│ └── get_params.py # parameter count script
│
├── outputs/
│ ├── mlp_reconstructions.png
│ ├── cnn_reconstructions.png
│ ├── train_loss.png
│ └── test_loss.png
│
├── main.py # entry point for training
└── README.md # this file


---

## ✅ Features

- Modular PyTorch implementation  
- MLP + CNN architectures  
- YAML-based configuration  
- W&B experiment logging  
- Shape-testing scripts  
- Reconstruction scripts for report  
- Parameter comparison utility  
- CPU / MPS auto-selection  
- Clean training loop+evaluation separation  

---

## ✅ Installation

Clone the repo:

```bash
git clone git@github.com:devvansh1997/MNIST_Autoencoder.git
cd MNIST_Autoencoder
```

Install Dependencies:
```bash
pip install torch torchvision wandb pyyaml matplotlib
```

and login to wandb:
```bash
wandb login
```