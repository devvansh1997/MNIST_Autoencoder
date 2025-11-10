import torch

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()          # put model in training mode
    total_loss = 0.0

    for images, _ in dataloader:
        images = images.to(device)

        # forward pass
        outputs = model(images)

        # compute reconstruction loss
        loss = loss_fn(outputs, images)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    for images, _ in dataloader:
        images = images.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, images)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)