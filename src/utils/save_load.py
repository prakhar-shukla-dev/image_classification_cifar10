import torch

def save_model(model, path="models/cnn_cifar10.pth"):
    torch.save(model.state_dict(), path)

def load_model(model, path="models/cnn_cifar10.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model