
from src.data.dataloader import get_dataloaders
from src.models.cnn_model import CNN
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.utils.save_load import save_model

def main():
    trainloader, testloader = get_dataloaders()

    model = CNN()

    train_model(model, trainloader, epochs=10)
    evaluate_model(model, testloader)

    save_model(model)

if __name__ == "__main__":
    main()