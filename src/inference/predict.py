
import torch
from PIL import Image
import torchvision.transforms as transforms

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]