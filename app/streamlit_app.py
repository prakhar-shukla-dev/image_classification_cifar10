
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from src.models.cnn_model import CNN

# Classes
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# Load model
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("models/cnn_cifar10.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# UI
st.title("🧠 Image Classifier (CIFAR-10)")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: {classes[predicted.item()]}")