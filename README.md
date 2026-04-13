# 🧠 Image Classification using CNN (CIFAR-10)

## 📌 Overview
This project implements an Image Classification model using Convolutional Neural Networks (CNN) in PyTorch. The model is trained on the CIFAR-10 dataset and can classify images into 10 categories.

## 🚀 Features
- Built using PyTorch
- Achieved **76.5% accuracy**
- Implemented data preprocessing and normalization
- Developed training & evaluation pipeline
- Deployed using Streamlit for real-time predictions

## 🧠 Classes
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## 🛠️ Tech Stack
- Python
- PyTorch
- Torchvision
- Streamlit
- Scikit-learn

## 📂 Project Structure
image_classification_cifar10/
│── src/
│── app/
│── models/
│── main.py
│── requirements.txt


## ▶️ How to Run

### Install dependencies
```bash
pip install -r requirements.txt

Train model:
python main.py

Run app: 
python -m streamlit run app/streamlit_app.py

📊 Results
Test Accuracy: 76.52%
Model: CNN (3 Conv layers + FC)
📸 Demo

(Add screenshots here)

🔮 Future Improvements
Use ResNet for better accuracy
Add confidence score
Deploy on cloud