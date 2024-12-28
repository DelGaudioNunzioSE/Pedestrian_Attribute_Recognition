import cv2
import torch
from ultralytics import YOLO  # Ensure you have the Ultralytics YOLO library installed
import os
import json
from torchvision import transforms
from Classifier.firstClassifier import CNNWithAttention
from PIL import Image

def load_image(image_path):
    # Carica l'immagine usando PIL
    img = Image.open(image_path).convert('RGB')  # Converti in RGB nel caso di immagini in scala di grigi
    img = transform(img)  # Applica le trasformazioni
    img = img.unsqueeze(0)  # Aggiungi una dimensione per il batch (1 immagine)
    return img

if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
        device = torch.device("mps")
else:
        device = torch.device("cpu")

print(f"Using device: {device}")

classifier_model = CNNWithAttention()   
classifier_model.to(device)
checkpoint = torch.load('./src/Classifier/Models/checkpoint.pth')
classifier_model.load_state_dict(checkpoint['model_state_dict'])

# Carica l'epoca (opzionale, per riprendere l'addestramento)
epoch = checkpoint['epoch']
losses = checkpoint['losses']  # Opzionale, per riprendere le perditclassifier_model.eval()

transform = transforms.Compose([transforms.ToTensor(),
				transforms.Resize((224, 224))])

# Carica l'immagine da un percorso
image_path = './src/Classifier/Datasets/validation_set/CAM17-2014-02-20-20140220171450-20140220172046-tarid93-frame1788-line1.jpg'  # Sostituisci con il percorso dell'immagine
img = load_image(image_path)
img=img.to(device)
gender, hat, bag = classifier_model(img)


gender_pred = torch.sigmoid(gender) > 0.5
hat_pred = torch.sigmoid(hat) > 0.5
bag_pred = torch.sigmoid(bag) > 0.5

print("Gender: ", gender_pred)
print("Hat: ", hat_pred)
print("Bag: ", bag_pred)
