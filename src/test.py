import cv2
import torch
from ultralytics import YOLO  # Ensure you have the Ultralytics YOLO library installed
import os
import json
from torchvision import transforms
from Classifier.classifier import CNNWithAttention
from PIL import Image
import torchvision.models as models
import torch.nn as nn 


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
# model = models.vgg16(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False

# input_features = model.classifier[0].in_features
# model.classifier = nn.Sequential(
#     nn.Linear(input_features, 256),
#     nn.ReLU(),
#     nn.Dropout(p=0.6),
#     nn.Linear(256, 1),
# )


checkpoint = torch.load('./src/Classifier/Models/checkpoint_4_try.pth')
classifier_model.load_state_dict(checkpoint['model_state_dict'])
classifier_model.to(device)
#model.classifier.to(device)

# Carica l'epoca (opzionale, per riprendere l'addestramento)
epoch = checkpoint['epoch']

transform= transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# Carica l'immagine da un percorso
image_path = './src/Classifier/Datasets/training_set/CAM10-2014-04-17-20140417133814-20140417134402-tarid301-frame2402-line1.jpg'  # Sostituisci con il percorso dell'immagine
img = load_image(image_path)
img=img.to(device)
gender, bag, hat = classifier_model(img)


gender_pred = torch.sigmoid(gender)
bag_pred = torch.sigmoid(bag) 
hat_pred = torch.sigmoid(hat) 

print("Gender: ", gender_pred)
print("Bag: ", bag_pred)
print("Hat: ", hat_pred)

