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
import numpy as np
from torchvision.transforms import functional as F


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

classifier_model = CNNWithAttention(hidden_dim=512)
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


checkpoint = torch.load('./src/Classifier/Models/HistogramEqualization_512_neurons_7_01_0818.pth')
classifier_model.load_state_dict(checkpoint['model_state_dict'])
classifier_model.to(device)
#model.classifier.to(device)

# Carica l'epoca (opzionale, per riprendere l'addestramento)
epoch = checkpoint['epoch']

class CLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    def __call__(self, img):
        img = np.array(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_clahe = self.clahe.apply(img_gray)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

class ContrastAdjustment:
    def __init__(self, contrast_factor=1.5):
        self.contrast_factor = contrast_factor
    
    def __call__(self, img):
        # Aumenta il contrasto (1.0 = nessuna modifica, >1 aumenta il contrasto)
        return F.adjust_contrast(img, self.contrast_factor)

class BrightnessTopHalf:
    def __init__(self, brightness_factor=1.5):
        self.brightness_factor = brightness_factor

    def __call__(self, img):
        # Converte in Tensor
        img_tensor = transforms.ToTensor()(img)
        _, h, w = img_tensor.shape
        
        # Calcola metà altezza
        mid = h // 2
        
        # Aumenta la luminosità solo nella metà superiore
        img_tensor[:, :mid, :] = torch.clamp(img_tensor[:, :mid, :] * self.brightness_factor, 0, 1)
        
        # Converte di nuovo in PIL Image
        return transforms.ToPILImage()(img_tensor)
    
class ContrastTopHalf:
    def __init__(self, contrast_factor=1.5):
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        # Converte in Tensor
        img_tensor = transforms.ToTensor()(img)
        _, h, w = img_tensor.shape
        
        # Calcola metà altezza
        mid = h // 2
        
        # Calcola media per il contrasto
        mean_top = img_tensor[:, :mid, :].mean(dim=[1, 2], keepdim=True)
        
        # Aumenta il contrasto solo nella metà superiore
        img_tensor[:, :mid, :] = torch.clamp((img_tensor[:, :mid, :] - mean_top) * self.contrast_factor + mean_top, 0, 1)
        
        # Converte di nuovo in PIL Image
        return transforms.ToPILImage()(img_tensor)

class ContrastTopHalf:
    def __init__(self, contrast_factor=1.5):
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        img_tensor = transforms.ToTensor()(img)
        _, h, w = img_tensor.shape
        mid = h // 2
        mean_top = img_tensor[:, :mid, :].mean(dim=[1, 2], keepdim=True)
        img_tensor[:, :mid, :] = torch.clamp((img_tensor[:, :mid, :] - mean_top) * self.contrast_factor + mean_top, 0, 1)
        return transforms.ToPILImage()(img_tensor)


class BrightnessShiftTopHalf:
    def __init__(self, shift=0.2):
        self.shift = shift

    def __call__(self, img):
        img_tensor = transforms.ToTensor()(img)
        _, h, _ = img_tensor.shape
        mid = h // 2
        img_tensor[:, :mid, :] = torch.clamp(img_tensor[:, :mid, :] + self.shift, 0, 1)
        return transforms.ToPILImage()(img_tensor)

class InvertTopHalf:
    def __call__(self, img):
        img_tensor = transforms.ToTensor()(img)
        _, h, _ = img_tensor.shape
        mid = h // 2
        img_tensor[:, :mid, :] = 1 - img_tensor[:, :mid, :]
        return transforms.ToPILImage()(img_tensor)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    InvertTopHalf(),  # Inverte i colori della parte superiore
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Carica l'immagine da un percorso
image_path = './src/Classifier/Datasets/validation_set/CAM17-2014-03-12-20140312171631-20140312172219-tarid110-frame2246-line2.jpg'  # Sostituisci con il percorso dell'immagine
img = load_image(image_path)
img=img.to(device)
gender, bag, hat = classifier_model(img)


gender_pred = torch.sigmoid(gender) 
hat_pred = torch.sigmoid(hat) 
bag_pred = torch.sigmoid(bag) 

print("Gender: ", gender_pred)
print("Bag: ", bag_pred)
print("Hat: ", hat_pred)
