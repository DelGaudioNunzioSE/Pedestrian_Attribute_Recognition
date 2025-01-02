from email.headerregistry import DateHeader
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

from classifier import CNNWithAttention
from SupportScripts.DataRefactor.readDataset import CSVDataset
from SupportScripts.adjustedLoss import *
from SupportScripts.device import device_selecter
from SupportScripts.tester import Tester

## Parametri
<<<<<<< HEAD
HOMEMADE_PATH = './src/Classifier/Datasets/canny_validation_set/'
MODEL= '_canny__6_02_0704'
IMAGE_TYPE = 'L'
BATCH_SIZE = int(512/2)
Number_of_neurons = int(512/2)
=======
MODEL= 'HistogramEqualization_512_neurons_7_01_0818'
IMAGE_TYPE = 'RGB'
BATCH_SIZE = int(32)
Number_of_neurons = int(512)
>>>>>>> 69811fee5974667f3d6efed79c5f505ce0a42bb2
#############################

class HistogramEqualization:
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Input deve essere un'immagine PIL.Image")
        return ImageOps.equalize(img)


if IMAGE_TYPE == 'RGB':
    transform = transforms.Compose([HistogramEqualization(),
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                ])
else:
    transform = transforms.Compose([#HistogramEqualization(),
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                ])


t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CSV_TEST_FILE='./src/Classifier/Datasets/validation_set.csv'
MODEL_PATH='./src/Classifier/Models/'+ MODEL +'.pth'
data = pd.read_csv(CSV_TEST_FILE, sep=';')
dataset_test= CSVDataset(csv_file=data, transform=transform, train=False, ImageType=IMAGE_TYPE,homade_path=HOMEMADE_PATH)
data_test= DataLoader(dataset_test, batch_size=BATCH_SIZE)
model = CNNWithAttention(hidden_dim=Number_of_neurons ,channel=IMAGE_TYPE) 
model.to('cuda')
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
tester=Tester(data_test, BATCH_SIZE)
for th in t:
    tester.test(model,t=th)
print('Model:'+ MODEL_PATH)