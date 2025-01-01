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
MODEL= 'checkpoint_3_31_0742'
IMAGE_TYPE = 'RGB'
BATCH_SIZE = int(512/2)
#############################



if IMAGE_TYPE == 'RGB':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                ])
else:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                ])



CSV_TEST_FILE='./src/Classifier/Datasets/validation_set.csv'
MODEL_PATH='./src/Classifier/Models/'+ MODEL +'.pth'
data = pd.read_csv(CSV_TEST_FILE, sep=';')
dataset_test= CSVDataset(csv_file=data, transform=transform, train=False, ImageType=IMAGE_TYPE)
data_test= DataLoader(dataset_test, batch_size=BATCH_SIZE)
model = CNNWithAttention(channel=IMAGE_TYPE) 
model.to('cuda')
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
tester=Tester(data_test, BATCH_SIZE)
tester.test(model)
print('Model:'+ MODEL_PATH)