from datetime import datetime
import cv2
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from SupportScripts.DataRefactor.reorderCSV import reorderCSV
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from SupportScripts.DataRefactor.readDataset import CSVDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
from PIL import Image, ImageOps
from torch.optim.lr_scheduler import CyclicLR

# OUR IMPORTS
from SupportScripts.checkpoint import checkpoint_fuction
from classifierTest import CNNWithAttention2
from SupportScripts.adjustedLoss import adjustedLoss, total_loss_fuction
from SupportScripts.tester import Tester
from SupportScripts.device import device_selecter

# auto information
DEVICE=device_selecter()
STARTING_TRAIN_TIME_STAMP= timestamp = datetime.now().strftime('%d_%H%M')

# Setup #########################################################################
LEARNING_COMMENT = '_7ciriprovo'
NUMBER_OF_NEURONS=int(512/2) 
TIMESTAMP = False
MODEL_PATH=None # if you wanto to start from a previous model

# Nunzio's
REORDER=True # Nunzio's reorder
IMAGE_TYPE='RGB' # RGB or L balck and white
HOMEMADE_IMGE_PATH = None # if we do a pre-image processing in another folder

# Paths
CSV_TRAINING_FILE='./src/Classifier/Datasets/training_set.csv'
CSV_NEW_TRAINING_FILE='./src/Classifier/Datasets/new_training_set.csv'

# Learning parameters
VALIDATION = True # if we have to compute validaton too

BATCH_SIZE = int(256) #Reduce if you have GPU's memory problems
VALIDATION_SIZE = 0.1
LEARNING_RATE = 0.00001
NUM_EPOCHS = 15
GENDER_LOSS_WEIGHT = 0.3
BAG_LOSS_WEIGHT = 0.4
HAT_LOSS_WEIGHT = 0.3
POS_WEIGHT_GENDER = torch.tensor([61000/24000], device=DEVICE) # 24000 1 61000 0
POS_WEIGHT_BAG  = torch.tensor([55168/10516], device=DEVICE)
POS_WEIGHT_HAT  = torch.tensor([(68629/14811)], device=DEVICE) 

#########################################################################################
class HistogramEqualization:
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Input is not PIL.Image")
        return ImageOps.equalize(img)

class CLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    def __call__(self, img):
        img = np.array(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_clahe = self.clahe.apply(img_gray)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)


##### DATA AUGMENTATION ######################################
####################################################
if IMAGE_TYPE=='RGB':
    print('Image type is RGB')
    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to a uniform size
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.RandomRotation(degrees=(-5, 5)),
        #CLAHE(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        #CLAHE(), # to simulate the same 'normalization'
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
else: 
    TRANSFORMS = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.7),
                                    transforms.RandomHorizontalFlip(),
                                    # HistogramEqualization(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    





##### DATASET ######################################
####################################################

if(REORDER==True):
    rcsv=reorderCSV(BATCH_SIZE=BATCH_SIZE ,FILE_PATH=CSV_TRAINING_FILE, NEW_FILE_PATH=CSV_NEW_TRAINING_FILE)
    DATASET_SIZE=rcsv.erase_invalid_row()
    rcsv.print_new_csv()


# Reading new dataset
data = pd.read_csv(CSV_NEW_TRAINING_FILE, sep=';')
train_data, val_data = train_test_split(data, test_size=VALIDATION_SIZE, random_state=42)

dataset_train = CSVDataset(csv_file=train_data, transform=TRAIN_TRANSFORMS, train=True, ImageType=IMAGE_TYPE,homade_path=HOMEMADE_IMGE_PATH)
dataset_valid = CSVDataset(csv_file=val_data, transform=VAL_TRANSFORMS, train=True, ImageType=IMAGE_TYPE,homade_path=HOMEMADE_IMGE_PATH)



#Dataset
data_train = DataLoader(dataset_train,batch_size=BATCH_SIZE) #batch di train
data_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE)




##### MODEL ######################################
####################################################
# Model creation
model = CNNWithAttention2()   
model.to(DEVICE)


if MODEL_PATH != None:
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])



# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) # Christian's
#scheduler = CyclicLR(optimizer, base_lr=LEARNING_RATE, max_lr=(LEARNING_RATE*10), step_size_up=NUM_EPOCHS/2, mode='triangular') # Nunzio's
###########################################################



print('Datatrain dimension:', len(data_train)*BATCH_SIZE) # len(data_train) = batch number

validator=Tester(data_valid, POS_WEIGHT_GENDER, POS_WEIGHT_BAG, POS_WEIGHT_HAT)#  VALIDATOR



###############################################
# TRAINING LOOP  ##########################################

for epoch in range(NUM_EPOCHS):
    
    print("We are in Epoch number: ", epoch)
    print(f'Learning rate: {scheduler.get_last_lr()[0]}')
    total_training_loss = 0 # Loss reset for evry epoch
    model.train() # Training mode ENABLED

    # Loop over the training batches
    for i, (images, labels) in enumerate(data_train):  # Ciclo sui batch di training
        # Forward pass
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        gender, bag, hat = model(images)  # FORWARD PASS

        loss_gender = adjustedLoss(gender, labels[:,0],pos_weight= POS_WEIGHT_GENDER)
        loss_bag = adjustedLoss(bag, labels[:,1],pos_weight=POS_WEIGHT_BAG)
        loss_hat = adjustedLoss(hat, labels[:,2],pos_weight=POS_WEIGHT_HAT)
        loss = total_loss_fuction(loss_gender, loss_bag, loss_hat, gender_weight = GENDER_LOSS_WEIGHT,  bag_weight=BAG_LOSS_WEIGHT, hat_weight=HAT_LOSS_WEIGHT)


        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_training_loss += loss.item()

        print(f'Loss for {i}° batch over', len(data_train),'for',epoch,'epoch', 'is:', loss.item())


    print('Saving model and optimizer...')
    checkpoint_fuction(TIMESTAMP, model, optimizer, epoch,comment=LEARNING_COMMENT)

    scheduler.step()

    # Validation
    if (VALIDATION == True):
        validator.test(model,GENDER_LOSS_WEIGHT, BAG_LOSS_WEIGHT,HAT_LOSS_WEIGHT)
        validator.plot(LEARNING_COMMENT) # save the plot








       


