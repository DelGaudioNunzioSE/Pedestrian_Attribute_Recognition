from datetime import datetime
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
from classifier import CNNWithAttention
from SupportScripts.adjustedLoss import adjustedLoss, total_loss_fuction
from SupportScripts.tester import Tester
from SupportScripts.device import device_selecter
from SupportScripts.calculateClassWeights import calculate_class_weights

# auto information
DEVICE=device_selecter()
STARTING_TRAIN_TIME_STAMP= timestamp = datetime.now().strftime('%d_%H%M')

# Setup #########################################################################
LEARNING_COMMENT ='HistogramEqualization_512_neurons'
NUMBER_OF_NEURONS=int(256*2)
DEBUG = False
TIMESTAMP = True
CLASS_WEIGHTS= False # Paolo's
MODEL_PATH=None # if you wanto to start from a previous model

# Nunzio's
REORDER=False # Nunzio's reorder
IMAGE_TYPE='RGB' # RGB or L balck and white

# Paths
CSV_TRAINING_FILE='./src/Classifier/Datasets/training_set.csv'
CSV_NEW_TRAINING_FILE='./src/Classifier/Datasets/new_training_set.csv'

# Learning parameters
VALIDATION = True # if we have to compute validaton too

BATCH_SIZE = 128 #Reduce if you have GPU's memory problems
VALIDATION_SIZE = 0.1
LEARNING_RATE = 0.00001
NUM_EPOCHS = 15
GENDER_LOSS_WEIGHT = 0.2
BAG_LOSS_WEIGHT = 0.6
HAT_LOSS_WEIGHT = 0.2
POS_WEIGHT_GENDER = torch.tensor([61000/24000], device=DEVICE) # 24000 1 61000 0
POS_WEIGHT_BAG  = torch.tensor([55168/10516], device=DEVICE)
POS_WEIGHT_HAT  = torch.tensor([(68629/14811)], device=DEVICE) 

#########################################################################################
class HistogramEqualization:
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Input deve essere un'immagine PIL.Image")
        return ImageOps.equalize(img)



##### DATA AUGMENTATION ######################################
####################################################
if IMAGE_TYPE=='L':
    TRANSFORMS = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                    transforms.RandomHorizontalFlip(),
                                    HistogramEqualization(),
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
    ])
else: 
    TRANSFORMS = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                    transforms.RandomHorizontalFlip(),
                                    HistogramEqualization(),
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) # resbet50 normalization
    ])





##### DATASET ######################################
####################################################

rcsv=reorderCSV(BATCH_SIZE=BATCH_SIZE ,FILE_PATH=CSV_TRAINING_FILE, NEW_FILE_PATH=CSV_NEW_TRAINING_FILE)
DATASET_SIZE=rcsv.erase_invalid_row()
CSV_TRAINING_FILE=CSV_NEW_TRAINING_FILE


# Reading new dataset
data = pd.read_csv(CSV_TRAINING_FILE, sep=';')
train_data, val_data = train_test_split(data, test_size=VALIDATION_SIZE, random_state=42)

dataset_train = CSVDataset(csv_file=train_data, transform=TRANSFORMS, train=True, ImageType=IMAGE_TYPE)
dataset_valid = CSVDataset(csv_file=val_data, transform=TRANSFORMS, train=True, ImageType=IMAGE_TYPE)




# Change class weight
print('Starting evaluating weights...')
if CLASS_WEIGHTS == True:
    class_weights = calculate_class_weights(dataset_train)
    sampler = WeightedRandomSampler(class_weights, len(dataset_train))
else:
    sampler = RandomSampler(dataset_train)
    valid_sampler = SequentialSampler(dataset_valid)


#Dataset
data_train = DataLoader(dataset_train,batch_size=BATCH_SIZE, sampler=sampler) #batch di train
data_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, sampler=valid_sampler)




##### MODELLO ######################################
####################################################
# Model creation
model = CNNWithAttention(channel=IMAGE_TYPE, hidden_dim=NUMBER_OF_NEURONS)   
model.to(DEVICE)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

if MODEL_PATH != None:
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

# model = models.vgg16(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False

# Usa il sigmoide all'interno, quindi non c'è bisogno di usarlo nella rete neurale
# E' più stabile di sigmoide seguito da BCE.
# criterion = nn.BCEWithLogitsLoss() #ottengo la loss per ogni campione


# Optimizer
# Scheduler che riduce il learning rate ogni 10 epoche di un fattore di 0.1
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) # Christian's
scheduler = CyclicLR(optimizer, base_lr=LEARNING_RATE, max_lr=(LEARNING_RATE*10), step_size_up=5, mode='triangular') # Nunzio's
###########################################################



print('Datatrain dimension:', len(data_train)*BATCH_SIZE) # len(data_train) = batch number
# VALIDATOR
validator=Tester(data_valid, POS_WEIGHT_GENDER, POS_WEIGHT_BAG, POS_WEIGHT_HAT)



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
        loss_hat = adjustedLoss(hat, labels[:,2],pos_weight=POS_WEIGHT_HAT) #unsqueeze ha fatto 32x1
        loss = total_loss_fuction(loss_gender=loss_gender, loss_bag=loss_bag, loss_hat=loss_hat, gender_weight = GENDER_LOSS_WEIGHT, bag_weight=BAG_LOSS_WEIGHT, hat_weight=HAT_LOSS_WEIGHT)
        #loss = gradnorm_loss(loss_gender, loss_hat, loss_bag)


        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_training_loss += loss.item()

        if(DEBUG==True):
                print('sono alla ', i)
                print('la loss gender è:', loss_gender.item(), 'labels:', labels[:,0])
                print('la loss bag è:', loss_bag.item(), 'labels:',  labels[:,1])
                print( 'la loss hat è:', loss_hat.item(), 'labels:',    labels[:,2])	
        print(f'Loss for {i}° batch over', len(data_train),'for',epoch,'epoch', 'is:', loss.item())


    print('Saving model and optimizer...')
    checkpoint_fuction(TIMESTAMP, model, optimizer, epoch,comment=LEARNING_COMMENT)

    scheduler.step()

    # Validation
    if (VALIDATION == True):
        validator.test(model,GENDER_LOSS_WEIGHT, BAG_LOSS_WEIGHT,HAT_LOSS_WEIGHT)
        validator.plot(LEARNING_COMMENT + STARTING_TRAIN_TIME_STAMP)







       


