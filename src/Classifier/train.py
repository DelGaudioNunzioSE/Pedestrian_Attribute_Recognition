import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from SupportScripts.DataRefactor.reorderCSV import reorderCSV
import pandas as pd
from torch.utils.data import DataLoader
from SupportScripts.DataRefactor.readDataset import CSVDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from collections import Counter
import numpy as np
from torch.utils.data import WeightedRandomSampler

# OUR IMPORTS
from SupportScripts.checkpoint import checkpoint_fuction
from classifier import CNNWithAttention
from tester import *
from SupportScripts.device import device_selecter



#
# Setup
DEBUG = False
TIMESTAMP = True
REORDER=False # Nunzio's reorder
IMAGE_TYPE='RGB'

DEVICE=device_selecter()


# Paths
CSV_TRAINING_FILE='./src/Classifier/Datasets/training_set.csv'
CSV_NEW_TRAINING_FILE='./src/Classifier/Datasets/new_training_set.csv'


# Learning parameters
VALIDATION = True # if we have to compute validaton too


# NON LI USARE !!!!!!
TEST_MEAN=  torch.tensor([0.4582, 0.4469, 0.4290])  # set None
TEST_STD= torch.tensor([0.2306, 0.2173, 0.2187]) # set None

BATCH_SIZE = 256 #Reduce if you have GPU's memory problems
TEST_SIZE = 0.2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 5
EPOTH_SAVE = 0 # from which epoch start to save the model
IMAGE_RESOLUTION = (224, 224) 

GENDER_LOSS_WEIGHT = 0.2
HAT_LOSS_WEIGHT = 0.4
BAG_LOSS_WEIGHT = 0.4

POS_WEIGHT_GENDER = torch.tensor([61000/24000], device=DEVICE) # 24000 1 61000 0
POS_WEIGHT_HAT  = torch.tensor([(68629/14811)], device=DEVICE) # 10500 1 55000 0
POS_WEIGHT_BAG  = torch.tensor([55168/10516], device=DEVICE) # 9600 1 # 69000 0


      



###########################################################
# START TRAINING






# Data Augmentation
# Compose = Composition of transformations
TRANSFORMS = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
])


# Model creation
model = CNNWithAttention()   
model.to(DEVICE)

# model = models.vgg16(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

# Reading new dataset
# Changing the dataset
if (REORDER == True):
        rcsv=reorderCSV(BATCH_SIZE=BATCH_SIZE ,FILE_PATH=CSV_TRAINING_FILE, NEW_FILE_PATH=CSV_NEW_TRAINING_FILE)
        DATASET_SIZE=rcsv.print_new_csv()
        CSV_TRAINING_FILE=CSV_NEW_TRAINING_FILE


# Reading new dataset
data = pd.read_csv(CSV_TRAINING_FILE, sep=';')
train_data, val_data = train_test_split(data, test_size=TEST_SIZE, random_state=42)

# Object to load images
# (csv_file -> in the first column there are the paths of the images)
dataset_train = CSVDataset(csv_file=train_data, transform=TRANSFORMS, train=True, mean=TEST_MEAN, std=TEST_STD, Normalize=True, ImageType=IMAGE_TYPE)
train_mean, train_std = dataset_train.return_mean_and_std()
dataset_valid = CSVDataset(csv_file=val_data, transform=TRANSFORMS, train=True, mean=train_mean, std=train_std, Normalize=True, ImageType=IMAGE_TYPE)
# TODO dataset_test = CSVDataset(csv_file='./src/Classifier/Datasets/validation_set.csv', transform=None, train=False)

# DataLoader
# Test DataLoader
# TODO TOERASE batch_sampler = CustomBatchSampler(dataset_train, batch_size=BATCH_SIZE)

def calculate_class_weights(dataset):
    """
    Calcola i pesi per bilanciare le classi per ogni task e assegna un peso per ogni campione.
    :param dataset: Dataset PyTorch
    :return: Array di pesi per ogni campione
    """
    # Calcolati da preprocess con seed=65464
    gender_dist = Counter({0: 49383, 1: 18952, -1: 6129})
    bag_dist = Counter({0: 44237, -1: 21829, 1: 8398})
    hat_dist = Counter({0: 54941, -1: 11838, 1: 7685})

    scale_factor = 1000
    gender_weights = {label: (1.0 / count) * scale_factor for label, count in gender_dist.items() if label != -1}
    bag_weights = {label: (1.0 / count) * scale_factor for label, count in bag_dist.items() if label != -1}
    hat_weights = {label: (1.0 / count) * scale_factor for label, count in hat_dist.items() if label != -1}

    sample_weights = []
    for i in range(len(dataset)):
        # Estrai le etichette del campione
        labels = np.array(dataset[i][1])

        # Calcola i pesi per ogni task, assegnando 0.0 se l'etichetta è -1
        gender_weight = gender_weights.get(labels[0], 0.0)
        bag_weight = bag_weights.get(labels[1], 0.0)
        hat_weight = hat_weights.get(labels[2], 0.0)

        # Se tutte le label sono -1, assegna peso 0.0
        if all(label == -1 for label in labels):
            combined_weight = 0.0
        else:
            # Calcola il peso combinato come media dei pesi validi
            combined_weight = np.mean([gender_weight, bag_weight, hat_weight])

        #print(labels,combined_weight)

        sample_weights.append(combined_weight)

    return np.array(sample_weights)


class_weights = calculate_class_weights(dataset_train)
sampler = WeightedRandomSampler(class_weights, len(dataset_train))

#bc = BalancedBatchSampler(train_data,32)
data_train = DataLoader(dataset_train,batch_size=BATCH_SIZE,sampler=sampler) #batch di train
# TODO data_test = DataLoader(dataset_test, batch_sampler=batch_sampler)

# Validation DataLoader
#batch_sampler_valid = CustomBatchSampler(dataset_valid, batch_size=BATCH_SIZE)
data_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE)


# Usa il sigmoide all'interno, quindi non c'è bisogno di usarlo nella rete neurale
# E' più stabile di sigmoide seguito da BCE.
# criterion = nn.BCEWithLogitsLoss() #ottengo la loss per ogni campione


# Optimizer
# Scheduler che riduce il learning rate ogni 10 epoche di un fattore di 0.1
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) 




###########################################################
# TRAINING LOOP  ##########################################

print('Datatrain dimension:', len(data_train)*BATCH_SIZE) # len(data_train) = batch number
validator=Tester(data_valid, POS_WEIGHT_GENDER, POS_WEIGHT_BAG, POS_WEIGHT_HAT)

for epoch in range(NUM_EPOCHS):
    
    print("We are in Epoch number: ", epoch)
    total_training_loss = 0 # Loss reset for evry epoch
    model.train() # Training mode ENABLED

    # Loop over the training batches
    for i, (images, labels) in enumerate(data_train):  # Ciclo sui batch di training
        # Forward pass
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        gender, bag, hat = model(images)  # FORWARD PASS

        loss_gender = adjustedLoss(gender, labels[:,0],pos_weight= POS_WEIGHT_GENDER)
        loss_bag = adjustedLoss(bag, labels[:,1],pos_weight=POS_WEIGHT_HAT)
        loss_hat = adjustedLoss(hat, labels[:,2],pos_weight=POS_WEIGHT_BAG) #unsqueeze ha fatto 32x1
        loss = total_loss_fuction(loss_gender, loss_hat, loss_bag, gender_weight = GENDER_LOSS_WEIGHT, hat_weight=HAT_LOSS_WEIGHT, bag_weight=BAG_LOSS_WEIGHT)
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
    model_to_validate=checkpoint_fuction(TIMESTAMP,model,optimizer,epoch)


    # Validation
    if (VALIDATION == True):
        validator.test(model_to_validate)


validator.plot()




       


