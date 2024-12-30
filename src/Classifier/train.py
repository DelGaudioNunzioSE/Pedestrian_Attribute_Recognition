from sklearn.metrics import f1_score
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from Classifier import reorderCSV
from firstClassifier import CNNWithAttention
import pandas as pd
from torch.utils.data import DataLoader
from readDataset import CSVDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision.models as models
from torch.optim import lr_scheduler
from balancedBatchSampler import BalancedBatchSampler
from collections import Counter
import numpy as np
from torch.utils.data import WeightedRandomSampler



#
# Setup
DEBUG=False
# torch setup
if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
else:
        DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}") 

############################################

# Paths
CSV_TRAINING_FILE='./src/Classifier/Datasets/training_set.csv'
CSV_NEW_TRAINING_FILE='./src/Classifier/Datasets/new_training_set.csv'


# Learning parameters
VALIDATION = True # if we have to compute validaton too

IMAGE_TYPE='RGB'

REORDER=False

# NON LI USARE !!!!!!
TEST_MEAN= None # torch.tensor([0.4582, 0.4469, 0.4290])  # set None
TEST_STD= None #torch.tensor([0.2306, 0.2173, 0.2187]) # set None

BATCH_SIZE = 512 #Reduce if you have GPU's memory problems
DATASET_SIZE = 92160  #Total number of samples: 92160
TEST_SIZE = 0.2
LEARNING_RATE = 0.1
NUM_EPOCHS = 10
EPOTH_SAVE = 0 # from which epoch start to save the model
IMAGE_RESOLUTION = (224, 224) 

GENDER_LOSS_WEIGHT = 0.2
HAT_LOSS_WEIGHT = 0.4
BAG_LOSS_WEIGHT = 0.4

POS_WEIGHT_GENDER = torch.tensor([61000/24000], device=DEVICE) # 24000 1 61000 0
POS_WEIGHT_HAT  = torch.tensor([(68629/14811)], device=DEVICE) # 10500 1 55000 0
POS_WEIGHT_BAG  = torch.tensor([55168/10516], device=DEVICE) # 9600 1 # 69000 0
############################################


# Lists to store losses and accuracies
losses_hat = [] 
losses_gender = [] 
losses_bag = [] 
losses_tot = [] 
total_training_loss = 0

accuracies_hat = []
accuracies_gender = [] 
accuracies_bag = []

val_losses_hat = []
val_losses_gender = []
val_losses_bag = [] 
val_losses_tot = []

val_accuracies_gender = [] 
val_accuracies_bag = [] 
val_accuracies_hat = [] 
val_accuracies_tot = [] 
############################################

# Checkpoint function
def checkpoint_fuction():
    timestamp = datetime.now().strftime('%d_%H%M')  # Formato: DDMM_HHMMSS

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,  # save the current epoch
        'losses': losses_tot,  # save the list of losses
    }
    checkpoint_filename = f'./src/Classifier/Models/checkpoint_epoch_try.pth'
    torch.save(checkpoint, checkpoint_filename)
    print("Model and optimizer saved successfully!")

       
# Loss Function
def adjustedLoss(prediction, labels, pos_weight ):

        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight) # object to evaluate sigmoid and then LOSS

        loss = criterion(prediction, labels.unsqueeze(1)) # evaluate loss

        mask = labels != -1 #prendo tutti gli indici delle labels -1
        valid_losses = loss[mask] #mi salvo le loss valide, con labels != -1
        mean_loss = valid_losses.mean() #ci faccio la media
        loss[~mask] = mean_loss #la sostituisco al posto delle labels -1

        batch_loss = loss.mean() #ritorno la media con le nuove loss

        return batch_loss


def total_loss_fuction(loss_gender,loss_hat,loss_bag, gender_weight = 1/3, hat_weight=1/3, bag_weight=1/3):
      if (gender_weight+hat_weight+bag_weight) != 1:
            print('Total weight is not 1!')

      total_loss= gender_weight * loss_gender + hat_weight * loss_hat + bag_weight * loss_bag
      return total_loss
      
def tpfpfn(pred,labels):
        tp = ((pred == 1) & (labels.unsqueeze(1) == 1)).sum().item()
        fp = ((pred == 1) & (labels.unsqueeze(1) == 0)).sum().item()
        fn = ((pred == 0) & (labels.unsqueeze(1) == 1)).sum().item()
        return tp,fp,fn

def fscore(tp,fp,fn):
        precision = tp / (tp + fp + 1e-8)
        recall_gender = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall_gender) / (precision + recall_gender + 1e-8)
        return f1

def gradnorm_loss(loss_gender, loss_hat, loss_bag):

        inv_loss_gender = 1 / loss_gender
        inv_loss_hat = 1 / loss_hat
        inv_loss_bag = 1 / loss_bag

        # Normalizzazione per rendere i pesi proporzionali
        total_inv_loss = inv_loss_gender + inv_loss_hat + inv_loss_bag

        fgender = inv_loss_gender / total_inv_loss
        fhat = inv_loss_hat / total_inv_loss
        fbag = inv_loss_bag / total_inv_loss

        # Calcolo della perdita totale con pesi dinamici
        loss = fgender * loss_gender + fhat * loss_hat + fbag * loss_bag
        return loss


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

# input_features = model.classifier[0].in_features
# model.classifier = nn.Sequential(
#     nn.Linear(input_features, 256),
#     nn.ReLU(),
#     nn.Dropout(p=0.6),
#     nn.Linear(256, 1),
# )
# model.to(DEVICE)

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
data = pd.read_csv(CSV_TRAINING_FILE, sep=';', nrows=DATASET_SIZE)
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

fgender, fhat, fbag = 1,1,1
for epoch in range(NUM_EPOCHS):
    
    print("We are in Epoch number: ", epoch)
    total_training_loss = 0 # Loss reset for evry epoch
    model.train() # Training mode ENABLED
    losses = []

    # Loop over the training batches
    for i, (images, labels) in enumerate(data_train):  # Ciclo sui batch di training
        # Forward pass
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        gender, hat, bag = model(images)  # FORWARD PASS
        #bag = model(images)

        loss_gender = adjustedLoss(gender, labels[:,0],pos_weight= POS_WEIGHT_GENDER)
        loss_bag = adjustedLoss(hat, labels[:,1],pos_weight=POS_WEIGHT_HAT)
        loss_hat = adjustedLoss(bag, labels[:,2],pos_weight=POS_WEIGHT_BAG) #unsqueeze ha fatto 32x1
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
                print( ' la loss hat è:', loss_hat.item(), 'labels:',    labels[:,1])
                print('la loss bag è:', loss_bag.item(), 'labels:',  labels[:,2])	
        print(f'Loss for {i}° branch over', len(data_train),'for',epoch,'epoch', 'is:', loss.item())


    # Save the model and the optimizer
    if (epoch > (EPOTH_SAVE-1)):
        print('Saving model and optimizer...')
        checkpoint_fuction()


    # Validation
    if (VALIDATION == True):
        model.eval()  # Impostiamo il modello in modalità di valutazione
        val_loss = 0.0
        val_acc_gender = 0.0
        val_acc_hat = 0.0
        val_acc_bag = 0.0
        total_samples = 0
        loss_val = 0
        tp_gender_tot = 0
        fp_gender_tot = 0
        fn_gender_tot = 0
        tp_hat_tot = 0
        fp_hat_tot = 0
        fn_hat_tot = 0
        tp_bag_tot = 0
        fp_bag_tot = 0
        fn_bag_tot = 0
        losses_val=[]

        w0,w1,w2 = fgender,fhat,fbag

        val_f1_gender = 0
        val_f1_hat = 0
        val_f1_bag = 0

        with torch.no_grad():  # Disabilita il calcolo dei gradienti per la validazione
                for i, (images, labels) in enumerate(data_valid):  # Ciclo sui batch di validazione
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE)
                        gender,hat,bag = model(images)

                        # Calcola le perdite
                        loss_gender = adjustedLoss(gender, labels[:, 0], pos_weight= POS_WEIGHT_GENDER)
                        loss_bag = adjustedLoss(hat, labels[:, 1], pos_weight= POS_WEIGHT_HAT)
                        loss_hat = adjustedLoss(bag, labels[:, 2], pos_weight= POS_WEIGHT_BAG)

                        #loss_val = gradnorm_loss(loss_gender, loss_hat, loss_bag)
                        loss_val = loss_gender + loss_hat + loss_bag
                        
                        val_loss += loss_val.item()

                        #Calcola l'accuratezza per ciascun output
                        gender_pred = torch.sigmoid(gender) > 0.5
                        accuracy_gender = (gender_pred.float() == labels[:, 0].unsqueeze(1)).float().mean()
                        tp_gender, fp_gender, fn_gender = tpfpfn(gender_pred,labels[:,0])
                        fgender = fscore(tp_gender, fp_gender, fn_gender)

                        bag_pred = torch.sigmoid(bag) > 0.5
                        accuracy_bag = (bag_pred.float() == labels[:, 1].unsqueeze(1)).float().mean()
                        tp_bag, fp_bag, fn_bag = tpfpfn(bag_pred,labels[:,1])
                        fbag = fscore(tp_bag, fp_bag, fn_bag)

                        hat_pred = torch.sigmoid(hat) > 0.5
                        accuracy_hat = (hat_pred.float() == labels[:, 2].unsqueeze(1)).float().mean()
                        tp_hat, fp_hat, fn_hat = tpfpfn(hat_pred,labels[:,2])
                        fhat = fscore(tp_hat, fp_hat, fn_hat)


                        val_acc_gender += accuracy_gender.item()
                        val_acc_hat += accuracy_hat.item()
                        val_acc_bag += accuracy_bag.item()

                        tp_gender_tot += tp_gender
                        fp_gender_tot += fp_gender
                        fn_gender_tot += fn_gender

                        tp_hat_tot += tp_hat
                        fp_hat_tot += fp_hat
                        fn_hat_tot += fn_hat

                        tp_bag_tot += tp_bag
                        fp_bag_tot += fp_bag
                        fn_bag_tot += fn_bag

                        total_samples += 1

                        print('Validation',i, 'over',len(data_valid))

    # Calcola la media delle perdite e delle accuratezze per la validazione
        val_loss /= total_samples
        val_acc_gender /= total_samples

        val_acc_hat /= total_samples
        val_acc_bag /= total_samples

        fgender = fscore(tp_gender_tot, fp_gender_tot, fn_gender_tot)
        fhat = fscore(tp_hat_tot, fp_hat_tot, fn_hat_tot)
        fbag = fscore(tp_bag_tot, fp_bag_tot, fn_bag_tot)

        # Salvo i valori di validazione per ogni epoca
        losses_tot.append(total_training_loss / len(data_train))
        val_losses_tot.append(val_loss)
        val_accuracies_gender.append(val_acc_gender)
        val_accuracies_hat.append(val_acc_hat)
        val_accuracies_bag.append(val_acc_bag)

        print(f"Train Loss: {losses_tot[-1]:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy (Gender): {val_acc_gender:.4f}")
        print(f"Validation Accuracy (Hat): {val_acc_hat:.4f}, Validation Accuracy (Bag): {val_acc_bag:.4f}")
        print(f"Tp (Gender): {tp_gender_tot:.4f}, Fp (Gender): {fp_gender_tot:.4f}, Fn (Gender): {fn_gender_tot}")
        print(f"Tp (Hat): {tp_hat_tot:.4f}, Fp (Hat): {fp_hat_tot:.4f}, Fn (hat): {fn_hat_tot}")
        print(f"Tp (Bag): {tp_bag_tot:.4f}, Fp (Bag): {fp_bag_tot:.4f}, Fn (Bag): {fn_bag_tot}")
        print(f"Fscore (Gender): {fgender:.2f}, Fscore (Hat): {fhat:.2f}, Fbag (Bag): {fbag}")
        print("Total Validation Samples: ", len(data_valid) * BATCH_SIZE)

        print("Epoch: ",epoch)




######################################################
# Plot Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_losses_tot) + 1), val_losses_tot, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.grid(True)
plt.legend()
plt.savefig('validation_loss.png')
plt.show()


# Plot della Validation Accuracy per Gender
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_accuracies_gender) + 1), val_accuracies_gender, label='Accuracy (Gender)', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy (Gender)')
plt.grid(True)
plt.legend()
plt.savefig('accuracy_gender.png')
plt.show()


# Plot della Validation Accuracy per Hat
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_accuracies_hat) + 1), val_accuracies_hat, label='Accuracy (Hat)', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy (Hat)')
plt.grid(True)
plt.legend()
plt.savefig('accuracy_hat.png')
plt.show()


# Plot della Validation Accuracy per Bag
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_accuracies_bag) + 1), val_accuracies_bag, label='Accuracy (Bag)', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy (Bag)')
plt.grid(True)
plt.legend()
plt.savefig('accuracy_bag.png')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(losses_tot, label='Train Loss', color='blue', marker='o')
plt.plot(val_losses_tot, label='Validation Loss', color='red', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('TrainVsValidation.png')
plt.show()




       


