import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from firstClassifier import CNNWithAttention
import pandas as pd
from torch.utils.data import DataLoader
from readDataset import CSVDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from reorderCSV import reorderCSV
from datetime import datetime


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
CSV_TRAINIG_FILE='./src/Classifier/Datasets/training_set.csv'
CSV_NEW_TRAINING_FILE='./src/Classifier/Datasets/new_training_set.csv'


# Learning parameters
BATCH_SIZE = 512 #Reduce if you have GPU's memory problems
DATASET_SIZE = 92160//3  #Total number of samples: 92160
TEST_SIZE = 0.3
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
EPOTH_SAVE = 0 # from which epoch start to save the model
IMAGE_RESOLUTION = (120, 300) 
POS_WEIGHT_GENDER = torch.tensor([61000/24000], device=DEVICE) # 24000 1 61000 0
POS_WEIGHT_HAT  = torch.tensor([55000/10500], device=DEVICE) # 10500 1 55000 0
POS_WEIGHT_BAG  = torch.tensor([69000/9600], device=DEVICE) # 9600 1 # 69000 0
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
    timestamp = datetime.now().strftime('%d-%H:%M')  # Formato: DDMM_HHMMSS

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,  # save the current epoch
        'losses': losses_tot,  # save the list of losses
    }
    checkpoint_filename = f'./src/Classifier/Models/checkpoint_epoch_{epoch}_{timestamp}.pth'
    torch.save(checkpoint, checkpoint_filename)
    print("Model and optimizer saved successfully!")

       
# Loss Function
def adjustedLoss(prediction, labels, pos_weight ):

        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight) # object to evaluate sigmoid and then LOSS

        loss = criterion(prediction, labels) # evaluate loss

        mask = labels != -1 #prendo tutti gli indici delle labels -1
        valid_losses = loss[mask] #mi salvo le loss valide, con labels != -1
        mean_loss = valid_losses.mean() #ci faccio la media
        loss[~mask] = mean_loss #la sostituisco al posto delle labels -1

        batch_loss = loss.mean() #ritorno la media con le nuove loss

        return batch_loss






###########################################################
# START TRAINING






# Data Augmentation
# Compose = Composition of transformations
TRANSFORMS = transforms.Compose([transforms.ToTensor(), # -> [C (number of channels), H (height), W (width)] 
				transforms.Resize(IMAGE_RESOLUTION) 
							   ])


# Changing the dataset
rcsv=reorderCSV(batch_size=BATCH_SIZE ,csv_file=CSV_TRAINIG_FILE, new_csv_file=CSV_NEW_TRAINING_FILE)
rcsv.print_new_csv()


# Reading new dataset
data = pd.read_csv(CSV_NEW_TRAINING_FILE, sep=';', nrows=200)
train_data, val_data = train_test_split(data, test_size=TEST_SIZE, random_state=42)

# Model creation
model = CNNWithAttention()   
model.to(DEVICE)

# Object to load images
# (csv_file -> in the first column there are the paths of the images)
dataset_train = CSVDataset(csv_file=train_data, transform=TRANSFORMS, train=True)
dataset_valid = CSVDataset(csv_file=val_data, transform=TRANSFORMS, train=True)
# TODO dataset_test = CSVDataset(csv_file='./src/Classifier/Datasets/validation_set.csv', transform=None, train=False)

# DataLoader
# Test DataLoader
# TODO TOERASE batch_sampler = CustomBatchSampler(dataset_train, batch_size=BATCH_SIZE)
data_train = DataLoader(dataset_train, batch_size=BATCH_SIZE) #batch di train
# TODO data_test = DataLoader(dataset_test, batch_sampler=batch_sampler)

# Validation DataLoader
#batch_sampler_valid = CustomBatchSampler(dataset_valid, batch_size=BATCH_SIZE)
data_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE)


# Usa il sigmoide all'interno, quindi non c'è bisogno di usarlo nella rete neurale
# E' più stabile di sigmoide seguito da BCE.
# criterion = nn.BCEWithLogitsLoss() #ottengo la loss per ogni campione


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



###########################################################
# TRAINING LOOP  ##########################################

print('Datatrain dimension:', len(data_train)*BATCH_SIZE) # len(data_train) = batch number

for epoch in range(NUM_EPOCHS):
    
    print("We are in Epoch number: ", epoch)
    total_training_loss = 0 # Loss reset for evry epoch
    model.train() # Training mode ENABLED

    # Loop over the training batches
    for i, (images, labels) in enumerate(data_train):  # Ciclo sui batch di training
        # Forward pass
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        gender, hat, bag = model(images)  # FORWARD PASS


        loss_gender = adjustedLoss(gender, labels[:,0],pos_weight= POS_WEIGHT_GENDER)
        loss_hat = adjustedLoss(hat, labels[:,1],pos_weight=POS_WEIGHT_HAT)
        loss_bag = adjustedLoss(bag, labels[:,2],pos_weight=POS_WEIGHT_BAG) #unsqueeze ha fatto 32x1
        loss = (1 / 3) * loss_gender + (1 / 3) * loss_hat + (1 / 3) * loss_bag  # i pesi devono essere dinamici

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
        print(f'Loss for {i}° branch over', len(data_train), 'is:', loss.item())


    # Save the model and the optimizer
    if (epoch > (EPOTH_SAVE-1)):
        print('Saving model and optimizer...')
        checkpoint_fuction()


    # Validation
    if (epoch + 1) % 5 == 0:
        model.eval()  # Impostiamo il modello in modalità di valutazione
        val_loss = 0.0
        val_acc_gender = 0.0
        val_acc_hat = 0.0
        val_acc_bag = 0.0
        total_samples = 0
        loss_val = 0

        with torch.no_grad():  # Disabilita il calcolo dei gradienti per la validazione
                for images, labels in data_valid:  # Ciclo sui batch di validazione
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE)
                        gender, hat, bag = model(images)

                        # Calcola le perdite
                        loss_gender = adjustedLoss(gender, labels[:, 0], pos_weight= POS_WEIGHT_GENDER)
                        loss_hat = adjustedLoss(hat, labels[:, 1], pos_weight= POS_WEIGHT_HAT)
                        loss_bag = adjustedLoss(bag, labels[:, 2], pos_weight= POS_WEIGHT_BAG)
                        loss_val = (1 / 3) * loss_gender + (1 / 3) * loss_hat + (1 / 3) * loss_bag

                        val_loss += loss_val.item()

                        # Calcola l'accuratezza per ciascun output
                        gender_pred = torch.sigmoid(gender) > 0.5
                        accuracy_gender = (gender_pred.float() == labels[:, 0].unsqueeze(1)).float().mean()

                        hat_pred = torch.sigmoid(hat) > 0.5
                        accuracy_hat = (hat_pred.float() == labels[:, 1].unsqueeze(1)).float().mean()

                        bag_pred = torch.sigmoid(bag) > 0.5
                        accuracy_bag = (bag_pred.float() == labels[:, 2].unsqueeze(1)).float().mean()

                        val_acc_gender += accuracy_gender.item()
                        val_acc_hat += accuracy_hat.item()
                        val_acc_bag += accuracy_bag.item()
                        total_samples += 1

    # Calcola la media delle perdite e delle accuratezze per la validazione
        val_loss /= total_samples
        val_acc_gender /= total_samples
        val_acc_hat /= total_samples
        val_acc_bag /= total_samples

        # Salvo i valori di validazione per ogni epoca
        losses_tot.append(total_training_loss / len(data_train))
        val_losses_tot.append(val_loss)
        val_accuracies_gender.append(val_acc_gender)
        val_accuracies_hat.append(val_acc_hat)
        val_accuracies_bag.append(val_acc_bag)
        print(f"Train Loss: {losses_tot[-1]:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy (Gender): {val_acc_gender:.4f}")
        print(f"Validation Accuracy (Hat): {val_acc_hat:.4f}, Validation Accuracy (Bag): {val_acc_bag:.4f}")
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
plt.show()

# Plot della Validation Accuracy per Gender
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_accuracies_gender) + 1), val_accuracies_gender, label='Accuracy (Gender)', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy (Gender)')
plt.grid(True)
plt.legend()
plt.show()

# Plot della Validation Accuracy per Hat
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_accuracies_hat) + 1), val_accuracies_hat, label='Accuracy (Hat)', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy (Hat)')
plt.grid(True)
plt.legend()
plt.show()

# Plot della Validation Accuracy per Bag
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_accuracies_bag) + 1), val_accuracies_bag, label='Accuracy (Bag)', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy (Bag)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(losses_tot, label='Train Loss', color='blue', marker='o')
plt.plot(val_losses_tot, label='Validation Loss', color='red', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,  # opzionale, per salvare l'epoca corrente
    'losses': losses_tot,  # opzionale, per salvare la lista delle perdite
}

torch.save(checkpoint, './src/Classifier/Models/checkpoint.pth')
print("Model and optimizer saved successfully!")


       


