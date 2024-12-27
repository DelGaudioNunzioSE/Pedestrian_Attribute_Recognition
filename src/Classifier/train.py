import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from firstClassifier import CNNWithAttention
import pandas as pd
from shuffleBatch import CustomBatchSampler
from torch.utils.data import DataLoader
from readDataset import CSVDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def adjustedLoss(prediction, labels):
        criterion = nn.BCEWithLogitsLoss(reduction='none') #object to evaluate sigmoid and then LOSS

        loss = criterion(prediction, labels[:, 0].unsqueeze(1)) # evaluate loss
        mask = labels != -1 #prendo tutti gli indici delle labels -1
        valid_losses = loss[mask] #mi salvo le loss valide, con labels != -1
        mean_loss = valid_losses.mean() #ci faccio la media
        loss[~mask] = mean_loss #la sostituisco al posto delle labels -1
        batch_loss = loss.mean() #ritorno la media con le nuove loss

        return batch_loss


if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
        device = torch.device("mps")
else:
        device = torch.device("cpu")

print(f"Using device: {device}") 


transform = transforms.Compose([transforms.ToTensor(),
				transforms.Resize((224, 224))
							   ]) #per ora solo questa

#lettura dataset
data = pd.read_csv('./src/Classifier/Datasets/training_set.csv',sep=';',nrows=10000) #TOGLIERE nrows!!
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


model = CNNWithAttention()   
model.to(device)
dataset_train = CSVDataset(csv_file=train_data, transform=transform, train=True) #potrei applicare trasformazioni per fare data augmentation
dataset_valid = CSVDataset(csv_file=val_data, transform=transform, train=True) #potrei applicare trasformazioni per fare data augmentation
#dataset_test = CSVDataset(csv_file='./src/Classifier/Datasets/validation_set.csv', transform=None, train=False) #potrei applicare trasformazioni per fare data augmentation


batch_sampler = CustomBatchSampler(dataset_train, batch_size=64)
data_train = DataLoader(dataset_train, batch_sampler=batch_sampler) #batch di train
#data_test = DataLoader(dataset_test, batch_sampler=batch_sampler)

batch_sampler_valid = CustomBatchSampler(dataset_valid, batch_size=64)
data_valid = DataLoader(dataset_valid, batch_sampler=batch_sampler_valid)


# Usa il sigmoide all'interno, quindi non c'è bisogno di usarlo nella rete neurale
# E' più stabile di sigmoide seguito da BCE.
criterion = nn.BCEWithLogitsLoss() #ottengo la loss per ogni campione

optimizer = optim.Adam(model.parameters(), lr=0.001)

losses_hat = [] 
losses_gender = [] 
losses_bag = [] 
losses_tot = [] 
loss_train = 0

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

# Train the model
num_epochs=50
for epoch in range(num_epochs):  # Ciclo su tutte le epoche
    loss_train = 0
    model.train()
    for i, (images, labels) in enumerate(data_train):  # Ciclo sui batch di training
        # Forward pass
        images = images.to(device)
        labels = labels.to(device)
        gender, hat, bag = model(images)  # 3 valori


        loss_gender = adjustedLoss(gender, labels[:,0].unsqueeze(1))
        loss_hat = adjustedLoss(hat, labels[:,1].unsqueeze(1))
        loss_bag = adjustedLoss(bag, labels[:,2].unsqueeze(1)) #unsqueeze ha fatto 32x1
        loss = (1 / 3) * loss_gender + (1 / 3) * loss_hat + (1 / 3) * loss_bag  # i pesi devono essere dinamici

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Applico un threshold per dire che quando la prob è superiore a 0.5 è classe 1.
        # Devo applicare anche il sigmoide perchè la loss lo applica internamente.
        gender_pred = torch.sigmoid(gender) > 0.5
        accuracy = (gender_pred == labels[:,0].unsqueeze(1)).float().mean()

        loss_train += loss.item()

       
    if (epoch + 1) % 2 == 0:
        model.eval()  # Impostiamo il modello in modalità di valutazione
        val_loss = 0.0
        val_acc_gender = 0.0
        val_acc_hat = 0.0
        val_acc_bag = 0.0
        total_samples = 0
        loss_val = 0

        with torch.no_grad():  # Disabilita il calcolo dei gradienti per la validazione
                for images, labels in data_valid:  # Ciclo sui batch di validazione
                        images = images.to(device)
                        labels = labels.to(device)
                        gender, hat, bag = model(images)

                        # Calcola le perdite
                        loss_gender = adjustedLoss(gender, labels[:, 0].unsqueeze(1))
                        loss_hat = adjustedLoss(hat, labels[:, 1].unsqueeze(1))
                        loss_bag = adjustedLoss(bag, labels[:, 2].unsqueeze(1))
                        loss_val = (1 / 3) * loss_gender + (1 / 3) * loss_hat + (1 / 3) * loss_bag

                        val_loss += loss_val.item()

                        # Calcola l'accuratezza per ciascun output
                        gender_pred = torch.sigmoid(gender) > 0.5
                        accuracy_gender = (gender_pred == labels[:, 0].unsqueeze(1)).float().mean()

                        hat_pred = torch.sigmoid(hat) > 0.5
                        accuracy_hat = (hat_pred == labels[:, 1].unsqueeze(1)).float().mean()

                        bag_pred = torch.sigmoid(bag) > 0.5
                        accuracy_bag = (bag_pred == labels[:, 2].unsqueeze(1)).float().mean()

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
        losses_tot.append(loss_train / len(data_train))
        val_losses_tot.append(val_loss)
        val_accuracies_gender.append(val_acc_gender)
        val_accuracies_hat.append(val_acc_hat)
        val_accuracies_bag.append(val_acc_bag)
        print(f"Train Loss: {losses_tot[-1]:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy (Gender): {val_acc_gender:.4f}")
        print(f"Validation Accuracy (Hat): {val_acc_hat:.4f}, Validation Accuracy (Bag): {val_acc_bag:.4f}")
        print("Epoch: ",epoch)

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


       


