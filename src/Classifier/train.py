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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
transform = transforms.Compose([transforms.ToTensor(),
							   transforms.Resize((224, 224))
							   ]) #per ora solo questa

#lettura dataset
model = CNNWithAttention()   
dataset = CSVDataset(csv_file='./src/Classifier/Datasets/training_set.csv', transform=transform, train=True) #potrei applicare trasformazioni per fare data augmentation
#dataset_test = CSVDataset(csv_file='./src/Classifier/Datasets/validation_set.csv', transform=None, train=False) #potrei applicare trasformazioni per fare data augmentation

val_percent = 0.2 # percentage of the data used for validation 
val_size = int(val_percent * len(dataset)) 
train_size = len(dataset) - val_size 

batch_sampler = CustomBatchSampler(dataset, batch_size=32)
data_train = DataLoader(dataset, batch_sampler=batch_sampler)

#data_test = DataLoader(dataset_test, batch_sampler=batch_sampler)


# Move the model to the GPU if available 
model.to(device)

# criterion_gender = nn.CrossEntropyLoss() 
# criterion_hat = nn.CrossEntropyLoss() 
# criterion_bag = nn.CrossEntropyLoss()

# Usa il sigmoide all'interno, quindi non c'è bisogno di usarlo nella rete neurale
# E' più stabile di sigmoide seguito da BCE.
criterion_gender = nn.BCEWithLogitsLoss()
criterion_hat = nn.BCEWithLogitsLoss()
criterion_bag = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


losses_hat = [] 
losses_gender = [] 
losses_bag = [] 
losses_tot = [] 

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
num_epochs=2 
for epoch in range(num_epochs):  # Ciclo su tutte le epoche
    for i, (images, labels) in enumerate(data_train):  # Ciclo sui batch di training
        # Forward pass
        images = images.to(device)
        labels = labels.to(device)
        gender, hat, bag = model(images)  # 3 valori

        labels[labels == -1] = 0 #TOGLIERE
        print(gender.size())

        loss_gender = criterion_gender(gender, labels[:,0].unsqueeze(1)) 
        loss_hat = criterion_hat(hat, labels[:,1].unsqueeze(1))
        loss_bag = criterion_bag(bag, labels[:,2].unsqueeze(1)) #unsqueeze ha fatto 32x1
        loss = (1 / 3) * loss_gender + (1 / 3) * loss_hat + (1 / 3) * loss_bag  # i pesi devono essere dinamici

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Applico un threshold per dire che quando la prob è superiore a 0.5 è classe 1.
        # Devo applicare anche il sigmoide perchè la loss lo applica internamente.
        gender_pred = torch.sigmoid(gender) > 0.5
        accuracy = (gender_pred == labels[:,0].unsqueeze(1)).float().mean()


        losses_tot.append(loss.item())
    print(epoch)
        # # Valutazione del modello sul set di validazione
        # val_loss = 0.0
        # val_acc = 0.0
        # with torch.no_grad():
        #     for images, labels in val_loader:
        #         labels = labels.to(device)
        #         images = images.to(device)
        #         outputs = model(images)
        #         loss = criterion(outputs, labels)
        #         val_loss += loss.item()

        #         _, predicted = torch.max(outputs.data, 1)
        #         total = labels.size(0)
        #         correct = (predicted == labels).sum().item()
        #         val_acc += correct / total
        #         val_accuracies.append(val_acc)
        
print(losses_tot)
