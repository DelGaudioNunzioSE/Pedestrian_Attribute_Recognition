import torch
from firstClassifier import CNNWithAttention

# 1. Carica il modello (assicurati di definire il modello prima se non l'hai fatto)
model = CNNWithAttention()  # Sostituisci con il tipo del tuo modello
model.load_state_dict(torch.load("Models/checkpoint_epoch_2_29_1915.pth"))  # Sostituisci con il percorso corretto del tuo file .pth

# 2. Imposta il modello in modalità di valutazione
model.eval()

# 3. Esegui la validazione
val_loss = 0.0
val_acc_gender = 0.0
val_acc_hat = 0.0
val_acc_bag = 0.0

with torch.no_grad():  # Disabilita il calcolo dei gradienti per la validazione
    for images, labels in data_valid:  # Ciclo sui batch di validazione
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        gender, hat, bag = model(images)

        # Calcola le perdite
        losses = [
            adjustedLoss(gender, labels[:, 0], pos_weight=POS_WEIGHT_GENDER),
            adjustedLoss(hat, labels[:, 1], pos_weight=POS_WEIGHT_HAT),
            adjustedLoss(bag, labels[:, 2], pos_weight=POS_WEIGHT_BAG)
        ]
        loss_val = sum(losses) / len(losses)

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

# Calcola la media delle perdite e delle accuratezze per la validazione
val_loss /= len(data_valid)  # Numero di batch
val_acc_gender /= len(data_valid)
val_acc_hat /= len(data_valid)
val_acc_bag /= len(data_valid)

# Salvo i valori di validazione per ogni epoca
losses_tot.append(total_training_loss / len(data_train))  # Assicurati che total_training_loss sia definito
val_losses_tot.append(val_loss)
val_accuracies_gender.append(val_acc_gender)
val_accuracies_hat.append(val_acc_hat)
val_accuracies_bag.append(val_acc_bag)

# Stampa i risultati
print(f"Train Loss: {losses_tot[-1]:.4f}, Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy (Gender): {val_acc_gender:.4f}")
print(f"Validation Accuracy (Hat): {val_acc_hat:.4f}, Validation Accuracy (Bag): {val_acc_bag:.4f}")
print("Epoch:", epoch)
