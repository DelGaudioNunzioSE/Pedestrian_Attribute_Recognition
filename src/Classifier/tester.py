from email.headerregistry import DateHeader
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from firstClassifier import CNNWithAttention
from readDataset import CSVDataset

class Tester:
    def __init__(self, model_path):
        self.device = self.device_selection() # device selection

        self.model = CNNWithAttention() 
        self.model.to(self.device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    
    def device_selection(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}") 


    def test(self,data_test):
        self.model.eval()  # Imposta il modello in modalità valutazione
        val_loss = 0.0
        val_acc_gender = 0.0
        val_acc_hat = 0.0
        val_acc_bag = 0.0
        val_f1_gender = 0.0
        val_f1_hat = 0.0
        val_f1_bag = 0.0
        total_samples = 0

        with torch.no_grad():  # Disabilita i gradienti
            for i, (images, labels) in enumerate(data_test):
                images, labels = images.to(self.device), labels.to(self.device)

                # Ottieni le predizioni
                gender, hat, bag = self.model(images)


                # Calcola le metriche
                gender_pred = torch.sigmoid(gender) > 0.5
                hat_pred = torch.sigmoid(hat) > 0.5
                bag_pred = torch.sigmoid(bag) > 0.5

                val_acc_gender += (gender_pred == labels[:, 0].unsqueeze(1)).float().mean().item()
                val_acc_hat += (hat_pred == labels[:, 1].unsqueeze(1)).float().mean().item()
                val_acc_bag += (bag_pred == labels[:, 2].unsqueeze(1)).float().mean().item()

                val_f1_gender += f1_score(labels[:, 0].cpu(), gender_pred.cpu(),average='binary')
                val_f1_hat += f1_score(labels[:, 1].cpu(), hat_pred.cpu(),average='binary')
                val_f1_bag += f1_score(labels[:, 2].cpu(), bag_pred.cpu(),average='binary')

                total_samples += 1
                print(f"Validation batch {i+1}/{len(data_test)}")

        # Calcola le medie
        val_loss /= total_samples
        val_acc_gender /= total_samples
        val_acc_hat /= total_samples
        val_acc_bag /= total_samples
        val_f1_gender /= total_samples
        val_f1_hat /= total_samples
        val_f1_bag /= total_samples


        # Stampa i risultati
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy (Gender): {val_acc_gender:.4f}, F1-Score: {val_f1_gender:.4f}")
        print(f"Validation Accuracy (Hat): {val_acc_hat:.4f}, F1-Score: {val_f1_hat:.4f}")
        print(f"Validation Accuracy (Bag): {val_acc_bag:.4f}, F1-Score: {val_f1_bag:.4f}")



TEST_MEAN = torch.tensor([0.4582, 0.4469, 0.4290])
TEST_STD = torch.tensor([0.2306, 0.2173, 0.2187])
IMAGE_TYPE = 'RGB'
BATCH_SIZE = 512


transform = transforms.Compose([transforms.ToTensor(),
				                transforms.Resize((224, 224)),
                                ])


CSV_TEST_FILE='./src/Classifier/Datasets/validation_set.csv'
MODEL_PATH='./src/Classifier/Models/checkpoint_epoch_2_30_0546.pth'
data = pd.read_csv(CSV_TEST_FILE, sep=';')
dataset_test= CSVDataset(csv_file=data, transform=transform, train=False, mean=TEST_MEAN, std=TEST_STD, Normalize=True, ImageType=IMAGE_TYPE)
data_test= DataLoader(dataset_test, batch_size=BATCH_SIZE)
tester=Tester(MODEL_PATH)
tester.test(data_test)

