import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from PIL import Image
import torchvision.transforms as transforms

class CSVDataset(Dataset):
    def __init__(self, csv_file, train, transform=None, mean=None, std =None, Normalize = True,ImageType='RGB'):

        self.image_type=ImageType
        
        self.data = csv_file #TOGLIERE nrows!!
        self.transform = transform #se vuoi fare augmentation
        self.train = train

        self.TRAIN_IMAGES_PATH = "./src/Classifier/Datasets/training_set/"
        self.VALIDATION_IMAGES_PATH = "./src/Classifier/Datasets/validation_set/"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if (mean is None or std is None) and Normalize: 
            self.mean, self.std = self.evaluate_mean_and_std()  # evaluate mean and std of the dataset
        else:
            self.mean, self.std =  mean, std 
                    # Add normalization transform

        # NORMALIZATION            
        if (Normalize==True):
            if self.transform is not None:
                self.transform = transforms.Compose([
                    self.transform,  # Trasformazioni già passate (come data augmentation)
                    transforms.Normalize(mean=self.mean, std=self.std)  # Aggiungi la normalizzazione
                ])
            else:
                # Se non ci sono trasformazioni personalizzate, usa la normalizzazione di default
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ])
        print('mean:',self.mean, 'std:',self.std)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data.iloc[idx, 0]  # Image path
        labels = self.data.iloc[idx, 1:4] # label0, label1, label2

        # Load image
        if self.train == True:
            img_path = self.TRAIN_IMAGES_PATH + img_path
        else:
            img_path = self.VALIDATION_IMAGES_PATH + img_path
        image = Image.open(img_path).convert(self.image_type)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Return image and labels
        return image, torch.tensor(labels, dtype=torch.float)

    


    def evaluate_mean_and_std(self):
        print('start evaluating mean and std')
        if self.image_type == 'RGB':
            mean = torch.zeros(3, device=self.device)  # Media per i 3 canali RGB
            std = torch.zeros(3, device=self.device)   # Deviazione standard per i 3 canali RGB
        else:
            mean = torch.zeros(1, device=self.device)  # Media per i 3 canali RGB
            std = torch.zeros(1, device=self.device)   # Deviazione standard per i 3 canali RGB
        n_samples = 0

        for idx in range(len(self.data)):
            img_path = self.data.iloc[idx, 0]
            img_path = self.TRAIN_IMAGES_PATH + img_path

            image = Image.open(img_path).convert(self.image_type)
            image = transforms.ToTensor()(image).to(self.device)  # Converte l'immagine in un tensore

            # Calcola la media e deviazione standard per ogni canale
            mean += image.mean(dim=(1, 2))  # Media su H e W
            std += image.std(dim=(1, 2))    # Deviazione standard su H e W
            n_samples += 1

        mean /= n_samples
        std /= n_samples

        return mean.cpu(), std.cpu()
    

    def return_mean_and_std(self):
        return self.mean, self.std