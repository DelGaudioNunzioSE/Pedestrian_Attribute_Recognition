import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from PIL import Image
import torchvision.transforms as transforms

class CSVDataset(Dataset):
    def __init__(self, csv_file, train, transform=None):
        self.data = pd.read_csv(csv_file,nrows=1000,sep=';') #TOGLIERE!!
        self.transform = transform #se vuoi fare augmentation
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]  # Percorso immagine
        labels = self.data.iloc[idx, 1:4] # label0, label1, label2
        if self.train == True:
            img_path = "./src/Classifier/Datasets/training_set/" + img_path
        else:
            img_path = "./src/Classifier/Datasets/validation_set/" + img_path
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(labels, dtype=torch.long)