import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from PIL import Image
import torchvision.transforms as transforms

class CSVDataset(Dataset):
    def __init__(self, csv_file, train, transform=None,ImageType='RGB',homade_path=None):

        self.image_type=ImageType
        
        self.data = csv_file #TOGLIERE nrows!!
        self.transform = transform #se vuoi fare augmentation
        self.train = train

        if homade_path == None:
            self.TRAIN_IMAGES_PATH = "./src/Classifier/Datasets/training_set/"
        else:
            self.TRAIN_IMAGES_PATH = homade_path
            
        self.VALIDATION_IMAGES_PATH = "./src/Classifier/Datasets/validation_set/"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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