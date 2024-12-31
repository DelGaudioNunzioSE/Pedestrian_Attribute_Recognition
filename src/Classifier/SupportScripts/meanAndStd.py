import torch
from PIL import Image
import torchvision.transforms as transforms


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