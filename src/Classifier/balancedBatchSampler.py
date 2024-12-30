import numpy as np
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Estrazione degli indici per le categorie positive, negative e con valori na
        self.hat_pos = np.where(dataset.iloc[:, 2] == 1)[0]   # Hat positive (colonna 2)
        self.bag_pos = np.where(dataset.iloc[:, 3] == 1)[0]   # Bag positive (colonna 3)
        self.gender_neg = np.where(dataset.iloc[:, 1] == 0)[0]  # Gender negative (colonna 1)
        self.hat_neg = np.where(dataset.iloc[:, 2] == 0)[0]    # Hat negative (colonna 2)
        self.bag_neg = np.where(dataset.iloc[:, 3] == 0)[0]    # Bag negative (colonna 3)
        self.gender_na = np.where(dataset.iloc[:, 1] == -1)[0] # Gender not available (colonna 1)

    def __iter__(self):
        while True:  # Ciclo infinito per restituire i batch
            batch_indices = []

            # Sovracampionamento delle categorie positive e bilanciamento con le categorie negative
            batch_indices.extend(np.random.choice(self.hat_pos, size=self.batch_size // 3, replace=True))
            batch_indices.extend(np.random.choice(self.bag_pos, size=self.batch_size // 3, replace=True))
            batch_indices.extend(np.random.choice(self.gender_neg, size=self.batch_size // 6, replace=True))
            batch_indices.extend(np.random.choice(self.hat_neg, size=self.batch_size // 6, replace=True))
            batch_indices.extend(np.random.choice(self.bag_neg, size=self.batch_size // 6, replace=True))
            batch_indices.extend(np.random.choice(self.gender_na, size=self.batch_size // 6, replace=True))

            # Mescolare gli indici per garantire che siano casuali
            np.random.shuffle(batch_indices)

            # Restituire gli indici del batch
            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size


