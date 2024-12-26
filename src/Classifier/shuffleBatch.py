import numpy as np
from torch.utils.data import Sampler

class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Legge le etichette e calcola quali sono diverse da -1
        self.labels = np.array(dataset.data.iloc[:, 1:4])  # label0, label1, label2
        self.valid_indices = np.where((self.labels != -1).any(axis=1))[0]  # Indici con label diversa da -1
        self.invalid_indices = np.where((self.labels == -1).all(axis=1))[0]  # Tutte le label -1
        
    def __iter__(self):
        np.random.shuffle(self.valid_indices)
        np.random.shuffle(self.invalid_indices)
        
        batch = []
        valid_idx_pointer = 0
        invalid_idx_pointer = 0
        
        while valid_idx_pointer < len(self.valid_indices):
            # Aggiunge un campione valido al batch
            batch.append(self.valid_indices[valid_idx_pointer])
            valid_idx_pointer += 1
            
            # Completa il batch con campioni casuali (anche con label -1)
            while len(batch) < self.batch_size and invalid_idx_pointer < len(self.invalid_indices):
                batch.append(self.invalid_indices[invalid_idx_pointer])
                invalid_idx_pointer += 1
            
            # Restituisce un batch completo
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Se ci sono campioni rimanenti, restituiscili
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(self.valid_indices) // self.batch_size
