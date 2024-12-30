import numpy as np
import pandas as pd

class ReorderCSV():
    def __init__(self, csv_file='./src/Classifier/Datasets/training_set.csv', 
                 new_csv_file='./src/Classifier/Datasets/new_training_set.csv', 
                 batch_size=32):
        self.dataset_csv = pd.read_csv(csv_file, sep=';', header=None)
        self.dataset = self.dataset_csv.to_numpy()

        # Rimuove le righe dove tutte le colonne 1, 2, 3 contengono -1
        maschera = np.any(self.dataset[:, 1:4] != -1, axis=1)
        self.dataset = self.dataset[maschera]

        np.random.shuffle(self.dataset)  # Mescola il dataset
        self.batch_size = batch_size
        self.new_dataset = np.empty((0, 4))
        self.new_csv_file = new_csv_file

    def create_balanced_batches(self):
        # Separazione delle classi per ciascuna colonna
        class_0_col2 = self.dataset[self.dataset[:, 1] == 0]
        class_1_col2 = self.dataset[self.dataset[:, 1] == 1]

        class_0_col3 = self.dataset[self.dataset[:, 2] == 0]
        class_1_col3 = self.dataset[self.dataset[:, 2] == 1]

        class_0_col4 = self.dataset[self.dataset[:, 3] == 0]
        class_1_col4 = self.dataset[self.dataset[:, 3] == 1]

        # Dimensione di ciascun sotto-batch
        third_batch = self.batch_size // 3 // 2

        while (len(class_0_col2) >= third_batch and len(class_1_col2) >= third_batch and
               len(class_0_col3) >= third_batch and len(class_1_col3) >= third_batch and
               len(class_0_col4) >= third_batch and len(class_1_col4) >= third_batch):
            
            # Seleziona porzioni equilibrate da ciascuna classe
            batch_0_col2 = class_0_col2[:third_batch]
            batch_1_col2 = class_1_col2[:third_batch]

            batch_0_col3 = class_0_col3[:third_batch]
            batch_1_col3 = class_1_col3[:third_batch]

            batch_0_col4 = class_0_col4[:third_batch]
            batch_1_col4 = class_1_col4[:third_batch]

            # Combina i batch
            temp_batch = np.vstack([batch_0_col2, batch_1_col2, 
                                    batch_0_col3, batch_1_col3, 
                                    batch_0_col4, batch_1_col4])

            # Controlla se ci sono sequenze con -1 in tutte le righe di una colonna
            for i in range(1, 4):
                if np.all(temp_batch[:, i] == -1):  # Se tutta una colonna è -1
                    # Inserisce una riga (ad esempio con valori di default, qui un array di zeri)
                    temp_batch = np.vstack([temp_batch, np.array([0, 0, 0, 0])])
                    temp_batch = temp_batch[:self.batch_size]  # Mantieni la dimensione batch_size

            # Elimina righe dove tutte le colonne contengono -1
            temp_batch = temp_batch[~np.all(temp_batch[:, 1:4] == -1, axis=1)]
            
            np.random.shuffle(temp_batch)  # Mescola il batch
            self.new_dataset = np.vstack([self.new_dataset, temp_batch])  # Aggiunge il batch finale

            # Rimuove le righe usate
            class_0_col2 = class_0_col2[third_batch:]
            class_1_col2 = class_1_col2[third_batch:]

            class_0_col3 = class_0_col3[third_batch:]
            class_1_col3 = class_1_col3[third_batch:]

            class_0_col4 = class_0_col4[third_batch:]
            class_1_col4 = class_1_col4[third_batch:]

    def print_new_csv(self):
        self.create_balanced_batches()
        new_csv = pd.DataFrame(self.new_dataset)
        new_csv.to_csv(self.new_csv_file, sep=';', index=False, header=False)
        print(f"Balanced dataset saved to {self.new_csv_file}")

    def __len__(self):
        return len(self.new_dataset) // self.batch_size

# Esempio di utilizzo
rcsv = ReorderCSV()
rcsv.print_new_csv()
