import numpy as np
import pandas as pd


class reorderCSV():

    def __init__(self, FILE_PATH= './src/Classifier/Datasets/training_set.csv',NEW_FILE_PATH='./src/Classifier/Datasets/new_training_set.csv', BATCH_SIZE = 32):
        self.dataset_csv = pd.read_csv(FILE_PATH, sep=';', header=None, dtype={'col1': str, 'col2': int, 'col3': int, 'col4': int})
        # convert the dataset to numpy
        self.dataset = self.dataset_csv.to_numpy()
        # remove rows with -1 for evry column
        
        np.random.shuffle(self.dataset) # shuffle on rows

        self.BATCH_SIZE = BATCH_SIZE

        self.new_dataset = np.empty((0, 4))  # inizialize the branch

        self.NEW_FILE_PATH = NEW_FILE_PATH

        
    def find_valid_row(self): # find the first valid row
        for idx, row in enumerate(self.dataset):
            if np.all(row != -1):
                return idx, row
        return None, None  # if the row is not found
    

    def _create_balanced_batches(self):
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

    
    def _create_batches(self):

        while len(self.dataset) >= self.BATCH_SIZE:

            # new possible batch
            temp_batch = self.dataset[:self.BATCH_SIZE, :]
            
            # all ok
            if((temp_batch != -1).any(axis=0).all()): # no only -1 in the collum (for every column)
                # append the batch
                self.dataset = self.dataset[self.BATCH_SIZE:, :] # erase the first batch
                self.new_dataset= np.vstack([self.new_dataset, temp_batch]) # <-

            # not all ok
            else:
                idx, _ = self.find_valid_row() # find a valid row in the remaining dataset
                if(idx == None):
                    print ('perfect row not found')
                    return None
                
                #swap the rows
                newrow = self.dataset[idx]
                self.dataset[idx] = self.dataset[self.BATCH_SIZE]
                self.dataset[self.BATCH_SIZE] = newrow

                # append the batch
                temp_batch = self.dataset[:self.BATCH_SIZE, :]
                self.new_dataset= np.vstack([self.new_dataset, temp_batch]) # <-



    def print_new_csv(self):
        self._create_batches() 
        new_csv= pd.DataFrame(self.new_dataset)
        new_csv.to_csv(self.NEW_FILE_PATH, sep=';', index=False, header=False)
        return new_csv.shape[0] # Return the number of rows in the new csv
    

    def print_belanced_new_csv(self):
        self._create_balanced_batches()
        new_csv = pd.DataFrame(self.new_dataset)
        new_csv.to_csv(self.NEW_FILE_PATH, sep=';', index=False, header=False)
        print(f"Balanced dataset saved to {self.NEW_FILE_PATH}")


    def erase_invalid_row(self):
        
        mask = np.all(self.dataset[:, 1:4] == -1, axis=1)
        self.dataset = self.dataset[~mask]

        new_csv= pd.DataFrame(self.dataset)
        new_csv.to_csv(self.NEW_FILE_PATH, sep=';', index=False, header=False)

        print('Erased invalid row')

        return new_csv.shape[0] # Return the number of rows in the new csv



    def __len__(self):
        return len(self.dataset) // self.BATCH_SIZE

# Esempio di utilizzo
rcsv = reorderCSV()
rcsv.print_new_csv()
