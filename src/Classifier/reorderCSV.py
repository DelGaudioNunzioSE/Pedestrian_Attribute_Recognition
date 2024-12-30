import numpy as np
import pandas as pd


class reorderCSV():

    def __init__(self, FILE_PATH= './src/Classifier/Datasets/training_set.csv',NEW_FILE_PATH='./src/Classifier/Datasets/new_training_set.csv', BATCH_SIZE = 32):
        self.dataset_csv = pd.read_csv(FILE_PATH, sep=';', header=None, dtype={'col1': str, 'col2': int, 'col3': int, 'col4': int})
        # convert the dataset to numpy
        self.dataset = self.dataset_csv.to_numpy()
        # remove rows with -1 for evry column
        mask = np.all(self.dataset[:, 1:4] == -1, axis=1)
        self.dataset = self.dataset[~mask]
        np.random.shuffle(self.dataset) # shuffle on rows

        self.BATCH_SIZE = BATCH_SIZE

        self.new_dataset = np.empty((0, 4))  # inizialize the branch

        self.NEW_FILE_PATH = NEW_FILE_PATH
    

        
    def find_valid_row(self): # find the first valid row
        for idx, row in enumerate(self.dataset):
            if np.all(row != -1):
                return idx, row
        return None, None  # if the row is not found
    


    def create_batches(self):

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


    # the main function
    def print_new_csv(self):
        self.create_batches() 
        new_csv= pd.DataFrame(self.new_dataset)
        new_csv.to_csv(self.NEW_FILE_PATH, sep=';', index=False, header=False)
        return new_csv.shape[0] # Return the number of rows in the new csv


    def __len__(self):
        return len(self.dataset) // self.BATCH_SIZE



rcsv=reorderCSV()
rcsv.print_new_csv()