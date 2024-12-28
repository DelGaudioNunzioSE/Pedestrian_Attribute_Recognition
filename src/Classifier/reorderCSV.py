import numpy as np
import pandas as pd


class reorderCSV():

    def __init__(self, csv_file= './src/Classifier/Datasets/training_set.csv',new_csv_file='./src/Classifier/Datasets/new_training_set.csv', batch_size = 32):
        self.dataset_csv = pd.read_csv(csv_file, sep=';', header=None, dtype={'col1': str, 'col2': int, 'col3': int, 'col4': int})

        self.dataset = self.dataset_csv.to_numpy()
        np.random.shuffle(self.dataset) # shuffle on rows
        self.batch_size = batch_size
        self.new_dataset = np.empty((0, 4))  # inizialaize the branch

        self.new_csv_file = new_csv_file
    

        
    def find_valid_row(self): # find the first valid row
        for idx, row in enumerate(self.newlabels):
            if np.all(row != -1):
                return idx, row
        return None, None  # if the row is not found
    


    def create_batches(self):

        while len(self.dataset) >= self.batch_size:

            # new possible batch
            temp_batch = self.dataset[:self.batch_size, :]
            
            # all ok
            if((temp_batch != -1).any(axis=0).all()): # no only -1 in the collum (for evry collum)
                # append the batch
                self.dataset = self.dataset[self.batch_size:, :] # erase the first batch
                self.new_dataset= np.vstack([self.new_dataset, temp_batch]) # <-

            # not all ok
            else:
                idx, _ = self.find_valid_row() # find a valid row in the remaining dataset
                if(idx == None):
                    print ('perfect row not found')
                    return None
                
                #swap the rows
                newrow = self.dataset[idx]
                self.dataset[idx] = self.dataset[self.batch_size]
                self.dataset[self.batch_size] = newrow

                # append the batch
                temp_batch = self.dataset[:self.batch_size, :]
                self.new_dataset= np.vstack([self.new_dataset, temp_batch]) # <-


    # the main function
    def print_new_csv(self):
        self.create_batches() 
        new_csv= pd.DataFrame(self.new_dataset)
        new_csv.to_csv(self.new_csv_file, sep=';', index=False, header=False)


    def __len__(self):
        return len(self.newlabels) // self.batch_size



rcsv=reorderCSV()
rcsv.print_new_csv()