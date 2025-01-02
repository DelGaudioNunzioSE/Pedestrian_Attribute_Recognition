import pandas as pd
import numpy as np

# Percorso del file CSV
CSV_TRAINING_FILE = './src/Classifier/Datasets/training_set.csv'
CSV_NEW_TRAINING_FILE = './src/Classifier/Datasets/new_training_set.csv'

BATCH_SIZE = 32

# Leggi il dataset
df = pd.read_csv(CSV_TRAINING_FILE, sep=';', dtype={'col1': str, 'col2': int, 'col3': int, 'col4': int})

# Filtra le righe dove la seconda colonna è uguale a 1
gender_pos = df[(df.iloc[:, 1] == 1) & (df.iloc[:,3] == 1)]
bag_pos = df[(df.iloc[:, 2] == 1) & (df.iloc[:, 1] == -1) & (df.iloc[:, 3] == -1)]
hat_pos = df[(df.iloc[:, 3] == 1)]

gender_neg = df[(df.iloc[:, 1] == 0) & (df.iloc[:,3] == 1)]
bag_neg = df[df.iloc[:, 2] == 0]
hat_neg = df[(df.iloc[:, 3] == 0)]

gender_na = df[(df.iloc[:, 1] == -1) & (df.iloc[:,3] == 1)]
bag_na = df[df.iloc[:, 2] == -1]
hat_na = df[(df.iloc[:, 3] == -1) & (df.iloc[:,3] == 1)]


for i in range(15000):
    df = pd.concat([df, bag_pos.sample(n=1)])
    print(i)




# Salva il risultato in un nuovo file CSV
df = df.sample(frac=1).reset_index(drop=True) #shuffle
new_csv= pd.DataFrame(df)
new_csv.to_csv(CSV_NEW_TRAINING_FILE, sep=';', index=False, header=False)


