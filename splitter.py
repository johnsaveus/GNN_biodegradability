from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
import os

class Data_Splitter():

    def __init__(self,
                 path,
                 train_ratio = 0.80,
                 val_ratio = 0.10):
        
        self.path = path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio 
        self.test_ratio = 1 - train_ratio - val_ratio

        assert (train_ratio + val_ratio + self.test_ratio) == 1, 'Fractions should add to 1'
        self._read_csv()

    def _read_csv(self):

        data_csv = pd.read_csv(self.path)
        self.smiles = data_csv['SMILES']
        self.y = data_csv['ReadyBiodegradability']

    def random_split(self):

        smiles_train, smiles_val_test, _ , y_val_test = train_test_split(self.smiles,
                                                                      self.y,
                                                                      test_size = self.val_ratio + self.test_ratio,
                                                                      random_state=42,
                                                                      stratify=self.y)
        train_ix = smiles_train.index

        smiles_val, smiles_test, _, _ = train_test_split(smiles_val_test,
                                                                  y_val_test,
                                                                  test_size=0.5,
                                                                  random_state=42,
                                                                  stratify=y_val_test)
        
        valid_ix = smiles_val.index
        test_ix = smiles_test.index

        self._save_ix(train_ix, valid_ix, test_ix)
        
    def _save_ix(self, train_ix, valid_ix, test_ix):
        
        ix_path = 'data/split_ix'
        if not os.path.exists(ix_path):
            os.makedirs(ix_path)

        train_ix_path = os.path.join(ix_path, 'train_ix.txt')
        valid_ix_path = os.path.join(ix_path, 'valid_ix.txt')
        test_ix_path =  os.path.join(ix_path, 'test_ix.txt')

        with open(train_ix_path, 'w') as file:
            for ix in train_ix:
                file.write(f"{ix}\n")

        with open(valid_ix_path, 'w') as file:
            for ix in valid_ix:
                file.write(f"{ix}\n")
        
        with open(test_ix_path, 'w') as file:
            for ix in test_ix:
                file.write(f"{ix}\n")

splitter = Data_Splitter(path = 'data/raw/smiles_rb.csv')
rand = splitter.random_split()