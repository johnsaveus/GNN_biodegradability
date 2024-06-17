from sklearn.model_selection import train_test_split
import pandas as pd
import os
from rdkit import Chem

class Data_Splitter():

    ''' Merge the 2 biodegradability datasets and split them stratified into train-val-test and saves their Ix
        Args:
            path_to_sdf (str): Path to Sdf file (1st dataset)
            path_to_csv (str): Path to Csv file (2nd dataset)
            train_ratio (float, optional) : Train fraction (default = 0.8)
            val_ratio (float, optional) : Validation fraction (default = 0.1)
    '''
    
    def __init__(self,
                 path_to_sdf : str,
                 path_to_csv : str,
                 train_ratio : float = 0.80,
                 val_ratio : float = 0.10):
        self.path_to_sdf = path_to_sdf
        self.path_to_csv = path_to_csv
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio 
        self.test_ratio = 1 - train_ratio - val_ratio
        assert (train_ratio + val_ratio + self.test_ratio) == 1, 'Fractions should add to 1'
        self._read_csv()
        self._read_sdf()

    def _read_csv(self):
        data_csv = pd.read_csv(self.path_to_csv)
        self.smiles_csv = data_csv['SMILES']
        self.y_csv = data_csv['ReadyBiodegradability']

    def _read_sdf(self):
        reader = Chem.SDMolSupplier(self.path_to_sdf)
        self.smiles_sdf = []  # Smiles from property
        self.y_sdf = [] # RB
        for ix, mol in enumerate(reader):
           properties = mol.GetPropsAsDict()
           if mol is not None:
              self.smiles_sdf.append(properties['SMILES'])
              self.y_sdf.append(properties['ReadyBiodegradability'])
           else:
              print(f"Invalid Mol file number {ix}")
        sdf_data = pd.DataFrame(
        {'SMILES':self.smiles_sdf,
         'ReadyBiodegradability': self.y_sdf})
        sdf_data.to_csv("raw/AllPublicnew.csv")
        self.smiles_sdf = sdf_data['SMILES']
        self.y_sdf = sdf_data['ReadyBiodegradability']

    def split(self):
        self._random_split(self.smiles_csv,
                           self.y_csv,
                           'csv')
        self._random_split(self.smiles_sdf,
                           self.y_sdf,
                           'sdf')
        
    def _random_split(self,
                     smiles,
                     y,
                     dataset_name):
        smiles_train, smiles_val_test, _ , y_val_test = train_test_split(smiles,
                                                                      y,
                                                                      test_size = self.val_ratio + self.test_ratio,
                                                                      random_state=42,
                                                                      stratify=y)
        train_ix = smiles_train.index
        smiles_val, smiles_test, _, _ = train_test_split(smiles_val_test,
                                                        y_val_test,
                                                        test_size=0.5,
                                                        random_state=42,
                                                        stratify=y_val_test)
        valid_ix = smiles_val.index
        test_ix = smiles_test.index
        self._save_ix(train_ix, valid_ix, test_ix, dataset_name)
        
    def _save_ix(self,
                train_ix,
                valid_ix, 
                test_ix,
                dataset_name):
        
        ix_path = 'split_ix'
        if not os.path.exists(ix_path):
            os.makedirs(ix_path)

        train_ix_path = os.path.join(ix_path, dataset_name + '_train_ix.txt')
        valid_ix_path = os.path.join(ix_path, dataset_name + '_valid_ix.txt')
        test_ix_path =  os.path.join(ix_path, dataset_name + '_test_ix.txt')

        with open(train_ix_path, 'w') as file:
            for ix in train_ix:
                file.write(f"{ix}\n")
        with open(valid_ix_path, 'w') as file:
            for ix in valid_ix:
                file.write(f"{ix}\n")
        with open(test_ix_path, 'w') as file:
            for ix in test_ix:
                file.write(f"{ix}\n")

if __name__ == '__main__':
    splitter = Data_Splitter(path_to_csv = 'raw/ECHA_biodegradability_new_compounds.csv',
                         path_to_sdf = 'raw/AllPublicnew.sdf')
    splitter.split()
