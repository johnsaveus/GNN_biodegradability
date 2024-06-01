from sklearn.model_selection import train_test_split
import pandas as pd
import warnings

class Splitter():

    def __init__(self,
                 path,
                 train_ratio = 0.7,
                 val_ratio = 0.15,
                 include_chirality = False):
        
        
        self.path = path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio 
        self.test_ratio = 1 - train_ratio - val_ratio
        self.inculde_chirality = include_chirality

        assert (train_ratio + val_ratio + self.test_ratio) == 1, 'Fractions should add to 1'
        self._read_csv()

    def _read_csv(self):

        data_csv = pd.read_csv(self.path)
        self.smiles = data_csv['SMILES']
        self.y = data_csv['ReadyBiodegradability']

    def random_split(self):

        smiles_train, smiles_val_test, y_train, y_val_test = train_test_split(self.smiles,
                                                                      self.y,
                                                                      test_size = self.val_ratio + self.test_ratio,
                                                                      random_state=42)

        smiles_val, smiles_test, y_val, y_test = train_test_split(smiles_val_test,
                                                                  y_val_test,
                                                                  test_size=0.5,
                                                                  random_state=42)
        
        return {'Train': [smiles_train, y_train],
                'Validation': [smiles_val, y_val],
                'Test' : [smiles_test, y_test]}
    
    def _generate_single_scaffold(self,
                                  smile):
    
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

        warnings.simplefilter('ignore')
        mol = Chem.MolFromSmiles(smile)

        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality= self.inculde_chirality)
        return scaffold 
    
    def _generate_all_scaffolds(self):

        scaffolds = {}
        for ix, smile in enumerate(self.smiles):
            scaffold = self. _generate_single_scaffold(smile)
            if scaffold is not None:
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = [ix]
                else:
                    scaffolds[scaffold].append(ix)

        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold,
                 scaffold_set) in sorted(scaffolds.items(),
                                         key=lambda x: (len(x[1]), x[1][0]),
                                         reverse=True)
        ]
        return scaffold_sets
    
    def scaffold_split(self):

        scaffold_sets = self._generate_all_scaffolds()

        train_cutoff = self.train_ratio * len(self.smiles)
        valid_cutoff = (self.train_ratio + self.val_ratio) * len(self.smiles)
        train_inds = []
        val_inds = []
        test_inds = []

        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(val_inds) + len(
                        scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    val_inds += scaffold_set
            else:
                train_inds += scaffold_set

        smiles_train, y_train = self.smiles[train_inds], self.y[train_inds]
        smiles_val, y_val = self.smiles[val_inds], self.y[val_inds]
        smiles_test, y_test = self.smiles[test_inds], self.y[test_inds]
    
        return {'Train': [smiles_train, y_train],
                'Validation': [smiles_val, y_val],
                'Test' : [smiles_test, y_test]}


splitter = Splitter(path = 'data/raw/smiles_rb.csv')
random = splitter.random_split()
scaf = splitter.scaffold_split()
print(len(scaf['Train'][0]))
print(len(scaf['Validation'][0]))
print(len(scaf['Test'][0]))

print(len(random['Train'][0]))
print(len(random['Validation'][0]))
print(len(random['Test'][0]))