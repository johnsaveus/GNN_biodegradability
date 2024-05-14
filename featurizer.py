from rdkit import Chem
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
   
        atom_feats = np.array(one_of_k_encoding_unk(
       atom.GetSymbol(),
       [
        'Rh',
        'Li',
        'Pd', 
        'V', 
        'B', 
        'Na',
        'S', 
        'Cl', 
        'Cu', 
        'Bi',
        'Mg',
        'Zr', 
        'P', 
        'I', 
        'K', 
        'C', 
        'Ca', 
        'Fe',
        'Mn', 
        'Cr',
        'N', 
        'La', 
        'Ba', 
        'Al',
        'F', 
        'Sn',
        'Hg', 
        'Br', 
        'Cs',
        'H',
        'Nd',
        'Ti', 
        'Ni', 
        'Co',
        'Sr',
        'O',
        'Ce', 
        'Y', 
        'Zn', 
        'Si', 
        'other'
        ])  + one_of_k_encoding_unk(atom.GetHybridization(), 
                [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP, 
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, 
                Chem.rdchem.HybridizationType.SP3D, 
                Chem.rdchem.HybridizationType.SP3D2,
                'other'
                ]) + one_of_k_encoding(atom.GetTotalNumHs(),
                                           [0, 1, 2, 3, 4]) + \
                    one_of_k_encoding_unk(atom.GetDegree(), 
                                      [0, 1, 2, 3, 4]) + \
                    [atom.GetFormalCharge()] + \
                    [atom.GetIsAromatic()])
        
        return atom_feats
    
def bond_features(bond):

    bt = bond.GetBondType()
    bond_feats = np.array([
    bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
    bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    bond.GetIsConjugated(),
    bond.IsInRing(),
    bond.GetStereo()])

    return bond_feats

def featurize(smile):

    mol = Chem.MolFromSmiles(smile)
    assert mol.GetNumAtoms() > 1
    "More than one atom should be present"

    node_features = np.asarray(
        [atom_features(atom) for atom in mol.GetAtoms()], 
        dtype = float
    )

    src , dest = [] , []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [start, end]
        dest += [end, start]
    edge_index = np.asarray([src, dest], dtype = int) # Needs to be in COO Format

    features = []
    for bond in mol.GetBonds():
        features += 2 * [bond_features(bond)]
    edge_features = np.asarray(features, dtype=float)

    return Data(node_features = node_features,
                edge_index = edge_index,
                edge_attr = edge_features)
    

class MolecularDataset(Dataset):

    def __init__(self,
                smiles,
                y):
         
        super(MolecularDataset, self).__init__()

    def _create(self):
        pass
    
    def _get_target(self, target):
        return np.asarray([target], dtype = int) 
    
smile = 'CCC'
feats = featurize(smile)
print(feats)
# Only for creation of validation and test
#df = pd.read_csv("data/smiles_rb.csv")
#smiles = df['SMILES']
#y = df['ReadyBiodegradability']
#print(len(df))

'''
dataset = MolecularDataset(smiles, y)
print(len(dataset.data))
print(dataset.data)'''

smiles = ['CCC']
