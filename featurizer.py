from rdkit import Chem
import pandas as pd
from torch_geometric.data import Data, Dataset
import torch
from tqdm import tqdm
import os

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
   
        atom_feats = torch.tensor(one_of_k_encoding_unk(
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
    bond_feats = [
    bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
    bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    bond.GetIsConjugated(),
    bond.IsInRing(),
    bond.GetStereo()]

    return bond_feats

class MolecularDataset(Dataset):

    def __init__(self):
        super(MolecularDataset, self).__init__()

    def create(self, raw_path):
        dataset = []
        self.data = pd.read_csv(raw_path).reset_index()
        for ix, mol in tqdm(self.data.iterrows(), total = self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol['SMILES'])
            if mol_obj is None or mol_obj.GetNumAtoms() <= 1:
            # Skip this molecule and continue to the next
                continue
            x = self._get_node_features(mol_obj)

            edge_index = self._get_adjacency(mol_obj)
            edge_attr = self._get_edge_features(mol_obj)
            label = self._get_label(mol['ReadyBiodegradability'])

            data = Data(x =x, 
                    edge_index = edge_index,
                    edge_attr = edge_attr,
                    y=label,
                    smiles=mol["SMILES"])
            
            dataset.append(data)
        torch.save(dataset, 'data/train.pt')
                       
    def _get_node_features(self, mol):

        node_feat = torch.stack(
        [atom_features(atom) for atom in mol.GetAtoms()])
        return node_feat
    
    def _get_adjacency(self, mol):

        src , dest = [] , []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]
        edge_index = torch.asarray([src, dest], dtype = int) # Needs to be in COO Format

        return edge_index
    
    def _get_edge_features(self, mol):

        features = []
        for bond in mol.GetBonds():
            features += 2 * [bond_features(bond)]
        edge_feat = torch.tensor(features, dtype=torch.float)

        return edge_feat

    def _get_label(self, label):
        return torch.asarray([label], dtype = int)
    
    
#dataset = MolecularDataset()
#dataset.create('data/smiles_rb.csv')
dataset = torch.load('data/train.pt')
print(len(dataset))
# Only for creation of validation and test
#df = pd.read_csv("data/smiles_rb.csv")
#smiles = df['SMILES']
#y = df['ReadyBiodegradability']
#print(len(df))

'''
dataset = MolecularDataset(smiles, y)
print(len(dataset.data))
print(dataset.data)'''
