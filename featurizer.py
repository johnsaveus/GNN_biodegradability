import torch
from rdkit import Chem
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data

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
   
    atom_feats = one_of_k_encoding_unk(
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
                    one_of_k_encoding(atom.GetDegree(), 
                                      [0, 1, 2, 3, 4]) + \
                    [atom.GetFormalCharge()] + \
                    [atom.GetIsAromatic]
    
    return torch.tensor(atom_feats)
 

def bond_features(bond):
   
   bt = bond.GetBondType()
   bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
        bond.GetStereo()]
   
   return bond_feats


def nodes_and_adjacency(smile, y):

    mol = Chem.MolFromSmiles(smile)

    node_features = [atom_features(atom) for atom in mol.GetAtoms()]
    node_features = torch.stack(node_features).float()

    bond_features = [bond_features(bond) for bond in mol.GetBonds()]
    bond_features = torch.stack(bond_features).float()
    # Create Adjacency Matrix
    ix1 , ix2 = [] , []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ix1 += [start, end]
        ix2 += [end, start]
    adj_norm = torch.asarray([ix1, ix2], dtype = torch.int64) # Needs to be in COO Format

    return Data(x = node_features,
                edge_index = adj_norm,
                edge_attr = bond_features,
                y = y)


class MolecularDataset(Dataset):

    def __init__(self,
                 smiles,
                 Y):
        
        self.smiles = smiles
        self.Y = Y

    def split():
        # Only for creation of validation and test
