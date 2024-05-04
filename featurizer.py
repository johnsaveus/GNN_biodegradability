import torch
from rdkit import Chem
import pandas as pd

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
                ]) + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                           [0, 1, 2, 3, 4]) + \
                    one_of_k_encoding(atom.GetDegree(), 
                                      [0, 1, 2, 3, 4]) + \
                    [atom.GetFormalCharge()] + \
                    [atom.GetIsAromatic]
    
    return atom_feats

'''
def bond_features(bond):
   
   bt = bond.GetBondType()
   bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
        bond.GetStereo()]
   
   return bond_feats
'''
