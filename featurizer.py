from rdkit import Chem
import pandas as pd
#from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
import torch
from tqdm import tqdm
import os
from rdkit.Chem import AllChem

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

class MolecularDataset(InMemoryDataset):

    def __init__(self,
                 root,
                 transform = None,
                 pre_transform = None,
                 pre_filter = None):
        super(MolecularDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0]) # 2.5 version

    def raw_file_names(self):
        return ['smiles_rb.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass 

    def process(self):
        dataset = []
        raw_path = os.path.join(self.raw_dir, 'smiles_rb.csv')
        self.data = pd.read_csv(raw_path).reset_index()
        for ix, mol in tqdm(self.data.iterrows(), total = self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol['SMILES'])
            if mol_obj is None or mol_obj.GetNumAtoms() <= 1:
            # Skip this molecule and continue to the next
                continue
            x = self._get_node_features(mol_obj)
            edge_index , bool_dense = self._get_adjacency(mol_obj)
            #edge_attr = self._get_edge_features(mol_obj)
            fingerprint = self._get_fingerprint(mol_obj)
            label = self._get_label(mol['ReadyBiodegradability'])
            num_nodes = mol_obj.GetNumAtoms()

            data = Data(
            x = x,
            edge_index = edge_index,
            fingerprint = fingerprint,
            y = label,
            smiles = mol['SMILES'],
            num_nodes = num_nodes)
            
            dataset.append(data)
            
        processed_dir = os.path.join(self.root, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        self.save(dataset, self.processed_paths[0])
                       
    def _get_node_features(self, mol):

        node_feat = torch.stack(
        [atom_features(atom) for atom in mol.GetAtoms()])
        node_feats = torch.tensor(node_feat, dtype = torch.float32)
        return node_feats
    
    def _get_adjacency(self, mol):
        
        src , dest = [] , []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]
        edge_index = torch.asarray([src, dest], dtype = torch.int) # Needs to be in COO Format
        # Convert to dense
        num_nodes = mol.GetNumAtoms()
        vals = torch.ones(edge_index.shape[1], dtype = torch.float32)
        sparse_adj = torch.sparse_coo_tensor(edge_index, vals, (num_nodes, num_nodes))
        dense_adj = sparse_adj.to_dense()
        bool_dense = dense_adj > 0
        return edge_index, bool_dense
    
    def _get_edge_features(self, mol):

        features = []
        for bond in mol.GetBonds():
            features += 2 * [bond_features(bond)]
        edge_feat = torch.tensor(features, dtype=torch.float)

        return edge_feat
    
    def _get_fingerprint(self, mol):

        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint_tensor = torch.tensor(list(fingerprint), dtype=torch.float32)

        return fingerprint_tensor

    def _get_label(self, label):
        return torch.asarray([label], dtype = int)
    
dataset = MolecularDataset(root = 'data')
dataset.load('data\processed\data.pt')
print(dataset[0])
#dataset.load()
#print(dataset[0])
#dataset.process()
#dataset = torch.load('data/processed/data.pt')
#print(slices)
#print(dataset[200])
#print(dataset[1].edge_index)
# Only for creation of validation and test
#df = pd.read_csv("data/smiles_rb.csv")
#smiles = df['SMILES']
#y = df['ReadyBiodegradability']
#print(len(df))

'''
dataset = MolecularDataset(smiles, y)
print(len(dataset.data))
print(dataset.data)'''
