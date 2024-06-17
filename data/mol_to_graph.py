from rdkit import Chem
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
import torch
from tqdm import tqdm
import os
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Disable Hydrogen warnings
RDLogger.DisableLog('rdApp.*')  

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
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
        'Y', 'La', 'Sn', 'Pb', 'K', 'Ti', 'N', 'Fe', 'Mg', 'P', 'Pt',
        'F', 'Hg', 'Li', 'I', 'Cl', 'Nd', 'O', 'Na', 'Zn',  'H', 'B',
        'Cr', 'Si', 'Al', 'C', 'Mo', 'Co', 'Pd', 'Bi',  'Zr', 'Ba', 
        'Tl', 'Mn', 'Ni', 'Br', 'Ca', 'S', 'Cs', 'Sr',  'Cu', 'Ce',
         'other'
        ])  + one_of_k_encoding_unk(atom.GetHybridization(), 
                [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP, 
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, 
                Chem.rdchem.HybridizationType.SP3D, 
                Chem.rdchem.HybridizationType.SP3D2
                ]) + one_of_k_encoding(atom.GetTotalNumHs(),
                                           [0, 1, 2, 3, 4]) + \
                    one_of_k_encoding(atom.GetDegree(), 
                                      [0, 1, 2, 3, 4, 5, 6]) + \
                    [atom.GetFormalCharge()] + \
                    one_of_k_encoding(atom.GetNumRadicalElectrons(),
                                      [0, 1, 2, 3, 4]) + \
                    [atom.GetIsAromatic()], dtype = torch.float32)
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

    '''Creates Molecular Datasets that will be used for training-inference
       
       Args:
            root (str): The path to the directory of the data folder (that contains raw, proccessed)
            path_to_ind_csv (str): Path to csv indexes saved for the specific dataset (train, val, test)
            path_to_ind_sdf (str): Path to sdf indexes saved for the specific dataset (train, val, test)
            save_name (str): Name of the dataset

        If the dataset is already created, it can be accessed with dataset.load(saved_path)
    '''
    def __init__(self,
                 root : str,
                 path_to_ind_csv : str,
                 path_to_ind_sdf : str,
                 save_name : str):
        
        self.path_to_ind_csv = path_to_ind_csv
        self.path_to_ind_sdf = path_to_ind_sdf
        self.save_name = save_name + '.pt'
        super(MolecularDataset, self).__init__(root)
        self.load(self.processed_paths[0]) # 2.5 version

    def raw_file_names(self):
        return ['ECHA_biodegradability_new_compounds.csv', 'AllPublicnew.csv']

    @property
    def processed_file_names(self):
        return [self.save_name]
    
    def download(self):
        pass 

    def _read_txt_ind(self, path):
        indices = []
        with open(path, 'r') as file:
            for line in file:
               indices.append(int(line.strip()))
        return indices

    def process(self):
        indices_csv = self._read_txt_ind(self.path_to_ind_csv)
        indices_sdf = self._read_txt_ind(self.path_to_ind_sdf)
        csv_data = pd.read_csv(self.raw_paths[0]).reset_index()
        sdf_data = pd.read_csv(self.raw_paths[1]).reset_index()
        csv_data = csv_data[['SMILES','ReadyBiodegradability']]
        csv_data['ReadyBiodegradability'] = csv_data['ReadyBiodegradability'].replace({'NRB': 0, 'RB': 1})
        csv_data = csv_data.iloc[indices_csv]
        sdf_data = sdf_data.iloc[indices_sdf]
        all_data = pd.concat([csv_data, sdf_data], axis = 0)
        dataset = []
        print(f"Creating {self.save_name} dataset")
        for _ , mol in tqdm(all_data.iterrows(), total = all_data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol['SMILES'])
            if mol_obj is None or mol_obj.GetNumAtoms() <= 1:
            # Skip this molecule and continue to the next
                continue
            x = self._get_node_features(mol_obj)
            edge_index  = self._get_adjacency(mol_obj)
            edge_attr = self._get_edge_features(mol_obj)
            fingerprint = self._get_maccs_fp(mol_obj)
            fingerprint = fingerprint.unsqueeze(dim=0)
            label = self._get_label(mol['ReadyBiodegradability'])
            num_nodes = mol_obj.GetNumAtoms()

            data = Data(
            x = x,
            edge_index = edge_index,
            edge_attr = edge_attr,
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
        node_feats = node_feat.clone().detach()
        return node_feats
    
    def _get_adjacency(self, mol):
        src , dest = [] , []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]
        edge_index = torch.asarray([src, dest], dtype = torch.int) # Needs to be in COO Format
        return edge_index 
    
    def _get_edge_features(self, mol):
        features = []
        for bond in mol.GetBonds():
            features += 2 * [bond_features(bond)]
        edge_feat = torch.tensor(features, dtype=torch.float)
        return edge_feat
    
    def _get_maccs_fp(self, mol):
        fp_maccs = list(AllChem.GetMACCSKeysFingerprint(mol))
        return torch.tensor(fp_maccs, dtype = torch.float32)
    
    def _get_label(self, label):
        return torch.asarray([label], dtype = int)
    
if __name__ == '__main__':
    dataset = MolecularDataset(root = '', 
                            path_to_ind_csv = 'split_ix/csv_train_ix.txt',
                            path_to_ind_sdf = 'split_ix/sdf_train_ix.txt',
                                save_name='train')
    dataset = MolecularDataset(root = '', 
                            path_to_ind_csv = 'split_ix/csv_valid_ix.txt',
                            path_to_ind_sdf = 'split_ix/sdf_valid_ix.txt',
                                save_name='valid')
    dataset = MolecularDataset(root = '', 
                            path_to_ind_csv = 'split_ix/csv_test_ix.txt',
                            path_to_ind_sdf = 'split_ix/sdf_test_ix.txt',
                                save_name='test')
    delete_paths = ['processed/pre_filter.pt', 'processed/pre_transform.pt']
    for path in delete_paths:
        if os.path.exists(path):
            os.remove(path)
    