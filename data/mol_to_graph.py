from rdkit import Chem
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
import torch
from tqdm import tqdm
import os
from rdkit.Chem import AllChem
import pubchempy as pcp
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

    def __init__(self,
                 root,
                 path_to_ind_csv,
                 path_to_ind_sdf,
                 save_name):
        
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
            edge_index , _  = self._get_adjacency(mol_obj)
            #edge_attr = self._get_edge_features(mol_obj)
            fingerprint = self._get_mixed_fp(mol_obj)
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
        node_feats = node_feat.clone().detach()
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
    
    def _get_mixed_fp(self, mol):

        fp_list = []
        #fp_pha = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        #fp_pubchem = self._get_pubchem_fp(mol)
        #fp_list.extend(fp_pha)
        fp_list.extend(fp_maccs)
        #fp_list.extend(fp_pubchem)

        return torch.tensor(fp_list, dtype = torch.float32)
    
    def _get_pubchem_fp(self, mol):

        smile = Chem.MolToSmiles(mol)
        pubchem_compound = pcp.get_compounds(smile, 'smiles')[0]
        feature = [int(bit) for bit in pubchem_compound.cactvs_fingerprint]
        return feature

    def _get_fingerprint(self, mol):

        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=100)
        fingerprint_tensor = torch.tensor(list(fingerprint), dtype=torch.float32)

        return fingerprint_tensor

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
    