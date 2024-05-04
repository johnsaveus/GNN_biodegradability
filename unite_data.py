from rdkit import Chem
import pandas as pd
import pickle

def parse_sdf(sdf_path:str):

    reader = Chem.SDMolSupplier(sdf_path)
    smiles_prop = []  # Smiles from property
    target = [] # RB
    for ix, mol in enumerate(reader):
        properties = mol.GetPropsAsDict()
        if mol is not None:
            #smiles_mol.append(Chem.MolToSmiles(mol)) # With canonical = False disimilarity drops to 1109
            smiles_prop.append(properties['SMILES'])
            target.append(properties['ReadyBiodegradability'])
        else:
            print(f"Invalid Mol file number {ix}")
    sdf_data = pd.DataFrame(
        {'SMILES':smiles_prop,
         'ReadyBiodegradability': target,
         'Reliability': 'Sdf'}
    )
    return sdf_data

def parse_csv(csv_path:str):
    
    df = pd.read_csv(csv_path)
    df = df[['ReadyBiodegradability', 'Reliability', 'SMILES']]
    replacements = {'RB': 1,
                    'NRB': 0}
    df['ReadyBiodegradability'] = df['ReadyBiodegradability'].replace(replacements)
    return df

def save_concat_df(sdf, csv):

    common  = pd.concat([sdf, csv], axis = 0, ignore_index = True)
    common.to_csv("data/smiles_rb.csv", index= False)

sdf_path = 'data/AllPublicnew.sdf'
csv_path = 'data/ECHA_biodegradability_new_compounds.csv'
sdf = parse_sdf(sdf_path)
csv = parse_csv(csv_path)
save_concat_df(sdf, csv)