import pandas as pd
import os
import sys
from rdkit import Chem
from hgraph.mol_graph import MolGraph
from tqdm import tqdm

def is_kekulizable(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        return True
    except:
        return False

def contains_only_common_atoms(smi, allowed_atoms):
    try:
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in allowed_atoms:
                return False
        return True
    except:
        return False
        

COMMON_ATOMS = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1)]

def preprocess_inputcsv(filename, df):
    out_dir = f"./data/{filename}"
    os.makedirs(out_dir, exist_ok=True)

    invalid_smiles = []         
    valid_smiles   = []        

    
    kek_mask = df["smiles"].apply(is_kekulizable)
    invalid_smiles.extend(df.loc[~kek_mask, "smiles"].tolist())
    df = df[kek_mask].copy() 

    
    allowed_atoms = {atom for atom, _ in COMMON_ATOMS}
    common_mask = df["smiles"].apply(lambda s: contains_only_common_atoms(s, allowed_atoms))
    invalid_smiles.extend(df.loc[~common_mask, "smiles"].tolist())
    df = df[common_mask].copy()

    for smi in tqdm(df["smiles"], desc="MolGraph filtering"):
        try:
            _ = MolGraph(smi)
            valid_smiles.append(smi)
        except Exception as e:
            invalid_smiles.append(smi)
            print(f"❌ Error on SMILES: {smi} | {e}")

    # Save
    pd.DataFrame(valid_smiles,   columns=["smiles"]).to_csv(f"{out_dir}/train.txt",   index=False, header=False)
    pd.DataFrame(invalid_smiles, columns=["smiles"]).to_csv(f"{out_dir}/invalid.txt", index=False, header=False)
    print(f"saved: {len(valid_smiles)}valid smiles → train.txt, {len(invalid_smiles)}invalid smiles → invalid.txt")
        

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preprocess SMILES CSV into train/invalid splits.")
    ap.add_argument("-i", "--input", required=True, help="Path to input CSV containing a 'smiles' column")
    ap.add_argument("-o", "--outname", default="dataset", help="Name for ./data/<outname>/ outputs")
    ap.add_argument("-c", "--col", default="smiles", help="Column name containing SMILES")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.col != "smiles" and args.col in df.columns:
        df = df.rename(columns={args.col: "smiles"})
    if "smiles" not in df.columns:
        raise SystemExit(f"No 'smiles' column found. Columns: {list(df.columns)}")

    preprocess_inputcsv(args.outname, df)
