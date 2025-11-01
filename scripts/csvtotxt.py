import pandas as pd
from pathlib import Path

def smiles_csv_txt(csv_path: str, out_path: str | None = None):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    preferred = ["unsat_SMILE", "sat_SMILE", "SMILES"]
    avail = {c.lower(): c for c in df.columns}

    target_col = None
    for name in preferred:
        key = name.lower()
        if key in avail:
            target_col = avail[key]
            break
    if target_col is None:
        raise KeyError(f"No preferred SMILES column found ({preferred}). Columns: {list(df.columns)}")

    smiles = df[target_col].dropna().astype(str).str.strip()

    if out_path is None:
        out_path = csv_path.with_suffix(f".{target_col}.txt")
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True) 


    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(smiles.tolist()))
        f.write("\n") 

    print(f"Wrote {len(smiles)} SMILES to: {out_path} (column used: {target_col})")
    return str(out_path)

if __name__ == "__main__":
    smiles_csv_txt("data/QM9/QM9-LOHC_new_molecules.csv", "data/QM9/QM9-LOHC_new_molecules.txt")

