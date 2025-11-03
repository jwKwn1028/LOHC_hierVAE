import re
import sys
from pathlib import Path
from typing import Optional, Iterable


_SMILES_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#\\/\.%]+$")

def _is_smiles_token(tok: str) -> bool:
    """Heuristic: token looks like a SMILES fragment."""
    if not tok:
        return False
    if tok.startswith("InChI="):
        return False
    return bool(_SMILES_RE.match(tok))

def extract_smiles_from_xyz(xyz_path: Path) -> Optional[str]:
    
    try:
        lines = [ln.strip() for ln in xyz_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    except Exception:
        return None

    
    while lines and not lines[-1]:
        lines.pop()

    
    inchi_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("InChI="):
            inchi_idx = i
            break


    candidates: list[str] = []
    if inchi_idx is not None and inchi_idx - 1 >= 0:
        candidates.append(lines[inchi_idx - 1])

    if not candidates:
        for i in range(len(lines) - 1, -1, -1):
            ln = lines[i]
            if not ln or ln.startswith("InChI="):
                continue
            toks = [t for t in re.split(r"\s+", ln) if t]
            if toks and all(_is_smiles_token(t) for t in toks):
                candidates.append(ln)
                break

    for cand in candidates:
        toks = [t for t in re.split(r"\s+", cand) if t]
        smiles_tokens = [t for t in toks if _is_smiles_token(t)]
        if smiles_tokens:
            return smiles_tokens[0]  # choose the first token as the SMILES

    return None

def iter_xyz_files(path: Path) -> Iterable[Path]:
    """Yield all .xyz files from a path (file or directory, recursively)."""
    if path.is_file() and path.suffix.lower() == ".xyz":
        yield path
    elif path.is_dir():
        yield from path.rglob("*.xyz")

def xyz_to_smiles_csv(input_path: str, output_txt: str) -> None:
    in_path = Path(input_path)
    out_path = Path(output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    smiles_list: list[str] = []
    for xyz_file in iter_xyz_files(in_path):
        smi = extract_smiles_from_xyz(xyz_file)
        if smi:
            smiles_list.append(smi)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("smiles\n")
        for s in smiles_list:
            f.write(f"{s}\n")

    print(f"Wrote {len(smiles_list)} SMILES to: {out_path}")

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python xyztocsv.py <xyz_file_or_dir> <output_csv>")
        sys.exit(1)
    xyz_to_smiles_csv(sys.argv[1], sys.argv[2])