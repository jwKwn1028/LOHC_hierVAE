import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdPartialCharges
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from typing import Any, Dict, List

# Source code from torch_geometric.utils.smiles 
x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
    'total_valence': list(range(0, 9)), #추가
    'implicit_valence': list(range(0, 9)), #추가
    'explicit_valence': list(range(0, 9)), #추가
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
    'is_in_ring': [False, True], #추가
    'is_aromatic': [False, True] #추가
}

def from_rdmol(mol: Any) -> 'torch_geometric.data.Data':
    r"""Converts a :class:`rdkit.Chem.Mol` instance to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.
    """
    assert isinstance(mol, Chem.Mol)
#    rdPartialCharges.ComputeGasteigerCharges(mol) #추가 -> high cost calculation, 추후 고려
    EState_mol = Chem.EState.EState.EStateIndices(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        row.append(x_map['total_valence'].index(atom.GetTotalValence())) #추가
        row.append(x_map['implicit_valence'].index(atom.GetImplicitValence())) #추가
        row.append(x_map['explicit_valence'].index(atom.GetExplicitValence())) #추가
        row.append(round(float(EState_mol[atom.GetIdx()]),7))# 추가
#        try:
#            row.append(float(atom.GetProp('_GasteigerCharge')[:7])) #추가
#        except:
#            row.append(0.0) #추가
        
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
        e.append(e_map['is_in_ring'].index(bond.IsInRing())) #추가
        e.append(e_map['is_aromatic'].index(bond.GetIsAromatic())) #추가
        
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    RDLogger.DisableLog('rdApp.*')  # type: ignore

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    data = from_rdmol(mol)
    data.smiles = smiles
    
    return data

def smiles_to_graph(smiles, label):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 그래프 생성
        graph = from_smiles(smiles)
        if graph is None:
            return None
        
        # 라벨 추가
        graph.y = torch.tensor([label], dtype=torch.float)
        graph.x = graph.x.to(torch.float)  # 노드 특징을 Float으로 변환
        graph.edge_index = graph.edge_index.to(torch.long)  # 엣지 인덱스는 Long 타입이어야 함
        graph.edge_attr = graph.edge_attr.to(torch.float)
        
#        # 분자 특성 추가
#        features = extract_all_mol_features(mol)        
#        graph.eFeature = features

        return graph
    
    except Exception as e:
        print(f"Error converting SMILES: {smiles}, Error: {e}")
        return None
    
def extract_all_mol_features(mol):  
    features = []
    for name, func in Descriptors.descList: # colleting 217 molecular features
        try:
            value = func(mol)
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            value = np.nan
        features.append(value)

    features = np.nan_to_num(features, nan=0.0)
    return torch.tensor(features, dtype=torch.float32).view(1, -1)
