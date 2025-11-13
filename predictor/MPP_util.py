import os
import sys
import pandas as pd
from tqdm import tqdm
import torch
import ast
import argparse
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from rdkit import Chem
from hgraph.mol_graph import MolGraph
from typing import List, Tuple, Dict, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Checkpoint/')))
from MPP import MM_Model, MultiModalDataset, collate_fn

DEFAULT_CKPT_MAP = {
    "esol": "ckpt/esolv0.ckpt",
    "freesolv": "ckpt/freesolvv0.ckpt",
    "lipo": "ckpt/lipov0.ckpt",
}


def canonicalize_smiles(smiles_list):
    canon_set = set()
    canon_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in canon_set:
            canon_set.add(canon)
            canon_smiles.append(canon)
    return canon_smiles


class MoleculeDatasetWithProperty(torch.utils.data.Dataset):
    def __init__(self, filepath, vocab, avocab, mode="single"):
        self.vocab = vocab
        self.avocab = avocab
        self.mode = mode

        self.smiles_list, self.prop_list = [], []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                parts = line.split('\t')
                smi = parts[0]
                '''
                self.smiles_list.append(smi)
                if mode == "single":
                    prop = parts[1]
                    self.prop_list.append(float(prop))
                elif mode == "multi":
                    props = list(map(float, parts[1:]))
                    self.prop_list.append(props)
                '''
                payload = parts[1:]  # SMILES 이후의 모든 컬럼
                # --- 유연 파싱 ---
                # 1) 리스트 문자열 한 컬럼: "[-5.27, -13.46, 0.153, 2.26]"
                if len(payload) == 1 and payload[0].startswith('[') and payload[0].endswith(']'):
                    try:
                        vals = ast.literal_eval(payload[0])
                    except Exception:
                        # 비상시: 대충 파싱
                        vals = [float(x) for x in payload[0].strip('[]').split(',')]
                    if not isinstance(vals, (list, tuple)):
                        vals = [float(vals)]
                    else:
                        vals = [float(v) for v in vals]

                # 2) 탭으로 나뉜 다컬럼: SMILES \t p1 \t p2 \t ...
                elif len(payload) > 1:
                    vals = [float(x) for x in payload]

                # 3) 스칼라 한 컬럼: SMILES \t p
                else:
                    vals = [float(payload[0])]

                # 저장 (mode에 따라 형태 결정)
                self.smiles_list.append(smi)
                if self.mode == "single":
                    self.prop_list.append(float(vals[0]))
                else:  # "multi"
                    self.prop_list.append(vals)

        # Filter valid molecules
        valid_smiles, valid_props = [], []
        for smi, prop in zip(self.smiles_list, self.prop_list):
            try:
                hmol = MolGraph(smi)
            except Exception as e:
                print(f"[SKIP] {smi} → MolGraph failed: {e}")
                continue
            ok = True
            for _, attr in hmol.mol_tree.nodes(data=True):
                smiles = attr['smiles']
                if attr['label'] not in vocab.vmap:
                    #self.vocab.vmap
                    print(f"[WARN] label '{attr['label']}' not in vocab for {smi}")
                    ok = False
                for _, s in attr['inter_label']:
                    if (smiles, s) not in vocab.vmap:
                        #self.vocab.vmap
                        print(f"[WARN] inter_label ({smiles}, {s}) not in vocab for {smi}")
                        ok = False
            if ok:
                valid_smiles.append(smi)
                valid_props.append(prop)

        print(f'[INFO] After pruning: {len(self.smiles_list)} → {len(valid_smiles)}')
        self.smiles_list = valid_smiles
        self.prop_list = valid_props

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if self.mode == "single":
            return self.smiles_list[idx], self.prop_list[idx]
        elif self.mode == "multi":
            return self.smiles_list[idx], torch.tensor(self.prop_list[idx], dtype=torch.float32)

    def collate_molgraph_with_props(self, batch):
        smiles_list, props = zip(*batch)
        mol_batch = MolGraph.tensorize(smiles_list, vocab=self.vocab, avocab=self.avocab)
        if self.mode == "single":
            prop_tensor = torch.tensor(props, dtype=torch.float32).unsqueeze(-1)#.cuda()
        elif self.mode == "multi":
            prop_tensor = torch.stack(props, dim=0).contiguous()
        return mol_batch, prop_tensor


### For single-property ###

class MPPWrapper:
    def __init__(self, checkpoint_path):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MM_Model.load_from_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)

    def predict(self, smiles_list):
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        
        dummy_targets = [0.0] * len(smiles_list)
        df = pd.DataFrame({'smiles': smiles_list, 'target': dummy_targets})

        dataset = MultiModalDataset(df, tokenizer=tokenizer, max_length = 200)
        dataloader = DataLoader(dataset, batch_size=256, collate_fn=collate_fn, num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=torch.cuda.is_available())

        all_preds, valid_smiles = [], []


        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                for k in ['graph', 'input_ids', 'attention_mask', 'img']:
                    batch[k] = batch[k].to(self.device)
                out = self.model(batch)
                all_preds.extend(torch.as_tensor(out).detach().cpu().flatten().tolist())
                valid_smiles.extend(batch['graph'].smiles)

        return list(zip(valid_smiles, all_preds))

### For multi--property ###
def normalize_scores(pairs, k_expected=None):
    out = []
    for item in pairs:   # item: (smi, preds_like)
        if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
        smi, preds = item[0], item[1]

        # 1. torch.Tensor → numpy array
        if hasattr(preds, "detach"):      
            preds = preds.detach().cpu().numpy()

        # 2. numpy array → Python list
        if hasattr(preds, "tolist"):      
            vec = list(map(float, preds.tolist()))

        # 3. 스칼라(single float)면 리스트로 감싸줌
        elif isinstance(preds, (list, tuple)):
            vec = [float(x) for x in preds]
        else:
            tail = item[1:]
            if all(isinstance(x, (int, float)) for x in tail):
                vec = [float(x) for x in tail]
            else:
                vec = [float(item)]

        # 4. 길이 확인 (예: props 3개인데 길이가 다르면 스킵)
        if k_expected is not None:
            if len(vec) >= k_expected:
                vec = vec[:k_expected]
            else:
                continue

        #out.append((smi, preds))   # 최종: (str, List[float])
        out.append((smi, vec))
    return out

def load_smiles(txt_path, *, canonicalize=False, keep_longest_fragment=False, verbose=True):
    valid = []
    invalid = []
    total = 0
    
    with open(txt_path, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            total += 1
            if keep_longest_fragment and "." in s:
                parts = [p.strip() for p in s.split(".") if p.strip()]
                if parts:
                    s = max(parts, key=len)
            m = Chem.MolFromSmiles(s)
            if m is None:
                 invalid.append((lineno, raw.rstrpi("\n")))
                 continue
            
            if canonicalize:
                s = Chem.MolToSmiles(m)
            valid.append(s)
    if verbose:
        n_bad = len(invalid)
        pct_bad = (100.0 * n_bad / total) if total > 0 else 0.0
        print(f"[load_smiles] Read {total} lines (non-empty). "
              f"Valid: {total - n_bad}, Invalid: {n_bad} "
              f"({pct_bad:.2f}% not SMILES).")
        if n_bad > 0:
            print("[load_smiles] Invalid entries (line_no: text):")
            for ln, bad in invalid:
                print(f"  {ln}: {bad}")

    return valid
            

class MultiMPPWrapper:
    def __init__(self, ckpt_paths):
        self.models = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for p in ckpt_paths:
            m = MM_Model.load_from_checkpoint(p)
            m.eval().to(self.device)
            self.models.append(m)
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    @torch.no_grad()
    def predict_raw(self, smiles_list):
        
        dummy_targets = [0.0] * len(smiles_list)
        df = pd.DataFrame({'smiles': smiles_list, 'target': dummy_targets})

        dataset = MultiModalDataset(df, tokenizer=self.tokenizer, max_length = 200)
        dataloader = DataLoader(dataset, batch_size=256, collate_fn=collate_fn, num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=torch.cuda.is_available())
        

        all_valid_smiles = []
        all_preds_by_model = [ [] for _ in self.models ]

        for batch in dataloader: 
            if batch is None:
                continue
            for k in ['graph', 'input_ids', 'attention_mask', 'img']:
                batch[k] = batch[k].to(self.device)
            for i, m in enumerate(self.models):
                out = m(batch)
                # all_preds_by_model[i].extend(out.detach().cpu().numpy().reshape(-1).tolist())
                all_preds_by_model[i].extend(torch.as_tensor(out).detach().cpu().flatten().tolist())
            all_valid_smiles.extend(batch['graph'].smiles)
        
        preds_matrix = list(zip(*all_preds_by_model))
        pairs = list(zip(all_valid_smiles, preds_matrix))
        
        return [(s, list(v)) for s, v in pairs]
    
    @torch.no_grad()
    def predict(self,smiles_list):
        pairs = self.predict_raw(smiles_list)
        return normalize_scores(pairs, k_expected=len(self.models))

### for Multi-property with sa score ###
def _compute_sa_scores(smiles_list):
    """
    SA(SAS) 점수 계산.
    1) 우선 RDKit의 sascorer / SA_Score가 있으면 표준 SA 점수 사용
    2) 없으면 RDKit 지표 기반의 proxy 점수로 대체
    Returns: (scores: List[float], mode: 'exact'|'proxy'|'none')
    """
    try:
        scorer = None
        # 1) Contrib sascorer 먼저 시도
        try:
            import props.sascorer
            def scorer(m):
                return props.sascorer.calculateScore(m)
        except Exception:
            # 2) rdkit.Chem.SA_Score 모듈 시도
            try:
                from rdkit.Chem import SA_Score as RDKitSAS
                if hasattr(RDKitSAS, "sascorer"):
                    def scorer(m):
                        return RDKitSAS.sascorer(m)
                elif hasattr(RDKitSAS, "calculateScore"):
                    def scorer(m):
                        return RDKitSAS.calculateScore(m)
            except Exception:
                scorer = None

        if scorer is not None:
            scores = []
            for s in tqdm(smiles_list, desc="[SA] RDKit SA_Score", leave=False):
                m = Chem.MolFromSmiles(s)
                if m is None:
                    scores.append(float("nan"))
                    continue
                try:
                    scores.append(float(scorer(m)))
                except Exception:
                    scores.append(float("nan"))
            return scores, "exact"

        # 3) Proxy 계산(간단 지표 기반)
        
        from rdkit.Chem import rdMolDescriptors as rdMD
        import math
        scores = []
        for s in tqdm(smiles_list, desc="[SA] proxy (fallback)", leave=False):
            m = Chem.MolFromSmiles(s)
            if m is None:
                scores.append(float("nan"))
                continue
            try:
                n_atoms = m.GetNumHeavyAtoms()
                n_rings = rdMD.CalcNumRings(m)
                n_arom = rdMD.CalcNumAromaticRings(m)
                n_spiro = rdMD.CalcNumSpiroAtoms(m)
                n_bridge = rdMD.CalcNumBridgeheadAtoms(m)
                chiral = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
                ri = m.GetRingInfo()
                max_ring = max((len(r) for r in ri.AtomRings()), default=0)
                macro = 1 if max_ring >= 8 else 0

                # 아주 러프한 proxy: 값이 1~10 근처로 나오게 스케일링
                sa = 0.6*math.log1p(n_atoms) + 0.5*n_rings + 0.3*n_arom \
                     + 0.3*(n_spiro+n_bridge) + 0.3*chiral + 1.0*macro
                sa = 1.0 + 9.0 * (sa / (sa + 5.0))  # 1~10로 부드럽게 압축
                sa = max(1.0, min(10.0, sa))
                scores.append(float(sa))
            except Exception:
                scores.append(float("nan"))
        return scores, "proxy"

    except Exception as e:
        print(f"[WARN] RDKit/SA 계산 실패: {e}")
        return [float("nan")] * len(smiles_list), "none"
    

_SA_ALIASES = {"sas","sa","sa_score","sascorer"}
def _is_sa_name(name: str) -> bool:
    return name.lower() in _SA_ALIASES

def ensure_list_format(src_path, dst_path, k_expected):
    """
    src: 'SMILES\tp1\tp2...\tpk' 또는 'SMILES\t[ ... ]' 형식 혼용 가능
    dst: 무조건 'SMILES\t[p1, p2, ..., pk]'로 변환하여 저장
    """
    # import re
    with open(src_path) as fin, open(dst_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line: 
                continue
            parts = line.split('\t')
            smi = parts[0]
            if len(parts) == 2 and parts[1].startswith('[') and parts[1].endswith(']'):
                # 이미 리스트 한 컬럼인 경우 그대로
                fout.write(line + '\n')
            else:
                # 탭 다컬럼 -> 리스트로 묶어서 쓰기
                vals = [float(x) for x in parts[1:]]
                if len(vals) != k_expected:
                    print(f"[WARN] {smi}: expected {k_expected} props, got {len(vals)}")
                fout.write(f"{smi}\t[{', '.join(f'{v:.6g}' for v in vals)}]\n")

def _clean_and_map(smiles_list):
    """
    원본 리스트 -> (정제된 canonical smiles 리스트, 원본 index -> 정제 index 매핑)
    - 소금/점(.) 있으면 가장 긴 fragment만 사용
    - RDKit canonical SMILES로 통일
    - 실패 시 None 기록
    """
    cleaned = []
    idx_map: list[Optional[int]] = []
    for s in smiles_list:
        if not isinstance(s, str):
            idx_map.append(None)
            continue
        s = s.strip()
        if not s:
            idx_map.append(None)
            continue
        if '.' in s:
            parts = [p.strip() for p in s.split('.') if p.strip()]
            if not parts:
                idx_map.append(None)
                continue
            s = max(parts, key=len)
        m = Chem.MolFromSmiles(s)
        if m is None:
            idx_map.append(None)
            continue
        cs = Chem.MolToSmiles(m)  # canonical
        idx_map.append(len(cleaned))
        cleaned.append(cs)
    return cleaned, idx_map

class HybridScorer:
    def __init__(self, props: List[str], ckpt_map: Dict[str,str], sa_fn=_compute_sa_scores):
        self.props = list(props)
        # 모델로 예측할 prop들만 props 순서대로 추출
        self.model_props = [p for p in self.props if (p in ckpt_map) and not _is_sa_name(p)]
        # 모델 ckpt들을 model_props 순서대로 배열
        ckpt_list = [ckpt_map[p] for p in self.model_props]
        self.nn = MultiMPPWrapper(ckpt_list) if ckpt_list else None
        self.need_sa = any(_is_sa_name(p) for p in self.props)
        self.sa_fn = sa_fn
        # model prop -> index 맵
        self.idx_of = {p:i for i,p in enumerate(self.model_props)}
    
    def predict(self, smiles_list: List[str]) -> List[Tuple[str, List[float]]]:
        # 0) 원본 -> 정제(canonical) + 매핑
        cleaned, idx_map = _clean_and_map(smiles_list)

        # 1) 모델 예측 (정제 리스트 기준, 키=canonical)
        model_map = {}
        if self.nn is not None and len(cleaned) > 0:
            nn_pairs = self.nn.predict_raw(cleaned)  # [(canon_smi, [vals in model_props order])]
            for cs, vec in nn_pairs:
                model_map[cs] = vec

        # 2) SA 계산 (정제 리스트 기준)
        sa_scores_clean = None
        if self.need_sa and len(cleaned) > 0:
            scores, *_mode = self.sa_fn(cleaned)  # (scores, mode) 호환
            if isinstance(scores, tuple):  # 혹시 (scores, mode) 그대로 올 경우 대비
                scores = scores[0]
            sa_scores_clean = [float(x) for x in scores]

        # 3) 원본 순서로 복원
        out = []
        for i, ori in enumerate(smiles_list):
            j = idx_map[i]
            if j is None:
                # 정제 실패 → 전부 NaN
                vec = [float('nan')] * len(self.props)
            else:
                key = cleaned[j]
                vec = []
                for p in self.props:
                    if _is_sa_name(p):
                        v = float('nan') if sa_scores_clean is None else sa_scores_clean[j]
                        vec.append(v)
                    elif p in self.idx_of:
                        mv = model_map.get(key)
                        vec.append(float('nan') if mv is None else float(mv[self.idx_of[p]]))
                    else:
                        vec.append(float('nan'))
            out.append((ori, vec))
        return out
        '''
        # 1) 모델 기반 예측(raw)
        model_map = {}
        if self.nn is not None and len(self.model_props) > 0:
            nn_pairs = self.nn.predict_raw(smiles_list)  # [(smi, [vals in model_props order])]
            for smi, vec in nn_pairs:
                model_map[smi] = vec

        # 2) SA 점수 (함수형)
        sa_scores = None
        if self.need_sa:
            scores, *_mode = self.sa_fn(smiles_list)  # _compute_sa_scores가 (scores, mode) 반환해도 OK
            # scores만 쓰면 됨
            if isinstance(scores, tuple):  # 안전장치
                scores = scores[0]
            sa_scores = scores

        # 3) props 순서대로 벡터 구성
        out = []
        for i, smi in enumerate(smiles_list):
            vec = []
            for p in self.props:
                if _is_sa_name(p):
                    vec.append(float('nan') if sa_scores is None else float(sa_scores[i]))
                elif p in self.idx_of:
                    mv = model_map.get(smi)
                    vec.append(float('nan') if mv is None else float(mv[self.idx_of[p]]))
                else:
                    vec.append(float('nan'))  # 정의되지 않은 prop
            out.append((smi, vec))
        return out
    '''
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True, help = "smiles.txt")
    ap.add_argument("--out", required=True, help="results.csv")
    ap.add_argument("--props", required=True, help='comma list of property')
    ap.add_argument("--ckpt_map", required=True, help="JSON mapping prop->ckpt")
    args = ap.parse_args()
    
    props = [p.strip() for p in args.props.split(",") if p.strip()]
    cli_ckpt_map = json.loads(args.ckpt_map)
    ckpt_map = DEFAULT_CKPT_MAP.copy()
    ckpt_map.update(cli_ckpt_map)
    smiles = load_smiles(args.smiles)
    scorer = HybridScorer(props, ckpt_map)
    pairs = scorer.predict(smiles)

    # Save: one column per prop (NaN if unavailable)
    df = pd.DataFrame(
        [(s, *vals) for s, vals in pairs],
        columns=["smiles"] + props
    )
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")