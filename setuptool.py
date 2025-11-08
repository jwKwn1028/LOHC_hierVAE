#!/usr/bin/env python3
import sys
import argparse
import random
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy
import torch
from rdkit import RDLogger

# --- your packages ---
# If MolGraph lives in hgraph.mol_graph in your environment, import from there for both commands.
from hgraph.mol_graph import MolGraph
from hgraph import common_atom_vocab, PairVocab


# ---------------------------
# Utilities
# ---------------------------
def chunked(lst, size):
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def to_numpy(tensors):
    def convert(x):
        return x.numpy() if isinstance(x, torch.Tensor) else x
    a, b, c = tensors
    b = ([convert(x) for x in b[0]], [convert(x) for x in b[1]])
    return a, b, c


def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)


def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y)  # no need of order for x


def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [list(map(int, c.split(','))) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,)  # no need of order for x


# ---------------------------
# VOCAB COMMAND
# ---------------------------
def _process_vocab_pairs(lines):
    pairs = set()
    for line in lines:
        s = line.strip("\r\n ")
        if not s:
            continue
        try:
            hmol = MolGraph(s)
        except Exception as e:
            print(f"Skipping problematic SMILES-rdkit_KekulizeException: {s} ({e})", file=sys.stderr)
            continue

        for _, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            pairs.add( attr['label'] )
            for _, s in attr['inter_label']:
                pairs.add( (smiles, s) )
    return pairs




def cmd_vocab(args):
    in_path = Path(args.input).expanduser().resolve()
    outdir: Path = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # pick your preferred naming; here: <stem>_vocab<suffix>
    out_path = outdir / f"{in_path.stem}_vocab{in_path.suffix}"

    # Use first two tokens per line (same as original)
    with in_path.open("r", encoding="utf-8") as f:
        mols = [mol for line in f for mol in line.split()[:2]]
    mols = list(set(mols))

    ncpu = max(int(args.ncpu), 1)
    batch_size = len(mols) // ncpu + 1
    batches = [mols[i:i+batch_size] for i in range(0, len(mols), batch_size)]

    with Pool(ncpu) as pool:
        results = pool.map(_process_vocab_pairs, batches)

    # merge & coerce to (str, str) (already done in worker, but safe)
    pairs = set()
    for P in results:
        pairs |= {(str(a), str(b)) for (a, b) in P}

    # Write exactly two columns per line (original script’s output form)
    with out_path.open("w", encoding="utf-8") as fh:
        for x, y in sorted(pairs, key=lambda t: (str(t[0]), str(t[1]))):
            fh.write(f"{x} {y}\n")


    print(f"[vocab] wrote {len(pairs)} pairs → {out_path}", file=sys.stderr)



# ---------------------------
# TENSORIZE COMMAND
# ---------------------------
def cmd_tensorize(args):
    # Keep RDKit quiet
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    train_path = Path(args.train).expanduser().resolve()
    vocab_path = Path(args.vocab).expanduser().resolve()
    outdir: Path = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load vocab file (expects space-separated pairs per line)
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_rows = [r.split() for r in (x.strip("\r\n ") for x in f) if r]
    vocab_rows = [r for r in vocab_rows if len(r) == 2]  # <-- keep only (x, y)
    pvocab = PairVocab(vocab_rows, cuda=False)

    ncpu = max(int(args.ncpu), 1)
    bs = max(int(args.batch_size), 1)

    with Pool(ncpu) as pool:
        random.seed(1)

        if args.mode == "pair":
            with train_path.open("r", encoding="utf-8") as f:
                data = [line.strip("\r\n ").split()[:2] for line in f if line.strip()]

            random.shuffle(data)
            batches = chunked(data, bs)
            func = partial(tensorize_pair, vocab=pvocab)
            all_data = pool.map(func, batches)

        elif args.mode == "cond_pair":
            with train_path.open("r", encoding="utf-8") as f:
                data = [line.strip("\r\n ").split()[:3] for line in f if line.strip()]

            random.shuffle(data)
            batches = chunked(data, bs)
            func = partial(tensorize_cond, vocab=pvocab)
            all_data = pool.map(func, batches)

        elif args.mode == "single":
            with train_path.open("r", encoding="utf-8") as f:
                data = [line.strip("\r\n ").split()[0] for line in f if line.strip()]

            random.shuffle(data)
            batches = chunked(data, bs)
            func = partial(tensorize, vocab=pvocab)
            all_data = pool.map(func, batches)

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    # Split and write pickles
    num_splits = max(len(all_data) // 1000, 1)
    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]
        out_path = outdir / f"tensors-{split_id}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
        print(f"[tensorize] Wrote {out_path}", file=sys.stderr)


# ---------------------------
# Main CLI
# ---------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Chemistry data utilities: vocab builder and tensorizer"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # vocab subcommand
    pv = sub.add_parser("vocab", help="Build vocab file from input data")
    pv.add_argument("--input", "-i", type=Path, required=True,
                    help="Path to input data file (text)")
    pv.add_argument("--outdir", type=Path, default=Path("vocab"),
                    help="Directory to write vocab file (default: ./vocab)")
    pv.add_argument("--ncpu", type=int, default=1, help="CPU workers (default: 1)")
    pv.set_defaults(func=cmd_vocab)

    # tensorize subcommand
    pt = sub.add_parser("tensorize", help="Tensorize dataset using a vocab")
    pt.add_argument("--train", required=True, type=Path, help="Training data file")
    pt.add_argument("--vocab", required=True, type=Path, help="Vocab file")
    pt.add_argument("--batch_size", type=int, default=32, help="Batch size")
    pt.add_argument("--mode", type=str, default="single",
                    choices=["pair", "cond_pair", "single"], help="Tensorization mode")
    pt.add_argument("--ncpu", type=int, default=8, help="CPU workers")
    pt.add_argument("--outdir", type=Path, default=Path("processed"), help="Output dir for tensors (default: ./processed)")
    pt.set_defaults(func=cmd_tensorize)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
