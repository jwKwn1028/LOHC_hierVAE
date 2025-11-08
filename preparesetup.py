# do the steps
# get vocab, preprocess, train, fintune, generate in one step
# merge get_vocab and preprocessing step.
import sys
import os
from pathlib import Path
import argparse 
from hgraph.mol_graph import MolGraph
from multiprocessing import Pool
import random
import pickle
from functools import partial
import torch
import numpy
from hgraph import common_atom_vocab, PairVocab  
import rdkit

def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        try:
            hmol = MolGraph(s)
        except Exception as e:
            print(f"Skipping problematic SMILES-rdkit_KekulizeException: {s} ({e})", file=sys.stderr)
            continue
        for _, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for _, s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab


def to_numpy(tensors):
    def convert(x):
        return x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)

def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) #no need of order for x

def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,) #no need of order for x

#i if vocab can contain both strings and 2-tuples
def to_line(item):
    if isinstance(item, tuple) and len(item) == 2:
        return f"{item[0]} {item[1]}\n"
    return f"{item}\n"




if __name__ == "__main__":


    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser(description="Generate vocab from <SMILES.txt> and preprocess for training")
    parser.add_argument('--ncpu', type=int, default=8)
    parser.add_argument('--input', required=True, help="path to input .txt containing smiles")
    parser.add_argument('--vocab', type=str, required=True, help="path to read generated vocab.txt")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='single', help="single, pair, cond_pair")
    parser.add_argument('--path', type=str, required=True, help="Directory to store preprocessed tensors")
    args = parser.parse_args()

    # Generating Vocab from smiles.txt file
    in_path = Path(args.input)
    outdir: Path = Path("vocab")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir/in_path.name
    
    with in_path.open("r", encoding="utf-8") as f:
        data = [mol for line in f for mol in line.split()[:2]]
    data = list(set(data))

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    with Pool(args.ncpu) as pool:
        vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    with out_path.open("w", encoding="utf-8") as smiles:
        smiles.writelines(to_line(smile) for smile in sorted(vocab, key=str))
    






    if args.mode == 'pair':
        #dataset contains molecule pairs
        with open(args.train) as f:
            pairdata = [line.strip("\r\n ").split()[:2] for line in f]

        random.shuffle(pairdata)

        batches = [pairdata[i : i + args.batch_size] for i in range(0, len(pairdata), args.batch_size)]
        func = partial(tensorize_pair, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'cond_pair':
        #dataset contains molecule pairs with conditions
        with open(args.train) as f:
            cond_pair_data = [line.strip("\r\n ").split()[:3] for line in f]

        random.shuffle(cond_pair_data)

        batches = [cond_pair_data[i : i + args.batch_size] for i in range(0, len(cond_pair_data), args.batch_size)]
        func = partial(tensorize_cond, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'single':
        #dataset contains single molecules
        with open(args.train) as f:
            single_data = [line.strip("\r\n ").split()[0] for line in f]

        random.shuffle(single_data)

        batches = [single_data[i : i + args.batch_size] for i in range(0, len(single_data), args.batch_size)]
        func = partial(tensorize, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)        