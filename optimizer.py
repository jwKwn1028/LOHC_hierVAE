#this module is for optimization over hyperparameters
#utilizing optuna and pytorch lightning
#below is the template code gpt provided

# hiervae_pl.py
"""
Drop-in starter kit to move your current training loop to PyTorch Lightning,
with hooks for KL annealing, LR anneal per N steps, W&B logging, and Optuna.

Usage (basic):
    python optimizer.py \
      --train train_processed_qm9/ \
      --vocab get_vocabresults/QM9loosekekulizevocab.txt \
      --save_dir ckpt/qm98 \
      --epoch 40 --batch_size 64 --latent_size 8 \
      --lr 5e-4 --clip_norm 5.0 --print_iter 100 \
      --anneal_iter 2000 --max_beta 0.2 --step_beta 0.02 \
      --warmup 3000 --kl_anneal_iter 300

SLURM example:
    srun --export=ALL python hiervae_pl.py ... (same args)

Notes:
- Assumes your `hgraph` package exposes: common_atom_vocab, PairVocab, HierVAE, DataFolder
- DataFolder appears to be an epoch-length iterable that yields ready-made batches.
  We wrap it into an IterableDataset so Lightning can attach a DataLoader.
- Validation is optional here (not in your original script). If you have val data,
  extend the DataModule accordingly and implement `validation_step`.
"""
from __future__ import annotations
import os
import math
# import random
import argparse
from dataclasses import dataclass
from typing import Any, Iterator

# import numpy as np
import torch
# import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Optional W&B / Optuna
try:
    import wandb  # noqa: F401
    from pytorch_lightning.loggers import WandbLogger
    _WANDB_OK = True
except Exception:
    _WANDB_OK = False

try:
    import optuna
    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
    _OPTUNA_OK = True
except Exception:
    _OPTUNA_OK = False

# --- your library ---
from hgraph import common_atom_vocab, PairVocab, HierVAE, DataFolder

# ------------------ Data plumbing ------------------
class HGraphIterable(IterableDataset):
    def __init__(self, root: str, batch_size: int):
        super().__init__()
        self.root = root
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Any]:
        # DataFolder provides a fresh iterable each epoch
        return iter(DataFolder(self.root, self.batch_size))

@dataclass
class Schedules:
    warmup: int
    kl_anneal_iter: int
    step_beta: float
    max_beta: float
    anneal_iter: int
    anneal_rate: float

# ------------------ LightningModule ------------------
class HierVAEPL(pl.LightningModule):
    def __init__(self,
                 vocab, atom_vocab,
                 rnn_type: str = 'LSTM',
                 hidden_size: int = 250,
                 embed_size: int = 250,
                 latent_size: int = 32,
                 depthT: int = 15,
                 depthG: int = 15,
                 diterT: int = 1,
                 diterG: int = 3,
                 dropout: float = 0.0,
                 lr: float = 1e-3,
                 clip_norm: float = 5.0,
                 print_iter: int = 50,
                 schedules: Schedules | None = None,
                 ):  # keep addl hparams if needed
        super().__init__()
        self.save_hyperparameters(ignore=[vocab, atom_vocab, schedules])
        self.vocab = vocab
        self.atom_vocab = atom_vocab

        # underlying model from your code
        args_like = argparse.Namespace(
            rnn_type=rnn_type, hidden_size=hidden_size, embed_size=embed_size,
            latent_size=latent_size, depthT=depthT, depthG=depthG,
            diterT=diterT, diterG=diterG, dropout=dropout,
            vocab=vocab, atom_vocab=atom_vocab,
        )
        self.model = HierVAE(args_like)

        # KL & LR schedules
        self.schedules = schedules or Schedules(
            warmup=10000, kl_anneal_iter=2000, step_beta=0.001, max_beta=1.0,
            anneal_iter=25000, anneal_rate=0.9
        )
        self.beta = 0.0
        self.lr = lr
        self.clip_norm = clip_norm
        self.print_iter = print_iter

    # ----- utility norms for logging -----
    @torch.no_grad()
    def param_norm(self) -> float:
        return math.sqrt(sum((p.norm().item() ** 2) for p in self.model.parameters()))

    @torch.no_grad()
    def grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return math.sqrt(total)

    # ----- core steps -----
    def forward(self, *batch):
        # Delegates to your model's forward that returns: loss, kl_div, wacc, iacc, tacc, sacc
        return self.model(*batch, beta=self.beta)

    def _update_beta(self):
        gs = self.global_step
        if gs >= self.schedules.warmup and self.schedules.kl_anneal_iter > 0:
            # how many anneal steps completed since warmup
            k = (gs - self.schedules.warmup) // self.schedules.kl_anneal_iter
            target = min(self.schedules.max_beta, k * self.schedules.step_beta)
            # monotonic non-decreasing
            if target > self.beta:
                self.beta = target

    def training_step(self, batch, batch_idx):
        self._update_beta()
        loss, kl_div, wacc, iacc, tacc, sacc = self.forward(*batch)

        # gradient clipping handled by Trainer(gradient_clip_val)
        # Log per-step
        self.log_dict({
            'train/loss': loss,
            'train/kl': kl_div,
            'train/wacc': wacc * 100.0,
            'train/iacc': iacc * 100.0,
            'train/tacc': tacc * 100.0,
            'train/sacc': sacc * 100.0,
            'train/beta': torch.tensor(self.beta, device=loss.device),
        }, on_step=True, on_epoch=False, prog_bar=True)

        # occasional norms for debug
        if (self.global_step + 1) % self.print_iter == 0:
            self.log('train/pnorm', self.param_norm(), prog_bar=False)
            self.log('train/gnorm', self.grad_norm(), prog_bar=False)
        return loss

    # (Optional) implement validation_step to drive Optuna pruning / model selection
    def validation_step(self, batch, batch_idx):
        loss, kl_div, wacc, iacc, tacc, sacc = self.forward(*batch)
        self.log_dict({
            'val/loss': loss,
            'val/kl': kl_div,
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.lr)

        # LR schedule: apply anneal_rate every `anneal_iter` steps
        def lr_lambda(global_step: int):
            if self.schedules.anneal_iter <= 0:
                return 1.0
            k = global_step // self.schedules.anneal_iter
            return (self.schedules.anneal_rate) ** k

        sch = LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': sch,
                'interval': 'step',
                'monitor': 'val/loss',
            }
        }

# ------------------ DataModule ------------------
class HGraphDataModule(pl.LightningDataModule):
    def __init__(self, train_root: str, batch_size: int, val_root: str | None = None):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size

    def train_dataloader(self):
        ds = HGraphIterable(self.train_root, self.batch_size)
        # DataFolder already makes batched samples, so keep batch_size=None
        return DataLoader(ds, batch_size=None, num_workers=0)

    def val_dataloader(self):
        if self.val_root is None:
            return None
        ds = HGraphIterable(self.val_root, self.batch_size)
        return DataLoader(ds, batch_size=None, num_workers=0)

# ------------------ CLI & entry ------------------

def build_vocab(vocab_path: str):
    vocab = [x.strip("\r\n ").split() for x in open(vocab_path)]
    return PairVocab(vocab)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--val', default=None, help='optional validation folder')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--seed', type=int, default=7)

    # model hparams
    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)

    # training hparams
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    parser.add_argument('--print_iter', type=int, default=50)

    # schedules
    parser.add_argument('--step_beta', type=float, default=0.001)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--kl_anneal_iter', type=int, default=2000)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=25000)

    # PL Trainer knobs
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32', help='32, 16, bf16')
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--limit_train_batches', type=float, default=1.0)

    # Optional W&B
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='hiervae')
    parser.add_argument('--wandb_run', type=str, default=None)

    # Optional Optuna quickstart
    parser.add_argument('--optuna', action='store_true', help='run a small Optuna search (requires --val)')
    parser.add_argument('--optuna_trials', type=int, default=10)

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    vocab = build_vocab(args.vocab)
    atom_vocab = args.atom_vocab

    schedules = Schedules(
        warmup=args.warmup,
        kl_anneal_iter=args.kl_anneal_iter,
        step_beta=args.step_beta,
        max_beta=args.max_beta,
        anneal_iter=args.anneal_iter,
        anneal_rate=args.anneal_rate,
    )

    # --- Logger ---
    logger = None
    if args.wandb and _WANDB_OK:
        logger = WandbLogger(project=args.wandb_project, name=args.wandb_run)
    elif args.wandb and not _WANDB_OK:
        print('[WARN] wandb not installed; proceeding without it.')

    # --- Callbacks ---
    ckpt_cb = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='model-{epoch:02d}-{global_step}',
        save_top_k=3,
        monitor='val/loss' if args.val else None,
        mode='min'
    )
    lr_cb = LearningRateMonitor(logging_interval='step')
    bar_cb = TQDMProgressBar(refresh_rate=10)

    # --- Data ---
    dm = HGraphDataModule(args.train, args.batch_size, args.val)

    def make_model(lr=args.lr, latent_size=args.latent_size, hidden_size=args.hidden_size,
                   embed_size=args.embed_size, depthT=args.depthT, depthG=args.depthG,
                   diterT=args.diterT, diterG=args.diterG, dropout=args.dropout):
        return HierVAEPL(
            vocab=vocab, atom_vocab=atom_vocab,
            rnn_type=args.rnn_type, hidden_size=hidden_size, embed_size=embed_size,
            latent_size=latent_size, depthT=depthT, depthG=depthG,
            diterT=diterT, diterG=diterG, dropout=dropout,
            lr=lr, clip_norm=args.clip_norm, print_iter=args.print_iter,
            schedules=schedules,
        )

    # ---------- Plain training or Optuna ----------
    if not args.optuna:
        model = make_model()
        # Optional: load checkpoint weights
        if args.load_model:
            ckpt = torch.load(args.load_model, map_location='cpu')
            state = ckpt[0] if isinstance(ckpt, tuple) else ckpt
            model.model.load_state_dict(state)

        trainer = pl.Trainer(
            max_epochs=args.epoch,
            devices=args.devices,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            precision=args.precision,
            accumulate_grad_batches=args.accumulate,
            gradient_clip_val=args.clip_norm,
            logger=logger,
            callbacks=[ckpt_cb, lr_cb, bar_cb],
            default_root_dir=args.save_dir,
            limit_train_batches=args.limit_train_batches,
        )
        trainer.fit(model, datamodule=dm)
        return

    # --------------- Minimal Optuna example ---------------
    assert args.val is not None, 'Optuna requires a validation set (`--val`).'
    if not _OPTUNA_OK:
        print('[WARN] optuna not installed; skipping search.')
        return

    def objective(trial: optuna.Trial):
        # sample a few knobs
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
        latent = trial.suggest_categorical('latent_size', [8, 16, 32, 48, 64])
        hidden = trial.suggest_categorical('hidden_size', [192, 224, 256, 320])
        dropout = trial.suggest_float('dropout', 0.0, 0.3)

        m = make_model(lr=lr, latent_size=latent, hidden_size=hidden, dropout=dropout)

        pruner_cb = PyTorchLightningPruningCallback(trial, monitor='val/loss')
        trainer = pl.Trainer(
            max_epochs=min(8, args.epoch),
            devices=args.devices,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            precision=args.precision,
            accumulate_grad_batches=args.accumulate,
            gradient_clip_val=args.clip_norm,
            logger=logger,
            callbacks=[ckpt_cb, lr_cb, bar_cb, pruner_cb],
            default_root_dir=args.save_dir,
            enable_checkpointing=False,
        )
        trainer.fit(m, datamodule=dm)
        return trainer.callback_metrics.get('val/loss').item()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.optuna_trials)

    @rank_zero_only
    def report_best():
        print('Best trial:', study.best_trial.number, study.best_params)

    report_best()


if __name__ == '__main__':
    main()
