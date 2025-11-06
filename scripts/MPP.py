import os
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, GraphNorm, global_add_pool, GATv2Conv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchvision as tv
from torchvision.models.densenet import _densenet
import time
import pandas as pd
import numpy as np
import random
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from func_gnn import smiles_to_graph
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    PairTensor,
    Size,
    SparseTensor
)
import typing
from typing import Callable, Optional, Union, Tuple


torch.set_float32_matmul_precision('high')

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AWSConv(MessagePassing):
    def __init__(self, 
                 M_nn: nn.Module, 
                 hidden_dim: int = 64,
                 eps: float = 0.,
                 train_eps: bool = False, 
                 edge_dim: Optional[int] = None,
                 **kwargs
                 ) -> None:
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.nn = M_nn
        self.initial_eps = eps
        
        if train_eps:
            self.eps = nn.Parameter(torch.empty(1)) # Make eps trainable
        else:
            self.register_buffer('eps', torch.empty(1))
        
        # Edge transform layer
        self.lin = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU()
        )

        # Custom intermediate MLPs
        self.nn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.SiLU()
        )
        
        self.nn3 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.SiLU()
        )
  
    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
            size: Size = None,
        ) -> Tensor:
    
        if isinstance(x, Tensor):
            x = (x, x)

        # out: [num_atom, hidden_dim]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        
        # x_r: [num_atom, hidden_dim]
        x_r = self.nn1(x[1])
        
        # out: [num_atom, hidden_dim*2]
        out = torch.cat([out,x_r],dim=1)
        
        # out: [num_atom, hidden_dim]
        out = self.nn3(out)
        
            
        return self.nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise None

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return self.nn2(torch.cat([x_i,x_j,edge_attr],dim=1))

class MultiModalDataset(Dataset):
    def __init__(self, 
                 dataframe,
                 tokenizer,
                 max_length: int = 200,
                 img_dim = 256,
                 ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_dim = img_dim
        self.img_transform = tv.transforms.ToTensor()
        self.smiles = dataframe.iloc[:,0].tolist()
        self.target = dataframe.iloc[:,1].tolist()
        #self.img_data_dir = img_data_dir

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi_i = self.smiles[idx]
        val_i = self.target[idx]
        
        graph = smiles_to_graph(smi_i,val_i)
        graph.smiles = smi_i
        if graph is None or graph.x.size(0) == 0 or graph.edge_index.size(1) == 0:
            return None
        
        # SMILES Tokenization (ChemBERTa)
        try:
            tokenized = self.tokenizer(
                smi_i,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
                )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
        except Exception:
            return None
    
        # Smiles -> img
        mol = Chem.MolFromSmiles(smi_i)
        if mol is None:
            raise ValueError("Invalid SMILES for image.")
        
        img = Draw.MolToImage(mol, size=(self.img_dim, self.img_dim))
        img = self.img_transform(img)

        target = torch.tensor(val_i, dtype=torch.float)
        
        return {
            'graph': graph,
            'input_ids': input_ids,
            "attention_mask": attention_mask,
            "img": img,
            "target": target
            }
    
class MM_Model(pl.LightningModule):
    def __init__(self, 
                 hidden_dim=128, 
                 lr = 1e-3,
                 num_gnn_layer = 5,
                 num_RL = 5,
                 reg_type = 'cat',
                 block_config=(4, 6, 8, 6),
                 Dr = 0.1
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.train_losses = []
        self.val_losses = []
        self.lr = lr
        self.start_time = None
        self.reg_type = reg_type

        # Graph encoder (AWSConv stack)
        self.gnn_pre = nn.Sequential(
            nn.Linear(13,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
            )
        
        self.gnn = nn.ModuleList()
        for _ in range(num_gnn_layer):
            self.gnn.append(AWSConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()),
                edge_dim=5,
                hidden_dim=hidden_dim))

        self.gnn_proj = nn.LayerNorm(hidden_dim)

        # Text encoder (ChemBERTa)
        self.text_encoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.text_proj = nn.LayerNorm(self.text_encoder.config.hidden_size*2)
        self.text_out = nn.Linear(self.text_encoder.config.hidden_size*2, hidden_dim)
        
        # Image encoder (DensenetLite)
        self.image_encoder = _densenet(12, block_config, 32, None, True)
        self.image_encoder.classifier = nn.Identity()
        self.image_proj = nn.LayerNorm(self.image_encoder.features[-1].num_features)
        self.image_out = nn.Linear(self.image_encoder.features[-1].num_features, hidden_dim)
        
        # Weighted fusion layers
        if reg_type == 'weighted':
            self.wg = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.wt = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.wi = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Cross-attention projection layers
        if reg_type == 'crossattention':
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                                    num_heads=8, 
                                                    batch_first=True)

        # Output MLP (shared for all types)
        if reg_type == 'cat':
            mlp_input_dim = hidden_dim * 3
        elif reg_type == 'weighted':
            mlp_input_dim = hidden_dim * 3
        elif reg_type == 'crossattention':
            mlp_input_dim = hidden_dim * 6
          
        self.reg_mlp = nn.ModuleList()
        self.reg_mlp.append(nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=Dr)
            ))
        
        for _ in range(num_RL-2):
            self.reg_mlp.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(p=Dr)
            ))

        self.reg_mlp.append(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        ))
            
        self.reg_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, batch):
        device = next(self.parameters()).device
        batch['graph'] = batch['graph'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['img'] = batch['img'].to(device)
        
        z_g = self.encode_graph(batch['graph'])
        z_t = self.encode_text(batch['input_ids'], batch['attention_mask'])
        z_i = self.encode_image(batch['img'])

        if self.reg_type == 'cat':
            z = torch.cat([z_g, z_t, z_i], dim=1)

        elif self.reg_type == 'weighted':
            g_proj = self.wg(z_g)
            t_proj = self.wt(z_t)
            i_proj = self.wi(z_i)
            z = torch.cat([g_proj, t_proj, i_proj], dim=1)

        elif self.reg_type == 'crossattention':
            z_g = z_g.unsqueeze(1)
            z_t = z_t.unsqueeze(1)
            z_i = z_i.unsqueeze(1)
            CA_gt, _ = self.cross_attn(z_g, z_t, z_t)
            CA_tg, _ = self.cross_attn(z_t, z_g, z_g)
            CA_gi, _ = self.cross_attn(z_g, z_i, z_i)
            CA_ig, _ = self.cross_attn(z_i, z_g, z_g)
            CA_ti, _ = self.cross_attn(z_t, z_i, z_i)
            CA_it, _ = self.cross_attn(z_i, z_t, z_t)
            z = torch.cat([CA_gt, CA_tg, CA_gi, CA_ig, CA_ti, CA_it], dim=1)
            # z = CA_gt + CA_tg + CA_gi + CA_ig + CA_ti + CA_it
            z = z.flatten(start_dim=1)
            
        for layer in self.reg_mlp:
            z = layer(z) # [batch, hidden_dim]
        
        z = self.reg_out(z) # [batch, 1]
        
        return z.squeeze(-1) # [batch]

    def encode_graph(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        
        x = self.gnn_pre(x)
        residual = x
        
        for conv in self.gnn:
            x = conv(x, edge_index, edge_attr) + residual
            residual = x
            
        x = global_add_pool(x, batch)
        
        return self.gnn_proj(x)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state
        cls_token = hidden_states[:, 0, :]
        mean_pooling = torch.mean(hidden_states, dim=1)
        
        combined = torch.cat((cls_token, mean_pooling), dim=1) 
        
        normalized_output = self.text_proj(combined)
        
        return self.text_out(normalized_output)

    def encode_image(self, img):
        x = self.image_encoder(img)
        x = self.image_proj(x)
        
        return self.image_out(x)

    def training_step(self, batch, batch_idx):
        if batch is None or batch['graph'] is None:
            return None 
        preds = self(batch)
        loss = F.mse_loss(preds, batch['target'])
        self.log("train_loss", loss, on_step =False, on_epoch=True, batch_size=batch["target"].size(0), prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss:
            self.train_losses.append(train_loss.item())
    
    def validation_step(self, batch, batch_idx):
        if batch is None or batch['graph'] is None:
            return None 
        preds = self(batch)
        loss = F.mse_loss(preds, batch['target'])
        self.log("val_loss", loss, on_step =False, on_epoch=True, batch_size=batch["target"].size(0), prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss:
            self.val_losses.append(val_loss.item())
    
    def plot_losses(self, target_name, seed, conv_type, idx):
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses, label="Train Loss", marker="o")
        plt.plot(self.val_losses, label="Validation Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"./{target_name}/{self.reg_type}/{idx}/{target_name}_{conv_type}{self.reg_type}_param{idx}_loss_plot_{seed}.png")
        plt.close()
    
    def on_train_start(self):
        self.start_time = time.time()
        print("Training started...")

    def on_train_end(self):
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.2f} seconds")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        #raise RuntimeError("모든 SMILES가 처리 실패했습니다. SMILES를 확인하세요.")
        return None

    graphs = [b['graph'] for b in batch]
    batch_graph = Batch.from_data_list(graphs)
    batch_graph.smiles = [g.smiles for g in graphs]
    return {
        'graph': torch_geometric.data.Batch.from_data_list(graphs),
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'img': torch.stack([b['img'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch])
    }

def evaluate_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            preds = model(batch)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch["target"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "preds": all_preds,
        "targets": all_targets
    }

def run_code():
    # controlable variables
    SEEDS = [17, 42, 2225]
    Smi_max_len = 200
    Batch_size = 256
    Patience = 50
    Num_epoch = 10000
    hy_params = [ # [HD, lr, num_RL, Dr]
        [128, 1e-3, 3, 0.1], # Ref
        [64, 1e-3, 3, 0.1], # HD
        [256, 1e-3, 3, 0.1],
        [512, 1e-3, 3, 0.1],
        [128, 1e-2, 3, 0.1], # LR
        [128, 1e-4, 3, 0.1],
        [128, 1e-5, 3, 0.1],
        [128, 1e-3, 1, 0.1], # RL
        [128, 1e-3, 5, 0.1],
        [128, 1e-3, 3, 0.0], # Dr
        [128, 1e-3, 3, 0.2]
        ]
    conv_type = 'GATv2'
    
    file_paths = [
            "/DATA/user_scratch/hwyook/Open_Database_test/freesolv_train.csv",
            "/DATA/user_scratch/hwyook/Open_Database_test/freesolv_val.csv",
            "/DATA/user_scratch/hwyook/Open_Database_test/freesolv_test.csv",
            "/DATA/user_scratch/hwyook/Open_Database_test/Lipophilicity_train.csv",
            "/DATA/user_scratch/hwyook/Open_Database_test/Lipophilicity_val.csv",
            "/DATA/user_scratch/hwyook/Open_Database_test/Lipophilicity_test.csv",
            "/DATA/user_scratch/hwyook/Open_Database_test/esol_train.csv",
            "/DATA/user_scratch/hwyook/Open_Database_test/esol_val.csv",
            "/DATA/user_scratch/hwyook/Open_Database_test/esol_test.csv",
            ]
    
    data_type = {'freesolv': 1, 'lipo': 2, 'esol': 3}
    
    for seed in SEEDS: 
        fix_seed(seed) # Set random number fixed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for Target_prop in ['freesolv', 'lipo', 'esol']: 
            df_list = []
            img_dir = []
            for file in file_paths[3*(data_type[Target_prop]-1):3*data_type[Target_prop]]:
                df = pd.read_csv(file, usecols=["smiles",Target_prop])
                df_list.append(df)
                img_dir.append('/DATA/user_scratch/hwyook/Open_Database_test/preprocessed_dataset/' + file[45:-4])
                    
            tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            
            train_set = MultiModalDataset(df_list[0], tokenizer, img_data_dir=img_dir[0],max_length = Smi_max_len)
            val_set = MultiModalDataset(df_list[1], tokenizer, img_data_dir=img_dir[1], max_length = Smi_max_len)
            test_set = MultiModalDataset(df_list[2], tokenizer, img_data_dir=img_dir[2], max_length = Smi_max_len)
            
            train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_set, batch_size=Batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True)
            test_loader = DataLoader(test_set, batch_size=Batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True)    
            
            for reg_type in ['cat', 'weighted', 'crossattention']:
                for idx, param in enumerate(hy_params):
                    model = MM_Model(
                        hidden_dim=param[0], 
                        lr = param[1],
                        num_gnn_layer = 5,
                        num_RL = param[2],
                        reg_type = reg_type,
                        block_config=(4, 6, 8, 6),
                        Dr = param[3]
                        )
                    
                    callbacks = [
                        EarlyStopping(monitor="val_loss", patience=Patience, mode="min"),
                        ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename=f"{Target_prop}_{conv_type}{reg_type}_param{idx}_best_model_{seed}")
                    ]
                    
                    trainer = pl.Trainer(max_epochs=Num_epoch, accelerator='auto', callbacks=callbacks, log_every_n_steps=1)
                    trainer.fit(model, train_loader, val_loader)
                    
                    # Save loss plot
                    model.plot_losses(target_name = Target_prop, seed = seed, conv_type = conv_type, idx = idx)
                    
                    # Load best model for evaluation
                    best_model_path = trainer.checkpoint_callback.best_model_path
                    best_model = MM_Model.load_from_checkpoint(
                        best_model_path,
                        hidden_dim=param[0], 
                        lr = param[1],
                        num_gnn_layer = 5,
                        num_RL = param[2],
                        reg_type = reg_type,
                        block_config=(4, 6, 8, 6),
                        Dr = param[3]
                        )
                    
                    # Evaluation
                    results = {}
                    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
                        results[split_name] = evaluate_predictions(best_model, loader, device)

                    # Parity plot
                    plt.figure(figsize=(6, 6))
                    plt.scatter(results['test']['targets'], results['test']['preds'], alpha=0.5)
                    plt.plot([min(results['test']['targets']), max(results['test']['targets'])],
                             [min(results['test']['targets']), max(results['test']['targets'])],
                             linestyle="--", color="red")  # y=x 기준선
                    plt.xlabel("Actual target")
                    plt.ylabel("Predicted target")
                    plt.title(f"Actual vs. Predicted {Target_prop}")
                    plt.savefig(f"./{Target_prop}/{reg_type}/{idx}/{Target_prop}_{conv_type}{reg_type}_param{idx}_parity_plot_{seed}.png")
                    plt.grid()
                    plt.close()
                
                    metrics = []
                    metrics.append({
                        "conv_type": conv_type,
                        "seed": seed,
                        "train_mae": results['train']['mae'],
                        "train_mse": results['train']['mse'],
                        "train_rmse": results['train']['rmse'],
                        "train_r2": results['train']['r2'],
                        "val_mae": results['val']['mae'],
                        "val_mse": results['val']['mse'],
                        "val_rmse": results['val']['rmse'],
                        "val_r2": results['val']['r2'],
                        "test_mae": results['test']['mae'],
                        "test_mse": results['test']['mse'],
                        "test_rmse": results['test']['rmse'],
                        "test_r2": results['test']['r2']
                    })
                
                    # Save results to CSV
                    metric_df = pd.DataFrame(metrics)
                    metric_df.to_csv(f'./{Target_prop}/{reg_type}/{idx}/{Target_prop}_{conv_type}{reg_type}_param{idx}_result_{seed}.csv', index=False)
        
                    # Save test results with SMILES
                    test_smiles = [b['graph']['smiles'] for b in test_loader.dataset]
                    test_df = pd.DataFrame({
                        "smiles": test_smiles,
                        "target": results['test']['targets'],
                        "prediction": results['test']['preds']
                    })
                    test_df.to_csv(f"./{Target_prop}/{reg_type}/{idx}/{Target_prop}_{conv_type}{reg_type}_param{idx}_test_predictions_{seed}.csv", index=False)
        
if __name__ == "__main__":
    run_code()
    