import pickle
from pathlib import Path
import sys
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from muon import SingleDeviceMuonWithAuxAdam

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.model import *
from src.training import train_gnn_epoch, validate_gnn_epoch, calculate_metrics, save_results, compute_rmse
from src.cross_validation import create_stratified_k_folds_by_distance, create_random_k_folds, save_cv_splits, load_cv_splits

# Configuration
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "galaxy_graphs.pkl"
CV_SPLIT_PATH = RESULTS_DIR / "cv_galaxy_splits.json"
GNN_RESULTS_DIR = RESULTS_DIR / "gnn"
GNN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_FOLDS = 5
SEED = 42

# GNN hyperparameters
GNN_CONFIG = {
    'lr': 1e-2, 
    'wd': 0, 
    'n_epochs': 150, 
    'batch_size': 8,
    # 'lr_step_down': [0.5, 0.85, 0.95],
    # 'lr_div_factor': 5,
    # '1cycle_pct_start': 0.2,
    # '1cycle_div_factor': 10,
    # '1cycle_final_div_factor': 1000,
    'muon_lr': 1e-2,
    'muon_wd': 1e-6,
    'n_layers': 1, 
    'hidden_channels': 256, 
    'latent_channels': 64,
    'n_unshared_layers': 16,
}

def main():
    
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_dict = pickle.load(f)

    galaxy_names = list(data_dict.keys())
    # galaxy_distances = pd.Series({name: data.u[0].item() for name, data in data_dict.items()}, name="D")

    # cv_splits = create_stratified_k_folds_by_distance(
    #     pd.DataFrame(index=galaxy_names).join(galaxy_distances), 
    #     k=K_FOLDS, 
    #     seed=SEED
    # )
    cv_splits = create_random_k_folds(
        pd.DataFrame(index=galaxy_names),
        k=K_FOLDS,
        seed=SEED
    )
    save_cv_splits(cv_splits, CV_SPLIT_PATH)

    all_preds_df = []
    all_metrics = []
    
    for i, fold in enumerate(tqdm(cv_splits, desc="Cross-validation folds")):
        train_galaxies, valid_galaxies = fold['train'], fold['valid']
        
        train_data = [data_dict[g] for g in train_galaxies]
        valid_data = [data_dict[g] for g in valid_galaxies]

        train_loader = DataLoader(train_data, shuffle=True)
        valid_loader = DataLoader(valid_data, shuffle=False)

        n_node_features = train_data[0].x.shape[-1]
        n_edge_features = train_data[0].edge_attr.shape[-1] if hasattr(train_data[0], "edge_attr") else 0
        n_graph_features = train_data[0].u.shape[-1] if hasattr(train_data[0], "u") else 0

        model = EdgeInteractionGNN(
            n_layers=GNN_CONFIG['n_layers'], 
            hidden_channels=GNN_CONFIG['hidden_channels'],
            latent_channels=GNN_CONFIG['latent_channels'], 
            n_unshared_layers=GNN_CONFIG['n_unshared_layers'],
            node_features=n_node_features,
            edge_features=n_edge_features,
            graph_features=n_graph_features,
        ).to(DEVICE)

        # optimizer = torch.optim.AdamW(model.parameters(), lr=GNN_CONFIG['lr'], weight_decay=GNN_CONFIG['wd'])

        hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]
        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=GNN_CONFIG["muon_lr"], weight_decay=GNN_CONFIG["muon_wd"]),
            dict(params=hidden_gains_biases, use_muon=False, lr=GNN_CONFIG["lr"], betas=(0.9, 0.95), weight_decay=GNN_CONFIG["wd"]),
        ]
        
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        
        train_losses, valid_losses, valid_rmses = [], [], []
        epoch_pbar = tqdm(range(GNN_CONFIG['n_epochs']), desc=f"Fold {i} Training", leave=False)
        
        for epoch in epoch_pbar:
            # if epoch in [int(frac * GNN_CONFIG['n_epochs']) for frac in GNN_CONFIG['lr_step_down']]:
            #     optimizer.param_groups[0]['lr'] /= GNN_CONFIG['lr_div_factor']
            
            train_loss = train_gnn_epoch(train_loader, model, optimizer, DEVICE)
            valid_loss, p, y = validate_gnn_epoch(valid_loader, model, DEVICE)

            # scheduler.step()

            valid_rmse = compute_rmse(p.flatten(), y.flatten())
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_rmses.append(valid_rmse)
            epoch_pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'valid_loss': f'{valid_loss:.4f}', 'val_rmse': f'{valid_rmse:.4f}'})

        # After all epochs, get final predictions and save the final model state
        _, p_valid, y_valid = validate_gnn_epoch(valid_loader, model, DEVICE)
        torch.save(model.state_dict(), GNN_RESULTS_DIR / f"cv_gnn_fold_{i}_model.pth")
        
        metrics = calculate_metrics(p_valid.flatten(), y_valid.flatten())
        all_metrics.append(metrics)
        tqdm.write(f"Fold {i} GNN Metrics: {metrics}")
        save_results(GNN_RESULTS_DIR / f"cv_gnn_fold_{i}_metrics.json", metrics)
        
        pred_df = pd.DataFrame({'y_pred': p_valid.ravel(), 'y_true': y_valid.ravel(), 'fold': i})
        all_preds_df.append(pred_df)
        save_results(GNN_RESULTS_DIR / f"cv_gnn_fold_{i}_predictions.csv", pred_df)
        
        # Plot and save loss curve for the fold
        plt.figure(figsize=(5, 4), dpi=120)
        plt.plot(valid_rmses, label="Valid RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE [dex]")
        plt.ylim(0.25, 0.75)
        plt.title(f"Fold {i} RMSE Curves")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.savefig(GNN_RESULTS_DIR / f"cv_gnn_fold_{i}_rmse_curve.png", bbox_inches='tight')
        plt.close()
        
    # Aggregate and plot final results
    final_df = pd.concat(all_preds_df)
    final_metrics_df = pd.DataFrame(all_metrics)
    aggregated_metrics = final_metrics_df.agg(['mean', 'std'])
    print("\n--- Aggregated GNN CV Metrics (mean +/- std) ---")
    for metric in aggregated_metrics.columns:
        mean, std = aggregated_metrics.loc['mean', metric], aggregated_metrics.loc['std', metric]
        print(f"  {metric:<15}: {mean:.4f} +/- {std:.4f}")
    
    plt.figure(figsize=(4, 4), dpi=150)
    plt.plot([5.5, 10.5], [5.5, 10.5], lw=1, c="0.7", ls='--', zorder=1)
    plt.scatter(final_df['y_true'], final_df['y_pred'], s=5, c=final_df['fold'], cmap='viridis', edgecolors='none', alpha=0.5)
    plt.grid(alpha=0.15)
    plt.xlabel("True log(age/yr)", fontsize=12)
    plt.ylabel("Predicted log(age/yr)", fontsize=12)
    plt.xlim(5.8, 10.3)
    plt.ylim(5.8, 10.3)
    mean_rmse = aggregated_metrics.loc['mean', 'rmse']
    mean_nmad = aggregated_metrics.loc['mean', 'nmad']
    plt.title(f"EdgeInteractionGNN (5-fold CV)\nRMSE={mean_rmse:.4f} dex, NMAD={mean_nmad:.4f} dex")
    plt.savefig(GNN_RESULTS_DIR / "cv_gnn_final_plot.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
