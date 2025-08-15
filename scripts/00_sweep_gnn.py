from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
import matplotlib.pyplot as plt
from muon import SingleDeviceMuonWithAuxAdam
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import sys
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import wandb
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data_processing import load_galaxy_data, load_galaxy_data_old, create_graph_from_df
from src.cross_validation import create_stratified_k_folds_by_distance, create_random_k_folds, save_cv_splits, load_cv_splits
from src.model import *
from src.training import train_gnn_epoch, validate_gnn_epoch, calculate_metrics, save_results, compute_rmse

# Configuration
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results-sweep-targeted"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "galaxy_graphs.pkl"
CV_SPLIT_PATH = RESULTS_DIR / "cv_galaxy_splits.json"
GNN_RESULTS_DIR = RESULTS_DIR / "gnn"
GNN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_FOLDS = 3
SEED = 42


# WandB config
project = "phangs-gnn-sweep-classes123"
N_SWEEP = 200

sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "mae"},
    "parameters": {
        'r_link_arcsec': {"values": [1, 3, 10, 30]},
        'adam_lr': {"values": [1e-3, 3e-3, 1e-2]},
        'adam_wd': {"values": [0]}, 
        'n_epochs': {"values": [100, 150, 200]}, 
        'batch_size': {"values": [2, 4]},
        'muon_lr': {"values": [1e-3, 3e-3, 1e-2]},
        'muon_wd': {"values": [1e-8, 1e-6]},
        'n_layers': {"values": [1]}, 
        'hidden_channels': {"values": [64, 128, 256]}, 
        'f_latent_channels': {"values": [0.25]},
        'n_unshared_layers': {"values": [4, 8, 16]},
    },
}

# Configuration
DATA_DIR = ROOT / "data"
CAT_DIR = DATA_DIR / "IR5"
RESULTS_DIR = ROOT / "results"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "galaxy_graphs.pkl"

ALL_GALAXIES = ["IC_1954", "IC_5332", "NGC_0685", "NGC_1087", "NGC_1097", "NGC_1317", "NGC_1365", "NGC_1385", "NGC_1433", "NGC_1512", "NGC_1566", "NGC_1792", "NGC_2775", "NGC_2835", "NGC_2903", "NGC_3351", "NGC_3627", "NGC_4254", "NGC_4298", "NGC_4303", "NGC_4321", "NGC_4535", "NGC_4536", "NGC_4548", "NGC_4569", "NGC_4571", "NGC_4654", "NGC_4689", "NGC_4826", "NGC_5068", "NGC_5248", "NGC_6744", "NGC_7496"]

RA_DEC_COLS = ["PHANGS_RA", "PHANGS_DEC"]
PHOT_COLS = ["PHANGS_F275W_VEGA", "PHANGS_F336W_VEGA", "PHANGS_F438W_VEGA", "PHANGS_F555W_VEGA", "PHANGS_F814W_VEGA", "PHANGS_CI"] #  
Y_COLS = ["cluster_log_age"]

def build_graphs(run):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Load galaxy metadata from Leroy + 2021
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        sample_metadata = Table.read(DATA_DIR / "Leroy+2021_table3.fits").to_pandas()
        sample_metadata["Galaxy"] = [s.decode('utf-8').strip() for s in sample_metadata["Name"]]
        sample_metadata["Galaxy"] = sample_metadata["Galaxy"].str.replace("NGC", "NGC_").str.replace("IC", "IC_")
        sample_metadata = sample_metadata.set_index("Galaxy").rename({"NGC_685": "NGC_0685"})
        
        galaxies_meta = sample_metadata.loc[ALL_GALAXIES]
        galaxies_meta["sin_i"] = np.sin(np.deg2rad(galaxies_meta["i"]))
        galaxies_meta["cos_pa"] = np.cos(np.deg2rad(galaxies_meta["PA"]))
        galaxies_meta["D"] = galaxies_meta["Dist"]

    # make graphs
    data_list = []
    for galaxy in tqdm(ALL_GALAXIES, desc="Processing galaxies"):
        meta_values = galaxies_meta.loc[galaxy][["sin_i", "D"]].values.astype(float) # skip cos_pa,
        # meta_values = [] 
        meta_RA_DEC = galaxies_meta.loc[galaxy][["RAJ2000", "DEJ2000"]].values.astype(float)

        df = load_galaxy_data(galaxy, CAT_DIR, PHOT_COLS, RA_DEC_COLS, source="human", include_class3=False)
        # df = load_galaxy_data_old(galaxy, "/home/john/research/phangs-star-clusters/data", PHOT_COLS, RA_DEC_COLS) # Turner ages
        if df is not None and not df.empty:
            # approximately convert kpc -> arcsec
            # R_LINK_ARCSEC = 3600 * np.rad2deg(1e-3 * R_LINK_KPC / meta_values[0]) 
            
            graph = create_graph_from_df(df, PHOT_COLS, Y_COLS, origin=meta_RA_DEC, r_link_arcsec=run.config["r_link_arcsec"], edge_features=["separation", "polar_angle"])
            graph.u = torch.tensor(meta_values, dtype=torch.float)
            graph.name = galaxy
            data_list.append(graph)

    data_dict = {name: d for name, d in zip(ALL_GALAXIES, data_list)}
    print("Number of node features: ", graph.x.shape[-1])
    print("Number of edge features: ", graph.edge_attr.shape[-1])
    print("Number of graph features: ", len(graph.u))

    # Save processed data
    with open(PROCESSED_DATA_PATH, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Processed data for {len(data_dict)} galaxies saved to {PROCESSED_DATA_PATH}")


def train_gnn_cv(run):

    GNN_CONFIG = run.config 
    
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_dict = pickle.load(f)

    galaxy_names = list(data_dict.keys())

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

        n_latent_channels = int(GNN_CONFIG['f_latent_channels'] * GNN_CONFIG['hidden_channels'])

        model = EdgeInteractionGNN(
            n_layers=GNN_CONFIG['n_layers'], 
            hidden_channels=GNN_CONFIG['hidden_channels'],
            latent_channels=n_latent_channels, 
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
            dict(params=hidden_gains_biases, use_muon=False, lr=GNN_CONFIG["adam_lr"], betas=(0.9, 0.95), weight_decay=GNN_CONFIG["adam_wd"]),
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
        # plt.figure(figsize=(5, 4), dpi=120)
        # plt.plot(valid_rmses, label="Valid RMSE")
        # plt.xlabel("Epoch")
        # plt.ylabel("RMSE [dex]")
        # plt.ylim(0.3, 0.8)
        # plt.title(f"Fold {i} RMSE Curves")
        # plt.grid(alpha=0.2)
        # plt.legend()
        # plt.savefig(GNN_RESULTS_DIR / f"cv_gnn_fold_{i}_rmse_curve.png", bbox_inches='tight')
        # plt.close()
        
    # Aggregate and compute final metrics
    final_df = pd.concat(all_preds_df)
    final_metrics = calculate_metrics(final_df.y_pred.to_numpy(), final_df.y_true.to_numpy())
    
    # plt.figure(figsize=(4, 4), dpi=150)
    # plt.plot([5.5, 10.5], [5.5, 10.5], lw=1, c="0.7", ls='--', zorder=1)
    # plt.scatter(final_df['y_true'], final_df['y_pred'], s=5, c=final_df['fold'], cmap='viridis', edgecolors='none', alpha=0.5)
    # plt.grid(alpha=0.15)
    # plt.xlabel("True log(age/yr)", fontsize=12)
    # plt.ylabel("Predicted log(age/yr)", fontsize=12)
    # plt.xlim(5.8, 10.3)
    # plt.ylim(5.8, 10.3)
    # mean_rmse = aggregated_metrics.loc['mean', 'rmse']
    # mean_nmad = aggregated_metrics.loc['mean', 'nmad']
    # plt.title(f"EdgeInteractionGNN ({K_FOLDS}-fold CV)\nRMSE={mean_rmse:.4f} dex, NMAD={mean_nmad:.4f} dex")
    # plt.savefig(GNN_RESULTS_DIR / "cv_gnn_final_plot.png", bbox_inches='tight')
    # plt.close()

    wandb.log(
        {
            "rmse": final_metrics["rmse"],
            "nmad": final_metrics["nmad"],
            "mae": final_metrics["mae"],
            "bias": final_metrics["bias"],
            "outlier_frac": final_metrics["outlier_frac"],
        }
    )
    
def main():
    with wandb.init(project=project) as run:
        build_graphs(run)
        train_gnn_cv(run)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    wandb.agent(sweep_id, function=main, count=N_SWEEP)