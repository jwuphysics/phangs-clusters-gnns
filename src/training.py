import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import median_abs_deviation
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_node, scatter
from typing import Tuple

def huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta=1.0) -> torch.Tensor:
    """Compute Huber loss while again masking out inf values."""
    finite_mask = (y_true > 0.) & (y_true.isfinite())
    
    if not finite_mask.any():
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    y_pred_masked = y_pred[finite_mask]
    y_true_masked = y_true[finite_mask]
    
    return F.huber_loss(y_pred_masked, y_true_masked, delta=delta, reduction="sum"), finite_mask.sum()

def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss while again masking out inf values"""
    finite_mask = (y_true > 0.) & (y_true.isfinite())
    
    if not finite_mask.any():
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    y_pred_masked = y_pred[finite_mask]
    y_true_masked = y_true[finite_mask]
    
    return F.mse_loss(y_pred_masked, y_true_masked, reduction="sum"), finite_mask.sum()


def gaussian_nll_loss(y_pred: torch.Tensor, y_true: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian NLL with per-node logvar (already averaged per-galaxy)."""
    finite_mask = (y_true > 0.) & (y_true.isfinite())
    
    if not finite_mask.any():
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True), 0

    y_pred = y_pred[finite_mask]
    y_true = y_true[finite_mask]
    logvar = logvar[finite_mask]
    
    nll = 0.5 * ((y_pred - y_true)**2 / (10**logvar) + logvar)
    
    return nll.sum(), finite_mask.sum()
    

def compute_rmse(preds, targs):
    """lil helper func"""
    finite_mask = (targs > 0.) & (np.isfinite(targs))
    y_pred_masked = preds[finite_mask]
    y_true_masked = targs[finite_mask]
    return np.mean((y_pred_masked - y_true_masked)**2)**0.5

    
# def train_gnn_epoch(dataloader, model, optimizer, device="cuda", augment_node_scatter=3e-4):

#     model.train()

#     loss_total = 0
#     for data in dataloader:
#         if augment_node_scatter is not None:
#             data_node_features_scatter = augment_node_scatter * torch.randn_like(data.x) * torch.std(data.x, dim=0)
#             data.x += data_node_features_scatter

            
#         data.to(device)

#         optimizer.zero_grad()
#         y_pred = model(data)

#         loss = mse_loss(y_pred, data.y)

#         loss.backward()
#         optimizer.step()
#         loss_total += loss.item()

#     return loss_total / len(dataloader)


# def validate_gnn_epoch(dataloader, model, device="cuda"):
#     model.eval()

#     uncertainties = []
#     loss_total = 0

#     y_preds = []
#     y_trues = []

#     for data in dataloader:
#         with torch.no_grad():
#             data.to(device)
#             y_pred = model(data)
#             loss = mse_loss(y_pred, data.y)

#             loss_total += loss.item()
#             y_preds += list(y_pred.detach().cpu().numpy())
#             y_trues += list(data.y.detach().cpu().numpy())

#     y_preds = np.concatenate(y_preds)
#     y_trues = np.array(y_trues)

#     return (
#         loss_total / len(dataloader),
#         y_preds,
#         y_trues,
#     )

def train_gnn_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    augment_node_scatter: float = 3e-4,
    clip_grad_norm: float = 3.0,
) -> float:
    """Train one epoch for GNN model.
    
    Args:
        dataloader: Data loader for training data (X, y tuples)
        model: Model to train
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    loss_total = 0
    num_samples = 0
    
    for data in (dataloader):
        if augment_node_scatter is not None:
            data_node_features_scatter = augment_node_scatter * torch.randn_like(data.x) * torch.std(data.x, dim=0)
            data.x += data_node_features_scatter
            assert not torch.isnan(data.x).any() 

        data.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        y_pred, logvar_pred = output.chunk(2, dim=1)
        
        assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
        
        # Average logvar per-galaxy, then broadcast back to nodes
        logvar_per_graph = scatter(logvar_pred, data.batch, dim=0, reduce='mean')
        logvar_per_node = logvar_per_graph[data.batch]

        loss, num_samp = gaussian_nll_loss(y_pred, data.y, logvar_per_node)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        
        optimizer.step()
        loss_total += loss.item()
        num_samples += num_samp.item()
        
    return loss_total / num_samples


def validate_gnn_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    device: str
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate GNN model.
    
    Args:
        dataloader: Validation data loader
        model: Model to validate
        device: Device to validate on
        
    Returns:
        Tuple of (loss, predictions, targets)
    """
    model.eval()
    loss_total = 0
    num_samples = 0
    y_preds = []
    y_trues = []
    
    for data in dataloader:
        with torch.no_grad():
            data.to(device)
            
            output = model(data)
            y_pred, logvar_pred = output.chunk(2, dim=1)
        
            assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
            
            logvar_per_graph = scatter(logvar_pred, data.batch, dim=0, reduce='mean')
            logvar_per_node = logvar_per_graph[data.batch]
    
            loss, num_samp = gaussian_nll_loss(y_pred, data.y, logvar_per_node)

            loss_total += loss.item()
            num_samples += num_samp.item()
            
            y_preds.append(y_pred.detach().cpu().numpy())
            y_trues.append(data.y.detach().cpu().numpy())
            
    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    
    return loss_total / num_samples, y_preds, y_trues


def validate_gnn_epoch_with_ids(
    dataloader: DataLoader,
    model: nn.Module,
    device: str
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate GNN model and extract cluster IDs and galaxy names.
    
    This function performs validation and also extracts the cluster_id and 
    galaxy name attributes from each graph, properly handling PyG's batching
    of custom attributes.
    
    Args:
        dataloader: Validation data loader
        model: Model to validate
        device: Device to validate on
        
    Returns:
        Tuple of (loss, predictions, targets, cluster_ids, galaxy_names)
    """
    import torch
    
    model.eval()
    loss_total = 0
    num_samples = 0
    y_preds = []
    y_trues = []
    all_cluster_ids = []
    all_galaxy_names = []
    
    for data in dataloader:
        with torch.no_grad():
            data.to(device)
            
            output = model(data)
            y_pred, logvar_pred = output.chunk(2, dim=1)
        
            assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
            
            logvar_per_graph = scatter(logvar_pred, data.batch, dim=0, reduce='mean')
            logvar_per_node = logvar_per_graph[data.batch]
    
            loss, num_samp = gaussian_nll_loss(y_pred, data.y, logvar_per_node)
            loss_total += loss.item()
            num_samples += num_samp.item()
            
            y_preds.append(y_pred.detach().cpu().numpy())
            y_trues.append(data.y.detach().cpu().numpy())
            
            # Extract cluster IDs
            # PyG batches custom numpy/list attributes as lists of the original arrays
            if hasattr(data, 'cluster_id'):
                if isinstance(data.cluster_id, list):
                    try:
                        batch_cluster_ids = np.concatenate([
                            c.cpu().numpy() if torch.is_tensor(c) else np.asarray(c)
                            for c in data.cluster_id
                        ])
                    except (ValueError, AttributeError):
                        batch_cluster_ids = np.concatenate(data.cluster_id)
                else:
                    batch_cluster_ids = data.cluster_id
                    if torch.is_tensor(batch_cluster_ids):
                        batch_cluster_ids = batch_cluster_ids.cpu().numpy()
                
                batch_cluster_ids = np.array(batch_cluster_ids).flatten()
                all_cluster_ids.append(batch_cluster_ids)
            
            # Extract galaxy names
            # data.name is typically a list of galaxy name strings (one per graph in batch)
            # We need to repeat each name for the number of nodes in that graph
            if hasattr(data, 'name') and isinstance(data.name, list):
                batch_galaxy_names = []
                for i, name in enumerate(data.name):
                    n_nodes = (data.batch == i).sum().item()
                    batch_galaxy_names.extend([name] * n_nodes)
                all_galaxy_names.append(np.array(batch_galaxy_names))
            
    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    cluster_ids = np.concatenate(all_cluster_ids) if all_cluster_ids else np.array([])
    galaxy_names = np.concatenate(all_galaxy_names) if all_galaxy_names else np.array([])
    
    return loss_total / num_samples, y_preds, y_trues, cluster_ids, galaxy_names


# def train_gnn_subgraph_epoch(
#     dataloader: DataLoader,
#     model: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     device: str = "cuda",
#     augment_node_scatter: float = 3e-4,
#     clip_grad_norm: float = 3.0,
# ) -> float:
#     """Train one epoch for GNN model using dynamic subgraph batching.
    
#     Args:
#         dataloader: Data loader for training data (X, y tuples)
#         model: Model to train
#         optimizer: Optimizer
#         device: Device to train on
        
#     Returns:
#         Average training loss for the epoch
#     """
#     model.train()
#     loss_total = 0
#     num_samples = 0
    
#     for data in (dataloader):
#         if augment_node_scatter is not None:
#             data_node_features_scatter = augment_node_scatter * torch.randn_like(data.x) * torch.std(data.x, dim=0)
#             data.x += data_node_features_scatter
#             assert not torch.isnan(data.x).any() 

#         data.to(device)
        
#         optimizer.zero_grad()
#         output = model(data)

#         y_pred, logvar_pred = output.chunk(2, dim=1)
        
#         # we only keep "seed node" predictions
#         y_pred = y_pred[:data.batch_size]
#         logvar_pred = logvar_pred[:data.batch_size]
        
#         assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
        
#         y_pred = y_pred.view(-1, data.y.shape[1] if len(data.y.shape) > 1 else 2)
#         logvar_pred = logvar_pred.mean()

#         loss, num_samp = gaussian_nll_loss(y_pred, data.y[:data.batch_size], logvar_pred)
#         loss.backward()
#         if clip_grad_norm is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        
#         optimizer.step()
#         loss_total += loss.item()
#         num_samples += num_samp.item()
        
#     return loss_total / num_samples


# def validate_gnn_subgraph_epoch(
#     dataloader: DataLoader,
#     model: nn.Module,
#     device: str
# ) -> Tuple[float, np.ndarray, np.ndarray]:
#     """Validate GNN model.
    
#     Args:
#         dataloader: Validation data loader
#         model: Model to validate
#         device: Device to validate on
        
#     Returns:
#         Tuple of (loss, predictions, targets)
#     """
#     model.eval()
#     loss_total = 0
#     num_samples = 0
#     y_preds = []
#     y_trues = []
    
#     for data in dataloader:
#         with torch.no_grad():
#             data.to(device)
            
#             output = model(data)
#             y_pred, logvar_pred = output.chunk(2, dim=1)
#             # we only keep "seed node" predictions
#             y_pred = y_pred[:data.batch_size]
#             logvar_pred = logvar_pred[:data.batch_size]
            
#             assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
            
#             y_pred = y_pred.view(-1, data.y.shape[1] if len(data.y.shape) > 1 else 2)
#             logvar_pred = logvar_pred.mean()
    
#             loss, num_samp = gaussian_nll_loss(y_pred, data.y[:data.batch_size], logvar_pred)
#             loss_total += loss.item()
#             num_samples += num_samp.item()
            
#             y_preds.append(y_pred.detach().cpu().numpy())
#             y_trues.append(data.y.detach().cpu().numpy())
            
#     y_preds = np.concatenate(y_preds, axis=0)
#     y_trues = np.concatenate(y_trues, axis=0)
    
#     return loss_total / num_samples, y_preds, y_trues
    
# def train_gnn_mixture_epoch(
#     dataloader: DataLoader,
#     model: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     device: str = "cuda",
#     augment: bool = True
# ) -> float:
#     """Train one epoch for GNN model.
    
#     Args:
#         dataloader: Data loader for training data (X, y tuples)
#         model: Model to train
#         optimizer: Optimizer
#         device: Device to train on
        
#     Returns:
#         Average training loss for the epoch
#     """
#     model.train()
#     loss_total = 0
    
#     for data in (dataloader):
#         if augment: # add random noise
#             data_node_features_scatter = 3e-4 * torch.randn_like(data.x[:, :-1]) * torch.std(data.x[:, :-1], dim=0)
#             data.x[:, :-1] += data_node_features_scatter
#             assert not torch.isnan(data.x).any() 

#         data.to(device)
        
#         optimizer.zero_grad()
#         output = model(data)

#         y_pred, logvar_pred, outlier_fraction = output.chunk(3, dim=1)
        
#         assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
        
#         y_pred = y_pred.view(-1, data.y.shape[1] if len(data.y.shape) > 1 else 2)
#         logvar_pred = logvar_pred.mean()
#         outlier_fraction = outlier_fraction.mean()

#         loss = mixture_gaussian_nll_loss(y_pred, data.y, logvar_pred, outlier_fraction)
#         # loss = gaussian_nll_loss(y_pred, data.y, logvar_pred)
#         loss.backward()
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()
#         loss_total += loss.item()
        
#     return loss_total / len(dataloader)


# def validate_gnn_mixture_epoch(
#     dataloader: DataLoader,
#     model: nn.Module,
#     device: str
# ) -> Tuple[float, np.ndarray, np.ndarray]:
#     """Validate GNN model.
    
#     Args:
#         dataloader: Validation data loader
#         model: Model to validate
#         device: Device to validate on
        
#     Returns:
#         Tuple of (loss, predictions, targets)
#     """
#     model.eval()
#     loss_total = 0
#     y_preds = []
#     y_trues = []
    
#     for data in dataloader:
#         with torch.no_grad():
#             data.to(device)
            
#             output = model(data)
#             y_pred, logvar_pred, outlier_fraction = output.chunk(3, dim=1)
        
#             assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
            
#             y_pred = y_pred.view(-1, data.y.shape[1] if len(data.y.shape) > 1 else 2)
#             logvar_pred = logvar_pred.mean()
#             outlier_fraction = outlier_fraction.mean()
    
#             loss = mixture_gaussian_nll_loss(y_pred, data.y, logvar_pred, outlier_fraction)
#             # loss = gaussian_nll_loss(y_pred, data.y, logvar_pred)
#             loss_total += loss.item()
            
#             y_preds.append(y_pred.detach().cpu().numpy())
#             y_trues.append(data.y.detach().cpu().numpy())
            
#     y_preds = np.concatenate(y_preds, axis=0)
#     y_trues = np.concatenate(y_trues, axis=0)
    
#     return loss_total / len(dataloader), y_preds, y_trues


def get_rf_predictions(X_train, y_train, X_valid):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train.ravel())
    p_valid = rf.predict(X_valid)
    return p_valid


def calculate_metrics(y_pred, y_true):
    y_pred, y_true = y_pred.ravel(), y_true.ravel()
    mask = ~np.isnan(y_true)
    y_pred_clean, y_true_clean = y_pred[mask], y_true[mask]

    residuals = y_pred_clean - y_true_clean

    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)

    nmad = median_abs_deviation(residuals, scale='normal')

    outlier_frac = np.mean(np.abs(residuals) > 3 * nmad) if nmad > 0 else 0.0

    metrics = {
        'rmse': rmse, 'nmad': nmad, 'mae': mae,
        'bias': bias, 'outlier_frac': outlier_frac
    }
    return metrics


def save_results(path, data):
    if isinstance(data, dict):
        json_data = {k: float(v) if isinstance(v, np.floating) else v for k, v in data.items()}
        with open(path, 'w') as f:
            json.dump(json_data, f, indent=4)
    elif isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False)
    else:
        raise TypeError(f"Unsupported data type for saving: {type(data)}")
