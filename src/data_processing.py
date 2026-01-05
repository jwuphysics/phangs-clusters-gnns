import numpy as np
import pandas as pd
import torch
import scipy
from astropy.table import Table
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def load_galaxy_data(galaxy_name, data_dir, phot_cols, ra_dec_cols, source="human", include_class3=False):
    phot_path = f"{data_dir}/catalogs/hlsp_phangs-cat_hst_uvis_{galaxy_name.lower().replace('_','')}_multi_v1_obs-{source}-cluster-class12.fits"
    ages_path = f"{data_dir}/catalogs/hlsp_phangs-cat_hst_uvis_{galaxy_name.lower().replace('_','')}_multi_v1_sed-ground-halpha-{source}-cluster-class12.fits"
    
    try:
        phot12 = Table.read(phot_path).to_pandas().set_index("ID_PHANGS_CLUSTER")
        ages12 = Table.read(ages_path).to_pandas().set_index("ID_PHANGS_CLUSTER")
    except FileNotFoundError:
        print(f"Catalogs not found for: {galaxy_name}")
        return None

    t12 = pd.DataFrame({
        "galaxy": galaxy_name,
        "cluster_id": ages12.index.values,
        "cluster_log_age": ages12["PHANGS_SED_AGE"].apply(np.log10).values + 6,
        "association": False,
    })
    t12 = t12.join(phot12[phot_cols + ra_dec_cols], how="inner", on="cluster_id")
    t12 = t12.rename({"PHANGS_RA": "ra", "PHANGS_DEC": "dec"}, axis=1)
    
    if include_class3:
        phot_path = f"{data_dir}/catalogs/hlsp_phangs-cat_hst_uvis_{galaxy_name.lower().replace('_','')}_multi_v1_obs-{source}-compact-association-class3.fits"
        ages_path = f"{data_dir}/catalogs/hlsp_phangs-cat_hst_uvis_{galaxy_name.lower().replace('_','')}_multi_v1_sed-ground-halpha-{source}-compact-association-class3.fits"
  
        phot3 = Table.read(phot_path).to_pandas().set_index("ID_PHANGS_CLUSTER")
        ages3 = Table.read(ages_path).to_pandas().set_index("ID_PHANGS_CLUSTER")

        # remove these ages, since we don't want to learn them (all very young of course) -- we just want the nodes in the graph!
        t3 = pd.DataFrame({
            "galaxy": galaxy_name,
            "cluster_id": ages3.index.values,
            "cluster_log_age": np.nan, # ages3["PHANGS_SED_AGE"].apply(np.log10).values + 6,
            "association": True,
        })
        t3 = t3.join(phot3[phot_cols + ra_dec_cols], how="inner", on="cluster_id")
        t3 = t3.rename({"PHANGS_RA": "ra", "PHANGS_DEC": "dec"}, axis=1)

        table = pd.concat([t12, t3], axis=0)
    else:
        table = t12
    
    return table


def load_galaxy_data_old(galaxy_name, data_dir, phot_cols, ra_dec_cols):
    """DEPRECATED VERSION using older dataset"""

    try:
        phot = Table.read(f"{data_dir}/PHANGS-CAT/catalogs/hlsp_phangs-cat_hst_uvis_{galaxy_name.lower().replace('_','')}_multi_v1_obs-human-cluster-class12.fits").to_pandas().set_index("ID_PHANGS_CLUSTER")
        ages = Table.read(f"{data_dir}/PHANGS_STAMPS/{galaxy_name}/{galaxy_name.replace('_', '').lower()}_data.fits")
    except FileNotFoundError:
        print(f"Catalogs not found for: {galaxy_name}")
        return None
    

    cluster_ids = [np.int32(id) for id in ages["ID"].data]
    table_log_ages = ages["LOG_AGE"].data
    decoded_log_ages = [age.decode('utf-8') for age in table_log_ages]
    
    log_ages_list = [float(age) for age in decoded_log_ages]
    cluster_log_ages = np.array(log_ages_list)

    table = pd.DataFrame({
        "galaxy": galaxy_name, 
        "cluster_id": cluster_ids, 
        "cluster_log_age": cluster_log_ages + 0,
    })
    table = table.join(phot[phot_cols + ra_dec_cols], how="inner", on="cluster_id")

    table = table.rename({"PHANGS_RA": "ra", "PHANGS_DEC": "dec"}, axis=1)    
    return table


def create_graph_from_df(df, x_cols, y_cols, r_link_arcsec=15, origin=None, epsilon=1e-8, edge_features=["separation", "polar_angle"]):
    x = torch.tensor(df[x_cols].values, dtype=torch.float)
    y = torch.tensor(df[y_cols].values, dtype=torch.float)

    if origin is None:
        origin = np.mean(df[["ra", "dec"]].values, axis=0)
        
    pos_gnomonic = np.vstack([
        ((df["ra"] - origin[0]) * np.cos(np.deg2rad(origin[1]))).values, 
        (df["dec"] - origin[1]).values
    ]).transpose() * 3600
    pos = torch.tensor(pos_gnomonic, dtype=torch.float)

    # include distance from center as feature
    # x = torch.concat([x, torch.linalg.norm(pos, dim=1).reshape(-1, 1)], dim=1)
    
    kd_tree = scipy.spatial.KDTree(pos, leafsize=10)
    edge_index_np = kd_tree.query_pairs(r=r_link_arcsec, output_type="ndarray")
    
    edge_index = torch.from_numpy(edge_index_np).t().contiguous().long()
    edge_index = to_undirected(edge_index)

    row, col = edge_index
    o = pos.mean(0).numpy()
    vec_row = pos[row] - o
    vec_col = pos[col] - o
    
    norm_row = torch.linalg.norm(vec_row, dim=1, keepdim=True)
    norm_col = torch.linalg.norm(vec_col, dim=1, keepdim=True)

    e_row = vec_row / (norm_row + epsilon)
    e_col = vec_col / (norm_col + epsilon)

    delta_pos = pos[row] - pos[col]
    dist = torch.linalg.norm(delta_pos, dim=1, keepdim=True)
    cos_angle = torch.sum(e_row * e_col, dim=1, keepdim=True).clamp(-1.0, 1.0)
    edge_attr = []
    if "separation" in edge_features:
        edge_attr.append(dist)
    if "polar_angle" in edge_features:
        edge_attr.append(cos_angle)
        
    edge_attr = torch.cat(edge_attr, dim=1) if len(edge_attr) > 0 else torch.tensor([])
    center = torch.tensor(origin, dtype=torch.float)
        
    graph_data = Data(x=x, y=y, pos=pos, edge_index=edge_index, edge_attr=edge_attr, center=center)
    return graph_data
