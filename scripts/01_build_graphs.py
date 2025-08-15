from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
import pickle
import numpy as np
from pathlib import Path
import sys
import pandas as pd
import torch
from tqdm import tqdm
import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data_processing import load_galaxy_data, load_galaxy_data_old, create_graph_from_df

# Configuration
DATA_DIR = ROOT / "data"
CAT_DIR = DATA_DIR / "IR5"
RESULTS_DIR = ROOT / "results"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "galaxy_graphs.pkl"

# Removed:
# No metadata:        NGC_1510
# ACS instead of WFC3: NGC_628C, NGC_628E, NGC_1300, NGC_1672, NGC_3621
# Causes NaNs (if we don't remove NaNs) or too sparse: NGC_1317 NGC_1559
# 

ALL_GALAXIES = ["IC_1954", "IC_5332", "NGC_0685", "NGC_1087", "NGC_1097", "NGC_1365", "NGC_1385", "NGC_1433", "NGC_1512", "NGC_1566", "NGC_1792", "NGC_2775", "NGC_2835", "NGC_2903", "NGC_3351", "NGC_3627", "NGC_4254", "NGC_4298", "NGC_4303", "NGC_4321", "NGC_4535", "NGC_4536", "NGC_4548", "NGC_4569", "NGC_4571", "NGC_4654", "NGC_4689", "NGC_4826", "NGC_5068", "NGC_5248", "NGC_6744", "NGC_7496"]

# old
# ALL_GALAXIES = ["NGC_1433", "NGC_2835", "NGC_1512", "NGC_3351", "NGC_7496", "NGC_5248", "NGC_4535", "NGC_1365", "NGC_5068", "NGC_4321", "NGC_3627", "NGC_4254", "NGC_1566", "IC_5332"]


edge_features = ["separation", "polar_angle"]
graph_features = ["sin_i", "D"] # skip... "cos_pa"


RA_DEC_COLS = ["PHANGS_RA", "PHANGS_DEC"]
PHOT_COLS = ["PHANGS_F275W_VEGA", "PHANGS_F336W_VEGA", "PHANGS_F438W_VEGA", "PHANGS_F555W_VEGA", "PHANGS_F814W_VEGA", "PHANGS_CI"] #  
Y_COLS = ["cluster_log_age"]

R_LINK_ARCSEC = 60
# R_LINK_KPC = 1.0

def main():
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

    # which edge features
    

    # make graphs
    data_list = []
    for galaxy in tqdm(ALL_GALAXIES, desc="Processing galaxies"):
        meta_values = galaxies_meta.loc[galaxy][graph_features].values.astype(float) if graph_features else [] 
        meta_RA_DEC = galaxies_meta.loc[galaxy][["RAJ2000", "DEJ2000"]].values.astype(float)

        df = load_galaxy_data(galaxy, CAT_DIR, PHOT_COLS, RA_DEC_COLS, source="human", include_class3=False)
        # df = load_galaxy_data_old(galaxy, "/home/john/research/phangs-star-clusters/data", PHOT_COLS, RA_DEC_COLS) # Turner ages
        if df is not None and not df.empty:
            # approximately convert kpc -> arcsec
            # R_LINK_ARCSEC = 3600 * np.rad2deg(1e-3 * R_LINK_KPC / meta_values[0]) 
            
            
            graph = create_graph_from_df(df, PHOT_COLS, Y_COLS, origin=meta_RA_DEC, r_link_arcsec=R_LINK_ARCSEC, edge_features=edge_features)
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
    print(f"Processed data for {len(data_dict)} galaxies ({sum([graph.x.shape[0] for graph in data_list])} clusters) saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
