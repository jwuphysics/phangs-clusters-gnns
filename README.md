# PHANGS Star Cluster Age Prediction with GNNs

This project uses a Graph Neural Network (GNN) to predict the ages of star clusters based on their photometric properties and their spatial relationships within their host galaxies.

## Project Structure

- `data/`: Contains raw and processed data.
    - `data/IR5/catalogs/hlsp_phangs-cat_hst_*.fits`: Raw catalog FITS files.
    - `data/processed/galaxy_graphs.pkl`: Stores the processed graph data.
    - `data/Leroy+2021_table3.fits`: Contains metadata for galaxy-level information.
- `results/`: Stores output from model training, including predictions, metrics, and plots for each cross-validation fold.
- `scripts/`: Executable Python scripts for running the pipeline.
- `src/`: Source code for data processing, model definition, and training.

## Workflow

0. (Optional) Sweep hyperparameters.
    ```bash
    python scripts/00_sweep_hyperparameters.py
    ```

2.  Build the graphs. 
    ```bash
    python scripts/01_build_graphs.py
    ```

    This script converts the raw FITS catalogs into a dictionary of galaxy graphs, while imposing selection criteria and included feature sets (node-, edge- and graph-level features).

3.  Train GNN. 
    ```bash
    python scripts/02_train_gnn_cv.py
    ```
    This will create a random-seeded k-fold cross-validation split, train the GNN on each fold, and save the results. By default the results are saved in `results/gnn/`

4.  Train RF baseline. 
    ```bash
    python scripts/03_train_rf_cv.py
    ```
    The Random Forest baseline scriptuses the same cross-validation splits as the GNN and performs reasonably well (without environmental information).
