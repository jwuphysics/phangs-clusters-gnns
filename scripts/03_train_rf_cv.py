import pickle
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.training import get_rf_predictions, calculate_metrics, save_results
from src.cross_validation import load_cv_splits

RESULTS_DIR = ROOT / "results"
PROCESSED_DATA_PATH = ROOT / "data/processed/galaxy_graphs.pkl"
CV_SPLIT_PATH = RESULTS_DIR / "cv_galaxy_splits.json"
RF_RESULTS_DIR = RESULTS_DIR / "rf"
RF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data_dict = pickle.load(f)

    cv_splits = load_cv_splits(CV_SPLIT_PATH)

    all_preds_df = []
    all_metrics = []

    for i, fold in enumerate(tqdm(cv_splits, desc="Cross-validation folds")):
        train_galaxies, valid_galaxies = fold['train'], fold['valid']

        X_train = np.concatenate([data_dict[g].x.numpy() for g in train_galaxies])
        y_train = np.concatenate([data_dict[g].y.numpy() for g in train_galaxies])
        X_valid = np.concatenate([data_dict[g].x.numpy() for g in valid_galaxies])
        y_valid = np.concatenate([data_dict[g].y.numpy() for g in valid_galaxies])

        y_train_mask = ~np.isnan(y_train.ravel())
        X_train, y_train = X_train[y_train_mask], y_train[y_train_mask]

        p_valid = get_rf_predictions(X_train, y_train, X_valid)

        metrics = calculate_metrics(p_valid, y_valid)
        all_metrics.append(metrics)
        print(f"Fold {i} RF Metrics: {metrics}")
        save_results(RF_RESULTS_DIR / f"cv_rf_fold_{i}_metrics.json", metrics)

        # Get cluster IDs and galaxy names for validation set
        valid_cluster_ids = np.concatenate([data_dict[g].cluster_id for g in valid_galaxies])
        valid_galaxy_names = np.concatenate([[g] * data_dict[g].x.shape[0] for g in valid_galaxies])

        pred_df = pd.DataFrame({
            'y_pred': p_valid.ravel(), 
            'y_true': y_valid.ravel(), 
            'cluster_id': valid_cluster_ids,
            'galaxy': valid_galaxy_names,
            'fold': i
        })
        all_preds_df.append(pred_df)
        save_results(RF_RESULTS_DIR / f"cv_rf_fold_{i}_predictions.csv", pred_df)

    final_df = pd.concat(all_preds_df)
    final_df.to_csv(RF_RESULTS_DIR / "predictions.csv", index=False)
    
    final_metrics_df = pd.DataFrame(all_metrics)
    aggregated_metrics = final_metrics_df.agg(['mean', 'std'])
    print("\n--- Aggregated Random Forest CV Metrics (mean +/- std) ---")
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
    plt.title(f"Random Forest (5-fold CV)\nRMSE={mean_rmse:.4f} dex, NMAD={mean_nmad:.4f} dex")
    plt.savefig(RF_RESULTS_DIR / "cv_rf_final_plot.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
