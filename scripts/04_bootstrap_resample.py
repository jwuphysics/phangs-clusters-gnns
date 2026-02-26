import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.training import (
    block_bootstrap_metrics, format_bootstrap_results
)

RESULTS_DIR = ROOT / "results"

RESULT_CONFIGS = {
    'gnn': {
        'name': 'GNN (T25)',
        'predictions': RESULTS_DIR / 'gnn' / 'predictions.csv',
        'output_dir': RESULTS_DIR / 'gnn',
    },
    'rf': {
        'name': 'Random Forest (T25)',
        'predictions': RESULTS_DIR / 'rf' / 'predictions.csv',
        'output_dir': RESULTS_DIR / 'rf',
    },
    'gnn_t21': {
        'name': 'GNN (T21)',
        'predictions': RESULTS_DIR / 'gnn_T21' / 'predictions.csv',
        'output_dir': RESULTS_DIR / 'gnn_T21',
    },
    'rf_t21': {
        'name': 'Random Forest (T21)',
        'predictions': RESULTS_DIR / 'rf_T21' / 'predictions.csv',
        'output_dir': RESULTS_DIR / 'rf_T21',
    },
    'gnn_no_edges': {
        'name': 'GNN (no edges)',
        'predictions': RESULTS_DIR / 'gnn_no_edges' / 'predictions.csv',
        'output_dir': RESULTS_DIR / 'gnn_no_edges',
    },
    'gnn_no_photometry': {
        'name': 'GNN (no photometry)',
        'predictions': RESULTS_DIR / 'gnn_no_photometry' / 'predictions.csv',
        'output_dir': RESULTS_DIR / 'gnn_no_photometry',
    },
}

N_BOOTSTRAP = 1000
SEED = 42


def process_results(config: dict, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Process a single results configuration and compute bootstrap metrics.
    
    Args:
        config: Dictionary with 'name', 'predictions', and 'output_dir' keys
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed
    
    Returns:
        Bootstrap results dictionary
    """
    predictions_path = config['predictions']
    output_dir = config['output_dir']
    name = config['name']
    
    if not predictions_path.exists():
        print(f"  Skipping {name}: {predictions_path} not found")
        return None
    
    df = pd.read_csv(predictions_path)
    
    if 'galaxy' not in df.columns:
        print(f"  Skipping {name}: 'galaxy' column not found for block-bootstrap")
        return None
    
    n_galaxies = df['galaxy'].nunique()
    n_samples = len(df)
    print(f"  Processing {name}: {n_samples} samples, {n_galaxies} galaxies")
    
    results = block_bootstrap_metrics(
        df, block_col='galaxy', n_bootstrap=n_bootstrap, seed=seed
    )
    
    # Save results
    output_path = output_dir / 'bootstrap_metrics.json'
    output_data = {
        'config': {
            'n_bootstrap': n_bootstrap,
            'seed': seed,
            'n_samples': n_samples,
            'n_galaxies': n_galaxies,
        },
        'metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics with block-bootstrap uncertainties"
    )
    parser.add_argument(
        '--model', '-m', type=str, default=None,
        choices=['gnn', 'rf', 'gnn_t21', 'rf_t21', 'gnn_no_edges', 'gnn_no_photometry', 'all'],
        help="Model to process. If not specified or 'all', processes all available."
    )
    parser.add_argument(
        '--n-bootstrap', '-n', type=int, default=N_BOOTSTRAP,
        help=f"Number of bootstrap iterations (default: {N_BOOTSTRAP})"
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    args = parser.parse_args()
    
    print(f"Block-bootstrapping results (n_bootstrap={args.n_bootstrap}, seed={args.seed})")
    
    if args.model is None or args.model == 'all':
        configs_to_process = RESULT_CONFIGS
    else:
        configs_to_process = {args.model: RESULT_CONFIGS[args.model]}
    
    all_results = {}
    for key, config in configs_to_process.items():
        results = process_results(config, n_bootstrap=args.n_bootstrap, seed=args.seed)
        if results is not None:
            all_results[key] = results
            print(f"\n  {config['name']}:")
            print(format_bootstrap_results(results, metrics=['rmse', 'nmad', 'mae', 'bias']))
            print("")
    
    # Print summary table
    if len(all_results) > 1:
        print("Summary")
        for key, results in all_results.items():
            name = RESULT_CONFIGS[key]['name']
            rmse = results['rmse']
            nmad = results['nmad']
            print(f"  {name:<20}: RMSE = {rmse['value']:.4f} +/- {rmse['std']:.4f}, "
                  f"NMAD = {nmad['value']:.4f} ± {nmad['std']:.4f}")


if __name__ == "__main__":
    main()
